"""Full-vocabulary diagnostics for an entropy-gated reverse-KL objective.

The probe is deliberately loss-free: both distributions are detached and the
returned tensors contain only additive statistics.  It is used to calibrate a
fixed absolute entropy threshold before enabling the training objective.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


DEFAULT_ENTROPY_THRESHOLDS = (
    0.01,
    0.02,
    0.03,
    0.05,
    0.075,
    0.10,
    0.125,
    0.15,
    0.20,
    0.25,
    0.30,
)

_BASE_SIZE = 9
_BASE_ELIGIBLE = 0
_BASE_SOURCE = 1
_BASE_MISSING_SOURCE = 2
_BASE_FINITE = 3
_BASE_NONFINITE = 4
_BASE_ENTROPY_SUM = 5
_BASE_RKL_SUM = 6
_BASE_TOP1_MISMATCH = 7
_BASE_NEGATIVE_RKL = 8


def compute_entropy_gated_ce_rkl_terms(
    student_logits: Tensor,
    teacher_logits: Tensor,
    labels: Tensor,
    *,
    entropy_threshold: float,
    rkl_cap: float,
    logit_clip: float = 50.0,
    cad_mode: bool = False,
    fkl_coef: float = 1.0,
    cj_power: float = 1.0,
) -> dict[str, Tensor]:
    """Return differentiable per-token CE/RKL hybrid loss terms.

    With ``cad_mode`` (Consistency Forcing, arXiv:2608.13925) the low-entropy
    branch becomes ``fkl_coef * FKL + (max_v q_v)^cj_power * capped RKL``: the
    forward-KL floor stabilizes the confidence-weighted reverse KL.
    """

    if student_logits.ndim != 2 or teacher_logits.ndim != 2:
        raise ValueError("student_logits and teacher_logits must be [tokens, vocab]")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            "student/teacher shape mismatch: "
            f"{tuple(student_logits.shape)} != {tuple(teacher_logits.shape)}"
        )
    if labels.ndim != 1 or labels.shape[0] != student_logits.shape[0]:
        raise ValueError(
            f"labels must be [tokens], got {tuple(labels.shape)} for "
            f"{tuple(student_logits.shape)} logits"
        )
    vocab_size = int(student_logits.shape[-1])
    if vocab_size < 2:
        raise ValueError("vocab size must be at least 2")
    if labels.numel() and (
        int(labels.min().item()) < 0 or int(labels.max().item()) >= vocab_size
    ):
        raise ValueError("labels contain values outside the vocabulary")
    if (
        not math.isfinite(entropy_threshold)
        or entropy_threshold < 0.0
        or entropy_threshold > 1.0
    ):
        raise ValueError(
            "entropy_threshold must be a finite normalized value in [0,1], "
            f"got {entropy_threshold}"
        )
    if not math.isfinite(rkl_cap) or rkl_cap <= 0.0:
        raise ValueError(f"rkl_cap must be finite and positive, got {rkl_cap}")
    if not math.isfinite(logit_clip) or logit_clip <= 0.0:
        raise ValueError(
            f"logit_clip must be finite and positive, got {logit_clip}"
        )
    if not math.isfinite(fkl_coef) or fkl_coef < 0.0:
        raise ValueError(
            f"fkl_coef must be finite and non-negative, got {fkl_coef}"
        )
    if not math.isfinite(cj_power) or cj_power < 0.0:
        raise ValueError(
            f"cj_power must be finite and non-negative, got {cj_power}"
        )

    student = torch.nan_to_num(
        student_logits.float(),
        nan=0.0,
        posinf=logit_clip,
        neginf=-logit_clip,
    ).clamp(min=-logit_clip, max=logit_clip)
    teacher = torch.nan_to_num(
        teacher_logits.detach().float(),
        nan=0.0,
        posinf=logit_clip,
        neginf=-logit_clip,
    ).clamp(min=-logit_clip, max=logit_clip)

    student_logp = F.log_softmax(student, dim=-1)
    teacher_logp = F.log_softmax(teacher, dim=-1)
    student_prob = student_logp.exp()
    teacher_prob = teacher_logp.exp()
    teacher_entropy = -(
        teacher_prob * teacher_logp
    ).sum(dim=-1) / math.log(vocab_size)
    low_entropy_mask = teacher_entropy.detach().le(float(entropy_threshold))

    ce = F.cross_entropy(student, labels.long(), reduction="none")
    rkl_raw = (
        student_prob * (student_logp - teacher_logp)
    ).sum(dim=-1)
    rkl_used = rkl_raw.clamp(min=0.0, max=float(rkl_cap))
    if cad_mode:
        fkl = (
            teacher_prob * (teacher_logp - student_logp)
        ).sum(dim=-1)
        cj = (
            teacher_prob.detach().max(dim=-1).values
            ** float(cj_power)
        )
        low_entropy_loss = float(fkl_coef) * fkl + cj * rkl_used
    else:
        low_entropy_loss = rkl_used
    loss = torch.where(low_entropy_mask, low_entropy_loss, ce)

    if not bool(torch.isfinite(loss).all()):
        raise FloatingPointError("entropy-gated CE/RKL produced non-finite loss")
    terms = {
        "loss": loss,
        "ce": ce,
        "rkl_raw": rkl_raw,
        "rkl_used": rkl_used,
        "teacher_entropy": teacher_entropy.detach(),
        "low_entropy_mask": low_entropy_mask,
        "high_entropy_mask": ~low_entropy_mask,
        "rkl_capped_mask": low_entropy_mask & rkl_raw.detach().gt(float(rkl_cap)),
    }
    if cad_mode:
        terms["fkl"] = fkl
        terms["cj"] = cj
    return terms


# Hybrid-loss accounting slots: 0 high-entropy CE sum, 1 high-entropy count,
# 2 raw RKL sum (low), 3 capped RKL sum (low), 4 low-entropy count, 5 capped
# count, 6 per-token loss sum, 7 FKL sum (low, cad_mode), 8 c_j sum (low,
# cad_mode), 9 weighted loss sum, 10 weight sum (w=1 when unweighted so both
# regimes share one schema).
ENTROPY_GATED_RKL_REPORT_SLOTS = 11


def accumulate_entropy_gated_rkl_report_(
    report: Tensor,
    terms: Mapping[str, Tensor],
    *,
    cad_mode: bool,
    token_weights: Tensor | None = None,
) -> Tensor:
    """Accumulate one chunk of hybrid CE/RKL accounting into ``report``."""

    if report.ndim != 1 or int(report.shape[0]) < ENTROPY_GATED_RKL_REPORT_SLOTS:
        raise ValueError(
            "entropy_gated_rkl report must have at least "
            f"{ENTROPY_GATED_RKL_REPORT_SLOTS} slots, got {tuple(report.shape)}"
        )
    loss = terms["loss"]
    low = terms["low_entropy_mask"]
    high = terms["high_entropy_mask"]
    report[0] += terms["ce"][high].detach().double().sum()
    report[1] += float(high.sum().item())
    report[2] += terms["rkl_raw"][low].detach().double().sum()
    report[3] += terms["rkl_used"][low].detach().double().sum()
    report[4] += float(low.sum().item())
    report[5] += float(terms["rkl_capped_mask"].sum().item())
    report[6] += loss.detach().double().sum()
    if cad_mode:
        report[7] += terms["fkl"][low].detach().double().sum()
        report[8] += terms["cj"][low].detach().double().sum()
    if token_weights is None:
        report[9] += loss.detach().double().sum()
        report[10] += float(loss.numel())
    else:
        weights = token_weights.detach()
        if weights.shape != loss.shape:
            raise ValueError(
                "entropy_gated_rkl token weights must match the per-token "
                f"loss shape: {tuple(weights.shape)} != {tuple(loss.shape)}"
            )
        report[9] += (loss.detach().double() * weights.double()).sum()
        report[10] += weights.double().sum()
    return report


def entropy_gated_rkl_parity_terms(
    report: Tensor,
    *,
    cad_mode: bool,
    weighted: bool,
) -> dict[str, float]:
    """Recompose parity quantities from a globally reduced report.

    Unweighted non-cad: numerator = high-CE sum + capped-RKL sum over the
    token count.  Unweighted cad: the low branch is fkl_coef*FKL + c_j*RKL,
    so the recomposition is the per-token loss sum itself.  Weighted
    (response_level_ce): numerator/denominator are the weighted loss sum and
    weight sum, matching the trained objective in all modes including cad.
    """

    if report.ndim != 1 or int(report.shape[0]) < ENTROPY_GATED_RKL_REPORT_SLOTS:
        raise ValueError(
            "entropy_gated_rkl report must have at least "
            f"{ENTROPY_GATED_RKL_REPORT_SLOTS} slots, got {tuple(report.shape)}"
        )
    values = report.detach().cpu().to(torch.float64)
    ce_sum = float(values[0])
    high_count = float(values[1])
    rkl_used_sum = float(values[3])
    low_count = float(values[4])
    loss_sum = float(values[6])
    weighted_loss_sum = float(values[9])
    weight_sum = float(values[10])
    total_count = high_count + low_count
    if weighted:
        numerator = weighted_loss_sum
        denominator = weight_sum
    elif cad_mode:
        numerator = loss_sum
        denominator = total_count
    else:
        numerator = ce_sum + rkl_used_sum
        denominator = total_count
    return {
        "total_count": total_count,
        "numerator": numerator,
        "denominator": denominator,
        "loss": numerator / denominator if denominator > 0.0 else 0.0,
        "weighted_loss_sum": weighted_loss_sum,
        "weight_sum": weight_sum,
    }


def entropy_gated_rkl_blocks_response_level_ce(
    *,
    entropy_gated_rkl_enable: bool,
    allow_aec_composition: bool,
) -> bool:
    """response_level_ce guard term: egrkl excluded unless opted in.

    allow_aec_composition unlocks BOTH directions: the egrkl guard's AEC term
    and this response_level_ce guard term (response-level AEC requires
    response_level_ce, so composing with it implies composing with the
    response-weighted CE denominator).
    """

    return bool(entropy_gated_rkl_enable) and not bool(allow_aec_composition)


def validate_entropy_gated_rkl_composition(
    *,
    adj_ent_enable: bool,
    allow_aec_composition: bool,
    negative_action_pg_update: bool,
    signed_trace_ce_update: bool,
    causal_wrong_ce_update: bool,
    block_tpf_enable: bool,
    response_tpf_ce_scale_enable: bool,
    selected_vocab_size: int,
    entropy_coef: float,
) -> None:
    """Fail fast on objectives that cannot compose with entropy_gated_rkl.

    AEC is value-decoupled from the hybrid loss (its hinge consumes only
    student-entropy sums, pair weights, and count/weight denominators, never
    the per-token CE/RKL values), so token-level AEC may be explicitly opted
    in via ``allow_aec_composition``.  Response-level AEC additionally
    requires response_level_ce, which stays excluded by its own guard.
    """

    incompatible = (
        (bool(adj_ent_enable) and not bool(allow_aec_composition))
        or bool(negative_action_pg_update)
        or bool(signed_trace_ce_update)
        or bool(causal_wrong_ce_update)
        or bool(block_tpf_enable)
        or bool(response_tpf_ce_scale_enable)
        or int(selected_vocab_size) != 0
        or float(entropy_coef) != 0.0
    )
    if incompatible:
        raise ValueError(
            "entropy_gated_rkl requires plain full-vocab trace-all-CE "
            "without AEC, signed/wrong objectives, or CE weighting"
        )


def rkl_histogram_edges(*, device: torch.device | None = None) -> Tensor:
    """Return fixed, high-resolution RKL histogram upper edges."""

    log_edges = torch.logspace(
        -6,
        2,
        steps=161,
        dtype=torch.float64,
        device=device,
    )
    anchors = torch.tensor(
        [1e-8, 1e-7, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.3, 0.5,
         1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0],
        dtype=torch.float64,
        device=device,
    )
    return torch.unique(torch.cat((anchors, log_edges))).sort().values


def compute_entropy_gated_rkl_probe_stats(
    student_logits: Tensor,
    teacher_logits: Tensor,
    *,
    eligible_count: int | None = None,
    entropy_thresholds: Sequence[float] = DEFAULT_ENTROPY_THRESHOLDS,
    logit_clip: float = 50.0,
) -> dict[str, Tensor]:
    """Compute detached token-level entropy and reverse-KL statistics.

    Entropy is ``H(q_teacher) / log(vocab_size)`` and reverse KL is
    ``KL(p_student || q_teacher)``.  No label, acceptance, or denoised-state
    mask participates in routing.
    """

    if student_logits.ndim != 2 or teacher_logits.ndim != 2:
        raise ValueError("student_logits and teacher_logits must be [tokens, vocab]")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            "student/teacher shape mismatch: "
            f"{tuple(student_logits.shape)} != {tuple(teacher_logits.shape)}"
        )
    if student_logits.shape[-1] < 2:
        raise ValueError("vocab size must be at least 2")
    thresholds = tuple(float(value) for value in entropy_thresholds)
    if not thresholds:
        raise ValueError("at least one entropy threshold is required")
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in thresholds):
        raise ValueError(f"entropy thresholds must be finite values in [0,1], got {thresholds}")
    if logit_clip <= 0.0 or not math.isfinite(logit_clip):
        raise ValueError(f"logit_clip must be finite and positive, got {logit_clip}")

    token_count = int(student_logits.shape[0])
    eligible = token_count if eligible_count is None else int(eligible_count)
    if eligible < token_count:
        raise ValueError(
            f"eligible_count={eligible} cannot be smaller than source token count={token_count}"
        )
    device = student_logits.device
    threshold_tensor = torch.tensor(thresholds, dtype=torch.float64, device=device)
    edges = rkl_histogram_edges(device=device)

    with torch.no_grad():
        student = torch.nan_to_num(
            student_logits.detach().float(),
            nan=0.0,
            posinf=logit_clip,
            neginf=-logit_clip,
        ).clamp(min=-logit_clip, max=logit_clip)
        teacher = torch.nan_to_num(
            teacher_logits.detach().float(),
            nan=0.0,
            posinf=logit_clip,
            neginf=-logit_clip,
        ).clamp(min=-logit_clip, max=logit_clip)

        student_logp = F.log_softmax(student, dim=-1)
        teacher_logp = F.log_softmax(teacher, dim=-1)
        student_prob = student_logp.exp()
        teacher_prob = teacher_logp.exp()
        entropy = -(teacher_prob * teacher_logp).sum(dim=-1)
        entropy = entropy / math.log(int(student_logits.shape[-1]))
        rkl = (student_prob * (student_logp - teacher_logp)).sum(dim=-1)

        finite = torch.isfinite(entropy) & torch.isfinite(rkl)
        negative_rkl = finite & rkl.lt(-1e-6)
        safe_entropy = entropy[finite].to(torch.float64)
        safe_rkl = rkl[finite].clamp_min(0.0).to(torch.float64)
        top1_mismatch = (
            student.argmax(dim=-1).ne(teacher.argmax(dim=-1)) & finite
        )
        safe_top1_mismatch = top1_mismatch[finite].to(torch.float64)

        base = torch.zeros(_BASE_SIZE, dtype=torch.float64, device=device)
        base[_BASE_ELIGIBLE] = float(eligible)
        base[_BASE_SOURCE] = float(token_count)
        base[_BASE_MISSING_SOURCE] = float(eligible - token_count)
        base[_BASE_FINITE] = float(finite.sum().item())
        base[_BASE_NONFINITE] = float((~finite).sum().item())
        base[_BASE_ENTROPY_SUM] = safe_entropy.sum()
        base[_BASE_RKL_SUM] = safe_rkl.sum()
        base[_BASE_TOP1_MISMATCH] = safe_top1_mismatch.sum()
        base[_BASE_NEGATIVE_RKL] = float(negative_rkl.sum().item())

        threshold_count = torch.zeros(
            len(thresholds), dtype=torch.float64, device=device
        )
        threshold_rkl_sum = torch.zeros_like(threshold_count)
        threshold_rkl_sq_sum = torch.zeros_like(threshold_count)
        threshold_top1_mismatch = torch.zeros_like(threshold_count)
        threshold_rkl_max = torch.zeros_like(threshold_count)
        threshold_rkl_hist = torch.zeros(
            (len(thresholds), int(edges.numel()) + 1),
            dtype=torch.float64,
            device=device,
        )
        for index, threshold in enumerate(threshold_tensor):
            low = safe_entropy.le(threshold)
            low_rkl = safe_rkl[low]
            threshold_count[index] = float(low_rkl.numel())
            if low_rkl.numel() == 0:
                continue
            threshold_rkl_sum[index] = low_rkl.sum()
            threshold_rkl_sq_sum[index] = low_rkl.square().sum()
            threshold_top1_mismatch[index] = safe_top1_mismatch[low].sum()
            threshold_rkl_max[index] = low_rkl.max()
            bins = torch.bucketize(low_rkl, edges)
            threshold_rkl_hist[index] = torch.bincount(
                bins,
                minlength=int(edges.numel()) + 1,
            ).to(torch.float64)

    return {
        "base": base,
        "threshold_count": threshold_count,
        "threshold_rkl_sum": threshold_rkl_sum,
        "threshold_rkl_sq_sum": threshold_rkl_sq_sum,
        "threshold_top1_mismatch": threshold_top1_mismatch,
        "threshold_rkl_max": threshold_rkl_max,
        "threshold_rkl_hist": threshold_rkl_hist,
    }


def merge_entropy_gated_rkl_probe_stats_(
    destination: dict[str, Tensor] | None,
    source: Mapping[str, Tensor],
) -> dict[str, Tensor]:
    """Merge one micro-batch probe into an update-local accumulator."""

    if destination is None:
        return {key: value.detach().clone() for key, value in source.items()}
    for key, value in source.items():
        if key not in destination:
            raise KeyError(f"probe destination is missing key {key!r}")
        if destination[key].shape != value.shape:
            raise ValueError(
                f"probe shape mismatch for {key}: "
                f"{tuple(destination[key].shape)} != {tuple(value.shape)}"
            )
        if key == "threshold_rkl_max":
            destination[key] = torch.maximum(destination[key], value.detach())
        else:
            destination[key].add_(value.detach())
    return destination


def all_reduce_entropy_gated_rkl_probe_stats_(
    stats: dict[str, Tensor],
) -> dict[str, Tensor]:
    """Reduce additive statistics across the actor world."""

    if not torch.distributed.is_initialized():
        return stats
    for key, value in stats.items():
        op = (
            torch.distributed.ReduceOp.MAX
            if key == "threshold_rkl_max"
            else torch.distributed.ReduceOp.SUM
        )
        torch.distributed.all_reduce(value, op=op)
    return stats


def _histogram_quantile(hist: Tensor, edges: Tensor, quantile: float) -> float:
    count = float(hist.sum().item())
    if count <= 0.0:
        return 0.0
    target = quantile * count
    index = int(torch.searchsorted(hist.cumsum(dim=0), target).item())
    if index < int(edges.numel()):
        return float(edges[index].item())
    return float("inf")


def entropy_gated_rkl_probe_metrics(
    stats: Mapping[str, Tensor],
    *,
    entropy_thresholds: Sequence[float] = DEFAULT_ENTROPY_THRESHOLDS,
    prefix: str = "entropy_gate_rkl_probe",
) -> dict[str, float]:
    """Convert globally reduced probe tensors into scalar metrics."""

    base = stats["base"].detach().cpu().to(torch.float64)
    counts = stats["threshold_count"].detach().cpu().to(torch.float64)
    sums = stats["threshold_rkl_sum"].detach().cpu().to(torch.float64)
    sq_sums = stats["threshold_rkl_sq_sum"].detach().cpu().to(torch.float64)
    mismatches = stats["threshold_top1_mismatch"].detach().cpu().to(torch.float64)
    maxima = stats["threshold_rkl_max"].detach().cpu().to(torch.float64)
    histograms = stats["threshold_rkl_hist"].detach().cpu().to(torch.float64)
    edges = rkl_histogram_edges(device=torch.device("cpu"))

    source = float(base[_BASE_SOURCE].item())
    finite = float(base[_BASE_FINITE].item())
    metrics = {
        f"{prefix}/eligible_tokens": float(base[_BASE_ELIGIBLE].item()),
        f"{prefix}/teacher_source_tokens": source,
        f"{prefix}/missing_teacher_source_tokens": float(
            base[_BASE_MISSING_SOURCE].item()
        ),
        f"{prefix}/finite_tokens": finite,
        f"{prefix}/nonfinite_tokens": float(base[_BASE_NONFINITE].item()),
        f"{prefix}/teacher_entropy_mean": (
            float(base[_BASE_ENTROPY_SUM].item()) / max(finite, 1.0)
        ),
        f"{prefix}/rkl_all_mean": (
            float(base[_BASE_RKL_SUM].item()) / max(finite, 1.0)
        ),
        f"{prefix}/top1_mismatch_all_frac": (
            float(base[_BASE_TOP1_MISMATCH].item()) / max(finite, 1.0)
        ),
        f"{prefix}/negative_rkl_tokens": float(base[_BASE_NEGATIVE_RKL].item()),
    }
    for index, threshold in enumerate(entropy_thresholds):
        count = float(counts[index].item())
        mean = float(sums[index].item()) / max(count, 1.0)
        variance = max(
            float(sq_sums[index].item()) / max(count, 1.0) - mean * mean,
            0.0,
        )
        label = f"{float(threshold):.3f}".replace(".", "p")
        hist = histograms[index]
        metrics.update(
            {
                f"{prefix}/threshold_{label}": float(threshold),
                f"{prefix}/threshold_{label}_raw_nats": (
                    float(threshold) * math.log(151936)
                ),
                f"{prefix}/threshold_{label}_rkl_tokens": count,
                f"{prefix}/threshold_{label}_rkl_frac": count / max(finite, 1.0),
                f"{prefix}/threshold_{label}_rkl_mean": mean,
                f"{prefix}/threshold_{label}_rkl_std": math.sqrt(variance),
                f"{prefix}/threshold_{label}_rkl_p50": _histogram_quantile(
                    hist, edges, 0.50
                ),
                f"{prefix}/threshold_{label}_rkl_p90": _histogram_quantile(
                    hist, edges, 0.90
                ),
                f"{prefix}/threshold_{label}_rkl_p95": _histogram_quantile(
                    hist, edges, 0.95
                ),
                f"{prefix}/threshold_{label}_rkl_p99": _histogram_quantile(
                    hist, edges, 0.99
                ),
                f"{prefix}/threshold_{label}_rkl_max": float(
                    maxima[index].item()
                ),
                f"{prefix}/threshold_{label}_top1_mismatch_frac": (
                    float(mismatches[index].item()) / max(count, 1.0)
                ),
            }
        )
        for cutoff in (1.0, 2.0, 5.0, 10.0, 20.0):
            first_above = int(torch.searchsorted(edges, cutoff, right=True).item())
            above = float(hist[first_above:].sum().item())
            cutoff_label = str(int(cutoff))
            metrics[
                f"{prefix}/threshold_{label}_rkl_gt_{cutoff_label}_frac"
            ] = above / max(count, 1.0)
    return metrics


__all__ = [
    "DEFAULT_ENTROPY_THRESHOLDS",
    "ENTROPY_GATED_RKL_REPORT_SLOTS",
    "accumulate_entropy_gated_rkl_report_",
    "all_reduce_entropy_gated_rkl_probe_stats_",
    "entropy_gated_rkl_blocks_response_level_ce",
    "entropy_gated_rkl_parity_terms",
    "compute_entropy_gated_ce_rkl_terms",
    "compute_entropy_gated_rkl_probe_stats",
    "entropy_gated_rkl_probe_metrics",
    "merge_entropy_gated_rkl_probe_stats_",
    "rkl_histogram_edges",
    "validate_entropy_gated_rkl_composition",
]
