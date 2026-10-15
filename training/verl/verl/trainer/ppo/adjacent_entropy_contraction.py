"""Adjacent Average Entropy Contraction objective.

Regularizes the actor by penalizing insufficient entropy drop between
consecutive denoising steps (t → t+1) within one response.

For each unique valid current step_key k with non-empty M_next mask:
    H_next(k) = mean normalized entropy of the current actor distribution
                over positions where adjacent_entropy_pair_mask == True and
                step_maps == k
    H_prev(k) = producer-captured rollout mean entropy (stop-grad scalar)
    pair_loss(k) = relu(H_next(k) - (1-rho) * H_prev(k).detach())^2

Global objective (equal per-pair weighting):
    L = lambda_H * sum_k pair_loss(k) / N_pair_global

Entropy is computed as:
    H(logits) = -sum_v softmax(logits)_v * log_softmax(logits)_v / log(vocab_size)

Performance contract:
  - H_next is computed from actor hidden states projected through lm_head in
    bounded chunks; the full [n_tokens, vocab_size] tensor is NOT materialized
    for all tokens simultaneously.
  - H_prev is a detached scalar per step_key; no gradient flows through it.

Integration contract with trace_all_ce global-denominator-scale path:
  - The forward pass returns (hinge_sum, pair_count) without normalization.
  - update_policy applies scaling so the final gradient contribution equals
    exactly lambda_H / N_pair_global per pair, matching the CE normalization.
  - Ghost micro-batches receive a graph-connected zero to preserve FSDP/SP
    graph structure.
  - Fail-closed unless objective=trace_all_ce, mode=replace, ppo_epochs=1,
    stream global denominator path active, payload complete, and denominators
    consistent.

Feature config location: generation_consistency_jsd.adjacent_entropy_contraction
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


__all__ = [
    "AdjacEntConfig",
    "compute_adj_entropy_hinge_reference",
    "compute_adj_entropy_hinge_hidden",
    "count_unique_valid_pairs_from_batch",
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AdjacEntConfig:
    """Configuration for adjacent average entropy contraction."""

    enable: bool = False
    rho: float = 0.05
    coef: float = 0.01
    projection_chunk_size: int = 512
    eps: float = 1e-8
    response_level: bool = False
    # scope: "unresolved" (default) | "all_valid" / "global"
    # "unresolved": H over unresolved (not-yet-accepted) positions (original behavior)
    # "all_valid":  H over all valid non-prompt response positions (global AEC)
    scope: str = "unresolved"
    # ceiling: pairs whose (detached) H_prev exceeds this normalized-entropy value are
    # exempted from the contraction hinge entirely. Rationale:
    # positions that fail to contract are exactly where the model has legitimate
    # uncertainty; penalizing them converts accuracy into apparent confidence.
    # 0.0 sentinel = disabled (original behavior).
    # ceiling gate semantics: H_prev > ceiling → exempt; H_prev == ceiling → participates.
    ceiling: float = 0.0
    # floor: pairs whose (detached) H_prev is strictly below this normalized-entropy value
    # are exempted from the contraction hinge. Rationale: positions that are already
    # very confident need no additional contraction pressure.
    # 0.0 sentinel = disabled (original behavior).
    # floor gate semantics: H_prev < floor → exempt; H_prev == floor → participates.
    # floor and ceiling are mutually exclusive: both > 0 raises ValueError.
    floor: float = 0.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "AdjacEntConfig":
        values = values or {}
        if not isinstance(values, Mapping):
            raise ValueError("adjacent_entropy_contraction must be a mapping")
        raw_scope = str(values.get("scope", "unresolved")).strip().lower()
        # Accept "global" as alias for "all_valid"
        if raw_scope == "global":
            raw_scope = "all_valid"
        return cls(
            enable=bool(values.get("enable", False)),
            rho=float(values.get("rho", 0.05)),
            coef=float(values.get("coef", 0.01)),
            projection_chunk_size=int(values.get("projection_chunk_size", 512) or 512),
            eps=float(values.get("eps", 1e-8)),
            response_level=bool(values.get("response_level", False)),
            scope=raw_scope,
            ceiling=float(values.get("ceiling", 0.0) or 0.0),
            floor=float(values.get("floor", 0.0) or 0.0),
        )

    def validate(self) -> None:
        if not (0.0 <= self.rho <= 1.0):
            raise ValueError(f"adjacent_entropy_contraction.rho must be in [0,1], got {self.rho}")
        if self.coef < 0.0:
            raise ValueError(f"adjacent_entropy_contraction.coef must be >= 0, got {self.coef}")
        if self.projection_chunk_size < 1:
            raise ValueError(
                f"adjacent_entropy_contraction.projection_chunk_size must be >= 1, "
                f"got {self.projection_chunk_size}"
            )
        if self.scope not in {"unresolved", "all_valid"}:
            raise ValueError(
                f"adjacent_entropy_contraction.scope must be 'unresolved' or 'all_valid', "
                f"got {self.scope!r}"
            )
        if self.ceiling < 0.0:
            raise ValueError(
                f"adjacent_entropy_contraction.ceiling must be >= 0, got {self.ceiling}"
            )
        if self.floor < 0.0:
            raise ValueError(
                f"adjacent_entropy_contraction.floor must be >= 0, got {self.floor}"
            )
        if self.floor > 0.0 and self.ceiling > 0.0:
            raise ValueError(
                f"adjacent_entropy_contraction.floor and ceiling are mutually exclusive; "
                f"both are enabled (floor={self.floor}, ceiling={self.ceiling}). "
                "Set exactly one to a positive value or both to 0.0 to disable."
            )


# ---------------------------------------------------------------------------
# Entropy helpers
# ---------------------------------------------------------------------------

def _normalized_entropy(logits: Tensor, vocab_size: int) -> Tensor:
    """Compute per-token entropy normalized by log(vocab_size).

    Args:
        logits: [N, V] float32 unnormalized logits
        vocab_size: vocabulary size V (used for normalization denominator)

    Returns:
        [N] float32 normalized entropy in [0, 1]
    """
    log_norm = math.log(max(vocab_size, 2))
    log_p = F.log_softmax(logits, dim=-1)
    p = log_p.exp()
    raw_h = -(p * log_p).sum(dim=-1)
    return raw_h / log_norm


# ---------------------------------------------------------------------------
# Reference helper (accepts full logits_t / logits_next tensors)
# ---------------------------------------------------------------------------

def compute_adj_entropy_hinge_reference(
    logits_t: Tensor,
    logits_next: Tensor,
    mask_t: Tensor,
    mask_next: Tensor,
    rho: float,
    vocab_size: Optional[int] = None,
    ceiling: float = 0.0,
    floor: float = 0.0,
) -> Tuple[Tensor, Tensor, Dict[str, float]]:
    """Reference implementation accepting pre-computed logit tensors.

    This function is for testing and validation only. Production code uses
    compute_adj_entropy_hinge_hidden() which avoids materializing full logits.

    Both transitions (t and t+1) correspond to the SAME logical step index k;
    mask_t selects the H_t positions and mask_next selects the H_next positions.

    Entropy gate semantics (applied to H_prev = mean entropy of mask_t positions):
    - floor > 0: exempt this pair if H_prev < floor (strict); equality participates.
    - ceiling > 0: exempt this pair if H_prev > ceiling (strict); equality participates.
    - floor and ceiling must not both be > 0 (caller must validate via AdjacEntConfig).
    - When exempt, hinge = 0, pair_count = 0; AEC contributes zero safely.
    - Normalization is over eligible (non-exempt) pairs only.

    Args:
        logits_t:   [N, V] float32 – logits for the previous state (t)
        logits_next: [N, V] float32 – logits for the current state (t+1)
        mask_t:     [N] bool – which positions belong to H_t computation
        mask_next:  [N] bool – which positions belong to H_next computation
        rho:        target relative entropy drop (e.g. 0.1 → 10% drop required)
        vocab_size: optional override; inferred from logits_next.shape[-1] if None
        ceiling:    entropy ceiling gate threshold (0.0 = disabled)
        floor:      entropy floor gate threshold (0.0 = disabled)

    Returns:
        (hinge_scalar, pair_count_scalar, metrics_dict)
        - hinge_scalar: differentiable scalar (grad through logits_next only)
        - pair_count_scalar: integer tensor (1 if eligible, else 0)
        - metrics_dict with entropy, hinge stats, and gate telemetry
    """
    V = vocab_size if vocab_size is not None else int(logits_next.shape[-1])
    graph_zero = logits_next.float().sum() * 0.0

    _empty_metrics: Dict[str, float] = {
        "adj_ent/avg_entropy_t": 0.0,
        "adj_ent/avg_entropy_next": 0.0,
        "adj_ent/relative_entropy_drop": 0.0,
        "adj_ent/violation_rate": 0.0,
        "adj_ent/valid_pair_count": 0.0,
        "adj_ent/hinge_sum": 0.0,
        "adj_ent/empty_mask": 1.0,
        "adj_ent/total_pairs": 0.0,
        "adj_ent/eligible_pairs": 0.0,
        "adj_ent/floor_exempt": 0.0,
        "adj_ent/ceiling_exempt": 0.0,
        "adj_ent/eligible_fraction": 0.0,
        "adj_ent/eligible_pair_fraction": 0.0,
        "adj_ent/eligible_hinge_violations": 0.0,
    }

    if not mask_t.any() or not mask_next.any():
        return graph_zero, torch.tensor(0, dtype=torch.long), _empty_metrics

    h_t = _normalized_entropy(logits_t[mask_t].float(), V)
    h_prev_mean = h_t.mean().detach()
    h_prev_val = float(h_prev_mean.item())

    # Entropy gate: determine if this single pair is eligible
    floor_exempt = (floor > 0.0 and h_prev_val < floor)
    ceiling_exempt = (ceiling > 0.0 and h_prev_val > ceiling)

    gate_telemetry: Dict[str, float] = {
        "adj_ent/total_pairs": 1.0,
        "adj_ent/floor_exempt": 1.0 if floor_exempt else 0.0,
        "adj_ent/ceiling_exempt": 1.0 if ceiling_exempt else 0.0,
    }

    if floor_exempt or ceiling_exempt:
        gate_telemetry["adj_ent/eligible_pairs"] = 0.0
        gate_telemetry["adj_ent/eligible_fraction"] = 0.0
        gate_telemetry["adj_ent/eligible_pair_fraction"] = 0.0
        gate_telemetry["adj_ent/eligible_hinge_violations"] = 0.0
        exempt_metrics = {**_empty_metrics, **gate_telemetry}
        exempt_metrics["adj_ent/avg_entropy_t"] = h_prev_val
        exempt_metrics["adj_ent/empty_mask"] = 0.0
        return graph_zero, torch.tensor(0, dtype=torch.long), exempt_metrics

    h_next = _normalized_entropy(logits_next[mask_next].float(), V)
    h_next_mean = h_next.mean()

    threshold = (1.0 - rho) * h_prev_mean
    excess = h_next_mean - threshold
    hinge = torch.relu(excess) ** 2

    violated = float((excess > 0).float().item())
    rel_drop = float(
        ((h_prev_mean - h_next_mean.detach()) / (h_prev_mean + 1e-8)).item()
    )

    metrics: Dict[str, float] = {
        "adj_ent/avg_entropy_t": h_prev_val,
        "adj_ent/avg_entropy_next": float(h_next_mean.detach().item()),
        "adj_ent/relative_entropy_drop": rel_drop,
        "adj_ent/violation_rate": violated,
        "adj_ent/valid_pair_count": 1.0,
        "adj_ent/hinge_sum": float(hinge.detach().item()),
        "adj_ent/empty_mask": 0.0,
        "adj_ent/total_pairs": 1.0,
        "adj_ent/eligible_pairs": 1.0,
        "adj_ent/floor_exempt": 0.0,
        "adj_ent/ceiling_exempt": 0.0,
        "adj_ent/eligible_fraction": 1.0,
        "adj_ent/eligible_pair_fraction": 1.0,
        "adj_ent/eligible_hinge_violations": violated,
    }

    return hinge, torch.tensor(1, dtype=torch.long), metrics


# ---------------------------------------------------------------------------
# Production helper (hidden-source projection)
# ---------------------------------------------------------------------------

def compute_adj_entropy_hinge_hidden(
    actor_hidden_rmpad: Tensor,
    lm_head: nn.Linear,
    pair_mask_rmpad: Tensor,
    prev_mean_rmpad: Tensor,
    pair_ids_rmpad: Tensor,
    rho: float,
    vocab_size: int,
    chunk_size: int = 512,
    eps: float = 1e-8,
    ceiling: float = 0.0,
    floor: float = 0.0,
) -> Tuple[Tensor, int, Dict[str, float]]:
    """Production entropy hinge using hidden-source projection.

    Only M_next hidden states are projected through lm_head (in bounded chunks),
    avoiding full [n_tokens, vocab_size] materialization.

    Entropy gate semantics (applied per pair to its detached H_prev):
    - floor > 0: exempt pair if H_prev < floor (strict); equality participates.
    - ceiling > 0: exempt pair if H_prev > ceiling (strict); equality participates.
    - floor and ceiling must not both be > 0 (caller must validate via AdjacEntConfig).
    - Exempt pairs contribute zero to hinge_sum and are excluded from pair_count.
    - Normalization is over eligible (non-exempt) pairs only.
    - If no eligible pairs remain, returns graph-connected zero safely.

    Telemetry keys added to metrics_dict:
    - adj_ent/total_pairs: all valid pre-gate pairs seen
    - adj_ent/eligible_pairs: pairs that passed the gate
    - adj_ent/floor_exempt: pairs exempted by floor gate
    - adj_ent/ceiling_exempt: pairs exempted by ceiling gate
    - adj_ent/eligible_fraction: eligible / total (0.0 if no pairs)
    - adj_ent/eligible_pair_fraction: eligible / total (count ratio; renamed from eligible_weight_mass
      to avoid confusion with the weight-mass metric emitted by dp_actor's response-level AEC path)
    - adj_ent/eligible_hinge_violations: hinge violations among eligible pairs

    Args:
        actor_hidden_rmpad: [n_tokens, hidden_size] float (actor hidden states,
            fast_hidden_ce mode output with remove_padding applied)
        lm_head: actor language model head
        pair_mask_rmpad: [n_tokens] bool – M_next positions
        prev_mean_rmpad: [n_tokens] float32 – H_t broadcast (constant per page)
        pair_ids_rmpad: [n_tokens] int64 – packed-batch-unique pair ID
        rho: target relative entropy drop
        vocab_size: vocabulary size for entropy normalization
        chunk_size: projection chunk size to bound peak memory
        eps: small constant for numerical stability
        ceiling: entropy ceiling gate threshold (0.0 = disabled)
        floor: entropy floor gate threshold (0.0 = disabled)

    Returns:
        (hinge_sum_differentiable, eligible_pair_count_int, metrics_dict)
    """
    graph_zero = actor_hidden_rmpad.float().sum() * 0.0

    _empty_metrics: Dict[str, float] = {
        "adj_ent/avg_entropy_t": 0.0,
        "adj_ent/avg_entropy_next": 0.0,
        "adj_ent/relative_entropy_drop": 0.0,
        "adj_ent/violation_rate": 0.0,
        "adj_ent/valid_pair_count": 0.0,
        "adj_ent/hinge_sum": 0.0,
        "adj_ent/empty_mask": 1.0,
        "adj_ent/gapped_counters": 0.0,
        "adj_ent/total_pairs": 0.0,
        "adj_ent/eligible_pairs": 0.0,
        "adj_ent/floor_exempt": 0.0,
        "adj_ent/ceiling_exempt": 0.0,
        "adj_ent/eligible_fraction": 0.0,
        "adj_ent/eligible_pair_fraction": 0.0,
        "adj_ent/eligible_hinge_violations": 0.0,
    }

    next_positions = torch.nonzero(pair_mask_rmpad, as_tuple=False).flatten()
    if next_positions.numel() == 0:
        return graph_zero, 0, dict(_empty_metrics)

    pair_ids_at_next = pair_ids_rmpad[next_positions]
    valid_pos_mask = pair_ids_at_next.ge(0)
    if not valid_pos_mask.any():
        return graph_zero, 0, dict(_empty_metrics)

    valid_next_positions = next_positions[valid_pos_mask]
    valid_pair_ids = pair_ids_at_next[valid_pos_mask]
    unique_pair_ids = torch.unique(valid_pair_ids)

    log_norm = math.log(max(vocab_size, 2))

    hinge_sum: Optional[Tensor] = None
    pair_count = 0
    total_pairs = 0
    floor_exempt_count = 0
    ceiling_exempt_count = 0
    h_t_values = []
    h_next_values = []
    gapped = 0

    for key_val in unique_pair_ids.tolist():
        key_val = int(key_val)
        key_mask = valid_pair_ids.eq(key_val)
        key_positions = valid_next_positions[key_mask]

        if key_positions.numel() == 0:
            gapped += 1
            continue

        h_prev_values = prev_mean_rmpad[key_positions].detach().float()
        if not bool(torch.isfinite(h_prev_values).all()):
            raise RuntimeError(
                f"adjacent entropy pair {key_val} has non-finite previous entropy"
            )
        h_prev_k = h_prev_values[0]
        if not bool(
            torch.allclose(
                h_prev_values,
                h_prev_k.expand_as(h_prev_values),
                atol=1e-6,
                rtol=1e-5,
            )
        ):
            raise RuntimeError(
                f"adjacent entropy pair {key_val} has inconsistent previous means"
            )

        total_pairs += 1
        h_prev_scalar = float(h_prev_k.item())

        # Entropy gate: exempt pairs outside the eligible window.
        # Equality with the threshold participates (strict comparison for exemption).
        if floor > 0.0 and h_prev_scalar < floor:
            floor_exempt_count += 1
            hinge_sum = (
                hinge_sum
                if hinge_sum is not None
                else graph_zero  # keep graph connected
            )
            continue
        if ceiling > 0.0 and h_prev_scalar > ceiling:
            ceiling_exempt_count += 1
            hinge_sum = (
                hinge_sum
                if hinge_sum is not None
                else graph_zero
            )
            continue

        h_next_pieces = []
        h_source = actor_hidden_rmpad[key_positions]
        n = h_source.shape[0]
        for chunk_start in range(0, n, chunk_size):
            h_chunk = h_source[chunk_start : chunk_start + chunk_size].to(
                dtype=lm_head.weight.dtype
            )
            logits_chunk = F.linear(h_chunk, lm_head.weight, lm_head.bias).float()
            log_p_chunk = F.log_softmax(logits_chunk, dim=-1)
            p_chunk = log_p_chunk.exp()
            ent_chunk = -(p_chunk * log_p_chunk).sum(dim=-1) / log_norm
            h_next_pieces.append(ent_chunk)

        h_next_k = torch.cat(h_next_pieces, dim=0).mean()
        threshold = (1.0 - rho) * h_prev_k
        excess = h_next_k - threshold
        hinge_k = torch.relu(excess) ** 2

        hinge_sum = hinge_k if hinge_sum is None else hinge_sum + hinge_k
        pair_count += 1
        h_t_values.append(h_prev_scalar)
        h_next_values.append(float(h_next_k.detach().item()))

    eligible_fraction = (
        float(pair_count) / float(total_pairs) if total_pairs > 0 else 0.0
    )

    if hinge_sum is None:
        # No pairs at all (all gapped, or all exempt with no eligible)
        m = dict(_empty_metrics)
        m["adj_ent/gapped_counters"] = float(gapped)
        m["adj_ent/total_pairs"] = float(total_pairs)
        m["adj_ent/eligible_pairs"] = 0.0
        m["adj_ent/floor_exempt"] = float(floor_exempt_count)
        m["adj_ent/ceiling_exempt"] = float(ceiling_exempt_count)
        m["adj_ent/eligible_fraction"] = eligible_fraction
        m["adj_ent/eligible_pair_fraction"] = eligible_fraction
        return graph_zero, 0, m

    if pair_count == 0:
        # All pairs were exempt; graph_zero keeps gradient graph connected.
        m = dict(_empty_metrics)
        m["adj_ent/gapped_counters"] = float(gapped)
        m["adj_ent/total_pairs"] = float(total_pairs)
        m["adj_ent/eligible_pairs"] = 0.0
        m["adj_ent/floor_exempt"] = float(floor_exempt_count)
        m["adj_ent/ceiling_exempt"] = float(ceiling_exempt_count)
        m["adj_ent/eligible_fraction"] = eligible_fraction
        m["adj_ent/eligible_pair_fraction"] = eligible_fraction
        return hinge_sum, 0, m

    avg_h_t = sum(h_t_values) / len(h_t_values)
    avg_h_next = sum(h_next_values) / len(h_next_values)
    rel_drop = (avg_h_t - avg_h_next) / max(avg_h_t, eps)
    violations = sum(
        1 for ht, hn in zip(h_t_values, h_next_values)
        if hn > (1.0 - rho) * ht
    )
    violation_rate = violations / pair_count

    metrics: Dict[str, float] = {
        "adj_ent/avg_entropy_t": avg_h_t,
        "adj_ent/avg_entropy_next": avg_h_next,
        "adj_ent/relative_entropy_drop": rel_drop,
        "adj_ent/violation_rate": violation_rate,
        "adj_ent/valid_pair_count": float(pair_count),
        "adj_ent/hinge_sum": float(hinge_sum.detach().item()),
        "adj_ent/empty_mask": 0.0,
        "adj_ent/gapped_counters": float(gapped),
        "_adj_ent/h_t_sum": float(sum(h_t_values)),
        "_adj_ent/h_next_sum": float(sum(h_next_values)),
        "_adj_ent/violation_count": float(violations),
        "adj_ent/total_pairs": float(total_pairs),
        "adj_ent/eligible_pairs": float(pair_count),
        "adj_ent/floor_exempt": float(floor_exempt_count),
        "adj_ent/ceiling_exempt": float(ceiling_exempt_count),
        "adj_ent/eligible_fraction": eligible_fraction,
        "adj_ent/eligible_pair_fraction": eligible_fraction,
        "adj_ent/eligible_hinge_violations": float(violations),
    }

    return hinge_sum, pair_count, metrics


# ---------------------------------------------------------------------------
# Pre-pass counting helper (used in update_policy before backward loop)
# ---------------------------------------------------------------------------

def count_unique_valid_pairs_from_batch(
    adj_pair_mask: Optional[Tensor],
    pair_ids: Optional[Tensor],
    response_mask: Optional[Tensor],
    prev_mean: Optional[Tensor] = None,
    floor: float = 0.0,
    ceiling: float = 0.0,
) -> Tuple[int, int]:
    """Count eligible transition pairs and CE tokens for the pre-pass denominator.

    Applies the same floor/ceiling gate as the actual AEC forward path so that
    the prepass denominator counts only ELIGIBLE pairs.

    Args:
        adj_pair_mask: [B, L] bool or None – M_next positions (response region)
        pair_ids:      [B, L] int64 or None – packed-batch-unique pair ID
        response_mask: [B, L] bool or None – valid response positions
        prev_mean:     [B, L] float32 or None – transported detached H_prev;
                       required for gate to take effect; None = no gating
        floor:         entropy floor gate threshold (0.0 = disabled)
        ceiling:       entropy ceiling gate threshold (0.0 = disabled)

    Returns:
        (eligible_pair_count, ce_token_count) as Python ints
        eligible_pair_count: unique (batch_idx, pair_id) pairs that pass the
            floor/ceiling gate.  When gates are disabled (both 0.0), equals the
            total valid unique pair count (original behavior).
    """
    ce_count = 0
    if response_mask is not None:
        ce_count = int(response_mask.bool().sum().item())

    pair_count = 0
    if adj_pair_mask is None or pair_ids is None:
        return pair_count, ce_count

    mask = adj_pair_mask.bool()
    if response_mask is not None:
        resp = response_mask.bool()
        if resp.shape == mask.shape:
            mask = mask & resp

    gate_active = (floor > 0.0 or ceiling > 0.0) and prev_mean is not None
    batch_size = adj_pair_mask.shape[0]
    unique_per_batch = set()
    for b in range(batch_size):
        b_mask = mask[b]
        b_keys = pair_ids[b][b_mask]
        b_keys_valid = b_keys[b_keys.ge(0)]
        if b_keys_valid.numel() == 0:
            continue
        for k in b_keys_valid.unique().tolist():
            k = int(k)
            if gate_active:
                pos = b_mask & pair_ids[b].eq(k)
                h_prev_val = float(prev_mean[b][pos].float().mean().item())
                if floor > 0.0 and h_prev_val < floor:
                    continue  # floor-exempt (strict < )
                if ceiling > 0.0 and h_prev_val > ceiling:
                    continue  # ceiling-exempt (strict > )
            unique_per_batch.add((b, k))
    pair_count = len(unique_per_batch)

    return pair_count, ce_count
