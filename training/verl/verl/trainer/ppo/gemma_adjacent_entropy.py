"""Adjacent Average Entropy Contraction (AEC) for the DiffusionGemma exact-replay path.

Gemma4Renoise re-noises every non-accepted canvas position at every denoise step and the
acceptance mask is recomputed from scratch each step, so there is no permanent
resolved/unresolved partition (the replay's ``valid_masks`` are all-True and the CE is applied
to all 256 canvas positions of every state).  The natural AEC unit is therefore the *canvas*
of one block: for two consecutive captured states (step t -> t+1) of the same block

    H_prev = mean_{p in M} H_t[p]          (actor entropy at step t, stop-grad)
    H_next = mean_{p in M} H_{t+1}[p]      (actor entropy at step t+1, with grad)
    hinge  = relu(H_next - (1 - rho) * H_prev) ** 2

with H the per-position entropy normalised by log(vocab).  ``M`` is either every canvas
position (``scope=all_valid``, default) or the positions whose canvas token changed between
the two states (``scope=renoised``: the positions the sampler did *not* accept at step t,
i.e. the closest analogue of LLaDA2's "unresolved" scope).  Pairs are formed from step
indices that differ by exactly one **in chronological order**: Gemma4Renoise's captured
``step`` counts down from ``max_denoising_steps`` to 1, so the chronologically next state has
step index ``t - 1`` (``step_direction=-1``; the actor walks each block's rows in descending
step index).  A detached carry of the last row of the previous wave
gives the pair that straddles two decoder waves of the same block slot, so no pair is lost
to chunking.  Unlike the LLaDA2 path, H_prev is the actor's own (detached) entropy rather
than a rollout-captured value: the fork does not capture rollout logits/entropies.

Global objective, matching the CE token-mean normalisation of the replay:
    L_aec = coef * sum_pairs hinge / N_pairs_total
where ``N_pairs_total`` is counted on the trainer before dispatch
(:func:`count_gemma_adjacent_pairs`) and transported in ``meta_info``.

Feature config: ``actor_rollout_ref.actor.generation_consistency_jsd.gemma_adjacent_entropy_contraction``
(``enable, rho, coef, scope, floor, ceiling``); parsed with :class:`AdjacEntConfig`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from verl.trainer.ppo.adjacent_entropy_contraction import AdjacEntConfig

__all__ = [
    "GemmaAdjacEntConfig",
    "GemmaAecState",
    "count_gemma_adjacent_pairs",
    "gemma_aec_wave_terms",
    "gemma_aec_metrics",
]

_SCOPES = ("all_valid", "renoised")


@dataclass(frozen=True)
class GemmaAdjacEntConfig:
    enable: bool = False
    rho: float = 0.2
    coef: float = 50.0
    scope: str = "all_valid"
    floor: float = 0.0
    ceiling: float = 0.0
    # step index delta from a state to its chronological successor (-1: Gemma4Renoise counts down)
    step_direction: int = -1

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "GemmaAdjacEntConfig":
        values = values or {}
        if not isinstance(values, Mapping):
            raise ValueError("gemma_adjacent_entropy_contraction must be a mapping")
        scope = str(values.get("scope", "all_valid")).strip().lower()
        if scope in ("global", "all"):
            scope = "all_valid"
        if scope in ("unresolved", "changed"):
            scope = "renoised"
        cfg = cls(
            enable=bool(values.get("enable", False)),
            rho=float(values.get("rho", 0.2)),
            coef=float(values.get("coef", 50.0)),
            scope=scope,
            floor=float(values.get("floor", 0.0) or 0.0),
            ceiling=float(values.get("ceiling", 0.0) or 0.0),
            step_direction=int(values.get("step_direction", -1) or -1),
        )
        if cfg.enable:
            cfg.validate()
        return cfg

    def validate(self) -> None:
        # reuse the LLaDA2 validation for the shared fields
        AdjacEntConfig(
            enable=True, rho=self.rho, coef=self.coef, floor=self.floor, ceiling=self.ceiling
        ).validate()
        if self.scope not in _SCOPES:
            raise ValueError(
                f"gemma_adjacent_entropy_contraction.scope must be one of {_SCOPES}, got {self.scope!r}"
            )
        if self.step_direction not in (-1, 1):
            raise ValueError("gemma_adjacent_entropy_contraction.step_direction must be -1 or 1")


@dataclass
class GemmaAecState:
    """Per-chunk-call accumulator (one actor micro-batch = a few block slots)."""

    pairs: int = 0
    violations: int = 0            # H_next > (1 - rho) * H_prev
    raw_violations: int = 0        # H_next > H_prev (no rho)
    floor_exempt: int = 0
    ceiling_exempt: int = 0
    empty_mask_pairs: int = 0
    h_prev_sum: float = 0.0
    h_next_sum: float = 0.0
    hinge_sum: float = 0.0
    mask_fraction_sum: float = 0.0  # |M| / 256 per pair (1.0 for all_valid)
    boundary_pairs: int = 0        # pairs formed with the previous wave's carry
    # block_slot -> (step_index, detached per-position normalised entropy [L], canvas [L])
    carry: dict = field(default_factory=dict)


def count_gemma_adjacent_pairs(tensor_batch: Mapping[str, torch.Tensor]) -> int:
    """Number of (t, t+1) step pairs over all real block slots (trainer-side denominator)."""
    row_valid = tensor_batch["gemma_replay_row_valid"].bool()
    step_indices = tensor_batch["gemma_replay_step_indices"].long()
    block_real = tensor_batch["gemma_replay_block_real"].bool()
    pairs = 0
    for slot in range(int(row_valid.shape[0])):
        if not bool(block_real[slot]):
            continue
        steps = sorted(step_indices[slot][row_valid[slot]].tolist())
        pairs += sum(1 for i in range(len(steps) - 1) if steps[i + 1] == steps[i] + 1)
    return int(pairs)


def _norm_entropy_rows(logits: torch.Tensor, log_norm: float) -> torch.Tensor:
    """[R, L, V] fp32 logits -> [R, L] normalised entropy."""
    log_p = F.log_softmax(logits, dim=-1)
    return -(log_p.exp() * log_p).sum(dim=-1) / log_norm


def _norm_entropy_rows_ckpt(logits: torch.Tensor, log_norm: float) -> torch.Tensor:
    # One checkpointed call per wave: the [R, L, V] log_softmax / softmax intermediates are
    # recomputed in backward instead of being kept alive next to the CE graph.
    return checkpoint(_norm_entropy_rows, logits, log_norm, use_reentrant=False)


def gemma_aec_wave_terms(
    *,
    logits: torch.Tensor,
    canvas_chunk: torch.Tensor,
    index_chunk: torch.Tensor,
    actual_rows: int,
    block_slot: int,
    cfg: GemmaAdjacEntConfig,
    state: GemmaAecState,
) -> torch.Tensor | None:
    """AEC hinge sum for one decoder wave (``logits``: [chunk, L, V] fp32 with grad).

    Rows ``0..actual_rows-1`` are real states of ``block_slot`` in chronological order
    (real_row_pack, walked in the direction of ``cfg.step_direction``) - pairs are consecutive
    rows whose step indices differ by exactly ``step_direction``, plus the boundary pair with
    the previous wave's carried last row.  Returns the differentiable hinge sum
    (un-normalised) or None when the wave produced no eligible pair.  Updates the carry for
    the block slot with the last real row.  Vectorised: one entropy pass over the wave, the
    "prev" side is the detached entropy of the same pass (or the carry), a handful of host
    syncs per wave.
    """
    if actual_rows <= 0:
        return None
    L = int(logits.shape[1])
    V = int(logits.shape[-1])
    log_norm = math.log(max(V, 2))
    device = logits.device
    steps = [int(s) for s in index_chunk[:actual_rows].detach().cpu().tolist()]
    d = int(cfg.step_direction)

    h_all = _norm_entropy_rows_ckpt(logits[:actual_rows], log_norm)  # [R, L] with grad
    h_det = h_all.detach()

    prev_vecs: list[torch.Tensor] = []
    prev_canvases: list[torch.Tensor] = []
    next_rows: list[int] = []
    boundary = 0
    carry = state.carry.get(block_slot)
    if carry is not None and carry[0] + d == steps[0]:
        prev_vecs.append(carry[1]); prev_canvases.append(carry[2]); next_rows.append(0); boundary = 1
    for i in range(actual_rows - 1):
        if steps[i] + d == steps[i + 1]:
            prev_vecs.append(h_det[i]); prev_canvases.append(canvas_chunk[i]); next_rows.append(i + 1)
    last = actual_rows - 1
    state.carry[block_slot] = (steps[last], h_det[last].clone(), canvas_chunk[last].detach().clone())
    if not next_rows:
        return None

    prev_h = torch.stack(prev_vecs, 0)                       # [P, L] detached
    next_idx = torch.tensor(next_rows, device=device, dtype=torch.long)
    if cfg.scope == "renoised":
        mask = canvas_chunk[next_idx].ne(torch.stack(prev_canvases, 0))   # [P, L] positions not accepted at t
    else:
        mask = torch.ones((len(next_rows), L), dtype=torch.bool, device=device)
    mask_f = mask.to(torch.float32)
    n_mask = mask_f.sum(dim=1)                               # [P]
    nonempty = n_mask.gt(0)
    denom = n_mask.clamp_min(1.0)
    h_prev_mean = (prev_h * mask_f).sum(dim=1) / denom       # [P] detached
    h_next_mean = (h_all[next_idx] * mask_f).sum(dim=1) / denom  # [P] with grad
    eligible = nonempty.clone()
    floor_exempt = torch.zeros_like(eligible)
    ceiling_exempt = torch.zeros_like(eligible)
    if cfg.floor > 0.0:
        floor_exempt = nonempty & h_prev_mean.lt(cfg.floor)
        eligible &= ~floor_exempt
    if cfg.ceiling > 0.0:
        ceiling_exempt = nonempty & h_prev_mean.gt(cfg.ceiling)
        eligible &= ~ceiling_exempt
    excess = h_next_mean - (1.0 - cfg.rho) * h_prev_mean
    hinge = torch.relu(excess) ** 2 * eligible.to(torch.float32)
    hinge_sum = hinge.sum()

    with torch.no_grad():
        elig_f = eligible.to(torch.float32)
        stats = torch.stack([
            elig_f.sum(),
            (excess.gt(0) & eligible).to(torch.float32).sum(),
            ((h_next_mean > h_prev_mean) & eligible).to(torch.float32).sum(),
            floor_exempt.to(torch.float32).sum(),
            ceiling_exempt.to(torch.float32).sum(),
            (~nonempty).to(torch.float32).sum(),
            (h_prev_mean * elig_f).sum(),
            (h_next_mean * elig_f).sum(),
            hinge_sum,
            (n_mask / float(L) * elig_f).sum(),
        ]).cpu().tolist()
    n_elig = int(stats[0])
    state.pairs += n_elig
    state.violations += int(stats[1])
    state.raw_violations += int(stats[2])
    state.floor_exempt += int(stats[3])
    state.ceiling_exempt += int(stats[4])
    state.empty_mask_pairs += int(stats[5])
    state.h_prev_sum += stats[6]
    state.h_next_sum += stats[7]
    state.hinge_sum += stats[8]
    state.mask_fraction_sum += stats[9]
    state.boundary_pairs += boundary if (n_elig > 0 and bool(eligible[0].item()) if boundary else False) else 0
    return hinge_sum if n_elig > 0 else None


def gemma_aec_metrics(state: GemmaAecState, *, cfg: GemmaAdjacEntConfig, total_pairs_hint: float, loss_value: float) -> dict[str, float]:
    pairs = max(state.pairs, 1)
    return {
        "adj_ent/violation_rate": state.violations / pairs if state.pairs else 0.0,
        "adj_ent/raw_violation_rate": state.raw_violations / pairs if state.pairs else 0.0,
        "adj_ent/avg_entropy_t": state.h_prev_sum / pairs if state.pairs else 0.0,
        "adj_ent/avg_entropy_next": state.h_next_sum / pairs if state.pairs else 0.0,
        "adj_ent/relative_entropy_drop": (
            (state.h_prev_sum - state.h_next_sum) / max(state.h_prev_sum, 1e-8) if state.pairs else 0.0
        ),
        "adj_ent/valid_pair_count": float(state.pairs),
        "adj_ent/boundary_pairs": float(state.boundary_pairs),
        "adj_ent/hinge_sum": state.hinge_sum,
        "adj_ent/floor_exempt": float(state.floor_exempt),
        "adj_ent/ceiling_exempt": float(state.ceiling_exempt),
        "adj_ent/empty_mask_pairs": float(state.empty_mask_pairs),
        "adj_ent/mask_fraction": state.mask_fraction_sum / pairs if state.pairs else 0.0,
        "adj_ent/total_pairs_hint": float(total_pairs_hint),
        "adj_ent/rho": cfg.rho,
        "adj_ent/coef": cfg.coef,
        "adj_ent/scope_renoised": 1.0 if cfg.scope == "renoised" else 0.0,
        "actor/gemma_aec/loss": loss_value,
    }
