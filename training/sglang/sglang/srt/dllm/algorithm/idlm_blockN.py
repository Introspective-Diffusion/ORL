"""
IDLM Block-N decoding: up to N tokens per forward with speculative verification.

Generalizes Block-2 to arbitrary N. Each forward has two modes:
  Cold start:  [t0, M, M, ..., M]        (1 + 2(N-1) positions, padded)
  Verify:      [pending, spec0..specK, M, M, ..., M]  (1 + K + (N-1) positions)

The SGLang block_size must be set to 2*N - 1 (the max input size for verify rounds).
N is controlled by the `gen_block_size` config key.

Speculative tokens are verified left-to-right using the standard spec decoding criterion:
    r = p(x) / q(x)
    if r >= 1: always accept
    if r <  1: accept with probability r, else resample from max(0, p-q)

Config keys (passed via --dllm-algorithm-config YAML):
  block_size:             int,   MUST be 2*gen_block_size - 1
  gen_block_size:         int,   tokens per step (1 clean + N-1 spec). Default 3.
  confidence_threshold:   float, default 0.0 (disabled, rely on spec verify)
  temperature:            float, default 1.0
  top_k:                  int,   default 50
  top_p:                  float, default 0.95
  use_spec_verify:        bool,  default true
"""

import logging
import time
from typing import Dict, List, Tuple, Union

import torch
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.base import DllmAlgorithm

try:
    from sglang.srt.dllm.algorithm.fused_verify_kernel import fused_spec_verify, fused_spec_verify_from_logits
    _HAS_FUSED_VERIFY = True
except ImportError:
    _HAS_FUSED_VERIFY = False
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


try:
    from flashinfer.sampling import top_k_top_p_sampling_from_probs as _fi_sample
    _HAS_FLASHINFER_SAMPLING = True
except ImportError:
    _HAS_FLASHINFER_SAMPLING = False


# Pre-allocated sampling parameter tensors (lazily initialized per device)
_SAMPLE_BUFS: Dict[torch.device, Dict[str, torch.Tensor]] = {}


def _get_sample_bufs(n: int, top_k: int, top_p: float, device: torch.device):
    """Get or create pre-allocated top_k/top_p tensors for flashinfer sampling."""
    bufs = _SAMPLE_BUFS.get(device)
    if bufs is None or bufs["size"] < n:
        new_size = max(n, 128)
        bufs = {
            "size": new_size,
            "top_ks": torch.full((new_size,), top_k, dtype=torch.int32, device=device),
            "top_ps": torch.full((new_size,), top_p, dtype=torch.float32, device=device),
        }
        _SAMPLE_BUFS[device] = bufs
    return bufs["top_ks"][:n], bufs["top_ps"][:n]


def _batched_sample(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    return_probs: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, ...]:
    """Batched sampling from [N, vocab_size] logits on GPU.

    Returns (token_ids [N], token_probs [N]) — both stay on GPU, no .item() sync.
    If return_probs=True, also returns the full probs [N, vocab_size] matrix.
    """
    if temperature <= 0:
        token_ids = logits.argmax(dim=-1)
        if return_probs:
            probs = F.softmax(logits, dim=-1)
            token_probs = probs.gather(1, token_ids.unsqueeze(1)).squeeze(1)
            return token_ids, token_probs, probs
        # Skip softmax when probs aren't needed (greedy path)
        token_probs = torch.ones(token_ids.shape[0], device=logits.device)
        return token_ids, token_probs

    scaled = logits if temperature == 1.0 else logits / temperature
    probs = F.softmax(scaled, dim=-1)

    if _HAS_FLASHINFER_SAMPLING:
        n = probs.shape[0]
        top_ks, top_ps = _get_sample_bufs(n, top_k, top_p, probs.device)
        token_ids = _fi_sample(
            probs.contiguous(), top_ks, top_ps, filter_apply_order="joint"
        )
        token_probs = probs.gather(1, token_ids.unsqueeze(1)).squeeze(1)
        if return_probs:
            return token_ids, token_probs, probs
        return token_ids, token_probs

    # Fallback: manual implementation
    if top_k > 0:
        topk_vals, _ = scaled.topk(top_k, dim=-1)
        scaled = scaled.masked_fill(scaled < topk_vals[:, -1:], float("-inf"))
    if top_p < 1.0:
        sorted_logits, sorted_idx = scaled.sort(dim=-1, descending=True)
        cum_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        mask = (cum_probs - sorted_logits.softmax(dim=-1)) >= top_p
        sorted_logits[mask] = float("-inf")
        scaled = sorted_logits.scatter(1, sorted_idx, sorted_logits)
    probs = F.softmax(scaled, dim=-1)
    token_ids = torch.multinomial(probs, num_samples=1).squeeze(1)
    token_probs = probs.gather(1, token_ids.unsqueeze(1)).squeeze(1)
    if return_probs:
        return token_ids, token_probs, probs
    return token_ids, token_probs


class IDLMBlockN(DllmAlgorithm):
    """
    Block-N speculative generation: 1 forward → 1 to N output tokens.

    Two modes per round (blk = 2*N - 1 positions):
      Verify:     [pending, spec0, ..., spec(N-2), M, M, ..., M]
                  → verify specs left-to-right, then sample N new tokens
      Cold start: [t0, M, M, ..., M]
                  → sample 1 clean + (N-1) spec, hold specs for next verify

    MASK KV is always freed after the forward (never persists in cache).
    Rejected specs cause a fallback to cold start with the corrected token.

    Communicates with the output processor via instance dicts:
      _dllm_write_override: per-request tokens to write to dllm_ids
      _kv_trim_info: per-request KV pool indices to free (GPU tensor)
      _advance_override: per-request variable dllm_block_offset advance
    """

    def __init__(self, config: DllmConfig):
        super().__init__(config)
        self.gen_block_size: int = config.algorithm_config.get("gen_block_size", 3)
        self.num_masks: int = self.gen_block_size - 1  # N-1 speculative tokens
        expected_blk = 2 * self.gen_block_size - 1
        assert self.block_size == expected_blk, (
            f"IDLMBlockN with gen_block_size={self.gen_block_size} requires "
            f"block_size={expected_blk}, got {self.block_size}"
        )
        self.temperature: float = config.algorithm_config.get("temperature", 1.0)
        self.top_k: int = config.algorithm_config.get("top_k", 50)
        self.top_p: float = config.algorithm_config.get("top_p", 0.95)
        self.confidence_threshold: float = config.algorithm_config.get(
            "confidence_threshold", 0.0
        )
        self.use_spec_verify: bool = config.algorithm_config.get(
            "use_spec_verify", True
        )
        # Relaxed verification: accept with prob min(1, p/(alpha*q))
        # alpha=1.0: standard verify, alpha<1: more lenient, alpha=0: always accept
        self.verify_alpha: float = config.algorithm_config.get(
            "verify_alpha", 1.0
        )
        # Number of speculative tokens to verify (default: all = num_masks)
        # Setting to 1 verifies only the first spec, auto-accepts the rest.
        self.verify_num_specs: int = min(
            config.algorithm_config.get("verify_num_specs", self.num_masks),
            self.num_masks,
        )
        # Fast verify mode: use logit-based threshold instead of full p/q ratio
        # Skips softmax computation for significant speedup
        self.fast_verify: bool = config.algorithm_config.get("fast_verify", False)
        # Logit threshold for fast verify: accept if logit(spec) is in top-K logits
        self.fast_verify_topk: int = config.algorithm_config.get("fast_verify_topk", 5)
        # Per-spec topk values for tiered verification (e.g., [7, 50] for strict spec0, lenient spec1)
        # If set, overrides fast_verify_topk for individual specs
        self.fast_verify_topk_per_spec: List[int] = config.algorithm_config.get(
            "fast_verify_topk_per_spec", []
        )
        # Output correction: when a spec token is accepted (in top-K) but isn't the
        # clean argmax, replace the OUTPUT token with the argmax for higher quality.
        # KV cache retains the original spec token (small mismatch, usually negligible).
        self.output_correction: bool = config.algorithm_config.get(
            "output_correction", False
        )

        # Per-request state (keyed by req_pool_idx)
        self._prev_last_logits: Dict[int, torch.Tensor] = {}
        # Greedy shortcut: store argmax token id directly instead of full logits
        self._prev_last_argmax: Dict[int, int] = {}
        self._pending: Dict[int, int] = {}
        self._spec_tokens: Dict[int, List[int]] = {}
        # Store full draft probability distributions for correct max(0, p-q) correction.
        # Each entry is a list of GPU tensors of shape [vocab_size], one per spec token.
        self._spec_draft_probs: Dict[int, List[torch.Tensor]] = {}
        self._force_next_token: Dict[int, int] = {}
        # Pre-allocated draft probs buffer: [max_slots, vocab_size] indexed by rpx
        # Lazily initialized on first use. Eliminates torch.stack overhead.
        self._draft_probs_buf: torch.Tensor = None  # [max_slots, vocab_size]
        self._draft_probs_buf_rpx: set = set()  # track which rpx slots are valid
        self._gumbel_seed_counter: int = 0  # incrementing seed for Gumbel RNG

        # Per-round signals to the output processor
        self._dllm_write_override: Dict[int, List[int]] = {}
        self._kv_trim_info: Dict[int, dict] = {}
        self._advance_override: Dict[int, int] = {}

        self._stats = {
            "total_forwards": 0,
            "total_tokens": 0,
            "accept_count": 0,
            "reject_count": 0,
        }
        self._timing = {
            "phase1_classify": 0.0,
            "phase2_forward": 0.0,
            "phase3_verify_sample": 0.0,
            "phase4_trim_assemble": 0.0,
            "timing_count": 0,
        }
        # Conditional LoRA: base-only for verify/committed, base+LoRA for MASK positions
        self.conditional_lora: bool = config.algorithm_config.get(
            "conditional_lora", False
        )

        self._timing_enabled = False  # Set True for profiling (adds GPU sync overhead!)
        logger.info(
            f"[IDLMBlockN] gen_block_size={self.gen_block_size}, "
            f"block_size={self.block_size}, num_masks={self.num_masks}, "
            f"spec_verify={self.use_spec_verify}, verify_alpha={self.verify_alpha}, "
            f"conditional_lora={self.conditional_lora}"
        )

    def cleanup_request(self, req_pool_idx: int):
        """Remove all per-request state for a finished request.

        Must be called when a request finishes to prevent stale state from

        being picked up by a new request that reuses the same req_pool_idx.
        """
        self._prev_last_logits.pop(req_pool_idx, None)
        self._prev_last_argmax.pop(req_pool_idx, None)
        self._pending.pop(req_pool_idx, None)
        self._spec_tokens.pop(req_pool_idx, None)
        self._spec_draft_probs.pop(req_pool_idx, None)
        self._force_next_token.pop(req_pool_idx, None)

    def _setup_conditional_lora(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
        extend_lens_cpu: List[int],
        is_prefill: List[bool],
        case_types: List[str],
        old_specs: List,
    ):
        """Set per-token LoRA mask: 0 for verify tokens, 1 for MASK tokens.

        Matches reference generate.py L3137-3142:
          lora_mask = [0]*n_verify + [1]*num_masks
        """
        lora_backend = model_runner.lora_manager.lora_backend

        # Fast path: when cuBLAS graph is captured and batch fits in mask buffer,
        # just update the mask tensor (avoids expensive segment computation).
        if lora_backend.cublas_graph_captured:
            total_tokens = sum(el for el in extend_lens_cpu if el > 0)
            if lora_backend.lora_mask is not None and total_tokens <= lora_backend.lora_mask.shape[0]:
                mask_values = []
                for bid, extend_len in enumerate(extend_lens_cpu):
                    if extend_len <= 0:
                        continue
                    req_lora_id = forward_batch.lora_ids[bid]
                    if req_lora_id is None or is_prefill[bid] or case_types[bid] == "P":
                        mask_values.extend([0.0] * extend_len)
                        continue
                    if case_types[bid] == "V":
                        base_len = 1 + len(old_specs[bid] or [])
                    else:
                        base_len = 1
                    draft_len = self.num_masks
                    pad_len = extend_len - base_len - draft_len
                    mask_values.extend([0.0] * base_len)
                    mask_values.extend([1.0] * draft_len)
                    mask_values.extend([0.0] * max(0, pad_len))
                lora_backend.update_lora_mask(mask_values)
                return
            # else: fall through to segment-based routing for large batches

        # Non-graph path: use segment-based routing (original csgmv path)
        blk = self.block_size
        lora_ids = forward_batch.lora_ids
        segment_ids: List[Union[str, None]] = []
        segment_lens: List[int] = []

        for bid, extend_len in enumerate(extend_lens_cpu):
            req_lora_id = lora_ids[bid]
            if extend_len <= 0:
                continue

            # Prefill or no LoRA → single segment, all base
            if req_lora_id is None or is_prefill[bid] or case_types[bid] == "P":
                segment_ids.append(None)
                segment_lens.append(extend_len)
                continue

            # Compute base_len (verify positions) and draft_len (MASK positions)
            if case_types[bid] == "V":
                base_len = 1 + len(old_specs[bid] or [])
            else:  # Cold start
                base_len = 1
            draft_len = self.num_masks
            pad_len = extend_len - base_len - draft_len

            # Base segment (committed + specs): no LoRA
            if base_len > 0:
                segment_ids.append(None)
                segment_lens.append(base_len)
            # MASK segment: apply LoRA
            if draft_len > 0:
                segment_ids.append(req_lora_id)
                segment_lens.append(draft_len)
            # Padding segment: no LoRA
            if pad_len > 0:
                segment_ids.append(None)
                segment_lens.append(pad_len)

        forward_batch.lora_segment_ids = segment_ids
        forward_batch.lora_segment_lens_cpu = segment_lens
        model_runner.lora_manager.prepare_lora_batch(forward_batch)

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
        overlap_fn=None,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], List[torch.Tensor], bool]:
        if self._timing_enabled:
            _t_run_start = time.perf_counter()
        batch_size = forward_batch.batch_size
        device = forward_batch.input_ids.device
        blk = self.block_size  # 2*N - 1

        # Compute per-request extend lengths and cumulative offsets
        # for mixed decode+prefill batches (variable-length ragged layout)
        _el = forward_batch.extend_seq_lens
        if _el is not None:
            extend_lens_cpu = (
                _el.tolist()
                if isinstance(_el, torch.Tensor)
                else list(_el)
            )
        else:
            extend_lens_cpu = [blk] * batch_size
        # Cumulative offsets: base[bid] = sum(extend_lens[:bid])
        base_offsets = [0] * batch_size
        for bid in range(1, batch_size):
            base_offsets[bid] = base_offsets[bid - 1] + extend_lens_cpu[bid - 1]

        # Detect inline prefill requests (extend_len != blk)
        is_prefill = [extend_lens_cpu[bid] != blk for bid in range(batch_size)]

        # Use cached CPU values from prepare_for_dllm_decode when available
        # (avoids GPU→CPU sync via torch.cat + tolist)
        _cached_rpx = getattr(forward_batch, 'dllm_rpx_cpu', None)
        if _cached_rpx is not None and len(_cached_rpx) == batch_size:
            req_pool_indices_cpu = _cached_rpx
            seq_lens_cpu = forward_batch.dllm_seq_lens_cpu
        else:
            # Fallback: GPU→CPU sync (first forward or non-decode-loop)
            _combined = torch.cat([
                forward_batch.req_pool_indices[:batch_size],
                forward_batch.seq_lens[:batch_size].to(forward_batch.req_pool_indices.dtype),
            ])
            _combined_cpu = _combined.tolist()
            req_pool_indices_cpu = _combined_cpu[:batch_size]
            seq_lens_cpu = [int(x) for x in _combined_cpu[batch_size:]]
        # In decode loop, extend_lens contain blk-sized entries (decode requests)
        # which always have MASKs. Skip the GPU sync for mask check.
        if any(el == blk for el in extend_lens_cpu):
            has_any_decode = True
        else:
            has_any_decode = (forward_batch.input_ids == self.mask_id).any().item()

        # ── Pure prefill ──────────────────────────────────────────────
        if not has_any_decode:
            # Conditional LoRA: prefill is base-only (no LoRA)
            if self.conditional_lora and forward_batch.lora_ids is not None:
                forward_batch.lora_segment_ids = [None] * batch_size
                forward_batch.lora_segment_lens_cpu = extend_lens_cpu
                model_runner.lora_manager.prepare_lora_batch(forward_batch)
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            full_logits = out.logits_output.full_logits

            prefix_lens = forward_batch.extend_prefix_lens
            offset = 0
            for bid in range(batch_size):
                seq_len = int(forward_batch.seq_lens[bid].item())
                prefix_len = (
                    int(prefix_lens[bid].item()) if prefix_lens is not None else 0
                )
                n_new = seq_len - prefix_len
                rpx = req_pool_indices_cpu[bid]
                last_idx = offset + n_new - 1
                self._prev_last_logits[rpx] = (
                    full_logits[last_idx].detach().clone()
                )
                offset += n_new

            self._stats["total_forwards"] += 1
            return out.logits_output, [], out.can_run_graph

        # ── Decode (possibly mixed with inline prefill) ───────────────
        self._dllm_write_override.clear()
        self._kv_trim_info.clear()
        self._advance_override.clear()

        num_masks = self.num_masks  # N - 1

        # Phase 1: Classify requests and fill input_ids
        case_types = []
        t0_tokens = []
        was_forced = [False] * batch_size
        old_specs = [None] * batch_size
        old_draft_probs = [None] * batch_size  # full draft prob distributions per spec

        pre_sample_logits = []
        pre_sample_bids = []

        for bid in range(batch_size):
            rpx = req_pool_indices_cpu[bid]

            # Skip inline prefill requests — they don't participate in decode
            if is_prefill[bid]:
                case_types.append('P')  # prefill
                t0_tokens.append(None)
                continue

            pending = self._pending.pop(rpx, None)
            specs = self._spec_tokens.pop(rpx, None)
            draft_probs = self._spec_draft_probs.pop(rpx, None)
            base = base_offsets[bid]

            if pending is not None and specs:
                forward_batch.input_ids[base + 0] = pending
                for si, sv in enumerate(specs):
                    forward_batch.input_ids[base + 1 + si] = sv
                case_types.append('V')
                t0_tokens.append(pending)
                old_specs[bid] = specs
                old_draft_probs[bid] = draft_probs
            else:
                forced = self._force_next_token.pop(rpx, None)
                if forced is not None:
                    forward_batch.input_ids[base] = forced
                    t0_tokens.append(forced)
                    was_forced[bid] = True
                else:
                    # Greedy shortcut: use cached argmax directly (no GPU tensor needed)
                    cached_argmax = self._prev_last_argmax.pop(rpx, None)
                    if cached_argmax is not None:
                        forward_batch.input_ids[base] = cached_argmax
                        t0_tokens.append(cached_argmax)
                    else:
                        prev = self._prev_last_logits.get(rpx)
                        if prev is not None:
                            pre_sample_logits.append(prev)
                            pre_sample_bids.append(bid)
                            t0_tokens.append(None)
                        else:
                            forward_batch.input_ids[base] = self.mask_id
                            t0_tokens.append(self.mask_id)
                case_types.append('C')

        # Batched pre-forward sampling for cold start t0
        if pre_sample_logits:
            sampled, _ = _batched_sample(
                torch.stack(pre_sample_logits),
                self.temperature, self.top_k, self.top_p,
            )
            sampled_cpu = sampled.tolist()
            for i, bid in enumerate(pre_sample_bids):
                tok = sampled_cpu[i]
                forward_batch.input_ids[base_offsets[bid]] = tok
                t0_tokens[bid] = tok

        if self._timing_enabled:
            _t_phase1_end = time.perf_counter()

        # Phase 2: Forward (GPU async — returns before GPU finishes)
        # Set up conditional LoRA segments: base-only for verify, base+LoRA for MASK
        if self.conditional_lora and forward_batch.lora_ids is not None:
            self._setup_conditional_lora(
                model_runner, forward_batch, extend_lens_cpu,
                is_prefill, case_types, old_specs,
            )
        forward_batch.dllm_force_causal = True
        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        forward_batch.dllm_force_causal = False

        # ── Overlap window: GPU is still computing, run CPU callback ──
        if overlap_fn is not None:
            overlap_fn()

        # Phase 3 starts here — first access to full_logits blocks until GPU done
        logits_output = out.logits_output
        full_logits = logits_output.full_logits
        if self._timing_enabled:
            torch.cuda.synchronize()  # For accurate phase timing (only when profiling)
            _t_phase2_end = time.perf_counter()

        # Phase 3: Batched post-forward — verify + sample + trim
        # seq_lens_cpu already computed above (combined tolist with req_pool_indices)
        req_to_token = model_runner.req_to_token_pool.req_to_token

        # Separate decode bids from prefill bids
        decode_bids = [bid for bid in range(batch_size) if not is_prefill[bid]]
        prefill_bids = [bid for bid in range(batch_size) if is_prefill[bid]]

        # ── Handle inline prefill requests ─────────────────────────
        # Save last logits for each prefill request (like pure prefill path)
        if prefill_bids:
            prefill_logit_indices = []
            for bid in prefill_bids:
                # Last token position in the ragged batch for this request
                last_idx = base_offsets[bid] + extend_lens_cpu[bid] - 1
                prefill_logit_indices.append(last_idx)
            prefill_idx_t = torch.tensor(
                prefill_logit_indices, dtype=torch.long, device=device
            )
            prefill_logits = full_logits[prefill_idx_t].detach().clone()
            for k, bid in enumerate(prefill_bids):
                rpx = req_pool_indices_cpu[bid]
                self._prev_last_logits[rpx] = prefill_logits[k]

        # ── Steps 1+2: Verify + sample (all GPU work, then single CPU sync) ──
        verify_bids = [bid for bid in decode_bids if case_types[bid] == 'V']
        cold_bids = [bid for bid in decode_bids if case_types[bid] == 'C']

        reject_at = {}       # bid -> rejected spec index
        corrected = {}       # bid -> corrected token id
        spec_corrections = {}  # bid -> list of corrected tokens per spec (output correction mode)
        gs = 1 + num_masks
        nv = 0
        vns = self.verify_num_specs

        # Build speculative sampling indices for ALL verify + cold bids
        sample_logit_indices = []
        sample_bid_roles = []
        for bid in verify_bids:
            n_specs = len(old_specs[bid])
            clean_idx = base_offsets[bid] + n_specs
            for j in range(gs):
                sample_logit_indices.append(clean_idx + j)
            sample_bid_roles.append(bid)
        for bid in cold_bids:
            base = base_offsets[bid]
            for j in range(gs):
                sample_logit_indices.append(base + j)
            sample_bid_roles.append(bid)

        # --- All GPU work: verify + correction + sample ---
        all_accepted_gpu = None
        all_corr_tokens_gpu = None
        all_sample_ids_gpu = None
        draft_probs_all = None
        if self._timing_enabled:
            _t_vs_start = time.perf_counter()

        # ── Fused greedy path: single gather + argmax for verify+sample ──
        if self.temperature <= 0 and not self.fast_verify:
            nv = len(verify_bids) if (self.use_spec_verify and verify_bids) else 0

            # Build combined index list: [verify_indices... | sample_indices...]
            all_gather_idx = []
            all_spec_vals = []
            if nv > 0:
                for bid in verify_bids:
                    base = base_offsets[bid]
                    for si in range(vns):
                        all_gather_idx.append(base + si)
                        all_spec_vals.append(old_specs[bid][si])

            verify_count = len(all_gather_idx)  # nv * vns
            combined_indices = all_gather_idx + sample_logit_indices

            if combined_indices:
                # Single gather + single argmax for everything
                combined_logits = full_logits[
                    torch.tensor(combined_indices, dtype=torch.long, device=device)
                ]
                combined_argmax = combined_logits.argmax(dim=-1)

                # Split results
                if verify_count > 0:
                    all_corr_tokens_gpu = combined_argmax[:verify_count]
                    all_spec_vals_t = torch.tensor(
                        all_spec_vals, dtype=torch.long, device=device
                    )
                    all_accepted_gpu = all_spec_vals_t == all_corr_tokens_gpu

                if sample_logit_indices:
                    all_sample_ids_gpu = combined_argmax[verify_count:]

            if self._timing_enabled:
                _t_vs_after_verify = time.perf_counter()
                _t_vs_after_sample = time.perf_counter()
        else:
            # ── Non-greedy path: separate verify + sample ──
            if self.use_spec_verify and verify_bids:
                nv = len(verify_bids)

                all_gather_idx = []
                all_spec_vals = []
                all_draft_prob_list = []
                _need_draft_probs = (
                    self.temperature > 0
                    and not self.fast_verify
                )
                for bid in verify_bids:
                    base = base_offsets[bid]
                    for si in range(vns):
                        all_gather_idx.append(base + si)
                        all_spec_vals.append(old_specs[bid][si])
                        if _need_draft_probs:
                            all_draft_prob_list.append(old_draft_probs[bid][si])

                all_clean_logits = full_logits[
                    torch.tensor(all_gather_idx, dtype=torch.long, device=device)
                ]
                all_spec_vals_t = torch.tensor(
                    all_spec_vals, dtype=torch.long, device=device
                )

                if self.fast_verify:
                    spec_logit_vals = all_clean_logits.gather(
                        1, all_spec_vals_t.unsqueeze(1)
                    ).squeeze(1)

                    if self.fast_verify_topk_per_spec and vns > 1:
                        per_spec_topk = self.fast_verify_topk_per_spec
                        all_accepted_gpu = torch.zeros(nv * vns, dtype=torch.bool, device=device)
                        for si in range(vns):
                            k = per_spec_topk[si] if si < len(per_spec_topk) else self.fast_verify_topk
                            si_indices = list(range(si, nv * vns, vns))
                            si_logits = all_clean_logits[si_indices]
                            si_spec_vals = spec_logit_vals[si_indices]
                            topk_vals_si, _ = si_logits.topk(k, dim=-1)
                            thresholds_si = topk_vals_si[:, -1]
                            accepted_si = si_spec_vals >= thresholds_si
                            for j, idx in enumerate(si_indices):
                                all_accepted_gpu[idx] = accepted_si[j]
                    else:
                        topk_vals, _ = all_clean_logits.topk(
                            self.fast_verify_topk, dim=-1
                        )
                        topk_thresholds = topk_vals[:, -1]
                        all_accepted_gpu = spec_logit_vals >= topk_thresholds
                    all_corr_tokens_gpu = all_clean_logits.argmax(dim=-1)
                elif _HAS_FUSED_VERIFY and self.verify_alpha > 0:
                    # Fused Triton kernel: softmax + ratio + Gumbel-max correction
                    all_draft_logits_t = torch.stack(all_draft_prob_list)
                    self._gumbel_seed_counter += 1
                    # Use from_logits variant: does softmax internally,
                    # avoids pre-computing draft probs in the previous step
                    all_accepted_gpu, all_corr_tokens_gpu = fused_spec_verify_from_logits(
                        all_clean_logits, all_draft_logits_t, all_spec_vals_t,
                        temperature=self.temperature,
                        alpha=self.verify_alpha,
                        gumbel_seed=self._gumbel_seed_counter,
                    )
                else:
                    # Standard verify: softmax + p/q ratio + correction
                    if self.temperature != 1.0:
                        all_clean_logits = all_clean_logits / self.temperature
                    all_clean_probs = F.softmax(all_clean_logits, dim=-1)

                    all_draft_probs_t = torch.stack(all_draft_prob_list)
                    all_p = all_clean_probs.gather(
                        1, all_spec_vals_t.unsqueeze(1)
                    ).squeeze(1)
                    all_q = all_draft_probs_t.gather(
                        1, all_spec_vals_t.unsqueeze(1)
                    ).squeeze(1)
                    alpha = self.verify_alpha
                    if alpha <= 0:
                        all_accepted_gpu = torch.ones(
                            nv * vns, dtype=torch.bool, device=device
                        )
                    else:
                        scaled_q = all_q * alpha
                        all_ratios = torch.where(
                            scaled_q > 0, all_p / scaled_q,
                            torch.zeros_like(all_p),
                        )
                        all_rands = torch.rand(nv * vns, device=device)
                        all_accepted_gpu = (all_ratios >= 1.0) | (
                            all_rands < all_ratios
                        )
                    corrected_dist = torch.clamp(
                        all_clean_probs - all_draft_probs_t, min=0
                    )
                    corrected_sums = corrected_dist.sum(dim=-1, keepdim=True)
                    corrected_dist = torch.where(
                        corrected_sums > 0,
                        corrected_dist / corrected_sums,
                        all_clean_probs,
                    )
                    all_corr_tokens_gpu = torch.multinomial(
                        corrected_dist, num_samples=1
                    ).squeeze(1)

            if self._timing_enabled:
                _t_vs_after_verify = time.perf_counter()

            # Speculative sampling for ALL verify + cold bids (before knowing verify results)
            if sample_logit_indices:
                all_sample_logits = full_logits[
                    torch.tensor(sample_logit_indices, dtype=torch.long, device=device)
                ]

                if self.temperature > 0:
                    # Split: sample only CLEAN positions (expensive top_k/top_p),
                    # argmax SPEC positions directly (cheap, they're just guesses)
                    clean_indices = list(range(0, len(sample_logit_indices), gs))
                    draft_indices = []
                    for group_start in range(0, len(sample_logit_indices), gs):
                        for m in range(num_masks):
                            draft_indices.append(group_start + 1 + m)

                    # Sample clean positions only (1 per request instead of gs)
                    clean_logits = all_sample_logits[clean_indices]
                    clean_ids, _ = _batched_sample(
                        clean_logits, self.temperature, self.top_k, self.top_p
                    )

                    # Argmax + softmax for draft positions (skip expensive sampling)
                    all_sample_ids_gpu = torch.empty(
                        len(sample_logit_indices), dtype=clean_ids.dtype, device=device
                    )
                    all_sample_ids_gpu[clean_indices] = clean_ids
                    if draft_indices:
                        draft_logits = all_sample_logits[draft_indices]
                        all_sample_ids_gpu[draft_indices] = draft_logits.argmax(dim=-1).to(clean_ids.dtype)
                        # Store scaled logits directly (defer softmax to next step's verify)
                        # fused_spec_verify_from_logits does softmax internally
                        draft_probs_all = draft_logits if self.temperature == 1.0 else draft_logits / self.temperature
                else:
                    all_sample_ids_gpu, _ = _batched_sample(
                        all_sample_logits, self.temperature, self.top_k, self.top_p
                    )

            if self._timing_enabled:
                _t_vs_after_sample = time.perf_counter()

        # --- Single GPU→CPU sync: pack everything ---
        gpu_parts = []
        verify_len = 0
        if all_accepted_gpu is not None:
            verify_len = nv * vns
            gpu_parts.append(all_accepted_gpu.to(torch.int32))
            gpu_parts.append(all_corr_tokens_gpu)
        sample_len = 0
        if all_sample_ids_gpu is not None:
            sample_len = all_sample_ids_gpu.shape[0]
            gpu_parts.append(all_sample_ids_gpu)

        if gpu_parts:
            _mega_packed = torch.cat(gpu_parts)
            _mega_cpu = _mega_packed.tolist()
        else:
            _mega_cpu = []

        if self._timing_enabled:
            _t_vs_after_sync = time.perf_counter()

        # --- CPU unpack ---
        if verify_len > 0:
            all_accepted_cpu = _mega_cpu[:verify_len]
            all_corr_cpu = _mega_cpu[verify_len:2 * verify_len]

            for k, bid in enumerate(verify_bids):
                base_k = k * vns
                for si in range(vns):
                    if not all_accepted_cpu[base_k + si]:
                        reject_at[bid] = si
                        corrected[bid] = all_corr_cpu[base_k + si]
                        break
                if self.output_correction:
                    spec_corrections[bid] = [
                        all_corr_cpu[base_k + si] for si in range(vns)
                    ]

        sampled_results = {}
        draft_probs_map = {}
        if sample_len > 0:
            all_ids_cpu = _mega_cpu[2 * verify_len:]

            offset = 0
            draft_offset = 0
            for bid in sample_bid_roles:
                if bid not in reject_at:
                    sampled_results[bid] = all_ids_cpu[offset:offset + gs]
                    if draft_probs_all is not None:
                        draft_probs_map[bid] = [
                            draft_probs_all[draft_offset + m]
                            for m in range(num_masks)
                        ]
                    else:
                        draft_probs_map[bid] = []
                offset += gs
                draft_offset += num_masks

        if self._timing_enabled:
            _t_phase3_end = time.perf_counter()

        # ── Step 3: Batched KV trim index lookup (decode only) ─────
        trim_counts = {}
        advances = {}
        for bid in decode_bids:
            if bid in reject_at:
                si = reject_at[bid]
                n_remaining = len(old_specs[bid]) - si
                trim_counts[bid] = n_remaining + num_masks
                advances[bid] = 1 + si
            elif case_types[bid] == 'V':
                trim_counts[bid] = num_masks
                advances[bid] = 1 + len(old_specs[bid])
            else:
                trim_counts[bid] = blk - 1
                advances[bid] = 1

        # Vectorized KV trim index computation (avoid per-element append)
        total_trim = sum(trim_counts[bid] for bid in decode_bids)
        if total_trim > 0:
            all_trim_rpx = [0] * total_trim
            all_trim_pos = [0] * total_trim
            _off = 0
            for bid in decode_bids:
                rpx = req_pool_indices_cpu[bid]
                sl = seq_lens_cpu[bid]
                tc = trim_counts[bid]
                for t in range(tc):
                    all_trim_rpx[_off] = rpx
                    all_trim_pos[_off] = sl - 1 - t
                    _off += 1
            all_kv_indices = req_to_token[all_trim_rpx, all_trim_pos]
        else:
            all_kv_indices = None

        # ── Step 4: Assemble outputs ──────────────────────────────
        next_token_ids_list = []
        kv_offset = 0

        # Batched gather of logits to save for decode requests
        _logit_save_indices = []
        _logit_save_bids = []
        for bid in decode_bids:
            base = base_offsets[bid]
            if bid in reject_at:
                _logit_save_indices.append(base + reject_at[bid])
            elif case_types[bid] == 'V':
                _logit_save_indices.append(base + len(old_specs[bid]))
            else:
                _logit_save_indices.append(base)
            _logit_save_bids.append(bid)

        _saved_logits = {}
        _saved_argmax = {}  # bid -> argmax token (greedy shortcut)
        if _logit_save_indices:
            _idx_t = torch.tensor(_logit_save_indices, dtype=torch.long, device=device)
            if self.temperature <= 0:
                # Greedy shortcut: just compute argmax, skip expensive clone
                _all_argmax = full_logits[_idx_t].argmax(dim=-1).tolist()
                for k, bid in enumerate(_logit_save_bids):
                    _saved_argmax[bid] = _all_argmax[k]
            else:
                _all_saved = full_logits[_idx_t].detach().clone()
                for k, bid in enumerate(_logit_save_bids):
                    _saved_logits[bid] = _all_saved[k]

        for bid in range(batch_size):
            rpx = req_pool_indices_cpu[bid]

            # Prefill requests: no output tokens, handled above
            if is_prefill[bid]:
                next_token_ids_list.append([])
                continue

            tc = trim_counts[bid]
            adv = advances[bid]

            if bid in reject_at:
                si = reject_at[bid]
                specs = old_specs[bid]
                ct = corrected[bid]
                # Output correction: replace accepted specs before rejection with clean argmax
                if bid in spec_corrections:
                    corr = spec_corrections[bid]
                    out_pre = [corr[j] if j < len(corr) else specs[j] for j in range(si)]
                else:
                    out_pre = list(specs[:si])
                output_tokens = out_pre + [ct]
                dllm_tokens = [t0_tokens[bid]] + out_pre + [ct]
                dllm_tokens += [self.mask_id] * (blk - len(dllm_tokens))
                self._force_next_token[rpx] = ct
                # Reject path: force_next_token handles t0, but still need logits
                # for verify round after the forced cold start
                if bid in _saved_argmax:
                    self._prev_last_argmax[rpx] = _saved_argmax[bid]
                else:
                    self._prev_last_logits[rpx] = _saved_logits[bid]
                self._stats["reject_count"] += 1

            elif case_types[bid] == 'V':
                specs = old_specs[bid]
                sr = sampled_results[bid]
                clean_token = sr[0]
                new_spec_tokens = sr[1:]
                # Output correction: replace spec tokens with clean argmax for quality
                if bid in spec_corrections:
                    corr = spec_corrections[bid]
                    # Replace verified specs with their clean corrections
                    out_specs = [corr[si] if si < len(corr) else specs[si]
                                 for si in range(len(specs))]
                else:
                    out_specs = list(specs)
                output_tokens = out_specs + [clean_token]
                dllm_tokens = [t0_tokens[bid]] + out_specs + [clean_token]
                dllm_tokens += [self.mask_id] * (blk - len(dllm_tokens))
                self._pending[rpx] = clean_token
                self._spec_tokens[rpx] = new_spec_tokens
                self._spec_draft_probs[rpx] = draft_probs_map.get(bid, [])
                if bid in _saved_argmax:
                    self._prev_last_argmax[rpx] = _saved_argmax[bid]
                else:
                    self._prev_last_logits[rpx] = _saved_logits[bid]
                self._stats["accept_count"] += 1

            else:
                t0 = t0_tokens[bid]
                sr = sampled_results[bid]
                clean_token = sr[0]
                new_spec_tokens = sr[1:]
                if was_forced[bid]:
                    output_tokens = [clean_token]
                else:
                    output_tokens = [t0, clean_token]
                dllm_tokens = [t0, clean_token] + new_spec_tokens
                dllm_tokens += [self.mask_id] * (blk - len(dllm_tokens))
                self._pending[rpx] = clean_token
                self._spec_tokens[rpx] = new_spec_tokens
                self._spec_draft_probs[rpx] = draft_probs_map.get(bid, [])
                if bid in _saved_argmax:
                    self._prev_last_argmax[rpx] = _saved_argmax[bid]
                else:
                    self._prev_last_logits[rpx] = _saved_logits[bid]

            next_token_ids_list.append(output_tokens)
            self._dllm_write_override[rpx] = dllm_tokens
            self._advance_override[rpx] = adv
            self._kv_trim_info[rpx] = {
                "kv_indices_gpu": all_kv_indices[kv_offset:kv_offset + tc] if all_kv_indices is not None else None,
                "trim_count": tc,
            }
            kv_offset += tc

        # Debug logging for single request
        if batch_size == 1 and not is_prefill[0]:
            rpx0 = req_pool_indices_cpu[0]
            ct = case_types[0]
            out_toks = next_token_ids_list[0] if next_token_ids_list else []
            adv = self._advance_override.get(rpx0, '?')
            tc = self._kv_trim_info.get(rpx0, {}).get('trim_count', '?')
            n_pend = self._pending.get(rpx0, '∅')
            n_specs = self._spec_tokens.get(rpx0, [])
            logger.info(
                f"[STEP] {ct} t0={t0_tokens[0]} "
                f"→ out={out_toks} adv={adv} trim={tc} "
                f"next: pend={n_pend} specs={n_specs[:3]}..."
            )

        # Stats + optional timing
        self._stats["total_forwards"] += 1
        self._stats["total_tokens"] += sum(len(t) for t in next_token_ids_list)
        if self._timing_enabled and has_any_decode:
            _t_run_end = time.perf_counter()
            self._timing["phase1_classify"] += (_t_phase1_end - _t_run_start)
            self._timing["phase2_forward"] += (_t_phase2_end - _t_phase1_end)
            self._timing["phase3_verify_sample"] += (_t_phase3_end - _t_phase2_end)
            self._timing["phase4_trim_assemble"] += (_t_run_end - _t_phase3_end)
            # Micro-timing for verify_sample breakdown
            self._timing.setdefault("vs_verify_gpu", 0.0)
            self._timing.setdefault("vs_sample_gpu", 0.0)
            self._timing.setdefault("vs_sync", 0.0)
            self._timing["vs_verify_gpu"] += (_t_vs_after_verify - _t_vs_start)
            self._timing["vs_sample_gpu"] += (_t_vs_after_sample - _t_vs_after_verify)
            self._timing["vs_sync"] += (_t_vs_after_sync - _t_vs_after_sample)
            self._timing["timing_count"] += 1
        if self._stats["total_forwards"] % 500 == 0:
            s = self._stats
            n = s["total_forwards"]
            tok_per_fwd = s["total_tokens"] / max(n, 1)
            total_decisions = s["accept_count"] + s["reject_count"]
            accept_rate = (
                s["accept_count"] / total_decisions * 100
                if total_decisions > 0
                else 0
            )
            timing_str = ""
            if self._timing_enabled:
                t = self._timing
                tc = max(t["timing_count"], 1)
                vs_detail = ""
                if t.get("vs_verify_gpu"):
                    vs_detail = (
                        f" [verify_gpu={t['vs_verify_gpu']/tc*1000:.2f}"
                        f" sample_gpu={t['vs_sample_gpu']/tc*1000:.2f}"
                        f" sync={t['vs_sync']/tc*1000:.2f}]"
                    )
                timing_str = (
                    f" timing(ms): classify={t['phase1_classify']/tc*1000:.2f} "
                    f"forward={t['phase2_forward']/tc*1000:.2f} "
                    f"verify_sample={t['phase3_verify_sample']/tc*1000:.2f}{vs_detail} "
                    f"trim_assemble={t['phase4_trim_assemble']/tc*1000:.2f}"
                )
            logger.info(
                f"[IDLMBlockN] N={self.gen_block_size}, fwd={n}, bs={batch_size}, "
                f"tok/fwd={tok_per_fwd:.2f}, accept={accept_rate:.1f}%{timing_str}"
            )

        return logits_output, next_token_ids_list, out.can_run_graph


Algorithm = IDLMBlockN
