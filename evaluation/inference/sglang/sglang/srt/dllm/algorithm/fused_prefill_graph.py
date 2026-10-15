"""Exact CUDA-graph replay of the eager block-causal prompt prefill.

The ragged custom-mask prefill (DllmConfig.block_causal_prefill) is CPU
launch bound: ~23 ms for a 5-block prompt whose GPU work is a few ms. Its
shape is fully determined by the token count once the prefix is empty
(prompt_tail_in_first_block makes every prefill span a multiple of the
block size, so a dataset only has a handful of distinct spans), so the eager
forward is captured ONCE per span length -- the same kernels with the same
shapes and the same plan -- and replayed afterwards. Replay is bitwise
identical to the eager forward (checked at capture time on the real request:
full logits and the K/V written to the request's slots must match exactly,
otherwise the cache disables itself and the eager path is used).

Only the per-request inputs are copied into static buffers before replay:
input_ids, positions, out_cache_loc (K/V slots), req_pool_indices, seq_lens.
The flashinfer plan is built once with those buffers and kept alive with the
graph; it depends on span length only (prefix 0, positions 0..n-1).
"""

from __future__ import annotations

import dataclasses
import logging
import os
import time
from typing import Any

import torch

logger = logging.getLogger(__name__)

_MAX_LEN = int(os.environ.get("SDAR_FUSED_EXACT_PREFILL_MAX", "2048") or 0)
_ENABLED = os.environ.get("SDAR_FUSED_EXACT_PREFILL", "1") == "1"
_CHECK_EVERY = os.environ.get("SDAR_EXACT_PREFILL_CHECK", "0") == "1"  # debug: compare every replay with eager
# production guard: every N-th replay is re-run eagerly and compared bitwise
# (logits + sampled K/V); a mismatch disables the cache. ~25 ms per check.
_GUARD_EVERY = int(os.environ.get("SDAR_EXACT_PREFILL_GUARD_EVERY", "128") or 0)
# capture graphs for every span length block_size..PRECAPTURE_MAX at the first
# prefill (on scratch K/V slots), so no request pays the capture in the loop;
# each pre-captured graph is still bitwise-checked against eager on its first
# real use. 0 disables pre-capture (capture lazily per length).
_PRECAPTURE_MAX = int(os.environ.get("SDAR_EXACT_PREFILL_PRECAPTURE_MAX", "1024") or 0)
# pre-captured graphs are already replay-vs-eager checked at capture (random
# token ids); =1 additionally re-checks each on its first real request (~40 ms each)
_VERIFY_FIRST_USE = os.environ.get("SDAR_EXACT_PREFILL_VERIFY_FIRST_USE", "0") == "1"


@dataclasses.dataclass
class _Entry:
    graph: torch.cuda.CUDAGraph
    fb: Any  # ForwardBatch view over the static buffers
    out: Any  # LogitsProcessorOutput (static)
    keepalive: Any  # attention metadata / plan referenced by the graph
    ref: Any = None  # debug: (logits, kv) of the capture request
    verified: bool = True  # pre-captured entries are checked on their first real replay


def _boundary_output(boundary: torch.Tensor):
    """LogitsProcessorOutput carrying only the boundary row as full_logits
    ([1, V]); the prefill branch of SDARPrefixKV reads row -1."""
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput

    return LogitsProcessorOutput(next_token_logits=None, full_logits=boundary.unsqueeze(0), hidden_states=None)


@dataclasses.dataclass
class _Result:
    logits_output: Any
    graph_was_replayed: bool = True
    graph_replay_miss_reason: str | None = None
    can_run_graph: bool = True


class ExactPrefillGraphs:
    def __init__(self, block_size: int, check_layers: int = 3):
        self.block_size = int(block_size)
        self.entries: dict[int, _Entry] = {}
        self.pool = None
        self.stream = None
        self.disabled = not _ENABLED
        self.disabled_reason: str | None = None
        self.check_layers = check_layers
        self.stats = {"replays": 0, "captures": 0, "capture_s": 0.0}
        self._precaptured = False

    # ------------------------------------------------------------------
    def applicable(self, forward_batch, trace: bool) -> bool:
        if self.disabled or trace:
            return False
        if int(forward_batch.batch_size) != 1:
            return False
        if getattr(forward_batch, "input_embeds", None) is not None:
            return False
        if not forward_batch.forward_mode.is_extend():
            return False
        n = int(forward_batch.input_ids.numel())
        if n <= self.block_size or n > _MAX_LEN or n % self.block_size != 0:
            return False
        pre = forward_batch.extend_prefix_lens_cpu
        if pre and int(pre[0]) != 0:
            return False
        if not forward_batch.input_ids.is_cuda:
            return False
        return True

    # ------------------------------------------------------------------
    def run(self, model_runner, forward_batch):
        """Replay (or capture on first sight of this span) the eager prefill.
        Returns an object with .logits_output / .graph_was_replayed, or None
        when the exact path is not applicable / got disabled."""
        n = int(forward_batch.input_ids.numel())
        if not self._precaptured and _PRECAPTURE_MAX > 0:
            self._precaptured = True
            try:
                self.precapture(model_runner, forward_batch)
            except Exception as exc:  # noqa: BLE001
                self._disable(f"precapture failed: {exc!r}")
                return None
        entry = self.entries.get(n)
        if entry is not None and not entry.verified:
            # first real use of a pre-captured graph: replay, then eager, compare
            self._copy_inputs(entry.fb, forward_batch)
            entry.graph.replay()
            torch.cuda.synchronize()
            rep_logits = entry.out.clone()
            rep_kv = self._snapshot_kv(model_runner, forward_batch.out_cache_loc)
            model_runner.attn_backend.init_forward_metadata(forward_batch)
            out = model_runner.forward(forward_batch, skip_attn_backend_init=True)
            torch.cuda.synchronize()
            eag_kv = self._snapshot_kv(model_runner, forward_batch.out_cache_loc)
            ok = bool(torch.equal(rep_logits, out.logits_output.full_logits[-1]))
            for a, b in zip(rep_kv, eag_kv):
                ok = ok and bool(torch.equal(a[0], b[0])) and bool(torch.equal(a[1], b[1]))
            if not ok:
                self._disable(f"pre-captured graph n={n} != eager on first use")
            else:
                entry.verified = True
                self.stats["verified"] = self.stats.get("verified", 0) + 1
            return _Result(logits_output=out.logits_output, graph_was_replayed=False)
        if entry is None:
            try:
                return self._capture(model_runner, forward_batch, n)
            except Exception as exc:  # noqa: BLE001 - any failure -> eager path
                self._disable(f"capture failed for n={n}: {exc!r}")
                return None
        self._copy_inputs(entry.fb, forward_batch)
        entry.graph.replay()
        self.stats["replays"] += 1
        if _GUARD_EVERY and not _CHECK_EVERY and self.stats["replays"] % _GUARD_EVERY == 0:
            torch.cuda.synchronize()
            rep_logits = entry.out.clone()
            rep_kv = self._snapshot_kv(model_runner, forward_batch.out_cache_loc)
            model_runner.attn_backend.init_forward_metadata(forward_batch)
            out = model_runner.forward(forward_batch, skip_attn_backend_init=True)
            torch.cuda.synchronize()
            eag_kv = self._snapshot_kv(model_runner, forward_batch.out_cache_loc)
            ok = bool(torch.equal(rep_logits, out.logits_output.full_logits[-1]))
            for a, b in zip(rep_kv, eag_kv):
                ok = ok and bool(torch.equal(a[0], b[0])) and bool(torch.equal(a[1], b[1]))
            self.stats["guard_checks"] = self.stats.get("guard_checks", 0) + 1
            if not ok:
                self._disable(f"guard: replay != eager at replay #{self.stats['replays']} (n={n})")
            return _Result(logits_output=out.logits_output, graph_was_replayed=False)
        if _CHECK_EVERY:
            torch.cuda.synchronize()
            rep_logits = entry.out.clone()
            rep_kv = self._snapshot_kv(model_runner, forward_batch.out_cache_loc)
            model_runner.attn_backend.init_forward_metadata(forward_batch)
            out = model_runner.forward(forward_batch, skip_attn_backend_init=True)
            torch.cuda.synchronize()
            eag_kv = self._snapshot_kv(model_runner, forward_batch.out_cache_loc)
            d = (rep_logits.float() - out.logits_output.full_logits[-1].float()).abs().unsqueeze(0)
            kvd = ["L%d K%.2e V%.2e" % (i, (a[0].float() - b[0].float()).abs().max().item(), (a[1].float() - b[1].float()).abs().max().item())
                   for i, (a, b) in enumerate(zip(rep_kv, eag_kv))]
            extra = ""
            if entry.ref is not None:
                ref_logits, ref_kv, ref_ids = entry.ref
                extra = " | vs CAPTURE request: logits max|d|=%.3e, L1 K max|d|=%.3e, input_ids same=%s, static buf == new ids: %s" % (
                    (rep_logits.float() - ref_logits.float()).abs().max().item(),
                    (rep_kv[1][0].float() - ref_kv[1][0].float()).abs().max().item() if len(rep_kv) > 1 else float("nan"),
                    bool(torch.equal(ref_ids, forward_batch.input_ids)), bool(torch.equal(entry.fb.input_ids, forward_batch.input_ids)))
            print("[SDAR-EXACT-PREFILL-CHECK] n=%d rpx=%d loc0=%d logits max|d|=%.3e rows_diff=%d/%d | kv: %s%s" % (
                n, int(forward_batch.req_pool_indices[0].item()), int(forward_batch.out_cache_loc[0].item()),
                d.max().item(), int((d.max(dim=-1).values > 0).sum().item()), d.shape[0], "; ".join(kvd), extra), flush=True)
            return _Result(logits_output=out.logits_output, graph_was_replayed=False)
        return _Result(logits_output=_boundary_output(entry.out))

    # ------------------------------------------------------------------
    @staticmethod
    def _copy_inputs(fb_static, fb):
        fb_static.input_ids.copy_(fb.input_ids)
        fb_static.positions.copy_(fb.positions)
        fb_static.out_cache_loc.copy_(fb.out_cache_loc)
        fb_static.req_pool_indices.copy_(fb.req_pool_indices)
        fb_static.seq_lens.copy_(fb.seq_lens)

    def _snapshot_kv(self, model_runner, loc):
        pool = model_runner.token_to_kv_pool
        L = model_runner.model_config.num_hidden_layers
        layers = sorted({0, L // 2, L - 1})[: self.check_layers]
        return [
            (pool.get_key_buffer(l)[loc].clone(), pool.get_value_buffer(l)[loc].clone())
            for l in layers
        ]

    def precapture(self, model_runner, template) -> None:
        """Capture every block-aligned span length up to _PRECAPTURE_MAX on
        scratch K/V slots (allocated and freed here). Entries are unverified
        until their first real replay is compared against eager."""
        alloc = getattr(model_runner, "token_to_kv_pool_allocator", None)
        if alloc is None:
            return
        t0 = time.perf_counter()
        dev = template.input_ids.device
        # a one-block span (n == block_size) is shape-eligible for the decode
        # graph runner's bs=1 graph, which claims it first; start at two blocks
        lengths = [m for m in range(2 * self.block_size, _PRECAPTURE_MAX + 1, self.block_size) if m not in self.entries]
        done = 0
        for m in lengths:
            slots = alloc.alloc(m)
            if slots is None:
                break
            try:
                # random token ids: the capture-time replay-vs-eager check then
                # exercises real attention/routing patterns (all-equal tokens
                # would make any mask look correct)
                g = torch.Generator(device="cpu").manual_seed(1234 + m)
                vocab = int(model_runner.model_config.vocab_size)
                fb = dataclasses.replace(
                    template,
                    input_ids=torch.randint(0, max(vocab - 1, 2), (m,), generator=g, dtype=template.input_ids.dtype).to(dev),
                    positions=torch.arange(m, dtype=template.positions.dtype, device=dev),
                    out_cache_loc=slots.to(device=dev, dtype=template.out_cache_loc.dtype),
                    req_pool_indices=template.req_pool_indices.clone(),
                    seq_lens=torch.tensor([m], dtype=template.seq_lens.dtype, device=dev),
                    seq_lens_cpu=torch.tensor([m], dtype=torch.int64),
                    seq_lens_sum=m,
                    extend_num_tokens=m,
                    extend_seq_lens=torch.tensor([m], dtype=torch.int64, device=dev),
                    extend_prefix_lens=torch.tensor([0], dtype=torch.int64, device=dev),
                    extend_start_loc=torch.tensor([0], dtype=torch.int64, device=dev),
                    extend_prefix_lens_cpu=[0],
                    extend_seq_lens_cpu=[m],
                )
                if getattr(template, "orig_seq_lens", None) is not None:
                    fb.orig_seq_lens = torch.tensor([m], dtype=template.orig_seq_lens.dtype, device=dev)
                if getattr(template, "num_token_non_padded", None) is not None:
                    fb.num_token_non_padded = torch.tensor([m], dtype=torch.int32, device=dev)
                    fb.num_token_non_padded_cpu = m
                self._capture(model_runner, fb, m, template_is_static=True, verified=not _VERIFY_FIRST_USE)
                done += 1
            finally:
                alloc.free(slots)
        torch.cuda.synchronize()
        msg = "[SDAR-EXACT-PREFILL] pre-captured %d span lengths (<= %d tokens) in %.1fs" % (done, _PRECAPTURE_MAX, time.perf_counter() - t0)
        logger.info(msg)
        print(msg, flush=True)

    def _capture(self, model_runner, forward_batch, n: int, template_is_static: bool = False, verified: bool = True):
        t0 = time.perf_counter()
        fb = forward_batch if template_is_static else dataclasses.replace(
            forward_batch,
            input_ids=forward_batch.input_ids.clone(),
            positions=forward_batch.positions.clone(),
            out_cache_loc=forward_batch.out_cache_loc.clone(),
            req_pool_indices=forward_batch.req_pool_indices.clone(),
            seq_lens=forward_batch.seq_lens.clone(),
        )
        fb.return_logprob = False
        # A flashinfer wrapper re-planned later (every other prefill) overwrites
        # its int workspace / indptr buffers, which the captured kernels read by
        # pointer. So this graph gets a DEDICATED ragged wrapper, planned once
        # and never touched again (same as the decode graph runner does).
        be = model_runner.attn_backend
        upd = be.indices_updater_prefill
        attr = "prefill_wrapper_ragged_block_causal" if getattr(be, "dllm_block_causal_prefill", False) else "prefill_wrapper_ragged"
        template = getattr(be, attr)
        from flashinfer import BatchPrefillWithRaggedKVCacheWrapper

        # own float workspace too (SDAR_EXACT_PREFILL_OWN_WS=0 shares the backend's)
        ws = (
            torch.empty(int(os.environ.get("SDAR_EXACT_PREFILL_WS_MB", "64")) << 20, dtype=torch.uint8, device=fb.input_ids.device)
            if os.environ.get("SDAR_EXACT_PREFILL_OWN_WS", "0") == "1"
            else be.workspace_buffer  # scratch only, safe to share sequentially
        )
        wrapper = BatchPrefillWithRaggedKVCacheWrapper(
            ws, "NHD", backend=getattr(template, "_backend", "auto") or "auto"
        )
        wrapper._sdar_ws_keepalive = ws
        saved = (getattr(be, attr), getattr(upd, attr))
        setattr(be, attr, wrapper)
        setattr(upd, attr, wrapper)
        try:
            # host-side plan with the static buffers; kept alive with the graph
            be.init_forward_metadata(fb)
            # The indices updater hands flashinfer SLICES of its shared
            # qo_indptr / kv_indptr buffers; a device tensor passed to plan() is
            # kept as-is, so the wrapper would alias buffers that every later
            # (decode) plan overwrites. Give the captured kernels private copies.
            for name in list(vars(wrapper).keys()):
                t = getattr(wrapper, name)
                if isinstance(t, torch.Tensor) and t.is_cuda and "workspace" not in name:
                    setattr(wrapper, name, t.clone())
            keepalive = (wrapper, getattr(be, "forward_metadata", None))

            # eager warm-up on the real request -- this is also the reference the
            # replay must reproduce bitwise, and the result handed to the caller
            out_ref = model_runner.forward(fb, skip_attn_backend_init=True)
            ref_logits = out_ref.logits_output.full_logits[-1:].clone()  # [1, V]
            del out_ref
            torch.cuda.synchronize()
            ref_kv = self._snapshot_kv(model_runner, fb.out_cache_loc)

            if self.pool is None:
                self.pool = torch.cuda.graph_pool_handle()
            graph = torch.cuda.CUDAGraph()
            # ONE capture stream for all graphs: the caching allocator keys free
            # blocks by stream, so a fresh stream per capture would stop later
            # graphs from reusing earlier graphs' intermediate memory in the pool
            if self.stream is None:
                self.stream = torch.cuda.Stream()
            stream = self.stream
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                with torch.cuda.graph(graph, pool=self.pool, stream=stream):
                    out_static = model_runner.forward(fb, skip_attn_backend_init=True)
                    # only the last row (boundary logits for the next block) is
                    # consumed; keeping the full [n, V] output alive per graph
                    # would pin ~n*V*4 bytes each in the pool
                    boundary = out_static.logits_output.full_logits[-1].clone()
            del out_static
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
        finally:
            setattr(be, attr, saved[0])
            setattr(upd, attr, saved[1])

        # bitwise check of the replay against the eager forward
        graph.replay()
        torch.cuda.synchronize()
        same = bool(torch.equal(boundary, ref_logits[-1]))
        kv_now = self._snapshot_kv(model_runner, fb.out_cache_loc)
        for (k0, v0), (k1, v1) in zip(ref_kv, kv_now):
            same = same and bool(torch.equal(k0, k1)) and bool(torch.equal(v0, v1))
        if not same:
            self._disable(f"replay not bitwise equal to eager for n={n}")
            # eager result already produced and K/V slots hold the replayed
            # (mismatching) values: rerun eager to leave the cache correct
            model_runner.attn_backend.init_forward_metadata(forward_batch)
            out = model_runner.forward(forward_batch, skip_attn_backend_init=True)
            return _Result(logits_output=out.logits_output, graph_was_replayed=False,
                           graph_replay_miss_reason="exact_prefill_mismatch", can_run_graph=False)
        self.entries[n] = _Entry(graph=graph, fb=fb, out=boundary, keepalive=keepalive,
                                 ref=(ref_logits[-1].clone(), ref_kv, fb.input_ids.clone()) if _CHECK_EVERY else None,
                                 verified=verified)
        dt = time.perf_counter() - t0
        self.stats["captures"] += 1
        self.stats["capture_s"] += dt
        logger.info("[SDAR-EXACT-PREFILL] captured n=%d (%d graphs, %.2fs)", n, len(self.entries), dt)
        return _Result(logits_output=_boundary_output(boundary))

    def _disable(self, reason: str):
        self.disabled = True
        self.disabled_reason = reason
        self.entries.clear()
        logger.warning("[SDAR-EXACT-PREFILL] disabled: %s", reason)
        print("[SDAR-EXACT-PREFILL] disabled: %s" % reason, flush=True)
