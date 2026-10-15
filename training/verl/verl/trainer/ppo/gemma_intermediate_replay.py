"""DiffusionGemma intermediate-step replay schema and deterministic batching."""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch

GEMMA_REPLAY_FORMAT = "gemma_intermediate_replay_v1"
GEMMA_REPLAY_TABLE_FORMAT = "gemma_intermediate_replay_table_handle_v1"
GEMMA_REPLAY_SCHEMA_VERSION = 2
GEMMA_REPLAY_SCHEMA_KEY = "gemma_replay_schema_version"

_SHM_FORMATS = {
    "gemma_intermediate_replay_shm_v1",
    "sglang_sdar_replay_shm_batch_v1",
    "split_streaming_t5_tensor_chunk_shm_v1",
}


@dataclass(frozen=True)
class GemmaReplaySample:
    payload: Any
    prompt_ids: torch.Tensor
    response_ids: torch.Tensor
    response_mask: torch.Tensor
    request_uid: str
    sample_uid: str
    response_uid: str


@dataclass(frozen=True)
class GemmaCorrectOnlySelection:
    """Response-level selection before replay tensor materialization."""

    samples: tuple[GemmaReplaySample, ...]
    source_responses: int
    kept_responses: int
    dropped_responses: int


def select_correct_gemma_replay_samples(
    samples: Sequence[GemmaReplaySample],
    correctness: Sequence[float | bool],
) -> GemmaCorrectOnlySelection:
    """Keep all captured rows for strict-correct responses and no others.

    Selection is intentionally response-level and happens before replay tensors
    are materialized.  It never ranks denoising states or applies GRPO/TPF
    weights.  The strict GSM8K adapter must provide exactly 0/1 correctness.
    """

    if len(samples) != len(correctness):
        raise ValueError(
            "DiffusionGemma correct-only selection length mismatch: "
            f"samples={len(samples)} correctness={len(correctness)}"
        )

    seen: set[str] = set()
    kept: list[GemmaReplaySample] = []
    for sample, raw_score in zip(samples, correctness, strict=True):
        response_uid = str(sample.response_uid)
        if not response_uid:
            raise RuntimeError(
                "DiffusionGemma correct-only baseline requires response_uid"
            )
        if response_uid in seen:
            raise RuntimeError(
                "DiffusionGemma correct-only baseline received duplicate "
                f"response_uid={response_uid!r}"
            )
        seen.add(response_uid)

        score = float(raw_score)
        if not math.isfinite(score) or score not in {0.0, 1.0}:
            raise RuntimeError(
                "DiffusionGemma correct-only baseline requires strict 0/1 "
                f"correctness, got {score!r} for response {response_uid!r}"
            )
        if score == 1.0:
            kept.append(sample)

    return GemmaCorrectOnlySelection(
        samples=tuple(kept),
        source_responses=len(samples),
        kept_responses=len(kept),
        dropped_responses=len(samples) - len(kept),
    )


@dataclass
class GemmaReplayTensorBatch:
    tensors: dict[str, torch.Tensor]
    row_manifest: list[dict[str, Any]]
    block_manifest: list[dict[str, Any]]
    metrics: dict[str, float]


@dataclass
class GemmaStreamWaveBatch:
    block_index: int
    stream_items: list[dict[str, Any]]
    metrics: dict[str, float]


def _sha256_int64_lanes(parts: Iterable[tuple[str, Any]]) -> tuple[int, int]:
    digest = hashlib.sha256()
    for label, value in parts:
        label_bytes = label.encode("utf-8")
        value_bytes = str(value).encode("utf-8")
        digest.update(len(label_bytes).to_bytes(4, "big"))
        digest.update(label_bytes)
        digest.update(len(value_bytes).to_bytes(8, "big"))
        digest.update(value_bytes)
    value = digest.digest()
    return (
        int.from_bytes(value[:8], "big", signed=True),
        int.from_bytes(value[8:16], "big", signed=True),
    )


def _stable_uid_hash_lanes(kind: str, value: Any) -> tuple[int, int]:
    return _sha256_int64_lanes(((kind, value),))


def stable_replay_identity_lanes(
    request_uid: Any,
    sample_uid: Any,
    response_uid: Any,
) -> tuple[int, int]:
    """Return two deterministic signed-int64 lanes for a replay identity."""
    return _sha256_int64_lanes(
        (
            ("request_uid", request_uid),
            ("sample_uid", sample_uid),
            ("response_uid", response_uid),
        )
    )


def _is_ray_object_ref(value: Any) -> bool:
    return value.__class__.__name__ == "ObjectRef" and value.__class__.__module__.startswith("ray")


def materialize_gemma_replay_payload(payload: Any) -> Any:
    """Resolve Ray/table handles while explicitly rejecting cross-node SHM."""
    if _is_ray_object_ref(payload):
        import ray

        return materialize_gemma_replay_payload(ray.get(payload))
    if isinstance(payload, list):
        return [materialize_gemma_replay_payload(item) for item in payload]
    if isinstance(payload, tuple):
        return tuple(materialize_gemma_replay_payload(item) for item in payload)
    if not isinstance(payload, dict):
        return payload

    payload_format = str(payload.get("format", ""))
    if payload_format in _SHM_FORMATS:
        raise RuntimeError(
            "DiffusionGemma cross-node replay does not support /dev/shm handles; "
            "use Ray tensors/table handles"
        )
    if payload_format == GEMMA_REPLAY_TABLE_FORMAT:
        ref = next(
            (
                payload.get(key)
                for key in ("ray_ref", "object_ref", "payload_ref", "tensor_payload", "payload")
                if payload.get(key) is not None
            ),
            None,
        )
        if ref is None:
            raise RuntimeError(
                "gemma_intermediate_replay_table_handle_v1 is missing a Ray payload reference"
            )
        resolved = materialize_gemma_replay_payload(ref)
        metadata = materialize_gemma_replay_payload(payload.get("metadata_payload", {}))
        if isinstance(resolved, dict) and isinstance(metadata, dict):
            return {**metadata, **resolved}
        return resolved

    return {key: materialize_gemma_replay_payload(value) for key, value in payload.items()}


class GemmaReplayResponseBuffer:
    """Collect block handles and release them according to the delivery policy."""

    def __init__(self, delivery: str = "final_response"):
        delivery = str(delivery).strip().lower()
        if delivery not in {"final_response", "block_stream"}:
            raise ValueError(
                "gemma_replay_delivery must be final_response or block_stream, "
                f"got {delivery!r}"
            )
        self.delivery = delivery
        self._responses: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _payload_blocks(payload: dict[str, Any]) -> list[dict[str, Any]]:
        blocks = payload.get("blocks")
        if blocks is None and payload.get("block_index") is not None:
            blocks = [payload]
        if not isinstance(blocks, (list, tuple)):
            raise RuntimeError("gemma_intermediate_replay_v1 payload is missing blocks")
        if not all(isinstance(block, dict) for block in blocks):
            raise RuntimeError("gemma_intermediate_replay_v1 blocks must be mappings")
        return list(blocks)

    @staticmethod
    def _semantic_blocks_equal(
        left: dict[str, Any],
        right: dict[str, Any],
    ) -> bool:
        semantic_keys = {
            "block_index",
            "block_size",
            "block_finalized",
            "final_target_ids",
            "target_valid_mask",
            "prefix_ids",
            "prefix_mask",
            "committed_prefix_ids",
            "committed_prefix_mask",
            "step_indices",
            "pre_forward_canvas_ids",
            "self_conditioning_signal",
            "self_conditioning_present",
            "valid_masks",
            "done_masks",
            "terminal_mask",
            "row_valid_mask",
            "steps_used",
            "captured_rows",
            "valid_token_count",
        }

        def equal_values(a: Any, b: Any) -> bool:
            if torch.is_tensor(a) or torch.is_tensor(b):
                if not torch.is_tensor(a) or not torch.is_tensor(b):
                    return False
                return (
                    a.shape == b.shape
                    and a.dtype == b.dtype
                    and torch.equal(a.detach().cpu(), b.detach().cpu())
                )
            if isinstance(a, dict) or isinstance(b, dict):
                if not isinstance(a, dict) or not isinstance(b, dict):
                    return False
                return a.keys() == b.keys() and all(
                    equal_values(a[key], b[key]) for key in a
                )
            if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
                if not isinstance(a, (list, tuple)) or not isinstance(
                    b, (list, tuple)
                ):
                    return False
                return len(a) == len(b) and all(
                    equal_values(x, y) for x, y in zip(a, b, strict=True)
                )
            return bool(a == b)

        keys = (left.keys() | right.keys()) & semantic_keys
        return all(
            key in left
            and key in right
            and equal_values(left[key], right[key])
            for key in keys
        )

    def add(self, payload: Any) -> list[dict[str, Any]]:
        payload = materialize_gemma_replay_payload(payload)
        if isinstance(payload, (list, tuple)):
            ready: list[dict[str, Any]] = []
            for item in payload:
                ready.extend(self.add(item))
            return ready
        if not isinstance(payload, dict) or payload.get("format") != GEMMA_REPLAY_FORMAT:
            raise RuntimeError(
                "DiffusionGemma producer replay is unavailable or has the wrong schema; "
                f"expected format={GEMMA_REPLAY_FORMAT!r}"
            )

        blocks = self._payload_blocks(payload)
        response_uid = str(
            payload.get("response_uid")
            or next((block.get("response_uid") for block in blocks if block.get("response_uid")), "")
        )
        if not response_uid:
            raise RuntimeError("DiffusionGemma replay payload is missing response_uid")

        state = self._responses.setdefault(
            response_uid,
            {
                "metadata": {},
                "blocks": {},
                "response_finished": False,
            },
        )
        metadata = {
            key: value
            for key, value in payload.items()
            if key not in {"blocks", "tensors", "tensor_handles"}
        }
        state["metadata"].update(metadata)
        for block in blocks:
            block_index = int(block["block_index"])
            if block_index in state["blocks"]:
                existing = state["blocks"][block_index]
                if self._semantic_blocks_equal(existing, block):
                    state["metadata"]["deduplicated_tp_blocks"] = int(
                        state["metadata"].get("deduplicated_tp_blocks", 0)
                    ) + 1
                    continue
                raise RuntimeError(
                    "conflicting duplicate DiffusionGemma replay block "
                    f"response_uid={response_uid} block={block_index}"
                )
            state["blocks"][block_index] = block
        state["response_finished"] = bool(
            state["response_finished"] or payload.get("response_finished", False)
        )

        if self.delivery == "block_stream":
            return [
                {
                    **state["metadata"],
                    "format": GEMMA_REPLAY_FORMAT,
                    "response_uid": response_uid,
                    "blocks": [state["blocks"].pop(index)],
                }
                for index in sorted(list(state["blocks"]))
            ]

        expected_blocks = int(state["metadata"].get("expected_blocks", 0) or 0)
        if expected_blocks <= 0:
            response_length = int(state["metadata"].get("response_length", 0) or 0)
            block_size = int(state["metadata"].get("block_size", 0) or 0)
            if response_length > 0 and block_size > 0:
                expected_blocks = (response_length + block_size - 1) // block_size
        if not state["response_finished"] or expected_blocks <= 0:
            return []
        expected_indices = list(range(expected_blocks))
        if sorted(state["blocks"]) != expected_indices:
            return []

        ready = {
            **state["metadata"],
            "format": GEMMA_REPLAY_FORMAT,
            "response_uid": response_uid,
            "response_finished": True,
            "expected_blocks": expected_blocks,
            "blocks": [state["blocks"][index] for index in expected_indices],
        }
        del self._responses[response_uid]
        return [ready]

    @property
    def pending_responses(self) -> int:
        return len(self._responses)


def finalize_gemma_replay_payload(payload: Any, delivery: str = "final_response") -> dict[str, Any]:
    """Return one finalized response envelope from producer output."""
    buffer = GemmaReplayResponseBuffer(delivery=delivery)
    ready = buffer.add(payload)
    if delivery == "final_response":
        if len(ready) != 1 or buffer.pending_responses:
            raise RuntimeError(
                "DiffusionGemma final_response delivery requires all replay blocks and "
                "response_finished=true before actor delivery"
            )
        return ready[0]
    if not ready:
        raise RuntimeError("DiffusionGemma block_stream payload contained no blocks")
    if len(ready) == 1:
        return ready[0]
    first = ready[0]
    return {
        **{key: value for key, value in first.items() if key != "blocks"},
        "blocks": [item["blocks"][0] for item in ready],
    }


def _as_cpu_tensor(value: Any, *, name: str, dtype: torch.dtype | None = None) -> torch.Tensor:
    if value is None:
        raise RuntimeError(f"DiffusionGemma replay is missing {name}")
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    value = value.detach().cpu().contiguous()
    return value.to(dtype=dtype) if dtype is not None else value


def _field(mapping: dict[str, Any], names: Iterable[str], *, required: bool = True) -> Any:
    tensors = mapping.get("tensors")
    for name in names:
        if name in mapping:
            return mapping[name]
        if isinstance(tensors, dict) and name in tensors:
            return tensors[name]
    if required:
        raise RuntimeError(f"DiffusionGemma replay is missing one of {tuple(names)}")
    return None


def _tensor_nbytes(value: Any, seen: set[int] | None = None) -> int:
    seen = seen if seen is not None else set()
    value_id = id(value)
    if value_id in seen:
        return 0
    seen.add(value_id)
    if torch.is_tensor(value):
        return int(value.numel()) * int(value.element_size())
    if isinstance(value, dict):
        return sum(_tensor_nbytes(item, seen) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_nbytes(item, seen) for item in value)
    return 0


def _gemma_replay_values_equal(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) or torch.is_tensor(right):
        return torch.is_tensor(left) and torch.is_tensor(right) and torch.equal(
            left.detach().cpu(),
            right.detach().cpu(),
        )
    if isinstance(left, dict) or isinstance(right, dict):
        if not isinstance(left, dict) or not isinstance(right, dict):
            return False
        if set(left) != set(right):
            return False
        return all(_gemma_replay_values_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
            return False
        if len(left) != len(right):
            return False
        return all(_gemma_replay_values_equal(a, b) for a, b in zip(left, right, strict=True))
    return left == right


class GemmaStreamWaveAlignedPacker:
    """Buffer streamed Gemma blocks into stable per-block FSDP waves."""

    def __init__(
        self,
        *,
        expected_sample_count: int,
        expected_block_count: int,
        allow_out_of_order: bool = False,
    ):
        self.expected_sample_count = int(expected_sample_count)
        self.expected_block_count = int(expected_block_count)
        if self.expected_sample_count <= 0 or self.expected_block_count <= 0:
            raise ValueError(
                "DiffusionGemma wave-aligned replay requires positive expected "
                f"samples/blocks, got {self.expected_sample_count}/{self.expected_block_count}"
            )
        self.allow_out_of_order = bool(allow_out_of_order)
        self._known_identities: set[tuple[str, str, str]] = set()
        self._expected_identities: tuple[tuple[str, str, str], ...] | None = None
        self._buckets: dict[int, dict[tuple[str, str, str], dict[str, Any]]] = {}
        self._seen_payloads: dict[tuple[tuple[str, str, str], int], dict[str, Any]] = {}
        self._bucket_first_ts: dict[int, float] = {}
        self._next_block_index = 0
        self._dispatched: set[int] = set()
        self._arrival_count = 0
        self._dedup_count = 0
        self._out_of_order_arrivals = 0
        self._buffered_peak_blocks = 0
        self._buffered_peak_bytes = 0
        self._dispatch_count = 0

    @staticmethod
    def _identity(payload: dict[str, Any], raw: dict[str, Any], fallback_index: int) -> tuple[str, str, str]:
        context = dict(raw.get("stream_context") or {})
        request_uid = str(
            payload.get("request_uid")
            or context.get("request_uid")
            or raw.get("request_id")
            or f"request-{fallback_index}"
        )
        sample_uid = str(payload.get("sample_uid") or context.get("sample_uid") or request_uid)
        response_uid = str(payload.get("response_uid") or context.get("response_uid") or request_uid)
        return request_uid, sample_uid, response_uid

    @staticmethod
    def _single_block_payload(payload: dict[str, Any], block: dict[str, Any]) -> dict[str, Any]:
        metadata = {
            key: value
            for key, value in payload.items()
            if key not in {"blocks", "tensors", "tensor_handles"}
        }
        return {
            **metadata,
            "format": GEMMA_REPLAY_FORMAT,
            "blocks": [block],
        }

    def _iter_payload_blocks(
        self,
        raw: dict[str, Any],
        item_index: int,
    ) -> list[tuple[tuple[str, str, str], int, dict[str, Any], dict[str, Any]]]:
        compact = raw.get("sglang_sdar_replay_tensors_compact")
        payload = materialize_gemma_replay_payload(compact)
        if isinstance(payload, (list, tuple)):
            payloads = list(payload)
        else:
            payloads = [payload]
        result = []
        for payload_item in payloads:
            if not isinstance(payload_item, dict) or payload_item.get("format") != GEMMA_REPLAY_FORMAT:
                raise RuntimeError(
                    "DiffusionGemma wave-aligned stream item did not materialize "
                    f"{GEMMA_REPLAY_FORMAT}"
                )
            identity = self._identity(payload_item, raw, item_index)
            for block in GemmaReplayResponseBuffer._payload_blocks(payload_item):
                block_index = int(block.get("block_index", -1))
                if not 0 <= block_index < self.expected_block_count:
                    raise RuntimeError(
                        "DiffusionGemma wave-aligned stream received unexpected block_index="
                        f"{block_index}; expected [0, {self.expected_block_count})"
                    )
                result.append(
                    (
                        identity,
                        block_index,
                        self._single_block_payload(payload_item, block),
                        block,
                    )
                )
        return result

    def _maybe_freeze_expected_identities(
        self,
        bucket: dict[tuple[str, str, str], dict[str, Any]] | None = None,
    ) -> None:
        if self._expected_identities is not None:
            return
        source = set(bucket or self._known_identities)
        if len(source) < self.expected_sample_count:
            return
        if len(source) != self.expected_sample_count:
            raise RuntimeError(
                "DiffusionGemma wave-aligned stream saw wrong response count: "
                f"{len(source)} != {self.expected_sample_count}"
            )
        self._expected_identities = tuple(sorted(source))

    def _is_complete(self, block_index: int) -> bool:
        bucket = self._buckets.get(int(block_index), {})
        self._maybe_freeze_expected_identities(bucket)
        if self._expected_identities is None:
            return False
        return tuple(sorted(bucket)) == self._expected_identities

    def _update_buffered_peaks(self) -> None:
        buffered_blocks = sum(len(bucket) for bucket in self._buckets.values())
        buffered_bytes = sum(
            _tensor_nbytes(item.get("sglang_sdar_replay_tensors_compact"))
            for bucket in self._buckets.values()
            for item in bucket.values()
        )
        self._buffered_peak_blocks = max(self._buffered_peak_blocks, buffered_blocks)
        self._buffered_peak_bytes = max(self._buffered_peak_bytes, buffered_bytes)

    def _ready_block_indices(self) -> list[int]:
        if self._expected_identities is None:
            return []
        if self.allow_out_of_order:
            return [
                block_index
                for block_index in sorted(self._buckets)
                if block_index not in self._dispatched and self._is_complete(block_index)
            ]
        ready = []
        while self._next_block_index < self.expected_block_count and self._is_complete(self._next_block_index):
            ready.append(self._next_block_index)
            self._next_block_index += 1
        return ready

    def _dispatch(self, block_index: int) -> GemmaStreamWaveBatch:
        bucket = self._buckets.pop(int(block_index))
        self._dispatched.add(int(block_index))
        wait_seconds = max(0.0, time.time() - float(self._bucket_first_ts.pop(int(block_index), time.time())))
        self._dispatch_count += 1
        assert self._expected_identities is not None
        stream_items = [bucket[identity] for identity in self._expected_identities]
        metrics = {
            "gemma_stream_wave_aligned/enabled": 1.0,
            "gemma_stream_wave_aligned/expected_responses": float(self.expected_sample_count),
            "gemma_stream_wave_aligned/expected_blocks": float(self.expected_block_count),
            "gemma_stream_wave_aligned/dispatch_count": 1.0,
            "gemma_stream_wave_aligned/dedup_count": float(self._dedup_count),
            "gemma_stream_wave_aligned/out_of_order_arrivals": float(self._out_of_order_arrivals),
            "gemma_stream_wave_aligned/buffered_peak_blocks": float(self._buffered_peak_blocks),
            "gemma_stream_wave_aligned/buffered_peak_bytes": float(self._buffered_peak_bytes),
            "gemma_stream_wave_aligned/wait_seconds": wait_seconds,
            "gemma_stream_wave_aligned/wait_seconds_max": wait_seconds,
            "gemma_stream_wave_aligned/final_coverage": float(
                len(self._dispatched) / float(self.expected_block_count)
            ),
            f"gemma_stream_wave_aligned/block_{block_index}/bucket_size": float(len(stream_items)),
        }
        return GemmaStreamWaveBatch(
            block_index=int(block_index),
            stream_items=stream_items,
            metrics=metrics,
        )

    def add_stream_items(self, stream_items: list[dict[str, Any]]) -> list[GemmaStreamWaveBatch]:
        ready: list[GemmaStreamWaveBatch] = []
        for raw in stream_items:
            if not isinstance(raw, dict) or raw.get("format") != "sglang_sdar_stream_chunk_v1":
                raise RuntimeError("DiffusionGemma wave-aligned packer received a non-stream queue item")
            item_index = self._arrival_count
            self._arrival_count += 1
            for identity, block_index, payload, block in self._iter_payload_blocks(raw, item_index):
                if self._expected_identities is not None and identity not in self._expected_identities:
                    raise RuntimeError(
                        "DiffusionGemma wave-aligned stream saw unexpected response identity "
                        f"{identity}; expected={self._expected_identities}"
                    )
                self._known_identities.add(identity)
                if len(self._known_identities) > self.expected_sample_count:
                    raise RuntimeError(
                        "DiffusionGemma wave-aligned stream saw too many response identities: "
                        f"{len(self._known_identities)} > {self.expected_sample_count}"
                    )
                self._maybe_freeze_expected_identities()

                seen_key = (identity, int(block_index))
                existing_payload = self._seen_payloads.get(seen_key)
                if existing_payload is not None:
                    if _gemma_replay_values_equal(existing_payload, payload):
                        self._dedup_count += 1
                        continue
                    raise RuntimeError(
                        "conflicting duplicate DiffusionGemma wave-aligned replay block "
                        f"identity={identity} block={block_index}"
                    )

                if int(block_index) < self._next_block_index and not self.allow_out_of_order:
                    raise RuntimeError(
                        "DiffusionGemma wave-aligned stream received a new block after its "
                        f"wave dispatched: identity={identity} block={block_index}"
                    )
                if int(block_index) > self._next_block_index:
                    self._out_of_order_arrivals += 1

                raw_for_builder = dict(raw)
                raw_for_builder["sglang_sdar_replay_tensors_compact"] = payload
                bucket = self._buckets.setdefault(int(block_index), {})
                if identity in bucket:
                    raise RuntimeError(
                        "duplicate DiffusionGemma wave-aligned bucket entry "
                        f"identity={identity} block={block_index}"
                    )
                bucket[identity] = raw_for_builder
                self._seen_payloads[seen_key] = payload
                self._bucket_first_ts.setdefault(int(block_index), time.time())
                # Touch block so malformed mappings fail before being buffered.
                int(block["block_index"])
            self._update_buffered_peaks()

        for block_index in self._ready_block_indices():
            ready.append(self._dispatch(block_index))
        return ready

    def close(self) -> list[GemmaStreamWaveBatch]:
        ready = [self._dispatch(block_index) for block_index in self._ready_block_indices()]
        if len(self._known_identities) != self.expected_sample_count:
            raise RuntimeError(
                "DiffusionGemma wave-aligned stream closed with wrong response count: "
                f"{len(self._known_identities)} != {self.expected_sample_count}"
            )
        if self._expected_identities is None:
            self._maybe_freeze_expected_identities()
        missing_blocks = [
            block_index
            for block_index in range(self.expected_block_count)
            if block_index not in self._dispatched
        ]
        if missing_blocks:
            coverage = {
                block_index: sorted(self._buckets.get(block_index, {}))
                for block_index in missing_blocks
            }
            raise RuntimeError(
                "DiffusionGemma wave-aligned stream closed with incomplete buckets: "
                f"missing_blocks={missing_blocks} coverage={coverage}"
            )
        if self._buckets:
            raise RuntimeError(
                "DiffusionGemma wave-aligned stream closed with unexpected buffered blocks: "
                f"{sorted(self._buckets)}"
            )
        return ready


def _masked_ids(ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    ids = ids.detach().cpu().long().view(-1)
    mask = mask.detach().cpu().bool().view(-1)
    if ids.shape != mask.shape:
        raise RuntimeError("DiffusionGemma final response ids/mask shape mismatch")
    return ids[mask]


def _normalize_block(
    *,
    block: dict[str, Any],
    sample: GemmaReplaySample,
    block_size: int,
    expected_blocks: int,
) -> dict[str, Any]:
    block_index = int(block.get("block_index", -1))
    if not 0 <= block_index < expected_blocks:
        raise RuntimeError(
            f"DiffusionGemma replay block_index={block_index} outside [0, {expected_blocks})"
        )
    if int(block.get("block_size", block_size) or block_size) != block_size:
        raise RuntimeError("DiffusionGemma replay block_size does not match actor block_size")
    if not bool(block.get("block_finalized", True)):
        raise RuntimeError(
            f"DiffusionGemma replay block {block_index} was exposed before its final target was committed"
        )

    response_ids = sample.response_ids.detach().cpu().long().view(-1)
    response_mask = sample.response_mask.detach().cpu().bool().view(-1)
    block_start = block_index * block_size
    block_end = block_start + block_size
    expected_target = response_ids[block_start:block_end]
    expected_target_mask = response_mask[block_start:block_end]
    if int(expected_target.numel()) < block_size:
        pad = block_size - int(expected_target.numel())
        expected_target = torch.cat((expected_target, torch.full((pad,), -100, dtype=torch.long)))
        expected_target_mask = torch.cat((expected_target_mask, torch.zeros((pad,), dtype=torch.bool)))

    target_ids = _as_cpu_tensor(
        _field(block, ("final_target_ids", "target_ids")),
        name="final_target_ids",
        dtype=torch.long,
    ).view(-1)
    target_mask_value = _field(
        block,
        ("target_valid_mask", "final_target_valid_mask"),
        required=False,
    )
    target_mask = (
        _as_cpu_tensor(target_mask_value, name="target_valid_mask", dtype=torch.bool).view(-1)
        if target_mask_value is not None
        else expected_target_mask.clone()
    )
    if tuple(target_ids.shape) != (block_size,) or tuple(target_mask.shape) != (block_size,):
        raise RuntimeError(
            "DiffusionGemma block target tensors must have shape "
            f"[{block_size}], got ids={tuple(target_ids.shape)} mask={tuple(target_mask.shape)}"
        )
    if not torch.equal(target_mask, expected_target_mask):
        raise RuntimeError(
            f"DiffusionGemma block {block_index} target mask does not match final response"
        )
    if not torch.equal(target_ids[target_mask], expected_target[target_mask]):
        raise RuntimeError(
            f"DiffusionGemma block {block_index} target is not its own final committed clean block"
        )

    prompt_ids = sample.prompt_ids.detach().cpu().long().view(-1)
    committed_response = _masked_ids(response_ids[:block_start], response_mask[:block_start])
    expected_prefix = torch.cat((prompt_ids, committed_response), dim=0)
    prefix_ids_value = _field(block, ("prefix_ids",), required=False)
    if prefix_ids_value is not None:
        prefix_ids = _as_cpu_tensor(prefix_ids_value, name="prefix_ids", dtype=torch.long).view(-1)
        prefix_mask_value = _field(block, ("prefix_mask",), required=False)
        if prefix_mask_value is not None:
            prefix_mask = _as_cpu_tensor(
                prefix_mask_value,
                name="prefix_mask",
                dtype=torch.bool,
            ).view(-1)
            if prefix_mask.shape != prefix_ids.shape:
                raise RuntimeError("DiffusionGemma prefix_ids/prefix_mask shape mismatch")
            prefix_ids = prefix_ids[prefix_mask]
        if not torch.equal(prefix_ids, expected_prefix):
            raise RuntimeError(
                f"DiffusionGemma block {block_index} prefix is not the exact committed rollout prefix"
            )
    else:
        committed_ids = _as_cpu_tensor(
            _field(block, ("committed_prefix_ids", "committed_response_ids")),
            name="committed_prefix_ids",
            dtype=torch.long,
        ).view(-1)
        committed_mask_value = _field(block, ("committed_prefix_mask",), required=False)
        if committed_mask_value is not None:
            committed_mask = _as_cpu_tensor(
                committed_mask_value,
                name="committed_prefix_mask",
                dtype=torch.bool,
            ).view(-1)
            if committed_mask.shape != committed_ids.shape:
                raise RuntimeError("DiffusionGemma committed prefix ids/mask shape mismatch")
            committed_ids = committed_ids[committed_mask]
        if not torch.equal(committed_ids, committed_response):
            raise RuntimeError(
                f"DiffusionGemma block {block_index} committed response prefix is not exact"
            )
        prefix_ids = expected_prefix
    if prefix_ids.numel() == 0:
        raise RuntimeError("DiffusionGemma replay cannot build a prefix cache from an empty prefix")

    step_indices = _as_cpu_tensor(
        _field(block, ("step_indices", "intermediate_step_indices")),
        name="step_indices",
        dtype=torch.long,
    ).view(-1)
    canvas_ids = _as_cpu_tensor(
        _field(block, ("pre_forward_canvas_ids", "canvas_ids")),
        name="pre_forward_canvas_ids",
        dtype=torch.long,
    )
    valid_masks = _as_cpu_tensor(
        _field(block, ("valid_masks", "valid_mask")),
        name="valid_masks",
        dtype=torch.bool,
    )
    done_masks = _as_cpu_tensor(
        _field(block, ("done_masks", "done_mask")),
        name="done_masks",
        dtype=torch.bool,
    )
    self_conditioning = _as_cpu_tensor(
        _field(block, ("self_conditioning_signal", "self_conditioning_signals")),
        name="self_conditioning_signal",
    )
    row_valid_value = _field(
        block,
        ("row_valid_mask", "intermediate_row_valid_mask"),
        required=False,
    )
    row_valid = (
        _as_cpu_tensor(row_valid_value, name="row_valid_mask", dtype=torch.bool).view(-1)
        if row_valid_value is not None
        else torch.ones_like(step_indices, dtype=torch.bool)
    )
    sc_present_value = _field(
        block,
        ("self_conditioning_present", "self_conditioning_present_mask"),
        required=False,
    )
    sc_present = (
        _as_cpu_tensor(
            sc_present_value,
            name="self_conditioning_present",
            dtype=torch.bool,
        ).view(-1)
        if sc_present_value is not None
        else self_conditioning.float().abs().sum(dim=(-1, -2)).gt(0)
    )

    steps = int(step_indices.numel())
    expected_step_shape = (steps, block_size)
    if tuple(canvas_ids.shape) != expected_step_shape:
        raise RuntimeError(
            f"DiffusionGemma canvas shape must be {expected_step_shape}, got {tuple(canvas_ids.shape)}"
        )
    if tuple(valid_masks.shape) != expected_step_shape or tuple(done_masks.shape) != expected_step_shape:
        raise RuntimeError("DiffusionGemma valid/done masks must align with canvas rows")
    if self_conditioning.ndim != 3 or tuple(self_conditioning.shape[:2]) != expected_step_shape:
        raise RuntimeError(
            "DiffusionGemma compact self-conditioning signal must be "
            f"[steps, block_size, hidden], got {tuple(self_conditioning.shape)}"
        )
    if tuple(row_valid.shape) != (steps,) or tuple(sc_present.shape) != (steps,):
        raise RuntimeError("DiffusionGemma row/self-conditioning masks must have one value per step")

    real_step_indices = step_indices[row_valid]
    if int(torch.unique(real_step_indices).numel()) != int(real_step_indices.numel()):
        raise RuntimeError(
            f"DiffusionGemma block {block_index} contains duplicate intermediate step indices"
        )
    # Keep semantic rows sorted by step while moving transport-padding rows to
    # the tail. Sorting raw indices would place the conventional -1 padding row
    # first and break row-slot alignment for canvas/self-conditioning tensors.
    max_step = int(step_indices[row_valid].max().item()) + 1 if bool(row_valid.any()) else 0
    sort_key = torch.where(row_valid, step_indices, torch.full_like(step_indices, max_step))
    order = torch.argsort(sort_key, stable=True)
    return {
        "block_index": block_index,
        "prefix_ids": prefix_ids,
        "target_ids": target_ids,
        "target_mask": target_mask,
        "step_indices": step_indices[order],
        "canvas_ids": canvas_ids[order],
        "valid_masks": valid_masks[order],
        "done_masks": done_masks[order],
        "self_conditioning": self_conditioning[order],
        "row_valid": row_valid[order],
        "sc_present": sc_present[order],
        "source_bytes": _tensor_nbytes(block),
        "timing": dict(block.get("timing", {}) or {}),
    }


def build_gemma_replay_tensor_batch(
    samples: list[GemmaReplaySample],
    *,
    response_length: int,
    block_size: int,
    delivery: str = "final_response",
    pad_token_id: int = 0,
) -> GemmaReplayTensorBatch:
    """Validate complete trajectories and build block-major actor tensors."""
    if response_length <= 0 or block_size <= 0 or response_length % block_size != 0:
        raise ValueError(
            "DiffusionGemma replay requires positive response_length divisible by block_size"
        )
    expected_blocks = response_length // block_size
    normalized_blocks: list[tuple[GemmaReplaySample, dict[str, Any]]] = []
    source_bytes = 0
    timing_totals: dict[str, float] = {}

    ordered_samples = sorted(
        samples,
        key=lambda item: (item.request_uid, item.sample_uid, item.response_uid),
    )
    for sample in ordered_samples:
        payload = finalize_gemma_replay_payload(sample.payload, delivery=delivery)
        if delivery == "final_response" and not bool(payload.get("response_finished", False)):
            raise RuntimeError("DiffusionGemma replay reached the actor before the response finished")
        active_positions = torch.nonzero(
            sample.response_mask.detach().cpu().bool().view(-1),
            as_tuple=False,
        ).flatten()
        active_extent = int(active_positions[-1].item()) + 1 if active_positions.numel() else 0
        sample_expected_blocks = max(
            1,
            (active_extent + block_size - 1) // block_size,
        )
        payload_expected = int(
            payload.get("expected_blocks", sample_expected_blocks)
            or sample_expected_blocks
        )
        expected_payload_blocks = (
            sample_expected_blocks if delivery == "final_response" else expected_blocks
        )
        if payload_expected != expected_payload_blocks:
            raise RuntimeError(
                "DiffusionGemma replay block count does not match the finalized response: "
                f"response_blocks={expected_payload_blocks} payload_blocks={payload_expected}"
            )
        blocks = GemmaReplayResponseBuffer._payload_blocks(payload)
        indices = sorted(int(block.get("block_index", -1)) for block in blocks)
        if len(indices) != len(set(indices)) or any(
            index < 0 or index >= expected_blocks for index in indices
        ):
            raise RuntimeError(
                f"DiffusionGemma response {sample.response_uid} has invalid/duplicate blocks {indices}"
            )
        if delivery == "final_response" and indices != list(
            range(sample_expected_blocks)
        ):
            raise RuntimeError(
                f"DiffusionGemma response {sample.response_uid} must contain blocks "
                f"0..{sample_expected_blocks - 1}, got {indices}"
            )
        if delivery == "block_stream" and not indices:
            raise RuntimeError(
                f"DiffusionGemma block-stream response {sample.response_uid} contains no blocks"
            )
        by_index = {int(block["block_index"]): block for block in blocks}
        for block_index in indices:
            normalized = _normalize_block(
                block=by_index[block_index],
                sample=sample,
                block_size=block_size,
                expected_blocks=expected_blocks,
            )
            normalized_blocks.append((sample, normalized))
            source_bytes += int(normalized["source_bytes"])
            for key, value in normalized["timing"].items():
                if isinstance(value, (int, float)):
                    timing_totals[str(key)] = timing_totals.get(str(key), 0.0) + float(value)

    if not normalized_blocks:
        raise RuntimeError("DiffusionGemma replay batch contains no blocks")
    hidden_sizes = {int(block["self_conditioning"].shape[-1]) for _, block in normalized_blocks}
    if len(hidden_sizes) != 1:
        raise RuntimeError(
            f"DiffusionGemma self-conditioning hidden size differs across blocks: {hidden_sizes}"
        )
    hidden_size = hidden_sizes.pop()
    max_prefix = max(int(block["prefix_ids"].numel()) for _, block in normalized_blocks)
    max_steps = max(int(block["step_indices"].numel()) for _, block in normalized_blocks)
    if max_steps <= 0:
        raise RuntimeError("DiffusionGemma replay batch contains no intermediate rows")

    block_count = len(normalized_blocks)
    input_ids = torch.full((block_count, max_prefix), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((block_count, max_prefix), dtype=torch.bool)
    position_ids = torch.zeros((block_count, max_prefix), dtype=torch.long)
    responses = torch.full((block_count, block_size), -100, dtype=torch.long)
    response_mask = torch.zeros((block_count, block_size), dtype=torch.bool)
    canvas_ids = torch.full(
        (block_count, max_steps, block_size),
        int(pad_token_id),
        dtype=torch.long,
    )
    step_indices = torch.full((block_count, max_steps), -1, dtype=torch.long)
    row_valid = torch.zeros((block_count, max_steps), dtype=torch.bool)
    valid_masks = torch.zeros((block_count, max_steps, block_size), dtype=torch.bool)
    done_masks = torch.zeros((block_count, max_steps, block_size), dtype=torch.bool)
    self_conditioning = torch.zeros(
        (block_count, max_steps, block_size, hidden_size),
        dtype=normalized_blocks[0][1]["self_conditioning"].dtype,
    )
    sc_present = torch.zeros((block_count, max_steps), dtype=torch.bool)
    block_indices = torch.empty((block_count,), dtype=torch.long)
    sample_indices = torch.empty((block_count,), dtype=torch.long)
    request_hashes = torch.empty((block_count, 2), dtype=torch.long)
    sample_hashes = torch.empty((block_count, 2), dtype=torch.long)
    response_hashes = torch.empty((block_count, 2), dtype=torch.long)
    identity_hashes = torch.empty((block_count, 2), dtype=torch.long)
    block_real = torch.ones((block_count,), dtype=torch.bool)
    reward = torch.full((block_count,), float("nan"), dtype=torch.float32)
    reward_ready = torch.zeros((block_count,), dtype=torch.bool)
    advantages = torch.full((block_count, max_steps), float("nan"), dtype=torch.float32)

    row_manifest: list[dict[str, Any]] = []
    block_manifest: list[dict[str, Any]] = []
    rows_by_block: dict[int, int] = {}
    rows_by_step: dict[int, int] = {}
    rows_by_block_step: dict[tuple[int, int], int] = {}
    expected_identities: set[tuple[str, str, str, int, int]] = set()

    sample_order = {
        (sample.request_uid, sample.sample_uid, sample.response_uid): index
        for index, sample in enumerate(ordered_samples)
    }
    for slot, (sample, block) in enumerate(normalized_blocks):
        prefix_len = int(block["prefix_ids"].numel())
        steps = int(block["step_indices"].numel())
        input_ids[slot, :prefix_len] = block["prefix_ids"]
        attention_mask[slot, :prefix_len] = True
        position_ids[slot, :prefix_len] = torch.arange(prefix_len, dtype=torch.long)
        responses[slot] = block["target_ids"]
        response_mask[slot] = block["target_mask"]
        canvas_ids[slot, :steps] = block["canvas_ids"]
        step_indices[slot, :steps] = block["step_indices"]
        row_valid[slot, :steps] = block["row_valid"]
        valid_masks[slot, :steps] = block["valid_masks"]
        done_masks[slot, :steps] = block["done_masks"]
        self_conditioning[slot, :steps] = block["self_conditioning"].to(self_conditioning.dtype)
        sc_present[slot, :steps] = block["sc_present"]
        block_indices[slot] = int(block["block_index"])
        sample_indices[slot] = sample_order[(sample.request_uid, sample.sample_uid, sample.response_uid)]
        request_hashes[slot] = torch.tensor(
            _stable_uid_hash_lanes("request_uid", sample.request_uid),
            dtype=torch.long,
        )
        sample_hashes[slot] = torch.tensor(
            _stable_uid_hash_lanes("sample_uid", sample.sample_uid),
            dtype=torch.long,
        )
        response_hashes[slot] = torch.tensor(
            _stable_uid_hash_lanes("response_uid", sample.response_uid),
            dtype=torch.long,
        )
        identity_hashes[slot] = torch.tensor(
            stable_replay_identity_lanes(
                sample.request_uid,
                sample.sample_uid,
                sample.response_uid,
            ),
            dtype=torch.long,
        )

        real_rows = int(block["row_valid"].sum().item())
        block_manifest.append(
            {
                "request_uid": sample.request_uid,
                "sample_uid": sample.sample_uid,
                "response_uid": sample.response_uid,
                "sample_index": int(sample_indices[slot].item()),
                "block_index": int(block["block_index"]),
                "block_slot": slot,
                "prefix_length": prefix_len,
                "intermediate_rows": real_rows,
            }
        )
        for local_row in range(steps):
            if not bool(block["row_valid"][local_row]):
                continue
            step_index = int(block["step_indices"][local_row].item())
            identity = (
                sample.request_uid,
                sample.sample_uid,
                sample.response_uid,
                int(block["block_index"]),
                step_index,
            )
            if identity in expected_identities:
                raise RuntimeError(f"duplicate DiffusionGemma replay row identity {identity}")
            expected_identities.add(identity)
            token_count = int(
                (
                    block["valid_masks"][local_row]
                    & block["target_mask"]
                ).sum().item()
            )
            row_manifest.append(
                {
                    "request_uid": sample.request_uid,
                    "sample_uid": sample.sample_uid,
                    "response_uid": sample.response_uid,
                    "sample_index": int(sample_indices[slot].item()),
                    "block_index": int(block["block_index"]),
                    "block_slot": slot,
                    "step_index": step_index,
                    "step_slot": local_row,
                    "active_token_steps": token_count,
                    "reward_attachment": None,
                    "advantage_attachment": None,
                }
            )
            rows_by_block[int(block["block_index"])] = (
                rows_by_block.get(int(block["block_index"]), 0) + 1
            )
            rows_by_step[step_index] = rows_by_step.get(step_index, 0) + 1
            block_step = (int(block["block_index"]), step_index)
            rows_by_block_step[block_step] = rows_by_block_step.get(block_step, 0) + 1

    token_mask = valid_masks & response_mask.unsqueeze(1) & row_valid.unsqueeze(-1)
    intermediate_rows = int(row_valid.sum().item())
    active_token_steps = int(token_mask.sum().item())
    if len(row_manifest) != intermediate_rows or len(expected_identities) != intermediate_rows:
        raise RuntimeError(
            "DiffusionGemma replay row accounting mismatch: "
            f"manifest={len(row_manifest)} unique={len(expected_identities)} tensor={intermediate_rows}"
        )
    manifest_tokens = sum(int(row["active_token_steps"]) for row in row_manifest)
    if manifest_tokens != active_token_steps:
        raise RuntimeError(
            "DiffusionGemma replay token accounting mismatch: "
            f"manifest={manifest_tokens} tensor={active_token_steps}"
        )

    tensors = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "responses": responses,
        "response_mask": response_mask,
        GEMMA_REPLAY_SCHEMA_KEY: torch.full(
            (block_count,),
            GEMMA_REPLAY_SCHEMA_VERSION,
            dtype=torch.long,
        ),
        "gemma_replay_canvas_ids": canvas_ids,
        "gemma_replay_step_indices": step_indices,
        "gemma_replay_row_valid": row_valid,
        "gemma_replay_valid_masks": valid_masks,
        "gemma_replay_done_masks": done_masks,
        "gemma_replay_self_conditioning_signal": self_conditioning,
        "gemma_replay_self_conditioning_present": sc_present,
        "gemma_replay_block_indices": block_indices,
        "gemma_replay_sample_indices": sample_indices,
        "gemma_replay_request_uid_hashes": request_hashes,
        "gemma_replay_sample_uid_hashes": sample_hashes,
        "gemma_replay_response_uid_hashes": response_hashes,
        "gemma_replay_identity_hashes": identity_hashes,
        "gemma_replay_block_real": block_real,
        # Reserved attachment points for future final-reward RL. They are not
        # read by the CE path.
        "gemma_replay_reward": reward,
        "gemma_replay_reward_ready": reward_ready,
        "gemma_replay_advantages": advantages,
    }
    tensor_bytes = _tensor_nbytes(tensors)
    metrics: dict[str, float] = {
        "gemma_replay/total_responses": float(len(ordered_samples)),
        "gemma_replay/total_blocks": float(block_count),
        "gemma_replay/intermediate_rows": float(intermediate_rows),
        "gemma_replay/active_token_steps": float(active_token_steps),
        "gemma_replay/row_coverage": 1.0,
        "gemma_replay/token_coverage": 1.0,
        "gemma_replay/target_mapping_coverage": 1.0,
        "gemma_replay/coverage": 1.0,
        "gemma_replay/coverage_percent": 100.0,
        "gemma_replay/source_tensor_bytes": float(source_bytes),
        "gemma_replay/actor_tensor_bytes": float(tensor_bytes),
        "gemma_replay/max_steps_per_block": float(max_steps),
        "gemma_replay/max_prefix_tokens": float(max_prefix),
        "gemma_replay/final_response_delivery": 1.0 if delivery == "final_response" else 0.0,
        "gemma_replay/block_stream_delivery": 1.0 if delivery == "block_stream" else 0.0,
        "gemma_replay/clean_filter_enabled": 0.0,
        "gemma_replay/reward_gating_enabled": 0.0,
        "gemma_replay/advantage_gating_enabled": 0.0,
        "gemma_replay/future_rl_attachment_ready": 1.0,
    }
    for block_index in range(expected_blocks):
        metrics[f"gemma_replay/block_{block_index}/rows"] = float(
            rows_by_block.get(block_index, 0)
        )
        metrics[f"gemma_replay/block_{block_index}/coverage"] = (
            1.0 if block_index in rows_by_block else 0.0
        )
    for step_index, count in sorted(rows_by_step.items()):
        metrics[f"gemma_replay/step_{step_index}/rows"] = float(count)
    for (block_index, step_index), count in sorted(rows_by_block_step.items()):
        metrics[f"gemma_replay/block_{block_index}/step_{step_index}/rows"] = float(count)
    for key, value in timing_totals.items():
        metrics[f"gemma_replay/producer_timing/{key}"] = float(value)

    return GemmaReplayTensorBatch(
        tensors=tensors,
        row_manifest=row_manifest,
        block_manifest=block_manifest,
        metrics=metrics,
    )
