"""Safety and profiling helpers for Qwen/SDAR FSDP1 weight streaming.

The feature is intentionally narrow:

* actor model type ``sdar`` / architecture ``SDARForCausalLM``;
* FSDP1 with one explicit FSDP wrapper per ``SDARDecoderLayer``;
* colocated SGLang rollout using the naive checkpoint backend.

It does not alias FSDP and SGLang storage. It avoids a full-model CUDA
``state_dict`` allocation by summoning the root-owned parameters and each
transformer layer as deterministic, non-overlapping FSDP units.
"""

from __future__ import annotations

import hashlib
import os
import time
import uuid
from collections.abc import Awaitable, Callable, Generator, Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import nn

from verl.utils.model import convert_weight_keys


def shared_weights_enabled() -> bool:
    """Return whether shared-weight profiling is enabled."""
    return os.getenv("VERL_ACTOR_ROLLOUT_SHARED_WEIGHTS", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def stream_export_enabled() -> bool:
    """Return whether Qwen/SDAR layered export is enabled."""
    return os.getenv("VERL_ACTOR_ROLLOUT_STREAM_EXPORT", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def hbm_snapshot() -> dict[str, int]:
    """Return current CUDA allocator counters in bytes; safe without CUDA."""
    if not torch.cuda.is_available():
        return {
            "allocated": 0,
            "reserved": 0,
            "max_allocated": 0,
            "max_reserved": 0,
            "free": 0,
            "total": 0,
        }

    device = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(device)
    return {
        "allocated": int(torch.cuda.memory_allocated(device)),
        "reserved": int(torch.cuda.memory_reserved(device)),
        "max_allocated": int(torch.cuda.max_memory_allocated(device)),
        "max_reserved": int(torch.cuda.max_memory_reserved(device)),
        "free": int(free),
        "total": int(total),
    }


class SharedWeightMetrics:
    """Per-worker counters for one or more actor-to-rollout hot syncs."""

    def __init__(self, actor_uuid: str) -> None:
        self.actor_uuid = actor_uuid
        self.instance_uuid = uuid.uuid4().hex
        self.weight_version = 0
        self.fallback_count = 0
        self.failure_count = 0
        self.total_allgather_bytes = 0
        self.total_copy_bytes = 0

    def begin_hot_sync(self) -> dict[str, Any]:
        """Capture pre-sync memory and reset peak counters for this sync."""
        snapshot = hbm_snapshot()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())
        return {
            "metrics_uuid": self.instance_uuid,
            "weight_version": self.weight_version,
            "hbm_before_allocated_bytes": snapshot["allocated"],
            "hbm_before_reserved_bytes": snapshot["reserved"],
            "hbm_before_free_bytes": snapshot["free"],
            "hbm_total_bytes": snapshot["total"],
        }

    def on_allgather_done(
        self,
        *,
        allgather_bytes: int,
        allgather_seconds: float,
        stream_export: bool,
        fsdp_layers_streamed: int,
        fsdp_units_streamed: int = 0,
        params_streamed: int = 0,
        fallback_reason: Optional[str],
    ) -> dict[str, Any]:
        """Record export metrics after the weight iterable is fully consumed."""
        self.total_allgather_bytes += int(allgather_bytes)
        if fallback_reason:
            self.fallback_count += 1
        snapshot = hbm_snapshot()
        return {
            "weight_version": self.weight_version,
            "allgather_bytes": int(allgather_bytes),
            "allgather_seconds": float(allgather_seconds),
            "stream_export": bool(stream_export),
            "fsdp_layers_streamed": int(fsdp_layers_streamed),
            "fsdp_units_streamed": int(fsdp_units_streamed),
            "params_streamed": int(params_streamed),
            "fallback_reason": fallback_reason,
            "fallback_count_total": self.fallback_count,
            "hbm_after_transfer_allocated_bytes": snapshot["allocated"],
            "hbm_after_transfer_reserved_bytes": snapshot["reserved"],
            "hbm_peak_allocated_bytes": snapshot["max_allocated"],
            "hbm_peak_reserved_bytes": snapshot["max_reserved"],
        }

    def on_hot_sync_done(
        self,
        *,
        copy_bytes: int,
        copy_seconds: float,
        update_seconds: float = 0.0,
    ) -> dict[str, Any]:
        """Advance the policy version after a successful SGLang update."""
        self.weight_version += 1
        self.total_copy_bytes += int(copy_bytes)
        snapshot = hbm_snapshot()
        return {
            "weight_version": self.weight_version,
            "copy_bytes": int(copy_bytes),
            "copy_seconds": float(copy_seconds),
            "update_seconds": float(update_seconds),
            "hbm_after_hot_sync_allocated_bytes": snapshot["allocated"],
            "hbm_after_hot_sync_reserved_bytes": snapshot["reserved"],
            "hbm_peak_allocated_bytes": snapshot["max_allocated"],
            "hbm_peak_reserved_bytes": snapshot["max_reserved"],
            "fallback_count_total": self.fallback_count,
            "failure_count_total": self.failure_count,
            "total_allgather_bytes": self.total_allgather_bytes,
            "total_copy_bytes": self.total_copy_bytes,
        }

    def on_hot_sync_failed(self, reason: str) -> dict[str, Any]:
        """Record a failed update without advancing the policy version."""
        self.failure_count += 1
        snapshot = hbm_snapshot()
        return {
            "weight_version": self.weight_version,
            "failure_reason": reason,
            "failure_count_total": self.failure_count,
            "hbm_after_failure_allocated_bytes": snapshot["allocated"],
            "hbm_after_failure_reserved_bytes": snapshot["reserved"],
        }


class StreamingExportUnavailable(RuntimeError):
    """Raised before collectives when the model cannot use the narrow stream path."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class _StateEntry:
    name: str
    tensor: torch.Tensor
    is_parameter: bool


@dataclass(frozen=True)
class _FSDPExportUnit:
    name: str
    module: nn.Module
    prefix: str
    is_root: bool


@dataclass(frozen=True)
class QwenFSDP1ExportPlan:
    """Eager, deterministic plan built before any SGLang weight is updated."""

    model: nn.Module
    units: tuple[_FSDPExportUnit, ...]
    expected_names: tuple[str, ...]
    signature: str
    layer_count: int


_NAME_SENTINEL = object()


def _join_name(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


def _canonical_weight_name(name: str, model: nn.Module) -> str:
    converted = convert_weight_keys({name: _NAME_SENTINEL}, model)
    if len(converted) != 1:
        raise RuntimeError(f"canonical weight conversion changed cardinality for {name!r}")
    converted_name, value = next(iter(converted.items()))
    if value is not _NAME_SENTINEL:
        raise RuntimeError(f"canonical weight conversion replaced the value for {name!r}")
    return converted_name


def _collect_owned_state_entries(
    module: nn.Module,
    *,
    prefix: str,
    fsdp_cls: type[nn.Module],
    encountered_fsdp: Optional[list[tuple[str, nn.Module]]] = None,
    include_parameters: bool = True,
) -> list[_StateEntry]:
    """Collect state_dict-equivalent entries without descending into FSDP units."""
    entries: list[_StateEntry] = []

    if include_parameters:
        for name, parameter in module._parameters.items():
            if parameter is not None:
                entries.append(
                    _StateEntry(_join_name(prefix, name), parameter, True)
                )

    non_persistent = module._non_persistent_buffers_set
    for name, buffer in module._buffers.items():
        if buffer is not None and name not in non_persistent:
            entries.append(_StateEntry(_join_name(prefix, name), buffer, False))

    for child_name, child in module._modules.items():
        if child is None:
            continue
        child_prefix = _join_name(prefix, child_name)
        if isinstance(child, fsdp_cls):
            if encountered_fsdp is not None:
                encountered_fsdp.append((child_prefix, child))
            continue
        entries.extend(
            _collect_owned_state_entries(
                child,
                prefix=child_prefix,
                fsdp_cls=fsdp_cls,
                encountered_fsdp=encountered_fsdp,
                include_parameters=include_parameters,
            )
        )

    return entries


def _unit_expected_raw_names(
    unit: _FSDPExportUnit,
    *,
    fsdp_cls: type[nn.Module],
) -> list[str]:
    """Read FSDP flat-parameter metadata without running a collective."""
    expected: list[str] = []
    handle = getattr(unit.module, "_handle", None)
    flat_parameter = getattr(handle, "flat_param", None)
    if flat_parameter is not None:
        expected.extend(
            _join_name(unit.prefix, name)
            for name in tuple(getattr(flat_parameter, "_fqns", ()) or ())
        )
        for shared in tuple(
            getattr(flat_parameter, "_shared_param_infos", ()) or ()
        ):
            shared_name = _join_name(shared.module_name, shared.param_name)
            expected.append(_join_name(unit.prefix, shared_name))

    owned_module = getattr(
        unit.module,
        "_fsdp_wrapped_module",
        unit.module,
    )
    expected.extend(
        entry.name
        for entry in _collect_owned_state_entries(
            owned_module,
            prefix=unit.prefix,
            fsdp_cls=fsdp_cls,
            include_parameters=False,
        )
    )
    return expected


# (config.model_type, model class name) -> decoder layer class that is one FSDP unit.
_SUPPORTED_STREAM_MODELS = {
    ("sdar", "SDARForCausalLM"): "SDARDecoderLayer",
    ("llada2_moe", "LLaDA2MoeModelLM"): "LLaDA2MoeDecoderLayer",
}


def build_qwen_fsdp1_export_plan(fsdp_module: nn.Module) -> QwenFSDP1ExportPlan:
    """Build root-plus-layer units for the exact Qwen/SDAR training model."""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    if not isinstance(fsdp_module, FSDP):
        raise StreamingExportUnavailable("not_fsdp1")

    model = getattr(fsdp_module, "_fsdp_wrapped_module", None)
    config = getattr(model, "config", None)
    # Extended from SDAR-only to LLaDA2 (llada2_moe). The layered
    # export needs nothing model-specific beyond "one FSDP wrapper per decoder
    # layer + root-owned rest"; LLaDA2's overlay declares
    # _no_split_modules = ["LLaDA2MoeDecoderLayer"] and is wrapped the same way.
    # On LLaDA2.1-mini the generic state_dict export costs ~10 s/step (DTensor
    # chunking of 1.5e4 params + one all-gather per param); the layered stream is
    # 20 layer-level all-gathers instead.
    model_type = getattr(config, "model_type", None) if model is not None else None
    layer_cls = _SUPPORTED_STREAM_MODELS.get((model_type, model.__class__.__name__ if model is not None else None))
    if model is None or layer_cls is None:
        raise StreamingExportUnavailable("qwen_sdar_model_required")

    no_split_modules = tuple(getattr(model, "_no_split_modules", ()) or ())
    if layer_cls not in no_split_modules:
        raise StreamingExportUnavailable("sdar_decoder_wrap_policy_required")

    body = getattr(model, "model", None)
    layers = getattr(body, "layers", None)
    if not isinstance(layers, nn.ModuleList) or not layers:
        raise StreamingExportUnavailable("model_layers_modulelist_required")

    layer_units: list[_FSDPExportUnit] = []
    for index, layer in enumerate(layers):
        if not isinstance(layer, FSDP):
            raise StreamingExportUnavailable(f"layer_{index}_not_fsdp1")
        inner_layer = getattr(layer, "_fsdp_wrapped_module", None)
        if inner_layer is None or inner_layer.__class__.__name__ != layer_cls:
            raise StreamingExportUnavailable(f"layer_{index}_not_sdar_decoder")
        layer_units.append(
            _FSDPExportUnit(
                name=f"layer_{index}",
                module=layer,
                prefix=f"model.layers.{index}",
                is_root=False,
            )
        )

    encountered_fsdp: list[tuple[str, nn.Module]] = []
    _collect_owned_state_entries(
        model,
        prefix="",
        fsdp_cls=FSDP,
        encountered_fsdp=encountered_fsdp,
    )
    # Every decoder layer must be its own FSDP unit, in order. Any *other* FSDP
    # unit (e.g. the separately wrapped self-conditioning adapter when
    # SC_WRAP_ADAPTER=1, which shows up as "self_conditioning") is exported as its
    # own unit too, in encounter order; nothing is assumed about its contents.
    layer_modules = {id(unit.module): unit for unit in layer_units}
    ordered_units: list[_FSDPExportUnit] = []
    seen_layers = 0
    for prefix, module in encountered_fsdp:
        unit = layer_modules.get(id(module))
        if unit is not None:
            if unit is not layer_units[seen_layers]:
                encountered_names = tuple(name for name, _ in encountered_fsdp)
                raise StreamingExportUnavailable(
                    f"non_layer_or_misordered_fsdp_units:{encountered_names!r}"
                )
            seen_layers += 1
            ordered_units.append(unit)
        else:
            ordered_units.append(
                _FSDPExportUnit(
                    name=f"extra_{prefix.replace('.', '_') or 'unnamed'}",
                    module=module,
                    prefix=prefix,
                    is_root=False,
                )
            )
    if seen_layers != len(layer_units):
        encountered_names = tuple(name for name, _ in encountered_fsdp)
        raise StreamingExportUnavailable(
            f"non_layer_or_misordered_fsdp_units:{encountered_names!r}"
        )

    units = (
        _FSDPExportUnit(name="root_owned", module=fsdp_module, prefix="", is_root=True),
        *ordered_units,
    )
    raw_names = [
        name
        for unit in units
        for name in _unit_expected_raw_names(unit, fsdp_cls=FSDP)
    ]

    if not raw_names:
        raise StreamingExportUnavailable("empty_export_plan")

    expected_names = tuple(
        sorted(_canonical_weight_name(name, model) for name in raw_names)
    )
    if len(expected_names) != len(set(expected_names)):
        raise StreamingExportUnavailable("duplicate_canonical_weight_names")

    signature_payload = "\n".join(
        [*(f"{unit.name}:{unit.prefix}" for unit in units), "--", *expected_names]
    )
    signature = hashlib.sha256(signature_payload.encode("utf-8")).hexdigest()
    return QwenFSDP1ExportPlan(
        model=model,
        units=units,
        expected_names=expected_names,
        signature=signature,
        layer_count=len(layer_units),
    )


def _synchronize_for_timing(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class FSDP1QwenParamStream:
    """Single-use stream over deterministic root and SDAR decoder FSDP units."""

    def __init__(
        self,
        plan: QwenFSDP1ExportPlan,
        device: int | str | torch.device,
    ) -> None:
        self.plan = plan
        self.device = (
            torch.device("cuda", device)
            if isinstance(device, int)
            else torch.device(device)
        )
        self.bytes_yielded = 0
        self.allgather_bytes = 0
        self.params_yielded = 0
        self.fsdp_units_streamed = 0
        self.fsdp_layers_streamed = 0
        self.allgather_seconds = 0.0
        self.elapsed_seconds = 0.0
        self.started = False
        self.completed = False
        self.failure_reason: Optional[str] = None
        self._generator: Optional[Generator[tuple[str, torch.Tensor], None, None]] = None

    def __iter__(self) -> Iterator[tuple[str, torch.Tensor]]:
        if self.started:
            raise RuntimeError("FSDP1QwenParamStream is single-use")
        self.started = True
        self._generator = self._iterate()
        return self._generator

    def close(self) -> None:
        if self._generator is not None:
            self._generator.close()

    def _assert_plan_matches_all_fsdp_ranks(self) -> None:
        if not torch.distributed.is_initialized():
            return
        root = self.plan.units[0].module
        process_group = getattr(root, "process_group", None)
        if isinstance(process_group, tuple):
            process_group = process_group[0]
        world_size = torch.distributed.get_world_size(group=process_group)
        if world_size <= 1:
            return
        signatures: list[Optional[str]] = [None] * world_size
        torch.distributed.all_gather_object(
            signatures,
            self.plan.signature,
            group=process_group,
        )
        if any(signature != self.plan.signature for signature in signatures):
            raise RuntimeError(
                "Qwen FSDP export plan differs across ranks; refusing to run "
                f"collectives in an unsafe order: {signatures!r}"
            )

    def _iterate(self) -> Generator[tuple[str, torch.Tensor], None, None]:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        started_at = time.perf_counter()
        emitted_name_set: set[str] = set()
        expected_name_set = set(self.plan.expected_names)
        try:
            self._assert_plan_matches_all_fsdp_ranks()
            for unit in self.plan.units:
                _synchronize_for_timing(self.device)
                gather_started_at = time.perf_counter()
                with FSDP.summon_full_params(
                    unit.module,
                    writeback=False,
                    recurse=False,
                ):
                    _synchronize_for_timing(self.device)
                    self.allgather_seconds += time.perf_counter() - gather_started_at

                    owned_module = getattr(
                        unit.module,
                        "_fsdp_wrapped_module",
                        unit.module,
                    )
                    encountered_fsdp: list[tuple[str, nn.Module]] = []
                    entries = _collect_owned_state_entries(
                        owned_module,
                        prefix=unit.prefix,
                        fsdp_cls=FSDP,
                        encountered_fsdp=encountered_fsdp,
                    )
                    if unit.is_root:
                        expected_children = tuple(
                            (child.prefix, child.module)
                            for child in self.plan.units
                            if not child.is_root
                        )
                        if tuple(encountered_fsdp) != expected_children:
                            raise RuntimeError(
                                "FSDP unit topology changed after export plan creation"
                            )
                    elif encountered_fsdp:
                        raise RuntimeError(
                            f"nested FSDP unit found inside {unit.name}: "
                            f"{tuple(name for name, _ in encountered_fsdp)!r}"
                        )

                    gathered_storages: set[tuple[str, int, int, int]] = set()
                    for entry in entries:
                        tensor = entry.tensor.detach()
                        if tensor.device != self.device:
                            tensor = tensor.to(self.device, non_blocking=True)

                        converted = convert_weight_keys(
                            {entry.name: tensor},
                            self.plan.model,
                        )
                        if len(converted) != 1:
                            raise RuntimeError(
                                "canonical weight conversion changed stream cardinality "
                                f"for {entry.name!r}"
                            )
                        name, tensor = next(iter(converted.items()))
                        if name not in expected_name_set:
                            raise RuntimeError(
                                f"stream emitted unplanned weight name {name!r}"
                            )
                        if name in emitted_name_set:
                            raise RuntimeError(
                                f"stream emitted duplicate weight name {name!r}"
                            )

                        tensor_bytes = tensor.numel() * tensor.element_size()
                        self.bytes_yielded += tensor_bytes
                        self.params_yielded += 1
                        emitted_name_set.add(name)
                        if entry.is_parameter:
                            storage_key = (
                                str(tensor.device),
                                int(tensor.data_ptr()),
                                int(tensor.numel()),
                                int(tensor.element_size()),
                            )
                            if storage_key not in gathered_storages:
                                gathered_storages.add(storage_key)
                                self.allgather_bytes += tensor_bytes
                        yield name, tensor

                self.fsdp_units_streamed += 1
                if not unit.is_root:
                    self.fsdp_layers_streamed += 1

            if emitted_name_set != expected_name_set:
                raise RuntimeError(
                    "stream ended with missing parameters: "
                    f"missing={sorted(expected_name_set - emitted_name_set)!r} "
                    f"extra={sorted(emitted_name_set - expected_name_set)!r}"
                )
            self.completed = True
        except BaseException as exc:
            self.failure_reason = type(exc).__name__
            raise
        finally:
            self.elapsed_seconds = time.perf_counter() - started_at


def stream_fsdp1_params_layered(
    fsdp_module: nn.Module,
    device: int | str | torch.device,
) -> FSDP1QwenParamStream:
    """Return a validated Qwen/SDAR stream; plan errors occur before collectives."""
    return FSDP1QwenParamStream(
        build_qwen_fsdp1_export_plan(fsdp_module),
        device,
    )


class _InstrumentedParamStream:
    """Count tensor bytes and elapsed wall time while preserving an iterable."""

    def __init__(self, iterable: Iterable[tuple[str, torch.Tensor]]) -> None:
        self._iterable = iterable
        self.bytes_yielded = 0
        self.params_yielded = 0
        self.elapsed_seconds = 0.0
        self.completed = False

    def __iter__(self) -> Iterator[tuple[str, torch.Tensor]]:
        started_at = time.perf_counter()
        try:
            for name, tensor in self._iterable:
                self.bytes_yielded += tensor.numel() * tensor.element_size()
                self.params_yielded += 1
                yield name, tensor
            self.completed = True
        finally:
            self.elapsed_seconds = time.perf_counter() - started_at

    def close(self) -> None:
        close = getattr(self._iterable, "close", None)
        if callable(close):
            close()


async def update_weights_with_failure_invalidation(
    update_weights: Callable[..., Awaitable[Any]],
    invalidate_weights: Callable[[], Awaitable[Any]],
    weights: Iterable[tuple[str, torch.Tensor]],
    **kwargs: Any,
) -> Any:
    """Invalidate rollout storage if a bucketed update fails after partial writes."""
    try:
        return await update_weights(weights, **kwargs)
    except BaseException as exc:
        close = getattr(weights, "close", None)
        if callable(close):
            close()
        try:
            await invalidate_weights()
        except BaseException as cleanup_exc:
            if hasattr(exc, "add_note"):
                exc.add_note(
                    "rollout weight invalidation also failed: "
                    f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                )
        raise


def assert_stream_offload_safe(
    stream: FSDP1QwenParamStream,
    *,
    update_returned: bool,
) -> None:
    """Reject FSDP offload until the stream completed and SGLang returned."""
    if not stream.completed or not update_returned:
        raise RuntimeError(
            "cannot offload FSDP shards before the streaming generator is fully "
            "consumed and rollout.update_weights() returns"
        )
