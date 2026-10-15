import logging
import os
import socket
import time
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from sglang.srt.entrypoints.engine import Engine
from sglang.srt.managers.io_struct import UpdateWeightsFromTensorReqInput
from sglang.srt.model_executor.model_runner import LocalSerializedTensor
from sglang.srt.utils import MultiprocessingSerializer
from sglang.srt.utils.common import get_free_port, init_custom_process_group


_DISTRIBUTED_UPDATE_GROUP = None
_DISTRIBUTED_UPDATE_GROUP_NAME = None
_DISTRIBUTED_UPDATE_GROUP_WORLD_SIZE = None

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


async def update_weights(
    engine: Engine,
    params_batch: list[tuple[str, torch.Tensor]],
    device_mesh_key: str,
    device_mesh: DeviceMesh,
    load_format: Optional[str] = None,
    weight_version: Optional[str] = None,
):
    """
    Update weights for the inference engine.
    This function is designed to be stateless, so that the caller process could keep the stateful engine.
    Example Use Case:
        - Multiple Producer Process will call this function in a SPMD style

    Args:
        engine: The inference engine created by the caller process.
        params_batch: A list of (name, tensor) tuples. We batched the tensors to avoid the overhead of cpu call.
        device_mesh_key: The key of the device mesh. Typically "tp" or "infer_tp"
        device_mesh: The device mesh.
        load_format: The format of the weights.
    """
    infer_tp_size = device_mesh[device_mesh_key].mesh.size()[0]
    infer_tp_rank = device_mesh[device_mesh_key].get_local_rank()

    if _distributed_update_weights():
        return await _update_weights_from_distributed_group(
            engine=engine,
            params_batch=params_batch,
            infer_tp_size=infer_tp_size,
            infer_tp_rank=infer_tp_rank,
            load_format=load_format,
            weight_version=weight_version,
        )

    if not _cpu_serialize_update_weights():
        from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions

        monkey_patch_torch_reductions()

    # [
    #   (name0, ipc_tensor0_tp0),
    #   (name1, ipc_tensor1_tp0),
    # ]
    named_tensors_batch = [
        (
            name,
            MultiprocessingSerializer.serialize(
                _preprocess_tensor_for_update_weights(tensor.detach())
            ),
        )
        for name, tensor in params_batch
    ]

    if infer_tp_rank == 0:
        gathered_serialized_batches = [None for _ in range(infer_tp_size)]
    else:
        gathered_serialized_batches = None

    # [
    #   [ (name0, ipc_tensor0_tp0), (name1, ipc_tensor1_tp0) ],
    #   [ (name0, ipc_tensor0_tp1), (name1, ipc_tensor1_tp1) ],
    # ]
    dist.gather_object(
        obj=named_tensors_batch,
        object_gather_list=gathered_serialized_batches,
        dst=device_mesh[device_mesh_key].mesh.tolist()[0],
        group=device_mesh[device_mesh_key].get_group(),
    )

    if infer_tp_rank == 0:
        # Use zip(*) to "transpose" the data structure.
        # After transpose, the data structure is like:
        # [
        #   ( (name0, ipc_tensor0_tp0), (name0, ipc_tensor0_tp1) ),
        #   ( (name1, ipc_tensor1_tp0), (name1, ipc_tensor1_tp1) ),
        # ]
        logical_tensors = zip(*gathered_serialized_batches, strict=True)

        named_tensors = [
            # [
            #   (name0, LocalSerializedTensor(values=[ipc_tensor0_tp0, ipc_tensor0_tp1])),
            #   (name1, LocalSerializedTensor(values=[ipc_tensor1_tp0, ipc_tensor1_tp1])),
            # ]
            (
                tensor_group[0][0],
                LocalSerializedTensor(
                    values=[rank_part[1] for rank_part in tensor_group]
                ),
            )
            for tensor_group in logical_tensors
        ]

        update_weights_request = UpdateWeightsFromTensorReqInput(
            serialized_named_tensors=[
                MultiprocessingSerializer.serialize(named_tensors)
                for _ in range(infer_tp_size)
            ],
            load_format=load_format,
            weight_version=weight_version,
        )

        return await engine.update_weights_from_tensor(update_weights_request)


def _preprocess_tensor_for_update_weights(tensor: torch.Tensor):
    """
    Preprocess the tensor for update weights.
    Example Use Case:
        - FSDP: we gather tensor by calling full_tensor in _preprocess_tensor_for_update_weights
        - Megatron: we do nothing here, assuming it is gathered when feed into this func

    Args:
        tensor: The tensor to be preprocessed.

    Returns:
        The full tensor if it is a DTensor, otherwise the original tensor.
    """
    if isinstance(tensor, DTensor):
        tensor = tensor.full_tensor()
    if _cpu_serialize_update_weights():
        return tensor.detach().to("cpu").contiguous()
    return tensor


def _cpu_serialize_update_weights() -> bool:
    return os.getenv("SGLANG_UPDATE_WEIGHTS_CPU_SERIALIZE", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _distributed_update_weights() -> bool:
    return os.getenv("SGLANG_UPDATE_WEIGHTS_DISTRIBUTED", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


async def _update_weights_from_distributed_group(
    engine: Engine,
    params_batch: list[tuple[str, torch.Tensor]],
    infer_tp_size: int,
    infer_tp_rank: int,
    load_format: Optional[str] = None,
    weight_version: Optional[str] = None,
):
    # Only the rollout-side infer_tp rank0 acts as the source rank. Other
    # ranks still consume the checkpoint-engine generator, but do not forward
    # duplicate full tensors to SGLang.
    if infer_tp_rank != 0:
        logger.info(
            "verl_sglang_update skip non-source rank: pid=%s infer_tp_rank=%s infer_tp_size=%s params=%s",
            os.getpid(),
            infer_tp_rank,
            infer_tp_size,
            len(params_batch),
        )
        return None

    group_name = _distributed_update_group_name()
    start_time = time.time()
    logger.info(
        "verl_sglang_update source begin: pid=%s cuda_device=%s infer_tp_size=%s group=%s params=%s",
        os.getpid(),
        torch.cuda.current_device() if torch.cuda.is_available() else "cpu",
        infer_tp_size,
        group_name,
        len(params_batch),
    )
    group = await _ensure_distributed_update_group(engine, infer_tp_size, group_name)
    logger.info(
        "verl_sglang_update group ready: pid=%s group=%s elapsed=%.2fs",
        os.getpid(),
        group_name,
        time.time() - start_time,
    )

    prepared_tensors: list[torch.Tensor] = []
    names: list[str] = []
    dtypes: list[str] = []
    shapes: list[list[int]] = []

    for name, tensor in params_batch:
        tensor = _preprocess_tensor_for_update_weights(tensor.detach())
        if tensor.device.type != "cuda":
            tensor = tensor.to("cuda", non_blocking=True)
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        prepared_tensors.append(tensor)
        names.append(name)
        dtypes.append(_torch_dtype_name(tensor.dtype))
        shapes.append([int(dim) for dim in tensor.shape])

    async def _server_receive():
        logger.info(
            "verl_sglang_update server receive begin: pid=%s group=%s params=%s bytes=%.2f MiB",
            os.getpid(),
            group_name,
            len(prepared_tensors),
            sum(t.numel() * t.element_size() for t in prepared_tensors) / (1024 * 1024),
        )
        result = await engine.update_weights_from_distributed(
            names=names,
            dtypes=dtypes,
            shapes=shapes,
            group_name=group_name,
            flush_cache=False,
            load_format=load_format,
            weight_version=weight_version,
        )
        if result and result.get("success") is False:
            raise RuntimeError(result.get("message", "SGLang distributed update failed"))
        logger.info(
            "verl_sglang_update server receive done: pid=%s group=%s result=%s",
            os.getpid(),
            group_name,
            result,
        )
        return result

    import asyncio

    server_task = asyncio.create_task(_server_receive())
    try:
        logger.info(
            "verl_sglang_update broadcast begin: pid=%s group=%s params=%s",
            os.getpid(),
            group_name,
            len(prepared_tensors),
        )
        await asyncio.to_thread(_broadcast_tensors_to_sglang, prepared_tensors, group)
        logger.info(
            "verl_sglang_update broadcast done: pid=%s group=%s elapsed=%.2fs",
            os.getpid(),
            group_name,
            time.time() - start_time,
        )
        return await server_task
    except Exception:
        if not server_task.done():
            server_task.cancel()
        raise


async def _ensure_distributed_update_group(engine: Engine, infer_tp_size: int, group_name: str):
    global _DISTRIBUTED_UPDATE_GROUP
    global _DISTRIBUTED_UPDATE_GROUP_NAME
    global _DISTRIBUTED_UPDATE_GROUP_WORLD_SIZE

    world_size = int(infer_tp_size) + 1
    if (
        _DISTRIBUTED_UPDATE_GROUP is not None
        and _DISTRIBUTED_UPDATE_GROUP_NAME == group_name
        and _DISTRIBUTED_UPDATE_GROUP_WORLD_SIZE == world_size
    ):
        return _DISTRIBUTED_UPDATE_GROUP

    master_address = _node_ip_address()
    master_port = int(get_free_port())
    backend = os.getenv("SGLANG_UPDATE_WEIGHTS_DISTRIBUTED_BACKEND", "nccl")
    logger.info(
        "verl_sglang_update init group begin: pid=%s group=%s master=%s:%s backend=%s world_size=%s source_cuda=%s",
        os.getpid(),
        group_name,
        master_address,
        master_port,
        backend,
        world_size,
        torch.cuda.current_device() if torch.cuda.is_available() else "cpu",
    )

    def _init_source_group():
        device_id = None
        if backend == "nccl" and torch.cuda.is_available():
            device_id = torch.device("cuda", torch.cuda.current_device())
        return init_custom_process_group(
            backend=backend,
            init_method=f"tcp://{master_address}:{master_port}",
            world_size=world_size,
            rank=0,
            group_name=group_name,
            device_id=device_id,
        )

    async def _init_sglang_group():
        logger.info(
            "verl_sglang_update init sglang group request begin: pid=%s group=%s master=%s:%s",
            os.getpid(),
            group_name,
            master_address,
            master_port,
        )
        result = await engine.init_weights_update_group(
            master_address=master_address,
            master_port=master_port,
            rank_offset=1,
            world_size=world_size,
            group_name=group_name,
            backend=backend,
        )
        if result and result.get("success") is False:
            raise RuntimeError(result.get("message", "SGLang update-group init failed"))
        logger.info(
            "verl_sglang_update init sglang group request done: pid=%s group=%s result=%s",
            os.getpid(),
            group_name,
            result,
        )
        return result

    import asyncio

    server_task = asyncio.create_task(_init_sglang_group())
    try:
        source_group = await asyncio.to_thread(_init_source_group)
        logger.info(
            "verl_sglang_update source group init done: pid=%s group=%s",
            os.getpid(),
            group_name,
        )
        await server_task
    except Exception:
        if not server_task.done():
            server_task.cancel()
        raise

    _DISTRIBUTED_UPDATE_GROUP = source_group
    _DISTRIBUTED_UPDATE_GROUP_NAME = group_name
    _DISTRIBUTED_UPDATE_GROUP_WORLD_SIZE = world_size
    return source_group


def _broadcast_tensors_to_sglang(tensors: list[torch.Tensor], group):
    for idx, tensor in enumerate(tensors):
        logger.info(
            "verl_sglang_update tensor broadcast begin: pid=%s idx=%s shape=%s dtype=%s bytes=%.2f MiB",
            os.getpid(),
            idx,
            tuple(tensor.shape),
            tensor.dtype,
            tensor.numel() * tensor.element_size() / (1024 * 1024),
        )
        dist.broadcast(tensor, src=0, group=group)
        logger.info("verl_sglang_update tensor broadcast done: pid=%s idx=%s", os.getpid(), idx)
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _distributed_update_group_name() -> str:
    return os.getenv("SGLANG_UPDATE_WEIGHTS_GROUP_NAME") or f"verl_sglang_update_{os.getpid()}"


def _torch_dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _node_ip_address() -> str:
    try:
        import ray

        return ray.util.get_node_ip_address().strip("[]")
    except Exception:
        return socket.gethostbyname(socket.gethostname())
