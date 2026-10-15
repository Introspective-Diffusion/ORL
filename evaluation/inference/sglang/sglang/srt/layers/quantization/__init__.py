# Adapted from https://raw.githubusercontent.com/vllm-project/vllm/v0.5.5/vllm/model_executor/layers/quantization/__init__.py
from __future__ import annotations

import builtins
import inspect
from typing import TYPE_CHECKING, Dict, Optional, Type

import torch


# Define empty classes as placeholders when vllm is not available
class DummyConfig:
    def override_quantization_method(self, *args, **kwargs):
        return None


CompressedTensorsConfig = DummyConfig

from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.utils import is_cuda, is_hip, is_npu, mxfp_supported


def _optional_import(module_name: str, attr_names: str | tuple[str, ...]):
    if isinstance(attr_names, str):
        attr_names = (attr_names,)
    try:
        module = __import__(module_name, fromlist=list(attr_names))
        values = tuple(getattr(module, attr_name) for attr_name in attr_names)
    except Exception:
        values = tuple(None for _ in attr_names)
    return values[0] if len(values) == 1 else values


AutoRoundConfig = _optional_import("sglang.srt.layers.quantization.auto_round", "AutoRoundConfig")
AWQConfig, AWQMarlinConfig = _optional_import("sglang.srt.layers.quantization.awq", ("AWQConfig", "AWQMarlinConfig"))
BitsAndBytesConfig = _optional_import("sglang.srt.layers.quantization.bitsandbytes", "BitsAndBytesConfig")
BlockInt8Config = _optional_import("sglang.srt.layers.quantization.blockwise_int8", "BlockInt8Config")
CompressedTensorsConfig = _optional_import("sglang.srt.layers.quantization.compressed_tensors.compressed_tensors", "CompressedTensorsConfig")
Fp8Config = _optional_import("sglang.srt.layers.quantization.fp8", "Fp8Config")
FBGEMMFp8Config = _optional_import("sglang.srt.layers.quantization.fpgemm_fp8", "FBGEMMFp8Config")
GGUFConfig = _optional_import("sglang.srt.layers.quantization.gguf", "GGUFConfig")
GPTQConfig, GPTQMarlinConfig = _optional_import("sglang.srt.layers.quantization.gptq", ("GPTQConfig", "GPTQMarlinConfig"))
ModelOptFp4Config, ModelOptFp8Config = _optional_import("sglang.srt.layers.quantization.modelopt_quant", ("ModelOptFp4Config", "ModelOptFp8Config"))
ModelSlimConfig = _optional_import("sglang.srt.layers.quantization.modelslim.modelslim", "ModelSlimConfig")
MoeWNA16Config = _optional_import("sglang.srt.layers.quantization.moe_wna16", "MoeWNA16Config")
Mxfp4Config = _optional_import("sglang.srt.layers.quantization.mxfp4", "Mxfp4Config")
PetitNvFp4Config = _optional_import("sglang.srt.layers.quantization.petit", "PetitNvFp4Config")
QoQConfig = _optional_import("sglang.srt.layers.quantization.qoq", "QoQConfig")
QuarkConfig = _optional_import("sglang.srt.layers.quantization.quark.quark", "QuarkConfig")
QuarkInt4Fp8Config = _optional_import("sglang.srt.layers.quantization.quark_int4fp8_moe", "QuarkInt4Fp8Config")
W4AFp8Config = _optional_import("sglang.srt.layers.quantization.w4afp8", "W4AFp8Config")
W8A8Fp8Config = _optional_import("sglang.srt.layers.quantization.w8a8_fp8", "W8A8Fp8Config")
W8A8Int8Config = _optional_import("sglang.srt.layers.quantization.w8a8_int8", "W8A8Int8Config")

_is_mxfp_supported = mxfp_supported()

if TYPE_CHECKING:
    from sglang.srt.layers.moe.topk import TopKOutput

# Base quantization methods
BASE_QUANTIZATION_METHODS: Dict[str, Type[QuantizationConfig]] = {
    "fp8": Fp8Config,
    "mxfp8": Fp8Config,
    "blockwise_int8": BlockInt8Config,
    "modelopt": ModelOptFp8Config,  # Auto-detect, defaults to FP8
    "modelopt_fp8": ModelOptFp8Config,
    "modelopt_fp4": ModelOptFp4Config,
    "w8a8_int8": W8A8Int8Config,
    "w8a8_fp8": W8A8Fp8Config,
    "awq": AWQConfig,
    "awq_marlin": AWQMarlinConfig,
    "bitsandbytes": BitsAndBytesConfig,
    "gguf": GGUFConfig,
    "gptq": GPTQConfig,
    "gptq_marlin": GPTQMarlinConfig,
    "moe_wna16": MoeWNA16Config,
    "compressed-tensors": CompressedTensorsConfig,
    "qoq": QoQConfig,
    "w4afp8": W4AFp8Config,
    "petit_nvfp4": PetitNvFp4Config,
    "fbgemm_fp8": FBGEMMFp8Config,
    "quark": QuarkConfig,
    "auto-round": AutoRoundConfig,
    "modelslim": ModelSlimConfig,
    "quark_int4fp8_moe": QuarkInt4Fp8Config,
}

BASE_QUANTIZATION_METHODS = {
    name: config_cls
    for name, config_cls in BASE_QUANTIZATION_METHODS.items()
    if config_cls is not None
}


if Mxfp4Config is not None and (is_cuda() or (_is_mxfp_supported and is_hip())):
    BASE_QUANTIZATION_METHODS.update(
        {
            "mxfp4": Mxfp4Config,
        }
    )

QUANTIZATION_METHODS = {**BASE_QUANTIZATION_METHODS}


def get_quantization_config(quantization: str) -> Type[QuantizationConfig]:
    if quantization not in QUANTIZATION_METHODS:
        raise ValueError(
            f"Invalid quantization method: {quantization}. "
            f"Available methods: {list(QUANTIZATION_METHODS.keys())}"
        )

    return QUANTIZATION_METHODS[quantization]


original_isinstance = builtins.isinstance
