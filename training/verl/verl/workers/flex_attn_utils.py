from typing import Optional, Tuple, Union

import torch
from packaging import version

from torch.nn.attention.flex_attention import flex_attention

_torch_version = torch.__version__


class WrappedFlexAttention:
    """
    We are doing a singleton class so that flex attention is compiled once when it's first called.
    """

    _instance = None
    _is_flex_compiled = False
    _compiled_flex_attention = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            # Create a new instance if one doesn't already exist
            cls._instance = super().__new__(cls)
        return cls._instance

    @torch.compiler.disable(recursive=False)
    def __init__(self, training):
        """
        Initialize or update the singleton instance.
        """
        if not self._is_flex_compiled or training != self.training:
            # In PyTorch 2.6.0, there's a known issue with flex attention compilation which may
            # cause errors. The suggested fix is to compile with "max-autotune-no-cudagraphs"
            # see https://github.com/pytorch/pytorch/issues/146260 for training
            self.training = training
            self._compiled_flex_attention = torch.compile(
                flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs"
            )
            self._is_flex_compiled = True

    def __call__(self):
        return self._compiled_flex_attention
