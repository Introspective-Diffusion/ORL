try:
    from .triton.fused_moe import fused_moe  # re-export
except ModuleNotFoundError:
    fused_moe = None

__all__ = ["fused_moe"]
