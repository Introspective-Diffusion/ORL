from sglang.srt.dllm.algorithm import get_algorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.server_args import ServerArgs


class DllmAlgorithm:

    def __init__(
        self,
        config: DllmConfig,
    ):
        self.block_size = config.block_size
        self.mask_id = config.mask_id
        # Subclasses may override _stats with more fields; these are the common ones.
        if not hasattr(self, "_stats"):
            self._stats = {
                "total_forwards": 0,
                "total_tokens": 0,
            }

    def get_stats(self) -> dict:
        """Return algorithm stats including TPF (tokens per forward)."""
        s = self._stats
        n = s["total_forwards"]
        return {
            **s,
            "tpf": s["total_tokens"] / max(n, 1),
        }

    def reset_stats(self):
        """Reset all counters to zero."""
        for k in self._stats:
            self._stats[k] = 0

    @staticmethod
    def from_server_args(server_args: ServerArgs):
        config = DllmConfig.from_server_args(server_args)
        return get_algorithm(config)
