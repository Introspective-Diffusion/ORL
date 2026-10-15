from typing import Any

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.server_args import ServerArgs


class DllmConfig:
    def __init__(
        self,
        algorithm: str,
        algorithm_config: dict[str, Any],
        block_size: int,
        mask_id: int,
        max_running_requests: int,
        causal_prefill: bool = False,
        block_causal_prefill: bool = False,
        prompt_tail_in_first_block: bool = False,
        sdar_rollout_cuda_graph: bool = False,
        sdar_rollout_cuda_graph_capture_sizes: list | None = None,
        sdar_rollout_cuda_graph_max_cache: int = 16,
        vocab_size: int | None = None,
        eos_token_ids: tuple[int, ...] | None = None,
    ):
        self.algorithm = algorithm
        self.algorithm_config = algorithm_config
        self.block_size = block_size
        self.mask_id = mask_id
        self.max_running_requests = max_running_requests
        # True vocabulary size from hf_config, used for entropy normalization.
        # None when constructed without a model config (e.g. in unit tests).
        self.vocab_size: int | None = vocab_size
        # Terminal token ids (hf_config + generation_config, merged by ModelConfig).
        # Consumed ONLY by the optional EOS edit shield in SDARPrefixKV. Empty when
        # constructed without a model config (e.g. unit tests).
        self.eos_token_ids: tuple[int, ...] = tuple(eos_token_ids or ())
        # If True, STAGING_PREFILL uses causal (lower-triangular) attention
        # instead of bidirectional. Required for SDAR models trained with
        # use_regular_causal=True where prompt tokens attend causally.
        self.causal_prefill = causal_prefill
        # If True, STAGING_PREFILL attends block-causally: every prompt token sees
        # all prompt tokens in its own `block_size` block (bidirectional within the
        # block) plus every earlier block, with blocks counted from position 0.
        # This is LLaDA2's native contract -- the official `generate` builds
        # `tril(num_blocks)` expanded by block_length over the whole sequence, and
        # the training kernel's prompt rows see `[0, next_block_end)`
        # (fast_diffusion_attention/diff_attn_noisy.py::_compute_ranges). Plain
        # `causal_prefill` differs from it by up to block_size-1 tokens of in-block
        # lookahead per prompt token. Takes precedence over causal_prefill for the
        # prompt; denoise and commit passes are unaffected. Implemented with a
        # flashinfer custom mask on a dedicated FA2 ragged prefill wrapper.
        # Requires prefix-cache matches aligned to block_size (a token's K/V now
        # depends on the tokens after it within its block), so run with
        # disable_radix_cache=True; the backend warns once on misaligned prefixes.
        self.block_causal_prefill = block_causal_prefill
        # If True, the block grid is anchored at sequence position 0 (absolute):
        # prefill covers only the whole prompt blocks, (len(prompt) // block_size)
        # * block_size tokens, and the prompt tail (len(prompt) % block_size tokens)
        # is carried into the first decode block as fixed non-mask tokens, so that
        # block starts at a multiple of block_size. This is upstream sglang's
        # behaviour (its adder prefills block_size tokens per round and asserts
        # `prefix_len % block_size == 0`), LLaDA2's official `generate`
        # (`prefill_blocks = L0 // block_length`), and verl's training layout
        # (`line_i = (len(prompt) // block) * block`, prompt tail moved into the
        # response segment). If False (the fork's default), the whole prompt is
        # prefilled and the first decode block starts right after it (relative
        # grid), which shifts every response block boundary by len(prompt) %
        # block_size relative to training whenever the prompt is not
        # block-aligned. Prompts shorter than one block already take the
        # INCOMING_DECODE path (whole prompt inside block 0) in both modes.
        self.prompt_tail_in_first_block = prompt_tail_in_first_block
        # SDAR rollout CUDA graph acceleration flags.
        self.sdar_rollout_cuda_graph = sdar_rollout_cuda_graph
        self.sdar_rollout_cuda_graph_capture_sizes = (
            sdar_rollout_cuda_graph_capture_sizes or [1, 2, 4, 8]
        )
        self.sdar_rollout_cuda_graph_max_cache = sdar_rollout_cuda_graph_max_cache

    @staticmethod
    def from_server_args(
        server_args: ServerArgs,
    ):
        if server_args.dllm_algorithm is None:
            return None

        model_config = ModelConfig.from_server_args(
            server_args,
            model_path=server_args.model_path,
            model_revision=server_args.revision,
        )
        DLLM_PARAMS = {
            "LLaDA2MoeModelLM": {"block_size": 32, "mask_id": 156895},
            "SDARForCausalLM": {"block_size": 4, "mask_id": 151669},
            "SDARMoeForCausalLM": {"block_size": 4, "mask_id": 151669},
            "Qwen3ForCausalLM": {"block_size": 4, "mask_id": 151669},
        }

        arch = model_config.hf_config.architectures[0]
        if arch in DLLM_PARAMS:
            params = DLLM_PARAMS[arch]
            block_size = params["block_size"]
            mask_id = params["mask_id"]
        else:
            raise RuntimeError(f"Unknown diffusion LLM: {arch}")

        # Prefer block_size from model config if available (e.g. block_size=1
        # models vs the default block_size=4 in DLLM_PARAMS).
        hf_block_size = getattr(model_config.hf_config, "block_size", None)
        if hf_block_size is not None:
            block_size = hf_block_size

        # SDAR models with use_regular_causal=True were trained with causal
        # attention for prompt (x0) tokens; prefill must use causal attention.
        causal_prefill = getattr(model_config.hf_config, "use_regular_causal", False)

        max_running_requests = (
            1
            if server_args.max_running_requests is None
            else server_args.max_running_requests
        )

        algorithm_config = {}
        if server_args.dllm_algorithm_config is not None:
            try:
                import yaml
            except ImportError:
                raise ImportError(
                    "Please install PyYAML to use YAML config files. "
                    "`pip install pyyaml`"
                )
            with open(server_args.dllm_algorithm_config, "r") as f:
                algorithm_config = yaml.safe_load(f)

            # Parse common algorithm configurations
            block_size = algorithm_config.get("block_size", block_size)

        # Allow YAML to override causal_prefill (some models have
        # use_regular_causal=None but still need causal prefill).
        if algorithm_config.get("causal_prefill") is not None:
            causal_prefill = algorithm_config["causal_prefill"]
        block_causal_prefill = bool(algorithm_config.get("block_causal_prefill", False))
        prompt_tail_in_first_block = bool(
            algorithm_config.get("prompt_tail_in_first_block", False)
        )

        hf_vocab_size = getattr(model_config.hf_config, "vocab_size", None)
        # ModelConfig already merges hf_config.eos_token_id with
        # hf_generation_config.eos_token_id and normalises int-or-list into a set.
        hf_eos_ids = getattr(model_config, "hf_eos_token_id", None)

        return DllmConfig(
            algorithm=server_args.dllm_algorithm,
            algorithm_config=algorithm_config,
            block_size=block_size,
            mask_id=mask_id,
            max_running_requests=max_running_requests,
            causal_prefill=causal_prefill,
            block_causal_prefill=block_causal_prefill,
            prompt_tail_in_first_block=prompt_tail_in_first_block,
            sdar_rollout_cuda_graph=getattr(server_args, "sdar_rollout_cuda_graph", False),
            sdar_rollout_cuda_graph_capture_sizes=getattr(
                server_args, "sdar_rollout_cuda_graph_capture_sizes", None
            ),
            sdar_rollout_cuda_graph_max_cache=getattr(
                server_args, "sdar_rollout_cuda_graph_max_cache", 16
            ),
            vocab_size=int(hf_vocab_size) if hf_vocab_size is not None else None,
            eos_token_ids=tuple(sorted(int(x) for x in hf_eos_ids)) if hf_eos_ids else (),
        )
