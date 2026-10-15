# Native Generation Clients

These clients preserve the runtime contracts used during validation. Prefer
the recipe-driven wrappers in `inference/adapters/`; importing these clients
directly can load GPU dependencies. None of them executes generated programs.

- `multi_gen.py`: LLaDA persistent SGLang worker, CUDA graph capture sizes
  `[1, 2]`, FlashInfer attention, radix cache disabled, memory fraction 0.60.
  It requires an explicit `ORL_ENGINE_ROOT` (the staged engine directory).
- `vllm_paper4.py`: DG GSM8K/HumanEval/MATH500 client. The adapter selects the
  frozen protocol; HumanEval uses a 2048-token cap.
- `vllm_full_extra_client.py`: original DG MBPP427 route, 4096-token cap and
  48-step sampler. Do not substitute the older MBPP378 branch.

`runtime.stage` copies these files into a fresh runtime tree and records their
SHA-256 values. The vLLM patches live in `inference/vllm/patches/`; Base and ORL
need separate environments because the ORL patch changes sampler behavior.
QwenDLM retains the pinned native clients in `repro/eval/`, validated by its
adapter before the fail-closed generation-only transformation is applied.

The adapters do not replace model-specific prompts, token counters or parsers
with a shared approximation. Standalone `inference/generate.py` is only a raw
local-server client, not a paper evaluation harness.
