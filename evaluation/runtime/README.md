# Runtime Preparation

Runtime engines are separate from model weights and frozen benchmark inputs.

- [SGLang](sglang/README.md): checksum-bound family-specific materialization.
- [vLLM](vllm/README.md): digest-pinned image and reviewed sampler patches.
- [Scoring](scoring/README.md): isolated scoring dependencies and source bindings.

Use `python -m runtime.stage --help` to assemble a runtime from this source tree (`scripts/prepare.sh runtimes` stages all three families). Follow the family-specific environment instructions; the CPU tests do not check CUDA availability or install engine dependencies. The original development-only formatter symlink is omitted from staged runtimes.

Staged runtimes keep the third-party source notices; materialization does not change their terms.
