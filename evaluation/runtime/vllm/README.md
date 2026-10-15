# DiffusionGemma runtime

The pinned vLLM image is recorded in `Dockerfile`. Base uses the HF-alignment
patch; ORL uses that alignment plus the adaptive scalar patch. Build in separate
images so a Base server cannot accidentally inherit the ORL sampler:

```sh
docker build -f runtime/vllm/Dockerfile --build-arg SAMPLER=base -t idlm-dg-base .
docker build -f runtime/vllm/Dockerfile --build-arg SAMPLER=orl -t idlm-dg-orl .
python3 -m runtime.stage --family dg --out outputs/runtime-dg
```

Build commands are recipes, not a claim that these new images were accepted.
The source-snippet guards reject an incompatible upstream vLLM version. These
patches deliberately target the pinned image's Python 3.12 installation.

Serve local weights with model alias `dg`, TP1, one active sequence, BF16,
TRITON_ATTN, prefix caching disabled, canvas 256 and 48 denoising steps. Apply
the role-specific generation config from `configs/paper/decoding/dg.yaml`;
the YAML registry is not itself a vLLM `generation_config.json` file. Main
tasks use context 8192; AIME/GPQA transfer uses context 32768.
Do not silently start a default server and assume it matches these settings.

Inside the appropriate patched GPU container with this checkout and local
weights mounted, prepare or launch the exact server:

```sh
python3 -m runtime.vllm.serve --model-root /models/dg-orl --role orl \
  --benchmark ifeval --out /outputs/dg-server --execute
```

Omit `--execute` to write only the launch plan and merged generation config.
Use a fresh output directory each time. The launcher binds loopback, checks
that Base/ORL sampler markers match, and writes `server_receipt.json` only after
health succeeds. It stays in the foreground until stopped and terminates the
server process group on exit. The receipt hashes the model inventory and sampler
source; its image digest is a declaration that must be checked against the
actual container image. It is not independent image attestation.
Pass this receipt to `repro.eval.transfer_generate --server-receipt ...`.
Use `--benchmark aime` or `gpqa` for their distinct long-context configuration.

The staged clients are source-contained and no longer require the historical
private runtime archive. They expect an already configured loopback server;
server provisioning is not hidden in the evaluation CLI. MBPP427 uses the
full-extra client with cap4096, not the old paper4 MBPP378/cap2048 route.
