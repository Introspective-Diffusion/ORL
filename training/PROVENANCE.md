# Provenance of `training/`

Every directory below was imported from the authors' private research
repository at the commits listed here (the hashes are kept for traceability).
The import was verified by comparing git tree hashes of the imported subtrees
against the source commits.

## Paper checkpoints and the training code that produced them

| Family | Paper checkpoint (internal run name) | Training code (private pins) |
|---|---|---|
| QwenDLM | `rct-cf325eg56-s350` (global_step_350) | `bec3c84b` (1 -> 325, then stage 2 326 -> 350) |
| LLaDA-2.1-mini | `lmix_q_steplin_c900` (global_step_625) | `3bfc1fb7` (launch 1, 1 -> 450), `8d71e204` (relaunch, 451 -> 625; same training code) |
| DiffusionGemma-26B-A4B | `dgaec_mixce_rho02_c100_allvalid_s450_20260912` (global_step_900) | `9bebef28` / `028ce292` (same training code), SGLang fork `1e4b811`, Transformers overlay `9a17b68a` |

## How the fused engine was built

1. QwenDLM + LLaDA: three-way merge of `bec3c84b` into `8d71e204` (merge base
   `5dea3efb`, the commit both lines branched from).
2. DiffusionGemma: three-way merge of the Gemma-specific engine delta
   `e3c88eca..028ce292` (training/, kernels/; `e3c88eca` is the last commit of
   the DiffusionGemma branch before Gemma-specific work) onto the result.

Conflict resolutions that change behavior are opt-in and keep each paper
recipe on its original semantics:

- `sglang/.../dllm/algorithm/sdar_prefix_kv.py`: new decoder option
  `confidence_temperature_first_step_full` for the `step_linear` confidence
  temperature schedule. `false` (default) is the QwenDLM schedule
  `T = max(T_min, T_max - d * iter)`; `true` is the LLaDA schedule
  `T = max(T_min, T_max - d * (iter - 1))`. The LLaDA recipe sets `true`.
- AEC empty-step handling (`ray_trainer.py`, `response_level_ce.py`,
  `dp_actor.py`): steps without pairable adjacent states skip the AEC term
  instead of raising; CE still trains. Both source lines already did this for
  the empty-mask case.
- `protocol.list_of_dict_to_dict_of_list`: union of keys for metrics, keys
  common to all ranks for `non_tensor_batch`.
- Self-conditioning telemetry in `dp_actor.py` (metrics only).

## Changes after the import

- `sglang*/sglang/srt/dllm/algorithm/sdar_gpu_replay_lease.py`: the CUDA-IPC
  lease ack lock is per user, `/tmp/sglang_sdar_gpu_replay_ack_<uid>.lock`
  (override: `SGLANG_SDAR_GPU_REPLAY_ACK_LOCK`), instead of one path shared by
  all users. The QwenDLM stage-2 recipe uses leases.
- QwenDLM recipes set `DLLM_STOP_TOKEN_IDS: ''`. The LLaDA line attaches
  tokenizer-derived stop ids to every DLLM rollout request that has none
  (`async_sglang_server._apply_dllm_stop_token_ids`); the QwenDLM engine
  (`bec3c84b`) never did, and an explicit empty value turns this off.
  LLaDA-2.1-mini keeps the automatic ids (`[156900, 156892]`, as in its training
  logs); DiffusionGemma passes explicit stop ids and is unaffected.
- `rewards/`: the two versions of the code-execution reward used by the paper
  runs are one module with two options (`code_exec.py`): `compute_score_lenient`
  (QwenDLM, `bec3c84b`) and `compute_score_strict` (LLaDA-2.1-mini and
  DiffusionGemma via `math_code_mix.py`, `3bfc1fb7`). Both entry points return the
  same results as the originals on the test inputs used during the port.
- `verl/verl/workers/fsdp_workers.py`: for the actor-to-rollout weight sync the
  fused engine exported every model with
  `FULL_STATE_DICT(offload_to_cpu=True, rank0_only=False)`, as the QwenDLM and
  LLaDA lines do. DiffusionGemma's line (`028ce292`) uses `SHARDED_STATE_DICT`
  (DTensors under the device-mesh FSDP1 wrap, gathered one at a time); with a
  full CPU copy of the 26B model on every rank the sync took 81-104 s per step
  instead of 19-36 s. DiffusionGemma uses the sharded export again. The synced
  weights are unchanged: with the same step-1 outcome, every step-2 metric is
  bit-identical to a run with the full export.
- `sglang/sglang/srt/dllm/algorithm/sdar_prefix_kv.py`: the asynchronously
  captured final tokens of a block are cloned at capture time.
- `verl/verl/experimental/agent_loop/agent_loop.py`: optional replay of a
  recorded run's final-token snapshots and reward verdicts
  (`ORL_QWEN_DLM_REPLAY_SNAPSHOTS`, `ORL_QWEN_DLM_REPLAY_REWARD`), used by
  `scripts/reproduce_qwen_dlm.sh`.
- Comments and log messages of the research code are in English, references
  to internal paths, machines and people are removed, and leftover debug output
  (a block-logging hook and a per-sync print in the LLaDA2 SGLang model,
  commented-out prints) is removed. Code changes made for
  portability, none of which affects the paper recipes:
  - DiffusionGemma's cross-node check no longer requires a particular interface
    name; it requires `NCCL_SOCKET_IFNAME` and `GLOO_SOCKET_IFNAME` to name the
    same interface, together with `NCCL_IB_DISABLE=1` as before.
  - The rollout judge's default API-key variable is `ROLLOUT_JUDGE_API_KEY`
    (the judge is off in all recipes).
  - `tools/data/build_code_level.py` takes the source dataset as `--source`, and
    `tools/data/build_code_1assert_20260906.py` carries the evaluation prompt
    itself instead of importing it from the research repository.

## Verification

- Imported subtrees: git tree hashes equal the source commits.
- Recipes: composed Hydra configs equal the configs printed by the paper runs up
  to the differences listed in `recipes/README.md`; launch environments equal the
  recorded ones except `SGLANG_SDAR_GPU_REPLAY_LEASE`, which the QwenDLM stage-1
  recipe sets to the paper value 0.
- Start models: `tools/make_scrp_base.py` rebuilds `checkpoint-38750-scrp8` byte
  for byte from `checkpoint-38750`; `tools/prepare_llada2_actor.sh` produces the
  actor directory used by the paper runs (same overlay code as md5
  `0952affd8fbd`; the release copy differs in comments only).
- Smoke tests with the fused engine (2 optimizer steps each, starting from the
  start models): QwenDLM stage 1, QwenDLM stage 2 (entropy-gated RKL on) and
  LLaDA-2.1-mini on 8 x H100; DiffusionGemma on 2 x 8 H100 (FSDP actor
  node + SGLang rollout node, cross-node NCCL weight sync, canvas AEC on).
- Behavior inherited across lines: every engine change one source line made
  after the common baseline, and that the other line inherits through the
  merge, was reviewed. Apart from the stop ids above, these changes are opt-in
  and off by default, telemetry only, or reached only where the original engine
  raised. One of them concerns prompts shorter than one block: blocks that
  start inside the prompt carry the prompt tokens they cover
  (`schedule_batch.py`). No paper training prompt is that short (the shortest
  is 337 tokens for QwenDLM and 141 tokens for LLaDA-2.1-mini's 32-token blocks).

## Numerical parity with the original engines

Every recipe was run on the fused engine and on the original engine of its
paper run (its pinned `verl/`, SGLang and kernels) from the same start model
and data, and every logged metric was compared step by step. Where the
original engine is itself deterministic, the two engines produce identical
metrics, and for QwenDLM identical weights; elsewhere the fused engine stays
within the variation of the original engine.

## Imported subtrees

| Release path | Source | Source path |
|---|---|---|
| `verl/verl/` | fused engine | `training/diffusion-reasoning-dapo-latent/verl/verl` |
| `verl/{LICENSE,Notice.txt,setup.py,pyproject.toml,requirements*.txt}` | fused engine | `training/diffusion-reasoning-dapo-latent/verl/` |
| `sglang/` | fused engine (QwenDLM + LLaDA rollouts) | `inference/sglang` |
| `sglang_dgemma/` | `1e4b811` (upstream `sgl-project/sglang` `11ffa55` + 3 commits) | `python/`, `LICENSE` |
| `kernels/fast-diffusion-attention/` | fused engine | `kernels/fast-diffusion-attention` |
| `overlays/sdar/` | model code shipped inside the `checkpoint-38750-scrp8` start model, with its comments translated to English | `modeling_sdar.py` (paper runs: md5 `ad3471c61d94`), `configuration_sdar.py`, `fused_linear_diffusion_cross_entropy.py` |
| `overlays/llada2/modeling_llada2_moe.py` | `8d71e204` | LLaDA-2.1-mini modeling overlay of the training runs (md5 `0952affd8fbd`, matching `modeling_md5` in the training launch logs; the release copy differs in comments only) |
| `overlays/diffusion_gemma/transformers/` | `9a17b68a` | DiffusionGemma Transformers overlay (identical to the runtime overlay used in training) |
