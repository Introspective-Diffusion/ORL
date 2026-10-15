# Frozen benchmark inputs

The exact prompts and references of the paper's main table and of two transfer benchmarks. `SHA256SUMS` pins every
file; scoring refuses inputs with a different hash.

| file | tasks | used by | source (license) |
|---|---|---|---|
| `qwen-dlm/gsm8k.jsonl`, `paper4/gsm8k.jsonl` | 1319 | QwenDLM / LLaDA, DiffusionGemma | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) test (MIT) |
| `qwen-dlm/math500.jsonl`, `paper4/math500.jsonl` | 500 | QwenDLM / LLaDA, DiffusionGemma | [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) (MIT) |
| `qwen-dlm/he.jsonl`, `paper4/humaneval.jsonl` | 164 | QwenDLM / LLaDA, DiffusionGemma | [HumanEval](https://github.com/openai/human-eval) (MIT) |
| `qwen-dlm/mbpp.jsonl`, `paper4/mbpp.jsonl` | 427 | QwenDLM / LLaDA, DiffusionGemma | [MBPP](https://github.com/google-research/google-research/tree/f82046ba5aabbbb427dbfd38a254d26bff08b533/mbpp) sanitized set (Apache-2.0) |
| `transfer/mmlu_pro.jsonl` | 12032 | LLaDA, DiffusionGemma (transfer) | [TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/tree/b189ec765aa7ed75c8acfea42df31fdae71f97be) test, revision `b189ec765aa7ed75c8acfea42df31fdae71f97be` (MIT) |
| `transfer/ifeval.jsonl` | 541 | LLaDA, DiffusionGemma (transfer) | [google-research/google-research](https://github.com/google-research/google-research/tree/08a8d6736475776f42ffac23b2c13111a28e5795/instruction_following_eval) `instruction_following_eval/data/input_data.jsonl`, commit `08a8d6736475776f42ffac23b2c13111a28e5795` (Apache-2.0) |

MBPP is the complete sanitized set (`sanitized-mbpp.json`, 427 tasks, 1,324 tests with their imports) of the Google
Research repository, whose Apache-2.0 license covers it; the Hugging Face release of MBPP is listed as CC BY 4.0.

The two families use different prompt formats, as in the paper; `paper4/` holds the one LLaDA and DiffusionGemma
share. `scripts/prepare.sh inputs` copies these files into the work directory and derives DiffusionGemma's four
interleaved GSM8K parts (`gsm8k_m4r0` ... `gsm8k_m4r3`).

The other transfer inputs are not bundled. `repro/eval/build_transfer_inputs.py` rebuilds AIME, LiveCodeBench (with its
tests) and GPQA from pinned Hugging Face revisions and checks their SHA256. GPQA asks that its examples not be shared
in plain text, so only its option order is stored and the questions come from the gated dataset, which needs a
Hugging Face login that has accepted its terms (see [TRANSFER_GENERATION.md](../repro/paper/TRANSFER_GENERATION.md)).
