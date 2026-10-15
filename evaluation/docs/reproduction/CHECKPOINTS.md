# Paper Checkpoints

The machine-readable source of truth is
[`configs/paper/checkpoints.yaml`](../../configs/paper/checkpoints.yaml).
Decoding is benchmark-specific; a checkpoint name alone is insufficient to
reproduce a cell.

| Role | Checkpoint | Hugging Face (revision) |
|---|---|---|
| `qwen-dlm-base` | QwenDLM SFT step 38750 | `yifanyu/QwenDLM-4B` (`94c4a956af73bb7be5ae5474e85a490a96ccb2b2`) |
| `qwen-dlm-orl` | QwenDLM + ORL, step 350 | `yifanyu/QwenDLM-4B-ORL` (`e12b90bf8267aa5d994bcf73996f00e00d484239`) |
| `llada-base` | LLaDA-2.1-mini | `inclusionAI/LLaDA2.1-mini` (`20e64e2ad21644d0e5248586ed9c942cdd45de0f`) |
| `llada-orl` | LLaDA-2.1-mini + ORL, step 625 | `yifanyu/LLaDA2.1-mini-ORL` (`eb9e57b7ced51a750ec4d94aa7d4cee46243f597`) |
| `dg-base` | DiffusionGemma-26B-A4B-it | `google/diffusiongemma-26B-A4B-it` (`f7f5b7f5fa82ffc52addd066915886d497f5517b`) |
| `dg-orl` | DiffusionGemma + ORL, step 900 | `yifanyu/DiffusionGemma-26B-A4B-it-ORL` (`a5f9c6d23c585eefacf3986a12c3e809061f01ce`) |

```bash
hf download yifanyu/QwenDLM-4B-ORL --revision e12b90bf8267aa5d994bcf73996f00e00d484239 \
  --local-dir $MODELS/QwenDLM-4B-ORL
```

Every weight, configuration and tokenizer file at these revisions is byte-identical to the model directory evaluated
for the paper; the ORL models are the paper's checkpoints. Download complete directories (tokenizer, config and
custom code included): recipes record the hash of every file. Evaluate the DiffusionGemma base model at the revision
above. DiffusionGemma training starts from revision `52de6b914ee1749a7d4933202505ddf5b414ec43`, which differs from it
only in `chat_template.jinja`; the ORL checkpoint carries the chat template of that training start.

These are the six measured main-table models; the transfer evaluation uses the four LLaDA-2.1-mini and DiffusionGemma
models. Literature baselines, training trajectories and appendix ablations are not part of this matrix.
