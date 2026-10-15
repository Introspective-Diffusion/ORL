#!/usr/bin/env python3
"""Align the vLLM :gemma DiffusionGemma sampler with the HF reference.

Apply inside `vllm/vllm-openai:gemma` before `vllm serve`. This patch:
1. seeds before the initial canvas (the stock custom sampler ignores request seed);
2. samples with HF's softmax + torch.multinomial instead of Gumbel-max;
3. compares argmax against previous history before writing it (stability=1 means
   two consecutive identical predictions, matching HF).

The patch is deliberately exact-version gated by source-snippet assertions.
"""

from pathlib import Path


PATH = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/"
    "model_executor/models/diffusion_gemma.py"
)


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one source match, found {count}")
    return source.replace(old, new)


def main() -> None:
    source = PATH.read_text()

    source = replace_once(
        source,
        """    def add_request(self, req_index: int, new_req_data: Any) -> None:
        self._req_id_to_index[new_req_data.req_id] = req_index
        self.diffusion_states.add_request(req_index)
""",
        """    def add_request(self, req_index: int, new_req_data: Any) -> None:
        self._req_id_to_index[new_req_data.req_id] = req_index
        sampling_params = new_req_data.sampling_params
        seed = getattr(sampling_params, "seed", None)
        if seed is not None:
            torch.manual_seed(int(seed))
            torch.cuda.manual_seed_all(int(seed))
        self.diffusion_states.add_request(req_index)
""",
        "request RNG",
    )

    source = replace_once(
        source,
        """    # ---- Phase 2: Temperature scaling + Gumbel-max sampling ----
    logits_3d = logits.reshape(num_decode, CL, -1).float()
    scaled = logits_3d / temp[:, None, None].clamp(min=1e-10)

    # Gumbel-max trick: argmax(logits/T + Gumbel) ~ sample from softmax(logits/T)
    u = torch.rand_like(scaled).clamp(min=1e-20)
    gumbel = -torch.log(-torch.log(u))
    # Zero noise when temp==0 (greedy)
    noisy = scaled + gumbel * (temp[:, None, None] > 0).float()
    new_tokens = noisy.view(-1, noisy.shape[-1]).argmax(dim=-1).view(num_decode, CL)
    argmax_tokens = (
        scaled.view(-1, scaled.shape[-1]).argmax(dim=-1).view(num_decode, CL)
    )

    # ---- Phase 3: Probs, self-conditioning, confidence ----
    log_probs = scaled.log_softmax(dim=-1)
    probs = log_probs.exp()
""",
        """    # ---- Phase 2: Temperature scaling + HF-reference sampling ----
    logits_3d = logits.reshape(num_decode, CL, -1).float()
    scaled = logits_3d / temp[:, None, None].clamp(min=1e-10)
    log_probs = scaled.log_softmax(dim=-1)
    probs = log_probs.exp()
    new_tokens = torch.multinomial(
        probs.reshape(num_decode * CL, -1), num_samples=1
    ).view(num_decode, CL)
    argmax_tokens = (
        scaled.view(-1, scaled.shape[-1]).argmax(dim=-1).view(num_decode, CL)
    )

    # ---- Phase 3: Probs, self-conditioning, confidence ----
""",
        "multinomial sampling",
    )

    source = replace_once(
        source,
        """    # History: write argmax_tokens for denoise requests at circular position
    hist_len = history_len_tensor[decode_slots]
    write_pos = hist_len % ST
    for i in range(ST):
        write_here = ((write_pos == i) & is_denoise).unsqueeze(1)
        history[decode_slots, i] = torch.where(
            write_here, argmax_tokens, history[decode_slots, i]
        )

    # Argmax canvas: update for denoise, preserve for commit
    argmax_canvas[decode_slots] = torch.where(
        is_denoise.unsqueeze(1), argmax_tokens, argmax_canvas[decode_slots]
    )

    # History length: increment for denoise, reset for commit
    new_hist_len = torch.where(is_denoise, hist_len + 1, hist_len.new_zeros(num_decode))
    history_len_tensor[decode_slots] = new_hist_len
""",
        """    # Compare current argmax against PREVIOUS history before writing it.
    hist_len = history_len_tensor[decode_slots]
    previous_equal = (history[decode_slots] == argmax_tokens[:, None, :]).all(
        dim=-1
    ).all(dim=-1)
    stable = previous_equal & (hist_len >= ST)

    write_pos = hist_len % ST
    for i in range(ST):
        write_here = ((write_pos == i) & is_denoise).unsqueeze(1)
        history[decode_slots, i] = torch.where(
            write_here, argmax_tokens, history[decode_slots, i]
        )

    argmax_canvas[decode_slots] = torch.where(
        is_denoise.unsqueeze(1), argmax_tokens, argmax_canvas[decode_slots]
    )
    new_hist_len = torch.where(is_denoise, hist_len + 1, hist_len.new_zeros(num_decode))
    history_len_tensor[decode_slots] = new_hist_len
""",
        "stability history",
    )

    source = replace_once(
        source,
        """    # ---- Phase 6: Stability + convergence ----
    ref = history[decode_slots, 0]
    mismatch = torch.zeros(num_decode, device=device, dtype=torch.int32)
    for h in range(1, ST):
        mismatch = mismatch + (ref != history[decode_slots, h]).sum(dim=-1).int()
    stable = mismatch == 0

    step_after = step_tensor[decode_slots]
""",
        """    # ---- Phase 6: Stability + convergence ----
    step_after = step_tensor[decode_slots]
""",
        "stability convergence",
    )

    PATH.write_text(source)
    print(f"patched HF-aligned DiffusionGemma sampler: {PATH}")


if __name__ == "__main__":
    main()
