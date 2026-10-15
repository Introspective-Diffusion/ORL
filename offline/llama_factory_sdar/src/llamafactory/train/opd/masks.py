import torch


def build_prompt_clean_noisy_mask(prefix_len: int, block_len: int, device=None) -> torch.Tensor:
    """Reference OPD attention mask for `[prefix | clean_block | noisy_block]`.

    Segments:
      A = committed prefix
      B = clean current block
      C = noisy current block

    Visibility:
      A -> A causal
      B -> A + B causal
      C -> A + C bidirectional
      C cannot see B to avoid current-clean leakage.
    """

    total_len = prefix_len + 2 * block_len
    mask = torch.zeros(total_len, total_len, dtype=torch.bool, device=device)
    prefix = slice(0, prefix_len)
    clean = slice(prefix_len, prefix_len + block_len)
    noisy = slice(prefix_len + block_len, total_len)

    if prefix_len > 0:
        causal_prefix = torch.tril(torch.ones(prefix_len, prefix_len, dtype=torch.bool, device=device))
        mask[prefix, prefix] = causal_prefix

    if block_len > 0:
        clean_rows = torch.arange(block_len, device=device)
        clean_cols = torch.arange(block_len, device=device)
        if prefix_len > 0:
            mask[clean, prefix] = True
        mask[clean, clean] = clean_cols.unsqueeze(0) <= clean_rows.unsqueeze(1)

        if prefix_len > 0:
            mask[noisy, prefix] = True
        mask[noisy, noisy] = True

    return mask
