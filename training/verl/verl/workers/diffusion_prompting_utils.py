from accelerate.logging import get_logger

logger = get_logger(__name__, log_level="INFO")

import torch


class UniversalPrompting():
    def __init__(self, text_tokenizer,
                 max_prompt_len=1024, max_gen_length=2048, ignore_id=-100):
        """
        :param text_tokenizer: original text tokenizer
        """
        self.text_tokenizer = text_tokenizer
        self.max_gen_length = max_gen_length
        self.max_prompt_len = max_prompt_len
        self.ignore_id = ignore_id

    def __call__(self, input, block_size: int):
        prompt_ids, response_ids = input
        if block_size is None or int(block_size) <= 0:
            raise ValueError("block_size must be a positive integer")
        block_size = int(block_size)

        pad_id = self.text_tokenizer.pad_token_id

        # --- inputs are token ids now: List[List[int]] ---
        # filter by prompt token length (equivalent to the former tokenizer(..., return_length=True))
        lengths = [len(x) for x in prompt_ids]
        keep_indices = [i for i, L in enumerate(lengths) if L <= self.max_prompt_len]
        drop_num = len(prompt_ids) - len(keep_indices)

        prompt_ids_list = [prompt_ids[i] for i in keep_indices]
        response_ids_list = [response_ids[i] for i in keep_indices]

        for p_ids, r_ids in zip(prompt_ids_list, response_ids_list):
            assert (len(p_ids) + len(r_ids)) % block_size == 0

        # Per-sample "line position" (end of the last whole block of the prompt)
        per_sample_line = []
        for p_ids in prompt_ids_list:
            p_len = len(p_ids)
            per_sample_line.append((p_len // block_size) * block_size)

        # One line position for the whole batch: the maximum (a multiple of block_size by construction)
        line_pos = max(per_sample_line) if len(per_sample_line) > 0 else 0

        # Build the aligned sequences and record positions (0-based index; end is exclusive)
        seqs = []
        prompt_end_positions = []  # true end of the prompt (including the prompt tail)
        seq_start_positions = []  # true start of the sequence (left padding before it)
        seq_end_positions = []  # true end of the sequence (right padding after it)

        for p_ids, r_ids, line_i in zip(prompt_ids_list, response_ids_list, per_sample_line):
            # 1) whole-block part of the prompt (left-padded to line_pos)
            p_full_blocks = p_ids[:line_i]  # len == line_i
            left_pad_len = line_pos - len(p_full_blocks)  # == line_pos - line_i
            if left_pad_len < 0:
                raise RuntimeError("Unexpected negative left padding length")
            assert (len(p_ids) + len(r_ids)) % block_size == 0
            prefix = ([pad_id] * left_pad_len) + p_full_blocks

            assert len(prefix) % block_size == 0

            # 2) remainder of the prompt (the tail that crosses a block), placed after the line
            p_tail = p_ids[line_i:]
            assert line_i % block_size == 0

            # 3) response (truncated to max_gen_length)
            if self.max_gen_length is not None and len(r_ids) + len(p_tail) > self.max_gen_length + block_size:
                r_ids = r_ids[: self.max_gen_length - len(p_tail) + block_size]
            assert (len(p_tail) + len(r_ids)) % block_size == 0

            temp_ids = prefix + p_tail + r_ids
            temp_labels = [self.ignore_id] * (len(prefix) + len(p_tail)) + r_ids
            assert len(temp_ids) == len(temp_labels)

            # Key point: the mask zeroes only the padding we inserted; original tokens (even if equal to pad_id) stay 1
            temp_attn = ([0] * left_pad_len) + ([1] * (len(p_full_blocks) + len(p_tail) + len(r_ids)))
            assert len(temp_attn) == len(temp_ids)

            seqs.append((temp_ids, temp_labels, temp_attn))

            seq_start_positions.append(left_pad_len)
            prompt_end_positions.append(len(prefix) + len(p_tail))
            seq_end_positions.append(len(temp_ids))
            assert len(temp_ids) % block_size == 0

        # pad to the longest sequence in the batch
        max_len = max((len(s[0]) for s in seqs), default=0)
        input_ids_rows = []
        label_rows = []
        attn_rows = []
        for s, l, a in seqs:
            if len(s) < max_len:
                pad_n = max_len - len(s)
                s = s + [pad_id] * pad_n
                l = l + [self.ignore_id] * pad_n
                a = a + [0] * pad_n
            else:
                s = s[:max_len]
                l = l[:max_len]
                a = a[:max_len]
            input_ids_rows.append(torch.tensor(s, dtype=torch.long).unsqueeze(0))
            label_rows.append(torch.tensor(l, dtype=torch.long).unsqueeze(0))
            attn_rows.append(torch.tensor(a, dtype=torch.long).unsqueeze(0))

        input_ids_lm = torch.cat(input_ids_rows, dim=0) if input_ids_rows else torch.empty((0, 0), dtype=torch.long)
        labels_lm = torch.cat(label_rows, dim=0) if label_rows else torch.empty((0, 0), dtype=torch.long)
        attention_mask_lm = torch.cat(attn_rows, dim=0) if attn_rows else torch.empty((0, 0), dtype=torch.long)

        prompt_end_positions = torch.tensor(prompt_end_positions,
                                            dtype=torch.long) if prompt_end_positions else torch.empty((0,),
                                                                                                       dtype=torch.long)
        seq_start_positions = torch.tensor(seq_start_positions,
                                           dtype=torch.long) if seq_start_positions else torch.empty((0,),
                                                                                                     dtype=torch.long)
        seq_end_positions = torch.tensor(seq_end_positions, dtype=torch.long) if seq_end_positions else torch.empty(
            (0,), dtype=torch.long)

        return (
            input_ids_lm,
            labels_lm,
            attention_mask_lm,
            line_pos,
            prompt_end_positions,
            seq_start_positions,
            seq_end_positions,
            drop_num,
            keep_indices
        )


if __name__ == '__main__':
    pass