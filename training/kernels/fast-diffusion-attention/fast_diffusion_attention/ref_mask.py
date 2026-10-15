"""Reference mask generator for diff_attn_varlen_with_noisy_input — verify against example.txt."""
import torch

def build_mask_for_doc(L0, L1, staircase_size):
    """Build the attention mask for a single document.

    Layout: [prompt(L0), clean_response(L1), noisy_response(L1)]
    Total length = L0 + 2*L1
    """
    S = staircase_size
    total = L0 + 2 * L1
    mask = torch.zeros(total, total, dtype=torch.bool)

    for qi in range(total):
        for ki in range(total):
            # Determine which part qi is in
            if qi < L0:
                # Part 1: Prompt
                q_part = 0
                q_block = qi // S  # block index within prompt
            elif qi < L0 + L1:
                # Part 2: Clean Response
                q_part = 1
                q_block = (qi - L0) // S  # block index within clean response
            else:
                # Part 3: Noisy Response
                q_part = 2
                q_block = (qi - L0 - L1) // S  # block index within noisy response

            # Determine which part ki is in
            if ki < L0:
                k_part = 0
                k_block = ki // S
            elif ki < L0 + L1:
                k_part = 1
                k_block = (ki - L0) // S
            else:
                k_part = 2
                k_block = (ki - L0 - L1) // S

            visible = False

            if q_part == 0:
                # Prompt query: can see prompt blocks <= own block (semi-autoregressive)
                if k_part == 0 and k_block <= q_block:
                    visible = True
            elif q_part == 1:
                # Clean response query
                if k_part == 0:
                    # Can see ALL prompt
                    visible = True
                elif k_part == 1 and k_block <= q_block:
                    # Can see clean response blocks <= own block (semi-autoregressive)
                    visible = True
            elif q_part == 2:
                # Noisy response query
                if k_part == 0:
                    # Can see ALL prompt
                    visible = True
                elif k_part == 1 and k_block < q_block:
                    # Can see clean response blocks BEFORE own block (strict <)
                    visible = True
                elif k_part == 2 and k_block == q_block:
                    # Can see own noisy block only
                    visible = True

            mask[qi, ki] = visible

    return mask


def build_full_mask(L0_list, L1_list, staircase_size):
    """Build the full block-diagonal attention mask for multiple documents."""
    doc_masks = []
    for L0, L1 in zip(L0_list, L1_list):
        doc_masks.append(build_mask_for_doc(L0, L1, staircase_size))

    total = sum(m.shape[0] for m in doc_masks)
    full_mask = torch.zeros(total, total, dtype=torch.bool)
    offset = 0
    for m in doc_masks:
        s = m.shape[0]
        full_mask[offset:offset+s, offset:offset+s] = m
        offset += s
    return full_mask


def load_example_mask(path="example.txt"):
    """Load the example mask from the tab-separated file."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            # Split by tab, treating empty entries as 0
            vals = []
            for v in line.split('\t'):
                v = v.strip()
                vals.append(int(v) if v else 0)
            rows.append(vals)

    # Determine total size
    max_len = max(len(r) for r in rows)
    n_rows = len(rows)
    total = max(max_len, n_rows)

    mask = torch.zeros(total, total, dtype=torch.bool)
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            mask[i, j] = bool(v)
    return mask


if __name__ == "__main__":
    # Example: staircase_size=2, doc1: L0=4, L1=6, doc2: L0=2, L1=2
    ref = build_full_mask([4, 2], [6, 2], staircase_size=2)
    example = load_example_mask("example.txt")

    print("Reference mask:")
    for i in range(ref.shape[0]):
        print("\t".join(str(int(ref[i, j].item())) for j in range(ref.shape[1])))

    print("\nExample mask:")
    for i in range(example.shape[0]):
        print("\t".join(str(int(example[i, j].item())) for j in range(example.shape[1])))

    match = torch.equal(ref, example)
    print(f"\nMasks match: {match}")
    if not match:
        diff = (ref != example)
        positions = diff.nonzero()
        print(f"Differences at {positions.shape[0]} positions:")
        for pos in positions[:20]:
            i, j = pos[0].item(), pos[1].item()
            print(f"  ({i},{j}): ref={int(ref[i,j])}, example={int(example[i,j])}")
