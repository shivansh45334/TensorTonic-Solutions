
def train_bpe(corpus: list[str], vocab_size: int) -> dict:
    """
    Returns a dictionary of learned vocab entries and ordered merges.
    """
    sequences = [list(s.encode("utf-8")) for s in corpus]

    token_bytes = {i: bytes([i]) for i in range(256)}
    vocab = []
    merges = []
    next_id = 256

    while next_id < vocab_size:
        pair_counts = {}

        # Count adjacent pairs within each string.
        for seq in sequences:
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

        if not pair_counts:
            break

        # Select the most frequent pair.
        # For ties, choose the lexicographically greater
        # pair of byte strings.
        best_pair = max(
            pair_counts,
            key=lambda pair: (
                pair_counts[pair],
                token_bytes[pair[0]],
                token_bytes[pair[1]]
            )
        )

        left, right = best_pair
        new_id = next_id

        merged_bytes = token_bytes[left] + token_bytes[right]
        token_bytes[new_id] = merged_bytes

        vocab.append([new_id, list(merged_bytes)])
        merges.append([left, right, new_id])

        # Merge matching pairs left to right, without overlap.
        for idx, seq in enumerate(sequences):
            new_seq = []
            i = 0

            while i < len(seq):
                if (
                    i + 1 < len(seq)
                    and seq[i] == left
                    and seq[i + 1] == right
                ):
                    new_seq.append(new_id)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1

            sequences[idx] = new_seq

        next_id += 1

    return {
        "vocab": vocab,
        "merges": merges
    }
