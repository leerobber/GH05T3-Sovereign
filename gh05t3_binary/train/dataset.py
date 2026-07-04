import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """Next-token-prediction dataset: each sample yields (x, y) where
    y[i] is the token that follows x[i] — x = ids[:-1], y = ids[1:].

    (A same-position target — y = x — combined with the model's
    unmasked self-attention would let every position simply attend to
    itself and "predict" its own already-visible token, converging to a
    low loss that reflects nothing about actual next-token prediction.
    Real next-token training needs both the shifted target here and a
    causal mask in the model, which GH05T3BinaryTransformer now applies
    internally.)
    """

    def __init__(self, texts, tokenizer, seq_len: int = 64):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.samples = []

        for t in texts:
            ids = tokenizer.encode(t)
            target_len = seq_len + 1
            if len(ids) < target_len:
                ids = ids + [0] * (target_len - len(ids))
            else:
                ids = ids[:target_len]
            self.samples.append(torch.tensor(ids, dtype=torch.long))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = self.samples[idx]
        x = ids[:-1]
        y = ids[1:]
        return x, y


class TokenStreamDataset(Dataset):
    """Next-token-prediction dataset for real corpora: concatenates all
    texts into one continuous token stream (joined with a space, tokenized
    as a whole so a BPE tokenizer can merge across boundaries naturally),
    then slices it into non-overlapping (seq_len+1)-length windows.

    TextDataset's per-line pad-or-truncate approach wastes most of a
    short line's slot on padding and silently drops anything past seq_len
    on a long one -- fine for a handful of demo lines, wasteful and
    lossy for a real multi-hundred-KB corpus. This uses every token.
    """

    def __init__(self, texts, tokenizer, seq_len: int = 64):
        full_text = " ".join(texts)
        ids = torch.tensor(tokenizer.encode(full_text), dtype=torch.long)
        self._init_from_ids(ids, seq_len)

    @classmethod
    def from_ids(cls, ids: torch.Tensor, seq_len: int) -> "TokenStreamDataset":
        """Builds a dataset directly from an already-tokenized id stream --
        used by build_train_val_token_datasets to construct disjoint train/
        val datasets from two slices of one corpus without re-tokenizing."""
        obj = cls.__new__(cls)
        obj._init_from_ids(ids, seq_len)
        return obj

    def _init_from_ids(self, ids: torch.Tensor, seq_len: int) -> None:
        self.ids = ids
        self.seq_len = seq_len
        self.num_chunks = max(0, (len(self.ids) - 1) // seq_len)
        if self.num_chunks == 0:
            raise ValueError(
                f"Corpus too short ({len(self.ids)} tokens) for seq_len={seq_len}; "
                "use a larger corpus, a smaller seq_len, or a smaller val_fraction."
            )

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.ids[start : start + self.seq_len + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def build_train_val_token_datasets(texts, tokenizer, seq_len: int, val_fraction: float = 0.1):
    """Tokenizes the corpus once, then splits the resulting id stream by
    position into a training prefix and a validation suffix -- disjoint by
    construction, so there's no leakage between the two chunked datasets
    built from them. Held-out loss (not just training loss) is the only
    way to tell "the model is learning" apart from "the model is
    memorizing a small corpus," especially while scaling capacity up on a
    fixed-size real corpus.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    full_text = " ".join(texts)
    ids = torch.tensor(tokenizer.encode(full_text), dtype=torch.long)

    split_idx = int(len(ids) * (1.0 - val_fraction))
    train_ids = ids[:split_idx]
    val_ids = ids[split_idx:]

    train_ds = TokenStreamDataset.from_ids(train_ids, seq_len)
    val_ds = TokenStreamDataset.from_ids(val_ids, seq_len)
    return train_ds, val_ds
