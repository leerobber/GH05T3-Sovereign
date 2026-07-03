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
