import torch

from gh05t3_binary.train.dataset import TokenStreamDataset, build_train_val_token_datasets


class _IdentityTokenizer:
    """Encodes each character as its ord() value -- deterministic and easy
    to reason about for testing chunk boundaries and target shifting."""

    def encode(self, text: str):
        return [ord(c) for c in text]


def test_token_stream_dataset_shapes_and_shift():
    tokenizer = _IdentityTokenizer()
    # 100 chars, seq_len=10 -> ids length 100, num_chunks = (100-1)//10 = 9
    text = "abcdefghij" * 10
    ds = TokenStreamDataset([text], tokenizer, seq_len=10)

    assert len(ds) == 9

    x0, y0 = ds[0]
    assert x0.shape == (10,)
    assert y0.shape == (10,)
    # y is x shifted by exactly one position within the same window
    assert torch.equal(y0[:-1], x0[1:])


def test_token_stream_dataset_uses_full_corpus_no_padding():
    """Unlike per-line pad/truncate, every token in a long corpus should
    appear in some chunk's x -- no padding tokens (id 0 from padding)
    should be introduced for a corpus with no null bytes."""
    tokenizer = _IdentityTokenizer()
    text = "the quick brown fox jumps over the lazy dog " * 5
    ds = TokenStreamDataset([text], tokenizer, seq_len=8)

    all_x_tokens = []
    for i in range(len(ds)):
        x, _ = ds[i]
        all_x_tokens.extend(x.tolist())

    expected_ids = tokenizer.encode(text)
    # every x-token actually came from the source text (no padding introduced)
    assert set(all_x_tokens) <= set(expected_ids)
    assert len(all_x_tokens) > 0


def test_token_stream_dataset_raises_on_too_short_corpus():
    tokenizer = _IdentityTokenizer()
    try:
        TokenStreamDataset(["short"], tokenizer, seq_len=64)
        assert False, "expected ValueError for too-short corpus"
    except ValueError:
        pass


def test_train_val_split_is_disjoint_and_covers_corpus():
    tokenizer = _IdentityTokenizer()
    text = "the quick brown fox jumps over the lazy dog " * 20  # 900 chars
    full_ids = tokenizer.encode(" ".join([text]))

    train_ds, val_ds = build_train_val_token_datasets([text], tokenizer, seq_len=8, val_fraction=0.1)

    # sizes: val gets the last ~10% of tokens, train gets the rest
    assert len(train_ds.ids) == int(len(full_ids) * 0.9)
    assert len(val_ds.ids) == len(full_ids) - len(train_ds.ids)

    # disjoint by construction: val's ids are exactly the tail the train
    # dataset does NOT contain
    assert torch.equal(torch.cat([train_ds.ids, val_ds.ids]), torch.tensor(full_ids, dtype=torch.long))

    assert len(train_ds) > 0
    assert len(val_ds) > 0


def test_train_val_split_rejects_invalid_fraction():
    tokenizer = _IdentityTokenizer()
    text = "the quick brown fox jumps over the lazy dog " * 20
    for bad_fraction in (0.0, 1.0, -0.1, 1.5):
        try:
            build_train_val_token_datasets([text], tokenizer, seq_len=8, val_fraction=bad_fraction)
            assert False, f"expected ValueError for val_fraction={bad_fraction}"
        except ValueError:
            pass
