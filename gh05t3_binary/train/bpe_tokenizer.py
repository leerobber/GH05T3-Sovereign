"""Real subword tokenizer wrapper around HuggingFace `tokenizers`
(ByteLevelBPETokenizer), same encode/decode/vocab_size interface as
SimpleCharTokenizer so it's a drop-in replacement for TextDataset and
train_binary.py. A trained BPE vocab keeps the embedding/output-head size
proportional to a small model (e.g. 4096 tokens) instead of the fixed
256-byte vocab, and produces token sequences a language model can
actually learn structure from -- byte-level tokenization of English text
mostly just re-learns spelling.
"""
from __future__ import annotations

import os

from tokenizers import ByteLevelBPETokenizer


class BPETokenizer:
    def __init__(self, tokenizer: ByteLevelBPETokenizer):
        self._tokenizer = tokenizer
        self.vocab_size = tokenizer.get_vocab_size()

    @classmethod
    def train(cls, corpus_path: str, vocab_size: int, save_path: str) -> "BPETokenizer":
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train(files=[corpus_path], vocab_size=vocab_size, min_frequency=2)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        tokenizer.save(save_path)
        return cls(tokenizer)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        from tokenizers import Tokenizer

        raw = Tokenizer.from_file(path)
        wrapper = cls.__new__(cls)
        wrapper._tokenizer = raw
        wrapper.vocab_size = raw.get_vocab_size()
        return wrapper

    def encode(self, text: str):
        return self._tokenizer.encode(text).ids

    def decode(self, ids) -> str:
        return self._tokenizer.decode(list(ids))
