class SimpleCharTokenizer:
    """Minimal byte-level tokenizer. vocab_size=256, matches the binary
    transformer's default vocab_size and binary_backend.py's naive
    tokenizer."""

    def __init__(self):
        self.vocab_size = 256

    def encode(self, text: str):
        return [ord(c) % 256 for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)
