import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from gh05t3_binary.oss.integration import GH05T3BinaryOSS
from gh05t3_binary.train.tokenizer import SimpleCharTokenizer
from gh05t3_binary.train.dataset import TextDataset, TokenStreamDataset


def _build_tokenizer_and_dataset(
    data_path: str, seq_len: int, tokenizer_type: str, vocab_size: int, tokenizer_path,
):
    with open(data_path, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f.readlines() if line.strip()]

    if tokenizer_type == "bpe":
        from gh05t3_binary.train.bpe_tokenizer import BPETokenizer

        if not tokenizer_path:
            raise ValueError("tokenizer_path is required when tokenizer_type='bpe'")

        if os.path.isfile(tokenizer_path):
            tokenizer = BPETokenizer.load(tokenizer_path)
        else:
            tokenizer = BPETokenizer.train(
                corpus_path=data_path, vocab_size=vocab_size, save_path=tokenizer_path,
            )
        dataset = TokenStreamDataset(texts, tokenizer, seq_len=seq_len)
    else:
        tokenizer = SimpleCharTokenizer()
        dataset = TextDataset(texts, tokenizer, seq_len=seq_len)

    return tokenizer, dataset


def train_binary_transformer(
    data_path: str,
    save_path: str = "binary_checkpoint.pt",
    epochs: int = 5,
    batch_size: int = 8,
    seq_len: int = 64,
    lr: float = 3e-4,
    state=None,
    tokenizer_type: str = "char",
    vocab_size: int = 256,
    tokenizer_path=None,
    num_layers: int = 4,
    dim: int = 256,
    num_heads: int = 4,
):
    """state, if given, is a backend.runtime.gh05t3_orchestrator.GH05T3State
    to update in-process. Progress is also always written to a
    training_state.json sidecar next to save_path -- that's the only way
    a separately-running process (e.g. an API server) can see it, since
    training is normally invoked as its own script/process with no shared
    memory with anything else.

    tokenizer_type: "char" (default, backward compatible -- SimpleCharTokenizer
    + per-line TextDataset) or "bpe" (real subword tokenizer, trained on
    data_path if tokenizer_path doesn't already exist, + TokenStreamDataset
    for full-corpus chunking instead of per-line pad/truncate).
    """
    tokenizer, dataset = _build_tokenizer_and_dataset(
        data_path, seq_len, tokenizer_type, vocab_size, tokenizer_path,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Use the tokenizer's *actual* vocab size for the model, not the
    # requested one -- BPE training can settle on a smaller vocab than
    # requested if the corpus doesn't have enough distinct content.
    model = GH05T3BinaryOSS(
        num_layers=num_layers,
        dim=dim,
        num_heads=num_heads,
        vocab_size=tokenizer.vocab_size,
        binary_ratio=0.95,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    print(f"Training on {device} with {len(dataset)} samples...")

    loss_curve: list = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)  # [B, S, vocab]
            loss = criterion(logits.reshape(-1, tokenizer.vocab_size), y.reshape(-1))
            loss.backward()
            # STE gradients are a rough approximation (they pass through the
            # non-differentiable sign()/round() as if it were identity) and
            # can spike sharply -- clipping is standard practice for
            # binary/quantized network training, not a workaround.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        avg = total_loss / len(loader)
        loss_curve.append(avg)
        print(f"Epoch {epoch+1}/{epochs} — loss={avg:.4f}")

        if state is not None:
            state.binary_training_loss_curve.append(avg)
            state.binary_training_epochs += 1
            state.binary_training_last_checkpoint = save_path

        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        torch.save(
            {
                "model_state": model.state_dict(),
                "vocab_size": tokenizer.vocab_size,
                "num_layers": num_layers,
                "dim": dim,
                "num_heads": num_heads,
                "tokenizer_type": tokenizer_type,
                "tokenizer_path": tokenizer_path if tokenizer_type == "bpe" else None,
            },
            save_path,
        )

        _write_training_state_sidecar(save_path, epoch + 1, avg, loss_curve)

    print(f"Training complete. Saved checkpoint to {save_path}")


def _write_training_state_sidecar(save_path: str, epochs_done: int, last_loss: float, loss_curve) -> None:
    """Writes gh05t3_binary/train/training_state.json (next to save_path's
    directory) so backend/api/binary_training.py can report real progress
    without needing an in-memory reference to this training run."""
    sidecar_path = os.path.join(os.path.dirname(save_path) or ".", "training_state.json")
    with open(sidecar_path, "w") as f:
        json.dump(
            {
                "epochs": epochs_done,
                "last_loss": last_loss,
                "loss_curve": loss_curve,
                "checkpoint": save_path,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--save", default="binary_checkpoint.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--tokenizer", choices=["char", "bpe"], default="char")
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)

    args = parser.parse_args()

    train_binary_transformer(
        data_path=args.data,
        save_path=args.save,
        epochs=args.epochs,
        batch_size=args.batch,
        seq_len=args.seq,
        lr=args.lr,
        tokenizer_type=args.tokenizer,
        vocab_size=args.vocab_size,
        tokenizer_path=args.tokenizer_path,
        num_layers=args.num_layers,
        dim=args.dim,
        num_heads=args.num_heads,
    )
