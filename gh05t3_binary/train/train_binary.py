import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from gh05t3_binary.oss.integration import GH05T3BinaryOSS
from gh05t3_binary.train.tokenizer import SimpleCharTokenizer
from gh05t3_binary.train.dataset import TextDataset


def train_binary_transformer(
    data_path: str,
    save_path: str = "binary_checkpoint.pt",
    epochs: int = 5,
    batch_size: int = 8,
    seq_len: int = 64,
    lr: float = 3e-4,
):
    tokenizer = SimpleCharTokenizer()

    with open(data_path, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f.readlines() if line.strip()]

    dataset = TextDataset(texts, tokenizer, seq_len=seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = GH05T3BinaryOSS(
        num_layers=4,
        dim=256,
        num_heads=4,
        vocab_size=tokenizer.vocab_size,
        binary_ratio=0.95,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    print(f"Training on {device} with {len(dataset)} samples...")

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
        print(f"Epoch {epoch+1}/{epochs} — loss={avg:.4f}")

        torch.save(
            {
                "model_state": model.state_dict(),
                "vocab_size": tokenizer.vocab_size,
                "num_layers": 4,
                "dim": 256,
                "num_heads": 4,
            },
            save_path,
        )

    print(f"Training complete. Saved checkpoint to {save_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--save", default="binary_checkpoint.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)

    args = parser.parse_args()

    train_binary_transformer(
        data_path=args.data,
        save_path=args.save,
        epochs=args.epochs,
        batch_size=args.batch,
        seq_len=args.seq,
        lr=args.lr,
    )
