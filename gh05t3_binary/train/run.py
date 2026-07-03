import os

import yaml

from gh05t3_binary.train.train_binary import train_binary_transformer

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

if __name__ == "__main__":
    with open(_CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    train_binary_transformer(
        data_path=cfg["dataset"],
        save_path=cfg["checkpoint"],
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        seq_len=cfg["seq_len"],
        lr=float(cfg["lr"]),
    )
