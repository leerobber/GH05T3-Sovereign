"""Exports a trained gh05t3_binary checkpoint's real 1-bit weights into a
packed u64 bitmask file, for a future SIMD inference kernel.

IMPORTANT architectural note, confirmed by reading BinaryLinear.forward()
directly (F.linear(x, binary_weight) -- x is used unmodified, only
binary_weight is sign()-quantized): this model is weight-only binarization
(BWN-style), not full XNOR-Net. Activations are never binarized. That
means XNOR+popcount -- which requires BOTH operands to be +-1 bits -- is
mathematically invalid here. The correct fast kernel for this checkpoint
is a *signed accumulation* (x_i if w_i==+1 else -x_i, summed), not
XNOR+popcount. This module only packs weight bits, which is valid and
useful regardless of which kernel consumes them; building that kernel is
a separate, later step.

Only true 1-bit layers (real gh05t3_binary.core.binary_layers.BinaryLinear
instances -- found by walking the real nn.Module tree, not by guessing at
key-name strings) are packed here. MultiBitLinear4 (4-bit, used for each
block's out_proj and the final head), TernaryLinear (2-bit, used for
attention's out_proj -- see pack_ternary.py for its own exporter), and
all full-precision parameters (embedding, LayerNorm, denorm, basis,
log_temperature) are explicitly left out and reported as skipped, not
silently dropped.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from gh05t3_binary.core.binary_layers import BinaryLinear
from gh05t3_binary.oss.integration import GH05T3BinaryOSS


def pack_signs_row_major(signs: torch.Tensor) -> np.ndarray:
    """signs: (out_features, in_features) tensor of +1.0/-1.0 values.
    Returns a (out_features, k_packed) uint64 array -- bit b of word w in
    row r is 1 if signs[r, w*64 + b] == +1, else 0. Rows are padded with
    0-bits (representing -1) if in_features isn't a multiple of 64; the
    corresponding activation elements must be zero-padded too by whatever
    consumes this (so the padding contributes nothing either way)."""
    out_features, in_features = signs.shape
    k_packed = (in_features + 63) // 64
    padded_len = k_packed * 64

    bits = (signs > 0).to(torch.uint8).numpy()  # 1 where +1, 0 where -1
    if padded_len != in_features:
        pad = np.zeros((out_features, padded_len - in_features), dtype=np.uint8)
        bits = np.concatenate([bits, pad], axis=1)

    bits = bits.reshape(out_features, k_packed, 64)
    weights = (1 << np.arange(64, dtype=np.uint64))
    packed = (bits.astype(np.uint64) * weights).sum(axis=2, dtype=np.uint64)
    return packed  # (out_features, k_packed) uint64


def unpack_signs_row_major(packed: np.ndarray, in_features: int) -> torch.Tensor:
    """Inverse of pack_signs_row_major, for round-trip verification.
    Returns a (out_features, in_features) tensor of +1.0/-1.0 values."""
    out_features, k_packed = packed.shape
    bits = np.zeros((out_features, k_packed * 64), dtype=np.uint8)
    for b in range(64):
        bits[:, b::64] = ((packed >> np.uint64(b)) & np.uint64(1)).astype(np.uint8)
    bits = bits[:, :in_features]
    return torch.from_numpy(bits).to(torch.float32) * 2.0 - 1.0  # 0/1 -> -1/+1


def find_binary_linear_layers(model: torch.nn.Module) -> list[tuple[str, BinaryLinear]]:
    return [(name, mod) for name, mod in model.named_modules() if isinstance(mod, BinaryLinear)]


def export_packed_checkpoint(checkpoint_path: str, output_dir: str) -> dict:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    model = GH05T3BinaryOSS(
        num_layers=ckpt["num_layers"],
        dim=ckpt["dim"],
        num_heads=ckpt["num_heads"],
        vocab_size=ckpt["vocab_size"],
        binary_ratio=1.0,
        stabilizer=ckpt.get("stabilizer", "mgc"),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    binary_layers = find_binary_linear_layers(model)
    total_params = sum(p.numel() for p in model.parameters())
    binary_params = sum(mod.weight.numel() for _, mod in binary_layers)

    os.makedirs(output_dir, exist_ok=True)
    bin_path = os.path.join(output_dir, "packed_weights.bin")
    meta_path = os.path.join(output_dir, "packed_weights.json")

    layers_meta = []
    with open(bin_path, "wb") as f:
        offset = 0
        for name, mod in binary_layers:
            out_features, in_features = mod.weight.shape
            with torch.no_grad():
                signs = torch.sign(mod.weight)
                # torch.sign(0.0) == 0.0, ambiguous for +-1 packing. Convention:
                # treat exact zero as +1. Verified this checkpoint has zero
                # such elements (see test_pack_weights.py), so this branch is
                # never actually exercised here -- it's a documented default
                # for the (currently unseen) edge case, not a silent guess.
                signs = torch.where(signs == 0, torch.ones_like(signs), signs)

            packed = pack_signs_row_major(signs)
            packed_bytes = packed.tobytes()
            f.write(packed_bytes)

            layers_meta.append({
                "name": name,
                "out_features": out_features,
                "in_features": in_features,
                "k_packed": packed.shape[1],
                "byte_offset": offset,
                "byte_length": len(packed_bytes),
            })
            offset += len(packed_bytes)

    # Walk every module (leaf or not) and collect params declared *directly*
    # on it (recurse=False) -- e.g. MagnitudeAwareINBL.denorm lives directly
    # on a module that also has a `binary_linear` child, so it would be
    # missed entirely if non-leaf modules were skipped here.
    skipped = []
    exported_names = {name for name, _ in binary_layers}
    for name, mod in model.named_modules():
        if name in exported_names:
            continue
        own_params = [p_name for p_name, _ in mod.named_parameters(recurse=False)]
        if own_params:
            skipped.append({"name": name, "type": type(mod).__name__, "params": own_params})

    metadata = {
        "source_checkpoint": os.path.abspath(checkpoint_path),
        "model_config": {
            "num_layers": ckpt["num_layers"], "dim": ckpt["dim"],
            "num_heads": ckpt["num_heads"], "vocab_size": ckpt["vocab_size"],
            "stabilizer": ckpt.get("stabilizer", "mgc"),
        },
        "total_model_params": total_params,
        "exported_binary_params": binary_params,
        "exported_binary_params_pct": round(100 * binary_params / total_params, 2),
        "packed_weights_file": os.path.basename(bin_path),
        "layers": layers_meta,
        "skipped_non_binary_layers": skipped,
    }

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def verify_round_trip(checkpoint_path: str, output_dir: str) -> bool:
    """Loads the exported .bin/.json back and confirms every packed layer
    unpacks to exactly the original sign(weight) -- proves the packing is
    lossless before anything downstream trusts it."""
    meta_path = os.path.join(output_dir, "packed_weights.json")
    bin_path = os.path.join(output_dir, "packed_weights.bin")

    with open(meta_path) as f:
        meta = json.load(f)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = GH05T3BinaryOSS(
        num_layers=ckpt["num_layers"], dim=ckpt["dim"], num_heads=ckpt["num_heads"],
        vocab_size=ckpt["vocab_size"], binary_ratio=1.0, stabilizer=ckpt.get("stabilizer", "mgc"),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    by_name = dict(find_binary_linear_layers(model))

    with open(bin_path, "rb") as f:
        raw = f.read()

    all_match = True
    for layer in meta["layers"]:
        chunk = raw[layer["byte_offset"]: layer["byte_offset"] + layer["byte_length"]]
        packed = np.frombuffer(chunk, dtype=np.uint64).reshape(layer["out_features"], layer["k_packed"])
        unpacked = unpack_signs_row_major(packed, layer["in_features"])

        original = torch.sign(by_name[layer["name"]].weight.detach())
        original = torch.where(original == 0, torch.ones_like(original), original)

        if not torch.equal(unpacked, original):
            print(f"MISMATCH in layer {layer['name']!r}")
            all_match = False

    return all_match


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="gh05t3_binary/train/checkpoints/binary_v2.pt")
    parser.add_argument("--out", default="gh05t3_binary/inference/checkpoints")
    args = parser.parse_args()

    meta = export_packed_checkpoint(args.checkpoint, args.out)
    print(f"Exported {len(meta['layers'])} binary layers "
          f"({meta['exported_binary_params']:,} / {meta['total_model_params']:,} params, "
          f"{meta['exported_binary_params_pct']}%) to {args.out}/")
    print(f"Skipped {len(meta['skipped_non_binary_layers'])} non-binary parameter groups "
          f"(full-precision or 4-bit -- see packed_weights.json for the full list).")

    print("Verifying round-trip losslessness...")
    ok = verify_round_trip(args.checkpoint, args.out)
    print("Round-trip verification:", "PASSED" if ok else "FAILED")
