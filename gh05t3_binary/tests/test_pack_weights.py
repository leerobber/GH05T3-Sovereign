import os

import torch

from gh05t3_binary.inference.pack_weights import (
    export_packed_checkpoint,
    find_binary_linear_layers,
    pack_signs_row_major,
    unpack_signs_row_major,
    verify_round_trip,
)
from gh05t3_binary.oss.integration import GH05T3BinaryOSS


def test_pack_unpack_round_trip_no_padding():
    torch.manual_seed(0)
    signs = torch.sign(torch.randn(5, 128))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    packed = pack_signs_row_major(signs)
    assert packed.shape == (5, 2)  # 128 / 64 = 2 words, no padding
    assert torch.equal(unpack_signs_row_major(packed, 128), signs)


def test_pack_unpack_round_trip_with_padding():
    torch.manual_seed(1)
    signs = torch.sign(torch.randn(3, 70))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    packed = pack_signs_row_major(signs)
    assert packed.shape == (3, 2)  # ceil(70/64) = 2 words
    assert torch.equal(unpack_signs_row_major(packed, 70), signs)


def test_export_real_small_checkpoint_round_trips(tmp_path):
    """End-to-end: build a small (untrained, random-weight) model, save it
    in the same checkpoint format train_binary.py produces, export its
    packed weights, and verify the round trip -- proves the export
    pipeline works against a real GH05T3BinaryOSS, not just synthetic
    tensors."""
    model = GH05T3BinaryOSS(num_layers=2, dim=128, num_heads=4, vocab_size=64, binary_ratio=1.0)
    ckpt_path = tmp_path / "tiny.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "vocab_size": 64, "num_layers": 2, "dim": 128, "num_heads": 4,
            "stabilizer": "mgc",
        },
        ckpt_path,
    )

    out_dir = tmp_path / "packed"
    meta = export_packed_checkpoint(str(ckpt_path), str(out_dir))

    expected_binary_layers = find_binary_linear_layers(model)
    assert len(meta["layers"]) == len(expected_binary_layers)
    assert meta["exported_binary_params"] > 0
    assert os.path.isfile(out_dir / "packed_weights.bin")
    assert os.path.isfile(out_dir / "packed_weights.json")

    assert verify_round_trip(str(ckpt_path), str(out_dir)) is True


def test_skipped_and_exported_cover_every_parameterized_module(tmp_path):
    """Every module in the model that owns at least one direct parameter
    must appear in exactly one of {exported layers, skipped list} -- proves
    the reporting fix (which used to silently miss non-leaf modules like
    MagnitudeAwareINBL that have both children and their own params) covers
    everything now."""
    model = GH05T3BinaryOSS(num_layers=2, dim=128, num_heads=4, vocab_size=64, binary_ratio=1.0)
    ckpt_path = tmp_path / "tiny.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "vocab_size": 64, "num_layers": 2, "dim": 128, "num_heads": 4,
            "stabilizer": "mgc",
        },
        ckpt_path,
    )
    meta = export_packed_checkpoint(str(ckpt_path), str(tmp_path / "packed"))

    exported_names = {layer["name"] for layer in meta["layers"]}
    skipped_names = {s["name"] for s in meta["skipped_non_binary_layers"]}

    for name, mod in model.named_modules():
        own_params = [p_name for p_name, _ in mod.named_parameters(recurse=False)]
        if not own_params:
            continue
        assert name in exported_names or name in skipped_names, (
            f"module {name!r} owns parameters but is neither exported nor reported as skipped"
        )
