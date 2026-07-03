"""Real HCM — 146 concept vectors @ 10,000 dims stored in MongoDB.
Projects to 2D with classic PCA (deterministic, numpy-only)."""
from __future__ import annotations
import hashlib
import numpy as np

DIMS = 10_000
ROOM_PALETTE = {
    "Identity": "#f59e0b",
    "Skills": "#22d3ee",
    "Projects": "#c4b5fd",
    "People": "#facc15",
    "Knowledge": "#10b981",
    "Decisions": "#e11d48",
}
ROOMS = list(ROOM_PALETTE.keys())


def seed_vector(label: str, dims: int = DIMS) -> np.ndarray:
    """Deterministic unit vector seeded by label hash."""
    seed = int.from_bytes(hashlib.sha256(label.encode()).digest()[:4], "big")
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dims).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-9
    return v


def make_seed_corpus(total: int = 146) -> list[dict]:
    """Create 146 concept vectors clustered by room (bias toward room mean)."""
    rng = np.random.default_rng(42)
    weights = np.array([9, 18, 11, 2, 29, 14], dtype=float)
    weights = weights / weights.sum()
    counts = (weights * total).round().astype(int)
    # patch to exact total
    counts[np.argmax(counts)] += total - counts.sum()

    concepts = []
    room_means = {r: seed_vector(f"room:{r}") for r in ROOMS}
    idx = 0
    for room, n in zip(ROOMS, counts):
        for j in range(int(n)):
            label = f"{room.lower()}-{j}"
            noise = rng.standard_normal(DIMS).astype(np.float32) * 0.18
            v = room_means[room] + noise
            v /= np.linalg.norm(v) + 1e-9
            concepts.append({"idx": idx, "label": label, "room": room, "vec": v})
            idx += 1
    return concepts


def pca_2d(vectors: np.ndarray) -> np.ndarray:
    """Project NxD float matrix to Nx2 via top-2 principal components."""
    X = vectors - vectors.mean(axis=0, keepdims=True)
    # For 146x10000, use the Gram trick: eig of X X^T (146x146)
    gram = X @ X.T
    vals, vecs = np.linalg.eigh(gram)
    # take top 2
    order = np.argsort(vals)[::-1][:2]
    top = vecs[:, order]
    # scale by sqrt eigenvalue
    scale = np.sqrt(np.maximum(vals[order], 1e-9))
    proj = top * scale
    return proj.astype(np.float32)


def build_cloud(concepts: list[dict]) -> list[dict]:
    mat = np.stack([c["vec"] for c in concepts])
    pts = pca_2d(mat)
    # normalize to [-1, 1]
    pts = pts - pts.mean(axis=0)
    m = np.abs(pts).max() + 1e-9
    pts = pts / m
    out = []
    for c, xy in zip(concepts, pts):
        out.append({
            "idx": c["idx"],
            "label": c["label"],
            "room": c["room"],
            "color": ROOM_PALETTE[c["room"]],
            "x": float(xy[0]),
            "y": float(xy[1]),
        })
    return out


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-9) * (np.linalg.norm(b) + 1e-9)))
