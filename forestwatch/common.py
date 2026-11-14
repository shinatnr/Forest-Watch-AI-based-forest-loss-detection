"""Shared helpers: seeds, device, band conventions and vegetation indices.

Image convention used everywhere: float32 numpy array, shape (C, H, W), values 0-1.
Two band orders ("kinds") are supported:
    "rgb"  : [R, G, B]            (PNG/JPG screenshots, drone images)
    "bgrn" : [B02, B03, B04, B08] (Sentinel-2 Blue, Green, Red, NIR at 10 m)
"""
import random
import numpy as np
import torch

EPS = 1e-6
_IDX = {"rgb": {"R": 0, "G": 1, "B": 2}, "bgrn": {"B": 0, "G": 1, "R": 2, "N": 3}}
N_BANDS = {"rgb": 3, "bgrn": 4}


def set_seed(seed: int = 42) -> None:
    """Fix all random seeds so runs are repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Use the GPU automatically if one is present, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def band(img: np.ndarray, kind: str, name: str) -> np.ndarray:
    return img[_IDX[kind][name]]


def veg_index(img: np.ndarray, kind: str) -> np.ndarray:
    """Vegetation index in roughly [-1, 1].
    bgrn: NDVI = (NIR - Red) / (NIR + Red).
    rgb : normalised excess-green (2G - R - B) / (R + G + B), a visible-light proxy for NDVI."""
    if kind == "bgrn":
        n, r = band(img, kind, "N"), band(img, kind, "R")
        return (n - r) / (n + r + EPS)
    r, g, b = (band(img, kind, c) for c in "RGB")
    return (2 * g - r - b) / (r + g + b + EPS)


def ndwi(img: np.ndarray, kind: str):
    """Water index (G - NIR)/(G + NIR); only defined for Sentinel-2 style input."""
    if kind != "bgrn":
        return None
    g, n = band(img, kind, "G"), band(img, kind, "N")
    return (g - n) / (g + n + EPS)


def evi(img: np.ndarray, kind: str):
    """Enhanced vegetation index; only defined when a NIR band exists."""
    if kind != "bgrn":
        return None
    n, r, b = (band(img, kind, c) for c in "NRB")
    den = n + 6 * r - 7.5 * b + 1
    den = np.where(np.abs(den) < 1e-3, 1e-3, den)                    # guard against division by ~0
    return np.clip(2.5 * (n - r) / den, -1, 1)


def model_input(img: np.ndarray, kind: str) -> np.ndarray:
    """Stack the raw bands plus the vegetation index (scaled to 0-1) -> the network input.
    rgb -> 4 channels, bgrn -> 5 channels (bands B02,B03,B04,B08 + NDVI)."""
    v = ((veg_index(img, kind) + 1) / 2).astype(np.float32)
    return np.concatenate([img, v[None]], axis=0).astype(np.float32)


def model_in_channels(kind: str) -> int:
    return N_BANDS[kind] + 1


def convert_kind(img: np.ndarray, src: str, dst: str) -> np.ndarray:
    """Convert between band orders. bgrn -> rgb works (drop NIR); rgb -> bgrn is impossible."""
    if src == dst:
        return img
    if src == "bgrn" and dst == "rgb":
        return img[[2, 1, 0]]
    raise ValueError(f"Cannot convert {src} input to {dst}: the NIR band is missing.")
