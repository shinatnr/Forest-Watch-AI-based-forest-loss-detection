"""Datasets: synthetic tiles (demo/tests), on-disk tiles (real data), scene loading, spatial split.

On-disk layout:  data/{train,val,test}/{A,B,mask}/<region>__<tile>.tif   (A=before, B=after)
The text before the first "__" is the REGION id; all tiles of one region must be in one split
(spatial independence; see prepare_data.py check/split).
"""
import os, glob, random
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter
from .common import N_BANDS, model_input

# ---------------------------------------------------------------- reading images
def _read_tif(path, window=None, nbands=None):
    import rasterio
    from rasterio.windows import Window
    with rasterio.open(path) as src:
        w = Window(*window) if window else None
        arr = src.read(window=w) if nbands is None else src.read(list(range(1, nbands + 1)), window=w)
        return arr, src.dtypes[0]


def read_any(path, window=None, nbands=None):
    """Return (C,H,W) float32 in 0-1. window=(col_off,row_off,width,height).
    GeoTIFF integer reflectance (Sentinel-2, 0-10000) is divided by 10000."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        arr, dt = _read_tif(path, window, nbands)
        arr = arr.astype(np.float32)
        if dt.startswith(("uint", "int")) and dt != "uint8":
            arr = arr / 10000.0
        elif dt == "uint8":
            arr = arr / 255.0
        arr = np.clip(arr, 0, 1)
    elif ext == ".npy":
        arr = np.load(path).astype(np.float32)
        if window:
            c, r, w, h = window; arr = arr[:, r:r + h, c:c + w]
    else:
        from PIL import Image
        a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32).transpose(2, 0, 1) / 255.0
        if window:
            c, r, w, h = window; a = a[:, r:r + h, c:c + w]
        arr = a
    return arr


def read_mask(path, window=None):
    """Binary mask (H,W) bool: any non-zero value is 'cleared' (works for 0/1 and 0/255 masks;
    masks must NOT be rescaled like images are)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        arr, _ = _read_tif(path, window, 1)
        return arr[0] > 0
    if ext == ".npy":
        a = np.load(path)
        a = a[0] if a.ndim == 3 else a
        if window:
            c, r, w, h = window; a = a[r:r + h, c:c + w]
        return a > 0
    from PIL import Image
    a = np.asarray(Image.open(path).convert("L"))
    if window:
        c, r, w, h = window; a = a[r:r + h, c:c + w]
    return a > 0


def image_size(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        import rasterio
        with rasterio.open(path) as s:
            return s.height, s.width
    if ext == ".npy":
        a = np.load(path, mmap_mode="r"); return a.shape[1], a.shape[2]
    from PIL import Image
    w, h = Image.open(path).size
    return h, w


def load_scene(path):
    """Load one user image for inference.
    Returns dict: img (C,H,W) 0-1, kind, transform, crs, scl (H,W int or None), georef (bool).
    GeoTIFF with 4 bands = B02,B03,B04,B08; with 5 bands the 5th is the Sentinel-2 SCL layer."""
    ext = os.path.splitext(path)[1].lower()
    out = dict(transform=None, crs=None, scl=None, georef=False)
    if ext in (".tif", ".tiff"):
        import rasterio
        with rasterio.open(path) as src:
            n = src.count
            arr = src.read().astype(np.float32)
            dt = src.dtypes[0]
            if src.crs is not None and src.transform is not None and not src.transform.is_identity:
                out.update(transform=src.transform, crs=src.crs, georef=True)
        if n >= 5:
            out["scl"] = arr[4].astype(np.int32); arr = arr[:4]
        elif n == 4:
            arr = arr[:4]
        elif n >= 3:
            arr = arr[:3]
        else:
            raise ValueError("GeoTIFF needs at least 3 bands (RGB) or 4 bands (B02,B03,B04,B08).")
        if dt != "uint8" and dt.startswith(("uint", "int")):
            arr = arr / 10000.0
        elif dt == "uint8":
            arr = arr / 255.0
        out["img"] = np.clip(arr, 0, 1).astype(np.float32)
        out["kind"] = "bgrn" if arr.shape[0] == 4 else "rgb"
    else:
        out["img"] = read_any(path)
        out["kind"] = "rgb"
    return out


# ---------------------------------------------------------------- augmentation / tensors
def augment(a, b, m, rng):
    """Same random flip + 90-degree rotation applied to BEFORE, AFTER and mask."""
    k = rng.randint(0, 3)
    a, b, m = (np.rot90(x, k, axes=(-2, -1)) for x in (a, b, m))
    if rng.random() < 0.5:
        a, b, m = (x[..., ::-1] for x in (a, b, m))
    if rng.random() < 0.5:
        a, b, m = (x[..., ::-1, :] for x in (a, b, m))
    return np.ascontiguousarray(a), np.ascontiguousarray(b), np.ascontiguousarray(m)


class _TileBase(Dataset):
    """Common __getitem__: raw numpy -> (optional augment) -> model_input tensors."""
    kind = "rgb"; train = False

    def raw(self, i):  # -> a (C,H,W), b (C,H,W), m (H,W) numpy
        raise NotImplementedError

    def __getitem__(self, i):
        a, b, m = self.raw(i)
        if self.train:
            a, b, m = augment(a, b, m[None], self._rng); m = m[0]
        xa, xb = model_input(a, self.kind), model_input(b, self.kind)
        return torch.from_numpy(xa), torch.from_numpy(xb), torch.from_numpy(m[None].astype(np.float32))


def select_tiles(change_frac, min_change=0.01, neg_frac=0.2, seed=42):
    """Class-imbalance filter: keep every tile with >= min_change changed pixels, and only a
    fraction `neg_frac` of the rest (so the model still sees 'no change' examples)."""
    rng = random.Random(seed)
    return [i for i, f in enumerate(change_frac) if f >= min_change or rng.random() < neg_frac]


# ---------------------------------------------------------------- synthetic data
def _blobs(size, rng, n, rmin, rmax):
    """Binary mask of n random smooth blobs (the 'clearings')."""
    yy, xx = np.mgrid[:size, :size]
    m = np.zeros((size, size), bool)
    for _ in range(n):
        cy, cx = rng.integers(0, size, 2)
        ry, rx = rng.integers(rmin, rmax, 2)
        ang = rng.random() * np.pi
        y, x = yy - cy, xx - cx
        u, v = x * np.cos(ang) + y * np.sin(ang), -x * np.sin(ang) + y * np.cos(ang)
        m |= (u / rx) ** 2 + (v / ry) ** 2 <= 1
    return m


def _render(cover, kind, rng, tint=0.0):
    """cover: 1 = forest, 0 = bare soil. Returns (C,H,W) image."""
    forest = np.array([0.10, 0.33, 0.12, 0.55]); bare = np.array([0.45, 0.33, 0.20, 0.25])  # R,G,B,NIR
    c = cover[None] * forest[:, None, None] + (1 - cover[None]) * bare[:, None, None]
    rgb = c[:3]; nir = c[3:4]
    rgb = rgb + tint
    img = rgb if kind == "rgb" else np.concatenate([rgb[[2, 1, 0]], nir + tint], 0)
    return img


def make_synthetic_pair(size=64, kind="rgb", rng=None, pos_prob=0.7):
    """One synthetic BEFORE/AFTER pair + change mask. Includes distractors that are NOT change:
    permanent bare patches, a season/brightness shift and sensor noise."""
    rng = rng or np.random.default_rng()
    tex = gaussian_filter(rng.random((size, size)), 2); tex = (tex - tex.min()) / (np.ptp(tex) + 1e-6)
    cover_a = 0.85 + 0.15 * tex
    cover_a[_blobs(size, rng, rng.integers(0, 3), 3, max(4, size // 10))] = 0.0   # permanent bare soil
    mask = np.zeros((size, size), bool)
    if rng.random() < pos_prob:
        mask = _blobs(size, rng, rng.integers(1, 4), 4, max(6, size // 5)) & (cover_a > 0.5)
    # cleared pixels become (partly) bare soil; clearings are not always perfectly bare
    cover_b = np.where(mask, rng.uniform(0.0, 0.3, mask.shape) * gaussian_filter(rng.random(mask.shape), 1.5) * 2, cover_a)
    # DISTRACTOR: seasonal browning / dry patches (cover drops partly but nothing was cleared)
    if rng.random() < 0.6:
        dry = gaussian_filter(rng.random((size, size)), size / 8); dry = dry > np.quantile(dry, 0.7)
        cover_b = np.where(dry & ~mask, cover_b * rng.uniform(0.45, 0.75), cover_b)
    a = _render(cover_a, kind, rng, tint=rng.normal(0, 0.02))
    b = _render(np.clip(cover_b, 0, 1), kind, rng, tint=rng.normal(0, 0.04))
    a = a * rng.uniform(0.9, 1.1) + rng.normal(0, 0.03, a.shape)             # illumination + sensor noise
    b = b * rng.uniform(0.85, 1.15) + rng.normal(0, 0.04, b.shape)
    return np.clip(a, 0, 1).astype(np.float32), np.clip(b, 0, 1).astype(np.float32), mask.astype(np.float32)


class SyntheticTiles(_TileBase):
    def __init__(self, n=200, size=64, kind="rgb", seed=0, train=False, min_change=0.0, neg_frac=1.0):
        rng = np.random.default_rng(seed)
        items = [make_synthetic_pair(size, kind, rng) for _ in range(n)]
        keep = select_tiles([m.mean() for _, _, m in items], min_change, neg_frac, seed) if min_change > 0 else range(n)
        self.items = [items[i] for i in keep]
        self.kind, self.train, self._rng = kind, train, random.Random(seed)

    def __len__(self): return len(self.items)
    def raw(self, i): return self.items[i]


# ---------------------------------------------------------------- on-disk data
class DiskTiles(_TileBase):
    """Tiles cut on a grid from data/<split>/{A,B,mask}. Large rasters are read window by window."""

    def __init__(self, split_dir, kind=None, tile=256, train=False, min_change=0.0, neg_frac=1.0, seed=42):
        fa = sorted(glob.glob(os.path.join(split_dir, "A", "*")))
        if not fa:
            raise FileNotFoundError(f"No files in {split_dir}/A. See README (data preparation).")
        self.index, self.change = [], []
        for p in fa:
            name = os.path.basename(p)
            pb, pm = os.path.join(split_dir, "B", name), os.path.join(split_dir, "mask", name)
            if not (os.path.exists(pb) and os.path.exists(pm)):
                continue
            h, w = image_size(p)
            for r in range(0, max(h - tile, 0) + 1, tile):
                for c in range(0, max(w - tile, 0) + 1, tile):
                    win = (c, r, min(tile, w), min(tile, h))
                    m = read_mask(pm, win)
                    self.index.append((p, pb, pm, win)); self.change.append(float(m.mean()))
        if kind is None:
            kind = "bgrn" if read_any(self.index[0][0], self.index[0][3], 4).shape[0] >= 4 else "rgb"
        self.kind = kind
        keep = select_tiles(self.change, min_change, neg_frac, seed) if min_change > 0 else range(len(self.index))
        self.index = [self.index[i] for i in keep]
        self.train, self._rng = train, random.Random(seed)

    def __len__(self): return len(self.index)

    def raw(self, i):
        pa, pb, pm, win = self.index[i]
        n = N_BANDS[self.kind]
        a, b = read_any(pa, win, n)[:n], read_any(pb, win, n)[:n]
        m = read_mask(pm, win).astype(np.float32)
        return a, b, m


def region_of(path):
    return os.path.basename(path).split("__")[0]


def split_by_region(files, ratios=(0.7, 0.15, 0.15), seed=42):
    """Group files by region id and assign WHOLE regions to train/val/test (no spatial leakage)."""
    regions = sorted({region_of(f) for f in files})
    random.Random(seed).shuffle(regions)
    n = len(regions); n_tr = max(1, round(ratios[0] * n)); n_va = max(1, round(ratios[1] * n)) if n > 2 else 0
    assign = {}
    for i, r in enumerate(regions):
        assign[r] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
    return {f: assign[region_of(f)] for f in files}
