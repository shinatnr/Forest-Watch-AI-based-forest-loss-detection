"""Classical detectors: Tier 0 NDVI difference, Tier 1 Random Forest / XGBoost per-pixel classifiers.
All detectors expose  predict_proba(a, b) -> (H,W) float map in 0-1  for numpy images (C,H,W)."""
import numpy as np
from .common import veg_index, ndwi, evi, N_BANDS, convert_kind


class NDVIDiff:
    """Tier 0: score = how much vegetation index dropped, only where it was vegetated before."""
    name = "ndvi_diff"
    meta = dict(type="ndvi", trained_on="none (rule-based)")

    def __init__(self, kind="rgb", gain=2.0):
        self.kind, self.gain = kind, gain
        self.forest_min = 0.3 if kind == "bgrn" else 0.15

    def predict_proba(self, a, b, kind=None):
        a, b = convert_kind(a, kind or self.kind, self.kind), convert_kind(b, kind or self.kind, self.kind)
        va, vb = veg_index(a, self.kind), veg_index(b, self.kind)
        return (np.clip((va - vb) * self.gain, 0, 1) * (va > self.forest_min)).astype(np.float32)


def pixel_features(a, b, kind):
    """Per-pixel feature matrix (H*W, F): bands of both dates, their differences, NDVI (+NDWI, EVI
    when a NIR band exists) of both dates and their differences."""
    feats = [a, b, b - a]
    va, vb = veg_index(a, kind), veg_index(b, kind)
    feats += [va[None], vb[None], (vb - va)[None]]
    for fn in (ndwi, evi):
        fa, fb = fn(a, kind), fn(b, kind)
        if fa is not None:
            feats += [fa[None], fb[None], (fb - fa)[None]]
    f = np.concatenate(feats, 0)
    return f.reshape(f.shape[0], -1).T


class PixelClassifier:
    """Tier 1: Random Forest or XGBoost on per-pixel features."""

    def __init__(self, algo="rf", kind="rgb", n_estimators=100, seed=42):
        from sklearn.ensemble import RandomForestClassifier
        self.algo, self.kind, self.name = algo, kind, algo
        self.meta = dict(type=algo, trained_on="unknown")
        if algo == "rf":
            self.clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=14, n_jobs=-1, random_state=seed)
        elif algo == "xgb":
            from xgboost import XGBClassifier
            self.clf = XGBClassifier(n_estimators=n_estimators, max_depth=6, learning_rate=0.1,
                                     tree_method="hist", n_jobs=2, random_state=seed, eval_metric="logloss")
        else:
            raise ValueError(algo)

    def fit(self, tiles, max_pixels=400_000, seed=42):
        """tiles: iterable of (a, b, mask) numpy. Samples a class-balanced set of pixels."""
        rng = np.random.default_rng(seed)
        X, y = [], []
        for a, b, m in tiles:
            f, lab = pixel_features(a, b, self.kind), m.ravel() > 0.5
            pos, neg = np.where(lab)[0], np.where(~lab)[0]
            k = min(len(pos), 300)
            sel = np.concatenate([rng.choice(pos, k, replace=False) if k else pos[:0],
                                  rng.choice(neg, min(len(neg), max(k, 100)), replace=False)])
            X.append(f[sel]); y.append(lab[sel])
        X, y = np.concatenate(X), np.concatenate(y)
        if len(y) > max_pixels:
            i = rng.choice(len(y), max_pixels, replace=False); X, y = X[i], y[i]
        if y.sum() == 0 or y.sum() == len(y):
            raise ValueError("Training pixels contain only one class.")
        self.clf.fit(X, y)
        return self

    def predict_proba(self, a, b, kind=None):
        a, b = convert_kind(a, kind or self.kind, self.kind), convert_kind(b, kind or self.kind, self.kind)
        h, w = a.shape[1:]
        return self.clf.predict_proba(pixel_features(a, b, self.kind))[:, 1].reshape(h, w).astype(np.float32)
