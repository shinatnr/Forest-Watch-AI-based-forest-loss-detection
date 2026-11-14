"""Inference: UNetDetector (tiled, overlap-blended), detector loading, and the full detection pipeline."""
import os, joblib
import numpy as np
import torch
from .common import convert_kind, model_input, get_device
from .models import SiameseUNet
from .train import load_meta
from . import preprocess as pp
from . import postprocess as post
from .baselines import NDVIDiff, PixelClassifier


class UNetDetector:
    def __init__(self, path, device=None, tile=256, overlap=32):
        self.meta = load_meta(path)
        if not self.meta:
            raise FileNotFoundError(f"Missing metadata file next to {path} (expected {os.path.splitext(path)[0]}.json)")
        self.device = device or get_device()
        self.kind, self.tile, self.overlap = self.meta["kind"], tile, overlap
        self.model = SiameseUNet(self.meta["in_ch"], self.meta["base"], self.meta["levels"], self.meta["mode"]).to(self.device)
        self.model.load_state_dict(torch.load(path, map_location=self.device)); self.model.eval()
        self.name = f"unet_{self.meta['mode']}"

    @torch.no_grad()
    def _run(self, xa, xb):
        xa, xb = torch.from_numpy(xa)[None].to(self.device), torch.from_numpy(xb)[None].to(self.device)
        return torch.sigmoid(self.model(xa, xb))[0, 0].cpu().numpy()

    def predict_proba(self, a, b, kind=None):
        """a, b: (C,H,W) numpy of band order `kind` (default: the model's). Returns (H,W) probabilities.
        Large scenes are processed in overlapping tiles and blended with a smooth window."""
        kind = kind or self.kind
        a, b = convert_kind(a, kind, self.kind), convert_kind(b, kind, self.kind)
        xa, xb = model_input(a, self.kind), model_input(b, self.kind)
        h, w = xa.shape[1:]; d = self.model.divisor
        ph, pw = (-h) % d, (-w) % d
        pad = ((0, 0), (0, ph), (0, pw))
        xa, xb = np.pad(xa, pad, mode="reflect"), np.pad(xb, pad, mode="reflect")
        H, W = xa.shape[1:]; t = self.tile
        if H <= t and W <= t:
            return self._run(xa, xb)[:h, :w]
        t = max(t - t % d, d); step = t - self.overlap
        ys = sorted(set(list(range(0, max(H - t, 0) + 1, step)) + [max(H - t, 0)]))
        xs = sorted(set(list(range(0, max(W - t, 0) + 1, step)) + [max(W - t, 0)]))
        win1 = np.clip(np.hanning(t + 2)[1:-1], 1e-3, None); win = np.outer(win1, win1).astype(np.float32)
        acc, wsum = np.zeros((H, W), np.float32), np.zeros((H, W), np.float32)
        for y in ys:
            for x in xs:
                th, tw = min(t, H - y), min(t, W - x)
                p = self._run(xa[:, y:y + th, x:x + tw], xb[:, y:y + th, x:x + tw])
                acc[y:y + th, x:x + tw] += p * win[:th, :tw]; wsum[y:y + th, x:x + tw] += win[:th, :tw]
        return (acc / np.maximum(wsum, 1e-6))[:h, :w]


class KindAdapter:
    """Wrap a classical detector so it accepts input of another band order (bgrn -> rgb)."""
    def __init__(self, det): self.det, self.name, self.meta, self.kind = det, det.name, det.meta, det.kind

    def predict_proba(self, a, b, kind=None):
        kind = kind or self.kind
        return self.det.predict_proba(convert_kind(a, kind, self.kind), convert_kind(b, kind, self.kind))


def load_detector(path, device=None):
    """Load a saved detector: *.pt -> Siamese U-Net / UNet-diff, *.joblib -> Random Forest / XGBoost."""
    if path.endswith(".pt"):
        return UNetDetector(path, device)
    obj = joblib.load(path); det = PixelClassifier(obj["algo"], obj["kind"]); det.clf = obj["clf"]
    det.meta = dict(load_meta(path), type=obj["algo"]); return KindAdapter(det)


def save_pixel_classifier(det, path, meta):
    from .train import save_meta
    joblib.dump(dict(algo=det.algo, kind=det.kind, clf=det.clf), path)
    det.meta = meta; save_meta(path, dict(meta, type=det.algo, kind=det.kind))


def is_validated(meta):
    """A detector counts as validated only if it was trained on REAL labelled data and has held-out test metrics."""
    return meta.get("trained_on") == "real" and "test_metrics" in meta


# ---------------------------------------------------------------- display helpers
def display_rgb(img, kind):
    """(C,H,W) -> (H,W,3) uint8 for display, with a percentile stretch for dark reflectance images."""
    rgb = (img[:3] if kind == "rgb" else img[[2, 1, 0]]).transpose(1, 2, 0)
    if rgb.max() < 0.6:
        lo, hi = np.percentile(rgb, (2, 98)); rgb = np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def red_overlay(rgb_u8, mask, alpha=0.55):
    out = rgb_u8.astype(np.float32).copy()
    out[mask] = (1 - alpha) * out[mask] + alpha * np.array([255, 0, 0], np.float32)
    return out.astype(np.uint8)


def _resize(img, shape):
    import cv2
    return cv2.resize(img.transpose(1, 2, 0), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR).reshape(shape[0], shape[1], -1).transpose(2, 0, 1)


# ---------------------------------------------------------------- full pipeline
def run_detection(before, after, detector, kind="rgb", threshold=0.5, pixel_size=10.0, alert_limit=10.0,
                  scl=None, cloud_valid=None, transform=None, crs=None, zones=None, third=None,
                  match_hist=False, min_area_ha=0.1, dates=None, cloud_masking=True):
    """before/after: (C,H,W) float 0-1 of band order `kind` (AFTER is resized to BEFORE if sizes differ).
    Returns a dict with mask, overlay, stats, patches, geojson, warnings and alert flags."""
    warnings = []
    h, w = before.shape[1:]
    if after.shape[1:] != (h, w):
        after = _resize(after, (h, w)); warnings.append("AFTER image was resized to the BEFORE image size.")
    if match_hist:
        after = pp.histogram_match(after, before)
    if dates:
        sw = pp.season_warning(*dates)
        if sw: warnings.append(sw)

    # --- valid-pixel (cloud) mask: True = usable
    valid = np.ones((h, w), bool); mask_source = "none"
    if cloud_masking:
        if cloud_valid is not None:
            valid, mask_source = cloud_valid, "uploaded cloud mask"
        elif scl is not None:
            valid, mask_source = pp.valid_from_scl(scl), "Sentinel-2 SCL"
        elif kind == "rgb":
            valid = pp.heuristic_cloud_valid(before, kind) & pp.heuristic_cloud_valid(after, kind)
            mask_source = "brightness/whiteness heuristic (approximate)"
    masked_pct = float(100.0 * (~valid).mean())
    if masked_pct > 20:
        warnings.append(f"{masked_pct:.1f}% of the scene is masked as cloud/shadow/no-data: results are unreliable.")

    prob = detector.predict_proba(before, after, kind)
    raw = (prob > threshold) & valid
    mask = post.clean_mask(raw, pixel_size, min_area_ha=min_area_ha)
    if third is not None:
        t_prob = detector.predict_proba(before, _resize(third, (h, w)) if third.shape[1:] != (h, w) else third, kind)
        mask = post.persistence_filter(mask, (t_prob > threshold) & valid)

    zone_list = zones
    if zones is not None and transform is None:
        warnings.append("Authorised zones ignored: the image has no georeferencing."); zone_list = None
    patches, gj = post.patches_from_mask(mask, pixel_size, transform, crs, zone_list)

    changed = int(mask.sum()); n_valid = int(valid.sum())
    hectares = changed * pixel_size ** 2 / 10000.0
    big = [p for p in patches if p["area_ha"] > alert_limit]
    stats = dict(image_size=f"{w}x{h}", changed_pixels=changed, pct_changed=100.0 * changed / max(n_valid, 1),
                 hectares=hectares, masked_pct=masked_pct, cloud_mask_source=mask_source, patches=len(patches),
                 suspicious_patches=sum(p["status"] == post.STATUS_SUSP for p in patches),
                 authorised_patches=sum(p["status"] == post.STATUS_AUTH for p in patches),
                 pixel_size=pixel_size, threshold=threshold, alert_limit_ha=alert_limit)
    rgb_b, rgb_a = display_rgb(before, kind), display_rgb(after, kind)
    return dict(mask=mask, prob=prob, valid=valid, stats=stats, patches=patches, geojson=gj, warnings=warnings,
                alert_total=hectares > alert_limit, alert_patches=[p["id"] for p in big],
                before_rgb=rgb_b, after_rgb=rgb_a, overlay=red_overlay(rgb_a, mask),
                mask_img=(mask * 255).astype(np.uint8), cloud_img=(~valid * 255).astype(np.uint8))
