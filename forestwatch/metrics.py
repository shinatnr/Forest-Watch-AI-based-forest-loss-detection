"""Change-class metrics (precision, RECALL, F1, IoU, Dice), size-stratified scores, threshold choice.
Overall accuracy is deliberately NOT reported: it is saturated by the 'no change' class."""
import numpy as np
from scipy import ndimage as ndi


def confusion(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    return dict(tp=int((pred & gt).sum()), fp=int((pred & ~gt).sum()), fn=int((~pred & gt).sum()))


def scores(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return dict(precision=p, recall=r, f1=f1, iou=iou, dice=f1,          # Dice == F1 for binary masks
                commission_error=(1 - p) if tp + fp else 0.0,            # false alarms among detections
                omission_error=(1 - r) if tp + fn else 0.0)              # missed clearings


def evaluate(probs, gts, threshold=0.5):
    """probs, gts: arrays (N,H,W). Pixel-level metrics of the change class."""
    c = confusion(np.asarray(probs) > threshold, np.asarray(gts) > 0.5)
    out = scores(**c); out.update(c); out["threshold"] = float(threshold)
    return out


def size_stratified(probs, gts, threshold=0.5, pixel_size=10.0, bins_ha=(1.0, 5.0)):
    """Metrics per clearing size (small < bins[0] ha <= medium < bins[1] ha <= large).
    For each ground-truth component, TP/FN are counted on its pixels; FP are predicted pixels
    within a 3-pixel margin of those components that are not ground truth anywhere."""
    names = ["small", "medium", "large"]
    acc = {n: dict(tp=0, fp=0, fn=0, n=0) for n in names}
    px_ha = pixel_size ** 2 / 10000.0
    for p, g in zip(np.asarray(probs), np.asarray(gts)):
        pred, gt = p > threshold, g > 0.5
        lab, n = ndi.label(gt)
        if n == 0:
            continue
        areas = ndi.sum(gt, lab, index=np.arange(1, n + 1)) * px_ha
        for k, a in enumerate(areas, start=1):
            name = names[0] if a < bins_ha[0] else names[1] if a < bins_ha[1] else names[2]
            comp = lab == k
            region = ndi.binary_dilation(comp, iterations=3)
            acc[name]["tp"] += int((pred & comp).sum()); acc[name]["fn"] += int((~pred & comp).sum())
            acc[name]["fp"] += int((pred & region & ~gt).sum()); acc[name]["n"] += 1
    return {n: dict(scores(v["tp"], v["fp"], v["fn"]), n_components=v["n"]) for n, v in acc.items()}


def _subsample(probs, gts, max_px=2_000_000, seed=0):
    p, g = np.asarray(probs).ravel(), (np.asarray(gts).ravel() > 0.5)
    if p.size > max_px:
        idx = np.random.default_rng(seed).choice(p.size, max_px, replace=False)
        p, g = p[idx], g[idx]
    return p, g


def pick_threshold_for_recall(probs, gts, target_recall=0.8):
    """Highest threshold whose recall on the VALIDATION set is >= target (an undetected clearing
    costs more than a false alarm). Returns (threshold, precision_at_threshold, recall_at_threshold)."""
    from sklearn.metrics import precision_recall_curve
    p, g = _subsample(probs, gts)
    if g.sum() == 0:
        return 0.5, 0.0, 0.0
    prec, rec, thr = precision_recall_curve(g, p)
    ok = np.where(rec[:-1] >= target_recall)[0]
    i = ok[-1] if len(ok) else 0
    # sklearn thresholds mean score >= t, but evaluate() uses score > t: step down one float so ties at t count
    return float(np.nextafter(thr[i], -np.inf)), float(prec[i]), float(rec[i])


def save_pr_curves(curves, path):
    """curves: {name: (probs, gts)} -> precision-recall plot saved to `path`."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for name, (probs, gts) in curves.items():
        p, g = _subsample(probs, gts)
        if g.sum() == 0:
            continue
        pr, rc, _ = precision_recall_curve(g, p)
        ax.plot(rc, pr, label=name)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision-recall (change class)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
