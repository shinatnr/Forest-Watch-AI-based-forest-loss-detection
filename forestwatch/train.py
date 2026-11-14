"""Training loop for the Siamese U-Net + helpers to save/load weights with a metadata sidecar (.json)."""
import json, os, time
import numpy as np
import torch
from torch.utils.data import DataLoader
from .models import SiameseUNet, BCEDiceLoss
from .metrics import evaluate


def save_meta(path, meta):
    with open(os.path.splitext(path)[0] + ".json", "w") as f:
        json.dump(meta, f, indent=2, default=float)


def load_meta(path):
    p = os.path.splitext(path)[0] + ".json"
    return json.load(open(p)) if os.path.exists(p) else {}


@torch.no_grad()
def _val_probs(model, ds, device, batch=16):
    model.eval(); P, G = [], []
    for xa, xb, m in DataLoader(ds, batch_size=batch):
        P.append(torch.sigmoid(model(xa.to(device), xb.to(device)))[:, 0].cpu().numpy()); G.append(m[:, 0].numpy())
    return np.concatenate(P), np.concatenate(G)


def fit(model, train_ds, val_ds, epochs, out_path, meta, lr=1e-3, batch=16, pos_weight=5.0,
        device=None, log=print):
    """Train, keep the weights with the best validation F1 (change class), save to out_path (+ .json)."""
    device = device or next(model.parameters()).device
    model.to(device)
    loss_fn = BCEDiceLoss(pos_weight).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loader = DataLoader(train_ds, batch_size=batch, shuffle=True, drop_last=len(train_ds) > batch)
    best, history, t0 = -1.0, [], time.time()
    for ep in range(1, epochs + 1):
        model.train(); tot = 0.0
        for xa, xb, m in loader:
            xa, xb, m = xa.to(device), xb.to(device), m.to(device)
            loss = loss_fn(model(xa, xb), m)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(m)
        P, G = _val_probs(model, val_ds, device)
        r = evaluate(P, G, 0.5); r["epoch"], r["loss"] = ep, tot / len(train_ds)
        history.append(r)
        log(f"epoch {ep} loss {r['loss']:.4f} | P {r['precision']:.3f} R {r['recall']:.3f} "
            f"F1 {r['f1']:.3f} IoU {r['iou']:.3f}")
        if r["f1"] > best:
            best = r["f1"]; torch.save(model.state_dict(), out_path)
            save_meta(out_path, dict(meta, best_val_f1=best, history=history, seconds=time.time() - t0))
    model.load_state_dict(torch.load(out_path, map_location=device))
    save_meta(out_path, dict(meta, best_val_f1=best, history=history, seconds=time.time() - t0))
    log(f"best validation F1: {best:.3f} -> saved to {out_path}")
    return history


def build_model(meta, device=None):
    m = SiameseUNet(meta["in_ch"], meta["base"], meta["levels"], meta["mode"])
    return m.to(device) if device else m
