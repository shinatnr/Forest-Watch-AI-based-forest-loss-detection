#!/usr/bin/env python3
"""Forest Watch command line.
  python deforestation_model.py demo                       # synthetic end-to-end test (CPU, ~1-2 min)
  python deforestation_model.py train --data data --epochs 30
  python deforestation_model.py predict --before b.png --after a.png --weights best.pt --out result
  python deforestation_model.py evaluate --data data --weights best.pt --split test
  python deforestation_model.py compare --data data --epochs 15
"""
import argparse, json, os, sys, time
import numpy as np
import torch
from PIL import Image

from forestwatch.common import set_seed, get_device, model_in_channels
from forestwatch.models import SiameseUNet
from forestwatch.data import SyntheticTiles, DiskTiles, load_scene
from forestwatch import metrics as M
from forestwatch.train import fit, save_meta, load_meta
from forestwatch.baselines import NDVIDiff, PixelClassifier
from forestwatch.infer import (UNetDetector, load_detector, save_pixel_classifier, run_detection, is_validated)

BANNER = "Synthetic data only: scores show the pipeline works, not real-world accuracy."


def collect(detector, ds):
    """Run a detector over every tile of a dataset -> (probs (N,H,W), gts (N,H,W))."""
    P, G = [], []
    for i in range(len(ds)):
        a, b, m = ds.raw(i)
        P.append(detector.predict_proba(a, b)); G.append(m)
    return np.stack(P), np.stack(G)


def save_result(res, out):
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    Image.fromarray(res["overlay"]).save(out + "_overlay.png"); Image.fromarray(res["mask_img"]).save(out + "_mask.png")
    json.dump(res["geojson"], open(out + "_patches.geojson", "w"))
    json.dump(dict(res["stats"], alert=bool(res["alert_total"]), warnings=res["warnings"]), open(out + "_stats.json", "w"), indent=2)


def cmd_demo(args):
    set_seed(args.seed); dev = get_device()
    kind, size = "rgb", 64
    print(f"[demo] device={dev}\n[demo] {BANNER}\n")
    tr = SyntheticTiles(200, size, kind, seed=1, train=True, min_change=0.01, neg_frac=0.25)
    va = SyntheticTiles(60, size, kind, seed=2)
    meta = dict(type="unet", kind=kind, in_ch=model_in_channels(kind), base=16, levels=3, mode="siamese",
                trained_on="synthetic", note=BANNER)
    model = SiameseUNet(meta["in_ch"], meta["base"], meta["levels"], meta["mode"]).to(dev)
    fit(model, tr, va, args.epochs, "best.pt", meta, lr=2e-3, device=dev)
    # classical models on the same synthetic data (so the web app can offer them)
    os.makedirs("weights", exist_ok=True)
    raw = [tr.raw(i) for i in range(len(tr))]
    for algo in ("rf", "xgb"):
        det = PixelClassifier(algo, kind).fit(raw)
        save_pixel_classifier(det, f"weights/{algo}_synthetic.joblib", dict(trained_on="synthetic", kind=kind, note=BANNER))
    # predict on one fresh tile
    te = SyntheticTiles(1, size, kind, seed=99)
    a, b, m = te.raw(0)
    det = UNetDetector("best.pt", dev)
    res = run_detection(a, b, det, kind, threshold=0.5, pixel_size=10, alert_limit=1.0, cloud_masking=False, min_area_ha=0.0)
    s = res["stats"]
    print(f"\nchanged pixels: {s['changed_pixels']} | estimated cleared area: {s['hectares']:.2f} ha")
    if res["alert_total"]:
        print(f"ALERT: cleared area exceeds {s['alert_limit_ha']} ha - flag for field verification")
    save_result(res, "results/demo"); print(f"\n{BANNER}\nsaved: best.pt, weights/*.joblib, results/demo_*")


def _meta_for(kind, args, mode):
    return dict(type="unet", kind=kind, in_ch=model_in_channels(kind), base=args.base, levels=args.levels, mode=mode,
                trained_on=args.trained_on, data=args.data)


def cmd_train(args):
    set_seed(args.seed); dev = get_device()
    tr = DiskTiles(os.path.join(args.data, "train"), tile=args.tile, train=True, min_change=0.01, neg_frac=args.neg_frac, seed=args.seed)
    va = DiskTiles(os.path.join(args.data, "val"), kind=tr.kind, tile=args.tile)
    print(f"device={dev} | train tiles {len(tr)} | val tiles {len(va)} | bands={tr.kind}")
    meta = _meta_for(tr.kind, args, args.mode)
    model = SiameseUNet(meta["in_ch"], meta["base"], meta["levels"], meta["mode"]).to(dev)
    fit(model, tr, va, args.epochs, args.out, meta, lr=args.lr, device=dev)
    # choose threshold on validation (target recall) and report held-out test metrics
    det = UNetDetector(args.out, dev); P, G = collect(det, va)
    thr, p, r = M.pick_threshold_for_recall(P, G, args.target_recall)
    meta = load_meta(args.out); meta["threshold"] = thr
    test_dir = os.path.join(args.data, "test")
    if os.path.isdir(test_dir):
        te = DiskTiles(test_dir, kind=tr.kind, tile=args.tile); Pt, Gt = collect(det, te)
        meta["test_metrics"] = M.evaluate(Pt, Gt, thr); print("test:", {k: round(v, 3) for k, v in meta["test_metrics"].items() if isinstance(v, float)})
    save_meta(args.out, meta)


def cmd_evaluate(args):
    det = load_detector(args.weights)
    ds = DiskTiles(os.path.join(args.data, args.split), kind=det.meta.get("kind"), tile=args.tile)
    P, G = collect(det, ds)
    thr = args.threshold if args.threshold is not None else det.meta.get("threshold", 0.5)
    out = dict(M.evaluate(P, G, thr), by_size=M.size_stratified(P, G, thr, args.pixel_size))
    print(json.dumps(out, indent=2))


def cmd_predict(args):
    dev = get_device()
    sa, sb = load_scene(args.before), load_scene(args.after)
    if args.detector == "ndvi":
        det = NDVIDiff(sa["kind"])
    else:
        det = load_detector(args.weights, dev)
    thr = args.threshold if args.threshold is not None else (det.meta.get("threshold", 0.5) if hasattr(det, "meta") else 0.5)
    res = run_detection(sa["img"], sb["img"], det, sa["kind"], thr, args.pixel_size, args.alert_limit,
                        scl=sa["scl"], transform=sa["transform"], crs=sa["crs"], match_hist=args.match_hist)
    save_result(res, args.out)
    s = res["stats"]
    print(f"detector: {getattr(det, 'name', args.detector)} | validated: {is_validated(getattr(det, 'meta', {}))}")
    print(f"changed pixels: {s['changed_pixels']} | {s['pct_changed']:.2f}% of valid pixels | {s['hectares']:.2f} ha | patches: {s['patches']}")
    for w in res["warnings"]: print("WARNING:", w)
    print("ALERT: cleared area above limit - flag for field verification" if res["alert_total"] else "No alert.")


def cmd_compare(args):
    """Tier 0 NDVI / Tier 1 RF, XGBoost / Tier 2 UNet-diff + Siamese U-Net on the SAME data.
    Thresholds are chosen on the validation set (target recall); metrics are reported on the test set."""
    import csv
    set_seed(args.seed); dev = get_device(); os.makedirs("results", exist_ok=True); os.makedirs("weights", exist_ok=True)
    tr_raw = DiskTiles(os.path.join(args.data, "train"), tile=args.tile, min_change=0.01, neg_frac=args.neg_frac, seed=args.seed)
    kind = tr_raw.kind
    va = DiskTiles(os.path.join(args.data, "val"), kind=kind, tile=args.tile)
    te = DiskTiles(os.path.join(args.data, "test"), kind=kind, tile=args.tile)
    print(f"bands={kind} | train {len(tr_raw)} val {len(va)} test {len(te)} tiles | device={dev}")
    dets = {"NDVI difference": NDVIDiff(kind)}
    train_tiles = [tr_raw.raw(i) for i in range(len(tr_raw))]
    for algo, label in (("rf", "Random Forest"), ("xgb", "XGBoost")):
        print("training", label); d = PixelClassifier(algo, kind).fit(train_tiles)
        save_pixel_classifier(d, f"weights/{algo}.joblib", dict(trained_on=args.trained_on, kind=kind, data=args.data)); dets[label] = d
    for mode, label in (("diff", "UNet-diff"), ("siamese", "Siamese U-Net")):
        print("training", label)
        tr = DiskTiles(os.path.join(args.data, "train"), kind=kind, tile=args.tile, train=True, min_change=0.01, neg_frac=args.neg_frac, seed=args.seed)
        meta = _meta_for(kind, args, mode); path = f"weights/unet_{mode}.pt"
        fit(SiameseUNet(meta["in_ch"], meta["base"], meta["levels"], mode).to(dev), tr, va, args.epochs, path, meta, lr=args.lr, device=dev, log=lambda s: print("   ", s))
        dets[label] = UNetDetector(path, dev)
    rows, curves, sizes = [], {}, {}
    for name, det in dets.items():
        Pv, Gv = collect(det, va); thr, _, _ = M.pick_threshold_for_recall(Pv, Gv, args.target_recall)
        Pt, Gt = collect(det, te); r = M.evaluate(Pt, Gt, thr)
        rows.append(dict(model=name, threshold=round(thr, 4), **{k: round(r[k], 4) for k in ("precision", "recall", "f1", "iou", "dice", "commission_error", "omission_error")}))
        curves[name] = (Pt, Gt); sizes[name] = M.size_stratified(Pt, Gt, thr, args.pixel_size)
        if hasattr(det, "meta") and name in ("UNet-diff", "Siamese U-Net"):
            mp = f"weights/unet_{'diff' if name == 'UNet-diff' else 'siamese'}.pt"; mt = load_meta(mp)
            mt.update(threshold=thr, test_metrics=r); save_meta(mp, mt)
    with open("results/comparison_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open("results/size_stratified.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "size", "n_components", "precision", "recall", "f1", "iou"])
        for n, d in sizes.items():
            for sz, v in d.items(): w.writerow([n, sz, v["n_components"], *(round(v[k], 4) for k in ("precision", "recall", "f1", "iou"))])
    M.save_pr_curves(curves, "results/pr_curve.png")
    json.dump(dict(args={k: v for k, v in vars(args).items() if k != 'f'}, rows=rows, by_size=sizes), open("results/compare_run.json", "w"), indent=2, default=float)
    print("\n" + "model".ljust(16) + "  thr    P      R      F1     IoU    Dice")
    for r in rows: print(r["model"].ljust(16), f"{r['threshold']:.3f}  {r['precision']:.3f}  {r['recall']:.3f}  {r['f1']:.3f}  {r['iou']:.3f}  {r['dice']:.3f}")
    print("\nsaved results/comparison_table.csv, size_stratified.csv, pr_curve.png" +
          ("\nNOTE: --trained-on is 'synthetic': these numbers are NOT real-world accuracy." if args.trained_on != "real" else ""))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    def common(q):
        q.add_argument("--seed", type=int, default=42)
    q = sub.add_parser("demo"); common(q); q.add_argument("--epochs", type=int, default=5); q.set_defaults(f=cmd_demo)
    def trainargs(q):
        common(q); q.add_argument("--data", default="data"); q.add_argument("--epochs", type=int, default=30)
        q.add_argument("--tile", type=int, default=256); q.add_argument("--base", type=int, default=32)
        q.add_argument("--levels", type=int, default=4); q.add_argument("--lr", type=float, default=1e-3)
        q.add_argument("--neg-frac", type=float, default=0.2); q.add_argument("--target-recall", type=float, default=0.8)
        q.add_argument("--pixel-size", type=float, default=10.0)
        q.add_argument("--trained-on", choices=["real", "synthetic"], default="real",
                       help="label stored with the weights; use 'synthetic' for generated test data")
    q = sub.add_parser("train"); trainargs(q); q.add_argument("--mode", choices=["siamese", "diff"], default="siamese")
    q.add_argument("--out", default="best.pt"); q.set_defaults(f=cmd_train)
    q = sub.add_parser("compare"); trainargs(q); q.set_defaults(f=cmd_compare)
    q = sub.add_parser("evaluate"); common(q); q.add_argument("--data", default="data"); q.add_argument("--weights", default="best.pt")
    q.add_argument("--split", default="test"); q.add_argument("--tile", type=int, default=256); q.add_argument("--threshold", type=float)
    q.add_argument("--pixel-size", type=float, default=10.0); q.set_defaults(f=cmd_evaluate)
    q = sub.add_parser("predict"); common(q); q.add_argument("--before", required=True); q.add_argument("--after", required=True)
    q.add_argument("--weights", default="best.pt"); q.add_argument("--detector", choices=["model", "ndvi"], default="model")
    q.add_argument("--out", default="result"); q.add_argument("--threshold", type=float); q.add_argument("--pixel-size", type=float, default=10.0)
    q.add_argument("--alert-limit", type=float, default=10.0); q.add_argument("--match-hist", action="store_true"); q.set_defaults(f=cmd_predict)
    args = p.parse_args(); args.f(args)


if __name__ == "__main__":
    main()
