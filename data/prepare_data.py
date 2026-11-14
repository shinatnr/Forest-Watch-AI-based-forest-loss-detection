#!/usr/bin/env python3
"""Dataset utilities for Forest Watch.

  python data/prepare_data.py synthetic --out data_synth       # generated GeoTIFF tiles (pipeline testing only)
  python data/prepare_data.py check --data data                # validate layout, class balance, spatial leakage
  python data/prepare_data.py split --src flat_dir --out data  # assign WHOLE regions to train/val/test

Layout:  <data>/{train,val,test}/{A,B,mask}/<region>__<tile>.tif   (A = before, B = after, mask = 1 where cleared)
"""
import argparse, glob, os, shutil, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from forestwatch.data import make_synthetic_pair, region_of, split_by_region, read_mask


def write_tif(path, arr, dtype):
    import rasterio
    from rasterio.transform import from_origin
    arr = arr if arr.ndim == 3 else arr[None]
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[1], width=arr.shape[2], count=arr.shape[0], dtype=dtype,
                       crs="EPSG:32643", transform=from_origin(500000, 1450000, 10, 10)) as dst:
        dst.write(arr.astype(dtype))


def cmd_synthetic(a):
    rng = np.random.default_rng(a.seed)
    plan = {"train": a.n_train, "val": a.n_val, "test": a.n_test}
    for split, n in plan.items():
        for sub in ("A", "B", "mask"):
            os.makedirs(os.path.join(a.out, split, sub), exist_ok=True)
        for i in range(n):
            region = f"{split}R{i // a.tiles_per_region:02d}"        # regions never shared between splits
            name = f"{region}__{i:04d}.tif"
            x, y, m = make_synthetic_pair(a.size, a.kind, rng)
            write_tif(os.path.join(a.out, split, "A", name), np.clip(x, 0, 1) * 10000, "uint16")
            write_tif(os.path.join(a.out, split, "B", name), np.clip(y, 0, 1) * 10000, "uint16")
            write_tif(os.path.join(a.out, split, "mask", name), m, "uint8")
    print(f"wrote synthetic tiles to {a.out} ({a.kind}, {a.size}px). SYNTHETIC: for testing the pipeline only.")


def cmd_check(a):
    ok, regions = True, {}
    for split in ("train", "val", "test"):
        d = os.path.join(a.data, split)
        if not os.path.isdir(d):
            print(f"[{split}] missing"); ok = split == "test" and ok; continue
        fa = sorted(glob.glob(os.path.join(d, "A", "*")))
        miss = [os.path.basename(f) for f in fa if not (os.path.exists(os.path.join(d, "B", os.path.basename(f))) and os.path.exists(os.path.join(d, "mask", os.path.basename(f))))]
        pos = tot = 0
        for f in fa[: a.max_files]:
            m = read_mask(os.path.join(d, "mask", os.path.basename(f)))
            pos += int(m.sum()); tot += m.size
        regions[split] = {region_of(f) for f in fa}
        print(f"[{split}] {len(fa)} files, {len(regions[split])} regions, changed pixels {100 * pos / max(tot, 1):.2f}%"
              + (f", MISSING B/mask for {len(miss)} files" if miss else ""))
        ok &= not miss and len(fa) > 0
    for s1, s2 in (("train", "val"), ("train", "test"), ("val", "test")):
        both = regions.get(s1, set()) & regions.get(s2, set())
        if both:
            ok = False; print(f"LEAKAGE: regions {sorted(both)[:5]} appear in both {s1} and {s2}. Split by region (see `split`).")
    print("OK" if ok else "PROBLEMS FOUND"); sys.exit(0 if ok else 1)


def cmd_split(a):
    files = sorted(glob.glob(os.path.join(a.src, "A", "*")))
    assign = split_by_region(files, tuple(a.ratios), a.seed)
    for f, split in assign.items():
        name = os.path.basename(f)
        for sub in ("A", "B", "mask"):
            os.makedirs(os.path.join(a.out, split, sub), exist_ok=True)
            shutil.copy(os.path.join(a.src, sub, name), os.path.join(a.out, split, sub, name))
    counts = {s: sum(1 for v in assign.values() if v == s) for s in ("train", "val", "test")}
    print("copied by region:", counts)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("synthetic"); q.add_argument("--out", default="data_synth"); q.add_argument("--kind", default="bgrn", choices=["rgb", "bgrn"])
    q.add_argument("--size", type=int, default=128); q.add_argument("--n-train", type=int, default=120); q.add_argument("--n-val", type=int, default=40)
    q.add_argument("--n-test", type=int, default=40); q.add_argument("--tiles-per-region", type=int, default=5); q.add_argument("--seed", type=int, default=0)
    q.set_defaults(f=cmd_synthetic)
    q = sub.add_parser("check"); q.add_argument("--data", default="data"); q.add_argument("--max-files", type=int, default=200); q.set_defaults(f=cmd_check)
    q = sub.add_parser("split"); q.add_argument("--src", required=True); q.add_argument("--out", default="data")
    q.add_argument("--ratios", type=float, nargs=3, default=[0.7, 0.15, 0.15]); q.add_argument("--seed", type=int, default=42); q.set_defaults(f=cmd_split)
    a = p.parse_args(); a.f(a)
