# Forest Watch: AI-based forest-loss detection from satellite image pairs

Compares a **BEFORE** and an **AFTER** image of the same area and highlights where forest was cleared: red overlay,
change mask, area in hectares, patch list (GeoJSON) and an alert. A Flask web page makes it usable without coding.

> **What this system does and does not do.** It detects forest *loss*. It **cannot prove that clearing was illegal**.
> With an optional authorised-felling map, loss outside the zones is labelled *suspicious - needs field verification*.
> Every alert needs ground verification.

## Pipeline

`input -> pre-process (resize, cloud/shadow mask, optional histogram match, NDVI) -> detector -> post-process (threshold,
open/close, minimum area, patches, authorised zones) -> overlay, mask, area, alert, history`

Three detector tiers share one interface `predict_proba(a, b, kind)`, so they are compared on identical data:

| Tier | Detector | File |
|---|---|---|
| 0 | NDVI difference (rule-based) | `forestwatch/baselines.py` |
| 1 | Random Forest / XGBoost on per-pixel features (bands, NDVI, NDWI, EVI of both dates + differences) | `forestwatch/baselines.py` |
| 2 | **Siamese U-Net** (shared encoder, feature difference `|f_before - f_after|` into a U-Net decoder) and the **UNet-diff** ablation | `forestwatch/models.py` |

Area: `hectares = changed_pixels x pixel_size^2 / 10000`.

## Setup (macOS / Linux, CPU is enough; a GPU is used automatically if present)

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Quick start

```bash
python deforestation_model.py demo     # ~30 s on CPU: trains on SYNTHETIC tiles, saves best.pt, predicts one tile
python app.py                          # then open http://127.0.0.1:5000
pytest                                 # 30 tests
```

The demo prints per-epoch precision / recall / F1 / IoU and the banner *"Synthetic data only: scores show the pipeline
works, not real-world accuracy."* It also saves small Random Forest / XGBoost models so the web page can offer them.

## Commands

```bash
python deforestation_model.py train    --data data --epochs 30                 # Siamese U-Net on real tiles -> best.pt
python deforestation_model.py train    --data data --mode diff --out weights/unet_diff.pt
python deforestation_model.py predict  --before b.tif --after a.tif --weights best.pt --out result --pixel-size 10 --alert-limit 10
python deforestation_model.py evaluate --data data --weights best.pt --split test        # change-class metrics + by clearing size
python deforestation_model.py compare  --data data --epochs 25                 # NDVI / RF / XGBoost / UNet-diff / Siamese U-Net
```

`compare` picks each model's threshold on the **validation** set to reach a target recall (default 0.8), reports **test**
metrics, and writes `results/comparison_table.csv`, `results/size_stratified.csv`, `results/pr_curve.png`.
Metrics are for the **change class only** (precision, recall, F1, IoU, Dice, commission and omission error). Overall accuracy
is not reported because the unchanged class saturates it. Size-stratified scores count false alarms only near each
ground-truth clearing, so they are not comparable with the overall table.

## Data preparation

Expected layout: `data/{train,val,test}/{A,B,mask}/<region>__<tile>.tif` (A = before, B = after, mask = 1 where cleared;
GeoTIFF bands B02, B03, B04, B08 as uint16 reflectance x 10000). **Split by region, never by pixel**: all tiles of a region
belong to one split (the text before `__` is the region id).

1. **Export from Google Earth Engine** (needs an account; `earthengine authenticate`; **not tested here**):
   `python gee_export.py --aoi 74.5,13.2,75.0,13.6 --before-year 2022 --after-year 2024 --name chikmagalur`.
   It exports SCL cloud-masked Sentinel-2 L2A median composites and labels from Hansen Global Forest Change (`lossyear`,
   `treecover2000 > 30`). *Limits:* Hansen labels are 30 m and noisy, so small clearings are missed or over-sized.
2. Put the files in `flat/{A,B,mask}/` named `<region>__<tile>.tif`, then
   `python data/prepare_data.py split --src flat --out data`.
3. `python data/prepare_data.py check --data data` (class balance, missing files, **region leakage between splits**).
4. Other datasets (for example the Isaienkov et al. Ukrainian clear-cut data, OSCD-style folders) work if converted to this layout.
5. To test the pipeline without real data: `python data/prepare_data.py synthetic --out data_synth`
   (`compare --data data_synth --tile 128 --pixel-size 30 --trained-on synthetic`).

## Web interface

Upload BEFORE / AFTER (PNG, JPG or GeoTIFF; a 5th GeoTIFF band is read as the Sentinel-2 SCL cloud layer), choose a
detector, pixel size, threshold (0.1 - 0.9), alert limit (ha). Optional: acquisition dates (warns if more than 60 days
apart in season), a cloud-mask image, authorised-felling zones (GeoJSON, EPSG:4326, needs a georeferenced GeoTIFF),
histogram matching. Downloads: overlay, mask, patches GeoJSON, one-page HTML report (print to PDF). Runs are stored in
SQLite (`instance/forestwatch.db`, table `detections`) and shown as a history table.
Detectors trained on synthetic data or rule-based are always shown as **not validated**.
For RGB screenshots without georeferencing the hectare value is only nominal (the page says so); prefer the percentage.

E-mail alerts are **off by default**: set `FW_EMAIL_ENABLED=1`, `FW_SMTP_HOST`, `FW_SMTP_PORT`, `FW_SMTP_USER`,
`FW_SMTP_PASS`, `FW_ALERT_FROM`, `FW_ALERT_TO`.

## Cloud and seasonal handling

Sentinel-2 SCL classes (no data, saturated, shadow, cloud, cirrus, snow) are masked and **excluded from the area count**;
the masked percentage is shown. For plain RGB images an approximate brightness/whiteness heuristic is used (it can also
flag sand, roofs or snow). Seasonal colour shifts are reduced by optional histogram matching and the date-gap warning.
Radar (Sentinel-1) fusion is not implemented (future work). A third-date regrowth filter exists as the `third` option of `run_detection()` (library only, not in the web page).

## Results so far (honest status)

No accuracy on real satellite data has been measured yet. The only numbers available come from the **synthetic** generator
(`python deforestation_model.py compare` on `data_synth`, 25 epochs, base 16, 3 levels, run in this build):

| Detector | Precision | Recall | F1 | IoU |
|---|---|---|---|---|
| NDVI difference | 0.553 | 0.969 | 0.704 | 0.543 |
| Random Forest | 0.968 | 0.802 | 0.877 | 0.781 |
| XGBoost | 0.971 | 0.802 | 0.878 | 0.783 |
| UNet-diff | 1.000 | 0.840 | 0.913 | 0.840 |
| Siamese U-Net | 0.995 | 0.810 | 0.893 | 0.806 |

These show that the pipeline and the comparison work. They are **not** real-world accuracy and do not rank the methods.

## Limitations

- Trained weights shipped here come from synthetic data; retrain on real labelled tiles before real use.
- Detects forest loss, not legality; fire, pests, landslides, harvesting and seasonal change can look like clearing.
- Hansen labels are coarse; clouds, shadows and thin haze are only partly handled; no field validation.
- Small clearings (a few pixels at 10 m) are the hardest case for every detector.
- `gee_export.py` and the Leaflet map (needs internet for tiles) were not exercised in the build environment.

## Project layout

```
deforestation_model.py   CLI: demo | train | predict | evaluate | compare
app.py  templates/       Flask site
gee_export.py            Earth Engine export (untested)
data/prepare_data.py     synthetic | check | split
forestwatch/             models, baselines, data, preprocess, postprocess, metrics, train, infer, alerts, db
tests/                   pytest (30 tests)
results/ weights/        outputs of demo/compare (synthetic)
```

## References

[1] R. Caye Daudt, B. Le Saux, A. Boulch, "Fully convolutional Siamese networks for change detection," IEEE ICIP, 2018, pp. 4063-4067.
[2] O. Ronneberger, P. Fischer, T. Brox, "U-Net: Convolutional networks for biomedical image segmentation," MICCAI, 2015, pp. 234-241.
