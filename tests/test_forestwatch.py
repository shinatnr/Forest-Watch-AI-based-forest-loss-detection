"""Forest Watch test-suite. IDs in comments map to requirement IDs (R1..R9) on the presentation."""
import os, sys, json, datetime
import numpy as np, pytest, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from forestwatch.common import convert_kind, model_input, set_seed
from forestwatch.models import SiameseUNet
from forestwatch.data import SyntheticTiles, make_synthetic_pair
from forestwatch import metrics as M, postprocess as post, preprocess as pp, db
from forestwatch.baselines import NDVIDiff
from forestwatch.infer import run_detection, UNetDetector
from forestwatch.train import fit


def _pair(kind="rgb", size=64, seed=3):
    rng = np.random.default_rng(seed)
    while True:
        a, b, m = make_synthetic_pair(size, kind, rng)
        if m.sum() > 50: return a, b, m


# ---- R3 area estimate
def test_T01_area_formula_matches_report():                       # R3
    assert post.patches_from_mask(np.pad(np.ones((10, 10), bool), 5), pixel_size=10)[0][0]["area_ha"] == 1.0
    assert 15070 * 10 ** 2 / 10000 == pytest.approx(150.7)       # figure quoted in the module report

def test_T02_alert_boundary_equal_limit_no_alert():               # R4 (boundary)
    a, b, _ = _pair()
    r = run_detection(a, b, NDVIDiff("rgb"), "rgb", pixel_size=10, alert_limit=1e9, cloud_masking=False)
    assert not r["alert_total"]
    hect = r["stats"]["hectares"]
    r2 = run_detection(a, b, NDVIDiff("rgb"), "rgb", pixel_size=10, alert_limit=hect, cloud_masking=False)
    assert not r2["alert_total"]                                  # equal to limit -> no alert (strict >)
    r3 = run_detection(a, b, NDVIDiff("rgb"), "rgb", pixel_size=10, alert_limit=hect - 0.01, cloud_masking=False)
    assert r3["alert_total"]

# ---- R2 model
@pytest.mark.parametrize("mode", ["siamese", "diff"])
def test_T03_model_output_shape(mode):                            # R2
    m = SiameseUNet(5, 8, 3, mode).eval()
    out = m(torch.rand(2, 5, 64, 64), torch.rand(2, 5, 64, 64))
    assert out.shape == (2, 1, 64, 64)

def test_T04_identical_dates_give_content_independent_output():   # R2 shared encoder / feature difference
    set_seed(0); m = SiameseUNet(4, 8, 3, "siamese").eval()
    x1, x2 = torch.rand(1, 4, 32, 32), torch.rand(1, 4, 32, 32)
    assert torch.allclose(m(x1, x1), m(x2, x2), atol=1e-5)       # no change -> same score whatever the scene looks like

def test_T05_unet_train_save_load_predict(tmp_path):              # R2
    set_seed(0)
    tr, va = SyntheticTiles(24, 64, "rgb", 1, True), SyntheticTiles(8, 64, "rgb", 2)
    meta = dict(type="unet", kind="rgb", in_ch=4, base=8, levels=3, mode="siamese", trained_on="synthetic")
    p = str(tmp_path / "w.pt")
    fit(SiameseUNet(4, 8, 3).to("cpu"), tr, va, 1, p, meta, device=torch.device("cpu"), log=lambda s: None)
    det = UNetDetector(p, torch.device("cpu")); a, b, _ = va.raw(0)
    prob = det.predict_proba(a, b)
    assert prob.shape == (64, 64) and 0 <= prob.min() and prob.max() <= 1

def test_T06_tiled_inference_large_scene(tmp_path):               # R2 / R1 large images
    set_seed(0); meta = dict(type="unet", kind="rgb", in_ch=4, base=8, levels=3, mode="siamese", trained_on="synthetic")
    p = str(tmp_path / "w.pt"); torch.save(SiameseUNet(4, 8, 3).state_dict(), p)
    from forestwatch.train import save_meta; save_meta(p, meta)
    det = UNetDetector(p, torch.device("cpu"), tile=128, overlap=16)
    a = np.random.rand(3, 300, 381).astype(np.float32)           # odd size, larger than one tile
    assert det.predict_proba(a, a).shape == (300, 381)

# ---- R1 input handling
def test_T07_after_image_resized_to_before():                     # R1
    a, b, _ = _pair(size=64)
    r = run_detection(a, np.random.rand(3, 50, 70).astype(np.float32), NDVIDiff("rgb"), "rgb", cloud_masking=False)
    assert r["mask"].shape == (64, 64) and any("resized" in w for w in r["warnings"])

def test_T08_negative_rgb_cannot_become_bgrn():                   # R1 (negative)
    with pytest.raises(ValueError):
        convert_kind(np.zeros((3, 8, 8), np.float32), "rgb", "bgrn")

# ---- R6 clouds / seasons
def test_T09_cloudy_pixels_excluded_from_area():                  # R6
    a, b, m = _pair("bgrn")
    scl = np.full(m.shape, 4, np.int32)
    base = run_detection(a, b, NDVIDiff("bgrn"), "bgrn", scl=scl, min_area_ha=0)["stats"]["changed_pixels"]
    scl[:, :] = 9                                                 # whole scene = high-probability cloud
    r = run_detection(a, b, NDVIDiff("bgrn"), "bgrn", scl=scl, min_area_ha=0)
    assert base > 0 and r["stats"]["changed_pixels"] == 0 and r["stats"]["masked_pct"] == 100

def test_T10_season_gap_warning():                                # R6 (boundary 60 days)
    d = datetime.date
    assert pp.season_warning(d(2020, 1, 15), d(2021, 7, 15)) is not None
    assert pp.season_warning(d(2020, 6, 1), d(2021, 7, 15)) is None
    assert pp.season_gap_days(d(2020, 12, 20), d(2021, 1, 10)) == 21    # wraps around the year end

# ---- post-processing
def test_T11_min_area_boundary():                                 # R3 / false-alarm filter
    m = np.zeros((40, 40), bool); m[2:12, 2:12] = True; m[20:29, 20:29] = True     # 10x10 (=1.00 ha) and 9x9 (=0.81 ha)
    out = post.clean_mask(m, 10, open_px=0, close_px=0, min_area_ha=1.0)
    assert out[5, 5] and not out[24, 24]

def test_T12_persistence_filter_drops_regrown_patch():
    m = np.zeros((30, 30), bool); m[2:8, 2:8] = True; m[15:25, 15:25] = True
    third = np.zeros_like(m); third[15:25, 15:25] = True          # first patch recovered by 3rd date
    out = post.persistence_filter(m, third)
    assert out[20, 20] and not out[4, 4]

# ---- R9 authorised vs suspicious
def test_T13_authorised_zone_labelling():                         # R9
    from rasterio.transform import from_origin
    from rasterio.crs import CRS
    from shapely.geometry import box, mapping
    tf = from_origin(75.0, 13.0, 0.0001, 0.0001); crs = CRS.from_epsg(4326)
    m = np.zeros((100, 100), bool); m[10:30, 10:30] = True; m[60:80, 60:80] = True
    zone = box(75.0005, 12.9980, 75.0035, 12.9995)                # covers only the first patch
    patches, gj = post.patches_from_mask(m, 10, tf, crs, [zone])
    st = sorted(p["status"] for p in patches)
    assert st == sorted([post.STATUS_AUTH, post.STATUS_SUSP])

def test_T14_zones_ignored_without_georeferencing():              # R9 (negative)
    from shapely.geometry import box
    a, b, _ = _pair()
    r = run_detection(a, b, NDVIDiff("rgb"), "rgb", zones=[box(0, 0, 1, 1)], cloud_masking=False)
    assert any("ignored" in w for w in r["warnings"])

# ---- R7 metrics
def test_T15_metrics_known_values():                              # R7
    pred = np.zeros((1, 10, 10)); gt = np.zeros((1, 10, 10)); pred[0, 0, :10] = 1; gt[0, 0, :8] = 1; gt[0, 1, :2] = 1
    r = M.evaluate(pred, gt, 0.5)                                 # tp=8 fp=2 fn=2
    assert (r["tp"], r["fp"], r["fn"]) == (8, 2, 2)
    assert r["precision"] == pytest.approx(0.8) and r["recall"] == pytest.approx(0.8) and r["iou"] == pytest.approx(8 / 12)

def test_T16_threshold_hits_target_recall():                      # R7
    rng = np.random.default_rng(0); g = (rng.random((20, 32, 32)) < 0.05)
    p = np.clip(g * 0.5 + rng.random(g.shape) * 0.5, 0, 1)
    thr, prec, rec = M.pick_threshold_for_recall(p, g, 0.8)
    assert rec >= 0.8

# ---- history / E-mail
def test_T17_history_roundtrip(tmp_path):                         # R4 monitoring
    a, b, _ = _pair(); r = run_detection(a, b, NDVIDiff("rgb"), "rgb", cloud_masking=False)
    p = str(tmp_path / "h.db"); db.add_detection("r1", "aoi", "ndvi_diff", False, r["stats"], r["alert_total"], p)
    h = db.history(5, p); assert len(h) == 1 and h[0]["detector"] == "ndvi_diff"

def test_T18_email_disabled_by_default(monkeypatch):              # R9 / safety
    from forestwatch import alerts
    monkeypatch.delenv("FW_EMAIL_ENABLED", raising=False)
    a, b, _ = _pair(); r = run_detection(a, b, NDVIDiff("rgb"), "rgb", cloud_masking=False)
    sent, msg = alerts.send_email_alert(r); assert not sent and "disabled" in msg
    assert "cannot prove" in alerts.alert_message(r)

def test_T19_threshold_with_saturated_scores():                   # R7 regression: scores tied at 1.0 must still be detected
    g = np.zeros((4, 16, 16), bool); g[:, :4, :4] = True
    p = np.where(g, 1.0, 0.0)                                     # saturated detector (like clipped NDVI score)
    thr, _, rec = M.pick_threshold_for_recall(p, g, 0.8)
    assert M.evaluate(p, g, thr)["recall"] == 1.0

def test_T20_uint8_01_mask_is_not_rescaled(tmp_path):             # data regression: 0/1 uint8 masks must stay binary
    import sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"))
    from prepare_data import write_tif
    from forestwatch.data import read_mask
    m = np.zeros((16, 16), np.uint8); m[:4, :4] = 1; p = str(tmp_path / "m.tif"); write_tif(p, m, "uint8")
    assert read_mask(p).sum() == 16

def test_T21_evi_finite_when_denominator_near_zero():            # regression: EVI must never produce inf/nan
    from forestwatch.common import evi
    img = np.zeros((4, 2, 2), np.float32); img[0] = 0.9; img[2] = 0.0; img[3] = 0.0      # B=0.9 -> den = 1 - 6.75 + ... near zero cases
    img[0] = (1 + 0.0) / 7.5                                       # makes n + 6r - 7.5b + 1 == 0 exactly
    assert np.isfinite(evi(img, "bgrn")).all()
