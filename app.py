"""Forest Watch web interface (Flask).  Run:  python app.py   then open http://127.0.0.1:5000

Upload a BEFORE and AFTER image, pick a detector, set pixel size / threshold / alert limit.
The page shows the red overlay, change mask, area in hectares, alerts and (for GeoTIFFs) a map.
IMPORTANT: the system detects forest LOSS. It cannot prove that clearing was illegal.
"""
import base64, glob, io, json, os, uuid, datetime
import numpy as np
from flask import Flask, render_template, request, send_from_directory, abort
from PIL import Image

from forestwatch import db, alerts
from forestwatch.data import load_scene
from forestwatch.baselines import NDVIDiff
from forestwatch.infer import load_detector, run_detection, is_validated
from forestwatch import postprocess as post

ROOT = os.path.dirname(os.path.abspath(__file__))
ALLOWED_IMG = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
_DETECTORS = {}


def available_detectors():
    """id -> label. NDVI baseline is always available; trained weights are found in best.pt and weights/."""
    out = {"ndvi": "NDVI baseline (rule-based, no training) - NOT validated"}
    paths = [p for p in [os.path.join(ROOT, "best.pt")] + sorted(glob.glob(os.path.join(ROOT, "weights", "*.pt"))
                                                               + glob.glob(os.path.join(ROOT, "weights", "*.joblib"))) if os.path.exists(p)]
    from forestwatch.train import load_meta
    for p in paths:
        m = load_meta(p); kind = m.get("type", "unet"); name = {"unet": f"U-Net ({m.get('mode', '?')})", "rf": "Random Forest", "xgb": "XGBoost"}.get(kind, kind)
        tag = "validated on real data" if is_validated(m) else f"trained on {m.get('trained_on', '?')} - NOT validated"
        out[os.path.relpath(p, ROOT)] = f"{name}: {os.path.basename(p)} ({m.get('kind', '?')} input; {tag})"
    return out


def get_detector(det_id, kind):
    if det_id == "ndvi":
        return NDVIDiff(kind)
    if det_id not in available_detectors():
        raise ValueError("Unknown detector.")
    if det_id not in _DETECTORS:
        _DETECTORS[det_id] = load_detector(os.path.join(ROOT, det_id))
    return _DETECTORS[det_id]


def _b64(arr):
    buf = io.BytesIO(); Image.fromarray(arr).save(buf, "PNG"); return base64.b64encode(buf.getvalue()).decode()


def _num(form, key, default, lo, hi):
    try:
        v = float(form.get(key, default))
    except ValueError:
        raise ValueError(f"'{key}' must be a number.")
    if not lo <= v <= hi:
        raise ValueError(f"'{key}' must be between {lo} and {hi}.")
    return v


def _save_upload(f, folder, label):
    ext = os.path.splitext(f.filename or "")[1].lower()
    if ext not in ALLOWED_IMG:
        raise ValueError(f"{label}: unsupported file type '{ext or 'none'}'. Use PNG, JPG or GeoTIFF.")
    path = os.path.join(folder, label + ext); f.save(path); return path


def _date(s):
    return datetime.date.fromisoformat(s) if s else None


def create_app(config=None):
    app = Flask(__name__)
    app.config.update(MAX_CONTENT_LENGTH=300 * 1024 * 1024, OUTPUT_DIR=os.path.join(ROOT, "instance", "outputs"))
    if config: app.config.update(config)

    @app.get("/")
    def index():
        return render_template("index.html", detectors=available_detectors(), history=db.history(15), email=alerts.email_enabled())

    @app.post("/detect")
    def detect():
        run_id = uuid.uuid4().hex[:10]; out_dir = os.path.join(app.config["OUTPUT_DIR"], run_id); os.makedirs(out_dir, exist_ok=True)
        try:
            fb, fa = request.files.get("before"), request.files.get("after")
            if not fb or not fb.filename or not fa or not fa.filename:
                raise ValueError("Please choose both a BEFORE and an AFTER image.")
            sb, sa = load_scene(_save_upload(fb, out_dir, "before_in")), load_scene(_save_upload(fa, out_dir, "after_in"))
            if sb["kind"] != sa["kind"]:
                raise ValueError(f"The two images have different band layouts ({sb['kind']} vs {sa['kind']}). Use the same kind of image for both.")
            kind = sb["kind"]
            pixel = _num(request.form, "pixel_size", 10, 0.1, 1000); thr = _num(request.form, "threshold", 0.5, 0.1, 0.9)
            limit = _num(request.form, "alert_limit", 10, 0, 1e9); min_area = _num(request.form, "min_area", 0.1, 0, 1e6)
            det_id = request.form.get("detector", "ndvi")
            try:
                det = get_detector(det_id, kind)
            except ValueError as e:
                raise ValueError(str(e))
            if det_id != "ndvi" and getattr(det, "kind", kind) == "bgrn" and kind == "rgb":
                raise ValueError("This detector needs a 4-band Sentinel-2 GeoTIFF (B02,B03,B04,B08); you uploaded an RGB image. Choose the NDVI baseline or an RGB-trained model.")
            cloud_valid = None
            fc = request.files.get("cloud_mask")
            if fc and fc.filename:
                from forestwatch.data import read_mask
                cm = read_mask(_save_upload(fc, out_dir, "cloud_in"))
                if cm.shape != sb["img"].shape[1:]:                  # resize mask to the BEFORE image (nearest neighbour)
                    h, w = sb["img"].shape[1:]
                    cm = np.array(Image.fromarray((cm * 255).astype(np.uint8)).resize((w, h), Image.NEAREST)) > 0
                cloud_valid = ~cm                                   # non-zero in the uploaded mask = cloud
            zones = None
            fz = request.files.get("zones")
            if fz and fz.filename:
                zones = post.load_zones(fz.read())
            dates = (_date(request.form.get("date_before")), _date(request.form.get("date_after")))
            res = run_detection(sb["img"], sa["img"], det, kind, thr, pixel, limit, scl=sb["scl"], cloud_valid=cloud_valid,
                                transform=sb["transform"], crs=sb["crs"], zones=zones, match_hist=bool(request.form.get("match_hist")),
                                min_area_ha=min_area, dates=dates if all(dates) else None)
        except Exception as e:  # noqa  (user-facing error page)
            return render_template("index.html", detectors=available_detectors(), history=db.history(15), email=alerts.email_enabled(), error=str(e)), 400

        meta = getattr(det, "meta", {}) or {}; validated = is_validated(meta)
        res["geojson"]["properties"]["run_id"] = run_id
        for nm, arr in (("before", res["before_rgb"]), ("after", res["after_rgb"]), ("overlay", res["overlay"]), ("mask", res["mask_img"])):
            Image.fromarray(arr).save(os.path.join(out_dir, nm + ".png"))
        json.dump(res["geojson"], open(os.path.join(out_dir, "patches.geojson"), "w"))
        caveat = kind == "rgb" and not sb["georef"]
        alert = res["alert_total"] or bool(res["alert_patches"])
        sent, mail_msg = (alerts.send_email_alert(res, f"run {run_id}") if alert else (False, ""))
        db.add_detection(run_id, request.form.get("aoi", "") or fb.filename, getattr(det, "name", det_id), validated, res["stats"], alert)
        ctx = dict(run_id=run_id, stats=res["stats"], patches=res["patches"], warnings=res["warnings"], alert=alert, alert_total=res["alert_total"],
                   alert_patches=res["alert_patches"], detector=getattr(det, "name", det_id), validated=validated, meta=meta, caveat=caveat,
                   georef=sb["georef"], geojson=json.dumps(res["geojson"]), mail_msg=mail_msg, has_zones=zones is not None,
                   img={k: _b64(res[v]) for k, v in (("before", "before_rgb"), ("after", "after_rgb"), ("overlay", "overlay"), ("mask", "mask_img"))})
        html = render_template("report.html", **ctx, standalone=True)
        open(os.path.join(out_dir, "report.html"), "w", encoding="utf-8").write(html)
        return render_template("result.html", **ctx)

    @app.get("/outputs/<run_id>/<path:name>")
    def outputs(run_id, name):
        if not run_id.isalnum(): abort(404)
        return send_from_directory(os.path.join(app.config["OUTPUT_DIR"], run_id), name, as_attachment=name.endswith((".geojson", ".html")))

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
