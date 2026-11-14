"""Flask app tests (R5 web interface, R9 honesty banners)."""
import io, os, sys
import numpy as np, pytest
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"))
from forestwatch.data import make_synthetic_pair
from forestwatch import db
import app as appmod


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "h.db"))
    return appmod.create_app({"TESTING": True, "OUTPUT_DIR": str(tmp_path / "out")}).test_client()


def _png(arr):
    b = io.BytesIO(); Image.fromarray((arr.transpose(1, 2, 0) * 255).astype(np.uint8)).save(b, "PNG"); b.seek(0); return b


def _pair_files(kind="rgb"):
    rng = np.random.default_rng(5)
    while True:
        a, b, m = make_synthetic_pair(96, kind, rng)
        if m.sum() > 100: return a, b


def test_A01_home_page_lists_ndvi_detector(client):
    r = client.get("/"); assert r.status_code == 200 and b"NDVI baseline" in r.data and b"Recent detections" in r.data


def test_A02_upload_pair_returns_result_alert_and_downloads(client):            # R5 R3 R4
    a, b = _pair_files()
    r = client.post("/detect", data={"before": (_png(a), "b.png"), "after": (_png(b), "a.png"), "detector": "ndvi",
                                     "pixel_size": "10", "threshold": "0.5", "alert_limit": "0.5"}, content_type="multipart/form-data")
    assert r.status_code == 200
    assert b"ALERT" in r.data and b"cleared area" in r.data and b"NOT validated" in r.data or b"Not validated" in r.data
    assert b"cannot prove" in r.data
    run = r.data.decode().split("/outputs/")[1].split("/")[0]
    for f in ("overlay.png", "mask.png", "patches.geojson", "report.html"):
        assert client.get(f"/outputs/{run}/{f}").status_code == 200
    assert b"Forest Watch report" in client.get(f"/outputs/{run}/report.html").data
    assert b"ndvi_diff" in client.get("/").data                                   # appears in history table


def test_A03_missing_file_is_a_clear_error(client):                              # negative
    r = client.post("/detect", data={"detector": "ndvi"}, content_type="multipart/form-data")
    assert r.status_code == 400 and b"choose both" in r.data


def test_A04_bad_extension_rejected(client):                                      # negative
    a, b = _pair_files()
    r = client.post("/detect", data={"before": (io.BytesIO(b"x"), "b.exe"), "after": (_png(b), "a.png")}, content_type="multipart/form-data")
    assert r.status_code == 400 and b"unsupported file type" in r.data


@pytest.mark.parametrize("field,val", [("threshold", "0.95"), ("threshold", "abc"), ("pixel_size", "-3")])
def test_A05_parameter_validation(client, field, val):                           # boundary / negative
    a, b = _pair_files()
    data = {"before": (_png(a), "b.png"), "after": (_png(b), "a.png"), "detector": "ndvi", field: val}
    r = client.post("/detect", data=data, content_type="multipart/form-data")
    assert r.status_code == 400 and field.encode() in r.data


def test_A06_geotiff_with_authorised_zones(client):                               # R9
    from prepare_data import write_tif
    a, b = _pair_files("bgrn")
    ba, bb = io.BytesIO(), io.BytesIO()
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        pa, pb = os.path.join(d, "a.tif"), os.path.join(d, "b.tif")
        write_tif(pa, a * 10000, "uint16"); write_tif(pb, b * 10000, "uint16")
        zone = b'{"type":"FeatureCollection","features":[{"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":[[[74.9,13.0],[75.1,13.0],[75.1,13.2],[74.9,13.2],[74.9,13.0]]]}}]}'
        r = client.post("/detect", data={"before": (open(pa, "rb"), "a.tif"), "after": (open(pb, "rb"), "b.tif"), "detector": "ndvi",
                                         "pixel_size": "10", "alert_limit": "100", "zones": (io.BytesIO(zone), "z.geojson")}, content_type="multipart/form-data")
    assert r.status_code == 200 and b"leaflet" in r.data.lower() and b"authorised" in r.data
