"""SQLite history of detection runs (table `detections`)."""
import os, sqlite3, datetime

DB_PATH = os.environ.get("FW_DB", os.path.join(os.path.dirname(os.path.dirname(__file__)), "instance", "forestwatch.db"))
SCHEMA = """CREATE TABLE IF NOT EXISTS detections (
  id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, run_id TEXT, aoi TEXT, detector TEXT, validated INTEGER,
  pixel_size REAL, threshold REAL, changed_pixels INTEGER, hectares REAL, pct_changed REAL, masked_pct REAL,
  patch_count INTEGER, suspicious_patches INTEGER, status TEXT)"""


def _conn(path=None):
    path = path or DB_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    c = sqlite3.connect(path); c.row_factory = sqlite3.Row; c.execute(SCHEMA); return c


def add_detection(run_id, aoi, detector, validated, stats, alert, path=None):
    with _conn(path) as c:
        c.execute("INSERT INTO detections (timestamp,run_id,aoi,detector,validated,pixel_size,threshold,changed_pixels,"
                  "hectares,pct_changed,masked_pct,patch_count,suspicious_patches,status) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                  (datetime.datetime.now().isoformat(timespec="seconds"), run_id, aoi, detector, int(validated),
                   stats["pixel_size"], stats["threshold"], stats["changed_pixels"], stats["hectares"], stats["pct_changed"],
                   stats["masked_pct"], stats["patches"], stats["suspicious_patches"], "ALERT" if alert else "OK"))


def history(limit=20, path=None):
    with _conn(path) as c:
        return [dict(r) for r in c.execute("SELECT * FROM detections ORDER BY id DESC LIMIT ?", (limit,))]
