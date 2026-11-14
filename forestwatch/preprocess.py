"""Pre-processing: cloud / shadow masks, histogram matching, season-gap check.
All masks here use the convention valid=True (usable pixel), invalid=False (cloud etc.)."""
from datetime import date
import numpy as np
from scipy import ndimage as ndi

# Sentinel-2 Scene Classification Layer classes to exclude
SCL_INVALID = (0, 1, 3, 8, 9, 10, 11)   # no data, saturated, cloud shadow, cloud med/high, cirrus, snow


def valid_from_scl(scl: np.ndarray) -> np.ndarray:
    return ~np.isin(scl, SCL_INVALID)


def valid_from_qa60(qa60: np.ndarray) -> np.ndarray:
    """QA60 bit 10 = opaque cloud, bit 11 = cirrus."""
    q = qa60.astype(np.int64)
    return ((q & (1 << 10)) == 0) & ((q & (1 << 11)) == 0)


def heuristic_cloud_valid(img: np.ndarray, kind: str) -> np.ndarray:
    """APPROXIMATE cloud mask for plain RGB images: bright, low-saturation (white/grey) pixels.
    Dilated slightly to cover cloud edges. May also flag bare sand, roofs or snow."""
    rgb = img[:3] if kind == "rgb" else img[[2, 1, 0]]
    bright = rgb.min(0) > 0.70
    grey = (rgb.max(0) - rgb.min(0)) < 0.12
    cloud = ndi.binary_dilation(bright & grey, iterations=2)
    return ~cloud


def histogram_match(after: np.ndarray, before: np.ndarray) -> np.ndarray:
    """Per-band histogram matching of AFTER to BEFORE: reduces brightness / seasonal colour shifts
    that are not real change (Isaienkov et al. use this idea)."""
    from skimage.exposure import match_histograms
    out = np.empty_like(after)
    for c in range(after.shape[0]):
        out[c] = match_histograms(after[c], before[c])
    return out.astype(np.float32)


def season_gap_days(d1: date, d2: date) -> int:
    """Gap in days between two dates ignoring the year (position in the seasonal cycle)."""
    # use a fixed non-leap year so day numbers line up (29 Feb -> 28 Feb)
    a, b = (date(2001, d.month, min(d.day, 28) if (d.month == 2) else d.day).timetuple().tm_yday for d in (d1, d2))
    diff = abs(a - b)
    return min(diff, 365 - diff)


def season_warning(d1, d2, limit_days: int = 60):
    """Return a warning string if the two acquisition dates are in different seasons, else None."""
    if d1 is None or d2 is None:
        return None
    gap = season_gap_days(d1, d2)
    if gap > limit_days:
        return (f"The two dates are {gap} days apart in the seasonal cycle (> {limit_days}). "
                "Vegetation colour differs between seasons, so false alarms are likely. "
                "Prefer images from the same time of year.")
    return None
