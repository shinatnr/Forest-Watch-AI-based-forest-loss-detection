"""Post-processing: morphology, minimum area, patch table, GeoJSON, authorised-zone overlay, persistence."""
import json
import numpy as np
from scipy import ndimage as ndi

STATUS_AUTH = "authorised"
STATUS_SUSP = "suspicious - needs field verification"
STATUS_UNKNOWN = "forest loss - legality unknown (no authorised-zone map given)"


def clean_mask(mask, pixel_size=10.0, open_px=1, close_px=1, min_area_ha=0.1):
    """Morphological open (removes speckle) + close (fills gaps), then drop patches smaller than min_area_ha."""
    m = mask.astype(bool)
    if open_px > 0:
        m = ndi.binary_opening(m, iterations=open_px)
    if close_px > 0:
        m = ndi.binary_closing(m, iterations=close_px)
    lab, n = ndi.label(m)
    if n:
        sizes = ndi.sum(m, lab, index=np.arange(1, n + 1))
        keep = np.isin(lab, np.where(sizes * pixel_size ** 2 / 10000.0 >= min_area_ha)[0] + 1)
        m = m & keep
    return m


def persistence_filter(mask, mask_third, min_overlap=0.5):
    """Drop patches that have recovered by a third (later) date: keep a patch only if at least
    `min_overlap` of its pixels are still flagged when comparing BEFORE with the third image."""
    lab, n = ndi.label(mask)
    out = np.zeros_like(mask, bool)
    for k in range(1, n + 1):
        comp = lab == k
        if (mask_third & comp).sum() / comp.sum() >= min_overlap:
            out |= comp
    return out


def load_zones(src):
    """Read authorised-felling zones from GeoJSON (EPSG:4326). Returns a list of shapely geometries."""
    from shapely.geometry import shape
    if isinstance(src, bytes) or (isinstance(src, str) and src.lstrip().startswith("{")):
        data = json.loads(src)                      # raw GeoJSON text / bytes
    else:
        data = json.load(open(src))                 # path to a .geojson file
    feats = data["features"] if data.get("type") == "FeatureCollection" else [data]
    geoms = [shape(f["geometry"] if "geometry" in f else f) for f in feats]
    return [g for g in geoms if not g.is_empty]


def patches_from_mask(mask, pixel_size=10.0, transform=None, crs=None, zones=None):
    """Label connected components; return list of patch dicts (+ GeoJSON FeatureCollection).
    Coordinates: EPSG:4326 if the image is georeferenced, otherwise pixel coordinates."""
    from rasterio import features
    from rasterio.transform import Affine
    from shapely.geometry import shape, mapping
    lab, n = ndi.label(mask)
    georef = transform is not None and crs is not None
    tf = transform if georef else Affine.identity()
    polys = {}
    for geom, val in features.shapes(lab.astype(np.int32), mask=lab > 0, transform=tf):
        g = geom
        if georef:
            from rasterio.warp import transform_geom
            g = transform_geom(crs, "EPSG:4326", geom)
        polys.setdefault(int(val), []).append(shape(g))
    patches, feats = [], []
    for k in range(1, n + 1):
        comp = lab == k
        npx = int(comp.sum())
        ys, xs = np.where(comp)
        from shapely.ops import unary_union
        poly = unary_union(polys.get(k, []))
        status = STATUS_UNKNOWN
        if zones is not None and georef:
            status = STATUS_AUTH if any(poly.intersects(z) for z in zones) else STATUS_SUSP
        c = poly.centroid
        p = dict(id=k, pixels=npx, area_ha=round(npx * pixel_size ** 2 / 10000.0, 4), status=status,
                 centroid=[round(c.x, 6), round(c.y, 6)], bbox_px=[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())])
        patches.append(p)
        feats.append(dict(type="Feature", geometry=mapping(poly), properties={k2: v for k2, v in p.items() if k2 != "bbox_px"}))
    gj = dict(type="FeatureCollection", features=feats,
              properties=dict(georeferenced=georef, coordinates="EPSG:4326 (lon, lat)" if georef else "image pixel coordinates"))
    return patches, gj
