#!/usr/bin/env python3
"""Export BEFORE / AFTER Sentinel-2 composites + Hansen forest-loss labels from Google Earth Engine to Google Drive.

NOT TESTED in the build environment (needs a Google Earth Engine account and network access).
Setup once:   pip install earthengine-api ;  earthengine authenticate
Run:          python gee_export.py --before-year 2022 --after-year 2024 --aoi 74.5,13.2,75.0,13.6 --name chikmagalur

What it exports (10 m, EPSG:4326 grid -> projected by GEE, cut into tiles by GEE `fileDimensions`):
  <name>_A.tif    BEFORE composite: median of cloud-masked (SCL) Sentinel-2 L2A, bands B2,B3,B4,B8 (uint16 reflectance*10000)
  <name>_B.tif    AFTER composite, same bands
  <name>_mask.tif 1 where Hansen GFC `lossyear` falls between before-year and after-year AND treecover2000 > 30 %

Known limitations (state these in your report):
  * Hansen labels are 30 m and noisy (resampled to 10 m here); small clearings are missed or over-sized.
  * Hansen `lossyear` ends at the latest GFC release; check the dataset version available for your years.
  * Composites mix dates within a season window; use the same months for BEFORE and AFTER to limit phenology change.
After export, download the Drive files into  flat/{A,B,mask}/<region>__<tile>.tif  and run
  python data/prepare_data.py split --src flat --out data   (splits by region)   then   python data/prepare_data.py check
"""
import argparse


def s2_composite(ee, aoi, year, months):
    start, end = f"{year}-{months[0]:02d}-01", f"{year}-{months[1]:02d}-28"
    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(aoi).filterDate(start, end)
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60)))

    def mask(img):                                          # drop SCL: no data, saturated, shadow, cloud, cirrus, snow
        scl = img.select("SCL")
        bad = scl.eq(0).Or(scl.eq(1)).Or(scl.eq(3)).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10)).Or(scl.eq(11))
        return img.updateMask(bad.Not())
    return col.map(mask).select(["B2", "B3", "B4", "B8"]).median().clip(aoi).unmask(0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--aoi", required=True, help="lon_min,lat_min,lon_max,lat_max (default example: Western Ghats, Karnataka)")
    p.add_argument("--before-year", type=int, required=True); p.add_argument("--after-year", type=int, required=True)
    p.add_argument("--months", type=int, nargs=2, default=[1, 4], help="season window used for BOTH dates (default Jan-Apr, dry season)")
    p.add_argument("--name", default="aoi"); p.add_argument("--drive-folder", default="forestwatch")
    p.add_argument("--min-treecover", type=int, default=30); p.add_argument("--project", default=None, help="Google Cloud project for ee.Initialize")
    a = p.parse_args()
    import ee
    ee.Initialize(project=a.project) if a.project else ee.Initialize()
    lon0, lat0, lon1, lat1 = map(float, a.aoi.split(","))
    aoi = ee.Geometry.Rectangle([lon0, lat0, lon1, lat1])
    A, B = s2_composite(ee, aoi, a.before_year, a.months), s2_composite(ee, aoi, a.after_year, a.months)
    hansen = ee.Image("UMD/hansen/global_forest_change_2023_v1_11")
    loss = hansen.select("lossyear")                        # value = year - 2000
    label = (loss.gte(a.before_year - 2000 + 1).And(loss.lte(a.after_year - 2000)).And(hansen.select("treecover2000").gt(a.min_treecover))).rename("mask")
    for img, suffix, dtype in ((A, "A", "uint16"), (B, "B", "uint16"), (label.unmask(0), "mask", "uint8")):
        img = img.toUint16() if dtype == "uint16" else img.toUint8()
        task = ee.batch.Export.image.toDrive(image=img, description=f"{a.name}_{suffix}", folder=a.drive_folder, fileNamePrefix=f"{a.name}_{suffix}",
                                             region=aoi, scale=10, crs="EPSG:4326", maxPixels=int(1e10), fileFormat="GeoTIFF")
        task.start(); print("started export:", f"{a.name}_{suffix}")
    print("Monitor at https://code.earthengine.google.com/tasks . Rename files to <region>__<tile>.tif when placing into A/B/mask.")


if __name__ == "__main__":
    main()
