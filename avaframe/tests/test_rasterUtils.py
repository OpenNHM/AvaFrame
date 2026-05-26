"""Tests for module rasterUtils"""

import pathlib
import numpy as np
import rasterio

import avaframe.in2Trans.rasterUtils as rasterUtils


def test_convertRasterToNcFile(tmp_path):
    inputDir = pathlib.Path(tmp_path, "testDir")
    inputDir.mkdir(exist_ok=True)

    # save test rasters as tif
    inputRasterNames = ["testRaster_t0.00", "testRaster_t0.01", "testRaster_t20.00"]

    testRaster = np.zeros((10, 10))
    cellsize = 10
    nrows, ncols = testRaster.shape

    header = {
        "cellsize": cellsize,
        "nrows": nrows,
        "ncols": ncols,
        "xllcenter": 0,
        "yllcenter": 0,
        "nodata_value": -9999,
        "driver": "GTiff",
        "crs": "EPSG:4326",
    }
    # convert lower-left center to upper-left corner
    x_ul = header["xllcenter"] - cellsize / 2
    y_ul = header["yllcenter"] + nrows * cellsize - cellsize / 2

    transform = rasterio.transform.from_origin(x_ul, y_ul, cellsize, cellsize)
    header["transform"] = transform
    for name in inputRasterNames:
        # write flipped raster, the read raster function does also flip the raster
        rasterUtils.writeResultToRaster(header, testRaster, inputDir / name, useCompression=False, flip=True)
    del testRaster

    inFileExt = "tif"
    inFileSuf = "test"
    outputDir = inputDir / "ncFiles"

    rasterUtils.convertRasterToNcFile(inputDir, inFileExt, inFileSuf, outputDir)

    ncFileNames = ["testRaster_0000", "testRaster_0001", "testRaster_0002"]
    for ncFileName in ncFileNames:
        ncPath = outputDir / f"{ncFileName}.nc"
        assert ncPath.is_file()
