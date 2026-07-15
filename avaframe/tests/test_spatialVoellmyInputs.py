"""Tests for module spatialVoellmyInputs"""

import pathlib
import tempfile
import shutil
import numpy as np
import rasterio
import shapefile
import pytest
from avaframe.in3Utils import spatialVoellmyInputs


def _makeSyntheticDEM(tmpDir, suffix=".asc"):
    """Create a small DEM raster for testing.

    Returns path to DEM, its transform, crs, and shape.
    """
    demPath = tmpDir / f"DEM{suffix}"
    data = np.arange(100, dtype=np.float32).reshape(10, 10)
    # North-up: 10x10 grid from (0,0) to (10,10)
    transform = rasterio.transform.from_bounds(0, 0, 10, 10, 10, 10)
    crs = "EPSG:32633"
    driver = "AAIGrid" if suffix == ".asc" else "GTiff"
    with rasterio.open(
        demPath,
        "w",
        driver=driver,
        height=10,
        width=10,
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data, 1)
    return demPath, transform, crs, data.shape


def _makeSyntheticShapefile(shpPath, fieldName, featureCoordsValues):
    """Create a shapefile with a single field and polygon features.

    featureCoordsValues: list of (coords_list, field_value) tuples.
    coords_list is list of (x, y) tuples forming a clockwise ring.
    """
    with shapefile.Writer(shpPath, shapeType=shapefile.POLYGON) as w:
        w.field(fieldName, "F", decimal=6)
        for coords, value in featureCoordsValues:
            w.poly([coords])
            w.record(value)


def _makeSpatialVoellmyShapefile(shpPath, features):
    """Create a *_spatialVoellmy.shp with 'mu' and 'xi' fields.

    features: list of (coords_list, mu_value, xi_value) tuples.
    coords_list is list of (x, y) tuples forming a clockwise ring.
    """
    with shapefile.Writer(shpPath, shapeType=shapefile.POLYGON) as w:
        w.field("mu", "F", decimal=6)
        w.field("xi", "F", decimal=6)
        for coords, muVal, xiVal in features:
            w.poly([coords])
            w.record(muVal, xiVal)


def test_generateMuXiRasters_asc():
    """Test raster generation with .asc DEM and shapefile."""
    tmpDir = pathlib.Path(tempfile.mkdtemp())
    try:
        # Setup: DEM
        demPath, transform, crs, demShape = _makeSyntheticDEM(tmpDir, ".asc")
        inputsDir = tmpDir / "Inputs"
        inputsDir.mkdir()
        shutil.move(str(demPath), str(inputsDir / "DEM.asc"))
        demPath = inputsDir / "DEM.asc"

        # Setup: spatialVoellmy shapefile with two polygons, each with mu and xi
        # Polygon 1: geographic (2..5, 7..9) -> rows 1-3, cols 2-5
        # Polygon 2: geographic (6..9, 2..5) -> rows 5-8, cols 6-9
        polyDir = inputsDir / "POLYGONS"
        polyDir.mkdir()
        shpPath = polyDir / "zones_spatialVoellmy.shp"
        _makeSpatialVoellmyShapefile(
            shpPath,
            [
                ([(2, 7), (5, 7), (5, 9), (2, 9), (2, 7)], 0.300, 3000.0),
                ([(6, 2), (9, 2), (9, 5), (6, 5), (6, 2)], 0.500, 5000.0),
            ],
        )

        # Setup: config
        import configparser

        cfg = configparser.ConfigParser()
        cfg["DEFAULTS"] = {"default_mu": "0.155", "default_xi": "4000."}

        # Run
        spatialVoellmyInputs.generateMuXiRasters(tmpDir, cfg)

        # Assert output files exist
        rastersDir = inputsDir / "RASTERS"
        muRasterPath = rastersDir / "raster_mu.asc"
        xiRasterPath = rastersDir / "raster_xi.asc"
        assert muRasterPath.exists()
        assert xiRasterPath.exists()

        # Assert mu raster values
        with rasterio.open(muRasterPath) as src:
            muData = src.read(1)
            assert src.transform == transform
            assert src.shape == demShape
            # Covered cells (interior of polygons)
            assert muData[2, 3] == pytest.approx(0.300)  # poly 1: row 2, col 3
            assert muData[6, 7] == pytest.approx(0.500)  # poly 2: row 6, col 7
            # Uncovered cell should have default
            assert muData[9, 0] == pytest.approx(0.155)  # row 9, col 0 outside

        # Assert xi raster values
        with rasterio.open(xiRasterPath) as src:
            xiData = src.read(1)
            assert xiData[2, 3] == pytest.approx(3000.0)
            assert xiData[6, 7] == pytest.approx(5000.0)
            assert xiData[9, 0] == pytest.approx(4000.0)

    finally:
        shutil.rmtree(tmpDir)


def test_generateMuXiRasters_tif():
    """Test raster generation with .tif DEM -- output should be .tif."""
    tmpDir = pathlib.Path(tempfile.mkdtemp())
    try:
        demPath, transform, crs, demShape = _makeSyntheticDEM(tmpDir, ".tif")
        inputsDir = tmpDir / "Inputs"
        inputsDir.mkdir()
        shutil.move(str(demPath), str(inputsDir / "DEM.tif"))

        polyDir = inputsDir / "POLYGONS"
        polyDir.mkdir()
        shpPath = polyDir / "zones_spatialVoellmy.shp"
        _makeSpatialVoellmyShapefile(
            shpPath,
            [([(2, 7), (5, 7), (5, 9), (2, 9), (2, 7)], 0.300, 3000.0)],
        )

        import configparser

        cfg = configparser.ConfigParser()
        cfg["DEFAULTS"] = {"default_mu": "0.155", "default_xi": "4000."}

        spatialVoellmyInputs.generateMuXiRasters(tmpDir, cfg)

        rastersDir = inputsDir / "RASTERS"
        muPath = rastersDir / "raster_mu.tif"
        xiPath = rastersDir / "raster_xi.tif"
        assert muPath.exists()
        assert xiPath.exists()
        assert muPath.suffix == ".tif"
        assert xiPath.suffix == ".tif"

    finally:
        shutil.rmtree(tmpDir)


def test_missingMuFieldRaises():
    """Test that missing 'mu' field in shapefile raises clear error."""
    tmpDir = pathlib.Path(tempfile.mkdtemp())
    try:
        demPath, _, _, _ = _makeSyntheticDEM(tmpDir, ".asc")
        inputsDir = tmpDir / "Inputs"
        inputsDir.mkdir()
        shutil.move(str(demPath), str(inputsDir / "DEM.asc"))
        polyDir = inputsDir / "POLYGONS"
        polyDir.mkdir()
        # Shapefile with wrong field name instead of 'mu'
        shpPath = polyDir / "zones_spatialVoellmy.shp"
        with shapefile.Writer(shpPath, shapeType=shapefile.POLYGON) as w:
            w.field("friction_mu", "F", decimal=6)
            w.field("xi", "F", decimal=6)
            w.poly([[(2, 7), (5, 7), (5, 9), (2, 9), (2, 7)]])
            w.record(0.3, 3000.0)

        import configparser

        cfg = configparser.ConfigParser()
        cfg["DEFAULTS"] = {"default_mu": "0.1", "default_xi": "300."}

        with pytest.raises(KeyError, match="mu"):
            spatialVoellmyInputs.generateMuXiRasters(tmpDir, cfg)
    finally:
        shutil.rmtree(tmpDir)


def test_missingDEMRaises():
    """Test that missing DEM raises an error."""
    tmpDir = pathlib.Path(tempfile.mkdtemp())
    try:
        inputsDir = tmpDir / "Inputs"
        inputsDir.mkdir()
        import configparser

        cfg = configparser.ConfigParser()
        cfg["DEFAULTS"] = {"default_mu": "0.1", "default_xi": "300."}
        with pytest.raises(FileNotFoundError):
            spatialVoellmyInputs.generateMuXiRasters(tmpDir, cfg)
    finally:
        shutil.rmtree(tmpDir)
