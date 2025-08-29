"""Tests for module preparePathGeneral"""

import numpy as np
import pytest
import configparser

import avaframe.ana5Utils.preparePathGeneral as prepPathGeneral


# ---------------------------------------------------------------------------
# updateSZProfile
# ---------------------------------------------------------------------------

def test_updateSZProfile_flatDEM():
    """ test that z is read as constant from a flat DEM and s is the cumulative 2D distance """
    cellsize = 1.0
    nrows, ncols = 100, 100
    zRaster = np.zeros((nrows, ncols))
    header = {"cellsize": cellsize, "nrows": nrows, "ncols": ncols, "xllcenter": 0.0, "yllcenter": 0.0}
    dem = {"rasterData": zRaster, "header": header}

    profile = {
        "x": np.array([10.0, 13.0, 13.0, 20.0]),
        "y": np.array([10.0, 10.0, 14.0, 14.0]),
    }
    expectedS = np.array([0.0, 3.0, 7.0, 14.0])

    result = prepPathGeneral.updateSZProfile(profile, dem)

    assert "z" in result
    assert "s" in result
    np.testing.assert_allclose(result["z"], [0.0, 0.0, 0.0, 0.0])
    # segment lengths: 3, 4, 7 (3-4-5 triangle then straight run) -> cumulative distance from 0
    np.testing.assert_allclose(result["s"], expectedS, atol=1e-6)
    # x, y are untouched
    np.testing.assert_allclose(result["x"], profile["x"])
    np.testing.assert_allclose(result["y"], profile["y"])


def test_updateSZProfile_slopedDEM():
    """ test that z values are correctly read (bilinearly interpolated) from a DEM with a linear x-gradient """
    cellsize = 1.0
    nrows, ncols = 100, 100
    # elevation ramp: z = x-coordinate everywhere (constant along y)
    zRaster = np.tile(np.arange(ncols) * cellsize, (nrows, 1)).astype(float)
    header = {"cellsize": cellsize, "nrows": nrows, "ncols": ncols, "xllcenter": 0.0, "yllcenter": 0.0}
    dem = {"rasterData": zRaster, "header": header}

    profile = {
        "x": np.array([10.0, 20.0, 35.5]),
        "y": np.array([50.0, 50.0, 50.0]),
    }

    result = prepPathGeneral.updateSZProfile(profile, dem)

    # since the ramp is linear, bilinear interpolation should exactly recover z == x
    np.testing.assert_allclose(result["z"], profile["x"], atol=1e-6)
    assert result["s"][0] == 0.0
    np.testing.assert_allclose(result["s"], [0.0, 10.0, 25.5], atol=1e-6)


# ---------------------------------------------------------------------------
# pathExtension
# ---------------------------------------------------------------------------

def _makeFlatDemAndCfg(extTopOption=2):
    cellsize = 1.0
    nrows, ncols = 200, 200
    zRaster = np.zeros((nrows, ncols))
    header = {"cellsize": cellsize, "nrows": nrows, "ncols": ncols, "xllcenter": 0.0, "yllcenter": 0.0}
    dem = {"rasterData": zRaster, "header": header}

    cfg = configparser.ConfigParser()
    cfg["PATH"] = {
        "extTopOption": str(extTopOption),
        "nCellsMinExtend": "1",
        "nCellsMaxExtend": "20",
        "factBottomExt": "0.1",
        "maxIterationExtBot": "10",
        "nBottomExtPrecision": "1",
    }
    return dem, cfg


def test_pathExtension_option2_extendsAndSetsIndices():
    """ test that pathExtension sets the mass-average indices and extends the profile at both ends """
    dem, cfg = _makeFlatDemAndCfg(extTopOption=2)

    profile = {
        "x": np.array([100.0, 105.0, 110.0, 115.0, 120.0]),
        "y": np.array([100.0, 100.0, 100.0, 100.0, 100.0]),
        "z": np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        "s": np.array([0.0, 5.0, 10.0, 15.0, 20.0]),
    }
    origLen = len(profile["x"])

    result = prepPathGeneral.pathExtension(profile, dem, cfg)

    # indices are set based on the ORIGINAL (pre-extension) length
    assert result["indStartMassAverage"] == 1
    assert result["indEndMassAverage"] == origLen - 2

    # profile grew by one point at the top and one at the bottom
    assert len(result["x"]) == origLen + 2
    # top extension moves in -x direction, bottom extension moves in +x direction
    assert result["x"][0] < profile["x"][1]  # extended top point is further "back"
    assert result["x"][-1] > profile["x"][-2]  # extended bottom point is further "forward"
    assert result["y"][0] == profile["y"][0]


def test_pathExtension_invalidTopOptionRaises():
    """ test that pathExtension raises ValueError for any extTopOption other than 2 """
    dem, cfg = _makeFlatDemAndCfg(extTopOption=0)

    profile = {
        "x": np.array([100.0, 105.0, 110.0]),
        "y": np.array([100.0, 100.0, 100.0]),
        "z": np.array([0.0, 0.0, 0.0]),
        "s": np.array([0.0, 5.0, 10.0]),
    }

    with pytest.raises(ValueError):
        prepPathGeneral.pathExtension(profile, dem, cfg)


# ---------------------------------------------------------------------------
# preparePathGeneralMain
# ---------------------------------------------------------------------------

def test_preparePathGeneralMain_shortProfile():
    """ test the short-circuit branch for a profile with 2 or fewer points """
    dem, cfg = _makeFlatDemAndCfg(extTopOption=2)

    profile = {
        "x": np.array([10.0, 20.0]),
        "y": np.array([10.0, 10.0]),
    }

    profileAveraged, profileExtended = prepPathGeneral.preparePathGeneralMain(profile, cfg, dem)

    # z and s were added by updateSZProfile
    assert "z" in profileAveraged
    assert "s" in profileAveraged
    np.testing.assert_allclose(profileAveraged["s"], [0.0, 10.0])

    # profileExtended is a copy with mass-average indices set to the short-profile defaults
    assert profileExtended["indStartMassAverage"] == 0
    assert profileExtended["indEndMassAverage"] == 1
    np.testing.assert_allclose(profileExtended["x"], profileAveraged["x"])

    # pathExtension/resamplePath must NOT have been invoked -> no extension happened,
    # i.e. length is unchanged from the input
    assert len(profileExtended["x"]) == 2


def test_preparePathGeneralMain_shortProfile_singlePoint():
    """ test the short-circuit branch for a profile with only 1 point """
    dem, cfg = _makeFlatDemAndCfg(extTopOption=2)

    profile = {
        "x": np.array([10.0]),
        "y": np.array([10.0]),
    }

    profileAveraged, profileExtended = prepPathGeneral.preparePathGeneralMain(profile, cfg, dem)

    assert profileExtended["indStartMassAverage"] == 0
    # max(len(x), 1) == 1 for a single-point profile
    assert profileExtended["indEndMassAverage"] == 0
