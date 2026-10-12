"""Tests for module debrisFunctions"""

import numpy as np
import pytest
import configparser
import pandas as pd
from pathlib import Path

import avaframe.com1DFA.debrisFunctions as debF


def test_addReleaseParticles():
    inputSimLines = {
        "releaseLine": {
            "Name": ["testTimeDepRel"],
            "Start": np.asarray([0.0]),
            "Length": np.asarray([5]),
            "type": "time dependent Release",
            "initializedFrom": "shapefile",
            "x": np.asarray(
                [
                    0,
                    10.0,
                    10.0,
                    0.0,
                    0.0,
                ]
            )
            - 2.5,
            "y": np.asarray([0.0, 0.0, 10.0, 10.0, 0.0]) - 2.5,
            "thicknessSource": ["csv file"],
            "thickness": 1,
        }
    }
    timeDepRelValues = {"thickness": np.array([1.0]), "velocity": np.array([0.0])}

    demHeader = {}
    demHeader["xllcenter"] = 0
    demHeader["yllcenter"] = 0
    demHeader["cellsize"] = 5.0
    demHeader["nodata_value"] = -9999
    demHeader["nrows"] = 7
    demHeader["ncols"] = 7
    dem = {"header": demHeader}
    dem["rasterData"] = np.ones((demHeader["nrows"], demHeader["ncols"]))
    dem["originalHeader"] = dem["header"]
    dem["areaRaster"] = np.ones((demHeader["nrows"], demHeader["ncols"]))
    dem["Nx"] = np.zeros_like(dem["rasterData"])
    dem["Ny"] = np.zeros_like(dem["rasterData"])
    dem["Nz"] = np.zeros_like(dem["rasterData"])

    cfg = configparser.ConfigParser()
    cfg["GENERAL"] = {
        "resType": "ppr|pft|pfv",
        "rho": "1000.",
        "gravAcc": "9.81",
        "cpIce": "2050",
        "TIni": "-10",
        "avalancheDir": "data/avaParabola",
        "massPerParticleDeterminationMethod": "MPPDH",
        "interpOption": "2",
        "initialiseParticlesFromFile": "False",
        "iniStep": "False",
        "seed": "12345",
        "sphKernelRadius": "1",
        "deltaTh": "1",
        "initPartDistType": "uniform",
        "thresholdPointInPoly": "0.001",
        "massPerPart": "1000",
        "thresholdPointInRel": "0",
    }

    particles = {
        "nPart": 3,
        "x": np.array([12, 20, 30]),
        "y": np.array([5, 10, 30]),
        "z": np.array([1, 1, 1]),
        "m": np.array([1000, 1000, 1000]),
        "idFixed": np.array([0, 0, 0]),
        "t": 1.0,
        "dt": 0.1,
    }
    nPart = particles["nPart"]
    particles["totalEnthalpy"] = (
        cfg["GENERAL"].getfloat("TIni") * cfg["GENERAL"].getfloat("cpIce")
        + cfg["GENERAL"].getfloat("gravAcc") * particles["z"]
    )
    particles["massPerPart"] = 1000
    particles["mTot"] = np.sum(particles["m"])
    particles["tPlot"] = 0
    particles["h"] = np.ones(nPart)
    particles["ux"] = np.zeros(nPart)
    particles["uy"] = np.zeros(nPart)
    particles["uz"] = np.zeros(nPart)
    particles["uAcc"] = np.zeros(nPart)
    particles["velocityMag"] = np.zeros(nPart)
    particles["trajectoryLengthXY"] = np.zeros(nPart)
    particles["trajectoryLengthXYCor"] = np.zeros(nPart)
    particles["trajectoryLengthXYZ"] = np.zeros(nPart)
    particles["trajectoryAngle"] = np.zeros(nPart)
    particles["stoppCriteria"] = False
    particles["peakForceSPH"] = 0.0
    particles["forceSPHIni"] = 0.0
    particles["peakMassFlowing"] = 0
    particles["xllcenter"] = dem["originalHeader"]["xllcenter"]
    particles["yllcenter"] = dem["originalHeader"]["yllcenter"]
    particles["nExitedParticles"] = 0.0
    particles["dmDet"] = np.zeros(nPart)
    particles["dmEnt"] = np.zeros(nPart)

    zPartArray0 = np.array([1, 1, 1])
    newParticleNumber = 4
    particlesTest = {
        "nPart": newParticleNumber + 3,
        "mTot": 7000,
        "x": np.append(particles["x"], np.array([0, 5, 0, 5])),
        "y": np.append(particles["y"], np.array([0, 0, 5, 5])),
        "z": np.append(particles["z"], np.ones([newParticleNumber])),
        "m": np.append(particles["m"], np.ones([newParticleNumber]) * 1000),
    }
    zPartArray0Test = np.ones(particlesTest["nPart"])

    particlesNewRel, zPartArray0NewRel = debF.addReleaseParticles(
        cfg, particles, inputSimLines, timeDepRelValues, dem, zPartArray0, timeDepRelIndex=0
    )

    assert np.all(np.equal(zPartArray0NewRel, zPartArray0Test))
    for key in particlesTest:
        if key in ["nPart", "mTot"]:
            assert particlesTest[key] == particlesNewRel[key]
        else:
            assert np.all(np.equal(particlesTest[key], particlesNewRel[key]))
    for key in ["ux", "uy", "uz", "velocityMag"]:
        assert np.all(np.equal(np.zeros(particlesTest["nPart"]), particlesNewRel[key]))

    cfg["GENERAL"]["deltaTh"] = "0.25"
    cfg["GENERAL"]["initPartDistType"] = "random"
    cfg["GENERAL"]["thresholdMassSplit"] = "1.5"

    particlesNewRel, zPartArray0NewRel = debF.addReleaseParticles(
        cfg, particles, inputSimLines, timeDepRelValues, dem, zPartArray0, timeDepRelIndex=0
    )
    assert particlesNewRel["nPart"] == 16 + 3
    for key in ["ux", "uy", "uz", "velocityMag", "x", "y", "z"]:
        assert len(particlesNewRel[key]) == particlesNewRel["nPart"]
    assert particlesNewRel["mTot"] == 7000

    particles["x"] = np.array([4, 10, 30])
    particles["y"] = np.array([5, 3, 30])

    with pytest.raises(ValueError):
        debF.addReleaseParticles(
            cfg, particles, inputSimLines, timeDepRelValues, dem, zPartArray0, timeDepRelIndex=0
        )


def test_addReleaseParticlesFromCoordinateCsv():
    """Coordinate CSV releases initialize particles and project vector velocity."""
    header = {"xllcenter": 0.0, "yllcenter": 0.0, "cellsize": 5.0, "nrows": 7, "ncols": 7}
    dem = {
        "header": header,
        "originalHeader": header,
        "rasterData": np.ones((7, 7)),
        "areaRaster": np.ones((7, 7)),
        "Nx": np.zeros((7, 7)),
        "Ny": np.zeros((7, 7)),
        "Nz": np.zeros((7, 7)),
    }
    cfg = configparser.ConfigParser()
    cfg["GENERAL"] = {
        "resType": "ppr|pft|pfv",
        "rho": "1000.",
        "gravAcc": "9.81",
        "cpIce": "2050",
        "TIni": "-10",
        "avalancheDir": "data/avaParabola",
        "massPerParticleDeterminationMethod": "MPPDIR",
        "interpOption": "2",
        "initialiseParticlesFromFile": "False",
        "iniStep": "False",
        "seed": "12345",
        "sphKernelRadius": "1",
        "deltaTh": "1",
        "initPartDistType": "uniform",
        "thresholdPointInPoly": "0.001",
        "massPerPart": "100000",
        "thresholdPointInRel": "0",
    }
    timeDepRelValues = {
        "timeStep": np.array([5.0, 5.0]),
        "thickness": np.array([1.5, 2.5]),
        "velocityX": np.array([3.0, -1.0]),
        "velocityY": np.array([-4.0, 2.0]),
        "velocityZ": np.array([5.0, 0.0]),
        "x": np.array([0.0, 5.0]),
        "y": np.array([0.0, 5.0]),
    }
    inputSimLines = {
        "releaseLine": {
            "initializedFrom": "csvfile",
            "timeDepRelValues": timeDepRelValues,
            "rasterData": np.zeros((3, 3)),
        }
    }
    particles = {
        "nPart": 3,
        "x": np.array([12.0, 20.0, 30.0]),
        "y": np.array([5.0, 10.0, 30.0]),
        "z": np.array([1.0, 1.0, 1.0]),
        "m": np.array([1000.0, 1000.0, 1000.0]),
        "idFixed": np.zeros(3),
        "t": 4.0,
        "dt": 0.5,
        "massPerPart": 1000.0,
        "mTot": 3000.0,
        "totalEnthalpy": np.full(3, -20490.19),
        "tPlot": 0,
        "h": np.ones(3),
        "ux": np.zeros(3),
        "uy": np.zeros(3),
        "uz": np.zeros(3),
        "uAcc": np.zeros(3),
        "velocityMag": np.zeros(3),
        "trajectoryLengthXY": np.zeros(3),
        "trajectoryLengthXYCor": np.zeros(3),
        "trajectoryLengthXYZ": np.zeros(3),
        "trajectoryAngle": np.zeros(3),
        "stoppCriteria": False,
        "peakForceSPH": 0.0,
        "forceSPHIni": 0.0,
        "peakMassFlowing": 0,
        "xllcenter": 0.0,
        "yllcenter": 0.0,
        "nExitedParticles": 0.0,
        "dmDet": np.zeros(3),
        "dmEnt": np.zeros(3),
    }
    particles, zPartArray0 = debF.addReleaseParticles(
        cfg,
        particles,
        inputSimLines,
        timeDepRelValues,
        dem,
        np.ones(3),
        timeDepRelIndex=np.array([0, 1]),
    )

    assert particles["nPart"] == 5
    assert np.array_equal(particles["x"], np.array([12.0, 20.0, 30.0, 0.0, 5.0]))
    assert np.array_equal(particles["y"], np.array([5.0, 10.0, 30.0, 0.0, 5.0]))
    assert np.array_equal(particles["ux"], np.array([0.0, 0.0, 0.0, 3.0, -1.0]))
    assert np.array_equal(particles["uy"], np.array([0.0, 0.0, 0.0, -4.0, 2.0]))
    assert np.array_equal(particles["uz"], np.array([0.0, 0.0, 0.0, 5.0, 0.0]))
    assert np.array_equal(zPartArray0, np.ones(5))


@pytest.fixture
def hydrographCfg(tmp_path):
    """Provide a temporary avalanche directory with a hydrograph input folder."""
    (tmp_path / "Inputs" / "OUTHYDR").mkdir(parents=True)
    cfg = configparser.ConfigParser()
    cfg["GENERAL"] = {"avalancheDir": str(tmp_path)}
    return cfg


def test_setupOutHydrographDF(hydrographCfg):
    """Read cross sections and initialize one zero-discharge column per station."""
    inputDir = hydrographCfg["GENERAL"]["avalancheDir"]

    csvFile = Path(inputDir) / "Inputs" / "OUTHYDR" / "crossSections.csv"
    crossSections = pd.DataFrame(
        {
            "station": [1, 1, 1, 2, 2, 2],
            "row": [397, 396, 395, 397, 396, 395],
            "column": [75, 75, 75, 163, 164, 165],
            "Nx": [1.0, 1.0, 1.0, 0.76975, 0.76975, 0.76975],
            "Ny": [0.0, 0.0, 0.0, 0.63833, 0.63833, 0.63833],
            "prjCsz": [0.86705, 1.86705, 1.99998, 1.31306, 2.72134, 2.17712],
        }
    )
    crossSections.to_csv(csvFile, index=False)

    # A horizontal DEM must have a normalized z component of 1 everywhere.
    dem = {
        "header": {
            "nrows": 751,
            "ncols": 1001,
            "cellsize": 2.0,
            "xllcenter": 999.000000000000,
            "yllcenter": -5001.000000000000,
            "nodata_value": -9999.0,
        },
        "rasterData": np.ones((751, 1001)),
    }

    outputDf, outIdx, nzNormed = debF.setupOutHydrographDF(hydrographCfg["GENERAL"], dem)

    pd.testing.assert_frame_equal(outIdx, crossSections)
    assert outputDf.index.name == "timestep"
    np.testing.assert_allclose(outputDf.index.to_numpy(), [0.0])
    assert list(outputDf.columns) == ["XS 1", "XS 2"]
    np.testing.assert_allclose(outputDf.iloc[0].to_numpy(), [0.0, 0.0])
    np.testing.assert_allclose(nzNormed, np.ones((751, 1001)))
    assert "Nx" not in dem  # Original DEM remains unchanged.


def test_setupOutHydrographDFMissingFile(hydrographCfg):
    """A missing input CSV must be reported explicitly."""
    dem = {"header": {}, "rasterData": np.zeros((751, 1001))}

    with pytest.raises(FileNotFoundError, match="No Output hydrograph csv-file"):
        debF.setupOutHydrographDF(hydrographCfg["GENERAL"], dem)


def test_collectOutputHydrograph():
    """Integrate flux over multiple cells and stations, incl. reversed normals."""
    outIdx = pd.DataFrame(
        {
            "station": [1, 1, 1, 2, 2, 2],
            "row": [397, 396, 395, 397, 396, 395],
            "column": [75, 75, 75, 163, 164, 165],
            "Nx": [1.0, 1.0, 1.0, 0.76975, 0.76975, 0.76975],
            "Ny": [0.0, 0.0, 0.0, 0.63833, 0.63833, 0.63833],
            "prjCsz": [0.86705, 1.86705, 1.99998, 1.31306, 2.72134, 2.17712],
        }
    )

    shape = (751, 1001)
    fields = {
        "Vx": np.ones(shape),
        "Vy": np.zeros(shape),
        "FT": np.ones(shape),
    }

    nzNormed = np.ones(shape)
    outputDf = pd.DataFrame({"XS 1": [0.0], "XS 2": [0.0]}, index=[0.0])
    outputDf.index.name = "timestep"
    # q = (Vx * Nx + Vy * Ny) * FT / nzNormed * distance
    qXS1 = 1 * 1 * 0.86705 + 1 * 1 * 1.86705 + 1 * 1 * 1.99998
    qXS2 = (
        (1 * 0.76975 + 0 * 0.63833) * 1 * 1.31306
        + (1 * 0.76975 + 0 * 0.63833) * 1 * 2.72134
        + (1 * 0.76975 + 0 * 0.63833) * 1 * 2.17712
    )

    t = 1.0
    result = debF.collectOutputHydrograph(outputDf, outIdx, nzNormed, fields, t)

    assert result is outputDf
    np.testing.assert_allclose(result.index.to_numpy(), [0.0, 1.0])
    assert result.loc[0.0, "XS 1"] == 0.0
    assert result.loc[0.0, "XS 2"] == 0.0
    assert result.loc[1.0, "XS 1"] == pytest.approx(qXS1)
    assert result.loc[1.0, "XS 2"] == pytest.approx(qXS2)


def test_collectOutputHydrographZeroVelocity():
    """A stationary flow must produce zero discharge even with nonzero thickness."""
    outIdx = pd.DataFrame(
        {"station": [1], "row": [0], "column": [0], "Nx": [1.0], "Ny": [0.0], "prjCsz": [2.0]}
    )

    outputDf = pd.DataFrame({"XS 1": [0.0]}, index=[0.0])
    nzNormed = np.ones((1, 1))
    fields = {"Vx": np.zeros((1, 1)), "Vy": np.zeros((1, 1)), "FT": np.ones((1, 1))}
    t = 1.0

    result = debF.collectOutputHydrograph(
        outputDf,
        outIdx,
        nzNormed,
        fields,
        t,
    )

    assert result.loc[1.0, "XS 1"] == pytest.approx(0.0)


"""
Test does not work because: When calling pytest, executing DFAfunctionsCython.upfateFieldsC() raises an error ("Fatal Python error: Aborted")
(see issue #1002?)
------------------------------

def test_initializeTimeDepRelease():
    timeDepRelValues = {
        "timeStep": np.array([0, 5, 10]),
        "thickness": np.array([1, 2, 3]),
        "velocity": np.array([0, 0, 0]),
    }
    inputSimLines = {
        "releaseLine": {
            "Name": ["testTimeDepRel"],
            "Start": np.asarray([0.0]),
            "Length": np.asarray([5]),
            "type": "time dependent Release",
            "x": np.asarray(
                [
                    0,
                    10.0,
                    10.0,
                    0.0,
                    0.0,
                ]
            )
            - 2.5,
            "y": np.asarray([0.0, 0.0, 10.0, 10.0, 0.0]) - 2.5,
            "thicknessSource": ["csv file"],
            "thickness": np.array([1]),
            "timeDepRelValues": timeDepRelValues,
        }
    }
    cfg = configparser.ConfigParser()
    cfg["GENERAL"] = {
        "resType": "ppr|pft|pfv",
        "rho": "1000.",
        "rhoEnt": "1000",
        "gravAcc": "9.81",
        "cpIce": "2050",
        "TIni": "-10",
        "avalancheDir": "data/avaParabola",
        "massPerParticleDeterminationMethod": "MPPDH",
        "interpOption": "2",
        "initialiseParticlesFromFile": "False",
        "iniStep": "False",
        "seed": "12345",
        "sphKernelRadius": "1",
        "deltaTh": "1",
        "initPartDistType": "uniform",
        "thresholdPointInPoly": "0.001",
        "massPerPart": "1000",
        "thresholdPointInRel": "0",
        "timeDependentRelease": "True",
    }

    demHeader = {}
    demHeader["xllcenter"] = 0
    demHeader["yllcenter"] = 0
    demHeader["cellsize"] = 5.0
    demHeader["nodata_value"] = -9999
    demHeader["nrows"] = 7
    demHeader["ncols"] = 7
    dem = {"header": demHeader}
    dem["rasterData"] = np.ones((demHeader["nrows"], demHeader["ncols"]))
    dem["originalHeader"] = dem["header"]
    dem["areaRaster"] = np.ones((demHeader["nrows"], demHeader["ncols"]))
    dem["Nx"] = np.zeros_like(dem["rasterData"])
    dem["Ny"] = np.zeros_like(dem["rasterData"])
    dem["Nz"] = np.zeros_like(dem["rasterData"])
    dem["headerNeighbourGrid"] = demHeader

    particles = {
        "nPart": 3,
        "x": np.array([10, 20, 30]),
        "y": np.array([5, 10, 30]),
        "z": np.array([1, 1, 1]),
        "m": np.array([1000, 1000, 1000]),
        "idFixed": np.array([0, 0, 0]),
        "t": 1.0,
        "dt": 0.1,
    }
    nPart = particles["nPart"]
    particles["totalEnthalpy"] = (
        cfg["GENERAL"].getfloat("TIni") * cfg["GENERAL"].getfloat("cpIce")
        + cfg["GENERAL"].getfloat("gravAcc") * particles["z"]
    )
    particles["massPerPart"] = 1000
    particles["mTot"] = np.sum(particles["m"])
    particles["tPlot"] = 0
    particles["h"] = np.ones(nPart)
    particles["ux"] = np.zeros(nPart)
    particles["uy"] = np.zeros(nPart)
    particles["uz"] = np.zeros(nPart)
    particles["uAcc"] = np.zeros(nPart)
    particles["velocityMag"] = np.zeros(nPart)
    particles["trajectoryLengthXY"] = np.zeros(nPart)
    particles["trajectoryLengthXYCor"] = np.zeros(nPart)
    particles["trajectoryLengthXYZ"] = np.zeros(nPart)
    particles["trajectoryAngle"] = np.zeros(nPart)
    particles["stoppCriteria"] = False
    particles["peakForceSPH"] = 0.0
    particles["forceSPHIni"] = 0.0
    particles["peakMassFlowing"] = 0
    particles["xllcenter"] = dem["originalHeader"]["xllcenter"]
    particles["yllcenter"] = dem["originalHeader"]["yllcenter"]
    particles["nExitedParticles"] = 0.0
    particles["dmDet"] = np.zeros(nPart)
    particles["dmEnt"] = np.zeros(nPart)
    particles["stoppedParticles"] = {}
    particles["stoppedParticles"]["m"] = np.empty(0)
    particles["stoppedParticles"]["x"] = np.empty(0)
    particles["stoppedParticles"]["y"] = np.empty(0)

    zPartArray0 = np.array([1, 1, 1])

    fields = {}
    fields["computeTA"] = False
    fields["computeKE"] = False
    fields["computeP"] = False
    fields["pfv"] = np.zeros_like(dem["rasterData"])
    fields["ppr"] = np.zeros_like(dem["rasterData"])
    fields["pft"] = np.zeros_like(dem["rasterData"])
    fields["pft"][[1, 2, 6], [2, 4, 6]] = 1
    fields["pta"] = np.zeros_like(dem["rasterData"])
    fields["pke"] = np.zeros_like(dem["rasterData"])
    fields["dmDet"] = np.zeros_like(dem["rasterData"])

    t = 10

    debF.initializeTimeDepRelease(cfg, inputSimLines, particles, fields, dem, zPartArray0, t)
"""
