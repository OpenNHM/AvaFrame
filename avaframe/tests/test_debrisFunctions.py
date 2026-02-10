"""Tests for module debrisFunctions"""

import numpy as np
import pytest
import configparser

import avaframe.com1DFA.debrisFunctions as debF


def test_addReleaseParticles():
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
            "thickness": 1,
        }
    }
    thickness = inputSimLines["releaseLine"]["thickness"]
    velocityMag = 0

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
        cfg, particles, inputSimLines, thickness, velocityMag, dem, zPartArray0
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
        cfg, particles, inputSimLines, thickness, velocityMag, dem, zPartArray0
    )
    assert particlesNewRel["nPart"] == 16 + 3
    for key in ["ux", "uy", "uz", "velocityMag", "x", "y", "z"]:
        assert len(particlesNewRel[key]) == particlesNewRel["nPart"]
    assert particlesNewRel["mTot"] == 7000

    particles["x"] = np.array([4, 10, 30])
    particles["y"] = np.array([5, 3, 30])

    with pytest.raises(ValueError):
        debF.addReleaseParticles(cfg, particles, inputSimLines, thickness, velocityMag, dem, zPartArray0)


def test_checkTimeDepRelease():
    timeDepRelValues = {
        "timeStep": np.array([0, 10, 20, 30, 35]),
        "thickness": np.array([1, 1, 1, 1, 1]),
        "velocity": np.array([10, 10, 10, 10, 10]),
    }
    timeDepRelCsv = "path2TimeDepRel.csv"
    # no error should occur
    debF.checkTimeDepRelease(timeDepRelValues, timeDepRelCsv)

    timeDepRelValues["timeStep"] = np.array([20, 10, 0, 30, 35])
    debF.checkTimeDepRelease(timeDepRelValues, timeDepRelCsv)

    # timesteps are not unique
    timeDepRelValues["timeStep"] = np.array([0, 10, 10, 30, 35])

    with pytest.raises(ValueError) as e:
        debF.checkTimeDepRelease(timeDepRelValues, timeDepRelCsv)
    assert ("The provided time dependent release timesteps in %s are not unique" % (timeDepRelCsv)) in str(
        e.value
    )

    timeDepRelValues["timeStep"] = np.array([0, 0])
    with pytest.raises(ValueError) as e:
        debF.checkTimeDepRelease(timeDepRelValues, timeDepRelCsv)
    assert ("The provided time dependent release timesteps in %s are not unique" % (timeDepRelCsv)) in str(
        e.value
    )

    # no timestep 0
    timeDepRelValues["timeStep"] = np.array([20, 15, 10, 30, 35])

    with pytest.raises(ValueError) as e:
        debF.checkTimeDepRelease(timeDepRelValues, timeDepRelCsv)
    assert (
        "If release is time dependent, a thickness needs to be provided for  time step 0 s in %s"
        % (timeDepRelCsv)
    ) in str(e.value)

    # thickness needs to be > 0
    timeDepRelValues["timeStep"] = np.array([20, 10, 0, 30, 35])
    timeDepRelValues["thickness"] = np.array([1, 0, 1, 1, 1])
    with pytest.raises(ValueError) as e:
        debF.checkTimeDepRelease(timeDepRelValues, timeDepRelCsv)
    assert ("For every release time step a thickness > 0") in str(e.value)

    timeDepRelValues["thickness"] = np.array([1, 1, -1, 1, 1])
    with pytest.raises(ValueError) as e:
        debF.checkTimeDepRelease(timeDepRelValues, timeDepRelCsv)
    assert ("For every release time step a thickness > 0") in str(e.value)

    # velocity needs to be >= 0
    timeDepRelValues["thickness"] = np.array([1, 1, 1, 1, 1])
    timeDepRelValues["velocity"] = np.array([10, 10, 10, -10, 10])
    with pytest.raises(ValueError) as e:
        debF.checkTimeDepRelease(timeDepRelValues, timeDepRelCsv)
    assert ("The initial velocity provided in %s can not be negative." % (timeDepRelCsv)) in str(e.value)


"""
Test does not word because: When calling pytest, executing DFAfunctionsCython.upfateFieldsC() raises an error ("Fatal Python error: Aborted")
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
