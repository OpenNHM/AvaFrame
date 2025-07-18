""" Tests for module debrisFunctions """

import numpy as np
import pytest
import configparser

import avaframe.com1DFA.debrisFunctions as debF


def test_addHydrographParticles():
    inputSimLines = {
        "hydrographAreaLine": {
            "Name": ["testHydr"],
            "Start": np.asarray([0.0]),
            "Length": np.asarray([5]),
            "type": "Hydrograph",
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
    thickness = inputSimLines["hydrographAreaLine"]["thickness"]
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
        "avalancheDir": "data/avaKotHYDR",
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
        "thresholdPointInHydr": "0",
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

    particlesHydr, zPartArray0Hydr = debF.addHydrographParticles(
        cfg, particles, inputSimLines, thickness, velocityMag, dem, zPartArray0
    )

    assert np.all(np.equal(zPartArray0Hydr, zPartArray0Test))
    for key in particlesTest:
        if key in ["nPart", "mTot"]:
            assert particlesTest[key] == particlesHydr[key]
        else:
            assert np.all(np.equal(particlesTest[key], particlesHydr[key]))
    for key in ["ux", "uy", "uz", "velocityMag"]:
        assert np.all(np.equal(np.zeros(particlesTest["nPart"]), particlesHydr[key]))

    cfg["GENERAL"]["deltaTh"] = "0.25"
    cfg["GENERAL"]["initPartDistType"] = "random"
    cfg["GENERAL"]["thresholdMassSplit"] = "1.5"

    particlesHydr, zPartArray0Hydr = debF.addHydrographParticles(
        cfg, particles, inputSimLines, thickness, velocityMag, dem, zPartArray0
    )
    assert particlesHydr["nPart"] == 16 + 3
    for key in ["ux", "uy", "uz", "velocityMag", "x", "y", "z"]:
        assert len(particlesHydr[key]) == particlesHydr["nPart"]
    assert particlesHydr["mTot"] == 7000

    particles["x"] = np.array([4, 10, 30])
    particles["y"] = np.array([5, 3, 30])

    with pytest.raises(ValueError):
        debF.addHydrographParticles(cfg, particles, inputSimLines, thickness, velocityMag, dem, zPartArray0)

"""
TODO: When calling pytest, the following function raises an error ("Fatal Python error: Aborted")
(see issue #1002?)

def test_releaseHydrograph():
    inputSimLines = {
        "hydrographAreaLine": {
            "Name": ["testHydr"],
            "Start": np.asarray([0.0]),
            "Length": np.asarray([5]),
            "type": "Hydrograph",
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
            "values": {
                "timeStep": np.array([0, 1]),
                "thickness": np.array([1, 1]),
                "velocity": np.array([0, 0]),
            },
        }
    }
    thickness = inputSimLines["hydrographAreaLine"]["thickness"]
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
    dem["headerNeighbourGrid"] = demHeader

    cfg = configparser.ConfigParser()
    cfg["GENERAL"] = {
        "resType": "ppr|pft|pfv",
        "rho": "1000.",
        "rhoEnt": "1000.",
        "gravAcc": "9.81",
        "cpIce": "2050",
        "TIni": "-10",
        "avalancheDir": "data/avaKotHYDR",
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
    }

    particles = {
        "nPart": 3,
        "x": np.array([10, 20, 30]),
        "y": np.array([5, 10, 30]),
        "z": np.array([1, 1, 1]),
        "m": np.array([1000, 1000, 1000]),
        "idFixed": np.array([0, 0, 0]),
        "dmDet": np.array([0, 0, 0]),
        "dmEnt": np.array([0, 0, 0]),
        "ux": np.array([0, 0, 0]),
        "uy": np.array([0, 0, 0]),
        "uz": np.array([0, 0, 0]),
        "trajectoryAngle": np.array([0, 0, 0]),
        "stoppedParticles": {
            "m": np.empty(0),
            "x": np.empty(0),
            "y": np.empty(0),
        },
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
    particles["uAcc"] = np.zeros(nPart)
    particles["velocityMag"] = np.zeros(nPart)
    particles["trajectoryLengthXY"] = np.zeros(nPart)
    particles["trajectoryLengthXYCor"] = np.zeros(nPart)
    particles["trajectoryLengthXYZ"] = np.zeros(nPart)
    particles["stoppCriteria"] = False
    particles["peakForceSPH"] = 0.0
    particles["forceSPHIni"] = 0.0
    particles["peakMassFlowing"] = 0
    particles["xllcenter"] = dem["originalHeader"]["xllcenter"]
    particles["yllcenter"] = dem["originalHeader"]["yllcenter"]
    particles["nExitedParticles"] = 0.0

    fields = {
        "computeTA": 0,
        "computeKE": 0,
        "computeP": 0,
        "pfv": np.zeros_like(dem["rasterData"]),
        "ppr": np.zeros_like(dem["rasterData"]),
        "pft": np.zeros_like(dem["rasterData"]),
        "pta": np.zeros_like(dem["rasterData"]),
        "pke": np.zeros_like(dem["rasterData"]),
        "dmDet": np.zeros_like(dem["rasterData"]),
    }
    fields["pft"][[2, 4, 6], [1, 2, 6]] = 1

    zPartArray0 = np.array([1, 1, 1])

    t = 0.00

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

    debF.updateParticlesHydrograph(cfg, inputSimLines, particles, fields, dem, zPartArray0, t)

"""


if __name__ == "__main__":
    test_addHydrographParticles()
