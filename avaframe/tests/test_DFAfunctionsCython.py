""""""

import numpy as np
import math
import configparser
import pytest
import matplotlib.tri as tri

# Local imports
import avaframe.com1DFA.DFAfunctionsCython as DFAfunC
import avaframe.com1DFA.DFAtools as DFAtls
import avaframe.com1DFA.DFAToolsCython as DFAtlsC
import avaframe.in3Utils.geoTrans as geoTrans


def test_getNeighborsC(capfd):
    """Test the grid search/particle location method"""
    header = {}
    header["ncols"] = 5
    header["nrows"] = 6
    header["cellsize"] = 1
    dem = {}
    dem["header"] = header
    dem["headerNeighbourGrid"] = header
    particles = {}
    particles["nPart"] = 18
    particles["x"] = np.array([1.6, 0.4, 1, 2, 1, 2, 0, 1, 0, 2, 0, 2, 1, 2, 3, 3, 4, 0])
    particles["y"] = np.array([2.6, 1.4, 0, 1, 3, 3, 2, 1, 0, 0, 3, 2, 2, 1, 1, 4, 5, 5])
    particles["z"] = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    particles["m"] = particles["z"]
    atol = 1e-10
    indPCell = np.array(
        [
            0.0,  # always an extra zero at the begining
            1,  # found 1 particle
            2,  # found 1 particle
            3,  # found 1 particle
            3,  # nothing happens
            3,  # nothing happens
            4,  # found 1 particle
            5,  # found 1 particle
            7,  # found 2 particles
            8,  # found 1 particle
            8,  # nothing happens
            9,  # found 1 particle
            10,  # found 1 particle
            11,  # found 1 particles
            11,
            11,  # nothing happens
            12,  # found 1 particle
            13,  # found 1 particle
            15,  # found 2 particle
            15,
            15,
            15,
            15,
            15,  # nothing happens
            16,  # found 1 particle
            16,  # nothing happens
            17,  # found 1 particle
            17,
            17,
            17,  # nothing happens
            18,
        ]
    )  # found 1 particle
    pInC = np.array([8, 2, 9, 1, 7, 13, 3, 14, 6, 12, 11, 10, 4, 5, 0, 15, 17, 16])

    particles = DFAfunC.getNeighborsC(particles, dem)
    #    print(particles['inCellDEM'])
    #    print(particles['indPartInCell'])
    #    print(particles['partInCell'])
    assert np.allclose(particles["indPartInCell"], indPCell, atol=atol)
    assert np.allclose(particles["partInCell"], pInC, atol=atol)


def test_computeEntMassAndForce(capfd):
    """Test the computeEntMassAndForce function"""
    dt = 0.1
    entrMassCell = 0
    areaPart = 1
    uMag = 10
    tau = 1000
    entEroEnergy = 5000
    rhoEnt = 100
    dm, areaEntrPart = DFAfunC.computeEntMassAndForce(
        dt, entrMassCell, areaPart, uMag, tau, entEroEnergy, rhoEnt
    )
    #    print(dm, areaEntrPart)
    assert dm == 0
    assert areaEntrPart == 0

    entrMassCell = 200
    entEroEnergy = 0
    dm, areaEntrPart = DFAfunC.computeEntMassAndForce(
        dt, entrMassCell, areaPart, uMag, tau, entEroEnergy, rhoEnt
    )
    #    print(dm, areaEntrPart)
    assert dm == 200
    assert areaEntrPart == 2

    entEroEnergy = 5000
    dm, areaEntrPart = DFAfunC.computeEntMassAndForce(
        dt, entrMassCell, areaPart, uMag, tau, entEroEnergy, rhoEnt
    )
    #    print(dm, areaEntrPart)
    assert dm == 0.2
    assert areaEntrPart == 1


def test_computeDetMass(capfd):
    """Test the computeDetMass function"""
    dt = 0.1
    detCell = 0
    areaPart = 1
    uMag = 10
    dmDet = DFAfunC.computeDetMass(dt, detCell, areaPart, uMag)
    assert dmDet == 0

    detCell = 10
    dmDet = DFAfunC.computeDetMass(dt, detCell, areaPart, uMag)
    #    print(dmDet)
    assert dmDet == -0.1


def test_computeResForce(capfd):
    """Test the computeResForce function"""

    areaPart = 1
    rho = 200
    cResCell = 1
    uMag = 10
    explicitFriction = 0
    # cRes
    resistanceType = 1
    cResPart = DFAfunC.computeResForce(areaPart, rho, cResCell, uMag, explicitFriction, resistanceType)
    print(cResPart)
    assert cResPart == -2000

    explicitFriction = 1
    cResPart = DFAfunC.computeResForce(areaPart, rho, cResCell, uMag, explicitFriction, resistanceType)
    print(cResPart)
    assert cResPart == -20000


def test_account4FrictionForce(capfd):
    """Test the account4FrictionForce function"""
    uxNew = 10
    uyNew = 0
    uzNew = 0
    m = 10
    dt = 0.1
    forceFrict = 100
    uMag = 10
    explicitFriction = 0
    uxNew, uyNew, uzNew, dtStop = DFAfunC.account4FrictionForce(
        uxNew, uyNew, uzNew, m, dt, forceFrict, uMag, explicitFriction
    )
    #    print(uxNew, uyNew, uzNew, dtStop)
    assert dtStop == 0.1
    assert uxNew == 5

    uxNew = 10
    uyNew = 0
    uzNew = 0
    explicitFriction = 1
    m = 0.5
    uxNew, uyNew, uzNew, dtStop = DFAfunC.account4FrictionForce(
        uxNew, uyNew, uzNew, m, dt, forceFrict, uMag, explicitFriction
    )
    #    print(uxNew, uyNew, uzNew, dtStop)
    #    print(dt*forceFrict/m)
    assert dtStop == 0.05
    assert uxNew == 0

    uxNew = 10
    uyNew = 0
    uzNew = 0
    m = 10
    uxNew, uyNew, uzNew, dtStop = DFAfunC.account4FrictionForce(
        uxNew, uyNew, uzNew, m, dt, forceFrict, uMag, explicitFriction
    )
    #    print(uxNew, uyNew, uzNew, dtStop)
    #    print(dt*forceFrict/m)
    assert dtStop == 0.1
    assert uxNew == 9.0


def test_updatePositionC():
    """test updating the position of particles"""

    # TODO: make test also if velocity in z not zero!! - so to test also when reprojecting onto surface

    # setup required input
    cfg = configparser.ConfigParser()
    cfg["GENERAL"] = {
        "stopCrit": "0.01",
        "stopCritIni": "0.1",
        "stopCritIniSmall": "1.001",
        "stopCritType": "kinEnergy",
        "uFlowingThreshold": "0.1",
        "gravAcc": "9.81",
        "velMagMin": "1.e-6",
        "rho": "100.",
        "interpOption": "2",
        "explicitFriction": "0",
        "centeredPosition": "1",
        "reprojMethodPosition": "2",
        "reprojectionIterations": "5",
        "thresholdProjection": "0.001",
        "dissDam": "1",
        "snowSlide": "1",
        "wetSnow": "1",
        "adaptSfcStopped": "0",
        "adaptSfcDetrainment": "0",
        "adaptSfcEntrainment": "0",
    }

    particles = {
        "dt": 1.0,
        "m": np.asarray([10.0, 10.0, 10.0]),
        "idFixed": np.asarray([0.0, 0.0, 0.0]),
        "trajectoryLengthXY": np.asarray([0.0, 0.0, 0.0]),
        "trajectoryLengthXYCor": np.asarray([0.0, 0.0, 0.0]),
        "trajectoryLengthXYZ": np.asarray([0.0, 0.0, 0.0]),
        "x": np.asarray([0.0, 1.0, 2.0]),
        "y": np.asarray([2.0, 3.0, 4.0]),
        "z": np.asarray([1.0, 1.0, 1.0]),
        "ux": np.asarray([1.0, 1.0, 1.0]),
        "uy": np.asarray([1.0, 1.0, 1.0]),
        "uz": np.asarray([0.0, 0.0, 0.0]),
        "uAcc": np.asarray([0.0, 0.0, 0.0]),
        "kineticEne": 0.0,
        "peakKinEne": 0.0,
        "peakForceSPH": 0.0,
        "forceSPHIni": 0.0,
        "nPart": 3,
        "peakMassFlowing": 0.0,
        "iterate": True,
        "totalEnthalpy": np.asarray([0.0, 0.0, 0.0]),
        "velocityMag": np.asarray([1.0, 1.0, 1.0]),
        "h": np.asarray([1.0, 1.0, 1.0]),
        "ID": np.array([0, 1, 2]),
        "stoppedParticles": {},
    }
    particles["potentialEne"] = np.sum(9.81 * particles["z"] * particles["m"])

    demHeader = {}
    demHeader["xllcenter"] = 0.0
    demHeader["yllcenter"] = 0.0
    demHeader["cellsize"] = 5.0
    demHeader["nodata_value"] = -9999
    demHeader["nrows"] = 10
    demHeader["ncols"] = 10
    dem = {"header": demHeader}
    dem["rasterData"] = np.ones((demHeader["nrows"], demHeader["ncols"]))
    dem["outOfDEM"] = np.zeros((demHeader["nrows"], demHeader["ncols"])).flatten()
    dem["Nx"] = np.zeros((demHeader["nrows"], demHeader["ncols"]))
    dem["Ny"] = np.zeros((demHeader["nrows"], demHeader["ncols"]))
    dem["Nz"] = np.ones((demHeader["nrows"], demHeader["ncols"]))

    force = {
        "forceZ": np.asarray([0.0, 0.0, 0.0]),
        "forceFrict": np.asarray([10.0, 10.0, 10.0]),
        "dM": np.asarray([0.0, 0.0, 0.0]),
        "dMDet": np.asarray([0.0, 0.0, 0.0]),
        "forceX": np.asarray([50.0, 50.0, 50.0]),
        "forceY": np.asarray([50.0, 50.0, 50.0]),
        "forceSPHX": np.asarray([50.0, 50.0, 50.0]),
        "forceSPHY": np.asarray([50.0, 50.0, 50.0]),
        "forceSPHZ": np.asarray([0.0, 0.0, 0.0]),
    }
    fields = {"FT": np.zeros((2, 2))}
    # crete a dummy dict (needed so that cython runs)
    wallLineDict = {
        "dam": 0,
        "nIterDam": 1,
        "cellsCrossed": np.zeros((dem["header"]["ncols"] * dem["header"]["nrows"])).astype(int),
    }
    for key in [
        "x",
        "y",
        "z",
        "xCrown",
        "yCrown",
        "zCrown",
        "xTangent",
        "yTangent",
        "zTangent",
    ]:
        wallLineDict[key] = np.ones((1)) * 1.0
    for key in ["nPoints", "height", "slope", "restitutionCoefficient"]:
        wallLineDict[key] = 0
    dem["damLine"] = wallLineDict
    typeStop = 0

    # kinetic energy new
    kinEneNew = 0.0
    potEneNew = 0.0
    for k in range(3):
        kinEneNew = kinEneNew + particles["m"][k] * np.sqrt(5.5**2 + 5.5**2 + 0**2) ** 2 * 0.5
        potEneNew = potEneNew + particles["m"][k] * 9.81 + 0.0

    particles = DFAfunC.updatePositionC(cfg["GENERAL"], particles, dem, force, fields, typeStop=typeStop)
    uAcc = (np.sqrt(5.5**2 + 5.5**2 + 0.0) - np.sqrt(1.0**2 + 1.0**2 + 0.0)) / 1.0
    velocityMag = np.asarray(
        [
            np.sqrt((5.5**2) + (5.5**2)),
            np.sqrt((5.5**2) + (5.5**2)),
            np.sqrt((5.5**2) + (5.5**2)),
        ]
    )

    assert np.array_equal(particles["m"], np.asarray([10.0, 10.0, 10.0]))
    assert np.array_equal(particles["x"], np.array([3.25, 4.25, 5.25]))
    assert np.array_equal(particles["y"], np.asarray([5.25, 6.25, 7.25]))
    assert np.array_equal(particles["z"], np.asarray([1.0, 1.0, 1.0]))
    assert np.allclose(particles["ux"], np.asarray([5.5, 5.5, 5.5]), atol=1.0e-4)
    assert np.allclose(particles["uy"], np.asarray([5.5, 5.5, 5.5]), atol=1.0e-4)
    assert np.allclose(particles["uAcc"], np.asarray([uAcc, uAcc, uAcc]), atol=1.0e-4)
    assert np.allclose(particles["velocityMag"], velocityMag, atol=1.0e-4)
    assert np.array_equal(particles["uz"], np.asarray([0.0, 0.0, 0.0]))
    assert particles["massEntrained"] == 0.0
    assert particles["massDetrained"] == 0.0
    assert particles["nPart"] == 3
    assert (kinEneNew - 1.0e-4) < particles["kineticEne"] < (kinEneNew + 1.0e-4)
    assert (potEneNew - 1.0e-4) < particles["potentialEne"] < (potEneNew + 1.0e-4)
    assert particles["iterate"] == True

    particles = {
        "dt": 1.0,
        "m": np.asarray([10.0, 10.0, 10.0]),
        "idFixed": np.asarray([0.0, 0.0, 0.0]),
        "trajectoryLengthXY": np.asarray([0.0, 0.0, 0.0]),
        "trajectoryLengthXYCor": np.asarray([0.0, 0.0, 0.0]),
        "trajectoryLengthXYZ": np.asarray([0.0, 0.0, 0.0]),
        "x": np.asarray([0.0, 1.0, 2.0]),
        "y": np.asarray([2.0, 3.0, 4.0]),
        "z": np.asarray([1.0, 1.0, 1.0]),
        "ux": np.asarray([1.0, 1.0, 1.0]),
        "uy": np.asarray([1.0, 1.0, 1.0]),
        "uz": np.asarray([0.0, 0.0, 0.0]),
        "uAcc": np.asarray([0.0, 0.0, 0.0]),
        "kineticEne": 0.0,
        "peakKinEne": 100000.0,
        "peakForceSPH": 0.0,
        "forceSPHIni": 0.0,
        "nPart": 3,
        "velocityMag": np.asarray([1.0, 1.0, 1.0]),
        "peakMassFlowing": 0.0,
        "iterate": True,
        "totalEnthalpy": np.asarray([0.0, 0.0, 0.0]),
        "h": np.asarray([1.0, 1.0, 1.0]),
        "ID": np.array([0, 1, 2]),
        "stoppedParticles": {},
    }
    particles["potentialEne"] = np.sum(9.81 * particles["z"] * particles["m"])

    # call function to be tested
    particles = DFAfunC.updatePositionC(cfg["GENERAL"], particles, dem, force, fields, typeStop=typeStop)

    assert np.array_equal(particles["m"], np.asarray([10.0, 10.0, 10.0]))
    assert np.array_equal(particles["x"], np.array([3.25, 4.25, 5.25]))
    assert np.array_equal(particles["y"], np.asarray([5.25, 6.25, 7.25]))
    assert np.array_equal(particles["z"], np.asarray([1.0, 1.0, 1.0]))
    assert np.allclose(particles["ux"], np.asarray([5.5, 5.5, 5.5]), atol=1.0e-4)
    assert np.allclose(particles["uy"], np.asarray([5.5, 5.5, 5.5]), atol=1.0e-4)
    assert np.allclose(particles["uAcc"], np.asarray([uAcc, uAcc, uAcc]), atol=1.0e-4)
    assert np.array_equal(particles["uz"], np.asarray([0.0, 0.0, 0.0]))
    assert particles["massEntrained"] == 0.0
    assert particles["nPart"] == 3
    assert (kinEneNew - 1.0e-4) < particles["kineticEne"] < (kinEneNew + 1.0e-4)
    assert (potEneNew - 1.0e-4) < particles["potentialEne"] < (potEneNew + 1.0e-4)
    assert particles["iterate"] == False

    particles = {
        "dt": 1.0,
        "m": np.asarray([10.0, 10.0, 10.0]),
        "idFixed": np.asarray([0.0, 0.0, 0.0]),
        "trajectoryLengthXY": np.asarray([0.0, 0.0, 0.0]),
        "trajectoryLengthXYCor": np.asarray([0.0, 0.0, 0.0]),
        "trajectoryLengthXYZ": np.asarray([0.0, 0.0, 0.0]),
        "x": np.asarray([0.0, 1.0, 2.0]),
        "y": np.asarray([2.0, 3.0, 4.0]),
        "z": np.asarray([1.0, 1.0, 1.0]),
        "ux": np.asarray([1.0, 1.0, 1.0]),
        "uy": np.asarray([1.0, 1.0, 1.0]),
        "uz": np.asarray([0.0, 0.0, 0.0]),
        "uAcc": np.asarray([0.0, 0.0, 0.0]),
        "kineticEne": 0.0,
        "peakKinEne": 10000.0,
        "peakForceSPH": 100000.0,
        "forceSPHIni": 1.0e5,
        "nPart": 3,
        "velocityMag": np.asarray([1.0, 1.0, 1.0]),
        "peakMassFlowing": 0.0,
        "iterate": True,
        "totalEnthalpy": np.asarray([0.0, 0.0, 0.0]),
        "h": np.asarray([1.0, 1.0, 1.0]),
        "ID": np.array([0, 1, 2]),
        "stoppedParticles": {},
    }
    particles["potentialEne"] = np.sum(9.81 * particles["z"] * particles["m"])
    typeStop = 1

    sphForceNew = 0.0
    kinEneNew = 0.0
    potEneNew = 0.0
    for k in range(3):
        sphForceNew = sphForceNew + particles["m"][k] * np.sqrt(50.0**2 + 50.0**2 + 0.0**2) ** 2.0
        kinEneNew = kinEneNew + particles["m"][k] * np.sqrt(11.0**2 + 11.0**2 + 0**2) ** 2 * 0.5
        potEneNew = potEneNew + particles["m"][k] * 9.81 + 0.0

    # call function to be tested
    particles = DFAfunC.updatePositionC(cfg["GENERAL"], particles, dem, force, fields, typeStop=typeStop)
    #    print('sph', particles['peakForceSPH'], sphForceNew)

    uAcc = (np.sqrt(11.0**2 + 11.0**2 + 0.0) - np.sqrt(1.0**2 + 1.0**2 + 0.0)) / 1.0
    velocityMag = np.asarray(
        [
            np.sqrt((11.0**2) + (11.0**2)),
            np.sqrt((11.0**2) + (11.0**2)),
            np.sqrt((11.0**2) + (11.0**2)),
        ]
    )

    assert np.array_equal(particles["m"], np.asarray([10.0, 10.0, 10.0]))
    assert np.array_equal(particles["x"], np.array([6.0, 7.0, 8.0]))
    assert np.array_equal(particles["y"], np.asarray([8.0, 9.0, 10.0]))
    assert np.array_equal(particles["z"], np.asarray([1.0, 1.0, 1.0]))
    assert np.allclose(particles["ux"], np.asarray([11.0, 11.0, 11.0]), atol=1.0e-4)
    assert np.allclose(particles["uy"], np.asarray([11.0, 11.0, 11.0]), atol=1.0e-4)
    assert np.allclose(particles["velocityMag"], velocityMag, atol=1.0e-4)
    assert np.array_equal(particles["uz"], np.asarray([0.0, 0.0, 0.0]))
    assert np.allclose(particles["uAcc"], np.asarray([uAcc, uAcc, uAcc]), atol=1.0e-4)
    assert particles["massEntrained"] == 0.0
    assert particles["nPart"] == 3
    assert (kinEneNew - 1.0e-4) < particles["kineticEne"] < (kinEneNew + 1.0e-4)
    assert (potEneNew - 1.0e-4) < particles["potentialEne"] < (potEneNew + 1.0e-4)
    assert particles["iterate"] == False

    particles = {
        "dt": 1.0,
        "m": np.asarray([10.0, 10.0, 10.0]),
        "idFixed": np.asarray([0.0, 0.0, 0.0]),
        "trajectoryLengthXY": np.asarray([0.0, 0.0, 0.0]),
        "trajectoryLengthXYCor": np.asarray([0.0, 0.0, 0.0]),
        "trajectoryLengthXYZ": np.asarray([0.0, 0.0, 0.0]),
        "x": np.asarray([0.0, 1.0, 2.0]),
        "y": np.asarray([2.0, 3.0, 4.0]),
        "z": np.asarray([1.0, 1.0, 1.0]),
        "ux": np.asarray([1.0, 1.0, 1.0]),
        "uy": np.asarray([1.0, 1.0, 1.0]),
        "uz": np.asarray([0.0, 0.0, 0.0]),
        "uAcc": np.asarray([0.0, 0.0, 0.0]),
        "kineticEne": 0.0,
        "peakKinEne": 10000.0,
        "peakForceSPH": 1000.0,
        "forceSPHIni": 1.0e5,
        "nPart": 3,
        "velocityMag": np.asarray([1.0, 1.0, 1.0]),
        "peakMassFlowing": 0.0,
        "iterate": True,
        "totalEnthalpy": np.asarray([0.0, 0.0, 0.0]),
        "h": np.asarray([1.0, 1.0, 1.0]),
        "ID": np.array([0, 1, 2]),
        "stoppedParticles": {},
    }
    particles["potentialEne"] = np.sum(9.81 * particles["z"] * particles["m"])
    typeStop = 1

    sphForceNew = 0.0
    kinEneNew = 0.0
    potEneNew = 0.0
    for k in range(3):
        sphForceNew = sphForceNew + particles["m"][k] * np.sqrt(50.0**2 + 50.0**2 + 0.0**2) ** 2.0
        kinEneNew = kinEneNew + particles["m"][k] * np.sqrt(11.0**2 + 11.0**2 + 0**2) ** 2 * 0.5
        potEneNew = potEneNew + particles["m"][k] * 9.81 + 0.0

    # call function to be tested
    particles = DFAfunC.updatePositionC(cfg["GENERAL"], particles, dem, force, fields, typeStop=typeStop)
    #    print('sph', particles['peakForceSPH'], sphForceNew)

    assert np.array_equal(particles["m"], np.asarray([10.0, 10.0, 10.0]))
    assert np.array_equal(particles["x"], np.array([6.0, 7.0, 8.0]))
    assert np.array_equal(particles["y"], np.asarray([8.0, 9.0, 10.0]))
    assert np.array_equal(particles["z"], np.asarray([1.0, 1.0, 1.0]))
    assert np.allclose(particles["ux"], np.asarray([11.0, 11.0, 11.0]), atol=1.0e-4)
    assert np.allclose(particles["uy"], np.asarray([11.0, 11.0, 11.0]), atol=1.0e-4)
    assert np.allclose(particles["uAcc"], np.asarray([uAcc, uAcc, uAcc]), atol=1.0e-4)
    assert np.array_equal(particles["uz"], np.asarray([0.0, 0.0, 0.0]))
    assert particles["massEntrained"] == 0.0
    assert particles["nPart"] == 3
    assert (kinEneNew - 1.0e-4) < particles["kineticEne"] < (kinEneNew + 1.0e-4)
    assert (potEneNew - 1.0e-4) < particles["potentialEne"] < (potEneNew + 1.0e-4)
    assert particles["iterate"] == True

    particles = {
        "dt": 1.0,
        "m": np.asarray([10.0, 10.0, 10.0]),
        "idFixed": np.asarray([0.0, 0.0, 0.0]),
        "trajectoryLengthXY": np.asarray([0.0, 0.0, 0.0]),
        "trajectoryLengthXYCor": np.asarray([0.0, 0.0, 0.0]),
        "trajectoryLengthXYZ": np.asarray([0.0, 0.0, 0.0]),
        "x": np.asarray([0.0, 1.0, 2.0]),
        "y": np.asarray([2.0, 3.0, 4.0]),
        "z": np.asarray([1.0, 1.0, 1.0]),
        "ux": np.asarray([0.0, 1.0, 1.0]),
        "uy": np.asarray([0.0, 1.0, 1.0]),
        "uz": np.asarray([0.0, 0.0, 0.0]),
        "uAcc": np.asarray([0.0, 0.0, 0.0]),
        "kineticEne": 0.0,
        "peakKinEne": 0.0,
        "peakForceSPH": 0.0,
        "forceSPHIni": 0.0,
        "nPart": 3,
        "peakMassFlowing": 0.0,
        "iterate": True,
        "totalEnthalpy": np.asarray([0.0, 0.0, 0.0]),
        "velocityMag": np.asarray([0.0, 1.0, 1.0]),
        "h": np.asarray([1.0, 1.0, 1.0]),
        "ID": np.array([0, 1, 2]),
        "stoppedParticles": {},
    }
    particles["potentialEne"] = np.sum(9.81 * particles["z"] * particles["m"])

    force = {
        "forceZ": np.asarray([0.0, 0.0, 0.0]),
        "forceFrict": np.asarray([10.0, 10.0, 10.0]),
        "dM": np.asarray([0.0, 0.0, 0.0]),
        "dMDet": np.asarray([0.0, 0.0, 0.0]),
        "forceX": np.asarray([0.0, 50.0, 50.0]),
        "forceY": np.asarray([0.0, 50.0, 50.0]),
        "forceSPHX": np.asarray([0.0, 50.0, 50.0]),
        "forceSPHY": np.asarray([0.0, 50.0, 50.0]),
        "forceSPHZ": np.asarray([0.0, 0.0, 0.0]),
    }

    cfg["GENERAL"]["adaptSfcStopped"] = "1"
    cfg["GENERAL"]["explicitFriction"] = "1"
    cfg["GENERAL"]["snowSlide"] = "0"
    typeStop = 1

    particles = DFAfunC.updatePositionC(cfg["GENERAL"], particles, dem, force, fields, typeStop=typeStop)
    assert np.array_equal(particles["stoppedParticles"]["m"], np.array([10.0]))
    assert particles["stoppedParticles"]["x"] == 0.0
    assert particles["stoppedParticles"]["y"] == 2.0
    assert particles["stoppedParticles"]["ID"] == 0
    assert np.array_equal(particles["ID"], np.array([1, 2]))
    assert particles["nPart"] == 2
    assert np.all(particles["velocityMag"] > 0)
    assert particles["massStopped"] == -np.sum(particles["stoppedParticles"]["m"])


def test_computeTrajectoryAngle():
    # first compute travel angle for each particle
    # get parent Id in order to  get the first z position
    parentID = np.array([0, 1, 2, 0])
    nPart = 4
    # get z0
    zPartArray0 = np.array([10.0, 9.0, 8.0])
    s = np.array([10.0, 10.0, 0.0, 10.0])
    z = np.array([0.0, 0.0, 0.0, 1.0])
    particles = {"nPart": nPart, "parentID": parentID, "trajectoryLengthXY": s, "z": z}
    particles = DFAfunC.computeTrajectoryAngleC(particles, zPartArray0)
    #    print(particles['trajectoryAngle'])
    gamma = particles["trajectoryAngle"]
    assert gamma[2] == 0
    assert gamma[0] == 45
    assert gamma[1] == pytest.approx(41.9872125, rel=1e-6)
    assert gamma[3] == pytest.approx(41.9872125, rel=1e-6)


def test_initializeBondsC():
    nPart = 3
    x = np.array([0.0, 1.0, 0.0])
    y = np.array([0.0, 0.0, 1.0])
    z = np.array([0.0, 0.0, 0.0])
    # original triangulation
    triangles = tri.Triangulation(x, y)
    particles = {"nPart": nPart, "x": x, "y": y, "z": z}
    particles = DFAfunC.initializeBondsC(particles, triangles)
    #    print(triangles.triangles)
    #    print(triangles.edges)
    #    print(particles['bondStart'])
    #    print(particles['bondDist'])
    #    print(particles['bondPart'])
    bondStart = particles["bondStart"]
    bondDist = particles["bondDist"]
    bondPart = particles["bondPart"]
    assert np.array_equal(bondStart, np.asarray([0, 2, 4, 6]))
    for k in range(nPart):
        # loop on all bonded particles
        neighbors = list()
        for ib in range(bondStart[k], bondStart[k + 1]):
            l = bondPart[ib]
            neighbors.append(l)

        neighbors.sort()
        if k == 0:
            assert neighbors == [1, 2]
        if k == 1:
            assert neighbors == [0, 2]
        if k == 2:
            assert neighbors == [0, 1]
    bondDist.sort()
    assert np.array_equal(bondDist, np.asarray([1, 1, 1, 1, math.sqrt(2), math.sqrt(2)]))


def test_removeBondsC():
    nPart = 3
    x = np.array([0.0, 1.0, 0.0])
    y = np.array([0.0, 0.0, 1.0])
    z = np.array([0.0, 0.0, 0.0])
    # original triangulation
    triangles = tri.Triangulation(x, y)
    particles = {"nPart": nPart, "x": x, "y": y, "z": z}
    particles = DFAfunC.initializeBondsC(particles, triangles)
    #    print(particles['bondStart'])
    #    print(particles['bondDist'])
    #    print(particles['bondPart'])

    # now remove one particle (here particle 1):
    keepParticle = np.array([1.0, 0.0, 1.0])
    nRemove = 1
    nBondRemove = DFAfunC.countRemovedBonds(particles, keepParticle, nRemove)
    #    print(nBondRemove)
    assert nBondRemove == 4
    particles = DFAfunC.removedBonds(particles, keepParticle, nRemove, nBondRemove)
    #    print(particles['bondStart'])
    #    print(particles['bondDist'])
    #    print(particles['bondPart'])
    bondStart = particles["bondStart"]
    bondDist = particles["bondDist"]
    bondPart = particles["bondPart"]
    assert np.array_equal(bondStart, np.asarray([0, 1, 1, 2]))
    for k in range(nPart):
        # loop on all bonded particles
        neighbors = list()
        for ib in range(bondStart[k], bondStart[k + 1]):
            l = bondPart[ib]
            neighbors.append(l)

        neighbors.sort()
        if k == 0:
            assert neighbors == [2]
        if k == 1:
            assert neighbors == []
        if k == 2:
            assert neighbors == [0]
    bondDist.sort()
    assert np.array_equal(bondDist, np.asarray([1, 1]))


def test_computeCohesionForceC():
    cfg = configparser.ConfigParser()
    cfg["GENERAL"] = {
        "cohesiveSurfaceTension": "50000",
        "cohesionMaxStrain": "0.2",
        "minDistCohesion": "1.0e-3",
    }
    nPart = 3
    x = np.array([0.0, 1.0, 0.0])
    y = np.array([0.0, 0.0, 1.0])
    z = np.array([0.0, 0.0, 0.0])
    ux = np.array([0.0, 0.0, 0.0])
    uy = np.array([0.0, 0.0, 0.0])
    uz = np.array([0.0, 0.0, 0.0])
    m = np.array([1.0, 1.0, 1.0])
    h = np.array([1.0, 1.0, 1.0])
    force = {}
    force["forceSPHX"] = np.zeros(np.shape(x))
    force["forceSPHY"] = np.zeros(np.shape(x))
    force["forceSPHZ"] = np.zeros(np.shape(x))
    # original triangulation
    triangles = tri.Triangulation(x, y)
    particles = {
        "nPart": nPart,
        "x": x,
        "y": y,
        "z": z,
        "ux": ux,
        "uy": uy,
        "uz": uz,
        "m": m,
        "h": h,
        "dt": 0.05,
    }
    particles = DFAfunC.initializeBondsC(particles, triangles)
    # bond breaking
    particles["x"][1] = 1.21
    #    print('Here')
    force, particles = DFAfunC.computeCohesionForceC(cfg["GENERAL"], particles, force)
    #    print('Here')
    #    print(particles['bondStart'])
    #    print(particles['bondDist'])
    #    print(particles['bondPart'])
    #    print(force['forceSPHX'])
    #    print(force['forceSPHY'])
    #    print(force['forceSPHZ'])
    bondStart = particles["bondStart"]
    bondDist = particles["bondDist"]
    bondPart = particles["bondPart"]
    for k in range(nPart):
        # loop on all bonded particles
        neighbors = list()
        for ib in range(bondStart[k], bondStart[k + 1]):
            l = bondPart[ib]
            neighbors.append(l)
            if k == 0:
                if l == 1:
                    assert bondDist[ib] == -1
            if k == 0:
                if l == 1:
                    assert bondDist[ib] == -1

        neighbors.sort()
        if k == 0:
            assert neighbors == [1, 2]
        if k == 1:
            assert neighbors == [0, 2]
        if k == 2:
            assert neighbors == [0, 1]
    bondDist.sort()
    assert np.array_equal(bondDist, np.asarray([-1, -1, 1, 1, math.sqrt(2), math.sqrt(2)]))
    assert force["forceSPHX"][0] == 0
    assert force["forceSPHY"][0] == 0
    assert force["forceSPHX"][1] < 0
    assert force["forceSPHY"][1] > 0
    assert force["forceSPHX"][2] > 0
    assert force["forceSPHY"][2] < 0
    assert force["forceSPHX"][1] == -force["forceSPHX"][2]
    assert force["forceSPHY"][1] == -force["forceSPHY"][2]

    # bound not breaking
    particles["x"][1] = 1
    particles = {
        "nPart": nPart,
        "x": x,
        "y": y,
        "z": z,
        "ux": ux,
        "uy": uy,
        "uz": uz,
        "m": m,
        "h": h,
        "dt": 0.05,
    }
    particles = DFAfunC.initializeBondsC(particles, triangles)
    particles["x"][1] = 1.09
    force, particles = DFAfunC.computeCohesionForceC(cfg["GENERAL"], particles, force)
    #    print(particles['bondStart'])
    #    print(particles['bondDist'])
    #    print(particles['bondPart'])
    #    print(force['forceSPHX'])
    #    print(force['forceSPHY'])
    #    print(force['forceSPHZ'])
    bondStart = particles["bondStart"]
    bondDist = particles["bondDist"]
    bondPart = particles["bondPart"]
    for k in range(nPart):
        # loop on all bonded particles
        neighbors = list()
        for ib in range(bondStart[k], bondStart[k + 1]):
            l = bondPart[ib]
            neighbors.append(l)

        neighbors.sort()
        if k == 0:
            assert neighbors == [1, 2]
        if k == 1:
            assert neighbors == [0, 2]
        if k == 2:
            assert neighbors == [0, 1]
    bondDist.sort()
    assert np.array_equal(bondDist, np.asarray([1, 1, 1, 1, math.sqrt(2), math.sqrt(2)]))

    assert force["forceSPHX"][0] > 0
    assert force["forceSPHY"][0] == 0
    assert force["forceSPHX"][1] < 0
    assert force["forceSPHY"][1] > 0
    assert force["forceSPHX"][2] > 0
    assert force["forceSPHY"][2] < 0
    assert force["forceSPHY"][1] == -force["forceSPHY"][2]

def test_computeForceC():
    cfg = configparser.ConfigParser()
    cfg["GENERAL"] = {
        "alpha1EtaObrienAndJulien": "0.650",
        "beta1EtaObrienAndJulien": "16.81",
        "alpha2TauyObrienAndJulien": "0.00886",
        "beta2TauyObrienAndJulien": "13.11",
        "alphaObrienAndJulien": "0.01",
        "rhoSediment": "2650.0",
        "sizeSediment": "0.002",
        "cvMaxSediment": "0.615",
        "cvSediment": "0.5",
        "rho": "2000.0",
        "rhoEnt": "100.0",
        "gravAcc": "9.81",
        "velMagMin": "1.0e-6",
        "depMin": "1.0e-6",
        "interpOption": "2",
        "explicitFriction": "0",
        "reprojMethodForce": "2",
        "reprojectionIterations": "5",
        "thresholdProjection": "0.001",
        "subgridMixingFactor": "100.00",
        "viscOption": "0", # 0: no viscosity -> ux as defined below
        "enthRef": "20000",
        "musamosat": "0.155",
        "tau0samosat": "0",
        "Rs0samosat": "0.222",
        "kappasamosat": "0.43",
        "Bsamosat": "4.13",
        "Rsamosat": "0.05",
        "musamosatsmall": "0.22",
        "tau0samosatsmall": "0",
        "Rs0samosatsmall": "0.222",
        "kappasamosatsmall": "0.43",
        "Bsamosatsmall": "4.13",
        "Rsamosatsmall": "0.05",
        "musamosatmedium": "0.17",
        "tau0samosatmedium": "0",
        "Rs0samosatmedium": "0.222",
        "kappasamosatmedium": "0.43",
        "Bsamosatmedium": "4.13",
        "Rsamosatmedium": "0.05",
        "entEroEnergy": "5000",
        "entShearResistance": "0",
        "entDefResistance": "0",
        "xsivoellmy": "4000.",
        "muvoellmy": "0.155",
        "xsivoellmyminshear": "4000.",
        "muvoellmyminshear": "0.155",
        "tau0voellmyminshear": "70",
        "mucoulomb": "0.155",
        "mucoulombminshear": "0.155",
        "tau0coulombminshear": "70",
        "alpha1EtaHerschelAndBulkley": "0.650",
        "beta1EtaHerschelAndBulkley": "16.81",
        "alpha2TauyHerschelAndBulkley": "0.00886",
        "beta2TauyHerschelAndBulkley": "13.11",
        "nHerschelAndBulkley": "1.5",
        "alpha1EtaBingham": "0.650",
        "beta1EtaBingham": "16.81",
        "alpha2TauyBingham": "0.00886",
        "beta2TauyBingham": "13.11",
        "curvAccInFriction": "1",
        "curvAccInTangent": "0",
        "curvAccInGradient": "0",
        "mu0wetsnow": "0.155",
        "xsiwetsnow": "4000.",
    }

    # set particles-parameters; dictionary at t - see 'com1DFA.DFAfunctionsCython.pyx.computeForceC()'
    # Minimal particles setup
    particles = {
            "nPart": 1,
            "x": np.array([0.0]),
            "y": np.array([0.0]),
            "z": np.array([0.0]),
            "ux": np.array([1.0]),
            "uy": np.array([0.0]),
            "uz": np.array([0.0]),
            "m": np.array([1.0]),
            "h": np.array([1.0]),
            "dt": 0.1,
            "ID": np.array([1]),
            "totalEnthalpy": np.array([0.0]),
            "indXDEM": np.array([0], dtype=np.int32),
            "indYDEM": np.array([0], dtype=np.int32),
    }

    # set dem-parameters; dictionary - see 'com1DFA.DFAfunctionsCython.pyx.computeForceC()'
    # Minimal dem setup
    dem = {
            "header": {
                "cellsize": 5,
                "nrows": 2,
                "ncols": 2,
            },
            "rasterData": np.ones((2, 2)),
            # normal vectors of surface
            "Nx": np.zeros((2, 2)),
            "Ny": np.zeros((2, 2)),
            "Nz": np.ones((2, 2)),
            "areaRaster": np.ones((2, 2)), # Area of grid cells
            "outOfDEM": np.zeros((4,), dtype=bool), # no data mask (used to find out of dem particles)
    }
    #dem["headerNeighbourGrid"] = dem["header"]

    # set fields parameters; dictionary - see 'com1DFA.DFAfunctionsCython.pyx.computeForceC()'
    # Minimal fields setup
    nrows = dem["header"]["nrows"]
    ncols = dem["header"]["ncols"]

    fields = {
            "Vx": np.zeros((nrows, ncols)),
            "Vy": np.zeros((nrows, ncols)),
            "Vz": np.zeros((nrows, ncols)),
            "entrMassRaster": np.zeros((nrows, ncols)),
            "entrEnthRaster": np.zeros((nrows, ncols)),
            "muField": np.zeros((nrows, ncols)),
            "xiField": np.zeros((nrows, ncols)),
            "detRaster": np.zeros((nrows, ncols)),
            "cResRaster": np.zeros((nrows, ncols)),
    }


    # initialize test array friction force
    test_tauArray = np.zeros(3) # saving tau of the three models
    test_forceFrict = np.zeros(particles['nPart'])

    # resistanceType
    resistanceType = 1 # default; since cResRaster = 0 there is no resistance

    # apply function
    # looping over models
    forceFrictArray = np.zeros(3)
    # i = index of result in array, j = frictType
    for i, j in enumerate(range(10,13)):
        particles, force,*_ = DFAfunC.computeForceC(cfg["GENERAL"], particles, fields, dem, j, resistanceType)
        forceFrictArray[i] = force['forceFrict'][0]
    
    ## Check output types and some expected values
    assert "forceX" in force
    assert "forceY" in force
    assert "forceZ" in force
    assert "forceFrict" in force
    assert isinstance(force["forceX"], np.ndarray)
    assert isinstance(force["forceFrict"], np.ndarray)

    # Mass should not change (no entrainment/detrainment)
    assert np.isclose(particles["m"][0], 1.0)
    # Output velocities should be unchanged (no artificial viscosity, no mass change)
    assert np.isclose(particles["ux"][0], 1.0)
    assert np.isclose(particles["uy"][0], 0.0)
    assert np.isclose(particles["uz"][0], 0.0)

    ## Testing rheologcial models for given input parameters
    # substitution of shear rate gamma
    test_uMag = DFAtlsC.norm(particles['ux'][0], particles['uy'][0], particles['uz'][0])
    test_shearRate = 3 *  test_uMag / particles['h']

    ## O'Brien & Julien
    # read input parameters
    test_alpha1EtaObrienAndJulien = float(cfg['GENERAL']['alpha1EtaObrienAndJulien'])
    test_beta1EtaObrienAndJulien = float(cfg['GENERAL']['beta1EtaObrienAndJulien'])
    test_alpha2TauyObrienAndJulien = float(cfg['GENERAL']['alpha2TauyObrienAndJulien'])
    test_beta2TauyObrienAndJulien = float(cfg['GENERAL']['beta2TauyObrienAndJulien'])
    test_cvSediment = float(cfg['GENERAL']['cvSediment'])
    test_cvMaxSediment = float(cfg['GENERAL']['cvMaxSediment'])
    test_rho = float(cfg['GENERAL']['rho'])
    test_alphaObrienAndJulien = float(cfg['GENERAL']['alphaObrienAndJulien'])
    test_rhoSediment = float(cfg['GENERAL']['rhoSediment'])
    test_sizeSediment = float(cfg['GENERAL']['sizeSediment'])
    # compute tau
    test_etaObrienAndJulien = test_alpha1EtaObrienAndJulien * math.exp(test_beta1EtaObrienAndJulien * test_cvSediment)
    test_tauyObrienAndJulien = test_alpha2TauyObrienAndJulien * math.exp(test_beta2TauyObrienAndJulien * test_cvSediment)
    test_lmObrienAndJulien = 0.4 * particles['h']
    test_lambdaBagnold = 1.0 / (math.pow(test_cvMaxSediment/test_cvSediment, 1.0 / 3.0) - 1.0)
    test_cObrienAndJulien = test_rho * test_lmObrienAndJulien * test_lmObrienAndJulien + test_alphaObrienAndJulien * test_rhoSediment * test_lambdaBagnold * test_lambdaBagnold * test_sizeSediment * test_sizeSediment  
    test_tauObrienAndJulien = test_tauyObrienAndJulien + test_etaObrienAndJulien * test_shearRate + test_cObrienAndJulien * (test_shearRate * test_shearRate)
    test_tauArray[0] = test_tauObrienAndJulien[0]

    ## Herschel and Bulkley
    # read input parameters
    test_alpha1EtaHerschelAndBulkley = float(cfg['GENERAL']['alpha1EtaHerschelAndBulkley'])
    test_beta1EtaHerschelAndBulkley = float(cfg['GENERAL']['beta1EtaHerschelAndBulkley'])
    test_alpha2TauyHerschelAndBulkley = float(cfg['GENERAL']['alpha2TauyHerschelAndBulkley'])
    test_beta2TauyHerschelAndBulkley = float(cfg['GENERAL']['beta2TauyHerschelAndBulkley'])
    test_nHerschelAndBulkley = float(cfg['GENERAL']['nHerschelAndBulkley'])
    # compute tau
    test_etaHerschelandBulkley = test_alpha1EtaHerschelAndBulkley * math.exp(test_beta1EtaHerschelAndBulkley * test_cvSediment)
    test_tauyHerschelandBulkley = test_alpha2TauyHerschelAndBulkley * math.exp(test_beta2TauyHerschelAndBulkley * test_cvSediment)
    test_tauHerschelAndBulkley = test_tauyHerschelandBulkley + test_etaHerschelandBulkley * math.pow(test_shearRate[0], test_nHerschelAndBulkley)
    test_tauArray[1] = test_tauHerschelAndBulkley

    ## Bingham
    # read input parameters
    test_alpha1EtaBingham = float(cfg['GENERAL']['alpha1EtaBingham'])
    test_beta1EtaBingham = float(cfg['GENERAL']['beta1EtaBingham'])
    test_alpha2TauyBingham = float(cfg['GENERAL']['alpha2TauyBingham'])
    test_beta2TauyBingham = float(cfg['GENERAL']['beta2TauyBingham'])
    # compute tau
    test_etaBingham = test_alpha1EtaBingham * math.exp(test_beta1EtaBingham * test_cvSediment)
    test_tauyBingham = test_alpha2TauyBingham * math.exp(test_beta2TauyBingham * test_cvSediment)
    test_tauBingham = test_tauyBingham + test_etaBingham * test_shearRate
    test_tauArray[2] = test_tauBingham[0]

    ## compute friction force
    # deduce area - see 'com1DFA.DFAfunctionsCython.pyx.computeForceC()'
    test_areaPart = particles['m'] / (test_rho * particles['h'])
    # adding bottom shear resistance contribution - see 'com1DFA.DFAfunctionsCython.pyx.computeForceC()'
    test_forceBotTang = - test_areaPart * test_tauArray
    # explicit friction = 0
    test_velMagMin = float(cfg['GENERAL']['velMagMin'])
    if test_uMag < test_velMagMin:
        test_uMagRes = test_velMagMin
    else:
        test_uMagRes = test_uMag
    test_forceFrictArray = test_forceFrict - test_forceBotTang/test_uMagRes

    assert (forceFrictArray == test_forceFrictArray).all()

def test_updateInitialVelocity():
    particles = {
        "x": np.array([10.0]),
        "y": np.array([10.0]),
        "ux": np.array([0.0]),
        "uy": np.array([0.0]),
        "uz": np.array([0.0]),
        "nPart": 1,
        "velocityMag": np.array([0.0]),
    }

    # incline plane
    dem = {
        "header": {"nrows": 5, "ncols": 5, "cellsize": 5},
        "rasterData": np.array(
            [
                [100, 100, 100, 100, 100],
                [95, 95, 95, 95, 95],
                [90, 90, 90, 90, 90],
                [85, 85, 85, 85, 85],
                [80, 80, 80, 80, 80],
            ]
        ),
    }

    cfg = configparser.ConfigParser()
    cfg["GENERAL"] = {"interpOption": "2"}

    velocityMag = 10.0
    dem = geoTrans.getNormalMesh(dem, num=1)

    particlesTest = {
        "x": np.array([10.0]),
        "y": np.array([10.0]),
        "ux": np.array([0.0]),
        "uy": np.array([np.sqrt(50)]),
        "uz": np.array([-np.sqrt(50)]),
        "nPart": 1,
        "velocityMag": np.array([velocityMag]),
    }

    particlesVelocity = DFAfunC.updateInitialVelocity(cfg["GENERAL"], particles, dem, velocityMag)
    for key in particlesTest:
        assert np.isclose(particlesTest[key], particlesVelocity[key], atol=1e-4)
    assert (
        DFAtls.norm(particlesVelocity["ux"], particlesVelocity["uy"], particlesVelocity["uz"]) == velocityMag
    )

    # flat plane
    dem["rasterData"] = np.ones((dem["header"]["ncols"], dem["header"]["nrows"]))
    dem = geoTrans.getNormalMesh(dem, num=1)
    particlesTest = {
        "x": np.array([10.0]),
        "y": np.array([10.0]),
        "ux": np.array([0.0]),
        "uy": np.array([0.0]),
        "uz": np.array([0.0]),
        "nPart": 1,
        "velocityMag": np.array([0.0]),
    }

    particlesVelocity = DFAfunC.updateInitialVelocity(cfg["GENERAL"], particles, dem, velocityMag)
    for key in particlesTest:
        assert np.isclose(particlesTest[key], particlesVelocity[key], atol=1e-3)


"""
TODO: When calling pytest, the following function raises an error ("Fatal Python error: Aborted")
(see issue #1002?)

def test_updateFieldsC():
    cfg = configparser.ConfigParser()
    cfg['GENERAL'] = {'rho': '200.',
                      'interpOption': '2'}
    header = {}
    header['nrows'] = 5
    header['ncols'] = 5
    header['cellsize'] = 1
    dem = {}
    dem['header'] = header
    dem['areaRaster'] = 25 * np.ones((header['nrows'], header['ncols']))

    particles = {}

    particles['m'] = np.array([100., 100., 100., 100.])
    particles['dmDet'] = np.array([1., 1., 1., 1.])
    particles['x'] = np.array([2., 1., 2., 1.])
    particles['y'] = np.array([2., 2., 1., 1.])
    particles['ux'] = np.array([10., 10., 10., 10.])
    particles['uy'] = np.array([10., 10., 10., 10.])
    particles['uz'] = np.array([10., 10., 10., 10.])
    particles['trajectoryAngle'] = np.array([10., 10., 10., 10.])

    fields = {}
    fields['computeTA'] = False
    fields['computeKE'] = False
    fields['computeP'] = False
    fields['pfv'] = np.zeros((1, 1))
    fields['ppr'] = np.zeros((1, 1))
    fields['pft'] = np.zeros((1, 1))
    fields['pta'] = np.zeros((1, 1))
    fields['pke'] = np.zeros((header['nrows'], header['ncols']))
    fields['dmDet'] = np.zeros((header['nrows'], header['ncols']))
    fields['dmDet'][[1, 2, 2], [0, 1, 2]] = 3
#    print(fields['dmDet'])
    dmDet_calculated = np.copy(fields['dmDet'])
    dmDet_calculated[[2, 1, 2, 1], [2, 2, 1, 1]] += 1
    particles, fields = DFAfunC.updateFieldsC(cfg['GENERAL'], particles, dem, fields)
#    print(fields['dmDet'])
#    print(dmDet_calculated)
    atol = 1e-10
    assert np.allclose(fields['dmDet'], dmDet_calculated, atol=atol)
    
    particles['x'] = np.array([2.5, 1.5, 2.5, 1.5])
    particles['y'] = np.array([2.5, 2.5, 1.5, 1.5])
    fields['dmDet'] = np.zeros((header['nrows'], header['ncols']))
    dmDet_calculated2 = np.array([[0, 0, 0, 0, 0],
                                  [0, 0.25, 0.5, 0.25, 0],
                                  [0, 0.5, 1, 0.5, 0],
                                  [0, 0.25, 0.5, 0.25, 0],
                                  [0, 0, 0, 0, 0]])
    particles, fields = DFAfunC.updateFieldsC(cfg['GENERAL'], particles, dem, fields)
    atol = 1e-10
    assert np.allclose(fields['dmDet'], dmDet_calculated2, atol=atol)
#    print(fields['dmDet'])
    """
