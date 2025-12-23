"""Tests for module ana1Tests"""
import math
import pathlib
import shutil

import numpy as np
import pytest

import avaframe.ana1Tests.analysisTools as anaTools
import avaframe.ana1Tests.energyLineTest as energyLineTest
import avaframe.ana1Tests.rotationTest as rotationTest
import avaframe.com1DFA.com1DFA as com1DFA
import avaframe.in3Utils.fileHandlerUtils as fU
from avaframe.in3Utils import cfgHandling
# Local imports
from avaframe.in3Utils import cfgUtils


# ############# Test analysis tools ##########
# ############################################


def test_ErrorNorm(capfd):
    """test L2Norm"""
    cellSize = 2
    cosAngle = 0.5
    localError = np.array([[0, 1, 2], [3, 4, 5]])
    normL2 = anaTools.L2Norm(localError, cellSize, cosAngle)
    atol = 1e-10
    assert normL2 == pytest.approx(np.sqrt(120), abs=atol)

    refArray = np.array([[2, 0, 1], [0, 1, 0]])
    errorL2, errorL2Rel, errorMax, errorMaxRel = anaTools.computeErrorAndNorm(
        localError, refArray, cellSize, cosAngle
    )

    assert errorL2 == pytest.approx(np.sqrt(120), abs=atol)
    assert errorL2Rel == pytest.approx(np.sqrt(60) / 4, abs=atol)
    assert errorMax == np.sqrt(5)
    assert errorMaxRel == pytest.approx(np.sqrt(5 / 2), abs=atol)

    errorL2, errorL2Rel, errorMax, errorMaxRel = anaTools.normL2Scal(
        refArray, localError, cellSize, cosAngle
    )
    assert errorL2 == pytest.approx(np.sqrt(49 * 8), abs=atol)
    assert errorL2Rel == pytest.approx(np.sqrt(49 / 6), abs=atol)
    assert errorMax == 5
    assert errorMaxRel == 5 / 2

    localError = {"fx": localError, "fy": localError, "fz": localError}
    refArray = {"fx": refArray, "fy": refArray, "fz": refArray}
    errorL2, errorL2Rel, errorMax, errorMaxRel = anaTools.normL2Vect(
        refArray, localError, cellSize, cosAngle
    )
    assert errorL2 == pytest.approx(np.sqrt(3 * 49 * 8), abs=atol)
    assert errorL2Rel == pytest.approx(np.sqrt(49 / 6), abs=atol)
    assert errorMax == pytest.approx(np.sqrt(3) * 5, abs=atol)
    assert errorMaxRel == pytest.approx(5 / 2, abs=atol)

    localError = np.array([[0, 1, 2], [3, 4, 5]])
    refArray = np.array([[0, 0, 0], [0, 0, 0]])
    errorL2, errorL2Rel, errorMax, errorMaxRel = anaTools.computeErrorAndNorm(
        localError, refArray, cellSize, cosAngle
    )

    assert errorL2 == pytest.approx(np.sqrt(120), abs=atol)
    assert errorL2Rel == pytest.approx(np.sqrt(120), abs=atol)
    assert errorMax == np.sqrt(5)
    assert errorMaxRel == np.sqrt(5)

    localError = np.array([[0, 0, 0], [0, 0, 0]])
    normL2 = anaTools.L2Norm(localError, cellSize, cosAngle)
    atol = 1e-6
    assert normL2 == 0


# ############# Test energy line test ########
# ############################################
def test_getIntersection(capfd):
    """test find intersection"""
    dirname = pathlib.Path(__file__).parents[0]
    energyLineTestCfgFile = dirname / ".." / "tests" / "data" / "testEnergyLine" / "energyLineTestCfg.ini"

    energyLineTestCfg, modInfo = cfgUtils.getModuleConfig(
        com1DFA, fileOverride=str(energyLineTestCfgFile), modInfo=True
    )

    mu = 1
    csz = 1
    avaProfileMass = {
        "s": np.array([0, 1, 2, 3.1, 3.6, 4.1, 4.5]),
        "z": np.array([4, 1, 0.5, -0.05, -0.30, -0.55, -0.75]),
    }  # slope 1/2
    slopeExt, sIntersection, zIntersection, coefExt = energyLineTest.getAlphaProfileIntersection(
        energyLineTestCfg, avaProfileMass, mu, csz
    )
#    print(sIntersection)
#    print(zIntersection)
#    print(slopeExt)
#    print(coefExt)
    atol = 1e-10
    assert sIntersection == pytest.approx(5, abs=atol)
    assert zIntersection == pytest.approx(-1, abs=atol)
    assert slopeExt == pytest.approx(-0.5, abs=atol)
    assert coefExt == pytest.approx(0.1111111111111, abs=atol)

    avaProfileMass = {
        "s": np.array([0, 1, 2, 3.1, 3.6, 4.1, 4.5]),
        "z": np.array([3, 1, 0.5, -0.05, -0.30, -0.55, -0.75]) + 1,
    }  # slope 1/2
    slopeExt, sIntersection, zIntersection, coefExt = energyLineTest.getAlphaProfileIntersection(
        energyLineTestCfg, avaProfileMass, mu, csz
    )
#    print(sIntersection)
#    print(zIntersection)
#    print(slopeExt)
#    print(coefExt)
    atol = 1e-10
    assert sIntersection == pytest.approx(3, abs=atol)
    assert zIntersection == pytest.approx(1, abs=atol)
    assert slopeExt == pytest.approx(-0.5, abs=atol)
    assert coefExt == pytest.approx(0.0, abs=atol)

    avaProfileMass = {
        "s": np.array([0, 1, 2, 3.1, 3.6, 4.1, 4.5]),
        "z": np.array([4, 1, 0.5, -0.05, 0.1, -0.55, -0.75]),
    }  # slope 1/2
    slopeExt, sIntersection, zIntersection, coefExt = energyLineTest.getAlphaProfileIntersection(
        energyLineTestCfg, avaProfileMass, mu, csz
    )
#    print(sIntersection)
#    print(zIntersection)
#    print(slopeExt)
#    print(coefExt)
    atol = 1e-10
    assert sIntersection == pytest.approx(10.600000000000238, abs=atol)
    assert zIntersection == pytest.approx(-6.600000000000238, abs=atol)
    assert slopeExt == pytest.approx(-0.9590163934426245, abs=atol)
    assert coefExt == pytest.approx(1.3555555555556085, abs=atol)


def test_getRunOutAngle(capfd):
    """test find intersection"""
    avaProfileMass = {"s": np.array([0, 1]), "z": np.array([0, -1])}
    runOutAngleRad, runOutAngleDeg = energyLineTest.getRunOutAngle(avaProfileMass)

    atol = 1e-10
    assert runOutAngleRad == pytest.approx(math.pi / 4, abs=atol)
    assert runOutAngleDeg == pytest.approx(45, abs=atol)

    avaProfileMass = {"s": np.array([0, 1]), "z": np.array([0, -math.sqrt(3)])}
    runOutAngleRad, runOutAngleDeg = energyLineTest.getRunOutAngle(avaProfileMass)

    atol = 1e-10
    assert runOutAngleRad == pytest.approx(math.pi / 3, abs=atol)
    assert runOutAngleDeg == pytest.approx(60, abs=atol)


def test_getEnergyInfo(capfd):
    """test find intersection"""
    dirname = pathlib.Path(__file__).parents[0]
    energyLineTestCfgFile = dirname / ".." / "tests" / "data" / "testEnergyLine" / "energyLineTestCfg.ini"

    energyLineTestCfg, modInfo = cfgUtils.getModuleConfig(
        com1DFA, fileOverride=str(energyLineTestCfgFile), modInfo=True
    )

    g = 9.81
    alphaDeg = 30
    csz = 5
    mu = np.tan(np.radians(alphaDeg))

    avaProfileMass = {
        "s": np.array([0, 2, 4, 6, 8, 10]),
        "z": np.array([10, 4, 3, 2, 1, 0]),
        "u2": 2 * g * np.array([0, 4, 3, 2, 1, 0]),
    }

    runOutAngleRad, runOutAngleDeg = energyLineTest.getRunOutAngle(avaProfileMass)
    slopeExt, sIntersection, zIntersection, coefExt = energyLineTest.getAlphaProfileIntersection(
        energyLineTestCfg, avaProfileMass, mu, csz
    )

    zEne, v2Path, sGeomL, zGeomL, errorEnergyTest = energyLineTest.getEnergyInfo(
        avaProfileMass, g, mu, sIntersection, zIntersection, runOutAngleDeg, alphaDeg
    )
#    print(zEne)
#    print(v2Path)
#    print(sGeomL)
#    print(zGeomL)
#    print(errorEnergyTest)
    atol = 1e-10
    assert errorEnergyTest["runOutZError"] == pytest.approx(-10, abs=atol)
    assert errorEnergyTest["runOutSError"] == pytest.approx(10, abs=atol)
    assert errorEnergyTest["runOutAngleError"] == pytest.approx(15, abs=atol)
    assert errorEnergyTest["rmseVelocityElevation"] == pytest.approx(2.559271214294478, abs=atol)


def test_mainEnergyLineTest(tmp_path):
    dirname = pathlib.Path(__file__).parents[0]
    energyLineTestCfgFile = dirname / ".." / "tests" / "data" / "testEnergyLine" / "energyLineTestCfg.ini"
    sourceDir = dirname / ".." / "data" / "avaSlide" / "Inputs"
    targetDir = tmp_path / "avaSlide" / "Inputs"
    avalancheDir = tmp_path / "avaSlide"

    shutil.copytree(sourceDir, targetDir)

    # Load avalanche directory from general configuration file
    cfgMain = cfgUtils.getGeneralConfig()
    cfgMain["MAIN"]["avalancheDir"] = str(avalancheDir)
    workPath = pathlib.Path(tmp_path, "Work", "energyLineTest")
    fU.makeADir(workPath)
    energyLineTestCfg = cfgUtils.getModuleConfig(energyLineTest, fileOverride=energyLineTestCfgFile)
    # ----------------
    # ++++++++++ set configurations for all the used modules and override ++++++++++++
    # get comDFA configuration and save to file
    com1DFACfg = cfgUtils.getModuleConfig(com1DFA, fileOverride='', modInfo=False, toPrint=False,
                                          onlyDefault=energyLineTestCfg['com1DFA_com1DFA_override'].getboolean('defaultConfig'))
    com1DFACfg, energyLineTestCfg = cfgHandling.applyCfgOverride(com1DFACfg, energyLineTestCfg, com1DFA,
                                                                 addModValues=False)
#    print(energyLineTestCfg['com1DFA_com1DFA_override']['dam'], com1DFACfg['GENERAL']['dam'])
    com1DFACfgFile = cfgUtils.writeCfgFile(avalancheDir, com1DFA, com1DFACfg, fileName='com1DFA_settings',
                                           filePath=workPath)
    dem, simDF, _ = com1DFA.runOrLoadCom1DFA(avalancheDir, cfgMain, runDFAModule=True, cfgFile=com1DFACfgFile)
    for simName in simDF.index:
        resultEnergyTest, savePath = energyLineTest.mainEnergyLineTest(avalancheDir, energyLineTestCfg, com1DFACfg,
                                                                       simName, dem)
#    print(resultEnergyTest)
    assert abs(resultEnergyTest["runOutSError"]) < 0.02
    assert abs(resultEnergyTest["runOutZError"]) < 0.02
    assert abs(resultEnergyTest["rmseVelocityElevation"]) < 0.02
    assert abs(resultEnergyTest["runOutAngleError"]) < 0.0003


# ############# Test rotation test ##########
# ############################################
def test_initializeRotationTestReport(tmp_path):
    """Test report initialization creates correct structure"""
    avalancheDir = tmp_path / "testAva"
    resTypeList = ["ppr", "pfv", "pft"]
    comModule = "com1DFA"
    refSimName = "testSim"
    flagMass = False

    report = rotationTest.initializeRotationTestReport(
        avalancheDir, resTypeList, comModule, refSimName, flagMass
    )

    # Check report structure
    assert "headerLine" in report
    assert report["headerLine"]["type"] == "title"
    assert report["headerLine"]["title"] == "Rotation Test for DFA Simulation"

    assert "avaName" in report
    assert report["avaName"]["type"] == "avaName"

    assert "time" in report
    assert report["time"]["type"] == "time"

    assert "Simulation Parameters" in report
    assert report["Simulation Parameters"]["type"] == "list"
    assert report["Simulation Parameters"]["DFA module"] == comModule
    assert report["Simulation Parameters"]["Reference simulation"] == refSimName

    # Check column names for result tables
    assert "Rotation test input simulations" in report
    assert "simName" in report["Rotation test input simulations"]["column names"]

    assert "Rotation test Energy line result table" in report
    assert "sDiff" in report["Rotation test Energy line result table"]["column names"]

    assert "Rotation test AIMEC result table" in report
    assert "sRunout" in report["Rotation test AIMEC result table"]["column names"]

    # Mass columns should not be present when flagMass is False
    assert "relMass" not in report["Rotation test AIMEC result table"]["column names"]


def test_initializeRotationTestReport_withMass(tmp_path):
    """Test report initialization with mass analysis enabled"""
    avalancheDir = tmp_path / "testAva"
    resTypeList = ["ppr", "pfv", "pft"]
    comModule = "com1DFA"
    refSimName = "testSim"
    flagMass = True

    report = rotationTest.initializeRotationTestReport(
        avalancheDir, resTypeList, comModule, refSimName, flagMass
    )

    # Mass columns should be present when flagMass is True
    assert "relMass" in report["Rotation test AIMEC result table"]["column names"]
    assert "finalMass" in report["Rotation test AIMEC result table"]["column names"]
    assert "entMass" in report["Rotation test AIMEC result table"]["column names"]
