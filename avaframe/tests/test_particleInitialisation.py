"""
    Pytest for particle initialisation functions
"""

#  Load modules
import numpy as np
import logging
import pytest
import configparser
import pathlib
import os
import copy
import pickle
import pandas as pd
import shutil


from avaframe.com1DFA import particleInitialisation as pI
import avaframe.in2Trans.rasterUtils as IOf
from avaframe.in3Utils import cfgUtils
import avaframe.com1DFA.com1DFA as com1DFA
import avaframe.com1DFA.damCom1DFA as damCom1DFA


def test_resetMassPerParticle():
    """test recomputing mass per particle"""

    # setup required input data
    cfg = configparser.ConfigParser()
    cfg["GENERAL"] = {"rho": "100."}

    areaRaster = np.ones((5, 6))
    dem = {"header": {"ncols": 6, "nrows": 5}, "areaRaster": areaRaster}
    partInCell = [7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8]
    indPartInCell = [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        8,
        16,
        16,
        16,
        16,
        16,
        16,
        16,
        16,
        16,
        16,
        16,
        16,
        16,
        16,
        16,
    ]

    particles = {"partInCell": partInCell, "nPart": 16, "indPartInCell": indPartInCell}
    relRaster = np.zeros((5, 6))
    relRaster[2, 2] = 2.0
    relRaster[2, 3] = 0.5
    relThField = ""

    # call function to be tested
    particles = pI.resetMassPerParticle(cfg, particles, dem, relRaster, relThField)

    #    print('particles new mass', particles['mIni'])
    testMass = np.ones(16)
    testMass[0:8] = 25.0
    testMass[8:] = 6.25

    assert len(particles["mIni"]) == 16
    assert np.sum(particles["mIni"]) == 250.0
    assert np.array_equal(particles["mIni"], testMass)


def test_createReleaseBuffer():
    """test generating a buffered release line"""

    # setup required input
    cfg = configparser.ConfigParser()
    cfg["GENERAL"] = {"sphKernelRadius": "2.", "bufferZoneFactor": "1."}

    inputSimLines = {
        "releaseLine": {
            "Length": np.asarray([5.0, 6.0]),
            "Start": np.asarray([0, 5.0]),
            "x": np.asarray([0.0, 1.0, 1.0, 0.0, 0.0, 10.0, 12.0, 12.0, 11.0, 10.0, 10.0]),
            "y": np.asarray([0.0, 0.0, 1.0, 1.0, 0.0, 10.0, 10.0, 12.0, 13.0, 12.0, 10.0]),
        }
    }

    # call function to be tested
    inputSimLines = pI.createReleaseBuffer(cfg, inputSimLines)

    releaseL = inputSimLines["releaseLineBuffer"]
    xL1 = releaseL["x"][releaseL["Start"][0]: releaseL["Start"][0] + releaseL["Length"][0]]
    xL2 = releaseL["x"][releaseL["Start"][1]: releaseL["Start"][1] + releaseL["Length"][1]]
    yL1 = releaseL["y"][releaseL["Start"][0]: releaseL["Start"][0] + releaseL["Length"][0]]
    yL2 = releaseL["y"][releaseL["Start"][1]: releaseL["Start"][1] + releaseL["Length"][1]]

    #    print('xl1', xL1)
    #    print('xl2', xL2)
    #    print('yl1', yL1)
    #    print('yl2', yL2)

    assert len(np.where(xL1 > 3.0)[0]) == 0
    assert len(np.where(xL1 < -2)[0]) == 0
    assert len(np.where(xL2 > 14.0)[0]) == 0
    assert len(np.where(xL2 < 8.0)[0]) == 0
    assert len(np.where(xL1 > 3.2)[0]) == 0
    assert len(np.where(xL2 > 14.2)[0]) == 0

    assert len(np.where(xL1 > 3.2)[0]) == 0
    assert len(np.where(xL1 < -2.2)[0]) == 0
    assert len(np.where(yL1 > 3.0)[0]) == 0
    assert len(np.where(yL1 < -2.0)[0]) == 0
    assert len(np.where(yL2 < 8.0)[0]) == 0
    assert len(np.where(yL2 > 15.2)[0]) == 0


def test_getIniPosition(tmp_path):
    """test get initial position"""

    # setup required input
    testDir = pathlib.Path(__file__).parents[0]
    testD = pathlib.Path(tmp_path)
    inputDir = testDir / "data" / "testCom1DFA"
    cfgFile = inputDir / "getIniP_com1DFACfg.ini"
    cfg, modInfo = cfgUtils.getModuleConfig(com1DFA, fileOverride=cfgFile, modInfo=True)
    # add avaDir
    cfg["GENERAL"]["avalancheDir"] = "tmp_path"

    # setup dem
    nCols = 10
    nRows = 9
    demOri = {
        "header": {"ncols": nCols, "nrows": nRows, "xllcenter": -15.5, "yllcenter": -17.5, "cellsize": 5.0},
        "rasterData": np.ones((nRows, nCols)),
    }

    # setup release area info
    relRaster = np.zeros((nRows, nCols))
    relRaster[3:6, 3:7] = 2.0
    inputSimLines = {
        "releaseLine": {
            "Length": np.asarray([5]),
            "Start": np.asarray([0]),
            "x": np.asarray([7.0, 17.0, 17.0, 7.0, 7.0]),
            "Name": ["rel1"],
            "thickness": [None],
            "header": demOri["header"],
            "y": np.asarray([5.0, 5.0, 10.0, 10.0, 5.0]),
            "rasterData": relRaster,
            "type": "Release",
        }
    }

    # get buffered line
    inputSimLines = pI.createReleaseBuffer(cfg, inputSimLines)
    relRaster2 = np.zeros((nRows, nCols))
    relRaster2[2, 3:7] = 2.0
    relRaster2[3:6, 2:8] = 2.0
    relRaster2[6, 3:7] = 2.0
    inputSimLines["releaseLineBuffer"]["rasterData"] = relRaster2

    # get mesh
    dem = com1DFA.initializeMesh(cfg["GENERAL"], demOri, 1)
    damFootLinePath = testD / "dam" / "damFootLine.shp"
    damLine = damCom1DFA.initializeWallLines(cfg["GENERAL"], dem, None, damFootLinePath)
    dem["damLine"] = damLine

    # initialize particles
    particles = com1DFA.initializeParticles(
        cfg["GENERAL"],
        inputSimLines["releaseLineBuffer"],
        dem,
        inputSimLines=inputSimLines,
        logName="",
        relThField="",
    )

    # initialize fields
    particles, fields = com1DFA.initializeFields(cfg, dem, particles, "")

    # call function to be tested
    particles, fields = pI.getIniPosition(cfg, particles, dem, fields, inputSimLines, "")

    assert len(np.where((particles["x"] + particles["xllcenter"]) < 7.0)[0]) == 0
    assert len(np.where((particles["x"] + particles["xllcenter"]) > 24.5)[0]) == 0
    assert len(np.where((particles["y"] + particles["yllcenter"]) < 5.0)[0]) == 0
    assert len(np.where((particles["y"] + particles["yllcenter"]) > 10.0)[0]) == 0
    assert len(np.where(particles["m"] != 1250.0)[0]) == 0
    assert particles["nPart"] == 16
    assert particles["mTot"] == 20000.0
    assert particles["peakKinEne"] == 0.0
    assert particles["kineticEne"] == 0.0
    assert particles["potentialEne"] == np.sum(9.81 * particles["m"] * particles["z"])
    assert particles["peakKinEne"] == 0.0
    assert particles["peakForceSPH"] == 0.0
    assert particles["forceSPHIni"] == 0.0
    assert particles["peakMassFlowing"] == 0
    assert "mIni" not in particles
