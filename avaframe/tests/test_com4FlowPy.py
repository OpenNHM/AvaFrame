"""
Pytest for module com4FlowPy
"""

#  Load modules
import numpy as np
import pathlib
import pytest
import pickle
import os
import rasterio
import geopandas as gpd
import configparser
import copy
import shutil

from avaframe.com4FlowPy import flowClass
import avaframe.com4FlowPy.flowCore as flowCore
import avaframe.com4FlowPy.splitAndMerge as SPAM
import avaframe.in2Trans.rasterUtils as IOf
import avaframe.runCom4FlowPy as runCom4FlowPy
from avaframe.com4FlowPy import com4FlowPy

import avaframe.runStandardTestsCom4FlowPy as runStandardTestsCom4

def test_add_os():
    cell = flowClass.Cell(
        1,
        1,
        np.array([[10, 10, 10], [10, 10, 10], [10, 10, 10]]),
        10,
        1,
        0,
        None,
        20,
        8,
        3e-4,
        270,
        startcell=True,
    )
    cell.add_os(0.2)
    assert cell.flux == 1.2


def test_reverseTopology():
    """
    testing flowCore.reverseTopology() for different
    examples of dir graphs
    """

    testGraph = {0: [1, 2, 3], 1: [2, 4], 2: [4, 5], 3: [2], 4: [], 5: [6], 6: []}

    testGraphReverse = {0: [], 1: [0], 2: [0, 1, 3], 3: [0], 4: [2, 1], 5: [2], 6: [5]}

    reverseGraphCalc = flowCore.reverseTopology(testGraph)

    for key, item in reverseGraphCalc.items():
        assert key in testGraphReverse.keys()
        setTestChildren = set(item)
        setCalcChildren = set(testGraphReverse[key])
        assert setTestChildren == setCalcChildren

    testGraph = {
        0: [1, 2, 3],
        1: [4],
        2: [5, 6],
        3: [7, 8],
        4: [9],
        5: [9, 10],
        6: [10],
        7: [11],
        8: [],
        9: [],
        10: [12],
        11: [],
        12: [],
    }

    testGraphReverse = {
        0: [],
        1: [0],
        2: [0],
        3: [0],
        4: [1],
        5: [2],
        6: [2],
        7: [3],
        8: [3],
        9: [4, 5],
        10: [5, 6],
        11: [7],
        12: [10],
    }

    reverseGraphCalc = flowCore.reverseTopology(testGraph)

    for key, item in reverseGraphCalc.items():
        assert key in testGraphReverse.keys()
        setTestChildren = set(item)
        setCalcChildren = set(testGraphReverse[key])
        assert setTestChildren == setCalcChildren


def test_backTracking():
    """
    testing flowCore.backTracking() for different
    examples of dir graphs - basic graphs are the same as
    in test_reverseTopology() with added 'infra' values
    as valueDicts
    """

    testGraph = {0: [1, 2, 3], 1: [2, 4], 2: [4, 5], 3: [2], 4: [], 5: [6], 6: []}

    testValsIn = {0: 0, 1: 0, 2: 0, 3: 0, 4: 3, 5: 0, 6: 2}

    testValsBT = {0: 3, 1: 3, 2: 3, 3: 3, 4: 3, 5: 2, 6: 2}

    calcValsBT = flowCore.backTracking(testGraph, testValsIn)

    for key, item in calcValsBT.items():
        assert calcValsBT[key] == testValsBT[key]

    testGraph = {
        0: [1, 2, 3],
        1: [4],
        2: [5, 6],
        3: [7, 8],
        4: [9],
        5: [9, 10],
        6: [10],
        7: [11],
        8: [],
        9: [],
        10: [12],
        11: [],
        12: [],
    }

    testValsIn = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 0,
        7: 0,
        8: 0,
        9: 1,
        10: 0,
        11: 3,
        12: 2,
    }

    testValsBT = {
        0: 3,
        1: 1,
        2: 2,
        3: 3,
        4: 1,
        5: 2,
        6: 2,
        7: 3,
        8: 0,
        9: 1,
        10: 2,
        11: 3,
        12: 2,
    }

    calcValsBT = flowCore.backTracking(testGraph, testValsIn)

    for key, item in calcValsBT.items():
        assert calcValsBT[key] == testValsBT[key]


def test_calculation():
    dem = np.array(
        [
            [40, 40, 40, 40, 40],
            [30, 30, 30, 30, 30],
            [20, 20, 20, 20, 20],
            [10, 10, 10, 10, 10],
            [0, 0, 0, 0, 0],
        ]
    )
    infra = None
    pra = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    alpha = 10
    exp = 99
    fluxTh = 0.001
    zDeltaMax = 8000
    nodata = -9999
    cellsize = 10
    infraBool = False
    forestBool = False
    variableParameters = {
        "varUmaxBool": False,
        "varUmaxArray": None,
        "varAlphaBool": False,
        "varAlphaArray": None,
        "varExponentBool": False,
        "varExponentArray": None,
    }
    fluxDistOldVersionBool = False
    previewMode = False
    outputs = ["travelLengthMin", "flux"]
    forestArray = None
    forestParams = None
    relOutputParams = {
        "relIdBool": False,
        "relIdArray": None,
        "relVolBool": False,
        "relVolArray": None,
    }
    rasterAttributes = {"cellsize": cellsize, "nodata": nodata}
    args = [
        dem,
        infra,
        pra,
        alpha,
        exp,
        fluxTh,
        zDeltaMax,
        rasterAttributes,
        infraBool,
        forestBool,
        variableParameters,
        fluxDistOldVersionBool,
        previewMode,
        forestArray,
        forestParams,
        outputs,
        relOutputParams,
        False,
        False,
        None,
    ]

    flux = np.ones_like(dem) * -9999.0
    flux[[1, 2, 3], [2, 2, 2]] = 1
    depFluxSum = np.zeros_like(dem)
    routFluxSum = np.where(flux == 1, 1.0, 0.0)
    travelLengthMin = np.ones_like(dem) * -9999.0
    travelLengthMin[1, 2] = 0
    travelLengthMin[2, 2] = np.sqrt(cellsize**2)
    travelLengthMin[3, 2] = 2 * np.sqrt(cellsize**2)
    results = flowCore.calculation(args)

    assert len(results) == 14
    assert np.all(results[1] == flux)
    assert np.all(results[10] == routFluxSum)
    assert np.all(results[11] == depFluxSum)
    assert np.all(results[8] == travelLengthMin)
    assert results[7] is None


def createTestRaster(pathTestFolder, rasterName):
    # first create test raster and save in test folder
    testRaster = np.zeros((10, 10))

    testRaster[2:5, 2:5] = 1
    testRaster[6:9, 5:9] = 2
    testRaster[5, 5:8] = 2
    testRaster[0, 0] = 3
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

    # write flipped raster, the read raster function does also flip the raster
    IOf.writeResultToRaster(header, testRaster, pathTestFolder / rasterName, useCompression=False, flip=True)
    del testRaster
    return header


def test_tileRaster(tmp_path):
    pathTestFolder = tmp_path / "data" / "testCom4"
    rasterName = "testRaster"
    ext = ".tif"

    tileName = "testTile"
    pathTempFolder = pathTestFolder / "tmp"
    xDim = 4
    yDim = 4
    U = 1
    if os.path.exists(pathTempFolder) is False:
        os.makedirs(pathTempFolder)

    createTestRaster(pathTestFolder, rasterName)

    SPAM.tileRaster(pathTestFolder / f"{rasterName}{ext}", tileName, pathTempFolder, xDim, yDim, U)
    mergedRaster = SPAM.mergeRaster(pathTempFolder, tileName)

    testData = IOf.readRaster(pathTestFolder / f"{rasterName}{ext}", noDataToNan=False)
    testRaster = testData["rasterData"]

    nTiles = pickle.load(open(pathTempFolder / "nTiles", "rb"))
    ext00 = pickle.load(open(pathTempFolder / "ext_0_0", "rb"))
    ext03 = pickle.load(open(pathTempFolder / "ext_0_3", "rb"))
    ext10 = pickle.load(open(pathTempFolder / "ext_1_0", "rb"))
    ext21 = pickle.load(open(pathTempFolder / "ext_2_1", "rb"))

    tile00 = np.load(pathTempFolder / "testTile_0_0.npy")
    tile03 = np.load(pathTempFolder / "testTile_0_3.npy")
    tile03Test = testRaster[0:yDim, 2 * xDim - 2 * U : 3 * xDim - 2 * U]
    tile00Test = testRaster[0:xDim, 0:yDim]

    assert np.all(testRaster == mergedRaster)
    assert nTiles == (3, 3)
    assert tile00.shape == (4, 4)
    assert np.all(tile00 == tile00Test)
    assert np.all(tile03 == tile03Test)
    assert ext00 == ((0, xDim), (0, yDim))
    assert ext03 == ((0, xDim), (2 * yDim - 2 * U, 3 * yDim - 2 * U))
    assert ext10 == ((xDim - 2 * U, 2 * xDim - 2 * U), (0, yDim))
    assert ext21 == (
        (2 * xDim - 4 * U, 3 * xDim - 4 * U),
        (yDim - 2 * U, 2 * yDim - 2 * U),
    )


def test_mergeDict(tmp_path):
    pathTestFolder = tmp_path / "data" / "testCom4"
    rasterName = "testRaster"
    ext = ".tif"
    pathRaster = pathTestFolder / (rasterName + ext)
    tileName = "testTile"
    pathTempFolder = pathTestFolder / "tmp"
    xDim = 4
    yDim = 4
    U = 1
    if os.path.exists(pathTempFolder) is False:
        os.makedirs(pathTempFolder)

    createTestRaster(pathTestFolder, rasterName)

    dictName = "testDict"

    SPAM.tileRaster(pathRaster, tileName, pathTempFolder, xDim, yDim, U)
    nTiles = pickle.load(open(pathTempFolder / "nTiles", "rb"))

    for i in range(nTiles[0] + 1):
        for j in range(nTiles[1] + 1):
            tile = np.load(pathTempFolder / f"{tileName}_{i}_{j}.npy")
            rows, cols = np.where(tile > 0)
            dictSmallTile = {(r, c): tile[r, c] for r, c in zip(rows, cols)}
            saveDict = open(pathTempFolder / ("%s_%s_%s.pickle" % (dictName, i, j)), "wb")
            pickle.dump(dictSmallTile, saveDict)
            saveDict.close()

    mergedDict = SPAM.mergeDict(pathTempFolder, dictName)

    mergedDictRef = {
        (0, 0): 3,
        (2, 2): 1,
        (2, 3): 1,
        (2, 4): 1,
        (3, 2): 1,
        (3, 3): 1,
        (3, 4): 1,
        (4, 2): 1,
        (4, 3): 1,
        (4, 4): 1,
        (6, 5): 2,
        (6, 6): 2,
        (6, 7): 2,
        (6, 8): 2,
        (7, 5): 2,
        (7, 6): 2,
        (7, 7): 2,
        (7, 8): 2,
        (8, 5): 2,
        (8, 6): 2,
        (8, 7): 2,
        (8, 8): 2,
        (5, 5): 2,
        (5, 6): 2,
        (5, 7): 2,
    }

    assert mergedDict.keys() == mergedDictRef.keys()
    for k in mergedDict:
        assert np.all(mergedDict[k] == mergedDictRef[k])

    # manipulate dictionaries that they overlap

    for i in range(nTiles[0] + 1):
        for j in range(nTiles[1] + 1):
            tile = np.load(pathTempFolder / f"{tileName}_{i}_{j}.npy")
            rows, cols = np.where(tile > 0)
            dictSmallTile = {(r, c): tile[r, c] for r, c in zip(rows, cols)}
            if i == 2 and j == 1:
                dictSmallTile[(1, 2)] = [1, 2]
                dictSmallTile[(1, 1)] = [1, 2]
            saveDict = open(pathTempFolder / ("%s_%s_%s.pickle" % (dictName, i, j)), "wb")
            pickle.dump(dictSmallTile, saveDict)
            saveDict.close()

    mergedDict = SPAM.mergeDict(pathTempFolder, dictName)

    mergedDictRef = {
        (5, 4): [1, 2],
        (5, 3): [1, 2],
        (0, 0): 3,
        (2, 2): 1,
        (2, 3): 1,
        (2, 4): 1,
        (3, 2): 1,
        (3, 3): 1,
        (3, 4): 1,
        (4, 2): 1,
        (4, 3): 1,
        (4, 4): 1,
        (6, 5): 2,
        (6, 6): 2,
        (6, 7): 2,
        (6, 8): 2,
        (7, 5): 2,
        (7, 6): 2,
        (7, 7): 2,
        (7, 8): 2,
        (8, 5): 2,
        (8, 6): 2,
        (8, 7): 2,
        (8, 8): 2,
        (5, 5): 2,
        (5, 6): 2,
        (5, 7): 2,
    }
    assert mergedDict.keys() == mergedDictRef.keys()
    for k in mergedDict:
        assert np.all(mergedDict[k] == mergedDictRef[k])


def test_mergeDictToRaster(tmp_path):
    pathTestFolder = tmp_path / "data" / "testCom4"
    rasterName = "testRaster"
    ext = ".tif"
    pathRaster = pathTestFolder / (rasterName + ext)
    tileName = "testTile"
    pathTempFolder = pathTestFolder / "tmp"
    xDim = 4
    yDim = 4
    U = 1
    if os.path.exists(pathTempFolder) is False:
        os.makedirs(pathTempFolder)

    createTestRaster(pathTestFolder, rasterName)

    dictName = "testDict"

    SPAM.tileRaster(pathRaster, tileName, pathTempFolder, xDim, yDim, U)
    nTiles = pickle.load(open(pathTempFolder / "nTiles", "rb"))

    for i in range(nTiles[0] + 1):
        for j in range(nTiles[1] + 1):
            tile = np.load(pathTempFolder / f"{tileName}_{i}_{j}.npy")
            rows, cols = np.where(tile > 0)
            dictSmallTile = {(r, c): tile[r, c] for r, c in zip(rows, cols)}
            saveDict = open(pathTempFolder / ("%s_%s_%s.pickle" % (dictName, i, j)), "wb")
            pickle.dump(dictSmallTile, saveDict)
            saveDict.close()

    mergedRaster = SPAM.mergeDictToRaster(pathTempFolder, dictName)

    testData = IOf.readRaster(pathTestFolder / "testRaster.tif")
    testRaster = testData["rasterData"]
    # due to no overlap:
    testRaster[testRaster > 0] = 1
    assert np.all(mergedRaster == testRaster)

    # manipulate dictionaries that they overlap

    for i in range(nTiles[0] + 1):
        for j in range(nTiles[1] + 1):
            tile = np.load(pathTempFolder / f"{tileName}_{i}_{j}.npy")
            rows, cols = np.where(tile > 0)
            dictSmallTile = {(r, c): tile[r, c] for r, c in zip(rows, cols)}
            if i == 2 and j == 1:
                dictSmallTile[(1, 2)] = [1, 2]
                dictSmallTile[(1, 1)] = [1, 2]
            saveDict = open(pathTempFolder / ("%s_%s_%s.pickle" % (dictName, i, j)), "wb")
            pickle.dump(dictSmallTile, saveDict)
            saveDict.close()

    mergedRaster = SPAM.mergeDictToRaster(pathTempFolder, dictName)
    testRaster[5, 4] = 2
    testRaster[5, 3] = 2

    assert np.all(mergedRaster == testRaster)


def test_mergeDictToPolygon(tmp_path):
    _testDIR = pathlib.Path(__file__).parent
    pathRefData = _testDIR / "data" / "testCom4"

    pathTestFolder = tmp_path / "data" / "testCom4"
    rasterName = "testRaster"
    ext = ".tif"
    pathRaster = pathTestFolder / (rasterName + ext)
    tileName = "testTile"
    pathTempFolder = pathTestFolder / "tmp"
    xDim = 4
    yDim = 4
    U = 1
    if os.path.exists(pathTempFolder) is False:
        os.makedirs(pathTempFolder)

    rasterHeader = createTestRaster(pathTestFolder, rasterName)

    dictName = "testDict"

    SPAM.tileRaster(pathRaster, tileName, pathTempFolder, xDim, yDim, U)
    nTiles = pickle.load(open(pathTempFolder / "nTiles", "rb"))

    for i in range(nTiles[0] + 1):
        for j in range(nTiles[1] + 1):
            tile = np.load(pathTempFolder / f"{tileName}_{i}_{j}.npy")
            rows, cols = np.where(tile > 0)
            dictSmallTile = {(r, c): tile[r, c] for r, c in zip(rows, cols)}
            saveDict = open(pathTempFolder / ("%s_%s_%s.pickle" % (dictName, i, j)), "wb")
            pickle.dump(dictSmallTile, saveDict)
            saveDict.close()

    gdfPathPolygons = SPAM.mergeDictToPolygon(pathTempFolder, dictName, rasterHeader)

    refPolygons = gpd.read_file(pathRefData / "refPolygon.geojson")

    assert len(gdfPathPolygons) == 3
    assert np.all(gdfPathPolygons["PRA_id"] == refPolygons["PRA_id"])
    assert np.all(gdfPathPolygons.geometry.geom_equals(refPolygons.geometry))

    # manipulate dictionaries that the polygons overlap

    for i in range(nTiles[0] + 1):
        for j in range(nTiles[1] + 1):
            tile = np.load(pathTempFolder / f"{tileName}_{i}_{j}.npy")
            rows, cols = np.where(tile > 0)
            dictSmallTile = {(r, c): tile[r, c] for r, c in zip(rows, cols)}
            if i == 2 and j == 1:
                dictSmallTile[(1, 2)] = [1, 2]
                dictSmallTile[(1, 1)] = [1, 2]
            saveDict = open(pathTempFolder / ("%s_%s_%s.pickle" % (dictName, i, j)), "wb")
            pickle.dump(dictSmallTile, saveDict)
            saveDict.close()

    gdfPathPolygons = SPAM.mergeDictToPolygon(pathTempFolder, dictName, rasterHeader)
    refPolygons = gpd.read_file(pathRefData / "refPolygon_manipulated.geojson")

    assert len(gdfPathPolygons) == 3
    assert np.all(gdfPathPolygons["PRA_id"] == refPolygons["PRA_id"])
    assert np.all(gdfPathPolygons.geometry.geom_equals(refPolygons.geometry))


def test_runCom4FlowPy(tmp_path):

    _avaframeDir = pathlib.Path(__file__).parents[1]
    avaFlowPyDir = str(_avaframeDir / "data" / "avaFlowPy" / "Inputs")
    avaTestDir = pathlib.Path(tmp_path, "avaFlowPyTest")
    avaTestDirInput = avaTestDir / "Inputs"
    shutil.copytree(avaFlowPyDir, avaTestDirInput)
    avalancheDir = str(avaTestDir)

    cfg = configparser.ConfigParser()
    cfg["GENERAL"] = {
        "infra": "False",
        "variableUmaxLim": "False",
        "variableAlpha": "False",
        "variableExponent": "False",
        "forest": "False",
        "alpha": "40",
        "exp": "8",
        "flux_threshold": "3.0e-4",
        "max_z": "200",
        "previewMode": "False",
        "fluxDistOldVersion": "False",
        "procPerCPUCore": "1",
        "chunkSize": "50",
        "maxChunks": "500",
        "cpuCount": "1",
        "tileSize": "15000",
        "tileOverlap": "5000",
        "thalwegreleasearea": "False",
        "calcThalweg": "False",
    }
    cfg["PATHS"] = {
        "outputFiles": "zDelta",
        "useCustomPaths": "False",
        "useCustomPathDEM": "False",
        "outputFileFormat": ".tif",
        "overwriteResults": "default",
    }

    resDictTest1 = {
        "simulationPerformed": True,
        "resultOverwritten": False,
        "message": "simulation completed, no previous results found."
    }
    resDict = runCom4FlowPy.main(avalancheDir=avalancheDir, cfg=copy.deepcopy(cfg))

    for key in resDictTest1:
        assert resDictTest1[key] == resDict[key]

    # second run
    resDictTest2 = {
        "uid": resDict["uid"],
        "simulationPerformed": False,
        "resultOverwritten": False,
        "message": "Simulation not performed! - results for exact same configuration already exist.",
    }
    resDict = runCom4FlowPy.main(avalancheDir=avalancheDir, cfg=copy.deepcopy(cfg))

    for key in resDictTest2:
        assert resDictTest2[key] == resDict[key]

    # search for one result file and remove it
    resFolder = (
        pathlib.Path(avalancheDir) / "Outputs" / "com4FlowPy" / "peakFiles" / "res_{}".format(resDict["uid"])
    )
    fileNames = os.listdir(resFolder)
    os.remove(resFolder / fileNames[0])
    # second run
    resDictTest21 = {
        "uid": resDict["uid"],
        "simulationPerformed": True,
        "resultOverwritten": True,
        "message": f"simulation completed, Leftover files from aborted run with same {resDict['uid']} overwritten.",
    }

    resDict = runCom4FlowPy.main(avalancheDir=avalancheDir, cfg=copy.deepcopy(cfg))

    for key in resDictTest21:
        assert resDictTest21[key] == resDict[key]

    # third run with changing cfg:
    cfg["PATHS"]["overwriteResults"] = "reRunAndOverwrite"
    resDict = runCom4FlowPy.main(avalancheDir=avalancheDir, cfg=copy.deepcopy(cfg))

    for key in resDictTest1:
        assert resDictTest1[key] == resDict[key]

    # fourth run with overwriting results
    resDictTest4 = {
        "uid": resDict["uid"],
        "simulationPerformed": True,
        "resultOverwritten": True,
        "message": "simulation completed, existing results overwritten.",
    }
    resDict = runCom4FlowPy.main(avalancheDir=avalancheDir, cfg=copy.deepcopy(cfg))

    for key in resDictTest4:
        assert resDictTest4[key] == resDict[key]

    # fifth run with changing cfg:
    cfg["PATHS"]["overwriteResults"] = "reRunAndBackup"
    resDict = runCom4FlowPy.main(avalancheDir=avalancheDir, cfg=copy.deepcopy(cfg))

    for key in resDictTest1:
        assert resDictTest1[key] == resDict[key]

    # sixth run with backuping results
    resDictTest6 = {
        "uid": resDict["uid"],
        "simulationPerformed": True,
        "resultOverwritten": True,
        "message": "simulation completed, existing results backed up.",
    }
    resDict = runCom4FlowPy.main(avalancheDir=avalancheDir, cfg=copy.deepcopy(cfg))

    for key in resDictTest6:
        assert resDictTest6[key] == resDict[key]

    # make same for custom paths
    _wDir = avalancheDir + "/Outputs/com4FlowPy"
    _demPath = avalancheDir + "/Inputs/dem.tif"
    _releasePath = avalancheDir + "/Inputs/REL/rel.shp"

    cfg["PATHS"] = {
        "outputFiles": "zDelta",
        "useCustomPaths": "True",
        "useCustomPathDEM": "False",
        "outputFileFormat": ".tif",
        "overwriteResults": "default",
        "workDir": _wDir,
        "demPath": _demPath,
        "releasePath": _releasePath,
        "relIdPath": "",
        "infraPath": "",
        "forestPath": "",
        "varUmaxPath": "",
        "varAlphaPath": "",
        "varExponentPath": "",
        "deleteTempFolder": "True",
        "outputNoDataValue": "-9999",
        "useCompression": "False",
    }

    resDictTest1 = {
        "simulationPerformed": True,
        "resultOverwritten": False,
        "message": "simulation completed, no previous results found.",
    }
    resDict = runCom4FlowPy.main(cfg=copy.deepcopy(cfg))

    for key in resDictTest1:
        assert resDictTest1[key] == resDict[key]

    # second run
    resDictTest2 = {
        "uid": resDict["uid"],
        "simulationPerformed": False,
        "resultOverwritten": False,
        "message": "Simulation not performed! - results for exact same configuration already exist.",
    }
    resDict = runCom4FlowPy.main(cfg=copy.deepcopy(cfg))

    for key in resDictTest2:
        assert resDictTest2[key] == resDict[key]

    # search for one result file and remove it
    resFolder = pathlib.Path(avalancheDir) / "Outputs" / "com4FlowPy" / "res_{}".format(resDict["uid"])
    fileNames = os.listdir(resFolder)
    os.remove(resFolder / fileNames[0])
    # second run
    resDictTest21 = {
        "uid": resDict["uid"],
        "simulationPerformed": True,
        "resultOverwritten": True,
        "message": f"simulation completed, Leftover files from aborted run with same {resDict['uid']} overwritten.",
    }

    resDict = runCom4FlowPy.main(avalancheDir=avalancheDir, cfg=copy.deepcopy(cfg))

    for key in resDictTest21:
        assert resDictTest21[key] == resDict[key]

    # second run with changing cfg:
    cfg["PATHS"]["overwriteResults"] = "reRunAndOverwrite"
    resDict = runCom4FlowPy.main(cfg=copy.deepcopy(cfg))

    for key in resDictTest1:
        assert resDictTest1[key] == resDict[key]

    # fourth run with overwriting results
    resDictTest4 = {
        "uid": resDict["uid"],
        "simulationPerformed": True,
        "resultOverwritten": True,
        "message": "simulation completed, existing results overwritten.",
    }
    resDict = runCom4FlowPy.main(cfg=copy.deepcopy(cfg))

    for key in resDictTest4:
        assert resDictTest4[key] == resDict[key]

    # fifth run with changing cfg:
    cfg["PATHS"]["overwriteResults"] = "reRunAndBackup"
    resDict = runCom4FlowPy.main(avalancheDir=avalancheDir, cfg=copy.deepcopy(cfg))

    for key in resDictTest1:
        assert resDictTest1[key] == resDict[key]

    # sixth run with backuping results
    resDictTest6 = {
        "uid": resDict["uid"],
        "simulationPerformed": True,
        "resultOverwritten": True,
        "message": "simulation completed, existing results backed up.",
    }
    resDict = runCom4FlowPy.main(avalancheDir=avalancheDir, cfg=copy.deepcopy(cfg))

    for key in resDictTest6:
        assert resDictTest6[key] == resDict[key]
def test_getMaskedRasters():
    raster = np.array(
        [
            [1, 2, 3],
            [4, 5, 0],
            [0, 7, 8],
        ]
    )
    mask = np.array(
        [
            [1, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ]
    )

    idsIn, idsOut = SPAM.getMaskedRasters(mask, raster)

    # inside mask: (0,0)=1, (0,1)=2, (1,0)=4, (2,2)=8
    assert list(idsIn) == [1, 2, 4, 8]
    # outside mask: (0,2)=3, (1,1)=5, (1,2)=0, (2,0)=0, (2,1)=7 -> zeros/negatives dropped
    assert list(idsOut) == [3, 5, 7]

    raster = np.array([[1, 2], [3, 4]])
    mask = np.ones_like(raster)

    idsIn, idsOut = SPAM.getMaskedRasters(mask, raster)

    assert list(idsIn) == [1, 2, 3, 4]
    assert list(idsOut) == []


def test_getTileEnds(tmp_path):
    # same raster/geometry as used implicitly by test_tileRaster, so the
    # expected exList/eyList can be cross-checked against the ext_i_j values
    # asserted there.
    pathTestFolder = tmp_path / "data" / "testCom4"
    rasterName = "testRaster"
    pathTempFolder = pathTestFolder / "tmp"
    os.makedirs(pathTempFolder, exist_ok=True)

    createTestRaster(pathTestFolder, rasterName)
    testData = IOf.readRaster(pathTestFolder / f"{rasterName}.tif", noDataToNan=False)
    relIdRaster = testData["rasterData"]

    xDim, yDim, U = 4, 4, 1

    exList, eyList = SPAM.getTileEnds(pathTempFolder, xDim, yDim, U, relIdRaster)

    assert exList == [6, 10]
    assert eyList == [6, 10]

    nTiles = pickle.load(open(pathTempFolder / "nTiles", "rb"))
    assert nTiles == (1, 1)

    # relId over entire raster

    relIdRaster = np.ones((10, 10))
    exList, eyList = SPAM.getTileEnds(pathTempFolder, xDim, yDim, U, relIdRaster)

    assert exList == [11]
    assert eyList == [11]
    nTiles = pickle.load(open(pathTempFolder / "nTiles", "rb"))
    assert nTiles == (0, 0)

    extentLarge = pickle.load(open(pathTempFolder / "extentLarge", "rb"))
    assert extentLarge == (10, 10)

    pathTempFolder = tmp_path / "tmp"
    os.makedirs(pathTempFolder, exist_ok=True)

    relIdRaster = np.zeros((10, 10))
    # release area (ID 5) straddles the naive x-tile boundary (column 3/4)
    relIdRaster[2:5, 2:6] = 5

    xDim, yDim, U = 4, 4, 1

    exList, eyList = SPAM.getTileEnds(pathTempFolder, xDim, yDim, U, relIdRaster)

    assert exList == [7, 9, 11]
    assert eyList == [6, 8, 10]

    relIdRaster = np.zeros((10, 10))
    # release area fully inside what would become the first naive tile
    relIdRaster[0:2, 0:2] = 7

    xDim, yDim, U = 4, 4, 1

    exList, eyList = SPAM.getTileEnds(pathTempFolder, xDim, yDim, U, relIdRaster)

    assert exList == [4, 6, 8, 10]
    assert eyList == [4, 6, 8, 10]


def test_tileRasterWithIndices(tmp_path):
    pathTestFolder = tmp_path / "data" / "testCom4"
    rasterName = "testRaster"
    ext = ".tif"
    pathTempFolder = pathTestFolder / "tmp"
    os.makedirs(pathTempFolder, exist_ok=True)

    createTestRaster(pathTestFolder, rasterName)

    fNameIn = pathTestFolder / f"{rasterName}{ext}"
    fNameOut = "testTile"
    U = 1
    # same end-indices SPAM.getTileEnds would produce for this raster/config
    exList = [4, 6, 8, 10]
    eyList = [4, 6, 8, 10]

    SPAM.tileRasterWithIndices(fNameIn, fNameOut, pathTempFolder, exList, eyList, U, isInit=False)

    testData = IOf.readRaster(fNameIn, noDataToNan=False)
    testRaster = testData["rasterData"]

    nTiles = pickle.load(open(pathTempFolder / "nTiles", "rb"))
    assert nTiles == (3, 3)

    # corner tile (0,0): rows 0:4, cols 0:4
    ext00 = pickle.load(open(pathTempFolder / "ext_0_0", "rb"))
    assert ext00 == ((0, 4), (0, 4))
    tile00 = np.load(pathTempFolder / "testTile_0_0.npy")
    assert tile00.shape == (4, 4)
    assert np.all(tile00 == testRaster[0:4, 0:4])

    # interior tile (1,2): rows 2:6, cols 4:8
    ext12 = pickle.load(open(pathTempFolder / "ext_1_2", "rb"))
    assert ext12 == ((2, 6), (4, 8))
    tile12 = np.load(pathTempFolder / "testTile_1_2.npy")
    assert np.all(tile12 == testRaster[2:6, 4:8])

    # bottom-right corner tile (3,3): rows 6:10, cols 6:10
    ext33 = pickle.load(open(pathTempFolder / "ext_3_3", "rb"))
    assert ext33 == ((6, 10), (6, 10))
    tile33 = np.load(pathTempFolder / "testTile_3_3.npy")
    assert np.all(tile33 == testRaster[6:10, 6:10])

    # with tiling  init
    pathTestFolder = tmp_path / "data" / "testCom4"
    rasterName = "testRaster"
    ext = ".tif"
    pathTempFolder = pathTestFolder / "tmp"
    os.makedirs(pathTempFolder, exist_ok=True)

    createTestRaster(pathTestFolder, rasterName)

    fNameIn = pathTestFolder / f"{rasterName}{ext}"
    fNameOut = "testTileInit"
    U = 1
    exList = [4, 6, 8, 10]
    eyList = [4, 6, 8, 10]

    SPAM.tileRasterWithIndices(fNameIn, fNameOut, pathTempFolder, exList, eyList, U, isInit=True)

    testData = IOf.readRaster(fNameIn, noDataToNan=False)
    testRaster = testData["rasterData"]
    # test raster only contains values >= 0, so any -9999 found below must
    # come from the edge-nulling logic, not from the source data
    assert not np.any(testRaster == -9999)

    # --- corner tile (0,0): east edge nulled (j != JMAX) and south edge
    #     nulled (i != IMAX); north/west untouched (i == 0, j == 0)
    tile00 = np.load(pathTempFolder / "testTileInit_0_0.npy")
    assert np.all(tile00[:, -U:] == -9999)  # east
    assert np.all(tile00[-U:, :] == -9999)  # south
    assert np.all(tile00[0, :-U] == testRaster[0, 0:4][:-U])  # north untouched
    assert np.all(tile00[:-U, 0] == testRaster[0:4, 0][:-U])  # west untouched

    # --- interior tile (1,2): all four edges nulled
    tile12 = np.load(pathTempFolder / "testTileInit_1_2.npy")
    assert np.all(tile12[:, -U:] == -9999)  # east
    assert np.all(tile12[0:U, :] == -9999)  # north
    assert np.all(tile12[:, 0:U] == -9999)  # west
    assert np.all(tile12[-U:, :] == -9999)  # south

    # --- bottom-right corner tile (3,3): north edge nulled (i != 0) and
    #     west edge nulled (j != 0); east/south untouched (j == JMAX, i == IMAX)
    tile33 = np.load(pathTempFolder / "testTileInit_3_3.npy")
    assert np.all(tile33[0:U, :] == -9999)  # north
    assert np.all(tile33[:, 0:U] == -9999)  # west
    assert np.all(tile33[-1, U:] == testRaster[9, 6:10][U:])  # south untouched
    assert np.all(tile33[U:, -1] == testRaster[6:10, 9][U:])  # east untouched

    # workflow: first get tiles, then make tiling
    createTestRaster(pathTestFolder, rasterName)
    testData = IOf.readRaster(pathTestFolder / f"{rasterName}.tif", noDataToNan=False)
    relIdRaster = testData["rasterData"]

    xDim, yDim, U = 4, 4, 1
    fNameIn = pathTestFolder / f"{rasterName}.tif"
    fNameOut = "testTile"

    exList, eyList = SPAM.getTileEnds(pathTempFolder, xDim, yDim, U, relIdRaster)
    SPAM.tileRasterWithIndices(fNameIn, fNameOut, pathTempFolder, exList, eyList, U, isInit=True)

    tile00 = np.load(pathTempFolder / "testTile_0_0.npy")
    tile01 = np.load(pathTempFolder / "testTile_0_1.npy")
    tile10 = np.load(pathTempFolder / "testTile_1_0.npy")
    tile11 = np.load(pathTempFolder / "testTile_1_1.npy")

    ids00 = np.unique(tile00)
    ids01 = np.unique(tile01)
    ids10 = np.unique(tile10)
    ids11 = np.unique(tile11)

    for id in ids00:
        if id <= 0:
            continue
        assert id not in ids01
        assert id not in ids10
        assert id not in ids11

    for id in ids10:
        if id <= 0:
            continue
        assert id not in ids01
        assert id not in ids00
        assert id not in ids11

    for id in ids01:
        if id <= 0:
            continue
        assert id not in ids00
        assert id not in ids10
        assert id not in ids11

def testCompareRasters(monkeypatch):
    """Test the comparison of two raster arrays.

    The test replaces ``rasterUtils.readRaster`` with a local fake
    implementation. This keeps the test self-contained and avoids reading
    actual raster files from disk.

    The test verifies that:

    * the cell-by-cell difference is calculated correctly;
    * non-identical rasters are reported as unequal;
    * the percentage of closely matching processed cells is correct;
    * both rasters are read with ``noDataToNan=False``;
    * the raster files are read in the expected order.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture used to temporarily replace
        ``rasterUtils.readRaster`` in the module under test.
    """
    rasterPath = pathlib.Path("raster.tif")
    referenceRasterPath = pathlib.Path("referenceRaster.tif")

    rasterData = np.array(
        [
            [1.0, 2.0, 0.0],
            [4.0, np.nan, 6.0],
        ]
    )
    referenceRasterData = np.array(
        [
            [1.0, 2.0001, 0.0],
            [5.0, np.nan, 6.0],
        ]
    )

    rasterDataByPath = {
        rasterPath: rasterData,
        referenceRasterPath: referenceRasterData,
    }

    readRasterCalls = []

    def _fakeReadRaster(requestedPath, noDataToNan):
        """Return predefined raster data for the requested path.

        Parameters
        ----------
        requestedPath : pathlib.Path
            Path of the raster requested by ``compareRasters``.
        noDataToNan : bool
            Value passed to the ``noDataToNan`` argument.

        Returns
        -------
        dict
            Dictionary containing the predefined NumPy array under the
            ``"rasterData"`` key.
        """
        readRasterCalls.append((requestedPath, noDataToNan))

        return {
            "rasterData": rasterDataByPath[requestedPath],
        }

    # Patch readRaster where compareRasters looks it up. The patch is
    # automatically removed by pytest after the test has completed.
    monkeypatch.setattr(
        runStandardTestsCom4.rasterUtils,
        "readRaster",
        _fakeReadRaster,
    )

    difference, areEqual, closePercentage = (
        runStandardTestsCom4.compareRasters(
            rasterPath,
            referenceRasterPath,
        )
    )

    expectedDifference = referenceRasterData - rasterData

    np.testing.assert_allclose(
        difference,
        expectedDifference,
        equal_nan=True,
    )

    assert areEqual is False

    # Four cells are selected by the positive-value mask:
    #
    #   1.0 versus 1.0       -> close
    #   2.0 versus 2.0001    -> close
    #   4.0 versus 5.0       -> not close
    #   6.0 versus 6.0       -> close
    #
    # The 0.0 pair and NaN pair are not selected by the mask.
    # Therefore, three out of four processed cells are close.
    assert closePercentage == pytest.approx(3 / 4)

    assert readRasterCalls == [
        (rasterPath, False),
        (referenceRasterPath, False),
    ]

def test_validateInputArray():
    praMask = (np.array([[1, 0], [1, 0]]) == 1)
    dataValid = np.array([[255, 0], [128, -1]])
    dataNan = np.array([[255.0, 0], [np.nan, -1]])
    dataHigh = np.array([[255.0, 0], [10000.0, -1]])

    assert com4FlowPy.validateInputArray(praMask, dataValid, maxValue=300)
    assert not com4FlowPy.validateInputArray(praMask, dataNan, maxValue=300.0)
    assert not com4FlowPy.validateInputArray(praMask, dataHigh, maxValue=300.0)


def makeValidParamRanges(minAlphaValid=0.0, maxAlphaValid=90.0, maxZDeltaValid=20000.0,
                         maxExpValid=600.0, minFluxThreshValid=1e-6, maxFluxThreshValid=1.0,
                         maxUMaxValid=100.0):
    """ build a valid parameter-ranges dict manually, with sensible defaults,
    overridable per test via explicit keyword arguments.
    """
    return {
        "minAlphaValid": minAlphaValid,
        "maxAlphaValid": maxAlphaValid,
        "maxZDeltaValid": maxZDeltaValid,
        "maxExpValid": maxExpValid,
        "minFluxThreshValid": minFluxThreshValid,
        "maxFluxThreshValid": maxFluxThreshValid,
        "maxUMaxValid": maxUMaxValid,
    }


def test_getValidParameterRanges():
    """ test that getValidParameterRanges falls back to its documented defaults
    when the .ini config section provides no overrides
    """
    cfgSetup = configparser.ConfigParser()
    cfgSetup["GENERAL"] = {}

    result = com4FlowPy.getValidParameterRanges(cfgSetup["GENERAL"])

    assert result["maxZDeltaValid"] == pytest.approx(20000.0)
    assert result["maxExpValid"] == pytest.approx(600.0)
    assert result["minFluxThreshValid"] == pytest.approx(1e-6)
    assert result["maxFluxThreshValid"] == pytest.approx(1.0)

    cfgSetup = configparser.ConfigParser()
    cfgSetup["GENERAL"] = {
        "maxZDeltaValid": "15000",
        "maxExpValid": "450",
        "minFluxThreshValid": "1e-5",
        "maxFluxThreshValid": "0.5",
    }

    result = com4FlowPy.getValidParameterRanges(cfgSetup["GENERAL"])

    assert result["maxZDeltaValid"] == pytest.approx(15000.0)
    assert result["maxExpValid"] == pytest.approx(450.0)
    assert result["minFluxThreshValid"] == pytest.approx(1e-5)
    assert result["maxFluxThreshValid"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def makeValidModelParameters(alpha=25.0, max_z=8000.0, exp=8, flux_threshold=0.003):
    """ build a set of global model parameters that pass all checkGlobalParameters checks """
    return {
        "alpha": alpha,
        "max_z": max_z,
        "exp": exp,
        "flux_threshold": flux_threshold,
    }


def makeModelPaths():
    return {
        "releasePathWork": "dummy_release.tif",
        "varAlphaPath": "dummy_alpha.tif",
        "varUmaxPath": "dummy_umax.tif",
        "varExponentPath": "dummy_exp.tif",
    }


# ---------------------------------------------------------------------------
# checkGlobalParameters
# ---------------------------------------------------------------------------

def test_checkGlobalParameters_alpha():
    """ test that an alpha value below minAlphaValid raises ValueError """
    modelParameters = makeValidModelParameters(alpha=-5.0)
    validParamRanges = makeValidParamRanges()

    with pytest.raises(ValueError, match="alpha"):
        com4FlowPy.checkGlobalParameters(modelParameters, makeModelPaths(), validParamRanges)

    """ test that an alpha value above maxAlphaValid raises ValueError """
    modelParameters = makeValidModelParameters(alpha=95.0)

    with pytest.raises(ValueError, match="alpha"):
        com4FlowPy.checkGlobalParameters(modelParameters, makeModelPaths(), validParamRanges)

    """ test that an alpha value  raises no ValueError """
    modelParameters = makeValidModelParameters(alpha=45.0)
    com4FlowPy.checkGlobalParameters(modelParameters, makeModelPaths(), validParamRanges)


def test_checkGlobalParameters_maxZ():
    """ test that max_z values outside [0, maxZDeltaValid] raise ValueError """
    modelParameters = makeValidModelParameters(max_z=-5)
    validParamRanges = makeValidParamRanges()

    with pytest.raises(ValueError, match="max_z"):
        com4FlowPy.checkGlobalParameters(modelParameters, makeModelPaths(), validParamRanges)

    modelParameters = makeValidModelParameters(max_z=25000.0)

    with pytest.raises(ValueError, match="max_z"):
        com4FlowPy.checkGlobalParameters(modelParameters, makeModelPaths(), validParamRanges)

    modelParameters = makeValidModelParameters(max_z=200.0)

    com4FlowPy.checkGlobalParameters(modelParameters, makeModelPaths(), validParamRanges)


def test_checkGlobalParameters_exp():
    """ test that exp values outside [0, maxExpValid] raise ValueError """
    modelParameters = makeValidModelParameters(exp=-1)
    validParamRanges = makeValidParamRanges()

    with pytest.raises(ValueError, match="exp"):
        com4FlowPy.checkGlobalParameters(modelParameters, makeModelPaths(), validParamRanges)

    modelParameters = makeValidModelParameters(exp=700)

    with pytest.raises(ValueError, match="exp"):
        com4FlowPy.checkGlobalParameters(modelParameters, makeModelPaths(), validParamRanges)

    """ test that exp at exactly 0 and at maxExpValid is accepted """

    com4FlowPy.checkGlobalParameters(makeValidModelParameters(exp=8), makeModelPaths(), validParamRanges)


def test_checkGlobalParameters_fluxThreshold():
    """ test that flux_threshold values outside [minFluxThreshValid, maxFluxThreshValid] raise ValueError """
    modelParameters = makeValidModelParameters(flux_threshold=0)
    validParamRanges = makeValidParamRanges()

    with pytest.raises(ValueError, match="flux_threshold"):
        com4FlowPy.checkGlobalParameters(modelParameters, makeModelPaths(), validParamRanges)

    modelParameters = makeValidModelParameters(flux_threshold=5.0)

    with pytest.raises(ValueError, match="flux_threshold"):
        com4FlowPy.checkGlobalParameters(modelParameters, makeModelPaths(), validParamRanges)

    """ test that flux_threshold at exactly the min and max valid values is accepted """

    com4FlowPy.checkGlobalParameters(
        makeValidModelParameters(flux_threshold=0.003),
        makeModelPaths(), validParamRanges
    )


def test_checkGlobalParameters_errorMessageMentionsValidRange():
    """ test that the raised error message includes the configured min/max bounds """
    validParamRanges = makeValidParamRanges(minAlphaValid=5.0, maxAlphaValid=60.0)
    modelParameters = makeValidModelParameters(alpha=100.0)

    with pytest.raises(ValueError, match=r"5\.0 and 60\.0"):
        com4FlowPy.checkGlobalParameters(modelParameters, makeModelPaths(), validParamRanges)


# ---------------------------------------------------------------------------
# checkVariableInputParameters
# ---------------------------------------------------------------------------

def test_checkVariableInputParameters_allDisabled_doesNotRaise(monkeypatch):
    """ test that when all 'var*Bool' flags are False, no raster is validated and no error is raised """
    praRaster = np.array([[0, 1], [1, 0]])
    monkeypatch.setattr(IOf, "readRaster", lambda path: {"rasterData": praRaster})

    modelParameters = {
        "varAlphaBool": False,
        "varUmaxBool": False,
        "varExponentBool": False,
    }
    validParamRanges = makeValidParamRanges()

    com4FlowPy.checkVariableInputParameters(modelParameters, makeModelPaths(), validParamRanges)


def test_checkVariableInputParameters_varAlpha(monkeypatch):
    """ test that a valid alpha raster passes validation without raising """
    praRaster = np.array([[0, 1], [1, 0]])
    alphaRaster = np.array([[np.nan, 25.0], [30.0, np.nan]])

    def fakeReadRaster(path):
        if "alpha" in path:
            return {"rasterData": alphaRaster}
        return {"rasterData": praRaster}

    monkeypatch.setattr(IOf, "readRaster", fakeReadRaster)

    modelParameters = {
        "varAlphaBool": True,
        "varUmaxBool": False,
        "varExponentBool": False,
    }
    validParamRanges = makeValidParamRanges()

    com4FlowPy.checkVariableInputParameters(modelParameters, makeModelPaths(), validParamRanges)

    """ test that an invalid alpha raster (per validateInputArray) raises ValueError """
    alphaRaster = np.array([[np.nan, 200.0], [30.0, np.nan]])

    monkeypatch.setattr(IOf, "readRaster", fakeReadRaster)

    modelParameters = {
        "varAlphaBool": True,
        "varUmaxBool": False,
        "varExponentBool": False,
    }
    validParamRanges = makeValidParamRanges()

    with pytest.raises(ValueError, match="alpha"):
        com4FlowPy.checkVariableInputParameters(modelParameters, makeModelPaths(), validParamRanges)


def test_checkVariableInputParameters_inPRA(monkeypatch):
    """check invalid value in PRA"""
    pra2Raster = np.array([[1, 1], [1, 0]])
    alpha2Raster = np.array([[np.nan, 25.0], [30.0, np.nan]])

    def fakeReadRaster(path):
        if "alpha" in path:
            return {"rasterData": alpha2Raster}
        return {"rasterData": pra2Raster}

    monkeypatch.setattr(IOf, "readRaster", fakeReadRaster)

    modelParameters = {
        "varAlphaBool": True,
        "varUmaxBool": False,
        "varExponentBool": False,
    }
    validParamRanges = makeValidParamRanges()

    with pytest.raises(ValueError):
        com4FlowPy.checkVariableInputParameters(modelParameters, makeModelPaths(), validParamRanges)


def test_checkVariableInputParameters_varUmax_typeUmax(monkeypatch):
    """ test that when varUmaxType is 'umax', validateInputArray is called with maxUMaxValid
    as the upper bound rather than maxZDeltaValid
    """
    praRaster = np.array([[1, 0]])
    umaxRaster = np.array([[50.0, 50.0]])

    monkeypatch.setattr(IOf, "readRaster", lambda path: {"rasterData": (
        praRaster if "release" in path else umaxRaster
    )})

    modelParameters = {
        "varAlphaBool": False,
        "varUmaxBool": True,
        "varUmaxType": "umax",
        "varExponentBool": False,
    }
    validParamRanges = makeValidParamRanges(maxUMaxValid=77.0, maxZDeltaValid=999.0)

    com4FlowPy.checkVariableInputParameters(modelParameters, makeModelPaths(), validParamRanges)

    """ test that an invalid u-max/zdelta raster raises ValueError mentioning its type """
    praRaster = np.array([[1]])
    umaxRaster = np.array([[999999.0]])

    monkeypatch.setattr(IOf, "readRaster", lambda path: {"rasterData": (
        praRaster if "release" in path else umaxRaster
    )})

    modelParameters = {
        "varAlphaBool": False,
        "varUmaxBool": True,
        "varUmaxType": "umax",
        "varExponentBool": False,
    }
    validParamRanges = makeValidParamRanges()

    with pytest.raises(ValueError, match="umax"):
        com4FlowPy.checkVariableInputParameters(modelParameters, makeModelPaths(), validParamRanges)

    """test np.nan value in PRA"""
    praRaster = np.array([[1, 0]])
    umaxRaster = np.array([[np.nan, 100]])

    monkeypatch.setattr(IOf, "readRaster", lambda path: {"rasterData": (
        praRaster if "release" in path else umaxRaster
    )})

    modelParameters = {
        "varAlphaBool": False,
        "varUmaxBool": True,
        "varUmaxType": "umax",
        "varExponentBool": False,
    }
    validParamRanges = makeValidParamRanges()

    with pytest.raises(ValueError, match="umax"):
        com4FlowPy.checkVariableInputParameters(modelParameters, makeModelPaths(), validParamRanges)


def test_checkVariableInputParameters_varUmax_typeZDelta_usesMaxZDeltaValid(monkeypatch):
    """ test that when varUmaxType is anything other than 'umax' (e.g. 'zdelta'),
    validateInputArray is called with maxZDeltaValid as the upper bound
    """
    praRaster = np.array([[1, 0]])
    umaxRaster = np.array([[500.0, np.nan]])

    monkeypatch.setattr(IOf, "readRaster", lambda path: {"rasterData": (
        praRaster if "release" in path else umaxRaster
    )})

    modelParameters = {
        "varAlphaBool": False,
        "varUmaxBool": True,
        "varUmaxType": "ZDelta",  # mixed case; function lowercases it internally
        "varExponentBool": False,
    }
    validParamRanges = makeValidParamRanges(maxUMaxValid=77.0, maxZDeltaValid=999.0)

    com4FlowPy.checkVariableInputParameters(modelParameters, makeModelPaths(), validParamRanges)


def test_checkVariableInputParameters_varExponent(monkeypatch):
    """ test that a valid exponent raster passes validation without raising """
    praRaster = np.array([[1, 0]])
    expRaster = np.array([[8.0, np.nan]])

    monkeypatch.setattr(IOf, "readRaster", lambda path: {"rasterData": (
        praRaster if "release" in path else expRaster
    )})

    modelParameters = {
        "varAlphaBool": False,
        "varUmaxBool": False,
        "varExponentBool": True,
    }
    validParamRanges = makeValidParamRanges()

    com4FlowPy.checkVariableInputParameters(modelParameters, makeModelPaths(), validParamRanges)

    """ test that an invalid exponent raster raises ValueError """
    praRaster = np.array([[1, 0]])
    expRaster = np.array([[900.0, np.nan]])

    monkeypatch.setattr(IOf, "readRaster", lambda path: {"rasterData": (
        praRaster if "release" in path else expRaster
    )})

    modelParameters = {
        "varAlphaBool": False,
        "varUmaxBool": False,
        "varExponentBool": True,
    }

    with pytest.raises(ValueError, match="exp"):
        com4FlowPy.checkVariableInputParameters(modelParameters, makeModelPaths(), validParamRanges)


def test_get_start_idx_sortedByAltitudeDescending():
    """ test that release pixels are returned sorted by altitude, highest first """
    dem = np.array([
        [100.0, 200.0, 150.0],
        [300.0, 50.0, 400.0],
        [10.0, 500.0, 20.0],
    ])
    release = np.array([
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ])

    row_list, col_list = flowCore.get_start_idx(dem, release)

    # release pixels are at (0,0)=100, (0,1)=200, (1,0)=300, (1,2)=400, (2,1)=500
    # sorted descending by altitude: 500, 400, 300, 200, 100
    expectedOrder = [(2, 1), (1, 2), (1, 0), (0, 1), (0, 0)]
    actualOrder = list(zip(row_list, col_list))

    assert actualOrder == expectedOrder


def test_get_start_idx_noReleasePixels():
    """ test that empty row/col lists are returned when there are no release pixels """
    dem = np.array([[100.0, 200.0], [300.0, 400.0]])
    release = np.zeros((2, 2), dtype=int)

    row_list, col_list = flowCore.get_start_idx(dem, release)

    assert len(row_list) == 0
    assert len(col_list) == 0


def test_get_start_idx_singleReleasePixel():
    """ test behavior with exactly one release pixel """
    dem = np.array([[10.0, 20.0], [30.0, 40.0]])
    release = np.array([[0, 0], [0, 1]])

    row_list, col_list = flowCore.get_start_idx(dem, release)

    assert list(row_list) == [1]
    assert list(col_list) == [1]


def test_get_start_idx_sortedByRelIdWhenThalwegRequested():
    """ test that with relIdArray and calcThalweg=True, pixels are grouped/sorted by release Id
    (descending), rather than purely by altitude
    """
    dem = np.array([
        [100.0, 200.0, 150.0],
        [300.0, 50.0, 400.0],
    ])
    release = np.array([
        [1, 1, 1],
        [1, 0, 1],
    ])
    relIdArray = np.array([
        [1, 2, 1],
        [2, 0, 3],
    ])

    row_list, col_list = flowCore.get_start_idx(dem, release, relIdArray=relIdArray, calcThalweg=True)

    # cells and their relId: (0,0)->1, (0,1)->2, (0,2)->1, (1,0)->2, (1,2)->3
    # sorted primarily by relId descending: relId 3 first, then relId 2s, then relId 1s
    resultRelIds = [relIdArray[r, c] for r, c in zip(row_list, col_list)]
    assert resultRelIds == sorted(resultRelIds, reverse=True)

    # relId 3 has exactly one cell -> must be (1,2)
    assert (row_list[0], col_list[0]) == (1, 2)

    # cells belonging to the same relId are contiguous in the output
    idxRelId1 = [i for i, rid in enumerate(resultRelIds) if rid == 1]
    assert idxRelId1 == list(range(min(idxRelId1), max(idxRelId1) + 1))
    idxRelId2 = [i for i, rid in enumerate(resultRelIds) if rid == 2]
    assert idxRelId2 == list(range(min(idxRelId2), max(idxRelId2) + 1))


def test_get_start_idx_relIdArrayProvidedButCalcThalwegFalse():
    """ test that relIdArray is ignored (falls back to altitude sort) when calcThalweg is False """
    dem = np.array([
        [100.0, 200.0],
        [300.0, 400.0],
    ])
    release = np.array([
        [1, 1],
        [1, 1],
    ])
    relIdArray = np.array([
        [5, 5],
        [1, 1],
    ])

    row_list, col_list = flowCore.get_start_idx(dem, release, relIdArray=relIdArray, calcThalweg=False)

    # falls back to pure altitude-descending sort, ignoring relIdArray
    expectedOrder = [(1, 1), (1, 0), (0, 1), (0, 0)]
    actualOrder = list(zip(row_list, col_list))
    assert actualOrder == expectedOrder


def test_split_release_evenSplit_byPixelCount():
    """ test the default (non-thalweg) branch: release cells split roughly evenly
    by cumulative pixel count into `pieces` chunks
    """
    release = np.zeros((2, 10), dtype=int)
    release[0, :] = 1  # 10 release pixels along the first row

    release_list = flowCore.split_release(release, pieces=2, relIdArray=None, calcThalweg=False)

    assert len(release_list) == 2
    for piece in release_list:
        assert piece.shape == release.shape

    # every original release pixel appears in exactly one piece
    totalReconstructed = sum(piece.sum() for piece in release_list)
    assert totalReconstructed == release.sum()

    # no overlap between pieces
    combined = np.zeros_like(release)
    for piece in release_list:
        assert np.all((combined & piece) == 0)  # no pixel assigned twice
        combined = combined | piece
    np.testing.assert_array_equal(combined, release)


def test_split_release_evenSplit_singlePiece():
    """ test that pieces=1 returns the whole release layer unchanged (in a single piece) """
    release = np.array([
        [1, 0, 1],
        [0, 1, 0],
    ])

    release_list = flowCore.split_release(release, pieces=1, relIdArray=None, calcThalweg=False)

    assert len(release_list) == 1
    np.testing.assert_array_equal(release_list[0], release)


def test_split_release_evenSplit_unevenPixelCount():
    """ test even splitting when the number of release pixels doesn't divide evenly into pieces """
    release = np.zeros((1, 7), dtype=int)
    release[0, :] = 1  # 7 release pixels, split into 3 pieces

    release_list = flowCore.split_release(release, pieces=3, relIdArray=None, calcThalweg=False)

    assert len(release_list) == 3
    totalReconstructed = sum(piece.sum() for piece in release_list)
    assert totalReconstructed == 7

    # no overlaps, full reconstruction
    combined = np.zeros_like(release)
    for piece in release_list:
        combined = combined | piece
    np.testing.assert_array_equal(combined, release)


def test_split_release_thalweg_groupsByReleaseId():
    """ test the thalweg branch: cells belonging to the same relId stay together in one chunk,
    and chunks are balanced by total cell count
    """
    release = np.ones((2, 6), dtype=int)
    # 3 release areas of sizes 6, 4, 2
    relIdArray = np.array([
        [1, 1, 1, 2, 2, 2],
        [1, 1, 1, 2, 3, 3],
    ])

    release_list = flowCore.split_release(release, pieces=2, relIdArray=relIdArray, calcThalweg=True)

    assert len(release_list) == 2

    # every cell belonging to a given relId must end up entirely within a single chunk
    for relId in np.unique(relIdArray):
        # find the mask of all cells belonging to this release Id
        cellsWithThisId = (relIdArray == relId)

        # count in how many of the output chunks these cells actually show up
        numChunksContainingId = 0
        for piece in release_list:
            pixelsInThisChunk = piece[cellsWithThisId]
            if np.any(pixelsInThisChunk > 0):
                numChunksContainingId += 1

        # a release area must not be split across multiple chunks
        assert numChunksContainingId == 1

    # every release pixel appears in exactly one piece, none lost/duplicated
    combined = np.zeros_like(release)
    for piece in release_list:
        assert np.all((combined & piece) == 0)
        combined = combined | piece
    np.testing.assert_array_equal(combined, release)


def test_split_release_thalweg_piecesClampedToUniqueIdCount():
    """ test that pieces is clamped down to the number of unique release Ids when
    there are fewer distinct release areas than requested pieces
    """
    release = np.array([
        [1, 1],
        [0, 0],
    ])
    relIdArray = np.array([
        [1, 1],
        [0, 0],
    ])
    # only 1 unique release id, but pieces=5 requested
    release_list = flowCore.split_release(release, pieces=5, relIdArray=relIdArray, calcThalweg=True)

    # should be clamped down to 1 chunk (np.minimum(pieces, len(uniqueIds)))
    assert len(release_list) == 1
    np.testing.assert_array_equal(release_list[0], release)


def test_split_release_thalweg_balancesChunkSizes():
    """ test that with several release areas of different sizes, the two most balanced
    combinations of areas end up in different chunks (greedy balancing by count)
    """
    release = np.ones((1, 15), dtype=int)
    # release areas of very different sizes: 10, 3, 2
    relIdArray = np.array([[1] * 10 + [2] * 3 + [3] * 2])

    release_list = flowCore.split_release(release, pieces=2, relIdArray=relIdArray, calcThalweg=True)

    counts = [piece.sum() for piece in release_list]
    assert sum(counts) == 15
    # the big area (10 cells) should end up alone in one chunk since combining it with
    # anything else would make that chunk more unbalanced (greedy assigns smallest chunk first)
    assert 10 in counts
    # other chunk should contain the two smaller areas (3+2=5)
    assert 5 in counts


if __name__ == "__main__":
    test_add_os()
    test_reverseTopology()
    test_backTracking()
    test_calculation()
    tmpDir = pathlib.Path(__file__).parent / "data" / "testCom4"
    test_tileRaster(tmpDir)
    test_mergeDict(tmpDir)
    test_mergeDictToRaster(tmpDir)
    test_mergeDictToPolygon(tmpDir)
    test_runCom4FlowPy(tmpDir)
    test_checkGlobalParameters_alpha()
    test_checkGlobalParameters_maxZ()
    test_checkGlobalParameters_exp()
    test_checkGlobalParameters_fluxThreshold()
    test_checkGlobalParameters_errorMessageMentionsValidRange()
    test_getMaskedRasters()
    test_getTileEnds(tmpDir)
    test_tileRasterWithIndices(tmpDir)
    test_get_start_idx_sortedByAltitudeDescending()
    test_get_start_idx_noReleasePixels()
    test_get_start_idx_singleReleasePixel()
    test_get_start_idx_sortedByRelIdWhenThalwegRequested()
    test_get_start_idx_relIdArrayProvidedButCalcThalwegFalse()
    test_split_release_evenSplit_byPixelCount()
    test_split_release_evenSplit_singlePiece()
    test_split_release_evenSplit_unevenPixelCount()
    test_split_release_thalweg_groupsByReleaseId()
    test_split_release_thalweg_piecesClampedToUniqueIdCount()
    test_split_release_thalweg_balancesChunkSizes()
