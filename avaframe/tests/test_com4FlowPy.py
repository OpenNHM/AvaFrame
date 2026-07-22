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

    testValsIn = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 1, 10: 0, 11: 3, 12: 2}

    testValsBT = {0: 3, 1: 1, 2: 2, 3: 3, 4: 1, 5: 2, 6: 2, 7: 3, 8: 0, 9: 1, 10: 2, 11: 3, 12: 2}

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
    pra = np.array([[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
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
    args = [
        dem,
        infra,
        pra,
        alpha,
        exp,
        fluxTh,
        zDeltaMax,
        nodata,
        cellsize,
        infraBool,
        forestBool,
        variableParameters,
        fluxDistOldVersionBool,
        previewMode,
        forestArray,
        forestParams,
        outputs,
        relOutputParams,
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
    assert ext21 == ((2 * xDim - 4 * U, 3 * xDim - 4 * U), (yDim - 2 * U, 2 * yDim - 2 * U))


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
    test_runCom4FlowPy()
