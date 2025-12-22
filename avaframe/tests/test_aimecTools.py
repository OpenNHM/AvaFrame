""" Tests for module ana3AIMEC aimecTools """

import pandas as pd
import numpy as np
import pathlib
import configparser
import pytest
import shutil

# Local imports
import avaframe.ana3AIMEC.aimecTools as aT
import avaframe.ana3AIMEC.ana3AIMEC as anaAI
import avaframe.in3Utils.fileHandlerUtils as fU
from avaframe.in3Utils import cfgUtils
import avaframe.in2Trans.rasterUtils as IOf


def test_fetchReferenceSimNo(tmp_path):
    """test fetchReferenceSimNo"""

    # setup required input
    avaDir = pathlib.Path(tmp_path, "testDir")
    testPath = avaDir / "Outputs" / "comModule" / "peakFiles"
    fU.makeADir(testPath)
    test1PPR = testPath / "testSim_no1_ppr.asc"
    test1PFT = testPath / "testSim_no1_pft.asc"
    test1PFV = testPath / "testSim_no1_pfv.asc"
    test2PPR = testPath / "testSim_no2_ppr.asc"
    test2PFT = testPath / "testSim_no2_pft.asc"
    test2PFV = testPath / "testSim_no2_pfv.asc"
    d = {
        "simName": ["testSim_no1", "testSim_no2"],
        "ppr": [test1PPR, test2PPR],
        "pft": [test1PFT, test2PFT],
        "pfv": [test1PFV, test2PFV],
    }
    inputsDF = pd.DataFrame(data=d, index=["testSim_no1", "testSim_no2"])
    cfgSetup = configparser.ConfigParser()
    cfgSetup["AIMECSETUP"] = {
        "resType": "pfv",
        "referenceSimName": "testSim_no2",
        "referenceSimValue": "",
        "varParList": "",
    }

    refSimHash, refSimName, inputsDF, colorParameter, valRef = aT.fetchReferenceSimNo(
        avaDir, inputsDF, "comModule", cfgSetup
    )
    assert refSimName == "testSim_no2"
    assert colorParameter is False
    assert valRef == ""

    cfgSetup["AIMECSETUP"]["referenceSimName"] = ""
    refSimHash, refSimName, inputsDF, colorParameter, valRef = aT.fetchReferenceSimNo(
        avaDir, inputsDF, "comModule", cfgSetup
    )
    assert refSimName == "testSim_no1"
    assert colorParameter is False
    assert inputsDF.loc[refSimHash, cfgSetup["AIMECSETUP"]["resType"]] == test1PFV


def test_computeCellSizeSL(tmp_path):
    """test fetchReferenceSimNo"""
    cfg = configparser.ConfigParser()
    cfg["AIMECSETUP"] = {"cellSizeSL": ""}
    cfgSetup = cfg["AIMECSETUP"]
    demHeader = {"cellsize": 1}

    # read the cell size from the header
    cellSizeSL = aT.computeCellSizeSL(cfgSetup, demHeader["cellsize"])
    assert cellSizeSL == 1

    # read the cell size from the cfg
    cfgSetup["cellSizeSL"] = "3"
    cellSizeSL = aT.computeCellSizeSL(cfgSetup, demHeader["cellsize"])
    assert cellSizeSL == 3

    # read the cell size from the cfg
    cfgSetup["cellSizeSL"] = "3.1"
    cellSizeSL = aT.computeCellSizeSL(cfgSetup, demHeader["cellsize"])
    assert cellSizeSL == 3.1

    # check error if no number provided but a character
    cfgSetup["cellSizeSL"] = "c"
    message = "cellSizeSL is read from the configuration file but should be a number, you provided: c"
    with pytest.raises(ValueError) as e:
        assert aT.computeCellSizeSL(cfgSetup, demHeader["cellsize"])
    assert str(e.value) == message


def test_addSurfaceParalleCoord(tmp_path):
    """test addSurfaceParalleCoord"""
    rasterTransfo = {"s": np.array([0, 3, 6, 9, 12]), "z": np.array([100, 96, 96, 92, 88])}
    rasterTransfo = aT.addSurfaceParalleCoord(rasterTransfo)
    tol = 1e-8
    sParaSol = np.array([0, 5, 8, 13, 18])
    testRes = np.allclose(rasterTransfo["sParallel"], sParaSol, atol=tol)
    #    print(rasterTransfo['sParallel'])
    assert testRes


def test_createReferenceDF(tmp_path):
    """test create a pandas dDF for reference data saved in pathDict"""

    pathDict = {
        "referencePoint": [pathlib.Path(tmp_path, "referenceTest_POINT.shp")],
        "referenceLine": [],
        "referencePolygon": [],
    }

    # call function to be tested
    referenceDF = aT.createReferenceDF(pathDict)

    assert len(referenceDF) == 1
    assert referenceDF["reference_name"].iloc[0] == "referenceTest_POINT"
    assert referenceDF["reference_filePath"].iloc[0] == pathlib.Path(tmp_path, "referenceTest_POINT.shp")
    assert referenceDF["dataType"].iloc[0] == "reference"
    assert "reference_resType" in referenceDF.columns.values


def test_computeRunoutPoint():
    """ " test compute the runout point diff between reference and sim"""

    # setup test
    resAnalysisDF = pd.DataFrame(
        data={
            "sRunout": [100.0, 110.0],
            "lRunout": [220, 400],
            "refSim_Diff_sRunout": [np.nan, np.nan],
            "refSim_Diff_lRunout": [np.nan, np.nan],
            "simName": ["simA", "simB"],
        },
        index=[10, 20],
    )
    refPoint = {"sRunout": 110.0, "lRunout": 200}

    # call function to be tested
    resAnalysisDF = aT.computeRunoutPointDiff(resAnalysisDF, refPoint, 10)

    assert np.isclose(resAnalysisDF["refSim_Diff_lRunout"].loc[10], -20)
    assert np.isclose(resAnalysisDF["refSim_Diff_sRunout"].loc[10], 10)
    assert np.isnan(resAnalysisDF["refSim_Diff_sRunout"].loc[20])
    assert np.isnan(resAnalysisDF["refSim_Diff_lRunout"].loc[20])


def test_addReferenceAnalysisTODF(tmp_path):
    """test adding reference analysis data to df"""

    # setup test
    refFile = pathlib.Path(tmp_path, "referenceTest_LINE.shp")
    referenceDF = pd.DataFrame(
        data={
            "reference_name": [refFile.stem],
            "reference_Type": [""],
            "reference_sRunout": [np.nan],
            "reference_lRunout": [np.nan],
            "reference_xRunout": [np.nan],
            "reference_yRunout": [np.nan],
        },
        index=[10],
    )
    refDataDict = {"sRunout": 110.0, "lRunout": 200.0, "xRunout": 100.0, "yRunout": 90}

    # call function to be tested
    referenceDF = aT.addReferenceAnalysisTODF(referenceDF, refFile, refDataDict)

    #    print('referenceDF ', referenceDF)

    for item in ["sRunout", "lRunout", "xRunout", "yRunout"]:
        assert np.isclose(referenceDF["reference_%s" % item].loc[10], refDataDict[item])


def test_analyzeDiffsRunoutLines(tmp_path):
    """test analyzing the difference in runout line computed from sim and derived from reference data set"""

    # setup inputs
    runoutLine = {
        "s": np.asarray([np.nan, np.nan, np.nan, 2, 3, 4, 5, 4, 3, np.nan, 1]),
        "l": np.asarray([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]),
    }
    refDataTransformed = {
        "refLine": {
            "s": np.asarray([np.nan, 1, 2, 3, 4, 4, 5, 4, 3, np.nan, np.nan]),
            "l": np.asarray([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]),
            "type": "line",
        }
    }
    resAnalysisDF = pd.DataFrame(
        data={
            "runoutLineDiff_line": [np.nan],
            "runoutLineDiff_line_pointsNotFoundInSim": [np.nan],
            "runoutLineDiff_line_pointsNotFoundInRef": [np.nan],
            "runoutLineDiff_line_RMSE": [np.nan],
            "simName": ["simA"],
        }
    )

    resAnalysisDF["runoutLineDiff_line"] = resAnalysisDF["runoutLineDiff_line"].astype(object)
    pathDict = {"projectName": "avaTest", "pathResult": pathlib.Path(tmp_path, "dir1")}
    cfg = configparser.ConfigParser()
    cfg["AIMECSETUP"] = {"runoutResType": "pfv", "thresholdValue": 1.0}

    # call function
    resAnalysisDF = aT.analyzeDiffsRunoutLines(
        cfg["AIMECSETUP"], runoutLine, refDataTransformed, resAnalysisDF, 0, pathDict
    )

    assert np.isnan(resAnalysisDF["runoutLineDiff_line"].loc[0][0])
    assert resAnalysisDF["runoutLineDiff_line_pointsNotFoundInSim"].loc[0] == "2/8"
    assert resAnalysisDF["runoutLineDiff_line_pointsNotFoundInRef"].loc[0] == "1/7"
    assert np.isclose(resAnalysisDF["runoutLineDiff_line_RMSE"].loc[0], (np.sqrt(2.0 / 6.0)))


def test_computeRunoutLine(tmp_path):
    """test computing the runout line from different data sets as furthest coordinate along s for each l"""

    avaName = "avaHockeyChannel"
    dirname = pathlib.Path(__file__).parents[0]
    sourceDir = dirname / ".." / "data" / avaName

    avalancheDir = tmp_path / avaName

    # Copy input to tmp dir
    shutil.copytree(sourceDir, avalancheDir)

    # setup inputs
    cfg = cfgUtils.getModuleConfig(anaAI, onlyDefault=True)
    cfgSetup = cfg["AIMECSETUP"]

    pathDict = {}
    pathDict = aT.readAIMECinputs(
        avalancheDir, pathDict, cfgSetup.getboolean("defineRunoutArea"), dirName="com1DFA"
    )
    demSource = pathDict["demSource"]
    dem = IOf.readRaster(demSource)

    rasterTransfo = aT.makeDomainTransfo(pathDict, dem, 5.0, cfgSetup)

    refRasterXY = np.zeros((dem["header"]["nrows"], dem["header"]["ncols"]))
    refRasterXY[195:200, 490] = 1.0
    refRasterXY[200:205, 491] = 1.0
    refRasterXY[205:209, 492] = 1.0
    refFile = pathlib.Path(avalancheDir, "referenceLine_LINE.shp")

    refRasterSL = aT.transform(
        {"header": dem["header"], "rasterData": refRasterXY}, refFile.stem, rasterTransfo, "bilinear"
    )
    transformedRasters = {"testRaster": refRasterSL}

    # call function to be tested
    runoutLine = aT.computeRunoutLine(
        cfgSetup, rasterTransfo, transformedRasters, "", "line", name="testRaster", basedOnMax=True
    )

    #    print('runoutLine', runoutLine)

    assert len(np.where(np.isnan(runoutLine["s"]) == False)[0]) == 14
    assert len(np.where(np.isnan(runoutLine["l"]) == False)[0]) == 14
    assert len(np.where(np.isnan(runoutLine["x"]) == False)[0]) == 14
    assert len(np.where(np.isnan(runoutLine["y"]) == False)[0]) == 14
    assert len(np.where(runoutLine["index"] == 485)[0]) == 4
    assert len(np.where(runoutLine["index"] == 484)[0]) == 5
    assert len(np.where(runoutLine["index"] == 483)[0]) == 5

    refRasterXY = np.zeros((dem["header"]["nrows"], dem["header"]["ncols"]))
    refRasterXY[200:205, 490] = 10.0
    refRasterXY[200:205, 491] = 5.0
    refRasterXY[200:205, 492] = 1.0

    refFile = pathlib.Path(avalancheDir, "referenceLine_LINE.shp")

    refRasterSL = aT.transform(
        {"header": dem["header"], "rasterData": refRasterXY}, refFile.stem, rasterTransfo, "bilinear"
    )
    transformedRasters = {"newRasterPPR": refRasterSL}

    # call function to be tested
    runoutLine = aT.computeRunoutLine(
        cfgSetup, rasterTransfo, transformedRasters, "", "line", name="", basedOnMax=False
    )

    #    print('runoutLine', runoutLine)

    assert len(np.where(np.isnan(runoutLine["s"]) == False)[0]) == 5
    assert len(np.where(np.isnan(runoutLine["l"]) == False)[0]) == 5
    assert len(np.where(np.isnan(runoutLine["x"]) == False)[0]) == 5
    assert len(np.where(np.isnan(runoutLine["y"]) == False)[0]) == 5
    assert len(np.where(runoutLine["index"] == 484)[0]) == 5

    refRasterXY = np.zeros((dem["header"]["nrows"], dem["header"]["ncols"]))
    refRasterXY[200:205, 490] = 0.4
    refRasterXY[200:205, 491] = 0.5
    refRasterXY[200:205, 492] = 0.1

    refFile = pathlib.Path(avalancheDir, "referenceLine_LINE.shp")

    refRasterSL = aT.transform(
        {"header": dem["header"], "rasterData": refRasterXY}, refFile.stem, rasterTransfo, "bilinear"
    )
    transformedRasters = {"newRasterPPR": refRasterSL}

    # call function to be tested
    runoutLine = aT.computeRunoutLine(
        cfgSetup, rasterTransfo, transformedRasters, "", "line", name="", basedOnMax=False
    )

    #    print('runoutLine', runoutLine)

    assert len(np.where(np.isnan(runoutLine["s"]) == False)[0]) == 0
    assert len(np.where(np.isnan(runoutLine["l"]) == False)[0]) == 0
    assert len(np.where(np.isnan(runoutLine["x"]) == False)[0]) == 0
    assert len(np.where(np.isnan(runoutLine["y"]) == False)[0]) == 0
    assert len(np.where(np.isnan(runoutLine["index"]) == False)[0]) == 0
    assert ("sRunout" in runoutLine.keys()) is False
    assert ("lRunout" in runoutLine.keys()) is False
    assert ("xRunout" in runoutLine.keys()) is False


def test_checkOverlapDBXY():
    """check if lines along coordinate grid do intersect"""

    # setup required input
    x1 = np.arange(0, 10, 1)
    y1 = np.arange(2, 10, 1)
    X, Y = np.meshgrid(x1, y1)
    rasterTransfo = {"gridx": X, "gridy": Y}

    flagOverlap = aT.checkOverlapDBXY(rasterTransfo)

    assert flagOverlap is True

    X[:, 0] = np.arange(0, 8, 1)
    rasterTransfo = {"gridx": X, "gridy": Y}

    with pytest.raises(AssertionError) as e:
        assert aT.checkOverlapDBXY(rasterTransfo)
    assert "New coordinate system has overlaps - first intersection at POINT (1 3)." in str(e.value)
