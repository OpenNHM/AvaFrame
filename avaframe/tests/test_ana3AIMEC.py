""" Tests for module ana3AIMEC """

import pandas as pd
import pathlib
import shutil

# Local imports
import avaframe.in3Utils.fileHandlerUtils as fU
import avaframe.ana3AIMEC.ana3AIMEC as ana3AIMEC
import avaframe.ana3AIMEC.aimecTools as aT
import avaframe.in2Trans.rasterUtils as IOf
from avaframe.com1DFA import particleTools as paT
from avaframe.in3Utils import cfgUtils
import rasterio

import pytest
from unittest.mock import Mock, patch
import numpy as np
from avaframe.ana3AIMEC.ana3AIMEC import postProcessReference


def test_getAimecInputs(capfd):
    """test readAIMECinputs and fetchReferenceSimNo"""
    dirname = pathlib.Path(__file__).parents[0]

    avalancheDir = pathlib.Path(dirname, "..", "data", "avaParabola")
    pathDict = {}
    pathDict = aT.readAIMECinputs(avalancheDir, pathDict, True, dirName="com1DFA")
    #    print(pathDict)
    assert "avaframe/tests/../data/avaParabola/Inputs/LINES/path_aimec.shp" in str(pathDict["profileLayer"])
    assert "avaframe/tests/../data/avaParabola/Inputs/POINTS/splitPoint.shp" in str(
        pathDict["splitPointSource"]
    )
    assert "avaframe/tests/../data/avaParabola/Inputs/DEM_PF_Topo.asc" in str(pathDict["demSource"])
    assert "avaframe/tests/../data/avaParabola/Outputs/ana3AIMEC/com1DFA" in str(pathDict["pathResult"])
    assert pathDict["projectName"] == "avaParabola"
    assert pathDict["pathName"] == "path_aimec"


def test_analyzeArea(tmp_path):
    """Simple test for module analyzeArea"""
    # get input data
    dirname = pathlib.Path(__file__).parents[0]
    dataRef = dirname / "data" / "refTestAimecTopo.asc"
    dataSim = dirname / "data" / "simTestAimecTopo.asc"
    dataMass = dirname / "data" / "000000.txt"
    dataMass1 = dirname / "data" / "000001.txt"
    pathDict = {}
    pathDict["projectName"] = "NameOfAvalanche"
    d = {}
    d["simName"] = ["refTestAimecTopo", "simTestAimecTopo"]
    d["ppr"] = [dataRef, dataSim]
    d["pft"] = [dataRef, dataSim]
    d["pfv"] = [dataRef, dataSim]
    d["massBal"] = [dataMass, dataMass1]
    inputsDF = pd.DataFrame(data=d, index=[0, 1])
    pathResult = tmp_path / "data"
    pathDict["pathResult"] = pathResult
    pathDict["dirName"] = "testAIMEC"
    pathDict["refSimRowHash"] = 0
    pathDict["refSimName"] = "refTestAimecTopo"
    pathDict["compType"] = ["singleModule", "com1DFA"]
    pathDict["contCmap"] = True
    pathDict["resTypeList"] = ["ppr", "pft", "pfv"]

    cfg = cfgUtils.getModuleConfig(ana3AIMEC)
    cfgSetup = cfg["AIMECSETUP"]
    cfgSetup["resType"] = "ppr"
    cfgSetup["thresholdValue"] = "0.9"
    cfgSetup["contourLevels"] = "0.1|0.5|1"
    cfgSetup["domainWidth"] = "600"
    cfgSetup["startOfRunoutAreaAngle"] = "10"
    cfg["PLOTS"]["velocityThreshold"] = "0.0001"

    avalData = np.array(([None] * 2))
    data = IOf.readRaster(dataRef)
    avalData[0] = np.transpose(data["rasterData"])
    data = IOf.readRaster(dataSim)
    avalData[1] = np.transpose(data["rasterData"])

    newRasters = {}
    newRasters["newRasterDEM"] = np.transpose(data["rasterData"])
    rasterTransfo = {}
    rasterTransfo["s"] = np.linspace(0, 499, 500)
    rasterTransfo["l"] = np.linspace(0, 99, 100)
    gridy, gridx = np.meshgrid(rasterTransfo["l"], rasterTransfo["s"])
    rasterTransfo["x"] = rasterTransfo["s"]
    rasterTransfo["y"] = 50 * np.ones(np.shape(rasterTransfo["s"]))
    rasterTransfo["z"] = 50 * np.ones(np.shape(rasterTransfo["s"]))
    rasterTransfo["gridx"] = gridx
    rasterTransfo["gridy"] = gridy
    rasterTransfo["rasterArea"] = np.ones((500, 100))
    rasterTransfo["indStartOfRunout"] = 400
    rasterTransfo["startOfRunoutAreaAngle"] = 10
    rasterTransfo["intersectionPoints"] = np.array(np.zeros((gridx.shape[0], gridy.shape[1])), dtype=bool)
    contourDict = {}

    # Analyze data
    # postprocess reference
    timeMass = None
    inputsDF, newRasters, timeMass, contourDict = ana3AIMEC.postProcessAIMEC(
        cfg, rasterTransfo, pathDict, inputsDF, newRasters, timeMass, 0, contourDict, {}
    )

    # postprocess other simulations
    for index, inputsDFrow in inputsDF.iterrows():
        if index != pathDict["refSimRowHash"]:
            inputsDF, newRasters, timeMass, contourDict = ana3AIMEC.postProcessAIMEC(
                cfg, rasterTransfo, pathDict, inputsDF, newRasters, timeMass, index, contourDict, {}
            )
    #    print(inputsDF['sRunout'])
    #    print(inputsDF['xRunout'])
    #    print(inputsDF['yRunout'])
    assert (
        (inputsDF["sRunout"][0] == 449)
        and (inputsDF["xRunout"][1] == 419)
        and (inputsDF["yRunout"][0] == 31)
        and (inputsDF["maxpprCrossMax"][1] == 1)
    )
    #    print(inputsDF['TP'])
    #    print(inputsDF['FN'])
    #    print(inputsDF['FP'])
    #    print(inputsDF['TN'])
    #    print(inputsDF['TN']+inputsDF['FP']+inputsDF['FN']+inputsDF['TP'])
    assert (
        (inputsDF["TP"][1] == 780)
        and (inputsDF["FN"][1] == 1670)
        and (inputsDF["FP"][1] == 200)
        and (inputsDF["TN"][1] == 7350)
    )


def test_makeDomainTransfo(tmp_path):
    """Simple test for module makeDomainTransfo"""
    # Extract input file locations
    pathDict = {}
    dir = pathlib.Path(__file__).parents[0]
    dirname = dir / "data" / "testAna3Aimec"
    pathData = dirname / "data"

    refDir = dirname / "LINES"
    profileLayer = list(refDir.glob("*aimec*.shp"))
    pathDict["profileLayer"] = profileLayer[0]

    refDir = dirname / "POINTS"
    splitPointLayer = list(refDir.glob("*.shp"))
    pathDict["splitPointSource"] = splitPointLayer[0]

    demSource = list(dirname.glob("*.asc"))
    pathDict["demSource"] = demSource[0]
    demSource = pathDict["demSource"]
    dem = IOf.readRaster(demSource)

    d = {}
    d["simName"] = ["testAimec_0", "testAimec_1", "testAimec_2", "testAimec_3", "testAimec_4"]
    d["ppr"] = [
        pathData / "testAimec_0.asc",
        pathData / "testAimec_1.asc",
        pathData / "testAimec_2.asc",
        pathData / "testAimec_3.asc",
        pathData / "testAimec_4.asc",
    ]
    d["pft"] = [
        pathData / "testAimec_0.asc",
        pathData / "testAimec_1.asc",
        pathData / "testAimec_2.asc",
        pathData / "testAimec_3.asc",
        pathData / "testAimec_4.asc",
    ]
    d["pfv"] = [
        pathData / "testAimec_0.asc",
        pathData / "testAimec_1.asc",
        pathData / "testAimec_2.asc",
        pathData / "testAimec_3.asc",
        pathData / "testAimec_4.asc",
    ]
    d["massBal"] = [dirname / "000001.txt"] * 5
    inputsDF = pd.DataFrame(data=d, index=[0, 1, 2, 3, 4])
    pathDict["contCmap"] = True

    pathResult = tmp_path / "results"
    pathDict["pathResult"] = pathResult

    pathDict["projectName"] = "testAna3Aimec"
    pathName = profileLayer[0].stem
    pathDict["pathName"] = pathName
    pathDict["dirName"] = "com1DFA"
    pathDict["refSimRowHash"] = 0
    pathDict["refSimName"] = "testAimec_0"
    pathDict["compType"] = ["singleModule", "com1DFA"]
    pathDict["resTypeList"] = ["ppr", "pft", "pfv"]

    cfg = cfgUtils.getModuleConfig(ana3AIMEC)
    cfgSetup = cfg["AIMECSETUP"]
    cfgSetup["startOfRunoutAreaAngle"] = "0"
    cfgSetup["domainWidth"] = "160"
    cfgSetup["resType"] = "ppr"
    cfgSetup["thresholdValue"] = "0.9"
    cfgSetup["contourLevels"] = "0.1|0.5|1"

    refSimRowHash = pathDict["refSimRowHash"]
    refResultSource = inputsDF.loc[refSimRowHash, cfgSetup["resType"]]
    refRaster = IOf.readRaster(refResultSource)
    refHeader = refRaster["header"]
    rasterTransfo = aT.makeDomainTransfo(pathDict, dem, refHeader["cellsize"], cfgSetup)

    assert rasterTransfo["gridx"][-1, 0] == 60
    assert rasterTransfo["gridx"][-1, -1] == 220
    assert rasterTransfo["gridy"][0, 0] == 180
    assert rasterTransfo["gridy"][0, -1] == 20
    assert rasterTransfo["gridy"][-1, -1] == 258

    # transform pressure_data, thickness_data and speed_data in new raster
    newRasters = {}
    # assign pressure data
    interpMethod = cfgSetup["interpMethod"]
    newRasterDEM = aT.transform(dem, pathDict["demSource"], rasterTransfo, interpMethod)
    newRasters["newRasterDEM"] = newRasterDEM
    contourDict = {}
    cfg["PLOTS"]["velocityThreshold"] = "0.0001"

    # Analyze data
    # postprocess reference
    timeMass = None
    inputsDF, newRasters, timeMass, contourDict = ana3AIMEC.postProcessAIMEC(
        cfg, rasterTransfo, pathDict, inputsDF, newRasters, timeMass, refSimRowHash, contourDict, {}
    )

    # postprocess other simulations
    for index, inputsDFrow in inputsDF.iterrows():
        if index != pathDict["refSimRowHash"]:
            resAnalysisDF, newRasters, timeMass, contourDict = ana3AIMEC.postProcessAIMEC(
                cfg, rasterTransfo, pathDict, inputsDF, newRasters, timeMass, index, contourDict, {}
            )

    for i in range(5):
        rasterSource = inputsDF["ppr"][i]
        sourceData = IOf.readRaster(rasterSource)
        rasterdata = sourceData["rasterData"]
        error = (resAnalysisDF["TP"][i] + resAnalysisDF["FP"][i] - np.nansum(rasterdata)) / (
            np.nansum(rasterdata) * 100
        )
        assert error < 0.4
        assert np.abs(resAnalysisDF["sRunout"][i] - (240 + 10 * (i + 1))) < 5


def test_mainAIMEC(tmp_path):
    # Extract input file locations
    pathDict = {}
    dir = pathlib.Path(__file__).parents[0]
    dirname = dir / "data" / "testAna3Aimec"
    pathData = dirname / "data"

    refDir = dirname / "LINES"
    profileLayer = list(refDir.glob("*aimec*.shp"))
    pathDict["profileLayer"] = profileLayer[0]

    refDir = dirname / "POINTS"
    splitPointLayer = list(refDir.glob("*.shp"))
    pathDict["splitPointSource"] = splitPointLayer[0]

    demSource = list(dirname.glob("*.asc"))
    pathDict["demSource"] = demSource[0]
    pathDict["avalancheDir"] = dirname

    d = {}
    d["simName"] = ["testAimec_0", "testAimec_1", "testAimec_2", "testAimec_3", "testAimec_4"]
    d["ppr"] = [
        pathData / "testAimec_0.asc",
        pathData / "testAimec_1.asc",
        pathData / "testAimec_2.asc",
        pathData / "testAimec_3.asc",
        pathData / "testAimec_4.asc",
    ]
    d["pft"] = [
        pathData / "testAimec_0.asc",
        pathData / "testAimec_1.asc",
        pathData / "testAimec_2.asc",
        pathData / "testAimec_3.asc",
        pathData / "testAimec_4.asc",
    ]
    d["pfv"] = [
        pathData / "testAimec_0.asc",
        pathData / "testAimec_1.asc",
        pathData / "testAimec_2.asc",
        pathData / "testAimec_3.asc",
        pathData / "testAimec_4.asc",
    ]
    d["massBal"] = [dirname / "000001.txt"] * 5
    inputsDF = pd.DataFrame(data=d, index=[0, 1, 2, 3, 4])
    pathDict["contCmap"] = True

    pathResult = tmp_path / "results"
    pathDict["pathResult"] = pathResult
    pathDict["valRef"] = ""

    pathDict["projectName"] = "testAna3Aimec"
    pathName = profileLayer[0].stem
    pathDict["pathName"] = pathName
    pathDict["dirName"] = "com1DFA"
    pathDict["refSimRowHash"] = 0
    pathDict["refSimName"] = "testAimec_0"
    pathDict["compType"] = ["singleModule", "com1DFA"]
    pathDict["resTypeList"] = ["ppr", "pft", "pfv"]
    pathDict["colorParameter"] = False

    cfg = cfgUtils.getModuleConfig(ana3AIMEC, onlyDefault=True)
    cfgSetup = cfg["AIMECSETUP"]
    cfgSetup["startOfRunoutAreaAngle"] = "0"
    cfgSetup["domainWidth"] = "160"
    cfgSetup["resType"] = "ppr"
    cfgSetup["thresholdValue"] = "0.9"
    cfgSetup["contourLevels"] = "0.1|0.5|1"
    cfg["PLOTS"]["velocityThreshold"] = "0.0001"

    rasterTransfo, inputsDF, plotDict, newRasters = ana3AIMEC.mainAIMEC(pathDict, inputsDF, cfg)

    assert rasterTransfo["gridx"][-1, 0] == 60
    assert rasterTransfo["gridx"][-1, -1] == 220
    assert rasterTransfo["gridy"][0, 0] == 180
    assert rasterTransfo["gridy"][0, -1] == 20
    assert rasterTransfo["gridy"][-1, -1] == 258


def test_aimecTransform():
    # Extract dem
    dem = {}
    dem["header"] = {}
    dem["header"]["xllcenter"] = 0
    dem["header"]["yllcenter"] = 0

    # Create rasterTransfo
    rasterTransfo = {}
    rasterTransfo["s"] = np.arange(0, 8, 1)
    rasterTransfo["l"] = np.arange(-2, 3, 1)
    gridy, gridx = np.meshgrid(rasterTransfo["l"], rasterTransfo["s"])
    rasterTransfo["gridx"] = gridx + 0.5
    rasterTransfo["gridy"] = gridy + 3.1

    # create particle
    particle = {}
    particle["x"] = [1, 7.6]
    particle["y"] = [0.6, 5.0]

    # run aimecTransform and test the output
    particle = ana3AIMEC.aimecTransform(rasterTransfo, particle, dem["header"])
    assert particle["lAimec"] == [-2, 2]
    assert particle["sAimec"] == [0, 7]

    # F = test_aimecTransform()
    # Extract input file locations
    pathDict = {}
    dir = pathlib.Path(__file__).parents[0]
    dirname = dir / "data" / "testAna3Aimec"

    # Extract dem
    demSource = list(dirname.glob("*.asc"))
    pathDict["demSource"] = demSource[0]
    demSource = pathDict["demSource"]
    dem = IOf.readRaster(demSource)

    # Create rasterTransfo
    rasterTransfo = {}
    rasterTransfo["s"] = np.linspace(0, 499, 500)
    rasterTransfo["l"] = np.linspace(0, 99, 100)
    gridy, gridx = np.meshgrid(rasterTransfo["l"], rasterTransfo["s"])
    rasterTransfo["x"] = rasterTransfo["s"]
    rasterTransfo["y"] = 50 * np.ones(np.shape(rasterTransfo["s"]))
    rasterTransfo["gridx"] = gridx
    rasterTransfo["gridy"] = gridy
    rasterTransfo["rasterArea"] = np.ones((500, 100))
    rasterTransfo["indStartOfRunout"] = 400
    rasterTransfo["startOfRunoutAreaAngle"] = 10

    # create particle
    particle = {}
    particle["x"] = np.linspace(200, 50, 400)
    particle["y"] = np.linspace(0, 50, 100)

    # run aimecTransform and test the output
    particle = ana3AIMEC.aimecTransform(rasterTransfo, particle, dem["header"])
    assert particle["lAimec"] != []
    assert particle["sAimec"] != []


def test_fullAimecAnalysis(tmp_path):
    # setup a case where the result peak fields don't have same extent and origin
    # to test whether the transformation is still working correctly
    avaDir = pathlib.Path(tmp_path, "avaAimecTest")
    testDir = avaDir / "testOutputs"
    fU.makeADir(testDir)
    dir = pathlib.Path(__file__).parents[0]
    lineDir = dir / "data" / "aimecInput" / "LINES"
    aimecInput = avaDir / "Inputs" / "LINES"
    shutil.copytree(lineDir, aimecInput)
    cellSize = 5.0
    nRows = 10
    nCols = 12
    nodata_value = np.nan
    xllcenter = 0.0
    yllcenter = 0.0

    # rasterio requires west, north
    # rasterio.transform.from_origin(west, north, xsize, ysize)
    transform = rasterio.transform.from_origin(xllcenter, yllcenter + nRows * cellSize, cellSize, cellSize)
    # crs = rasterio.crs.CRS.from_epsg(31287)
    crs = rasterio.crs.CRS()

    demHeader = {
        "cellsize": cellSize,
        "nrows": nRows,
        "ncols": nCols,
        "nodata_value": nodata_value,
        "xllcenter": xllcenter,
        "yllcenter": yllcenter,
        "driver": "AAIGrid",
        "crs": crs,
        "transform": transform,
    }
    demData = np.tile(np.flip(np.arange(12)), (10, 1))
    demName = avaDir / "Inputs" / "testDEM"
    IOf.writeResultToRaster(demHeader, demData, demName, flip=True)

    res1 = np.zeros((nRows, nCols))
    res1[3:7, 3:8] = 10.0
    res1[3:7, 7:10] = 7.0
    res1File = avaDir / "testOutputs" / "test1_01T_null_fullDEM_pft"
    IOf.writeResultToRaster(demHeader, res1, res1File, flip=True)

    xllcenter = 5.0
    yllcenter = 10.0
    nRows = 8
    nCols = 10
    # rasterio requires west, north
    # rasterio.transform.from_origin(west, north, xsize, ysize)
    transform = rasterio.transform.from_origin(xllcenter, yllcenter + nRows * cellSize, cellSize, cellSize)
    crs = rasterio.crs.CRS()

    demHeader2 = {
        "cellsize": cellSize,
        "nrows": nRows,
        "ncols": nCols,
        "nodata_value": nodata_value,
        "xllcenter": xllcenter,
        "yllcenter": yllcenter,
        "driver": "AAIGrid",
        "crs": crs,
        "transform": transform,
    }

    res2 = np.zeros((nRows, nCols))
    res2[1:5, 2:7] = 10.0
    res2[1:5, 6:9] = 7.0
    resFile2 = avaDir / "testOutputs" / "test1_01T_null_partDEM_pft"
    IOf.writeResultToRaster(demHeader2, res2, resFile2, flip=True)

    cfg = cfgUtils.getModuleConfig(ana3AIMEC, onlyDefault=True)
    cfgSetup = cfg["AIMECSETUP"]
    cfgSetup["defineRunoutArea"] = "False"
    cfgSetup["resType"] = "pft"
    cfgSetup["thresholdValue"] = "7.5"
    cfg["FLAGS"]["flagMass"] = "False"
    cfgSetup["runoutResType"] = "pft"
    cfgSetup["resTypes"] = "pft"
    cfg["PLOTS"]["compResType1"] = "deltaSXY"
    cfg["PLOTS"]["compResType2"] = "runoutAngle"

    rasterTransfo, resAnalysisDF, plotDict, newRasters, pathDict = ana3AIMEC.fullAimecAnalysis(
        avaDir, cfg, inputDir=testDir, demFileName=""
    )

    assert resAnalysisDF["pftFieldMax"].iloc[0] == resAnalysisDF["pftFieldMax"].iloc[1]
    assert resAnalysisDF["sRunout"].iloc[0] == resAnalysisDF["sRunout"].iloc[1]
    assert resAnalysisDF["lRunout"].iloc[0] == resAnalysisDF["lRunout"].iloc[1]
    assert resAnalysisDF["simName"].iloc[0] != resAnalysisDF["simName"].iloc[1]
    assert resAnalysisDF["pftFieldMax"].iloc[0] == resAnalysisDF["pftFieldMax"].iloc[1]
    assert resAnalysisDF["pftFieldMean"].iloc[0] == resAnalysisDF["pftFieldMean"].iloc[1]

    xllcenter = 5.0
    yllcenter = 10.0
    nRows = 8
    nCols = 10

    transform = rasterio.transform.from_origin(xllcenter, yllcenter + nRows * cellSize, cellSize, cellSize)
    crs = rasterio.crs.CRS()

    demHeader2 = {
        "cellsize": cellSize,
        "nrows": nRows,
        "ncols": nCols,
        "nodata_value": nodata_value,
        "xllcenter": xllcenter,
        "yllcenter": yllcenter,
        "driver": "AAIGrid",
        "crs": crs,
        "transform": transform,
    }

    res2 = np.zeros((nRows, nCols))
    res2[1:5, 2:7] = 10.0
    res2[1:5, 6:9] = 6.0
    resFile2 = avaDir / "testOutputs" / "test1_01T_null_partDEM_pft"
    IOf.writeResultToRaster(demHeader2, res2, resFile2, flip=True)

    rasterTransfo, resAnalysisDF, plotDict, newRasters, pathDict = ana3AIMEC.fullAimecAnalysis(
        avaDir, cfg, inputDir=testDir, demFileName=""
    )

    assert resAnalysisDF["pftFieldMax"].iloc[0] == resAnalysisDF["pftFieldMax"].iloc[1]
    assert resAnalysisDF["sRunout"].iloc[0] == resAnalysisDF["sRunout"].iloc[1]
    assert resAnalysisDF["lRunout"].iloc[0] == resAnalysisDF["lRunout"].iloc[1]
    assert resAnalysisDF["simName"].iloc[0] != resAnalysisDF["simName"].iloc[1]
    assert resAnalysisDF["pftFieldMax"].iloc[0] == resAnalysisDF["pftFieldMax"].iloc[1]
    assert resAnalysisDF["pftFieldMean"].iloc[0] != resAnalysisDF["pftFieldMean"].iloc[1]


def test_postProcessReference(tmp_path):
    """test computing the runout line from different data sets as furthest coordinate along s for each l"""

    avaName = "avaHockeyChannel"
    dirname = pathlib.Path(__file__).parents[0]
    sourceDir = dirname / ".." / "data" / avaName
    refFileTestDir = dirname / "data" / "referenceAIMECTest"
    avalancheDir = tmp_path / avaName
    refDataDir = avalancheDir / "Inputs" / "REFDATA"

    # Copy input to tmp dir
    shutil.copytree(sourceDir, avalancheDir)
    shutil.copytree(refFileTestDir, refDataDir)

    # setup inputs
    cfg = cfgUtils.getModuleConfig(ana3AIMEC, onlyDefault=True)
    cfgSetup = cfg["AIMECSETUP"]

    pathDict = {}
    pathDict = aT.readAIMECinputs(
        avalancheDir, pathDict, cfgSetup.getboolean("defineRunoutArea"), dirName="com1DFA"
    )
    demSource = pathDict["demSource"]
    dem = IOf.readRaster(demSource)

    rasterTransfo = aT.makeDomainTransfo(pathDict, dem, 5.0, cfgSetup)

    referenceDF = aT.createReferenceDF(pathDict)
    newRasters = {}

    # function to test
    refDataTransformed, referenceDF = ana3AIMEC.postProcessReference(
        cfg, rasterTransfo, pathDict, referenceDF, newRasters
    )

    assert np.isclose(refDataTransformed["refLine"]["xRunout"], 4239.6903)
    assert np.isclose(refDataTransformed["refLine"]["yRunout"], -3950)
    assert np.isclose(refDataTransformed["refLine"]["sRunout"], 3198.7449)
    assert np.isclose(refDataTransformed["refLine"]["lRunout"], -50.0)
    assert len(np.where(refDataTransformed["refLine"]["runoutLineFound"] == False)[0]) == 0
    maxIndex = np.amax(refDataTransformed["refLine"]["index"])

    assert np.isclose(rasterTransfo["s"][int(maxIndex)], 3198.7449)


# From here it is AI created


@pytest.fixture
def mockConfig(mocker):
    """Mock configparser object with AIMECSETUP section"""
    cfg = mocker.MagicMock()
    aimecSetup = mocker.MagicMock()

    # Configure AIMECSETUP section
    aimecSetup.getfloat.return_value = 2.0
    aimecSetup.__getitem__.side_effect = lambda key: {"interpMethod": "bilinear"}[key]

    cfg.__getitem__.side_effect = lambda section: {"AIMECSETUP": aimecSetup}[section]

    return cfg


@pytest.fixture
def mockRasterTransfo():
    """Mock raster transformation dictionary"""
    return {
        "dem": {
            "header": {"cellsize": 1.0, "xllcenter": 0.0, "yllcenter": 0.0},
            "rasterData": np.array([[1, 2], [3, 4]]),
            "originalHeader": {},
        }
    }


@pytest.fixture
def mockPathDict(tmp_path):
    """Mock path dictionary with temporary files"""
    return {
        "referenceLine": [tmp_path / "line1.shp"],
        "referencePoint": [tmp_path / "point1.shp"],
        "referencePolygon": [tmp_path / "poly1.shp"],
    }


@pytest.fixture
def mock_path_dict_line_only(tmp_path):
    """Mock path dictionary with ONLY line reference"""
    return {
        "referenceLine": [tmp_path / "line1.shp"],
        "referencePoint": [],  # Empty list for other types
        "referencePolygon": [],
    }


@pytest.fixture
def mockReferenceDf(mocker):
    """Mock DataFrame that supports assignment"""
    return mocker.MagicMock()


@pytest.fixture
def mockNewRasters():
    """Real dictionary for raster storage"""
    return {}


def test_postProcessReference_referencePointMultiplePointsError(
    mocker, mockConfig, mockRasterTransfo, mockPathDict, mockReferenceDf, mockNewRasters
):
    """Test error when multiple points are provided in reference file"""
    # Mock all shapefile reading functions
    mocker.patch("avaframe.in2Trans.shpConversion.readLine", return_value={"x": [0], "y": [0]})
    mocker.patch("avaframe.in2Trans.shpConversion.readPoints", return_value={"x": [1, 2], "y": [3, 4]})
    mocker.patch(
        "avaframe.in2Trans.shpConversion.readLine", return_value={"x": [0], "y": [0]}
    )  # For polygon
    mocker.patch("avaframe.in3Utils.geoTrans.prepareLine", return_value=({"x": [0], "y": [0]}, None))
    # Mock downstream processing to short-circuit execution
    mocker.patch("avaframe.in3Utils.geoTrans.projectOnGrid", return_value=({"x": [0], "y": [0]}, None))
    mocker.patch("avaframe.ana3AIMEC.aimecTools.transform")
    mocker.patch("avaframe.ana3AIMEC.aimecTools.computeRunoutLine")
    mocker.patch("avaframe.out3Plot.outAIMEC.referenceLinePlot")

    with pytest.raises(AssertionError) as excInfo:
        postProcessReference(mockConfig, mockRasterTransfo, mockPathDict, mockReferenceDf, mockNewRasters)

    assert "More than one point in reference data file" in str(excInfo.value)


def test_postProcessReference_processReferenceLine(
    mocker, mockConfig, mockRasterTransfo, mockPathDict, mockReferenceDf, mockNewRasters
):
    """Test processing of reference line data"""
    # Use line-only path dictionary
    pathDict = {"referenceLine": [Mock()], "referencePoint": [], "referencePolygon": []}  # Mock Path object

    # Setup mocks
    mocker.patch("avaframe.in2Trans.shpConversion.readLine", return_value={"x": [0], "y": [0]})
    mocker.patch("avaframe.in3Utils.geoTrans.prepareLine", return_value=({"x": [0], "y": [0]}, None))
    mocker.patch("avaframe.in3Utils.geoTrans.projectOnGrid", return_value=(None, np.array([[1]])))
    mocker.patch("avaframe.ana3AIMEC.aimecTools.transform", return_value=np.array([[1]]))
    mocker.patch(
        "avaframe.ana3AIMEC.aimecTools.computeRunoutLine", return_value={"sRunout": 10.0, "lRunout": 5.0}
    )
    mocker.patch("avaframe.ana3AIMEC.aimecTools.addReferenceAnalysisTODF", return_value=mockReferenceDf)
    plot_mock = mocker.patch("avaframe.out3Plot.outAIMEC.referenceLinePlot")

    # Execute function
    ref_data, updated_df = postProcessReference(
        mockConfig, mockRasterTransfo, pathDict, mockReferenceDf, mockNewRasters
    )

    # Verify results
    assert "refLine" in ref_data
    assert mockNewRasters["refLine"].shape == (1, 1)
    plot_mock.assert_called_once()
