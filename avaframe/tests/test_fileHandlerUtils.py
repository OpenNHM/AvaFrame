"""
    Pytest for fileHandlerUtils

    This file is part of Avaframe.

 """

#  Load modules
import numpy as np
import os
from avaframe.in3Utils import fileHandlerUtils as fU
import avaframe.in2Trans.rasterUtils as rasterUtils
import pytest
import shutil
import pathlib
import configparser
import rasterio
import json


def test_makeADir(tmp_path):
    """ test make directory """

    # create temporary directory
    avaName = 'testAva'
    avaDir = os.path.join(tmp_path, avaName)
    fU.makeADir(avaDir)
    avaDir2 = os.path.join(tmp_path, avaName, 'test')

    dirTrue = os.path.isdir(avaDir)
    dirFalse = os.path.isdir(avaDir2)

    assert dirTrue
    assert dirFalse is False


def test_readLogFile():
    """ Test if logDict is generated correctly """

    # Test function
    dirPath = os.path.dirname(__file__)
    logName = os.path.join(dirPath, 'data', 'testExpLog.txt')
    cfg = configparser.ConfigParser()
    cfg = {'varPar': 'RelTh'}
    logDict = fU.readLogFile(logName, cfg)
    logDictMu = fU.readLogFile(logName)

    assert logDict['noSim'][4] == 5
    assert logDict['simName'][2] == 'release1HS2_null_dfa'
    assert logDict['RelTh'][2] == 4.0
    assert logDictMu['noSim'][4] == 5
    assert logDictMu['simName'][2] == 'release1HS2_null_dfa'
    assert logDictMu['Mu'][2] == 4.0


def test_extractLogInfo():
    """ test extracting info from logFile """

    # setup required input
    dirPath = pathlib.Path(__file__).parents[0]
    logName = dirPath / 'data' / 'logTest.log'

    # call function to be tested
    logDict = fU.extractLogInfo(logName)

#    print('logDict', logDict)
    # define test results
    time = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 301.1])
    mass = np.asarray([1.99393e+07, 1.99393e+07, 1.99393e+07, 1.99393e+07, 1.99393e+07, 1.99393e+07,
                       2.02876e+07])
    entrMass = np.asarray([0., 0., 0., 0., 0., 0., 0.])
    stopTime = 78.4059
    stopCrit = 'kinetic energy 1.00 of peak KE'

    assert logDict['indRun'] == [0, 7]
    assert np.array_equal(logDict['time'], time)
    assert np.array_equal(logDict['mass'], mass)
    assert np.array_equal(logDict['entrMass'], entrMass)
    assert logDict['stopTime'] == stopTime
    assert logDict['stopCrit'] == stopCrit


def test_checkIfFileExists():
    """ test if a file exists and if not throw error """

    # setup required input
    dirPath = pathlib.Path(__file__).parents[0]
    avaTestName = 'avaHockeyChannelPytest'
    testPath = dirPath / '..' / '..' / 'benchmarks' / avaTestName
    pathData = testPath / 'Outputs' / 'com1DFAOri' / 'peakFiles' / 'release1HS_ent_dfa_67dc2dc10a_pft.asc'

    # call function to be tested
    with pytest.raises(FileNotFoundError) as e:
        assert fU.checkIfFileExists(pathData, fileType='')
    assert str(e.value) == ('No  file found called: %s' % str(pathData))

    # call function to be tested
    with pytest.raises(FileNotFoundError) as e:
        assert fU.checkIfFileExists(pathData, fileType='log info')
    assert str(e.value) == ('No log info file found called: %s' % str(pathData))

    # call function to be tested
    pathData2 = 'test/dataTest'
#    print('pathDatastr', pathData2)
    with pytest.raises(FileNotFoundError) as e:
        assert fU.checkIfFileExists(pathData, fileType='log info')
    assert str(e.value) == ('No log info file found called: %s' % str(pathData))


def test_makeSimDF():
    """ Test if simulation dataFrame is generated correctly """

    # Test function
    dirPath = os.path.dirname(__file__)
    inputDir = os.path.join(dirPath, 'data', 'testSim')
    cfg = configparser.ConfigParser()
    cfg = {'varPar': 'test'}
    dataDF = fU.makeSimDF(inputDir, simID=cfg['varPar'])

    assert dataDF['names'][0] == 'releaseTest1_0.888_entres_dfa_ppr'
    assert dataDF['releaseArea'][0] == 'releaseTest1'
    assert dataDF['simType'][0] == 'entres'
    assert dataDF['resType'][0] == 'ppr'
    assert dataDF['cellSize'][0] == 5.0
    assert dataDF['test'][0] == '0.888'

    inputDir = os.path.join(dirPath, 'data', 'testSim1')
    dataDF = fU.makeSimDF(inputDir, simID=cfg['varPar'])
    assert dataDF['names'][0] == 'releaseTest1_test_AF_0.888_entres_dfa_ppr'
    assert dataDF['releaseArea'][0] == 'releaseTest1_test'
    assert dataDF['simType'][0] == 'entres'
    assert dataDF['resType'][0] == 'ppr'
    assert dataDF['cellSize'][0] == 5.0
    assert dataDF['test'][0] == '0.888'


def test_makeSimFromResDF():
    """ Test if simulation dataFrame is generated correctly """

    # Test function
    dirPath = os.path.dirname(__file__)
    inputDir = os.path.join(dirPath, '..', '..', 'benchmarks', 'avaHelixChannelWetSnowTest')
    dataDF, resTypeList = fU.makeSimFromResDF(inputDir, 'comModule', inputDir=inputDir)
#    print(resTypeList)
#    print(dataDF.columns)
#    print(dataDF.index)
    assert dataDF['simName'].iloc[0] == 'release1HX_f6942a9a69_C_L_ent_dfa'
    assert dataDF['releaseArea'].iloc[0] == 'release1HX'
    assert dataDF['simType'].iloc[0] == 'ent'
    assert dataDF['cellSize'].iloc[0] == 5.0
    assert dataDF['modelType'].iloc[0] == 'dfa'
    assert dataDF['simModified'].iloc[0] == 'C'

    inputDir = os.path.join(dirPath, 'data', 'testSim1')
    dataDF2, resTypeList = fU.makeSimFromResDF(inputDir, 'comModule', inputDir=inputDir)
    assert dataDF2['simName'].iloc[0] == 'releaseTest1_test_AF_0.888_entres_dfa'
    assert dataDF2['releaseArea'].iloc[0] == 'releaseTest1_test'
    assert dataDF2['simHash'].iloc[0] == '0.888'
    assert dataDF2['cellSize'].iloc[0] == 5.0
    assert dataDF2['simType'].iloc[0] == 'entres'
    assert dataDF2['modelType'].iloc[0] == 'dfa'
    assert dataDF2['simModified'].iloc[0] == 'not specified'
    assert ('simModified' in dataDF2.columns ) == True



def test_exportcom1DFAOrigOutput(tmp_path):
    """ Test if export of result files works """

    # Create input directoy structure
    dirPath = os.path.dirname(__file__)
    avaName = 'avaParabola'
    avaNameTest = 'avaParabolaPyest'
    avaDir = os.path.join(tmp_path, avaName)
    outDir = os.path.join(avaDir, 'Work', 'com1DFAOrig', 'FullOutput_RelTh_1.25000', 'release1PF_entres_dfa', 'raster')
    os.makedirs(avaDir)
    os.makedirs(outDir)

    # copy inut data from benchmarks folder to tmp_path and rename correctly
    resType = ['ppr', 'pft', 'pfv']
    for m in resType:
        if m == 'pfv':
            avaData = os.path.join(dirPath, '..', '..', 'benchmarks', avaNameTest,
                               'release1PF_entres_dfa_1.25000_pfv.asc')
            input = os.path.join(avaDir, 'Work', 'com1DFAOrig', 'FullOutput_RelTh_1.25000',
                                'release1PF_entres_dfa', 'raster', 'release1PF_entres_dfa_pv.asc')
        elif m == 'pft':
            avaData = os.path.join(dirPath, '..', '..', 'benchmarks', avaNameTest,
                               'release1PF_entres_dfa_1.25000_pft.asc')
            input = os.path.join(avaDir, 'Work', 'com1DFAOrig', 'FullOutput_RelTh_1.25000',
                                'release1PF_entres_dfa', 'raster', 'release1PF_entres_dfa_pfd.asc')
        else:
            avaData = os.path.join(dirPath, '..', '..', 'benchmarks', avaNameTest,
                                'release1PF_entres_dfa_1.25000_%s.asc' % m)
            input = os.path.join(avaDir, 'Work', 'com1DFAOrig', 'FullOutput_RelTh_1.25000',
                                 'release1PF_entres_dfa', 'raster', 'release1PF_entres_dfa_%s.asc' % m)
        shutil.copy(avaData, input)
    avaData = os.path.join(dirPath, '..', '..', 'benchmarks', avaNameTest,
                           'ExpLog.txt')
    input = os.path.join(avaDir, 'Work', 'com1DFAOrig', 'ExpLog.txt')
    shutil.copy(avaData, input)
    avaData = os.path.join(dirPath, '..', '..', 'benchmarks', avaNameTest,
                           'test.html')
    input = os.path.join(avaDir, 'Work', 'com1DFAOrig', 'FullOutput_RelTh_1.25000',
                        'release1PF_entres_dfa.html')
    shutil.copy(avaData, input)

    # Set cfg
    cfg = configparser.ConfigParser()
    cfg = {'varPar': 'RelTh'}

    # Call function to test
    fU.exportcom1DFAOrigOutput(avaDir, cfg)
    # load exported file
    pprTest = np.loadtxt(os.path.join(avaDir, 'Outputs', 'com1DFAOrig', 'peakFiles',
                         'release1PF_entres_dfa_1.25000_ppr.asc'), skiprows=6)

    # load initial file
    pprBench = np.loadtxt(os.path.join(dirPath, '..', '..', 'benchmarks', avaNameTest,
                                       'release1PF_entres_dfa_1.25000_ppr.asc'), skiprows=6)
    # Compare result to reference solution
    testRes = np.allclose(pprTest, pprBench, atol=1.e-12)

    assert testRes is True


def test_splitIniValueToArraySteps():
    """ Test if splitting into an array works fine  """

    cfgValues = '1.0|2.5|3.8'
    cfgValuesList = np.asarray([1.0, 2.5, 3.8])

    cfgValues2 = '0:10:5'
    cfgValuesList2 = np.asarray([0., 2.5, 5., 7.5, 10.])

    # call function to be tested
    items = fU.splitIniValueToArraySteps(cfgValues)
    items2 = fU.splitIniValueToArraySteps(cfgValues2)
    items3 = fU.splitIniValueToArraySteps(cfgValues, returnList=True)

    assert len(items) == len(cfgValuesList)
    assert items[0] == cfgValuesList[0]
    assert items[1] == cfgValuesList[1]
    assert items[2] == cfgValuesList[2]
    assert len(items2) == len(cfgValuesList2)
    assert items2[0] == cfgValuesList2[0]
    assert items2[1] == cfgValuesList2[1]
    assert items2[2] == cfgValuesList2[2]
    assert items2[3] == cfgValuesList2[3]
    assert items2[4] == cfgValuesList2[4]
    assert len(items3) == len(cfgValuesList)
    assert items3[0] == '1.0'
    assert items3[1] == '2.5'
    assert items3[2] == '3.8'
    assert isinstance(items3, list)


    cfgValues4 = '10$50$11'
    cfgValues5 = '10$-50$11'
    cfgValues6 = '10$+50$11'

    # call function to be tested
    items4 = fU.splitIniValueToArraySteps(cfgValues4)
    items5 = fU.splitIniValueToArraySteps(cfgValues5)
    items6 = fU.splitIniValueToArraySteps(cfgValues6)

    assert np.array_equal(items4, np.linspace(5., 15., 11))
    assert np.array_equal(items5, np.linspace(10., 5., 11))
    assert np.array_equal(items6, np.linspace(10., 15., 11))

    cfgValues4 = '10$50$11&20'
    cfgValues6 = '10:50:11&40'

    # call function to be tested
    items4 = fU.splitIniValueToArraySteps(cfgValues4)
    items6 = fU.splitIniValueToArraySteps(cfgValues6)

    assert np.array_equal(items4, np.append(np.linspace(5., 15., 11), 20))
    assert np.array_equal(items6, np.append(np.linspace(10, 50, 11), 40))


def test_splitTimeValueToArrayInterval():
    """ Test if splitting into an array works fine  """

    cfgValues = '1.0|2.5|3.8'
    cfgValuesList = np.asarray([1.0, 2.5, 3.8])

    cfgValues1 = '0.|2.5|3.8'
    cfgValuesList1 = np.asarray([0., 2.5, 3.8])

    cfgValues2 = '0:5'
    cfgValuesList2 = np.asarray([0., 5., 10., 15.])

    cfgValues3 = ''
    cfgValuesList3 = np.asarray([40.])

    cfgValues4 = '0:22'
    cfgValuesList4 = np.asarray([0., 20.])

    cfgValues5 = '0'
    cfgValuesList5 = np.asarray([0.])

    cfg = configparser.ConfigParser()
    cfg['GENERAL'] = {'tEnd': '20'}
    cfgGen = cfg['GENERAL']

    # call function to be tested
    cfgGen['tSteps'] = cfgValues
    items = fU.splitTimeValueToArrayInterval(cfgGen['tSteps'], cfgGen.getfloat('tEnd'))
    cfgGen['tSteps'] = cfgValues1
    items1 = fU.splitTimeValueToArrayInterval(cfgGen['tSteps'], cfgGen.getfloat('tEnd'))
    cfgGen['tSteps'] = cfgValues2
    items2 = fU.splitTimeValueToArrayInterval(cfgGen['tSteps'], cfgGen.getfloat('tEnd'))
    cfgGen['tSteps'] = cfgValues3
    items3 = fU.splitTimeValueToArrayInterval(cfgGen['tSteps'], cfgGen.getfloat('tEnd'))
    cfgGen['tSteps'] = cfgValues4
    items4 = fU.splitTimeValueToArrayInterval(cfgGen['tSteps'], cfgGen.getfloat('tEnd'))
    cfgGen['tSteps'] = cfgValues5
    items5 = fU.splitTimeValueToArrayInterval(cfgGen['tSteps'], cfgGen.getfloat('tEnd'))

    assert len(items) == len(cfgValuesList)
    assert items[0] == cfgValuesList[0]
    assert items[1] == cfgValuesList[1]
    assert items[2] == cfgValuesList[2]
    assert len(items1) == len(cfgValuesList1)
    assert items1[0] == cfgValuesList1[0]
    assert items1[1] == cfgValuesList1[1]
    assert items1[2] == cfgValuesList1[2]
    assert len(items2) == len(cfgValuesList2)
    assert items2[0] == cfgValuesList2[0]
    assert items2[1] == cfgValuesList2[1]
    assert items2[2] == cfgValuesList2[2]
    assert items2[3] == cfgValuesList2[3]
    assert len(items4) == len(cfgValuesList4)
    assert items4[0] == cfgValuesList4[0]
    assert items4[1] == cfgValuesList4[1]
    assert len(items5) == len(cfgValuesList5)
    assert items5[0] == cfgValuesList5[0]


def test_getFilterDict():
    """ test generation of filter dictionary """

    cfg = configparser.ConfigParser()
    cfg.optionxform = str
    cfg['GENERAL'] = {'tEnd': '20'}
    cfg['FILTER'] = {'relTh': '1:2:3', 'entH': 200, 'simType': '', 'secRelArea': 'True', 'mu': '<0.1'}

    parametersDict = fU.getFilterDict(cfg, 'FILTER')

    noKey = 'simType' in parametersDict

#    print('parametersDict', parametersDict)

    assert np.allclose(parametersDict['relTh'], np.asarray([1, 1.5, 2]), atol=1e-10)
    assert noKey is False
    assert parametersDict['entH'] == [200.]
    assert parametersDict['secRelArea'] == ['True']
    assert parametersDict['mu'] == ['<0.1']

    parametersDict = fU.getFilterDict(cfg, 'TESTS')

    assert parametersDict == {}


def test_fetchFlowFields():
    """ test fetching fields in a folder """

    # setup required input
    # get input data
    dirPath = pathlib.Path(__file__).parents[0]
    avaName = 'avaHockeyChannelPytest'
    flowFieldsDir = dirPath / '..' / '..' / 'benchmarks' / avaName / 'Outputs' / 'com1DFA' / 'peakFiles'
    suffix = 'ppr'

    # call function to be tested
    flowFields = fU.fetchFlowFields(flowFieldsDir, suffix=suffix)
    flowFields = sorted(flowFields)
#    print('flowFields', flowFields, sorted(flowFields))

    assert flowFields[0].stem == 'release1HS_0dcd58fc86_ent_dfa_ppr'
    assert flowFields[1].stem == 'release2HS_3d519adab0_ent_dfa_ppr'
    assert len(flowFields) == 2

    # call function to be tested
    flowFields = fU.fetchFlowFields(flowFieldsDir)
    flowFields = sorted(flowFields)
#    print('flowFields', flowFields, sorted(flowFields))

    assert flowFields[0].stem == 'release1HS_0dcd58fc86_ent_dfa_pft'
    assert len(flowFields) == 6


# --- Multi-layer makeSimFromResDF tests ---

RASTER_HEADER = """ncols         10
nrows         10
xllcenter     0
yllcenter     0
cellsize      5
NODATA_value  -9999
"""


def _createTestRaster(filepath):
    """Create a minimal valid raster file for testing"""
    with open(filepath, "w") as f:
        f.write(RASTER_HEADER)
        for _ in range(10):
            f.write(" ".join(["0"] * 10) + "\n")


def test_makeSimFromResDF_multiLayer(tmp_path):
    """Test that multi-layer files produce layer-suffixed columns"""
    peakDir = tmp_path / "peakFiles"
    peakDir.mkdir()

    # Create multi-layer result files: sim_L1_ppr, sim_L1_pfv, sim_L2_ppr, sim_L2_pfv
    simBase = "release1_abc123_com8_C_null_psa"
    for layer in ["L1", "L2"]:
        for resType in ["ppr", "pfv"]:
            _createTestRaster(peakDir / f"{simBase}_{layer}_{resType}.asc")

    dataDF, resTypeListAll = fU.makeSimFromResDF(str(tmp_path), "com8", inputDir=str(peakDir))

    # Layer-suffixed columns should exist
    assert "ppr_l1" in dataDF.columns
    assert "ppr_l2" in dataDF.columns
    assert "pfv_l1" in dataDF.columns
    assert "pfv_l2" in dataDF.columns

    # Layer-suffixed columns should contain file paths (not NaN)
    assert not dataDF["ppr_l1"].isnull().any()
    assert not dataDF["ppr_l2"].isnull().any()
    assert not dataDF["pfv_l1"].isnull().any()
    assert not dataDF["pfv_l2"].isnull().any()

    # Layers metadata column should exist and be populated
    assert "layers" in dataDF.columns
    layersVal = dataDF["layers"].iloc[0]
    assert "L1" in layersVal
    assert "L2" in layersVal

    # resTypeListAll should contain layer-suffixed names
    assert "ppr_l1" in resTypeListAll
    assert "ppr_l2" in resTypeListAll
    assert "pfv_l1" in resTypeListAll
    assert "pfv_l2" in resTypeListAll


def test_makeSimFromResDF_singleLayer_unchanged(tmp_path):
    """Test that single-layer files still produce standard columns (backward compat)"""
    peakDir = tmp_path / "peakFiles"
    peakDir.mkdir()

    # Create standard single-layer result files
    simBase = "release1_abc123_com1_C_null_dfa"
    for resType in ["ppr", "pfv", "pft"]:
        _createTestRaster(peakDir / f"{simBase}_{resType}.asc")

    dataDF, resTypeListAll = fU.makeSimFromResDF(str(tmp_path), "com1", inputDir=str(peakDir))

    # Standard columns should exist
    assert "ppr" in dataDF.columns
    assert "pfv" in dataDF.columns
    assert "pft" in dataDF.columns

    # Standard columns should contain file paths
    assert not dataDF["ppr"].isnull().any()
    assert not dataDF["pfv"].isnull().any()
    assert not dataDF["pft"].isnull().any()

    # Layers metadata column should be NaN for single-layer
    if "layers" in dataDF.columns:
        assert dataDF["layers"].isnull().all()

    # resTypeListAll should contain standard names
    assert "ppr" in resTypeListAll
    assert "pfv" in resTypeListAll
    assert "pft" in resTypeListAll


def test_makeSimFromResDF_multiLayer_simName_reconstructed(tmp_path):
    """Test that simName is correctly reconstructed without layer component"""
    peakDir = tmp_path / "peakFiles"
    peakDir.mkdir()

    simBase = "release1_abc123_com8_C_null_psa"
    _createTestRaster(peakDir / f"{simBase}_L1_ppr.asc")
    _createTestRaster(peakDir / f"{simBase}_L2_ppr.asc")

    dataDF, _ = fU.makeSimFromResDF(str(tmp_path), "com8", inputDir=str(peakDir))

    # Should have exactly one row (one simulation, two layers)
    assert len(dataDF) == 1
    # simName should NOT contain the layer
    assert "L1" not in dataDF["simName"].iloc[0]
    assert "L2" not in dataDF["simName"].iloc[0]
    assert dataDF["simName"].iloc[0] == simBase


# --- Multi-layer makeSimDF tests ---


def test_makeSimDF_layerColumn(tmp_path):
    """makeSimDF should include a layer column populated from parseSimName"""
    peakDir = tmp_path / "peakFiles"
    peakDir.mkdir()

    _createTestRaster(peakDir / "release1_abc123_null_psa_L1_ppr.asc")
    _createTestRaster(peakDir / "release1_abc123_null_psa_L2_ppr.asc")
    _createTestRaster(peakDir / "release1_abc123_null_psa_L1_pfv.asc")

    dataDF = fU.makeSimDF(str(peakDir))

    assert "layer" in dataDF.columns
    # 3 files -> 3 rows
    assert len(dataDF) == 3
    layers = sorted(dataDF["layer"].tolist())
    assert layers == ["L1", "L1", "L2"]


def test_makeSimDF_layerColumn_singleLayer(tmp_path):
    """makeSimDF layer column should be empty string for single-layer files"""
    peakDir = tmp_path / "peakFiles"
    peakDir.mkdir()

    _createTestRaster(peakDir / "release1_abc123_null_dfa_ppr.asc")
    _createTestRaster(peakDir / "release1_abc123_null_dfa_pfv.asc")

    dataDF = fU.makeSimDF(str(peakDir))

    assert "layer" in dataDF.columns
    assert all(v == "" for v in dataDF["layer"].tolist())


def test_checkResultFolderFilesExist(tmp_path):
    """ test make directory """

    avaName = 'testRes'
    resDir = pathlib.Path(tmp_path) / avaName
    folderExist, filesExist = fU.checkResultFolderFilesExist(resDir)

    assert folderExist is False
    assert filesExist is False

    fU.makeADir(resDir)
    folderExist, filesExist = fU.checkResultFolderFilesExist(resDir)
    assert folderExist
    assert filesExist is False

    fU.makeADir(resDir)
    folderExist, filesExist = fU.checkResultFolderFilesExist(resDir, outputnames=["zdelta"])
    assert folderExist
    assert filesExist is False

    # create a result file
    # first create test raster and save in test folder
    rasterName = "resultTest_123_zdelta"
    testRaster = np.zeros((10, 10))

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

    rasterUtils.writeResultToRaster(header, testRaster, resDir / rasterName, useCompression=True, flip=True)

    folderExist, filesExist = fU.checkResultFolderFilesExist(resDir, outputnames=["zdelta"])
    assert folderExist
    assert filesExist

    folderExist, filesExist = fU.checkResultFolderFilesExist(resDir, outputnames=["zdelta", "flux"])
    assert folderExist
    assert filesExist is False

    rasterName = "resultTest_456_flux"
    rasterUtils.writeResultToRaster(header, testRaster, resDir / rasterName, useCompression=True, flip=True)

    folderExist, filesExist = fU.checkResultFolderFilesExist(resDir, outputnames=["zdelta", "flux"])
    assert folderExist
    assert filesExist


def test_searchCom4ResDir(tmp_path):
    """ test searchCom4ResDir function for locating com4FlowPy Result Directories"""
    simHash = "abc123"
    outputPath = pathlib.Path(tmp_path) / "Outputs"

    # no folder exists yet
    resFolder = fU.searchCom4ResDir(outputPath, simHash)
    assert resFolder is None

    # create folder directly in outputPath
    directResFolder = outputPath / f"res_{simHash}"
    fU.makeADir(directResFolder)
    resFolder = fU.searchCom4ResDir(outputPath, simHash)
    assert resFolder == directResFolder

    # remove it and create in peakFiles subfolder instead
    shutil.rmtree(directResFolder)
    peakResFolder = outputPath / "peakFiles" / f"res_{simHash}"
    fU.makeADir(peakResFolder)
    resFolder = fU.searchCom4ResDir(outputPath, simHash)
    assert resFolder == peakResFolder

    # different simHash should not be found
    resFolder = fU.searchCom4ResDir(outputPath, "otherHash")
    assert resFolder is None


def test_deleteCom4Results(tmp_path):
    """ test deleting com4FlowPy results folder and json file """
    outputPath = pathlib.Path(tmp_path) / "Outputs"
    simHash = "abc123"

    jsonFile = outputPath / f"{simHash}.json"
    resFolder = outputPath / f"res_{simHash}"
    fU.makeADir(resFolder)
    with open(jsonFile, "w") as f:
        json.dump({"simHash": simHash}, f)
    (resFolder / "dummy.txt").write_text("dummy")

    assert jsonFile.is_file()
    assert resFolder.is_dir()

    # deleting a non-existing simHash should not raise and not touch existing files
    fU.deleteCom4Results(outputPath, "otherHash")
    assert jsonFile.is_file()
    assert resFolder.is_dir()

    # deleting existing simHash removes both json and folder
    fU.deleteCom4Results(outputPath, simHash)
    assert not jsonFile.exists()
    assert not resFolder.exists()

    # calling again should not raise even though nothing is left to delete
    fU.deleteCom4Results(outputPath, simHash)
    assert not jsonFile.exists()
    assert not resFolder.exists()

    """ test deleting com4FlowPy results folder located in peakFiles subfolder """
    outputPath = pathlib.Path(tmp_path) / "Outputs"
    simHash = "peak123"

    jsonFile = outputPath / f"{simHash}.json"
    resFolder = outputPath / "peakFiles" / f"res_{simHash}"
    fU.makeADir(resFolder)
    with open(jsonFile, "w") as f:
        json.dump({"simHash": simHash}, f)

    assert jsonFile.is_file()
    assert resFolder.is_dir()

    fU.deleteCom4Results(outputPath, "otherHash")
    assert jsonFile.is_file()
    assert resFolder.is_dir()

    fU.deleteCom4Results(outputPath, simHash)
    assert not jsonFile.exists()
    assert not resFolder.exists()

    fU.deleteCom4Results(outputPath, simHash)
    assert not jsonFile.exists()
    assert not resFolder.exists()


def test_backupCom4Results(tmp_path):
    """ test backing up com4FlowPy results folder and json file """
    outputPath = pathlib.Path(tmp_path) / "Outputs"
    backupPath = outputPath / "backup"
    simHash = "abc123"

    jsonFile = outputPath / f"{simHash}.json"
    resFolder = outputPath / f"res_{simHash}"
    fU.makeADir(resFolder)
    with open(jsonFile, "w") as f:
        json.dump({"simHash": simHash}, f)
    (resFolder / "dummy.txt").write_text("dummy")

    assert jsonFile.exists()
    assert resFolder.exists()

    fU.backupCom4Results(outputPath, simHash)

    # originals should be gone
    assert not jsonFile.exists()
    assert not resFolder.exists()

    # backups should exist
    backupJson = backupPath / f"{simHash}.json"
    backupFolder = backupPath / f"res_{simHash}"
    assert backupJson.is_file()
    assert backupFolder.is_dir()
    assert (backupFolder / "dummy.txt").is_file()

    """ test that backing up twice does not overwrite previous backup """

    jsonFile = outputPath / f"{simHash}.json"
    resFolder = outputPath / f"res_{simHash}"
    fU.makeADir(resFolder)
    with open(jsonFile, "w") as f:
        json.dump({"simHash": simHash}, f)
    (resFolder / "dummy.txt").write_text("dummy")

    fU.backupCom4Results(outputPath, simHash)
    assert not (outputPath / f"{simHash}.json").exists()
    assert not (outputPath / f"res_{simHash}").exists()

    # both original and "(1)" suffixed backups should exist
    assert (backupPath / f"{simHash}.json").is_file()
    assert (backupPath / f"res_{simHash}").is_dir()
    assert (backupPath / f"{simHash}(1).json").is_file()
    assert (backupPath / f"res_{simHash}(1)").is_dir()

    """ test backing up when no results exist yet does not raise """
    outputPath = pathlib.Path(tmp_path)
    simHash = "doesNotExist"

    fU.backupCom4Results(outputPath, simHash)

    backupPath = outputPath / "backup"
    assert backupPath.is_dir()
    assert not (backupPath / f"{simHash}.json").exists()
    assert not (backupPath / f"res_{simHash}").exists()
