"""
    Pytest for com8MoTPSA postprocess layer renaming

    This file is part of Avaframe.
"""

import configparser
import pathlib
import shutil

import pytest
from unittest.mock import patch

import avaframe.com1DFA.com1DFA as com1DFA
from avaframe.com8MoTPSA import com8MoTPSA
from avaframe.in3Utils import cfgUtils


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


def _createMockRawOutputFiles(workDir, simKey, simType):
    """Create mock raw MoT-PSA output files matching the expected naming pattern.

    MoT-PSA appends raw suffixes (_p1_max, _p2_max, etc.) to the output root,
    which is the simKey itself. The simKey already contains simType_modelType,
    so the replace pattern straddles the key/suffix boundary.
    """
    simWorkDir = workDir / simKey
    simWorkDir.mkdir(parents=True)

    # MoT-PSA appends these suffixes to the key (output root)
    rawSuffixes = ["p1_max", "p2_max", "h1_max", "h2_max", "s1_max", "s2_max"]
    for suffix in rawSuffixes:
        _createTestRaster(simWorkDir / ("%s_%s.asc" % (simKey, suffix)))

    # DataTime.txt needed by postprocess
    (simWorkDir / "DataTime.txt").write_text("0.0\n1.0\n")


def _createConfigIndicatorDirs(avalancheDir):
    """Create configurationFilesDone and configurationFilesLatest dirs.

    In a real run these are created by initialiseRunDirs during project
    initialisation; the postprocess function assumes they exist.
    """
    for saveDir in ["configurationFilesDone", "configurationFilesLatest"]:
        configDir = avalancheDir / "Outputs" / "com8MoTPSA" / "configurationFiles" / saveDir
        configDir.mkdir(parents=True)


@patch("avaframe.com8MoTPSA.com8MoTPSA.oP.plotAllPeakFields")
@patch("avaframe.com8MoTPSA.com8MoTPSA.rU.readRaster", return_value={"header": {"cellsize": 5}})
def test_com8MoTPSAPostprocess_layerNaming(mockReadRaster, mockPlot, tmp_path):
    """Test that postprocess produces files with L1/L2 layer naming"""
    avalancheDir = tmp_path / "testAva"
    workDir = avalancheDir / "Work" / "com8MoTPSA"

    simKey = "release1_abc123_com8_C_null_psa"
    simType = "null"

    # Create mock raw output files
    _createMockRawOutputFiles(workDir, simKey, simType)
    _createConfigIndicatorDirs(avalancheDir)

    # Build mock simDict and cfgMain
    simDict = {
        simKey: {
            "simType": simType,
        }
    }
    cfgMain = {
        "MAIN": {"avalancheDir": str(avalancheDir)},
        "FLAGS": {"showPlot": "False", "savePlot": "True"},
    }
    inputSimFiles = {"demFile": str(tmp_path / "fake_dem.asc")}

    # Run postprocess
    com8MoTPSA.com8MoTPSAPostprocess(simDict, cfgMain, inputSimFiles)

    # Check output files in peakFiles directory
    peakDir = avalancheDir / "Outputs" / "com8MoTPSA" / "peakFiles"
    outputFiles = sorted([f.name for f in peakDir.glob("*.asc")])

    # Expected: L1/L2 layer naming, modelType stays psa for both layers
    expectedFiles = sorted([
        "%s_L1_ppr.asc" % simKey,
        "%s_L2_ppr.asc" % simKey,
        "%s_L1_pfd.asc" % simKey,
        "%s_L2_pfd.asc" % simKey,
        "%s_L1_pfv.asc" % simKey,
        "%s_L2_pfv.asc" % simKey,
    ])

    assert outputFiles == expectedFiles


@patch("avaframe.com8MoTPSA.com8MoTPSA.oP.plotAllPeakFields")
@patch("avaframe.com8MoTPSA.com8MoTPSA.rU.readRaster", return_value={"header": {"cellsize": 5}})
def test_com8MoTPSAPostprocess_entSimType(mockReadRaster, mockPlot, tmp_path):
    """Test layer naming with entrainment simType"""
    avalancheDir = tmp_path / "testAva"
    workDir = avalancheDir / "Work" / "com8MoTPSA"

    simKey = "release1_abc123_com8_C_ent_psa"
    simType = "ent"

    _createMockRawOutputFiles(workDir, simKey, simType)
    _createConfigIndicatorDirs(avalancheDir)

    simDict = {simKey: {"simType": simType}}
    cfgMain = {
        "MAIN": {"avalancheDir": str(avalancheDir)},
        "FLAGS": {"showPlot": "False", "savePlot": "True"},
    }
    inputSimFiles = {"demFile": str(tmp_path / "fake_dem.asc")}

    com8MoTPSA.com8MoTPSAPostprocess(simDict, cfgMain, inputSimFiles)

    peakDir = avalancheDir / "Outputs" / "com8MoTPSA" / "peakFiles"
    outputFiles = sorted([f.name for f in peakDir.glob("*.asc")])

    expectedFiles = sorted([
        "%s_L1_ppr.asc" % simKey,
        "%s_L2_ppr.asc" % simKey,
        "%s_L1_pfd.asc" % simKey,
        "%s_L2_pfd.asc" % simKey,
        "%s_L1_pfv.asc" % simKey,
        "%s_L2_pfv.asc" % simKey,
    ])

    assert outputFiles == expectedFiles


def _setupCom8Preprocess(tmp_path, forestEffects):
    """Copy avaKot inputs and run com1DFA preprocess for com8 with res simType."""
    testDir = pathlib.Path(__file__).parents[0]
    avaSrc = testDir / ".." / "data" / "avaKot"
    avaDir = tmp_path / "avaKot"
    shutil.copytree(avaSrc / "Inputs", avaDir / "Inputs")

    cfgMain = configparser.ConfigParser()
    cfgMain["MAIN"] = {"avalancheDir": str(avaDir), "nCPU": "1"}
    cfgMain["FLAGS"] = {"showPlot": "False", "savePlot": "False"}

    cfgCom8 = cfgUtils.getModuleConfig(com8MoTPSA)
    cfgCom8["GENERAL"]["simTypeList"] = "res"
    cfgCom8["FOREST_EFFECTS"]["Forest effects"] = forestEffects

    simDict, _, inputSimFiles, _ = com1DFA.com1DFAPreprocess(cfgMain, cfgCom8, module=com8MoTPSA)
    return simDict, inputSimFiles, cfgMain


def test_com8MoTPSAPreprocess_forestEffectsAuto(tmp_path):
    """Forest effects auto resolves to no when no bhd raster is present."""
    simDict, inputSimFiles, cfgMain = _setupCom8Preprocess(tmp_path, "auto")

    rcfFiles = com8MoTPSA.com8MoTPSAPreprocess(simDict, inputSimFiles, cfgMain)

    assert len(rcfFiles) == 1
    cfgSim = list(simDict.values())[0]["cfgSim"]
    assert cfgSim["FOREST_EFFECTS"]["Forest effects"] == "no"
    assert cfgSim["File names"]["Forest density filename"] == "-"
    assert cfgSim["File names"]["Tree diameter filename"] == "-"


def test_com8MoTPSAPreprocess_forestEffectsNonAuto(tmp_path):
    """Forest effects other than auto raises AssertionError."""
    simDict, inputSimFiles, cfgMain = _setupCom8Preprocess(tmp_path, "no")

    with pytest.raises(AssertionError):
        com8MoTPSA.com8MoTPSAPreprocess(simDict, inputSimFiles, cfgMain)
