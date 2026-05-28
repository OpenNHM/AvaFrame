"""Tests for com9MoTVoellmy module"""

import configparser
import logging
import pathlib
import shutil
from unittest.mock import patch

import pytest

from avaframe.com9MoTVoellmy import com9MoTVoellmy
from avaframe.in3Utils import cfgUtils


def test_com9MoTVoellmyTask_windows(tmp_path):
    """Test com9MoTVoellmyTask on Windows platform"""
    rcfFile = tmp_path / "test_config.rcf"
    rcfFile.write_text("test config")

    with (
        patch("os.name", "nt"),
        patch("os.chdir") as mockChdir,
        patch("os.path.dirname") as mockDirname,
        patch("os.path.abspath") as mockAbspath,
        patch("avaframe.com9MoTVoellmy.com9MoTVoellmy.mT.runAndCheckMoT") as mockRun,
    ):
        mockAbspath.return_value = "/fake/module/path"
        mockDirname.return_value = "/fake/module"

        # Call the function
        result = com9MoTVoellmy.com9MoTVoellmyTask(rcfFile)

        # Verify chdir was called
        mockChdir.assert_called_once_with("/fake/module")

        # Verify runAndCheckMoT was called with Windows executable
        mockRun.assert_called_once()
        command = mockRun.call_args[0][0]
        assert command[0] == "MoT-Voellmy_win.exe"
        assert command[1] == rcfFile

        # Verify return value
        assert result == command


def test_com9MoTVoellmyTask_linux(tmp_path):
    """Test com9MoTVoellmyTask on Linux platform"""
    rcfFile = tmp_path / "test_config.rcf"
    rcfFile.write_text("test config")

    with (
        patch("os.name", "posix"),
        patch("platform.system", return_value="Linux"),
        patch("os.chdir") as mockChdir,
        patch("os.path.dirname") as mockDirname,
        patch("os.path.abspath") as mockAbspath,
        patch("avaframe.com9MoTVoellmy.com9MoTVoellmy.mT.runAndCheckMoT") as mockRun,
    ):
        mockAbspath.return_value = "/fake/module/path"
        mockDirname.return_value = "/fake/module"

        # Call the function
        result = com9MoTVoellmy.com9MoTVoellmyTask(rcfFile)

        # Verify chdir was called
        mockChdir.assert_called_once_with("/fake/module")

        # Verify runAndCheckMoT was called with Linux executable
        mockRun.assert_called_once()
        command = mockRun.call_args[0][0]
        assert command[0] == "./MoT-Voellmy_linux.exe"
        assert command[1] == rcfFile

        # Verify return value
        assert result == command


def test_com9MoTVoellmyTask_macOS_uses_mac_binary(tmp_path):
    """Test that com9MoTVoellmyTask uses the macOS binary on Darwin"""
    rcfFile = tmp_path / "test_config.rcf"
    rcfFile.write_text("test config")

    with (
        patch("os.name", "posix"),
        patch("platform.system", return_value="Darwin"),
        patch("os.chdir"),
        patch("os.path.dirname"),
        patch("os.path.abspath"),
        patch("avaframe.com9MoTVoellmy.com9MoTVoellmy.mT.runAndCheckMoT") as mock_run,
    ):
        command = com9MoTVoellmy.com9MoTVoellmyTask(rcfFile)
        assert command[0] == "./MoT-Voellmy_mac.exe"


def test_com9MoTVoellmyTask_verifyLogging(tmp_path, caplog):
    """Test that com9MoTVoellmyTask logs the simulation run"""
    rcfFile = tmp_path / "test_config.rcf"
    rcfFile.write_text("test config")

    with (
        patch("os.name", "nt"),
        patch("os.chdir"),
        patch("os.path.dirname"),
        patch("os.path.abspath"),
        patch("avaframe.com9MoTVoellmy.com9MoTVoellmy.mT.runAndCheckMoT"),
    ):
        with caplog.at_level(logging.INFO):
            com9MoTVoellmy.com9MoTVoellmyTask(rcfFile)

        # Verify log message
        assert any(
            "Run simulation" in record.message and str(rcfFile) in record.message
            for record in caplog.records
        )


def test_com9MoTVoellmyTask_rcfFilePathHandling(tmp_path):
    """Test that rcfFile path is passed correctly regardless of type"""
    # Test with pathlib.Path
    rcfFilePath = tmp_path / "config.rcf"
    rcfFilePath.write_text("test")

    # Test with string path
    rcfFileStr = str(tmp_path / "config2.rcf")
    pathlib.Path(rcfFileStr).write_text("test")

    with (
        patch("os.name", "nt"),
        patch("os.chdir"),
        patch("os.path.dirname"),
        patch("os.path.abspath"),
        patch("avaframe.com9MoTVoellmy.com9MoTVoellmy.mT.runAndCheckMoT") as mockRun,
    ):
        # Test with pathlib.Path
        com9MoTVoellmy.com9MoTVoellmyTask(rcfFilePath)
        assert mockRun.call_args[0][0][1] == rcfFilePath

        # Test with string
        mockRun.reset_mock()
        com9MoTVoellmy.com9MoTVoellmyTask(rcfFileStr)
        assert mockRun.call_args[0][0][1] == rcfFileStr


def test_com9MoTVoellmyPostprocess_directoryCreation(tmp_path):
    """Test that postprocess creates necessary output directories"""
    avalancheDir = tmp_path / "avaTest"
    avalancheDir.mkdir()

    # Setup config
    cfgMain = configparser.ConfigParser()
    cfgMain["MAIN"] = {"avalancheDir": str(avalancheDir)}
    cfgMain["FLAGS"] = {}

    # Setup simDict with one simulation
    simDict = {"sim001": {"cfgSim": configparser.ConfigParser()}}

    # Create mock work directory with no files
    workDir = avalancheDir / "Work" / "com9MoTVoellmy" / "sim001"
    workDir.mkdir(parents=True)

    with (
        patch("avaframe.com9MoTVoellmy.com9MoTVoellmy.mT.copyMoTFiles"),
        patch("avaframe.com9MoTVoellmy.com9MoTVoellmy.mT.copyMoTDirs"),
        patch("avaframe.com9MoTVoellmy.com9MoTVoellmy.oP.plotAllPeakFields"),
    ):
        # Call function
        com9MoTVoellmy.com9MoTVoellmyPostprocess(simDict, cfgMain)

        # Verify output directory created
        outputDirPeakFile = avalancheDir / "Outputs" / "com9MoTVoellmy" / "peakFiles"
        assert outputDirPeakFile.exists()
        assert outputDirPeakFile.is_dir()

        # Verify report directory created
        reportDir = avalancheDir / "Outputs" / "com9MoTVoellmy" / "reports"
        assert reportDir.exists()
        assert reportDir.is_dir()


def test_com9MoTVoellmyPostprocess_filesCopied(tmp_path):
    """Test that postprocess copies all expected files"""
    avalancheDir = tmp_path / "avaTest"
    avalancheDir.mkdir()

    # Setup config
    cfgMain = configparser.ConfigParser()
    cfgMain["MAIN"] = {"avalancheDir": str(avalancheDir)}
    cfgMain["FLAGS"] = {}

    # Setup simDict with two simulations
    simDict = {
        "sim001": {"cfgSim": configparser.ConfigParser()},
        "sim002": {"cfgSim": configparser.ConfigParser()},
    }

    # Create mock work directories
    for simKey in simDict.keys():
        workDir = avalancheDir / "Work" / "com9MoTVoellmy" / simKey
        workDir.mkdir(parents=True)

    with (
        patch("avaframe.com9MoTVoellmy.com9MoTVoellmy.mT.copyMoTFiles") as mockCopyFiles,
        patch("avaframe.com9MoTVoellmy.com9MoTVoellmy.mT.copyMoTDirs"),
        patch("avaframe.com9MoTVoellmy.com9MoTVoellmy.oP.plotAllPeakFields"),
    ):
        # Call function
        com9MoTVoellmy.com9MoTVoellmyPostprocess(simDict, cfgMain)

        outputDirPeakFile = avalancheDir / "Outputs" / "com9MoTVoellmy" / "peakFiles"

        # Verify copyMoTFiles called for each simulation and file type
        # For each sim: ppr, pfd, pfv files = 3 calls per sim * 2 sims = 6 calls
        assert mockCopyFiles.call_count == 6

        # Verify correct file types copied for first simulation
        workDir1 = avalancheDir / "Work" / "com9MoTVoellmy" / "sim001"
        mockCopyFiles.assert_any_call(workDir1, outputDirPeakFile, "p_max", "ppr")
        mockCopyFiles.assert_any_call(workDir1, outputDirPeakFile, "h_max", "pfd")
        mockCopyFiles.assert_any_call(workDir1, outputDirPeakFile, "s_max", "pfv")

        # Verify correct file types copied for second simulation
        workDir2 = avalancheDir / "Work" / "com9MoTVoellmy" / "sim002"
        mockCopyFiles.assert_any_call(workDir2, outputDirPeakFile, "p_max", "ppr")
        mockCopyFiles.assert_any_call(workDir2, outputDirPeakFile, "h_max", "pfd")
        mockCopyFiles.assert_any_call(workDir2, outputDirPeakFile, "s_max", "pfv")


def test_com9MoTVoellmyPostprocess_directoriesCopied(tmp_path):
    """Test that postprocess copies timestep directories"""
    avalancheDir = tmp_path / "avaTest"
    avalancheDir.mkdir()

    # Setup config
    cfgMain = configparser.ConfigParser()
    cfgMain["MAIN"] = {"avalancheDir": str(avalancheDir)}
    cfgMain["FLAGS"] = {}

    # Setup simDict
    simDict = {"sim001": {"cfgSim": configparser.ConfigParser()}}

    # Create mock work directory
    workDir = avalancheDir / "Work" / "com9MoTVoellmy" / "sim001"
    workDir.mkdir(parents=True)

    with (
        patch("avaframe.com9MoTVoellmy.com9MoTVoellmy.mT.copyMoTFiles"),
        patch("avaframe.com9MoTVoellmy.com9MoTVoellmy.mT.copyMoTDirs") as mockCopyDirs,
        patch("avaframe.com9MoTVoellmy.com9MoTVoellmy.oP.plotAllPeakFields"),
    ):
        # Call function
        com9MoTVoellmy.com9MoTVoellmyPostprocess(simDict, cfgMain)

        outputDirPeakFile = avalancheDir / "Outputs" / "com9MoTVoellmy" / "peakFiles"

        # Verify copyMoTDirs called for s and h directories
        assert mockCopyDirs.call_count == 2
        mockCopyDirs.assert_any_call(workDir, outputDirPeakFile, "sim001", "s")
        mockCopyDirs.assert_any_call(workDir, outputDirPeakFile, "sim001", "h")


def test_com9MoTVoellmyPostprocess_plotsGenerated(tmp_path):
    """Test that postprocess generates plots"""
    avalancheDir = tmp_path / "avaTest"
    avalancheDir.mkdir()

    # Setup config
    cfgMain = configparser.ConfigParser()
    cfgMain["MAIN"] = {"avalancheDir": str(avalancheDir)}
    cfgMain["FLAGS"] = {"plotAllPeakFields": "True"}

    # Setup simDict
    simDict = {"sim001": {"cfgSim": configparser.ConfigParser()}}

    # Create mock work directory
    workDir = avalancheDir / "Work" / "com9MoTVoellmy" / "sim001"
    workDir.mkdir(parents=True)

    with (
        patch("avaframe.com9MoTVoellmy.com9MoTVoellmy.mT.copyMoTFiles"),
        patch("avaframe.com9MoTVoellmy.com9MoTVoellmy.mT.copyMoTDirs"),
        patch("avaframe.com9MoTVoellmy.com9MoTVoellmy.oP.plotAllPeakFields") as mockPlot,
    ):
        # Call function
        com9MoTVoellmy.com9MoTVoellmyPostprocess(simDict, cfgMain)

        # Verify plotAllPeakFields was called
        # Note: avalancheDir gets converted to string in the function
        mockPlot.assert_called_once_with(str(avalancheDir), cfgMain["FLAGS"], "com9MoTVoellmy")


def test_com9MoTVoellmyPostprocess_multipleSimulations(tmp_path):
    """Test postprocess with multiple simulations"""
    avalancheDir = tmp_path / "avaTest"
    avalancheDir.mkdir()

    # Setup config
    cfgMain = configparser.ConfigParser()
    cfgMain["MAIN"] = {"avalancheDir": str(avalancheDir)}
    cfgMain["FLAGS"] = {}

    # Setup simDict with three simulations
    simDict = {
        "sim001": {"cfgSim": configparser.ConfigParser()},
        "sim002": {"cfgSim": configparser.ConfigParser()},
        "sim003": {"cfgSim": configparser.ConfigParser()},
    }

    # Create mock work directories
    for simKey in simDict.keys():
        workDir = avalancheDir / "Work" / "com9MoTVoellmy" / simKey
        workDir.mkdir(parents=True)

    with (
        patch("avaframe.com9MoTVoellmy.com9MoTVoellmy.mT.copyMoTFiles") as mockCopyFiles,
        patch("avaframe.com9MoTVoellmy.com9MoTVoellmy.mT.copyMoTDirs") as mockCopyDirs,
        patch("avaframe.com9MoTVoellmy.com9MoTVoellmy.oP.plotAllPeakFields"),
    ):
        # Call function
        com9MoTVoellmy.com9MoTVoellmyPostprocess(simDict, cfgMain)

        # Verify copyMoTFiles called correct number of times: 3 file types * 3 sims = 9
        assert mockCopyFiles.call_count == 9

        # Verify copyMoTDirs called correct number of times: 2 dirs * 3 sims = 6
        assert mockCopyDirs.call_count == 6


def test_com9MoTVoellmyIntegrationRun(tmp_path, caplog):
    """Integration test: run com9MoTVoellmy with avaKot (null) and verify output files"""
    testDir = pathlib.Path(__file__).parents[0]
    avaSrc = testDir / ".." / "data" / "avaKot"
    avaDir = tmp_path / "avaKot"
    shutil.copytree(avaSrc / "Inputs", avaDir / "Inputs")

    binPath = pathlib.Path(com9MoTVoellmy.__file__).parent / "MoT-Voellmy_linux.exe"
    if not binPath.exists():
        pytest.skip("MoT-Voellmy binary not found -- run `pixi run compilemot`")

    cfgMain = configparser.ConfigParser()
    cfgMain["MAIN"] = {"avalancheDir": str(avaDir), "nCPU": "auto", "CPUPercent": "90"}
    cfgMain["FLAGS"] = {"showPlot": "False", "savePlot": "True"}

    cfgCom9 = cfgUtils.getModuleConfig(com9MoTVoellmy)
    cfgCom9["GENERAL"]["simTypeList"] = "null"

    with caplog.at_level(logging.INFO):
        com9MoTVoellmy.com9MoTVoellmyMain(cfgMain, cfgInfo=cfgCom9)

    # Verify no errors during run (only main process log is captured)
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errors) == 0, f"Errors in log: {[e.message for e in errors]}"

    # Verify parallel computation ran (logged in main process)
    assert any(
        "Overall" in r.message and "computation took" in r.message
        for r in caplog.records
    ), "No parallel computation summary found"

    # Verify log says simulations were performed
    assert any(
        "The following simulations will be performed" in r.message
        for r in caplog.records
    )

    # Verify output files exist
    peakDir = avaDir / "Outputs" / "com9MoTVoellmy" / "peakFiles"
    assert peakDir.exists()
    ascFiles = sorted(peakDir.glob("*.asc"))
    assert len(ascFiles) >= 3, f"Expected >= 3 asc files, got {len(ascFiles)}: {ascFiles}"

    cfgDir = avaDir / "Outputs" / "com9MoTVoellmy" / "configurationFiles"
    rcfFiles = sorted(cfgDir.glob("*.rcf"))
    assert len(rcfFiles) >= 1, f"No RCF files found in {cfgDir}"
