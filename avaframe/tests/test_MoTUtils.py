import pytest
import numpy as np
import pathlib
from unittest.mock import patch, MagicMock, Mock, call
from avaframe.in3Utils import MoTUtils
import os
import platform


# Note 3.7.25: these tests are mostly AI-generated

def test_rewriteDEMtoZeroValues(tmp_path):
    # Create mock input data
    mockDemData = {
        "rasterData": np.array([[1.0, np.nan, 3.0],
                              [np.nan, 5.0, 6.0]]),
        "header": {
            "nodata_value": np.nan
        }
    }
    
    expectedRasterData = np.array([[1.0, 0.0, 3.0],
                                   [0.0, 5.0, 6.0]])
    
    # Create a mock Path object    avaTestDir = "avaHockeyChannelPytest"
    # mockPath = pathlib.Path(tmp_path)
    # outFile  = pathlib.Path(tmp_path, 'testDir', 'testDem.asc')
    # print(outFile)
    mockPath = MagicMock(
        spec=pathlib.Path,
        parent=MagicMock(
            spec=pathlib.Path,
            name="testDir"
        ),
        name="testDem.asc"
    )
    # mockPath = MagicMock(spec=pathlib.Path)
    # print(mockPath)
    # mockPath.stem            # = "testDem"
    
    # Mock the rasterUtils functions
    with patch('avaframe.in2Trans.rasterUtils.readRaster') as mockRead:
        with patch('avaframe.in2Trans.rasterUtils.writeResultToRaster') as mockWrite:
            # Set up the mock return value for readRaster
            mockRead.return_value = mockDemData
            
            # Call the function
            MoTUtils.rewriteDEMtoZeroValues(mockPath)
            # MoTUtils.rewriteDEMtoZeroValues(outFile)

            # Verify readRaster was called with correct argument
            mockRead.assert_called_once_with(mockPath)
            # mockRead.assert_called_once_with(outFile)

            # Verify the data was modified correctly
            np.testing.assert_array_equal(
                mockDemData["rasterData"],
                expectedRasterData
            )
            
            # Verify nodata value was updated
            assert mockDemData["header"]["nodata_value"] == 0.0
            
            # Verify writeResultToRaster was called with correct arguments
            mockWrite.assert_called_once()
            call_args = mockWrite.call_args[0]
            assert call_args[0] == mockDemData["header"]


def test_CopyMoTFiles(tmp_path):
    # Create temporary source and destination directories
    workDir = tmp_path / "work"
    outputDir = tmp_path / "output"
    workDir.mkdir()
    outputDir.mkdir()

    # Create some test files in work directory
    testFiles = [
        "test_p_max_001.txt",
        "test_p_max_002.txt",
        "another_p_max_file.txt"
    ]

    for fileName in testFiles:
        (workDir / fileName).touch()

    # Test parameters
    searchString = "p_max"
    replaceString = "ppr"

    # Create expected target file names
    expectedTargets = [
        outputDir / name.replace(searchString, replaceString)
        for name in testFiles
    ]

    # Mock shutil.copy2 to avoid actual file copying
    with patch('shutil.copy2') as mockCopy:
        # Run the function
        MoTUtils.copyMoTFiles(workDir, outputDir, searchString, replaceString)

        # Check that copy2 was called the correct number of times
        assert mockCopy.call_count == len(testFiles)

        # Verify each copy operation
        # Get actual calls and sort them by source path
        actualCopies = sorted(mockCopy.call_args_list,
                              key=lambda x: str(x[0][0]))
        sourceFilePaths = sorted(workDir.glob(f"*{searchString}*"))
        expectedTargets = sorted(expectedTargets)

        for i in range(len(testFiles)):
            callArgs = actualCopies[i][0]
            assert callArgs[0] == sourceFilePaths[i]
            assert callArgs[1] == expectedTargets[i]


def test_CopyMoTFiles_EmptyDirectory(tmp_path):
    # Test behavior with empty source directory
    workDir = tmp_path / "empty_work"
    outputDir = tmp_path / "empty_output"
    workDir.mkdir()
    outputDir.mkdir()

    with patch('shutil.copy2') as mockCopy:
        MoTUtils.copyMoTFiles(workDir, outputDir, "p_max", "ppr")
        mockCopy.assert_not_called()

def test_CopyMoTFiles_NoMatchingFiles(tmp_path):
    # Test behavior when no files match the search string
    workDir = tmp_path / "work"
    outputDir = tmp_path / "output"
    workDir.mkdir()
    outputDir.mkdir()

    # Create files that don't match the search pattern
    (workDir / "test1.txt").touch()
    (workDir / "test2.txt").touch()

    with patch('shutil.copy2') as mockCopy:
        MoTUtils.copyMoTFiles(workDir, outputDir, "p_max", "ppr")
        mockCopy.assert_not_called()

class MockProcess:
    def __init__(self, outputLines):
        self.outputLines = outputLines
        self.currentLine = 0
        self.stdout = self

    def readline(self):
        if self.currentLine >= len(self.outputLines):
            return ""
        line = self.outputLines[self.currentLine]
        self.currentLine += 1
        return line

    def poll(self):
        return None if self.currentLine < len(self.outputLines) else 0

@pytest.mark.parametrize("osName,platformSystem,expectedShell", [
    ("nt", "Windows", True),
    ("posix", "Darwin", False),
    ("posix", "Linux", False)
])
def test_RunAndCheckMoT_OSSpecific(osName, platformSystem, expectedShell):
    with patch('os.name', osName), \
            patch('platform.system', return_value=platformSystem), \
            patch('subprocess.Popen') as mockPopen:

        mockPopen.return_value = MockProcess([])
        MoTUtils.runAndCheckMoT("testCommand")

        mockPopen.assert_called_once()
        assert mockPopen.call_args[1]['shell'] == expectedShell



def test_RunAndCheckMoT_CommandTypes():
    testCases = [
        "singleCommand",
        ["command", "with", "arguments"],
        "command with spaces"
    ]

    for command in testCases:
        with patch('subprocess.Popen') as mockPopen:
            mockPopen.return_value = MockProcess([])
            MoTUtils.runAndCheckMoT(command)

            mockPopen.assert_called_once()
            assert mockPopen.call_args[0][0] == command

def test_RunAndCheckMoT_ProcessExit():
    # Test proper handling of process termination
    testOutput = ["Line 1\n", "Line 2\n"]

    with patch('subprocess.Popen') as mockPopen, \
            patch('avaframe.in3Utils.MoTUtils.log.info') as mockLog:

        mockPopen.return_value = MockProcess(testOutput)
        MoTUtils.runAndCheckMoT("testCommand")

        # Verify all output was processed
        assert mockLog.call_count == 2
        mockLog.assert_has_calls([
            call("Line 1"),
            call("Line 2")
        ])


def test_copyMoTDirs(tmp_path):
    """Test copying timestep directories from work to output directory"""
    # Create source and destination directories
    workDir = tmp_path / "work"
    outputDir = tmp_path / "output"
    workDir.mkdir()
    outputDir.mkdir()

    # Create source directory with files
    sourceDirS = workDir / "s"
    sourceDirS.mkdir()
    (sourceDirS / "file1.txt").write_text("content1")
    (sourceDirS / "file2.txt").write_text("content2")

    # Create a subdirectory that should NOT be copied (only files)
    subDir = sourceDirS / "subdir"
    subDir.mkdir()
    (subDir / "nested.txt").write_text("nested")

    simKey = "sim001"
    dirName = "s"

    # Call the function
    MoTUtils.copyMoTDirs(workDir, outputDir, simKey, dirName)

    # Verify output directory structure created
    expectedTargetDir = outputDir / "timesteps" / simKey / dirName
    assert expectedTargetDir.exists()
    assert expectedTargetDir.is_dir()

    # Verify files were copied
    assert (expectedTargetDir / "file1.txt").exists()
    assert (expectedTargetDir / "file2.txt").exists()
    assert (expectedTargetDir / "file1.txt").read_text() == "content1"
    assert (expectedTargetDir / "file2.txt").read_text() == "content2"

    # Verify subdirectories were NOT copied
    assert not (expectedTargetDir / "subdir").exists()


def test_copyMoTDirs_nonexistentSource(tmp_path):
    """Test that function handles gracefully when source directory doesn't exist"""
    workDir = tmp_path / "work"
    outputDir = tmp_path / "output"
    workDir.mkdir()
    outputDir.mkdir()

    simKey = "sim001"
    dirName = "h"

    # Call function with non-existent source directory
    MoTUtils.copyMoTDirs(workDir, outputDir, simKey, dirName)

    # Function should not create target directory if source doesn't exist
    expectedTargetDir = outputDir / "timesteps" / simKey / dirName
    assert not expectedTargetDir.exists()


def test_copyMoTDirs_emptySourceDir(tmp_path):
    """Test copying empty source directory"""
    workDir = tmp_path / "work"
    outputDir = tmp_path / "output"
    workDir.mkdir()
    outputDir.mkdir()

    # Create empty source directory
    sourceDirH = workDir / "h"
    sourceDirH.mkdir()

    simKey = "sim002"
    dirName = "h"

    # Call the function
    MoTUtils.copyMoTDirs(workDir, outputDir, simKey, dirName)

    # Verify target directory created even if empty
    expectedTargetDir = outputDir / "timesteps" / simKey / dirName
    assert expectedTargetDir.exists()
    assert expectedTargetDir.is_dir()

    # Verify it's empty
    assert len(list(expectedTargetDir.iterdir())) == 0


def test_setVariableFrictionParameters_bothFilesFound(tmp_path):
    """Test setting friction parameters when both mu and k files are found"""
    import configparser

    # Setup test directories - inputsDir should be the Inputs folder
    inputsDir = tmp_path / "Inputs"
    rastersDir = inputsDir / "RASTERS"
    rastersDir.mkdir(parents=True)
    workInputDir = tmp_path / "Work" / "Input"
    workInputDir.mkdir(parents=True)

    # Create mock mu and k files
    muFile = rastersDir / "test_mu.asc"
    kFile = rastersDir / "test_k.asc"
    muFile.write_text("mock mu data")
    kFile.write_text("mock k data")

    # Setup config
    cfg = configparser.ConfigParser()
    cfg["Physical_parameters"] = {
        "Dry-friction coefficient (-)": "0.5",
        "Turbulent drag coefficient (-)": "0.3",
        "Parameters": "auto"
    }
    cfg["INPUT"] = {
        "muFile": "RASTERS/test_mu.asc",
        "kFile": "RASTERS/test_k.asc"
    }

    # Setup inputSimFiles
    inputSimFiles = {
        "entResInfo": {
            "mu": "Yes",
            "k": "Yes",
            "muRemeshed": "No",
            "kRemeshed": "No"
        }
    }

    # Call function
    result = MoTUtils.setVariableFrictionParameters(cfg, inputSimFiles, workInputDir, inputsDir)

    # Verify parameters were set to variable
    assert result["Physical_parameters"]["Parameters"] == "variable"
    assert str(muFile) in result["Physical_parameters"]["Dry-friction coefficient (-)"]
    assert str(kFile) in result["Physical_parameters"]["Turbulent drag coefficient (-)"]


def test_setVariableFrictionParameters_filesNotFound(tmp_path):
    """Test setting friction parameters when files are not found"""
    import configparser

    inputsDir = tmp_path / "Inputs" / "RASTERS"
    inputsDir.mkdir(parents=True)
    workInputDir = tmp_path / "Work" / "Input"
    workInputDir.mkdir(parents=True)

    # Setup config
    cfg = configparser.ConfigParser()
    cfg["Physical_parameters"] = {
        "Dry-friction coefficient (-)": "0.5",
        "Turbulent drag coefficient (-)": "0.3",
        "Parameters": "auto"
    }
    cfg["INPUT"] = {
        "muFile": "RASTERS/test_mu.asc",
        "kFile": "RASTERS/test_k.asc"
    }

    # Setup inputSimFiles with files not found
    inputSimFiles = {
        "entResInfo": {
            "mu": "No",
            "k": "No",
            "muRemeshed": "No",
            "kRemeshed": "No"
        }
    }

    # Call function
    result = MoTUtils.setVariableFrictionParameters(cfg, inputSimFiles, workInputDir, inputsDir)

    # Verify parameters were set to constant
    assert result["Physical_parameters"]["Parameters"] == "constant"
    assert result["Physical_parameters"]["Dry-friction coefficient (-)"] == "0.5"
    assert result["Physical_parameters"]["Turbulent drag coefficient (-)"] == "0.3"


def test_setVariableFrictionParameters_withRemeshed(tmp_path):
    """Test setting friction parameters with remeshed files"""
    import configparser

    inputsDir = tmp_path / "Inputs"
    rastersDir = inputsDir / "RASTERS"
    rastersDir.mkdir(parents=True)
    workInputDir = tmp_path / "Work" / "Input"
    workInputDir.mkdir(parents=True)

    # Create mock remeshed files
    muFile = rastersDir / "test_remeshed_mu.asc"
    kFile = rastersDir / "test_remeshed_k.asc"
    muFile.write_text("mock mu data")
    kFile.write_text("mock k data")

    # Setup config
    cfg = configparser.ConfigParser()
    cfg["Physical_parameters"] = {
        "Dry-friction coefficient (-)": "0.5",
        "Turbulent drag coefficient (-)": "0.3",
        "Parameters": "auto"
    }
    cfg["INPUT"] = {
        "muFile": "RASTERS/test_remeshed_mu.asc",
        "kFile": "RASTERS/test_remeshed_k.asc"
    }

    # Setup inputSimFiles with remeshed files
    inputSimFiles = {
        "entResInfo": {
            "mu": "Yes",
            "k": "Yes",
            "muRemeshed": "Yes",
            "kRemeshed": "Yes"
        }
    }

    # Call function
    result = MoTUtils.setVariableFrictionParameters(cfg, inputSimFiles, workInputDir, inputsDir)

    # Verify parameters were set to variable
    assert result["Physical_parameters"]["Parameters"] == "variable"

    # Verify remeshed files were copied to work directory
    expectedMuPath = workInputDir / "test_remeshed_mu_mu.asc"
    expectedKPath = workInputDir / "test_remeshed_k_k.asc"
    assert expectedMuPath.exists()
    assert expectedKPath.exists()
    assert str(expectedMuPath) in result["Physical_parameters"]["Dry-friction coefficient (-)"]
    assert str(expectedKPath) in result["Physical_parameters"]["Turbulent drag coefficient (-)"]


def test_setVariableEntrainmentParameters_withEntrainment(tmp_path):
    """Test setting entrainment parameters when entrainment is enabled"""
    import configparser

    # Setup config
    cfg = configparser.ConfigParser()
    cfg["ENTRAINMENT"] = {
        "Entrainment": "auto",
        "Bed strength profile": "variable"
    }
    cfg["INPUT"] = {}

    # Setup inputSimFiles with entrainment enabled
    inputSimFiles = {
        "entResInfo": {
            "flagEnt": "Yes",
            "tauC": "Yes"
        }
    }

    # Call function
    result = MoTUtils.setVariableEntrainmentParameters(cfg, inputSimFiles, None, None)

    # Verify entrainment set to TJEM
    assert result["ENTRAINMENT"]["Entrainment"] == "TJEM"
    assert result["ENTRAINMENT"]["Bed strength profile"] == "constant"


def test_setVariableEntrainmentParameters_noEntrainment(tmp_path):
    """Test setting entrainment parameters when entrainment is disabled"""
    import configparser

    # Setup config
    cfg = configparser.ConfigParser()
    cfg["ENTRAINMENT"] = {
        "Entrainment": "auto",
        "Bed strength profile": "variable"
    }
    cfg["INPUT"] = {}

    # Setup inputSimFiles with entrainment disabled
    inputSimFiles = {
        "entResInfo": {
            "flagEnt": "No",
            "tauC": "No"
        }
    }

    # Call function
    result = MoTUtils.setVariableEntrainmentParameters(cfg, inputSimFiles, None, None)

    # Verify entrainment set to none
    assert result["ENTRAINMENT"]["Entrainment"] == "none"


def test_setVariableForestParameters_withForest(tmp_path):
    """Test setting forest parameters when forest files are found"""
    import configparser

    inputsDir = tmp_path / "Inputs"
    rastersDir = inputsDir / "RASTERS"
    rastersDir.mkdir(parents=True)
    workInputDir = tmp_path / "Work" / "Input"
    workInputDir.mkdir(parents=True)

    # Create mock bhd file
    bhdFile = rastersDir / "test_bhd.asc"
    bhdFile.write_text("mock bhd data")

    # Create mock resistance file
    resFile = inputsDir / "RES" / "resistance.shp"
    resFile.parent.mkdir(parents=True)
    resFile.write_text("mock resistance")

    # Setup config
    cfg = configparser.ConfigParser()
    cfg["FOREST_EFFECTS"] = {
        "Forest effects": "auto"
    }
    cfg["File names"] = {
        "Forest density filename": "",
        "Tree diameter filename": ""
    }
    cfg["INPUT"] = {
        "bhdFile": "RASTERS/test_bhd.asc"
    }

    # Setup inputSimFiles with forest enabled
    inputSimFiles = {
        "entResInfo": {
            "flagRes": "Yes",
            "bhd": "Yes"
        },
        "resFile": resFile
    }

    # Call function
    result = MoTUtils.setVariableForestParameters(cfg, inputSimFiles, workInputDir, inputsDir)

    # Verify forest effects enabled
    assert result["FOREST_EFFECTS"]["Forest effects"] == "yes"
    assert str(resFile) in result["File names"]["Forest density filename"]
    assert str(bhdFile) in result["File names"]["Tree diameter filename"]


def test_setVariableForestParameters_noForest(tmp_path):
    """Test setting forest parameters when forest is disabled"""
    import configparser

    inputsDir = tmp_path / "Inputs" / "RASTERS"
    inputsDir.mkdir(parents=True)
    workInputDir = tmp_path / "Work" / "Input"
    workInputDir.mkdir(parents=True)

    # Setup config
    cfg = configparser.ConfigParser()
    cfg["FOREST_EFFECTS"] = {
        "Forest effects": "auto"
    }
    cfg["File names"] = {
        "Forest density filename": "",
        "Tree diameter filename": ""
    }
    cfg["INPUT"] = {}

    # Setup inputSimFiles with forest disabled
    inputSimFiles = {
        "entResInfo": {
            "flagRes": "No",
            "bhd": "No"
        }
    }

    # Call function
    result = MoTUtils.setVariableForestParameters(cfg, inputSimFiles, workInputDir, inputsDir)

    # Verify forest effects disabled
    assert result["FOREST_EFFECTS"]["Forest effects"] == "no"
    assert result["File names"]["Forest density filename"] == "-"
    assert result["File names"]["Tree diameter filename"] == "-"


def test_MoTGenerateConfigs(tmp_path):
    """Test MoTGenerateConfigs function"""
    import configparser
    import sys
    from unittest.mock import MagicMock

    # Create a mock module
    mockModule = MagicMock()
    mockModule.__name__ = "avaframe.com9MoTVoellmy.com9MoTVoellmy"

    # Setup config
    cfgMain = configparser.ConfigParser()
    cfgMain["MAIN"] = {
        "avalancheDir": str(tmp_path / "avaTest"),
        "simTypeList": "null"
    }

    cfgInfo = None

    # Mock the com1DFA functions
    mockSimDict = {
        "sim001": {
            "cfgSim": configparser.ConfigParser()
        }
    }
    mockInputSimFiles = {
        "demFile": tmp_path / "dem.asc",
        "relFiles": []
    }

    with patch('avaframe.in3Utils.MoTUtils.com1DFA.com1DFAPreprocess') as mockPreprocess:

        mockPreprocess.return_value = (mockSimDict, tmp_path / "out", mockInputSimFiles, None)

        # Call function
        simDict, inputSimFiles = MoTUtils.MoTGenerateConfigs(cfgMain, cfgInfo, mockModule)

        # Verify results
        assert simDict == mockSimDict
        assert inputSimFiles == mockInputSimFiles
        mockPreprocess.assert_called_once()


def test_RunAndCheckMoT_HighTimeStepCount():
    """Test that time step counter logs after 100 steps"""
    # Create output with 105 lines containing "Step" to trigger printCounter > 100
    testOutput = [f"Step {i}\n" for i in range(1, 106)]

    with patch('subprocess.Popen') as mockPopen, \
            patch('avaframe.in3Utils.MoTUtils.log.info') as mockLog:

        mockPopen.return_value = MockProcess(testOutput)
        MoTUtils.runAndCheckMoT("testCommand")

        # Verify that the time step logging was triggered
        # After 101 steps, printCounter should exceed 100 and log a message
        logCalls = [call for call in mockLog.call_args_list
                   if "Process is running" in str(call)]
        assert len(logCalls) > 0, "Expected 'Process is running' log message after 100 steps"

        # Verify the message contains time step information
        firstLogCall = logCalls[0]
        assert "Reported time steps:" in str(firstLogCall)


def test_RunAndCheckMoT_MessageFiltering():
    """Test that specific keywords are filtered from output"""
    # Create output with lines containing keywords that should be filtered
    testOutput = [
        "Normal line 1\n",
        "find_dt calculation\n",
        "h1 value\n",
        "h2 value\n",
        "write_data operation\n",
        "update_boundaries process\n",
        "V_tot volume\n",
        "Normal line 2\n",
    ]

    with patch('subprocess.Popen') as mockPopen, \
            patch('avaframe.in3Utils.MoTUtils.log.info') as mockLog:

        mockPopen.return_value = MockProcess(testOutput)
        MoTUtils.runAndCheckMoT("testCommand")

        # Verify only non-filtered messages were logged
        loggedMessages = [call[0][0] for call in mockLog.call_args_list]

        # These should be logged
        assert "Normal line 1" in loggedMessages
        assert "Normal line 2" in loggedMessages

        # These should NOT be logged (filtered out)
        assert not any("find_dt" in msg for msg in loggedMessages)
        assert not any("h1" in msg for msg in loggedMessages)
        assert not any("h2" in msg for msg in loggedMessages)
        assert not any("write_data" in msg for msg in loggedMessages)
        assert not any("update_boundaries" in msg for msg in loggedMessages)
        assert not any("V_tot" in msg for msg in loggedMessages)