import os
import platform
import subprocess
import logging
import pathlib
import shutil

import numpy as np

from avaframe.com1DFA import com1DFA as com1DFA
from avaframe.com1DFA import com1DFATools as com1DFATools
from avaframe.in2Trans import rasterUtils as rU

log = logging.getLogger(__name__)


def rewriteDEMtoZeroValues(demFile):
    """Set all NaN values in a DEM raster to zero and update the nodata value.

    This function reads a DEM raster file, replaces all NaN values with 0.0,
    updates the nodata value in the header to 0.0, and writes the modified
    raster back to a new file.

    Parameters
    ----------
    demFile : pathlib.Path
        Path to the input DEM raster file

    Returns
    -------
    None
        Writes a new raster file with zero values instead of NaN values.
        The output file is saved in the same directory as the input file,
        using the same stem name.

    Notes
    -----
    The function uses the rasterUtils module for reading and writing raster data.
    The output raster is flipped during writing.
    """
    demData = rU.readRaster(demFile)
    demData["rasterData"][np.isnan(demData["rasterData"])] = 0.0
    demData["header"]["nodata_value"] = 0.0
    newFileName = demFile.parent / demFile.stem
    rU.writeResultToRaster(demData["header"], demData["rasterData"], newFileName, flip=True)


def runAndCheckMoT(command):
    """Execute MoT command and monitor its output.

    This function runs a MoT command as a subprocess and monitors its output,
    filtering and logging specific messages while tracking time steps.

    Parameters
    ----------
    command : str or list
        The command to execute. Can be a string or list of arguments.

    Returns
    -------
    None
        Function runs the command and logs output but does not return a value.

    Notes
    -----
    - Uses different shell settings based on operating system
    - Filters output to reduce noise from common status messages
    - Logs time step progress every 100 steps
    - Handles UTF-8 encoding with replacement of invalid characters
    """
    if os.name == "nt":
        useShell = True
    elif platform.system() == "Darwin":
        useShell = False
    else:
        useShell = False

    # This starts the subprocess
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=useShell,
        encoding="utf-8",
        errors="replace",
        universal_newlines=True,
    )

    printCounter = 0
    counter = 1

    while True:
        realtimeOutput = process.stdout.readline()

        if realtimeOutput == "" and process.poll() is not None:
            break

        if realtimeOutput:
            line = realtimeOutput.strip()

            # do not pollute output window with time step prints
            # TODO: hacky for now
            if "Step" in line:
                counter = counter + 1
                printCounter = printCounter + 1
                if printCounter > 100:
                    # print('\r' + line, flush=True, end='')
                    msg = "Process is running. Reported time steps: " + str(counter)
                    log.info(msg)
                    printCounter = 0

            elif "find_dt" in line:
                continue
            elif "h1" in line:
                continue
            elif "h2" in line:
                continue
            elif "write_data" in line:
                continue
            elif "update_boundaries" in line:
                continue
            elif "V_tot" in line:
                continue
            else:
                log.info(line)


def MoTGenerateConfigs(cfgMain, cfgInfo, currentModule):
    """
    Creates configuration objects for com8MoTPSA and com9MoTVoellmy.

    Parameters
    ------------
    cfgMain: configparser object
        main configuration of AvaFrame
    cfgInfo: str or pathlib Path or configparser object
        path to configuration file if overwrite is desired - optional
        if not local (if available) or default configuration will be loaded
        if cfgInfo is a configparser object take this as initial config
    currentModule: module object
        is being passed to cfgUtils.writeCfgFile to create the correct the cfg


    Returns
    --------
    simDict: dict
        dictionary with one key per simulation to perform including its config object
    inputSimFiles: dict
        dictionary with input files info
    """

    # Route based on cfgInfo type: directory Path = batch mode, otherwise single config mode
    if isinstance(cfgInfo, pathlib.Path) and cfgInfo.is_dir():
        # Batch mode - cfgInfo is directory path (from getModuleConfig with batchCfgDir)
        simDict, inputSimFiles, simDFExisting, outDir = com1DFATools.createSimDictFromCfgs(
            cfgMain, cfgInfo, module=currentModule
        )
    else:
        # Single config mode - cfgInfo is ConfigParser, file path string, or empty string
        simDict, outDir, inputSimFiles, simDFExisting = com1DFA.com1DFAPreprocess(
            cfgMain, cfgInfo, module=currentModule
        )

    return simDict, inputSimFiles


def copyMoTFiles(workDir, outputDir, searchString, replaceString):
    """
    Copy and rename MoT result files from work directory to output directory.

    Parameters
    ----------
    workDir : pathlib.Path
        Source directory containing the original p_max files
    outputDir : pathlib.Path
        Destination directory where renamed ppr files will be copied to
    searchString : str
        String pattern to search for in the original filenames
    replaceString : str
        String to replace the searchString with in the new filenames

    Returns
    -------
    None
        Files are copied to the destination directory with renamed extensions
    """
    varFiles = list(workDir.glob("*" + searchString + "*"))
    targetFiles = [pathlib.Path(str(f.name).replace(searchString, replaceString)) for f in varFiles]
    targetFiles = [outputDir / f for f in targetFiles]

    for source, target in zip(varFiles, targetFiles):
        shutil.copy2(source, target)


def copyMoTDirs(workDir, outputDir, simKey, dirName):
    """
    Copy timestep directory from work directory to output directory.

    Parameters
    ----------
    workDir : pathlib.Path
        Source work directory containing the simulation results
    outputDir : pathlib.Path
        Destination directory where timestep directories will be copied to
    simKey : str
        Simulation key used to construct source and target paths
    dirName : str
        Directory name to copy (e.g., 's' or 'h')

    Returns
    -------
    None
        Directory is copied to the destination directory structure

    Notes
    -----
    Creates a timesteps/{simKey}/{dirName} subdirectory structure in the output directory.
    Only copies files, not subdirectories within the specified directory.
    """
    outputDirTimesteps = outputDir / "timesteps" / str(simKey)
    outputDirTimesteps.mkdir(parents=True, exist_ok=True)

    sourceDirPath = workDir / dirName
    if sourceDirPath.exists():
        targetDirPath = outputDirTimesteps / dirName
        targetDirPath.mkdir(parents=True, exist_ok=True)

        for sourceFile in sourceDirPath.glob("*"):
            if sourceFile.is_file():
                shutil.copy2(sourceFile, targetDirPath / sourceFile.name)


def setVariableFrictionParameters(cfg, inputSimFiles, workInputDir, inputsDir):
    """set file paths in cfg object for friction parameters (required if option variable is set)
    if _mu, _k files found in Inputs/RASTERS have to be remeshed, copy remeshed files
    to workInputDir with new file name ending _mu, _k

    Parameters
    -----------
    cfg: configparser object
        configuration info for simulation
    inputSimFiles: dict
        dictionary with info on all input data found; here mu, k file and if remeshed
    workInputDir: pathlib path
        pathlib path to work Inputs folder for current simulation
    inputsDir: pathlib path
        path to avalancheDir/Inputs where original input data and remeshed rasters are stored

    Returns
    --------
    cfg: configparser object
        updated configuration info for simulation with file paths to friction parameters
    """

    fricParameters = {"mu": "Dry-friction coefficient (-)", "k": "Turbulent drag coefficient (-)"}

    if inputSimFiles["entResInfo"]["mu"] == "Yes" and inputSimFiles["entResInfo"]["k"] == "Yes":

        for fric in ["mu", "k"]:
            fricFile = inputsDir / cfg["INPUT"]["%sFile" % fric]

            # check first if remeshed files should be used
            if (
                "_remeshed" in cfg["INPUT"]["%sFile" % fric]
                and inputSimFiles["entResInfo"]["%sRemeshed" % fric] == "Yes"
            ):
                fricFilePathNew = workInputDir / (fricFile.stem + "_%s" % fric + fricFile.suffix)
                shutil.copy2(fricFile, fricFilePathNew)
                cfg["Physical_parameters"][fricParameters[fric]] = str(fricFilePathNew)
                log.info(
                    "Remeshed %s file copied to %s and set for %s"
                    % (fric, str(fricFilePathNew), fricParameters[fric])
                )
            else:
                cfg["Physical_parameters"][fricParameters[fric]] = str(fricFile)

        cfg["Physical_parameters"]["Parameters"] = "variable"

    else:
        # TODO FSO implement if setting is variable or constant that if variable but file not found then error
        message = "Mu and k file not found in Inputs/RASTERS - check if file ending is correct (_mu, _k) - setting constant values of configuration file"
        log.warning(message)

        message2 = "Setting %s to constant value of %s, and %s to %s" % (
            fricParameters["mu"],
            cfg["Physical_parameters"][fricParameters["mu"]],
            fricParameters["k"],
            cfg["Physical_parameters"][fricParameters["k"]],
        )
        log.warning(message2)

        cfg["Physical_parameters"]["Parameters"] = "constant"

        # log.error(message)
        # raise FileNotFoundError(message)

    return cfg


def setVariableEntrainmentParameters(cfg, inputSimFiles, workInputDir, inputsDir):
    """set file path in cfg object for entrainment parameters (required if option variable is set)
    if _b0 , _tauc files found in Inputs/RASTERS and Inputs/ENT

    Parameters
    -----------
    cfg: configparser object
        configuration info for simulation
    inputSimFiles: dict
        dictionary with info on all input data found; here b0, tauc file and if remeshed
    workInputDir: pathlib path
        pathlib path to work Inputs folder for current simulation
    inputsDir: pathlib path
        path to avalancheDir/Inputs where original input data and remeshed rasters are stored

    Returns
    --------
    cfg: configparser object
        updated configuration info for simulation with file paths to friction parameters
    """

    if inputSimFiles["entResInfo"]["flagEnt"] == "Yes" and inputSimFiles["entResInfo"]["tauC"] == "Yes":
        cfg["ENTRAINMENT"]["Entrainment"] = "TJEM"
        cfg["ENTRAINMENT"]["Bed strength profile"] = "constant"
    else:
        cfg["ENTRAINMENT"]["Entrainment"] = "none"

    return cfg


def setVariableForestParameters(cfg, inputSimFiles, workInputDir, inputsDir):
    """set file paths in cfg object for forest parameters.
    if _nd, _bhd files found in Inputs/RASTERS have to be remeshed, copy remeshed files
    to workInputDir with new file name ending _nd, _bhd

    Parameters
    -----------
    cfg: configparser object
        configuration info for simulation
    inputSimFiles: dict
        dictionary with info on all input data found; here nd, bhd file and if remeshed
    workInputDir: pathlib path
        pathlib path to work Inputs folder for current simulation
    inputsDir: pathlib path
        path to avalancheDir/Inputs where original input data and remeshed rasters are stored

    Returns
    --------
    cfg: configparser object
        updated configuration info for simulation with file paths to forest parameters
    """

    if inputSimFiles["entResInfo"]["flagRes"] == "Yes" and inputSimFiles["entResInfo"]["bhd"] == "Yes":
        treeDiamFile = inputsDir / cfg["INPUT"]["bhdFile"]
        cfg["FOREST_EFFECTS"]["Forest effects"] = "yes"
        # TODO  Make this remeshed compatible
        cfg["File names"]["Forest density filename"] = str(inputSimFiles["resFile"])
        cfg["File names"]["Tree diameter filename"] = str(treeDiamFile)
    else:
        cfg["FOREST_EFFECTS"]["Forest effects"] = "no"
        cfg["File names"]["Forest density filename"] = "-"
        cfg["File names"]["Tree diameter filename"] = "-"

    # forestParameters = {"nd": "Forest density filename", "bhd": "Tree diameter filename"}
    # if inputSimFiles["entResInfo"]["nd"] == "Yes" and inputSimFiles["entResInfo"]["bhd"] == "Yes":
    #
    #     for forestParam in ["nd", "bhd"]:
    #         forestFile = inputsDir / cfg["INPUT"]["%sFile" % forestParam]
    #
    #         # check first if remeshed files should be used
    #         if (
    #             "_remeshed" in cfg["INPUT"]["%sFile" % forestParam]
    #             and inputSimFiles["entResInfo"]["%sRemeshed" % forestParam] == "Yes"
    #         ):
    #             forestFilePathNew = workInputDir / (
    #                 forestFile.stem + "_%s" % forestParam + forestFile.suffix
    #             )
    #             shutil.copy2(forestFile, forestFilePathNew)
    #             cfg["File names"][forestParameters[forestParam]] = str(forestFilePathNew)
    #             log.info(
    #                 "Remeshed %s file copied to %s and set for %s"
    #                 % (forestParam, str(forestFilePathNew), forestParameters[forestParam])
    #             )
    #         else:
    #             cfg["File names"][forestParameters[forestParam]] = str(forestFile)
    #
    #     cfg["FOREST_EFFECTS"]["Forest effects"] = "yes"
    #
    # else:
    #     # TODO FSO implement if setting is variable or constant that if variable but file not found then error
    #     message = "nd and bhd file not found in Inputs/RASTERS - check if file ending is correct (_nd, _bhd) - setting forest effects to no"
    #     log.warning(message)
    #
    #     cfg["FOREST_EFFECTS"]["Forest effects"] = "no"
    #     cfg["File names"]["Forest density filename"] = "-"
    #     cfg["File names"]["Tree diameter filename"] = "-"

    return cfg
