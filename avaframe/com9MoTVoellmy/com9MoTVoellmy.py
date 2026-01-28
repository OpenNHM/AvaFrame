import os
import platform
import logging
import pathlib
import time
import sys

if os.name == "nt":
    from multiprocessing.pool import ThreadPool as Pool
else:
    from multiprocessing import Pool

import avaframe.com1DFA.com1DFA as com1DFA
from avaframe.in3Utils import cfgUtils
from avaframe.in2Trans import rasterUtils as rU
from avaframe.in1Data import getInput as gI
import avaframe.in3Utils.fileHandlerUtils as fU
from avaframe.out1Peak import outPlotAllPeak as oP
from avaframe.in3Utils.cfgUtils import cfgToRcf
import avaframe.in3Utils.MoTUtils as mT


# create a local logger
log = logging.getLogger(__name__)


def com9MoTVoellmyMain(cfgMain, cfgInfo=None):
    """Run MoT-Voellmy simulations using specified configurations.

    This function executes MoT-Voellmy simulations by handling
    preprocessing, parallel execution, and postprocessing steps.

    Parameters
    ----------
    cfgMain : configparser.ConfigParser
        Main configuration settings for the simulation
    cfgInfo : str or pathlib.Path or configparser.ConfigParser, optional
        Additional configuration information, by default None
        Can be:
        - Path to configuration file for overrides
        - ConfigParser object with initial configuration
        - None to use local/default configuration

    Returns
    -------
    None
        Results are written to the output directory structure

    Notes
    -----
    The function performs several key steps:
    - Generates configuration settings for all simulations
    - Preprocesses input data and creates RCF configuration files
    - Executes simulations in parallel using multiple processes
    - Handles DEM preparation by converting NaN values to zeros
    """

    # Get all necessary information from the configuration files
    currentModule = sys.modules[__name__]  # As if you would to import com9MoTVoellmy
    simDict, inputSimFiles = mT.MoTGenerateConfigs(cfgMain, cfgInfo, currentModule)

    # convert DEM from nan to 0 values
    # TODO: suggest MoT-PSA to handle nan values
    mT.rewriteDEMtoZeroValues(inputSimFiles["demFile"])

    log.info("The following simulations will be performed")
    for key in simDict:
        log.info("Simulation: %s" % key)

    # Preprocess the simulations, mainly creating the rcf files
    rcfFiles = com9MoTVoellmyPreprocess(simDict, inputSimFiles, cfgMain)

    # And now we run the simulations
    startTime = time.time()

    log.info("--- STARTING (potential) PARALLEL PART ----")

    # Get number of CPU Cores wanted
    nCPU = cfgUtils.getNumberOfProcesses(cfgMain, len(rcfFiles))

    # Create parallel pool and run
    # with multiprocessing.Pool(processes=nCPU) as pool:
    with Pool(processes=nCPU) as pool:
        results = pool.map(com9MoTVoellmyTask, rcfFiles)
        pool.close()
        pool.join()

    timeNeeded = "%.2f" % (time.time() - startTime)
    log.info("Overall (parallel) com9MoTVoellmy computation took: %s s " % timeNeeded)
    log.info("--- ENDING (potential) PARALLEL PART ----")

    # Postprocess the simulations
    com9MoTVoellmyPostprocess(simDict, cfgMain)


def com9MoTVoellmyPostprocess(simDict, cfgMain):
    """Post-process MoT-Voellmy simulation results.

    This function handles post-processing tasks after MoT-Voellmy simulations complete,
    including copying result files to output directories and generating visualization plots.

    Parameters
    ----------
    simDict : dict
        Dictionary containing simulation configurations, with one entry per simulation
    cfgMain : configparser.ConfigParser
        Main configuration settings for the simulation
    inputSimFiles : dict
        Dictionary containing paths to input files (DEM, release areas, etc.)

    Returns
    -------
    None
        Results are written to the output directory structure

    Notes
    -----
    The function performs several key tasks:
    - Creates output directory structure
    - Copies simulation result files (DataTime.txt, ppr, pfd, pfv files)
    - Renames files according to conventions
    - Generates visualization plots of peak fields
    """
    avalancheDir = cfgMain["MAIN"]["avalancheDir"]

    # Copy max files to output directory
    outputDirPeakFile = pathlib.Path(avalancheDir) / "Outputs" / "com9MoTVoellmy" / "peakFiles"
    fU.makeADir(outputDirPeakFile)

    for key in simDict:
        workDir = pathlib.Path(avalancheDir) / "Work" / "com9MoTVoellmy" / str(key)

        # Copy ppr files
        mT.copyMoTFiles(workDir, outputDirPeakFile, "p_max", "ppr")

        # Copy pfd files
        mT.copyMoTFiles(workDir, outputDirPeakFile, "h_max", "pfd")

        # Copy pfv files
        mT.copyMoTFiles(workDir, outputDirPeakFile, "s_max", "pfv")

        # Copy timestep directories to timesteps subfolder
        mT.copyMoTDirs(workDir, outputDirPeakFile, key, "s")
        mT.copyMoTDirs(workDir, outputDirPeakFile, key, "h")

    # create plots and report
    modName = __name__.split(".")[-1]
    reportDir = pathlib.Path(avalancheDir, "Outputs", modName, "reports")
    fU.makeADir(reportDir)

    # Generate plots for all peakFiles
    oP.plotAllPeakFields(avalancheDir, cfgMain["FLAGS"], modName)


def com9MoTVoellmyTask(rcfFile):
    """Execute a single MoT-PSA simulation using the provided configuration file.

    Parameters
    ----------
    rcfFile : pathlib.Path
        Path to the RCF configuration file for the simulation

    Returns
    -------
    list
        The command that was executed as a list containing the executable path
        and configuration file path

    Notes
    -----
    Changes to the directory containing this module before executing the simulation.
    Uses runAndCheckMoT to execute and monitor the simulation process.
    """
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    if os.name == "nt":
        exeName = "MoT-Voellmy_win.exe"
    elif platform.system() == "Darwin":
        message = "MoT-Voellmy does not support MacOS at the moment"
        log.error(message)
        raise OSError(message)
    else:
        exeName = "./MoT-Voellmy_linux.exe"

    command = [exeName, rcfFile]
    log.info("Run simulation: %s" % rcfFile)
    mT.runAndCheckMoT(command)
    return command


def com9MoTVoellmyPreprocess(simDict, inputSimFiles, cfgMain):
    """Preprocess data for MoT-PSA simulations.

    This function prepares the input data and configuration files needed to run
    MoT-PSA simulations. It processes release areas, creates required directories,
    and generates configuration files for each simulation.

    Parameters
    ----------
    simDict : dict
        Dictionary containing simulation configurations with one entry per simulation
    inputSimFiles : dict
        Dictionary containing paths to input files (DEM, release areas, etc.)
    cfgMain : configparser.ConfigParser
        Main configuration settings for the simulation

    Returns
    -------
    list
        List of pathlib.Path objects pointing to generated RCF configuration files
        that will be used to run the simulations

    Notes
    -----
    The function performs several key steps:
    - Creates working directories for each simulation
    - Processes release areas and converts them to raster format
    - Generates configuration files in RCF format
    - Handles both single and multiple release scenarios
    """
    # Load avalanche directory from general configuration file
    avalancheDir = cfgMain["MAIN"]["avalancheDir"]
    # set inputsDir where original input data and remeshed rasters are stored
    inputsDir = pathlib.Path(avalancheDir) / "Inputs"

    # create required Work und Outputs directories in avalancheDir
    workDir = pathlib.Path(avalancheDir) / "Work" / "com9MoTVoellmy"
    cfgFileDir = pathlib.Path(avalancheDir) / "Outputs" / "com9MoTVoellmy" / "configurationFiles"
    fU.makeADir(cfgFileDir)
    rcfFiles = list()

    # loop over all simulation to be performed
    for key in simDict:
        # Generate command and run via subprocess.run
        # Configuration that needs adjustment

        # Generate the work and data dirs for the current simHash
        # save derived fields from polygons, optionally zeroRasters and remeshedRasters to that folder
        cuWorkDir = workDir / key
        workInputDir = cuWorkDir / "Input"
        workOutputDir = cuWorkDir / key
        fU.makeADir(cuWorkDir)
        fU.makeADir(workInputDir)

        # load configuration object for current sim
        cfg = simDict[key]["cfgSim"]
        log.info("Prepare simulation configuration for key %s" % key)

        # select release area input data according to a chosen release scenario
        inputSimFiles = gI.selectReleaseFile(inputSimFiles, cfg["INPUT"]["releaseScenario"])

        # create the required input from input files
        # if release, entrainment area are provided as shapefile - read shapefile attributes and values for current sim
        # if provided by raster - load raster data
        # load DEM and dem file type information
        demOri, inputSimLines = com1DFA.prepareInputData(inputSimFiles, cfg)
        demOri["originalHeader"] = demOri["header"]
        demSuffix = rU.getRasterFileTypeFromHeader(demOri["header"])

        # set thickness values for the release area, entrainment areas
        relName, inputSimLines, badName = com1DFA.prepareReleaseEntrainment(
            cfg, inputSimFiles["releaseScenario"], inputSimLines
        )

        # RELEASE AREA - fetch path to release raster
        releaseName, inputSimLines["releaseLine"] = gI.deriveLineRaster(
            cfg,
            inputSimLines["releaseLine"],
            demOri,
            workInputDir,
            inputsDir,
            "rel",
            rasterFileType=demSuffix,
        )

        # ENTRAINMENT AREA - fetch path to entrainment (bedDepth) raster
        if "ent" in key:
            saveZeroRaster = False
        else:
            saveZeroRaster = True

        bedDepthName, inputSimLines["entLine"] = gI.deriveLineRaster(
            cfg,
            inputSimLines["entLine"],
            demOri,
            workInputDir,
            inputsDir,
            "ent",
            rasterFileType=demSuffix,
            saveZeroRaster=saveZeroRaster,
        )

        # TODO: is this check if release and entrainment have overlap required?
        # if "ent" in key:
        #     log.info("Check for overlap?")
        #
        #     # check if entrainment and release area have overlap
        #     _ = geoTrans.checkOverlap(
        #         inputSimLines["entLine"]["rasterData"],
        #         inputSimLines["releaseLine"]["rasterData"],
        #         "Entrainment",
        #         "Release",
        #         crop=False,
        #     )

        # BED SHEAR - fetch path to tauC raster
        bedShearDict = {
            "initializedFrom": "raster",
            "fileName": inputSimLines["tauCFile"],
        }
        if inputSimLines["entResInfo"]["tauC"] == "Yes":
            saveZeroRaster = False
        else:
            saveZeroRaster = True

        bedShearName, bedShearDict = gI.deriveLineRaster(
            cfg,
            bedShearDict,
            demOri,
            workInputDir,
            inputsDir,
            "tauC",
            rasterFileType=demSuffix,
            saveZeroRaster=saveZeroRaster,
        )

        # set configuration for MoT-Voellmy
        cfg["Run information"]["Area of Interest"] = cfgMain["MAIN"]["avalancheDir"]
        cfg["Run information"]["UTM zone"] = "32N"
        cfg["Run information"]["EPSG geodetic datum code"] = "31287"
        cfg["Run information"]["Run name"] = cfgMain["MAIN"]["avalancheDir"]
        cfg["File names"]["Grid filename"] = str(pathlib.Path(inputsDir / cfg["INPUT"]["DEM"]))
        cfg["File names"]["Release depth filename"] = str(releaseName)
        cfg["File names"]["Bed depth filename"] = str(bedDepthName)
        cfg["File names"]["Bed shear strength filename"] = str(bedShearName)
        cfg["File names"]["Output filename root"] = str(workOutputDir)

        # if _mu and _k files in avalancheDir/Inputs/RASTERS found - set paths to mu and k files
        # if not found then mu and k are set constant to values provided in cfg
        if cfg["Physical_parameters"]["Parameters"] == "auto":
            cfg = mT.setVariableFrictionParameters(cfg, inputSimFiles, workInputDir, inputsDir)
        else:
            # TODO FSO allow for options constant and variable
            message = "Currently only available option is auto for %s" % (
                '["Physical_parameters"]["Parameters"]'
            )
            log.error(message)
            raise AssertionError(message)

        if cfg["ENTRAINMENT"]["Entrainment"] == "auto":
            cfg = mT.setVariableEntrainmentParameters(cfg, inputSimFiles, workInputDir, inputsDir)
        else:
            message = "Currently only available option is auto for %s" % ('["ENTRAINMENT"]["Entrainment"]')
            log.error(message)
            raise AssertionError(message)

        # if a file in Inputs/RES is found (forest density info) and _bhd file in avalancheDir/Inputs/RASTERS found - set paths to nd and bhd files

        # if not found then forest effects are set to no
        if cfg["FOREST_EFFECTS"]["Forest effects"] == "auto":
            cfg = mT.setVariableForestParameters(cfg, inputSimFiles, workInputDir, inputsDir)
        else:
            message = "Currently only available option is auto for %s" % (
                '["FOREST_EFFECTS"]["Forest effects"]'
            )
            log.error(message)
            raise AssertionError(message)
        # elif cfg["FOREST_EFFECTS"]["Forest effects"] == "no":
        #     cfg["File names"]["Forest density filename"] = "-"
        #     cfg["File names"]["Tree diameter filename"] = "-"
        # else:
        #     # if forest effects set to yes but files not found - error will be raised by setVariableForestParameters
        #     cfg = mT.setVariableForestParameters(cfg, inputSimFiles, workInputDir, inputsDir)

        rcfFileName = cfgFileDir / (str(key) + ".rcf")

        currentModule = sys.modules[__name__]
        cfgUtils.writeCfgFile(avalancheDir, currentModule, cfg, str(key))
        cfgToRcf(cfg, rcfFileName)
        rcfFiles.append(rcfFileName)
        log.info("rcf and ini file written for key %s-------------------------" % key)
    return rcfFiles
