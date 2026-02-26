import os
import sys
import platform
import logging
import numpy as np
import pathlib
import time
import shutil

from avaframe.in3Utils.cfgUtils import cfgToRcf

from multiprocessing import Pool

import avaframe.com1DFA.com1DFA as com1DFA
from avaframe.in3Utils import cfgUtils
from avaframe.in2Trans import rasterUtils as rU
from avaframe.ana4Stats import probAna
from avaframe.in1Data import getInput as gI
import avaframe.in3Utils.fileHandlerUtils as fU
from avaframe.out1Peak import outPlotAllPeak as oP
import avaframe.in3Utils.MoTUtils as mT
from avaframe.in3Utils.initializeProject import _checkForFolderAndDelete

# create local logger
log = logging.getLogger(__name__)


def com8MoTPSAMain(cfgMain, cfgInfo=None, returnSimName=None):
    """Run the full MoT-PSA workflow: generate configs, run simulations in parallel, postprocess.

    Parameters
    ----------
    cfgMain : configparser.ConfigParser
        main AvaFrame configuration (avalancheDir, nCPU, plot flags)
    cfgInfo : dict or None, optional
        override configuration info passed to MoTGenerateConfigs
    returnSimName : any, optional
        if not None, return the first simDict key after running
    """
    # Get all necessary information from the configuration files
    currentModule = sys.modules[__name__]
    simDict, _, inputSimFiles, _ = com1DFA.com1DFAPreprocess(cfgMain, cfgInfo, module=currentModule)

    # convert DEM from nan to 0 values
    # TODO: suggest MoT-PSA to handle nan values
    mT.rewriteDEMtoZeroValues(inputSimFiles["demFile"])

    log.info("The following simulations will be performed")
    for key in simDict:
        log.info("Simulation: %s" % key)

    # Preprocess the simulations, mainly creating the rcf files
    rcfFiles = com8MoTPSAPreprocess(simDict, inputSimFiles, cfgMain)

    # Check if there is simulation to be run
    if not rcfFiles:
        log.warning("There is no simulation to be performed for releaseScenario")
        return None

    # And now we run the simulations
    startTime = time.time()

    log.info("--- STARTING (potential) PARALLEL PART ----")

    # Split into chunks to postprocess and clean up working directory incrementally
    # Get chunkSize from probAnaCfg.ini, if it is empty use 10 as default
    cfgProbAna = cfgUtils.getModuleConfig(probAna)
    chunkSize = cfgProbAna.get('GENERAL', 'chunkSize', fallback='')
    if chunkSize == '':
        chunkSize = 10
    else:
        chunkSize = int(chunkSize)

    rcfChunks = [rcfFiles[i:i + chunkSize] for i in range(0, len(rcfFiles), chunkSize)]

    for rcfFilesChunk in rcfChunks:
        simNamesChunk = [p.stem for p in rcfFilesChunk]

        nCPU = cfgUtils.getNumberOfProcesses(cfgMain, len(rcfFilesChunk))

        with Pool(processes=nCPU) as pool:
            results = pool.map(com8MoTPSATask, rcfFilesChunk)
            pool.close()
            pool.join()

        timeNeeded = "%.2f" % (time.time() - startTime)
        log.info("Overall (parallel) com8MoTPSA computation took: %s s ", timeNeeded)
        log.info("--- ENDING (potential) PARALLEL PART ----")

        # Postprocess the simulations
        com8MoTPSAPostprocess(simNamesChunk, cfgMain, inputSimFiles)

        # Delete folder in Work directory after postprocessing to reduce memory costs
        avaDir = cfgMain["MAIN"]["avalancheDir"]
        for sim in simNamesChunk:
            folderName = "Work/com8MoTPSA/" + sim
            _checkForFolderAndDelete(avaDir, folderName)

    if returnSimName is not None and simDict:
        return next(iter(simDict))
    return None


def copyRawToLayerPeakFiles(workDir, outputDirPeakFile):
    """Rename and copy raw MoT-PSA output files to peakFiles with L1/L2 layer naming.

    MoT-PSA produces raw suffixes (p1/p2_max, h1/h2_max, s1/s2_max) that map
    to AvaFrame result types (ppr, pfd, pfv). The digit in the raw suffix encodes
    the layer: 1 -> L1 (dense flow), 2 -> L2 (powder snow).

    Example: simKey_null_psa_p1_max.asc -> simKey_null_psa_L1_ppr.asc

    Parameters
    ----------
    workDir : pathlib.Path
        simulation work directory containing raw MoT-PSA output files
    outputDirPeakFile : pathlib.Path
        target directory for renamed peak files
    """
    # Each entry: (glob pattern, raw L1 suffix, raw L2 suffix, AvaFrame resType)
    layerRenameMap = [
        ("*p?_max*", "p1_max", "p2_max", "ppr"),
        ("*h?_max*", "h1_max", "h2_max", "pfd"),
        ("*s?_max*", "s1_max", "s2_max", "pfv"),
    ]
    for globPattern, rawL1, rawL2, resType in layerRenameMap:
        rawFiles = list(workDir.glob(globPattern))
        # Replace raw suffixes with layer naming (e.g. p1_max -> L1_ppr, p2_max -> L2_ppr)
        targetFiles = [
            pathlib.Path(str(f.name).replace(rawL1, "L1_%s" % resType).replace(rawL2, "L2_%s" % resType))
            for f in rawFiles
        ]
        # Prepend output directory and copy
        targetFiles = [outputDirPeakFile / f for f in targetFiles]
        for source, target in zip(rawFiles, targetFiles):
            shutil.copy2(source, target)


def com8MoTPSAPostprocess(simNames, cfgMain, inputSimFiles):
    """Postprocess MoT-PSA results: rename outputs to L1/L2 peak files, generate plots and reports.

    For each simulation, copies DataTime.txt and renames raw MoT-PSA output files
    (p1/p2_max, h1/h2_max, s1/s2_max) to AvaFrame layer naming (L1/L2 + ppr/pfd/pfv).

    Parameters
    ----------
    simNames : list
        list of simulation name strings
    cfgMain : configparser.ConfigParser
        main AvaFrame configuration (avalancheDir, plot flags)
    inputSimFiles : dict
        input file paths, must contain "demFile"
    """
    avalancheDir = cfgMain["MAIN"]["avalancheDir"]
    # Copy max files to output directory

    outputDir = pathlib.Path(avalancheDir) / "Outputs" / "com8MoTPSA"
    outputDirPeakFile = pathlib.Path(avalancheDir) / "Outputs" / "com8MoTPSA" / "peakFiles"
    fU.makeADir(outputDirPeakFile)

    for key in simNames:
        workDir = pathlib.Path(avalancheDir) / "Work" / "com8MoTPSA" / str(key)

        # Copy DataTime.txt
        dataTimeFile = workDir / "DataTime.txt"
        shutil.copy2(dataTimeFile, outputDir / (str(key) + "_DataTime.txt"))

        copyRawToLayerPeakFiles(workDir, outputDirPeakFile)

        # Write config indicator files to track completed simulations
        configFileName = "%s.ini" % key
        for saveDir in ["configurationFilesDone", "configurationFilesLatest"]:
            configDir = pathlib.Path(avalancheDir, "Outputs", "com8MoTPSA", "configurationFiles", saveDir)
            with open((configDir / configFileName), "w") as fi:
                fi.write("see directory configurationFiles for info on config")

    # create plots and report
    modName = __name__.split(".")[-1]
    reportDir = pathlib.Path(avalancheDir, "Outputs", modName, "reports")
    fU.makeADir(reportDir)

    dem = rU.readRaster(inputSimFiles["demFile"])
    # Generate plots for all peakFiles
    oP.plotAllPeakFields(avalancheDir, cfgMain["FLAGS"], modName, demData=dem)


def com8MoTPSATask(rcfFile):
    """Run a single MoT-PSA simulation by invoking the MoT-PSA executable with an rcf file.

    Parameters
    ----------
    rcfFile : str or pathlib.Path
        path to the .rcf configuration file for this simulation

    Returns
    -------
    list
        the command that was executed (["./MoT-PSA", rcfFile])
    """
    # TODO: Obvious...
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    command = ["./MoT-PSA", rcfFile]
    # command = ['/home/felix/Versioning/AvaFrame/avaframe/com8MoTPSA/MoT-PSA', rcfFile]
    log.info("Run simulation: %s" % rcfFile)
    mT.runAndCheckMoT(command)
    return command


def com8MoTPSAPreprocess(simDict, inputSimFiles, cfgMain):
    """Prepare all MoT-PSA simulations: derive input rasters, set config paths, write rcf files.

    For each simulation in simDict, processes release/entrainment/bed shear/deposition
    input data, configures MoT-PSA file paths and parameters, and writes the .rcf
    configuration file needed by the MoT-PSA executable.

    Parameters
    ----------
    simDict : dict
        simulation dictionary keyed by simKey, each value contains "cfgSim" and "simType"
    inputSimFiles : dict
        input file paths (DEM, release scenarios, etc.)
    cfgMain : configparser.ConfigParser
        main AvaFrame configuration (avalancheDir)

    Returns
    -------
    list of pathlib.Path
        paths to the generated .rcf files, one per simulation
    """
    # Load avalanche directory from general configuration file
    avalancheDir = cfgMain["MAIN"]["avalancheDir"]
    # set inputsDir where original input data and remeshed rasters are stored
    inputsDir = pathlib.Path(avalancheDir) / "Inputs"

    workDir = pathlib.Path(avalancheDir) / "Work" / "com8MoTPSA"
    cfgFileDir = pathlib.Path(avalancheDir) / "Outputs" / "com8MoTPSA" / "configurationFiles"
    fU.makeADir(cfgFileDir)
    rcfFiles = list()

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

        # select release area input data according to chosen release scenario
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
        # TODO: split releaseheight -> question NGI
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

        # TODO: NGI shall this also be read from inputs?

        # RELEASE LAYER 2
        releaseL2Dict = None
        releaseL2Name, _ = gI.deriveLineRaster(
            cfg,
            releaseL2Dict,
            demOri,
            workInputDir,
            inputsDir,
            "releaseLayer2",
            rasterFileType=demSuffix,
            saveZeroRaster=True,
        )
        # BED DEPOSITION
        bedDepositionDict = None
        bedDepoName, _ = gI.deriveLineRaster(
            cfg,
            bedDepositionDict,
            demOri,
            workInputDir,
            inputsDir,
            "bedDepo",
            rasterFileType=demSuffix,
            saveZeroRaster=True,
        )

        # set configuration for MoT-PSA
        cfg["Run information"]["Area of Interest"] = cfgMain["MAIN"]["avalancheDir"]
        cfg["Run information"]["UTM zone"] = "32N"
        cfg["Run information"]["EPSG geodetic datum code"] = "31287"
        cfg["Run information"]["Run name"] = cfgMain["MAIN"]["avalancheDir"]
        cfg["File names"]["Grid filename"] = str(pathlib.Path(inputsDir / cfg["INPUT"]["DEM"]))
        cfg["File names"]["Release depth 1 filename"] = str(releaseName)
        cfg["File names"]["Release depth 2 filename"] = str(releaseL2Name)
        cfg["File names"]["Bed depth filename"] = str(bedDepthName)
        cfg["File names"]["Bed deposition filename"] = str(bedDepoName)
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

        rcfFileName = cfgFileDir / (str(key) + ".rcf")
        currentModule = sys.modules[__name__]
        cfgUtils.writeCfgFile(avalancheDir, currentModule, cfg, str(key))
        cfgToRcf(cfg, rcfFileName)
        rcfFiles.append(rcfFileName)
        log.info("rcf and ini file written for key %s-------------------------" % key)
    return rcfFiles
