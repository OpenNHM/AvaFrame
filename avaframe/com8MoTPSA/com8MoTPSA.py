import os
import sys
import platform
import logging
import numpy as np
import pathlib
import time
import shutil

from avaframe.in3Utils.cfgUtils import cfgToRcf

if os.name == "nt":
    from multiprocessing.pool import ThreadPool as Pool
elif platform.system() == "Darwin":
    from multiprocessing.pool import ThreadPool as Pool
else:
    from multiprocessing import Pool

import avaframe.com1DFA.com1DFA as com1DFA
from avaframe.in3Utils import cfgUtils
from avaframe.in2Trans import rasterUtils as rU
from avaframe.in1Data import getInput as gI
import avaframe.in3Utils.fileHandlerUtils as fU
from avaframe.out1Peak import outPlotAllPeak as oP
import avaframe.in3Utils.MoTUtils as mT

# create local logger
log = logging.getLogger(__name__)


def com8MoTPSAMain(cfgMain, cfgInfo=None):
    # Get all necessary information from the configuration files
    currentModule = sys.modules[__name__]
    simDict, inputSimFiles = mT.MoTGenerateConfigs(cfgMain, cfgInfo, currentModule)

    # convert DEM from nan to 0 values
    # TODO: suggest MoT-PSA to handle nan values
    mT.rewriteDEMtoZeroValues(inputSimFiles["demFile"])

    log.info("The following simulations will be performed")
    for key in simDict:
        log.info("Simulation: %s" % key)

    # Preprocess the simulations, mainly creating the rcf files
    rcfFiles = com8MoTPSAPreprocess(simDict, inputSimFiles, cfgMain)

    # And now we run the simulations
    startTime = time.time()

    log.info("--- STARTING (potential) PARALLEL PART ----")

    # Get number of CPU Cores wanted
    nCPU = cfgUtils.getNumberOfProcesses(cfgMain, len(rcfFiles))

    # Create parallel pool and run
    # with multiprocessing.Pool(processes=nCPU) as pool:
    with Pool(processes=nCPU) as pool:
        results = pool.map(com8MoTPSATask, rcfFiles)
        pool.close()
        pool.join()

    timeNeeded = "%.2f" % (time.time() - startTime)
    log.info("Overall (parallel) com8MoTPSA computation took: %s s " % timeNeeded)
    log.info("--- ENDING (potential) PARALLEL PART ----")

    # Postprocess the simulations
    com8MoTPSAPostprocess(simDict, cfgMain, inputSimFiles)


def com8MoTPSAPostprocess(simDict, cfgMain, inputSimFiles):
    avalancheDir = cfgMain["MAIN"]["avalancheDir"]
    # Copy max files to output directory

    outputDir = pathlib.Path(avalancheDir) / "Outputs" / "com8MoTPSA"
    outputDirPeakFile = pathlib.Path(avalancheDir) / "Outputs" / "com8MoTPSA" / "peakFiles"
    fU.makeADir(outputDirPeakFile)

    for key in simDict:
        workDir = pathlib.Path(avalancheDir) / "Work" / "com8MoTPSA" / str(key)

        # identify simType
        simType = simDict[key]["simType"]

        # Copy DataTime.txt
        dataTimeFile = workDir / "DataTime.txt"
        shutil.copy2(dataTimeFile, outputDir / (str(key) + "_DataTime.txt"))

        # TODO: functionize it
        # Copy ppr files
        pprFiles = list(workDir.glob("*p?_max*"))

        targetFiles = [pathlib.Path(str(f.name).replace("null_p1_max", "null_ppr_L1")) for f in pprFiles]
        targetFiles = [pathlib.Path(str(f).replace("null_p2_max", "null_ppr_L2")) for f in targetFiles]

        targetFiles = [outputDirPeakFile / f for f in targetFiles]
        for source, target in zip(pprFiles, targetFiles):
            shutil.copy2(source, target)

        # Copy pfd files
        pfdFiles = list(workDir.glob("*h?_max*"))

        targetFiles = [pathlib.Path(str(f.name).replace("null_h1_max", "null_pfd_L1")) for f in pfdFiles]
        targetFiles = [pathlib.Path(str(f).replace("null_h2_max", "null_pfd_L2")) for f in targetFiles]

        targetFiles = [outputDirPeakFile / f for f in targetFiles]
        for source, target in zip(pfdFiles, targetFiles):
            shutil.copy2(source, target)

        # Copy pfv files
        pfvFiles = list(workDir.glob("*s?_max*"))

        targetFiles = [pathlib.Path(str(f.name).replace("null_s1_max", "null_pfv_L1")) for f in pfvFiles]
        targetFiles = [pathlib.Path(str(f).replace("null_s2_max", "null_pfv_L2")) for f in targetFiles]

        targetFiles = [outputDirPeakFile / f for f in targetFiles]
        for source, target in zip(pfvFiles, targetFiles):
            shutil.copy2(source, target)

    # create plots and report
    modName = __name__.split(".")[-1]
    reportDir = pathlib.Path(avalancheDir, "Outputs", modName, "reports")
    fU.makeADir(reportDir)

    dem = rU.readRaster(inputSimFiles["demFile"])
    # Generate plots for all peakFiles
    oP.plotAllPeakFields(avalancheDir, cfgMain["FLAGS"], modName, demData=dem)


def com8MoTPSATask(rcfFile):
    # TODO: Obvious...
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    command = ["./MoT-PSA", rcfFile]
    # command = ['/home/felix/Versioning/AvaFrame/avaframe/com8MoTPSA/MoT-PSA', rcfFile]
    log.info("Run simulation: %s" % rcfFile)
    mT.runAndCheckMoT(command)
    return command


def com8MoTPSAPreprocess(simDict, inputSimFiles, cfgMain):
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
