"""
Main functions for python DFA kernel
"""

import copy
import logging
import math
import os
import pathlib
import pickle
import platform
import time
from datetime import datetime
from functools import partial
from itertools import product

import matplotlib.tri as tri
import numpy as np
import pandas as pd
from shapely.geometry import Polygon as sPolygon


if os.name == "nt":
    from multiprocessing.pool import ThreadPool as Pool
elif platform.system() == "Darwin":
    from multiprocessing.pool import ThreadPool as Pool
else:
    from multiprocessing import Pool

# Local imports
from avaframe.version import getVersion
import avaframe.in2Trans.shpConversion as shpConv
import avaframe.in3Utils.geoTrans as geoTrans
from avaframe.in3Utils import initializeProject as iP
import avaframe.com1DFA.timeDiscretizations as tD
import avaframe.out3Plot.outCom1DFA as outCom1DFA
import avaframe.com1DFA.DFAtools as DFAtls
import avaframe.com1DFA.com1DFATools as com1DFATools
import avaframe.com1DFA.particleTools as particleTools
import avaframe.com1DFA.DFAfunctionsCython as DFAfunC
import avaframe.com1DFA.DFAToolsCython as DFAtllsC
import avaframe.com1DFA.damCom1DFA as damCom1DFA
import avaframe.in2Trans.rasterUtils as IOf
import avaframe.in3Utils.fileHandlerUtils as fU
from avaframe.in3Utils import cfgUtils
import avaframe.out3Plot.outDebugPlots as debPlot
import avaframe.com1DFA.deriveParameterSet as dP
import avaframe.com1DFA.com1DFA as com1DFA
from avaframe.in1Data import getInput as gI
from avaframe.out1Peak import outPlotAllPeak as oP
from avaframe.log2Report import generateReport as gR
from avaframe.com1DFA import particleInitialisation as pI
from avaframe.com1DFA import checkCfg
from avaframe.ana5Utils import distanceTimeAnalysis as dtAna
import avaframe.out3Plot.outDistanceTimeAnalysis as dtAnaPlots
import threading

#######################################
# Set flags here
#######################################
# create local logger
log = logging.getLogger(__name__)
cfgAVA = cfgUtils.getGeneralConfig()
debugPlot = cfgAVA["FLAGS"].getboolean("debugPlot")


def com1DFAPreprocess(cfgMain, typeCfgInfo, cfgInfo, module=com1DFA):
    """preprocess information from configuration, read input data and gather into inputSimFiles,
    create one config object for each of all desired simulations,
    create dataFrame with one line per simulations of already existing sims in avalancheDir

    Parameters
    ------------
    cfgMain: configparser object
        main configuration of AvaFrame
    typeCfgInfo: str
        name of type of cfgInfo (cfgFromFile or cfgFromObject)
    cfgInfo: str or pathlib Path or configparser object
        path to configuration file if overwrite is desired - optional
        if not local (if available) or default configuration will be loaded
        if cfgInfo is a configparser object take this as initial config
    module: module
        module to be used for task (optional)

    Returns
    --------
    simDict: dict
        dictionary with one key per simulation to perform including its config object
    inputSimFiles: dict
        dictionary with input files info
    outDir: str
        path to store outputs
    """

    avalancheDir = cfgMain["MAIN"]["avalancheDir"]

    # read initial configuration
    if typeCfgInfo in ["cfgFromFile", "cfgFromDefault"]:
        cfgStart = cfgUtils.getModuleConfig(module, fileOverride=cfgInfo, toPrint=False)
    elif typeCfgInfo == "cfgFromObject":
        cfgStart = cfgInfo

    # fetch input data and create work and output directories
    inputSimFilesAll, outDir, simDFExisting, simNameExisting = com1DFATools.initializeInputs(
        avalancheDir, cfgStart["GENERAL"].getboolean("cleanRemeshedRasters"), module
    )

    # create dictionary with one key for each simulation that shall be performed
    simDict = dP.createSimDict(avalancheDir, module, cfgStart, inputSimFilesAll, simNameExisting)

    return simDict, outDir, inputSimFilesAll, simDFExisting


def com1DFAMain(cfgMain, cfgInfo=""):
    """preprocess information from ini and run all desired simulations, create outputs and reports

    Parameters
    ------------
    cfgMain: configparser object
        main configuration of AvaFrame
    cfgInfo: str or pathlib Path or configparser object
        path to configuration file if overwrite is desired - optional
        if not local (if available) or default configuration will be loaded
        if cfgInfo is a configparser object take this as initial config
        if path to a directory is provided - load one or multiple override configuration files

    Returns
    --------
    dem: dict
        dictionary with dem header and raster data (that has been used for the final run)
    plotDict: dict
        information on result plot paths
    reportDictList: list
        list of report dictionaries for all performed simulations
    simDF: pandas dataFrame
        configuration dataFrame of the simulations computed (if no simulation computed, configuration dataFrame
        of the already existing ones)
    """

    avalancheDir = cfgMain["MAIN"]["avalancheDir"]

    # fetch type of cfgInfo
    typeCfgInfo = com1DFATools.checkCfgInfoType(cfgInfo)
    if typeCfgInfo == "cfgFromDir":
        # preprocessing to create configuration objects for all simulations to run by reading multiple cfg files
        simDict, inputSimFiles, simDFExisting, outDir = com1DFATools.createSimDictFromCfgs(cfgMain, cfgInfo)
    else:
        # preprocessing to create configuration objects for all simulations to run
        simDict, outDir, inputSimFiles, simDFExisting = com1DFAPreprocess(cfgMain, typeCfgInfo, cfgInfo)

    log.info("The following simulations will be performed")
    for key in simDict:
        log.info("Simulation: %s" % key)
        exportFlag = simDict[key]["cfgSim"]["EXPORTS"].getboolean("exportData")
        adaptDemPlot = simDict[key]["cfgSim"]["GENERAL"].getboolean("adaptDemPlot")

    # initialize reportDict list
    reportDictList = list()

    # is there any simulation to run?
    if bool(simDict):
        # reset simDF and timing
        simDF = pd.DataFrame()
        tCPUDF = pd.DataFrame()
        dem = dict()

        startTime = time.time()

        log.info("--- STARTING (potential) PARALLEL PART ----")
        # Get number of CPU Cores wanted
        nCPU = cfgUtils.getNumberOfProcesses(cfgMain, len(simDict))

        # Supply compute task with inputs
        com1DFACoreTaskWithInput = partial(com1DFACoreTask, simDict, inputSimFiles, avalancheDir, outDir)

        # Create parallel pool and run
        # with multiprocessing.Pool(processes=nCPU) as pool:
        with Pool(processes=nCPU) as pool:
            results = pool.map(com1DFACoreTaskWithInput, simDict)
            pool.close()
            pool.join()

        # Split results to according structures
        for result in results:
            simDF = pd.concat([simDF, result[0]], axis=0)
            tCPUDF = pd.concat([tCPUDF, result[1]], axis=0)
            dem = result[2]  # only last dem is used
            reportDictList.append(result[3])

        timeNeeded = "%.2f" % (time.time() - startTime)
        log.info("Overall (parallel) com1DFA computation took: %s s " % timeNeeded)
        log.info("--- ENDING (potential) PARALLEL PART ----")

        # TODO: needs to be moved inside the outPlotAllPeakFunction
        # dem for plot chosen there
        dem = com1DFATools.chooseDemPlot(dem, adaptedDemBackground=adaptDemPlot)
        # postprocessing: writing report, creating plots
        dem, plotDict, reportDictList, simDFNew = com1DFAPostprocess(
            simDF,
            tCPUDF,
            simDFExisting,
            cfgMain,
            dem,
            reportDictList,
            exportData=exportFlag,
        )

        return dem, plotDict, reportDictList, simDFNew

    else:
        log.warning("There is no simulation to be performed for releaseScenario")

        return 0, {}, [], ""


def com1DFACoreTask(simDict, inputSimFiles, avalancheDir, outDir, cuSim):
    """This is a subdivision of com1DFAMain to allow for parallel execution.
    Please read this in the context of the com1DFAMain function.
    """

    simDF = pd.DataFrame()
    tCPUDF = pd.DataFrame()

    # load configuration object for current sim
    cfg = simDict[cuSim]["cfgSim"]

    # check configuraton for consistency and plausibility
    checkCfg.checkCfgConsistency(cfg)

    # fetch simHash for current sim
    simHash = simDict[cuSim]["simHash"]

    log.info("%s runs as process: %s, %s" % (cuSim, os.getpid(), threading.current_thread().ident))

    # append configuration to dataframe
    simDF = cfgUtils.appendCgf2DF(simHash, cuSim, cfg, simDF)

    # log simulation name
    log.info("Run simulation: %s" % cuSim)

    # ++++++++++PERFORM com1DFA SIMULAITON++++++++++++++++
    (
        dem,
        reportDict,
        cfgFinal,
        tCPU,
        nPartInitial,
    ) = com1DFA.com1DFACore(cfg, avalancheDir, cuSim, inputSimFiles, outDir, simHash=simHash)

    simDF.at[simHash, "nPart"] = str(int(nPartInitial))

    # append time to data frame
    tCPUDF = cfgUtils.appendTcpu2DF(simHash, tCPU, tCPUDF)

    # create hash to check if configuration didn't change
    simHashFinal = cfgUtils.cfgHash(cfgFinal)
    if simHashFinal != simHash:
        cfgUtils.writeCfgFile(avalancheDir, com1DFA, cfg, fileName="%s_butModified" % simHash)
        message = "Simulation configuration has been changed since start"
        log.error(message)
        raise AssertionError(message)

    # return simDF, tCPUDF, simDFExisting, cfg, cfgMain, dem, reportDictList
    return simDF, tCPUDF, dem, reportDict


def com1DFAPostprocess(simDF, tCPUDF, simDFExisting, cfgMain, dem, reportDictList, exportData):
    """postprocessing of simulation results: save configuration to csv, create plots and report

    Parameters
    -----------
    simDF: pandas DataFrame
        dataframe with one line per simulation and info on parameters used
    tCPUDF:
        computation time
    simDFExisting: pandas DataFrame
        dataframe with one line per simulation and info on parameters used before
        simulations have been performed
    cfgMain: configparser object
        global avaframe config
    dem: dict
        dem dictionary
    reportDictList: list
        list of dictionaries for each simulation with info for report creation
    exportData: bool
        if True result fields are exported and plots generated

    Returns
    --------
    dem: dict
        dictionary with dem header and raster data (that has been used for final sim)
    plotDict: dict
        information on result plot paths
    reportDictList: list
        list of report dictionaries for all performed simulations
    simDFNew: pandas dataFrame
        configuration dataFrame of the simulations computed and the ones that have been already
        in the Outputs folder (if no simulation computed, configuration dataFrame
        of the already existing ones)
    """

    modName = "com1DFA"
    avalancheDir = cfgMain["MAIN"]["avalancheDir"]

    # prepare for writing configuration info
    simDF = cfgUtils.convertDF2numerics(simDF)
    # add cpu time info to the dataframe
    simDF = simDF.join(tCPUDF)

    # write the actually simulated sims to a separate csv file,
    # this is used for the qgis connector
    cfgUtils.writeAllConfigurationInfo(avalancheDir, simDF, specDir="", csvName="latestSims.csv")

    # append new simulations configuration to old ones (if they exist),
    # return total dataFrame and write it to csv
    simDFNew = pd.concat([simDF, simDFExisting], axis=0)
    cfgUtils.writeAllConfigurationInfo(avalancheDir, simDFNew, specDir="")

    # write the actually simulated sims to a separate csv file
    cfgUtils.writeAllConfigurationInfo(avalancheDir, simDFNew, specDir="")

    # create plots and report
    reportDir = pathlib.Path(avalancheDir, "Outputs", modName, "reports")
    fU.makeADir(reportDir)

    # Generate plots for all peakFiles
    if exportData:
        # TODO: if adaptedDEM this needs to be changed!!
        plotDict = oP.plotAllPeakFields(avalancheDir, cfgMain["FLAGS"], modName)
    else:
        plotDict = ""
        # create contour line plot
        reportDictList, _ = outCom1DFA.createContourPlot(reportDictList, avalancheDir, simDF)

    if cfgMain["FLAGS"].getboolean("createReport"):
        # write report
        reportDictList = gR.checkAndCleanReportDictOnWinIssue872(reportDictList)

        gR.writeReport(
            reportDir,
            reportDictList,
            cfgMain["FLAGS"].getboolean("reportOneFile"),
            plotDict,
        )

    return dem, plotDict, reportDictList, simDFNew


def com1DFACore(cfg, avaDir, cuSimName, inputSimFiles, outDir, simHash=""):
    """Run main com1DFA model

    This will compute a dense flow avalanche with the settings specified in cfg and the name cuSimName

    Parameters
    ----------
    cfg : configparser object
        configuration object for simulation to be performed
    cuSimName: str
        name of simulation
    inputSimFiles: dict
        dictionary with input files, release scenario chosen according to inputSimFiles['releaseScenario']
    avaDir : str or pathlib object
        path to avalanche directory
    outDir: str or pathlib object
        path to Outputs
    simHash: str
        unique sim ID

    Returns
    -------
    reportDictList : list
        list of dictionaries that contain information on simulations that can be used for report generation
    dem: dict
        dictionary with info on header and dem data
    reportDict: dict
        dictionary that contains information on simulation that can be used for report generation
    cfg: configparser object
        configuration object for simulation to be performed
    infoDict['tCPU']: dict
        info on cpu timing
    particlesList: list
        list of particle dictionaries for all saving time steps
    """

    # select release area input data according to chosen release scenario
    inputSimFiles = gI.selectReleaseFile(inputSimFiles, cfg["INPUT"]["releaseScenario"])

    # create required input from input files
    demOri, inputSimLines = prepareInputData(inputSimFiles, cfg)

    if cfg["GENERAL"].getboolean("iniStep"):
        # append buffered release Area
        inputSimLines = pI.createReleaseBuffer(cfg, inputSimLines)

    # set thickness values for the release area, entrainment and secondary release areas
    relName, inputSimLines, badName = prepareReleaseEntrainment(
        cfg, inputSimFiles["releaseScenario"], inputSimLines
    )

    log.debug("Perform %s simulation" % cuSimName)

    # +++++++++PERFORM SIMULAITON++++++++++++++++++++++
    # for timing the sims
    startTime = time.time()

    # initialize particles, fields, dem
    particles, fields, dem, reportAreaInfo = initializeSimulation(
        cfg, outDir, demOri, inputSimLines, cuSimName
    )
    nPartInitial = particles["nPart"]

    # add reportAreaInfo to inputSimLines
    inputSimLines["reportAreaInfo"] = reportAreaInfo

    # ------------------------
    #  Start time step computation
    Tsave, infoDict, contourDictXY = DFAIterate(
        cfg, particles, fields, dem, inputSimLines, outDir, cuSimName, simHash=simHash
    )

    # write mass balance to File
    writeMBFile(infoDict, avaDir, cuSimName)

    tCPUDFA = "%.2f" % (time.time() - startTime)
    log.info(("cpu time DFA = %s s" % (tCPUDFA)))

    # write report dictionary
    reportDict = createReportDict(avaDir, cuSimName, relName, inputSimLines, cfg, reportAreaInfo)
    # add time and mass info to report
    reportDict = reportAddTimeMassInfo(reportDict, tCPUDFA, infoDict)

    if cfg["EXPORTS"].getboolean("exportData") == False:
        reportDict["contours"] = contourDictXY

    # write text file to Outputs/com1DFA/configurationFilesDone to indicate that this simulation has been performed
    configFileName = "%s.ini" % cuSimName
    for saveDir in ["configurationFilesDone", "configurationFilesLatest"]:
        configDir = pathlib.Path(avaDir, "Outputs", "com1DFA", "configurationFiles", saveDir)
        with open((configDir / configFileName), "w") as fi:
            fi.write("see directory configurationFiles for info on config")
        fi.close()

    return dem, reportDict, cfg, infoDict["tCPU"], nPartInitial


def prepareReleaseEntrainment(cfg, rel, inputSimLines):
    """set thickness values for release, secondary release and entrainment
        set flag to append _AF to release scenario name if it includes an underscore

    Parameters
    ----------
    cfg : dict
        configuration parameters - keys: relTh, secRelArea, secondaryRelTh
    rel : str
        path to release file
    inputSimLines: dict
        dictionary with dictionaries with input data infos (releaseLine, entLine, ...)

    Returns
    -------
    relName : str
        release name
    inputSimLines : dict
        dictionary with dictionaries with input data infos now updated with thickness values (releaseLine, entLine, ...)
    badName : boolean
        changed release name
    """

    # Set release areas and release thickness
    relName = rel.stem
    badName = False
    if "_" in relName:
        badName = True
        log.warning(
            "Release area scenario file name includes an underscore \
        the suffix _AF will be added for the simulation name"
        )

    # set release thickness
    if cfg["INPUT"]["relThFile"] == "":
        releaseLine = setThickness(cfg, inputSimLines["releaseLine"], "relTh")
        inputSimLines["releaseLine"] = releaseLine
    log.debug("Release area scenario: %s - perform simulations" % (relName))

    if cfg["GENERAL"].getboolean("iniStep"):
        # set release thickness for buffer
        releaseLineBuffer = setThickness(cfg, inputSimLines["releaseLineBuffer"], "relTh")
        inputSimLines["releaseLineBuffer"] = releaseLineBuffer

    if (
        cfg.getboolean("GENERAL", "secRelArea")
        and inputSimLines["entResInfo"]["flagSecondaryRelease"] == "Yes"
    ):
        if cfg["INPUT"]["secondaryRelThFile"] == "":
            secondaryReleaseLine = setThickness(cfg, inputSimLines["secondaryReleaseLine"], "secondaryRelTh")
            inputSimLines["secondaryReleaseLine"] = secondaryReleaseLine
    else:
        inputSimLines["entResInfo"]["flagSecondaryRelease"] = "No"
        secondaryReleaseLine = None
        inputSimLines["secondaryReleaseLine"] = secondaryReleaseLine

    if cfg["GENERAL"]["simTypeActual"] in ["ent", "entres"] and cfg["INPUT"]["entThFile"] == "":
        # set entrainment thickness
        entLine = setThickness(cfg, inputSimLines["entLine"], "entTh")
        inputSimLines["entLine"] = entLine

    return relName, inputSimLines, badName


def setThickness(cfg, lineTh, typeTh):
    """set thickness in line dictionary of release, entrainment, second. release

    Parameters
    -----------
    cfg: config parser
        configuration settings
    lineTh: dict
        dictionary with info on line (e.g. release area line)
    typeTh: str
        type of thickness to be set (e.g. relTh for release thickness)

    Returns
    --------
    lineTh: dict
        updated dictionary with new key: thickness and thicknessSource
    """

    # create thickness flag name
    thFlag = typeTh + "FromFile"
    # set thickness source info
    if cfg["GENERAL"].getboolean(thFlag):
        if cfg["INPUT"]["thFromIni"] != "" and typeTh in cfg["INPUT"]["thFromIni"]:
            lineTh["thicknessSource"] = ["ini file"] * len(lineTh["thickness"])
        else:
            lineTh["thicknessSource"] = ["shp file"] * len(lineTh["thickness"])
    else:
        lineTh["thicknessSource"] = ["ini file"] * len(lineTh["thickness"])

    # set thickness value info read from ini file that has been updated from shp or ini previously
    for count, id in enumerate(lineTh["id"]):
        if cfg["GENERAL"].getboolean(thFlag):
            thName = typeTh + id
            lineTh["thickness"][count] = cfg["GENERAL"].getfloat(thName)

        else:
            thName = typeTh
            lineTh["thickness"][count] = cfg["GENERAL"].getfloat(thName)

    return lineTh


def prepareInputData(inputSimFiles, cfg):
    """Fetch input data

    Parameters
    ----------
    inputSimFiles : dict
        dictionary containing

        - relFile : str, path to release area file
        - demFile : str, path to dem file in Inputs/
        - secondaryRelFile : str, path to secondaryRelease file
        - entFiles : str, path to entrainment file
        - resFile : str, path to resistance file
        - entResInfo : flag dict
        flag if Yes entrainment and/or resistance areas found and used for simulation
        flag True if a Secondary Release file found and activated

    cfg: configparser object
        configuration for simType and secondary rel

    Returns
    -------
    demOri : dict
        dictionary with dem info (header original origin), raster data correct mesh cell size
        this dem has been remeshed/read from remeshed if chosen cell size is not equal to cell size
        of DEM in Inputs/

    inputSimLines : dict
        dictionary containing

        - releaseLine : dict, release line dictionary
        - secondaryReleaseLine : dict, secondaryRelease line dictionary
        - entLine : dict, entrainment line dictionary
        - resLine : dict, resistance line dictionary
        - entrainmentArea : str, entrainment file name
        - resistanceArea : str, resistance file name
        - entResInfo : flag dict
        flag if Yes entrainment and/or resistance areas found and used for simulation
        flag True if a Secondary Release file found and activated

    """

    # load data
    entResInfo = inputSimFiles["entResInfo"].copy()
    relFile = inputSimFiles["releaseScenario"]

    # get dem dictionary - already read DEM with correct mesh cell size
    demOri = gI.initializeDEM(cfg["GENERAL"]["avalancheDir"], demPath=cfg["INPUT"]["DEM"])
    dOHeader = demOri["header"]

    # read data from relThFile if needed, already with correct mesh cell size
    # TODO: remove if not required anymore
    # relThFieldData, _ = gI.initializeRelTh(cfg, dOHeader)

    if cfg["INPUT"]["relThFile"] == "":
        # get line from release area polygon
        releaseLine = shpConv.readLine(relFile, "release1", demOri)
        releaseLine["file"] = relFile
        releaseLine["type"] = "Release"
        releaseLine["initializedFrom"] = "shapefile"
        # check for holes in release area polygons
        gI.checkForMultiplePartsShpArea(
            cfg["GENERAL"]["avalancheDir"], releaseLine, "com1DFA", type="release"
        )
        relThFieldData = ""
    else:
        relRasterPath = pathlib.Path(cfg["GENERAL"]["avalancheDir"], "Inputs", cfg["INPUT"]["relThFile"])
        relRasterDict = IOf.readRaster(relRasterPath)
        relThFieldData = relRasterDict["rasterData"]
        releaseLine = {
            "rasterData": relRasterDict["rasterData"],
            "file": relRasterPath,
            "type": "Release from raster",
        }
        releaseLine["initializedFrom"] = "raster"
        releaseLine["Name"] = "from raster"
        releaseLine["thickness"] = "from raster"
        log.info("Set %s for relThField" % relRasterPath)

    # get line from secondary release area polygon
    if cfg["GENERAL"].getboolean("secRelArea"):
        if entResInfo["flagSecondaryRelease"] == "Yes":
            if cfg["INPUT"]["secondaryRelThFile"] == "":
                secondaryReleaseFile = inputSimFiles["secondaryRelFile"]
                secondaryReleaseLine = shpConv.readLine(secondaryReleaseFile, "", demOri)
                secondaryReleaseLine["fileName"] = secondaryReleaseFile
                secondaryReleaseLine["type"] = "Secondary release"
                secondaryReleaseLine["initializedFrom"] = "shapefile"
                secondaryReleaseArea = secondaryReleaseFile.name
                # check for holes in secondary release area polygons
                gI.checkForMultiplePartsShpArea(
                    cfg["GENERAL"]["avalancheDir"],
                    secondaryReleaseLine,
                    "com1DFA",
                    type="secondary release",
                )
            else:
                secRelRasterPath = pathlib.Path(
                    cfg["GENERAL"]["avalancheDir"], "Inputs", cfg["INPUT"]["secondaryRelThFile"]
                )
                secrelRasterDict = IOf.readRaster(secRelRasterPath)
                secondaryReleaseLine = {
                    "rasterData": secrelRasterDict["rasterData"],
                    "fileName": secRelRasterPath,
                    "type": "Secondary release from raster",
                }
                secondaryReleaseLine["initializedFrom"] = "raster"
                secondaryReleaseLine["Name"] = "from raster"
                secondaryReleaseLine["thickness"] = "from raster"
                secondaryReleaseArea = secRelRasterPath.name
        else:
            message = "No secondary release file found"
            log.error(message)
            raise FileNotFoundError(message)
    else:
        secondaryReleaseLine = None
        secondaryReleaseArea = ""
        # set False
        entResInfo["flagSecondaryRelease"] = "No"

    # get line from entrainment area polygon
    if cfg["GENERAL"]["simTypeActual"] in ["ent", "entres"]:
        if cfg["INPUT"]["entThFile"] == "":
            entFile = inputSimFiles["entFile"]
            entLine = shpConv.readLine(entFile, "", demOri)
            entrainmentArea = entFile.name
            entLine["fileName"] = entFile
            entLine["type"] = "Entrainment"
            entLine["initializedFrom"] = "shapefile"
            # check for holes in entrainment area polygons
            gI.checkForMultiplePartsShpArea(
                cfg["GENERAL"]["avalancheDir"], entLine, "com1DFA", type="entrainment"
            )
        else:
            entRasterPath = pathlib.Path(cfg["GENERAL"]["avalancheDir"], "Inputs", cfg["INPUT"]["entThFile"])
            entLineDict = IOf.readRaster(entRasterPath)
            entLine = {
                "rasterData": entLineDict["rasterData"],
                "fileName": entRasterPath,
                "type": "Entraiment from raster",
            }
            entLine["initializedFrom"] = "raster"
            entLine["Name"] = "from raster"
            entLine["thickness"] = "from raster"
            entrainmentArea = entRasterPath.name
    else:
        entLine = None
        entrainmentArea = ""

    # get line from resistance area polygon or raster
    if cfg["GENERAL"]["simTypeActual"] in ["entres", "res"]:
        if inputSimFiles["entResInfo"]["resFileType"] == ".shp":
            resFile = inputSimFiles["resFile"]
            resLine = shpConv.readLine(resFile, "", demOri)
            resistanceArea = resFile.name
            resLine["fileName"] = resFile
            resLine["type"] = "Resistance"
            resLine["initializedFrom"] = "shapefile"
            # check for holes in resistance area polygons
            gI.checkForMultiplePartsShpArea(
                cfg["GENERAL"]["avalancheDir"], resLine, "com1DFA", type="resistance"
            )
        else:
            resRasterPath = pathlib.Path(cfg["GENERAL"]["avalancheDir"], "Inputs", cfg["INPUT"]["resFile"])
            resLineDict = IOf.readRaster(resRasterPath)
            resLine = {
                "rasterData": resLineDict["rasterData"],
                "fileName": resRasterPath,
                "type": "Resistance from raster",
            }
            resLine["initializedFrom"] = "raster"
            resLine["Name"] = "from raster"
            resLine["thickness"] = "from raster"
            resistanceArea = resRasterPath.name
    else:
        resLine = None
        resistanceArea = ""

    # get line from dam
    if cfg["GENERAL"].getboolean("dam"):
        if entResInfo["dam"] == "Yes":
            damFile = inputSimFiles["damFile"]
            damLine = shpConv.readLine(damFile, "", demOri)
            damLine["fileName"] = [damFile]
            damLine["type"] = "Dam"
        else:
            message = "Take dam is set, but no dam file found, skipping it"
            log.debug(message)
            damLine = None
    else:
        damLine = None

    inputSimLines = {
        "releaseLine": releaseLine,
        "secondaryReleaseLine": secondaryReleaseLine,
        "entLine": entLine,
        "resLine": resLine,
        "damLine": damLine,
        "entrainmentArea": entrainmentArea,
        "resistanceArea": resistanceArea,
        "secondaryReleaseArea": secondaryReleaseArea,
        "entResInfo": entResInfo,
        "relThField": relThFieldData,
        "muFile": inputSimFiles["muFile"],
        "xiFile": inputSimFiles["xiFile"],
        "kFile": inputSimFiles["kFile"],
        "tauCFile": inputSimFiles["tauCFile"],
    }

    return demOri, inputSimLines


def createReportDict(avaDir, logName, relName, inputSimLines, cfg, reportAreaInfo):
    """create simulaton report dictionary

    Parameters
    ----------
    logName : str
        simulation scenario name
    relName : str
        release name
    relDict : dict
        release dictionary
    cfg : configparser
        simulation configuration
    entrainmentArea : str
        entrainment file name
    resistanceArea : str
        resistance file name

    Returns
    -------
    reportST : dict
        simulation scenario dictionary
    """

    # load parameters
    dateTimeInfo = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    entInfo = reportAreaInfo["entrainment"]
    resInfo = reportAreaInfo["resistance"]
    entrainmentArea = inputSimLines["entrainmentArea"]
    resistanceArea = inputSimLines["resistanceArea"]
    relDict = inputSimLines["releaseLine"]
    secRelAreaInfo = reportAreaInfo["secRelArea"]
    if secRelAreaInfo == "No":
        secRelAreaFlag = "No"
    else:
        secRelAreaFlag = "Yes"

    # Get default cfg and convert to dict for comparison
    cfgGen = cfg["GENERAL"]

    # Create dictionary
    reportST = {}
    reportST = {
        "headerLine": {"type": "title", "title": "com1DFA Simulation"},
        "avaName": {"type": "avaName", "name": str(avaDir)},
        "simName": {"type": "simName", "name": logName},
        "time": {"type": "time", "time": dateTimeInfo},
        "Simulation Parameters": {
            "type": "list",
            "Program version": getVersion(),
            "Parameter set": "Default",
            "Release Area Scenario": relName,
            "Entrainment": entInfo,
            "Resistance": resInfo,
            "Secondary release area": secRelAreaFlag,
            "Density [kgm-3]": cfgGen["rho"],
            "Friction model": cfgGen["frictModel"],
        },
        "Release Area": {
            "type": "columns",
            "Release area scenario": relName,
            "Release Area": relDict["Name"],
            "Release thickness [m]": relDict["thickness"],
        },
    }

    # add frict parameters
    if cfgGen["frictModel"].lower() == "samosat":
        reportST["Friction model"] = {
            "type": "columns",
            "model": "samosAT",
            "mu": cfgGen["musamosat"],
            "tau0": cfgGen["tau0samosat"],
            "Rs0": cfgGen["Rs0samosat"],
            "kappa": cfgGen["kappasamosat"],
            "R": cfgGen["Rsamosat"],
            "B": cfgGen["Bsamosat"],
        }
    elif cfgGen["frictModel"].lower() == "samosatsmall":
        reportST["Friction model"] = {
            "type": "columns",
            "model": "samosATSmall",
            "mu": cfgGen["musamosatsmall"],
            "tau0": cfgGen["tau0samosatsmall"],
            "Rs0": cfgGen["Rs0samosatsmall"],
            "kappa": cfgGen["kappasamosatsmall"],
            "R": cfgGen["Rsamosatsmall"],
            "B": cfgGen["Bsamosatsmall"],
        }
    elif cfgGen["frictModel"].lower() == "samosatmedium":
        reportST["Friction model"] = {
            "type": "columns",
            "model": "samosATMedium",
            "mu": cfgGen["musamosatmedium"],
            "tau0": cfgGen["tau0samosatmedium"],
            "Rs0": cfgGen["Rs0samosatmedium"],
            "kappa": cfgGen["kappasamosatmedium"],
            "R": cfgGen["Rsamosatmedium"],
            "B": cfgGen["Bsamosatmedium"],
        }
    elif cfgGen["frictModel"].lower() == "voellmy":
        reportST["Friction model"] = {
            "type": "columns",
            "model": "Voellmy",
            "mu": cfgGen["muvoellmy"],
            "xi": cfgGen["xsivoellmy"],
        }
    elif cfgGen["frictModel"].lower() == "coulomb":
        reportST["Friction model"] = {
            "type": "columns",
            "model": "Coulomb",
            "mu": cfgGen["mucoulomb"],
        }
    elif cfgGen["frictModel"].lower() == "wetsnow":
        reportST["Friction model"] = {
            "type": "columns",
            "model": "wetsnow",
            "mu": cfgGen["mu0wetsnow"],
            "xi": cfgGen["mu0wetsnow"],
        }
    elif cfgGen["frictModel"].lower() == "spatialvoellmy":
        reportST["Friction model"] = {
            "type": "columns",
            "model": "spatialVoellmy",
            "mu file": inputSimLines["muFile"].name,
            "xi file": inputSimLines["xiFile"].name,
        }
    elif cfgGen["frictModel"].lower() == "obrienandjulien":
        reportST["Friction model"] = {
            "type": "columns",
            "model": "O´Brien and Julien",
            "alpha": cfgGen["alphaObrienAndJulien"],
            "Cmax": cfgGen["cvMaxSediment"],
            "Cv": cfgGen["cvSediment"],
            "alpha1": cfgGen["alpha1EtaObrienAndJulien"],
            "beta1": cfgGen["beta1EtaObrienAndJulien"],
            "alpha2": cfgGen["alpha2TauyObrienAndJulien"],
            "beta2": cfgGen["beta2TauyObrienAndJulien"],
        }
    elif cfgGen["frictModel"].lower() == "herschelandbulkley":
        reportST["Friction model"] = {
            "type": "columns",
            "model": "Herschel and Bulkley",
            "n": cfgGen["nHerschelAndBulkley"],
            "alpha1": cfgGen["alpha1EtaHerschelAndBulkley"],
            "beta1": cfgGen["beta1EtaHerschelAndBulkley"],
            "alpha2": cfgGen["alpha2TauyHerschelAndBulkley"],
            "beta2": cfgGen["beta2TauyHerschelAndBulkley"],
        }
    elif cfgGen["frictModel"].lower() == "bingham":
        reportST["Friction model"] = {
            "type": "columns",
            "model": "Bingham",
            "alpha1": cfgGen["alpha1EtaBingham"],
            "beta1": cfgGen["beta1EtaBingham"],
            "alpha2": cfgGen["alpha2TauyBingham"],
            "beta2": cfgGen["beta2TauyBingham"],
        }

    # check if secondary release area
    if secRelAreaFlag == "Yes":
        reportST["Secondary release Area"] = secRelAreaInfo

    # Check if parameter set is modified from default, and add section to report
    if "_C_" in logName:
        reportST["Simulation Parameters"]["Parameter set"] = "Changed"
        reportST.update({"Parameters changed from default": {"type": "list"}})
        cfgDict = cfgUtils.convertConfigParserToDict(cfg)
        _, changedVals = com1DFATools.compareSimCfgToDefaultCfgCom1DFA(cfgDict)

        for key, val in changedVals.items():
            valStr = val["new_value"] + " (default is " + val["old_value"] + ")"

            reportST["Parameters changed from default"][key] = valStr

    if entInfo == "Yes":
        entDict = inputSimLines["entLine"]
        reportST.update(
            {
                "Entrainment area": {
                    "type": "columns",
                    "Entrainment area scenario": entrainmentArea,
                    "Entrainment thickness [m]": entDict["thickness"],
                    "Entrainment density [kgm-3]": cfgGen["rhoEnt"],
                }
            }
        )
    if resInfo == "Yes":
        reportST.update(
            {
                "Resistance area": {
                    "type": "columns",
                    "Resistance area scenario": resistanceArea,
                }
            }
        )

    reportST["Release Area"].update(reportAreaInfo["Release area info"])

    return reportST


def reportAddTimeMassInfo(reportDict, tCPUDFA, infoDict):
    """Add time and mass info to report"""

    # add mass info
    reportDict["Simulation Parameters"].update({"Initial mass [kg]": ("%.2f" % infoDict["initial mass"])})
    reportDict["Simulation Parameters"].update({"Final mass [kg]": ("%.2f" % infoDict["final mass"])})
    reportDict["Simulation Parameters"].update(
        {"Entrained mass [kg]": ("%.2f" % infoDict["entrained mass"])}
    )
    reportDict["Simulation Parameters"].update(
        {"Entrained volume [m3]": ("%.2f" % infoDict["entrained volume"])}
    )

    # add stop info
    reportDict["Simulation Parameters"].update(infoDict["stopInfo"])

    # add computation time to report dict
    reportDict["Simulation Parameters"].update({"Computation time [s]": tCPUDFA})

    return reportDict


def initializeMesh(cfg, demOri, num):
    """Create rectangular mesh

    Reads the DEM information, computes the normal vector field and
    boundries to the DEM. Also generates the grid for the neighbour search

    Parameters
    ----------
    demOri : dict
        dictionary with initial dem information
    num : int
        chose between 4, 6 or 8 (using then 4, 6 or 8 triangles) or
        1 to use the simple cross product method

    Returns
    -------
    dem : dict
        dictionary relocated in (0,0) and completed with normal field and
        boundaries as well as neighbour search grid information
    """

    # set origin to 0, 0 for computations, store original origin
    dem = setDEMoriginToZero(demOri)
    dem["originalHeader"] = demOri["header"].copy()
    dem["originalRasterData"] = dem["rasterData"].copy()

    # read dem header
    headerDEM = dem["header"]
    nColsDEM = headerDEM["ncols"]
    nRowsDEM = headerDEM["nrows"]
    cszDEM = headerDEM["cellsize"]

    # get normal vector of the grid mesh
    dem = geoTrans.getNormalMesh(dem, num=num)

    # Prepare SPH grid
    headerNeighbourGrid = {}
    cszNeighbourGrid = cfg.getfloat("sphKernelRadius")
    headerNeighbourGrid["cellsize"] = cszNeighbourGrid
    headerNeighbourGrid["ncols"] = np.ceil(nColsDEM * cszDEM / cszNeighbourGrid)
    headerNeighbourGrid["nrows"] = np.ceil(nRowsDEM * cszDEM / cszNeighbourGrid)
    headerNeighbourGrid["xllcenter"] = 0
    headerNeighbourGrid["yllcenter"] = 0
    dem["headerNeighbourGrid"] = headerNeighbourGrid

    # get real Area
    dem = DFAtls.getAreaMesh(dem, num)
    projArea = nColsDEM * nRowsDEM * cszDEM * cszDEM
    areaRaster = dem["areaRaster"]
    log.debug("Largest cell area: %.2f m²" % (np.nanmax(areaRaster)))
    log.debug("Projected Area : %.2f" % projArea)
    log.debug("Total Area : %.2f" % np.nansum(areaRaster))

    return dem


def setDEMoriginToZero(demOri):
    """set origin of DEM to 0,0"""

    dem = copy.deepcopy(demOri)
    dem["header"]["xllcenter"] = 0
    dem["header"]["yllcenter"] = 0

    return dem


def initializeSimulation(cfg, outDir, demOri, inputSimLines, logName):
    """create simulaton report dictionary

    Parameters
    ----------
    cfg : str
        simulation scenario name
    outDir : pathlib path
        path to output directory (to save the dam foot line if a dam is used)
    demOri : dict
        dictionary with original dem
    inputSimLines : dict
        releaseLine : dict
            release line dictionary
        releaseLineBuffer: dict
            release line buffer dictionary - optional if iniStep True
        secondaryReleaseLine : dict
            secondary release line dictionary
        entLine : dict
            entrainment line dictionary
        resLine : dict
            resistance line dictionary
    logName : str
        simulation scenario name

    Returns
    -------
    particles : dict
        particles dictionary at initial time step
        list of secondary release particles to be used
    fields : dict
        fields dictionary at initial time step
    dem : dict
        dictionary with new dem (lower left center at origin)
    """
    cfgGen = cfg["GENERAL"]
    methodMeshNormal = cfg.getfloat("GENERAL", "methodMeshNormal")
    thresholdPointInPoly = cfgGen.getfloat("thresholdPointInPoly")
    # this is from the current release scenario set in prepareInputData - where inputSimLines is created using releaseScenario info
    relThField = inputSimLines["relThField"]

    # -----------------------
    # Initialize mesh
    log.debug("Initializing Mesh")
    dem = initializeMesh(cfgGen, demOri, methodMeshNormal)

    # ------------------------
    log.debug("Initializing main release area")
    # process release info to get it as a raster
    if cfg["GENERAL"].getboolean("iniStep"):
        message = (
            "iniStep=True is currently not supported due to recent refactoring of thickness handling. "
            "Please set iniStep=False in your configuration file. "
            "Support for iniStep will be restored in a future update."
        )
        log.error(message)
        raise NotImplementedError(message)
        releaseLine = inputSimLines["releaseLineBuffer"]
        releaseLineReal = inputSimLines["releaseLine"]
        # check if release features overlap between features
        geoTrans.prepareArea(releaseLineReal, dem, thresholdPointInPoly, combine=True, checkOverlap=True)
        buffer1 = (
            cfg["GENERAL"].getfloat("sphKernelRadius")
            * cfg["GENERAL"].getfloat("additionallyFixedFactor")
            * cfg["GENERAL"].getfloat("bufferZoneFactor")
        )
        if len(relThField) == 0:
            # if no release thickness field or function - set release according to shapefile or ini file
            # this is a list of release rasters that we want to combine
            releaseLineReal = geoTrans.prepareArea(
                releaseLineReal,
                dem,
                buffer1,
                thList=releaseLineReal["thickness"],
                combine=True,
                checkOverlap=False,
            )
        else:
            # if relTh provided - set release thickness with field or function
            releaseLineReal = geoTrans.prepareArea(
                releaseLineReal, dem, buffer1, combine=True, checkOverlap=False
            )

    else:
        releaseLine = inputSimLines["releaseLine"]
        # create release area raster if not read from file
        if inputSimLines["releaseLine"]["initializedFrom"] == "shapefile":
            # check if release features overlap between features
            geoTrans.prepareArea(releaseLine, dem, thresholdPointInPoly, combine=True, checkOverlap=True)

            # if no release thickness field or function - set release according to shapefile or ini file
            # this is a list of release rasters that we want to combine
            releaseLine = geoTrans.prepareArea(
                releaseLine,
                dem,
                np.sqrt(2),
                thList=releaseLine["thickness"],
                combine=True,
                checkOverlap=False,
            )

    # set relRaster
    relRaster = releaseLine["rasterData"]
    log.info("Release area initialized using %s " % releaseLine["initializedFrom"])

    # compute release area
    header = dem["header"]
    csz = header["cellsize"]
    # for area computation use smaller threshold to identify raster cells that lie within release line
    # as for creating particles a bigger radius is chosen as particles that lie outside are removed afterwards
    releaseInfoDict = copy.deepcopy(releaseLine)
    relAreaActualList, relAreaProjectedList, releaseInfoDict = gI.computeAreasFromRasterAndLine(
        releaseInfoDict, dem
    )
    relAreaProjected = np.sum(relAreaProjectedList)
    relAreaActual = np.sum(relAreaActualList)
    reportAreaInfo = {
        "Release area info": {
            "Projected Area [m2]": "%.2f" % (relAreaProjected),
            "Actual Area [m2]": "%.2f" % (relAreaActual),
        }
    }

    # ------------------------
    # initialize simulation
    # create primary release area particles and fields
    # TODO: check if same header if release read from raster
    releaseLine["header"] = dem["originalHeader"]
    inputSimLines["releaseLine"]["header"] = dem["originalHeader"]
    # export release area raster to file
    if cfg["EXPORTS"].getboolean("exportRasters"):
        outDir = pathlib.Path(cfgGen["avalancheDir"], "Outputs", "internalRasters")
        fU.makeADir(outDir)
        useCompression = cfg["EXPORTS"].getboolean("useCompression")
        IOf.writeResultToRaster(
            dem["originalHeader"],
            relRaster,
            (outDir / "releaseRaster"),
            flip=True,
            useCompression=useCompression,
        )
        log.info(
            "Release area raster derived from %s saved to %s"
            % (releaseLine["initializedFrom"], str(outDir / "releaseRaster"))
        )
    particles = initializeParticles(
        cfgGen,
        releaseLine,
        dem,
        inputSimLines=inputSimLines,
        logName=logName,
        relThField=relThField,
    )
    particles, fields = initializeFields(cfg, dem, particles, releaseLine)

    reportAreaInfo["Release area info"]["Model release volume [m3]"] = "%.2f" % (
        particles["mTot"] / cfgGen.getfloat("rho")
    )

    # initialize Dam
    damLine = inputSimLines["damLine"]
    # FSO: disabled writing of damFootLine inside the initializeWallLines.
    # This is not threadsafe and leads to errors on multiprocessing
    # damFootLinePath = outDir / 'dam' / 'damFootLine.shp'
    # damLine = damCom1DFA.initializeWallLines(cfgGen, dem, damLine, damFootLinePath)
    damLine = damCom1DFA.initializeWallLines(cfgGen, dem, damLine, "")
    dem["damLine"] = damLine
    if inputSimLines["damLine"] is None:
        reportAreaInfo["dam"] = "No"
    else:
        reportAreaInfo["dam"] = "Yes"

    # perform initialisation step for redistributing particles
    if cfg["GENERAL"].getboolean("iniStep"):
        startTimeIni = time.time()
        particles, fields = pI.getIniPosition(cfg, particles, dem, fields, inputSimLines, relThField)
        tIni = time.time() - startTimeIni
        log.info(
            "Ini step for initialising particles finalized, total mass: %.2f, number of particles: %d"
            % (np.sum(particles["m"]), particles["nPart"])
        )
        log.debug("Time needed for ini step: %.2f s" % (tIni))
    # ------------------------
    # process secondary release info to get it as a list of rasters
    secondaryReleaseInfo, reportAreaInfo = initializeSecRelease(
        inputSimLines, dem, relRaster, reportAreaInfo
    )
    particles["secondaryReleaseInfo"] = secondaryReleaseInfo

    # initialize entrainment and resistance
    # get info of simType and whether or not to initialize resistance and entrainment
    simTypeActual = cfgGen["simTypeActual"]
    entrMassRaster, entrEnthRaster, reportAreaInfo = initializeMassEnt(
        dem,
        simTypeActual,
        inputSimLines["entLine"],
        reportAreaInfo,
        thresholdPointInPoly,
        cfg,
    )

    # check if entrainment and release overlap
    entrMassRaster = geoTrans.checkOverlap(entrMassRaster, relRaster, "Entrainment", "Release", crop=True)
    entrEnthRaster = geoTrans.checkOverlap(entrEnthRaster, relRaster, "Entrainment", "Release", crop=True)
    # check for overlap with the secondary release area
    if secondaryReleaseInfo["flagSecondaryRelease"] == "Yes":
        for secIndex, secRelRaster in enumerate(secondaryReleaseInfo["rasterData"]):
            entrMassRaster = geoTrans.checkOverlap(
                entrMassRaster,
                secRelRaster,
                "Entrainment",
                "Secondary release ",
                crop=True,
            )
            entrEnthRaster = geoTrans.checkOverlap(
                entrEnthRaster,
                secRelRaster,
                "Entrainment",
                "Secondary release ",
                crop=True,
            )
            # export secondary release raster used for computations (after cutting potential overlap with release)
            if cfg["EXPORTS"].getboolean("exportRasters"):
                outDir = pathlib.Path(cfg["GENERAL"]["avalancheDir"], "Outputs", "internalRasters")
                useCompression = cfg["EXPORTS"].getboolean("useCompression")
                IOf.writeResultToRaster(
                    dem["originalHeader"],
                    secRelRaster,
                    (outDir / ("secondaryReleaseRaster_%d" % secIndex)),
                    flip=True,
                    useCompression=useCompression,
                )
                log.info(
                    "SecondaryRelease area raster derived from %s saved to %s"
                    % (
                        inputSimLines["entResInfo"]["secondaryRelThFileType"],
                        str(outDir / ("secondaryReleaseRaster_%d" % secIndex)),
                    )
                )
    # export entrainment raster used for computations (after cutting potential overlap with release or secondary release)
    if cfg["EXPORTS"].getboolean("exportRasters"):
        outDir = pathlib.Path(cfg["GENERAL"]["avalancheDir"], "Outputs", "internalRasters")
        useCompression = cfg["EXPORTS"].getboolean("useCompression")
        IOf.writeResultToRaster(
            dem["originalHeader"],
            entrMassRaster / cfg["GENERAL"].getfloat("rhoEnt"),
            (outDir / "entrainmentRaster"),
            flip=True,
            useCompression=useCompression,
        )
        log.info(
            "Entrainment area raster derived from %s saved to %s"
            % (inputSimLines["entResInfo"]["entThFileType"], str(outDir / "entrainmentRaster"))
        )

    # surfacic entrainment mass available (unit kg/m²)
    fields["entrMassRaster"] = entrMassRaster
    fields["entrEnthRaster"] = entrEnthRaster
    entreainableMass = np.nansum(fields["entrMassRaster"] * dem["areaRaster"])
    log.info("Mass available for entrainment: %.2f kg" % (entreainableMass))

    log.debug("Initializing resistance area")
    cResRaster, detRaster, reportAreaInfo = initializeResistance(
        cfgGen,
        dem,
        simTypeActual,
        inputSimLines["resLine"],
        reportAreaInfo,
        thresholdPointInPoly,
    )
    fields["cResRaster"] = cResRaster
    fields["detRaster"] = detRaster
    fields["cResRasterOrig"] = cResRaster
    fields["detRasterOrig"] = detRaster

    for fric in ["mu", "xi"]:
        if (inputSimLines[fric + "File"] == None) or (
            cfg["GENERAL"]["frictModel"].lower() != "spatialvoellmy"
        ):
            fields[fric + "Field"] = np.asarray([[np.nan], [np.nan]])
        else:
            fricField = IOf.readRaster(
                pathlib.Path(
                    cfg["GENERAL"]["avalancheDir"],
                    "Inputs",
                    cfg["INPUT"]["%sFile" % fric],
                )
            )
            fields[fric + "Field"] = fricField["rasterData"]

    # plot release area scenario
    outCom1DFA.plotReleaseScenarioView(
        cfgGen["avalancheDir"],
        releaseLine,
        relThField,
        reportAreaInfo,
        dem,
        ("Release Scenario %s" % logName),
        logName,
        inputSimLines,
    )

    return particles, fields, dem, reportAreaInfo


def initializeParticles(cfg, releaseLine, dem, inputSimLines="", logName="", relThField="", thName="rel"):
    """Initialize DFA simulation

    Create particles and fields dictionary according to config parameters
    release raster and dem

    Parameters
    ----------
    cfg: configparser
        configuration for DFA simulation
    releaseLine: dict
        dictionary with info on release
    dem : dict
        dictionary with dem information
    inputSimLines: dictionary
        info on input files; real releaseline info required for iniStep
    relThField: 2D numpy array
        if the release thickness is not uniform, give here the releaseRaster
    thName: str
        name rel, secondaryRel

    Returns
    -------
    particles : dict
        particles dictionary at initial time step
    fields : dict
        fields dictionary at initial time step
    """

    # get simulation parameters
    rho = cfg.getfloat("rho")
    gravAcc = cfg.getfloat("gravAcc")
    cpIce = cfg.getfloat("cpIce")
    TIni = cfg.getfloat("TIni")
    avaDir = cfg["avalancheDir"]
    massPerParticleDeterminationMethod = cfg["massPerParticleDeterminationMethod"]
    interpOption = cfg.getfloat("interpOption")

    # read dem header
    header = dem["header"]
    ncols = header["ncols"]
    nrows = header["nrows"]
    csz = header["cellsize"]
    # if the release is not constant but given by a varying function, we need both the mask giving the cells
    # to be initialized and the raster giving the flow thickness value
    relRasterMask = releaseLine["rasterData"]
    if len(relThField) == 0:
        relRaster = releaseLine["rasterData"]
    else:
        log.info("Release thickness read from %sThFile" % (thName))
        relRaster = relThField
    areaRaster = dem["areaRaster"]

    # get the initialization method used
    relThForPart = getRelThFromPart(cfg, releaseLine, relThField, thName)
    massPerPart, nPPK = com1DFATools.getPartInitMethod(cfg, csz, relThForPart)

    # initialize arrays
    partPerCell = np.zeros(np.shape(relRaster), dtype=np.int64)
    # find all non empty cells (meaning release area)
    indRelY, indRelX = np.nonzero(relRasterMask)
    if inputSimLines != "":
        indRelYReal, indRelXReal = np.nonzero(inputSimLines["releaseLine"]["rasterData"])
    else:
        indRelYReal, indRelXReal = np.nonzero(relRaster)
    iReal = list(zip(indRelYReal, indRelXReal))
    # get approximate ratio between projected and real release area
    # because relRasterMask has a none 0 value where the release is but we want a 1 there
    realArea = np.sum(areaRaster * np.where(relRasterMask > 0, 1, 0))
    projectedArea = csz * csz * np.size(indRelY)
    ratioArea = projectedArea / realArea

    # make option available to read initial particle distribution from file
    if cfg.getboolean("initialiseParticlesFromFile"):
        if cfg.getboolean("iniStep"):
            message = (
                "If initialiseParticlesFromFile is used, iniStep cannot be performed - chose only one option"
            )
            log.error(message)
            raise AssertionError(message)
        particles, hPartArray = particleTools.initialiseParticlesFromFile(
            cfg, avaDir, releaseLine["file"].stem
        )
    else:
        # initialize random generator
        rng = np.random.default_rng(cfg.getint("seed"))

        nPart = 0
        xPartArray = np.empty(0)
        yPartArray = np.empty(0)
        mPartArray = np.empty(0)
        aPartArray = np.empty(0)
        hPartArray = np.empty(0)
        idFixed = np.empty(0)
        if len(relThField) != 0 and cfg.getboolean("iniStep"):
            # set release thickness to a constant value for initialisation
            relRaster = np.where(relRaster > 0.0, cfg.getfloat("%sTh" % thName), 0.0)
            log.warning("%sThField!= 0, but relRaster set to %sTh value (from ini)" % (thName, thName))
        # loop on non empty cells
        for indRelx, indRely in zip(indRelX, indRelY):
            # compute number of particles for this cell
            hCell = relRaster[indRely, indRelx]
            aCell = areaRaster[indRely, indRelx]
            xPart, yPart, mPart, n, aPart = particleTools.placeParticles(
                hCell,
                aCell,
                indRelx,
                indRely,
                csz,
                massPerPart,
                nPPK,
                rng,
                cfg,
                ratioArea,
            )
            nPart = nPart + n
            partPerCell[indRely, indRelx] = n
            # initialize particles position, mass, height...
            xPartArray = np.append(xPartArray, xPart)
            yPartArray = np.append(yPartArray, yPart)
            mPartArray = np.append(mPartArray, mPart * np.ones(n))
            aPartArray = np.append(aPartArray, aPart * np.ones(n))
            if (indRely, indRelx) in iReal:
                idFixed = np.append(idFixed, np.zeros(n))
            else:
                idFixed = np.append(idFixed, np.ones(n))

        hPartArray = DFAtllsC.projOnRaster(
            xPartArray, yPartArray, relRaster, csz, ncols, nrows, interpOption
        )
        hPartArray = np.asarray(hPartArray)
        # for the MPPKR option use hPart and aPart to define the mass of the particle (this means, within a cell
        # partticles have the same area but may have different flow thickness which means a different mass)
        if massPerParticleDeterminationMethod == "MPPKR":
            mPartArray = rho * aPartArray * hPartArray
        # create dictionnary to store particles properties
        particles = {}
        particles["nPart"] = nPart
        particles["x"] = xPartArray
        particles["y"] = yPartArray
        # adding z component
        particles, _ = geoTrans.projectOnRaster(dem, particles, interp="bilinear")
        particles["m"] = mPartArray
        particles["idFixed"] = idFixed
    # initialize enthalpy
    particles["totalEnthalpy"] = TIni * cpIce + gravAcc * particles["z"]
    particles["massPerPart"] = massPerPart
    particles["mTot"] = np.sum(particles["m"])
    particles["tPlot"] = 0
    particles["h"] = hPartArray
    particles["ux"] = np.zeros(np.shape(hPartArray))
    particles["uy"] = np.zeros(np.shape(hPartArray))
    particles["uz"] = np.zeros(np.shape(hPartArray))
    particles["uAcc"] = np.zeros(np.shape(hPartArray))
    particles["velocityMag"] = np.zeros(np.shape(hPartArray))
    particles["trajectoryLengthXY"] = np.zeros(np.shape(hPartArray))
    particles["trajectoryLengthXYCor"] = np.zeros(np.shape(hPartArray))
    particles["trajectoryLengthXYZ"] = np.zeros(np.shape(hPartArray))
    particles["trajectoryAngle"] = np.zeros(np.shape(hPartArray))
    particles["stoppCriteria"] = False
    mPartArray = particles["m"]
    kineticEne = np.sum(0.5 * mPartArray * DFAtls.norm2(particles["ux"], particles["uy"], particles["uz"]))
    particles["kineticEne"] = kineticEne
    particles["potentialEne"] = np.sum(gravAcc * mPartArray * particles["z"])
    particles["peakKinEne"] = kineticEne
    particles["peakForceSPH"] = 0.0
    particles["forceSPHIni"] = 0.0
    particles["peakMassFlowing"] = 0
    particles["simName"] = logName
    particles["xllcenter"] = dem["originalHeader"]["xllcenter"]
    particles["yllcenter"] = dem["originalHeader"]["yllcenter"]
    particles["nExitedParticles"] = 0.0
    particles["dmDet"] = np.zeros(np.shape(hPartArray))
    particles["dmEnt"] = np.zeros(np.shape(hPartArray))

    # remove particles that might lay outside of the release polygon
    if (
        not cfg.getboolean("iniStep")
        and not cfg.getboolean("initialiseParticlesFromFile")
        and len(relThField) == 0
    ):
        particles = geoTrans.checkParticlesInRelease(
            particles, releaseLine, cfg.getfloat("thresholdPointInPoly")
        )
        log.info("Particles that lie outside of release polygon removed")

    # add a particles ID:
    # integer ranging from 0 to nPart in the initialisation.
    # Everytime that a new particle is created, it gets a new ID that is > nID
    # where nID is the number of already used IDs
    # (enable tracking of particles even if particles are added or removed)
    # unique identifier for each particle
    particles["ID"] = np.arange(particles["nPart"], dtype=np.int64)
    # keep track of the identifier (usefull to add identifier to newparticles)
    particles["nID"] = particles["nPart"]
    # keep track of parents (usefull for new particles created after splitting)
    particles["parentID"] = np.arange(particles["nPart"])

    # initialize time
    t = 0
    particles["t"] = t

    relCells = np.size(indRelY)
    partPerCell = particles["nPart"] / relCells

    if massPerParticleDeterminationMethod != "MPPKR":
        # we need to set the nPPK
        aTot = np.sum(particles["m"] / (rho * particles["h"]))
        # average number of particles per kernel radius
        nPPK = particles["nPart"] * math.pi * csz**2 / aTot
    particles["nPPK"] = nPPK

    log.info(
        "Initialized particles. MTot = %.2f kg, %s particles in %.2f cells."
        % (particles["mTot"], particles["nPart"], relCells)
    )
    log.info(
        "Mass per particle = %.2f kg and particles per cell = %.2f."
        % (particles["mTot"] / particles["nPart"], partPerCell)
    )

    if debugPlot:
        debPlot.plotPartIni(particles, dem)

    # space for deposited particles TODO: extra dict for them?
    particles["stoppedParticles"] = {
        "x": np.empty(0),
        "y": np.empty(0),
        "m": np.empty(0),
        "ID": np.empty(0),
        "velocityMag": np.empty(0),
    }
    return particles


def getRelThFromPart(cfg, releaseLine, relThField, thName):
    """get release thickness for initialising particles - use max value

    Parameters
    -----------
    cfg: configparser object
        configuration settings
    releaseLine: dict
        info on releaseline (thickness)
    relThField: numpy array or str
        release thickness field if used, else empty string
    thName: str
        name of thickness info: rel, secondaryRel

    Returns
    --------
    relThForPart: float
        max value of release thickness
    """

    if len(relThField) != 0:
        relThForPart = np.amax(relThField)
    elif cfg.getboolean("%sThFromFile" % thName):
        relThForPart = np.amax(np.asarray(releaseLine["thickness"], dtype=float))
    else:
        relThForPart = cfg.getfloat("%sTh" % thName)

    return relThForPart


def initializeFields(cfg, dem, particles, releaseLine):
    """Initialize fields and update particles flow thickness. Eventually build bond array if snowSlide is activated

    Parameters
    ----------
    cfg: configparser
        configuration for DFA simulation
    dem : dict
        dictionary with dem information
    particles : dict
        particles dictionary at initial time step

    Returns
    -------
    particles : dict
        particles dictionary at initial time step updated with the flow thickness
    fields : dict
        fields dictionary at initial time step
    releaseLine : dict
        release line dictionary
    """
    # read config
    cfgGen = cfg["GENERAL"]
    # what result types are desired as output (we need this to decide which fields we actually need to compute)
    resTypes = fU.splitIniValueToArraySteps(cfgGen["resType"])
    resTypesReport = fU.splitIniValueToArraySteps(cfg["REPORT"]["plotFields"])
    resTypesLast = list(set(resTypes + resTypesReport))
    # read dem header
    header = dem["header"]
    ncols = header["ncols"]
    nrows = header["nrows"]
    # initialize fields
    fields = {}
    fields["pfv"] = np.zeros((nrows, ncols))
    fields["pft"] = np.zeros((nrows, ncols))
    fields["FV"] = np.zeros((nrows, ncols))
    fields["FT"] = np.zeros((nrows, ncols))
    fields["FM"] = np.zeros((nrows, ncols))
    fields["Vx"] = np.zeros((nrows, ncols))
    fields["Vy"] = np.zeros((nrows, ncols))
    fields["Vz"] = np.zeros((nrows, ncols))
    fields["dmDet"] = np.zeros((nrows, ncols))
    fields["FTStop"] = np.zeros((nrows, ncols))
    fields["FTDet"] = np.zeros((nrows, ncols))
    fields["FTEnt"] = np.zeros((nrows, ncols))
    fields["sfcChange"] = np.zeros((nrows, ncols))
    fields["sfcChangeTotal"] = np.zeros((nrows, ncols))
    fields["demAdapted"] = np.zeros((nrows, ncols))
    # for optional fields, initialize with dummys (minimum size array). The cython functions then need something
    # even if it is empty to run properly
    if ("TA" in resTypesLast) or ("pta" in resTypesLast):
        fields["pta"] = np.zeros((nrows, ncols))
        fields["TA"] = np.zeros((nrows, ncols))
        fields["computeTA"] = True
        log.debug("Computing Travel Angle")
    else:
        fields["pta"] = np.zeros((1, 1))
        fields["TA"] = np.zeros((1, 1))
        fields["computeTA"] = False
    if "pke" in resTypesLast:
        fields["pke"] = np.zeros((nrows, ncols))
        fields["computeKE"] = True
        log.debug("Computing Kinetic energy")
    else:
        fields["pke"] = np.zeros((1, 1))
        fields["computeKE"] = False
    if ("P" in resTypesLast) or ("ppr" in resTypesLast):
        fields["P"] = np.zeros((nrows, ncols))
        fields["ppr"] = np.zeros((nrows, ncols))
        fields["computeP"] = True
        log.debug("Computing Pressure")
    else:
        fields["P"] = np.zeros((1, 1))
        fields["ppr"] = np.zeros((1, 1))
        fields["computeP"] = False

    particles = DFAfunC.getNeighborsC(particles, dem)
    particles, fields = DFAfunC.updateFieldsC(cfgGen, particles, dem, fields)

    if cfgGen.getint("snowSlide") == 1:
        # Initialize the bonds between particles if the snowSlide is activated
        # get particles
        x = particles["x"]
        y = particles["y"]
        # get release outline (in order to remove spurious bonds)
        xOutline = releaseLine["x"] - dem["originalHeader"]["xllcenter"]
        yOutline = releaseLine["y"] - dem["originalHeader"]["yllcenter"]
        # original triangulation (make delaunay triangulation on points)
        triangles = tri.Triangulation(x, y)

        # Cleaning up the triangles (remove unwanted bonds)
        # masking triangles exiting the release (plan small buffer to be sure to keep all inner edges)
        # this happends on non-convex release areas
        outline = sPolygon(zip(xOutline, yOutline)).buffer(cfgGen.getfloat("thresholdPointInPoly"))
        mask = [
            not outline.contains(sPolygon(zip(x[tri], y[tri]))) for tri in triangles.get_masked_triangles()
        ]
        triangles.set_mask(mask)
        # masking triangles with sidelength bigger than some threshold
        # (this removes spurious bonds that appear on the sides of the release polygon)
        triang = triangles.triangles
        # Mask off unwanted triangles.
        xtri = x[triang] - np.roll(x[triang], 1, axis=1)
        ytri = y[triang] - np.roll(y[triang], 1, axis=1)
        maxi = np.max(np.sqrt(xtri**2 + ytri**2), axis=1)
        if cfgGen["initPartDistType"] == "triangular":
            # if it is a regular triangular mesh, remove all bonds longer than the triangle side size (np.nanmin(maxi))
            maxRadius = np.nanmin(maxi) * 1.1
        else:
            # otherwise remove bonds that are longer than sqrt(5)csz (worst case senario with one particle per cell)
            maxRadius = dem["header"]["cellsize"] * math.sqrt(5)
        triangles.set_mask(maxi > maxRadius)

        # build the bond array from the triagular mesh (put it to a format that suits us for cython)
        particles = DFAfunC.initializeBondsC(particles, triangles)
        # debugg plot
        if debugPlot:
            debPlot.plotBondsSnowSlideFinal(cfg, particles, dem)
    return particles, fields


def initializeSecRelease(inputSimLines, dem, relRaster, reportAreaInfo):
    """Initialize secondary release area

    Parameters
    ----------
    inputSimLines : dict
        dict with

        - entResInfo : dict, with the flagSecondaryRelease
        - secondaryReleaseLine : dict, secondary release line dictionary

    dem: dict
        dem dictionary
    relRaster: 2D numpy array
        release Raster (to check overlap)
    reportAreaInfo: dict
        simulation area information dictionary

    Returns
    -------
    secondaryReleaseInfo: dict
        inputSimLines['secondaryReleaseLine'] dictionary completed with

        - header: the dem original header
        - rasterData: list of secondary release rasters (without the overlapping part with the release)
        - flagSecondaryRelease: 'Yes' if a secondary release is there

    reportAreaInfo: dict
        updated simulation area information dictionary

    """
    if inputSimLines["entResInfo"]["flagSecondaryRelease"] == "Yes":
        secondaryReleaseInfo = inputSimLines["secondaryReleaseLine"]
        log.info("Initializing secondary release area: %s" % secondaryReleaseInfo["fileName"])
        log.info("Secondary release area features: %s" % (secondaryReleaseInfo["Name"]))
        secondaryReleaseInfo["header"] = dem["originalHeader"]

        if secondaryReleaseInfo["initializedFrom"] == "shapefile":
            # fetch secondary release areas
            secondaryReleaseInfo = geoTrans.prepareArea(
                secondaryReleaseInfo,
                dem,
                np.sqrt(2),
                thList=secondaryReleaseInfo["thickness"],
                combine=False,
                checkOverlap=False,
            )
        # remove overlaping parts of the secondary release area with the main release areas
        noOverlaprasterList = []
        if secondaryReleaseInfo["initializedFrom"] == "raster":
            noOverlaprasterList = [
                geoTrans.checkOverlap(
                    secondaryReleaseInfo["rasterData"],
                    relRaster,
                    "Secondary release " + secondaryReleaseInfo["Name"],
                    "Release",
                    crop=True,
                )
            ]
            secondaryReleaseInfo["Name"] = [secondaryReleaseInfo["Name"]]
        else:
            for secRelRatser, secRelName in zip(
                secondaryReleaseInfo["rasterData"], secondaryReleaseInfo["Name"]
            ):
                noOverlaprasterList.append(
                    geoTrans.checkOverlap(
                        secRelRatser,
                        relRaster,
                        "Secondary release " + secRelName,
                        "Release",
                        crop=True,
                    )
                )
        secondaryReleaseInfo["flagSecondaryRelease"] = "Yes"
        # replace the rasterData with noOverlaprasterList (which is the list of rasterData without the overlapping
        # part with the release)
        secRelInfoCopy = copy.deepcopy(secondaryReleaseInfo)
        nameListSecRel = secRelInfoCopy["Name"]
        thicknessListSecRel = secRelInfoCopy["thickness"]
        secondaryReleaseInfo["rasterData"] = noOverlaprasterList
        reportAreaInfo["secRelArea"] = {
            "type": "columns",
            "Secondary release area scenario": secRelInfoCopy["fileName"].stem,
            "features": nameListSecRel,
            "thickness [m]": thicknessListSecRel,
            "features released at time [s]": [],
        }
    else:
        secondaryReleaseInfo = {}
        secondaryReleaseInfo["flagSecondaryRelease"] = "No"
        reportAreaInfo["secRelArea"] = "No"

    return secondaryReleaseInfo, reportAreaInfo


def initializeMassEnt(dem, simTypeActual, entLine, reportAreaInfo, thresholdPointInPoly, cfg):
    """Initialize mass for entrainment

    Parameters
    ----------
    dem: dict
        dem dictionary
    simTypeActual: str
        simulation type
    entLine: dict
        entrainment line dictionary
    reportAreaInfo: dict
        simulation area information dictionary
    thresholdPointInPoly: float
        threshold val that decides if a point is in the polygon, on the line or
        very close but outside
    cfg: config parser
        configuration

    Returns
    -------
    entrMassRaster : 2D numpy array
        raster of available mass for entrainment
    reportAreaInfo: dict
        simulation area information dictionary completed with entrainment area info
    """
    # read dem header
    header = dem["originalHeader"]
    ncols = header["ncols"]
    nrows = header["nrows"]
    if "ent" in simTypeActual:
        entrainmentArea = entLine["fileName"]
        log.info("Initializing entrainment area: %s" % (entrainmentArea))
        log.info("Entrainment area features: %s" % (entLine["Name"]))
        if entLine["initializedFrom"] == "shapefile":
            entLine = geoTrans.prepareArea(entLine, dem, thresholdPointInPoly, thList=entLine["thickness"])
        entrMassRaster = entLine["rasterData"]
        # ToDo: not used in samos but implemented
        # tempRaster = cfg['GENERAL'].getfloat('entTempRef') + (dem['rasterData'] - cfg['GENERAL'].getfloat('entMinZ'))
        # * cfg['GENERAL'].getfloat('entTempGrad')
        # entrEnthRaster = np.where(tempRaster < 0, tempRaster*cfg['GENERAL'].getfloat('cpIce'),
        #                           tempRaster*cfg['GENERAL'].getfloat('cpWtr')/cfg['GENERAL'].getfloat('hFusion'))
        entrEnthRaster = np.where(
            entrMassRaster > 0,
            cfg["GENERAL"].getfloat("entTempRef") * cfg["GENERAL"].getfloat("cpIce"),
            0,
        )
        reportAreaInfo["entrainment"] = "Yes"
    else:
        entrMassRaster = np.zeros((nrows, ncols))
        entrEnthRaster = np.zeros((nrows, ncols))
        reportAreaInfo["entrainment"] = "No"

    entrMassRaster = entrMassRaster * cfg["GENERAL"].getfloat("rhoEnt")

    return entrMassRaster, entrEnthRaster, reportAreaInfo


def initializeResistance(cfg, dem, simTypeActual, resLine, reportAreaInfo, thresholdPointInPoly):
    """Initialize resistance matrix

    Parameters
    ----------
    dem: dict
        dem dictionary
    simTypeActual: str
        simulation type
    resLine: dict
        resistance line dictionary
    reportAreaInfo: dict
        simulation area information dictionary
    thresholdPointInPoly: float
        threshold val that decides if a point is in the polygon, on the line or
        very close but outside

    Returns
    -------
    cResRaster : 2D numpy array
        raster of resistance coefficients
    detRaster : 2D numpy array
        raster of detrainment coefficients
    reportAreaInfo: dict
        simulation area information dictionary completed with entrainment area info
    """
    K = cfg.getfloat("detK")
    detrainment = cfg.getboolean("detrainment")

    # read dem header
    header = dem["originalHeader"]
    ncols = header["ncols"]
    nrows = header["nrows"]

    if simTypeActual in ["entres", "res"]:
        ResModel = cfg["ResistanceModel"].lower()
        if ResModel == "default":
            cRes = cfg.getfloat("cResH")
        else:
            message = "Resistance model %s not a valid option" % ResModel
            log.error(message)
            raise AssertionError(message)

        resistanceArea = resLine["fileName"]
        log.info("Initializing resistance area: %s" % (resistanceArea))
        log.info("Resistance area features: %s" % (resLine["Name"]))
        if resLine["initializedFrom"] == "shapefile":
            resLine = geoTrans.prepareArea(resLine, dem, thresholdPointInPoly)
            mask = resLine["rasterData"]
        else:
            log.info(
                "Resistance area where values > 0, no resistance everywhere else based on raster file %s"
                % resLine["fileName"]
            )
            mask = np.where(resLine["rasterData"] > 0, 1, 0)
        # Combine constants (d, cw, sres) to one parameter cRes
        cResRaster = cRes * mask
        reportAreaInfo["resistance"] = "Yes"

        if detrainment:
            log.info("Initializing detrainment (resistance) area: %s" % (resistanceArea))
            log.info("Detrainment (Resistance) area features: %s" % (resLine["Name"]))
            detRaster = K * mask
            reportAreaInfo["detrainment"] = "Yes"
        else:
            detRaster = np.zeros((nrows, ncols))
            reportAreaInfo["detrainment"] = "No"
    else:
        cResRaster = np.zeros((nrows, ncols))
        reportAreaInfo["resistance"] = "No"
        detRaster = np.zeros((nrows, ncols))
        reportAreaInfo["detrainment"] = "No"

    return cResRaster, detRaster, reportAreaInfo


def DFAIterate(cfg, particles, fields, dem, inputSimLines, outDir, cuSimName, simHash=""):
    """Perform time loop for DFA simulation
     Save results at desired intervals

    Parameters
    ----------
    cfg: configparser
        configuration for DFA simulation
    particles : dict
        particles dictionary at initial time step

        - secondaryReleaseParticles : list, of secondary release area particles dictionaries at initial time step

    fields : dict
        fields dictionary at initial time step
    dem : dict
        dictionary with dem information
    inputSimLines : dict
        dictionary with input data dictionaries (releaseLine, entLine, ...)

    Returns
    -------
    particlesList : list
        list of particles dictionary
    fieldsList : list
        list of fields dictionary (for each time step saved)
    tCPU : dict
        computation time dictionary
    infoDict : dict
        Dictionary of all simulations carried out

    """

    cfgGen = cfg["GENERAL"]
    # Initialise cpu timing
    tCPU = {
        "timeLoop": 0,
        "timeForce": 0.0,
        "timeForceSPH": 0.0,
        "timePos": 0.0,
        "timeNeigh": 0.0,
        "timeField": 0.0,
    }

    # Load configuration settings
    tEnd = cfgGen.getfloat("tEnd")
    dtSave = fU.splitTimeValueToArrayInterval(cfgGen["tSteps"], tEnd)
    sphOption = cfgGen.getint("sphOption")
    log.debug("using sphOption %s:" % sphOption)
    # desired output fields
    resTypes = fU.splitIniValueToArraySteps(cfgGen["resType"])
    # add particles to the results type if trackParticles option is activated
    if cfg.getboolean("TRACKPARTICLES", "trackParticles"):
        resTypes = list(set(resTypes + ["particles"]))
    # make sure to save all desiered resuts for first and last time step for
    # the report
    resTypesReport = fU.splitIniValueToArraySteps(cfg["REPORT"]["plotFields"])
    resTypesLast = list(set(resTypes + resTypesReport))
    # derive friction type
    # turn friction model into integer
    frictModelsList = [
        "samosat",
        "coulomb",
        "voellmy",
        "wetsnow",
        "samosatsmall",
        "samosatmedium",
        "voellmyminshear",
        "coulombminshear",
        "spatialvoellmy",
        "obrienandjulien",
        "herschelandbulkley",
        "bingham",
    ]
    frictModel = cfgGen["frictModel"].lower()
    frictType = frictModelsList.index(frictModel) + 1
    log.debug("Friction Model used: %s, %s" % (frictModelsList[frictType - 1], frictType))

    # turn resistance model into integer
    ResModel = cfgGen["ResistanceModel"].lower()
    ResModelsList = [
        "default",
    ]
    resistanceType = ResModelsList.index(ResModel) + 1
    log.debug("Resistance Model used: %s, %s" % (ResModelsList[resistanceType - 1], resistanceType))

    # Initialise Lists to save fields and add initial time step
    contourDictXY = None
    timeM = []
    massEntrained = []
    massDetrained = []
    massStopped = []
    massTotal = []
    pfvTimeMax = []

    # setup a result fields info data frame to save max values of fields and avalanche front
    resultsDF = setupresultsDF(resTypesLast, cfg["VISUALISATION"].getboolean("createRangeTimeDiagram"))

    # Add different time stepping options here
    log.debug("Use standard time stepping")
    # Initialize time and counters
    nSave = 1
    tCPU["nSave"] = nSave
    nIter = 1
    nIter0 = 1
    countParticleCsv = 0
    particles["iterate"] = True
    t = particles["t"]
    log.debug("Saving results for time step t = %f s", t)

    # export initial time step
    if cfg["EXPORTS"].getboolean("exportData"):
        exportFields(cfg, t, fields, dem, outDir, cuSimName, TSave="initial")

        if "particles" in resTypes:
            outDirData = outDir / "particles"
            fU.makeADir(outDirData)
            savePartToPickle(particles, outDirData, cuSimName)

    # export particles properties for visulation
    if cfg["VISUALISATION"].getboolean("writePartToCSV"):
        particleTools.savePartToCsv(
            cfg["VISUALISATION"]["visuParticleProperties"],
            [particles],
            outDir,
            countParticleCsv=countParticleCsv,
        )
        countParticleCsv = countParticleCsv + 1

    # export particles dictionaries of saving time steps

    zPartArray0 = copy.deepcopy(particles["z"])

    # create range time diagram
    # check if range-time diagram should be performed, if yes - initialize
    if cfg["VISUALISATION"].getboolean("createRangeTimeDiagram"):
        demRT = dtAna.setDemOrigin(dem)
        mtiInfo, dtRangeTime, cfgRangeTime = dtAna.initializeRangeTime(dtAna, cfg, demRT, simHash)
        # fetch initial time step too
        mtiInfo, dtRangeTime = dtAna.fetchRangeTimeInfo(
            cfgRangeTime, cfg, dtRangeTime, t, demRT["header"], fields, mtiInfo
        )
        cfgRangeTime["GENERAL"]["simHash"] = simHash

    # add initial time step to Tsave array
    Tsave = [0]
    # derive time step for first iteration
    if cfgGen.getboolean("sphKernelRadiusTimeStepping"):
        dtSPHKR = tD.getSphKernelRadiusTimeStep(dem, cfgGen)
        dt = dtSPHKR
    else:
        # get time step
        dt = cfgGen.getfloat("dt")
    particles["dt"] = dt
    t = t + dt

    # Start time step computation
    while t <= tEnd * (1.0 + 1.0e-13) and particles["iterate"]:
        startTime = time.time()
        log.debug("Computing time step t = %f s, dt = %f s" % (t, dt))
        # Perform computations
        particles, fields, zPartArray0, tCPU, dem = computeEulerTimeStep(
            cfgGen,
            particles,
            fields,
            zPartArray0,
            dem,
            tCPU,
            frictType,
            resistanceType,
            inputSimLines["reportAreaInfo"],
        )
        # set max values of fields to dataframe
        if cfg["VISUALISATION"].getboolean("createRangeTimeDiagram"):
            rangeValue = mtiInfo["rangeList"][-1]
        else:
            rangeValue = ""
        resultsDF = addMaxValuesToDF(resultsDF, fields, t, resTypesLast, rangeValue=rangeValue)

        tCPU["nSave"] = nSave
        particles["t"] = t

        # write mass balance info
        massEntrained.append(particles["massEntrained"])
        massDetrained.append(particles["massDetrained"])
        massStopped.append(particles["massStopped"])
        massTotal.append(particles["mTot"])
        timeM.append(t)
        pfvTimeMax.append(np.nanmax(fields["FV"]))
        # print progress to terminal
        print("time step t = %f s\r" % t, end="")

        # create range time diagram
        # determine avalanche front and flow characteristics in respective coodrinate system
        if cfg["VISUALISATION"].getboolean("createRangeTimeDiagram") and t >= dtRangeTime[0]:
            mtiInfo, dtRangeTime = dtAna.fetchRangeTimeInfo(
                cfgRangeTime, cfg, dtRangeTime, t, demRT["header"], fields, mtiInfo
            )

            # create plots for tt diagram animation
            if cfgRangeTime["PLOTS"].getboolean("animate") and cfg["VISUALISATION"].getboolean("TTdiagram"):
                TTResType = cfgRangeTime["GENERAL"]["rangeTimeResType"]
                dtAnaPlots.animationPlot(
                    demRT,
                    fields[TTResType],
                    demRT["header"]["cellsize"],
                    TTResType,
                    cfgRangeTime,
                    mtiInfo,
                    t,
                )

        # make sure the array is not empty
        if t >= (dtSave[0] - 1.0e-8):
            Tsave.append(t)
            log.debug("Saving results for time step t = %f s", t)
            log.debug("MTot = %f kg, %s particles" % (particles["mTot"], particles["nPart"]))
            log.debug(("cpu time Force = %s s" % (tCPU["timeForce"] / nIter)))
            log.debug(("cpu time ForceSPH = %s s" % (tCPU["timeForceSPH"] / nIter)))
            log.debug(("cpu time Position = %s s" % (tCPU["timePos"] / nIter)))
            log.debug(("cpu time Neighbour = %s s" % (tCPU["timeNeigh"] / nIter)))
            log.debug(("cpu time Fields = %s s" % (tCPU["timeField"] / nIter)))

            # Result parameters to be exported
            if cfg["EXPORTS"].getboolean("exportData"):
                exportFields(cfg, t, fields, dem, outDir, cuSimName, TSave="intermediate")

                # export particles dictionaries of saving time steps
                if "particles" in resTypes:
                    savePartToPickle(particles, outDirData, cuSimName)

            # export particles properties for visulation
            if cfg["VISUALISATION"].getboolean("writePartToCSV"):
                particleTools.savePartToCsv(
                    cfg["VISUALISATION"]["visuParticleProperties"],
                    [particles],
                    outDir,
                    countParticleCsv=countParticleCsv,
                )
                countParticleCsv = countParticleCsv + 1

            # remove saving time steps that have already been saved
            dtSave = updateSavingTimeStep(dtSave, cfg["GENERAL"], t)

            # debug plot
            if debugPlot:
                debPlot.plotBondsSnowSlideFinal(cfg, particles, dem, inputSimLines)

        # derive time step
        if cfgGen.getboolean("sphKernelRadiusTimeStepping"):
            dt = dtSPHKR
        else:
            # get time step
            dt = cfgGen.getfloat("dt")
        particles["dt"] = dt

        t = t + dt
        nIter = nIter + 1
        nIter0 = nIter0 + 1
        tCPUtimeLoop = time.time() - startTime
        tCPU["timeLoop"] = tCPU["timeLoop"] + tCPUtimeLoop
    tCPU["nIter"] = nIter
    log.info("Ending computation at time t = %f s", t - dt)
    log.debug("Saving results for time step t = %f s", t - dt)
    log.info("MTot = %f kg, %s particles" % (particles["mTot"], particles["nPart"]))
    log.debug("Computational performances:")
    log.debug(("cpu time Force = %s s" % (tCPU["timeForce"] / nIter)))
    log.debug(("cpu time ForceSPH = %s s" % (tCPU["timeForceSPH"] / nIter)))
    log.debug(("cpu time Position = %s s" % (tCPU["timePos"] / nIter)))
    log.debug(("cpu time Neighbour = %s s" % (tCPU["timeNeigh"] / nIter)))
    log.debug(("cpu time Fields = %s s" % (tCPU["timeField"] / nIter)))
    log.debug(("cpu time timeLoop = %s s" % (tCPU["timeLoop"] / nIter)))
    log.debug(
        (
            "cpu time total other = %s s"
            % (
                (
                    tCPU["timeForce"]
                    + tCPU["timeForceSPH"]
                    + tCPU["timePos"]
                    + tCPU["timeNeigh"]
                    + tCPU["timeField"]
                )
                / nIter
            )
        )
    )
    Tsave.append(t - dt)

    # debug plot
    if debugPlot:
        debPlot.plotBondsSnowSlideFinal(cfg, particles, dem, inputSimLines)

    # create infoDict for report and mass log file
    infoDict = {
        "massEntrained": massEntrained,
        "massDetrained": massDetrained,
        "massStopped": massStopped,
        "timeStep": timeM,
        "massTotal": massTotal,
        "tCPU": tCPU,
        "final mass": massTotal[-1],
        "initial mass": massTotal[0],
        "entrained mass": np.sum(massEntrained),
        "detrained mass": np.sum(massDetrained),
        "entrained volume": (np.sum(massEntrained) / cfgGen.getfloat("rhoEnt")),
        "pfvTimeMax": pfvTimeMax,
    }

    # determine if stop criterion is reached or end time
    stopCritNotReached = particles["iterate"]
    avaTime = particles["t"]
    stopCritPer = cfgGen.getfloat("stopCrit") * 100.0
    # update info dict with stopping info for report
    if stopCritNotReached:
        infoDict.update(
            {
                "stopInfo": {
                    "Stop criterion": "end Time reached: %.2f" % avaTime,
                    "Avalanche run time [s]": "%.2f" % avaTime,
                }
            }
        )
    else:
        infoDict.update(
            {
                "stopInfo": {
                    "Stop criterion": "< %.2f percent of PKE" % stopCritPer,
                    "Avalanche run time [s]": "%.2f" % avaTime,
                }
            }
        )

    # create range time diagram
    # export data for range-time diagram
    if cfg["VISUALISATION"].getboolean("createRangeTimeDiagram"):
        lastTimeStep = t - dt
        # first append final time step
        mtiInfo, dtRangeTime = dtAna.fetchRangeTimeInfo(
            cfgRangeTime,
            cfg,
            dtRangeTime,
            lastTimeStep,
            demRT["header"],
            fields,
            mtiInfo,
        )
        dtAna.exportData(mtiInfo, cfgRangeTime, "com1DFA")

    # save resultsDF to file
    resultsDFPath = pathlib.Path(cfgGen["avalancheDir"], "Outputs", "com1DFA", "resultsDF_%s.csv" % simHash)
    resultsDF.to_csv(resultsDFPath)

    if cfg["EXPORTS"].getboolean("exportData"):
        exportFields(cfg, t, fields, dem, outDir, cuSimName, TSave="final")

        # export particles dictionaries of saving time steps
        if "particles" in resTypes:
            savePartToPickle(particles, outDirData, cuSimName)
    else:
        # fetch contourline info
        contourDictXY = outCom1DFA.fetchContCoors(
            dem["header"],
            fields[cfg["VISUALISATION"]["contourResType"]],
            cfg["VISUALISATION"],
            cuSimName,
        )

    # save contour line for each sim
    contourDictXY = outCom1DFA.fetchContCoors(
        dem["header"],
        fields[cfg["VISUALISATION"]["contourResType"]],
        cfg["VISUALISATION"],
        cuSimName,
    )
    outDirDataCont = outDir / "contours"
    fU.makeADir(outDirDataCont)
    saveContToPickle(contourDictXY, outDirDataCont, cuSimName)

    # export particles properties for visulation
    if cfg["VISUALISATION"].getboolean("writePartToCSV"):
        particleTools.savePartToCsv(
            cfg["VISUALISATION"]["visuParticleProperties"],
            [particles],
            outDir,
            countParticleCsv=countParticleCsv,
        )
        countParticleCsv = countParticleCsv + 1

    if particles["nExitedParticles"] != 0.0:
        log.warning(
            "%d particles have been removed during simulation because they exited the domain"
            % particles["nExitedParticles"]
        )

    return Tsave, infoDict, contourDictXY


def setupresultsDF(resTypes, cfgRangeTime):
    """setup result fields max values dataframe for initial time step
    for all resTypes used and optional for avalanche front

    Parameters
    -----------
    resTypes: list
        list of all resultTypes
    cfgRangeTime: bool
        config info if range time diagram should be performed and rangeList is available

    Returns
    --------
    resultsDF: dataframe
        data frame with on line for iniital time step and max and mean values of fields
    """
    # TODO catch empty resTypes
    resDict = {"timeStep": [0.0]}
    for resT in resTypes:
        if resT != "particles" and resT != "FTDet":
            resDict["max" + resT] = [0.0]
    if cfgRangeTime:
        resDict["rangeList"] = [0.0]
    resultsDF = pd.DataFrame.from_dict(resDict)
    resultsDF = resultsDF.set_index("timeStep")

    return resultsDF


def addMaxValuesToDF(resultsDF, fields, timeStep, resTypes, rangeValue=""):
    """add max values of peakFields to dataframe and optionally rangeValue

    Parameters
    -----------
    fields: dict
        dict with all result type fields
    resultsDF: dataframe
        data frame with on line for each time step and max and mean values of fields
    timeStep: float
        computation time step
    resTypes: list
        list of all resultTypes
    rangeValue: float
        avalanche front location -optional

    Returns
    --------
    resultsDF: data frame
        updated data frame
    """

    newLine = []
    for resT in resTypes:
        if resT != "particles" and resT != "FTDet":
            newLine.append(np.nanmax(fields[resT]))

    if rangeValue != "":
        newLine.append(rangeValue)
    resultsDF.loc[timeStep] = newLine

    return resultsDF


def updateSavingTimeStep(dtSave, cfg, t):
    """update saving time step list

    Parameters
    -----------
    dtSave: list
        list of time steps that shall be saved
    cfg: configparser object
        configuration settings, end time step
    t: float
        actual time step

    Returns
    --------
    dtSave: list
        updated list of saving time steps

    """

    if dtSave.size == 1:
        dtSave = np.asarray([2 * cfg.getfloat("tEnd")])
    else:
        indSave = np.where(dtSave > (t + 1.0e-8))
        dtSave = dtSave[indSave]

    return dtSave


def appendFieldsParticles(fieldsList, particlesList, particles, fields, resTypes):
    """append fields and optionally particle dictionaries to list for export

    Parameters
    ------------
    particles: dict
        dictionary with particle properties
    fields: dict
        dictionary with all result type fields
    resTypes: list
        list with all result types that shall be exported

    Returns
    -------
    Fields: list
        updated list with desired result type fields dictionary
    Particles: list
        updated list with particles dicionaries
    """

    fieldAppend = {}
    for resType in resTypes:
        if resType == "particles":
            particlesList.append(copy.deepcopy(particles))
        elif resType != "":
            fieldAppend[resType] = copy.deepcopy(fields[resType])
    fieldsList.append(fieldAppend)

    return fieldsList, particlesList


def writeMBFile(infoDict, avaDir, logName):
    """write mass balance info to file

    Parameters
    -----------
    infoDict: dict
        info on mass
    avaDir: str or pathlib path
        path to avalanche directory
    logName: str
        simulation name
    """

    t = infoDict["timeStep"]
    massEntrained = infoDict["massEntrained"]
    massDetrained = infoDict["massDetrained"]
    massStopped = infoDict["massStopped"]
    massTotal = infoDict["massTotal"]
    massDetrainedTotal = np.zeros(len(massDetrained))
    for m in range(1, len(massDetrained)):
        massDetrainedTotal[m] = massDetrainedTotal[m - 1] + massDetrained[m]

    # create mass plot
    outCom1DFA.massPlot(infoDict, massDetrainedTotal, t, avaDir, logName)

    # write mass balance info to log file
    massDir = pathlib.Path(avaDir, "Outputs", "com1DFA")
    fU.makeADir(massDir)
    with open(massDir / ("mass_%s.txt" % logName), "w") as mFile:
        mFile.write("time, current, entrained, detrained, detrainedTotal, stopped\n")
        for m in range(len(t)):
            mFile.write(
                "%.02f,    %.06f,    %.06f,   %.06f,    %.06f,    %.06f\n"
                % (
                    t[m],
                    massTotal[m],
                    massEntrained[m],
                    massDetrained[m],
                    massDetrainedTotal[m],
                    massStopped[m],
                )
            )


def computeEulerTimeStep(
    cfg, particles, fields, zPartArray0, dem, tCPU, frictType, resistanceType, reportAreaInfo
):
    """compute next time step using an euler forward scheme

    Parameters
    ----------
    cfg: configparser
        configuration for DFA simulation
    particles : dict
        particles dictionary at t
    fields : dict
        fields dictionary at t
    zPartArray0 : dict
        z coordinate of particles at t=0s
    dem : dict
        dictionary with dem information
    tCPU : dict
        computation time dictionary
    frictType: int
        indicator for chosen type of friction model
    resistanceType: int
        identifier for chosen type of resistance model

    Returns
    -------
    particles : dict
        particles dictionary at t + dt
    fields : dict
        fields dictionary at t + dt
    tCPU : dict
        computation time dictionary
    dem: dict
        dictionary with dem information including the adapted DEM
    reportAreaInfo: dict
        updated secondaryReleaseInfo dictionary with report area information
    """

    # update cRes and detK rasters according to thresholds of FV and FT
    particles["tPlot"] = particles["tPlot"] + 1
    # only if entres or res sim and detrainment is used
    if cfg["simTypeActual"] in ["entres", "res"] and (
        cfg["ResistanceModel"].lower() == "default" and cfg.getboolean("detrainment")
    ):
        # update resistance area fields using thresholds
        fields = com1DFATools.updateResCoeffFields(fields, cfg)
        if debugPlot:
            outCom1DFA.plotResFields(fields, cfg, particles["tPlot"], dem, particles["mTot"])

    # get forces
    startTime = time.time()
    # loop version of the compute force
    log.debug("Compute Force C")
    particles, force, fields = DFAfunC.computeForceC(cfg, particles, fields, dem, frictType, resistanceType)
    tCPUForce = time.time() - startTime
    tCPU["timeForce"] = tCPU["timeForce"] + tCPUForce

    # compute lateral force (SPH component of the calculation)
    startTime = time.time()
    if cfg.getint("sphOption") == 0:
        force["forceSPHX"] = np.zeros(np.shape(force["forceX"]))
        force["forceSPHY"] = np.zeros(np.shape(force["forceY"]))
        force["forceSPHZ"] = np.zeros(np.shape(force["forceZ"]))
    else:
        log.debug("Compute Force SPH C")
        particles, force = DFAfunC.computeForceSPHC(
            cfg, particles, force, dem, cfg.getint("sphOption"), gradient=0
        )
    tCPUForceSPH = time.time() - startTime
    tCPU["timeForceSPH"] = tCPU["timeForceSPH"] + tCPUForceSPH

    # add bonding force if required (if snowSlide is activated)
    if cfg.getint("snowSlide") == 1:
        force, particles = DFAfunC.computeCohesionForceC(cfg, particles, force)

    # update velocity and particle position
    startTime = time.time()
    # particles = updatePosition(cfg, particles, dem, force)
    log.debug("Update position C")
    particles = DFAfunC.updatePositionC(cfg, particles, dem, force, fields, typeStop=0)
    tCPUPos = time.time() - startTime
    tCPU["timePos"] = tCPU["timePos"] + tCPUPos
    # Split particles
    if cfg.getint("splitOption") == 0:
        # split particles with too much mass
        # this only splits particles that grew because of entrainment
        log.debug("Split particles")
        particles = particleTools.splitPartMass(particles, cfg)
    elif cfg.getint("splitOption") == 1:
        # split merge operation
        # first update fields (compute grid values) because we need the h of the particles to get the aPart
        # ToDo: we could skip the update field and directly do the split merge. This means we would use the old h
        startTime = time.time()
        log.debug("update Fields C")
        particles, fields = DFAfunC.updateFieldsC(cfg, particles, dem, fields)
        tcpuField = time.time() - startTime
        tCPU["timeField"] = tCPU["timeField"] + tcpuField
        # Then split merge particles
        particles = particleTools.splitPartArea(particles, cfg, dem)
        particles = particleTools.mergePartArea(particles, cfg, dem)

    # release secondary release area?
    if particles["secondaryReleaseInfo"]["flagSecondaryRelease"] == "Yes":
        particles, zPartArray0, reportAreaInfo = releaseSecRelArea(
            cfg, particles, fields, dem, zPartArray0, reportAreaInfo
        )

    # get particles location (neighbours for sph)
    startTime = time.time()
    log.debug("get Neighbours C")
    particles = DFAfunC.getNeighborsC(particles, dem)

    tCPUNeigh = time.time() - startTime
    tCPU["timeNeigh"] = tCPU["timeNeigh"] + tCPUNeigh

    # update fields (compute grid values)
    startTime = time.time()
    log.debug("update Fields C")
    if fields["computeTA"]:
        particles = DFAfunC.computeTrajectoryAngleC(particles, zPartArray0)
    particles, fields = DFAfunC.updateFieldsC(cfg, particles, dem, fields)

    # adapt DEM considering erosion and deposition
    # only adapt DEM when in one grid cell the changing height > threshold
    thresholdAdaptSfc = cfg.getfloat("thresholdAdaptSfc")
    adaptStop = cfg.getboolean("adaptSfcStopped") and np.any(abs(fields["FTStop"]) > thresholdAdaptSfc)
    adaptDet = cfg.getboolean("adaptSfcDetrainment") and np.any(abs(fields["FTDet"]) > thresholdAdaptSfc)
    adaptEnt = cfg.getboolean("adaptSfcEntrainment") and np.any(abs(fields["FTEnt"]) > thresholdAdaptSfc)
    if particles["t"] > 0:
        if adaptStop or adaptDet or adaptEnt:
            dem, fields = adaptDEM(dem, fields, cfg)

    tCPUField = time.time() - startTime
    tCPU["timeField"] = tCPU["timeField"] + tCPUField

    return particles, fields, zPartArray0, tCPU, dem


def releaseSecRelArea(cfg, particles, fields, dem, zPartArray0, reportAreaInfo):
    """Release secondary release area if trigered
    Initialize particles of the trigured secondary release area and add them
    to the simulation (particles dictionary)
    """

    secondaryReleaseInfo = particles["secondaryReleaseInfo"]
    flowThicknessField = fields["FT"]
    secRelRasterList = secondaryReleaseInfo["rasterData"]
    secRelRasterNameList = secondaryReleaseInfo["Name"]
    count = 0
    indexRel = []
    for secRelRaster, secRelRasterName in zip(secRelRasterList, secRelRasterNameList):
        # do the two arrays intersect (meaning a flowing particle entered the
        # secondary release area)
        mask = (secRelRaster > 0) & (flowThicknessField > 0)
        if mask.any():
            # create secondary release area particles
            log.info("Initializing secondary release area feature %s" % secRelRasterName)
            if secondaryReleaseInfo["initializedFrom"] == "shapefile":
                secRelInfo = shpConv.extractFeature(secondaryReleaseInfo, count)
                secRelInfo["rasterData"] = secRelRaster
                secRelParticles = initializeParticles(cfg, secRelInfo, dem, thName="secondaryRel")
            else:
                secondaryReleaseInfo["rasterData"] = secRelRaster
                secRelParticles = initializeParticles(
                    cfg, secondaryReleaseInfo, dem, relThField=secRelRaster, thName="secondaryRel"
                )
            # release secondary release area by just appending the particles
            log.info(
                "Releasing secondary release area %s at t = %.2f s" % (secRelRasterName, particles["t"])
            )
            particles = particleTools.mergeParticleDict(particles, secRelParticles)
            # save index of secRel feature
            indexRel.append(secRelRasterName)
            # save initial z position for travel angle computation
            zPartArray0 = np.append(zPartArray0, copy.deepcopy(secRelParticles["z"]))
            reportAreaInfo["secRelArea"]["features released at time [s]"].append(
                "%s_t=%.2f" % (secRelRasterName, particles["t"])
            )
        count = count + 1

    secondaryReleaseInfo["rasterData"] = secRelRasterList
    particles["secondaryReleaseInfo"] = secondaryReleaseInfo
    for item in indexRel:
        iR = secRelRasterNameList.index(item)
        # remove it from the secondary release area list
        secRelRasterList.pop(iR)
        if secondaryReleaseInfo["initializedFrom"] == "shapefile":
            secondaryReleaseInfo = shpConv.removeFeature(secondaryReleaseInfo, iR)
        else:
            secondaryReleaseInfo["thickness"] = ""
            secondaryReleaseInfo["fileName"] = ""
            secondaryReleaseInfo["header"] = ""
        secRelRasterNameList.pop(iR)

    # update secondaryReleaseInfo
    secondaryReleaseInfo["rasterData"] = secRelRasterList
    particles["secondaryReleaseInfo"] = secondaryReleaseInfo

    return particles, zPartArray0, reportAreaInfo


def savePartToPickle(dictList, outDir, logName):
    """Save each dictionary from a list to a pickle in outDir; works also for one dictionary instead of list
    Note: particle coordinates are still in com1DFA reference system with origin 0,0

    Parameters
    ---------
    dictList: list or dict
        list of dictionaries or single dictionary
    outDir: str
        path to output directory
    logName : str
        simulation Id
    """

    if isinstance(dictList, list):
        for dict in dictList:
            fi = open(outDir / ("particles_%s_%09.4f.pickle" % (logName, dict["t"])), "wb")
            pickle.dump(dict, fi)
            fi.close()
    else:
        fi = open(outDir / ("particles_%s_%09.4f.pickle" % (logName, dictList["t"])), "wb")
        pickle.dump(dictList, fi)
        fi.close()


def trackParticles(cfgTrackPart, dem, particlesList):
    """track particles from initial area

    Find all particles in an initial area. Find the same particles in
    the other time steps (+ the children if they were splitted).
    Extract time series of given properties of the tracked particles

    Parameters
    -----------
    cfgTrackPart: configParser
        centerTrackPartPoint : str
            centerTrackPartPoint of the location of the particles to
            track (x|y coordinates)
        radius : str
            radius of the circle around point
        particleProperties: str
            list of particles properties to extract ('x', 'y', 'ux', 'm'...)
    dem: dict
        dem dictionary
    particlesList: list
        list of particles dictionary

    Returns
    -------
    particlesList : list
        Particles list of dict updated with the 'trackedParticles' array
        (in the array, ones for particles that are tracked, zeros otherwise)
    trackedPartProp: dict
        dictionary with time series of the wanted properties for tracked
        particles
    track: boolean
        False if no particles are tracked
    """

    # read particle properties to be extracted
    particleProperties = cfgTrackPart["particleProperties"]
    if particleProperties == "":
        particleProperties = ["x", "y", "z", "ux", "uy", "uz", "m", "h"]
    else:
        particleProperties = set(["x", "y", "z", "ux", "uy", "uz", "m", "h"] + particleProperties.split("|"))
    # read location of particle to be tracked
    radius = cfgTrackPart.getfloat("radius")
    centerList = cfgTrackPart["centerTrackPartPoint"]
    centerList = centerList.split("|")
    centerTrackPartPoint = {
        "x": np.array([float(centerList[0])]),
        "y": np.array([float(centerList[1])]),
    }
    centerTrackPartPoint["x"] = centerTrackPartPoint["x"] - dem["originalHeader"]["xllcenter"]
    centerTrackPartPoint["y"] = centerTrackPartPoint["y"] - dem["originalHeader"]["yllcenter"]

    # start by finding the particles to be tracked
    particles2Track, track = particleTools.findParticles2Track(
        particlesList[0], centerTrackPartPoint, radius
    )
    if track:
        # find those same particles and their children in the particlesList
        particlesList, nPartTracked = particleTools.getTrackedParticles(particlesList, particles2Track)

        # extract the wanted properties for the tracked particles
        trackedPartProp = particleTools.getTrackedParticlesProperties(
            particlesList, nPartTracked, particleProperties
        )
    else:
        trackedPartProp = None

    return particlesList, trackedPartProp, track


def readFields(
    inDir,
    resType,
    simName="",
    flagAvaDir=True,
    comModule="com1DFA",
    timeStep="",
    atol=1.0e-6,
):
    """Read ascii files within a directory and return List of dictionaries

    Parameters
    -----------
    inDir: str
        path to input directory
    resType: list
        list of desired result types, if string converted to list
    simName : str
        simulation name
    flagAvaDir: bool
        if True inDir corresponds to an avalanche directory and pickles are
        read from avaDir/Outputs/com1DFA/particles
    comModule: str
        module that computed the particles
    timeStep: float or list of floats
        desired time step if difference to time step of field file is smaller than atol
        field is found - optional
    atol: float
        look for matching time steps with atol tolerance - default is atol=1.e-6

    Returns
    -------
    fieldsList : list
        list of fields dictionaries
    fieldHeader: dict
        raster header corresponding to first element in fieldsList
    timeList: list
        tme corresponding to elements in fieldsList

    """

    if isinstance(resType, list) is False:
        resType = [resType]

    if flagAvaDir:
        inDir = pathlib.Path(inDir, "Outputs", comModule, "peakFiles", "timeSteps")

    # initialise list of fields dictionaries
    fieldsList = []
    timeList = []
    first = True
    for r in resType:
        # search for all files within directory
        if simName:
            name = "*" + simName + "*_" + r + "*.*"
        else:
            name = "*_" + r + "*.*"
        FieldsNameList = list(inDir.glob(name))
        timeListTemp = [float(element.stem.split("_t")[-1]) for element in FieldsNameList]
        FieldsNameList = [x for _, x in sorted(zip(timeListTemp, FieldsNameList))]
        count = 0
        for fieldsName in FieldsNameList:
            t = float(fieldsName.stem.split("_t")[-1])
            if timeStep == "" or np.isclose(timeStep, t, atol=atol).any():
                # initialize field Dict
                if first:
                    fieldsList.append({})
                    timeList.append(t)
                field = IOf.readRaster(fieldsName)
                fieldsList[count][r] = field["rasterData"]
                count = count + 1
        first = False

    if count == 0:
        log.warning("No matching fields found in %s" % inDir)
        fieldHeader = None
        fieldsList = []
    else:
        fieldHeader = field["header"]

    return fieldsList, fieldHeader, timeList


def exportFields(
    cfg,
    timeStep,
    fields,
    dem,
    outDir,
    cuSimName,
    TSave="intermediate",
    resTypesForced=[],
):
    """export result fields to Outputs directory according to result parameters and time step
    that can be specified in the configuration file
    option intermediate or final, if final also plotFields for report are exported if intermediate only resTypes are exported

    Parameters
    -----------
    cfg: dict
        configurations
    timeStep: float
        acutal time step
    fields: dict
        dictionary with resTypes fields
    dem: dict
        dictionary with dem info
    outDir: str
        outputs Directory
    cuSimName: str
        name of current simulation
    Tsave: str
        indicator if time step is initial, intermediate or final - to decide which resTypes shall be exported
    resTypesForced: list
        list of resTypes that overwrite info from configuration regarding resTypes to be exported

    Returns
    --------
    exported peak fields are saved in Outputs/com1DFA/peakFiles
    """

    resTypesGen = fU.splitIniValueToArraySteps(cfg["GENERAL"]["resType"])
    resTypesReport = fU.splitIniValueToArraySteps(cfg["REPORT"]["plotFields"])
    if "particles" in resTypesGen:
        resTypesGen.remove("particles")
    if "particles" in resTypesReport:
        resTypesReport.remove("particles")

    if TSave == "final" or TSave == "initial":
        # for last time step we need to add the report fields
        resTypes = list(set(resTypesGen + resTypesReport))
    else:
        resTypes = resTypesGen

    if resTypesForced != []:
        resTypes = resTypesForced
    for resType in resTypes:
        if resType == "FTDet":
            dmDet = fields["dmDet"]
            resField = dmDet / (cfg["GENERAL"].getfloat("rho") * dem["areaRaster"])
        elif TSave == "final" and resType == "sfcChange":
            resField = fields["sfcChangeTotal"]
        else:
            resField = fields[resType]
        if resType == "ppr":
            # convert from Pa to kPa
            resField = resField * 0.001
        if resType == "pke":
            # convert from J/cell to kJ/m²
            # (by dividing the peak kinetic energy per cell by the real area of the cell)
            resField = resField * 0.001 / dem["areaRaster"]

        dataName = cuSimName + "_" + resType + "_" + "t%.2f" % (timeStep)
        # create directory
        outDirPeak = outDir / "peakFiles" / "timeSteps"
        fU.makeADir(outDirPeak)
        outFile = outDirPeak / dataName
        useCompression = cfg["EXPORTS"].getboolean("useCompression")
        IOf.writeResultToRaster(
            dem["originalHeader"], resField, outFile, flip=True, useCompression=useCompression
        )
        log.debug(
            "Results parameter: %s has been exported to Outputs/peakFiles for time step: %.2f "
            % (resType, timeStep)
        )

        if TSave == "final":
            log.debug(
                "Results parameter: %s exported to Outputs/peakFiles for time step: %.2f - FINAL time step "
                % (resType, timeStep)
            )
            dataName = cuSimName + "_" + resType
            # create directory
            outDirPeakAll = outDir / "peakFiles"
            fU.makeADir(outDirPeakAll)
            outFile = outDirPeakAll / dataName
            useCompression = cfg["EXPORTS"].getboolean("useCompression")
            IOf.writeResultToRaster(
                dem["originalHeader"], resField, outFile, flip=True, useCompression=useCompression
            )


def prepareVarSimDict(standardCfg, inputSimFiles, variationDict, simNameExisting="", module=com1DFA):
    """Prepare a dictionary with simulations that shall be run with varying parameters following the variation dict

    Parameters
    -----------
    standardCfg : configParser object
        default configuration or local configuration
    inputSimFiles: dict
        info dict on available input data
    variationDict: dict
        dictionary with parameter to be varied as key and list of it's values
    simNameExisting: list
        list of simulation names that already exist (optional). If provided,
        only carry on simulations that do not exist
    module: module
        module to be used for task (optional)

    Returns
    -------
    simDict: dict
        dicionary with info on simHash, releaseScenario, release area file path,
        simType and contains full configuration configparser object for simulation run
    """

    # extract the name of the module
    modName = module.__name__.split(".")[-1]

    # get list of simulation types that are desired
    if "simTypeList" in variationDict:
        simTypeList = variationDict["simTypeList"]
        del variationDict["simTypeList"]
    else:
        simTypeList = standardCfg["GENERAL"]["simTypeList"].split("|")
    # get a list of simulation types that are desired AND available
    standardCfg, simTypeList = getSimTypeList(standardCfg, simTypeList, inputSimFiles)

    # set simTypeList (that has been checked if available) as parameter in variationDict
    variationDict["simTypeList"] = simTypeList
    # create a dataFrame with all possible combinations of the variationDict values
    variationDF = pd.DataFrame(product(*variationDict.values()), columns=variationDict.keys())

    # generate a dictionary of full simulation info for all simulations to be performed
    # simulation info must contain: simName, releaseScenario, relFile, configuration as dictionary
    simDict = {}

    # loop over all simulations that shall be performed according to variationDF
    # one row per simulation
    log.info("Start working on variations")
    for row in variationDF.itertuples():
        log.info("New line in variationDF-------")
        # convert full configuration to dict
        cfgSim = cfgUtils.convertConfigParserToDict(standardCfg)

        # create release scenario name for simulation name
        rel, cfgSim, relThFile = gI.fetchReleaseFile(
            inputSimFiles,
            row._asdict()["releaseScenario"],
            cfgSim,
            variationDict["releaseScenario"],
        )
        relName = rel.stem
        if "_" in relName:
            relNameSim = relName + "_AF"
        else:
            relNameSim = relName

        # update info for parameters that are given in variationDF
        for parameter in variationDict:
            # add simType
            cfgSim["GENERAL"]["simTypeActual"] = row._asdict()["simTypeList"]
            # update parameter value - now only single value for each parameter
            keyList = [
                "relThPercentVariation",
                "entThPercentVariation",
                "secondaryRelThPercentVariation",
                "relThRangeVariation",
                "entThRangeVariation",
                "secondaryRelThRangeVariation",
                "relThRangeFromCiVariation",
                "entThRangeFromCiVariation",
                "secondaryRelThRangeFromCiVariation",
                "relThDistVariation",
                "entThDistVariation",
                "secondaryRelThDistVariation",
            ]
            if parameter in keyList:
                # set thickness value according to percent variation info
                cfgSim = dP.setThicknessValueFromVariation(
                    parameter, cfgSim, cfgSim["GENERAL"]["simTypeActual"], row
                )
            elif parameter == "releaseScenario":
                cfgSim["INPUT"][parameter] = row._asdict()[parameter]
            else:
                cfgSim["GENERAL"][parameter] = row._asdict()[parameter]

        # update INPUT section - delete non relevant parameters
        if cfgSim["GENERAL"]["simTypeActual"] not in ["ent", "entres"]:
            cfgSim["INPUT"].pop("entrainmentScenario", None)
            cfgSim["INPUT"].pop("entThId", None)
            cfgSim["INPUT"].pop("entThThickness", None)
            cfgSim["INPUT"].pop("entThCi95", None)
        if cfgSim["GENERAL"]["secRelArea"] == "False":
            cfgSim["INPUT"].pop("secondaryReleaseScenario", None)
            cfgSim["INPUT"].pop("secondaryRelThId", None)
            cfgSim["INPUT"].pop("secondaryRelThThickness", None)
            cfgSim["INPUT"].pop("secondaryRelThCi95", None)

        # check if DEM in Inputs has desired mesh size
        pathToDem = dP.checkRasterMeshSize(cfgSim, inputSimFiles["demFile"], "DEM")
        cfgSim["INPUT"]["DEM"] = pathToDem
        dem = IOf.readRaster(pathlib.Path(cfgSim["GENERAL"]["avalancheDir"], "Inputs", pathToDem))

        # check extent of inputs read from raster have correct extent and cellSize
        # first release area
        if inputSimFiles["entResInfo"]["relThFileType"] in [".asc", ".tif"]:
            pathToRel, pathToRelFull, remeshedRel = dP.checkExtentAndCellSize(cfgSim, relThFile, dem, "rel")
            cfgSim["INPUT"]["relThFile"] = pathToRel
            inputSimFiles["entResInfo"]["relRemeshed"] = remeshedRel

        # secondary release area
        if (
            inputSimFiles["entResInfo"]["secondaryRelThFileType"] in [".asc", ".tif"]
            and cfgSim["GENERAL"]["secRelArea"] == "True"
        ):
            pathToSecRel, pathToSecRelFull, remeshedSecRel = dP.checkExtentAndCellSize(
                cfgSim, inputSimFiles["secondaryRelThFile"], dem, "secondaryRel"
            )
            cfgSim["INPUT"]["secondaryRelThFile"] = pathToSecRel
            inputSimFiles["entResInfo"]["secondaryRelRemeshed"] = remeshedSecRel

        if modName == "com1DFA":
            # check if spatialVoellmy is chosen that friction fields have correct extent
            if cfgSim["GENERAL"]["frictModel"].lower() == "spatialvoellmy":
                dem = IOf.readRaster(pathlib.Path(cfgSim["GENERAL"]["avalancheDir"], "Inputs", pathToDem))
                for fric in ["mu", "xi"]:
                    if inputSimFiles["entResInfo"][fric] == "Yes":
                        pathToFric, _, remeshedFric = dP.checkExtentAndCellSize(
                            cfgSim, inputSimFiles["%sFile" % fric], dem, fric
                        )
                        cfgSim["INPUT"]["%sFile" % fric] = pathToFric
                        inputSimFiles["entResInfo"]["%sRemeshed" % fric] = remeshedFric
                    else:
                        message = (
                            "spatialVoellmy friction model: %s file in Inputs/RASTERS with file ending _%s not found"
                            % (fric, fric)
                        )
                        log.error(message)
                        raise FileNotFoundError(message)

            # add info about dam file path to the cfg
            if cfgSim["GENERAL"]["dam"] == "True" and inputSimFiles["damFile"] is not None:
                cfgSim["INPUT"]["DAM"] = str(pathlib.Path("DAM", inputSimFiles["damFile"].name))

        # if tauC, mu, k used in com8 and com9 check extent of cellSize
        if modName in ["com8MoTPSA", "com9MoTVoellmy"]:
            dem = IOf.readRaster(pathlib.Path(cfgSim["GENERAL"]["avalancheDir"], "Inputs", pathToDem))

            if inputSimFiles["entResInfo"]["tauC"] == "Yes":
                pathToFric, pathToFricFull, remeshedFric = dP.checkExtentAndCellSize(
                    cfgSim, inputSimFiles["tauCFile"], dem, "tauC"
                )
                cfgSim["INPUT"]["tauCFile"] = pathToFric
                inputSimFiles["entResInfo"]["tauCRemeshed"] = remeshedFric

            if inputSimFiles["entResInfo"]["bhd"] == "Yes":
                pathToFric, pathToFricFull, remeshedFric = dP.checkExtentAndCellSize(
                    cfgSim, inputSimFiles["bhdFile"], dem, "bhd"
                )
                cfgSim["INPUT"]["bhdFile"] = pathToFric
                inputSimFiles["entResInfo"]["bhdRemeshed"] = remeshedFric

            # check if physical parameters = variable is chosen that friction fields have correct extent
            if cfgSim["Physical_parameters"]["Parameters"] == "auto":
                for fric in ["mu", "k"]:
                    if inputSimFiles["entResInfo"][fric] == "Yes":
                        pathToFric, pathToFricFull, remeshedFric = dP.checkExtentAndCellSize(
                            cfgSim, inputSimFiles["%sFile" % fric], dem, fric
                        )
                        cfgSim["INPUT"]["%sFile" % fric] = pathToFric
                        inputSimFiles["entResInfo"]["%sRemeshed" % fric] = remeshedFric

            # check if forest effects = auto is chosen that forest parameter fields have correct extent
            if "res" in row._asdict()["simTypeList"] and inputSimFiles["resFile"] is not None:
                if (
                    cfgSim["FOREST_EFFECTS"]["Forest effects"] == "auto"
                    and inputSimFiles["entResInfo"]["bhd"] == "Yes"
                ):
                    pathToForest, pathToForestFull, remeshedForest = dP.checkExtentAndCellSize(
                        cfgSim, inputSimFiles["%sFile" % "bhd"], dem, "bhd"
                    )
                    cfgSim["INPUT"]["%sFile" % "bhd"] = pathToForest
                    inputSimFiles["entResInfo"]["%sRemeshed" % "bhd"] = remeshedForest

        # add info about entrainment file path to the cfg
        if "ent" in row._asdict()["simTypeList"] and inputSimFiles["entFile"] is not None:
            if inputSimFiles["entResInfo"]["entThFileType"] != ".shp":
                pathToEnt, pathToEntFull, remeshedEnt = dP.checkExtentAndCellSize(
                    cfgSim, inputSimFiles["entThFile"], dem, "ent"
                )
                cfgSim["INPUT"]["entThFile"] = pathToEnt
                inputSimFiles["entResInfo"]["entRemeshed"] = remeshedEnt
            cfgSim["INPUT"]["entrainmentScenario"] = str(pathlib.Path("ENT", inputSimFiles["entFile"].name))

        # add info about resistance file path to the cfg
        if "res" in row._asdict()["simTypeList"] and inputSimFiles["resFile"] is not None:
            if inputSimFiles["entResInfo"]["resFileType"] != ".shp":
                pathToRes, pathToResFull, remeshedRes = dP.checkExtentAndCellSize(
                    cfgSim, inputSimFiles["resFile"], dem, "res"
                )
                cfgSim["INPUT"]["resFile"] = pathToRes
                inputSimFiles["entResInfo"]["resRemeshed"] = remeshedRes
            cfgSim["INPUT"]["resistanceScenario"] = str(pathlib.Path("RES", inputSimFiles["resFile"].name))

        # add thickness values if read from shp and not varied
        cfgSim = dP.appendThicknessToCfg(cfgSim)

        # check differences to default and add indicator to name
        defID, _ = com1DFATools.compareSimCfgToDefaultCfgCom1DFA(cfgSim, module)

        # predefine different size classification indices
        frictIndi = None
        volIndi = None

        pathToDemFull = pathlib.Path(cfgSim["GENERAL"]["avalancheDir"], "Inputs", pathToDem)

        if modName == "com1DFA":
            # if frictModel is samosATAuto compute release vol
            if cfgSim["GENERAL"]["frictModel"].lower() == "samosatauto":
                relVolume = fetchRelVolume(rel, cfgSim, pathToDemFull, inputSimFiles["secondaryRelFile"])
            else:
                relVolume = ""

            # check sphKernelRadius setting
            cfgSim = checkCfg.checkCellSizeKernelRadius(cfgSim)

            # only keep friction model parameters that are used
            cfgSim = checkCfg.checkCfgFrictionModel(cfgSim, inputSimFiles, relVolume=relVolume)

            # set frictModelIndicator, this needs to happen AFTER checkCfgFrictModel
            frictIndi = com1DFATools.setFrictTypeIndicator(cfgSim)

        elif modName == "com8MoTPSA":
            relVolume = fetchRelVolume(rel, cfgSim, pathToDemFull, inputSimFiles["secondaryRelFile"])

            # set Volume class identificator
            volIndi = setVolumeIndicator(cfgSim, relVolume)

        # convert back to configParser object
        cfgSimObject = cfgUtils.convertDictToConfigParser(cfgSim)
        # create unique hash for simulation configuration
        simHash = cfgUtils.cfgHash(cfgSimObject)

        simName = "_".join(
            filter(
                None,
                [
                    relNameSim,
                    simHash,
                    defID,
                    frictIndi or volIndi,
                    row._asdict()["simTypeList"],
                    cfgSim["GENERAL"]["modelType"],
                ],
            )
        )

        # check if simulation exists. If yes do not append it
        if simName not in simNameExisting:
            simDict[simName] = {
                "simHash": simHash,
                "releaseScenario": relName,
                "simType": row._asdict()["simTypeList"],
                "relFile": rel,
                "cfgSim": cfgSimObject,
            }
            if modName == "com1DFA":
                # write configuration file, dont need to write cfg file for com8MoTPSA (does this later when creating rcf file)
                cfgUtils.writeCfgFile(
                    cfgSimObject["GENERAL"]["avalancheDir"],
                    com1DFA,
                    cfgSimObject,
                    fileName=simName,
                )
        else:
            log.warning("Simulation %s already exists, not repeating it" % simName)

    log.info("Done preparing variations -----")
    # TODO: maybe treat this in some other way, i.e. adding an "finalDEM" or similar
    inputSimFiles.pop("demFile")
    inputSimFiles["demFile"] = pathToDemFull

    return simDict


def getSimTypeList(standardCfg, simTypeList, inputSimFiles):
    """Define available simulation types of requested types

    Parameters
    -----------
    standardCfg : configParser object
        default configuration or local configuration
    simTypeList: List
        list of simTypes to conpute (ent, null...)
    inputSimFiles: dict
        info dict on available input data

    Returns
    --------
    standardCfg : configParser object
        configuration with updated 'secRelArea' depending on if a secondary release file is available or not
    simTypeList: list
        list of requested simTypes where also the required input data is available
    """

    # read entrainment resistance info
    entResInfo = inputSimFiles["entResInfo"]

    # check if set simType is a valid option
    validSimTypesStr = "available|null|ent|entres|res"
    validSimTypes = validSimTypesStr.split("|")
    validArray = [True if item in validSimTypes else False for item in simTypeList]
    if False in validArray:
        message = "A non-valid entry found in simType, valid Types are %s" % validSimTypesStr
        log.error(message)
        raise AssertionError(message)

    # define simulation type
    if "available" in simTypeList:
        if entResInfo["flagEnt"] == "Yes" and entResInfo["flagRes"] == "Yes":
            simTypeList.append("entres")
        elif entResInfo["flagEnt"] == "Yes" and entResInfo["flagRes"] == "No":
            simTypeList.append("ent")
        elif entResInfo["flagEnt"] == "No" and entResInfo["flagRes"] == "Yes":
            simTypeList.append("res")
        # always add null simulation
        simTypeList.append("null")
        simTypeList.remove("available")

    # remove duplicate entries
    simTypeList = set(simTypeList)
    simTypeList = sorted(list(simTypeList), reverse=False)

    if "ent" in simTypeList or "entres" in simTypeList:
        if entResInfo["flagEnt"] == "No":
            message = "No entrainment file found"
            log.error(message)
            raise FileNotFoundError(message)
    if "res" in simTypeList or "entres" in simTypeList:
        if entResInfo["flagRes"] == "No":
            message = "No resistance file found"
            log.error(message)
            raise FileNotFoundError(message)
    if standardCfg["GENERAL"].getboolean("secRelArea"):
        if entResInfo["flagSecondaryRelease"] == "No":
            standardCfg["GENERAL"]["secRelArea"] = "False"
        else:
            log.info("Using the secondary release area file: %s" % inputSimFiles["secondaryRelFile"])

    return standardCfg, simTypeList


def runOrLoadCom1DFA(avalancheDir, cfgMain, runDFAModule=True, cfgFile="", deleteOutput=True):
    """Run or load DFA results depending on runDFAModule=True or False

    Parameters
    -----------
    avalancheDir: pathlib path
        avalanche directory path
    cfgMain : configParser object
        main avaframe configuration
    runDFAModule: bool
        True to run the DFA simulation Falso to load the results dataFrame and dem
    cfgFile: str or pathlib path
        path to cfgFile to read overall configuration - optional if not provided the local or default config is used
    deleteOutput: Boolean
        True to delete the com1DFA output dir before running com1DFA (used only if runDFAModule=True)

    Returns
    --------
    dem: dict
        dem dictionary
    simDF: dataFrame
        simulation results dataframe
    resTypeList: list
        list of output files resTypes available (ppr, pft...)
    """
    if runDFAModule:
        # clean avalanche directory
        iP.cleanModuleFiles(avalancheDir, com1DFA, deleteOutput=deleteOutput)
        # Run the DFA simulation
        dem, _, _, simDF = com1DFA.com1DFAMain(cfgMain, cfgInfo=cfgFile)
    else:
        # read simulation dem
        demOri = gI.readDEM(avalancheDir)
        dem = com1DFA.setDEMoriginToZero(demOri)
        dem["originalHeader"] = demOri["header"].copy()
        # load DFA results
        simDF, _ = cfgUtils.readAllConfigurationInfo(avalancheDir)
        if simDF is None:
            message = "Did not find any com1DFA simulations in %s/Outputs/com1DFA/" % avalancheDir
            log.error(message)
            raise FileExistsError(message)

    dataDF, resTypeList = fU.makeSimFromResDF(avalancheDir, "com1DFA", inputDir="", simName="")
    simDF = simDF.reset_index().merge(dataDF, on="simName").set_index("index")
    return dem, simDF, resTypeList


def fetchRelVolume(releaseFile, cfg, pathToDem, secondaryReleaseFile, radius=0.01):
    """compute release area volume using release line and thickness info and dem
    if in config settings secRelArea is True - also include secondary release area in
    release volume estimate

    Parameters
    -----------
    releaseFile: pathlib path
        path to release area shp file
    cfg: dict
        config settings of current sim
    pathToDem: pathlib path
        path to dem file used for current sim
    releaseFile: pathlib path, None
        path to secondary release area shp file or None if not available
    radius : float
        include all cells which center is in the release line or close enough

    Returns
    ---------
    relVolume: float
        volume of release area in m3


    """

    # convert back to configParser object
    cfg = cfgUtils.convertDictToConfigParser(cfg)

    # read simulation dem
    demVol = IOf.readRaster(pathToDem, noDataToNan=True)
    demVol["originalHeader"] = demVol["header"].copy()
    methodMeshNormal = cfg["GENERAL"].getfloat("methodMeshNormal")
    # get normal vector of the grid mesh
    demVol = geoTrans.getNormalMesh(demVol, num=methodMeshNormal)
    demVol = DFAtls.getAreaMesh(demVol, methodMeshNormal)

    # compute volume of release area
    relVolume = initializeRelVol(cfg, demVol, releaseFile, radius, releaseType="primary")

    if cfg["GENERAL"]["secRelArea"] == "True":
        # compute volume of secondary release area
        secondaryRelVolume = initializeRelVol(
            cfg, demVol, secondaryReleaseFile, radius, releaseType="secondary"
        )

        totalVolume = relVolume + secondaryRelVolume
        log.info(
            "release volume is: %.2f m3 and secondary release volume is: %.2f m3 - total volume: %.2f m3 - based on %.2f meter grid"
            % (relVolume, secondaryRelVolume, totalVolume, demVol["header"]["cellsize"])
        )

        relVolume = relVolume + secondaryRelVolume
    else:
        log.info(
            "%.2f meter grid based release volume is: %.2f m3" % (demVol["header"]["cellsize"], relVolume)
        )

    return relVolume


def initializeRelVol(cfg, demVol, releaseFile, radius, releaseType="primary"):
    """initialize release line and apply thickness to compute release volume

    Parameters
    cfg: dict
        configuration settings
    demVol: dct
        dictionary with info on DEM data, area
    releaseFile: pathlib path
        path to release area shp file
    radius: float
        include all cells which center is in the release line or close enough
    releaseType: str
        name of release area type, e.g. primary, secondary

    Returns
    ---------
    relVolume: float
        volume of the release area

    """

    if releaseType == "primary":
        typeTh = "relTh"
    else:
        typeTh = "secondaryRelTh"

    # check if release thickness provided as field or constant value
    if cfg["INPUT"][(typeTh + "File")] != "":
        # read relThField from file
        relThFilePath = pathlib.Path(cfg["GENERAL"]["avalancheDir"], "Inputs", cfg["INPUT"][typeTh + "File"])
        relThFieldFull = IOf.readRaster(relThFilePath)
        relThField = relThFieldFull["rasterData"]

        # mask the relThField with raster from polygon
        releaseLineMask = np.ma.masked_where(relThField == 0.0, relThField)
        relVolumeField = releaseLineMask * demVol["areaRaster"]
        relVolume = np.nansum(relVolumeField)

    else:
        # create release line
        releaseLine = {}
        releaseLine = shpConv.readLine(releaseFile, "release1", demVol)
        # check if release features overlap between features
        thresholdPointInPoly = cfg["GENERAL"].getfloat("thresholdPointInPoly")
        geoTrans.prepareArea(releaseLine, demVol, thresholdPointInPoly, combine=True, checkOverlap=True)
        releaseLine["type"] = "Release"
        # set thickness values on releaseLine
        releaseLine = setThickness(cfg, releaseLine, typeTh)
        # when creating raster from polygon apply release thickness
        releaseLine = geoTrans.prepareArea(
            releaseLine,
            demVol,
            radius,
            thList=releaseLine["thickness"],
            combine=True,
            checkOverlap=False,
        )

        # compute release volume using raster and dem area
        relVolume = np.nansum(releaseLine["rasterData"] * demVol["areaRaster"])

    return relVolume


def setVolumeIndicator(simCfg, relVolume):
    """Sets the Volume indicator for the simname based on the threshold defined in ini file

    Parameters
    -----------
    simCfg: dict
        simulation configuration
    relVolume: float
        Volume of the release area in m^3

    Returns
    --------
    VolIndi: str
        S, M or L

    """
    if relVolume < float(simCfg["GENERAL"]["volClassSmall"]):
        volIndi = "S"
    elif relVolume > float(simCfg["GENERAL"]["volClassMedium"]):
        volIndi = "L"
    else:
        volIndi = "M"

    return volIndi


def saveContToPickle(contourDictXY, outDir, cuSimName):
    """save contourline x, y coordinates dictionary to a pickle

    Parameters
    ------------
    contourDictXY: dict
        dictionary with key simName and dict with x, y coordinates of contour line of specified level
    outDir: pathlib path
        path to dir where pickle shall be saved
    cuSimName: str
        name of current simulation where this contourline is derived from
    """

    fi = open(outDir / ("contourDictXY_%s.pickle" % (cuSimName)), "wb")
    pickle.dump(contourDictXY, fi)
    fi.close()


def adaptDEM(dem, fields, cfg):
    """adapt topography in respect to erosion and deposition

    Parameters
    dem: dict
        dictionary with info on DEM data
    fields : dict
        fields dictionary
    cfg: dict
        configuration settings

    Returns
    ---------
    dem: dict
        dictionary with info on DEM data containing adapted topography
    fields : dict
        fields dictionary containing adapted DEM
    """

    ZDEM = dem["rasterData"].copy()
    FTDet = fields["FTDet"]
    FTStop = fields["FTStop"]
    FTEnt = fields["FTEnt"]
    sfcChangeTotal = fields["sfcChangeTotal"]
    sfcChange = np.zeros_like(FTDet)
    ZDEMadapt = ZDEM

    _, _, NzNormed = DFAtls.normalize(dem["Nx"].copy(), dem["Ny"].copy(), dem["Nz"].copy())

    if cfg.getboolean("adaptSfcStopped"):
        # compute thickness to depth
        depthStop = FTStop / NzNormed
        ZDEMadapt += depthStop
        sfcChange += depthStop
    if cfg.getboolean("adaptSfcDetrainment"):
        # compute thickness to depth
        depthDet = FTDet / NzNormed
        ZDEMadapt += depthDet
        sfcChange += depthDet
    if cfg.getboolean("adaptSfcEntrainment"):
        # compute thickness to depth
        depthEnt = FTEnt / NzNormed
        ZDEMadapt += depthEnt
        sfcChange += depthEnt

    dem["rasterData"] = ZDEMadapt
    fields["demAdapted"] = ZDEMadapt

    num = cfg.getfloat("methodMeshNormal")
    dem = geoTrans.getNormalMesh(dem, num=num)
    dem = DFAtls.getAreaMesh(dem, num)

    fields["sfcChange"] = sfcChange
    fields["sfcChangeTotal"] = sfcChangeTotal + sfcChange

    return dem, fields
