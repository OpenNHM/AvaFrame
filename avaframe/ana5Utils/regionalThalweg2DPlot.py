import numpy as np
import pathlib
import matplotlib.pyplot as plt
import logging
import json

import avaframe.ana5Utils.regionalThalwegTools as tools
from avaframe.in3Utils import fileHandlerUtils as fU
import avaframe.out3Plot.outAIMEC as outAIMEC
from avaframe.in3Utils import cfgUtils
from avaframe.in3Utils import cfgHandling
from avaframe.ana3AIMEC import ana3AIMEC
import avaframe.ana5Utils.preparePathGeneral as pathGen
import avaframe.in1Data.getInput as gI
from avaframe.ana5Utils import DFAPathGeneration

log = logging.getLogger(__name__)


def regionalThalweg2DPlotMain(avalanchedir, cfg):
    """
    read in Input data and general function for 2D thalweg plot

    Parameters
    -----------
    avalanchedir: str
        Path to the th avalanche directory
    cfg: configparser Object
        contains configuration settings
    """
    avalanchedir = pathlib.Path(avalanchedir)

    simhash = cfg["GENERAL"].get("simHash")
    module = cfg["GENERAL"].get("modName")
    startRow = cfg["GENERAL"].get("startRow")
    startCol = cfg["GENERAL"].get("startCol")
    relId = cfg["GENERAL"].get("relId")

    pathToOutput = avalanchedir / "Outputs" / module / "peakFiles" / f"res_{simhash}"
    savePath = pathToOutput / "ThalwegPlots"
    fU.makeADir(savePath)
    pathDict = {"avalancheDir": avalanchedir, "pathToOutput": pathToOutput, "savePath": savePath}

    # open json file that contains com4FlowPy parameters
    pathToCom4Json = avalanchedir / "Outputs" / module / simhash
    with open(f"{pathToCom4Json}.json", "r") as file:
        com4Cfg = json.load(file)

    demDict = gI.readDEM(avalanchedir)
    # TODO: Check if flipping DEM is needed!(gI.readDem flips the raster.)
    # demDict["rasterData"] = np.flipud(demDict["rasterData"])

    # check which thalweg is plotted
    if startRow != "" and startCol != "" and relId != "":
        message = "When choosing one thalweg that is plotted, only select with startcell coordinates or release Id!"
        log.error(message)
        raise ValueError(message)
    plotAllThalwegs = False
    if startCol != "" or startRow != "":
        startCol = np.int16(startCol)
        startRow = np.int16(startRow)
    elif relId != "":
        relId = np.int16(relId)
    else:
        plotAllThalwegs = True

    centerOf = cfg["GENERAL"].get("centerOfVariable")
    if centerOf == "":
        plotAllCenterOf = True
    else:
        plotAllCenterOf = False

    pathDict["titleVariables"] = {
        "startRow": startRow,
        "startCol": startCol,
        "relId": relId,
        "centerOf": centerOf,
        "simHash": simhash,
    }
    if com4Cfg["GENERAL"]["addThalwegExtension"] != "True":
        cfgDFAPath = cfgUtils.getModuleConfig(
            DFAPathGeneration,
            onlyDefault=cfg["ana5Utils_DFAPathGeneration_override"].getboolean("defaultConfig"),
        )
        # and override with settings from config
        cfgDFAPath, cfg = cfgHandling.applyCfgOverride(
            cfgDFAPath, cfg, DFAPathGeneration, addModValues=False
        )

    # FlowPy output: thalweg data
    if plotAllCenterOf:
        log.info(f"Plot all thalweg data that can be found in {pathToOutput}/ThalwegData.")
        files = sorted(list((pathToOutput / "thalwegData").glob(f"thalwegData_*.pickle")))
    elif plotAllThalwegs:
        log.info(
            f"Plot all thalwegs averaged with {centerOf} that can be found in {pathToOutput}/ThalwegData."
        )
        files = sorted(list((pathToOutput / "thalwegData").glob(f"thalwegData_{centerOf}_*.pickle")))
        if len(files) == 0:
            message = f"There is no thalweg data computed with {centerOf} in{pathToOutput}/ThalwegData."
            log.error(message)
            raise FileNotFoundError(message)
    else:
        dataThalweg = tools.readThalwegData(pathToOutput / "thalwegData", pathDict["titleVariables"])
        if com4Cfg["GENERAL"]["addThalwegExtension"] != "True":
            _, dataThalwegExtended = pathGen.preparePathGeneralMain(dataThalweg, cfgDFAPath, demDict)
            for variable in ["x", "y", "z", "s"]:
                dataThalweg[variable] = dataThalwegExtended[variable]

        plotThalweg2D(pathDict, cfg, dataThalweg)
        plotThalwegAltitude(pathDict, dataThalweg)
        plotDFAGenerationLocation(pathDict, dataThalweg, rasterVariable="fpTravelAngleMax")

    if plotAllThalwegs or plotAllCenterOf:
        for thalwegDataFile in files:
            stem = thalwegDataFile.stem
            nameParts = stem.split("_")
            if len(nameParts) == 4:
                _, centerOf, startRow, startCol = stem.split("_")
            elif len(nameParts) == 3:
                _, centerOf, relId = stem.split("_")
            pathDict["titleVariables"]["startRow"] = startRow
            pathDict["titleVariables"]["startCol"] = startCol
            pathDict["titleVariables"]["centerOf"] = centerOf
            pathDict["titleVariables"]["relId"] = relId
            # startRow = int(startRow)
            # startCol = int(startCol)
            dataThalweg = np.load(thalwegDataFile, allow_pickle="TRUE")
            if com4Cfg["GENERAL"]["addThalwegExtension"] != "True":
                _, dataThalwegExtended = pathGen.preparePathGeneralMain(dataThalweg, cfgDFAPath, demDict)
                for variable in ["x", "y", "z", "s"]:
                    dataThalweg[variable] = dataThalwegExtended[variable]
            print(dataThalweg["z"])

            plotThalweg2D(pathDict, cfg, dataThalweg)
            plotThalwegAltitude(pathDict, dataThalweg)
            plotDFAGenerationLocation(pathDict, dataThalweg, rasterVariable="fpTravelAngleMax")


def plotThalweg2D(pathDict, cfg, dataThalweg):
    """
    saves 2D thalweg plot:
    top panel: position of the thalweg in the field
    bottom panel: 2 dimensional representation

    Parameters
    ------------
    pathDict: dict
        contains the simulation paths
    cfg: configparser Object
        contains configuration settings
    dataThalweg: numpy array
        thalweg data that are saved in the simulation (averaged x-, y-coordinates, zdelta, ..)

    """
    variable = cfg["GENERAL"].get("plotVariable")
    thalwegPra = cfg["GENERAL"].getboolean("thalwegPra")
    size = cfg["GENERAL"].get("avalancheSize")
    savePath = pathDict["savePath"]
    centerOf = pathDict["titleVariables"]["centerOf"]

    if thalwegPra:
        folder = pathlib.Path(pathDict["pathToOutput"] / "thalwegData")
        files = list(folder.glob(f"thalwegData_{centerOf}*"))
        x = []
        y = []
        indStartAverageThalweg = []
        indEndAverageThalweg = []
        dataStartAverageThalweg = {}
        dataEndAverageThalweg = {}

        for thalwegFile in files:
            data = np.load(thalwegFile, allow_pickle="TRUE")
            newX = np.array(data["x"])
            newY = np.array(data["y"])

            y.append(newY)
            x.append(newX)
            if cfg["GENERAL"].getboolean("addThalwegExtension"):
                indStartAverageThalweg.append(data["indexStartAverageData"])
                indEndAverageThalweg.append(data["indexEndAverageData"])
                for key in data["startAverageData"].keys():
                    if key not in dataStartAverageThalweg.keys():
                        dataStartAverageThalweg[key] = data["startAverageData"][key]
                        dataEndAverageThalweg[key] = data["endAverageData"][key]
                    else:
                        dataStartAverageThalweg[key] = np.append(
                            dataStartAverageThalweg[key], data["startAverageData"][key]
                        )
                        dataEndAverageThalweg[key] = np.append(
                            dataEndAverageThalweg[key], data["endAverageData"][key]
                        )
    else:
        y = np.array(dataThalweg[f"y"])
        x = np.array(dataThalweg[f"x"])
        if cfg["GENERAL"].getboolean("addThalwegExtension"):
            indStartAverageThalweg = dataThalweg["indexStartAverageData"]
            indEndAverageThalweg = dataThalweg["indexEndAverageData"]
            dataStartAverageThalweg = dataThalweg["startAverageData"]
            dataEndAverageThalweg = dataThalweg["endAverageData"]

    if cfg["GENERAL"].getboolean("addThalwegExtension"):
        averageThalweg = {
            "indStartAverageThalweg": indStartAverageThalweg,
            "indEndAverageThalweg": indEndAverageThalweg,
            "dataStartAverageThalweg": dataStartAverageThalweg,
            "dataEndAverageThalweg": dataEndAverageThalweg,
        }
    else:
        averageThalweg = None
    # PLOT
    fig, axs = plt.subplots(2, 1)

    fig.set_figheight(10)
    fig.tight_layout(pad=3.0)
    fig.set_figwidth(8)

    fig, axs[0] = tools.makeFieldPlot(
        axs[0], fig, pathDict, variable, x, y, averageThalweg, thalwegPra=thalwegPra
    )
    axs[1] = tools.makeThalwegPlot(axs[1], dataThalweg, pathDict, centerOf=centerOf)

    if size != "":
        axs[0].set_title(f"Avalanche size: {size}")

    outFileNamePart = tools.getOutFileNamePartly(pathDict["titleVariables"])
    outFileName = f"Thalweg2D_{outFileNamePart}.png"
    fig.savefig(savePath / outFileName)
    log.info(f"saved plot: {(savePath / outFileName)}")


def plotDFAGenerationLocation(pathDict, profile, rasterVariable="fpTravelAngleMax"):
    savePath = pathDict["savePath"]

    fig, ax1 = plt.subplots(figsize=(10, 8), dpi=150, constrained_layout=True)
    avaProfile = profile["resampleProfile"]
    ax1 = tools.DFAThalwegPlot(ax1, avaProfile, pathDict, rasterVariable)

    outFileNamePart = tools.getOutFileNamePartly(pathDict["titleVariables"])
    outFileName = f"DFA_thalwegLocation_{outFileNamePart}.png"
    fig.savefig(savePath / outFileName)
    log.info(f"saved plot: {(savePath / outFileName)}")


def plotThalwegAltitude(pathDict, dataThalweg):
    """
    plot the AIMEC thalweg-altitude plot

    Parameters
    """
    dataThalweg["indStartOfRunout"] = 0
    dataThalweg["startOfRunoutAreaAngle"] = False

    velocityThalweg = tools.zDelta2velocity(dataThalweg["zDelta"])
    pftCrossMax = dataThalweg["flux"] * 10
    # pftCrossMax = np.ones_like(velocityThalweg) * 10

    cfg = cfgUtils.getModuleConfig(ana3AIMEC)
    cfgPlots = cfg["PLOTS"]

    simName = str(pathDict["avalancheDir"]).split("/")[-1]

    outFileNamePart = tools.getOutFileNamePartly(pathDict["titleVariables"])
    pathDict["projectName"] = outFileNamePart
    pathDict["pathResult"] = str(pathDict["savePath"])
    # TODO: we could divide the function outAIMEC.plotVelThAlongThalweg to enable modifications, e.g. the pft representation
    outAIMEC.plotVelThAlongThalweg(pathDict, dataThalweg, pftCrossMax, velocityThalweg, cfgPlots, simName)
