import numpy as np
import pathlib
import matplotlib.pyplot as plt
import logging

import avaframe.ana5Utils.regionalThalwegTools as tools
from avaframe.in3Utils import fileHandlerUtils as fU
import avaframe.out3Plot.outAIMEC as outAIMEC
from avaframe.in3Utils import cfgUtils
from avaframe.ana3AIMEC import ana3AIMEC

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
    simhash = cfg["GENERAL"].get("simHash")
    module = cfg["GENERAL"].get("modName")
    avalanchedir = pathlib.Path(avalanchedir)

    startRow = cfg["GENERAL"].get("startRow")
    startCol = cfg["GENERAL"].get("startCol")
    relId = cfg["GENERAL"].get("relId")

    pathToOutput = avalanchedir / "Outputs" / module / "peakFiles" / f"res_{simhash}"
    savePath = pathToOutput / "ThalwegPlots"
    fU.makeADir(savePath)
    pathDict = {"avalancheDir": avalanchedir, "pathToOutput": pathToOutput, "savePath": savePath}

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
        plotThalweg2D(pathDict, cfg, dataThalweg)
        plotThalwegAltitude(pathDict, dataThalweg)

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
            plotThalweg2D(pathDict, cfg, dataThalweg)
            plotThalwegAltitude(pathDict, dataThalweg)


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

        for thalwegFile in files:
            data = np.load(thalwegFile, allow_pickle="TRUE")
            newX = np.array(data["x"])
            newY = np.array(data["y"])

            y.append(newY)
            x.append(newX)
    else:
        y = np.array(dataThalweg[f"y"])
        x = np.array(dataThalweg[f"x"])

    # PLOT
    fig, axs = plt.subplots(2, 1)

    fig.set_figheight(10)
    fig.tight_layout(pad=3.0)
    fig.set_figwidth(8)

    fig, axs[0] = tools.makeFieldPlot(axs[0], fig, pathDict, variable, x, y, thalwegPra=thalwegPra)
    axs[1] = tools.makeThalwegPlot(axs[1], dataThalweg, centerOf=centerOf)

    if size != "":
        axs[0].set_title(f"Avalanche size: {size}")

    outFileNamePart = tools.getOutFileNamePartly(pathDict["titleVariables"])
    outFileName = f"Thalweg2D_{outFileNamePart}.png"
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
    pftCrossMax = dataThalweg["flux"] * 100
    # pftCrossMax = np.ones_like(velocityThalweg) * 10

    cfg = cfgUtils.getModuleConfig(ana3AIMEC)
    cfgPlots = cfg["PLOTS"]

    simName = str(pathDict["avalancheDir"]).split("/")[-1]

    outFileNamePart = tools.getOutFileNamePartly(pathDict["titleVariables"])
    pathDict["projectName"] = outFileNamePart
    pathDict["pathResult"] = str(pathDict["savePath"])
    # TODO: we could divide the function outAIMEC.plotVelThAlongThalweg to enable modifications, e.g. the pft representation
    outAIMEC.plotVelThAlongThalweg(pathDict, dataThalweg, pftCrossMax, velocityThalweg, cfgPlots, simName)
