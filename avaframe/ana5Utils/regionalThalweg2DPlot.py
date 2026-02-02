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

    #TODO: instead of startRow and startCol take PRA ID

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

    pathToOutput = avalanchedir / "Outputs" / module / "peakFiles" / f"res_{simhash}"
    savePath = pathToOutput / "ThalwegPlots"
    fU.makeADir(savePath)
    pathDict = {"avalancheDir": avalanchedir, "pathToOutput": pathToOutput, "savePath": savePath}

    if startCol == "" or startRow == "":
        plotAllThalwegs = True
    else:
        plotAllThalwegs = False
        startCol = np.int16(startCol)
        startRow = np.int16(startRow)

    centerOf = cfg["GENERAL"].get("centerOfVariable")
    if centerOf == "":
        plotAllCenterOf = True
    else:
        plotAllCenterOf = False

    # TODO: add rel ID
    pathDict["titleVariables"] = {"startRow": startRow, "startCol": startCol, "centerOf": centerOf, "simHash": simhash}

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
        dataThalweg = tools.readThalwegData(
            pathToOutput / "thalwegData", startRow, startCol, centerOf=centerOf
        )
        plotThalweg2D(pathDict, cfg, dataThalweg)
        plotThalwegAltitude(pathDict, dataThalweg)

    if plotAllThalwegs or plotAllCenterOf:
        for thalwegDataFile in files:
            stem = thalwegDataFile.stem
            _, centerOf, startRow, startCol = stem.split("_")
            pathDict["titleVariables"]["startRow"] = startRow
            pathDict["titleVariables"]["startCol"] = startCol
            pathDict["titleVariables"]["centerOf"] = centerOf
            startRow = int(startRow)
            startCol = int(startCol)
            dataThalweg = np.load(thalwegDataFile, allow_pickle="TRUE")
            plotThalweg2D(
                pathDict,
                cfg,
                dataThalweg)
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
    startRow = pathDict["titleVariables"]["startRow"]
    startCol = pathDict["titleVariables"]["startCol"]

    if thalwegPra:
        folder = pathlib.Path(pathDict["pathToOutput"] / "thalwegData")
        files = list(folder.glob(f"thalwegData_{centerOf}*"))
        x = np.empty(0)
        y = np.empty(0)

        for thalwegFile in files:
            data = np.load(thalwegFile, allow_pickle="TRUE")
            newX = np.array(data[f"x"])
            newY = np.array(data[f"y"])

            y = np.concatenate((y, newY))
            x = np.concatenate((x, newX))
    else:
        y = np.array(dataThalweg[f"y"])
        x = np.array(dataThalweg[f"x"])

    # PLOT
    fig, axs = plt.subplots(2, 1)

    fig.set_figheight(10)
    fig.tight_layout(pad=3.0)
    fig.set_figwidth(8)

    fig, axs[0] = tools.makeFieldPlot(
        axs[0], fig, pathDict, variable, x, y, thalwegPra=thalwegPra
    )
    axs[1] = tools.makeThalwegPlot(axs[1], dataThalweg, centerOf=centerOf)

    if size != "":
        axs[0].set_title(f"Avalanche size: {size}")
    fig.savefig(f"{savePath}/Thalweg{centerOf}_{startRow}_{startCol}.png")
    log.info(f"saved plot: {savePath}/Thalweg{centerOf}_{startRow}_{startCol}.png")


def plotThalwegAltitude(pathDict, dataThalweg):
    """
    plot the AIMEC thalweg-altitude plot

    Parameters
    """
    dataThalweg["indStartOfRunout"] = 0
    dataThalweg["startOfRunoutAreaAngle"] = False

    velocityThalweg = tools.zDelta2velocity(dataThalweg["zDelta"])
    pftCrossMax = dataThalweg["flux"] * 100

    cfg = cfgUtils.getModuleConfig(ana3AIMEC)
    cfgPlots = cfg['PLOTS']

    # TODO: also add pra ID to simname
    simName = pathDict["titleVariables"]["centerOf"]

    pathDict["projectName"] = str(pathDict["avalancheDir"]).split("/")[-1] + "_" + pathDict["titleVariables"]["startRow"] + "_" + pathDict["titleVariables"]["startCol"]
    pathDict["pathResult"] = str(pathDict["savePath"])
    # TODO: we could divide the function outAIMEC.plotVelThAlongThalweg to enable modifications, e.g. the pft representation
    outAIMEC.plotVelThAlongThalweg(pathDict, dataThalweg, pftCrossMax, velocityThalweg, cfgPlots, simName)