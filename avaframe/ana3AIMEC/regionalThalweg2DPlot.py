import numpy as np
import pathlib
import matplotlib.pyplot as plt

import avaframe.ana3AIMEC.regionalThalwegTools as tools
from avaframe.in3Utils import fileHandlerUtils as fU


def regionalThalweg2DPlotMain(avalanchedir, cfg, size=None):
    """
    saves Plot of thalweg:
    top pannel of the position in the raster
    bottom panel of the 2 dimensional representation
    #TODO: instead of startRow and startCol take PRA ID

    Parameters
    -----------
    avalanchedir: str
        Path to the th avalanche directory
    cfg: configparser Object
        contains configuration settings
    size: float
        avalanche size of the pathaxs[1]
    """
    simhash = cfg["GENERAL"].get("simHash")
    module = cfg["GENERAL"].get("modName")
    avalanchedir = pathlib.Path(avalanchedir)

    startRow = cfg["GENERAL"].get("startRow")
    startCol = cfg["GENERAL"].get("startCol")
    if startCol == "" or startRow == "":
        plotAllPras = True
    else:
        plotAllPras = False
        startCol = np.int16(startCol)
        startRow = np.int16(startRow)

    centerOf = cfg["GENERAL"].get("centerOfVariable")
    if centerOf == "":
        plotAllThalwegs = True
    else:
        plotAllThalwegs = False

    pathToOutput = avalanchedir / "Outputs" / module / "peakFiles" / f"res_{simhash}"
    savePath = pathToOutput / "ThalwegPlots"
    fU.makeADir(savePath)

    # FlowPy output: thalweg data
    if plotAllThalwegs:
        files = sorted(list((pathToOutput / "thalwegData").glob(f"thalwegData_*.pickle")))
    elif plotAllPras:
        files = sorted(list((pathToOutput / "thalwegData").glob(f"thalwegData_{centerOf}_*.pickle")))
    else:
        dataThalweg = tools.readThalwegData(
            pathToOutput / "thalwegData", startRow, startCol, centerOf=centerOf
        )
        plotThalweg2D(
            avalanchedir, cfg, pathToOutput, savePath, dataThalweg, startRow, startCol, centerOf, size=size
        )

    if plotAllPras or plotAllThalwegs:
        for thalwegDataFile in files:
            stem = thalwegDataFile.stem
            _, centerOf, startRow, startCol = stem.split("_")
            startRow = int(startRow)
            startCol = int(startCol)
            dataThalweg = np.load(thalwegDataFile, allow_pickle="TRUE")
            plotThalweg2D(
                avalanchedir,
                cfg,
                pathToOutput,
                savePath,
                dataThalweg,
                startRow,
                startCol,
                centerOf,
                size=size,
            )


def plotThalweg2D(
        avalanchedir, cfg, pathToOutput, savePath, dataThalweg, startRow, startCol, centerOf, size=None
):
    """ """
    variable = cfg["GENERAL"].get("plotVariable")
    thalwegPra = cfg["GENERAL"].getboolean("thalwegPra")

    if thalwegPra:
        folder = pathlib.Path(pathToOutput / "thalwegData")
        files = list(folder.glob(f"thalwegData_{centerOf}*"))
        x = np.empty(0)
        y = np.empty(0)

        for thalwegFile in files:
            data = np.load(thalwegFile, allow_pickle="TRUE")
            newX = np.array(data[f"x"])
            newY = np.array(data[f"y"])
            # if centerOf in ["CoE", "CoZd"]:
            #    newX = newX[1:]
            #    newY = newY[1:]

            y = np.concatenate((y, newY))
            x = np.concatenate((x, newX))
    else:
        y = np.array(dataThalweg[f"y"])
        x = np.array(dataThalweg[f"x"])

    # PLOT
    fig, axs = plt.subplots(2, 1)  # (3,1)

    fig.set_figheight(10)
    fig.tight_layout(pad=3.0)
    fig.set_figwidth(8)

    fig, axs[0] = tools.makeFieldPlot(
        axs[0], fig, avalanchedir, pathToOutput, variable, x, y, centerOf=centerOf, thalwegPra=thalwegPra
    )
    axs[1] = tools.makeThalwegPlot(axs[1], dataThalweg, centerOf=centerOf)

    if size is not None:
        axs[0].set_title(f"size: {size}")
    if savePath is not None:
        fig.savefig(f"{savePath}/Thalweg{centerOf}_{startRow}_{startCol}.png")
