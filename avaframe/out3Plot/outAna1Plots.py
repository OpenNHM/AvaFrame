# imports
import numpy as np
import pandas as pds
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import logging
import copy

# local imports
import avaframe.in2Trans.rasterUtils as IOf
import avaframe.com1DFA.DFAtools as DFAtls
import avaframe.com1DFA.com1DFA as com1DFA
from avaframe.in1Data import getInput as gI
import avaframe.out3Plot.plotUtils as pU
import avaframe.out3Plot.outCom1DFA as outCom1DFA

# create local logger
# change log level in calling module to DEBUG to see log messages
log = logging.getLogger(__name__)


# Simi Sol plots
def saveSimiSolProfile(cfgMain, cfgSimi, fields, limits, simiDict, tSave, header, outDirTest, simHash):
    """Generate plots of the comparison of DFA solution and simiSol

    Parameters
    -----------
    cfgMain: configparser
        main cfg
    cfgSimi: configparser
        simiSol cfg
    fieldsList: list
        list of fields Dictionaries
    solSimi: dict
        dictionary with similiarty solution
    tSave: float
        time coresponding to fields
    header: dict
        field header dictionary
    outDirTest: str or pathlib
        output directory
    simHash: str
        hash of the curent simulation
    """
    # get particle parameters
    xCenter = simiDict["xCenter"]
    # get info on DEM extent
    ncols = header["ncols"]
    nrows = header["nrows"]
    xllc = header["xllcenter"]
    yllc = header["yllcenter"]
    csz = header["cellsize"]

    xArrayFields = np.linspace(xllc, xllc + (ncols - 1) * csz, ncols)
    yArrayFields = np.linspace(yllc, yllc + (nrows - 1) * csz, nrows)

    comSol = {
        "xArrayFields": xArrayFields,
        "yArrayFields": yArrayFields,
        "fields": fields,
    }
    comSol["outDirTest"] = outDirTest
    comSol["tSave"] = tSave

    # profile in flow direction
    fig1 = plt.figure(figsize=(4 * pU.figW, 2 * pU.figH))
    fig1.suptitle(
        "Similarty solution test, t = %.2f s (simulation %s)" % (tSave, simHash),
        fontsize=20,
    )
    ax1 = plt.subplot2grid((1, 2), (0, 0))
    comSol["indFinal"] = int(nrows * 0.5) - 1
    ax1, ax11 = _plotVariable(ax1, cfgSimi, simiDict, comSol, limits, "xaxis")
    ax1.set_title("Profile in flow direction (y = 0 m)")

    # profile across flow direction
    comSol["indFinal"] = int(np.round((xCenter - xllc) / csz) + 1)
    ax2 = plt.subplot2grid((1, 2), (0, 1))
    ax2, ax22 = _plotVariable(ax2, cfgSimi, simiDict, comSol, limits, "yaxis")
    ax2.set_title("Profile across flow direction (x = %.2f m)" % xCenter)

    pU.saveAndOrPlot(
        {"pathResult": outDirTest / "pics"},
        "compareProfileSimiSol%s_%.2f." % (simHash, tSave),
        fig1,
    )


def makeContourSimiPlot(avalancheDir, simHash, fieldFT, limits, simiDict, fieldHeader, tSave, outDirTest):
    """ """
    fig, ax1 = plt.subplots(sharex=True, figsize=(pU.figW * 4, pU.figH * 2))
    # make flow momentum comparison plot
    ax1 = addContour2Plot(ax1, fieldFT, simiDict, fieldHeader, limits, nLevels=16)

    ax1.set_title(pU.cfgPlotUtils["nameFT"] + " contours at t = %.2f s" % tSave)
    pU.putAvaNameOnPlot(ax1, avalancheDir)
    pU.saveAndOrPlot(
        {"pathResult": outDirTest / "pics"},
        "compareContourSimiSol%s_%.2f." % (simHash, tSave),
        fig,
    )

    return fig


def _plotVariable(ax1, cfg, simiDict, comSol, limits, axis, particles=False):
    """Plot flow thickness and velocity for similarity solution and simulation results in axis direction

    Parameters
    -----------
    ax1: matplotlib axis object
    cfg: configParser
        similarity solution cfg
    simiDict: dict
        dictionary with similiarty solution for h, u, and xCenter at required time step
    comSol: dict
        dictionary of simulation results and info (particles, fields, indices, time step)
    axis: str
        xaxis or yaxis

    """
    # com1DFA results
    fields = comSol["fields"]
    xArrayFields = comSol["xArrayFields"]
    yArrayFields = comSol["yArrayFields"]
    # particle properties
    if particles:
        x = comSol["x"]
        y = comSol["y"]
        h = comSol["h"]
        v = comSol["v"]
    indFinal = comSol["indFinal"]
    # similarity solution results
    vSimi = simiDict["vSimi"]
    vxSimi = simiDict["vxSimi"]
    vySimi = simiDict["vySimi"]
    vzSimi = simiDict["vzSimi"]
    hSimi = simiDict["hSimi"]
    xCenter = simiDict["xCenter"]
    ax2 = ax1.twinx()

    # get limits
    widthX = limits["widthX"]
    widthY = limits["widthY"]
    maxFT = limits["maxFT"]
    minMz = limits["minMz"]
    maxFM = limits["maxFM"]

    if axis == "xaxis":
        ax1.axvline(x=xCenter, linestyle=":", color="b")
        FT = fields["FT"][indFinal, :]
        hSimi = hSimi[indFinal, :]
        # DFA simulation
        l1 = ax1.plot(xArrayFields, FT, "k", label="h")
        l2 = ax2.plot(
            xArrayFields,
            FT * fields["FV"][indFinal, :],
            "g",
            label=getLabel("", "", dir="", vert=True),
        )
        l3 = ax2.plot(
            xArrayFields,
            FT * fields["Vx"][indFinal, :],
            "m",
            label=getLabel("", "", dir="x"),
        )
        l4 = ax2.plot(
            xArrayFields,
            FT * fields["Vy"][indFinal, :],
            "b",
            label=getLabel("", "", dir="y"),
        )
        l5 = ax2.plot(
            xArrayFields,
            FT * fields["Vz"][indFinal, :],
            "c",
            label=getLabel("", "", dir="z"),
        )
        if particles:
            ax1.plot(x, h, ".k", linestyle="None", label="part flow thickness")
            ax2.plot(
                x,
                h * v,
                ".g",
                linestyle="None",
                label=getLabel("part", "", dir="", vert=True),
            )
        # similarity solution
        ax1.plot(xArrayFields, hSimi, "--k")
        ax2.plot(xArrayFields, hSimi * vSimi[indFinal, :], "--g")
        ax2.plot(xArrayFields, hSimi * vxSimi[indFinal, :], "--m")
        ax2.plot(xArrayFields, hSimi * vySimi[indFinal, :], "--b")
        ax2.plot(xArrayFields, hSimi * vzSimi[indFinal, :], "--c")
        ax1.set_xlabel("x in (m)")
        ax1.text(xCenter + 5, -0.05, "x = %.2f m" % (xCenter), color="b")
        ax1.set_xlim([xCenter - widthX, xCenter + widthX])
        ax1.set_ylim([-0.05, maxFT])
        ax2.set_ylim([minMz, maxFM])
        # mere legends of both axes
        lns = l1 + l2 + l3 + l4 + l5
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc="upper right")

    elif axis == "yaxis":
        FT = fields["FT"][:, indFinal]
        hSimi = hSimi[:, indFinal]
        # DFA simulation
        l1 = ax1.plot(yArrayFields, FT, "k")
        l2 = ax2.plot(yArrayFields, FT * fields["FV"][:, indFinal], "g")
        l3 = ax2.plot(yArrayFields, FT * fields["Vx"][:, indFinal], "m")
        l4 = ax2.plot(yArrayFields, FT * fields["Vy"][:, indFinal], "b")
        l5 = ax2.plot(yArrayFields, FT * fields["Vz"][:, indFinal], "c")
        if particles:
            ax1.plot(y, h, ".k", linestyle="None")
            ax2.plot(y, h * v, ".g", linestyle="None")
        # similarity solution
        ax1.plot(yArrayFields, hSimi, "--k")
        ax2.plot(yArrayFields, hSimi * vSimi[:, indFinal], "--g")
        ax2.plot(yArrayFields, hSimi * vxSimi[:, indFinal], "--m")
        ax2.plot(yArrayFields, hSimi * vySimi[:, indFinal], "--b")
        ax2.plot(yArrayFields, hSimi * vzSimi[:, indFinal], "--c")
        ax1.set_xlabel("y in (m)")
        ax1.set_xlim([-widthY, widthY])
        ax1.set_ylim([-0.05, maxFT])
        ax2.set_ylim([minMz, maxFM])

    ax1.set_ylabel(pU.cfgPlotUtils["nameFT"] + " (" + pU.cfgPlotUtils["unitFT"] + ")")
    ax1.grid(color="grey", linestyle="-", linewidth=0.25, alpha=0.5)
    color = "tab:green"
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.grid(color="tab:green", linestyle="-", linewidth=0.25, alpha=0.5)
    ax2.set_ylabel(getLabel("", "", dir="") + " (" + pU.cfgPlotUtils["unitFTV"] + ")", color=color)

    props = dict(boxstyle="round", alpha=0)
    text = "analytical solution (dashed line) \n numerical solution (full line)"
    ax1.text(0.05, 0.95, text, transform=ax1.transAxes, verticalalignment="top", bbox=props)

    return ax1, ax2


def plotSimiSolSummary(
    avalancheDir,
    timeList,
    fieldsList,
    fieldHeader,
    simiDict,
    hErrorL2Array,
    hErrorLMaxArray,
    vhErrorL2Array,
    vhErrorLMaxArray,
    outDirTest,
    simDFrow,
    simHash,
    cfgSimi,
):
    """Plot sumary figure of the similarity solution test

    Parameters
    -----------
    avalancheDir: pathlib path
        path to avalanche directory
    timeList: list
        list of time steps
    fieldsList: list
        list of fields dictionaries
    timeList: list
        list of time steps
    fieldHeader: dict
        header dictionary with info about the extend and cell size
    simiDict: dict
        analytic solution dictionary
    fieldHeader: dict
        header dictionary with info about the extend and cell size
    hErrorL2Array: numpy array
        L2 error on flow thickness for saved time steps
    hErrorLMaxArray: numpy array
        LMax error on flow thickness for saved time steps
    vErrorL2Array: numpy array
        L2 error on flow velocity for saved time steps
    vErrorLMaxArray: numpy array
        LMax error on flow velocity for saved time steps
    outDirTest: pathlib path
        path output directory (where to save the figures)
    simDFrow: dataFrame
        data frame row corresponding to simHash
    simHash: str
        com1DFA simulation id
    cfgSimi: configParser object
        SIMISOL configuration

    """
    paramInfo = cfgSimi["paramInfo"].split("|")
    tSave = cfgSimi.getfloat("tSave")
    relativ = cfgSimi.getboolean("relativError")
    indT = min(np.searchsorted(timeList, tSave), min(len(timeList) - 1, len(fieldsList) - 1))
    tSave = timeList[indT]

    xCenter = simiDict["xCenter"]

    # get plots limits
    limits = getPlotLimits(cfgSimi, [fieldsList[indT]], fieldHeader)

    # get info on DEM extent
    ncols = fieldHeader["ncols"]
    nrows = fieldHeader["nrows"]
    xllc = fieldHeader["xllcenter"]
    yllc = fieldHeader["yllcenter"]
    csz = fieldHeader["cellsize"]

    xArrayFields = np.linspace(xllc, xllc + (ncols - 1) * csz, ncols)
    yArrayFields = np.linspace(yllc, yllc + (nrows - 1) * csz, nrows)

    comSol = {
        "xArrayFields": xArrayFields,
        "yArrayFields": yArrayFields,
        "fields": fieldsList[indT],
    }
    comSol["outDirTest"] = outDirTest
    comSol["tSave"] = tSave

    # create figures and plots
    fig = plt.figure(figsize=(pU.figW * 4, pU.figH * 2))
    fig.suptitle(
        "Similarty solution test, t = %.2f s (simulation %s)" % (tSave, simHash),
        fontsize=30,
    )
    # make comparison profile plot in flow direction
    ax1 = plt.subplot2grid((2, 6), (0, 0), colspan=3)
    comSol["indFinal"] = int(nrows * 0.5) - 1
    ax1, ax11 = _plotVariable(ax1, cfgSimi, simiDict, comSol, limits, "xaxis")
    ax1.set_title("Profile in flow direction (y = 0 m)")

    # make flow momentum comparison plot
    ax2 = plt.subplot2grid((2, 6), (0, 3), colspan=3)
    comSol["indFinal"] = int(np.round((xCenter - xllc) / csz) + 1)
    ax2, ax22 = _plotVariable(ax2, cfgSimi, simiDict, comSol, limits, "yaxis")
    ax2.set_title("Profile across flow direction (x = %.2f m)" % xCenter)

    # make bird view plot
    ax3 = plt.subplot2grid((2, 6), (1, 0), colspan=2)
    ax3 = addContour2Plot(ax3, fieldsList[indT]["FT"], simiDict, fieldHeader, limits)
    ax3.set_title(pU.cfgPlotUtils["nameFT"] + " contours")
    pU.putAvaNameOnPlot(ax3, avalancheDir)

    # make error plot
    ax4 = plt.subplot2grid((2, 6), (1, 2), colspan=2)
    title = "\n between analytical solution and com1DFA"
    title = getTitleError(relativ, title)
    ax4.set_title(title)
    ax4, ax5 = addErrorTime(
        ax4,
        timeList,
        hErrorL2Array,
        hErrorLMaxArray,
        vhErrorL2Array,
        vhErrorLMaxArray,
        relativ,
        tSave,
    )

    ax7 = plt.subplot2grid((2, 6), (1, 4), colspan=2)
    ax7.axis("off")
    ax7.invert_yaxis()
    text = ""
    for param in paramInfo:
        text = text + (param + " = %.2f" % simDFrow[param]) + "\n"
    ax7.text(
        0.5,
        0.5,
        text,
        transform=ax7.transAxes,
        ha="center",
        va="center",
        fontsize=pU.fs,
    )

    outFileName = "_".join([simHash, "SimiSolTest"])
    pU.saveAndOrPlot({"pathResult": outDirTest / "pics"}, outFileName, fig)


# Dam Break plots
def _plotVariableDamBreak(var, x, xMiddle, dtInd, dtStep, label):

    fig = plt.figure(figsize=(pU.figW, pU.figH))
    plt.title("Dry-Bed")
    plt.plot(x, var[:, 0], "k--", label="t = 0s")
    plt.plot(x, var[:, dtInd], label="t = %.1fs" % dtStep)
    plt.xlabel("x-coordinate (m)")
    plt.ylabel(label)
    plt.legend()

    return fig


def plotDamAnaResults(t, x, xMiddle, h, u, tSave, cfg, outDirTest):
    """Create plots of the analytical solution for the given settings,
    including an animation

    """

    # index of time steps
    dtInd = np.searchsorted(t, tSave)
    name = pU.cfgPlotUtils["nameFT"] + "(" + pU.cfgPlotUtils["unitFT"] + ")"
    fig = _plotVariableDamBreak(h, x, xMiddle, dtInd, tSave, name)
    outputName = "damBreakFlowThickness"
    pU.saveAndOrPlot({"pathResult": outDirTest / "pics"}, outputName, fig)

    name = pU.cfgPlotUtils["nameFV"] + "(" + pU.cfgPlotUtils["unitFV"] + ")"
    fig = _plotVariableDamBreak(u, x, xMiddle, dtInd, tSave, name)
    outputName = "damBreakFlowVelocity"
    pU.saveAndOrPlot({"pathResult": outDirTest / "pics"}, outputName, fig)


def plotComparisonDam(cfg, simHash, fields0, fieldsT, header, solDam, tSave, limits, outDirTest):
    """Generate plots that compare the simulation results to the analytical solution

    Parameters
    -----------
    cfgC: configParser object
        configuration setting for avalanche simulation including DAMBREAK section
    simHash: str
        simulation hash
    fields0: dict
        initial time step field dictionary
    fieldsT: dict
        tSave field dictionary
    header: dict
        fields header dictionary
    solDam: dict
        analytic solution dictionary:
            tAna: 1D numpy array
                time array
            h0: float
                release thickness
            hAna: 2D numpy array
                Flow thickness (rows for x and columns for time)
            uAna: 2D numpy array
                flow velocity (rows for x and columns for time)
            xAna: 2D numpy array
                extent of domain in the horizontal plane coordinate system (rows for x and columns for time)
            xMidAna: 1D numpy array
                middle of the material in x dir in the horizontal plane coordinate system
                (used to compute the error)
    tSave: float
        time step of analaysis
    limits: dict
        y extend for profile plots
    outDirTest: pathli path
        path to output directory

    """
    phi = cfg.getfloat("phi")
    phiRad = np.radians(phi)
    # Load data
    dataIniFT = fields0["FT"]
    dataAnaFT = fieldsT["FT"]
    dataIniVx = fields0["Vx"]
    dataIniVy = fields0["Vy"]
    dataIniVz = fields0["Vz"]
    dataAnaVx = fieldsT["Vx"]
    dataAnaVy = fieldsT["Vy"]
    dataAnaVz = fieldsT["Vz"]
    if cfg.getboolean("projectVelocity"):
        # project velocity on inclined plane
        dataIniV = DFAtls.scalProd(dataIniVx, dataIniVy, dataIniVz, np.cos(phiRad), 0, -np.sin(phiRad))
        dataAnaV = DFAtls.scalProd(dataAnaVx, dataAnaVy, dataAnaVz, np.cos(phiRad), 0, -np.sin(phiRad))
    else:
        dataIniV = DFAtls.norm(dataIniVx, dataIniVy, dataIniVz)
        dataAnaV = DFAtls.norm(dataAnaVx, dataAnaVy, dataAnaVz)

    # Location of Profiles
    cellSize = header["cellsize"]
    ny = dataAnaFT.shape[0]
    nx = dataAnaFT.shape[1]
    xllc = header["xllcenter"]
    nx_loc = int(ny * 0.5)

    # set x Vector
    x = np.arange(xllc, xllc + nx * cellSize, cellSize)
    y = np.zeros(len(x))
    y[x < 0] = solDam["h0"]
    y[x >= 0] = 0.0
    y[x < -120] = 0.0
    # setup index for time of analyitcal solution
    indTime = np.searchsorted(solDam["tAna"], tSave)

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, sharex=True, figsize=(pU.figW * 4, pU.figH * 2))
    ax1 = _plotDamProfile(
        ax1,
        x,
        y,
        nx_loc,
        cfg,
        dataIniFT,
        dataAnaFT,
        solDam["xAna"],
        solDam["xMidAna"],
        solDam["hAna"],
        indTime,
        limits["maxFT"],
        pU.cfgPlotUtils["nameFT"],
        pU.cfgPlotUtils["unitFT"],
    )
    ax1.set_title(pU.cfgPlotUtils["nameFT"] + " profile")

    y = np.zeros(len(x))
    ax2 = _plotDamProfile(
        ax2,
        x,
        y,
        nx_loc,
        cfg,
        dataIniV,
        dataAnaV,
        solDam["xAna"],
        solDam["xMidAna"],
        solDam["uAna"],
        indTime,
        limits["maxFV"],
        pU.cfgPlotUtils["nameFV"],
        pU.cfgPlotUtils["unitFV"],
    )
    ax2.set_title(pU.cfgPlotUtils["nameFV"] + " profile")
    ax2.legend(loc="upper left")

    y = np.zeros(len(x))
    ax3 = _plotDamProfile(
        ax3,
        x,
        y,
        nx_loc,
        cfg,
        dataIniFT * dataIniV,
        dataAnaFT * dataAnaV,
        solDam["xAna"],
        solDam["xMidAna"],
        solDam["hAna"] * solDam["uAna"],
        indTime,
        limits["maxFM"],
        pU.cfgPlotUtils["nameFTV"],
        pU.cfgPlotUtils["unitFTV"],
    )
    ax3.set_title(getLabel(pU.cfgPlotUtils["nameFTV"] + " profile", "", dir="", vert=True))

    fig.suptitle("Simulation %s, t = %.2f s" % (simHash, tSave), fontsize=30)
    outputName = "compareDamBreak%s_%.02f" % (simHash, tSave)
    pU.saveAndOrPlot({"pathResult": outDirTest / "pics"}, outputName, fig)

    return ax1, ax2, ax3


def _plotDamProfile(ax, x, y, nx_loc, cfg, data1, data2, xAna, xMidAna, yAna, indT, yMax, label, unit):
    """generate plots"""
    scaleCoef = cfg.getfloat("scaleCoef")
    ax.plot(x, y, "grey", linestyle="--")
    ax.plot(x, data1[nx_loc, :], "k--", label="initial")
    ax.plot(x, data2[nx_loc, :], "b", label="numerical")
    ax.plot(xAna, yAna[:, indT], "r-", label="analytical")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("%s (%s)" % (label, unit))
    ax.set_xlim([cfg.getfloat("xStart"), cfg.getfloat("xEnd")])
    ax.set_ylim([-0.05, max(yMax, scaleCoef * max(yAna[:, indT]))])
    # ax.axvline(xMidAna[indT], color='grey', linestyle='--')
    ax.axvspan(
        xMidAna[indT],
        cfg.getfloat("xEnd"),
        color="grey",
        alpha=0.3,
        lw=0,
        label="error computation \n domain",
    )

    return ax


def plotDamBreakSummary(
    avalancheDir,
    timeList,
    fieldsList,
    fieldHeader,
    solDam,
    hErrorL2Array,
    hErrorLMaxArray,
    vhErrorL2Array,
    vhErrorLMaxArray,
    outDirTest,
    simDFrow,
    simHash,
    cfg,
):
    """Plot sumary figure of the damnreak test

    Parameters
    -----------
    avalancheDir: pathlib path
        path to avalanche directory
    timeList: list
        list of time steps
    fieldsList: list
        list of fields dictionaries
    timeList: list
        list of time steps
    fieldHeader: dict
        header dictionary with info about the extend and cell size
    solDam: dict
        analytic solution dictionary
    fieldHeader: dict
        header dictionary with info about the extend and cell size
    hErrorL2Array: numpy array
        L2 error on flow thickness for saved time steps
    hErrorLMaxArray: numpy array
        LMax error on flow thickness for saved time steps
    vErrorL2Array: numpy array
        L2 error on flow velocity for saved time steps
    vErrorLMaxArray: numpy array
        LMax error on flow velocity for saved time steps
    outDirTest: pathlib path
        path output directory (where to save the figures)
    simDFrow: dataFrame
        data frame row corresponding to simHash
    simHash: str
        com1DFA simulation id
    cfg: configParser object
        configuration setting for avalanche simulation including DAMBREAK section

    """
    # Initialise DEM
    demFile = gI.getDEMPath(avalancheDir)
    demOri = IOf.readRaster(demFile, noDataToNan=True)
    dem = com1DFA.initializeMesh(cfg["GENERAL"], demOri, cfg["GENERAL"].getint("methodMeshNormal"))
    dem["header"]["xllcenter"] = dem["originalHeader"]["xllcenter"]
    dem["header"]["yllcenter"] = dem["originalHeader"]["yllcenter"]

    cfgDam = cfg["DAMBREAK"]
    phi = cfgDam.getfloat("phi")
    phiRad = np.radians(phi)
    tSave = cfgDam.getfloat("tSave")
    relativ = cfgDam.getboolean("relativError")
    paramInfo = cfgDam["paramInfo"].split("|")
    indT = min(np.searchsorted(timeList, tSave), min(len(timeList) - 1, len(fieldsList) - 1))
    tSave = timeList[indT]
    # Load data
    fields0 = fieldsList[0]
    fieldsT = fieldsList[indT]
    dataIniFT = fields0["FT"]
    dataFT = fieldsT["FT"]
    dataIniVx = fields0["Vx"]
    dataIniVy = fields0["Vy"]
    dataIniVz = fields0["Vz"]
    dataVx = fieldsT["Vx"]
    dataVy = fieldsT["Vy"]
    dataVz = fieldsT["Vz"]
    if cfgDam.getboolean("projectVelocity"):
        # project velocity on inclined plane
        dataIniV = DFAtls.scalProd(dataIniVx, dataIniVy, dataIniVz, np.cos(phiRad), 0, -np.sin(phiRad))
        dataV = DFAtls.scalProd(dataVx, dataVy, dataVz, np.cos(phiRad), 0, -np.sin(phiRad))
    else:
        dataIniV = DFAtls.norm(dataIniVx, dataIniVy, dataIniVz)
        dataV = DFAtls.norm(dataVx, dataVy, dataVz)

    # Location of Profiles
    cellSize = fieldHeader["cellsize"]
    ny = dataFT.shape[0]
    nx = dataFT.shape[1]
    xllc = fieldHeader["xllcenter"]
    yllc = fieldHeader["yllcenter"]
    nx_loc = int(ny * 0.5)

    xEnd = cfgDam.getfloat("xEnd")
    yStart = cfgDam.getfloat("yStart")
    yEnd = cfgDam.getfloat("yEnd")

    # set x Vector
    x = np.arange(xllc, xllc + nx * cellSize, cellSize)
    y = np.zeros(len(x))
    y[x < 0] = solDam["h0"]
    y[x >= 0] = 0.0
    y[x < -120] = 0.0

    # setup index for time of analyitcal solution
    indTime = np.searchsorted(solDam["tAna"], tSave)
    # get plots limits
    limits = getPlotLimits(cfgDam, fieldsList[:indT], fieldHeader)

    # create figures and plots
    fig = plt.figure(figsize=(pU.figW * 4, pU.figH * 2))
    fig.suptitle("DamBreak test, t = %.2f s (simulation %s)" % (tSave, simHash), fontsize=30)
    # make flow thickness comparison plot
    ax1 = plt.subplot2grid((2, 6), (0, 0), colspan=2)
    ax1 = _plotDamProfile(
        ax1,
        x,
        y,
        nx_loc,
        cfgDam,
        dataIniFT,
        dataFT,
        solDam["xAna"],
        solDam["xMidAna"],
        solDam["hAna"],
        indTime,
        limits["maxFT"],
        pU.cfgPlotUtils["nameFT"],
        pU.cfgPlotUtils["unitFT"],
    )
    ax1.set_title(pU.cfgPlotUtils["nameFT"] + " profile")

    # make flow momentum comparison plot
    ax2 = plt.subplot2grid((2, 6), (0, 4), colspan=2)
    y = y * 0
    ax2 = _plotDamProfile(
        ax2,
        x,
        y,
        nx_loc,
        cfgDam,
        dataIniFT * dataIniV,
        dataFT * dataV,
        solDam["xAna"],
        solDam["xMidAna"],
        solDam["hAna"] * solDam["uAna"],
        indTime,
        limits["maxFM"],
        pU.cfgPlotUtils["nameFTV"],
        pU.cfgPlotUtils["unitFTV"],
    )
    ax2.set_title(getLabel(pU.cfgPlotUtils["nameFTV"] + " profile", "", dir="", vert=True))

    # make flow velocity comparison plot
    ax3 = plt.subplot2grid((2, 6), (0, 2), colspan=2)
    ax3 = _plotDamProfile(
        ax3,
        x,
        y,
        nx_loc,
        cfgDam,
        dataIniV,
        dataV,
        solDam["xAna"],
        solDam["xMidAna"],
        solDam["uAna"],
        indTime,
        limits["maxFV"],
        pU.cfgPlotUtils["nameFV"],
        pU.cfgPlotUtils["unitFV"],
    )
    ax3.set_title(pU.cfgPlotUtils["nameFV"] + " profile")
    plt.legend(loc="upper left")

    # make bird view plot
    ax6 = plt.subplot2grid((2, 6), (1, 0), colspan=2)
    ax6, extent, cbar0, cs1 = outCom1DFA.addResult2Plot(ax6, fieldHeader, dataFT, "FT")
    cbar0.ax.set_ylabel(pU.cfgPlotUtils["nameFT"])
    ax6 = outCom1DFA.addDem2Plot(ax6, dem, what="slope", extent=extent)
    rowsMin, rowsMax, colsMin, colsMax = pU.constrainPlotsToData(
        dataFT,
        fieldHeader["cellsize"],
        extentOption=True,
        constrainedData=False,
        buffer="",
    )
    # draw rectangle corresponding to the error measurement domain
    # Create a Rectangle patch
    rect = patches.Rectangle(
        (solDam["xMidAna"][indTime], yStart),
        xEnd - solDam["xMidAna"][indTime],
        yEnd - yStart,
        linewidth=3,
        linestyle="dashed",
        edgecolor="None",
        facecolor="gray",
        alpha=0.2,
        zorder=200,
        label="error computation domain",
    )
    # Add the patch to the Axes
    ax6.add_patch(rect)
    ax6.set_ylim([rowsMin + yllc, rowsMax + yllc])
    ax6.set_xlim([colsMin + xllc, colsMax + xllc])
    ax6.set_xlabel("x (m)")
    ax6.set_ylabel("y (m)")
    leg = ax6.legend(loc="upper right")
    leg.set(zorder=200)
    # ax3.set_title(pU.cfgPlotUtils['nameFT'])
    pU.putAvaNameOnPlot(ax6, avalancheDir)

    # make error plot
    ax4 = plt.subplot2grid((2, 6), (1, 2), colspan=2)
    title = "\nbetween analytical solution and com1DFA"
    title = getTitleError(relativ, title)
    ax4.set_title(title)
    ax4, ax5 = addErrorTime(
        ax4,
        timeList,
        hErrorL2Array,
        hErrorLMaxArray,
        vhErrorL2Array,
        vhErrorLMaxArray,
        relativ,
        tSave,
    )

    ax7 = plt.subplot2grid((2, 6), (1, 4), colspan=2)
    ax7.axis("off")
    ax7.invert_yaxis()
    text = ""
    for param in paramInfo:
        text = text + (param + " = %.2f" % simDFrow[param]) + "\n"
    ax7.text(
        0.5,
        0.5,
        text,
        transform=ax7.transAxes,
        ha="center",
        va="center",
        fontsize=pU.fs,
    )

    outFileName = "_".join([simHash, "DamBreakTest"])
    pU.saveAndOrPlot({"pathResult": outDirTest / "pics"}, outFileName, fig)


# Genaral plots
def addErrorTime(
    ax1,
    time,
    hErrorL2Array,
    hErrorLMaxArray,
    vhErrorL2Array,
    vhErrorLMaxArray,
    relativ,
    t,
):
    """plot error between a given com1DFA sol and the analytic sol
    function of time on ax1 and ax2

    Parameters
    -----------
    ax1: matplotlib axis
        axis where the erro should be ploted
    time: 1D numpy array
        time array
    hErrorL2Array: 1D numpy array
        flow thickness L2 error array
    hErrorLMaxArray: 1D numpy array
        flow thickness LMax error array
    vhErrorL2Array: 1D numpy array
        flow momentum L2 error array
    vhErrorLMaxArray: 1D numpy array
        flow momentum LMax error array
    relativ: str
    t: float
        time for vertical line
    """
    ax2 = ax1.twinx()
    ax1.plot(time, hErrorL2Array, "k-", label="h L2 error")
    ax1.plot(time, hErrorLMaxArray, "k--", label="h LMax error")
    ax1.set_xlabel("time in (s)")
    ax1.set_ylabel(getTitleError(relativ, " on h"))
    ax1.legend(loc="upper left")
    ax1.grid(color="grey", linestyle="-", linewidth=0.25, alpha=0.5)

    color = "tab:green"
    ax2.plot(time, vhErrorL2Array, "g-", label=getLabel("", " L2 error", dir=""))
    ax2.plot(time, vhErrorLMaxArray, "g--", label=getLabel("", " LMax error", dir=""))
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.set_ylabel(getTitleError(relativ, getLabel(" on", "", dir="")), color=color)
    ax2.legend(loc="lower right")
    ax2.grid(color="tab:green", linestyle="-", linewidth=0.25, alpha=0.5)

    ax1.axvline(t, color="grey", linestyle="--")
    ax1.set_yscale("log")
    ax2.set_yscale("log")
    minY, _ = ax1.get_ylim()
    ax1.text(t, minY, "%.2f s" % (t))

    return ax1, ax2


def plotErrorTime(
    time,
    hErrorL2Array,
    hErrorLMaxArray,
    vhErrorL2Array,
    vhErrorLMaxArray,
    relativ,
    t,
    outputName,
    outDirTest,
):
    """plot and save error between a given com1DFA sol and the analytic sol
    function of time

    Parameters
    -----------
    time: 1D numpy array
        time array
    hErrorL2Array: 1D numpy array
        flow thickness L2 error array
    hErrorLMaxArray: 1D numpy array
        flow thickness LMax error array
    vhErrorL2Array: 1D numpy array
        flow momentum L2 error array
    vhErrorLMaxArray: 1D numpy array
        flow momentum LMax error array
    relativ: str
    t: float
        time for vertical line
    outputName: str
        figure name
    outDirTest: str or pathlib
        output directory
    """
    title = " between similarity solution and com1DFA \n(simulation %s)" % (outputName)
    title = getTitleError(relativ, title)
    fig1, ax1 = plt.subplots(figsize=(2 * pU.figW, 2 * pU.figH))
    ax1.set_title(title)
    ax1, ax2 = addErrorTime(
        ax1,
        time,
        hErrorL2Array,
        hErrorLMaxArray,
        vhErrorL2Array,
        vhErrorLMaxArray,
        relativ,
        t,
    )
    pU.saveAndOrPlot({"pathResult": outDirTest / "pics"}, "Error_time_" + str(outputName), fig1)

    return fig1


def addContour2Plot(ax1, fieldFT, simiDict, fieldHeader, limits, nLevels=9):
    """Make a contour plot of flow thickness for analytical solution and simulation result"""
    # get info on DEM extent
    ncols = fieldHeader["ncols"]
    nrows = fieldHeader["nrows"]
    xllc = fieldHeader["xllcenter"]
    yllc = fieldHeader["yllcenter"]
    csz = fieldHeader["cellsize"]

    xCenter = simiDict["xCenter"]

    xArrayFields = np.linspace(xllc, xllc + (ncols - 1) * csz, ncols)
    yArrayFields = np.linspace(yllc, yllc + (nrows - 1) * csz, nrows)
    X, Y = np.meshgrid(xArrayFields, yArrayFields)
    # get limits
    widthX = limits["widthX"]
    widthY = limits["widthY"]
    contourLevels = np.linspace(0, limits["maxFT"], nLevels)
    contourLevels = contourLevels[:-1]
    # make plot
    cmap, _, ticks, norm = pU.makeColorMap(
        pU.cmapThickness, np.nanmin(fieldFT), np.nanmax(fieldFT), continuous=True
    )
    cmap.set_under(color="w")
    cs1 = ax1.contour(X, Y, fieldFT, levels=contourLevels, origin="lower", cmap=cmap, linewidths=2)
    cs2 = ax1.contour(
        X,
        Y,
        simiDict["hSimi"],
        levels=contourLevels,
        origin="lower",
        cmap=cmap,
        linewidths=2,
        linestyles="dashed",
    )
    CB = pU.addColorBar(
        cs1,
        ax1,
        ticks,
        pU.cfgPlotUtils["unitFT"],
        title=pU.cfgPlotUtils["nameFT"],
        extend="both",
        pad=0.05,
        tickLabelsList="",
    )
    # make colorbar lines thicker
    lines1 = CB.ax.get_children()
    lines1[1].set_linewidths(5)
    # ax3.clabel(CS, inline=1, fontsize=8)
    h1, _ = cs1.legend_elements()
    h2, _ = cs2.legend_elements()
    ax1.legend([h1[-1], h2[-1]], ["simulation", "analytical"])
    ax1.set_ylim([-widthY, widthY])
    ax1.set_xlim([-widthX + xCenter, widthX + xCenter])
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    # add vertical and horizontal line showing the location of the profile plots cuts
    ax1.axvline(xCenter, color="b", linestyle=":")
    ax1.axhline(0, color="r", linestyle=":")
    ax1.text(xCenter + 5, -widthY, "x = %.2f m" % (xCenter), color="b")
    ax1.text(-widthX + xCenter, 5, "y = 0 m", color="r")

    return ax1


def plotErrorConvergence(
    simDF,
    outDirTest,
    cfgSimi,
    xField,
    yField,
    coloredBy,
    sizedBy,
    logScale=False,
    fit=False,
):
    """plot error between all com1DFA sol and analytic sol
    function of whatever you want

    The coloredBy, sizedBy can not be corresponding to non numeric parameters.

    Parameters
    -----------
    simDF: dataFrame
        the simulation data with the postprocessing results
    outDirTest: str or pathlib
        output directory
    cfgSimi: configparser
        the cfg
    xField: str
        column of the simDF to use for the x axis
    yField: str
        column of the simDF to use for the y axis
    coloredBy: str
        column of the simDF to use for the colors
    sizedBy: str
        column of the simDF to use for the marker size
    logScale: boolean
        If you want a loglog scale
    fit: boolean
        if True add power law regression
    """
    tSave = simDF["timeError"].iloc[0]
    relativ = cfgSimi.getboolean("relativError")
    cmap, _, ticks, norm = pU.makeColorMap(
        pU.cmapAvaframeCont,
        min(simDF[coloredBy]),
        max(simDF[coloredBy]),
        continuous=pU.contCmap,
    )
    fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(3 * pU.figW, 2 * pU.figH))
    # get the sizing function
    sizeList = simDF[sizedBy].unique()
    lenSize = len(sizeList)
    minSize = np.nanmin(sizeList)
    maxSize = np.nanmax(sizeList)
    if lenSize > 1:
        sizeList = (simDF[sizedBy].to_numpy() - minSize) / (maxSize - minSize) * 70 + 10
    else:
        sizeList = np.array([100])
    # make the scatter plot
    scatter1 = ax1.scatter(
        simDF[xField],
        simDF[yField],
        c=simDF[coloredBy],
        s=sizeList,
        cmap=cmap,
        norm=norm,
        marker=pU.markers[0],
        alpha=1,
    )  # , edgecolors='k')

    # #########################################
    # If you want to add some regression lines
    colorValueList = simDF[coloredBy].unique()
    lenColor = len(colorValueList)
    if fit:
        for colorValue in colorValueList:
            simDFNew = simDF[simDF[coloredBy] == colorValue]
            color = cmap(norm(simDFNew[coloredBy][0]))
            xArray = simDFNew[xField]
            hErrorL2 = simDFNew[yField]
            p, rSquaredH, _, _, _ = np.polyfit(np.log(xArray), np.log(hErrorL2), deg=1, full=True)
            p1H = p[0]
            p0H = np.exp(p[1])
            ax1.plot(xArray, p0H * xArray**p1H, color=color)
            if np.size(rSquaredH) == 0:
                rSquaredH = np.nan
            log.debug(
                "power law fit sphKernelRadius = %.2f m: hErrorL2 = %.1f * Npart^{%.2f}, r=%.2f"
                % (colorValue, p0H, p1H, rSquaredH)
            )

    if logScale:
        ax1.set_yscale("log")
        ax1.set_xscale("log")

    fig1.suptitle("Convergence of DFA simulation for the similarity solution test at t = %.2fs" % tSave)
    ax1.set_title(getTitleError(relativ, r" L2 on flow thickness"))
    ax1.set_xlabel(xField)
    ax1.set_ylabel(getTitleError(relativ, r" L2 on flow thickness"))
    if lenColor <= 10:
        lenColor = None
    legend1 = ax1.legend(*scatter1.legend_elements(num=lenColor), loc="upper center", title=coloredBy)
    ax1.add_artist(legend1)

    # produce a legend with a cross section of sizes from the scatter
    if lenSize <= 10:
        lenSize = None
    kw = dict(
        prop="sizes",
        color=scatter1.cmap(0.7),
        func=lambda s: (s - 10) * (maxSize - minSize) / 70 + minSize,
    )
    legend3 = ax1.legend(*scatter1.legend_elements(num=lenSize, **kw), loc="upper right", title=sizedBy)
    ax1.grid(color="grey", linestyle="-", linewidth=0.25, alpha=0.5)
    ax1.grid(color="grey", which="minor", linestyle="--", linewidth=0.25, alpha=0.5)
    pU.saveAndOrPlot({"pathResult": outDirTest / "pics"}, "ErrorLog%ds" % int(tSave), fig1)
    return fig1, ax1


def plotPresentation(
    simDF,
    outDirTest,
    cfgSimi,
    xField,
    yField,
    coloredBy,
    sizedBy,
    logScale=False,
    fit=False,
):
    """plot error between all com1DFA sol in simDF and analytic sol
    function of whatever you want

    The coloredBy, sizedBy can not be corresponding to non numeric parameters.

    Parameters
    -----------
    simDF: dataFrame
        the simulation data with the postprocessing results
    outDirTest: str or pathlib
        output directory
    cfgSimi: configparser
        the cfg
    xField: str
        column of the simDF to use for the x axis
    yField: str
        column of the simDF to use for the y axis
    coloredBy: str
        column of the simDF to use for the colors
    sizedBy: str
        column of the simDF to use for the marker size
    logScale: boolean
        If you want a loglog scale
    fit: boolean
        if True add power law regression
    """
    tSave = simDF["timeError"].iloc[0]
    relativ = cfgSimi.getboolean("relativError")
    cmap, _, ticks, norm = pU.makeColorMap(
        pU.cmapAvaframeCont,
        min(simDF[coloredBy]),
        max(simDF[coloredBy]),
        continuous=pU.contCmap,
    )
    fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(3 * pU.figW, 2 * pU.figH))
    if logScale:
        ax1.set_yscale("log")
        ax1.set_xscale("log")
    # get the sizing function
    sizeList = simDF[sizedBy].unique()
    lenSize = len(sizeList)
    minSize = np.nanmin(sizeList)
    maxSize = np.nanmax(sizeList)
    if lenSize > 1:
        sizeList = (simDF[sizedBy].to_numpy() - minSize) / (maxSize - minSize) * 70 + 10
    else:
        sizeList = np.array([100])

    # #########################################
    # If you want to add some regression lines
    colorValueList = -np.sort(-simDF[coloredBy].unique())
    lenColor = len(colorValueList)
    if lenColor <= 10:
        lenColor = None
    if lenSize <= 10:
        lenSize = None
    count = 0
    simDFOld = ""
    for colorValue in colorValueList:
        simDFNew = simDF[simDF[coloredBy] == colorValue]
        xArray = simDFNew[xField]
        hErrorL2 = simDFNew[yField]
        if count >= 1:
            simDFNew = pds.concat([simDFNew, simDFOld])
        colorList = simDFNew[coloredBy].unique()
        lenColor = len(colorList)
        if lenColor <= 10:
            lenColor = None
        if len(sizeList) > 1:
            sizeList = (simDFNew[sizedBy].to_numpy() - minSize) / (maxSize - minSize) * 70 + 10
        else:
            sizeList = np.array([100])
        # make the scatter plot
        scatter1 = ax1.scatter(
            simDFNew[xField],
            simDFNew[yField],
            c=simDFNew[coloredBy],
            s=sizeList,
            cmap=cmap,
            norm=norm,
            marker=pU.markers[0],
            alpha=1,
        )
        fig1.suptitle("Convergence of DFA simulation for the similarity solution test at t = %.2fs" % tSave)
        ax1.set_title(getTitleError(relativ, r" L2 on flow thickness"))
        ax1.set_xlabel(xField)
        ax1.set_ylabel(getTitleError(relativ, r" L2 on flow thickness"))
        legend1 = ax1.legend(*scatter1.legend_elements(num=lenColor), loc="upper center", title=coloredBy)
        ax1.add_artist(legend1)

        # produce a legend with a cross section of sizes from the scatter
        kw = dict(
            prop="sizes",
            color=scatter1.cmap(0.7),
            func=lambda s: (s - 10) * (maxSize - minSize) / 70 + minSize,
        )
        legend3 = ax1.legend(*scatter1.legend_elements(num=lenSize, **kw), loc="upper right", title=sizedBy)
        ax1.grid(color="grey", linestyle="-", linewidth=0.25, alpha=0.5)
        ax1.grid(color="grey", which="minor", linestyle="--", linewidth=0.25, alpha=0.5)
        pU.saveAndOrPlot({"pathResult": outDirTest / "pics"}, "ErrorPresentation%d" % count, fig1)
        if fit:
            p, rSquaredH, _, _, _ = np.polyfit(np.log(xArray), np.log(hErrorL2), deg=1, full=True)
            p1H = p[0]
            p0H = np.exp(p[1])
            color = cmap(norm(simDFNew[coloredBy].iloc[0]))
            ax1.plot(xArray, p0H * xArray**p1H, color=color)
            infoText = "%s = %.2f" % (coloredBy, simDFNew[coloredBy].iloc[0])
            ax1.text(
                max(1.05 * xArray),
                p0H * max(xArray) ** p1H,
                infoText,
                color=color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.5),
            )

            if np.size(rSquaredH) == 0:
                rSquaredH = np.nan
            log.debug(
                "power law fit sphKernelRadius = %.2f m: hErrorL2 = %.1f * Npart^{%.2f}, r=%.2f"
                % (colorValue, p0H, p1H, rSquaredH)
            )

            pU.saveAndOrPlot(
                {"pathResult": outDirTest / "pics"},
                "ErrorPresentation%dFit" % count,
                fig1,
            )

        count = count + 1
        simDFOld = copy.deepcopy(simDFNew)
        legend1.remove()
    return fig1, ax1


def plotTimeCPULog(simDF, outDirTest, cfgSimi, xField, coloredBy, sizedBy, logScale=False):
    """plot computation time function of nParts
    function of whatever (ini parameter given in the simDF) you want

    Parameters
    -----------
    simDF: dataFrame
        the simulation data with the postprocessing results
    outDirTest: str or pathlib
        output directory
    cfgSimi: configparser
        the cfg
    xField: str
        column of the simDF to use for the x axis
    coloredBy: str
        column of the simDF to use for the colors
    sizedBy: str
        column of the simDF to use for the marker size
    logScale: boolean
        If you want a loglog scale
    """
    colorList = simDF[coloredBy].unique()
    tSave = simDF["timeError"].iloc[0]
    cmap, _, ticks, norm = pU.makeColorMap(
        pU.cmapAvaframeCont,
        min(simDF[coloredBy]),
        max(simDF[coloredBy]),
        continuous=pU.contCmap,
    )
    # get the sizing function
    sizeList = simDF[sizedBy].unique()
    minSize = np.nanmin(sizeList)
    maxSize = np.nanmax(sizeList)
    if len(sizeList) > 1:
        sizeList = (simDF[sizedBy].to_numpy() - minSize) / (maxSize - minSize) * 70 + 10
    else:
        sizeList = np.array([100])
    fig1, ax1 = plt.subplots(figsize=(2 * pU.figW, 2 * pU.figH))
    nameList = [
        "timeLoop",
        "timeForce",
        "timeForceSPH",
        "timePos",
        "timeNeigh",
        "timeField",
    ]
    for count, name in enumerate(nameList):
        scatter = ax1.scatter(
            simDF[xField],
            simDF[name],
            c=simDF[coloredBy],
            s=sizeList,
            cmap=cmap,
            marker=pU.markers[count],
            alpha=1,
            edgecolors="k",
        )

    for colorValue in colorList:
        simDFNew = simDF[simDF[coloredBy] == colorValue]
        nPart = simDFNew[xField]
        timeLoop = simDFNew["timeLoop"]
        timeForce = simDFNew["timeForce"]
        timeForceSPH = simDFNew["timeForceSPH"]
        timePos = simDFNew["timePos"]
        timeNeigh = simDFNew["timeNeigh"]
        timeField = simDFNew["timeField"]
        p = np.polyfit(np.log(simDFNew[xField]), np.log(timeLoop), deg=1)
        p11 = p[0]
        p01 = np.exp(p[1])
        p = np.polyfit(np.log(simDFNew[xField]), np.log(timeForce), deg=1)
        p12 = p[0]
        p02 = np.exp(p[1])
        p = np.polyfit(np.log(simDFNew[xField]), np.log(timeForceSPH), deg=1)
        p13 = p[0]
        p03 = np.exp(p[1])
        p = np.polyfit(np.log(simDFNew[xField]), np.log(timePos), deg=1)
        p14 = p[0]
        p04 = np.exp(p[1])
        p = np.polyfit(np.log(simDFNew[xField]), np.log(timeNeigh), deg=1)
        p15 = p[0]
        p05 = np.exp(p[1])
        p = np.polyfit(np.log(simDFNew[xField]), np.log(timeField), deg=1)
        p16 = p[0]
        p06 = np.exp(p[1])
        handles1 = []
        hl = ax1.plot(nPart, p01 * nPart**p11, "k", label="timeLoop")
        handles1.append(hl[0])
        hl = ax1.plot(nPart, p02 * nPart**p12, "g", label="timeForce")
        handles1.append(hl[0])
        hl = ax1.plot(nPart, p03 * nPart**p13, "r", label="timeForceSPH")
        handles1.append(hl[0])
        hl = ax1.plot(nPart, p04 * nPart**p14, "b", label="timePos")
        handles1.append(hl[0])
        hl = ax1.plot(nPart, p05 * nPart**p15, "m", label="timeNeigh")
        handles1.append(hl[0])
        hl = ax1.plot(nPart, p06 * nPart**p16, "c", label="timeField")
        handles1.append(hl[0])
        log.debug(
            "power law fit sphKernelRadius = %.2f m: timeLoop = %.1f * nPart^{%.2f}" % (colorValue, p01, p11)
        )
        log.debug(
            "power law fit sphKernelRadius = %.2f m: timeForce = %.1f * nPart^{%.2f}"
            % (colorValue, p02, p12)
        )
        log.debug(
            "power law fit sphKernelRadius = %.2f m: timeForceSPH = %.1f * nPart^{%.2f}"
            % (colorValue, p03, p13)
        )
        log.debug(
            "power law fit sphKernelRadius = %.2f m: timePos = %.1f * nPart^{%.2f}" % (colorValue, p04, p14)
        )
        log.debug(
            "power law fit sphKernelRadius = %.2f m: timeNeigh = %.1f * nPart^{%.2f}"
            % (colorValue, p05, p15)
        )
        log.debug(
            "power law fit sphKernelRadius = %.2f m: timeField = %.1f * nPart^{%.2f}"
            % (colorValue, p06, p16)
        )

    ax1.set_yscale("log")
    ax1.set_xscale("log")
    ax1.set_title("CPU time")
    ax1.set_xlabel("number of particles")
    ax1.set_ylabel("time (s)")
    # Adding legend and titles
    legend = ax1.legend(handles=handles1, loc="upper left")
    ax1.add_artist(legend)
    legend1 = ax1.legend(*scatter.legend_elements(), loc="lower left", title=coloredBy)
    ax1.add_artist(legend1)

    # produce a legend with a cross section of sizes from the scatter
    kw = dict(
        prop="sizes",
        color=scatter.cmap(0.7),
        func=lambda s: (s - 10) * (maxSize - minSize) / 70 + minSize,
    )
    legend2 = ax1.legend(*scatter.legend_elements(**kw), loc="upper right", title=sizedBy)
    ax1.grid(color="grey", linestyle="-", linewidth=0.25, alpha=0.5)
    ax1.grid(color="grey", which="minor", linestyle="--", linewidth=0.25, alpha=0.5)
    pU.saveAndOrPlot({"pathResult": outDirTest / "pics"}, "timeCPU%ds" % int(tSave), fig1)


def getPlotLimits(cfgSimi, fieldsList, fieldHeader):
    """Get x and y axis limits for the profile and contour plots

    Parameters
    -----------
    cfgSimi: configparser
        simiSol cfg
    fieldsList: list
        list of fields Dictionaries
    fieldHeader: dict
        field header dictionary
    """
    scaleCoef = cfgSimi.getfloat("scaleCoef")
    rowsMin, rowsMax, colsMin, colsMax = pU.constrainPlotsToData(
        fieldsList[-1]["FT"],
        fieldHeader["cellsize"],
        extentOption=True,
        constrainedData=False,
        buffer="",
    )
    widthX = scaleCoef * round(colsMax - colsMin) / 2
    widthY = scaleCoef * round(rowsMax - rowsMin) / 2

    maxFT = 0
    maxFV = 0
    minMz = 0
    maxFM = 0
    for fields in fieldsList:
        maxFT = max(maxFT, np.nanmax(fields["FT"]))
        maxFV = max(maxFT, np.nanmax(fields["FV"]))
        minMz = min(minMz, np.nanmin(fields["Vz"] * fields["FT"]))
        maxFM = max(maxFM, np.nanmax(fields["FV"] * fields["FT"]))

    maxFT = scaleCoef * maxFT
    minMz = scaleCoef * minMz
    maxFM = scaleCoef * maxFM
    limits = {
        "widthX": widthX,
        "widthY": widthY,
        "maxFT": maxFT,
        "maxFV": maxFV,
        "minMz": minMz,
        "maxFM": maxFM,
    }

    return limits


def getTitleError(relativ, ending=""):
    """Get error plot title (relativ error or not?)"""
    if relativ:
        return "Relative error difference" + ending
    else:
        return "Error difference" + ending


def getLabel(start, end, dir="", vert=True):
    """Get error plot title (relativ error or not?)"""
    if dir:
        return start + r" $h \bar{u}_" + dir + r"$ " + end
    else:
        if vert:
            return start + r" $\vert h \mathbf{\bar{u}} \vert $" + end
        else:
            return start + r" $h \mathbf{\bar{u}}$ " + end


def last_nonzero(arr, axis, invalid_val=-1):
    """Get index of last non zero value

    Parameters
    -----------
    arr: numpy array
        data array
    axis: int
        axis along which you want to get the index

    Returns
    --------
    index of last non zero in axis direction

    """
    mask = arr != 0
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)


def first_nonzero(arr, axis, invalid_val=-1):
    """Get index of first non zero value

    Parameters
    -----------
    arr: numpy array
        data array
    axis: int
        axis along which you want to get the index

    Returns
    --------
    index of first non zero in axis direction
    """
    mask = arr != 0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)
