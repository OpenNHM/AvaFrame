# -*- coding: utf-8 -*-
"""

Tools to extract information on the avalanche simulations run in the Output files

"""

# Python imports
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import configparser
from matplotlib.ticker import FormatStrFormatter
import pandas as pd
import math
import seaborn as sns
import logging
from cmcrameri import cm
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches
import pickle

# Local imports
import avaframe.out3Plot.plotUtils as pU
import avaframe.in3Utils.fileHandlerUtils as fU
import avaframe.out3Plot.outDistanceTimeAnalysis as dtAnaPlots
import avaframe.out1Peak.outPlotAllPeak as oP
from avaframe.out3Plot import outAIMEC

log = logging.getLogger(__name__)


def plotParticleThalwegAltitudeVelocity(
    avalancheDir, simIndex, simDF, rasterTransfo, dictVelAltThalweg, resTypePlots, modName, demData
):
    """plot peak flow fields and velocity thalweg envelope

    Parameters
    ------------
    avalancheDir: pathlib path or str
        path to avalanche directory
    simIndex: str
        index of current simulation
    rasterTransfo: dict
        info on domain transformation from xy to sl thalweg
    dictVelAltThalweg: dict
        info on velocity and altitude of particles along thalweg s
    resTypePlots: list
        list of result types that shall be plotted, one plot for each resType
    modName: str
        name of com module used to perform sims
    demData: dict
        dict of dem from sim
    """

    # convert avalancheDir to pathlib path
    avaDir = pathlib.Path(avalancheDir)
    # fetch name of simulation
    simName = simDF["simName"].loc[simIndex]

    # create a plot for result variables in resTypePlots
    for resType in resTypePlots:

        # initialize figure
        fig, ax = plt.subplots(1, 2, figsize=(pU.figW + 10, pU.figH + 3))

        # ax[0]
        # add peak field and thalweg
        ax[0], rowsMinPlot, colsMinPlot = addPeakFieldConstrained(
            avaDir, modName, simName, resType, demData, ax[0], alpha=1.0
        )
        ax[0].plot(
            rasterTransfo["avaPath"]["x"],
            rasterTransfo["avaPath"]["y"],
            "-y",
            zorder=20,
            linewidth=1.0,
            path_effects=[pe.Stroke(linewidth=2, foreground="k"), pe.Normal()],
            label="thalweg",
        )
        ax[0].legend(loc="upper right")
        ax[0].set_xlabel("x [m]")
        ax[0].set_ylabel("y [m]")

        # ax[1]
        # plot the thalweg diagram
        # First Y axis
        cmap = cm.vik  # colormap
        ax[1].plot(
            dictVelAltThalweg["sXYThalweg"],
            dictVelAltThalweg["maxVelocity"],
            color=cmap(0.01),
            label="Max and min velocity",
        )  # max
        ax[1].plot(
            dictVelAltThalweg["sXYThalweg"], dictVelAltThalweg["minVelocity"], color=cmap(0.01)
        )  # min
        ax[1].plot(
            dictVelAltThalweg["sXYThalweg"],
            dictVelAltThalweg["meanVelocity"],
            linestyle="dashed",
            color=cmap(0.7),
            markersize=0.5,
            label="Mean velocity",
            linewidth=0.5,
        )  # mean
        ax[1].plot(
            dictVelAltThalweg["sXYThalweg"],
            dictVelAltThalweg["medianVelocity"],
            linestyle="dotted",
            color=cmap(1.0),
            markersize=0.7,
            label="Median velocity",
            linewidth=0.7,
        )  # median
        # filling the space between the max and min
        ax[1].fill_between(
            dictVelAltThalweg["sXYThalweg"],
            dictVelAltThalweg["minVelocity"],
            dictVelAltThalweg["maxVelocity"],
            color=cmap(0.2),
            alpha=0.2,
            zorder=1,
        )

        # y label
        ax[1].set_ylabel("Velocity [m/s]", fontsize=22)
        ax[1].tick_params(axis="both", labelsize=15)

        # Second Y axis
        ax2 = ax[1].twinx()
        # Plotting the altitude envelope - dashed lines
        ax2.plot(
            dictVelAltThalweg["sXYThalweg"],
            dictVelAltThalweg["maxZ"],
            color="red",
            linestyle="dashed",
            linewidth=0.4,
            label="Max and min altitude",
        )
        ax2.plot(
            dictVelAltThalweg["sXYThalweg"],
            dictVelAltThalweg["minZ"],
            color="red",
            linestyle="dashed",
            linewidth=0.4,
        )
        ax2.fill_between(
            dictVelAltThalweg["sXYThalweg"],
            dictVelAltThalweg["minZ"],
            dictVelAltThalweg["maxZ"],
            hatch="|||||||",
            facecolor="black",
        )
        # X and Y labels
        ax[1].set_xlabel("$s_{xy}$[m]\n\n", fontsize=22)
        ax[1].xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
        ax2.set_ylabel("Altitude [m]", fontsize=22)
        ax2.tick_params(axis="both", labelsize=15)

        # add legend
        fig.legend(loc="lower center", ncol=4, fancybox=True, shadow=True, fontsize=15)

        # save and or plot
        outDir = avaDir / "Outputs" / "out3Plot" / "particleAnalysis"
        fU.makeADir(outDir)
        name = "particleThalwegAltitudeVelocity"
        plotName = "%s_%s_%s" % (name, simName, resType)
        plotPath = pU.saveAndOrPlot({"pathResult": outDir}, plotName, fig)


def plotThalwegTimeAltitudes(
    avalancheDir,
    simIndex,
    simDF,
    rasterTransfo,
    pfvCrossMax,
    modName,
    demSim,
    mtiInfo,
    cfgRangeTime,
    velocityThreshold,
    measuredData="",
):
    """Create plot showing the resType peak field with thalweg,
    thalweg vs altitude with max peak field values along thalweg derived from peak fields
    and the tt-diagram

    Parameters
    -----------
    avalancheDir: str or pathlib path
        path to avalanche directory
    simIndex: str
        index of current sim in simDF
    simDF: dataFrame
        dataframe with one row per simulation, and all model config parameters
    rasterTransfo: dict
        dict with info on transformation from cartesian to thalweg coordinate system
    dictRaster: dict
        dict with info on peak fields
    modName: str
        name of com module used to perform sims
    demSim: dict
        dict with info on dem used for sims
    measuredData: dict
        dict data: with time series of measured data (same format as trackedPartProp)
             labelName: str name of label for plots
             t: 1D array of time steps
    """

    # Load all info from the peak files
    avaDir = pathlib.Path(avalancheDir)
    # fetch name of simulation
    simName = simDF["simName"].loc[simIndex]

    # check if measured data available to be included
    mDataAvailable = False
    if measuredData != "":
        mDataAvailable = True
        nData = measuredData["x"].shape[1]
        cmapData = cm.hawaii
        zMaxM = np.nanmax(measuredData["z"])
        tExt = np.nanmax(measuredData["t"])
    else:
        zMaxM = 0.0
        tExt = 0.0

    # initialize figure
    fig = plt.figure(figsize=(pU.figW + 10, pU.figH + 3))
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[:, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    # add thalweg-altitude plot to axes
    ax1 = outAIMEC.addThalwegAltitude(ax1, rasterTransfo, pfvCrossMax, velocityThreshold, zMaxM=zMaxM)
    # optionally add measurements
    if mDataAvailable:
        g = pU.gravityAcc
        measuredData["velAlt"] = measuredData["z"] + (measuredData["velocityMag"] ** 2.0) / (2.0 * g)
        addTrOrMe(ax1, measuredData, "sAimec", "velAlt", cmapData, label=False, lineStyle="--")
        addTrOrMe(ax1, measuredData, "sAimec", "z", cmapData, label=False)

    # ax2
    # add peak file plot
    ax2, rowsMinPlot, colsMinPlot = addPeakFieldConstrained(
        avaDir, modName, simName, "pfv", demSim, ax2, alpha=1.0
    )
    # add the thalweg
    ax2.plot(
        rasterTransfo["avaPath"]["x"],
        rasterTransfo["avaPath"]["y"],
        "-y",
        zorder=20,
        linewidth=1.0,
        path_effects=[pe.Stroke(linewidth=2, foreground="k"), pe.Normal()],
        label="thalweg",
    )
    # optional add measurements
    if mDataAvailable:
        addTrOrMe(ax2, measuredData, "xOrig", "yOrig", cmapData, label=True, zorder=22)

    # # add labels and legend
    ax2.set_xlabel("x [m] \n\n", fontsize=20)
    ax2.set_ylabel("y [m] \n\n", fontsize=20)
    ax2.tick_params(axis="both", labelsize=15)
    ax2.legend(loc="lower left")

    # ax3 thalweg time diagram
    ax3, rangeTimeResType = dtAnaPlots.addRangeTimePlotToAxes(mtiInfo, cfgRangeTime, ax3)
    maxY = np.nanmax(mtiInfo["rangeList"])
    if mDataAvailable:
        addTrOrMe(ax3, measuredData, "t", "sAimec", cmapData, label=True)
        maxY = max(maxY, np.nanmax(measuredData["sAimec"]))

    # set limits of axes to data extent
    ax3.set_ylim([maxY * 1.1, 0])
    xExt = max(tExt, np.nanmax(mtiInfo["timeList"]))
    ax3.set_xlim([0 - 2, xExt * 1.1])

    # save and or plot
    avaDir = pathlib.Path(avalancheDir)
    outDir = avaDir / "Outputs" / "out3Plot" / "particleAnalysis"
    fU.makeADir(outDir)
    plotName = "thalwegTimeAltitudes_%s" % (simName)
    plotPath = pU.saveAndOrPlot({"pathResult": outDir}, plotName, fig)
    log.info("Plot for %s successfully saved at %s" % (plotName, str(plotPath)))


def plotParticleMotionTracking(
    avalancheDir,
    simName,
    dictVelAltThalweg,
    trackedPartProp,
    dictVelEnvelope,
    demSim,
    modName,
    rasterTransfo,
    measuredData="",
):
    """Create plot showing particle properties over time and along avalanche thalweg
    in light blue envelope for all particles (filled between min and max values)
    in dark blue the values for tracked particles

    panel 1: map view of flow variable peak field
    panel 2: particle trajectoryLengthXYZ vs time
    panel 3: particle velocityMagnitude vs time
    panel 4: particle acceleration vs time
    panel 5: particle trajectoryLengthXYZ vs thalweg Sxy
    panel 6: particle velocity vs thalweg Sxy
    panel 7: particle acceleration vs thalweg Sxy

    Parameters
    -----------
    avalancheDir: pathlib path or str
        path to avalanche directory
    simName: str
        name of simulation
    dictVelAltThalweg: dict
        dict with velocity and altitude envelope info for all particles
    trackedPartProp: dict
        dict with time series of tracked particle properties
    dictVelEnvelope: dict
        dict with velocity envelope info
    demSim: dict
        dict with sim dem info
    modName: str
        name of computational module that has been used to produce the sims
    measuredData: dict
        dict data: with time series of measured data (same format as trackedPartProp)
             labelName: str name of label for plots
             t: 1D array of time steps
    """

    # create pathlib path
    avaDir = pathlib.Path(avalancheDir)

    # check if measured data available to be included
    mDataAvailable = False
    if measuredData != "":
        mDataAvailable = True
        nData = measuredData["x"].shape[1]
        cmapData = cm.hawaii

    # setup figure with subplots 7 panels
    fig = plt.figure(figsize=(pU.figW + 15, pU.figH + 3))
    gs = fig.add_gridspec(3, 3)
    ax1 = fig.add_subplot(gs[0:3, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[2, 1])
    ax5 = fig.add_subplot(gs[0, 2])
    ax6 = fig.add_subplot(gs[1, 2])
    ax7 = fig.add_subplot(gs[2, 2])

    # ax1
    # add peak file plot
    ax1, rowsMinPlot, colsMinPlot = addPeakFieldConstrained(
        avaDir, modName, simName, "pfv", demSim, ax1, alpha=0.2, oneColor=cm.vik
    )

    # add tracked particles locations over time
    xllcenter = demSim["header"]["xllcenter"]
    yllcenter = demSim["header"]["yllcenter"]
    cmap = cm.vik
    ax1.plot(
        trackedPartProp["x"][:, 0] + xllcenter,
        trackedPartProp["y"][:, 0] + yllcenter,
        zorder=1,
        linewidth=1.0,
        color=cmap(0.25),
        label="tracked particles",
    )
    ax1.plot(
        trackedPartProp["x"] + xllcenter,
        trackedPartProp["y"] + yllcenter,
        zorder=2,
        linewidth=1.0,
        color=cmap(0.25),
    )
    ax1.plot(
        rasterTransfo["avaPath"]["x"],
        rasterTransfo["avaPath"]["y"],
        "-y",
        zorder=20,
        linewidth=1.0,
        path_effects=[pe.Stroke(linewidth=2, foreground="k"), pe.Normal()],
        label="thalweg",
    )
    if mDataAvailable:
        addTrOrMe(ax1, measuredData, "xOrig", "yOrig", cmapData, label=True, zorder=4)

    # add blue path for legend entry for addPeakFieldConstrained imshow plot
    bluePatch = mpatches.Patch(color=cm.vik(0.25), label="all particles", alpha=0.2)
    # where some data has already been plotted to ax
    handles, labels = ax1.get_legend_handles_labels()
    handles.append(bluePatch)
    # labels and ticks
    ax1.set_xlabel("x [m] \n\n", fontsize=22)
    ax1.set_ylabel("y [m]", fontsize=22)
    ax1.tick_params(axis="both", labelsize=13)
    ax1.legend(handles=handles, loc="upper left")

    # ax2
    # # plot the trajectoryLengthXYZ vs time
    ax2.plot(
        trackedPartProp["t"],
        trackedPartProp["trajectoryLengthXYZ"],
        zorder=2,
        linewidth=1.0,
        color=cmap(0.25),
    )
    ax2.plot(
        trackedPartProp["t"][:],
        trackedPartProp["trajectoryLengthXYZ"][:, 0],
        zorder=1,
        linewidth=1.0,
        color=cmap(0.25),
        label="tracked particles",
    )
    if mDataAvailable:
        addTrOrMe(ax2, measuredData, "t", "trajectoryLengthXYZ", cmapData, label=False)
    ax2.fill_between(
        dictVelEnvelope["Time"],
        dictVelEnvelope["SxyzMin"],
        dictVelEnvelope["SxyzMax"],
        color=cmap(0.2),
        alpha=0.2,
        zorder=0,
    )
    # labels and ticks
    ax2.set_xlabel("Time [s] \n\n", fontsize=15)
    ax2.set_ylabel("$trajectory_{XYZ}$ [m]", fontsize=15)
    ax2.tick_params(axis="both", labelsize=13)

    # ax3
    # plot velocity Magnitude of particles vs time
    cmap = cm.vik
    ax3.plot(
        trackedPartProp["t"],
        trackedPartProp["velocityMag"],
        zorder=2,
        linewidth=1.0,
        color=cmap(0.25),
        alpha=0.5,
    )
    if mDataAvailable:
        addTrOrMe(ax3, measuredData, "t", "velocityMag", cmapData, label=False)
    ax3.fill_between(
        dictVelEnvelope["Time"],
        dictVelEnvelope["Min"],
        dictVelEnvelope["Max"],
        color=cmap(0.2),
        alpha=0.2,
        zorder=0,
    )
    # labels and ticks
    ax3.set_xlabel("Time [s] \n\n", fontsize=15)
    ax3.set_ylabel("Velocity [m/s]", fontsize=15)
    ax3.tick_params(axis="both", labelsize=13)

    # ax4
    # plot acceleration of particles vs time
    ax4.fill_between(
        dictVelEnvelope["Time"],
        np.nanmin(dictVelEnvelope["Acc"], axis=1),
        np.nanmax(dictVelEnvelope["Acc"], axis=1),
        color=cmap(0.2),
        alpha=0.2,
        zorder=0,
        label="all particles",
    )
    ax4.plot(
        trackedPartProp["t"][:],
        trackedPartProp["uAcc"][:, 0],
        zorder=1,
        linewidth=1.0,
        color=cmap(0.25),
        label="tracked particles",
    )
    ax4.plot(trackedPartProp["t"], trackedPartProp["uAcc"], zorder=1, linewidth=1.0, color=cmap(0.25))
    if mDataAvailable:
        addTrOrMe(ax4, measuredData, "t", "uAcc", cmapData, label=False)
    # labels and ticks
    ax4.set_xlabel("Time [s] \n\n", fontsize=15)
    ax4.set_ylabel("Acceleration [m/s²]", fontsize=15)
    ax4.tick_params(axis="both", labelsize=13)

    # ax5
    # plot travel length along thalweg
    ax5.fill_between(
        dictVelAltThalweg["sXYThalweg"],
        dictVelAltThalweg["minSxyz"],
        dictVelAltThalweg["maxSxyz"],
        color=cmap(0.2),
        alpha=0.2,
        zorder=0,
    )
    ax5.plot(trackedPartProp["sAimec"], trackedPartProp["trajectoryLengthXYZ"], color=cmap(0.25), zorder=1)
    if mDataAvailable:
        addTrOrMe(ax5, measuredData, "sAimec", "trajectoryLengthXYZ", cmapData, label=False)
    # labels and ticks
    ax5.set_xlabel("$S_{xy}$ (thalweg) [m]", fontsize=15)
    ax5.set_ylabel("$trajectory_{XYZ}$ [m]", fontsize=15)
    ax5.tick_params(axis="both", labelsize=13)

    # ax6
    # plot velocity along thalweg
    l2 = ax6.fill_between(
        dictVelAltThalweg["sXYThalweg"],
        dictVelAltThalweg["minVelocity"],
        dictVelAltThalweg["maxVelocity"],
        color=cmap(0.2),
        alpha=0.2,
        zorder=0,
        label="all particles",
    )
    ax6.plot(trackedPartProp["sAimec"], trackedPartProp["velocityMag"], color=cmap(0.25), zorder=1)
    ax6.plot(
        trackedPartProp["sAimec"][:, 0],
        trackedPartProp["velocityMag"][:, 0],
        color=cmap(0.25),
        zorder=1,
        label="tracked particles",
    )
    if mDataAvailable:
        addTrOrMe(ax6, measuredData, "sAimec", "velocityMag", cmapData, label=True)
    # labels and ticks
    ax6.set_xlabel("$S_{xy}$ [m]", fontsize=15)
    ax6.set_ylabel("Velocity [m/s]", fontsize=15)
    ax6.tick_params(axis="both", labelsize=13)

    # ax7
    # plot acceleration along the thalweg
    ax7.fill_between(
        dictVelAltThalweg["sXYThalweg"],
        dictVelAltThalweg["maxAcc"],
        dictVelAltThalweg["minAcc"],
        color=cmap(0.2),
        alpha=0.2,
        zorder=0,
    )
    ax7.plot(trackedPartProp["sAimec"], trackedPartProp["uAcc"], color=cmap(0.25), zorder=1)  # max
    if mDataAvailable:
        addTrOrMe(ax7, measuredData, "sAimec", "uAcc", cmapData, label=False)
    # labels and ticks
    ax7.set_xlabel("$S_{xy}$ [m]", fontsize=15)
    ax7.set_ylabel("Acceleration [m/s²]", fontsize=15)
    ax7.tick_params(axis="both", labelsize=13)

    # save and or plot
    outDir = avaDir / "Outputs" / "out3Plot" / "particleAnalysis"
    fU.makeADir(outDir)
    plotName = "particleMotionTracking_%s" % (simName)
    plotPath = pU.saveAndOrPlot({"pathResult": outDir}, plotName, fig)
    log.info("Plot for %s successfully saved at %s" % (plotName, str(plotPath)))


def addTrOrMe(ax, pDict, prop1, prop2, cmap, label=False, zorder=1, lineStyle="-"):
    """add a line plot of x: prop1, y: prop2 and label if label=True

    Parameters
    -----------
    ax: matplotlib axes
        axes where plot should be added to
    prop1: str
        name of property in pDict used for x axis
    prop2: str
        name of property in pDict used for y axis
    cmap: matplotlib colormap
        cmap to be used for multiple particles in pDict
    label: bool
        if True add label to lines for legend
    zorder: int
        order of the plot object on the axes
    """

    # create an index array for colormap to divide into equal intervals according to number of particles to be plotted
    cmapDiv = np.linspace(0, 1, pDict[prop2].shape[1])
    for ind in range(pDict[prop2].shape[1]):
        if label:
            labelStr = pDict["label"][ind]
        else:
            labelStr = ""
        # if property is t, only 1D vector, for all others 2d array
        if prop1 == "t":
            ax.plot(
                pDict[prop1][:],
                pDict[prop2][:, ind],
                color=cmap(cmapDiv[ind]),
                zorder=zorder,
                linewidth=1.0,
                label=labelStr,
                linestyle=lineStyle,
            )
        else:
            ax.plot(
                pDict[prop1][:, ind],
                pDict[prop2][:, ind],
                color=cmap(cmapDiv[ind]),
                zorder=zorder,
                linewidth=1.0,
                label=labelStr,
                linestyle=lineStyle,
            )


def addPeakFieldConstrained(avaDir, modName, simName, resType, demData, ax, alpha, oneColor=""):
    """use out1Peak functions to add plot of a peak field

    Parameters
    ------------
    avaDir: pathlib path
        path to avalanche dir
    modName: str
        name of computational module used to produce result peak fields
    simName: str
        name of simulation
    resType: str
        result variable name to look for peak field
    demData: dict
        dict with info on dem used to run sim of peak field data
    ax: matplotlib axes object
        axes where plot shall be added to
    alpha: float
        value for transparency from 0-1
    oneColor: empty str
        optional to add a color for a single color for field

    Returns
    --------
    ax: matplotlib axes object
        axes with the peak field plot added
    rowsMinPlot: float
        minimum row extent of the plotted data in plot coordinates
    colsMinPlot: float
        minimum column extent of the plotted data in plot coordinates
    """

    # prepare dem data
    demField = np.where(
        demData["rasterData"] == demData["header"]["nodata_value"], np.nan, demData["rasterData"]
    )

    # set input dir
    inputDir = avaDir / "Outputs" / modName / "peakFiles"

    # fetch all sims in inputDir with info on result files
    simResDF, resTypeList = fU.makeSimFromResDF(avaDir, modName, inputDir=inputDir)

    # find peakFile for resType
    peakFilePath = simResDF[resType].loc[simResDF["simName"] == simName].values[0]

    # fetch cell size
    cellSize = simResDF["cellSize"].loc[simResDF["simName"] == simName].values[0]

    # add constrained peak result field data plot to axes
    ax, rowsMinPlot, colsMinPlot, _ = oP.addConstrainedDataField(
        peakFilePath, resType, demField, ax, cellSize, alpha=alpha, oneColor=oneColor
    )

    return ax, rowsMinPlot, colsMinPlot


def velocityEnvelope(particlesTimeArrays):
    """compute the velocity envelope from particle values

    Parameters
    -----------
    particlesTimeArrays: dict
        dict with time series for particle properties

    Returns
    --------
    dictVelEnvelope: dict
        max, mean, min values of velocity over all particles for each time step
        velocity, acceleration, min and max of trajectoryLengthXYZ of particles for each time step
    """

    # Preparing time, maximum values, minimum values, mean, median
    Max = np.zeros(len(particlesTimeArrays["t"]))
    Min = np.zeros(len(particlesTimeArrays["t"]))
    Mean = np.zeros(len(particlesTimeArrays["t"]))
    Median = np.zeros(len(particlesTimeArrays["t"]))
    SxyzMax = np.zeros(len(particlesTimeArrays["t"]))
    SxyzMin = np.zeros(len(particlesTimeArrays["t"]))

    # loop over time steps and compute max, min, ...
    for j in range(len(particlesTimeArrays["t"])):
        Max[j] = np.amax(particlesTimeArrays["velocityMag"][j, :])
        Min[j] = np.amin(particlesTimeArrays["velocityMag"][j, :])
        Mean[j] = np.mean(particlesTimeArrays["velocityMag"][j, :])
        Median[j] = np.median(particlesTimeArrays["velocityMag"][j, :])
        SxyzMax[j] = np.nanmax(particlesTimeArrays["trajectoryLengthXYZ"][j, :])
        SxyzMin[j] = np.nanmin(particlesTimeArrays["trajectoryLengthXYZ"][j, :])

    dictVelEnvelope = {}
    dictVelEnvelope["Velocity"] = particlesTimeArrays["velocityMag"]
    dictVelEnvelope["Acc"] = particlesTimeArrays["uAcc"]
    dictVelEnvelope["Mean"] = Mean
    dictVelEnvelope["Median"] = Median
    dictVelEnvelope["Min"] = Min
    dictVelEnvelope["Max"] = Max
    dictVelEnvelope["Time"] = particlesTimeArrays["t"]
    dictVelEnvelope["SxyzMax"] = SxyzMax
    dictVelEnvelope["SxyzMin"] = SxyzMin

    return dictVelEnvelope


def velocityEnvelopeThalweg(particlesTimeArrays):
    """function to generate the velocity envelope along the thalweg from simulated flows

    Parameters
    -----------
    particlesTimeArrays: dict
        dict with time series of particle properties

    Returns
    --------
    dictVelAltThalweg: dict
        dict with min, max, mean, ... values of particles elevation, velocity and acceleration
        for each thalweg coordinate

    """

    time = particlesTimeArrays["t"]
    sBetaPoint = particlesTimeArrays["sBetaPoint"]

    elevation = particlesTimeArrays["z"].flatten()
    sXY = particlesTimeArrays["sAimec"].flatten()

    velocityMag = particlesTimeArrays["velocityMag"].flatten()
    uAcc = particlesTimeArrays["uAcc"].flatten()
    trajectoryLengthXYZ = particlesTimeArrays["trajectoryLengthXYZ"].flatten()

    # sort particle coordinates in thalweg system and only keep unique values
    sXYSorted = np.unique(sXY)

    # get max, min, mean, median values of elevation, velocity mag and trajectorylengthXYZ along thalweg
    maxVelocity = np.zeros(len(sXYSorted))
    minVelocity = np.zeros(len(sXYSorted))
    medianVelocity = np.zeros(len(sXYSorted))
    meanVelocity = np.zeros(len(sXYSorted))
    maxAcc = np.zeros(len(sXYSorted))
    minAcc = np.zeros(len(sXYSorted))
    medianAcc = np.zeros(len(sXYSorted))
    meanAcc = np.zeros(len(sXYSorted))
    maxZ = np.zeros(len(sXYSorted))
    minZ = np.zeros(len(sXYSorted))
    maxSxyz = np.zeros(len(sXYSorted))
    minSxyz = np.zeros(len(sXYSorted))

    # create array of properties along S_xy (thalweg/aimec) coordinate
    for indS, sXYCoor in enumerate(sXYSorted):
        sXYInd = np.where(sXY == sXYCoor)
        V = velocityMag[sXYInd[0]]
        A = uAcc[sXYInd[0]]
        S = trajectoryLengthXYZ[sXYInd[0]]
        maxVelocity[indS] = np.nanmax(velocityMag[sXYInd[0]])
        minVelocity[indS] = np.nanmin(velocityMag[sXYInd[0]])
        meanVelocity[indS] = np.nanmean(velocityMag[sXYInd[0]])
        medianVelocity[indS] = np.nanmedian(velocityMag[sXYInd[0]])
        maxZ[indS] = np.nanmax(elevation[sXYInd[0]])
        minZ[indS] = np.nanmin(elevation[sXYInd[0]])
        maxSxyz[indS] = np.nanmax(trajectoryLengthXYZ[sXYInd[0]])
        minSxyz[indS] = np.nanmin(trajectoryLengthXYZ[sXYInd[0]])
        maxAcc[indS] = np.nanmax(uAcc[sXYInd[0]])
        minAcc[indS] = np.nanmin(uAcc[sXYInd[0]])
        meanAcc[indS] = np.nanmean(uAcc[sXYInd[0]])
        medianAcc[indS] = np.nanmedian(uAcc[sXYInd[0]])

    dictVelAltThalweg = {}
    dictVelAltThalweg["sXYPart"] = particlesTimeArrays["sAimec"]
    dictVelAltThalweg["maxSxyz"] = maxSxyz
    dictVelAltThalweg["minSxyz"] = minSxyz
    dictVelAltThalweg["sXYThalweg"] = sXYSorted
    dictVelAltThalweg["sBetaPoint"] = sBetaPoint
    dictVelAltThalweg["meanVelocity"] = meanVelocity
    dictVelAltThalweg["medianVelocity"] = medianVelocity
    dictVelAltThalweg["minVelocity"] = minVelocity
    dictVelAltThalweg["maxVelocity"] = maxVelocity
    dictVelAltThalweg["minZ"] = minZ
    dictVelAltThalweg["maxZ"] = maxZ
    dictVelAltThalweg["meanAcc"] = meanAcc
    dictVelAltThalweg["medianAcc"] = medianAcc
    dictVelAltThalweg["minAcc"] = minAcc
    dictVelAltThalweg["maxAcc"] = maxAcc

    return dictVelAltThalweg


def readMeasuredParticleData(avalancheDir, demHeader, pData=""):
    """fetch data on measured particles from pickle in Inputs/measuredParticles
    only one pickle file allowed if no pData (name of particle file) is provided

    Parameters
    ----------
    avalancheDir: pathlib path
        path to avalanche directory
    demHeader: dict
        currently xllcenter, yllcenter required to set x, y coordinates to origin 0,0
    pData: str
        name of mesured particle file

    Returns
    ---------
    mParticles: dict
        dict with info on measured particles properties (veloctiyMag, x, y, z, uAcc, t)
        all properties except t are of shape: mxn matrix, where m refers to the time steps and n to the
        individual particles t is a vector of the time step values corresponding to m
        label is a list of names of the labels for the measured particles corresponding to n
    """

    inputDir = pathlib.Path(avalancheDir, "Inputs", "measuredParticles")
    if pData != "":
        partFileP = inputDir / pData
        if partFileP.is_file() is False:
            message = "measured particle file: %s does not exist" % (str(partFileP))
            log.error(message)
            raise FileNotFoundError(message)
    else:
        partFiles = list(inputDir.glob("*.pickle"))
        partFileP = partFiles[0]
    if len(partFiles) > 1:
        message = "Multiple measured particle files found in %s, first one found selected: %s" % (
            str(inputDir),
            partFiles[0].name,
        )
        log.warning(message)

    mParticles = pickle.load(open(partFileP, "rb"))

    # TODO: check if we will keep this or change the particles from com1DFA to have xllcenter, yllcenter as origin too
    mParticles["xOrig"] = mParticles["x"]
    mParticles["yOrig"] = mParticles["y"]
    mParticles["x"] = mParticles["x"] - demHeader["xllcenter"]
    mParticles["y"] = mParticles["y"] - demHeader["yllcenter"]

    return mParticles
