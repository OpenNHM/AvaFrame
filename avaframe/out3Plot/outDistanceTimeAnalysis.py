""" functions to plot range-time diagrams """

import numpy as np
from matplotlib import pyplot as plt
import logging
import pathlib
from cmcrameri import cm
from matplotlib.colors import LightSource
import pandas as pd

# Local imports
import avaframe.out3Plot.plotUtils as pU
import avaframe.ana5Utils.distanceTimeAnalysis as dtAna
import avaframe.in3Utils.geoTrans as gT
# from avaframe.Tools import NodeTools

log = logging.getLogger(__name__)


def plotRangeTime(mtiInfo, cfgRangeTime):
    """plot range-time diagram with avalanche front and colorcoded average values of result parameter

    Parameters
    -----------
    mtiInfo: dict
        dictionary with average values for range to reference point (mti), timeStep list,
        list with distance to reference point of avalanche front (rangeList)
    cfgRangeTime: configparser object
        configuration settings for range time diagram - here used avalancheDir, rangeTimeResType,
        simHash, and from plots width, height, lw, ..

    """

    # create plot
    fig = plt.figure(figsize=(pU.figW, pU.figH))
    ax = fig.add_subplot(1, 1, 1)
    plt.title(mtiInfo["plotTitle"])

    # add plot to axes
    ax, rangeTimeResType = addRangeTimePlotToAxes(mtiInfo, cfgRangeTime, ax)

    # set path for saving figure
    outDir = pathlib.Path(cfgRangeTime["GENERAL"]["avalancheDir"], "Outputs", "ana5Utils")
    outFileName = mtiInfo["type"] + "_" + rangeTimeResType + "_" + cfgRangeTime["GENERAL"]["simHash"]
    plotPath = pU.saveAndOrPlot({"pathResult": outDir}, outFileName, fig)


def addRangeTimePlotToAxes(mtiInfo, cfgRangeTime, ax):
    """add plot range-time diagram with avalanche front and colorcoded average values of result parameter

    Parameters
    -----------
    mtiInfo: dict
        dictionary with average values for range to reference point (mti), timeStep list,
        list with distance to reference point of avalanche front (rangeList)
    cfgRangeTime: configparser object
        configuration settings for range time diagram - here used avalancheDir, rangeTimeResType,
        simHash, and from plots width, height, lw, ..
    ax: matplotlib axes object
        axes where plot shall be added to

    """
    # fetch required input info
    mti = mtiInfo["mti"]
    rangeGates = mtiInfo["rangeGates"]
    timeList = mtiInfo["timeList"]
    rangeList = mtiInfo["rangeList"]
    rangeTimeResType = cfgRangeTime["GENERAL"]["rangeTimeResType"]
    maxVel, rangeVel, timeVel = dtAna.approachVelocity(mtiInfo)

    # in case time steps are not ordered - the colormesh x and y need to be ordered
    timeIndex = np.argsort(np.array(timeList))
    timeListNew = np.array(timeList)[timeIndex]
    mti = mti[:, timeIndex]

    # fetch velocity legend style info
    width = cfgRangeTime["PLOTS"].getfloat("width")
    height = cfgRangeTime["PLOTS"].getfloat("height")
    lw = cfgRangeTime["PLOTS"].getfloat("lw")
    textsize = cfgRangeTime["PLOTS"].getfloat("textsize")


    pc = ax.pcolormesh(timeListNew, rangeGates, mti, cmap=pU.cmapRangeTime)
    ax.plot(timeList, rangeList, ".", color="black", markersize=4, label="avalanche front")
    ax.set_xlabel("Time (s)")
    # add y label axis
    if mtiInfo["type"] == "thalwegTime":
        sTypeCapital = mtiInfo["sType"][0].upper() + mtiInfo["sType"][1:]
        if cfgRangeTime['GENERAL'].getboolean('originStart'):
            ax.set_ylabel("$S_{XY}$ (m)")
        else:
            labelY = r'$\beta_{%.1f°}$' % mtiInfo['betaPointAngle']
            ax.set_ylabel("%s distance to %s (m)" % (sTypeCapital, labelY))
    else:
        ax.set_ylabel("Distance to %s (m)" % mtiInfo["referencePointName"])

    # add colorbar and infobox
    unit = pU.cfgPlotUtils["unit" + rangeTimeResType]
    if cfgRangeTime["GENERAL"]["maxOrMean"].lower() == "max":
        avgType = "max"
    else:
        avgType = "avg."
    cName = "%s " % avgType + pU.cfgPlotUtils["name" + rangeTimeResType]
    pU.addColorBar(pc, ax, None, unit, title=cName)
    pU.putAvaNameOnPlot(ax, cfgRangeTime["GENERAL"]["avalancheDir"])
    # add range time velocity legend
    rangeTimeVelocityLegend(ax, maxVel, width, height, lw, textsize)

    # add max velocity location
    ax.plot(timeVel, rangeVel, "r*", label="max velocity location")
    # FSO: is this needed?
    # cbar.ax.axhline(y=maxVel, color='r', lw=1, label='max velocity')

    # add info on avalanche front in legend
    ax.legend(facecolor="grey", framealpha=0.2, loc="lower right", fontsize=8)

    # if tt-diagram add beta point info
    if mtiInfo["type"] == "thalwegTime":
        # invert y axis as ava flow starts from minus distance to beta point
        ax.invert_yaxis()
        ax.axhline(
            y=0.0,
            color="gray",
            linestyle="--",
            linewidth=1,
            alpha=0.5,
            label="beta point: %.1f°" % mtiInfo["betaPointAngle"],
        )

    return ax, rangeTimeResType


def radarFieldOfViewPlot(radarFov, radarRange, cfgRangeTime, rangeGates, dem):
    """Create radar field of view plot

    Parameters
    -----------
    radarFov: numpy array
        list with radar location and end point of field of view, x and y coors
    radarRange: masked array
        masked array of DEM with radar field of view - showing distance to radar
    cfgRangeTime: configparser object
        configuration settings section - here used avalancheDir, simHash,
            aperture angle [degree]
    rangeGates: numpy array
        range gates of radar field of view
    dem: dict
        dictionary with dem header and data
    """

    # get input parameters
    aperture = cfgRangeTime["GENERAL"].getfloat("aperture")

    # fetch header info - required for creating coordinate grid
    xllc = dem["header"]["xllcenter"]
    yllc = dem["header"]["yllcenter"]
    ncols = dem["header"]["ncols"]
    nrows = dem["header"]["nrows"]
    cellSize = dem["header"]["cellsize"]

    # Set coordinate grid with given origin
    X, Y = gT.makeCoordinateGrid(xllc, yllc, cellSize, ncols, nrows)

    # load required input parameters for contour plot
    gateContours = cfgRangeTime["PLOTS"].getint("gateContours")

    # get field of view with radar location and aperture angle
    xR = gT.rotate(radarFov, aperture)
    xL = gT.rotate(radarFov, -aperture)
    radarFovArrow = np.diff(radarFov) * 0.2

    # create plot of radar range values and field of view
    fig = plt.figure(figsize=(pU.figW, pU.figH))
    ax1 = plt.subplot(111)
    cmapRadar = pU.cmapRadarFOV
    cmapRadar.set_bad("w", alpha=0.0)
    pc = plt.pcolormesh(X, Y, radarRange, cmap=cmapRadar)
    ax1.plot(radarFov[0], radarFov[1], marker="o", color="k", zorder=1)
    ax1.contour(X, Y, radarRange, rangeGates[::gateContours], colors=["brown"], alpha=0.5, linewidths=0.5)
    ax1.plot(xR[0], xR[1], marker="o", color="k")
    ax1.plot(xL[0], xL[1], marker="o", color="k")
    ax1.arrow(
        x=radarFov[0][0],
        y=radarFov[1][0],
        dx=radarFovArrow[0][0],
        dy=radarFovArrow[1][0],
        fc="r",
        ec="r",
        width=10,
        zorder=2,
    )
    ax1.set_title("Radar range")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    plt.xticks(rotation=45)
    # add colorbar and infoboxes
    unit = "m"
    cName = "range"
    pU.addColorBar(pc, ax1, None, unit, title=cName)
    pU.putAvaNameOnPlot(ax1, cfgRangeTime["GENERAL"]["avalancheDir"])
    pU.putInfoBox(ax1, "- range gates [%d]" % gateContours, location="lowerRight", color="brown")

    # create plot of DEM and radar field of view
    fig2 = plt.figure(figsize=(pU.figW, pU.figH))
    # add second panel with DEM and radar location
    ax2 = plt.subplot(111)
    # pc0 = plt.pcolormesh(X, Y, dem['rasterData'], cmap=cm.grayC)
    ls = LightSource(azdeg=315, altdeg=45)
    ax2.imshow(
        ls.hillshade(
            dem["rasterData"], vert_exag=10, dx=dem["rasterData"].shape[1], dy=dem["rasterData"].shape[0]
        ),
        cmap="gray",
        extent=[np.amin(X), np.amax(X), np.amin(Y), np.amax(Y)],
        origin="lower",
    )
    CS = ax2.contour(X, Y, dem["rasterData"], colors=["brown"], alpha=0.75, linewidths=0.5)
    # add radar field of view
    ax2.plot(radarFov[0], radarFov[1], marker="o", color="k", zorder=1)
    ax2.plot(xR[0], xR[1], marker="o", color="k")
    ax2.plot(xL[0], xL[1], marker="o", color="k")
    ax2.set_title("DEM with radar location")
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("y (m)")
    plt.xticks(rotation=45)
    # add infoboxes
    pU.putAvaNameOnPlot(ax2, cfgRangeTime["GENERAL"]["avalancheDir"])
    pU.putInfoBox(ax2, "- elevation (m)", location="lowerRight", color="brown")
    ax2.clabel(CS, CS.levels, inline=True, fontsize=10)

    # set path for saving figure
    outDir = pathlib.Path(cfgRangeTime["GENERAL"]["avalancheDir"], "Outputs", "ana5Utils")
    outFileName = "radarFieldOfViewPlot_%s" % cfgRangeTime["GENERAL"]["simHash"]
    plotPath = pU.saveAndOrPlot({"pathResult": outDir}, outFileName, fig)
    outFileName = "radarFieldOfViewPlotDEM_%s" % cfgRangeTime["GENERAL"]["simHash"]
    plotPath = pU.saveAndOrPlot({"pathResult": outDir}, outFileName, fig2)


def plotRangeRaster(slRaster, rasterTransfo, cfgRangeTime, rangeRaster, cLower):
    """create plot of distance from start of runout area point extracted avalanche front and result field

    Parameters
    -----------
    slRaster: numpy array
        transformed result field
    rasterTransfo: dict
        info on coordinate transformation
    cfgRangeTime: configparser object
        configuration settings of range time diagram
    rangeRaster: np array
        distance to runout area point in s coordinate
    cLower: int
        index of avalanche front
    """

    # fetch unit for result parameter
    resType = cfgRangeTime["rangeTimeResType"]
    unitRes = pU.cfgPlotUtils["unit" + resType]

    # fetch s. l coordinates
    l = rasterTransfo["l"]
    s = rasterTransfo["s"]
    indStartOfRunout = rasterTransfo["indStartOfRunout"]

    # create figure
    fig = plt.figure(figsize=(pU.figW * 1.5, pU.figH))
    ax1 = plt.subplot(111)
    plt.title("Distance to reference point and %s field" % pU.cfgPlotUtils["name" + resType])
    # create colormap for range field
    cmapRange, _, ticksRange, normRange = pU.makeColorMap(
        cm.grayC.reversed(), np.nanmin(rangeRaster), np.nanmax(rangeRaster), continuous=pU.contCmap
    )
    # add imshow plot of range
    ref0, im1 = pU.NonUnifIm(
        ax1, l, s, rangeRaster, "l (m)", "s (m)", extent=[l.min(), l.max(), s.min(), s.max()], cmap=cmapRange
    )
    # add horizontal line indicating location of start of runout area and avalanche front
    ax1.axhline(
        y=s[indStartOfRunout],
        color="w",
        linestyle="--",
        label="start of run-out area point : %.1f °" % rasterTransfo["startOfRunoutAreaAngle"],
    )
    if np.isnan(cLower):
        log.debug("No avalanche front found for this time step")
    else:
        ax1.axhline(
            y=s[cLower],
            color="y",
            linestyle="--",
            label="avalanche front based on %s limit %s" % (resType, cfgRangeTime["thresholdResult"]),
        )

    # add result field masked with threshold
    bmaskRes = slRaster < cfgRangeTime.getfloat("thresholdResult")
    maskResType = np.ma.array(slRaster, mask=bmaskRes)
    # create colormap
    cmapRes, _, ticksRes, normRes = pU.makeColorMap(
        pU.colorMaps[resType], np.nanmin(maskResType), np.nanmax(maskResType), continuous=True
    )
    cmapRes.set_bad(alpha=0.0)
    ref2, im2 = pU.NonUnifIm(
        ax1, l, s, maskResType, "l (m)", "s (m)", extent=[l.min(), l.max(), s.min(), s.max()], cmap=cmapRes
    )

    # add legend and colorbar
    ax1.legend(loc="center", facecolor="black", framealpha=0.2)
    pU.addColorBar(im1, ax1, ticksRange, "m", pad=0.005)
    pU.addColorBar(im2, ax1, ticksRes, unitRes)

    # create plot name and location for saving and save/plot
    outDir = pathlib.Path(cfgRangeTime["avalancheDir"], "Outputs", "ana5Utils")
    pathDict = {"pathResult": outDir}
    outFileName = "distanceToStartOfRunoutArea_%s_%s" % (resType, cfgRangeTime["simHash"])
    pU.saveAndOrPlot(pathDict, outFileName, fig)


def plotMaskForMTI(cfgRangeTime, bmaskRange, bmaskAvaRadar, bmaskAvaRadarRangeslice, mtiInfo):
    """plot masks used to compute averages for range-time diagram with radar field of view

    Parameters
    -----------
    cfgRangeTime: configparser settings
        configuration settings for range time diagram, avalancheDir
    bmaskRange: numpy mask
        mask for radar range slice for range gate and rgWidth
    bmaskAvaRadar: numpy mask
        mask for result parameter below threshold and out of radar field of view
    bmaskAvaRadarRangeslice: numpy mask
        full mask combination
    """

    # get extent of masks
    headerInfo = mtiInfo["demOriginal"]["header"]
    x0 = headerInfo["xllcenter"]
    y0 = headerInfo["yllcenter"]
    x1 = headerInfo["xllcenter"] + headerInfo["cellsize"] * (bmaskRange.shape[1] - 1)
    y1 = headerInfo["yllcenter"] + headerInfo["cellsize"] * (bmaskRange.shape[0] - 1)
    shiftCorner = 0.5 * headerInfo['cellsize']
    # create extent for plot of raster data using cell corners - lower left (-shiftCorner) to upper right (+shiftCorner) from cell centers
    extentPlot = [x0-shiftCorner, x1+shiftCorner, y0-shiftCorner, y1+shiftCorner]
    # dimensions
    nx = bmaskRange.shape[1]
    ny = bmaskRange.shape[0]

    # create figure with three panels
    fig = plt.figure(figsize=(pU.figW * 3, pU.figH))
    # add radar range gate mask
    ax1 = fig.add_subplot(131)
    ax1.set_title("Radar range gate mask")
    pU.checkExtentFormat(extentPlot, headerInfo['cellsize'], bmaskRange, plotName="")
    ax1.imshow(bmaskRange, extent=extentPlot, origin="lower", aspect=nx / ny)
    # add masked avalanche result field with threshold and radar field of view
    ax2 = fig.add_subplot(132)
    ax2.set_title("RadarFOV and resultThreshold mask")
    ax2.imshow(bmaskAvaRadar, extent=extentPlot, origin="lower", aspect=nx / ny)
    # add combined mask
    ax3 = fig.add_subplot(133)
    ax3.set_title("Combined mask")
    ax3.imshow(bmaskAvaRadarRangeslice, extent=extentPlot, origin="lower", aspect=nx / ny)

    # create plot name and location for saving and save/plot
    outDir = pathlib.Path(cfgRangeTime["avalancheDir"], "Outputs", "ana5Utils")
    pathDict = {"pathResult": outDir}
    outFileName = "rangeTimeMasks_%s" % cfgRangeTime["simHash"]
    pU.saveAndOrPlot(pathDict, outFileName, fig)


def getMaxVelocityPoint(width, height, maxVel, diagVel):
    """get point of max approach velocity in axes coordinates

    Parameters
    -----------
    width: float
        fractional percentage of legend width
    height: float
        fractional percentage of legend height
    maxVel: float
        maximum approach velocity value
    diagVel: float
        x/y ratio of axes

    Returns
    --------
    point: tuple
        x, y axes coordinates of maximum velocity
    """

    # along bottom line
    if maxVel > diagVel:
        point = [(1 - width) + width / maxVel * diagVel, 1 - height]
    else:
        # along re vert line
        point = [1, 1 - height * maxVel / diagVel]

    return point


def rangeTimeVelocityLegend(ax, maxVel, width, height, lw, textsize):
    """set legend in range time diagram for velocity in terms of steepness of front position
        connects to figure callback to react to zoom/pan events

    Parameters
    ----------
    ax: matplotlib axes object
        figure axes object
    maxVel: float
        maximum approach velocity value
    width : float , *0.25
        fractional percentage of legend width
    height: float
        fractional percentage of legend height
    lw: float
        linewidth for legend line
    textsize: float
        textsize for legend entries
    """

    # keep track of which text elements belong to this legend thing...
    inititalNrOfTextElements = len(ax.texts)

    # get legend points in axes coordinates
    point = getVelocityPoints(1, 1, width, height, 1, 1)

    # add title and boundary box and diagonal lines using point
    lVmax = addTitleBox(ax, width, height, lw, point, textsize, maxVel=True)

    # ---- internal functions ----
    def zoomLvlChange(eventAx, first=False):
        """
        Function to annotate the diagonal lines with velocity values.
        Flag first decides if the text elements are created (must be called once)
        or if just the text is changed (figure callback)
        """

        # get current axes limits
        xdMax = max(eventAx.get_xlim())  # timesteps[-1]
        ydMax = max(eventAx.get_ylim())  # max(rangegates)
        xdIntervall = max(eventAx.get_xlim()) - min(eventAx.get_xlim())
        ydIntervall = max(eventAx.get_ylim()) - min(
            eventAx.get_ylim()
        )  # y_max-rangegates[0] # from min to max
        vDiag = ydIntervall / xdIntervall

        # calculate current velocities of diagonal lines
        dataPoint = getVelocityPoints(xdMax, ydMax, width, height, xdIntervall, ydIntervall)
        textElementNr = addVelocityValues(
            ax, dataPoint, point, first=first, inititalNrOfTextElements=inititalNrOfTextElements
        )

        # make or update max velocity line and text
        maxVelPoint = getMaxVelocityPoint(width, height, maxVel, vDiag)
        if maxVel < vDiag:
            xLoc = maxVelPoint[0] - 0.04
            yLoc = maxVelPoint[1] - 0.004
        else:
            xLoc = maxVelPoint[0]
            yLoc = maxVelPoint[1] + 0.004
        if first:
            eventAx.text(
                xLoc,
                yLoc,
                "max %.1f" % maxVel,
                size=5,
                fontweight="bold",
                color="r",
                transform=ax.transAxes,
                gid=len(point) - 2,
                va="center",
                ha="left",
                bbox=dict(facecolor="white", alpha=0.5),
            )
        else:
            ax.texts[textElementNr + 1].set_position((xLoc, yLoc))
            lVmax.set_data([point[0][0], maxVelPoint[0]], [point[0][1], maxVelPoint[1]])

    # ---- internal functions end ----

    # make the text, e.g. the mandatory first=True call
    zoomLvlChange(ax, first=True)

    # connect callback
    ax.callbacks.connect("xlim_changed", zoomLvlChange)
    ax.callbacks.connect("ylim_changed", zoomLvlChange)


def addTitleBox(ax, width, height, lw, point, textsize, maxVel=True):
    """add velocity legend title and boundary box and diagonal lines
    if maxVel=True add max velocity line in red

    Parameters
    -----------
    ax: matplotlib axes object
        axes object for legend
    width: float
        fractional percentage of legend width
    height: float
        fractional percentage of legend height
    lw: float
        linewidth
    point: list
        list of points coordinates for velocity legend
    textsize: float
        size of legend entries
    maxVel: bool
        if True add max velocity line in red
    """

    # add legend title
    ax.text(
        point[1][0],
        point[1][1],
        "Approach \n velocity (m/s)",
        size=textsize,
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontweight="bold",
        bbox=dict(facecolor="white", alpha=0.5),
    )

    # plot boundary of velocity box
    ax.plot(
        [point[0][0], point[1][0]],
        [point[0][1], point[1][1]],
        color="k",
        linewidth=lw,
        transform=ax.transAxes,
    )  # top hor
    ax.plot(
        [point[1][0], point[5][0]],
        [point[1][1], point[5][1]],
        color="k",
        linewidth=lw,
        transform=ax.transAxes,
    )  # re vert
    ax.plot(
        [point[5][0], point[9][0]],
        [point[5][1], point[9][1]],
        color="k",
        linewidth=lw,
        transform=ax.transAxes,
    )  # bottom hor
    ax.plot(
        [point[9][0], point[0][0]],
        [point[9][1], point[0][1]],
        color="k",
        linewidth=lw,
        transform=ax.transAxes,
    )  # le vert

    # these are the diagonal lines referring to velocity
    for j in range(2, len(point) - 1):
        x1 = [point[0][0], point[j][0]]
        y1 = [point[0][1], point[j][1]]
        ax.plot(x1, y1, color="k", linewidth=lw, transform=ax.transAxes)

    if maxVel:
        # draw red line for max velocity
        xdIntervall = max(ax.get_xlim()) - min(ax.get_xlim())
        ydIntervall = max(ax.get_ylim()) - min(ax.get_ylim())
        vDiag = ydIntervall / xdIntervall
        pVmax = getMaxVelocityPoint(width, height, maxVel, vDiag)
        lVmax = ax.plot(
            [point[0][0], pVmax[0]],
            [point[0][1], pVmax[1]],
            color="r",
            linewidth=lw * 1.5,
            transform=ax.transAxes,
        )[
            0
        ]  # return handle not list

        return lVmax


def addVelocityValues(ax, dataPoint, point, first=True, inititalNrOfTextElements=np.nan):
    """add velocity values as labels

    Parameters
    -----------
    ax: matplotlib axis object
    dataPoint: list
        list of points coordinates for velocity label locations
    point: list
        list of points coordinates for velocity lines
    first: bool
        if True text elements are created - initially required, if False then only changed
    inititalNrOfTextElements:

    """

    # compute velocity values
    velocity = []
    for i in range(2, len(dataPoint) - 1):
        deltaS = dataPoint[0][1] - dataPoint[i][1]
        deltaT = dataPoint[i][0] - dataPoint[0][0]
        velocity.append(deltaS / deltaT)

    textElementNr = inititalNrOfTextElements
    # add velocity labels
    for j in range(2, len(point) - 1):
        x1 = [point[0][0], point[j][0]]
        y1 = [point[0][1], point[j][1]]

        textElementNr = textElementNr + 1
        if first:
            if j < 5:  # text along vertical axes
                ax.text(
                    x1[1] + 0.01,
                    y1[1] + 0.002,
                    "%.1f" % velocity[j - 2],
                    size=5,
                    fontweight="bold",
                    color="k",
                    transform=ax.transAxes,
                    gid=j,
                    va="center",
                    ha="left",
                )
            else:  # text along horizontal axes
                ax.text(
                    x1[1],
                    y1[1] - 0.01,
                    "%.1f" % velocity[j - 2],
                    size=5,
                    fontweight="bold",
                    color="k",
                    transform=ax.transAxes,
                    gid=j,
                    va="top",
                    ha="center",
                )
        else:
            ax.texts[textElementNr].set_text("%.1f" % velocity[j - 2])

    return textElementNr


def getVelocityPoints(xMax, yMax, width, height, xinterval, yinterval):
    """get points for legend creation (box and sloping lines)

    if xMax, yMax, xinterval, yinterval are all 1 -> points refer to
    axes coordinates, that range from 0 to 1

    if given true values, e.g. in seconds and meters, points are in
    data coordinates and can be used for velocity calculus etc.

    Note: used to be an internal function to rangeTimeVelocityLegend,
          so use with care.

    Parameters
    -----------
    xMax: float
        max x extent of data (timeSteps)
    yMax: float
        max y extent of data (rangeGates)
    width: float
        fractional percentage of legend width
    height: float
        fractional percentage of legend height
    xinterval: float
        interval on x
    yinterval: float
        interval on y

    Returns
    --------
    points: list
        list of point coordinates
    """

    points = []

    points.append([xMax - width * xinterval, yMax])  # p0 left, top
    points.append([xMax, yMax])  # p1 right, top
    points.append([xMax, yMax - height / 4.0 * yinterval])  # p2 right, 0.75*top
    points.append([xMax, yMax - height / 2.0 * yinterval])  # p3 right, 0.5*top
    points.append([xMax, yMax - height / 4.0 * 3.0 * yinterval])  # p4 right, 0.25*top
    points.append([xMax, yMax - height * yinterval])  # p5 right, bottom
    points.append([xMax - width / 4 * xinterval, yMax - height * yinterval])  # p6 0.75*right, bottom
    points.append([xMax - width / 2 * xinterval, yMax - height * yinterval])  # p7 0.5*right, bottom
    points.append([xMax - width / 4 * 3 * xinterval, yMax - height * yinterval])  # p8 0.25*right, bottom
    points.append([xMax - width * xinterval, yMax - height * yinterval])  # p9 left, bottom

    return points


def animationPlot(demData, data, cellSize, resType, cfgRangeTime, mtiInfo, timeStep):
    """3 panel plot: result in x,y, result in s, l, tt diagram

    Parameters
    -----------
    demData: dict
        info on dem with header and rasterData
    data: numpy array
        result type data
    cellSize: float
        cell size of result type and dem
    resType: str
        name of result type, e.g. FV, FT
    cfgRangeTime: configparser object
        configuration of range time diagram settings
    mtiInfo: dict
        info dictionary for color plot of tt diagram, also info on domain transformation into s,l
    timeStep: float
        actual time step of sim result
    """

    # read required configuration for final plots, e.g. result min and max for colorbar range
    resMin = cfgRangeTime["ANIMATE"].getfloat("resMin")
    resMax = cfgRangeTime["ANIMATE"].getfloat("resMax")

    # fetch unit of result type
    unit = pU.cfgPlotUtils["unit%s" % resType]

    # choose colormap with min max result type range
    cmapRes, col, ticksRes, normRes = pU.makeColorMap(
        pU.colorMaps[resType], resMin, resMax, continuous=pU.contCmap
    )

    # set up 3 panel plot
    fig = plt.figure(figsize=(pU.figW * 3, pU.figH))

    # +++++++++++++++PANEL 1++++++++++++++++++
    # result field in x,y with s,l domain on top
    ax1 = fig.add_subplot(131)
    # check if nodata_value is found and if replace with nans for plotting
    demField = np.where(
        demData["rasterData"] == demData["header"]["nodata_value"], np.nan, demData["rasterData"]
    )
    # initialize x, y vectors of result field domain
    xllc = demData["header"]["xllcenter"]
    yllc = demData["header"]["yllcenter"]
    shiftCorner = demData["header"]["cellsize"] * 0.5
    # get vectors for cell center positions
    x = np.linspace(xllc, xllc + (demData["header"]["ncols"]-1) * cellSize, demData["header"]["ncols"])
    y = np.linspace(yllc, yllc + (demData["header"]["nrows"]-1) * cellSize, demData["header"]["nrows"])

    # mask data for plot where it is zero
    data = np.ma.masked_where(data == 0.0, data)

    # set alpha to zero
    cmapRes.set_bad(alpha=0)
    # uncomment this to set the under value for discrete cmap transparent
    # cmap.set_under(alpha=0)

    # add DEM hillshade with contour lines
    ls, CS = pU.addHillShadeContours(
        ax1, demField, cellSize, [x.min(), x.max(), y.min(), y.max()], colors=["white"]
    )

    # add peak field data
    extentPlot = [x.min()-shiftCorner, x.max()+shiftCorner, y.min()-shiftCorner, y.max()+shiftCorner]
    pU.checkExtentFormat(extentPlot,  demData["header"]['cellsize'], data, plotName="")
    im1 = ax1.imshow(
        data,
        cmap=cmapRes,
        norm=normRes,
        extent=extentPlot,
        origin="lower",
        aspect="equal",
        zorder=4,
    )
    cName1 = "%s (%s)" % (pU.cfgPlotUtils["name" + resType], pU.cfgPlotUtils["unit" + resType])
    pU.addColorBar(im1, ax1, ticksRes, None, title=cName1)

    # add domain transformation info
    # read avaPath with scale
    rasterTransfo = mtiInfo["rasterTransfo"]
    xPath = rasterTransfo["x"]
    yPath = rasterTransfo["y"]
    # read domain boundaries with scale
    cellSizeSL = rasterTransfo["cellSizeSL"]
    DBXl = rasterTransfo["DBXl"] * cellSizeSL
    DBXr = rasterTransfo["DBXr"] * cellSizeSL
    DBYl = rasterTransfo["DBYl"] * cellSizeSL
    DBYr = rasterTransfo["DBYr"] * cellSizeSL
    # add indication for run out area based on beta point but fetch for x, y coordinates
    indStartOfRunout = rasterTransfo["indStartOfRunout"]
    betaPointX = rasterTransfo["gridx"][indStartOfRunout]
    betaPointY = rasterTransfo["gridy"][indStartOfRunout]
    # add indication for avalanche front based on s,l but fetch for x, y coordinates
    cLower = mtiInfo["cLower"]
    if np.isnan(cLower):
        log.debug("No avalanche front found for this time step")
    else:

        frontLineX = rasterTransfo["gridx"][cLower]
        frontLineY = rasterTransfo["gridy"][cLower]
        ax1.plot(frontLineX, frontLineY, "k:", linewidth=2, zorder=5, label="avalanche front")

    # add lines to plot about path following domain
    ax1.plot(xPath, yPath, "k--", zorder=5, label="thalweg")
    ax1.plot(DBXl, DBYl, "k-", label="s,l domain")
    ax1.plot(DBXr, DBYr, "k-")
    ax1.plot([DBXl, DBXr], [DBYl, DBYr], "k-")
    ax1.plot(betaPointX, betaPointY, "b--", zorder=5, label="beta Point")
    ax1.legend(loc="upper right")
    # label x, y axes
    if cfgRangeTime["ANIMATE"].getboolean("xyEastNorth"):
        ax1.set_xlabel("East x (m)")
        ax1.set_ylabel("North y (m)")
    else:
        ax1.set_xlabel("x (m)")
        ax1.set_ylabel("y (m)")
    # optional add title to panel
    if cfgRangeTime["ANIMATE"].getboolean("panelTitles"):
        ax1.set_title("simulation x,y extent at t=%.2fs" % timeStep)

    # add avaName
    pU.putAvaNameOnPlot(ax1, cfgRangeTime["GENERAL"]["avalancheDir"], date=False)

    # +++++++++++++++PANEL 2#############
    # result field bigger than threshold  in s,l with ava front and runout area start
    ax2 = fig.add_subplot(132)

    # fetch avalanche front info
    cLower = mtiInfo["cLower"]
    slRaster = mtiInfo["slRaster"]
    rasterTransfo = mtiInfo["rasterTransfo"]
    rangeRaster = mtiInfo["rangeRaster"]
    sType = mtiInfo["sType"]

    # fetch s. l coordinates
    l = rasterTransfo["l"]
    indStartOfRunout = rasterTransfo["indStartOfRunout"]
    # determine if sParallel or sProjected has been used
    if sType.lower() == "parallel":
        s = rasterTransfo["sParallel"] - rasterTransfo["sParallel"][indStartOfRunout]
        sLabel = "Parallel distance to beta point (m)"
    elif sType.lower() == "projected":
        s = rasterTransfo["s"] - rasterTransfo["s"][indStartOfRunout]
        sLabel = "Projected distance to beta point (m)"

    # create figure
    # add line indicating location of start of runout area and avalanche front
    ax2.axhline(
        y=s[indStartOfRunout],
        color="b",
        linestyle="--",
        label="beta point (%.1f°)" % mtiInfo["betaPointAngle"],
    )
    # only plot front if a front found - e.g. if FV is zero in first time step no front found
    if np.isnan(cLower):
        log.debug("No avalanche front found for this time step")
    else:
        ax2.axhline(
            y=s[cLower],
            color="k",
            linewidth=2.5,
            linestyle=(0, (0.1, 2)),
            dash_capstyle="round",
            label="avalanche front",
        )

    # add result field masked with threshold
    bmaskRes = slRaster < cfgRangeTime["GENERAL"].getfloat("thresholdResult")
    maskResType = np.ma.array(slRaster, mask=bmaskRes)

    # plot masked result type, add norm res to scale to colorbar
    cmapRes.set_bad(alpha=0.0)
    ref2, im3 = pU.NonUnifIm(
        ax2,
        l,
        s,
        maskResType,
        "l (m)",
        sLabel,
        extent=[l.min(), l.max(), s.min(), s.max()],
        cmap=cmapRes,
        norm=normRes,
        origin="lower",
    )

    # add elevation contours, first get z coordinates of s,l points
    zPoints = {"x": rasterTransfo["gridx"], "y": rasterTransfo["gridy"]}
    zPoints, _ = gT.projectOnRaster(demData, zPoints)
    ls, CS = pU.addHillShadeContours(
        ax2,
        zPoints["z"],
        rasterTransfo["cellSizeSL"],
        [l.min(), l.max(), s.min(), s.max()],
        colors=["gray"],
        onlyContours=True,
        extentCenters=False,
    )

    # invert y axis
    ax2.invert_yaxis()

    # add legend and colorbar
    ax2.legend(loc="lower right", facecolor="white", framealpha=0.2)

    # optionally add title
    if cfgRangeTime["ANIMATE"].getboolean("panelTitles"):
        ax2.set_title("s,l domain extent")

    # +++++++++++++++PANEL 3++++++++++++++++++
    # tt-diagram of result field bigger than threshold
    ax3 = fig.add_subplot(133)

    # fetch required input info
    mti = mtiInfo["mti"]
    rangeGates = mtiInfo["rangeGates"]
    timeList = mtiInfo["timeList"]
    rangeList = mtiInfo["rangeList"]

    # in case time steps are not ordered - the colormesh x and y need to be ordered
    timeIndex = np.argsort(np.array(timeList))
    timeListNew = np.array(timeList)[timeIndex]
    mti = mti[:, timeIndex]

    # create fig
    pc = ax3.pcolormesh(timeListNew, rangeGates, mti, vmin=resMin, vmax=resMax, cmap=pU.cmapRangeTime)
    ax3.plot(timeList, rangeList, ".", color="black", markersize=4, label="avalanche front")
    ax3.set_xlabel("Time (s)")
    sTypeCapital = sType[0].upper() + sType[1:]
    ax3.set_ylabel("%s distance to %s (m)" % (sTypeCapital, mtiInfo["referencePointName"]))

    # add colorbar and infobox
    cName = "%s (%s)" % (
        cfgRangeTime["GENERAL"]["maxOrMean"] + " " + pU.cfgPlotUtils["name" + resType],
        pU.cfgPlotUtils["unit" + resType],
    )
    pU.addColorBar(pc, ax3, None, None, title=cName)
    width = cfgRangeTime["PLOTS"].getfloat("width")
    height = cfgRangeTime["PLOTS"].getfloat("height")
    lw = cfgRangeTime["PLOTS"].getfloat("lw")
    textsize = cfgRangeTime["PLOTS"].getfloat("textsize")
    xMin = cfgRangeTime["ANIMATE"].getfloat("xMin")
    xMax = cfgRangeTime["ANIMATE"].getfloat("xMax")
    yMax = cfgRangeTime["ANIMATE"].getfloat("yMax")
    yMin = cfgRangeTime["ANIMATE"].getfloat("yMin")
    xinterval = xMax - xMin
    yinterval = yMax - yMin
    kPlot = yinterval / xinterval

    # fetch points for drawing velocity legend
    point = getVelocityPoints(1, 1, width, height, 1, 1)

    # add title and boundary box and diagonal lines using point
    lVmax = addTitleBox(ax3, width, height, lw, point, textsize, maxVel=False)

    # compute velocity values
    dataPoint = getVelocityPoints(xMax, yMax, width, height, xinterval, yinterval)

    # add velocity legend labels
    _ = addVelocityValues(ax3, dataPoint, point)

    # set limits for plot (depends on final time step)
    ax3.set_ylim([yMin, yMax])
    ax3.set_xlim([xMin, xMax])

    # add info on avalanche front in legend
    plt.legend(facecolor="grey", framealpha=0.2, loc="lower right", fontsize=8)

    # if tt-diagram add beta point info
    # invert y axis as ava flow starts from minus distance to beta point
    ax3.invert_yaxis()
    ax3.axhline(y=0.0, color="b", linestyle="--", label="beta point: %.1f°" % mtiInfo["betaPointAngle"])

    # optional - add title to panel
    if cfgRangeTime["ANIMATE"].getboolean("panelTitles"):
        ax3.set_title("tt-diagram")

    # # set path for saving figure
    outDir = pathlib.Path(cfgRangeTime["GENERAL"]["avalancheDir"], "Outputs", "ana5Utils")
    outFileName = (
        mtiInfo["type"] + "_" + resType + "_" + cfgRangeTime["GENERAL"]["simHash"] + "_%08.3f" % timeStep
    )
    plotPath = pU.saveAndOrPlot({"pathResult": outDir}, outFileName, fig)
