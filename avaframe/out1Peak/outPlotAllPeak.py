"""
Functions to plot 2D simulation results: plot of all peak files at once

"""

import logging
import numpy as np
import matplotlib

matplotlib.use("agg")
from matplotlib import pyplot as plt
import pathlib
from mpl_toolkits.axes_grid1 import make_axes_locatable

import avaframe.out3Plot.plotUtils as pU
import avaframe.in1Data.getInput as gI
from avaframe.in3Utils import fileHandlerUtils as fU
import avaframe.in2Trans.rasterUtils as IOf
import avaframe.in3Utils.geoTrans as gT
import avaframe.in3Utils.cfgUtils as cfgUtils

import rasterio
import rasterio.plot
import contextily as ctx
import geopandas as gpd

# create local logger
log = logging.getLogger(__name__)


def plotAllPeakFields(avaDir, cfgFLAGS, modName, demData=""):
    """Plot all peak fields and return dictionary with paths to plots
    with DEM in background

    Parameters
    ----------
    avaDir : str
        path to avalanche directoy
    cfgFLAGS : str
        general configuration, required to define if plots saved to reports directoy
    modName : str
        name of module that has been used to produce data to be plotted
    demData: dictionary
        optional - if not the dem in the avaDir/Inputs folder has been used but a different one

    Returns
    -------
    plotDict : dict
        dictionary with info on plots, like path to plot
    """

    # Load all infos on simulations
    avaDir = pathlib.Path(avaDir)
    inputDir = avaDir / "Outputs" / modName / "peakFiles"
    inDir = avaDir / "Inputs"
    peakFilesDF = fU.makeSimDF(inputDir, avaDir=avaDir)

    if modName in ["com1DFA", "com9MoTVoellmy"] and demData == "":
        configurationDF = cfgUtils.createConfigurationInfo(avaDir, comModule=modName)
        configurationDF = configurationDF.rename(columns={"resType": "resTypeList"})
        peakFilesDF = (
            peakFilesDF.reset_index().merge(configurationDF, on=["simName"]).set_index("index")
        )

    if demData == "":
        demFile = gI.getDEMPath(avaDir)
        demDataRaster = IOf.readRaster(demFile, noDataToNan=True)
        demDataField = demDataRaster["rasterData"]
    else:
        # check if nodata_value is found and if replace with nans for plotting
        demDataField = np.where(
            demData["rasterData"] == demData["header"]["nodata_value"],
            np.nan,
            demData["rasterData"],
        )
    demField = demDataField

    # Output directory
    if cfgFLAGS.getboolean("ReportDir"):
        outDir = avaDir / "Outputs" / modName / "reports"
        fU.makeADir(outDir)
    else:
        outDir = avaDir / "Outputs" / "out1Peak"
        fU.makeADir(outDir)

    # Initialise plot dictionary with simulation names
    plotDict = {}
    for sName in peakFilesDF["simName"]:
        plotDict[sName] = {}

    # Loop through peakFiles and generate plot
    # for m in range(len(peakFilesDF["names"])):
    for ind1, row in peakFilesDF.iterrows():
        # Load names and paths of peakFiles
        name = row["names"]
        fileName = row["files"]
        resType = row["resType"]
        simType = row["simType"]

        log.debug("now plot %s:" % (fileName))

        plotName = outDir / ("%s.%s" % (name, pU.outputFormat))

        # only produce a plot if it does not already exists
        # make sure to remove the Outputs folder if you want to regenerate the plot
        # this enables to append simulations to an already existing output without regenerating all plots
        if not plotName.is_file():
            # for comModules load DEM used for computation
            if demData == "" and modName in ["com1DFA", "com9MoTVoellmy"]:
                demFile = inDir / row["DEM"]
                demDataRaster = IOf.readRaster(demFile, noDataToNan=True)
                demDataField = demDataRaster["rasterData"]
                demField = demDataField

            # Figure  shows the result parameter data
            fig, ax = plt.subplots(figsize=(pU.figW, pU.figH))

            # fetch cellSize of peak field
            cellSize = row["cellSize"]

            # add peak field data now
            ax, rowsMinPlot, colsMinPlot, extentCellCorners = addConstrainedDataField(
                fileName, resType, demField, ax, cellSize
            )

            # Add background map
            if cfgFLAGS.getboolean("showOnlineBackground"):
                rasterInfo = IOf.readRaster(fileName, noDataToNan=True)
                providers = ctx.providers.flatten()
                srcCrs = rasterInfo["header"]["crs"]
                if srcCrs is None:
                    message = "chosen basemap: %s not applicable for CRS: %s" % (
                        str(cfgFLAGS["mapProvider"]),
                        srcCrs,
                    )
                    log.error(message)
                    raise AssertionError(message)
                ctx.add_basemap(
                    ax,
                    crs=srcCrs,
                    source=providers[str(cfgFLAGS["mapProvider"])],
                    zorder=2,
                )

            # if entrainment or resistance area is considered in simulation, show extent of entrainment or resistance area
            colorOutline = {"ent": "white", "res": "green"}
            for sType in ["ent", "res"]:
                if sType in simType:
                    sFile, sInfo, fileSType = gI.getAndCheckInputFiles(
                        inDir, sType.upper(), sType, fileExt=["shp", "asc", "tif"]
                    )
                    if fileSType == ".shp":
                        sarea = gpd.read_file(sFile)
                        sarea.plot(
                            ax=ax,
                            zorder=12,
                            edgecolor=colorOutline[sType],
                            linewidth=2,
                            facecolor="none",
                            label=("%s area" % sType),
                            alpha=0.8,
                        )
                    elif fileSType in [".asc", ".tif"]:
                        sarea = IOf.readRaster(sFile, noDataToNan=True)
                        xGrid, yGrid, _, _ = gT.makeCoordGridFromHeader(sarea["header"])
                        contourDictXY = pU.fetchContourCoords(xGrid, yGrid, sarea["rasterData"], 0.001)
                        for key in contourDictXY:
                            ax.plot(
                                contourDictXY[key]["x"],
                                contourDictXY[key]["y"],
                                color=colorOutline[sType],
                                zorder=12,
                            )

            # set limit to axis from constrainedData
            ax.set_xlim(extentCellCorners[0], extentCellCorners[1])
            ax.set_ylim(extentCellCorners[2], extentCellCorners[3])

            # if available zoom into area provided by crop shp file in Inputs/CROPSHAPE
            cropFile, cropInfo, _ = gI.getAndCheckInputFiles(
                inDir, "POLYGONS", "cropFile", fileExt="shp", fileSuffix="_cropshape"
            )
            if cropInfo != "No":
                focus = gpd.read_file(cropFile)
                focus.plot(
                    ax=ax,
                    zorder=12,
                    edgecolor="red",
                    linewidth=2,
                    facecolor="none",
                    alpha=0,
                )
                extent = focus.total_bounds
                ax.set_xlim(extent[0], extent[2])
                ax.set_ylim(extent[1], extent[3])

            # add title, labels and ava Info
            title = str("%s" % name)
            ax.set_title(title)
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            pU.putAvaNameOnPlot(ax, avaDir)

            # save and or show figure
            plotPath = pU.saveAndOrPlot({"pathResult": outDir}, plotName.stem, fig)
            plotDict[row["simName"]].update({row["resType"]: plotPath})

    return plotDict


def addConstrainedDataField(fileName, resType, demField, ax, cellSize, alpha=1.0, oneColor=""):
    """find fileName data, constrain data and demField to where there is data,
    create colormap, define extent, add hillshade contours, add to axes
    and add colorbar

    Parameters
    -----------
    fileName: pathlib path
        path to data
    resType: str
        name of result variable type
    demField: numpy ndarray
        array of dem data
    ax: matplotlib axes object
        axes where to add data plot to
    cellSize: float
        cellSize of data
    alpha: float
        from 0 transparent to 1 opaque for plot of constrained data
    oneColor: str
        optional to add a color for a single color for field

    Return
    --------
    ax: matplotlib axes object
        axes updated
    """

    # Load data
    raster = IOf.readRaster(fileName, noDataToNan=True)
    data = raster["rasterData"]

    # constrain data to where there is data
    rowsMin, rowsMax, colsMin, colsMax = pU.constrainPlotsToData(data, cellSize)
    # dataConstrained = data[rowsMin : rowsMax + 1, colsMin : colsMax + 1]
    # demConstrained = demField[rowsMin : rowsMax + 1, colsMin : colsMax + 1]

    dataConstrained = np.ma.masked_where(data == 0.0, data)
    unit = pU.cfgPlotUtils["unit%s" % resType]

    # Set extent of peak file
    ny = data.shape[0]
    nx = data.shape[1]

    # choose colormap
    cmap, col, ticks, norm = pU.makeColorMap(
        pU.colorMaps[resType],
        np.amin(dataConstrained),
        np.amax(dataConstrained),
        continuous=pU.contCmap,
    )
    cmap.set_bad(alpha=0)
    # uncomment this to set the under value for discrete cmap transparent
    # cmap.set_under(alpha=0)

    # set extent in meters using cellSize and llcenter location
    extentCellCentersIm, extentCellCornersIm = pU.createExtentMinMax(
        data, raster["header"], originLLCenter=True
    )
    (
        extentCellCenters,
        extentCellCorners,
        rowsMinPlot,
        rowsMaxPlot,
        colsMinPlot,
        colsMaxPlot,
    ) = pU.createExtent(rowsMin, rowsMax, colsMin, colsMax, raster["header"])

    # add DEM hillshade with contour lines
    # set extent in meters using cellSize and llcenter location
    _, _ = pU.addHillShadeContours(ax, demField, cellSize, extentCellCentersIm)

    # add peak field data
    if oneColor != "":
        dataOneColor = np.where(dataConstrained > 0.0, np.amax(data) * 0.25, np.nan)
        ax.imshow(
            dataOneColor,
            cmap=oneColor,
            norm=norm,
            extent=extentCellCornersIm,
            origin="lower",
            aspect="equal",
            zorder=4,
            alpha=alpha,
        )
    else:
        im1 = ax.imshow(
            dataConstrained,
            cmap=cmap,
            norm=norm,
            extent=extentCellCornersIm,
            origin="lower",
            aspect="equal",
            zorder=4,
            alpha=alpha,
        )
        if len(np.nonzero(data)[0]) > 0.0:
            # add Colorbar
            fig = ax.get_figure()
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im1, cax=cax)

    return ax, rowsMinPlot, colsMinPlot, extentCellCorners


def plotAllFields(avaDir, inputDir, outDir, unit="", constrainData=True):
    """Plot all fields within given directory and save to outDir
    here the extent of the axis is provided in meters and origin lower left corner coordinate

    Parameters
    ----------
    avaDir : str
        path to avalanche directoy
    inputDir : str
        path to input directoy
    outDir : str
        path to directoy where plots shall be saved to
    unit: str
        unit of result type
    constrainData: bool
        if True - raster data is constrained to where data is nonzero + bufferZone (read from plotUtils ini file)
    """

    # Load all infos on simulations
    inputDir = pathlib.Path(inputDir)
    outDir = pathlib.Path(outDir)
    if outDir.is_dir() is False:
        # create out dir if not already existing
        outDir.mkdir()
    peakFiles = list(inputDir.glob("*.asc")) + list(inputDir.glob("*.tif"))

    # Loop through peakFiles and generate plot
    for filename in peakFiles:
        # Load data
        raster = IOf.readRaster(filename)
        data = raster["rasterData"]
        data = np.ma.masked_where(data == 0.0, data)
        name = filename.stem

        # get header info for file writing
        header = raster["header"]
        cellSize = header["cellsize"]

        # Set extent of peak file
        ny = data.shape[0]
        nx = data.shape[1]
        Ly = ny * cellSize
        Lx = nx * cellSize

        # Figure  shows the result parameter data
        fig = plt.figure(figsize=(pU.figW, pU.figH))
        fig, ax = plt.subplots()
        # choose colormap
        cmap, _, ticks, norm = pU.makeColorMap(
            pU.cmapNN, np.amin(data), np.amax(data), continuous=pU.contCmap
        )
        cmap.set_bad("w")

        if constrainData:
            rowsMin, rowsMax, colsMin, colsMax = pU.constrainPlotsToData(data, cellSize)
            dataConstrained = data[rowsMin : rowsMax + 1, colsMin : colsMax + 1]
            data = np.ma.masked_where(dataConstrained == 0.0, dataConstrained)
            # set extent in meters using cellSize and llcenter location
            extentCellCenters, extentCellCorners, _, _, _, _ = pU.createExtent(
                rowsMin, rowsMax, colsMin, colsMax, header
            )
            im1 = ax.imshow(
                data,
                cmap=cmap,
                norm=norm,
                extent=extentCellCorners,
                origin="lower",
                aspect=nx / ny,
            )
        else:
            extentCellCenters, extentCellCorners = pU.createExtentMinMax(data, header, originLLCenter=True)
            im1 = ax.imshow(
                data,
                cmap=cmap,
                norm=norm,
                extent=extentCellCorners,
                origin="lower",
                aspect=nx / ny,
            )

        pU.addColorBar(im1, ax, ticks, unit)

        title = str("%s" % name)
        ax.set_title(title)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")

        plotName = "%s" % (name)

        pU.putAvaNameOnPlot(ax, avaDir)

        # save and or show figure
        plotPath = pU.saveAndOrPlot({"pathResult": outDir}, plotName, fig)
