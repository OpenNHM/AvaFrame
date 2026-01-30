# Tools for plots (partially copied from AvaFrame)


import numpy as np
import os
import pathlib
import matplotlib.pyplot as plt
import rasterio
import logging
from cmcrameri import cm as cmapCrameri
from matplotlib.colors import LightSource
from matplotlib.colors import BoundaryNorm
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

import avaframe.in2Trans.rasterUtils as rasterUtils

# create local logger
log = logging.getLogger(__name__)


def getRasterFile(path):
    """
    read in raster (*.asc or *.tif)

    Parameters:
    -----------
    path: str
        path to raster file or folder containing raster

    Returns:
    -----------
    output: numpy array
        raster
    """
    path = pathlib.Path(path)

    try:
        raster = rasterio.open(path)
        filePath = path
    except:
        files = sorted(list(path.glob("*.asc")))
        if len(files) == 0:
            files = sorted(list(path.glob("*.tif")))
        raster = rasterio.open(files[0])
        # filePath = pathlib.Path(files[0])
        filePath = files[0]
    return filePath


def createExtent(rowsMin, rowsMax, colsMin, colsMax, header, originLLCenter=True):
    """create extent from min to max cols and rows for plots

    Parameters
    -----------
    colsMin, colsMax, rowsMin, rowsMax: int
        index of minimum column and row and maximum column and row of data array
        Note: max is the index of the maximum column that should still be INCLUDED
    header: dict
        header of data array: llcenter coordinates, cellsize
    originLLCenter: bool
        if True the origin is set to lower left cell center / corner if False, 0,0 as origin of lower left center

    Returns
    ---------
    extentCellCenters: list
        extent corresponding to the cell center coordinates in meters
    extentCellCorners: list
        extent corresponding to the cell corner coordinates (lower left corner, uper right corner)
    colsMinPlot, colsMaxPlot, rowsMinPlot, rowsMaxPlot: float
        coordinates in meters corresponding to provided indices
        Note: colsMaxPlot and rowsMaxPlot is colsMax+1*cellSize, rowsMax+1*cellSize
    """

    cellSize = header["cellsize"]

    # set extent in meters using cellSize and llcenter location
    if originLLCenter:
        rowsMinPlot = rowsMin * cellSize + header["yllcenter"]
        rowsMaxPlot = rowsMax * cellSize + header["yllcenter"]
        colsMinPlot = colsMin * cellSize + header["xllcenter"]
        colsMaxPlot = colsMax * cellSize + header["xllcenter"]
    else:
        rowsMinPlot = rowsMin * cellSize
        rowsMaxPlot = rowsMax * cellSize
        colsMinPlot = colsMin * cellSize
        colsMaxPlot = colsMax * cellSize

    # create extent of cell corners lower left to upper right in meters
    extentCellCorners = [
        colsMinPlot - 0.5 * cellSize,
        colsMaxPlot + 0.5 * cellSize,
        rowsMinPlot - 0.5 * cellSize,
        rowsMaxPlot + 0.5 * cellSize,
    ]

    # create extent for cell centers lower left to upper right in meters
    extentCellCenters = [colsMinPlot, colsMaxPlot, rowsMinPlot, rowsMaxPlot]

    return extentCellCenters, extentCellCorners, rowsMinPlot, rowsMaxPlot, colsMinPlot, colsMaxPlot


def constrainPlotsToData(inputData, cellSize, extentOption=False, constrainedData=False, buffer="150"):
    """constrain input raster dataset to where there is data plus buffer zone

    Parameters
    -----------
    inputData: numpy array
        raster data to be plotted
    cellSize: float
        cellsize of raster data
    extentOption: bool
        if True rows and columns limits converted to actual extent in meters
    buffer: float
        buffer for constraining data in meters - optional if not provided read from ini file
    constrainedData: bool
        if True - also return constrained input data array

    Returns
    --------
    rowsMin, rowsMax: int or float
        indices where there is data in x direction (or min/max extent in meters)
    colsMin, colsMax: int or float
        indices where there is data in y direction (or min/max extent in meters)
    dataConstrained: numpy array
        constrained array where there is data
    """

    # check if buffer is given as input or needs to be read from ini file
    plotBuffer = int(buffer) / cellSize

    ind = np.where(inputData > 0)
    if len(ind[0]) > 0:
        rowsMin = max(np.amin(ind[0]) - plotBuffer, 0)
        rowsMax = min(np.amax(ind[0]) + plotBuffer, inputData.shape[0] - 1)
        colsMin = max(np.amin(ind[1]) - plotBuffer, 0)
        colsMax = min(np.amax(ind[1]) + plotBuffer, inputData.shape[1] - 1)
    else:
        # if not constrained as data found at the margins of the array - constrain to nrows-1 or ncols-1 as +1 is used as
        # index to then constrain the array
        rowsMin = 0
        rowsMax = inputData.shape[0] - 1
        colsMin = 0
        colsMax = inputData.shape[1] - 1

    if extentOption:
        rowsMinPlot = rowsMin * cellSize
        rowsMaxPlot = rowsMax * cellSize
        colsMinPlot = colsMin * cellSize
        colsMaxPlot = colsMax * cellSize
        if constrainedData:
            dataConstrained = inputData[rowsMin: rowsMax + 1, colsMin: colsMax + 1]
            return rowsMinPlot, rowsMaxPlot, colsMinPlot, colsMaxPlot, dataConstrained
        else:
            return rowsMinPlot, rowsMaxPlot, colsMinPlot, colsMaxPlot
    else:
        if constrainedData:
            dataConstrained = inputData[rowsMin: rowsMax + 1, colsMin: colsMax + 1]
            return rowsMin, rowsMax, colsMin, colsMax, dataConstrained
        else:
            return rowsMin, rowsMax, colsMin, colsMax


def checkExtentFormat(extentPlot, cellSize, data, plotName=""):
    """check if extent that is provided to imshow plot is in line with cellSize nrows, ncols

    Parameters
    -----------
    extentPlot: list
        list of xmin, xmax, ymin, ymax as actual limits of x and y axis
    cellSize: float
        cell size of cells
    data: numpy nd array
        data array to be plotted in imshow
    plotName: str
        optional for log message
    """

    extentFlagX = np.isclose(((extentPlot[1] - extentPlot[0]) / cellSize), data.shape[1])
    extentFlagY = np.isclose(((extentPlot[3] - extentPlot[2]) / cellSize), data.shape[0])

    if (extentFlagX == False) or (extentFlagY == False):
        message = "Extent of dem data for hillshade and provided extent for figure axes in meters are of wrong shape"
        raise AssertionError(message)


def makeCoordinateGrid(xllc, yllc, csz, ncols, nrows):
    """Create grid

    Parameters
    -----------
    xllc, yllc: float
        x and y coordinate of the lower left center
    csz: float
        cell size
    ncols, nrows: int
        number of columns and rows
    Returns
    --------
    xGrid, yGrid: 2D numpy arrays
        2D vector of x and y values for mesh center coordinates (produced using meshgrid)
    """

    xEnd = (ncols - 1) * csz
    yEnd = (nrows - 1) * csz

    xp = np.linspace(0, xEnd, ncols) + xllc
    yp = np.linspace(0, yEnd, nrows) + yllc

    xGrid, yGrid = np.meshgrid(xp, yp)
    return xGrid, yGrid


def addHillShadeContours(
        ax, data, cellSize, extent, colors=["gray"], onlyContours=False, extentCenters=True
):
    """add hillshade and contours for given DEM data

    Parameters
    -----------
    ax: matplotlib axes
        axes of plot
    data: numpy array
        dem data
    cellSize: float
        cell size of data
    extent: list
        if extentCenters = True:
            extent [x0, x1, y0, y1] x0, y0 lower left center, x1, y1 upper right center
            for extent of imshow plot take into account shift of 0.5cellSize (-0.5*cellsize to get left edge for min row and
            +0.5*cellsizeright edge for rowMax); this is done by computing extentCorners
        elif extentCenters = False:
            extent[x0, x1, y0, y1] - coordinates for extent of x, y axes of imshow plot
    colors: list
        optional, colors for elevation contour lines
    onlyContours: bool
        if True add only contour lines but no hillshade
    """

    azimuthDegree = 315
    elevationDegree = 15
    vertExag = 10
    hillshadeContLevs = 15

    if onlyContours:
        ls = None
    else:
        # create lightSource
        ls = LightSource(azdeg=azimuthDegree, altdeg=elevationDegree)

        # add hillshade to axes
        # create extent for minCol-maxCol+1, minRow-maxRow+1
        if extentCenters:
            extentPlot = [
                extent[0] - 0.5 * cellSize,
                extent[1] + 0.5 * cellSize,
                extent[2] - 0.5 * cellSize,
                extent[3] + 0.5 * cellSize,
            ]
            checkExtentFormat(extentPlot, cellSize, data)

        else:
            extentPlot = extent

        im1 = ax.imshow(
            ls.hillshade(data, vert_exag=vertExag, dx=data.shape[1], dy=data.shape[0]),
            cmap="gray",
            extent=extentPlot,
            origin="lower",
            aspect="equal",
            zorder=1,
        )

    # create x,y coors for data array
    nrows, ncols = data.shape
    X, Y = makeCoordinateGrid(extent[0], extent[2], cellSize, ncols, nrows)

    # add contour lines
    CS = ax.contour(
        X,
        Y,
        data,
        colors=colors,
        levels=hillshadeContLevs,
        alpha=1.0,
        linewidths=0.5,
        zorder=2,
    )

    # add labels
    ax.clabel(CS, CS.levels, inline=True, fontsize=8, zorder=3)

    # add info box with indication of label meaning

    return ls, CS


def zDelta2velocity(zDelta):
    """compute velocity from energy line hight
    Parameters
    -----------
    zDelta: numpy float or array
        energy line height

    Returns
    -----------
    velocity: numpy float or array
        velocity comuted frm zDelta
    """
    velocity = (zDelta * 2 * 9.81) ** 0.5
    return velocity


def readThalwegData(path, startRow, startCol, centerOf="CoE"):
    """
    load thalweg data

    Parameters:
    -----------
    path: str
        OutputPath of the FlowPy simulation
    startRow: int
        row number (y coordinate of raster) of the starting cell of the thalweg/path
    startCol: int
        column number (x coordinate of raster) of the starting cell of the thalweg/path

    Returns:
    -----------
    data: dict
        thalweg data of one thalweg
    """

    data = np.load(f"{path}/thalwegData_{centerOf}_{startRow}_{startCol}.pickle", allow_pickle="TRUE")
    return data


def getInputPath(path):
    """
    get Path of Inputs folder
    if there are more, the first is picked

    Parameters:
    -----------
    path: str
        OutputPath of the FlowPy simulation

    Returns:
    -----------
    path_inputs: str
        path of Inputs folder
    """

    # Find the index of the last occurrence of '/'
    output_index = path.rfind("Outputs")
    if output_index >= 0:
        pathFolder = f"{path[:output_index]}"
    elif output_index == -1:
        pathFolder = path
    # Path to Inputs folder
    path_inputs = f"{pathFolder}/Inputs"

    return path_inputs


def getOutputFile(path, variable):
    """
    find output-raster file (*.asc or *.tif)

    Parameters:
    -----------
    path: str
        path to raster file or folder containing raster
    variable: str
        output variable that is searched for (name is in file name)

    Returns:
    -----------
    files[0]: str
        file name of searched output file
    """

    path = pathlib.Path(path)
    files = sorted(list(path.glob(f"*{variable}.asc")))
    if len(files) == 0:
        files = sorted(list(path.glob(f"*{variable}.tif")))
    return files[0]


def parameterOfAllThalwegs(path, variable):
    """
    get the thalweg data

    Parameters:
    -----------
    path: str
        Thalweg-Output Path of the FlowPy simulation
    variable: str
        name of thalweg parameter

    Returns:
    -----------
    variableValues: list
        thalweg values of the parameter variable of all thalwegs
    """

    variableValues = []
    for filename in os.listdir(path):
        # Check if the filename starts with 'thalweg'
        if filename.startswith("thalwegData"):
            # Construct full file path
            file_path = os.path.join(path, filename)
            data = np.load(file_path, allow_pickle="TRUE")
            # print(data)
            variableValues.append(data[variable])
    return variableValues


def maxParameterOfAllThalwegs(path, variableList, centerOf):
    """
    get thalweg data (maximum per thalweg)

    Parameters:
    -----------
    path: str
        Thalweg-Output Path of the FlowPy simulation
    variable: str
        name of thalweg parameter

    Returns:
    -----------
    variableValues: list
        maximum values of the parameter variable of all thalwegs
    """
    if type(variableList) == str:
        variableList = [variableList]
    variableValues = {}
    for variable in variableList:
        variableValues[variable] = []
    for filename in os.listdir(path):
        # Check if the filename starts with 'thalweg'
        if filename.startswith(f"thalwegData_{centerOf}"):
            # Construct full file path
            file_path = os.path.join(path, filename)
            data = np.load(file_path, allow_pickle="TRUE")
            # print(data)
            for variable in variableList:
                variableValues[variable].append(np.nanmax(data[variable]))
    return variableValues


def plot_hists(region, variable, xlabel, bins=[]):
    """
    brauch ma die??
    """
    path = get_path_output(region)
    var = np.loadtxt(f"{path}/values_{variable}.csv", delimiter=",")

    # Plot the histogram with density normalized by total count
    fig, ax = plt.subplots(tight_layout=True)
    if len(bins) > 0:
        hist, bin_edges, _ = ax.hist(var, bins, edgecolor="white")
    else:
        hist, bin_edges, _ = ax.hist(var, edgecolor="white")

    # Normalize the histogram by total count
    hist_normalized = hist / np.sum(hist)

    # Convert y-axis values to percentages
    hist_percentages = hist_normalized * 100

    # RUNOUT LENGTH!!:
    runout_l1000 = np.sum(var > 1000) / np.sum(var > 0) * 100

    # Plot the histogram with y-axis values as percentages
    fig, ax = plt.subplots(tight_layout=True)
    values = bin_edges[:-1] + np.diff(bin_edges)[0] / 2
    colors = ["#B22222" if value > 900 else "#4169E1" for value in values]  # for runout length!!!
    ax.bar(
        bin_edges[:-1] + np.diff(bin_edges)[0] / 2,
        hist_percentages,
        width=np.diff(bin_edges),
        edgecolor="white",
        color=colors,
    )
    ax.text(1500, 18, f"{round(runout_l1000)}%", color="#B22222", fontsize=14, fontweight="bold")

    # Add labels and title
    plt.ylabel("Relative number in [%]")
    plt.xlabel(xlabel)
    # plt.title(region)

    # Save the figure
    fig.savefig(f"{path}/hist_{variable}_postprocess.png")

    # Show the plot
    plt.show()


def getDataBoxplots(path, variable, centerOf):
    """
    get the thalweg data

    Parameters:
    -----------
    path: str
        OutputPath of the FlowPy simulation
    variable: str
        name of output variable that is analysed and plotted (e.g, impressure, travelLength)

    Returns:
    -----------
    data: numpy array
        maximum value of the parameter varName of all thalwegs
    """

    data = ""

    if variable == "velocity":
        varName = "velocity"
        variable = f"zDelta"

    elif variable == "impressure":
        varName = "impressure"
        variable = f"zDelta"
    else:
        varName = f"{variable}"

    dataDict = maxParameterOfAllThalwegs(f"{path}/thalwegData", variable, centerOf)
    data = np.array(dataDict[variable])
    if varName == "velocity":
        data = zDelta2velocity(data)

    if varName == "impressure":
        velo = zDelta2velocity(data)

        rho = 200  # km m-3
        data = rho * velo ** 2 * 1e-3

    return data


def plotBoxplot(
        path, varName, ylabel, size_class=None, centerOf="CoE", log_scale=False, savePath=None, title=""
):
    """
    shows and potentially saves Violinplot and Boxplot

    Parameters:
    -----------
    path: str
        OutputPath of the FlowPy simulation
    varName: str
        name of output variable that is analysed and plotted (e.g, impressure, travelLength)
    dataNan: np.array
        data that is analysed and plotted (can contain nans)
    ylabel: str
        ylabel of plot
    size_class: list
        values for boundaries between the avalanche sizes for the background colors, if None: no background colors (default=None)
    centerOf: str
        which center of is used (possible:'CoE' (default), 'CoZd', 'CoF')
    log_scale: bool
        yaxis could be logarithmic (if log_scale == True)
    savePath: str
        if not None (=default), the Figure is saved at this path
    title: str
        title for the plot
    """
    dataNan = getDataBoxplots(path, varName, centerOf)

    data = np.delete(dataNan, np.where(np.isnan(dataNan)))
    fig, ax2 = plt.subplots()  # figsize = [4,5])
    # fig.tight_layout()
    labels = [f" (n = {len(data)})"]
    if log_scale:
        ax2.set_yscale("log")
    ax2.violinplot([data])
    ax2.boxplot([data], whis=0, widths=0.07, showfliers=False, medianprops={"color": "blue"})
    ax2.set_xticks(np.arange(1, len(labels) + 1), labels=labels, fontsize=13)
    ax2.set_xlim(0.25, len(labels) + 0.75)

    # Color background
    if size_class != None:
        y_min, y_max = ax2.get_ylim()
        ax2.axhspan(0, size_class[0], facecolor="#008B8B", alpha=0.2)  # Avalanche size 1
        ax2.axhspan(size_class[0], size_class[1], facecolor="#4682B4", alpha=0.2)  # size 2
        ax2.axhspan(size_class[1], size_class[2], facecolor="#6495ED", alpha=0.2)  # size 3
        ax2.axhspan(size_class[2], size_class[3], facecolor="#CD5C5C", alpha=0.2)  # size 4
        ax2.axhspan(size_class[3], y_max, facecolor="#B22222", alpha=0.2)  # size 5

        if varName == "impressure":
            class_lab = "$C_{ip}$"
        elif varName == "path_area":
            class_lab = "$B_{aa}$"
        elif varName == "travelLength":
            class_lab = "$E_{rl}$"

        ax2.text(
            1.5,
            0 + (size_class[0] * 0.75),
            f"{class_lab} 1",
            ha="center",
            va="center",
            color="#008B8B",
            fontsize=13,
        )
        ax2.text(
            1.5,
            size_class[0] + (size_class[1] - size_class[0]) / 2,
            f"{class_lab} 2",
            ha="center",
            va="center",
            color="#4682B4",
            fontsize=13,
        )
        ax2.text(
            1.5,
            size_class[1] + (size_class[2] - size_class[1]) / 2,
            f"{class_lab} 3",
            ha="center",
            va="center",
            color="#6495ED",
            fontsize=13,
        )
        ax2.text(
            1.5,
            size_class[2] + (size_class[3] - size_class[2]) / 2,
            f"{class_lab} 4",
            ha="center",
            va="center",
            color="#CD5C5C",
            fontsize=13,
        )
        ax2.text(
            1.5,
            size_class[3] + (y_max - size_class[3]) / 2,
            f"{class_lab} 5",
            ha="center",
            va="center",
            color="#B22222",
            fontsize=13,
        )
        ax2.set_yticks(size_class)
        ax2.set_yticklabels(size_class, fontsize=13)

    plt.ylabel(ylabel, fontsize=13)

    if title == "":
        title = f"thalwege {centerOf}"
    plt.title(title)
    plt.grid(True)
    if savePath is not None:
        fig.savefig(f"{savePath}/{avaframeName}_Thalweg_{varName}{centerOf}.png")
    plt.show()

    """
    print(f'Median:{np.median(data)}')
    print(f'Mean:{np.mean(data)}')
    print(f'75% percentile:{np.percentile(data,75)}')
    print(f'90% percentile:{np.percentile(data,90)}')
    print(f'80% percentile:{np.percentile(data,80)}')
    """


def plotField(ax, fig, path, pathToOutput, variable, thalwegPra=False):
    """plots hillshade of the DEM and the output raster of the simulation zoomed in to the simulation extent

    Parameters:
    -----------
    ax: matplotlib axis
        axis in which the hillshade and output raster is plotted
    fig: matplotlib figure
        figure to that the plot belongs to
    path: str
        Path to the avalanche directory
    variable: str
        output variable that is plotted (of whole simulation)

    Returns:
    -----------
    ax: matplotlib axis
        axis containing hillshade and output raster of simulation
    """

    pathInput = path / "Inputs"
    demPath = getRasterFile(pathInput)
    praPath = getRasterFile(pathInput / "REL")
    demDict = rasterUtils.readRaster(demPath, flip=False)
    dem = demDict["rasterData"]
    header = demDict["header"]
    cellSize = header["cellsize"]
    clabel = {
        "zdelta": "zDelta [m]",
        "fpTravelAngle": "travel angle [°]",
        "travelLength": "travel length [m]",
        "velocityMax": "velocity [m/s]",
    }

    file = getOutputFile(pathToOutput, variable)
    rasterDict = rasterUtils.readRaster(file, flip=False)
    raster = rasterDict["rasterData"]
    rasterPraDict = rasterUtils.readRaster(praPath, flip=False)
    rasterPra = rasterPraDict["rasterData"]

    rowsMin, rowsMax, colsMin, colsMax = constrainPlotsToData(raster, header["cellsize"])
    rowsMin = int(rowsMin)
    rowsMax = int(rowsMax)
    colsMin = int(colsMin)
    colsMax = int(colsMax)
    dataConstrained = raster[rowsMin: rowsMax + 1, colsMin: colsMax + 1]
    demConstrained = dem[rowsMin: rowsMax + 1, colsMin: colsMax + 1]
    praConstrained = rasterPra[rowsMin: rowsMax + 1, colsMin: colsMax + 1]

    data = np.ma.masked_where(dataConstrained == 0.0, dataConstrained)
    dataConstrained = np.ma.masked_where(dataConstrained == 0.0, dataConstrained)

    # set 0 and smaller to np.nan
    praConstrained = np.where(praConstrained > 0, 1.0, np.nan)

    # Set extent of peak file
    ny = data.shape[0]
    nx = data.shape[1]
    Ly = ny * cellSize
    Lx = nx * cellSize

    extentCellCenters, extentCellCorners, rowsMinPlot, rowsMaxPlot, colsMinPlot, colsMaxPlot = createExtent(
        rowsMin, rowsMax, colsMin, colsMax, header
    )

    _, _ = addHillShadeContours(ax, demConstrained, cellSize, extentCellCenters)

    extent = extentCellCenters
    extentPlot = [
        extent[0] - 0.5 * cellSize,
        extent[1] + 0.5 * cellSize,
        extent[2] - 0.5 * cellSize,
        extent[3] + 0.5 * cellSize,
    ]

    CS = ax.contour(
        demConstrained, levels=np.arange(0, 3500, 100), extent=extentPlot, colors="dimgrey", linewidths=0.5
    )
    ax.clabel(CS, CS.levels[::2], inline=True, fontsize=9)
    # dataOneColor = np.where(dataConstrained > 0.0, np.amax(data)*0.25, np.nan)
    colorsS = ["#FFCEF4", "#FFA7A8", "#C19A1B", "#578B21", "#007054", "#004960", "#201158"]
    cmapS = cmapCrameri.batlow.reversed()
    levels = 7
    bounds = np.linspace(
        np.nanmin(dataConstrained), np.nanmax(dataConstrained), levels + 1
    )  # Define boundaries
    norm = BoundaryNorm(bounds, ncolors=cmapS.N, clip=True)  # Create a norm based on the boundaries

    f = ax.imshow(
        dataConstrained,
        cmap=cmapS,
        norm=norm,
        extent=extentCellCorners,
        origin="lower",
        aspect="equal",
        zorder=4,
        alpha=0.7,
    )
    fig.colorbar(f, ax=ax, label=clabel[variable])
    if thalwegPra:
        cmapPra = ListedColormap(["magenta"])
        cmapPra.set_bad(color="none")
        # normPra = BoundaryNorm([0.5, 1.5], cmapPra.N)
        ax.imshow(
            praConstrained,
            cmap=cmapPra,
            extent=extentCellCorners,
            origin="lower",
            aspect="equal",
            zorder=3,
            alpha=0.5,
            interpolation="none",
        )

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    return ax


"""
def segmentationPra(path, variable, method_thalweg='max', method_PRA='max', returnGroup=False, centerOf='coE'):
    '''
    Gives one value (computed by a selected method) for an avalanche (of combined thalwegs)

    Parameters:
    ---------------
    path: str
        Path to the output folder of the FlowPy simulation
    variable: str
        parameter name (in thalweg data) that is analysed (returned)
    method_thalweg: str
        method how the values of one thalweg are computed (default: 'max')
    method_PRA: str
        method how the values of the thalweg are computed (default: 'max', possible: 'sum', 'mean')
    returnGroup: bool
        if True, the output contains the value of raster PRA (area of contigous PRA)

    Returns:
    ---------------
    max_variable: dict or numpy array
        one value for one avalanche (thalwegs aggregated) (if returnGroup==True: dict contains area of PRA)
    '''
    inputPath = getInputPath(path)
    coordsPRA = getContigousPras(inputPath)
    max_variable = np.array([])
    if returnGroup == True:
        max_variable = {}
    for group in coordsPRA:
        max_PRA = np.array([])
        row_coords = coordsPRA[group][0]
        col_coords = coordsPRA[group][1]
        a = 0
        for row, col in zip(row_coords, col_coords):
            try:
                data = readThalwegData(f'{path}/thalwegData', row, col, centerOf=centerOf)
                if method_thalweg == 'max':
                    thalweg_value = np.max(data[variable])
                elif method_thalweg == None:
                    thalweg_value = data[variable]
                max_PRA = np.append(max_PRA, thalweg_value)
            except:
                a += 1

        try:
            if method_PRA == 'max':
                PRA_value = np.max(max_PRA)
            elif method_PRA == 'sum':
                PRA_value = np.sum(max_PRA)
            elif method_PRA == 'mean':
                PRA_value = np.mean(max_PRA)

            if returnGroup == True:
                max_variable[group] = PRA_value
            else:
                max_variable = np.append(max_variable, PRA_value)
        except:
            PRA_value = np.nan

        if a > 0:
            print(f'{a} files not found for size {group}')

    return max_variable


def getContigousPras(inputPath):
    idsRaster = readRaster(f'{inputPath}/RELid')
    rel = readRaster(f'{inputPath}/REL')
    ids0 = np.unique(idsRaster)
    ids = np.delete(ids0, np.where(ids0 == 0))

    segmentedPras = {}
    for i in ids:
        coords = np.where(idsRaster == i)
        segmentedPras[rel[coords][0]] = coords
    return segmentedPras
"""


def makeFieldPlot(ax, fig, path, pathToOutput, variable, xThalweg, yThalweg, centerOf="", thalwegPra=False):
    """make a raster plot for FlowPy output

    Parameters
    -----------
    ax: matplotlib axis
        Axis for the plot
    fig: matplotlib figure
        Figure for the plot
    path: str
        Path to the avalanche directory
    pathToOutput: Path
        Path to Output folder
    variable: str
        output variable that is plotted (of whole simulation)
    centerOf: str
        which center of is used (possible: '' (default),'CoE', 'CoZd', 'CoF')

    Returns
    -----------
    fig: matplotlib figure
        Figure containg the plot
    ax: matplotlib axis
        Axis containg the plot
    """

    ax = plotField(ax, fig, path, pathToOutput, variable, thalwegPra=thalwegPra)
    ax.scatter(xThalweg, yThalweg, c="r", s=0.3, zorder=5, label=f"thalweg {centerOf}")
    ax.scatter(xThalweg[0], yThalweg[0], c="b", s=2.0, zorder=6, label="startcell")

    ax.legend()
    if thalwegPra:
        # add PRA path to existing legend
        handles, labels = ax.get_legend_handles_labels()
        praPath = Patch(facecolor="magenta", edgecolor="magenta", label="PRA", alpha=0.7)
        handles.append(praPath)
        labels.append(praPath.get_label())
        ax.legend(handles=handles, labels=labels)
    return fig, ax


def makeThalwegPlot(ax, dataThalweg, centerOf=""):
    """make a twodimensional thalweg plot for FlowPy output

    Parameters
    -----------
    ax: matplotlib axis
        Axis for the plot
    dataThalweg: dict
        contains thalweg data:
        s or travelLength (np.array): travel length along thalweg
        z or altitude (np.array): altitude along thalweg
        zDelta (np.array): velocity altitude along thalweg
        alpha (float or int): input parameter of the simulation: alpha angle
        exp (float or int): input parameter of the simulation: exponent
        zDeltaMax (float or int): input parameter of the simulation: zDelta Maximum threshold
    centerOf: str
        which center of is used (possible: '' (default),'CoE', 'CoZd', 'CoF')

    Returns
    -----------
    ax: matplotlib axis
        Axis containg the thalweg plot
    """

    try:
        s = np.array(dataThalweg[f"travelLength"])
    except:
        s = np.array(dataThalweg[f"s"])
    try:
        z = np.array(dataThalweg[f"altitude"])
    except:
        z = np.array(dataThalweg[f"z"])
    zdelta = np.array(dataThalweg[f"zDelta"])

    # get FlowPy input parameter
    alpha = dataThalweg["alpha"]
    exp = dataThalweg["exponent"]
    zDeltaMax = dataThalweg["zDeltaMax"]

    s_max = s[zdelta == max(zdelta)]
    z_max = z[zdelta == max(zdelta)]
    zdelta_max = zdelta[zdelta == max(zdelta)]

    # calculate tatsaechlicher runout angle
    angle_rad = np.arctan((max(z) - min(z)) / (max(s) - min(s)))
    angle_degrees = np.rad2deg(angle_rad)

    ds = max(s) - min(s)
    dh = ds * np.tan(np.deg2rad(alpha))

    ax.hlines(max(z) - dh, ds * 0.85, ds, colors="k", linestyles="dotted", linewidths=0.7)

    ax.plot(s, z, c="gray", linestyle="-", label=f"""$z_{{{centerOf}}}$""")
    ax.plot(s, [d + z for d, z in zip(z, zdelta)], "r", label=f"""$z^{{vel}}_{{{centerOf}}}$""")

    ax.vlines(
        s_max[0],
        z_max[0],
        z_max[0] + zdelta_max[0],
        label="$v_{max}$ = " + str(np.round(np.sqrt(zdelta_max[0] * 2 * 9.81), 1)) + " m/s",
    )
    ax.plot(
        [s[0], s[-1]],
        [z[0], z[-1]],
        color="lightgrey",
        linestyle="--",
        linewidth=1,
        label=rf"""$\alpha_{{eff}}$ = {np.round(angle_degrees, 1)}°""",
    )
    ax.plot(
        [0, ds],
        [max(z), max(z) - dh],
        "k--",
        linewidth=0.7,
        label=rf"""$\alpha_{{input}}$ = {np.round(alpha, 1)}°""",
    )
    ax.plot(
        [s[0], s[-1]],
        [min(z)] * 2,
        color="grey",
        linewidth=1,
        linestyle="--",
        label=rf"""$\Delta$s = {np.round(s[-1] - s[0], 1)} m""",
    )
    ax.vlines(
        x=0,
        ymin=z[-1],
        ymax=z[0],
        color="silver",
        linestyle="--",
        linewidth=1,
        label=(rf"$\Delta z = {np.round(z[0] - z[-1], 1)}$$m$"),
    )

    # ax.text(s_max[0] + 1, z_max[0] + zdelta_max[0]/2, '$v_{max}$ = ' + str(np.round(np.sqrt(zdelta_max[0] * 2 * 9.81),1)) + ' m/s', va = 'center')
    # ax.text((max(s)/5*4), min(z) + (max(z) - min(z)) / 22, fr'{angle_degrees:.1f}°', fontsize=11, ha='center')
    # ax.text((ds*0.88), (max(z)-dh) * 1.05, fr'{alpha:.1f}°', fontsize=11, ha='center')
    ax.set(xlabel=f"""$s_{{{centerOf}}}$ in [m]""")
    ax.set(ylabel="elevation in [m]")
    ax.legend()

    ax.text(
        max(s) * 0.5,
        max(z) * 0.95,
        f"""model parameters: \n alpha: {alpha}° \n exp: {np.round(exp, 1)} \n $Z^{{vel}}_{{max}}$: {np.round(zDeltaMax, 1)} m \n $v_{{max}}$: {round(np.sqrt(zDeltaMax * 2 * 9.81), 1)} m/s""",
        va="top",
        ha="left",
    )

    return ax


def plotThalweg_wetAndDry(path, resName, startRow, startCol, size=None, centerOf="CoE", savePath=None):
    """
    shows and potentially saves Plot of thalweg:
    2 dimensional representation for dry and wet parametrisation
    the

    Parameters:
    -----------
    path: str
        path to the data folder
    resName: str
        name of the folder containing the FlowPy results
    startRow: int
        row number (y coordinate of raster) of the starting cell of the thalweg/path
    startCol: int
        column number (x coordinate of raster) of the starting cell of the thalweg/path
    size: float
        avalanche size of the path
    centerOf: str
        which center of is used (possible:'CoE' (default), 'CoZd', 'CoF')
    savePath: str
        if not None (=default), the Figure is saved at this path
    """

    fig, axs = plt.subplots(1)  # (3,1)
    fig.tight_layout(pad=3.0)

    for ava in ["dry", "wet"]:
        pathOutput = f"{path}/{ava}/Outputs/com4FlowPy/peakFiles/{resName}"
        data = readThalwegData(f"{pathOutput}/thalwegData", startRow, startCol, centerOf=centerOf)
        try:
            s = np.array(data[f"travelLength"])
        except:
            s = np.array(data[f"s"])
        try:
            z = np.array(data[f"altitude"])
        except:
            z = np.array(data[f"z"])
        zdelta = np.array(data[f"zDelta"])

        alpha = data["alpha"]
        exp = data["exponent"]
        zDeltaMax = data["zDeltaMax"]

        s_max = s[zdelta == max(zdelta)]
        z_max = z[zdelta == max(zdelta)]
        zdelta_max = zdelta[zdelta == max(zdelta)]

        angle_rad = np.arctan((max(z) - min(z)) / (max(s) - min(s)))
        angle_degrees = np.rad2deg(angle_rad)

        ds = max(s) - min(s)
        dh = ds * np.tan(np.deg2rad(alpha))

        axs.hlines(
            max(z) - dh, ds * 0.85, ds, colors="k", linestyles="dotted", linewidths=0.7
        )  # , 'k:')#, linewidth=0.7)
        axs.text((ds * 0.88), (max(z) - dh) * 1.05, rf"{alpha:.0f}°", fontsize=11, ha="center")

        axs.plot([s[0], s[-1]], [min(z)] * 2, "k", linewidth=0.5)

        p = axs.plot(s, [d + z for d, z in zip(z, zdelta)], label=f"""$z^{{vel}}_{{{centerOf}}}$, {ava}""")
        axs.vlines(s_max[0], z_max[0], z_max[0] + zdelta_max[0], color=p[0].get_color(), linestyle="--")
        axs.text(
            s_max[0] + 1,
            z_max[0] + zdelta_max[0] / 2,
            f"""$v_{{max}}$ = {np.round(np.sqrt(zdelta_max[0] * 2 * 9.81), 1):.0f} m/s""",
            va="center",
        )
        axs.plot([0, ds], [max(z), max(z) - dh], "k:", linewidth=0.7)
        axs.plot(
            [s[0], s[-1]],
            [z[0], z[-1]],
            "--",
            linewidth=0.5,
            color=p[0].get_color(),
            label=rf"""$\alpha_{{eff}}$ = {np.round(angle_degrees, 1)}°""",
        )

        # axs.text((max(s)/4.3*4), min(z) + (max(z) - min(z)) / 90, fr'{angle_degrees:.0f}°', fontsize=10, ha='center')
        # Inputparameters
        if ava == "dry":
            textPosition = [0.66, 0.45]
        elif ava == "wet":
            textPosition = [0.66, 0.22]
        axs.text(
            0,
            max(z) * textPosition[1],
            f"""{ava}: \n alpha: {alpha:.0f}° \n exp: {np.round(exp, 1):.0f} \n max $Z^{{vel}}$: {np.round(zDeltaMax, 1):.0f} m (= {round(np.sqrt(zDeltaMax * 2 * 9.81), 1):.0f} m/s)""",
            va="top",
            ha="left",
            fontsize=9,
            color=p[0].get_color(),
        )

    axs.plot(s, z, c="gray", linestyle="-", label=f"""$z_{{{centerOf}}}$""")

    axs.set(xlabel=f"""$s_{{{centerOf}}}$ in [m]""")
    axs.set(ylabel="elevation in [m]")
    axs.legend()

    axs.set_title(f"size: {size}")
    if savePath is not None:
        fig.savefig(f"{savePath}/{avaframeName}_Thalweg{centerOf}_wetAndDry_{startRow}_{startCol}.png")
