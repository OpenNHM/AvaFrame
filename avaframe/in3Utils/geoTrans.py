""" Opperations and transformations of rasters and lines
"""

import logging
import math
import pathlib
import numpy as np
import scipy as sp
import scipy.interpolate
import shapely as shp
import rasterio
import rasterio.warp
from rasterio.enums import Resampling
import copy
from scipy.interpolate import splprep, splev
import matplotlib.path as mpltPath
from shapely import LineString, Point, distance, MultiPoint

# Local imports
import avaframe.in3Utils.fileHandlerUtils as fU
import avaframe.in2Trans.rasterUtils as rU
from avaframe.com1DFA import particleTools


# create local logger
log = logging.getLogger(__name__)


def projectOnRaster(dem, Points, interp="bilinear", inData="rasterData", outData="z"):
    """Projects Points on raster
    using a bilinear or nearest interpolation and returns the z coord (no for loop)

    Parameters
    -------------
    dem: dict
        dem dictionary
    Points: dict
        Points dictionary (x,y)
    interp: str
        interpolation option, between nearest or bilinear
    inData: str
        key in the dem dict of the 2D field to use for the interpolation.
    outData: str
        key in the Points dict toe updat with the interpolated data.

    Returns
    -------
    Points: dict
        Points dictionary with z coordinate added or updated
    ioob: int
        number of out of bounds indexes
    """
    header = dem["header"]
    rasterdata = dem[inData]
    xllc = header["xllcenter"]
    yllc = header["yllcenter"]
    cellsize = header["cellsize"]
    xcoor = Points["x"]
    ycoor = Points["y"]

    zcoor, ioob = projectOnGrid(xcoor, ycoor, rasterdata, csz=cellsize, xllc=xllc, yllc=yllc, interp=interp)
    Points[outData] = zcoor
    return Points, ioob


def projectOnGrid(x, y, Z, csz=1, xllc=0, yllc=0, interp="bilinear", getXYField=False):
    """Projects Z onto points (x,y)
    using a bilinear or nearest interpolation and returns the z coord

    Parameters
    -------------
    x: array
        x coord of the points to project
    y: array
        y coord of the points to project
    Z : 2D numpy array
        raster data
    csz: float
        cellsize corresponding to the raster data
    xllc: float
        x coord of the lower left center of the raster
    yllc: float
        y coord of the lower left center of the raster
    interp: str
        interpolation option, between nearest or bilinear
    getXYField: bool
        also return raster with dimension of raster data mark all cells that are used for interpolation of
        x, y coordinates of points to project

    Returns
    -------
    z : 2D numpy array
        projected data on the raster data
    ioob: int
        number of out of bounds indexes
    """
    nrow, ncol = np.shape(Z)
    zField = np.zeros((nrow, ncol))
    # initialize outputs
    z = np.ones((np.shape(x))) * np.nan
    dx = np.ones((np.shape(x))) * np.nan
    dy = np.ones((np.shape(x))) * np.nan
    f11 = np.ones((np.shape(x))) * np.nan
    f12 = np.ones((np.shape(x))) * np.nan
    f21 = np.ones((np.shape(x))) * np.nan
    f22 = np.ones((np.shape(x))) * np.nan

    # find coordinates in normalized ref (origin (0,0) and cellsize 1)
    Lxx = (x - xllc) / csz
    Lyy = (y - yllc) / csz
    Lx = copy.deepcopy(Lxx)
    Ly = copy.deepcopy(Lyy)

    # find out of bound indexes
    if interp == "nearest":
        Lx[np.where((Lxx <= -0.5))] = np.nan
        Ly[np.where((Lxx <= -0.5))] = np.nan
        Lx[np.where(Lxx >= (ncol - 0.5))] = np.nan
        Ly[np.where(Lxx >= (ncol - 0.5))] = np.nan
        Lx[np.where(Lyy <= -0.5)] = np.nan
        Ly[np.where(Lyy <= -0.5)] = np.nan
        Lx[np.where(Lyy >= (nrow - 0.5))] = np.nan
        Ly[np.where(Lyy >= (nrow - 0.5))] = np.nan
    elif interp == "bilinear":
        Lx[np.where((Lxx < 0))] = np.nan
        Ly[np.where((Lxx < 0))] = np.nan
        Lx[np.where(Lxx >= (ncol - 1))] = np.nan
        Ly[np.where(Lxx >= (ncol - 1))] = np.nan
        Lx[np.where(Lyy < 0)] = np.nan
        Ly[np.where(Lyy < 0)] = np.nan
        Lx[np.where(Lyy >= (nrow - 1))] = np.nan
        Ly[np.where(Lyy >= (nrow - 1))] = np.nan

    # find index of not nan value
    mask = ~np.isnan(Lx + Ly)
    maskInd = np.argwhere(~np.isnan(Lx + Ly))[:, 0]
    itot = len(Lx)
    iinb = len(maskInd)
    ioob = itot - iinb

    # find coordinates of the 4 nearest cornes on the raster
    Lx0 = np.floor(Lx)
    Ly0 = np.floor(Ly)
    Lx1 = Lx0 + 1
    Ly1 = Ly0 + 1
    # prepare for bilinear interpolation(do not take out of bound into account)
    if interp == "nearest":
        dx[mask] = np.round(Lx[mask])
        dy[mask] = np.round(Ly[mask])
        z[mask] = Z[dy[mask].astype("int"), dx[mask].astype("int")]
        zField[dy[mask].astype("int"), dx[mask].astype("int")] = 1.0
    elif interp == "bilinear":
        dx[mask] = Lx[mask] - Lx0[mask]
        dy[mask] = Ly[mask] - Ly0[mask]
        f11[mask] = Z[Ly0[mask].astype("int"), Lx0[mask].astype("int")]
        f12[mask] = Z[Ly1[mask].astype("int"), Lx0[mask].astype("int")]
        f21[mask] = Z[Ly0[mask].astype("int"), Lx1[mask].astype("int")]
        f22[mask] = Z[Ly1[mask].astype("int"), Lx1[mask].astype("int")]
        # using bilinear interpolation on the cell
        z = f11 * (1 - dx) * (1 - dy) + f21 * dx * (1 - dy) + f12 * (1 - dx) * dy + f22 * dx * dy
        zField[Ly0[mask].astype("int"), Lx0[mask].astype("int")] = 1.0
        zField[Ly1[mask].astype("int"), Lx0[mask].astype("int")] = 1.0
        zField[Ly0[mask].astype("int"), Lx1[mask].astype("int")] = 1.0
        zField[Ly1[mask].astype("int"), Lx1[mask].astype("int")] = 1.0

    if getXYField:
        return z, zField
    else:
        return z, ioob


def resizeData(raster, rasterRef):
    """
    Reproject raster on a grid of shape rasterRef, raster adapts cellsize and extend of rasterRef

    Parameters
    ----------
    raster : dict
        raster dictionary
    rasterRef : dict
        reference raster dictionary

    Returns
    -------
    data : 2D numpy array
        reprojected data
    dataRef : 2D numpy array
        reference data
    """
    if rU.isEqualASCheader(raster["header"], rasterRef["header"]):
        return raster["rasterData"], rasterRef["rasterData"]
    else:
        headerRef = rasterRef["header"]
        ncols = headerRef["ncols"]
        nrows = headerRef["nrows"]
        csz = headerRef["cellsize"]
        xllc = headerRef["xllcenter"]
        yllc = headerRef["yllcenter"]
        xgrid = np.linspace(xllc, xllc + (ncols - 1) * csz, ncols)
        ygrid = np.linspace(yllc, yllc + (nrows - 1) * csz, nrows)
        X, Y = np.meshgrid(xgrid, ygrid)
        Points = {"x": X, "y": Y}
        Points, _ = projectOnRaster(raster, Points, interp="bilinear")
        raster["rasterData"] = Points["z"]
        return raster["rasterData"], rasterRef["rasterData"]


def remeshData(rasterDict, cellSizeNew, remeshOption="griddata", interpMethod="cubic", larger=True):
    """compute raster data on a new mesh with cellSize using the specified remeshOption.

        remeshOption are to choose between 'griddata' or 'RectBivariateSpline'
        Only the 'griddata' works properly if the input data contains noData points,
        'RectBivariateSpline' is faster but fails if input data contains noData points.
        The new mesh is as big or smaller as the original mesh if larger is False and bigger if larger is True

    Parameters
    ----------
    rasterDict : dict
        raster dictionary (with header and rasterData)
    cellSizeNew : float
        mesh size of new mesh
    remeshOption: str
        method used to remesh ('griddata' or 'RectBivariateSpline')
        Check the scipy documentation for more details
        default is 'griddata'
    interpMethod: str
        interpolation order to use for the interpolation ('linear', 'cubic' or 'quintic')
    larger: Boolean
        if true (default) output grid is at least as big as the input

    Returns
    -------
    data : dict
        remeshed data dict with data as numpy array and header info

    """
    header = rasterDict["header"]

    # fetch shape info and get new mesh info
    xGrid, yGrid, _, _ = makeCoordGridFromHeader(header)
    xGridNew, yGridNew, ncolsNew, nrowsNew = makeCoordGridFromHeader(
        header, cellSizeNew=cellSizeNew, larger=larger
    )
    z = rasterDict["rasterData"]
    log.info(
        "Remeshed data extent difference x: %f and y %f"
        % (xGrid[-1, -1] - xGridNew[-1, -1], yGrid[-1, -1] - yGridNew[-1, -1])
    )
    if remeshOption == "griddata":
        xGrid = xGrid.flatten()
        yGrid = yGrid.flatten()
        zCopy = np.copy(z).flatten()
        # make sure to remove the nans (no data points) from the input
        mask = np.where(~np.isnan(zCopy))
        xGrid = xGrid[mask]
        yGrid = yGrid[mask]
        z = zCopy[mask]
        zNew = sp.interpolate.griddata(
            (xGrid, yGrid), z, (xGridNew, yGridNew), method=interpMethod, fill_value=header["nodata_value"]
        )
    elif remeshOption == "RectBivariateSpline":
        if np.isnan(z).any():
            message = 'Data to remesh contains NaNs. Can not interpole with "RectBivariateSpline".'
            log.error(message)
            raise ValueError(message)
        if interpMethod == "linear":
            k = 1
        elif interpMethod == "cubic":
            k = 3
        elif interpMethod == "quintic":
            k = 5
        else:
            message = "There is no %s interpolation method available for RectBivariateSpline" % interpMethod
            log.error(message)
            raise NameError(message)
        zNew = sp.interpolate.RectBivariateSpline(yGrid[:, 0], xGrid[0, :], z, ky=k, kx=k)(
            yGridNew[:, 0], xGridNew[0, :], grid=True
        )
        # zNew = zNew.reshape(np.shape(xGrid))

    # create header of remeshed DEM
    # set new header
    headerRemeshed = header
    headerRemeshed["cellsize"] = cellSizeNew
    headerRemeshed["ncols"] = ncolsNew
    headerRemeshed["nrows"] = nrowsNew
    xllcenter = header["xllcenter"]
    yllcenter = header["yllcenter"]
    transform = rasterio.transform.from_origin(
        xllcenter - cellSizeNew / 2.0,
        (yllcenter - cellSizeNew / 2.0) + nrowsNew * cellSizeNew,
        cellSizeNew,
        cellSizeNew,
    )
    headerRemeshed["transform"] = transform
    # create remeshed raster dictionary
    remeshedRaster = {"rasterData": zNew, "header": headerRemeshed}

    return remeshedRaster


def remeshDataRio(rasterFile, cellSizeNew, larger=True):
    """resample raster data using rasterio to change effective cell size to cellSizeNew by specifying an output array
    of specified size, default resampling option is set to cubic

    Parameters
    -------------
    rasterFile: pathlib Path
        path to file with raster data options asci or tif
    cellSizeNew: float
        desired spatial resolution of new raster dataset, cellsize
    larger: Boolean
        if true (default) output grid is at least as big as the input

    Returns
    ---------
    remeshedRaster: dict
        dictionary with header and rasterdata with new cellSize
    """
    # open raster data
    src = rasterio.open(rasterFile, "r")

    # First part is to be able to replicate our own remeshing
    header = rU.getHeaderFromRaster(src)
    _, _, width, height = makeCoordGridFromHeader(header, cellSizeNew=cellSizeNew, larger=larger)

    targetTransform = rasterio.transform.from_origin(
        header["xllcenter"] - cellSizeNew / 2.0,
        (header["yllcenter"] - cellSizeNew / 2.0) + height * cellSizeNew,
        cellSizeNew,
        cellSizeNew,
    )

    # Check if crs is none, as warp.reproject NEEDS a valid crs
    # If none is available, set any. The original is taken again when
    # the header is reset lower down
    # TODO: require a default crs in config
    if src.crs is None:
        srcCrs = rasterio.crs.CRS.from_epsg(31287)
    else:
        srcCrs = src.crs

    data, transform = rasterio.warp.reproject(
        source=src.read(),
        destination=np.empty((src.count, height, width)) * np.nan,
        src_transform=src.transform,
        dst_transform=targetTransform,
        src_crs=srcCrs,
        dst_crs=srcCrs,
        src_nodata=src.nodata,
        dst_nodata=src.nodata,
        resampling=Resampling.cubic,
    )

    data = np.where(data == src.nodata, np.nan, data)
    # create header of resampled data
    # set new header
    headerRemeshed = rU.getHeaderFromRaster(src)
    src.close()

    headerRemeshed["cellsize"] = transform[0]
    headerRemeshed["transform"] = transform
    headerRemeshed["ncols"] = data[0].shape[1]
    headerRemeshed["nrows"] = data[0].shape[0]
    headerRemeshed["nodata_value"] = np.nan

    # create remeshed raster dictionary
    remeshedRaster = {"rasterData": data[0], "header": headerRemeshed}

    return remeshedRaster


def remeshRaster(rasterFile, cfgSim, typeIndicator="DEM", onlySearch=False, legacy=False):
    """change raster cell size by reprojecting on a new grid - first check if remeshed raster available

    the new raster is as big or smaller as the original raster and saved to Inputs/remeshedRasters as
    remeshedTYPEINDICATORcellSize

    Interpolation is based on rasterio rsampling with a cubic method. Here would be the place
    to change the order of the interpolation or to switch to another interpolation method.

    We provide the legacy option via the legacy parameter, then the
    interpolation is based on griddata with a cubic method.

    Parameters
    ----------
    rasterFile: str or pathlib path
        file path to DEM in Inputs/
    cfgSim : configParser
        meshCellSizeThreshold : threshold under which no remeshing is done
        meshCellSize : desired cell size
    typeIndicator: str
        type of raster, possible options DEM or RELTH
    onlySearch: bool
        if True - only searching for remeshed DEM but not remeshing if not found
    legacy: bool
        if True use old resampling options (griddata and RectBivariateSpline)

    Returns
    -------
    pathRaster : str
        path of raster with desired cell size relative to Inputs/

    """
    # first check if remeshed raster is available
    pathRaster, rasterFound, allRasterNames = searchRemeshedRaster(rasterFile.stem, cfgSim)
    if rasterFound or onlySearch:
        return pathRaster

    # -------- if no remeshed raster found - remesh
    # fetch info on raster file
    raster = rU.readRaster(rasterFile)
    headerRaster = raster["header"]
    # read raster header info
    cszRaster = headerRaster["cellsize"]
    # fetch info on desired meshCellSize
    cszRasterNew = float(cfgSim["GENERAL"]["meshCellSize"])

    # start remesh
    log.info(
        "Remeshing the input raster (of cell size %.2g m) to a cell size of %.2g m"
        % (cszRaster, cszRasterNew)
    )
    if legacy:
        remeshedRaster = remeshData(
            raster, cszRasterNew, remeshOption="griddata", interpMethod="cubic", larger=False
        )
        log.info("Legacy option used for remeshing")
        flipArg = True
    else:
        log.info("Using rasterio resampling")
        remeshedRaster = remeshDataRio(rasterFile, cszRasterNew, larger=False)
        flipArg = False

    # save remeshed raster
    pathToRaster = pathlib.Path(cfgSim["GENERAL"]["avalancheDir"], "Inputs", "remeshedRasters")
    fU.makeADir(pathToRaster)

    outFile = pathToRaster / (
            "%s_remeshed%s%.2f" % (rasterFile.stem, typeIndicator, remeshedRaster["header"]["cellsize"])
    )

    writtenFile = rU.writeResultToRaster(
        remeshedRaster["header"], remeshedRaster["rasterData"], outFile, flip=flipArg
    )
    log.info("Saved remeshed raster to %s" % writtenFile)
    pathRaster = str(pathlib.Path("remeshedRasters", writtenFile.name))

    return pathRaster


def searchRemeshedRaster(rasterName, cfgSim, typeIndicator="DEM"):
    """search if remeshed raster with correct name and cell size already available

    Parameters
    -----------
    rasterName: str
        name of raster file in Inputs/
    typeIndicator: str
        which type of raster to handle, possible options Raster or RELTH
    cfgSim: configparser object
        configuration settings: avaDir, meshCellSize, meshCellSizeThreshold

    Returns
    --------
    pathRaster: pathlib path
        to Raster
    rasterFound: bool
        flag if dem is found
    allRasterNames: list
        of all names of dems found in Inputs/remeshedRasters
    """

    # path to remeshed Raster folder
    pathToRasters = pathlib.Path(cfgSim["GENERAL"]["avalancheDir"], "Inputs", "remeshedRasters")
    rasterFound = False
    pathRaster = ""
    allRasterNames = []

    # fetch info on desired meshCellSize
    meshCellSize = float(cfgSim["GENERAL"]["meshCellSize"])
    meshCellSizeThreshold = float(cfgSim["GENERAL"]["meshCellSizeThreshold"])

    # check if Raster is available
    if pathToRasters.is_dir():
        # look for rasters and check if cellSize within tolerance and origin matches
        rasterFiles = list(pathToRasters.glob("*remeshed%s*" % typeIndicator))
        allRasterNames = [d.name for d in rasterFiles]
        for rasterF in rasterFiles:
            headerRaster = rU.readRasterHeader(rasterF)
            if (
                    abs(meshCellSize - headerRaster["cellsize"]) < meshCellSizeThreshold
                    and rasterName in rasterF.stem
            ):
                log.info(
                    "Remeshed Raster found: %s cellSize: %.5f" % (rasterF.name, headerRaster["cellsize"])
                )
                rasterFound = True
                pathRaster = str(pathlib.Path("remeshedRasters", rasterF.name))
                continue
            else:
                log.debug(
                    "Remeshed raster found %s with cellSize %.2f - not used"
                    % (rasterF, headerRaster["cellsize"])
                )

    else:
        log.debug("Directory %s does not exist" % pathToRasters)

    return pathRaster, rasterFound, allRasterNames


def computeS(avaPath):
    """compute s coordinate given a path (x, y)

    Parameters
    -----------
    avaPath: dict
        path dictionary with x and y coordinates as 1D numpy arrays

    Returns
    --------
    avaPath: dict
        path dictionary updated with s coordinate
    """
    xcoord = avaPath["x"]
    ycoord = avaPath["y"]
    n = np.size(xcoord)
    # compute s
    dxs = xcoord[1:n] - xcoord[0: n - 1]
    dys = ycoord[1:n] - ycoord[0: n - 1]
    # deduce the distance in s direction
    ds2 = dxs * dxs + dys * dys
    ds = np.sqrt(ds2)
    scoord = np.cumsum(ds)
    avaPath["s"] = np.insert(scoord, 0, 0)
    return avaPath


def prepareLineStrict(dem, avapath, distance, Point=None):
    """Resample and project line on dem, with strict settings, i.e. to follow
    the line as close as possible. Calls prepareLine
    """
    avaProfile, projPoint = prepareLine(dem, avapath, distance, Point, k=1, s=0.0)
    return avaProfile, projPoint


def prepareLine(dem, avapath, distance, Point=None, k=3, s=None):
    """Resample and project line on dem
    1- Resample the avapath line with an interval of approximately distance in meters
    between points (projected distance on the horizontal plane).
    2- Make avalanche profile out of the path (affect a z value using the dem)
    3- Get projection of points on the profile (closest point)

    Parameters
    -----------
    dem: dict
        dem dictionary
    avapath: dict
        line dictionary
    distance: float
        resampling distance
    Point: dict
        a point dictionary (optional, can contain several point)
    k: int
        Degree of the spline for splprep. Set to splprep default of 3, use 1
        if you want to lower the level of spline (3 is cubic)
    s: float
        A smoothing condition for splprep. Defaults to None (i.e. splprep default), set to 0 if you want
        to minimize the distance between new line and old line


    Returns
    -------
    avaProfile: dict
        the resampled avapath with the z coordinate
    projPoint: dict
        point dictionary projected on the profile (if several points
        were give in input, only the closest point to the profile
        is projected)
    """

    # fetch x, y coors from avapath
    x = avapath["x"]
    y = avapath["y"]

    # check if duplicate points in avapath coordinates
    indexNonDup = np.where(np.abs(np.diff(x)) + np.abs(np.diff(y)) > 0)
    xcoor = x[indexNonDup]
    xNew = np.append(xcoor, x[-1])
    ycoor = y[indexNonDup]
    yNew = np.append(ycoor, y[-1])

    # create a B-spline with scipy for given x, y line
    if len(xNew) <= 3 and k != 1:
        tck, u = splprep([xNew, yNew], k=len(xNew) - 1, s=s)
        log.warning(
            "Path is defined by only %d points - degree of spline is set to %d" % (len(xNew), len(xNew) - 1)
        )
    else:
        tck, u = splprep([xNew, yNew], k=k, s=s)

    # compute accumulated distance along spline of x, y
    s = computeLengthOfLine2D(xNew, yNew)
    # compute number of desired points along spline of x, y as a function of
    # length of the spline and the desired resample distance of the line
    nPoints = np.ceil(s[-1] / distance) + 1
    # evaluate spline for nPoints
    uPoints = np.linspace(0, 1, int(nPoints))
    xcoornew, ycoornew = splev(uPoints, tck)
    # compute accumulated distance along spline of xcoornew, ycoornew
    sNew = computeLengthOfLine2D(xcoornew, ycoornew)
    # start with 0
    sNew = np.append([0], sNew)

    resampAvaPath = avapath
    resampAvaPath["x"] = xcoornew
    resampAvaPath["y"] = ycoornew
    resampAvaPath, _ = projectOnRaster(dem, resampAvaPath)
    resampAvaPath["s"] = sNew
    avaProfile = resampAvaPath

    # find split point by computing the distance to the line
    if Point:
        projPoint = findSplitPoint(avaProfile, Point)
    else:
        projPoint = None

    return avaProfile, projPoint


def computeLengthOfLine2D(x, y):
    """compute distance along a line in 2D

    Parameters
    ------------
    x, y: np array
        x, y coordinates of line

    Returns
    ---------
    s: np array
        accumulated distance measured along line from point to point

    """
    dx = np.diff(x)
    dy = np.diff(y)
    s = np.sqrt(dx**2 + dy**2)
    s = s.cumsum()

    return s


def findPointOnDEM(dem, vDirX, vDirY, vDirZ, zHighest, xFirst, yFirst, zFirst):
    """find point on dem given a direction and a z value to reach

    Parameters
    -----------
    dem: dict
        dem dict
    vDirX, vDirY, vDirZ: floats
        x, y and z components of the direction in which to extend
    zHighest: float
        z value to reach
    xFirst, yFirst, zFirst: floats
        x, y and z coordinates of the starting point

    Returns
    --------
    xExtTop, yExtTop, zExtTop:floats
        x, y and z coordinates of the point found
    """
    header = dem["header"]
    xllc = header["xllcenter"]
    yllc = header["yllcenter"]
    csz = header["cellsize"]
    zRaster = dem["rasterData"]
    gamma = (zHighest - zFirst) / vDirZ * np.linspace(0.25, 2, 100)
    xArray = xFirst + gamma * vDirX
    yArray = yFirst + gamma * vDirY
    zArray, _ = projectOnGrid(xArray, yArray, zRaster, csz=csz, xllc=xllc, yllc=yllc, interp="bilinear")
    idx = np.nanargmin(np.abs(zArray - np.array([zHighest])))
    xExtTop = np.array([xFirst + gamma[idx] * vDirX])
    yExtTop = np.array([yFirst + gamma[idx] * vDirY])
    zExtTop = np.array([zArray[idx]])
    return xExtTop, yExtTop, zExtTop


def findSplitPoint(avaProfile, Points):
    """Finds the closest point in Points to the avaProfile and returns
    its projection on avaProfile.

    Parameters
    -----------
    avaProfile: dict
        line dictionary with x and y coordinates
    Points: dict
        a point dictionary

    Returns
    -------
    projPoint: dict
        point dictionary projected on the profile (if several points
        were give in input, only the closest point to the profile
        is projected)
    """

    xcoor = avaProfile["x"]
    ycoor = avaProfile["y"]

    indSplit = findClosestPoint(xcoor, ycoor, Points)
    projPoint = {}
    projPoint["x"] = avaProfile["x"][indSplit]
    projPoint["y"] = avaProfile["y"][indSplit]
    projPoint["z"] = avaProfile["z"][indSplit]
    projPoint["s"] = avaProfile["s"][indSplit]
    projPoint["indSplit"] = indSplit
    return projPoint


def findClosestPoint(xcoor, ycoor, pointsDict):
    """find the closest point of pointDict along line defined by xcoor and ycoor - only xy plane!

    Parameters
    -----------
    xcoor, ycoor: np array
        x and y coordinates of line
    pointsDict: dict
        a dictionary with coordinates of points with keys x and y

    Returns
    --------
    indSplit: int
        index of closest point found on the line
    """

    Dist = np.empty((0))
    IndSplit = np.empty((0))
    for i in range(len(pointsDict["x"])):
        dist = np.sqrt((xcoor - pointsDict["x"][i]) ** 2 + (ycoor - pointsDict["y"][i]) ** 2)
        indSplit = np.argmin(dist)
        IndSplit = np.append(IndSplit, indSplit)
        Dist = np.append(Dist, dist[indSplit])

    ind = np.argmin(Dist)
    indSplit = int(IndSplit[ind])

    return indSplit


def computeAlongLineDistance(line, dim="2D"):
    """compute distance along a dict of coordinates or a shapely lineString incrementally

    Parameters
    ------------
    line: lineString shapely or dict
        lineString object or dict with x, y, z keys and np.arrays as items
    dim: str
        2D only in xy, 3D in xyz

    Returns
    --------
    distancePoints: list
        list of starting (at zero) distance along this line
    """

    # fetch coordinates of line
    if isinstance(line, dict):
        x = line["x"]
        y = line["y"]
        if dim.lower() == "3d":
            z = line["z"]
    elif isinstance(line, shp.LineString):
        x = np.asarray([coord[0] for coord in line.coords])
        y = np.asarray([coord[1] for coord in line.coords])
        if dim.lower() == "3d":
            z = np.asarray([coord[1] for coord in line.coords])

    # compute distance of points along line
    distancePoints = [0]
    for i in range(len(x) - 1):
        if dim.lower() != "2d":
            distancePoints.append(
                distancePoints[i]
                + np.sqrt((x[i + 1] - x[i]) ** 2 + ((y[i + 1] - y[i]) ** 2) + (z[i + 1] - z[i]) ** 2)
            )
        else:
            distancePoints.append(
                distancePoints[i] + np.sqrt((x[i + 1] - x[i]) ** 2 + ((y[i + 1] - y[i]) ** 2))
            )

    return distancePoints


def checkProfile(avaProfile, projSplitPoint=None):
    """check that the avalanche profiles goes from top to bottom
    flip it if not and adjust the splitpoint in consequence

    Parameters
    -----------
    avaProfile: dict
        line dictionary with x and y coordinates
    projSplitPoint: dict
        a point dictionary already projected on the avaProfile

    Returns
    -------
    avaProfile: dict
        avaProfile, fliped if needed
    projSplitPoint: dict
        point dictionary
    """
    if projSplitPoint:
        indSplit = projSplitPoint["indSplit"]
    if avaProfile["z"][-1] > avaProfile["z"][0]:
        log.info("Profile reversed")
        avaProfile["x"] = np.flip(avaProfile["x"])
        avaProfile["y"] = np.flip(avaProfile["y"])
        avaProfile["z"] = np.flip(avaProfile["z"])
        try:
            L = avaProfile["s"][-1]
            avaProfile["s"] = L - np.flip(avaProfile["s"])
        except KeyError:
            pass

        if projSplitPoint:
            indSplit = len(avaProfile["x"]) - indSplit - 1
            projSplitPoint["indSplit"] = indSplit
            avaProfile["indSplit"] = indSplit
        else:
            projSplitPoint = None
            avaProfile["indSplit"] = None

    return projSplitPoint, avaProfile


def findAngleProfile(tmp, ds, dsMin):
    """
    Find the beta point: first point under the beta value given in
    prepareAngleProfile. Make sure that at least dsMin meters behind the point
    are also under the beta value otherwise keep searching

    Parameters
    ----------
    tmp: 1D numpy array
        index array of point in profile with slope bellow the given beta angle
        and bellow the splitPoint
    ds: 1D numpy array
        distance between points discribed in tmp
    dsMin: float
        threshold distance [m] for looking for the beta point (at least dsMin meters below
        beta degres)

    Returns
    -------
    idsAnglePoint: int
        index of beta point
    """
    noPointFoundMessage = "No point found. Check the angle and threshold distance."
    i = 0
    condition = True
    if np.size(tmp) == 0:
        raise IndexError(noPointFoundMessage)
    while i <= np.size(tmp) and condition:
        ind = tmp[i]
        j = 0
        dist = 0
        while dist < dsMin:
            try:
                condition = condition and (tmp[i + j + 1] == ind + j + 1)
                dist = dist + ds[i + j]
            except IndexError:
                raise IndexError(noPointFoundMessage)
            if not condition:
                i = i + j + 1
                break
            j = j + 1
        if condition:
            idsAnglePoint = ind
            break
        condition = True
    return idsAnglePoint


def prepareAngleProfile(beta, avaProfile, raiseWarning=True):
    """Prepare inputs for findAngleProfile function
    Read profile (s, z), compute the slope Angle
    look for points for which the slope is under the given Beta value and
    that are located downstream of the splitPoint

    Parameters
    ----------
    beta: float
        beta angle in degrees
    avaProfile: dict
        profile dictionary, s, z and a split point(optional)
    raiseWarning: bool
        True to raise eventual warnings
    Returns
    -------
    angle: 1D numpy array
        profile angle
    tmp: 1D numpy array
        index array of point in profile with slope bellow the given beta angle
        and bellow the splitPoint
    ds: 1D numpy array
        distance between points discribed in tmp
    """

    s = avaProfile["s"]
    z = avaProfile["z"]
    try:
        indSplit = avaProfile["indSplit"]
        sSplit = s[indSplit]
    except KeyError:
        if raiseWarning:
            log.warning("No split Point given!")
        sSplit = 0
    ds = np.abs(s - np.roll(s, 1))
    dz = np.roll(z, 1) - z
    ds[0] = ds[1]
    dz[0] = dz[1]
    angle = np.rad2deg(np.arctan2(dz, ds))
    # get all values where Angle < beta but >0
    # get index of first occurance and go one back to get previous value
    # (i.e. last value above beta)
    # tmp = x[(angle < beta) & (angle > 0.0) & (x > 450)]
    tmp = np.where((angle <= beta) & (s > sSplit))
    tmp = np.asarray(tmp).flatten()
    ds = ds[tmp]
    return angle, tmp, ds


def isCounterClockWise(path):
    """Determines if a polygon path is mostly clockwise or counter clockwise

    https://stackoverflow.com/a/45986805/15887086

    Parameters
    ----------
    path: matplotlib.path
        polygon path
    Returns
    -------
    isCounterCloc1: int
        1 if the path is counter clockwise, 0 otherwise
    """
    v = path.vertices - path.vertices[0, :]
    a = np.arctan2(v[1:, 1], v[1:, 0])
    isCounterClock = (a[1:] >= a[:-1]).astype(int).mean() >= 0.5
    return isCounterClock


def getCellsAlongLine(header, lineDict, addBuffer=True):
    """Find all raster cells crossed by the line
    line has to be entierly contained on the raster extend. If addBuffer is True, add neighbour cells to the result
    based on https://stackoverflow.com/a/35808540/15887086

    Parameters
    ----------
    header: dict
        raster header
    lineDict: dict
        line dictionary
    addBuffer: boolean
        True to add a 1 cell buffer around the line
    Returns
    -------
    lineDict: dict
        line dictionary updated with the "cellsCrossed" 1D array (boolean array of 0 and 1 if the cell is crossed by the
        line or in its neigborhood)
    """
    ncols = header["ncols"]
    nrows = header["nrows"]
    xllc = header["xllcenter"]
    yllc = header["yllcenter"]
    csz = header["cellsize"]
    # normalize line coordinates
    xArray = (lineDict["x"] - xllc) / csz
    yArray = (lineDict["y"] - yllc) / csz
    # loop on line points
    cellsCrossed = np.zeros((ncols * nrows), dtype=np.int64)
    for i in range(np.size(xArray) - 1):
        xA = xArray[i]
        xB = xArray[i + 1]
        yA = yArray[i]
        yB = yArray[i + 1]
        dx = xB - xA
        dy = yB - yA
        sx = np.sign(dx)
        sy = np.sign(dy)
        # add starting point cell to cell list
        indX = round(xA)
        indY = round(yA)
        indCell = indX + ncols * indY
        cellsCrossed[indCell] = 1
        if addBuffer:
            cellsCrossed, _, _ = getNeighborCells(indX, indY, ncols, nrows, cellsCrossed)
        # find next intersection with vertical and horizontal axis
        tIx = dy * (indX + sx / 2 - xA) if dx != 0 else float("+inf")
        tIy = dx * (indY + sy / 2 - yA) if dy != 0 else float("+inf")
        indXB = round(xB)
        indYB = round(yB)
        while (indX, indY) != (indXB, indYB):
            # NB if tIx == tIy we increment both x and y
            (movx, movy) = (abs(tIx) <= abs(tIy), abs(tIy) <= abs(tIx))

            if movx:
                # intersection is at (indX + sx, yA + tIx / dx^2)
                indX += sx
                tIx = dy * (indX + sx / 2 - xA)

            if movy:
                # intersection is at (xA + tIy / dy^2, indY + sy)
                indY += sy
                tIy = dx * (indY + sy / 2 - yA)

            indX = round(indX)
            indY = round(indY)
            indCell = indX + ncols * indY
            cellsCrossed[indCell] = 1
            if addBuffer:
                cellsCrossed, _, _ = getNeighborCells(indX, indY, ncols, nrows, cellsCrossed)
    lineDict["cellsCrossed"] = cellsCrossed
    return lineDict


def getNeighborCells(indX, indY, ncols, nrows, cellsArray):
    """Find the neighbour cells to a given cell

    Parameters
    ----------
    indX: int
        x index of the cell for which you want to find the direct neighbors
    indY: int
        y index of the cell for which you want to find the direct neighbors
    ncols: int
        number of cols in the raster
    nrows: int
        number of rows in the raster
    cellsArray: 1D int array
        boolean array of 0 and 1 if the cell is crossed by the line or in its neigborhood
    Returns
    -------
    cellsArray: 1D int array
        updated boolean array of 0 and 1 if the cell is crossed by the line or in its neigborhood
    """
    indXList = []
    indYList = []
    for i in [-1, 0, 1]:
        if (indX + i < ncols) & (indX + i >= 0):
            for j in [-1, 0, 1]:
                if (indY + j < nrows) & (indY + j >= 0):
                    indCell = (indX + i) + ncols * (indY + j)
                    cellsArray[indCell] = 1
                    indXList.append(indX + i)
                    indYList.append(indY + j)
    return cellsArray, indXList, indYList


def path2domain(xyPath, rasterTransfo):
    """Creates a domain (irregular raster) along a path,
    given the path xyPath, a domain width and a raster cellsize

    Parameters:
    -------------
    xyPath: dict
        line dictionary with coordinates x and y
    rasterTransfo: dict
        rasterTransfo['w']: float
            Domain width
        rasterTransfo['cellSizeSL']: float
            cellsize expected for the new raster

    Returns:
    ---------
    rasterTransfo: dict
        rasterTransfo updated with xp, yp Arrays determining a path of width w along a line

        rasterTransfo['DBXl']:
            x coord of the left boundary
        rasterTransfo['DBXr']:
            x coord of the right boundary
        rasterTransfo['DBYl']:
            y coord of the left boundary
        rasterTransfo['DBYr']:
            y coord of the right boundary

    [Fischer2013] Fischer, Jan-Thomas. (2013).
    A novel approach to evaluate and compare computational snow avalanche
    simulation.
    Natural Hazards and Earth System Sciences.
    13. 1655-. 10.5194/nhess-13-1655-2013.
    Uwe Schlifkowitz/ BFW, June 2011
    """
    csz = rasterTransfo["cellSizeSL"]
    x = xyPath["x"]
    y = xyPath["y"]
    # compute the non dimensional width
    w = rasterTransfo["domainWidth"] / 2 / csz
    # remove scaling due to cellsize
    x = x / csz
    y = y / csz

    # Difference between x- bzw. y-Coordinates of Polyline
    # first and last  Vertex: Difference between this and the next
    # other vertices: Difference between previous and next
    dx = np.array((x[1] - x[0]))
    dy = np.array((y[1] - y[0]))
    n = len(x)
    for i in range(2, n):
        dx = np.append(dx, (x[i] - x[i - 2]) / 2.0)
        dy = np.append(dy, (y[i] - y[i - 2]) / 2.0)

    dx = np.append(dx, x[-1] - x[-2])
    dy = np.append(dy, y[-1] - y[-2])

    # Direction of normal vector of difference,
    # a.k.a. bisecting line of angle
    d = np.arctan2(dy, dx) + math.pi / 2

    # x- and y-Coordinates (left and right) of path edges,
    # total width w
    # x-KOO[left right]
    DBXl = np.array((x + w * np.cos(d)))
    DBXr = np.array((x + w * np.cos(d + math.pi)))
    # y-KOO[left right]
    DBYl = np.array((y + w * np.sin(d)))
    DBYr = np.array((y + w * np.sin(d + math.pi)))

    rasterTransfo["DBXl"] = DBXl
    rasterTransfo["DBXr"] = DBXr
    rasterTransfo["DBYl"] = DBYl
    rasterTransfo["DBYr"] = DBYr

    return rasterTransfo


def areaPoly(X, Y):
    """Gauss's area formula to calculate polygon area

    Parameters
    ----------
    X: 1D numpy array
        x coord of the vertices
    Y: 1D numpy array
        y coord of the vertices
    (Without repeating the first vertex!!!)
    Returns
    -------
    area: float
        Area of the polygon
    """

    X = np.append(X, X[0])
    Y = np.append(Y, Y[0])
    area = 0
    for i in range(np.size(X) - 1):
        area = area + (X[i] * Y[i + 1] - Y[i] * X[i + 1]) / 2
    return area


def prepareArea(line, dem, radius, thList="", combine=True, checkOverlap=True):
    """convert shape file polygon to raster

    Parameters
    ----------
    line: dict
        line dictionary
    dem : dict
        dictionary with dem information
    radius : float
        include all cells which center is in the polygon or close enough
    thList: list
        thickness values for all features in the line dictionary
    combine : Boolean
        if True sum up the rasters in the area list to return only 1 raster
        if False return the list of distinct area rasters
        this option works only if thList is not empty
    checkOverlap : Boolean
        if True check if features are overlapping and return an error if it is the case
        if False check if features are overlapping and average the value for overlapping areas
        (Attention: if combine is set to False, you do not see the result of the averaging
        since the list of raters was not affected by the averaging step)

    Returns
    -------
    updates the line dictionary with the rasterData:
        contains either

        -  Raster: 2D numpy array, raster of the area (returned if relRHlist is empty OR if combine is set
        to True)
        - RasterList: list, list of 2D numpy array rasters (returned if relRHlist is not empty AND
        if combine is set to False)

    """
    NameRel = line["Name"]
    StartRel = line["Start"]
    LengthRel = line["Length"]
    RasterList = []

    for i in range(len(NameRel)):
        name = NameRel[i]
        start = StartRel[i]
        end = start + LengthRel[i]
        avapath = {
            "x": line["x"][int(start) : int(end)],
            "y": line["y"][int(start) : int(end)],
            "Name": name,
        }
        # if relTh is given - set relTh
        if thList != "":
            log.info(
                "%s feature %s, thickness: %.2f - read from %s"
                % (line["type"], name, thList[i], line["thicknessSource"][i])
            )
            Raster = polygon2Raster(dem["originalHeader"], avapath, radius, th=thList[i])
        else:
            Raster = polygon2Raster(dem["originalHeader"], avapath, radius)
        RasterList.append(Raster)

    # if RasterList not empty check for overlap between features
    Raster = np.zeros(np.shape(dem["rasterData"]))
    for rast in RasterList:
        ind1 = Raster > 0
        ind2 = rast > 0
        indMatch = np.logical_and(ind1, ind2)
        if indMatch.any():
            # if there is an overlap, raise error
            if checkOverlap:
                message = "Features %s are overlapping - this is not allowed" % NameRel
                log.error(message)
                raise AssertionError(message)
            else:
                # if there is an overlap, take average of values for the overlapping cells
                Raster = np.where(((Raster > 0) & (rast > 0)), (Raster + rast) / 2, Raster + rast)
        else:
            Raster = Raster + rast

    if combine:
        line["rasterData"] = Raster
    else:
        line["rasterData"] = RasterList

    return line


def polygon2Raster(demHeader, Line, radius, th=""):
    """convert line to raster

    Parameters
    ----------
    demHeader: dict
        dem header dictionary
    Line : dict
        line dictionary
    radius : float
        include all cells which center is in the polygon or close enough
    th: float
        thickness value ot the line feature

    Returns
    -------
    Mask : 2D numpy array
        updated raster
    """
    # adim and center dem and polygon
    ncols = demHeader["ncols"]
    nrows = demHeader["nrows"]
    xllc = demHeader["xllcenter"]
    yllc = demHeader["yllcenter"]
    csz = demHeader["cellsize"]
    xCoord0 = (Line["x"] - xllc) / csz
    yCoord0 = (Line["y"] - yllc) / csz
    if (xCoord0[0] == xCoord0[-1]) and (yCoord0[0] == yCoord0[-1]):
        xCoord = np.delete(xCoord0, -1)
        yCoord = np.delete(yCoord0, -1)
    else:
        xCoord = copy.deepcopy(xCoord0)
        yCoord = copy.deepcopy(yCoord0)

    # get the raster corresponding to the polygon
    polygon = np.stack((xCoord, yCoord), axis=-1)
    path = mpltPath.Path(polygon)
    # add a tolerance to include cells for which the center is on the lines
    # for this we need to know if the path is clockwise or counterclockwise
    # to decide if the radius should be positive or negative in contains_points
    is_ccw = isCounterClockWise(path)
    r = radius * is_ccw - radius * (1 - is_ccw)
    x = np.linspace(0, ncols - 1, ncols)
    y = np.linspace(0, nrows - 1, nrows)
    X, Y = np.meshgrid(x, y)
    X = X.flatten()
    Y = Y.flatten()
    points = np.stack((X, Y), axis=-1)
    mask = path.contains_points(points, radius=r)
    Mask = mask.reshape((nrows, ncols)).astype(int)
    # thickness field is provided, then return array with ones
    if th != "":
        log.debug("REL set from dict, %.2f" % th)
        Mask = np.where(Mask > 0, th, 0.0)
    else:
        Mask = np.where(Mask > 0, 1.0, 0.0)

    return Mask


def checkParticlesInRelease(particles, line, radius):
    """remove particles laying outside the polygon

    Parameters
    ----------
    particles : dict
        particles dictionary
    line: dict
        line dictionary
    radius: float
        threshold val that decides if a point is in the polygon, on the line or
        very close but outside

    Returns
    -------
    particles : dict
        particles dictionary where particles outside of the polygon have been removed
    """
    Mask = getParticlesInPolygon(particles, line, radius)
    # also remove particles with negative mass
    mask = np.where(particles["m"] <= 0, False, True)
    Mask = np.logical_and(Mask, mask)
    nRemove = len(Mask) - np.sum(Mask)
    if nRemove > 0:
        particles = particleTools.removePart(particles, Mask, nRemove, "")
        log.debug("removed %s particles because they are not within the release polygon" % (nRemove))

    return particles


def getParticlesInPolygon(particles, line, radius):
    """
    check which particles are within a polygon (including a buffer)

    Parameters
    ----------
    particles : dict
        particles dictionary
    line: dict
        line dictionary of polygon
    radius: float
        this value is added to the polygon coordinates to decide whether a particles
        is located within or outside of the polygon

    Returns
    -------
    Mask: numpy array
        is True at particle indices that are inside the polygon, and False for outside the polygon
    """
    NameRel = line["Name"]
    StartRel = line["Start"]
    LengthRel = line["Length"]
    Mask = np.full(np.size(particles["x"]), False)
    for i in range(len(NameRel)):
        name = NameRel[i]
        start = StartRel[i]
        end = start + LengthRel[i]
        avapath = {
            "x": line["x"][int(start) : int(end)],
            "y": line["y"][int(start) : int(end)],
            "Name": name,
        }
        # get an array that have at the same indices of a particle a True if the particle is within the polygon
        # and a False if it is outside the polygon (including a buffer (radius))
        mask = pointInPolygon(line["header"], particles, avapath, radius)
        Mask = np.logical_or(Mask, mask)
    return Mask


def pointInPolygon(demHeader, points, Line, radius):
    """find particles within a polygon

    Parameters
    ----------
    demHeader: dict
        dem header dictionary
    points: dict
        points to check
    Line : dict
        line dictionary
    radius: float
        threshold val that decides if a point is in the polygon, on the line or
        very close but outside

    Returns
    -------
    Mask : 1D numpy array
        Mask of particles that are within the polygon
    """
    xllc = demHeader["xllcenter"]
    yllc = demHeader["yllcenter"]
    xCoord0 = Line["x"] - xllc
    yCoord0 = Line["y"] - yllc
    if (xCoord0[0] == xCoord0[-1]) and (yCoord0[0] == yCoord0[-1]):
        xCoord = np.delete(xCoord0, -1)
        yCoord = np.delete(yCoord0, -1)
    else:
        xCoord = copy.deepcopy(xCoord0)
        yCoord = copy.deepcopy(yCoord0)

    # get the raster corresponding to the polygon
    polygon = np.stack((xCoord, yCoord), axis=-1)
    path = mpltPath.Path(polygon)
    # add a tolerance to include cells for which the center is on the lines
    # for this we need to know if the path is clockwise or counter clockwise
    # to decide if the radius should be positif or negatif in contains_points
    is_ccw = isCounterClockWise(path)
    r = radius * is_ccw - radius * (1 - is_ccw)
    points2Check = np.stack((points["x"], points["y"]), axis=-1)
    mask = path.contains_points(points2Check, radius=r)
    mask = np.where(mask > 0, True, False)

    return mask


def checkOverlap(toCheckRaster, refRaster, nameToCheck, nameRef, crop=False):
    """Check if two rasters overlap

    Parameters
    ----------
    toCheckRaster : 2D numpy array
        Raster to check
    refRaster : 2D numpy array
        reference Raster
    nameToCheck: str
        name of raster that might overlap
    nameRef: str
        name of reference raster
    crop : boolean
        if True, remove overlaping part and send a warning
    Returns
    -------
    toCheckRaster: 2D numpy array
        if crop is True, return toCheckRaster without the overlaping part and send a
        warning if needed
        if crop is False, return error if Rasters overlap otherwise return toCheckRaster
    """
    mask = (toCheckRaster > 0) & (refRaster > 0)
    if mask.any():
        if crop:
            toCheckRaster[mask] = 0
            message = "%s area feature overlapping with %s area - removing the overlapping part" % (
                nameToCheck,
                nameRef,
            )
            log.warning(message)
        else:
            message = "%s area features overlapping with %s area - this is not allowed" % (
                nameToCheck,
                nameRef,
            )
            log.error(message)
            raise AssertionError(message)

    return toCheckRaster


def cartToSpherical(X, Y, Z):
    """convert from cartesian to spherical coordinates

    Parameters
    -----------
    X: float
        x coordinate
    Y: float
        y coordinate
    Z: float
        z coordinate

    Returns
    ---------
    r: float
        radius
    phi: float
        azimuth angle [degrees]
    theta: float
        for elevation angle defined from Z-axis down [degrees]
    """

    xy = X**2 + Y**2
    r = np.sqrt(xy + Z**2)
    # for elevation angle defined from Z-axis down
    theta = np.arctan2(np.sqrt(xy), Z)
    theta = np.degrees(theta)
    # azimuth: 0 degree is south
    phi = np.arctan2(X, Y)
    phi = np.degrees(phi)

    return r, phi, theta


def rotateRaster(rasterDict, theta, deg=True):
    """rotate clockwise a raster arround (0, 0) with theta angle

    Parameters
    -----------
    rasterDict: dict
        raster dictionary
    theta: float
        rotation angle of the vector from start point to end point - degree default

    deg: bool
        if true theta is converted to rad from degree

    Returns
    --------
    rotatedRaster: dict
        rotated raster dictionary
    """

    # convert to rad if provided as degree
    if deg:
        theta = np.radians(theta)

    # create raster grid with origin 0,0
    header = rasterDict["header"]
    xllc = header["xllcenter"]
    yllc = header["yllcenter"]
    ncols = header["ncols"]
    nrows = header["nrows"]
    csz = header["cellsize"]
    X, Y = makeCoordinateGrid(xllc, yllc, csz, ncols, nrows)

    # rotate Grid
    xTheta = np.cos(theta) * X + np.sin(theta) * Y
    yTheta = -np.sin(theta) * X + np.cos(theta) * Y

    # project data on this new grid
    rotatedZ, _ = projectOnGrid(
        xTheta, yTheta, rasterDict["rasterData"], csz=csz, xllc=xllc, yllc=yllc, interp="bilinear"
    )

    rotatedRaster = {"header": header, "rasterData": rotatedZ}
    return rotatedRaster


def rotate(locationPoints, theta, deg=True):
    """rotate a vector provided as start and end point with theta angle
    rotation counter-clockwise

    Parameters
    -----------
    locationPoints: list
        list of lists with x,y coordinate of start and end point of a line
    theta: float
        rotation angle of the vector from start point to end point - degree default
    deg: bool
        if true theta is converted to rad from degree

    Returns
    --------
    rotatedLine: list
        list of lists of x,y coordinates of start and end point of rotated vector
    """

    # convert to rad if provided as degree
    if deg:
        theta = np.radians(theta)

    # create vector with origin 0,0
    vector = np.diff(locationPoints)

    # create rotation matrix
    # counterclockwise rotation
    rotationMatrix = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )

    # rotate vector
    vectorRot = np.dot(rotationMatrix, vector)

    # create rotated line as list of start and end point
    rotatedLine = [
        [locationPoints[0][0], float(locationPoints[0][0] + vectorRot[0][0])],  # x
        [locationPoints[1][0], float(locationPoints[1][0] + vectorRot[1][0])],  # y
    ]

    return rotatedLine


def makeCoordGridFromHeader(rasterHeader, cellSizeNew=None, larger=False):
    """Get x and y (2D) grid description vectors for a mesh
    with a given number of rows and columns, lower left center and cellSize.
    If 'cellSizeNew' is not None use cellSizeNew instead of rasterHeader['cellsize']
    Make sure the new grid is at least as big as the old one if larger=True
    (can happen if 'cellSizeNew' is not None)

    Parameters
    -----------
    rasterHeader: dict
        ratser header with info on ncols, nrows, csz, xllcenter, yllcenter, nodata_value
    cellSizeNew: float
        If not None, use cellSizeNew as cell size
    larger: boolean
        If True, make sure the extend of the (xGrid, yGrid) is larger or equal than the
        header one
    Returns
    --------
    xGrid, yGrid: 2D numpy arrays
        2D vector of x and y values for mesh center coordinates (produced using meshgrid)
    ncols, nrows: int
        number of columns and rows

    """
    ncols = rasterHeader["ncols"]
    nrows = rasterHeader["nrows"]
    xllc = rasterHeader["xllcenter"]
    yllc = rasterHeader["yllcenter"]
    csz = rasterHeader["cellsize"]
    # if a new cell size is provided, compute the new ncols and nrows
    if cellSizeNew is not None:
        xExtent = (ncols - 1) * csz
        yExtent = (nrows - 1) * csz
        ncolsNew = int(xExtent / cellSizeNew + 1)
        nrowsNew = int(yExtent / cellSizeNew + 1)
        # get rid of the case cellSizeNew = csz (which would lead to a too large grid)
        if larger and ((ncolsNew - 1) * cellSizeNew < xExtent):
            ncols = ncolsNew + 1
            nrows = nrowsNew + 1
        else:
            ncols = ncolsNew
            nrows = nrowsNew
        csz = cellSizeNew
    # create the grid
    xGrid, yGrid = makeCoordinateGrid(xllc, yllc, csz, ncols, nrows)
    return xGrid, yGrid, ncols, nrows


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


def snapPtsToLine(dbData, projstr, lineName, pointsList):
    """snap points to line in dataframe only considering x, y plane!

    Parameters
    -----------
    dbData: pandas dataframe
        dataframe with geometry info of events
    lineName: str
        name of line column except projstr
    pointsList: list
        list with point column names except projstr
    projstr: str
        projection string to append to all names

    Returns
    --------
    dbData: pandas dataframe
        updated dataframe with ..._snapped point column
    """

    for pt in pointsList:
        dbData[pt + "_" + projstr + "_snapped"] = np.empty(len(dbData))
        dbData["distanceXY"] = np.empty(len(dbData))

    for index, row in dbData.iterrows():
        xycoor = np.asarray(dbData.loc[index, ("%s_%s_resampled" % (lineName, projstr))].coords.xy)
        zcoorTemp = dbData.loc[index, ("%s_%s_resampled" % (lineName, projstr))].coords
        zcoor = np.asarray([coord[2] for coord in zcoorTemp])

        for pt in pointsList:
            pointsDict = {
                "x": [dbData.loc[index, ("%s_%s" % (pt, projstr))].x],
                "y": [dbData.loc[index, ("%s_%s" % (pt, projstr))].y],
            }

            indSplit = findClosestPoint(xycoor[0, :], xycoor[1, :], pointsDict)
            projPoint = shp.Point(xycoor[0, indSplit], xycoor[1, indSplit], zcoor[indSplit])
            dbData.loc[index, (pt + "_" + projstr + "_snapped")] = projPoint

    return dbData


def getNormalMesh(dem, num=4):
    """Compute normal to surface at grid points

    Get the normal vectors to the surface defined by a DEM.
    Either by adding the normal vectors of the adjacent triangles for each
    points (using 4, 6 or 8 adjacent triangles). Or use the next point in
    x direction and the next in y direction to define two vectors and then
    compute the cross product to get the normal vector

    - moved from com1DFA/DFAtools

    Normal computation on rectangular grid explanation for num parameter
       4 triangles method             6 triangles method             8 triangles method
    ++-----U------UR-----++-...    ++-----++-----++-----++-...    ++-----++-----++-----++-...
    ||   //||\\   ||   //||        ||   //|| 2 //||   //||        ||\\ 2 || 3 //||   //||         Y
    ||  // || \\  ||  // ||        ||  // ||  // ||  // ||        || \\  ||  // ||  // ||        //\\
    || //  ||  \\ || //  || //     || //  || //  || //  || //     ||  \\ || //  || //  || //      ||
    ||// 1 || 2 \\||//   ||//      ||// 1 ||// 3 ||//   ||//      || 1 \\||// 4 ||//   ||//       ||
    ++-----P------L------++-...    ++-----++-----++-----++-...    ++-----++-----++-----++-... -------> X
    ||\\ 4 || 3 //||   //||        || 6 //|| 4 //||   //||        || 8 //||\\ 5 ||   //||
    || \\  ||  // ||  // ||        ||  // ||  // ||  // ||        ||  // || \\  ||  // ||
    ||  \\ || //  || //  || //     || //  || //  || //  || //     || //  ||  \\ || //  || //
    ||   \\||//   ||//   ||//      ||// 5 ||//   ||//   ||//      ||// 7 || 6 \\||//   ||//
    ++-----++-----++-----++-...    ++-----++-----++-----++-...    ++-----++-----++-----++-...
    ||   //||   //||   //||        ||   //||   //||   //||        ||   //||   //||   //||
    4 Options available : -1: simple cross product method (with the diagonals P-UR and U-L)
                          -4: 4 triangles method
                          -6: 6 triangles method
                          -8: 8 triangles method

    Parameters
    ----------
        dem: dict
            header :
                dem header (cellsize, ncols, nrows)
            rasterData : 2D numpy array
                elevation at grid points
        num: int
            chose between 4, 6 or 8 (using then 4, 6 or 8 triangles) or
            1 to use the simple cross product method (with the diagonals)

    Returns
    -------
    dem: dict
        dem dict updated with:
            Nx: 2D numpy array
                x component of the normal vector field on grid points
            Ny: 2D numpy array
                y component of the normal vector field on grid points
            Nz: 2D numpy array
                z component of the normal vector field on grid points
            outOfDEM: 2D boolean numpy array
                True if the cell is out the dem, False otherwise
    """
    # read dem header
    header = dem["header"]
    ncols = header["ncols"]
    nrows = header["nrows"]
    csz = header["cellsize"]
    # read rasterData
    z = dem["rasterData"]
    n, m = np.shape(z)
    Nx = np.ones((n, m))
    Ny = np.ones((n, m))
    Nz = np.ones((n, m))
    # first and last row, first and last column are inacurate
    if num == 4:
        # filling the inside of the matrix
        # normal calculation with 4 triangles
        # (Zl - Zr) / csz
        Nx[1: n - 1, 1: m - 1] = (z[1: n - 1, 0: m - 2] - z[1: n - 1, 2:m]) / csz
        # (Zd - Zu) * csz
        Ny[1: n - 1, 1: m - 1] = (z[0: n - 2, 1: m - 1] - z[2:n, 1: m - 1]) / csz
        Nz = 2 * Nz
        # filling the first col of the matrix
        # -2*(Zr - Zp) / csz
        Nx[1: n - 1, 0] = -2 * (z[1: n - 1, 1] - z[1: n - 1, 0]) / csz
        # (Zd - Zu) / csz
        Ny[1: n - 1, 0] = (z[0: n - 2, 0] - z[2:n, 0]) / csz
        # filling the last col of the matrix
        # 2*(Zl - Zp) / csz
        Nx[1: n - 1, m - 1] = 2 * (z[1: n - 1, m - 2] - z[1: n - 1, m - 1]) / csz
        # (Zd - Zu) / csz
        Ny[1: n - 1, m - 1] = (z[0: n - 2, m - 1] - z[2:n, m - 1]) / csz
        # filling the first row of the matrix
        # (Zl - Zr) / csz
        Nx[0, 1: m - 1] = (z[0, 0: m - 2] - z[0, 2:m]) / csz
        # -2*(Zu - Zp) / csz
        Ny[0, 1: m - 1] = -2 * (z[1, 1: m - 1] - z[0, 1: m - 1]) / csz
        # filling the last row of the matrix
        # (Zl - Zr) / csz
        Nx[n - 1, 1: m - 1] = (z[n - 1, 0: m - 2] - z[n - 1, 2:m]) / csz
        # 2*(Zd - Zp) / csz
        Ny[n - 1, 1: m - 1] = 2 * (z[n - 2, 1: m - 1] - z[n - 1, 1: m - 1]) / csz
        # filling the corners of the matrix
        Nx[0, 0] = -(z[0, 1] - z[0, 0]) / csz
        Ny[0, 0] = -(z[1, 0] - z[0, 0]) / csz
        Nz[0, 0] = 1
        Nx[n - 1, 0] = -(z[n - 1, 1] - z[n - 1, 0]) / csz
        Ny[n - 1, 0] = (z[n - 2, 0] - z[n - 1, 0]) / csz
        Nz[n - 1, 0] = 1
        Nx[0, m - 1] = (z[0, m - 2] - z[0, m - 1]) / csz
        Ny[0, m - 1] = -(z[1, m - 1] - z[0, m - 1]) / csz
        Nz[0, m - 1] = 1
        Nx[n - 1, m - 1] = (z[n - 1, m - 2] - z[n - 1, m - 1]) / csz
        Ny[n - 1, m - 1] = (z[n - 2, m - 1] - z[n - 1, m - 1]) / csz
        Nz[n - 1, m - 1] = 1

    if num == 6:
        # filling the inside of the matrix
        # normal calculation with 6 triangles
        # (2*(Zl - Zr) - Zur + Zdl + Zu - Zd) / csz
        Nx[1: n - 1, 1: m - 1] = (
                                         2 * (z[1: n - 1, 0: m - 2] - z[1: n - 1, 2:m])
                                         - z[2:n, 2:m]
                                         + z[0: n - 2, 0: m - 2]
                                         + z[2:n, 1: m - 1]
                                         - z[0: n - 2, 1: m - 1]
                                 ) / csz
        # (2*(Zd - Zu) - Zur + Zdl - Zl + Zr) / csz
        Ny[1: n - 1, 1: m - 1] = (
                                         2 * (z[0: n - 2, 1: m - 1] - z[2:n, 1: m - 1])
                                         - z[2:n, 2:m]
                                         + z[0: n - 2, 0: m - 2]
                                         - z[1: n - 1, 0: m - 2]
                                         + z[1: n - 1, 2:m]
                                 ) / csz
        Nz = 6 * Nz
        # filling the first col of the matrix
        # (- 2*(Zr - Zp) + Zu - Zur ) / csz
        Nx[1: n - 1, 0] = (-2 * (z[1: n - 1, 1] - z[1: n - 1, 0]) + z[2:n, 0] - z[2:n, 1]) / csz
        # (Zd - Zu + Zr - Zur) / csz
        Ny[1: n - 1, 0] = (z[0: n - 2, 0] - z[2:n, 0] + z[1: n - 1, 1] - z[2:n, 1]) / csz
        Nz[1: n - 1, 0] = 3
        # filling the last col of the matrix
        # (2*(Zl - Zp) + Zdl - Zd) / csz
        Nx[1: n - 1, m - 1] = (
                                      2 * (z[1: n - 1, m - 2] - z[1: n - 1, m - 1]) + z[0: n - 2, m - 2] - z[0: n - 2,
                                                                                                           m - 1]
                              ) / csz
        # (Zd - Zu + Zdl - Zl) / csz
        Ny[1: n - 1, m - 1] = (
                                      z[0: n - 2, m - 1] - z[2:n, m - 1] + z[0: n - 2, m - 2] - z[1: n - 1, m - 2]
                              ) / csz
        Nz[1: n - 1, m - 1] = 3
        # filling the first row of the matrix
        # (Zl - Zr + Zu - Zur) / csz
        Nx[0, 1: m - 1] = (z[0, 0: m - 2] - z[0, 2:m] + z[1, 1: m - 1] - z[1, 2:m]) / csz
        # (-2*(Zu - Zp) + Zr - Zur) / csz
        Ny[0, 1: m - 1] = (-2 * (z[1, 1: m - 1] - z[0, 1: m - 1]) + z[0, 2:m] - z[1, 2:m]) / csz
        Nz[0, 1: m - 1] = 3
        # filling the last row of the matrix
        # (Zl - Zr + Zdl - Zd) / csz
        Nx[n - 1, 1: m - 1] = (
                                      z[n - 1, 0: m - 2] - z[n - 1, 2:m] + z[n - 2, 0: m - 2] - z[n - 2, 1: m - 1]
                              ) / csz
        # (2*(Zd - Zp) + Zdl - Zl) / csz
        Ny[n - 1, 1: m - 1] = (
                                      2 * (z[n - 2, 1: m - 1] - z[n - 1, 1: m - 1]) + z[n - 2, 0: m - 2] - z[n - 1,
                                                                                                           0: m - 2]
                              ) / csz
        Nz[n - 1, 1: m - 1] = 3
        # filling the corners of the matrix
        Nx[0, 0] = (z[1, 0] - z[1, 1] - (z[0, 1] - z[0, 0])) / csz
        Ny[0, 0] = (z[0, 1] - z[1, 1] - (z[1, 0] - z[0, 0])) / csz
        Nz[0, 0] = 2
        Nx[n - 1, 0] = -(z[n - 1, 1] - z[n - 1, 0]) / csz
        Ny[n - 1, 0] = (z[n - 2, 0] - z[n - 1, 0]) / csz
        Nz[n - 1, 0] = 1
        Nx[0, m - 1] = (z[0, m - 2] - z[0, m - 1]) / csz
        Ny[0, m - 1] = -(z[1, m - 1] - z[0, m - 1]) / csz
        Nz[0, m - 1] = 1
        Nx[n - 1, m - 1] = (z[n - 1, m - 2] - z[n - 1, m - 1] + z[n - 2, m - 2] - z[n - 2, m - 1]) / csz
        Ny[n - 1, m - 1] = (z[n - 2, m - 1] - z[n - 1, m - 1] + z[n - 2, m - 2] - z[n - 1, m - 2]) / csz
        Nz[n - 1, m - 1] = 2

    if num == 8:
        # filling the inside of the matrix
        # normal calculation with 8 triangles
        # (2*(Zl - Zr) + Zul - Zur + Zdl - Zdr) / csz
        Nx[1: n - 1, 1: m - 1] = (
                                         2 * (z[1: n - 1, 0: m - 2] - z[1: n - 1, 2:m])
                                         + z[2:n, 0: m - 2]
                                         - z[2:n, 2:m]
                                         + z[0: n - 2, 0: m - 2]
                                         - z[0: n - 2, 2:m]
                                 ) / csz
        # (2*(Zd - Zu) - Zul - Zur + Zdl + Zdr) / csz
        Ny[1: n - 1, 1: m - 1] = (
                                         2 * (z[0: n - 2, 1: m - 1] - z[2:n, 1: m - 1])
                                         - z[2:n, 0: m - 2]
                                         - z[2:n, 2:m]
                                         + z[0: n - 2, 0: m - 2]
                                         + z[0: n - 2, 2:m]
                                 ) / csz
        Nz = 8 * Nz
        # filling the first col of the matrix
        # (- 2*(Zr - Zp) + Zu - Zur + Zd - Zdr) / csz
        Nx[1: n - 1, 0] = (
                                  -2 * (z[1: n - 1, 1] - z[1: n - 1, 0])
                                  + z[2:n, 0]
                                  - z[2:n, 1]
                                  + z[0: n - 2, 0]
                                  - z[0: n - 2, 1]
                          ) / csz
        # (Zd - Zu + Zdr - Zur) / csz
        Ny[1: n - 1, 0] = (z[0: n - 2, 0] - z[2:n, 0] + z[0: n - 2, 1] - z[2:n, 1]) / csz
        Nz[1: n - 1, 0] = 4
        # filling the last col of the matrix
        # (2*(Zl - Zp) + Zdl - Zd + Zul - Zu) / csz
        Nx[1: n - 1, m - 1] = (
                                      2 * (z[1: n - 1, m - 2] - z[1: n - 1, m - 1])
                                      + z[0: n - 2, m - 2]
                                      - z[0: n - 2, m - 1]
                                      + z[2:n, m - 2]
                                      - z[2:n, m - 1]
                              ) / csz
        # (Zd - Zu + Zdl - Zul) / csz
        Ny[1: n - 1, m - 1] = (
                                      z[0: n - 2, m - 1] - z[2:n, m - 1] + z[0: n - 2, m - 2] - z[2:n, m - 2]
                              ) / csz
        Nz[1: n - 1, m - 1] = 4
        # filling the first row of the matrix
        # (Zl - Zr + Zul - Zur) / csz
        Nx[0, 1: m - 1] = (z[0, 0: m - 2] - z[0, 2:m] + z[1, 0: m - 2] - z[1, 2:m]) / csz
        # (-2*(Zu - Zp) + Zr - Zur + Zl - Zul) / csz
        Ny[0, 1: m - 1] = (
                                  -2 * (z[1, 1: m - 1] - z[0, 1: m - 1])
                                  + z[0, 2:m]
                                  - z[1, 2:m]
                                  + z[0, 0: m - 2]
                                  - z[1, 0: m - 2]
                          ) / csz
        Nz[0, 1: m - 1] = 4
        # filling the last row of the matrix
        # (Zl - Zr + Zdl - Zdr) / csz
        Nx[n - 1, 1: m - 1] = (
                                      z[n - 1, 0: m - 2] - z[n - 1, 2:m] + z[n - 2, 0: m - 2] - z[n - 2, 2:m]
                              ) / csz
        # (2*(Zd - Zp) + Zdl - Zl + Zdr - Zr) / csz
        Ny[n - 1, 1: m - 1] = (
                                      2 * (z[n - 2, 1: m - 1] - z[n - 1, 1: m - 1])
                                      + z[n - 2, 0: m - 2]
                                      - z[n - 1, 0: m - 2]
                                      + z[n - 2, 2:m]
                                      - z[n - 1, 2:m]
                              ) / csz
        Nz[n - 1, 1: m - 1] = 4
        # filling the corners of the matrix
        Nx[0, 0] = (z[1, 0] - z[1, 1] - (z[0, 1] - z[0, 0])) / csz
        Ny[0, 0] = (z[0, 1] - z[1, 1] - (z[1, 0] - z[0, 0])) / csz
        Nz[0, 0] = 2
        Nx[n - 1, 0] = (-(z[n - 1, 1] - z[n - 1, 0]) + z[n - 2, 0] - z[n - 2, 1]) / csz
        Ny[n - 1, 0] = (z[n - 2, 1] - z[n - 1, 1] + z[n - 2, 0] - z[n - 1, 0]) / csz
        Nz[n - 1, 0] = 2
        Nx[0, m - 1] = (z[1, m - 2] - z[1, m - 1] + z[0, m - 2] - z[0, m - 1]) / csz
        Ny[0, m - 1] = (-(z[1, m - 1] - z[0, m - 1]) + z[0, m - 2] - z[1, m - 2]) / csz
        Nz[0, m - 1] = 2
        Nx[n - 1, m - 1] = (z[n - 1, m - 2] - z[n - 1, m - 1] + z[n - 2, m - 2] - z[n - 2, m - 1]) / csz
        Ny[n - 1, m - 1] = (z[n - 2, m - 1] - z[n - 1, m - 1] + z[n - 2, m - 2] - z[n - 1, m - 2]) / csz
        Nz[n - 1, m - 1] = 2

    if num == 1:
        # using the simple cross product
        z1 = np.append(z, z[:, -2].reshape(n, 1), axis=1)
        n1, m1 = np.shape(z1)
        z2 = np.append(z1, z1[-2, :].reshape(1, m1), axis=0)
        n2, m2 = np.shape(z2)

        Nx = (
                -((z2[0: n2 - 1, 1:m2] - z2[1:n2, 0: m2 - 1]) + (z2[1:n2, 1:m2] - z2[0: n2 - 1, 0: m2 - 1]))
                * csz
        )
        Ny = (
                -((z2[1:n2, 1:m2] - z2[0: n2 - 1, 0: m2 - 1]) - (z2[0: n2 - 1, 1:m2] - z2[1:n2, 0: m2 - 1]))
                * csz
        )
        Nz = 2 * Nz * csz * csz

        # Nx = - (z2[0:n2-1, 1:m2] - z2[0:n2-1, 0:m2-1]) / csz
        # Ny = - (z2[1:n2, 0:m2-1] - z2[0:n2-1, 0:m2-1]) / csz
        Ny[n - 1, 0: m - 1] = -Ny[n - 1, 0: m - 1]
        Nx[0: n - 1, m - 1] = -Nx[0: n - 1, m - 1]
        Ny[n - 1, m - 1] = -Ny[n - 1, m - 1]
        Nx[n - 1, m - 1] = -Nx[n - 1, m - 1]
        # TODO, Try to replicate samosAT notmal computation
        # if method num=1 is used, the normals are computed at com1DFA (original) cell center
        # this corresponds to our cell vertex
        # Create com1DFA (original) vertex grid
        x = np.linspace(-csz / 2.0, (ncols - 1) * csz - csz / 2.0, ncols)
        y = np.linspace(-csz / 2.0, (nrows - 1) * csz - csz / 2.0, nrows)
        X, Y = np.meshgrid(x, y)
        # interpolate the normal from com1DFA (original) center to his vertex
        # this means from our vertex to our centers
        Nx, Ny, NzCenter = getNormalArray(X, Y, Nx, Ny, Nz, csz)
        # this is for tracking mesh cell with actual data
        NzCenter = np.where(np.isnan(Nx), Nz, NzCenter)
        Nz = NzCenter

    # if no normal available, put 0 for Nx and Ny and 1 for Nz
    dem["Nx"] = np.where(np.isnan(Nx), 0.0, 0.5 * Nx)
    dem["Ny"] = np.where(np.isnan(Ny), 0.0, 0.5 * Ny)
    dem["Nz"] = 0.5 * Nz
    # build no data mask (used to find out of dem particles)
    outOfDEM = np.where(np.isnan(dem["rasterData"]), 1, 0).astype(bool).flatten()
    dem["outOfDEM"] = outOfDEM
    return dem


def getNormalArray(x, y, Nx, Ny, Nz, csz):
    """Interpolate vector field from grid to unstructured points

    Originally created to get the normal vector at location (x,y) given the
    normal vector field on the grid. Grid has its origin in (0,0).
    Can be used to interpolate any vector field.
    Interpolation using a bilinear interpolation

    Parameters
    ----------
        x: numpy array
            location in the x location of desired interpolation
        y: numpy array
            location in the y location of desired interpolation
        Nx: 2D numpy array
            x component of the vector field at the grid nodes
        Ny: 2D numpy array
            y component of the vector field at the grid nodes
        Nz: 2D numpy array
            z component of the vector field at the grid nodes
        csz: float
            cellsize of the grid

    Returns
    -------
        nx: numpy array
            x component of the interpolated vector field at position (x, y)
        ny: numpy array
            y component of the interpolated vector field at position (x, y)
        nz: numpy array
            z component of the interpolated vector field at position (x, y)
    """
    nrow, ncol = np.shape(Nx)
    # by default bilinear interpolation of the Nx, Ny, Nz of the grid
    nx, _ = projectOnGrid(x, y, Nx, csz=csz)
    ny, _ = projectOnGrid(x, y, Ny, csz=csz)
    nz, _ = projectOnGrid(x, y, Nz, csz=csz)
    return nx, ny, nz


def cellsAffectedLine(header, pointsXY, typePoints):
    """find which cells of data area affected by points

    Parameters
    -----------
    header: dict
        info on array llcenter, nrows, ncols, cellsize
    data: numpy nd array
        data array
    points: dict
        dictionary with x, y coordinates
    typePoints: str
        type of coordinates in pointsXY, options: linestring, point
    """

    xllc = header["xllcenter"]
    yllc = header["yllcenter"]
    cellSize = header["cellsize"]
    x = np.linspace(xllc, xllc + (header["ncols"] - 1) * cellSize, header["ncols"])
    y = np.linspace(yllc, yllc + (header["nrows"] - 1) * cellSize, header["nrows"])
    xx, yy = np.meshgrid(x, y)

    linecoords = np.zeros((len(pointsXY["x"]), 2))
    linecoords[:, 0] = pointsXY["x"]
    linecoords[:, 1] = pointsXY["y"]

    if typePoints.lower() == "linestring":
        line = LineString(linecoords)
    elif typePoints.lower() == "point" and len(pointsXY["x"]) == 1:
        line = Point(linecoords)
    elif typePoints.lower() == "point" and len(pointsXY["x"]) < 1:
        line = MultiPoint(linecoords)

    mask = np.zeros((header["nrows"], header["ncols"]))
    for ind, m in enumerate(xx[:, 0]):
        for ind2, k in enumerate(xx[0, :]):
            if distance(line, Point([xx[ind, ind2], yy[ind, ind2]])) <= np.sqrt(cellSize**2 + cellSize**2):
                mask[ind, ind2] = 1.0

    return mask, xx, yy
