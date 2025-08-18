"""
Raster (ascii and tif) file reader and handler

"""

import logging
import rasterio
import numpy as np

# create local logger
log = logging.getLogger(__name__)


def readRaster(fname, noDataToNan=True):
    """Read raster file in .asc or .tif format.
    Returns a dict with a header and the data in it.
    Header is based on avaframe header info with llcenter info

    Parameters
    -----------

    fname: pathlib object
        path to ascii/tif file
    noDataToNan: bool
        if True convert nodata_values to nan and set nodata_value to nan

    Returns
    --------
    data: dict
        -header: class
            information that is stored in header (ncols, nrows, xllcenter, yllcenter, nodata_value, transform,
            crs)
        -rasterData : 2D numpy array
                2D numpy array of raster matrix
    """

    log.debug("Reading raster file : %s", fname)

    raster = rasterio.open(fname)
    rasterData = raster.read(1).astype(np.float64)
    header = getHeaderFromRaster(raster)
    raster.close()

    data = {}
    data["header"] = header
    if noDataToNan:
        rasterData[rasterData == header["nodata_value"]] = np.nan
        data["header"]["nodata_value"] = np.nan
    data["rasterData"] = np.flipud(rasterData)

    return data


def getHeaderFromRaster(raster):
    """convert rasterio raster info to header info

    Parameters
    ----------
    raster: rasterio raster
        read by rasterio

    Returns
    -------
    header: dict
        header info
    """
    header = {}
    header["ncols"] = raster.width
    header["nrows"] = raster.height
    header["cellsize"] = raster.transform[0]
    header["xllcenter"] = (raster.transform * (0, 0))[0] + header["cellsize"] / 2.0
    header["yllcenter"] = (raster.transform * (0, raster.height))[1] + header["cellsize"] / 2.0
    header["nodata_value"] = raster.nodata
    header["crs"] = raster.crs
    header["driver"] = raster.driver
    header["transform"] = raster.transform

    return header


def transformFromASCHeader(header):
    """convert header info to raster transform info

    Parameters
    ----------
    header: dict
        header info

    Returns
    -------
    transfrom: dict
        rasterio transform info
    """
    # rasterio requires west, north
    # rasterio.transform.from_origin(west, north, xsize, ysize)
    xllCenter = header["xllcenter"]
    yllCenter = header["yllcenter"]
    cellSize = header["cellsize"]
    nRows = header["nrows"]

    transform = rasterio.transform.from_origin(
        xllCenter - cellSize / 2.0, (yllCenter - cellSize / 2.0) + nRows * cellSize, cellSize, cellSize
    )

    return transform


def readRasterHeader(fname):
    """return a class with information from an ascii file header

    Parameters
    -----------

    fname: str or pathlib object
        path to ascii file

    Returns
    --------
    headerInfo: class
        information that is stored in header (ncols, nrows, xllcenter, yllcenter, nodata_value)
    """

    raster = rasterio.open(fname)
    header = getHeaderFromRaster(raster)
    raster.close()

    return header


def isEqualASCheader(headerA, headerB):
    """Test if two headers (A,B) are the same (except for noData Values)

    Parameters
    -----------
    headerA: class
    headerB: class

    Returns
    --------
    boolean: True if header A and B are equal (disregrad the noData field)
    """
    a = headerA
    b = headerB
    return (
        (a["ncols"] == b["ncols"])
        and (a["nrows"] == b["nrows"])
        and (a["xllcenter"] == b["xllcenter"])
        and (a["yllcenter"] == b["yllcenter"])
        and (a["cellsize"] == b["cellsize"])
    )


def writeResultToRaster(header, resultArray, outFileName, flip=False, useCompression=True):
    """Write 2D array to a raster file with header and save to location of outFileName

    Parameters
    ----------
    header : class
        class with methods that give cellsize, nrows, ncols, xllcenter
        yllcenter, nodata_value, driver, transfrom, crs
    resultArray : 2D numpy array
        2D numpy array of values that shall be written to file
    outFileName : str
        path incl. name of file to be written
    flip: boolean
        if True, flip the rows of the resultArray when writing. AF considers the first line in a data array to be the
        southernmost one. Some formats (e.g. tif) have the northernmost line first
    useCompression: boolean
        True if compression should be used on writing tiff files (lzw)


    Returns
    -------
    outFile: path
        to file being written
    """

    driver = header["driver"]
    if driver not in ("AAIGrid", "GTiff"):
        raise ValueError(f"Unsupported driver: {driver}")

    extMap = {"AAIGrid": ".asc", "GTiff": ".tif"}
    outFile = outFileName.parent / (outFileName.name + extMap[driver])

    commonKwargs = {
        "driver": driver,
        "crs": header["crs"],
        "nodata": header["nodata_value"],
        "transform": header["transform"],
        "height": resultArray.shape[0],
        "width": resultArray.shape[1],
        "count": 1,
        "dtype": resultArray.dtype,
    }

    extraKwargs = {}
    if useCompression and driver == "GTiff":
        extraKwargs = {"compress": "lzw"}

    with rasterio.open(outFile, "w", **commonKwargs, **extraKwargs) as rasterOut:
        data = np.flipud(resultArray) if flip else resultArray
        rasterOut.write(data, 1)

    return outFile


def getRasterFileTypeFromHeader(header):
    """derive raster filetype from header
    Parameters
    ------------
    header: dict
        dictionary with header information

    Returns
    ---------
    fileType: str
        raster file type, current options are .asc, .tif

    """

    if header["driver"] == "AAIGrid":
        fileType = ".asc"
    elif header["driver"] == "GTiff":
        fileType = ".tif"
    else:
        log.error("Unsupported driver for DEM %s" % header["driver"])
        raise AssertionError("Unsupported driver for DEM %s" % header["driver"])

    return fileType
