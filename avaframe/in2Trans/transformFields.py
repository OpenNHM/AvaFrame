"""
ASCII file reader and handler

"""

import numpy as np
import logging
import pathlib

# local imports
import avaframe.com1DFA.DFAtools as DFAtls
from avaframe.in3Utils import geoTrans as gT
import avaframe.in2Trans.rasterUtils as IOf

# create local logger
log = logging.getLogger(__name__)


def convertDepthThickness(inputDict, demDictOrig, typeOfInput="depth"):
    """convert depthField to thicknessField or the other way around depending on type, using a DEM to compute the slope
    angle required for transformation
    also writes field to a new directory called transformed where depthFile/thicknessFile is located

    Parameters
    -----------
    demDictOrig: dict
        dictionary with dem header and rasterData (numpy nd array of z values)
    inputDict: dict
        dictionary with depthField/thicknessField header and rasterData (numpy nd array of depth values)
    typeOfInput: str
        provided inputDict type: thickness or depth

    Returns
    --------
    outputDict: dict
        dictionary with header and thickness/depth field numpy array as rasterData key

    """
    # get normal vector of the grid mesh
    demDict = {"header": demDictOrig["header"].copy(), "rasterData": demDictOrig["rasterData"]}
    demDict = gT.getNormalMesh(demDict)
    _, _, NzNormed = DFAtls.normalize(demDict["Nx"], demDict["Ny"], demDict["Nz"])

    # if resType field cellSize/extent is different to DEM reproject raster on a grid of shape DEM
    inputRasterNew, demData = gT.resizeData(inputDict, demDict)
    demDict["header"]["nodata_value"] = inputDict["header"]["nodata_value"]

    # multiply depth with cos(slopeAngle)
    if typeOfInput == "depth":
        outField = inputRasterNew * NzNormed
    elif typeOfInput == "thickness":
        outField = inputRasterNew / NzNormed
    else:
        message = "Type for thickness/depth conversion is: %s - not valid" % typeOfInput
        log.error(message)
        raise AssertionError(message)

    slopeAngleField = np.rad2deg(np.arccos(NzNormed))

    # create thickness dict
    outputDict = {"header": demDict["header"], "rasterData": outField}

    return outputDict, inputRasterNew, slopeAngleField


def fetchPointValuesFromField(dataDF, xyPoints, resType, interpMethod="bilinear"):
    """derive field values at xyPoints using a interpMethod (options: nearest and bilinear)

    Parameters
    -----------
    dataDF: pandas dataFrame
        dataframe with info on simulations (including result file paths,configuration)
    xyPoints: dict
        dictionary with keys x, y and point of interest coordinates
    resType: str
        name of result type (pfv, pft, ...)
    interpMethod: str
        interpolation method to derive values of field at xyPoints

    Returns
    ---------
    dataDF: pandas dataFrame
        updated pandas dataFrame with new column pointValues_resType providing value of resType at xyPoint

    """

    for index, row in dataDF.iterrows():

        # read field
        field = IOf.readRaster(row[resType])

        value, _ = gT.projectOnRaster(
            field, xyPoints, interp=interpMethod, inData="rasterData", outData="value"
        )
        dataDF.loc[index, ("pointValues_%s" % resType)] = value["value"][0]

    return dataDF
