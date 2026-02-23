"""
generate talweg from x and y coordinates
"""

import numpy as np
import pathlib
import matplotlib.pyplot as plt
import logging
import json

from avaframe.in3Utils import cfgUtils
from avaframe.in3Utils import logUtils
from avaframe.ana5Utils import regionalThalweg2DPlot
import avaframe.in3Utils.geoTrans as gT
from avaframe.ana5Utils import DFAPathGeneration
import avaframe.ana5Utils.regionalThalwegTools as tools

log = logging.getLogger(__name__)
logName = "runRegionalThalweg2DPlot"


def xyToProfile(x, y, dem):
    """
    for given coordinates (of the talweg) read z values from DEM and compute distance between coordinates

    Parameters
    ------------
    x: np.array
        x coordinates
    y: np.array
        y coordinates
    dem: dict
        contains dem data
    """
    demHeader = dem["header"]

    z, _ = gT.projectOnGrid(
        x,
        y,
        dem["rasterData"],
        csz=demHeader["cellsize"],
        xllc=demHeader["xllcenter"],
        yllc=demHeader["yllcenter"],
    )
    s = gT.computeLengthOfLine2D(x, y)
    profile = {"x": x, "y": y, "z": z, "s": s}

    return profile


def preparePathGeneralMain(profile, cfgDFAPath, dem):
    """ """
    # normalize x and y coordinates
    demHeader = dem["header"]
    # normalize x and y coordinates
    # x = profile["x"] - demHeader["xllcenter"]
    # y = profile["y"] - demHeader["yllcenter"]
    x = profile["x"]
    y = profile["y"]

    # get profile with normalized x and y coordinates and z and s values
    profileAveraged = xyToProfile(x, y, dem)

    profileExtended = profileAveraged.copy()
    # if extTopOption == 2, particlesIni are not used!!
    cfgDFAPath["PATH"]["extTopOption"] = "2"
    particlesIni = {}
    profileExtended = DFAPathGeneration.extendDFAPath(cfgDFAPath["PATH"], profileExtended, dem, particlesIni)

    return profileAveraged, profileExtended
