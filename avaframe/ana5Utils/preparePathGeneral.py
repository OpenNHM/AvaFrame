"""
generate thalweg from x and y coordinates (including extension to top and bottom and resampling)
"""

import numpy as np
import logging
import copy
# local imports
import avaframe.in3Utils.geoTrans as gT
from avaframe.ana5Utils import DFAPathGeneration
# create local logger
log = logging.getLogger(__name__)


def preparePathGeneralMain(profile, cfgDFAPath, dem):
    """
     prepare thalweg from x and y coordinates:
     1. read z coordinates from DEM and compute horizontally projected distance
     2. extend path to bottom and top
     3. resample path points

     Parameters
     -------------
     profile: dict
        contains x and y coordinates of thalweg location
    cfgDFAPath: configparser object
        configuration for DFA path generation
    dem: dict
        dictionary with header and raster data of elevation model

    Returns
    -------------
    profileAveraged: dict
        s and z coordinates are added (x and y original) to input profile
    profileExtended: dict
        x, y, s, z of extended and resampled path
    """
    # get profile with normalized x and y coordinates and z and s values
    profileAveraged = updateSZProfile(profile, dem)
    profileExtended = copy.deepcopy(profileAveraged)

    # skip profile that only contains one or two points
    if len(profileAveraged["x"]) <= 2:
        profileExtended["indStartMassAverage"] = 0
        profileExtended["indEndMassAverage"] = max(len(profileExtended["x"]) - 1, 0)
        return profileAveraged, profileExtended

    # if extTopOption == 2, particlesIni are not used!!
    profileExtended = pathExtension(profileExtended, dem, cfgDFAPath)
    profileExtended = updateSZProfile(profileExtended, dem)
    # resample profile/ path and save in an extra dictionary
    profileExtended = DFAPathGeneration.resamplePath(cfgDFAPath["PATH"], dem, profileExtended)

    # add input parameters to extended profile if they exist
    for inputPara in ["alpha", "exponent", "zDeltaMax"]:
        if inputPara in profile.keys():
            profileExtended[inputPara] = profile[inputPara]

    return profileAveraged, profileExtended


def pathExtension(profile, demDict, cfgPathGen):
    """
    extend thalweg to top and bottom of path

    Parameters
    ------------
    profile: dict
        thalweg data
    demDict: dict
        DEM data
    cfgPathGen: confiparser object
        configuration setup for DFA Path generation

    Returns
    -------------
    profile: dict
        thalweg data that are extended to top and bottom
    """

    profile["indStartMassAverage"] = 1
    # do not use the last two points because the last points are weird sometimes
    profile["indEndMassAverage"] = np.size(profile["x"]) - 2

    if cfgPathGen["PATH"].getint("extTopOption") != 2:
        # TODO: if we provide particlesIni, the other options would also work.
        message = "Up to now only top-extension option 2 works!"
        log.error(message)
        raise ValueError(message)

    profile = DFAPathGeneration.extendProfileTop(
        cfgPathGen["PATH"].getint("extTopOption"),
        {},
        profile,
        dem=demDict,
        cfg=cfgPathGen["PATH"],
        considerLLC=True,
    )

    # extend the bottom
    profile = DFAPathGeneration.extendProfileBottom(cfgPathGen["PATH"], demDict, profile, considerLLC=True)

    return profile


def updateSZProfile(profile, dem):
    """
    for given coordinates (of the talweg) read z values
    from DEM and compute distance between coordinates

    Parameters
    ------------
    profile: dict
        contains at least x and y coordinates
    dem: dict
        contains dem data

    Returns
    -----------
    profile: dict
        profile with added a and z values
    """
    x = profile["x"]
    y = profile["y"]

    demHeader = dem["header"]

    z, _ = gT.projectOnGrid(
        x,
        y,
        dem["rasterData"],
        csz=demHeader["cellsize"],
        xllc=demHeader["xllcenter"],
        yllc=demHeader["yllcenter"],
    )
    s = np.append([0], gT.computeLengthOfLine2D(x, y))
    profile["z"] = z
    profile["s"] = s

    return profile
