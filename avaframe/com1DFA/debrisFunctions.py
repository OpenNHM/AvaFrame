"""
Functions that are used specifically for modeling debris flows (within DebrisFrame).
"""

# Load modules
import logging
import numpy as np
import copy

import avaframe.com1DFA.DFAfunctionsCython as DFAfunC
import avaframe.com1DFA.particleTools as particleTools
import avaframe.in3Utils.geoTrans as geoTrans
import avaframe.com1DFA.com1DFA as com1DFA
import avaframe.in2Trans.shpConversion as shpConv
from avaframe.in1Data import getInput as gI
import avaframe.in3Utils.cfgUtils as cfgUtils

# create local logger
# change log level in calling module to DEBUG to see log messages
log = logging.getLogger(__name__)


def initializeTimeDepRelease(cfg, inputSimLines, particles, fields, dem, zPartArray0, t, atol=1e-05):
    """
    Update particles with "new" particles initialised by a time dependent release.

    Parameters
    ---------
    cfg: configparser
        configuration settings
    inputSimLines : dict
        dictionary with input data dictionaries (releaseLine,...)
    particles : dict
        particles dictionary at timestep t that are in the flow already
    fields: dict
        fields dictionary at timestep t
    dem: dict
        dictionary with info on DEM data
    zPartArray0: dict
        dictionary containing z - value of particles at timestep 0
    t: float
        timestep of iteration
    atol: float
        look for matching time steps with atol tolerance - default is atol=1.e-5

    Returns
    ---------
    particles: dict
        particles dictionary at t including the new released particles
    fields: dict
        updated fields dictionary at t including the new released particles
    zPartArray0: dict
        dictionary containing z - value of particles at timestep 0
    """
    timeDepRelValues = inputSimLines["releaseLine"]["timeDepRelValues"]
    if np.isclose(t, timeDepRelValues["timeStep"], atol=atol, rtol=0).any():
        iTup = np.where(np.isclose(t, timeDepRelValues["timeStep"], atol=atol, rtol=0))
        # iTup is a tuple containing an array with one value in the first position, so we can extract the index:
        i = iTup[0]

        # similar workflow to secondary release!
        particles, zPartArray0 = addReleaseParticles(
            cfg,
            particles,
            inputSimLines,
            timeDepRelValues,
            dem,
            zPartArray0,
            timeDepRelIndex=i,
        )
        particles = DFAfunC.getNeighborsC(particles, dem)

        # update fields (compute grid values)
        if fields["computeTA"]:
            particles = DFAfunC.computeTrajectoryAngleC(particles, zPartArray0)
        particles, fields = DFAfunC.updateFieldsC(cfg["GENERAL"], particles, dem, fields)

    return particles, fields, zPartArray0


def addReleaseParticles(
        cfg, particles, inputSimLines, timeDepRelValues, dem, zPartArray0, timeDepRelIndex
):
    """
    add new particles initialized by a time dependent release to particles that are in the flow already

    Parameters
    ---------
    cfg: configparser object
        configuration settings
    particles : dict
        particles dictionary at t that are in the flow already
    inputSimLines : dict
        dictionary with input data dictionaries (releaseLine,...)
    timeDepRelValues: dict
        time dependent release values
    dem: dict
        dictionary with info on DEM data
    zPartArray0: numpy array
        z - value of particles at timestep 0

    Returns
    ---------
    particles: dict
        particles dictionary at t including the current released particles
    zPartArray0: dict
        dictionary containing z - value of particles at timestep 0
    """
    thickness = timeDepRelValues["thickness"][timeDepRelIndex]
    if "velocity" in timeDepRelValues:
        velocityMag = timeDepRelValues["velocity"][timeDepRelIndex]

    relLine = copy.deepcopy(inputSimLines["releaseLine"])
    relLine["header"] = dem["originalHeader"].copy()
    if relLine["initializedFrom"] == "csvfile":
        relLine["rasterData"] = gI.timeDepRelCoordsToRaster(
            relLine["timeDepRelValues"], timeDepRelIndex, dem["originalHeader"], parameter="thickness"
        )
        relThField = relLine["rasterData"]
    else:
        relThField = ""
        relLine = geoTrans.prepareArea(
            relLine,
            dem,
            np.sqrt(2),
            thList=[thickness] * len(relLine["Name"]),
            combine=True,
            checkOverlap=False,
        )

        # check if already existing particles are within the release polygon
        # it's possible that there are still a few particles in the polygon with low velocities
        # TODO: could think of a threshold of number of particles that are still allowed in the polygons?
        mask = geoTrans.getParticlesInPolygon(particles, relLine, cfg["GENERAL"].getfloat("thresholdPointInRel"))
        if np.sum(mask) > 0:
            message = (
                    "Already existing particles are within the release polygon, which can cause numerical instabilities (at timestep: %02f s)"
                    % (particles["t"] + particles["dt"])
            )
            log.error(message)
            raise ValueError(message)

    particlesRelease = com1DFA.initializeParticles(
        cfg["GENERAL"],
        relLine,
        dem,
        relThField=relThField,
        timestep=particles["t"] + particles["dt"]
    )

    if "velocity" in timeDepRelValues and "x" not in timeDepRelValues:
        particlesRelease = DFAfunC.updateInitialVelocity(cfg["GENERAL"], particlesRelease, dem, velocityMag)



    elif "velocityX" in timeDepRelValues and "x" in timeDepRelValues:
        for uComp, timedepParameter in zip(["ux", "uy", "uz"], ["velocityX", "velocityY", "velocityZ"]):
            raster = gI.timeDepRelCoordsToRaster(relLine["timeDepRelValues"], timeDepRelIndex,
                                                 dem["originalHeader"], parameter=timedepParameter
                                                 )
            rasterDict = {"header": dem["header"], "rasterData": raster}
            particlesRelease, _ = geoTrans.projectOnRaster(rasterDict,
                                                           particlesRelease, outData=uComp)

        particlesRelease["uMag"] = np.sqrt(
            particlesRelease["ux"] ** 2 + particlesRelease["uy"] ** 2 + particlesRelease["uz"] ** 2)

    particles = particleTools.mergeParticleDict(particles, particlesRelease)
    # save initial z position for travel angle computation
    zPartArray0 = np.append(zPartArray0, copy.deepcopy(particlesRelease["z"]))
    return particles, zPartArray0


def defineReleaseLineFromCoordinates(relFile, timeDepRelValues, demHeader):
    """
    define the release line and its thickness raster from coordinates read from time dependent release values

    Parameters
    ----------
    relFile: pathlib.Path
        directory to release file (csv file)
    timeDepRelValues: dict
        time dependent release values
    demHeader: dict
        header of DEM

    Returns
    -------
    releaseLine: dict
        dictionary for release line containing thickness raster data
    relThFieldData: numpy array
        release thickness raster data
    """
    releaseLine = {
        "file": relFile,
        "Name": [relFile.stem],
        "initializedFrom": "csvfile"
    }
    initialIndex = np.where(timeDepRelValues["timeStep"] == 0)[0]
    releaseLine["rasterData"] = gI.timeDepRelCoordsToRaster(timeDepRelValues, initialIndex, demHeader,
                                                            parameter="thickness")
    relThFieldData = releaseLine["rasterData"]
    # TODO: define thickness for output report, now mean of thickness values
    releaseLine["thickness"] = np.nanmean(np.where(relThFieldData == 0, np.nan, relThFieldData))

    return releaseLine, relThFieldData


def prepareTimeDepRelLine(releaseLine, cfg):
    """
    read time dependent release values and return them as a dictionary containing:
    - timestep
    - thickness
    - velocity
    additionally, the values are saved into the configurationFiles folder

    Parameters
    ----------
    releaseLine: dict
        contains information of release line
    cfg: configparser object
        configuration for simType
    demOri : dict
        dictionary with dem info (header original origin), raster data correct mesh cell size

    Returns
    -------
    releaseLine: dict
        added time dependent release values: timeStep, thickness, velocity
    """

    try:
        releaseLine["values"], timeDepRelValuesDF = gI.getTimeDepRelCsv(cfg["INPUT"]["timeDepRelCsv"])
        releaseLine["thicknessSource"] = ["csv file"]
    except:
        message = "Provide a valid csv file containing time dependent release values"
        log.error(message)
        raise FileNotFoundError(message)
    # write the time dependent values into configurationFiles folder
    cfgUtils.writeReleaseCsvFile(cfg, timeDepRelValuesDF)

    return releaseLine


def checkTravelledDistance(cfgGen, timeDepRelValues, timeDepRelCsv):
    """
    not used at the moment (related to timeStepDistance in the configuration file)!
    check if time steps of time dependent release are not too close so that the particle density cannot become too high
    check that particles moved out of release area before new particles are initialized
    time between release time steps
    first timestep is skipped since this is always ok.

    Parameters
    -----------
    cfgGen: configparser
        configuration settings, section "GENERAL"
    timeDepRelValues: dict
        contains time dependent release values: timestep, thickness, velocity
    timeDepRelCsv: str
        directory to csv table containing time dependent release values
    """
    timeStepUnique = np.unique(timeDepRelValues["timeStep"])
    if timeStepUnique.ndim > 0:
        relDT = np.append(timeDepRelValues["timeStep"], 0) - np.append(0, timeDepRelValues["timeStep"])
        vel = np.where(np.array(timeDepRelValues["velocity"]) > 0, np.array(timeDepRelValues["velocity"]), 1)
        distance = vel[:-1] * relDT[1:-1]

        if np.any(distance < cfgGen.getfloat("timeStepDistance")):
            message = "Please select timesteps with greater spacing in %s." % (timeDepRelCsv)
            # TODO: error or warning?
            log.error(message)
            raise ValueError(message)
