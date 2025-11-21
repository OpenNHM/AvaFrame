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

# create local logger
# change log level in calling module to DEBUG to see log messages
log = logging.getLogger(__name__)


def releaseHydrograph(cfg, inputSimLines, particles, fields, dem, zPartArray0, t, atol=1e-05):
    """
    Update particles with "new" particles initialised by a hydrograph.

    Parameters
    ---------
    cfg: configparser
        configuration settings
    inputSimLines : dict
        dictionary with input data dictionaries (releaseLine, hydrographAreaLine,...)
    particles : dict
        particles dictionary at t that are in the flow already
    fields: dict
        fields dictionary at t
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
        particles dictionary at t including the hydrograph particles
    fields: dict
        updated fields dictionary at t including the hydrograph particles
    zPartArray0: dict
        dictionary containing z - value of particles at timestep 0
    """
    hydrValues = inputSimLines["hydrographAreaLine"]["values"]
    if np.isclose(t, hydrValues["timeStep"], atol=atol, rtol=0).any():
        iTup = np.where(np.isclose(t, hydrValues["timeStep"], atol=atol, rtol=0))
        # iTup is a tuple containing an array with one value in the first position, so we can extract the index:
        i = iTup[0].item()
        log.info(
            "add hydrograph at timestep: %f s with thickness %s m and velocity %s m/s"
            % (t, hydrValues["thickness"][i], hydrValues["velocity"][i])
        )
        # similar workflow to secondary release!
        particles, zPartArray0 = addHydrographParticles(
            cfg,
            particles,
            inputSimLines,
            hydrValues["thickness"][i],
            hydrValues["velocity"][i],
            dem,
            zPartArray0,
        )
        particles = DFAfunC.getNeighborsC(particles, dem)
        # update fields (compute grid values)
        if fields["computeTA"]:
            particles = DFAfunC.computeTrajectoryAngleC(particles, zPartArray0)
        particles, fields = DFAfunC.updateFieldsC(cfg["GENERAL"], particles, dem, fields)

    return particles, fields, zPartArray0


def addHydrographParticles(cfg, particles, inputSimLines, thickness, velocityMag, dem, zPartArray0):
    """
    add new particles initialized by a hydrograph to particles that are in the flow already

    Parameters
    ---------
    cfg: configparser object
        configuration settings
    particles : dict
        particles dictionary at t that are in the flow already
    inputSimLines : dict
        dictionary with input data dictionaries (releaseLine, hydrographAreaLine,...)
    thickness: float
        thickness of incoming hydrograph
    velocityMag: float
        velocity of incoming hydrograph
    dem: dict
        dictionary with info on DEM data
    zPartArray0: numpy array
        z - value of particles at timestep 0

    Returns
    ---------
    particles: dict
        particles dictionary at t including the hydrograph particles
    zPartArray0: dict
        dictionary containing z - value of particles at timestep 0
    """
    hydrLine = inputSimLines["hydrographAreaLine"]
    hydrLine["header"] = dem["originalHeader"].copy()
    hydrLine = geoTrans.prepareArea(
        hydrLine,
        dem,
        np.sqrt(2),
        thList=[thickness],
        combine=True,
        checkOverlap=False,
    )

    # check if already existing particles are within the hydrograph polygon
    # it's possible that there are still a few particles in the polygon with low velocities
    # TODO: could think of a threshold of number of particles that are still allowed in the polygons?
    mask = geoTrans.getParticlesInPolygon(
        particles, hydrLine, cfg["GENERAL"].getfloat("thresholdPointInHydr")
    )
    if np.sum(mask) > 0:
        # if there is at least one particle within the polygon (including the buffer):
        message = (
            "Already existing particles are within the hydrograph polygon, which can cause numerical instabilities (at timestep: %02f s)"
            % (particles["t"] + particles["dt"])
        )
        # timestep in particles is not updated yet
        log.error(message)
        raise ValueError(message)

    particlesHydrograph = com1DFA.initializeParticles(
        cfg["GENERAL"],
        hydrLine,
        dem,
    )
    particlesHydrograph = DFAfunC.updateInitialVelocity(
        cfg["GENERAL"], particlesHydrograph, dem, velocityMag
    )

    particles = particleTools.mergeParticleDict(particles, particlesHydrograph)
    # save initial z position for travel angle computation
    zPartArray0 = np.append(zPartArray0, copy.deepcopy(particlesHydrograph["z"]))
    return particles, zPartArray0


def checkHydrograph(hydrographValues, hydrCsv, cfgGen):
    """
    check if hydrograph satisfied the following requirements:
    - hydrograph-timesteps are unique
    - provided release-thickness is larger than zero
    - the hydrograph-timesteps are not too close (that the particle density becomes too high)

    Parameters
    -----------
    hydrCsv: str
        directory to csv table containing hydrograph values
    hydrographValues: dict
        contains hydrograph values: timestep, thickness, velocity
    cfgGen: configparser Object
        GENERAL section of the configuration file
    """
    # check if timesteps are unique
    timeStepUnique = np.unique(hydrographValues["timeStep"])
    if timeStepUnique.ndim == 0:
        if timeStepUnique != hydrographValues["timeStep"]:
            message = "The provided hydrograph time steps in %s are not unique" % (hydrCsv)
            log.error(message)
            raise ValueError(message)
    elif len(timeStepUnique) != len(hydrographValues["timeStep"]):
        message = "The provided hydrograph timesteps in %s are not unique" % (hydrCsv)
        log.error(message)
        raise ValueError(message)

    # check that hydrograph thickness > 0
    for th in hydrographValues["thickness"]:
        if th <= 0:
            message = "For every release time step a thickness > 0 needs to be provided in %s" % (hydrCsv)
            log.error(message)
            raise ValueError(message)

    # check if a timestep = 0 is provided, when no REL area is used
    if cfgGen.getboolean("hydrograph") and cfgGen.getboolean("noRelArea"):
        if 0 not in timeStepUnique:
            message = (
                "If no release area is released, a thickness needs to be provided for  time step 0 s in %s"
                % (hydrCsv)
            )
            log.error(message)
            raise ValueError(message)


def preparehydrographAreaLine(inputSimFiles, demOri, cfg):
    """
    read hydrograph polygon and values

    Parameters
    ----------
    inputSimFiles : dict
        dictionary containing
        - hydrographFile: str, path to hydrograph polygon file
        - hydrographCsv: str, path to hydrograph values (csv-)file
    cfg: configparser object
        configuration for simType
    demOri : dict
        dictionary with dem info (header original origin), raster data correct mesh cell size

    Returns
    -------
    hydrLine: dict
        contains hydrograph outline and values, among other things:
        - x, y, z
        - values: timeStep, thickness, velocity
    """
    try:
        hydrFile = inputSimFiles["hydrographFile"][0]
        hydrLine = shpConv.readLine(hydrFile, "", demOri)
        hydrLine["fileName"] = hydrFile
        hydrLine["type"] = "Hydrograph"
        gI.checkForMultiplePartsShpArea(
            cfg["GENERAL"]["avalancheDir"], hydrLine, "com1DFA", type="hydrograph"
        )
    except:
        message = "No hydrograph shp file found"
        log.error(message)
        raise FileNotFoundError(message)

    try:
        hydrLine["values"] = gI.getHydrographCsv(inputSimFiles["hydrographCsv"])
        hydrLine["thicknessSource"] = ["csv file"]
    except:
        message = "No hydrograph csv file found"
        log.error(message)
        raise FileNotFoundError(message)

    checkHydrograph(hydrLine["values"], inputSimFiles["hydrographCsv"], cfg["GENERAL"])

    return hydrLine


def checkTravelledDistance(cfgGen, hydrographValues, hydrCsv):
    """
    not used at the moment (related to timeStepDistance in the configuration file)!
    check if time steps of hydrograph are not to close that the particle density becomes too high
    check that particles moved out of hydrograph area before new particles are initialized
    time between hydrograph time steps
    first timestep is skipped since this is always ok.

    Parameters
    -----------
    hydrCsv: str
        directory to csv table containing hydrograph values
    cfgGen: configparser
        configuration settings, section "GENERAL"
    hydrographValues: dict
        contains hydrograph values: timestep, thickness, velocity
    """
    timeStepUnique = np.unique(hydrographValues["timeStep"])
    if timeStepUnique.ndim > 0:
        hydrDT = np.append(hydrographValues["timeStep"], 0) - np.append(0, hydrographValues["timeStep"])
        vel = np.where(np.array(hydrographValues["velocity"]) > 0, np.array(hydrographValues["velocity"]), 1)
        distance = vel[:-1] * hydrDT[1:-1]

        if np.any(distance < cfgGen.getfloat("timeStepDistance")):
            message = "Please select timesteps with greater spacing in %s." % (hydrCsv)
            # TODO: error or warning?
            log.error(message)
            raise ValueError(message)
