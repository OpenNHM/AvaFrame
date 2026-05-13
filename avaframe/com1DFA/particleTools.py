"""
Tools for handling particles, splitting, merging and tracking.
"""

# Load modules
import logging
import numpy as np
import pandas as pd
import numbers
import math
import pathlib
import pickle

# Local imports
import avaframe.in3Utils.fileHandlerUtils as fU
import avaframe.com1DFA.DFAtools as DFAtls
import avaframe.in3Utils.geoTrans as geoTrans
import avaframe.com1DFA.DFAfunctionsCython as DFAfunC


# create local logger
# change log level in calling module to DEBUG to see log messages
log = logging.getLogger(__name__)


def initialiseParticlesFromFile(cfg, avaDir, releaseScenario):
    # TODO: this is for development purposes, change or remove in the future
    # If initialisation from file

    if cfg["particleFile"]:
        inDirPart = pathlib.Path(cfg["particleFile"])
    else:
        inDirPart = pathlib.Path(avaDir, "Outputs", "com1DFAOrig")

    searchDir = inDirPart / "particles"
    inDirPart = list(searchDir.glob(("*" + releaseScenario + "_" + "*" + cfg["simTypeActual"] + "*")))
    if inDirPart == []:
        messagePart = (
            "Initialise particles from file - no particles file found for releaseScenario: %s and simType: %s"
            % (releaseScenario, cfg["simTypeActual"])
        )
        log.error(messagePart)
        raise FileNotFoundError(messagePart)
    elif len(inDirPart) > 1:
        log.warning("More than one file found for Initialise particle from file: took %s" % inDirPart[0])
        inDirPart = inDirPart[0]
    else:
        inDirPart = inDirPart[0]

    log.info("Initial particle distribution read from file!! %s" % (inDirPart))
    Particles, timeStepInfo = readPartFromPickle(inDirPart)
    particles = Particles[0]
    xPartArray = particles["x"]
    hPartArray = np.ones(len(xPartArray))
    particles["nPart"] = len(xPartArray)
    particles["trajectoryLengthXY"] = np.zeros(np.shape(xPartArray))
    particles["trajectoryLengthXYZ"] = np.zeros(np.shape(xPartArray))
    particles["idFixed"] = np.zeros(np.shape(xPartArray))
    return particles, hPartArray


def placeParticles(hCell, aCell, indx, indy, csz, massPerPart, nPPK, rng, cfg, ratioArea):
    """Create particles in given cell

    Compute number of particles to create in a given cell.
    Place particles in cell according to the chosen pattern (random semirandom
    or ordered)

    Parameters
    ----------
    hCell: float
        snow thickness in cell
    aCell: float
        cell area
    indx: int
        column index of the cell
    indy: int
        row index of the cell
    csz : float
        cellsize
    massPerPart : float
        maximum mass per particle
    nPPK: int
        number of particles per kernel radius (used only if massPerParticleDeterminationMethod = MPPKR)
    cfg: configParser
        com1DFA general configParser
    ratioArea: float
        ratio between projected release area and real release area (used for the triangular initialization)
        limitations appear if there are multiple release areas feature (the ratio stands for the average of all release
        areas so it is not specific to each feature).
    Returns
    -------
    xPart : 1D numpy array
        x position of particles
    yPart : 1D numpy array
        y position of particles
    mPart : 1D numpy array
        mass of particles
    n : int
        number of particles created
    aPart : int
        particle area
    """

    rho = cfg.getfloat("rho")
    thresholdMassSplit = cfg.getfloat("thresholdMassSplit")
    initPartDistType = cfg["initPartDistType"].lower()
    massPerParticleDeterminationMethod = cfg["massPerParticleDeterminationMethod"]
    volCell = aCell * hCell
    massCell = volCell * rho
    if initPartDistType == "random":
        if massPerParticleDeterminationMethod == "MPPKR":
            # impose a number of particles within a kernel radius so impose number of particles in a cell
            nFloat = nPPK * aCell / (math.pi * csz**2)
        else:
            # number of particles needed (floating number)
            nFloat = massCell / massPerPart

        n = int(np.floor(nFloat))
        # adding 1 with a probability of the residual proba
        proba = nFloat - n
        if rng.random(1) < proba:
            n = n + 1
        n = np.maximum(n, 1)
        # make sure we do not violate the (massCell / n) < thresholdMassSplit x massPerPart rule
        if (massCell / n) / massPerPart >= thresholdMassSplit:
            n = n + 1
    elif initPartDistType == "semirandom" or initPartDistType == "uniform":
        n1 = (np.ceil(np.sqrt(massCell / massPerPart))).astype("int")
        n = n1 * n1
        d = csz / n1
        pos = np.linspace(0.0, csz - d, n1) + d / 2.0
        x, y = np.meshgrid(pos, pos)
        x = x.flatten()
        y = y.flatten()
    elif initPartDistType == "triangular":
        if massPerParticleDeterminationMethod == "MPPKR":
            # impose a number of particles within a kernel radius so impose number of particles in a cell
            # (regardless of the slope)
            nPPC = nPPK / math.pi
        else:
            # number of particles needed (floating number)
            # ToDo: this only works if the release thickness is constant in a release area!!!
            nPPC = hCell * (csz**2 / ratioArea) * rho / massPerPart
        n = int(np.floor(nPPC))
        indx = indx - 1 / 2
        indy = indy - 1 / 2
        # compute triangles properties
        Aparticle = csz**2 / n
        sTri = math.sqrt(Aparticle / (math.sqrt(3) / 2))
        hTri = sTri * math.sqrt(3) / 2
        jMin = int(indy * csz / hTri)
        if jMin * hTri < indy * csz:
            jMin = jMin + 1
    else:
        log.warning(
            "Chosen value for initial particle distribution type not available: %s uniform is used instead"
            % initPartDistType
        )

    mPart = massCell / n
    aPart = aCell / n

    # TODO make this an independent function
    #######################
    # start ###############
    if initPartDistType == "random":
        # place particles randomly in the cell
        xPart = csz * (rng.random(n) - 0.5 + indx)
        yPart = csz * (rng.random(n) - 0.5 + indy)
    elif initPartDistType == "semirandom" or initPartDistType == "uniform":
        # place particles equaly distributed
        xPart = csz * (-0.5 + indx) + x
        yPart = csz * (-0.5 + indy) + y
        if initPartDistType == "semirandom":
            # place particles equaly distributed with a small variation
            xPart = xPart + (rng.random(n) - 0.5) * d
            yPart = yPart + (rng.random(n) - 0.5) * d
    elif initPartDistType == "triangular":
        xPart = np.empty(0)
        yPart = np.empty(0)
        n = 0
        j = jMin
        iTemp = int(indx * csz / sTri)
        if iTemp * sTri < indx * csz:
            iTemp = iTemp + 1
        while j * hTri < (indy + 1) * csz:
            i = iTemp - 1 / 2 * j % 2
            while i * sTri < (indx + 1) * csz:
                if i * sTri >= indx * csz and j * hTri >= indy * csz:
                    xPart = np.append(xPart, i * sTri)
                    yPart = np.append(yPart, j * hTri)
                    n = n + 1
                i = i + 1
            j = j + 1

    return xPart, yPart, mPart, n, aPart


def removePart(particles, mask, nRemove, reasonString="", snowSlide=0):
    """remove given particles

    Parameters
    ----------
    particles : dict
        particles dictionary
    mask : 1D numpy array
        particles to keep
    nRemove : int
        number of particles removed
    reasonString: str
        reason why removing particles - for log message

    Returns
    -------
    particles : dict
        particles dictionary
    """
    if reasonString != "":
        log.debug("removed %s particles %s" % (nRemove, reasonString))
    if reasonString == "because they exited the domain":
        particles["nExitedParticles"] = particles["nExitedParticles"] + nRemove
    nPart = particles["nPart"]
    if snowSlide == 1:
        # if snowSlide is activated, we need to remove the particles as well as the bonds accordingly
        # we do this first befor nPart changes
        nBondRemove = DFAfunC.countRemovedBonds(particles, mask, nRemove)
        particles = DFAfunC.removedBonds(particles, mask, nRemove, nBondRemove)
    for key in particles:
        if key == "nPart":
            particles["nPart"] = particles["nPart"] - nRemove
        elif key == "stoppedParticles" or key == "bondStart" or key == "bondPart" or key == "bondDist":
            # bond arrays are already rebuilt by removedBonds with new-indexed data
            continue
        # for all keys in particles that are arrays of size nPart do:
        elif type(particles[key]).__module__ == np.__name__:
            if np.size(particles[key]) == nPart:
                particles[key] = np.extract(mask, particles[key])

    particles["mTot"] = np.sum(particles["m"])

    return particles


def addParticles(particles, nAdd, ind, mNew, xNew, yNew, zNew):
    """add particles

    Parameters
    ----------
    particles : dict
        particles dictionary
    nAdd : int
        number of particles added (one particles is modified, nAdd are added)
    ind : int
        index of particle modified
    mNew: float
        new mass of the particles
    xNew: numpy array
        new x position of the particles
    yNew: numpy array
        new y position of the particles
    zNew: numpy array
        new z position of the particles

    Returns
    -------
    particles : dict
        particles dictionary with modified particle and new ones
    """
    # get old values
    nPart = particles["nPart"]
    nID = particles["nID"]
    # update total number of particles and number of IDs used so far
    particles["nPart"] = particles["nPart"] + nAdd
    particles["nID"] = nID + nAdd
    # log.info('Spliting particle %s in %s' % (ind, nAdd))
    for key in particles:
        # update splitted particle mass
        # first update the old particle
        particles["m"][ind] = mNew
        particles["x"][ind] = xNew[0]
        particles["y"][ind] = yNew[0]
        particles["z"][ind] = zNew[0]
        # add new particles at the end of the arrays
        # for all keys in particles that are arrays of size nPart do:
        if type(particles[key]).__module__ == np.__name__:
            # create unique ID for the new particles
            if key == "ID":
                particles["ID"] = np.append(particles["ID"], np.arange(nID, nID + nAdd, 1))
            elif key == "x":
                particles[key] = np.append(particles[key], xNew[1:])
            elif key == "y":
                particles[key] = np.append(particles[key], yNew[1:])
            elif key == "z":
                particles[key] = np.append(particles[key], zNew[1:])
            elif key == "bondStart":
                # no bonds for added particles:
                nBondsParts = np.size(particles["bondPart"])
                particles[key] = np.append(particles[key], nBondsParts * np.ones((nAdd)))
            # set the parent properties to new particles due to splitting
            elif np.size(particles[key]) == nPart:
                particles[key] = np.append(particles[key], particles[key][ind] * np.ones((nAdd)))
            # ToDo: maybe also update the h smartly
    return particles


def splitPartMass(particles, cfg):
    """Split big particles

    Split particles bigger than thresholdMassSplit x massPerPart
    place the new particle in the flow direction at distance epsilon x rPart
    (this means splitting happens only if particles grow -> entrainment)

    Parameters
    ----------
    particles : dict
        particles dictionary
    cfg : configParser
        GENERAL configuration for com1DFA

    Returns
    -------
    particles : dict
        particles dictionary

    """
    rho = cfg.getfloat("rho")
    thresholdMassSplit = cfg.getfloat("thresholdMassSplit")
    distSplitPart = cfg.getfloat("distSplitPart")
    massPerPart = particles["massPerPart"]
    mPart = particles["m"]
    # decide which particles to split
    nSplit = np.ceil(mPart / (massPerPart * thresholdMassSplit))
    Ind = np.where(nSplit > 1)[0]
    # loop on particles to split
    for ind in Ind:
        # compute new mass (split particle in 2)
        mNew = mPart[ind] / 2  # nSplit[ind]
        nAdd = 1  # (nSplit[ind]-1).astype('int')
        xNew, yNew, zNew = getSplitPartPositionSimple(particles, rho, distSplitPart, ind)
        # add new particles
        particles = addParticles(particles, nAdd, ind, mNew, xNew, yNew, zNew)

    particles["mTot"] = np.sum(particles["m"])
    return particles


def splitPartArea(particles, cfg, dem):
    """Split big particles

    Split particles to keep enough particles within the kernel radius.
    place the new particle in the flow direction at distance epsilon x rPart

    Parameters
    ----------
    particles : dict
        particles dictionary
    cfg : configParser
        GENERAL configuration for com1DFA
    dem : dict
        dem dictionary

    Returns
    -------
    particles : dict
        particles dictionary

    """
    # get cfg info
    rho = cfg.getfloat("rho")
    sphKernelRadius = cfg.getfloat("sphKernelRadius")
    cMinNPPK = cfg.getfloat("cMinNPPK")
    cMinMass = cfg.getfloat("cMinMass")
    nSplit = cfg.getint("nSplit")
    # get dem info
    csz = dem["header"]["cellsize"]
    Nx = dem["Nx"]
    Ny = dem["Ny"]
    Nz = dem["Nz"]
    # get the threshold area over which we split the particle
    massPerPart = particles["massPerPart"]
    nPPK = particles["nPPK"]
    aMax = math.pi * sphKernelRadius**2 / (cMinNPPK * nPPK)
    mMin = massPerPart * cMinMass
    # get particle area
    mPart = particles["m"]
    hPart = particles["h"]
    aPart = mPart / (rho * hPart)
    # find particles to split
    tooBig = np.where((aPart > aMax) & (mPart / nSplit > mMin))[0]
    # count new particles
    nNewPart = 0
    # loop on particles to split
    for ind in tooBig:
        # compute new mass and particles to add
        mNew = mPart[ind] / nSplit
        nAdd = nSplit - 1
        nNewPart = nNewPart + nAdd
        # get position of new particles
        xNew, yNew, zNew = getSplitPartPosition(cfg, particles, aPart, Nx, Ny, Nz, csz, nSplit, ind)
        # add new particles
        particles = addParticles(particles, nAdd, ind, mNew, xNew, yNew, zNew)
    log.debug("Added %s because of splitting" % (nNewPart))

    particles["mTot"] = np.sum(particles["m"])
    return particles


def mergePartArea(particles, cfg, dem):
    """merge small particles

    merge particles to avoid too many particles within the kernel radius.
    place the new merge particle between the two old ones. The new position and velocity are the
    mass averaged ones

    Parameters
    ----------
    particles : dict
        particles dictionary
    cfg : configParser
        GENERAL configuration for com1DFA
    dem : dict
        dem dictionary

    Returns
    -------
    particles : dict
        particles dictionary

    """
    # get cfg info
    rho = cfg.getfloat("rho")
    sphKernelRadius = cfg.getfloat("sphKernelRadius")
    cMaxNPPK = cfg.getfloat("cMaxNPPK")
    # get the threshold area under which we merge the particle
    nPPK = particles["nPPK"]
    aMin = math.pi * sphKernelRadius**2 / (cMaxNPPK * nPPK)
    # get particle area
    mPart = particles["m"]
    hPart = particles["h"]
    xPart = particles["x"]
    yPart = particles["y"]
    zPart = particles["z"]
    aPart = mPart / (rho * hPart)
    # find particles to merge
    tooSmall = np.where(aPart < aMin)[0]
    keepParticle = np.ones((particles["nPart"]), dtype=bool)
    nRemoved = 0
    # loop on particles to merge
    for ind in tooSmall:
        if keepParticle[ind]:
            # find nearest particle
            rMerge, neighbourInd = getClosestNeighbour(
                xPart, yPart, zPart, ind, sphKernelRadius, keepParticle
            )
            # only merge a particle if it is closer thant the kernel radius
            if rMerge < sphKernelRadius:
                # remove neighbourInd from tooSmall if possible
                keepParticle[neighbourInd] = False
                nRemoved = nRemoved + 1
                # compute new mass and particles to add
                mNew = mPart[ind] + mPart[neighbourInd]
                # compute mass averaged values
                for key in ["x", "y", "z", "ux", "uy", "uz"]:
                    particles[key][ind] = (
                        mPart[ind] * particles[key][ind] + mPart[neighbourInd] * particles[key][neighbourInd]
                    ) / mNew
                particles["m"][ind] = mNew
                # ToDo: mabe also update h

    particles = removePart(particles, keepParticle, nRemoved, reasonString="")  # 'because of colocation')
    return particles


def getClosestNeighbour(xPartArray, yPartArray, zPartArray, ind, sphKernelRadius, keepParticle):
    """find closest neighbour

    Parameters
    ----------
    xPartArray: numpy array
        x position of the particles
    yPartArray: numpy array
        y position of the particles
    zPartArray: numpy array
        z position of the particles
    ind : int
        index of particle modified
    sphKernelRadius: float
        kernel radius
    keepParticle: numpy array
        boolean array telling if particles are kept or merged

    Returns
    -------
    rMerge : float
        distance to the closest neighbour
    neighbourInd : int
        index of closest neighbour
    """
    r = DFAtls.norm(
        xPartArray - xPartArray[ind],
        yPartArray - yPartArray[ind],
        zPartArray - zPartArray[ind],
    )
    # make sure you don't find the particle itself
    r[ind] = 2 * sphKernelRadius
    # make sure you don't find a particle that has already been merged
    r[~keepParticle] = 2 * sphKernelRadius
    # find nearest particle
    neighbourInd = np.argmin(r)
    rMerge = r[neighbourInd]
    return rMerge, neighbourInd


def mergeParticleDict(particles1, particles2):
    """Merge two particles dictionary

    Parameters
    ----------
    particles1 : dict
        first particles dictionary
    particles2 : dict
        second particles dictionary

    Returns
    -------
    particles : dict
        merged particles dictionary

    """
    particles = {}
    nPart1 = particles1["nPart"]
    # loop on the keys from particles1 dicionary
    for key in particles1:
        # deal with specific cases
        # nPart: just sum them up
        if key == "nPart":
            particles["nPart"] = particles1["nPart"] + particles2["nPart"]
        # massPerPart, should stay unchanged. If ever they are different take
        # the minimum
        # ToDo: are we sure we want the minimum?
        elif key == "massPerPart":
            particles["massPerPart"] = min(particles1["massPerPart"], particles2["massPerPart"])
        # now if the value is a numpy array and this key is also in particles2
        elif (key in particles2) and (type(particles1[key]).__module__ == np.__name__):
            # deal with the specific cases:
            # in the case of ID or 'parentID' we assume that both particles1 and
            # particles2 were initialized with an ID and parentID starting at 0
            # here when we merge the 2 arrays we make sure to shift the value
            # of particles2 so that the ID stays a unique identifier and
            # that the parentID is consistent with this shift.
            if (key == "ID") or (key == "parentID"):
                particles[key] = np.append(particles1[key], (particles2[key] + particles1["nID"]))
            # general case where the key value is an array with as many elements
            # as particles
            elif np.size(particles1[key]) == nPart1:
                particles[key] = np.append(particles1[key], particles2[key])
            # if the array is of size one, (potential energy, mTot...) we just
            # sum the 2 values
            else:
                particles[key] = particles1[key] + particles2[key]
        # the key is in both dictionaries, it is not an array but it is a
        # number (int, double, float) then we sum the 2 values
        elif (key in particles2) and (isinstance(particles1[key], numbers.Number)):
            if "llcenter" in key:
                log.debug("Not modifying particles llcenter coordinate")
                particles[key] = particles1[key]
            else:
                particles[key] = particles1[key] + particles2[key]
        # finally, if the key is only in particles1 then we give this value to
        # the new particles
        else:
            particles[key] = particles1[key]
    return particles


def getSplitPartPosition(cfg, particles, aPart, Nx, Ny, Nz, csz, nSplit, ind):
    """Compute the new particle position due to splitting

    Parameters
    ----------
    cfg : configParser
        GENERAL configuration for com1DFA
    particles : dict
        particles dictionary
    aPart : numpy array
        particle area array
    Nx : numpy 2D array
        x component of the normal vector on the grid
    Ny : numpy 2D array
        y component of the normal vector on the grid
    Nz : numpy 2D array
        z component of the normal vector on the grid
    csz : float
        grid cell size
    nSplit : int
        in how many particles do we split?
    ind : int
        index of the particle to split

    Returns
    -------
    xNew : numpy array
        x components of the splitted particles
    yNew : numpy array
        y components of the splitted particles
    zNew : numpy array
        z components of the splitted particles
    """
    rng = np.random.default_rng(int(cfg["seed"]))
    x = particles["x"]
    y = particles["y"]
    z = particles["z"]
    ux = particles["ux"]
    uy = particles["uy"]
    uz = particles["uz"]
    rNew = np.sqrt(aPart[ind] / (math.pi * nSplit))
    alpha = 2 * math.pi * (np.arange(nSplit) / nSplit + rng.random(1))
    cos = rNew * np.cos(alpha)
    sin = rNew * np.sin(alpha)
    nx, ny, nz = geoTrans.getNormalArray(np.array([x[ind]]), np.array([y[ind]]), Nx, Ny, Nz, csz)
    e1x, e1y, e1z, e2x, e2y, e2z = getTangenVectors(
        nx, ny, nz, np.array([ux[ind]]), np.array([uy[ind]]), np.array([uz[ind]])
    )
    xNew = x[ind] + cos * e1x + sin * e2x
    yNew = y[ind] + cos * e1y + sin * e2y
    zNew = z[ind] + cos * e1z + sin * e2z
    # toDo: do we need to reproject the particles on the dem?
    return xNew, yNew, zNew


def getSplitPartPositionSimple(particles, rho, distSplitPart, ind):
    """Compute the new particle potion due to splitting

    Parameters
    ----------
    particles : dict
        particles dictionary
    rho : float
        density
    distSplitPart : float
        distance coefficient
    ind : int
        index of the particle to split

    Returns
    -------
    xNew : numpy array
        x components of the splitted particles
    yNew : numpy array
        y components of the splitted particles
    zNew : numpy array
        z components of the splitted particles
    """
    mPart = particles["m"][ind]
    hPart = particles["h"][ind]
    xPart = particles["x"][ind]
    yPart = particles["y"][ind]
    zPart = particles["z"][ind]
    uxPart = particles["ux"][ind]
    uyPart = particles["uy"][ind]
    uzPart = particles["uz"][ind]
    # get the area of the particle as well as the distance expected between the old and new particle
    # note that if we did not update the particles FT, we use here the h from the previous time step
    aPart = mPart / (rho * hPart)
    rNew = distSplitPart * np.sqrt(aPart / math.pi)
    cos = rNew * np.array([-1, 1])
    # compute velocity mag to get the direction of the flow (e_1)
    uMag = DFAtls.norm(uxPart, uyPart, uzPart)
    xNew = xPart + cos * uxPart / uMag
    yNew = yPart + cos * uyPart / uMag
    zNew = zPart + cos * uzPart / uMag
    # toDo: do we need to reproject the particles on the dem?
    return xNew, yNew, zNew


def getTangenVectors(nx, ny, nz, ux, uy, uz):
    """Compute the tangent vector to the surface

    If possible, e1 is in the velocity direction, if not possible,
    use the tangent vector in x direction for e1 (not that any other u vector could be provided,
    it does not need to be the velocity vector, it only needs to be in the tangent plane)

    Parameters
    ----------
    nx : float
        x component of the normal vector
    ny : float
        y component of the normal vector
    nz : float
        z component of the normal vector
    ux : float
        x component of the velocity vector
    uy : float
        y component of the velocity vector
    uz : float
        z component of the velocity vector

    Returns
    -------
    e1x : float
        x component of the first tangent vector
    e1y : float
        y component of the first tangent vector
    e1z : float
        z component of the first tangent vector
    e2x : float
        x component of the second tangent vector
    e2y : float
        y component of the second tangent vector
    e2z : float
        z component of the second tangent vector
    """
    # compute the velocity magnitude
    velMag = DFAtls.norm(ux, uy, uz)
    if velMag > 0:
        e1x = ux / velMag
        e1y = uy / velMag
        e1z = uz / velMag
    else:
        # if vector u is zero use the tangent vector in x direction for e1
        e1x = np.array([1])
        e1y = np.array([0])
        e1z = -nx / nz
        e1x, e1y, e1z = DFAtls.normalize(e1x, e1y, e1z)
    # compute the othe tengent vector
    e2x, e2y, e2z = DFAtls.crossProd(nx, ny, nz, e1x, e1y, e1z)
    e2x, e2y, e2z = DFAtls.normalize(e2x, e2y, e2z)
    return e1x, e1y, e1z, e2x, e2y, e2z


def findParticles2Track(particles, center, radius):
    """Find particles within a circle arround a given point

    Parameters
    ----------
    particles : dict
        particles dictionary (with the 'parentID' array)
    center : dict
        point dictionary:
            x : x coordinate
            y : y coordinate
            z : z coordinate
    radius : float
        radius of the circle around point

    Returns
    -------
    particles2Track : numpy array
        array with Parent ID of particles to track
    track: boolean
        False if no particles are tracked
    """
    track = True
    x = particles["x"]
    y = particles["y"]
    xc = center["x"]
    yc = center["y"]
    r = DFAtls.norm(x - xc, y - yc, 0)
    index = np.where(r <= radius)
    particles2Track = particles["parentID"][index]
    log.info("Tracking %d particles" % len(index[0]))
    if len(index[0]) < 1:
        log.warning("Found No particles to track ")
        track = False

    return particles2Track, track


def getTrackedParticles(particlesList, particles2Track):
    """Track particles along time given the parentID of the particles to track

    Parameters
    ----------
    particlesList : list
        list of particles dictionaries (with the 'parentID' array)
    particles2Track : numpy array
        array with the parentID of the particles to track

    Returns
    -------
    particlesList : list
        list of particles dictionaries updated with the 'trackedParticles'
        array (in the array, the ones correspond to the particles that
        are tracked)
    nPartTracked : int
        total number of tracked particles
    """
    nPartTracked = np.size(particles2Track)
    # add trackedParticles array to the particles dictionary for every saved time step
    for particles in particlesList:
        # find index of particles to track
        index = [ind for ind, parent in enumerate(particles["parentID"]) if parent in particles2Track]
        nPartTracked = max(nPartTracked, len(index))
        trackedParticles = np.zeros(particles["nPart"])
        trackedParticles[index] = 1
        particles["trackedParticles"] = trackedParticles
    return particlesList, nPartTracked


def getTrackedParticlesProperties(particlesList, nPartTracked, properties):
    """Get the desired properties for the tracked particles

    Parameters
    ----------
    particlesList : list
        list of particles dictionaries (with the 'parentID' array)
    nPartTracked : int
        total number of tracked particles
    properties : list
        list of strings

    Returns
    -------
    trackedPartProp : dict
        dictionary with 2D numpy arrays corresponding to the time series of the
        properties for the tracked particles (for example if
        properties = ['x', 'y'], the dictionary will have the keys 't',
        'x' and 'y'. trackedPartProp['x'] will be a 2D numpy array, each line
        corresponds to the 'x' time series of a tracked particle)
    """
    # buid time series for desired properties of tracked particles
    nTimeSteps = len(particlesList)
    trackedPartProp = {}
    trackedPartProp["t"] = np.zeros(nTimeSteps)
    newProperties = []

    # initialize
    for key in properties:
        if key in particlesList[0]:
            trackedPartProp[key] = np.zeros((nTimeSteps, nPartTracked))
            newProperties.append(key)
        else:
            log.warning("%s is not a particle property" % key)

    # extract wanted properties and build the time series
    trackedPartID = []
    for particles, nTime in zip(particlesList, range(nTimeSteps)):
        trackedParticles = particles["trackedParticles"]
        trackedPartProp["t"][nTime] = particles["t"]
        index = np.where(trackedParticles == 1)
        for ind, id in zip(index[0], particles["ID"][index]):
            if id not in trackedPartID:
                trackedPartID.append(id)
            indCol = trackedPartID.index(id)
            for key in newProperties:
                trackedPartProp[key][nTime, indCol] = particles[key][ind]

    return trackedPartProp


def readPartFromPickle(inDir, simName="", flagAvaDir=False, comModule="com1DFA"):
    """Read pickles within a directory and return List of dicionaries read from pickle

    Parameters
    -----------
    inDir: str
        path to input directory
    simName : str
        simulation name
    flagAvaDir: bool
        if True inDir corresponds to an avalanche directory and pickles are
        read from avaDir/Outputs/com1DFA/particles
    comModule: str
        module that computed the particles
    """

    if flagAvaDir:
        inDir = pathlib.Path(inDir, "Outputs", comModule, "particles")

    # search for all pickles within directory
    if simName:
        name = "*" + simName + "*.pickle"
    else:
        name = "*.pickle"
    PartDicts = sorted(list(inDir.glob(name)))

    # initialise list of particle dictionaries
    Particles = []
    timeStepInfo = []
    for particles in PartDicts:
        particles = pickle.load(open(particles, "rb"))
        Particles.append(particles)
        timeStepInfo.append(particles["t"])

    return Particles, timeStepInfo


def savePartToCsv(particleProperties, dictList, outDir, countParticleCsv=None):
    """Save each particle dictionary from a list to a csv file;
    works also for one dictionary instead of list

    Parameters
    ---------
    particleProperties: str
        all particle properties that shall be saved to csv file
        (e.g.: m, velocityMagnitude, ux,..)
    dictList: list or dict
        list of dictionaries or single dictionary
    outDir: str
        path to output directory; particlesCSV will be created in this
        outDir
    countParticleCsv: int
        number of particlesDict to be saved according to time step - ranging from 0...n where n is the total number
        of saved particleDicts - if list of dicts is saved this parameter is ignored
    """

    # set output directory
    outDir = outDir / "particlesCSV"
    fU.makeADir(outDir)

    # read particle properties to be saved
    particleProperties = particleProperties.split("|")

    # write particles locations and properties to csv file
    nParticles = len(dictList)
    if nParticles == 1:
        if countParticleCsv is None:
            message = "Indicator for step in timeseries for particlesDict to be saved is required, set countPartCsv"
            log.error(message)
            raise AssertionError(message)
        else:
            count = countParticleCsv
    else:
        count = 0

    for m in range(nParticles):
        if nParticles == 1:
            particles = dictList[0]
        else:
            particles = dictList[count]
        simName = particles["simName"]
        csvData = {}
        csvData["X"] = particles["x"] + particles["xllcenter"]
        csvData["Y"] = particles["y"] + particles["yllcenter"]
        csvData["Z"] = particles["z"]

        for partProp in particleProperties:
            if partProp == "velocityMagnitude":
                ux = particles["ux"]
                uy = particles["uy"]
                uz = particles["uz"]
                csvData[partProp] = DFAtls.norm(ux, uy, uz)
            else:
                if partProp in particles:
                    csvData[partProp] = particles[partProp]
                else:
                    log.warning("Particle property %s does not exist" % (partProp))
        csvData["time"] = particles["t"]

        # create pandas dataFrame and save to csv
        outFile = outDir / ("particles%s.csv.%04d" % (simName, count))
        particlesData = pd.DataFrame(data=csvData)
        particlesData.to_csv(outFile, index=False)
        count = count + 1


def reshapeParticlesDicts(particlesList, propertyList):
    """reshape particlesList from one dict per time step with all particle properties for each particle,
    to one dict with an array of property values for all time steps for each particle
    shape: nx x ny; nx time steps, ny number of particles

    Parameters
    -----------
    particlesList: list
        list of particle dicts, one dict per time step
    propertyList: list
        list of property names that shall be reshaped and saved to particlesTimeArrays

    Returns
    --------
    particlesTimeArrays: dict
        dict with time series of properties of particles
        key: property, item: timeSteps x particlesID array of property values
    """

    # create particlesTimeArrays
    particlesTimeArrays = {}

    idParticles = [p["nPart"] for p in particlesList]
    if len(list(set(idParticles))) > 1:
        message = "Number of particles changed throughout simulation"
        log.error(message)
        raise AssertionError(message)

    for props in propertyList:
        if isinstance(particlesList[0][props], int) or isinstance(particlesList[0][props], float):
            particlesTimeArrays[props] = np.zeros(len(particlesList))
            particlesTimeArrays[props][:] = np.asarray([p[props] for p in particlesList])
        else:
            particlesTimeArrays[props] = np.zeros((len(particlesList), particlesList[0]["nPart"]))
            for idx in particlesList[0]["ID"]:
                particlesTimeArrays[props][:, idx] = np.asarray([p[props][idx] for p in particlesList])

    return particlesTimeArrays


def savePartDictToPickle(partDict, fName):
    """save a single dict to a pickle

    Parameters
    -----------
    partDict: dict
        dict with info
    fName: pathlib path
        path to saving location
    """

    fi = open(fName, "wb")
    pickle.dump(partDict, fi)
    fi.close()
