"""
    Tools for generating an avalanche path from a DFA simulation
"""

# Load modules
import math
import heapq
import numpy as np
import logging
import pathlib
import shutil
from scipy.ndimage import distance_transform_edt

# Local imports
from avaframe.in1Data import getInput as gI
import avaframe.in2Trans.rasterUtils as IOf
import avaframe.in2Trans.shpConversion as shpConv
from avaframe.in3Utils import cfgUtils
from avaframe.in3Utils import fileHandlerUtils as fU
from avaframe.in3Utils import cfgHandling
import avaframe.in3Utils.initializeProject as initProj
import avaframe.in3Utils.geoTrans as gT
import avaframe.com1DFA.particleTools as particleTools
import avaframe.com1DFA.DFAtools as DFAtls
from avaframe.com1DFA import com1DFA
import avaframe.out3Plot.outDebugPlots as debPlot
from avaframe.out3Plot import outCom3Plots

# create local logger
# change log level in calling module to DEBUG to see log messages
log = logging.getLogger(__name__)
cfgAVA = cfgUtils.getGeneralConfig()
debugPlot = cfgAVA['FLAGS'].getboolean('debugPlot')

def generatePathAndSplitpoint(avalancheDir, cfgDFAPath, cfgMain, runDFAModule):
    """
    Parameters:
    avalancheDir (str):
        path to the avalanche directory
    cfgDFAPath (configParser object):
        DFAPath configuration
    cfgMain (configParser object):
        main avaframe configuration
    runDFAModule (bool):
        whether to run the DFA module (True) or load existing results (False)
    Returns
    -------
    massAvgPath: pathlib
        file path to the mass-averaged path result saved as a shapefile
    splitPoint: pathlib
        file path to the split point result saved as a shapefile
    """
    if runDFAModule: # call DFA module to perform simulations with overrides from DFAPath config
            # Clean avalanche directory of old work and output files from module
            initProj.cleanModuleFiles(avalancheDir, com1DFA, deleteOutput=True)
            # create and read the default com1DFA config (no local is read)
            com1DFACfg = cfgUtils.getModuleConfig(com1DFA, avalancheDir, toPrint=False,
                                                  onlyDefault=cfgDFAPath['com1DFA_com1DFA_override'].getboolean(
                                                      'defaultConfig'))
            # and override with settings from DFAPath config
            com1DFACfg, cfgDFAPath = cfgHandling.applyCfgOverride(com1DFACfg, cfgDFAPath, com1DFA,
                                                                         addModValues=False)
            outDir = pathlib.Path(avalancheDir, 'Outputs', 'ana5Utils', 'DFAPath')
            fU.makeADir(outDir)
            # write configuration to file for documentation
            com1DFACfgFile = outDir / 'com1DFAPathGenerationCfg.ini'
            with open(com1DFACfgFile, 'w') as configfile:
                com1DFACfg.write(configfile)
            # call com1DFA and perform simulations
            dem, plotDict, reportDictList, simDF = com1DFA.com1DFAMain(cfgMain, cfgInfo=com1DFACfg)
    else: # read existing simulation results
        # read simulation dem
        demOri = gI.readDEM(avalancheDir)
        dem = com1DFA.setDEMoriginToZero(demOri)
        dem['originalHeader'] = demOri['header'].copy()
        # load DFA results (use runCom1DFA to generate these results for example)
        # here is an example with com1DFA but another DFA computational module can be used
        # as long as it produces some pta, particles or FT, FM and FV results
        # create dataFrame of results (read DFA data)
        simDF, _ = cfgUtils.readAllConfigurationInfo(avalancheDir)
        if simDF is None:
            message = "Did not find any com1DFA simulations in %s/Outputs/com1DFA/" % avalancheDir
            log.error(message)
            raise FileExistsError(message)

    # the peak flow thickness fields are only needed when the path is extended to the deposit
    # front; parse the peak files once here instead of re-reading them for every simulation
    extendToFront = cfgDFAPath['PATH'].getint('extBottomOption', fallback=0) == 1
    peakFilesDF = None
    if extendToFront:
        peakFilesDir = pathlib.Path(avalancheDir, 'Outputs', 'com1DFA', 'peakFiles')
        peakFilesDF = fU.makeSimDF(peakFilesDir, avaDir=avalancheDir)

    for simName, simDFrow in simDF.iterrows():
        log.info('Computing avalanche path from simulation: %s', simName)
        pathFromPart = cfgDFAPath['PATH'].getboolean('pathFromPart')
        resampleDistance = cfgDFAPath['PATH'].getfloat('nCellsResample') * dem['header']['cellsize']
        # peak flow thickness field, only needed when extending the path to the deposit front
        fieldFT = None
        if extendToFront:
            fieldFT = readPeakFT(peakFilesDF, simName)
        # get the mass average path
        avaProfileMass, particlesIni = generateMassAveragePath(avalancheDir, pathFromPart, simName, dem,
                                                               addVelocityInfo=cfgDFAPath['PATH'].getboolean('addVelocityInfo'))
        avaProfileMass, _ = gT.prepareLine(dem, avaProfileMass, distance=resampleDistance, Point=None)
        avaProfileMass['indStartMassAverage'] = 1
        avaProfileMass['indEndMassAverage'] = np.size(avaProfileMass['x'])
        # make the parabolic fit
        parabolicFit = getParabolicFit(cfgDFAPath['PATH'], avaProfileMass, dem)
        # here the avaProfileMass given in input is overwritten and returns only an x, y, z extended profile
        avaProfileMass = extendDFAPath(cfgDFAPath['PATH'], avaProfileMass, dem, particlesIni,
                                       fieldFT=fieldFT)
        # resample path and keep track of start and end of mass averaged part
        avaProfileMass = resamplePath(cfgDFAPath['PATH'], dem, avaProfileMass)
        # get split point
        splitPoint = getSplitPoint(cfgDFAPath['PATH'], avaProfileMass, parabolicFit)
        # make analysis and generate plots
        _ = outCom3Plots.generateCom1DFAPathPlot(avalancheDir, cfgDFAPath['PATH'], avaProfileMass, dem,
                                                 parabolicFit, splitPoint, simName)
        # now save the path and split point as shapefiles
        avaPath,splitPoint = saveSplitAndPath(avalancheDir, simDFrow, splitPoint, avaProfileMass, dem)

    return avaPath, splitPoint

def generateMassAveragePath(avalancheDir, pathFromPart, simName, dem, addVelocityInfo=False, flagAvaDir=True,
                            comModule='com1DFA'):
    """ extract path from fields or particles

    Parameters
    -----------
    avalancheDir: pathlib
        avalanche directory pathlib path
    pathFromPart: boolean
        compute path from particles if True, from fields (FT, FM, FV) if False
    simName: str
        simulation name
    dem: dict
        com1DFA simulation dictionary
    addVelocityInfo: boolean
        True to add (u2, ekin, totEKin) to result
        Will only work if the particles (ux, uy, uz) exist or if 'FV' exists
    flagAvaDir: bool
        if True avalancheDir corresponds to an avalanche directory and data is
        read from avaDir/Outputs/comModule/particles or avaDir/Outputs/comModule/peakFiles
        depending on if pathFromPart is True or False
    comModule: str
        module that computed the particles or fields

    Returns
    --------
    avaProfileMass: dict
        mass averaged profile (x, y, z, 's')
        if addVelocityInfo is True, kinetic energy and velocity information are added to
        the avaProfileMass dict (u2, ekin, totEKin)
    particlesIni: dict
        x, y coord of the initial particles or flow thickness field
    """
    if pathFromPart:
        particlesList, timeStepInfo = particleTools.readPartFromPickle(avalancheDir, simName=simName, flagAvaDir=True,
                                                                       comModule='com1DFA')
        particlesIni = particlesList[0]
        log.info('Using particles to generate avalanche path profile')
        # postprocess to extract path and energy line
        avaProfileMass = getMassAvgPathFromPart(particlesList, addVelocityInfo=addVelocityInfo)
    else:
        particlesList = ''
        # read field
        fieldName = ['FT', 'FM']
        if addVelocityInfo:
            fieldName.append('FV')
        fieldsList, fieldHeader, timeList = com1DFA.readFields(avalancheDir, fieldName, simName=simName,
                                                               flagAvaDir=True, comModule='com1DFA')
        # get fields header
        ncols = fieldHeader['ncols']
        nrows = fieldHeader['nrows']
        csz = fieldHeader['cellsize']
        # we want the origin to be in (0, 0) as it is in the avaProfile that comes in
        X, Y = gT.makeCoordinateGrid(0, 0, csz, ncols, nrows)
        indNonZero = np.where(fieldsList[0]['FT'] > 0)
        # convert this data in a particles style (dict with x, y, z info)
        particlesIni = {'x': X[indNonZero], 'y': Y[indNonZero]}
        particlesIni, _ = gT.projectOnRaster(dem, particlesIni)
        log.info('Using fields to generate avalanche path profile')
        # postprocess to extract path and energy line
        avaProfileMass = getMassAvgPathFromFields(fieldsList, fieldHeader, dem)

    return avaProfileMass, particlesIni


def getMassAvgPathFromPart(particlesList, addVelocityInfo=False):
    """ compute mass averaged path from particles

    Also returns the averaged velocity and kinetic energy associated
    If addVelocityInfo is True, information about velocity and kinetic energy is computed

    Parameters
    -----------
    particlesList: list
        list of particles dict
    addVelocityInfo: boolean
        True to add (u2, ekin, totEKin) to result

    Returns
    --------
    avaProfileMass: dict
        mass averaged profile (x, y, z, 's', 'sCor')
        if addVelocityInfo is True, kinetic energy and velocity information are added to
        the avaProfileMass dict (u2, ekin, totEKin)
    """

    propList = ['x', 'y', 'z', 's', 'sCor']
    propListPart = ['x', 'y', 'z', 'trajectoryLengthXY', 'trajectoryLengthXYCor']
    avaProfileMass = {}
    # do we have velocity info?
    if addVelocityInfo:
        propList.append('u2')
        propList.append('ekin')
        propListPart.append('u2')
        propListPart.append('ekin')
        avaProfileMass['totEKin'] = np.empty((0, 1))
    # initialize other properties
    for prop in propList:
        avaProfileMass[prop] = np.empty((0, 1))
        avaProfileMass[prop + 'std'] = np.empty((0, 1))

    # loop on each particle dictionary (ie each time step saved)
    for particles in particlesList:
        if particles['nPart'] > 0:
            m = particles['m']
            if addVelocityInfo:
                ux = particles['ux']
                uy = particles['uy']
                uz = particles['uz']
                u = DFAtls.norm(ux, uy, uz)
                u2Array = u*u
                kineticEneArray = 0.5*m*u2Array
                particles['u2'] = u2Array
                particles['ekin'] = kineticEneArray

            # mass-averaged path
            avaProfileMass = appendAverageStd(propList, avaProfileMass, particles, m, naming=propListPart)

            if addVelocityInfo:
                avaProfileMass['totEKin'] = np.append(avaProfileMass['totEKin'], np.nansum(kineticEneArray))

    return avaProfileMass


def getMassAvgPathFromFields(fieldsList, fieldHeader, dem):
    """ compute mass averaged path from fields

    Also returns the averaged velocity and kinetic energy associated
    The dem and fieldsList (FT, FM and FV) need to have identical dimensions and cell size.
    If FV is not provided, information about velocity and kinetic energy is not computed

    Parameters
    -----------
    fieldsList: list
        time sorted list of fields dict
    fieldHeader: dict
        field header dict
    dem: dict
        dem dict

    Returns
    --------
    avaProfileMass: dict
        mass averaged profile (x, y, z, 's')
        if 'FV' in fieldsList, kinetic energy and velocity information are added to
        the avaProfileMass dict (u2, ekin, totEKin)
    """
    # get DEM
    demRaster = dem['rasterData']
    # get fields header
    ncols = fieldHeader['ncols']
    nrows = fieldHeader['nrows']
    xllc = fieldHeader['xllcenter']
    yllc = fieldHeader['yllcenter']
    csz = fieldHeader['cellsize']
    X, Y = gT.makeCoordinateGrid(xllc, yllc, csz, ncols, nrows)

    propList = ['x', 'y', 'z']
    avaProfileMass = {}
    # do we have velocity info?
    addVelocityInfo = False
    if 'FV' in fieldsList[0]:
        propList.append('u2')
        propList.append('ekin')
        avaProfileMass['totEKin'] = np.empty((0, 1))
        addVelocityInfo = True
    # initialize other properties
    for prop in propList:
        avaProfileMass[prop] = np.empty((0, 1))
        avaProfileMass[prop + 'std'] = np.empty((0, 1))
    # loop on each field dictionary (ie each time step saved)
    for field in fieldsList:
        # find cells with snow
        nonZeroIndex = np.where(field['FT'] > 0)
        xArray = X[nonZeroIndex]
        yArray = Y[nonZeroIndex]
        zArray, _ = gT.projectOnGrid(xArray, yArray, demRaster, csz=csz, xllc=xllc, yllc=yllc)
        mArray = field['FM'][nonZeroIndex]
        particles = {'x': xArray, 'y': yArray, 'z': zArray}
        if addVelocityInfo:
            uArray = field['FV'][nonZeroIndex]
            u2Array = uArray*uArray
            kineticEneArray = 0.5*mArray*u2Array
            particles['u2'] = u2Array
            particles['ekin'] = kineticEneArray

        # mass-averaged path
        avaProfileMass = appendAverageStd(propList, avaProfileMass, particles, mArray)

        if addVelocityInfo:
            avaProfileMass['totEKin'] = np.append(avaProfileMass['totEKin'], np.nansum(kineticEneArray))

    avaProfileMass['x'] = avaProfileMass['x'] - xllc
    avaProfileMass['y'] = avaProfileMass['y'] - yllc

    # compute s
    avaProfileMass = gT.computeS(avaProfileMass)
    return avaProfileMass


def readPeakFT(peakFilesDF, simName):
    """ get the peak flow thickness field (pft) of one simulation from the peak file dataframe

    Parameters
    -----------
    peakFilesDF: pandas DataFrame
        peak files of all simulations (from fU.makeSimDF), parsed once for the whole run
    simName: str
        simulation name

    Returns
    --------
    fieldFT: numpy array
        peak flow thickness raster (same grid as the simulation dem),
        None if no pft peak field is available for this simulation
    """
    # the simulation can be identified by its full name or by its hash (the index of the
    # configuration dataframe iterated in generatePathAndSplitpoint is the hash)
    isSim = (peakFilesDF['simName'] == simName) | (peakFilesDF['simID'] == simName)
    index = peakFilesDF.index[isSim & (peakFilesDF['resType'] == 'pft')]
    if len(index) == 0:
        log.warning('No pft peak field found for simulation %s' % simName)
        return None
    return IOf.readRaster(peakFilesDF.loc[index[0], 'files'])['rasterData']


def extendDFAPath(cfg, avaProfile, dem, particlesIni, fieldFT=None):
    """ extend the DFA path at the top and bottom
    avaProfile with x, y, z, s information

    Parameters
    -----------
    cfg: configParser
        configuration object with:

        - extTopOption: int, how to extend towards the top? 0 for heighst point method, a for largest runout method
        - extBottomOption: int, how to extend towards the bottom? 0 for the fixed-length extrapolation
        method, 1 for the least-cost extension to the deposit front (requires fieldFT)
        - nCellsResample: int, resampling length is given by nCellsResample*demCellSize
        - nCellsMinExtend: int, when extending towards the bottom, take points at more
        than nCellsMinExtend*demCellSize from last point to get the direction
        - nCellsMaxExtend: int, when extending towards the bottom, take points
        at less than nCellsMaxExtend*demCellSize from last point to get the direction
        - factBottomExt: float, extend the profile from factBottomExt*sMax

    avaProfile: dict
        profile to be extended
    dem: dict
        dem dict
    particlesIni: dict
        initial particles dict
    fieldFT: numpy array, optional
        peak flow thickness field on the dem grid, only used if extBottomOption = 1

    Returns
    --------
    avaProfile: dict
        extended profile at top and bottom (x, y, z).
    """
    # resample the profile
    resampleDistance = cfg.getfloat('nCellsResample') * dem['header']['cellsize']
    avaProfile, _ = gT.prepareLine(dem, avaProfile, distance=resampleDistance, Point=None)
    avaProfile = extendProfileTop(cfg.getint('extTopOption'), particlesIni, avaProfile)
    if cfg.getint('extBottomOption', fallback=0) == 1:
        if fieldFT is None:
            log.warning('extBottomOption is 1 but no flow thickness field was provided, '
                        'falling back to the fixed-length bottom extension')
            avaProfile = extendProfileBottom(cfg, dem, avaProfile)
        else:
            avaProfile = extendProfileToFront(cfg, dem, avaProfile, fieldFT)
    else:
        avaProfile = extendProfileBottom(cfg, dem, avaProfile)
    return avaProfile


def extendProfileTop(extTopOption, particlesIni, profile):
    """ extend the DFA path at the top (release)

    Either towards the highest point in particlesIni (extTopOption = 0)
    or the point leading to the longest runout (extTopOption = 1)

    Parameters
    -----------
    extTopOption: int
        decide how to extend towards the top
        if 0, extrapolate towards the highest point in the release
        if 1, extrapolate towards the point leading to the lonest runout
    particlesIni: dict
        initial particles dict
    profile: dict
        profile to extend

    Returns
    --------
    profile: dict
        extended profile
    """
    if extTopOption == 0:
        # get highest particle
        indTop = np.argmax(particlesIni['z'])
        xExtTop = particlesIni['x'][indTop]
        yExtTop = particlesIni['y'][indTop]
        zExtTop = particlesIni['z'][indTop]
        dx = xExtTop - profile['x'][0]
        dy = yExtTop - profile['y'][0]
        ds = np.sqrt(dx**2 + dy**2)
    elif extTopOption == 1:
        # get point with the most important runout gain
        # get first particle of the path
        xFirst = profile['x'][0]
        yFirst = profile['y'][0]
        zFirst = profile['z'][0]
        # get last particle of the path
        sLast = profile['s'][-1]
        zLast = profile['z'][-1]
        # compute runout angle for averaged path
        tanAngle = (zFirst-zLast)/sLast
        # compute ds
        dx = particlesIni['x'] - xFirst
        dy = particlesIni['y'] - yFirst
        ds = np.sqrt(dx**2 + dy**2)
        # compute dz
        dz = particlesIni['z'] - zFirst
        # remove the elevation needed to match the runout angle
        dz1 = dz - tanAngle * ds
        # get the particle with the highest potential
        indTop = np.argmax(dz1)
        xExtTop = particlesIni['x'][indTop]
        yExtTop = particlesIni['y'][indTop]
        zExtTop = particlesIni['z'][indTop]
        ds = ds[indTop]

    # extend profile
    profile['x'] = np.append(xExtTop, profile['x'])
    profile['y'] = np.append(yExtTop, profile['y'])
    profile['z'] = np.append(zExtTop, profile['z'])
    profile['s'] = np.append(0, profile['s'] + ds)
    if debugPlot:
        debPlot.plotPathExtTop(profile, particlesIni, xFirst, yFirst, zFirst, dz1)
    return profile


def extendProfileBottom(cfg, dem, profile):
    """ extend the DFA path at the bottom (runout area)

    Find the direction in which to extend considering the last point of the profile
    and a few previous ones but discarding the ones that are too close
    (nCellsMinExtend* csz < distFromLast <= nCellsMaxExtend * csz).
    Extend in this diretion for a distance factBottomExt * length of the path.

    Parameters
    -----------
    cfg: configParser
        nCellsMinExtend: int, when extending towards the bottom, take points
        at more than nCellsMinExtend*demCellSize from last point to get the direction

        nCellsMaxExtend: int, when extending towards the bottom, take points at
        less than nCellsMaxExtend*demCellSize from last point to get the direction
        factBottomExt: float, extend the profile from factBottomExt*sMax
    dem: dict
        dem dict
    profile: dict
        profile to extend

    Returns
    --------
    profile: dict
        extended profile
    """
    header = dem['header']
    csz = header['cellsize']
    zRaster = dem['rasterData']
    # get last point
    xLast = profile['x'][-1]
    yLast = profile['y'][-1]
    sLast = profile['s'][-1]
    # compute distance from last point:
    r = DFAtls.norm(profile['x']-xLast, profile['y']-yLast, 0)
    # find the previous points
    extendMinDistance = cfg.getfloat('nCellsMinExtend') * csz
    extendMaxDistance = cfg.getfloat('nCellsMaxExtend') * csz
    pointsOfInterestLast = np.where((r < extendMaxDistance) & (r > extendMinDistance))[0]
    xInterest = profile['x'][pointsOfInterestLast]
    yInterest = profile['y'][pointsOfInterestLast]

    # check if points are found to compute direction of extension
    if len(xInterest) > 0:
        # find the direction in which we need to extend the path
        vDirX = xLast - xInterest
        vDirY = yLast - yInterest
        vDirX, vDirY, vDirZ = DFAtls.normalize(np.array([vDirX]), np.array([vDirY]), 0*np.array([vDirY]))
        vDirX = np.sum(vDirX)
        vDirY = np.sum(vDirY)
        vDirZ = np.sum(vDirZ)
        vDirX, vDirY, vDirZ = DFAtls.normalize(np.array([vDirX]), np.array([vDirY]), np.array([vDirZ]))
        # extend in this direction
        factExt = cfg.getfloat('factBottomExt')
        gamma = factExt * sLast / np.sqrt(vDirX**2 + vDirY**2)
        xExtBottom = np.array([xLast + gamma * vDirX])
        yExtBottom = np.array([yLast + gamma * vDirY])
        # project on DEM
        zExtBottom, _ = gT.projectOnGrid(xExtBottom, yExtBottom, zRaster, csz=csz)
        # Dicothomie method to find the last point on the extention and on the dem
        if np.isnan(zExtBottom):
            factExt = factExt/2
            stepSize = factExt
            isOut = True
        else:
            isOut = False
            stepSize = 0
        count = 0
        # remember last point found inside
        factLast = 0
        while count < cfg.getint('maxIterationExtBot') and stepSize * sLast > cfg.getint('nBottomExtPrecision')*csz:
            count = count + 1
            gamma = factExt * sLast / np.sqrt(vDirX**2 + vDirY**2)
            xExtBottom = np.array([xLast + gamma * vDirX])
            yExtBottom = np.array([yLast + gamma * vDirY])
            # project on DEM
            zExtBottom, _ = gT.projectOnGrid(xExtBottom, yExtBottom, zRaster, csz=csz)
            stepSize = stepSize/2
            if np.isnan(zExtBottom):
                factExt = factExt - stepSize
                isOut = True
            else:
                # remember last point found inside
                factLast = factExt
                factExt = factExt + stepSize
                isOut = False

        if isOut:
            # the last iteration is not in the domain, fall back to last point in domain
            factExt = factLast
            gamma = factExt * sLast / np.sqrt(vDirX**2 + vDirY**2)
            xExtBottom = np.array([xLast + gamma * vDirX])
            yExtBottom = np.array([yLast + gamma * vDirY])
            # project on DEM
            zExtBottom, _ = gT.projectOnGrid(xExtBottom, yExtBottom, zRaster, csz=csz)
        log.info('found extention after %d iterations, precision is %.2f m' % (count, stepSize * sLast))

        # extend profile
        profile['x'] = np.append(profile['x'], xExtBottom)
        profile['y'] = np.append(profile['y'], yExtBottom)
        profile['z'] = np.append(profile['z'], zExtBottom)
        profile['s'] = np.append(profile['s'], sLast + np.sqrt((xLast-xExtBottom)**2 + (yLast-yExtBottom)**2))

    else:
        log.warning('Path not extended at bottom as no point of interest for computing direction \
            of where to extend path is found')

    if debugPlot:
        debPlot.plotPathExtBot(profile, xInterest, yInterest, 0*yInterest, xLast, yLast)
    return profile


def extendProfileToFront(cfg, dem, profile, fieldFT):
    """ extend the DFA path at the bottom to the front of the deposit

    Locate the front of the deposit in the flow thickness field (findFlowFront)
    and extend the profile to it along a least-cost path over the dem
    (leastCostPath). In contrast to extendProfileBottom, the extension follows
    the terrain and the deposit and stops at the front instead of extrapolating
    a straight line of fixed relative length. If the front cannot be located or
    reached, the profile falls back to the fixed-length extension
    (extendProfileBottom).

    Parameters
    -----------
    cfg: configParser
        configuration object with:

        - ftThreshold: float, minimum flow thickness (m) for a cell to belong to the flow footprint
        - lowFrontFraction: float, fraction of the flow elevation range defining the front band
        - upSlopePenalty: float, cost multiplier on the positive elevation gain of an edge
        - flowDistPenalty: float, cost multiplier on the distance (m) of a cell from the flow footprint
    dem: dict
        dem dict
    profile: dict
        profile to extend
    fieldFT: numpy array
        flow thickness field (peak or last time step) on the same grid as the dem

    Returns
    --------
    profile: dict
        extended profile (x, y, z, s)
    """
    header = dem['header']
    csz = header['cellsize']
    zRaster = dem['rasterData']
    if fieldFT.shape != zRaster.shape:
        message = 'Flow thickness field and dem do not have the same shape'
        log.error(message)
        raise AssertionError(message)
    # get last point
    xLast = profile['x'][-1]
    yLast = profile['y'][-1]
    sLast = profile['s'][-1]
    # locate the deposit front
    frontRow, frontCol = findFlowFront(fieldFT, zRaster, cfg.getfloat('ftThreshold'),
                                       cfg.getfloat('lowFrontFraction'))
    if frontRow is None:
        log.warning('No flow cell above ftThreshold was found, '
                    'falling back to the fixed-length bottom extension')
        return extendProfileBottom(cfg, dem, profile)
    # cell of the last profile point (profile and field share the dem grid)
    startRow = min(max(int(round((yLast - header['yllcenter']) / csz)), 0), zRaster.shape[0] - 1)
    startCol = min(max(int(round((xLast - header['xllcenter']) / csz)), 0), zRaster.shape[1] - 1)
    cellPath = leastCostPath((startRow, startCol), (frontRow, frontCol), fieldFT, zRaster, csz,
                             cfg.getfloat('ftThreshold'), cfg.getfloat('upSlopePenalty'),
                             cfg.getfloat('flowDistPenalty'))
    if len(cellPath) < 2:
        log.warning('No least-cost path to the deposit front was found, '
                    'falling back to the fixed-length bottom extension')
        return extendProfileBottom(cfg, dem, profile)
    # drop the first cell (the path end itself) and convert to coordinates; the extension
    # points are cell centers of valid dem cells, so z is read directly from the raster
    rows, cols = np.array(cellPath[1:]).T
    xExtBottom = header['xllcenter'] + cols * csz
    yExtBottom = header['yllcenter'] + rows * csz
    zExtBottom = zRaster[rows, cols]
    dx = np.diff(np.append(xLast, xExtBottom))
    dy = np.diff(np.append(yLast, yExtBottom))
    sExtBottom = sLast + np.cumsum(np.sqrt(dx**2 + dy**2))
    log.info('Path extended to the deposit front (%.0f m beyond the mass averaged path end)'
             % (sExtBottom[-1] - sLast))

    # extend profile
    profile['x'] = np.append(profile['x'], xExtBottom)
    profile['y'] = np.append(profile['y'], yExtBottom)
    profile['z'] = np.append(profile['z'], zExtBottom)
    profile['s'] = np.append(profile['s'], sExtBottom)
    return profile


def findFlowFront(fieldFT, demRaster, ftThreshold, lowFrontFraction):
    """ locate the front of the deposit in a flow thickness field

    The front is the flow-thickness-weighted centroid of the flow cells lying
    in the lowest lowFrontFraction of the flow elevation range. The band always
    contains the lowest flow cell; for a flat deposit it covers the whole
    footprint, so the front falls back to the centroid of the deposit. If the
    centroid falls outside the flow footprint (e.g. between two deposit lobes),
    the front is snapped to the nearest cell of the band.

    Parameters
    -----------
    fieldFT: numpy array
        flow thickness field
    demRaster: numpy array
        dem raster of the same shape
    ftThreshold: float
        minimum flow thickness (m) for a cell to belong to the flow footprint
    lowFrontFraction: float
        fraction of the flow elevation range defining the front band

    Returns
    --------
    frontRow, frontCol: int
        cell of the front, (None, None) if there is no flow above ftThreshold
    """
    flow = (fieldFT > ftThreshold) & np.isfinite(demRaster)
    if not flow.any():
        return None, None
    rows, cols = np.nonzero(flow)
    elev = demRaster[rows, cols]
    lowBand = elev <= elev.min() + lowFrontFraction * (elev.max() - elev.min())
    weight = fieldFT[rows, cols] * lowBand
    # weight.sum() is always positive: the lowest flow cell is in lowBand and, being a flow
    # cell, carries fieldFT > ftThreshold >= 0, so no divide-by-zero guard is needed
    frontRow = int(round(np.sum(rows * weight) / weight.sum()))
    frontCol = int(round(np.sum(cols * weight) / weight.sum()))
    if not flow[frontRow, frontCol]:
        # centroid outside the footprint (e.g. two deposit lobes): snap to the band
        bandInd = np.flatnonzero(lowBand)
        iNear = bandInd[np.argmin((rows[bandInd] - frontRow)**2 + (cols[bandInd] - frontCol)**2)]
        frontRow, frontCol = int(rows[iNear]), int(cols[iNear])
    return frontRow, frontCol


def leastCostPath(startCell, goalCell, fieldFT, demRaster, csz, ftThreshold, upSlopePenalty,
                  flowDistPenalty):
    """ Dijkstra least-cost path between two cells of the dem grid

    The cost of an edge is its horizontal length plus a penalty on the positive
    elevation gain and a penalty on the distance (m) of the target cell from
    the flow footprint, so that the path descends along the deposit.

    Parameters
    -----------
    startCell, goalCell: tuple
        (row, col) of the start and goal cells
    fieldFT: numpy array
        flow thickness field used to build the flow footprint
    demRaster: numpy array
        dem raster of the same shape
    csz: float
        cell size (m)
    ftThreshold: float
        minimum flow thickness (m) for a cell to belong to the flow footprint
    upSlopePenalty: float
        cost multiplier on the positive elevation gain of an edge
    flowDistPenalty: float
        cost multiplier on the distance (m) of a cell from the flow footprint

    Returns
    --------
    cellPath: list
        (row, col) cells from start to goal, empty if the goal is unreachable
    """
    if upSlopePenalty < 0 or flowDistPenalty < 0:
        message = 'The penalty parameters of the least-cost path must not be negative'
        log.error(message)
        raise AssertionError(message)
    # distance (m) from the flow footprint, used to keep the path on the deposit
    distToFlow = distance_transform_edt(~(fieldFT > ftThreshold), sampling=csz)
    nrows, ncols = demRaster.shape
    demValid = np.isfinite(demRaster)
    neighbours = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
    diagDist = csz * math.sqrt(2.)
    costGrid = np.full((nrows, ncols), np.inf)
    costGrid[startCell] = 0.
    previousCell = np.full((nrows, ncols, 2), -1, dtype=np.int64)
    queue = [(0., int(startCell[0]), int(startCell[1]))]
    while queue:
        cost, row, col = heapq.heappop(queue)
        if (row, col) == (goalCell[0], goalCell[1]):
            break
        if cost > costGrid[row, col]:
            continue
        for dRow, dCol in neighbours:
            nRow, nCol = row + dRow, col + dCol
            if 0 <= nRow < nrows and 0 <= nCol < ncols and demValid[nRow, nCol]:
                # no diagonal moves across the corner of two nodata cells
                if dRow and dCol and not (demValid[row, nCol] and demValid[nRow, col]):
                    continue
                dz = demRaster[nRow, nCol] - demRaster[row, col]
                edge = ((diagDist if (dRow and dCol) else csz) + max(dz, 0.) * upSlopePenalty
                        + distToFlow[nRow, nCol] * flowDistPenalty)
                if cost + edge < costGrid[nRow, nCol]:
                    costGrid[nRow, nCol] = cost + edge
                    previousCell[nRow, nCol] = (row, col)
                    heapq.heappush(queue, (cost + edge, nRow, nCol))
    if previousCell[goalCell[0], goalCell[1], 0] < 0 and tuple(goalCell) != tuple(startCell):
        return []
    cellPath = []
    row, col = int(goalCell[0]), int(goalCell[1])
    while row >= 0:
        cellPath.append((row, col))
        row, col = int(previousCell[row, col, 0]), int(previousCell[row, col, 1])
    cellPath.reverse()
    return cellPath


def getParabolicFit(cfg, avaProfile, dem):
    """fit a parabola on a set of (s, z) points

    first and last point match. Last constraint is given by fitOption (see below)

    Parameters
    -----------
    cfg: configParser
        configuration object with:
        fitOption: int
        how to optimize the fit
        0 minimize distance between points
        1 match the end slope
    avaProfile: dict
        profile to be extended
    dem: dict
        dem dict

    Returns
    --------
    parabolicFit: dict
        a, b, c coefficients of the parabolic fit (y = a*a*x + b*x + c)
    """
    s = avaProfile['s']
    sE = s[-1]
    z = avaProfile['z']
    z0 = z[0]
    zE = z[-1]
    # same start and end point, minimize distance between curves
    if cfg.getfloat('fitOption') == 0:
        SumNom = np.sum(s*(s-sE)*((zE-z0)/sE*s+z0-z))
        SumDenom = s*(s-sE)
        SumDenom = np.dot(SumDenom, SumDenom)
        a = - SumNom/SumDenom
        b = (zE-z0)/sE - a*sE
    elif cfg.getfloat('fitOption') == 1:
        angleProf, tmpProf, dsProf = gT.prepareAngleProfile(10, avaProfile, raiseWarning=False)
        r = avaProfile['s'] - avaProfile['s'][-1]
        resampleDistance = cfg.getfloat('nCellsSlope') * dem['header']['cellsize']
        pointsOfInterestLast = np.where(np.abs(r) < resampleDistance)
        slope = np.nansum(angleProf[pointsOfInterestLast])/np.size(pointsOfInterestLast)
        slope = -np.tan(np.radians(slope))
        a = (slope*sE + (z0 - zE))/(sE*sE)
        b = -slope - 2*(z0 - zE)/sE
    c = z0
    parabolicFit = {'a': a, 'b': b, 'c': c}
    return parabolicFit


def getSplitPoint(cfg, avaProfile, parabolicFit):
    """ find the split point corresponding to an avalanche profile, with parabolic fit and the slopeSplitPoint

    Parameters
    -----------
    cfg: configParser
        configuration object with

        - slopeSplitPoint: float, desired slope at split point (in degrees)
        - dsMin: float, threshold distance [m] for looking for the
        - split point (at least dsMin meters below split point angle)

    avaProfile: dict
        profile to be extended
    parabolicFit: dict
        a, b, c coeficients of the parabolic fit (y = a*a*x + b*x + c)

    Returns
    --------
    splitPoint: dict
        (x, y, z, zPra, s) at split point location.
    """
    indFirst = avaProfile['indStartMassAverage']
    indEnd = avaProfile['indEndMassAverage']
    s0 = avaProfile['s'][indFirst]
    sEnd = avaProfile['s'][indEnd]
    s = avaProfile['s']
    z = avaProfile['z']
    sNew = s - s0
    zPara = parabolicFit['a']*sNew*sNew+parabolicFit['b']*sNew+parabolicFit['c']
    parabolicProfile = {'s': sNew, 'z': zPara}

    anglePara, tmpPara, dsPara = gT.prepareAngleProfile(cfg.getfloat('slopeSplitPoint'), parabolicProfile,
                                                        raiseWarning=False)
    try:
        indSplitPoint = gT.findAngleProfile(tmpPara, dsPara, cfg.getfloat('dsMin'))
        splitPoint = {'x': avaProfile['x'][indSplitPoint], 'y': avaProfile['y'][indSplitPoint],
                      'z': z[indSplitPoint], 'zPara': zPara[indSplitPoint], 's': sNew[indSplitPoint]}
    except IndexError:
        noSplitPointFoundMessage = ('Automated split point generation failed as no point where slope is less than %s°'
                                    'was found, setting split point at the top. Correct split point manually.'
                                    % cfg.getfloat('slopeSplitPoint'))
        splitPoint = {'x': avaProfile['x'][0], 'y': avaProfile['y'][0],
                      'z': z[0], 'zPara': zPara[0], 's': sNew[0], 'isTopSplitPoint': True}
        log.warning(noSplitPointFoundMessage)
    if debugPlot:
        angleProf, tmpProf, dsProf = gT.prepareAngleProfile(cfg.getfloat('slopeSplitPoint'), avaProfile)
        debPlot.plotFindAngle(avaProfile, angleProf, parabolicProfile, anglePara, s0, sEnd, splitPoint, indSplitPoint)
    return splitPoint


def resamplePath(cfg, dem, avaProfile):
    """
    resample the path profile and keep track of indStartMassAverage and indEndMassAverage

    Parameters
    -----------
    cfg: configParser
        configuration object with

        - nCellsResample: int, resampling length is given by nCellsResample*demCellSize

    dem: dict
        dem dict
    avaProfile: dict
        profile to be resampled

    Returns
    --------
    avaProfile: dict
        resampled path profile
    """
    resampleDistance = cfg.getfloat('nCellsResample') * dem['header']['cellsize']
    indFirst = avaProfile['indStartMassAverage']
    indEnd = avaProfile['indEndMassAverage']
    s0 = avaProfile['s'][indFirst]
    sEnd = avaProfile['s'][indEnd]
    avaProfile, _ = gT.prepareLine(dem, avaProfile, distance=resampleDistance, Point=None)
    # make sure we get the good start and end point... prepareLine might make a small error on the s coord
    indFirst = np.argwhere(avaProfile['s'] >= s0 - resampleDistance/3)[0][0]
    # look for the first point in the extension and take the one before; if the extension is
    # shorter than a resample step, the mass averaged part reaches the last point
    indEndCandidates = np.argwhere(avaProfile['s'] >= sEnd + resampleDistance/3)
    indEnd = indEndCandidates[0][0]-1 if len(indEndCandidates) > 0 else np.size(avaProfile['s'])-1
    avaProfile['indStartMassAverage'] = indFirst
    avaProfile['indEndMassAverage'] = indEnd
    return avaProfile


def saveSplitAndPath(avalancheDir, simDFrow, splitPoint, avaProfileMass, dem):
    """
    Save avaPath and split point to directory

    Parameters
    -----------
    avalancheDir: pathlib
        avalanche directory pathlib path
    simDFrow: pandas series
        row of the simulation dataframe corresponding to the current simulation analyzed
    splitPoint: dict
        point dictionary
    avaProfileMass: dict
        line dictionary
    dem: dict
        com1DFA simulation dictionary
    Returns
    --------
    pathAB: pathlib
        file path to the saved shapefile for the mass-averaged path
    splitAB: pathlib
        file path to the saved shapefile for the split point
    """
    # put path back in original location
    if splitPoint != '':
        splitPoint['x'] = splitPoint['x'] + dem['originalHeader']['xllcenter']
        splitPoint['y'] = splitPoint['y'] + dem['originalHeader']['yllcenter']
    avaProfileMass['x'] = avaProfileMass['x'] + dem['originalHeader']['xllcenter']
    avaProfileMass['y'] = avaProfileMass['y'] + dem['originalHeader']['yllcenter']
    # get projection from release shp layer
    simName = simDFrow['simName']
    relName = cfgUtils.parseSimName(simName)["releaseName"]
    inProjection = pathlib.Path(avalancheDir, 'Inputs', 'REL', relName + '.prj')
    # save profile in Inputs
    pathAB = pathlib.Path(avalancheDir, 'Outputs', 'ana5Utils', 'DFAPath', 'massAvgPath_%s_AB_aimec' % simName)
    name = 'massAvaPath'
    shpConv.writeLine2SHPfile(avaProfileMass, name, pathAB)
    if inProjection.is_file():
        shutil.copy(inProjection, pathAB.with_suffix('.prj'))
    else:
        message = ('No projection layer for shp file %s' % inProjection)
        log.warning(message)
    log.info('Saved path to: %s', pathAB)
    if splitPoint != '':
        splitAB = pathlib.Path(avalancheDir, 'Outputs', 'ana5Utils', 'DFAPath', 'splitPointParabolicFit_%s_AB_aimec' % simName)
        name = 'parabolaSplitPoint'
        shpConv.writePoint2SHPfile(splitPoint, name, splitAB)
        if inProjection.is_file():
            shutil.copy(inProjection, splitAB.with_suffix('.prj'))
        log.info('Saved split point to: %s', splitAB)
    return pathAB,splitAB

def weightedAvgAndStd(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return (average, math.sqrt(variance))


def appendAverageStd(propList, avaProfile, particles, weights, naming=''):
    """ append averaged to path

    Parameters
    -----------
    propList: list
        list of properties to average and append
    avaProfile: dict
        path
    particles: dict
        particles dict
    weights: numpy array
        array of weights (same size as particles)
    naming: list
        optional - list of properties at source (if different than final naming in avaProfile)

    Returns
    --------
    avaProfile: dict
        averaged profile
    """
    propListNames = naming if naming != '' else propList
    for prop, propName in zip(propList, propListNames):
        avg, std = weightedAvgAndStd(particles[propName], weights)
        avaProfile[prop] = np.append(avaProfile[prop], avg)
        avaProfile[prop + 'std'] = np.append(avaProfile[prop + 'std'], std)
    return avaProfile
