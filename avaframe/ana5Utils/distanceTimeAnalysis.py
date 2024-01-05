""" functions to convert to a different coordinate system and
    produce a range-time diagram from simulation results
    options:
    1) range-time diagram from radar's field of view capped to this
    2) thalweg-time diagram from beta point of view along avalanche path
"""


import numpy as np
import logging
import pickle
import pathlib

# Local imports
from avaframe.in3Utils import cfgUtils
import avaframe.in3Utils.geoTrans as gT
import avaframe.in2Trans.rasterUtils as IOf
import avaframe.out3Plot.outDistanceTimeAnalysis as dtAnaPlots
import avaframe.ana3AIMEC.aimecTools as aT
import avaframe.com1DFA.com1DFA as com1DFA
import avaframe.in3Utils.fileHandlerUtils as fU

# create local logger
# change log level in calling module to DEBUG to see log messages
log = logging.getLogger(__name__)


def createThalwegTimeInfoFromSimResults(avalancheDir, cfgRangeTime, modName, index, simDF, dem):
    """ top level function to create mtiInfo dict from com Module simulation results
        required are flow variable fields for different time steps

        Parameters
        -----------
        avalancheDir: str or pathlib path
            path to avalanche directory
        cfgRangeTime: configparser object
            configuration settings for range time diagram
        modName: str
            name of computational module that has been used to create ava sim
        index: str
            index of current sim in simDF
        simDF: dataFrame
            dataframe with info on sim configuration
        dem: dict
            dictionary with simulation dem

        Returns
        --------
        cfgRangeTime: configparser object
            configuration settings for range time
        mtiInfo: dict
            dictionary with info for creating range time plots

    """

    log.info('Perform range-time diagram for simulation: %s' % index)

    # add simHash info
    cfgRangeTime['GENERAL']['simHash'] = index

    # setup required data
    mtiInfo = setupThalwegTimeDiagram(dem, cfgRangeTime)
    mtiInfo['plotTitle'] = 'thalweg-time diagram %s' % index
    mtiInfo['textbox'] = 'beta point: %.2f, %.2f' % (mtiInfo['betaPoint'][0],
                                                     mtiInfo['betaPoint'][1])

    # fetch all flow parameter result fields
    flowFieldsDir = pathlib.Path(avalancheDir, 'Outputs', modName, 'peakFiles', 'timeSteps')
    simNameSuffix = simDF['simName'].loc[index] + '_' + cfgRangeTime['GENERAL']['rangeTimeResType']
    flowFields = fU.fetchFlowFields(flowFieldsDir, suffix=simNameSuffix)

    if len(flowFields) == 0:
        fU.fileNotFoundMessage(('No flow variable results found in %s - consider first running avalanche simulations' %
                                flowFieldsDir))

    for flowField in flowFields:

        # read flow field data
        flowFieldDict = IOf.readRaster(flowField)
        flowF = flowFieldDict['rasterData']

        # extract avalanche front distance to radar and average values of range gates for mti plot
        mtiInfo = extractFrontAndMeanValuesTT(cfgRangeTime, flowF, dem['header'], mtiInfo)
        timeStep, _ = fetchTimeStepFromName(flowField)
        mtiInfo['timeList'].append(timeStep[0])

    return cfgRangeTime, mtiInfo


def getRadarLocation(cfg):
    """ fetch radar location and direction of of field of view coordinates

        Parameters
        ------------
        cfg: configparser object
            configuration settings

        Returns
        --------
        radarFov: numpy array
            array with x and y coordinates of radar location and point in direction of field of view

    """

    # fetch coordinate info
    locationInfo = cfg['GENERAL']['radarLocation'].split('|')

    if len(locationInfo) != 4:
        message = ('Radar location is invalid, required format x0|x1|y0|y1 not: %s' %
                   cfg['GENERAL']['radarLocation'])
        log.error(message)
        raise AssertionError(message)

    # set radar location and point in direction of field of view as list
    radarFov = np.asarray([[float(locationInfo[0]), float(locationInfo[1])], [float(locationInfo[2]),
                                                                              float(locationInfo[3])]])

    return radarFov


def setDemOrigin(demSims):
    """ reset original origin to simulation DEM - required if for ava sim is set to something different

        Parameters
        ------------
        demSims: dict
            dictionary with header and data of DEM used to run ava simulations
            must contain a key called originalHeader - where the original origin is given

        Returns
        --------
        demOriginal: dict
            updated DEM with xllcenter and yllcenter set to original origin
            cellsize and nrows and columns kept from simulation DEM
    """
    if 'originalHeader' not in demSims:
        message = 'DEM dictionary does not have originalHeader key - required to get original origin'
        log.error(message)
        raise KeyError(message)

    # fetch dem data and header - use simulation dem data but xllcenter und yllcenter from original dem
    demOriginal = {'header': demSims['originalHeader'], 'rasterData': demSims['rasterData']}

    return demOriginal


def setupRangeTimeDiagram(dem, cfgRangeTime):
    """ create setup for creating range-time diagrams from avalanche simulation results
        with respect to a radar's field of view

        Parameters
        -----------
        dem: dict
            DEM dictionary with header and raster data
        cfgRangeTime: configparser object
            configuration settings for range-time diagrams

        Returns
        --------
        rangeMasked: numpy masked array
            radar range retrieved from DEM and masked with radar's field of view
    """

    # load parameters from configuration
    rgWidth = cfgRangeTime['GENERAL'].getfloat('rgWidth')

    # fetch radar field of view coordinates
    radarFov = getRadarLocation(cfgRangeTime)

    # create masked array of radar range with DEM extent using radar field of view and range gates
    rangeMasked, rangeGates = radarMask(dem, radarFov, cfgRangeTime['GENERAL'].getfloat('aperture'),
                                        cfgRangeTime)

    # create range gates and initialise mti array
    rArray = np.ma.getdata(rangeMasked)
    nRangeGate = len(rangeGates)
    mti = np.zeros((nRangeGate, 1))

    # fetch field of view mask
    bmaskRadar = np.ma.getmask(rangeMasked)

    # add all info to mti dict
    mtiInfo = {'mti': mti, 'rangeGates': rangeGates, 'rArray': rArray, 'bmaskRadar': bmaskRadar,
               'rangeList': [], 'timeList': [], 'rangeMasked': rangeMasked, 'radarFov': radarFov,
               'demOriginal': dem, 'type': 'rangeTime', 'referencePointName': 'radar', 'DEM': dem}

    return mtiInfo


def radarMask(demOriginal, radarFov, aperture, cfgRangeTime):
    """ function to create masked array of radar range using dem and radar's field of view

        Parameters
        -----------
        dem: dict
            DEM dictionary with header info and rasterData
        radarFov: numpy array
            radars field of view min and max points
        aperture: float
            aperature angle
        cfgRangeTime: configparser object
            configuration settings here used: avalancheDir info

        Returns
        --------
        radarRange: numpy masked array
            masked array of radar range covering full DEM extent
        rangeGates: numpy array
            range gates
    """

    # fetch header info - required for creating coordinate grid
    xllc = demOriginal['header']['xllcenter']
    yllc = demOriginal['header']['yllcenter']
    ncols = demOriginal['header']['ncols']
    nrows = demOriginal['header']['nrows']
    cellSize = demOriginal['header']['cellsize']
    rasterdata = demOriginal['rasterData']

    # Set coordinate grid with given origin
    X, Y = gT.makeCoordinateGrid(xllc, yllc, cellSize, ncols, nrows)

    if (np.any(radarFov[0] < np.amin(X)) or np.any(radarFov[0] > np.amax(X))
            or np.any(radarFov[1] < np.amin(Y)) or np.any(radarFov[1] > np.amax(Y))):
        message = ('Radar location outside of DEM, x= %.2f, %.2f, y= %.2f, %.2f' %
                   (radarFov[0][0], radarFov[0][1], radarFov[1][0], radarFov[1][1]))
        log.error(message)
        raise AssertionError(message)

    # define radar field of view as dict
    radarpath = {'x': np.asarray(radarFov[0]),
                 'y': np.asarray(radarFov[1])}
    # add z-coordinate from dem data
    radarProfile = gT.projectOnRaster(demOriginal, radarpath, interp='bilinear')

    # radar location - position of radar [0] and point in direction of field of view [1]
    rX = radarProfile[0]['x'][0]
    rY = radarProfile[0]['y'][0]
    rZ = radarProfile[0]['z'][0]
    # a point that gives direction - point in direction of field of view
    rDx = radarProfile[0]['x'][1]
    rDy = radarProfile[0]['y'][1]
    rDz = radarProfile[0]['z'][1]

    # convert centerline of radar field of view to spherical coordinates
    rD, phiD, thetaD = gT.cartToSpherical(X=rDx-rX, Y=rDy-rY, Z=rDz-rZ)

    # domain in spherical coordinates with center beeing radar position
    r, phi, theta = gT.cartToSpherical(X=X-rX, Y=Y-rY, Z=rasterdata-rZ)

    # create mask where in spherical coordinates azimuth angle is bigger than field of view of radar
    # this is defined by the radar position where it is looking and the aperature angle
    maskPhi = ~np.logical_and(phi > phiD - aperture, phi < phiD + aperture)

    # compute radar range field (distance from radar -masked with field of view)
    # TODO: check if we need to mask also where there are nans in dem
    radarRange = np.ma.array(r, mask=maskPhi)

    # create range gates
    minR = int(np.floor(np.nanmin(radarRange)))
    maxR = int(np.ceil(np.nanmax(radarRange)))
    rgWidth = cfgRangeTime['GENERAL'].getfloat('rgWidth')
    rangeGates = np.arange(minR+rgWidth/2, maxR+rgWidth/2, rgWidth)

    return radarRange, rangeGates


def extractFrontAndMeanValuesRadar(cfgRangeTime, flowF, mtiInfo):
    """ extract average values of flow parameter result at certain distance intervals (range gates)
        add info to be used for colorcoding range-time diagram

        Parameters
        -----------
        cfgRangeTime: configparser object
            configuration settings for creating range-time diagram
        flowF: numpy array
            flow parameter result field
        mtiInfo: dict
            info here used: rArray, bmaskRadar rangeMasked, rangeGates, mti, timeList

        Returns
        --------
        mtiInfo: dict
            updated mtiInfo dict with info on mti values
    """

    # load parameters from configuration file and mtiInfo
    rgWidth = cfgRangeTime['GENERAL'].getfloat('rgWidth')
    threshold = cfgRangeTime['GENERAL'].getfloat('thresholdResult')
    rangeGates = mtiInfo['rangeGates']
    rArray = mtiInfo['rArray']
    rangeMasked = mtiInfo['rangeMasked']
    mti = mtiInfo['mti']
    mtiNew = np.zeros((len(rangeGates), 1))
    maxOrMean = cfgRangeTime['GENERAL']['maxOrMean']

    # mask range with radar field of view and treshold of flow variable result
    maskAva, bmaskAvaRadar, rMaskedAvaRadar = maskRangeFull(flowF, threshold, rangeMasked)

    # ++++++++Extract front location with respect to radar ++++++++++++++
    # get line of sight distance to identify front, use threshold to mask flowF
    # line of sight min of masked array
    losDistance = rMaskedAvaRadar.min()
    if isinstance(losDistance, np.ma.core.MaskedArray):
        losDistance = np.nan

    # update lists of time step and front location
    mtiInfo['rangeList'].append(losDistance)

    # +++++++Extract average values at range gates +++++++++
    # min and max radar range of masked radar range
    if not rMaskedAvaRadar.mask.all():
        #TODO why int?
        #minRAva = int(np.floor(rmaskAva_radar.min()))
        #maxRAva = int(np.ceil(rmaskAva_radar.max()))
        minRAva = rMaskedAvaRadar.min()
        maxRAva = rMaskedAvaRadar.max()

        # create array of capped range gates
        smallRangeGatesIndex = np.where((rangeGates >= minRAva) & (rangeGates <= maxRAva))[0]

        # loop over each range gate within visible part to populate mti array
        for indexRI in smallRangeGatesIndex:
            # create mask for range slice within range gate
            bmaskRange = ~np.logical_and(rArray > rangeGates[indexRI]-rgWidth/2,
                                         rArray < rangeGates[indexRI]+rgWidth/2)
            bmaskAvaRadarRangeslice = ~np.logical_and(~bmaskRange, ~bmaskAvaRadar)
            if np.any(~bmaskAvaRadarRangeslice):
                # only update if range_slice mask is not empty
                if maxOrMean.lower() == 'max':
                    mtiValue = np.amax(np.ma.array(np.ma.getdata(maskAva), mask=bmaskAvaRadarRangeslice))
                else:
                    mtiValue = np.mean(np.ma.array(np.ma.getdata(maskAva), mask=bmaskAvaRadarRangeslice))
                mtiNew[indexRI] = mtiValue
                if cfgRangeTime['PLOTS'].getboolean('debugPlot'):
                    dtAnaPlots.plotMaskForMTI(cfgRangeTime['GENERAL'], bmaskRange, bmaskAvaRadar,
                                              bmaskAvaRadarRangeslice, mtiInfo)
    else:
        log.debug('No avalanche data bigger threshold in masked radar range array')

    # add average values of this time step to full mti array
    if mtiInfo['timeList'] == []:
        mti = mtiNew
    else:
        mti = np.hstack((mti, mtiNew))

    # update mtiInfo dict
    mtiInfo['mti'] = mti

    return mtiInfo


def maskRangeFull(flowF, threshold, rangeMasked):
    """ mask range (already masked with radar field of view) also with avalanche result field
        where NOT above threshold

        Parameters
        ------------
        flowF: numpy array
            flow variable result field
        threshold: float
            threshold of result field - masked where values NOT above this threshold
        rangeMasked: numpy masked array
            range to radar location masked with radar field of view

        Returns
        --------
        maskAva: numpy mask
            mask where result field NOT above threshold
        bmaskAvaRadar: numpy mask
            mask where result field NOT above threshold and NOT in radar field of view
        rMaskedAvaRadar: numpy masked array
            masked range array with NOT in radar field of view and result field NOT above threshold
    """

    # get mask of results not above threshold
    maskAva = np.ma.masked_where(~(flowF > threshold), flowF)
    bmaskAva = np.ma.getmask(maskAva)

    # get mask of radar field of view
    maskRadarFOV = np.ma.getmask(rangeMasked)

    # and join to a mask with "visible part of avalanche" - mask where not field of view and not
    # ava result above threshold
    bmaskAvaRadar = ~np.logical_and(~maskRadarFOV, ~bmaskAva)
    # mask range with this mask to obtain 'visible part of avalanche' ranges
    rMaskedAvaRadar = np.ma.array(np.ma.getdata(rangeMasked), mask=bmaskAvaRadar)  # masked ranges

    return maskAva, bmaskAvaRadar, rMaskedAvaRadar


def setupThalwegTimeDiagram(dem, cfgRangeTime):
    """ initialize parameters and prerequisites for generating thalweg-time diagrams

        Parameters
        -----------
        dem: dict
            dictionary with info on dem header and data
        cfgRangeTime: confiparser object
            configuration settings of range-time diagram

        Returns
        --------
        mtiInfo: dict
            dictionary with arrays, lists for range-time diagram:
            mti for colorcoding (average values of range gates/ crossprofiles)
            timeList time step info
            rangeList range info
            rangeGates info on cross profile locations
            rangeRaster - info on distance from beta point in path following coordinate system
            betaPoint - coordinates of start of runout zone point
            betaPointAngle - angle of beta point used
    """

    # set required parameters from/to configuration
    avaDir = cfgRangeTime['GENERAL']['avalancheDir']
    cfgSetup = cfgRangeTime['GENERAL']
    resType = cfgRangeTime['GENERAL']['rangeTimeResType']
    cfgSetup['resType'] = resType

    # fetch info on avalanche path, split point, path to dem
    pathDict = {}
    pathDict = aT.readAIMECinputs(avaDir, pathDict, cfgSetup.getboolean('defineRunoutArea'), dirName='com1DFA')

    # generate data required to perform domain tranformation
    rasterTransfo = aT.makeDomainTransfo(pathDict, dem, dem['header']['cellsize'], cfgSetup)

    # tranform DEM data
    log.debug("Assigning dem data to deskewed raster")
    newRasterDEM = aT.transform(dem, 'dem', rasterTransfo, cfgSetup['interpMethod'])

    # create raster with info on distance from beta point
    # fetch info on distance of runout area start point index in s array
    indStartOfRunout = rasterTransfo['indStartOfRunout']
    # create range raster with respect to runout area start point
    # minus in upstream direction of runout area start point
    if cfgRangeTime['GENERAL']['sType'].lower() == 'parallel':
        rangeGates = rasterTransfo['sParallel'][:] - rasterTransfo['sParallel'][indStartOfRunout]
    elif cfgRangeTime['GENERAL']['sType'].lower() == 'projected':
        if cfgRangeTime['GENERAL'].getboolean('originStart'):
            rangeGates = rasterTransfo['s'][:]
        else:
            rangeGates = rasterTransfo['s'][:] - rasterTransfo['s'][indStartOfRunout]
    else:
        message = ('sType for tt-diagram is invalid, valid options are: projected and parallel' %
            cfgRangeTime['GENERAL']['sType'])
        log.error(message)
        raise AssertionError
    rangeRaster = np.repeat([rangeGates], newRasterDEM.shape[1], axis=0).T

    # create dictionary with info on front distance, mean values, range distance array
    mtiInfo = {'mti': '', 'rangeList': [], 'timeList': []}
    # mtiInfo['rangeGates'] = rasterTransfo['s'][indStartOfRunout] - rasterTransfo['s'][:]
    mtiInfo['rangeGates'] = rangeGates
    mtiInfo['betaPoint'] = [rasterTransfo['x'][indStartOfRunout], rasterTransfo['y'][indStartOfRunout]]
    mtiInfo['betaPointAngle'] = rasterTransfo['startOfRunoutAreaAngle']
    mtiInfo['rangeRaster'] = rangeRaster
    mtiInfo['rasterTransfo'] = rasterTransfo
    mtiInfo['type'] = 'thalwegTime'
    mtiInfo['z'] = rasterTransfo['z']
    mtiInfo['referencePointName'] = 'beta point'
    mtiInfo['sType'] = cfgRangeTime['GENERAL']['sType']

    return mtiInfo


def extractFrontAndMeanValuesTT(cfgRangeTime, flowF, demHeader, mtiInfo):
    """ extract avalanche front and mean values of flow parameter result field
        used for colorcoding range-time diagram

        Parameters
        ------------
        cfgRangeTime: configparser object
            configuration settings
        flowF: numpy nd array
            flow parameter result field
        demHeader: dict
            dictionary with info on DEM header used for locating fields
        mtiInfo: dict
            dictionary with arrays, lists for range-time diagram:
            mti for colorcoding (average values of range gates/ crossprofiles)
            timeList time step info
            rangeList range info
            rangeGates info on cross profile locations
            rangeRaster - info on distance from beta point in path following coordinate system
            rasterTransfo - information on domain transformation to path following coordinate system

        Returns
        --------
        mtiInfo: dict
            updated dictionary with info on average value along crossprofiles (mti)
            and distance to front from reference point along path (rangeList)
    """

    # setup input parameters
    rasterTransfo = mtiInfo['rasterTransfo']

    # add header to result field - required for coordinate transformation
    fieldFull = {'header': demHeader, 'rasterData': flowF}

    # transorm result field into avalanche path following coordinate system
    slRaster = aT.transform(fieldFull, 'field', rasterTransfo, cfgRangeTime['GENERAL']['interpMethod'])

    # fetch raster area and compute mean, max values for each cross-profile
    # TODO: average over cells – weighted with cell area (aimec function)
    rasterArea = rasterTransfo['rasterArea']
    maxaCrossMax, aCrossMax, aCrossMean = aT.getMaxMeanValues(slRaster, rasterArea)
    # use the max or the mean of each cross section
    if cfgRangeTime['GENERAL']['maxOrMean'].lower() == 'max':
        aCross = aCrossMax
    else:
        aCross = aCrossMean

    # add max or mean values for each cross-section for actual time step to mti values array
    # reshape is required as rows- max/mean values for each crossprofile and cols: time steps
    if mtiInfo['timeList'] == []:
        mtiInfo['mti'] = aCross.reshape(-1, 1)

    else:
        mtiInfo['mti'] = np.hstack((mtiInfo['mti'], aCross.reshape(-1, 1)))

    # extract avalanche front distance to reference point in path following coordinate system
    indStartOfRunout = rasterTransfo['indStartOfRunout']
    lindex = np.nonzero(slRaster > cfgRangeTime['GENERAL'].getfloat('thresholdResult'))[0]
    if lindex.any():
        cUpper = min(lindex)
        cLower = max(lindex)
        if mtiInfo['sType'].lower() == 'parallel':
            rangeValue = rasterTransfo['sParallel'][cLower] - rasterTransfo['sParallel'][indStartOfRunout]
        elif mtiInfo['sType'].lower() == 'projected':
            if cfgRangeTime['GENERAL'].getboolean('originStart'):
                rangeValue = rasterTransfo['s'][cLower]
            else:
                rangeValue = rasterTransfo['s'][cLower] - rasterTransfo['s'][indStartOfRunout]
        else:
            message = ('sType for tt-diagram is invalid, valid options are: projected and parallel' %
                cfgRangeTime['GENERAL']['sType'])
            log.error(message)
            raise AssertionError
        mtiInfo['rangeList'].append(rangeValue)
        # if animation plot shall be created add transformation info to mtiInfo dict for plots
        if cfgRangeTime['PLOTS'].getboolean('animate'):
            mtiInfo['slRaster'] = slRaster
            mtiInfo['rasterTransfo'] = rasterTransfo
            mtiInfo['cLower'] = cLower
    else:
        cLower = np.nan
        mtiInfo['rangeList'].append(np.nan)
        if cfgRangeTime['PLOTS'].getboolean('animate'):
            mtiInfo['slRaster'] = slRaster
            mtiInfo['rasterTransfo'] = rasterTransfo
            mtiInfo['cLower'] = cLower

    # plot avalanche front and transformed raster field
    if cfgRangeTime['PLOTS'].getboolean('debugPlot'):
        dtAnaPlots.plotRangeRaster(slRaster, rasterTransfo, cfgRangeTime['GENERAL'],
                                   mtiInfo['rangeRaster'], cLower)

    return mtiInfo


def initializeRangeTime(modName, cfg, dem, simHash, configDir):
    """ initialize generation of range-time diagram for visualizing simulation data

        Parameters
        -----------
        modName:
            module name
        cfg: configparser object
            configuration settings from computational module
        dem: dict
            dictionary with DEM header and data
        simHash: str
            unique simulation ID
        configDir: str or pathlib path
            path to configuration directory - optional if not provided has to be empty string

        Returns
        --------
        mtiInfo: dict
            info on time steps (timeList) and distance (rangeList) of avalanche front
            averaged values of result parameter (mti) for each range gate (rangeGates) for colorcoding
            x, y origin of DEM, cellSize and plotTitle
            optional info on masks for radar field of view (rangeMasked), etc.
    """

    # fetch configuration and add info
    cfgRangeTime = cfgUtils.getModuleConfig(modName, configDir)

    cfgRangeTime['GENERAL']['tEnd'] = cfg['GENERAL']['tEnd']
    cfgRangeTime['GENERAL']['avalancheDir'] = cfg['GENERAL']['avalancheDir']
    cfgRangeTime['GENERAL']['simHash'] = simHash
    # fetch time steps for creating range time diagram
    dtRangeTime = fU.splitTimeValueToArrayInterval(cfgRangeTime['GENERAL']['distanceTimeSteps'],
                                                   cfg['GENERAL'].getfloat('tEnd'))

    if cfg['VISUALISATION'].getboolean('TTdiagram'):
        mtiInfo = setupThalwegTimeDiagram(dem, cfgRangeTime)
        mtiInfo['plotTitle'] = 'thalweg-time diagram %s' % simHash
        mtiInfo['textbox'] = 'beta point: %.2f, %.2f' % (mtiInfo['betaPoint'][0],
                                                         mtiInfo['betaPoint'][1])
    else:
        mtiInfo = setupRangeTimeDiagram(dem, cfgRangeTime)
        mtiInfo['plotTitle'] = 'range-time diagram %s' % simHash

    mtiInfo['xOrigin'] = dem['header']['xllcenter']
    mtiInfo['yOrigin'] = dem['header']['yllcenter']
    mtiInfo['cellSize'] = dem['header']['cellsize']
    mtiInfo['dem'] = dem

    return mtiInfo, dtRangeTime, cfgRangeTime


def fetchRangeTimeInfo(cfgRangeTime, cfg, dtRangeTime, t, demHeader, fields, mtiInfo):
    """ determine avalanche front and average values of flow parameter along path
    update mtiInfo dictionary with this information

    Parameters
    -----------
    cfgRangeTime: configparser
        configuration settings for range-time diagram
    cfg: configParser object
        configuration settings of computational module
    dtRangeTime: list
        list of time steps where avalanche front shall be exported
    t: float
        actual simulation time step
    demHeader: dict
        dictionary with DEM header
    fields: dict
        dictionary with flow parameter result fields
    mtiInfo: dict
        info on time steps (timeList) and distance (rangeList) of avalanche front
        averaged values of result parameter (mti) for each range gate (rangeGates) for colorcoding
        optional info on masks for radar field of view (rangeMasked), etc.

    Returns
    --------
    mtiInfo: dict
        updated dictionary
    dtRangeTime: list
        updated list of time steps
    """

    # load result type for fields
    rangeTimeResType = cfgRangeTime['GENERAL']['rangeTimeResType']

    # log info
    log.debug('Processing frame %d of time step: %.2f for range-time plot' % (len(mtiInfo['timeList'])+1, t))

    # extract front and average values of result parameter
    if cfg['VISUALISATION'].getboolean('TTdiagram'):
        mtiInfo = extractFrontAndMeanValuesTT(cfgRangeTime, fields[rangeTimeResType], demHeader, mtiInfo)

    else:
        # extract front in simulation result for each time step and average values along range gates
        # for colorcoding range time plot
        mtiInfo = extractFrontAndMeanValuesRadar(cfgRangeTime, fields[rangeTimeResType], mtiInfo)

    # append time step info
    mtiInfo['timeList'].append(t)

    # remove saving time steps that have already been saved
    dtRangeTime = com1DFA.updateSavingTimeStep(dtRangeTime, cfgRangeTime['GENERAL'], t)

    return mtiInfo, dtRangeTime


def exportData(mtiInfo, cfgRangeTime, modName):
    """ save all info about range, time steps, average values to pickle for producing plots

        Parameters
        ------------
        mtiInfo: dict
            dictionary with rangeList, timeList, mti (average values along range gates or
            cross profiles)
        cfgRangeTime: configparser
            configuration settings of distance time
        modName: str
            name of computational module
    """

    # create path for saving
    dictPath = pathlib.Path(cfgRangeTime['GENERAL']['avalancheDir'], 'Outputs', modName,
                            'distanceTimeAnalysis')
    fU.makeADir(dictPath)
    outDict = dictPath / ('mtiInfo_%s.p' % cfgRangeTime['GENERAL']['simHash'])

    # append configuration info to dict
    cfgDict = cfgUtils.convertConfigParserToDict(cfgRangeTime)
    mtiInfo['configurationSettings'] = cfgDict

    # write dictionary to pickle
    with open(outDict, 'wb') as dictRangeTime:
        pickle.dump(mtiInfo, dictRangeTime)


def importMTIData(avaDir, modName, inputDir='', simHash=''):
    """ import mtiInfo data for creating range time plots, if no inputDir is provided,
        pickles are fetched from avaDir/Outputs/modName/distanceTimeAnalysis
        multiple pickles allowed- these carry also configuration info for distanceTimeAnalysis

        Parameters
        -----------
        avaDir: pathlib path or str
            path to avalanche directory
        modName: str
            name of computational module that has been used to produce flow variable fields
        inputDir: pathlib path or str
            optional: path to pickle location
        simHash: str
            optional simulation ID

        Returns
        --------
        mtiInfoDicts: list
            list of mtiInfo dictionaries where key name has been added with file Path
    """
    if inputDir == '':
        inputDir = pathlib.Path(avaDir, 'Outputs', modName, 'distanceTimeAnalysis')
    else:
        # make sure it is a pathlib path
        inputDir = pathlib.Path(inputDir)

    # fetch all files
    if simHash == '':
        mtiInfoPickles = list(inputDir.glob('*.p'))
    else:
        mtiInfoPickles = list(inputDir.glob('*%s.p' % simHash))

    if len(mtiInfoPickles) == 0:
        fU.fileNotFoundMessage(('No mtiInfo dictionary found in %s - consider first running avalanche simulations' %
                                inputDir))

    # create list of all mtiInfo dicts found
    mtiInfoDicts = []
    for infoD in mtiInfoPickles:
        with open(infoD, 'rb') as infoDict:
            mtiDict = pickle.load(infoDict)
            mtiDict['name'] = infoD
            mtiInfoDicts.append(mtiDict)

    return mtiInfoDicts


def fetchTimeStepFromName(pathNames):
    """ split path name to fetch info on time step

        Parameters
        -----------
        pathName: pathlib path or list of paths
            path to file including time step info as _t%.2f_

        Returns
        -------
        timeStep: list
            actual time steps
        indexTime: numpy array
            index of timeStep list to order them in ascending order
    """

    if isinstance(pathNames, list) is False:
        pathNames = [pathNames]

    timeSteps = []
    for fName in pathNames:
        timeSteps.append(float(fName.stem.split('_t')[1]))

    indexTime = np.argsort(np.array(timeSteps))

    return timeSteps, indexTime


def approachVelocity(mtiInfo):
    """ compute maximal approach velocity based on front location (range) and time step

        - cleans nan in range
        - neglects anything behind maximal runout
        - neglects non-unique range values, e.g. the front did not move
        - cleans approach velocity: velocity in cell under test can not be higher
            than the double of the mean from the surrounding cells


        Parameters
        -----------
        mtiInfo: dict
            info on distance to front and time steps

        Returns
        --------
        maxVel: float
            max value of approach velocity
        rangeVel: float
            range of max value of approach velocity
        timeVel: float
            time of max value of approach velocity
    """

    # load lists
    timeList = mtiInfo['timeList']
    rangeList = mtiInfo['rangeList']

    # as these might be not in ascending order timestep wise - order first
    timeListSorted = np.asarray(timeList)[np.argsort(timeList)]
    # convert to only positive distance values (move reference point to end of ava)
    # to get correct distance increments for velocity computation
    # nanmin required as if FV is zero range is nan
    if mtiInfo['type'] == 'rangeTime':
        # if in rangeList is measured in line of sight to the radar, in order to find the max runout close to the end of
        # the array, reference point is moved to closest found distance to radar and measured from start towards radar
        rangeListSorted = (np.asarray(rangeList)[np.argsort(timeList)] - abs(np.nanmax(np.asarray(rangeList)))) * -1.
    else:
        rangeListSorted = np.asarray(rangeList)[np.argsort(timeList)] + abs(np.nanmin(np.asarray(rangeList)))
    maxVel = 0.0
    rangeVel = 0.0
    timeVel = 0.0

    # remove nans in range
    nanIdx = np.argwhere(~np.isnan(rangeListSorted)).squeeze()
    # remove anything behind runout
    rmaxIdx = np.arange(0, np.nanargmax(rangeListSorted)+1)
    idx = np.intersect1d(nanIdx, rmaxIdx)

    rangeListSortedSmall = rangeListSorted[idx]
    timeListSortedSmall = timeListSorted[idx]
    # remove non-unique range values
    rangeListSortedUnique, uniqueIdx = np.unique(rangeListSortedSmall.round(decimals=2), return_index=True)
    timeListSortedUnique = timeListSortedSmall[uniqueIdx]

    # calculate approach velocity
    appVel = np.diff(rangeListSortedUnique)/np.diff(timeListSortedUnique)
    rangeAppVel = rangeListSortedUnique[0:-1]
    timeAppVel = timeListSortedUnique[0:-1]

    # clean approach velocity: Idea is that the velocity in one cell can not be bigger
    # than the double of the mean in the surrounding cells
    idxAppVel = []
    for i in range(len(appVel)-1):
        if i == 0:  # first cell
            vSurrounding = appVel[1]
        elif i == len(appVel)-1:  # last cell
            vSurrounding = appVel[i-1]
        else:  # other cells, take mean
            vSurrounding = (appVel[i-1] + appVel[i+1]) / 2
        if appVel[i] <= 2*vSurrounding:
            idxAppVel.append(i)
    appVelClean = appVel[idxAppVel]
    rangeAppVelClean = rangeAppVel[idxAppVel]
    timeAppVelClean = timeAppVel[idxAppVel]

    idxMaxVel = np.argmax(appVelClean)
    maxVel = appVelClean[idxMaxVel]
    if mtiInfo['type'] == 'rangeTime':
        rangeVel = rangeAppVelClean[idxMaxVel] * -1. + abs(np.nanmax(np.asarray(rangeList)))
    else:
        rangeVel = rangeAppVelClean[idxMaxVel] - abs(np.nanmin(np.asarray(rangeList)))
    timeVel = timeAppVelClean[idxMaxVel]

    return maxVel, rangeVel, timeVel
