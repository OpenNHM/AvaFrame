"""
    Main logic for AIMEC post processing
"""

import logging
import pathlib
import numpy as np
import pandas as pd
import copy


# Local imports
import avaframe.in2Trans.shpConversion as shpConv
import avaframe.in2Trans.rasterUtils as IOf
from avaframe.in3Utils import cfgUtils
from avaframe.in3Utils import cfgHandling
import avaframe.in1Data.getInput as gI
import avaframe.in3Utils.fileHandlerUtils as fU
import avaframe.in3Utils.geoTrans as geoTrans
import avaframe.com1DFA.DFAtools as DFAtools
import avaframe.out3Plot.outAIMEC as outAimec
import avaframe.out3Plot.plotUtils as pU


# create local logger
log = logging.getLogger(__name__)

# -----------------------------------------------------------
# Aimec read inputs tools
# -----------------------------------------------------------


def readAIMECinputs(avalancheDir, pathDict, defineRunoutArea, dirName='com1DFA'):
    """ Read inputs for AIMEC postprocessing

    Reads the required geometry files location for AIMEC postpocessing
    given an avalanche directory; avalanche path, DEM
    - optional requirement: splitPoint if start of runout area is defined according to startOfRunoutAreaAngle

    Parameters
    ----------
    avalancheDir : str
        path to directory of avalanche to analyze
    pathDict: dict
        dictionary with info required e.g. reference sim name, comparison type, color variation info
    defineRunoutArea: bool
        if True also splitPoint is fetched
    dirName: str
        name of desired results directory (avalancheDir/Outputs/dirName)

    Returns
    -------
    pathDict : dict
        updated dictionary with path to geometry input data
    """

    refDir = pathlib.Path(avalancheDir, 'Inputs', 'LINES')
    profileLayer = list(refDir.glob('*aimec*.shp'))
    try:
        message = ('There should be exactly one path_aimec.shp file containing the avalanche path in %s/Inputs/LINES/' %
                   avalancheDir)
        assert len(profileLayer) == 1, message
    except AssertionError:
        raise
    pathDict['profileLayer'] = profileLayer[0]

    if defineRunoutArea:
        refDir = pathlib.Path(avalancheDir, 'Inputs', 'POINTS')
        splitPointLayer = list(refDir.glob('*.shp'))
        try:
            message = 'There should be exactly one .shp file containing the split point in %s/Inputs/POINTS/' % avalancheDir
            assert len(splitPointLayer) == 1, message
        except AssertionError:
            raise
        pathDict['splitPointSource'] = splitPointLayer[0]
    else:
        pathDict['splitPointSource'] = None

    refDir = pathlib.Path(avalancheDir, 'Inputs')
    # check for DEM
    if 'demFileName' not in pathDict.keys():
        demSource = list(refDir.glob('*.asc')) + list(refDir.glob('*.tif'))
    elif pathDict['demFileName'] == '':
        demSource = list(refDir.glob('*.asc')) + list(refDir.glob('*.tif'))
    else:
        demSource = list(refDir.glob(pathDict['demFileName']))
    try:
        assert len(demSource) == 1, 'There should be exactly one topography .asc file in %s/Inputs/' % avalancheDir
    except AssertionError:
        raise
    pathDict['demSource'] = demSource[0]

    if dirName != '':
        pathResult = pathlib.Path(avalancheDir, 'Outputs', 'ana3AIMEC', dirName)
    else:
        pathResult = pathlib.Path(avalancheDir, 'Outputs', 'ana3AIMEC')
    pathDict['pathResult'] = pathResult

    # check for reference data
    referenceDir= pathlib.Path(avalancheDir, 'Inputs')

    referenceTypes = {"referenceLine": "LINE", "referencePolygon": "POLY",'referencePoint': 'POINT'}
    for refType in referenceTypes:
        referenceFile, availableFile, _ = gI.getAndCheckInputFiles(
            referenceDir,
            "REFDATA",
            referenceTypes[refType],
            fileExt="shp",
            fileSuffix=referenceTypes[refType],
        )
        if availableFile == 'Yes':
            # add file paths to pathDict
            pathDict[refType] = [referenceFile]
        else:
            pathDict[refType] = []

    projectName = pathlib.Path(avalancheDir).stem
    pathDict['projectName'] = projectName
    pathName = profileLayer[0].stem
    pathDict['pathName'] = pathName
    pathDict['avalancheDir'] = avalancheDir

    return pathDict


def fetchReferenceSimNo(avaDir, inputsDF, comModule, cfg, inputDir=''):
    """ Define reference simulation used for aimec analysis.

        if the configuration files are available and a varParList is provided, the simulations
        are ordered and the referenceSimValue is used to define the reference.

        otherwise, if a referenceSimName is provided, the reference is based on this name

        otherwise, the first file found is used as reference

        Parameters
        -----------
        avaDir : str
            path to avalanche directory
        inputsDF: dataFrame
            dataFrame with simulations to analyze and path to result files
        comModule: str
            computational module used to produce the results to analyze
        cfg: configParser object
            configuration for aimec - referenceSimValue, varParList used here
        inputDir: str or pathlib path
            optional- directory where peak files are located, if '',
            avaDir/Outputs/comMod/peakFiles is set

        Returns
        --------
        refSimRowHash: str
            dataframe hash (not simHash) of the simulation used as reference
        refSimName: str
            name of the simulation used as reference
        inputsDF:dataFrame
            dataFrame with simulations to analyze and path to result files
            If com1DFA = comModule and a variation parameter was specified, the
            com1DFA configuration is merged to the inputsDF
        colorVariation: boolean
            True if a color variation should be applied in the plots
        valRef: str
            value of vapParList[0] used to define reference sim
    """

    cfgSetup = cfg['AIMECSETUP']
    if inputDir != '':
        inputDir = pathlib.Path(inputDir)
    else:
        inputDir = pathlib.Path(avaDir, 'Outputs', comModule, 'peakFiles')
    if not inputDir.is_dir():
        message = 'Input directory %s does not exist - check anaMod' % inputDir
        log.error(message)
        raise NotADirectoryError(message)

    referenceSimValue = cfgSetup['referenceSimValue']
    referenceSimName = cfgSetup['referenceSimName']
    colorVariation = False
    # look for a configuration
    try:
        # load dataFrame for all configurations
        configurationDF = cfgUtils.createConfigurationInfo(avaDir, comModule=comModule)
        # Merge inputsDF with the configurationDF. Make sure to keep the indexing from inputs and to merge on 'simName'
        inputsDF = inputsDF.reset_index().merge(configurationDF, on=["simName"]).set_index("index")
        configFound = True
    except (NotADirectoryError, FileNotFoundError) as e:
        if cfgSetup['varParList'] != '' and (any(item in inputsDF.columns.tolist() for item in cfgSetup['varParList'].split('|')) == False):
            message = ('No configuration directory found. This is needed for sorting simulation according to '
                       '%s' % cfgSetup['varParList'])
            raise (message)
        elif cfgSetup['varParList'] != '' and any(item in inputsDF.columns.tolist() for item in cfgSetup['varParList'].split('|')):
            configFound = True
        else:
            configFound = False

            # filter simulations
    if cfg.has_section('FILTER'):
        parametersDict = fU.getFilterDict(cfg, 'FILTER')
        simNameList = cfgHandling.filterSims(avaDir, parametersDict, specDir='')
        inputsDF = inputsDF[inputsDF['simName'].isin(simNameList)]
        log.info('%d simulations found matching filter criteria' % inputsDF.shape[0])

    # set reference value to empty string
    valRef = ''
    # if the simulations come from com1DFA, it is possible to order the files and define a reference
    if configFound and cfgSetup['varParList'] != '':
        # fetch parameters that shall be used for ordering
        varParList = cfgSetup['varParList'].split('|')
        # order simulations
        varParList, inputsDF = cfgHandling.orderSimulations(varParList, cfgSetup.getboolean('ascendingOrder'), inputsDF)
        colorVariation = True
        # now look for the reference
        if referenceSimValue != '':
            # if a referenceSimValue is provided, find the corresponding simulation
            # get the value of the first parameter used for ordering (this will be usefull for colorcoding in plots)
            refSimRowHash, refSimName, valRef = defineRefOnSimValue(referenceSimValue, varParList, inputsDF)

        elif cfgSetup['referenceSimName'] != '':
            # else if a referenceSimName is provided, find the corresponding simulation - set
            # simulation with referenceSimName in name as referene simulation
            refSimRowHash, refSimName = defineRefOnSimName(referenceSimName, inputsDF)
        else:
            # if no referenceSimValue is provided, we assume the first simulation after reordering is the reference
            # reference simulation
            refSimRowHash = inputsDF.index[0]
            refSimName = inputsDF.loc[refSimRowHash, 'simName']
            log.info(('Reference simulation is the first simulation after reordering according to %s in ascending order'
                      '= %s and corresponds to simulation %s')
                     % (varParList, cfgSetup.getboolean('ascendingOrder'), refSimName))

    elif referenceSimName != '':
        # else if a referenceSimName is provided, find the corresponding simulation - set
        # simulation with referenceSimName in name as referene simulation
        refSimRowHash, refSimName = defineRefOnSimName(referenceSimName, inputsDF)

    else:
        # if no ordering is done, no referenceSimValue nor referenceSimName are given, take the first simulation
        # as reference
        refSimRowHash = inputsDF.index[0]
        refSimName = inputsDF.loc[refSimRowHash, 'simName']
        message = ('No information on how to define the reference was provided, taking simulation %s as reference.'
                   % refSimName)
        log.warning(message)

    return refSimRowHash, refSimName, inputsDF, colorVariation, valRef


def defineRefOnSimValue(referenceSimValue, varParList, inputsDF):
    """ Search for reference based on configuration parameter and value

     Raise an error if no simulation is found

    Parameters
    ----------
    referenceSimValue : str
        reference value to use to define the reference
    varParList: list
        list of parameter used for ordering the simulations
    inputsDF: dataFrame
        simulation dataFrame (with configuration parameters includeds)

    Returns
    -------
    refSimRowHash : index
        dataframe hash (not simHash) of the simulation used as reference
    refSimName : str
        name of THE reference simulation
    valRef: str
        value of vapParList[0] used to define reference sim
    """
    sortingParameter = inputsDF[varParList[0]].to_list()
    # if a simulation has an empty field for varParList[0], we get a nan or empty string
    # but the type of this no data should be the same as the other fields in the column
    # get the type ot this parameter
    typeCP = type(sortingParameter[0])
    try:
        if typeCP == str:
            sortingValues = [x.lower() for x in sortingParameter]
            # look for matching string (case unsensitive)
            indexRef = sortingValues.index(typeCP(referenceSimValue.lower()))
            valRef = sortingParameter[indexRef]
        elif typeCP in [float, int]:
            # check if thicknessValue read from shp
            if np.isnan(sortingParameter).any() and varParList[0] in ['relTh', 'entTh', 'secondaryRelTh']:
                sortingParameter = inputsDF[(varParList[0] + '0')].to_list() + sortingParameter
            colorValues = np.asarray(sortingParameter)
            # look for closest value
            indexRef = np.nanargmin(np.abs(colorValues - typeCP(referenceSimValue)))
            valRef = colorValues[indexRef]
        else:
            indexRef = sortingParameter.index(typeCP(referenceSimValue))
            valRef = sortingParameter[indexRef]
        # there might be multiple simulations matching the referenceSimValue, we take the first one
        refSimRowHash = inputsDF[inputsDF[varParList[0]] == valRef].index
        refSimRowHash, refSimName = checkMultipleSimFound(refSimRowHash, inputsDF)
        log.info(('Reference simulation is based on %s = %s - closest value '
                  'found is: %s and corresponds to simulation %s')
                 % (varParList[0], referenceSimValue, str(valRef), refSimName))
    except ValueError:
        message = 'Did not find any simulation matching %s = %s.' % (varParList[0], referenceSimValue)
        log.error(message)
        raise ValueError(message)
    return refSimRowHash, refSimName, str(valRef)


def defineRefOnSimName(referenceSimName, inputsDF):
    """ Search for reference based on simulation name

    Raise an error if no simulation is found

    Parameters
    ----------
    referenceSimName : str
        string to look for in the simulations name
    inputsDF: dataFrame
        simulation dataFrame

    Returns
    -------
    refSimRowHash : index
        dataframe hash (not simHash) of the simulation used as reference
    refSimName : str
        name of THE reference simulation
    """
    refSimRowHash = inputsDF.index[inputsDF['simName'].str.contains('%s' % referenceSimName).to_list()]
    if len(refSimRowHash) == 0:
        message = ('Found no simulation matching the provided referenceSimName = %s.'
                   % referenceSimName)
        log.error(message)
        raise IndexError(message)
    refSimRowHash, refSimName = checkMultipleSimFound(refSimRowHash, inputsDF)
    log.info('Reference simulation is based on provided simName: %s and corresponds to simulation %s'
             % (referenceSimName, refSimName))
    return refSimRowHash, refSimName


def checkMultipleSimFound(refSimRowHash, inputsDF, error=False):
    """ Check if more than one file can be chosen as reference

    Log warning or error if it is the case

    Parameters
    ----------
    refSimRowHash : index
        dataframe hash (not simHash) of the simulation used as reference
    inputsDF: dataFrame
        simulation dataFrame
    error: bool
        if True raise an error if multiple simulations where found, otherwise a simple warning

    Returns
    -------
    refSimRowHash : index
        dataframe hash (not simHash) of the simulation used as reference
    refSimName : str
        name of THE reference simulation
    """
    if len(refSimRowHash) > 1:
        if not error:
            message = ('Found multiple simulations (%s) matching the reference criterion, '
                       'taking the first one as reference' % len(refSimRowHash))
            log.warning(message)
        else:
            message = ('Found multiple simulations (%s) matching the reference criterion, '
                       'there should be only one reference' % len(refSimRowHash))
            log.error(message)
            raise NameError(message)
    refSimRowHash = refSimRowHash[0]
    refSimName = inputsDF.loc[refSimRowHash, 'simName']
    return refSimRowHash, refSimName


def checkAIMECinputs(cfgSetup, pathDict):
    """ Check inputs before running AIMEC postprocessing

    Make sure that the available data satisfies what is required in the ini file

    Parameters
    ----------
    cfgSetup : configParser
        aimec configuration
    pathDict: dict
        aimec input dictionary (path to inputs and refSimName and hash)

    Returns
    -------
    pathDict : dict
        aimec input dictionary updated with the resTypeList (list of result file types to take into account)
    """
    # check that the resTypes and runoutResType asked for in the ini file are available for all simulations
    # if not, raise an error
    # what we have for all simulations
    resTypeList = pathDict['resTypeList']

    print("resType")
    # what we need for all simulations
    if cfgSetup['resTypes'] != '':
        # required res types
        resTypesWanted = cfgSetup['resTypes'].split('|')
    else:
        resTypesWanted = copy.deepcopy(pathDict['resTypeList'])

    # add the runout res type to what we need
    resTypesWanted.append(cfgSetup['runoutResType'])
    resTypesWanted = set(resTypesWanted)
    # check for differences
    match = resTypesWanted & set(resTypeList)  # what is in both
    diff = list(resTypesWanted.difference(match))  # what is in resTypesWanted but not in both
    if diff != []:
        message = '%s result type should be available for all simulations to analyze.' % diff
        log.error(message)
        raise FileNotFoundError(message)
    else:
        pathDict['resTypeList'] = resTypesWanted
    log.info('Analyzing %s results types. Runout based on %s result type' %
             (pathDict['resTypeList'], cfgSetup['runoutResType']))
    return pathDict


def computeCellSizeSL(cfgSetup, refCellSize):
    """ Get the new (s, l) coordinate cell size
        read by default the reference result file cell size.
        If a 'cellSizeSL' is specified in cfgSetup then use this one

        Parameters
        -----------
        refCellSize: float
            cell size of reference simulation
        cfgSetup: configParser object
            configuration for aimec - with field cellSizeSL

        Returns
        --------
        cellSizeSL: float
            cell size to be used for the (s, l) coordinates
    """
    if cfgSetup['cellSizeSL'] == '':
        cellSizeSL = float(refCellSize)
        log.info('cellSizeSL is read from the reference header and is : %.2f m' % cellSizeSL)
    else:
        try:
            cellSizeSL = cfgSetup.getfloat('cellSizeSL')
            log.info('cellSizeSL is read from the configuration file and is : %.2f m' % cellSizeSL)
        except ValueError:
            message = ('cellSizeSL is read from the configuration file but should be a number, you provided: %s'
                       % cfgSetup['cellSizeSL'])
            log.error(message)
            raise ValueError(message)

    return cellSizeSL


def makeDomainTransfo(pathDict, dem, refCellSize, cfgSetup):
    """ Make domain transformation

    This function returns the information about the domain transformation
    Data given on a regular grid is projected on a nonuniform grid following
    a polyline to end up with "straightend raster"

    Parameters
    ----------
    pathDict : dict
        dictionary with paths to dem and lines for Aimec analysis
    dem: dict
        dem dictionary with header and raster data
    refCellSize: float
        cell-size corresponding to reference
    cfgSetup : configparser
        configparser with ana3AIMEC settings defined in ana3AIMECCfg.ini
        regarding domain transformation (domain width w, and new cellsize,
        startOfRunoutAreaAngle or interpolation method and
        referenceFile to get header info)

    Returns
    -------
    rasterTransfo: dict
        domain transformation information:
            gridx: 2D numpy array
                x coord of the new raster points in old coord system
            gridy: 2D numpy array
                y coord of the new raster points in old coord system
            s: 1D numpy array
                new coord system in the polyline direction
            l: 1D numpy array
                new coord system in the cross direction
            x: 1D numpy array
                coord of the resampled polyline in old coord system
            y: 1D numpy array
                coord of the resampled polyline in old coord system
            rasterArea: 2D numpy array
                real area of the cells of the new raster
            indStartOfRunout: int
                index for start of the runout area (in s)
                if defineRunoutArea is False - indStartOfRunout=0 (start of thalweg)
    """
    w = cfgSetup.getfloat('domainWidth')
    # get the cell size for the (s, l) raster
    cellSizeSL = computeCellSizeSL(cfgSetup, refCellSize)
    # Initialize transformation dictionary
    rasterTransfo = {}
    rasterTransfo['domainWidth'] = w
    rasterTransfo['cellSizeSL'] = cellSizeSL
    # read avaPath
    avaPath, splitPoint = setAvaPath(pathDict, dem)
    rasterTransfo['avaPath'] = avaPath
    rasterTransfo['splitPoint'] = splitPoint

    # Get new Domain Boundaries DB
    # input: ava path
    # output: Left and right side points for the domain
    rasterTransfo = geoTrans.path2domain(avaPath, rasterTransfo)

    # Make transformation matrix
    rasterTransfo = makeTransfoMat(rasterTransfo)

    # calculate the real area of the new cells as well as the scoord
    dem = geoTrans.getNormalMesh(dem, 1)
    dem = DFAtools.getAreaMesh(dem, 1)
    rasterTransfo = getSArea(rasterTransfo, dem)

    # put back the scale due to the desired cellsize and get x,y of resample avapath
    rasterTransfo = scalePathWithCellSize(rasterTransfo, cellSizeSL)

    # get z coordinate of the center line
    rasterTransfo, _ = geoTrans.projectOnRaster(dem, rasterTransfo)

    # add surface parallel coordinate (sParallel)
    rasterTransfo = addSurfaceParalleCoord(rasterTransfo)

    # add info on start of runout area
    rasterTransfo = findStartOfRunoutArea(dem, rasterTransfo, cfgSetup, splitPoint)

    # add dem info to rasterTransfo
    rasterTransfo['dem'] = dem

    return rasterTransfo


def splitSection(DB, i):
    """ Splits the ith segment of domain boundary DB in the s direction
    (direction of the path)

    Parameters
    ----------
    DB: dict
        domain Boundary dictionary:
            DBXl: 1D numpy array
                x coord of the left boundary
            DBXr: 1D numpy array
                x coord of the right boundary
            DBYl: 1D numpy array
                y coord of the left boundary
            DBYr: 1D numpy array
                y coord of the right boundary
    i: int
        index of the segment of DB to split

    Returns
    -------
    (x,y) coordinates of the ith left and right splited Boundaries
    bxl: 1D numpy array
    byl: 1D numpy array
    bxr: 1D numpy array
    byr: 1D numpy array
    m: int
        number of elements on the new segments
    """
    # left edge
    xl0 = DB['DBXl'][i]
    xl1 = DB['DBXl'][i+1]
    yl0 = DB['DBYl'][i]
    yl1 = DB['DBYl'][i+1]
    dxl = xl1 - xl0
    dyl = yl1 - yl0
    Vl = np.array((dxl, dyl))
    zl = np.linalg.norm(Vl)

    # right edge
    xr0 = DB['DBXr'][i]
    xr1 = DB['DBXr'][i+1]
    yr0 = DB['DBYr'][i]
    yr1 = DB['DBYr'][i+1]
    dxr = xr1 - xr0
    dyr = yr1 - yr0
    Vr = np.array((dxr, dyr))
    zr = np.linalg.norm(Vr)

    # number of segments
    m = int(max(np.ceil(zl), np.ceil(zr))+1)
    # make left segment
    bxl = np.linspace(xl0, xl1, m)
    byl = np.linspace(yl0, yl1, m)
    # make right segment
    bxr = np.linspace(xr0, xr1, m)
    byr = np.linspace(yr0, yr1, m)

    return bxl, byl, bxr, byr, m


def makeTransfoMat(rasterTransfo):
    """ Make transformation matrix.

    Takes a Domain Boundary and finds the (x,y) coordinates of the new
    raster point (the one following the path)

    input rasterTransfo contains



    Parameters
    ----------
    rasterTransfo: dict
        dictionary containing

        - domainWidth: float
        - cellSize: float
        - DBXl: 1D numpy array, x coord of the left boundary
        - DBXr: 1D numpy array, x coord of the right boundary
        - DBYl: 1D numpy array, y coord of the left boundary
        - DBYr: 1D numpy array, y coord of the right boundary

    Returns
    -------
    rasterTransfo: dict
        rasterTransfo dictionary updated with

        - gridx: 2D numpy array, x coord of the new raster points in old coord system
        - gridy: 2D numpy array, y coord of the new raster points in old coord system
        - l: 1D numpy array, new coord system in the cross direction

    """
    w = rasterTransfo['domainWidth']
    cellSize = rasterTransfo['cellSizeSL']
    # number of points describing the avaPath
    n_pnt = np.shape(rasterTransfo['DBXr'])[0]
    # Working with no dimentions
    # (the cellsize scaling will be readded at the end)
    # lcoord is the distance from the polyline (cross section)
    # the maximum step size should be smaller then the cellsize
    nTot = np.ceil(w/cellSize)
    # take the next odd integer. This ensures that the lcoord = 0 exists
    nTot = int(nTot+1) if ((nTot % 2) == 0) else int(nTot)
    n2Tot = int(np.floor(nTot/2))
    lcoord = np.linspace(-n2Tot, n2Tot, nTot)  # this way, 0 is in lcoord

    # initialize new_rasters
    newGridRasterX = np.array([])  # xcoord of the points of the new raster
    newGridRasterY = np.array([])  # ycoord of the points of the new raster
    # loop on each section of the path
    for i in range(n_pnt-1):
        # split edges in segments
        bxl, byl, bxr, byr, m = splitSection(rasterTransfo, i)
        # bxl, byl, bxr, byr reprensent the s direction (olong path)
        # loop on segments of section
        for j in range(m-1):
            # this is the cross section segment (l direction)
            x = np.linspace(bxl[j], bxr[j], nTot)  # line coordinates x
            y = np.linspace(byl[j], byr[j], nTot)  # line coordinates y
            # save x and y coordinates of the new raster points
            if i == 0 and j == 0:
                newGridRasterX = x.reshape(1, nTot)
                newGridRasterY = y.reshape(1, nTot)
            else:
                newGridRasterX = np.append(newGridRasterX, x.reshape(1, nTot),
                                           axis=0)
                newGridRasterY = np.append(newGridRasterY, y.reshape(1, nTot),
                                           axis=0)

    # add last column
    x = np.linspace(bxl[m-1], bxr[m-1], nTot)  # line coordinates x
    y = np.linspace(byl[m-1], byr[m-1], nTot)  # line coordinates y
    newGridRasterX = np.append(newGridRasterX, x.reshape(1, nTot), axis=0)
    newGridRasterY = np.append(newGridRasterY, y.reshape(1, nTot), axis=0)

    rasterTransfo['l'] = lcoord
    rasterTransfo['gridx'] = newGridRasterX
    rasterTransfo['gridy'] = newGridRasterY

    return rasterTransfo


def getSArea(rasterTransfo, dem):
    """ Get the s curvilinear coordinate and area on the new raster

    Find the scoord corresponding to the transformation and the Area of
    the cells of the new raster

    Parameters
    ----------
    rasterTransfo: dict
        dictionary containing

        - domainWidth: float
        - cellSize: float
        - gridx: 2D numpy array, x coord of the new raster points in old coord system
        - gridy: 2D numpy array, y coord of the new raster points in old coord system

    dem: dict
        dem dictionary with normal and area information

    Returns
    -------
    rasterTransfo: dict
        rasterTransfo dictionary updated with

        - s: 1D numpy array, new coord system in the polyline direction
        - rasterArea: 2D numpy array, real area of the cells of the new raster

    """
    xcoord = rasterTransfo['gridx']
    ycoord = rasterTransfo['gridy']
    # add ghost lines and columns to the coord matrix
    # in order to perform dx and dy calculation
    n, m = np.shape(xcoord)
    xcoord = np.append(xcoord, xcoord[:, -2].reshape(n, 1), axis=1)
    ycoord = np.append(ycoord, ycoord[:, -2].reshape(n, 1), axis=1)
    n, m = np.shape(xcoord)
    xcoord = np.append(xcoord, xcoord[-2, :].reshape(1, m), axis=0)
    ycoord = np.append(ycoord, ycoord[-2, :].reshape(1, m), axis=0)
    n, m = np.shape(xcoord)
    # calculate dx and dy for each point in the s direction
    dxs = xcoord[1:n, 0:m-1]-xcoord[0:n-1, 0:m-1]
    dys = ycoord[1:n, 0:m-1]-ycoord[0:n-1, 0:m-1]
    # deduce the distance in s direction
    Vs2 = (dxs*dxs + dys*dys)
    Vs = np.sqrt(Vs2)

    # Method 1
    # calculate area of each cell using Gauss's area formula
    # sum_i{x_(i)*y_(i+1) + y_(i)*x_(i+1)}
    # i = 1 : top left in the matrix : [0:n-1, 0:m-1]
    x1 = xcoord[0:n-1, 0:m-1]
    y1 = ycoord[0:n-1, 0:m-1]
    # i = 2 : top right in the matrix : [1:n, 0:m-1]
    x2 = xcoord[1:n, 0:m-1]
    y2 = ycoord[1:n, 0:m-1]
    # i = 3 : bottom right in the matrix : [1:n, 1:m]
    x3 = xcoord[1:n, 1:m]
    y3 = ycoord[1:n, 1:m]
    # i = 4 : bottom left in the matrix : [0:n-1, 1:m]
    x4 = xcoord[0:n-1, 1:m]
    y4 = ycoord[0:n-1, 1:m]

    area = np.abs(x1*y2-y1*x2 + x2*y3-y2*x3 + x3*y4-y3*x4 + x4*y1-y4*x1)/2

    # save Area matrix
    demCellSize = dem['header']['cellsize']
    xllcenter = dem['header']['xllcenter']
    yllcenter = dem['header']['yllcenter']
    # area corection coef due to slope (1 if cell is horizontal, >1 if sloped)
    demAreaCoef = dem['areaRaster']/(demCellSize*demCellSize)
    # project on ney grid
    areaCoef, _ = geoTrans.projectOnGrid(rasterTransfo['gridx']*demCellSize, rasterTransfo['gridy']*demCellSize,
                                         demAreaCoef, csz=demCellSize, xllc=xllcenter, yllc=yllcenter)
    # real area
    rasterTransfo['rasterArea'] = area * areaCoef
    # get scoord
    ds = Vs[:, int(np.floor(m/2))-1]
    scoord = np.cumsum(ds)-ds[0]
    rasterTransfo['s'] = scoord

    return rasterTransfo


def transform(data, name, rasterTransfo, interpMethod):
    """ Transfer data from old raster to new raster

    Assign value to the points of the new raster (after domain transormation)

    Parameters
    ----------
    data: dict
        raster dictionary to transform
    name: str
        name of data to transform
    rasterTransfo: dict
        transformation information
    interpMethod: str
        interpolation method to chose between 'nearest' and 'bilinear'

    Returns
    -------
    newData: 2D numpy array
        new_data = z, pressure or thickness... corresponding to fname on
        the new raster
    """
    # read tranformation info
    newGridRasterX = rasterTransfo['gridx']
    newGridRasterY = rasterTransfo['gridy']

    n, m = np.shape(newGridRasterX)
    xx = newGridRasterX
    yy = newGridRasterY
    Points = {}
    Points['x'] = xx.flatten()
    Points['y'] = yy.flatten()
    iib = len(Points['x'])
    Points, ioob = geoTrans.projectOnRaster(data, Points, interp=interpMethod)
    newData = Points['z'].reshape(n, m)
    log.debug('Data-file: %s - %d raster values transferred - %d out of original raster'
              'bounds!' % (name, iib-ioob, ioob))

    return newData


def assignData(fnames, rasterTransfo, interpMethod):
    """ Transfer data from old raster to new raster

    Loops through paths in fnames and calls transfom
    Transform affects values to the points of the new raster
    (after domain transormation)

    Parameters
    ----------
    fnames: list of str
        list of path to rasterfile to transform
    rasterTransfo: dict
        transformation information
    interpMethod: str
        interpolation method to chose between 'nearest' and 'bilinear'

    Returns
    -------
    newData: 2D numpy array
        new_data = z, pressure or thickness... corresponding to fname on
        the new raster
    """

    maxtopo = len(fnames)
    avalData = np.array(([None] * maxtopo))

    log.debug('Transfer data of %d file(s) from old to new raster' % maxtopo)
    for i in range(maxtopo):
        fname = fnames[i]
        name = fname.name
        data = IOf.readRaster(fname)
        avalData[i] = transform(data, name, rasterTransfo, interpMethod)

    return avalData


def analyzeMass(fnameMass, simRowHash, refSimRowHash, resAnalysisDF, time=None):
    """ analyze Mass data

    Parameters
    ----------
    fnameMass: str
        path to mass data to analyze
    simRowHash: str
        simulation dataframe hash
    refSimRowHash: str
        dataframe hash (not simHash) of the simulation used as reference
    resAnalysisDF: dataFrame
        results from Aimec Analysis to update with mass info
    time: None or 1D numpy array
        None at the first call of the function, then a 1D array
        that will be used to analyze all other mass results

    Returns
    -------
    resAnalysisDF: dataFrame
        results from Aimec Analysis updated with mass inifo:
            relMass: float
                release mass
            entMass: float
                entrained mass
            finalMass: float
                final mass
            relativMassDiff: float
                the final mass diff with ref (in %)
            growthIndex: float
                growth index
            growthGrad: float
                growth gradient
            entMassFlowArray: 1D numpy array
                entrained mass function of time
            totalMassArray: 1D numpy array
                total mass function of time
        time: 1D numpy array
            time array for mass analysis
    """
    log.debug('Analyzing mass')
    log.debug('{: <10} {: <10} {: <10}'.format('Sim number ', 'GI ', 'GR '))
    # analyze mass
    releasedMass, entrainedMass, finalMass, grIndex, grGrad, entMassFlow, totalMass, time = readWrite(fnameMass, time)
    resAnalysisDF.loc[simRowHash, 'relMass'] = releasedMass
    resAnalysisDF.loc[simRowHash, 'finalMass'] = finalMass
    resAnalysisDF.loc[simRowHash, 'entMass'] = entrainedMass
    resAnalysisDF.loc[simRowHash, 'growthIndex'] = grIndex
    resAnalysisDF.loc[simRowHash, 'growthGrad'] = grGrad

    releasedMassRef = resAnalysisDF.loc[refSimRowHash, 'relMass']
    finalMassRef = resAnalysisDF.loc[refSimRowHash, 'finalMass']
    relativMassDiff = (finalMass-finalMassRef)/finalMassRef*100
    if not (releasedMass == releasedMassRef):
        log.warning('Release masses differ between simulations!')
    log.info('{: <10} {:<10.4f} {:<10.4f}'.format(*[resAnalysisDF.loc[simRowHash, 'simName'], grIndex, grGrad]))
    resAnalysisDF.loc[simRowHash, 'relativMassDiff'] = relativMassDiff

    if simRowHash == refSimRowHash:
        resAnalysisDF = pd.concat([resAnalysisDF,
                                   pd.DataFrame({'entMassFlowArray': np.nan},
                                                index=resAnalysisDF.index)],
                                  axis=1).copy()
        resAnalysisDF['entMassFlowArray'] = resAnalysisDF['entMassFlowArray'].astype(object)
        resAnalysisDF = pd.concat([resAnalysisDF,
                                   pd.DataFrame({'totalMassArray': np.nan},
                                                index=resAnalysisDF.index)],
                                  axis=1).copy()
        resAnalysisDF['totalMassArray'] = resAnalysisDF['totalMassArray'].astype(object)

    resAnalysisDF.at[simRowHash, 'entMassFlowArray'] = entMassFlow
    resAnalysisDF.at[simRowHash, 'totalMassArray'] = totalMass
    return resAnalysisDF, time


def computeRunOut(cfgSetup, rasterTransfo, resAnalysisDF, transformedRasters, simRowHash):
    """ Compute runout based on peak field results

    Parameters
    ----------
    cfgSetup: confiParser
        aimec analysis configuration
    rasterTransfo: dict
        transformation information
    resAnalysisDF : dataFrame
        analysis results from aimec containing

        - PResCrossMax: 1D numpy array, max of the peak result in each cross section
        - PResCrossMean: 1D numpy array, mean of the peak result in each cross section

    transformedRasters: dict
        dict with transformed dem and peak results
    simRowHash: str
        simulation dataframe hash

    Returns
    -------
    resAnalysisDF : dataFrame
        result dataFrame updated for each simulation with

        - xRunout: float, x coord of the runout point
          measured from the beginning of the path. run-out
          calculated with the MAX result in each cross section
        - yRunout: float, y coord of the runout point
          measured from the beginning of the path. run-out
          calculated with the MAX result in each cross section
        - sRunout: float, runout distance measured from the beginning of the path.
          run-out calculated with the MAX result in each cross section
        - xMeanRunout: float, x coord of the runout point
          measured from the beginning of the path. run-out
          calculated with the MEAN result in each cross section
        - yMeanRunout: float, y coord of the runout point
          measured from the beginning of the path. run-out
          calculated with the MEAN result in each cross section
        - sMeanRunout: float, runout distance measured from the beginning of the path.
          run-out calculated with the MEAN result in each cross section
        - elevRel: float, elevation of the release area (based on first point with
          peak field > thresholdValue)
        - deltaZ: float, elevation fall difference between elevRel and altitude of
          run-out point

    """

    # read inputs
    scoord = rasterTransfo['s']
    lcoord = rasterTransfo['l']
    zThalweg = rasterTransfo['z']
    n = np.shape(lcoord)[0]
    n = int(np.floor(n/2)+1)
    x = rasterTransfo['x']
    y = rasterTransfo['y']
    gridx = rasterTransfo['gridx']
    gridy = rasterTransfo['gridy']

    runoutResType = cfgSetup['runoutResType']
    thresholdValue = cfgSetup.getfloat('thresholdValue')
    transformedDEMRasters = transformedRasters['newRasterDEM']
    PResRasters = transformedRasters['newRaster' + runoutResType.upper()]
    PResCrossMax = resAnalysisDF.loc[simRowHash, runoutResType + 'CrossMax']
    PResCrossMean = resAnalysisDF.loc[simRowHash, runoutResType + 'CrossMean']

    log.debug('Computing runout')
    lindex = np.nonzero(PResCrossMax > thresholdValue)[0]
    if lindex.any():
        cUpper = min(lindex)
        cLower = max(lindex)
        runoutFound = True
    else:
        log.warning('No max values > threshold found. threshold = %4.2f, too high?' % thresholdValue)
        cUpper = 0
        cLower = 0
        runoutFound = False
    # search in mean values
    lindex = np.nonzero(PResCrossMean > thresholdValue)[0]
    if lindex.any():
        cUpperm = min(lindex)
        cLowerm = max(lindex)
        flagMeanFound = True
    else:
        log.warning('No average values > threshold found. threshold = %4.2f, too high?' % thresholdValue)
        cUpperm = 0
        cLowerm = 0
        flagMeanFound = False

    if runoutFound:
        resAnalysisDF.loc[simRowHash, "sRunout"] = scoord[cLower]
        index = np.nanargmax(PResRasters[cLower, :])
        resAnalysisDF.loc[simRowHash, "lRunout"] = lcoord[index]
        resAnalysisDF.loc[simRowHash, "xRunout"] = gridx[cLower, index]
        resAnalysisDF.loc[simRowHash, "yRunout"] = gridy[cLower, index]
        resAnalysisDF.loc[simRowHash, "runoutAngle"] = np.rad2deg(
            np.arctan((zThalweg[cUpper] - zThalweg[cLower]) / (scoord[cLower] - scoord[cUpper]))
        )
    else:
        resAnalysisDF.loc[simRowHash, "sRunout"] = np.nan
        resAnalysisDF.loc[simRowHash, "lRunout"] = np.nan
        resAnalysisDF.loc[simRowHash, "xRunout"] = np.nan
        resAnalysisDF.loc[simRowHash, "yRunout"] = np.nan
        resAnalysisDF.loc[simRowHash, "runoutAngle"] = np.nan

    resAnalysisDF.loc[simRowHash, 'deltaSXY'] = scoord[cLower] - scoord[cUpper]

    if flagMeanFound:
        resAnalysisDF.loc[simRowHash, "zRunout"] = zThalweg[cLower]
        resAnalysisDF.loc[simRowHash, "sMeanRunout"] = scoord[cLowerm]
        resAnalysisDF.loc[simRowHash, "xMeanRunout"] = x[cLowerm]
        resAnalysisDF.loc[simRowHash, "yMeanRunout"] = y[cLowerm]
    else:
        resAnalysisDF.loc[simRowHash, "zRunout"] = np.nan
        resAnalysisDF.loc[simRowHash, "sMeanRunout"] = np.nan
        resAnalysisDF.loc[simRowHash, "xMeanRunout"] = np.nan
        resAnalysisDF.loc[simRowHash, "yMeanRunout"] = np.nan

    resAnalysisDF.loc[simRowHash, "zRelease"] = zThalweg[cUpper]
    resAnalysisDF.loc[simRowHash, 'elevRel'] = zThalweg[cUpper]
    resAnalysisDF.loc[simRowHash, 'deltaZ'] = zThalweg[cUpper] - zThalweg[cLower]
    resAnalysisDF.loc[simRowHash, "runoutFound"] = runoutFound

    return resAnalysisDF


def computeRunoutLine(cfgSetup, rasterTransfo, transformedRasters, simRowHash, actualType, name='', basedOnMax=False):
    """ compute the runout line as for each l coordinate the furthest affected s coordinate using the desired
        resType and thresholdvalue, or if basedOnMax searching for the max value (used for reference line, point)
        also add runout point based on max s value of runout line

        Parameters
        -----------
        cfgSetup: configparser object
            configuration settings, here: runoutResType and tresholdValue
        rasterTransfo: dict
            information on raster transformation
        transformedRasters: dict
            transformed rasters from simulation
        simRowHash: hash
            index of current simulation
        actualType: str
            options: simulation, line, point, poly
        name: str
            optional name to find raster in transfromedRasters dict
        basedOnMax: bool
            if not threshold of runoutResType is used but max value along s for each l

        Returns
        ---------
        runoutLine : dict
            coordinates of runout line in s, l, and x, y, index of rasterTransfo['s']mname and type
            coordinates of runout point (max s extent) in s, l
        """

    # get info on coordinate grids
    gridx = rasterTransfo['gridx']
    gridy = rasterTransfo['gridy']

    # fetch raster data
    if name == '':
        runoutResType = cfgSetup['runoutResType']
        anaRaster = transformedRasters['newRaster' + runoutResType.upper()]
        name = simRowHash
    else:
        anaRaster = transformedRasters[name]

    # set to 0 where nans and flip order of rows to search for furthest point in s
    anaRaster = np.where(np.isnan(anaRaster), 0, anaRaster)
    anaRasterFlip = np.flip(anaRaster, axis=0)

    # setup runout line
    thresholdValue = cfgSetup.getfloat('thresholdValue')
    runoutLine = {'s': np.zeros(rasterTransfo['l'].shape) * np.nan, 'l': np.zeros(rasterTransfo['l'].shape) * np.nan,
                  'x': np.zeros(rasterTransfo['l'].shape) * np.nan, 'y': np.zeros(rasterTransfo['l'].shape) * np.nan,
                  'runoutLineFound': np.zeros(rasterTransfo['l'].shape) * np.nan}
    runoutLineIndex = np.zeros(rasterTransfo['l'].shape) * np.nan

    # loop over each l coordinate to find max s coordinate
    sMaxInd = np.nan
    lMaxInd = np.nan
    sMax = 0
    oneIndexFound = False
    for ind, lCoor in enumerate(rasterTransfo['l']):
        indexFound = False
        if basedOnMax and (len(np.nonzero(anaRaster[:,ind])[0]) > 0):
            index2 = np.argmax(anaRasterFlip[:,ind])
            index1 = anaRaster.shape[0]-index2
            indexFound = True
            oneIndexFound =True
        elif (basedOnMax == False) and (len(np.nonzero(anaRaster[:,ind]> thresholdValue)[0]) > 0):
            index1 = max(np.nonzero(anaRaster[:,ind] > thresholdValue)[0])
            indexFound = True
            oneIndexFound = True
        if indexFound:
            runoutLineIndex[ind] = index1
            runoutLine['s'][ind] = (rasterTransfo['s'][index1])
            runoutLine['l'][ind] = lCoor
            runoutLine['x'][ind] = gridx[index1, ind]
            runoutLine['y'][ind] = gridy[index1, ind]
            runoutLine['runoutLineFound'][ind] = True
            if rasterTransfo['s'][index1] > sMax:
                sMaxInd = index1
                lMaxInd = ind
                sMax = rasterTransfo['s'][index1]
        else:
            runoutLine['runoutLineFound'][ind] = False

    # add info on name, type
    runoutLine['index'] = runoutLineIndex
    runoutLine['name'] = name
    runoutLine['type'] = actualType
    if oneIndexFound:
        runoutLine['sRunout'] = np.nanmax(runoutLine['s'])
        runoutLine['lRunout'] = rasterTransfo['l'][lMaxInd]
        runoutLine['xRunout'] = gridx[sMaxInd, lMaxInd]
        runoutLine['yRunout'] = gridy[sMaxInd, lMaxInd]
    else:
        log.warning('For new Raster %s no runout line is found' % name)

    return runoutLine


def analyzeField(simRowHash, rasterTransfo, transformedRaster, dataType, resAnalysisDF):
    """ analyze transformed field

    analyze transformed raster: compute the Max and Mean values in each cross section
    as well as the overall maximum

    Parameters
    ----------
    simRowHash: str
        simulation dataframe hash
    rasterTransfo: dict
        transformation information
    transformedRaster: 2D numpy array
        raster after transformation
    dataType: str
        type of the data to analyze ('ppr', 'pft', 'pfv', ...)
    resAnalysisDF: dataFrame
        result dataFrame to be updated

    Returns
    -------
    Updates the resAnalysisDF input dataFrame with:
        -maxaCrossMax: float
            overall maximum
        -aCrossMax: 1D numpy array
            containing the max of the field in each cross section
        -aCrossMean: 1D numpy array
            containing the mean of the field in each cross section
    """
    # read inputs
    rasterArea = rasterTransfo['rasterArea']

    # initialize Arrays
    log.debug('Analyzing %s' % (dataType))
    log.debug('{: <10} {: <10}'.format('Sim number ', 'maxCrossMax '))

    # Max Mean in each Cross-Section for each field
    maxaCrossMax, aCrossMax, aCrossMean = getMaxMeanValues(transformedRaster, rasterArea)
    log.debug('{: <10} {:<10.4f}'.format(*[simRowHash, maxaCrossMax]))

    resAnalysisDF.loc[simRowHash, 'max' + dataType + 'CrossMax'] = maxaCrossMax
    resAnalysisDF.at[simRowHash, dataType + 'CrossMax'] = aCrossMax
    resAnalysisDF.at[simRowHash, dataType + 'CrossMean'] = aCrossMean

    return resAnalysisDF


def analyzeArea(rasterTransfo, resAnalysisDF, simRowHash, newRasters, cfg, pathDict, contourDict, referenceType='newRefRaster'):
    """Compare area results to reference.

    Compute True positive, False negative... areas.

    Parameters
    ----------
    rasterTransfo: dict
        transformation information
    resAnalysisDF: dataFrame
        dataFrame containing Aimec results to update
    simRowHash: str
        simulation dataframe hash
    newRasters: dict
        dict with tranformed raster for reference and curent simulation
    cfg: confiParser
        numerical value of the limit to use for the runout computation
        as well as the levels for the contour line plot
    pathDict: dict
        path to data dem data and lines for aimec analysis
    contourDict: dict
        dictionary with one key per sim and its x, y coordinates for contour line of runoutresType
        for thresholdValue
    referenceType: str
        to decide which key to use for newRasters (transformed rasters)

    Returns
    -------
    resAnalysisDF: dataFrame
        dataFrame containing Aimec results updated with:
            TP: float
                ref = True sim2 = True
            FN: float
                ref = False sim2 = True
            FP: float
                ref = True sim2 = False
            TN: float
                ref = False sim2 = False
    contourDict: dict
        dictionary with one key per sim and its x, y coordinates for contour line of runoutresType
        for thresholdValue - updated
    """

    cfgSetup = cfg['AIMECSETUP']
    cfgPlots = cfg['PLOTS']
    runoutResType = cfgSetup['runoutResType']
    refSimRowHash = pathDict['refSimRowHash']
    cellarea = rasterTransfo['rasterArea']
    indStartOfRunout = rasterTransfo['indStartOfRunout']
    thresholdValue = cfgSetup.getfloat('thresholdValue')
    contourLevels = fU.splitIniValueToArraySteps(cfgSetup['contourLevels'])
    simName = resAnalysisDF.loc[simRowHash, 'simName']
    # rasterinfo
    nStart = indStartOfRunout
    # inputs for plot
    inputs = {}
    inputs['runoutLength'] = resAnalysisDF.loc[refSimRowHash, 'sRunout']
    inputs['refData'] = newRasters[referenceType + runoutResType.upper()]
    inputs['nStart'] = nStart
    inputs['runoutResType'] = runoutResType

    thresholdArray = contourLevels
    thresholdArray = np.append(thresholdArray, thresholdValue)
    inputs['thresholdArray'] = thresholdArray
    inputs['diffLim'] = cfgSetup.getfloat('diffLim')

    rasterdata = newRasters['newRaster' + runoutResType.upper()]

    # take first simulation as reference
    refMask = copy.deepcopy(inputs['refData'])
    # prepare mask for area resAnalysis
    refMask = np.where(np.isnan(refMask), 0, refMask)
    refMask = np.where(refMask <= thresholdValue, 0, refMask)
    refMask = np.where(refMask > thresholdValue, 1, refMask)
    # comparison rasterdata with mask
    log.debug('{: <15} {: <15} {: <15} {: <15} {: <15}'.format('Sim number ', 'TP ', 'FN ', 'FP ', 'TN'))
    compRasterMask = copy.deepcopy(rasterdata)
    # prepare mask for area resAnalysis
    compRasterMask = np.where(np.isnan(compRasterMask), 0, compRasterMask)
    compRasterMask = np.where(compRasterMask <= thresholdValue, 0, compRasterMask)
    compRasterMask = np.where(compRasterMask > thresholdValue, 1, compRasterMask)

    tpInd = np.where((refMask[nStart:] == 1) & (compRasterMask[nStart:] == 1))
    fpInd = np.where((refMask[nStart:] == 0) & (compRasterMask[nStart:] == 1))
    fnInd = np.where((refMask[nStart:] == 1) & (compRasterMask[nStart:] == 0))
    tnInd = np.where((refMask[nStart:] == 0) & (compRasterMask[nStart:] == 0))

    # subareas
    tp = np.nansum(cellarea[tpInd[0] + nStart, tpInd[1]])
    fp = np.nansum(cellarea[fpInd[0] + nStart, fpInd[1]])
    fn = np.nansum(cellarea[fnInd[0] + nStart, fnInd[1]])
    tn = np.nansum(cellarea[tnInd[0] + nStart, tnInd[1]])

    # take reference (first simulation) as normalizing area
    areaSum = tp + fn

    if areaSum > 0:
        log.debug('{: <15} {:<15.4f} {:<15.4f} {:<15.4f} {:<15.4f}'.format(
                  *[simName, tp/areaSum, fn/areaSum, fp/areaSum, tn/areaSum]))
    if tp + fp == 0:
        log.warning('Simulation %s did not reach the run-out area for threshold %.2f %s' %
                    (simName, thresholdValue, pU.cfgPlotUtils['unit' + cfgSetup['runoutResType']]))

    resAnalysisDF.loc[simRowHash, 'TP'] = tp
    resAnalysisDF.loc[simRowHash, 'TN'] = tn
    resAnalysisDF.loc[simRowHash, 'FP'] = fp
    resAnalysisDF.loc[simRowHash, 'FN'] = fn
    # inputs for plot
    inputs['compData'] = rasterdata
    # masked data for the dataThreshold given in the ini file
    inputs['refRasterMask'] = refMask
    inputs['compRasterMask'] = compRasterMask
    inputs['simName'] = simName

    compPlotPath = None
    if (
        simRowHash != refSimRowHash
        and cfgPlots.getboolean("extraPlots")
        and resAnalysisDF.loc[simRowHash, "runoutFound"]
    ):
        # only plot comparisons of simulations to reference
        compPlotPath = outAimec.visuComparison(rasterTransfo, inputs, pathDict)
    elif not resAnalysisDF.loc[simRowHash, "runoutFound"] and cfgPlots.getboolean("extraPlots"):
        log.warning(
            "ContourComparisonToReference plot not generated as only 0 values for comparison simulation: %s"
            % resAnalysisDF.loc[simRowHash, "simName"]
        )
    # add contourlines to contourDict
    contourDict = outAimec.fetchContourLines(rasterTransfo, inputs, cfgSetup.getfloat('thresholdValue'), contourDict)

    return resAnalysisDF, compPlotPath, contourDict


def readWrite(fname_ent, time):
    """ Get mass balance information
        Read mass balance files to get mass properties of the simulation
        (total mass, entrained mass...). Checks for mass conservation

        Parameters
        ----------
        fname_ent: list of str
            list of pah to mass balance files
        Returns
        -------
        relMass: float
            release mass
        entMassentMassFlow: float
            entrained mass flow (kg/s)
        finalMass: float
            final mass
        growthIndex: float
        growthGrad: float
    """

    #    load data
    #    time, total mass, entrained mass
    massTime = np.loadtxt(fname_ent, delimiter=',', skiprows=1)
    timeSimulation = massTime[:, 0]
    dt = timeSimulation[1:] - timeSimulation[:-1]
    timeResults = [massTime[0, 0], massTime[-1, 0]]
    totMassResults = [massTime[0, 1], massTime[-1, 1]]
    relMass = totMassResults[0]
    if time is None:
        time = np.arange(0, int(timeResults[1]), 0.1)
    entMassFlow = np.interp(time, massTime[1:, 0], massTime[1:, 2]/dt)
    totalMass = np.interp(time, massTime[:, 0], massTime[:, 1])
    entrainedMass = np.sum(massTime[:, 2])
    finalMass = totMassResults[1]

    # check mass balance
    log.info('Total mass change between first and last time step in sim %s is: %.1f kg' %
             (fname_ent.stem, totMassResults[1] - relMass))
    log.info('Total entrained mass in sim %s is: %.1f kg' %
             (fname_ent.stem, entrainedMass))
    if (totMassResults[1] - relMass) == 0:
        diff = np.abs((totMassResults[1] - relMass) - entrainedMass)
        if diff > 0:
            log.warning('Conservation of mass is not satisfied')
            log.warning('Total mass change and total entrained mass differ from %.4f kg' % (diff))
        else:
            log.info('Total mass change and total entrained mass differ from %.4f kg' % (diff))
    else:
        diff = np.abs((totMassResults[1] - relMass) - entrainedMass)/(totMassResults[1] - relMass)
        if diff*100 > 0.05:
            log.warning('Conservation of mass is not satisfied')
            log.warning('Total mass change and total entrained mass differ from %.4f %%' % (diff*100))
        else:
            log.info('Total mass change and total entrained mass differ from %.4f %%' % (diff*100))

    # growth results
    growthIndex = totMassResults[1]/totMassResults[0]
    growthGrad = (totMassResults[1] - totMassResults[0]) / (timeResults[1] - timeResults[0])

    return relMass, entrainedMass, finalMass, growthIndex, growthGrad, entMassFlow, totalMass, time


def getMaxMeanValues(rasterdataA, rasterArea):
    """Compute average, max values in each cross section for a given input raster

    Read mass balance files to get mass properties of the simulation
    (total mass, entrained mass...). Checks for mass conservation

    Parameters
    ----------
    rasterdataA: 2D numpy array
        raster data
    rasterArea: 2D numpy array
        raster area corresponding to rasterdataA

    Returns
    -------
    mma: float
        maximum maximum of rasterdataA
    aCrossMax: 1D numpy array
        max of rasterdataA in each cross section
    aCrossMean: 1D numpy array
        mean of rasterdataA in each cross section (area weighted)
    """
    # get mean max for each cross section for A field
    rasterArea = np.where(np.where(np.isnan(rasterdataA), 0, rasterdataA) > 0, rasterArea, 0)
    areaSum = np.nansum(rasterArea, axis=1)
    areaSum = np.where(areaSum > 0, areaSum, 1)
    aCrossMean = np.nansum(rasterdataA*rasterArea, axis=1)/areaSum
    # aCrossMean = np.nanmean(rasterdataA, axis=1)
    aCrossMax = np.nanmax(rasterdataA, 1)
    # maximum of field a
    maxaCrossMax = np.nanmax(aCrossMax)

    return maxaCrossMax, aCrossMax, aCrossMean


def setAvaPath(pathDict, dem):
    """ fetch path shapefile, prepare for AIMEC, set z-coordinate

        Parameters
        -----------
        pathDict: dict
            info on path to aimec path, split Point,
        dem: dict
            dictionary with DEM header and data

        Returns
        --------
        avaPath: dict
            info on aimec path coordinates, ...
        splitPoint: dict
            info on split point coordinates

    """

    # fetch input parameters
    profileLayer = pathDict['profileLayer']
    splitPointSource = pathDict['splitPointSource']
    defaultName = pathDict['projectName']

    # read avaPath
    avaPath = shpConv.readLine(profileLayer, defaultName, dem)
    # read split point
    if splitPointSource != None:
        splitPoint = shpConv.readPoints(splitPointSource, dem)
    else:
        splitPoint = None
    # add 'z' coordinate to the avaPath
    avaPath, _ = geoTrans.projectOnRaster(dem, avaPath)
    # reverse avaPath if necessary
    _, avaPath = geoTrans.checkProfile(avaPath, projSplitPoint=None)

    log.debug('Creating new raster along polyline: %s' % profileLayer)

    return avaPath, splitPoint


def scalePathWithCellSize(rasterTransfo, cellSizeSL):
    """ use desired cellsize to scale ava path and create s, l coordinates
        and coordinates of the resampled polyline in the old coord systems x, y

        Parameters
        ----------
        rasterTransfo: dict
            domain transformation info
        cellSizeSL: float
            desired cell size of sl raster

        Returns
        --------
        rasterTransfo: dict
            updated dictionary with new coordinate system
    """

    # put back the scale due to the desired cellsize
    rasterTransfo['s'] = rasterTransfo['s']*cellSizeSL
    rasterTransfo['l'] = rasterTransfo['l']*cellSizeSL
    rasterTransfo['gridx'] = rasterTransfo['gridx']*cellSizeSL
    rasterTransfo['gridy'] = rasterTransfo['gridy']*cellSizeSL
    rasterTransfo['rasterArea'] = rasterTransfo['rasterArea']*cellSizeSL*cellSizeSL
    # (x,y) coordinates of the resampled avapth (centerline where l = 0)
    n = np.shape(rasterTransfo['l'])[0]
    indCenter = int(np.floor(n/2))
    rasterTransfo['x'] = rasterTransfo['gridx'][:, indCenter]
    rasterTransfo['y'] = rasterTransfo['gridy'][:, indCenter]

    return rasterTransfo


def addSurfaceParalleCoord(rasterTransfo):
    """ Add the surface parallel coordinate (along flow path taking elevation into account)

    Parameters
    ----------
    rasterTransfo: dict
        domain transformation info

    Returns
    --------
    rasterTransfo: dict
        updated dictionary with the surface parallel coordinate 'sParallel'
    """
    dz = np.diff(rasterTransfo['z'], prepend=rasterTransfo['z'][0])
    ds = np.diff(rasterTransfo['s'], prepend=rasterTransfo['s'][0])
    dsParallel2 = (ds*ds + dz*dz)
    dsParallel = np.sqrt(dsParallel2)
    sParallel = np.cumsum(dsParallel)-dsParallel[0]
    rasterTransfo['sParallel'] = sParallel
    return rasterTransfo


def findStartOfRunoutArea(dem, rasterTransfo, cfgSetup, splitPoint):
    """ find start of runout area point using splitPoint
        add info on x, y coordinates of point, angle, index
        - if defineRunoutArea=False - start of runout area is equal to the start of the thalweg
        -> in this case the entire SL domain represents the runout area

        Parameters
        -----------
        dem: dict
            dictionary with DEM header and data
        rasterTransfo: dict
            domain transformation info
        cfgSetup: configparser object
            configuration for aimec
        splitPoint: dict
            dictionary with split Point coordinates

        Returns
        --------
        rasterTransfo: dict
            updated dictionary with info on start of runout location,...

    """

    if splitPoint != None:
        # fetch input parameters from config
        startOfRunoutAreaAngle = cfgSetup.getfloat('startOfRunoutAreaAngle')

        # add 'z' coordinate to the centerline
        rasterTransfo, _ = geoTrans.projectOnRaster(dem, rasterTransfo)

        # find projection of split point on the centerline
        projPoint = geoTrans.findSplitPoint(rasterTransfo, splitPoint)
        rasterTransfo['indSplit'] = projPoint['indSplit']
        rasterTransfo['projSplitPoint'] = projPoint

        # prepare find start of runout area points
        angle, tmp, ds = geoTrans.prepareAngleProfile(startOfRunoutAreaAngle, rasterTransfo)

        # find the runout point: first point under startOfRunoutAreaAngle
        indStartOfRunout = geoTrans.findAngleProfile(tmp, ds, cfgSetup.getfloat('dsMin'))
        rasterTransfo['startOfRunoutAreaAngle'] = angle[indStartOfRunout]
        rasterTransfo['labelRunout'] = ('start of runout area: ' + (r'$\beta_{%.1f °}$' %
                                                                    rasterTransfo['startOfRunoutAreaAngle']))
    else:
        log.info('DefineRunoutArea is set to False - start of runout area set to start of thalweg')
        indStartOfRunout = 0
        rasterTransfo['startOfRunoutAreaAngle'] = np.nan
        rasterTransfo['labelRunout'] = 'start of runout area'

    # add info to rasterTransfo dict
    rasterTransfo['indStartOfRunout'] = indStartOfRunout
    rasterTransfo['xBetaPoint'] = rasterTransfo['x'][indStartOfRunout]
    rasterTransfo['yBetaPoint'] = rasterTransfo['y'][indStartOfRunout]

    log.info('Start of run-out area at the %.2f ° point of coordinates (%.2f, %.2f)' %
             (rasterTransfo['startOfRunoutAreaAngle'], rasterTransfo['xBetaPoint'], rasterTransfo['yBetaPoint']))

    return rasterTransfo


def addFieldsToDF(inputsDF, layerTypesList):
    """add fields that will be added in aimec analysis to dataframe

    Parameters
    -----------
    inputsDF: pandas DataFrame
        DataFrame where fields should be added as empty columns
        fields are:
    cfgSetup: configparser object
        configuration settings, here used: includeReference if inputs from Inputs/REFDATA shall be included and
        used to compare sim results to
    layerTypesList: list
        list of available layer types in simulations to be analysed
    Returns
    ---------
    inputsDF: pandas DataFrame
        updated DataFrame

    """
    nanFieldsNames = [
        "sRunout",
        "lRunout",
        "xRunout",
        "yRunout",
        "deltaSXY",
        "runoutAngle",
        "zRelease",
        "zRunout",
        "sMeanRunout",
        "xMeanRunout",
        "yMeanRunout",
        "elevRel",
        "deltaZ",
        "refSim_Diff_sRunout",
        "refSim_Diff_lRunout",
        "dataType",
        "runoutLineDiff_line_RMSE",
        "runoutLineDiff_poly_RMSE",
    ]

    emptyStrFieldsNames = [
        "runoutLineDiff_line_pointsNotFoundInSim",
        "runoutLineDiff_line_pointsNotFoundInRef",
        "runoutLineDiff_poly_pointsNotFoundInSim",
        "runoutLineDiff_poly_pointsNotFoundInRef",
        "runoutFound",
    ]

    nanFields = []
    emptyStrFields = []
    for layerType in layerTypesList:
        nanFields = [nanF + "_" + layerType for nanF in nanFieldsNames]
        emptyStrFields = [emptySF + "_" + layerType for emptySF in emptyStrFieldsNames]

    for item in nanFields:
        inputsDF = pd.concat([inputsDF, pd.DataFrame({item: np.nan}, index=inputsDF.index)], axis=1).copy()

    for item in emptyStrFields:
        inputsDF = pd.concat([inputsDF, pd.DataFrame({item: ''}, index=inputsDF.index)], axis=1).copy()

    # add that datatype is simulation
    inputsDF['dataType'] = ['simulation']*len(inputsDF)

    return inputsDF


def createReferenceDF(pathDict):
    """ create data frame with one row per reference data set found in Inputs/REFDATA
        Parameters
        -----------
        pathDict: dict
            dictionary with info paths to reference datasets

        Returns
        ---------
        referenceDF: pandas DataFrame
            DataFrame with one row per reference dataset

    """

    # add parameters that provide info on reference data and analysis that will be performed on this data
    nanFields = ['reference_Type', 'reference_resType', 'reference_sRunout', 'reference_lRunout',
                 'reference_xRunout', 'reference_yRunout', 'reference_name', 'reference_filePath',
                 'dataType']

    # create empty dataframe
    referenceDF = pd.DataFrame(columns=nanFields)

    # add one row for each reference dataset
    refData = [pathDict['referenceLine'], pathDict['referencePolygon'], pathDict['referencePoint']]
    for ref in refData:
        if ref != []:
            for refFile in ref:
                referenceName = refFile.stem
                newLine = pd.DataFrame([[referenceName, refFile]], columns=['reference_name', 'reference_filePath'], index=[referenceName])
                hashRef = pd.util.hash_pandas_object(newLine)
                newLine = newLine.set_index(hashRef)
                referenceDF = pd.concat([referenceDF, newLine], ignore_index=False)
                referenceDF.loc[hashRef, 'dataType'] = 'reference'

    # TODO add here if additional info read from shp or a textfile?

    return referenceDF


def computeRunoutPointDiff(resAnalysisDF, refData, simRowHash):
    """ compute difference between runout point of simulation and runout point of refData

        Parameters
        -------------
        resAnalysisDF: data frame
            one row per simulation (config info, sim results and analysis)
        refData: dict
            dictionary of reference data set
        simRowHash:
            index of row in data frame for current simulation

        Returns
        --------
        resAnalysisDF: data frame
            updated DF with runout point difference

    """

    for item in ['sRunout', 'lRunout']:
        simPoint = resAnalysisDF.loc[simRowHash, item]
        refPoint = refData[item]

        diffRunout = refPoint - simPoint

        resAnalysisDF.loc[simRowHash, 'refSim_Diff_%s' % item] = diffRunout
        log.info('%s Runout diff for %s is %.2f' % (item, resAnalysisDF.loc[simRowHash, 'simName'], diffRunout))

    return resAnalysisDF


def findSLCoors(rasterTransfo, refData, referenceType):
    """ find closest s,l coordinates to given x,y coordinates using transformation matrix

        Parameters
        -----------
        rasterTransfo
        refData

    """

    xx = rasterTransfo['gridx']
    yy = rasterTransfo['gridy']
    nrows = xx.shape[0]
    ncols = xx.shape[1]
    sizeXX = xx.size
    indexArray = np.arange(sizeXX).reshape(nrows, ncols)

    coors = np.column_stack((np.reshape(xx, sizeXX),  np.reshape(yy, sizeXX)))

    # initialize index1Array
    index1Array = None
    if referenceType.lower() == 'point':
        # arrange x,y coordinates of ref Data into a 2D array for finding closest coordinates of coors
        test = np.zeros((len(coors), 2))
        test[:, 0] = refData['x']
        test[:, 1] = refData['y']

        indexBool = np.isclose(coors, test)
        index1Array = [np.where(np.all(indexBool==True, axis=1) == True)[0][0]]

    elif referenceType.lower() == 'line':
        indRowArray = []
        indColArray = []
        index1Array = []
        for x1, y1 in zip(refData['x'], refData['y']):
            # arrange x,y coordinates of ref Data into a 2D array for finding closest coordinates of coors
            test = np.zeros((len(coors), 2))
            test[:, 0] = x1
            test[:, 1] = y1
            indexBool = np.isclose(coors, test)

            if len(np.where(np.all(indexBool==True, axis=1))[0]) > 1:
                index1 = np.where(np.all(indexBool==True, axis=1) == True)[0][0]
                indRowArray.append(np.where(indexArray==index1)[0][0])
                indColArray.append(np.where(indexArray==index1)[1][0])
                index1Array.append(index1)

    if index1Array is None:
        Lcoors = None
        Scoors = None
    else:
        gridL, gridS = np.meshgrid(rasterTransfo['l'], rasterTransfo['s'])
        slCoors = np.column_stack((np.reshape(gridL, gridL.size),  np.reshape(gridS, gridS.size)))
        Lcoors = slCoors[index1Array,0]
        Scoors = slCoors[index1Array,1]

    return Lcoors, Scoors


def addReferenceAnalysisTODF(referenceDF, refFile, refDataDict):
    """ add reference analysis info to referenceDF

        Parameters
        ------------
        referenceDF: data frame
            one row per reference dataset
        refFile: pathlib Path
            file path of reference dataset file
        refDataDict: dict
            dict with info on reference data set and computed runout point

        Returns
        ----------
        referenceDF: data frame
            updated data DF with info on runout point coordinate
    """

    for item in ['sRunout', 'lRunout', 'xRunout', 'yRunout']:
        referenceDF.loc[referenceDF['reference_name'] == refFile.stem, 'reference_%s' % item] = refDataDict[item]

    return referenceDF


def analyzeDiffsRunoutLines(cfgSetup, runoutLine, refDataTransformed, resAnalysisDF, simRowHash, pathDict):
    """ analyze difference between runout line derived from simRaster and reference datasets (lines, polygons)

        Parameters
        -------------
        runoutLine: dict
            info on runout line derived from simRaster
        refDataTransformed: dict
            info on runout line derived from reference datasets
        resAnalysisDF: data frame
            one row per simulation
        simRowHash: ID
            index of row for current sim in resAnalysisDF
        pathDict: dict
            dict with info on where to save plots

        Returns
        ----------
        resAnalysisDF: data frame
            updated DF
    """

    for item in refDataTransformed:
        refLine = refDataTransformed[item]

        if refLine['type'] in ['line', 'poly']:

            # simName is
            simName = resAnalysisDF.loc[simRowHash, 'simName']

            # add runoutLine differences as vector
            resAnalysisDF.at[simRowHash, 'runoutLineDiff_%s' % refLine['type']] = np.asarray(runoutLine['s'] - refLine['s'])

            # compute difference between runoutLine from simulation and reference data
            diffAll = runoutLine['s'] - refLine['s']
            # check where both lines have points to compute RMSE
            diffNoNans = diffAll[np.where((np.isnan(refLine['s']) == False) & (np.isnan(runoutLine['s']) == False))]
            # check number of points where runoutLine sim has no points but reference Line has points
            runoutLineNoPoints = len(np.where(np.isnan(runoutLine['s'][np.where(np.isnan(refLine['s']) == False)]))[0])
            runoutLineAllPoints = len(np.where(np.isnan(runoutLine['s']) == False)[0])
            # check number of points where refLine has no points but runout line sim has
            refLineNoPoints = len(np.where(np.isnan(refLine['s'][np.where(np.isnan(runoutLine['s']) == False)]))[0])
            refLineAllPoints = len(np.where(np.isnan(refLine['s']) == False)[0])
            runoutStr = '%d/%d' % (runoutLineNoPoints, refLineAllPoints)
            refLineStr = '%d/%d' % (refLineNoPoints, runoutLineAllPoints)

            # compute RMSE between runout line and refLine
            # set RMSE to nan if no runout line found for sim to compare to reference
            if np.all(np.isnan(runoutLine["s"])):
                RMSE = np.nan
            else:
                RMSE = np.sqrt(np.sum(diffNoNans**2) / len(diffNoNans))

            # plot differences in runout lines
            if len(np.where((np.isnan(runoutLine["s"]) == False))[0]) > 0:
                outAimec.plotRunoutLineComparisonToReference(
                    cfgSetup, refLine, runoutLine, pathDict, simName, runoutStr, refLineStr, RMSE, diffNoNans
                )
            else:
                log.warning(
                    "No runout line found for simulation: %s, not creating comparison to reference plot"
                    % simName
                )

            resAnalysisDF.at[simRowHash, 'runoutLineDiff_%s_pointsNotFoundInSim' % refLine['type']] = runoutStr
            resAnalysisDF.at[simRowHash, 'runoutLineDiff_%s_pointsNotFoundInRef' % refLine['type']] = refLineStr
            resAnalysisDF.at[simRowHash, 'runoutLineDiff_%s_RMSE' % refLine['type']] = RMSE

        else:
            log.info('For reference data type %s, runout line comparison is not available' % refLine['type'])

    return resAnalysisDF
