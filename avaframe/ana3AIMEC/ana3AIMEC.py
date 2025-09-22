"""
    AIMEC post processing workflow
"""

import logging
import numpy as np
import pathlib

# Local imports
from avaframe.in2Trans import shpConversion
from avaframe.ana3AIMEC import aimecTools
from avaframe.ana3AIMEC import dfa2Aimec
import avaframe.in2Trans.rasterUtils as IOf
import avaframe.out3Plot.outAIMEC as outAimec
import avaframe.in1Data.getInput as gI
from avaframe.in3Utils import geoTrans
from avaframe.com1DFA import com1DFA
import avaframe.in3Utils.fileHandlerUtils as fU


# create local logger
log = logging.getLogger(__name__)


def fullAimecAnalysis(avalancheDir, cfg, inputDir='', demFileName=''):
    """ fetch all data required to run aimec analysis, setup pathDict and reference sim,
        read the inputs, perform checks if all required data is available, run main aimec

        Parameters
        ------------
        avalancheDir: str or pathlib path
            path to avalanche directory
        cfg: configparser object
            main aimec configuration settings
        inputDir: str or pathlib path
            optional- directory where peak files are located, if '',
            avaDir/Outputs/comMod/peakFiles is set
        demFileName: pathlib path
            path to dem file

        Returns
        --------
        rasterTransfo: dict
            domain transformation information
        resAnalysisDF: dataFrame
            results of ana3AIMEC analysis
        plotDict: dict
            dictionary for report
        newRasters: dict
            dictionary containing pressure, velocity and flow thickness rasters after
            transformation for the reference and the current simulation

    """

    cfgSetup = cfg['AIMECSETUP']
    anaMod = cfgSetup['anaMod']

    # Setup input from computational module
    inputsDF, resTypeList, resTypeAndLayerTypeList, layerTypeList = dfa2Aimec.mainDfa2Aimec(
        avalancheDir, anaMod, cfg, inputDir=inputDir
    )

    # define reference simulation
    refSimRowHash, refSimName, inputsDF, colorParameter, valRef = aimecTools.fetchReferenceSimNo(avalancheDir, inputsDF, anaMod,
                                                                                         cfg, inputDir=inputDir)
    pathDict = {
        "refSimRowHash": refSimRowHash,
        "refSimName": refSimName,
        "compType": ["singleModule", anaMod],
        "colorParameter": colorParameter,
        "resTypeList": resTypeList,
        "valRef": valRef,
        "demFileName": demFileName,
        "layerTypeList": layerTypeList,
        "resTypeAndLayerTypeList": resTypeAndLayerTypeList,
    }
    pathDict = aimecTools.readAIMECinputs(avalancheDir, pathDict, cfgSetup.getboolean('defineRunoutArea'),
                                          dirName=anaMod)
    pathDict = aimecTools.checkAIMECinputs(cfgSetup, pathDict)
    log.info("Running ana3AIMEC model on test case DEM \n %s \n with profile \n %s ",
             pathDict['demSource'], pathDict['profileLayer'])
    # Run AIMEC postprocessing
    rasterTransfo, resAnalysisDF, plotDict, newRasters = mainAIMEC(pathDict, inputsDF, cfg)

    return rasterTransfo, resAnalysisDF, plotDict, newRasters, pathDict


def mainAIMEC(pathDict, inputsDF, cfg):
    """ Main logic for AIMEC postprocessing

    Reads the required files location for ana3AIMEC postprocessing
    given an avalanche directory
    Make the domain transformation corresponding to the input avalanche path
    Transform 2D fields (pressure, flow thickness ...)
    Analyze transformed rasters and mass
    Produce plots and report

    Parameters
    ----------
    pathDict : dict
        dictionary with paths to dem and lines for Aimec analysis
    inputsDF : dataFrame
        dataframe with simulations to analyze and associated path to raster data
    cfg : configparser
        configparser with ana3AIMEC settings defined in ana3AIMECCfg.ini

    Returns
    -------
    rasterTransfo: dict
        domain transformation information
    resAnalysisDF: dataFrame
        results of ana3AIMEC analysis
    plotDict: dict
        dictionary for report
    """

    # Extract input config parameters
    cfgSetup = cfg['AIMECSETUP']
    cfgFlags = cfg['FLAGS']
    cfgPlots = cfg['PLOTS']
    interpMethod = cfgSetup['interpMethod']

    log.info('Prepare data for post-processing')
    # Read input dem
    demSource = pathDict['demSource']
    dem = IOf.readRaster(demSource)
    # get hash of the reference
    refSimRowHash = pathDict['refSimRowHash']
    # read reference file and raster and config
    refResultSource = inputsDF.loc[
        refSimRowHash, cfgSetup["runoutResType"] + "_" + cfgSetup["referenceSimLayer"]
    ]
    refRaster = IOf.readRaster(refResultSource)
    refRasterData = refRaster['rasterData']
    refHeader = refRaster['header']

    log.debug('Data-file %s analysed and domain transformation done' % demSource)

    # Make domain transformation
    log.debug("Creating new deskewed raster and preparing new raster assignment function")
    log.debug('Analyze and make domain transformation on Data-file %s' % demSource)
    rasterTransfo = aimecTools.makeDomainTransfo(pathDict, dem, refHeader['cellsize'], cfgSetup)

    # ####################################################
    # visualisation
    # TODO: needs to be moved somewhere else
    slRaster = aimecTools.transform(refRaster, refResultSource, rasterTransfo, interpMethod)
    newRasters = {}
    log.debug("Assigning dem data to deskewed raster")
    newRasters['newRasterDEM'] = aimecTools.transform(dem, demSource, rasterTransfo, interpMethod)

    inputData = {}
    inputData['slRaster'] = slRaster
    inputData['xyRaster'] = refRasterData
    inputData['xyHeader'] = refHeader
    outAimec.visuTransfo(rasterTransfo, inputData, cfgSetup, pathDict)

    # postprocess reference...
    contourDict = {}
    timeMass = None
    # add fields that will be filled in analysis
    resAnalysisDF = aimecTools.addFieldsToDF(inputsDF, pathDict["layerTypeList"])

    # if includeReference add reference data to resAnalysisDF
    if cfgSetup.getboolean('includeReference'):
        referenceDF = aimecTools.createReferenceDF(pathDict)
        refDataTransformed, referenceDF = postProcessReference(cfg, rasterTransfo, pathDict, referenceDF, newRasters)
    else:
        refDataTransformed = {}

    # postprocess reference simulation
    resAnalysisDF, newRasters, timeMass, contourDict = postProcessAIMEC(cfg, rasterTransfo, pathDict, resAnalysisDF, newRasters,
                                                           timeMass, refSimRowHash, contourDict, refDataTransformed)
    # postprocess other simulations
    for simRowHash, inputsDFrow in inputsDF.iterrows():
        if simRowHash != refSimRowHash:
            resAnalysisDF, newRasters, timeMass, contourDict = postProcessAIMEC(cfg, rasterTransfo, pathDict, resAnalysisDF,
                                                                   newRasters, timeMass, simRowHash, contourDict,
                                                                                refDataTransformed)
            pathDict['simRowHash'] = simRowHash

    # -----------------------------------------------------------
    # result visualisation + report
    # -----------------------------------------------------------
    plotDict = {}
    log.info('Visualisation of AIMEC results')

    # plot the contour lines of all sims for the thresholdValue of runoutResType
    outAimec.plotContoursTransformed(contourDict, pathDict, rasterTransfo, cfgSetup, inputsDF)

    if sorted(pathDict['resTypeList']) == sorted(['ppr', 'pft', 'pfv']) and cfgPlots.getboolean('extraPlots'):
        outAimec.visuSimple(cfgSetup, rasterTransfo, resAnalysisDF, newRasters, pathDict)
    if len(resAnalysisDF.index) == 2 and sorted(pathDict['resTypeList']) == sorted(['ppr', 'pft', 'pfv']):
        plotName = outAimec.visuRunoutComp(rasterTransfo, resAnalysisDF, cfgSetup, pathDict)

    plotName = outAimec.visuRunoutStat(rasterTransfo, inputsDF, resAnalysisDF, newRasters, cfgSetup, pathDict)

    areaDict = {('area comparison plot ' + resAnalysisDF.loc[k, 'simName']):
                v for k, v in resAnalysisDF['areasPlot'].to_dict().items()}
    plotDict['areasPlot'] = {'Aimec area analysis': areaDict}
    if cfgFlags.getboolean('flagMass'):
        massDict = {('mass comparison plot ' + resAnalysisDF.loc[k, 'simName']):
                    v for k, v in resAnalysisDF['massPlotName'].to_dict().items()}
        plotDict['massAnalysisPlot'] = {'Aimec mass analysis': massDict}

    if cfgPlots.getboolean('extraPlots'):
        outAimec.resultVisu(cfgSetup, inputsDF, pathDict, cfgFlags, rasterTransfo, resAnalysisDF)

    plotDict['slCompPlot'] = {'Aimec comparison of mean and max values along path': {'AIMEC '
                                                                                     + cfgSetup['runoutResType']
                                                                                     + ' analysis': plotName}}

    log.info('Writing results to file')
    outAimec.resultWrite(pathDict, cfg, rasterTransfo, resAnalysisDF)

    # create comparison plots of two scalar result variables or derived quantities
    if len(resAnalysisDF.index) > 1:
        compResTypes1 = cfgPlots['compResType1'].split('|')
        compResTypes2 = cfgPlots['compResType2'].split('|')
        for indRes, compResType in enumerate(compResTypes1):
            outAimec.plotMaxValuesComp(pathDict, resAnalysisDF, compResType, compResTypes2[indRes],
                                       hue=cfgPlots['scenarioName'])

    return rasterTransfo, resAnalysisDF, plotDict, newRasters


def postProcessAIMEC(cfg, rasterTransfo, pathDict, resAnalysisDF, newRasters, timeMass, simRowHash, contourDict, refDataTransformed):
    """ Apply domain transformation and analyse result data (for example pressure, thickness, velocity...)

    Apply the domain tranformation to peak results
    Analyse them.
    Calculate runout, Max Peak Values, Average Peak Values...
    Get mass and entrainment

    The output resAnalysisDF contains:

    -maxACrossMax: float
        max max A (A depends on what is in resTypes)
    -ACrossMax: 1D numpy array
        max A in each cross section (A depends on what is in resTypes)
    -ACrossMean: 1D numpy array
        mean A in each cross section (A depends on what is in resTypes)
    -xRunout: float
        x coord of the runout point calculated from the
        MAX peak result in each cross section (runoutResType provided in the ini file)
    -yRunout: float
        y coord of the runout point calculated from the
        MAX peak result in each cross section (runoutResType provided in the ini file)
    -sRunout: float
        projected runout distance calculated from the
        MAX peak result in each cross section (runoutResType provided in the ini file)
    -xMeanRunout: float
        x coord of the runout point calculated from the
        MEAN peak result in each cross section (runoutResType provided in the ini file)
    -yMeanRunout: float
        y coord of the runout point calculated from the
        MEAN peak result in each cross section (runoutResType provided in the ini file)
    -sMeanRunout: float
        projected runout distance calculated from the
        MEAN peak result in each cross section (runoutResType provided in the ini file)
    -elevRel: float
        elevation of the release area (based on first point with
        peak field > thresholdValue)
    -deltaH: float
        elevation fall difference between elevRel and altitude of
        run-out point

    -TP: float
        ref = True sim2 = True
    -FN: float
        ref = False sim2 = True
    -FP: float
        ref = True sim2 = False
    -TN: float
        ref = False sim2 = False
    -resTypeFieldMax: float
        max value of peak field resType raster (original raster read from result file (no coordinate transformation)
    -resTypeFieldMin: float
        min value of peak field resType raster (original raster read from result file (no coordinate transformation)
        0 values are masked
    -resTypeFieldMean: float
        mean value of peak field resType raster (original raster read from result file (no coordinate transformation)
        0 values are masked
    -resTypeFieldStd: float
        std value of peak field resType raster (original raster read from result file (no coordinate transformation)
        0 values are masked


    if mass analysis is performed

    -relMass: float
        release mass
    -entMass: float
        entrained mass
    -finalMass: float
        final mass
    -relativMassDiff: float
        the final mass diff with ref (in %)
    -growthIndex: float
        growth index
    -growthGrad: float
        growth gradient

    Parameters
    ----------
    cfg: configParser objec
        parameters for AIMEC analysis
    rasterTransfo: dict
        transformation information
    pathDict : dict
        dictionary with paths to dem and lines for Aimec analysis
    resAnalysisDF : dataFrame
        dataframe with simulations to analyze and associated path to raster data
    newRasters: dict
        dictionary containing pressure, velocity and flow thickness rasters after
        transformation for the reference and the current simulation
    timeMass: 1D numpy array
        time array for mass analysis (if flagMass=True, otherwise None)
    simRowHash: str
        dataframe hash of the current simulation to analyze
    contourDict: dict
        dictionary with one key per sim and its x, y coordinates for contour line of runoutresType
        for thresholdValue
    refDataTransformed: dict
        dictionary with info on reference data sets

    Returns
    -------
    resAnalysisDF: dataFrame
        input DF with results from Aimec Analysis updated with results from current simulation
    contourDict: dict
        dictionary with one key per sim and its x, y coordinates for contour line of runoutresType
        for thresholdValue - updated with info for current simulation

    """
    cfgSetup = cfg['AIMECSETUP']
    cfgFlags = cfg['FLAGS']
    cfgPlots = cfg['PLOTS']
    interpMethod = cfgSetup['interpMethod']
    flagMass = cfgFlags.getboolean('flagMass')
    refSimRowHash = pathDict['refSimRowHash']
    resTypeList = pathDict['resTypeList']
    layerTypeList = pathDict["layerTypeList"]
    resTypeAndLayerTypeList = pathDict["resTypeAndLayerTypeList"]

    # apply domain transformation
    log.info('Analyzing data in path coordinate system')
    for resType in resTypeAndLayerTypeList:
        log.debug("Assigning %s data to deskewed raster" % resType)
        inputFiles = resAnalysisDF.loc[simRowHash, resType]
        if isinstance(inputFiles, pathlib.PurePath):
            rasterData = IOf.readRaster(inputFiles)
            newRaster = aimecTools.transform(rasterData, inputFiles, rasterTransfo, interpMethod)
            newRasters['newRaster' + resType.upper()] = newRaster
            if simRowHash == refSimRowHash:
                newRasters['newRefRaster' + resType.upper()] = newRaster
                resAnalysisDF[resType + 'CrossMax'] = np.nan
                resAnalysisDF[resType + 'CrossMax'] = resAnalysisDF[resType + 'CrossMax'].astype(object)
                resAnalysisDF[resType + 'CrossMin'] = np.nan
                resAnalysisDF[resType + 'CrossMin'] = resAnalysisDF[resType + 'CrossMax'].astype(object)
                resAnalysisDF[resType + 'CrossMean'] = np.nan
                resAnalysisDF[resType + 'CrossMean'] = resAnalysisDF[resType + 'CrossMax'].astype(object)
                if cfgSetup.getboolean('includeReference'):
                    resAnalysisDF['runoutLineDiff_line'] = np.nan
                    resAnalysisDF['runoutLineDiff_line'] = resAnalysisDF['runoutLineDiff_line'].astype(object)
                    resAnalysisDF['runoutLineDiff_poly'] = np.nan
                    resAnalysisDF['runoutLineDiff_poly'] = resAnalysisDF['runoutLineDiff_line'].astype(object)

            # only extract values if nonzero values found in rasterData; if only 0 values set to 0
            if np.any(rasterData["rasterData"]):
                # add max, min and std values of result fields
                resAnalysisDF.at[simRowHash, resType + "FieldMax"] = np.nanmax(rasterData["rasterData"])
                # for mean and min values and std, only take peak field values != 0
                maskedRaster = np.where(
                    rasterData["rasterData"] < cfgSetup.getfloat("minValueField"),
                    np.nan,
                    rasterData["rasterData"],
                )
                resAnalysisDF.at[simRowHash, resType + "FieldMin"] = np.nanmin(maskedRaster)
                resAnalysisDF.at[simRowHash, resType + "FieldMean"] = np.nanmean(maskedRaster)
                resAnalysisDF.at[simRowHash, resType + "FieldStd"] = np.nanstd(maskedRaster)
            else:
                resAnalysisDF.at[simRowHash, resType + "FieldMax"] = 0.0
                resAnalysisDF.at[simRowHash, resType + "FieldMin"] = 0.0
                resAnalysisDF.at[simRowHash, resType + "FieldMean"] = 0.0
                resAnalysisDF.at[simRowHash, resType + "FieldStd"] = 0.0

            # analyze all fields
            resAnalysisDF = aimecTools.analyzeField(
                simRowHash, rasterTransfo, newRaster, resType, resAnalysisDF
            )

    # compute runout based on runoutResType
    resAnalysisDF = aimecTools.computeRunOut(cfgSetup, rasterTransfo, resAnalysisDF, newRasters, simRowHash)
    runoutLine = aimecTools.computeRunoutLine(cfgSetup, rasterTransfo, newRasters, simRowHash, 'simulation', name='')

    if 'refPoint' in refDataTransformed:
        # compute differences between runout points
        resAnalysisDF = aimecTools.computeRunoutPointDiff(resAnalysisDF, refDataTransformed['refPoint'], simRowHash)

    # plot comparison between runout lines
    outAimec.compareRunoutLines(cfgSetup, refDataTransformed, newRasters['newRaster'+cfgSetup['runoutResType'].upper()],
                                runoutLine, rasterTransfo, resAnalysisDF.loc[simRowHash], pathDict)

    # analyze distribution of diffs between runout lines
    resAnalysisDF = aimecTools.analyzeDiffsRunoutLines(cfgSetup, runoutLine, refDataTransformed, resAnalysisDF, simRowHash, pathDict)

    if flagMass:
        fnameMass = resAnalysisDF.loc[simRowHash, 'massBal']
        resAnalysisDF, timeMass = aimecTools.analyzeMass(fnameMass, simRowHash, refSimRowHash, resAnalysisDF, time=timeMass)

        massPlotName = outAimec.visuMass(resAnalysisDF, pathDict, simRowHash, refSimRowHash, timeMass)
        resAnalysisDF.loc[simRowHash, 'massPlotName'] = massPlotName
    else:
        timeMass = None

    # TODO also include comparison to reference area if provided by polygon or raster (referenceType name of newRasters)
    resAnalysisDF, compPlotPath, contourDict = aimecTools.analyzeArea(rasterTransfo, resAnalysisDF, simRowHash, newRasters, cfg,
                                                 pathDict, contourDict, referenceType='newRefRaster')

    resAnalysisDF.loc[simRowHash, 'areasPlot'] = compPlotPath

    if 'pftCrossMax' in resAnalysisDF.columns and 'pfvCrossMax' in resAnalysisDF.columns:
        outAimec.plotVelThAlongThalweg(pathDict, rasterTransfo, resAnalysisDF['pftCrossMax'].loc[simRowHash],
                                       resAnalysisDF['pfvCrossMax'].loc[simRowHash], cfgPlots,
                                       resAnalysisDF['simName'].loc[simRowHash])

    return resAnalysisDF, newRasters, timeMass, contourDict


def postProcessReference(cfg, rasterTransfo, pathDict, referenceDF, newRasters):
    """ Apply domain transformation and analyse reference data, e.g. compute runout point

    Apply the domain tranformation to reference data sets
    Analyse them.
    Calculate runout

    The output referenceDF contains:

    -xRunout: float
        x coord of the runout point calculated
    -yRunout: float
        y coord of the runout point calculated
    -sRunout: float
        runout point in s coordinate
    -lRunout: float
        runout point in l coordinate

    Parameters
    ----------
    cfg: configParser objec
        parameters for AIMEC analysis
    rasterTransfo: dict
        transformation information
    pathDict : dict
        dictionary with paths to dem and lines for Aimec analysis and reference data file paths
    referenceDF : dataFrame
        dataframe one row per reference data set
    newRasters: dict
        dictionary containing pressure, velocity and flow thickness rasters after
        transformation for the reference and the current simulation

    Returns
    -------
    referenceDataTransformed: dict
        dictionary with refLine, refPoint, refPoly info on reference data set and computed runout line or point
        - transformed and computed in s l coordinate system
    referenceDF: dataFrame
       updated df
    """

    cfgSetup = cfg['AIMECSETUP']
    resampleLineFactor = cfgSetup.getfloat('resampleLineFactor')
    interpMethod = cfgSetup['interpMethod']

    # apply domain transformation
    log.info('Analyzing reference data in path coordinate system')

    # read reference LINE, RUNOUTPOINT, POLY and convert to s, l coordinate system (intermediate step: create raster)
    # first read DEM (needed for transformation)
    dem = rasterTransfo['dem']
    refDataTransformed = {}

    for ind, refFile in enumerate(pathDict['referenceLine']):
        log.debug('Convert line: %s to raster' % refFile.stem)
        refLine = shpConversion.readLine(refFile, "referenceLine", dem)
        refLine, _ = geoTrans.prepareLine(dem, refLine, distance=(dem['header']['cellsize']/resampleLineFactor), Point=None, k=1)

        # derive raster from line
        _, refRasterXY = geoTrans.projectOnGrid(refLine['x'], refLine['y'], dem['rasterData'],
                                              csz=dem['header']['cellsize'], xllc=dem['header']['xllcenter'],
                                              yllc=dem['header']['yllcenter'], interp="bilinear", getXYField=True)

        # transform xy raster into sl raster and compute where line lies in this raster
        refRasterSL = aimecTools.transform({'header': dem['header'], 'rasterData': refRasterXY}, refFile.stem, rasterTransfo, interpMethod)
        newRasters['refLine'] = refRasterSL
        refLineSL = aimecTools.computeRunoutLine(cfgSetup, rasterTransfo, newRasters, '',
                                                                'line', ('refLine'), basedOnMax=True)
        refDataTransformed['refLine'] = refLineSL

        # create plot from reference line transformation
        outAimec.referenceLinePlot(refRasterXY, dem, refLine, rasterTransfo, refRasterSL, refLineSL, pathDict)

        # add info to DF
        referenceDF = aimecTools.addReferenceAnalysisTODF(referenceDF, refFile, refLineSL)

    for ind, refFile in enumerate(pathDict['referencePoint']):
        log.debug('Convert point: %s to raster' % refFile.stem)
        refPoint = shpConversion.readPoints(refFile, dem)

        # check if more than one point provided
        if len(refPoint['x']) > 1:
            message = 'More than one point in reference data file: %s - only one is allowed' % refFile.stem
            log.error(message)
            raise AssertionError(message)

        # derive raster from point
        _, refRasterXY = geoTrans.projectOnGrid(refPoint['x'], refPoint['y'], dem['rasterData'],
                                              csz=dem['header']['cellsize'], xllc=dem['header']['xllcenter'],
                                              yllc=dem['header']['yllcenter'], interp="nearest", getXYField=True)

        # transform xy raster into sl raster and compute where point lies in this raster
        refRasterSL = aimecTools.transform({'header': dem['header'], 'rasterData': refRasterXY}, refFile.stem, rasterTransfo, interpMethod)
        newRasters['refPoint'] = refRasterSL
        refPointSL = aimecTools.computeRunoutLine(cfgSetup, rasterTransfo, newRasters, '',
                                                                 'point', ('refPoint'), basedOnMax=True)
        refDataTransformed['refPoint'] = refPointSL

        # add info to DF
        referenceDF = aimecTools.addReferenceAnalysisTODF(referenceDF, refFile, refPointSL)

        # create plot from reference line transformation
        outAimec.referenceLinePlot(refRasterXY, dem, refPoint, rasterTransfo, refRasterSL, refPointSL, pathDict)

    for ind, refFile in enumerate(pathDict['referencePolygon']):
        log.debug('Convert point: %s to raster' % refFile.stem)

        # derive area from polygon in xy coordinates
        dem['originalHeader'] = dem['header']
        refPoly = shpConversion.readLine(refFile, "referencepoly", dem)
        refPoly = geoTrans.prepareArea(refPoly,
            dem,
            np.sqrt(dem['header']['cellsize']**2),
            thList='',
            combine=True,
            checkOverlap=False,
        )
        refRasterXY = refPoly['rasterData']

        # transform xy raster into sl raster and compute where runout line is in this raster
        refRasterSL = aimecTools.transform({'header': dem['header'], 'rasterData': refRasterXY}, refFile.stem, rasterTransfo, interpMethod)
        newRasters['refPoly'] = refRasterSL
        refPolySL = aimecTools.computeRunoutLine(cfgSetup, rasterTransfo, newRasters, '', 'poly', 'refPoly',
                                                 basedOnMax=True)
        refDataTransformed['refPoly'] = refPolySL

        # add info to DF
        referenceDF = aimecTools.addReferenceAnalysisTODF(referenceDF, refFile, refPolySL)

        # create plot from reference line transformation
        outAimec.referenceLinePlot(refRasterXY, dem, refPoly, rasterTransfo, refRasterSL, refPolySL, pathDict)

    return refDataTransformed, referenceDF


def aimecRes2ReportDict(resAnalysisDF, reportD, benchD, pathDict):
    """ Gather aimec results and append them to report the dictionary

    Parameters
    ----------
    resAnalysisDF : dataFrame
        dataframe with aimec analysis results
    reportD: dict
        report dictionary to be updated
    benchD: dict
        benchmark dictionary to be updated
    pathDict : dict
        dictionary with refSimRowHash

    Returns
    --------
    reportD: dict
        report dictionary updated with the 'Aimec analysis' section
    benchD: dict
        benchmark dictionary updated with the 'Aimec analysis' section
    """
    refSimRowHash = pathDict['refSimRowHash']
    resAnalysisDFRef = resAnalysisDF.loc[refSimRowHash].squeeze().to_dict()
    resAnalysisDFComp = resAnalysisDF.loc[resAnalysisDF.index != refSimRowHash].squeeze().to_dict()
    reportD['Aimec analysis'] = {'type': 'list', 'runout [m]': resAnalysisDFComp['sRunout'],
                                 'max peak pressure [kPa]': resAnalysisDFComp['maxpprCrossMax'],
                                 'max peak flow thickness [m]': resAnalysisDFComp['maxpftCrossMax'],
                                 'max peak flow velocity [ms-1]': resAnalysisDFComp['maxpfvCrossMax']}

    benchD['Aimec analysis'] = {'type': 'list', 'runout [m]': resAnalysisDFRef['sRunout'],
                                'max peak pressure [kPa]': resAnalysisDFRef['maxpprCrossMax'],
                                'max peak flow thickness [m]': resAnalysisDFRef['maxpftCrossMax'],
                                'max peak flow velocity [ms-1]': resAnalysisDFRef['maxpfvCrossMax']}
    # add mass info
    if "relMass" in resAnalysisDF.columns:
        reportD['Aimec analysis'].update({'Initial mass [kg]': resAnalysisDFComp['relMass']})
        reportD['Aimec analysis'].update({'Final mass [kg]': resAnalysisDFComp['finalMass']})
        reportD['Aimec analysis'].update({'Entrained mass [kg]': resAnalysisDFComp['entMass']})
        benchD['Aimec analysis'].update({'Initial mass [kg]': resAnalysisDFRef['relMass']})
        benchD['Aimec analysis'].update({'Final mass [kg]': resAnalysisDFRef['finalMass']})
        benchD['Aimec analysis'].update({'Entrained mass [kg]': resAnalysisDFRef['entMass']})

    return reportD, benchD


def aimecTransform(rasterTransfo, particle, demHeader, timeSeries=False):
    """ Adding s projected and l projected in thalweg/aimec coordinate system to the particle dictionary

        Parameters
        ----------
        rasterTransfo: dict
            domain transformation information
        particle: dict
            dictionary with particle properties, particles x,y coordinate origin is 0,0
        demHeader: dict
            dict with info on dem cellSize, xllcenter, ..
        timeSeries: bool
            if timeSeries consider that particles dict is provided as time series
            if False particles dict is provided for one time step only

        Returns
        -------
        particle: dict
            particle dictionary with s grid and l grid added
    """

    xllcenter = demHeader['xllcenter']
    yllcenter = demHeader['yllcenter']

    if timeSeries:
        particle['lAimec'] = np.zeros((particle['x'].shape[0],particle['x'].shape[1]))
        particle['sAimec'] = np.zeros((particle['x'].shape[0],particle['x'].shape[1]))
        for timeInd in range(particle['x'].shape[0]):
            sList, lList = computeSLParticles(rasterTransfo, demHeader,
                particle['x'][timeInd,:], particle['y'][timeInd,:])

            #TODO: consider renaming to ana3AIMEC_s, ana3AIMEC_l
            particle['lAimec'][timeInd, :] = lList
            particle['sAimec'][timeInd, :] = sList
            particle['sBetaPoint'] = rasterTransfo['s'][rasterTransfo['indStartOfRunout']]
            particle['beta'] = rasterTransfo['startOfRunoutAreaAngle']
    else:
        sList, lList = computeSLParticles(rasterTransfo, demHeader,
            particle['x'], particle['y'])

        #TODO: consider renaming to ana3AIMEC_s, ana3AIMEC_l
        particle['lAimec'] = lList
        particle['sAimec'] = sList

    return(particle)


def computeSLParticles(rasterTransfo, demHeader, particlesX, particlesY):
    """ find the closest s, l coordinates in the aimec/thalweg coordinate system to a particles x,y location

        Parameters
        -----------
        rasterTransfo: dict
            info on rasterTransformation, here gridx, gridy coordinates
        demHeader: dict
            dict with info on dem xllcenter, yllcenter
        particlesX: np array
            x coordinates of particles location for one time step but all particles (origin 0,0)
        particlesY: np array
            Y coordinates of particles location for one time step but all particles (origin 0,0)

        Returns
        --------
        sList, lList: list
            list of s, l coordinates for respective particle
    """

    xllcenter = demHeader['xllcenter']
    yllcenter = demHeader['yllcenter']
    sList = []
    lList = []

    # loop over all particles
    for x, y in zip(particlesX, particlesY):
        # calculating the distance between the particle position and the grid points
        distance = np.sqrt((x+xllcenter-rasterTransfo['gridx'])**2 + (y+yllcenter-rasterTransfo['gridy'])**2)
        # Finding the coordinates of the grid point which minimizes the difference
        (sIndex, lIndex) = np.unravel_index(np.argmin(distance, axis=None), distance.shape)
        if np.isnan(x) or np.isnan(y):
            lList.append(np.nan)
            sList.append(np.nan)
        else:
            lList.append(rasterTransfo['l'][lIndex])
            sList.append(rasterTransfo['s'][sIndex])


    return sList, lList


def addSLToParticles(avaDir, cfgAimec, demFileName, particlesList, saveToPickle=False):
    """ add aimec (thalweg) s,l coordinates to particle dicts
        use dem from sim corresponding to particle dicts by providing demFileName
        if demFileName='' standard dem in avaDir/Inputs is used

        Parameters
        ------------
        avaDir: pathlib path
            path to avalanche directory
        cfgAimec: configparser object
            configuration settings of aimec
        demFileName: str
            path including fileName and extension to demFile located in avaDir/Inputs
        particlesList: list
            list of particles dicts of sim
        saveToPickle: bool
            if updated particle dicts shall be saved to pickle

        Returns
        --------
        particlesList: list
            list of particles dicts updated with s,l
        rasterTransfo: dict
            dictionary with info on rasterTransformation

    """

    # create path dict
    pathDict = {}
    pathDict = aimecTools.readAIMECinputs(avaDir, pathDict, cfgAimec['AIMECSETUP'].getboolean('defineRunoutArea'),
        dirName=cfgAimec['AIMECSETUP']['anaMod'])

    # fetch dem of sim
    demFilePath = gI.getDEMFromConfig(avaDir, fileName=demFileName)
    dem = IOf.readRaster(demFilePath)
    dem['originalHeader'] = dem['header']

    # create rasterTransfo info
    # use cell size of dem, as dem has to match cellsize of sim results if fetched using getDEMFromConfig
    rasterTransfo = aimecTools.makeDomainTransfo(pathDict, dem, dem['header']['cellsize'],
        cfgAimec['AIMECSETUP'])

    for particleDict in particlesList:
        particleDict = aimecTransform(rasterTransfo, particleDict, dem['header'], timeSeries=False)
        particleDict['sBetaPoint'] = rasterTransfo['s'][rasterTransfo['indStartOfRunout']]
        particleDict['beta'] = rasterTransfo['startOfRunoutAreaAngle']
        simName = particleDict['simName']
        if saveToPickle:
            # save pickle file
            outDir = pathlib.Path(avaDir, 'Outputs', 'ana3AIMEC', 'particles')
            fU.makeADir(outDir)
            com1DFA.savePartToPickle(particleDict, outDir, simName)

    if saveToPickle:
        log.info('Saving particles updated with s,l to directory: %s' % outDir)

    # add header of dem that is used in rasterTransfo to dict
    rasterTransfo['demHeader'] = dem['header']

    return particlesList, rasterTransfo, dem
