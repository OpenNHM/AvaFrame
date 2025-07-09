#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
com4FlowPy main function
mainly handling input of data, model params
and output of model results
"""
# Load modules
import pathlib
import numpy as np
from datetime import datetime
import logging
import pickle
import shutil
import os
import sys

# Local imports (avaFrame API)
from avaframe.in1Data import getInput as gI
import avaframe.in3Utils.initialiseDirs as inDirs
from avaframe.in3Utils import fileHandlerUtils as fU
import avaframe.in2Trans.shpConversion as shpConv
import avaframe.in2Trans.rasterUtils as IOf
import avaframe.in3Utils.geoTrans as gT

# com4FlowPy Libraries
import avaframe.com4FlowPy.flowCore as fc
import avaframe.com4FlowPy.splitAndMerge as SPAM

# create local logger
log = logging.getLogger(__name__)


def com4FlowPyMain(cfgPath, cfgSetup):
    """com4FlowPy main function performs the model run and writes results to disk:
        * reading of input data and model Parameters
        * calculation
        * writing of result files

    the function assumes that all necessary directories inside cfgPath have
    already been created (workDir, tempDir)

    Parameters
    -----------
    cfgPath: configparser.SectionProxy Object
        contains paths to input data (from .ini file)
    cfgSetup: configparser.SectionProxy Object
        "GENERAL" model configs (from .ini file)

    """
    _startTime = datetime.now().replace(microsecond=0)  # used for timing model runtime

    modelParameters = {}
    # Flow-Py parameters
    modelParameters["alpha"] = cfgSetup.getfloat("alpha")  # float(cfgSetup["alpha"])
    modelParameters["exp"] = cfgSetup.getfloat("exp")  # float(cfgSetup["exp"])
    modelParameters["flux_threshold"] = cfgSetup.getfloat("flux_threshold")  # float(cfgSetup["flux_threshold"])
    modelParameters["max_z"] = cfgSetup.getfloat("max_z")  # float(cfgSetup["max_z"])

    # Flags for use of Forest and/or Infrastructure
    modelParameters["infraBool"] = cfgSetup.getboolean("infra")
    modelParameters["forestBool"] = cfgSetup.getboolean("forest")
    modelParameters["forestInteraction"] = cfgSetup.getboolean("forestInteraction")
    # modelParameters["infra"]  = cfgSetup["infra"]
    # modelParameters["forest"] = cfgSetup["forest"]

    # Flag for preview mode
    modelParameters["previewMode"] = cfgSetup.getboolean("previewMode")

    # Flags for use of dynamic input parameters
    modelParameters["varUmaxBool"] = cfgSetup.getboolean("variableUmaxLim")
    modelParameters["varAlphaBool"] = cfgSetup.getboolean("variableAlpha")
    modelParameters["varExponentBool"] = cfgSetup.getboolean("variableExponent")

    # Flag for use of old flux distribution version
    modelParameters["fluxDistOldVersionBool"] = cfgSetup.getboolean("fluxDistOldVersion")

    # Tiling Parameters used for calculation of large model-domains
    tilingParameters = {}
    tilingParameters["tileSize"] = cfgSetup.getfloat("tileSize")  # float(cfgSetup["tileSize"])
    tilingParameters["tileOverlap"] = cfgSetup.getfloat("tileOverlap")  # float(cfgSetup["tileOverlap"])

    # Paths
    modelPaths = {}
    modelPaths["outDir"] = cfgPath["outDir"]
    modelPaths["workDir"] = cfgPath["workDir"]
    modelPaths["demPath"] = cfgPath["demPath"]
    modelPaths["releasePath"] = cfgPath["releasePath"]

    modelPaths["resDir"] = cfgPath["resDir"]
    modelPaths["tempDir"] = cfgPath["tempDir"]
    modelPaths["uid"] = cfgPath["uid"]
    modelPaths["timeString"] = cfgPath["timeString"]
    modelPaths["outputFileList"] = cfgPath["outputFiles"].split('|')

    modelPaths["outputFileFormat"] = cfgPath["outputFileFormat"]
    if modelPaths["outputFileFormat"] in [".asc", ".ASC"]:
        modelPaths["outputFileFormat"] = '.asc'
    else:
        modelPaths["outputFileFormat"] = '.tif'

    # check if 'customDirs' are used - alternative is 'default' AvaFrame Folder Structure
    modelPaths["useCustomDirs"] = True if cfgPath["customDirs"] == "True" else False
    # check if the temp folder, where intermediate results are stored, should be deleted after writing output files
    modelPaths["deleteTempFolder"] = True if cfgPath["deleteTemp"] == "True" else False

    # Multiprocessing Options
    MPOptions = {}
    MPOptions["nCPU"] = cfgSetup.getint("cpuCount")  # int(cfgSetup["cpuCount"]) #number of CPUs to use
    MPOptions["procPerCPU"] = cfgSetup.getint("procPerCPUCore")  # int(cfgSetup["procPerCPUCore"]) #processes per core
    MPOptions["chunkSize"] = cfgSetup.getint("chunkSize")  # int(cfgSetup["chunkSize"]) # default task size for MP
    MPOptions["maxChunks"] = cfgSetup.getint("maxChunks")  # int(cfgSetup["maxChunks"]) # max number of tasks for MP

    # check if calculation with infrastructure
    if modelParameters["infraBool"]:
        modelPaths["infraPath"] = cfgPath["infraPath"]
    else:
        modelPaths["infraPath"] = ""

    forestParams = {}
    # check if calculation with forest
    if modelParameters["forestBool"]:

        forestParams["forestModule"] = cfgSetup.get("forestModule")
        modelPaths["forestPath"] = cfgPath["forestPath"]
        # 'forestFriction' and 'forestDetrainment' parameters
        forestParams["maxAddedFriction"] = cfgSetup.getfloat("maxAddedFrictionFor")  # float(cfgSetup["maxAddedFr.."])
        forestParams["minAddedFriction"] = cfgSetup.getfloat("minAddedFrictionFor")  # float(cfgSetup["minAddedFr.."])
        forestParams["velThForFriction"] = cfgSetup.getfloat("velThForFriction")  # float(cfgSetup["velThForFriction"])
        forestParams["maxDetrainment"] = cfgSetup.getfloat("maxDetrainmentFor")  # float(cfgSetup["maxDetrainmentFor"])
        forestParams["minDetrainment"] = cfgSetup.getfloat("minDetrainmentFor")  # float(cfgSetup["minDetrainmentFor"])
        forestParams["velThForDetrain"] = cfgSetup.getfloat("velThForDetrain")  # float(cfgSetup["velThForDetrain"])
        # 'forestFrictionLayer' parameter
        forestParams["fFrLayerType"] = cfgSetup.get("forestFrictionLayerType")
        # skipForestDist - no forest friciton effect assumed while distance along path <= skipForestDist
        forestParams["skipForestDist"] = cfgSetup.getfloat("skipForestDist")

    else:
        modelPaths["forestPath"] = ""
        modelParameters["forestInteraction"] = False

    # check if with u_max limit
    if modelParameters["varUmaxBool"]:
        modelParameters["varUmaxType"] = cfgSetup.get("varUmaxParameter")
        modelPaths["varUmaxPath"] = cfgPath["varUmaxPath"]
    else:
        modelPaths["varUmaxPath"] = ""
    # check if we use the variable alpha layer
    if modelParameters["varAlphaBool"]:
        modelPaths["varAlphaPath"] = cfgPath["varAlphaPath"]
    else:
        modelPaths["varAlphaPath"] = ""
    # check if we use the variable exponent layer
    if modelParameters["varExponentBool"]:
        modelPaths["varExponentPath"] = cfgPath["varExponentPath"]
    else:
        modelPaths["varExponentPath"] = ""

    # TODO: provide some kind of check for the model Parameters
    #       i.e. * sensible value ranges
    #            * contradicting options ...

    # write model parameters paths, etc. to logfile
    startLogging(modelParameters, forestParams, modelPaths, MPOptions)

    # check if release file is given als .shp and convert to .tif/.asc in that case
    # NOTE-TODO: maybe also handle this in ../runCom4FlowPy.py
    modelPaths = checkConvertReleaseShp2Tif(modelPaths)

    # check if input layers have same x,y dimensions
    checkInputLayerDimensions(modelParameters, modelPaths)

    # check if input parameters are within physically sensible ranges
    checkInputParameterValues(modelParameters, modelPaths)

    # get information on cellsize and nodata value from demHeader
    rasterAttributes = {}
    
    demHeader = IOf.readRasterHeader(modelPaths["demPath"])
    rasterAttributes["nodata"] = demHeader["nodata_value"]
    rasterAttributes["cellsize"] = demHeader["cellsize"]

    # tile input layers and write tiles (pickled np.arrays) to temp Folder
    nTiles = tileInputLayers(modelParameters, modelPaths, rasterAttributes, tilingParameters)

    # now run the model for all tiles and save the results for each tile to the temp Folder
    performModelCalculation(nTiles, modelParameters, modelPaths, rasterAttributes, forestParams, MPOptions)

    # merge results for the tiles stored in Temp Folder and write Output files
    mergeAndWriteResults(modelPaths, modelParameters)

    _endTime = datetime.now().replace(microsecond=0)

    log.info("==================================")
    log.info(":::> Total time needed: " + str(_endTime - _startTime) + " <:::")
    log.info("==================================")

    if (modelPaths["useCustomDirs"] is True) and (modelPaths["deleteTempFolder"] is True):
        deleteTempFolder(modelPaths["tempDir"])


def startLogging(modelParameters, forestParams, modelPaths, MPOptions):
    """performs logging at the start of the simulation

    Parameters
    -----------
    modelParameters: dict
        model input parameters (from .ini - file)
    forestParams: dict
        input parameters regarding forest interaction (from .ini - file)
    modelPaths: dict
        paths to input files, workDir, resDir, outputFileFormat, etc. (from .ini - file)
    MPOptions: dict
        contains parameters for multiprocessing (from .ini - file)
    """
    # Start of Calculation (logging...)
    log.info("==================================")
    log.info("Starting calculation ...")
    log.info("==================================")
    log.info(f"{'Alpha Angle:' : <20}{modelParameters['alpha'] : <5}")
    log.info(f"{'Exponent:' : <20}{modelParameters['exp'] : <5}")
    log.info(f"{'Flux Threshold:' : <20}{modelParameters['flux_threshold'] : <5}")
    log.info(f"{'Max Z_delta:' : <20}{modelParameters['max_z'] : <5}")
    log.info("------------------------")
    # Also log the used input-files
    log.info(f"{'DEM:' : <5}{'%s'%modelPaths['demPath'] : <5}")
    log.info(f"{'REL:' : <5}{'%s'%modelPaths['releasePath'] : <5}")
    # log.info("DEM: {}".format(modelPaths["demPath"]))
    # log.info("REL: {}".format(modelPaths["releasePath"]))
    log.info("------------------------")
    if modelParameters["forestBool"]:
        log.info("Calculation using forestModule: {}".format(forestParams["forestModule"]))
        log.info(f"{'FOREST LAYER:' : <14}{'%s'%modelPaths['forestPath'] : <5}")
        log.info("-----")
        for param, value in forestParams.items():
            log.info(f"{'%s:'%param : <20}{value : <5}")
        log.info(f"{'forestInteraction : ' : <20}{'%s'%modelParameters['forestInteraction'] : <5}")
        log.info("------------------------")
    if modelParameters["varAlphaBool"]:
        log.info("Calculation using variable Alpha")
        log.info(f"{'ALPHA LAYER:' : <14}{'%s'%modelPaths['varAlphaPath'] : <5}")
        log.info("------------------------")
    if modelParameters["varUmaxBool"]:
        log.info("Calculation using variable uMax Limit")
        log.info(f"{'UMAX LAYER:' : <14}{'%s'%modelPaths['varUmaxPath'] : <5}")
        log.info("------------------------")
    if modelParameters["varExponentBool"]:
        log.info("Calculation using variable Alpha")
        log.info(f"{'ALPHA LAYER:' : <14}{'%s'%modelPaths['varExponentPath'] : <5}")
        log.info("------------------------")
    if modelParameters["infraBool"]:
        log.info("calculation with Infrastructure")
        log.info(f"{'INFRA LAYER:' : <14}{'%s'%modelPaths['infraPath'] : <5}")
        log.info("------------------------")
    if modelParameters["previewMode"]:
        log.info("!!! previewMode is ON !!!! - mind when interpreting results!!!")
        log.info("------------------------")
    if modelParameters["fluxDistOldVersionBool"]:
        log.info("Calculation using old (BUGGY!!) version of flux distribution!")
        log.info("------------------------")
    for param, value in MPOptions.items():
        log.info(f"{'%s:'%param : <20}{value : <5}")
        # log.info("{}:\t{}".format(param,value))
    log.info("------------------------")
    log.info(f"{'WorkDir:' : <12}{'%s'%modelPaths['workDir'] : <5}")
    log.info(f"{'ResultsDir:' : <12}{'%s'%modelPaths['resDir'] : <5}")
    # log.info("WorkDir: {}".format(modelPaths["workDir"]))
    # log.info("ResultsDir: {}".format(modelPaths["resDir"]))
    log.info("========================")


def checkInputLayerDimensions(modelParameters, modelPaths):
    """check if all input layers have the same size
    and can be read from the provided paths

    Parameters
    -----------
    modelParameters: dict
        model input parameters (from .ini - file)
    modelPaths: dict
        contains paths to input files
    """
    # Check if Layers have same size!!!
    try:
        log.info("checking input layer alignment ...")

        ext = os.path.splitext(modelPaths["demPath"])[1]

        _demHeader = IOf.readRasterHeader(modelPaths["demPath"])
        _relHeader = IOf.readRasterHeader(modelPaths["releasePathWork"])

        if _demHeader["ncols"] == _relHeader["ncols"] and _demHeader["nrows"] == _relHeader["nrows"]:
            log.info("DEM and Release Layer ok!")
        else:
            log.error("Error: Release Layer doesn't match DEM!")
            sys.exit(1)

        if modelParameters["infraBool"]:
            _infraHeader = IOf.readRasterHeader(modelPaths["infraPath"])
            if _demHeader["ncols"] == _infraHeader["ncols"] and _demHeader["nrows"] == _infraHeader["nrows"]:
                log.info("Infra Layer ok!")
            else:
                log.error("Error: Infra Layer doesn't match DEM!")
                sys.exit(1)

        if modelParameters["forestBool"]:
            _forestHeader = IOf.readRasterHeader(modelPaths["forestPath"])
            if _demHeader["ncols"] == _forestHeader["ncols"] and _demHeader["nrows"] == _forestHeader["nrows"]:
                log.info("Forest Layer ok!")
            else:
                log.error("Error: Forest Layer doesn't match DEM!")
                sys.exit(1)

        if modelParameters["varUmaxBool"]:
            _varUmaxHeader = IOf.readRasterHeader(modelPaths["varUmaxPath"])
            if _demHeader["ncols"] == _varUmaxHeader["ncols"] and _demHeader["nrows"] == _varUmaxHeader["nrows"]:
                log.info("uMax Limit Layer ok!")
            else:
                log.error("Error: uMax Limit Layer doesn't match DEM!")
                sys.exit(1)

        if modelParameters["varAlphaBool"]:
            _varAlphaHeader = IOf.readRasterHeader(modelPaths["varAlphaPath"])
            if _demHeader["ncols"] == _varAlphaHeader["ncols"] and _demHeader["nrows"] == _varAlphaHeader["nrows"]:
                log.info("variable Alpha Layer ok!")
            else:
                log.error("Error: variable Alpha Layer doesn't match DEM!")
                sys.exit(1)

        if modelParameters["varExponentBool"]:
            _varExponentHeader = IOf.readRasterHeader(modelPaths["varExponentPath"])
            if _demHeader["ncols"] == _varExponentHeader["ncols"] and _demHeader["nrows"] == _varExponentHeader["nrows"]:
                log.info("variable exponent Layer ok!")
            else:
                log.error("Error: variable exponent Layer doesn't match DEM!")
                sys.exit(1)

        log.info("========================")

    except Exception as ex:
        log.error("could not read all required Input Layers, please re-check files and paths provided in .ini files")
        log.error('Error occured: %s' % ex)
        # return
        sys.exit(1)


def checkInputParameterValues(modelParameters, modelPaths):
    """check if the input parameters alpha, uMaxLimit/ zDeltaMaxLimit, exponent
    are within a physically sensible range

    Parameters
    -----------
    modelParameters: dict
        model input parameters (from .ini - file)
    modelPaths: dict
        contains paths to input files
    """
    alpha = modelParameters['alpha']
    if (alpha < 0 or alpha > 90):
        log.error("Error: Alpha value is not within a physically sensible range ([0,90]).")
        sys.exit(1)

    zDelta = modelParameters['max_z']
    if (zDelta < 0 or zDelta > 8848):
        log.error("Error: zDeltaMaxLimit value is not within a physically sensible range ([0,8848]).")
        sys.exit(1)

    exp = modelParameters['exp']
    if exp < 0:
        log.error("Error: Exponent value is not within a physically sensible range (> 0).")
        sys.exit(1)

    _checkVarParams = True

    if modelParameters["varAlphaBool"]:
        data = IOf.readRaster(modelPaths["varAlphaPath"])
        rasterValues = data["rasterData"] 
        rasterValues[rasterValues < 0] = np.nan  # handle different noData values
        if np.any(rasterValues > 90, where=~np.isnan(rasterValues)):
            log.error("Error: Not all Alpha-raster values are within a physically sensible range ([0,90]),\
                 in respective startcells the general alpha angle is used.")
            _checkVarParams = False

    if modelParameters["varUmaxBool"]:
        data = IOf.readRaster(modelPaths["varUmaxPath"])
        rasterValues = data["rasterData"] 
        rasterValues[rasterValues < 0] = np.nan
        if modelParameters["varUmaxType"].lower() == 'umax':
            _maxVal = 1500  # ~sqrt(8848*2*9.81)
        else:
            _maxVal = 8848
        if np.any(rasterValues > _maxVal, where=~np.isnan(rasterValues)):
            log.error("Error: Not all zDeltaMaxLimit-raster values are within a physically sensible range \
                ([0, 8848 m] or [0, 1500 m/s]), in respective startcells the general zDeltaMax value is used.")
            _checkVarParams = False

    if _checkVarParams:
        log.info("All input parameters are within a physically sensible range.")
        log.info("========================")
    else:
        log.info("NOT ALL variable input parameter rasters are within physically sensible ranges.")
        log.info("!!PLEASE RE-CHECK Input Rasters and handle Results with Care!!")
        log.info("========================")


def tileInputLayers(modelParameters, modelPaths, rasterAttributes, tilingParameters):
    """computes the number of tiles (_tileCols, _tileRows) and tileOverlap (_U)
    based on input layer dimensions and tilingParameters,
    divides all used input layers into tiles
    and saves the tiles to the temp folder

    the function is a wrapper around the code in splitAndMerge.py,
    where the actual tiling is handled.

    Parameters
    -----------
    modelParameters: dict
        model input parameters (from .ini - file)
    modelPaths: dict
        contains paths to input files
    rasterAttributes: dict
        contains (header) information about the rasters (that are the same for all rasters)
    tilingParameters: dict
        parameters relevant for tiling (from .ini - file)

    Returns
    -----------
    nTiles: tuple
        nTiles[0]: maximum index of tiles along rows
        nTiles[1]: maximum index of tiles along columns
        actual number of tiles = (nTiles[0] + 1) * (nTiles[1] + 1)
    """
    _tileCOLS = int(tilingParameters["tileSize"] / rasterAttributes["cellsize"])
    _tileROWS = int(tilingParameters["tileSize"] / rasterAttributes["cellsize"])
    _U = int(tilingParameters["tileOverlap"] / rasterAttributes["cellsize"])

    log.info("Start Tiling...")
    log.info("---------------------")

    SPAM.tileRaster(modelPaths["demPath"], "dem", modelPaths["tempDir"], _tileCOLS, _tileROWS, _U)
    SPAM.tileRaster(modelPaths["releasePathWork"], "init", modelPaths["tempDir"], _tileCOLS, _tileROWS, _U, isInit=True)

    if modelParameters["infraBool"]:
        SPAM.tileRaster(modelPaths["infraPath"], "infra", modelPaths["tempDir"], _tileCOLS, _tileROWS, _U)
    if modelParameters["varUmaxBool"]:
        SPAM.tileRaster(modelPaths["varUmaxPath"], "varUmax", modelPaths["tempDir"], _tileCOLS, _tileROWS, _U)
    if modelParameters["varAlphaBool"]:
        SPAM.tileRaster(modelPaths["varAlphaPath"], "varAlpha", modelPaths["tempDir"], _tileCOLS, _tileROWS, _U)
    if modelParameters["varExponentBool"]:
        SPAM.tileRaster(modelPaths["varExponentPath"], "varExponent", modelPaths["tempDir"], _tileCOLS, _tileROWS, _U)
    if modelParameters["forestBool"]:
        SPAM.tileRaster(modelPaths["forestPath"], "forest", modelPaths["tempDir"], _tileCOLS, _tileROWS, _U)
    log.info("Finished Tiling All Input Rasters.")
    log.info("==================================")

    nTiles = pickle.load(open(modelPaths["tempDir"] / "nTiles", "rb"))

    return nTiles


def performModelCalculation(nTiles, modelParameters, modelPaths, rasterAttributes, forestParams, MPOptions):
    """wrapper around fc.run()
    handles passing of model paths, configurations to fc.run()
    also responsible for processing input-data tiles in sequence

    Parameters
    -----------
    nTiles: tuple
        number of tiles
    modelParameters: dict
        model input parameters (from .ini - file)
    modelPaths: dict
        contains paths to input files
    rasterAttributes: dict
        contains (header) information about the rasters (that are the same for all rasters)
    forestParams: dict
        input parameters regarding forest interaction (from .ini - file)
    MPOptions: dict
        contains parameters for multiprocessing (from .ini - file)
    """

    optList = []

    for i in range(nTiles[0] + 1):
        for j in range(nTiles[1] + 1):
            optList.append((i, j, modelParameters, modelPaths, rasterAttributes, forestParams, MPOptions))

    log.info(" >> Start Calculation << ")
    log.info("-------------------------")
    # Calculation, i.e. iterating over the list of Tiles which have to be processed with fc.run()
    for i, optTuple in enumerate(optList):
        log.info("processing tile %i of %i" % (i + 1, len(optList)))
        fc.run(optTuple)
        log.info("finished   tile %i of %i" % (i + 1, len(optList)))
        log.info("-------------------------")

    log.info(" >> Calculation Finished << ")
    log.info("==================================")


def mergeAndWriteResults(modelPaths, modelOptions):
    """function handles merging of results for all tiles inside the temp Folder
    and also writing result files to the resultDir

    Parameters
    -----------
    modelPaths: dict
        contains paths to input files
    modelOptions: dict
        contains model input parameters (from .ini - file)
    """
    _uid = modelPaths["uid"]
    _outputs = set(modelPaths['outputFileList'])

    log.info(" merging results ...")
    log.info("-------------------------")

    # Merge calculated tiles
    affectedCells = SPAM.mergeRaster(modelPaths["tempDir"], "res_affected")
    zDelta = SPAM.mergeRaster(modelPaths["tempDir"], "res_z_delta")
    flux = SPAM.mergeRaster(modelPaths["tempDir"], "res_flux")
    cellCounts = SPAM.mergeRaster(modelPaths["tempDir"], "res_count", method="sum")
    zDeltaSum = SPAM.mergeRaster(modelPaths["tempDir"], "res_z_delta_sum", method="sum")
    routFluxSum = SPAM.mergeRaster(modelPaths["tempDir"], "res_rout_flux_sum", method="sum")
    depFluxSum = SPAM.mergeRaster(modelPaths["tempDir"], "res_dep_flux_sum", method="sum")
    fpTaMax = SPAM.mergeRaster(modelPaths["tempDir"], "res_fp_max")
    fpTaMin = SPAM.mergeRaster(modelPaths["tempDir"], "res_fp_min", method="min")
    slTa = SPAM.mergeRaster(modelPaths["tempDir"], "res_sl")
    travelLengthMax = SPAM.mergeRaster(modelPaths["tempDir"], "res_travel_length_max")
    travelLengthMin = SPAM.mergeRaster(modelPaths["tempDir"], "res_travel_length_min", method="min")

    if modelOptions["infraBool"]:
        backcalc = SPAM.mergeRaster(modelPaths["tempDir"], "res_backcalc")

    if modelOptions["forestInteraction"]:
        forestInteraction = SPAM.mergeRaster(modelPaths["tempDir"], "res_forestInt", method='min')

    # Write Output Files to Disk
    log.info("-------------------------")
    log.info(" writing output files ...")
    log.info("-------------------------")
    _oF = modelPaths["outputFileFormat"]
    _ts = modelPaths["timeString"]

    demHeader = IOf.readRasterHeader(modelPaths["demPath"])
    outputHeader = demHeader.copy()
    outputHeader["nodata_value"]=-9999

    if _oF == ".asc":
        outputHeader["driver"] = "AAIGrid"
    elif _oF == ".tif":
        outputHeader["driver"] = "GTiff"

    if 'flux' in _outputs:
        output = IOf.writeResultToRaster(outputHeader, flux,
                                         modelPaths["resDir"] / "com4_{}_{}_flux".format(_uid, _ts), flip=True)
    if 'zDelta' in _outputs:
        defineNotAffectedCells(zDelta, affectedCells)
        output = IOf.writeResultToRaster(outputHeader, zDelta,
                                         modelPaths["resDir"] / "com4_{}_{}_zdelta".format(_uid, _ts), flip=True)
    if 'cellCounts' in _outputs:
        defineNotAffectedCells(cellCounts, affectedCells)
        output = IOf.writeResultToRaster(outputHeader, cellCounts,
                                         modelPaths["resDir"] / "com4_{}_{}_cellCounts".format(_uid, _ts), flip=True)
    if 'zDeltaSum' in _outputs:
        defineNotAffectedCells(zDeltaSum, affectedCells)
        output = IOf.writeResultToRaster(outputHeader, zDeltaSum,
                                         modelPaths["resDir"] / "com4_{}_{}_zDeltaSum".format(_uid, _ts), flip=True)
    if 'routFluxSum' in _outputs:
        defineNotAffectedCells(routFluxSum, affectedCells)
        output = IOf.writeResultToRaster(outputHeader, routFluxSum,
                                         modelPaths["resDir"] / "com4_{}_{}_routFluxSum".format(_uid, _ts), flip=True)
    if 'depFluxSum' in _outputs:
        defineNotAffectedCells(depFluxSum, affectedCells)
        output = IOf.writeResultToRaster(outputHeader, depFluxSum,
                                         modelPaths["resDir"] / "com4_{}_{}_depFluxSum".format(_uid, _ts), flip=True)
    if "fpTravelAngle" in _outputs or "fpTravelAngleMax" in _outputs:
        output = IOf.writeResultToRaster(
            outputHeader,
            fpTaMax,
            modelPaths["resDir"] / "com4_{}_{}_fpTravelAngleMax".format(_uid, _ts),
            flip=True,
        )
    if "fpTravelAngleMin" in _outputs:
        output = IOf.writeResultToRaster(
            outputHeader,
            fpTaMin,
            modelPaths["resDir"] / "com4_{}_{}_fpTravelAngleMin".format(_uid, _ts),
            flip=True,
        )
    if 'slTravelAngle' in _outputs:
        output = IOf.writeResultToRaster(outputHeader, slTa,
                                         modelPaths["resDir"] / "com4_{}_{}_slTravelAngle".format(_uid, _ts), flip=True)
    if "travelLength" in _outputs or "travelLengthMax" in _outputs:
        output = IOf.writeResultToRaster(
            outputHeader,
            travelLengthMax,
            modelPaths["resDir"] / "com4_{}_{}_travelLengthMax".format(_uid, _ts),
            flip=True,
        )
    if "travelLengthMin" in _outputs:
        output = IOf.writeResultToRaster(
            outputHeader,
            travelLengthMin,
            modelPaths["resDir"] / "com4_{}_{}_travelLengthMin".format(_uid, _ts),
            flip=True,
        )

    # NOTE:
    # if not modelOptions["infraBool"]:  # if no infra
    # io.output_raster(modelPaths["demPath"], modelPaths["resDir"] / ("cell_counts%s" %(output_format)),cell_counts)
    # io.output_raster(modelPaths["demPath"], modelPaths["resDir"] / ("z_delta_sum%s" %(output_format)),z_delta_sum)
    if modelOptions["infraBool"]:  # if infra
        output = IOf.writeResultToRaster(outputHeader, backcalc,
                                         modelPaths["resDir"] / "com4_{}_{}_backcalculation".format(_uid, _ts), flip=True)
    if modelOptions["forestInteraction"]:
        output = IOf.writeResultToRaster(outputHeader, forestInteraction,
                                         modelPaths["resDir"] / "com4_{}_{}_forestInteraction".format(_uid, _ts), flip=True)
    del output


def checkConvertReleaseShp2Tif(modelPaths):
    """function checks if release area is a .shp file and tries to convert to tif in that case

    Parameters
    -----------
    modelPaths: dict
        contains modelPaths

    Returns
    -----------
    modelPaths: dict
        contains paths including ["releasePathWork"]
    """
    # the release is a shp polygon, we need to convert it to a raster
    # releaseLine = shpConv.readLine(releasePath, 'releasePolygon', demDict)
    if modelPaths["releasePath"].suffix == ".shp":

        dem = IOf.readRaster(modelPaths["demPath"])
        demHeader = dem["header"]
        dem['originalHeader'] = demHeader

        releaseLine = shpConv.SHP2Array(modelPaths["releasePath"], "releasePolygon")
        thresholdPointInPoly = 0.01
        releaseLine = gT.prepareArea(releaseLine, dem, thresholdPointInPoly, combine=True, checkOverlap=False)
        # give the same header as the dem
        releaseArea = np.flipud(releaseLine["rasterData"])
        if demHeader["driver"] == "AAIGrid":
            modelPaths["releasePathWork"] = modelPaths["workDir"] / "release.asc"
        if demHeader["driver"] == "GTiff":
            modelPaths["releasePathWork"] = modelPaths["workDir"] / "release.tif"
        # we use the nodata_value of the DEM for the output release raster
        # NOTE: nodata_value does not matter anyways - since gT.prepareArea() returns a
        # raster with values of '0' indicating 'no release area' - so there actually should
        # not be any 'no_data' values in the created release raster file
        IOf.writeResultToRaster(demHeader, releaseArea, modelPaths["workDir"] / "release")
        del releaseLine
    else:
        modelPaths["releasePathWork"] = modelPaths["releasePath"]

    return modelPaths


def deleteTempFolder(tempFolderPath):
    """delete tempFolder containing the pickled np.arrays of the input data and output data tiles.
    should be called after all merged model results are written to disk.
    performs a few checks to make sure the folder is indeed a com4FlowPy tempFolder, i.e.
        - does not contain subfolders
        - no other file-extensions than '.npy' and ''

    Parameters
    -----------
    tempFolderPath: str
        path to temp folder
    """

    log.info("+++++++++++++++++++++++")
    log.info("deleteTempFolder = True in (local_)com4FlowPyCfg.ini")

    log.info("... checking if folder is a com4FlowPy temp Folder")
    # check if path exists and is directory
    isDir = os.path.isdir(tempFolderPath)
    validTemp = True

    for f in os.scandir(tempFolderPath):
        # check if there's a nested folder inside tempDir
        if f.is_dir():
            validTemp = False
            break
        # check if all files are either .npy or start with "ext, "nTi"
        elif f.is_file():
            if not f.path.endswith(".npy"):
                if not f.name[:3] in ["ext", "nTi"]:
                    validTemp = False
                    break

    if isDir and validTemp:
        log.info("Tempfolder checked: isDir:{} isTemp:{}".format(isDir, validTemp))
        try:
            shutil.rmtree(tempFolderPath)
            log.info("Deleted temp folder {}".format(tempFolderPath))
        except OSError as e:
            log.info("deletion of temp folder {} failed".format(tempFolderPath))
            print("Error: %s : %s" % (tempFolderPath, e.strerror))
    else:
        log.info("deletion of temp folder {} failed".format(tempFolderPath))
        log.info(" isDir:{} isTemp:{}}".format(isDir, validTemp))

    log.info("+++++++++++++++++++++++")


def defineNotAffectedCells(raster, affectedCells, noDataValue=-9999):
    """
    define not affected cells as -9999

    Parameters
    -----------
    raster: np.array
        raster whose not affected cells are specified
    affectedCells: np.array
        mask for affected cells
    noDataValue: float
        value for not affected cells (default: -9999)

    Returns
    -----------
    raster: np. array
        raster with not affected cells have the value noDataValue
    """
    raster[affectedCells == 0] = noDataValue
    return raster
