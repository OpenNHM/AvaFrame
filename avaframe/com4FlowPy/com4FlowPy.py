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
    modelParameters["flux_threshold"] = cfgSetup.getfloat(
        "flux_threshold"
    )  # float(cfgSetup["flux_threshold"])
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
    modelParameters["calcGeneration"] = cfgSetup.getboolean("calcGeneration")
    modelParameters["calcThalweg"] = cfgSetup.getboolean("calcThalweg")
    if modelParameters["calcThalweg"]:
        modelParameters["thalwegReleaseArea"] = cfgSetup.getboolean("thalwegReleaseArea")
        modelParameters["thalwegSaveRam"] = cfgSetup.getboolean("thalwegSaveRam")
        modelParameters["videoRelId"] = cfgSetup.get("videoRelId", fallback="")
        modelParameters["videoDataVariable"] = cfgSetup.get("videoDataVariable")
    else:
        modelParameters["thalwegReleaseArea"] = False
        modelParameters["thalwegSaveRam"] = False
    modelParameters["thalwegVariables"] = cfgSetup.get("thalwegVariables")
    modelParameters["thalwegCenterOf"] = cfgSetup.get("thalwegCenterOf")

    if modelParameters["thalwegSaveRam"]:
        if modelParameters["videoRelId"] != "":
            message = f"If thalwegSaveRam is True, no video data is stored, please check the configuration settings!"
            log.error(message)
            raise ValueError(message)

    # modelParameters["infra"]  = cfgSetup["infra"]
    # modelParameters["forest"] = cfgSetup["forest"]

    # compute engine: "python" (default, Cell-based) or "numba" (JIT kernel)
    modelParameters["engine"] = cfgSetup.get("engine", "python").strip().lower()

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
    modelPaths["outputFileList"] = cfgPath["outputFiles"].split("|")
    modelPaths["outputNoDataValue"] = cfgPath["outputNoDataValue"]
    modelPaths["useCompression"] = cfgPath["useCompression"]

    modelPaths["outputFileFormat"] = cfgPath["outputFileFormat"]
    if modelPaths["outputFileFormat"] in [".asc", ".ASC"]:
        modelPaths["outputFileFormat"] = ".asc"
    else:
        modelPaths["outputFileFormat"] = ".tif"
    modelPaths["thalwegDir"] = cfgPath["thalwegDir"]

    # check if 'customDirs' are used - alternative is 'default' AvaFrame Folder Structure
    modelPaths["useCustomDirs"] = True if cfgPath["customDirs"] == "True" else False
    # check if the temp folder, where intermediate results are stored, should be deleted after writing output files
    modelPaths["deleteTempFolder"] = True if cfgPath["deleteTemp"] == "True" else False

    # Multiprocessing Options
    MPOptions = {}
    MPOptions["nCPU"] = cfgSetup.getint("cpuCount")  # int(cfgSetup["cpuCount"]) #number of CPUs to use
    MPOptions["procPerCPU"] = cfgSetup.getint(
        "procPerCPUCore"
    )  # int(cfgSetup["procPerCPUCore"]) #processes per core
    MPOptions["chunkSize"] = cfgSetup.getint(
        "chunkSize"
    )  # int(cfgSetup["chunkSize"]) # default task size for MP
    MPOptions["maxChunks"] = cfgSetup.getint(
        "maxChunks"
    )  # int(cfgSetup["maxChunks"]) # max number of tasks for MP

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
        forestParams["maxAddedFriction"] = cfgSetup.getfloat(
            "maxAddedFrictionFor"
        )  # float(cfgSetup["maxAddedFr.."])
        forestParams["minAddedFriction"] = cfgSetup.getfloat(
            "minAddedFrictionFor"
        )  # float(cfgSetup["minAddedFr.."])
        forestParams["velThForFriction"] = cfgSetup.getfloat(
            "velThForFriction"
        )  # float(cfgSetup["velThForFriction"])
        forestParams["maxDetrainment"] = cfgSetup.getfloat(
            "maxDetrainmentFor"
        )  # float(cfgSetup["maxDetrainmentFor"])
        forestParams["minDetrainment"] = cfgSetup.getfloat(
            "minDetrainmentFor"
        )  # float(cfgSetup["minDetrainmentFor"])
        forestParams["velThForDetrain"] = cfgSetup.getfloat(
            "velThForDetrain"
        )  # float(cfgSetup["velThForDetrain"])
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

    # conditions if relId is used
    _outputPolygon = "relIdPolygon" in modelPaths["outputFileList"]
    _outputCount = "relIdCount" in modelPaths["outputFileList"]
    if _outputPolygon or _outputCount or modelParameters["thalwegReleaseArea"]:
        modelPaths["relIdPath"] = cfgPath["relIdPath"]
        modelParameters["outputRelIdBool"] = True
    else:
        modelPaths["relIdPath"] = ""
        modelParameters["outputRelIdBool"] = False

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
    validParamRanges = getValidParameterRanges(cfgSetup)
    checkInputParameterValues(modelParameters, modelPaths, validParamRanges)

    # get information on cellsize and nodata value from demHeader
    rasterAttributes = {}

    demHeader = IOf.readRasterHeader(modelPaths["demPath"])
    rasterAttributes["nodata"] = demHeader["nodata_value"]
    rasterAttributes["cellsize"] = demHeader["cellsize"]
    rasterAttributes["xllcenter"] = demHeader["xllcenter"]
    rasterAttributes["yllcenter"] = demHeader["yllcenter"]
    rasterAttributes["nrows"] = demHeader["nrows"]

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
            message = "Error: Release Layer doesn't match DEM!"
            log.error(message)
            raise ValueError(message)

        if modelParameters["outputRelIdBool"]:
            _relIdHeader = IOf.readRasterHeader(modelPaths["relIdPath"])
            if _demHeader["ncols"] == _relIdHeader["ncols"] and _demHeader["nrows"] == _relIdHeader["nrows"]:
                log.info("Release ID Layer ok!")
            else:
                message = "Error: Release ID Layer doesn't match DEM!"
                log.error(message)
                raise ValueError(message)

        if modelParameters["infraBool"]:
            _infraHeader = IOf.readRasterHeader(modelPaths["infraPath"])
            if _demHeader["ncols"] == _infraHeader["ncols"] and _demHeader["nrows"] == _infraHeader["nrows"]:
                log.info("Infra Layer ok!")
            else:
                message = "Error: Infra Layer doesn't match DEM!"
                log.error(message)
                raise ValueError(message)

        if modelParameters["forestBool"]:
            _forestHeader = IOf.readRasterHeader(modelPaths["forestPath"])
            if (
                _demHeader["ncols"] == _forestHeader["ncols"]
                and _demHeader["nrows"] == _forestHeader["nrows"]
            ):
                log.info("Forest Layer ok!")
            else:
                message = "Error: Forest Layer doesn't match DEM!"
                log.error(message)
                raise ValueError(message)

        if modelParameters["varUmaxBool"]:
            _varUmaxHeader = IOf.readRasterHeader(modelPaths["varUmaxPath"])
            if (
                _demHeader["ncols"] == _varUmaxHeader["ncols"]
                and _demHeader["nrows"] == _varUmaxHeader["nrows"]
            ):
                log.info("uMax Limit Layer ok!")
            else:
                message = "Error: uMax Limit Layer doesn't match DEM!"
                log.error(message)
                raise ValueError(message)

        if modelParameters["varAlphaBool"]:
            _varAlphaHeader = IOf.readRasterHeader(modelPaths["varAlphaPath"])
            if (
                _demHeader["ncols"] == _varAlphaHeader["ncols"]
                and _demHeader["nrows"] == _varAlphaHeader["nrows"]
            ):
                log.info("variable Alpha Layer ok!")
            else:
                message = "Error: variable Alpha Layer doesn't match DEM!"
                log.error(message)
                raise ValueError(message)

        if modelParameters["varExponentBool"]:
            _varExponentHeader = IOf.readRasterHeader(modelPaths["varExponentPath"])
            if (
                _demHeader["ncols"] == _varExponentHeader["ncols"]
                and _demHeader["nrows"] == _varExponentHeader["nrows"]
            ):
                log.info("variable exponent Layer ok!")
            else:
                message = "Error: variable exponent Layer doesn't match DEM!"
                log.error(message)
                raise ValueError(message)

        log.info("========================")

    except Exception as ex:
        log.error(
            "could not read all required Input Layers, please re-check files and paths provided in .ini files"
        )
        log.error("Error occured: %s" % ex)
        # return
        raise ValueError(
            "could not read all required Input Layers, please re-check files and paths provided in .ini files"
        )

def checkInputParameterValues(modelParameters, modelPaths, validParamRanges):
    """check if the input parameters are valid
    are valid and within physically sensible limits
    alpha, uMaxLimit/ zDeltaMaxLimit, exponent
    are within a physically sensible range

    Parameters
    -----------
    modelParameters: dict
        model input parameters (from .ini - file)
    modelPaths: dict
        contains paths to input files
    """

    log.info("checking input data validity")
    # check if engine parameter provided in the .ini matches available options
    engine = modelParameters["engine"]
    validEngines = {"python","numba"}

    if engine not in validEngines:
        raise ValueError(
            f"Invalid engine '{engine}'. "
            f"Valid engine options are {sorted(validEngines)}"
        )
    
    # checking value ranges of (global) input parameters
    # alpha, max_z, exp, flux_threshold
    checkGlobalParameters(modelParameters, modelPaths, validParamRanges)
    # check spatially varying input parameters for validity and completeness
    checkVariableInputParameters(modelParameters, modelPaths, validParamRanges)


def validateInputArray(praMask, data, maxValue, minValue=0):
    """validates the input array, by checking the required valued based on the provided pra
        
    Parameters
    -----------
    praMask: np.ndarray
        pra input array 
    data: np.ndarray
        data array which will be tested
    maxValue: int | float
        max value, for range validation of data entires
    minValue: int | float, optional
        min value, for range validation of data entires, by default 0

    Returns
    --------
    bool
        if input is valid
    """
    dataMask = data[praMask]
    return (dataMask.min() >= minValue) and (dataMask.max() <= maxValue)


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
    if modelParameters["thalwegReleaseArea"]:
        _relIdRasterDict = IOf.readRaster(modelPaths["relIdPath"])
        _relIdRaster = _relIdRasterDict["rasterData"]
        exList, eyList = SPAM.getTileEnds(modelPaths["tempDir"], _tileCOLS, _tileROWS, _U, _relIdRaster)

        SPAM.tileRasterWithIndices(modelPaths["demPath"], "dem", modelPaths["tempDir"], exList, eyList, _U)
        SPAM.tileRasterWithIndices(
            modelPaths["releasePathWork"],
            "init",
            modelPaths["tempDir"],
            exList,
            eyList,
            _U,
            isInit=True,
        )

        if modelParameters["infraBool"]:
            SPAM.tileRasterWithIndices(
                modelPaths["infraPath"],
                "infra",
                modelPaths["tempDir"],
                exList,
                eyList,
                _U,
            )
        if modelParameters["varUmaxBool"]:
            SPAM.tileRasterWithIndices(
                modelPaths["varUmaxPath"],
                "varUmax",
                modelPaths["tempDir"],
                exList,
                eyList,
                _U,
            )
        if modelParameters["varAlphaBool"]:
            SPAM.tileRasterWithIndices(
                modelPaths["varAlphaPath"],
                "varAlpha",
                modelPaths["tempDir"],
                exList,
                eyList,
                _U,
            )
        if modelParameters["varExponentBool"]:
            SPAM.tileRasterWithIndices(
                modelPaths["varExponentPath"],
                "varExponent",
                modelPaths["tempDir"],
                exList,
                eyList,
                _U,
            )
        if modelParameters["forestBool"]:
            SPAM.tileRasterWithIndices(
                modelPaths["forestPath"],
                "forest",
                modelPaths["tempDir"],
                exList,
                eyList,
                _U,
            )
        if modelParameters["outputRelIdBool"]:
            SPAM.tileRasterWithIndices(
                modelPaths["relIdPath"],
                "relId",
                modelPaths["tempDir"],
                exList,
                eyList,
                _U,
            )

    else:
        SPAM.tileRaster(
            modelPaths["demPath"],
            "dem",
            modelPaths["tempDir"],
            _tileCOLS,
            _tileROWS,
            _U,
        )
        SPAM.tileRaster(
            modelPaths["releasePathWork"],
            "init",
            modelPaths["tempDir"],
            _tileCOLS,
            _tileROWS,
            _U,
            isInit=True,
        )

        if modelParameters["infraBool"]:
            SPAM.tileRaster(
                modelPaths["infraPath"],
                "infra",
                modelPaths["tempDir"],
                _tileCOLS,
                _tileROWS,
                _U,
            )
        if modelParameters["varUmaxBool"]:
            SPAM.tileRaster(
                modelPaths["varUmaxPath"],
                "varUmax",
                modelPaths["tempDir"],
                _tileCOLS,
                _tileROWS,
                _U,
            )
        if modelParameters["varAlphaBool"]:
            SPAM.tileRaster(
                modelPaths["varAlphaPath"],
                "varAlpha",
                modelPaths["tempDir"],
                _tileCOLS,
                _tileROWS,
                _U,
            )
        if modelParameters["varExponentBool"]:
            SPAM.tileRaster(
                modelPaths["varExponentPath"],
                "varExponent",
                modelPaths["tempDir"],
                _tileCOLS,
                _tileROWS,
                _U,
            )
        if modelParameters["forestBool"]:
            SPAM.tileRaster(
                modelPaths["forestPath"],
                "forest",
                modelPaths["tempDir"],
                _tileCOLS,
                _tileROWS,
                _U,
            )
        if modelParameters["outputRelIdBool"]:
            SPAM.tileRaster(
                modelPaths["relIdPath"],
                "relId",
                modelPaths["tempDir"],
                _tileCOLS,
                _tileROWS,
                _U,
            )

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
            optList.append(
                (
                    i,
                    j,
                    modelParameters,
                    modelPaths,
                    rasterAttributes,
                    forestParams,
                    MPOptions,
                )
            )

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
    _outputs = set(modelPaths["outputFileList"])
    _outputNoDataValue = modelPaths["outputNoDataValue"]

    demHeader = IOf.readRasterHeader(modelPaths["demPath"])
    outputHeader = demHeader.copy()
    outputHeader["nodata_value"] = _outputNoDataValue
    _oF = modelPaths["outputFileFormat"]
    _ts = modelPaths["timeString"]

    useCompression = modelPaths["useCompression"]

    if _oF == ".asc":
        outputHeader["driver"] = "AAIGrid"
    elif _oF == ".tif":
        outputHeader["driver"] = "GTiff"

    log.info(" merging and writing results ...")
    log.info("-------------------------")

    # Merge calculated tiles
    # compute cellCounts and don't delete because it is used for defining not affected cells
    # other rasters (and polygons) are deleted after writing (to reduce computation time and used RAM)
    cellCounts = SPAM.mergeRaster(modelPaths["tempDir"], "res_count", method="sum")
    if "cellCounts" in _outputs:
        cellCounts = defineNotAffectedCells(cellCounts, cellCounts, noDataValue=_outputNoDataValue)
        output = IOf.writeResultToRaster(
            outputHeader,
            cellCounts,
            modelPaths["resDir"] / "com4_{}_{}_cellCounts".format(_uid, _ts),
            flip=True,
            useCompression=useCompression,
        )
        del output
        log.info("com4_{}_{}_cellCounts  is written".format(_uid, _ts))

    if "zDelta" in _outputs:
        zDelta = SPAM.mergeRaster(modelPaths["tempDir"], "res_z_delta")
        zDelta = defineNotAffectedCells(zDelta, cellCounts, noDataValue=_outputNoDataValue)
        output = IOf.writeResultToRaster(
            outputHeader,
            zDelta,
            modelPaths["resDir"] / "com4_{}_{}_zdelta".format(_uid, _ts),
            flip=True,
            useCompression=useCompression,
        )
        del zDelta
        del output
        log.info("com4_{}_{}_zdelta  is written".format(_uid, _ts))

    if "flux" in _outputs:
        flux = SPAM.mergeRaster(modelPaths["tempDir"], "res_flux")
        flux = defineNotAffectedCells(flux, cellCounts, noDataValue=_outputNoDataValue)
        output = IOf.writeResultToRaster(
            outputHeader,
            flux,
            modelPaths["resDir"] / "com4_{}_{}_flux".format(_uid, _ts),
            flip=True,
            useCompression=useCompression,
        )
        del flux
        del output
        log.info("com4_{}_{}_flux  is written".format(_uid, _ts))

    if "zDeltaSum" in _outputs:
        zDeltaSum = SPAM.mergeRaster(modelPaths["tempDir"], "res_z_delta_sum", method="sum")
        zDeltaSum = defineNotAffectedCells(zDeltaSum, cellCounts, noDataValue=_outputNoDataValue)
        output = IOf.writeResultToRaster(
            outputHeader,
            zDeltaSum,
            modelPaths["resDir"] / "com4_{}_{}_zDeltaSum".format(_uid, _ts),
            flip=True,
            useCompression=useCompression,
        )
        del zDeltaSum
        del output
        log.info("com4_{}_{}_zDeltaSum  is written".format(_uid, _ts))

    if "routFluxSum" in _outputs:
        routFluxSum = SPAM.mergeRaster(modelPaths["tempDir"], "res_rout_flux_sum", method="sum")
        routFluxSum = defineNotAffectedCells(routFluxSum, cellCounts, noDataValue=_outputNoDataValue)
        output = IOf.writeResultToRaster(
            outputHeader,
            routFluxSum,
            modelPaths["resDir"] / "com4_{}_{}_routFluxSum".format(_uid, _ts),
            flip=True,
            useCompression=useCompression,
        )
        del routFluxSum
        del output
        log.info("com4_{}_{}_routFluxSum  is written".format(_uid, _ts))

    if "depFluxSum" in _outputs:
        depFluxSum = SPAM.mergeRaster(modelPaths["tempDir"], "res_dep_flux_sum", method="sum")
        depFluxSum = defineNotAffectedCells(depFluxSum, cellCounts, noDataValue=_outputNoDataValue)
        output = IOf.writeResultToRaster(
            outputHeader,
            depFluxSum,
            modelPaths["resDir"] / "com4_{}_{}_depFluxSum".format(_uid, _ts),
            flip=True,
            useCompression=useCompression,
        )
        del depFluxSum
        del output
        log.info("com4_{}_{}_depFluxSum  is written".format(_uid, _ts))

    if "fpTravelAngle" in _outputs or "fpTravelAngleMax" in _outputs:
        fpTaMax = SPAM.mergeRaster(modelPaths["tempDir"], "res_fp_max")
        fpTaMax = defineNotAffectedCells(fpTaMax, cellCounts, noDataValue=_outputNoDataValue)
        output = IOf.writeResultToRaster(
            outputHeader,
            fpTaMax,
            modelPaths["resDir"] / "com4_{}_{}_fpTravelAngleMax".format(_uid, _ts),
            flip=True,
            useCompression=useCompression,
        )
        del fpTaMax
        del output
        log.info("com4_{}_{}_fpTravelAngleMax is written".format(_uid, _ts))

    if "fpTravelAngleMin" in _outputs:
        fpTaMin = SPAM.mergeRaster(modelPaths["tempDir"], "res_fp_min", method="min")
        fpTaMin = defineNotAffectedCells(fpTaMin, cellCounts, noDataValue=_outputNoDataValue)
        output = IOf.writeResultToRaster(
            outputHeader,
            fpTaMin,
            modelPaths["resDir"] / "com4_{}_{}_fpTravelAngleMin".format(_uid, _ts),
            flip=True,
            useCompression=useCompression,
        )
        del fpTaMin
        del output
        log.info("com4_{}_{}_fpTravelAngleMin is written".format(_uid, _ts))

    if "slTravelAngle" in _outputs:
        slTa = SPAM.mergeRaster(modelPaths["tempDir"], "res_sl")
        slTa = defineNotAffectedCells(slTa, cellCounts, noDataValue=_outputNoDataValue)
        output = IOf.writeResultToRaster(
            outputHeader,
            slTa,
            modelPaths["resDir"] / "com4_{}_{}_slTravelAngle".format(_uid, _ts),
            flip=True,
            useCompression=useCompression,
        )
        del slTa
        del output
        log.info("com4_{}_{}_slTravelAngle is written".format(_uid, _ts))

    if "travelLength" in _outputs or "travelLengthMax" in _outputs:
        travelLengthMax = SPAM.mergeRaster(modelPaths["tempDir"], "res_travel_length_max")
        travelLengthMax = defineNotAffectedCells(travelLengthMax, cellCounts, noDataValue=_outputNoDataValue)
        output = IOf.writeResultToRaster(
            outputHeader,
            travelLengthMax,
            modelPaths["resDir"] / "com4_{}_{}_travelLengthMax".format(_uid, _ts),
            flip=True,
            useCompression=useCompression,
        )
        del travelLengthMax
        del output
        log.info("com4_{}_{}_travelLengthMax is written".format(_uid, _ts))

    if "travelLengthMin" in _outputs:
        travelLengthMin = SPAM.mergeRaster(modelPaths["tempDir"], "res_travel_length_min", method="min")
        travelLengthMin = defineNotAffectedCells(travelLengthMin, cellCounts, noDataValue=_outputNoDataValue)
        output = IOf.writeResultToRaster(
            outputHeader,
            travelLengthMin,
            modelPaths["resDir"] / "com4_{}_{}_travelLengthMin".format(_uid, _ts),
            flip=True,
            useCompression=useCompression,
        )
        del travelLengthMin
        del output
        log.info("com4_{}_{}_travelLengthMin is written".format(_uid, _ts))

    if modelOptions["infraBool"]:
        backcalc = SPAM.mergeRaster(modelPaths["tempDir"], "res_backcalc")
        backcalc = defineNotAffectedCells(backcalc, cellCounts, noDataValue=_outputNoDataValue)
        output = IOf.writeResultToRaster(
            outputHeader,
            backcalc,
            modelPaths["resDir"] / "com4_{}_{}_backcalculation".format(_uid, _ts),
            flip=True,
            useCompression=useCompression,
        )
        del backcalc
        del output
        log.info("com4_{}_{}_backcalculation is written".format(_uid, _ts))

    if modelOptions["forestInteraction"]:
        forestInteraction = SPAM.mergeRaster(modelPaths["tempDir"], "res_forestInt", method="min")
        forestInteraction = defineNotAffectedCells(
            forestInteraction, cellCounts, noDataValue=_outputNoDataValue
        )
        output = IOf.writeResultToRaster(
            outputHeader,
            forestInteraction,
            modelPaths["resDir"] / "com4_{}_{}_forestInteraction".format(_uid, _ts),
            flip=True,
            useCompression=useCompression,
        )
        del forestInteraction
        del output
        log.info("com4_{}_{}_forestInteraction is written".format(_uid, _ts))

    if "relIdPolygon" in _outputs:
        pathPolygons = SPAM.mergeDictToPolygon(modelPaths["tempDir"], "res_startCellIdDict", outputHeader)
        pathPolygons.to_file(
            modelPaths["resDir"] / "com4_{}_{}_pathPolygons.geojson".format(_uid, _ts),
            driver="GeoJSON",
        )
        del pathPolygons
        log.info("com4_{}_{}_relIdPolygon is written".format(_uid, _ts))

    if "relIdCount" in _outputs:
        countRelId = SPAM.mergeDictToRaster(modelPaths["tempDir"], "res_startCellIdDict")
        countRelId = defineNotAffectedCells(countRelId, cellCounts, noDataValue=_outputNoDataValue)
        output = IOf.writeResultToRaster(
            outputHeader,
            countRelId,
            modelPaths["resDir"] / "com4_{}_{}_relIdCount".format(_uid, _ts),
            flip=True,
            useCompression=useCompression,
        )
        del countRelId
        del output
        log.info("com4_{}_{}_relIdCount is written".format(_uid, _ts))

    # NOTE:
    # if not modelOptions["infraBool"]:  # if no infra
    # io.output_raster(modelPaths["demPath"], modelPaths["resDir"] / ("cell_counts%s" %(output_format)),cell_counts)
    # io.output_raster(modelPaths["demPath"], modelPaths["resDir"] / ("z_delta_sum%s" %(output_format)),z_delta_sum)


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
        dem["originalHeader"] = demHeader

        releaseLine = shpConv.SHP2Array(modelPaths["releasePath"], "releasePolygon")
        thresholdPointInPoly = 0.01
        releaseLine = gT.prepareArea(
            releaseLine, dem, thresholdPointInPoly, combine=True, checkOverlap=False
        )
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
        IOf.writeResultToRaster(
            demHeader,
            releaseArea,
            modelPaths["workDir"] / "release",
            useCompression=modelPaths["useCompression"],
        )
        del releaseLine
    else:
        modelPaths["releasePathWork"] = modelPaths["releasePath"]

    return modelPaths


def deleteTempFolder(tempFolderPath):
    """delete tempFolder containing the pickled np.arrays of the input and output data tiles or other .pickle files.
    should be called after all merged model results are written to disk.
    performs a few checks to make sure the folder is indeed a com4FlowPy tempFolder, i.e.
        - does not contain subfolders
        - no other file-extensions than '.npy', '.pickle' and ''

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
        # check if all files are either .npy, .pickle or start with "ext, "nTi"
        elif f.is_file():
            if not (f.path.endswith(".npy") or f.path.endswith(".pickle")):
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
        log.info(" isDir:{} isTemp:{}".format(isDir, validTemp))

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
    raster[affectedCells <= 0] = noDataValue
    return raster

def getValidParameterRanges(cfgSetup):
    """
    read valid parameter ranges from cfgSetup
    and provide default fallbacks if they are missing.

    Parameters
    -----------
    cfgSetup: configparser.SectionProxy Object
        "GENERAL" model configs (from .ini file)

    Returns
    -----------
    validParamRanges: dict
        dictionary with the parameter limits used by the input parameter
        checker functions
    """
    validParamRanges = {}

    # get valid data ranges from cfg - if they are not provided use default fallbacks
    validParamRanges['minAlphaValid']      = cfgSetup.getfloat("minAlphaValid", 0)
    validParamRanges['maxAlphaValid']      = cfgSetup.getfloat("maxAlphaValid", 90)
    validParamRanges['maxUMaxValid']       = cfgSetup.getfloat("maxUMaxValid", 630)
    validParamRanges['maxZDeltaValid']     = cfgSetup.getfloat("maxZDeltaValid", 20000)
    validParamRanges['maxExpValid']        = cfgSetup.getfloat("maxExpValid", 600)
    validParamRanges['minFluxThreshValid'] = cfgSetup.getfloat("minFluxThreshValid", 1e-6)
    validParamRanges['maxFluxThreshValid'] = cfgSetup.getfloat("maxFluxThreshValid", 1)

    log.info("retreiving valid parameter ranges:")
    log.info(24*"-")
    for key, val in validParamRanges.items():
        log.info(f"{key}: {val}")
    log.info("========================")

    return validParamRanges

def checkGlobalParameters(modelParameters, modelPaths, validParamRanges):
    """
    checks global input parameters for valid input ranges
    
    Parameters
    -----------
    modelParameters: dict
        model input parameters (from .ini - file)
    modelPaths: dict
        contains paths to input files
    validParameterRanges: dict
        contains valid parameter ranges to be checked against

    Returns
    -----------
    raises a ValueError if any check is unsuccessful
    """

    minAlphaValid      = validParamRanges['minAlphaValid']
    maxAlphaValid      = validParamRanges['maxAlphaValid']
    maxZDeltaValid     = validParamRanges['maxZDeltaValid']
    maxExpValid        = validParamRanges['maxExpValid']
    minFluxThreshValid = validParamRanges['minFluxThreshValid']
    maxFluxThreshValid = validParamRanges['maxFluxThreshValid']

    # checking value ranges of (global) input parameters
    # alpha, max_z, exp, flux_threshold
    alpha = modelParameters["alpha"]
    if alpha < minAlphaValid or alpha > maxAlphaValid:
        log.error(
            f"Error: Alpha value is not within a physically sensible range ([{minAlphaValid}, {maxAlphaValid}])."
        )
        raise ValueError(
            f"Invalid (global) model parameter 'alpha' provided: {alpha}\n"
            f"please provide values for 'alpha' between {minAlphaValid} and {maxAlphaValid}.\n"
            f"HINT: if you insist on using values outside of [{minAlphaValid}, {maxAlphaValid}] you can adapt "
            "the values in 'getValidParameterRanges()' in com4FlowPy/com4FlowPy.py"
        )

    zDelta = modelParameters["max_z"]
    if zDelta < 0 or zDelta > maxZDeltaValid:
        log.error(
            f"Error: zDeltaMaxLimit value is not within a physically sensible range ([0,{maxZDeltaValid}])."
        )
        raise ValueError(
            f"Invalid (global) model parameter 'max_z' provided: {zDelta} meters\n"
            f"please provide values for 'max_z' between 0 and {maxZDeltaValid} meters.\n"
            f"HINT: if you insist on using values outside of [0, {maxZDeltaValid}] you can adapt "
            "the values in 'getValidParameterRanges()' in com4FlowPy/com4FlowPy.py"
        )

    exp = modelParameters["exp"]
    if exp < 0 or exp > maxExpValid:
        log.error(
            f"Error: Exponent value is not within a physically sensible range (> 0 and <={maxExpValid})."
        )
        raise ValueError(
            f"Invalid (global) model parameter 'exp' provided: ({exp})\n"
            f"please provide values for 'exp' between 0 and {maxExpValid}.\n"
            f"HINT: if you insist on using values outside of [>0, <={maxExpValid}] you can adapt "
            "the values in 'getValidParameterRanges()' in com4FlowPy/com4FlowPy.py"
        )

    fluxTh = modelParameters["flux_threshold"]
    if fluxTh < minFluxThreshValid or fluxTh > maxFluxThreshValid:
        log.error(
            f"Error: flux_threshold value is not within a physically sensible range ([{minFluxThreshValid}, {maxFluxThreshValid}])."
        )
        raise ValueError(
            f"Invalid (global) model parameter 'flux_threshold' provided: {fluxTh}\n"
            f"please provide values for 'flux_threshold' between {minFluxThreshValid} and {maxFluxThreshValid}.\n"
            f"HINT: if you insist on using values outside of [{minFluxThreshValid}, {maxFluxThreshValid}] you can adapt "
            "the values in 'getValidParameterRanges()' in com4FlowPy/com4FlowPy.py"
        )
    log.info("global input parameters are complete and within the valid range.")


def checkVariableInputParameters(modelParameters, modelPaths, validParamRanges):
    """
    checks spatially variable input parameters for valid input ranges and completness
    
    Parameters
    -----------
    modelParameters: dict
        model input parameters (from .ini - file)
    modelPaths: dict
        contains paths to input files
    validParameterRanges: dict
        contains valid parameter ranges to be checked against

    Returns
    -----------
    raises a ValueError if any check is unsuccessful
    """
    minAlphaValid  = validParamRanges['minAlphaValid']
    maxAlphaValid  = validParamRanges['maxAlphaValid']
    maxZDeltaValid = validParamRanges['maxZDeltaValid']
    maxUMaxValid   = validParamRanges['maxUMaxValid']
    maxExpValid    = validParamRanges['maxExpValid']

    # read the release cells raster
    pras = IOf.readRaster(modelPaths["releasePathWork"])["rasterData"]
    # convert to binary following the > 0 convention used
    praMask = (pras > 0)

    if modelParameters["varAlphaBool"]:
        data = IOf.readRaster(modelPaths["varAlphaPath"])
        rasterValues = data["rasterData"]
        isValid = validateInputArray(praMask, rasterValues, maxAlphaValid, minAlphaValid)

        if not isValid:
            log.error(
                f"Error: Some Alpha-raster values are incomplete or outside a physically sensible range ([0,{maxAlphaValid}])"
            )
            raise ValueError(
            f"Missing or invalid values for spatially varying model parameter 'alpha' found in {modelPaths['varAlphaPath']}\n"
            f"please provide values for 'alpha' between {minAlphaValid} and {maxAlphaValid} for all release cells.\n"
            f"HINT: if you insist on using values outside of [{minAlphaValid}, {maxAlphaValid}] you can adapt "
            "the values in 'getValidParameterRanges()' in com4FlowPy/com4FlowPy.py"
        )

    if modelParameters["varUmaxBool"]:
        data = IOf.readRaster(modelPaths["varUmaxPath"])
        rasterValues = data["rasterData"]
        _type = modelParameters["varUmaxType"].lower()
        if _type == "umax":
            _maxVal = maxUMaxValid
        else:
            _maxVal = maxZDeltaValid
        isValid = validateInputArray(praMask, rasterValues, _maxVal)

        if not isValid:
            log.error(
                f"Error: Some raster values of type '{_type}' are incomplete or outside a physically sensible range."
            )
            raise ValueError(
            f"Missing or invalid values for spatially varying model parameter '{_type}' found in {modelPaths['varUmaxPath']}\n"
            f"please provide values for '{_type}' between 0 and {_maxVal} for all release cells.\n"
            f"HINT: if you insist on using values outside of [0, {_maxVal}] you can adapt "
            "the values in 'getValidParameterRanges()' in com4FlowPy/com4FlowPy.py"
        )
    
    if modelParameters["varExponentBool"]:
        data = IOf.readRaster(modelPaths["varExponentPath"])
        rasterValues = data["rasterData"]

        isValid = validateInputArray(praMask, rasterValues, maxExpValid)

        if not isValid:
            log.error(
                f"Error: Some raster values of type 'exp' are incomplete or outside the allowed range [0, {maxExpValid}]."
            )
            raise ValueError(
            f"Missing or invalid values for spatially varying model parameter 'exp' found in {modelPaths['varExponentPath']}\n"
            f"please provide values for 'exp' between 0 and {maxExpValid} for all release cells.\n"
            f"HINT: if you insist on using values outside of [0, {maxExpValid}] you can adapt "
            "the values in 'getValidParameterRanges()' in com4FlowPy/com4FlowPy.py"
        )

    log.info("spatially variable input parameters are complete and within the valid range.")
    log.info(24*"=")