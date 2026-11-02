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

    if "relIdPolygon" in modelPaths["outputFileList"] or "relIdCount" in modelPaths["outputFileList"]:
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
    if modelParameters["forestBool"]:
        checkInputParameterValues(modelParameters, modelPaths, validParamRanges, forestParams)
    else:
        checkInputParameterValues(modelParameters, modelPaths, validParamRanges)

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
            if (
                _demHeader["ncols"] == _forestHeader["ncols"]
                and _demHeader["nrows"] == _forestHeader["nrows"]
            ):
                log.info("Forest Layer ok!")
            else:
                log.error("Error: Forest Layer doesn't match DEM!")
                sys.exit(1)

        if modelParameters["varUmaxBool"]:
            _varUmaxHeader = IOf.readRasterHeader(modelPaths["varUmaxPath"])
            if (
                _demHeader["ncols"] == _varUmaxHeader["ncols"]
                and _demHeader["nrows"] == _varUmaxHeader["nrows"]
            ):
                log.info("uMax Limit Layer ok!")
            else:
                log.error("Error: uMax Limit Layer doesn't match DEM!")
                sys.exit(1)

        if modelParameters["varAlphaBool"]:
            _varAlphaHeader = IOf.readRasterHeader(modelPaths["varAlphaPath"])
            if (
                _demHeader["ncols"] == _varAlphaHeader["ncols"]
                and _demHeader["nrows"] == _varAlphaHeader["nrows"]
            ):
                log.info("variable Alpha Layer ok!")
            else:
                log.error("Error: variable Alpha Layer doesn't match DEM!")
                sys.exit(1)

        if modelParameters["varExponentBool"]:
            _varExponentHeader = IOf.readRasterHeader(modelPaths["varExponentPath"])
            if (
                _demHeader["ncols"] == _varExponentHeader["ncols"]
                and _demHeader["nrows"] == _varExponentHeader["nrows"]
            ):
                log.info("variable exponent Layer ok!")
            else:
                log.error("Error: variable exponent Layer doesn't match DEM!")
                sys.exit(1)

        log.info("========================")

    except Exception as ex:
        log.error(
            "could not read all required Input Layers, please re-check files and paths provided in .ini files"
        )
        log.error("Error occured: %s" % ex)
        # return
        sys.exit(1)

def checkInputParameterValues(modelParameters, modelPaths, validParamRanges, forestParameters=None):
    """check if the input parameters are valid
    and within physically sensible limits.
    currently the function checks:
    - alpha, exp, flux_threshold, max_z/u_max (for global and spatially varying parameters)
    - if "forestBool" (calculation with a forest-module), also forestModule Parameters and input data are checked.
    
    Parameters
    -----------
    modelParameters: dict
        model input parameters (from .ini - file)
    modelPaths: dict
        contains paths to input files
    validParamRanges: dict
        contains valid parameter ranges (from .ini - file)
    forestParameters: dict
        (optional) if modelParameters["forstBool"] == True then this dict has to be provided
        otherwhise the function will fail with a ValueError
    
    Returns
    -----------
    Function raises ValueErrors and writes error message to logFile if a check fails.
    Upon successful check an affirmative message is written to the logfile
    """

    log.info("checking completeness and validity of input data")

    if modelParameters["forestBool"]:
        if forestParameters is None:
            _msg = (
                "If modelParameters['forestBool']=='True' you have to provide forestParameters dictionary "
                "when calling com4FlowPy.checkInputParameterValues()."
            )
            log.error(_msg)
            raise ValueError(_msg)
        else:
            # if forest functionality is used - also check these input parameters and files
            checkForestInputParameters(forestParameters, modelParameters, modelPaths)

    
    # check if engine parameter provided in the .ini matches available options
    engine = modelParameters["engine"]
    validEngines = {"python","numba"}

    if engine not in validEngines:
        _msg = (
            f"Invalid engine '{engine}' defined in config >> "
            f"Valid engine options are {sorted(validEngines)}"
        )
        log.error(_msg)
        raise ValueError(_msg)
    
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
        max value, for range validation of data entries
    minValue: int | float, optional
        min value, for range validation of data entries, by default 0

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

    SPAM.tileRaster(modelPaths["demPath"], "dem", modelPaths["tempDir"], _tileCOLS, _tileROWS, _U)
    SPAM.tileRaster(
        modelPaths["releasePathWork"], "init", modelPaths["tempDir"], _tileCOLS, _tileROWS, _U, isInit=True
    )

    if modelParameters["infraBool"]:
        SPAM.tileRaster(modelPaths["infraPath"], "infra", modelPaths["tempDir"], _tileCOLS, _tileROWS, _U)
    if modelParameters["varUmaxBool"]:
        SPAM.tileRaster(
            modelPaths["varUmaxPath"], "varUmax", modelPaths["tempDir"], _tileCOLS, _tileROWS, _U
        )
    if modelParameters["varAlphaBool"]:
        SPAM.tileRaster(
            modelPaths["varAlphaPath"], "varAlpha", modelPaths["tempDir"], _tileCOLS, _tileROWS, _U
        )
    if modelParameters["varExponentBool"]:
        SPAM.tileRaster(
            modelPaths["varExponentPath"], "varExponent", modelPaths["tempDir"], _tileCOLS, _tileROWS, _U
        )
    if modelParameters["forestBool"]:
        SPAM.tileRaster(modelPaths["forestPath"], "forest", modelPaths["tempDir"], _tileCOLS, _tileROWS, _U)
    if modelParameters["outputRelIdBool"]:
        SPAM.tileRaster(modelPaths["relIdPath"], "relId", modelPaths["tempDir"], _tileCOLS, _tileROWS, _U)
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
            modelPaths["resDir"] / "com4_{}_{}_relIdPolygon.geojson".format(_uid, _ts), driver="GeoJSON"
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
        _msg = (
            f"Invalid (global) model parameter 'alpha' provided ({alpha}°) >> "
            f"please provide values for 'alpha' between {minAlphaValid}° and {maxAlphaValid}°."
        )
        log.error(_msg)
        raise ValueError(_msg)

    zDelta = modelParameters["max_z"]
    if zDelta < 0 or zDelta > maxZDeltaValid:
        _msg = (
            f"Provided (global) 'max_z' value is not within a physically sensible range ([0, {maxZDeltaValid}]) >> "
            f"please provide values for 'max_z' between 0 and {maxZDeltaValid} meters."
        )
        log.error(_msg)
        raise ValueError(_msg)

    exp = modelParameters["exp"]
    if exp < 0 or exp > maxExpValid:
        _msg = (
            f"Error: (global) 'exp' value is not within a physically sensible range (> 0 and <={maxExpValid}) >> "
            f"please provide values for 'exp' between 0 and {maxExpValid}."
        )
        log.error(_msg)
        raise ValueError(_msg)

    fluxTh = modelParameters["flux_threshold"]
    if fluxTh < minFluxThreshValid or fluxTh > maxFluxThreshValid:
        _msg = (
            f"Error: 'flux_threshold' value is not within a physically sensible range ([{minFluxThreshValid}, {maxFluxThreshValid}]) >> "
            f"please provide values for 'flux_threshold' between {minFluxThreshValid} and {maxFluxThreshValid}.\n"
        )
        log.error(_msg)
        raise ValueError(_msg)

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
            _msg = (
                f"Error: Some Alpha-raster values are incomplete or outside a physically sensible range ([0,{maxAlphaValid}]) >> "
                f"please provide values for 'alpha' between {minAlphaValid}° and {maxAlphaValid}° for all release cells."
            )
            log.error(_msg)
            raise ValueError(_msg)

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
            _msg = (
                f"Error: Some raster values of type '{_type}' are incomplete or outside a physically sensible range >> "
                f"please provide values for '{_type}' between 0 and {_maxVal} for all release cells."
            )
            log.error(_msg)
            raise ValueError(_msg)
    
    if modelParameters["varExponentBool"]:
        data = IOf.readRaster(modelPaths["varExponentPath"])
        rasterValues = data["rasterData"]

        isValid = validateInputArray(praMask, rasterValues, maxExpValid)

        if not isValid:
            _msg = (
                f"Error: Some raster values of type 'exp' are incomplete or outside the allowed range [0, {maxExpValid}]. >> "
                f"please provide values for 'exp' between 0 and {maxExpValid} for all release cells."
            )
            log.error(_msg)
            raise ValueError(_msg)

    log.info("spatially variable input parameters are complete and within the valid range.")
    log.info(24*"=")

def checkForestInputParameters(forestParameters, modelParameters, modelPaths):
    """
    checks forest related input parameters for valid input ranges and completness
    
    Parameters
    -----------
    forestParameters: dict
        forest related input parameters (from .ini - file)
    modelParameters: dict
        model input parameters (from .ini - file)
    modelPaths: dict
        contains paths to input files (here forest related files)

    Returns
    -----------
    raises a ValueError if any check is unsuccessful
    """
    _module = forestParameters["forestModule"]

    _minAF = forestParameters['minAddedFriction']
    _maxAF = forestParameters['maxAddedFriction']
    _velLimFr = forestParameters['velThForFriction']
    _minDET = forestParameters['minDetrainment']
    _maxDET = forestParameters['maxDetrainment']
    _velLimDt = forestParameters['velThForDetrain']

    if forestParameters['skipForestDist'] < 0:
        _msg = (
            f"Invalid value for 'skipForestDist': {forestParameters['skipForestDist']} found in config >> "
            "please provide non-negative scalar values for the parameter"
        )
        log.error(_msg)
        raise ValueError(_msg)

    if modelParameters['varAlphaBool']:
        # if variable alpha is used, then we derive the max value
        _pras = IOf.readRaster(modelPaths["releasePathWork"])["rasterData"]
        _praMask = (_pras > 0)
        _data = IOf.readRaster(modelPaths["varAlphaPath"])
        _rasterValues = _data["rasterData"]
        _alphaMax = np.max(_rasterValues[_praMask])
    else:
        # if global alpha is used, we just use this value
        _alphaMax = modelParameters["alpha"]

    fsi = IOf.readRaster(modelPaths["forestPath"])["rasterData"]

    if _module == 'forestFriction' or _module == 'forestDetrainment':
        # FSI <= 0 is assumed to be 0/noData in flowClass.py
        # convert to binary following the > 0 convention used
        fsiMask = (fsi > 0)
        # check if the provided FSI-Layer / Forest Array is not empty
        if not np.any(fsiMask):
            _msg = (
                f"All values in the provided forest layer {modelPaths['forestPath']} are <= 0. >> "
                "if you want to disable forest interaction, define 'forest=False' in the config file; "
                "otherwhise provide a forest layer with valid forest cells (>0)."
            )
            log.error(_msg)
            raise ValueError(_msg)
        
        # check if FSI is provided and in correct scaling (0,...,1)
        isValid = validateInputArray(fsiMask, fsi, 1)

        if not isValid:
            _msg = (
                f"Invalid forest Parameters ('FSI') found in  {modelPaths['forestPath']} >> "
                "please provide FSI in a scaling/range between 0 and 1 for all forest cells,"
                "when using 'forestFriction' or 'forestDetrainment' options."
            )
            log.error(_msg)
            raise ValueError(_msg)
        
        # check that alpha + addedFriction is <= 90°
        if _alphaMax + _maxAF >= 90:
            _msg = (
                "maximum total friction (alpha + 'forestFriction') exceeding 90° "
                f"(maximum) alpha (no forest) provided = {_alphaMax}°, "
                f"maximum forest Friction provided = {_maxAF}° >> "
                "please use parameter combinations, that do not exceed max. 90°"
            )
            log.error(_msg)
            raise ValueError(_msg)

        # check if velocity limit is non-negative
        if _velLimFr < 0:
            _msg = (
                f"Invalid forest Parameters ('velThForFriction': {_velLimFr}) provided in com4FlowPy config >> "
                "please provide velocity limits that are >= 0 m/s."
            )
            log.error(_msg)
            raise ValueError(_msg)
        
        # check that minAddedFriction <= maxAddedFriction
        if _minAF > _maxAF:
            _msg = (
                "Invalid forest Parameters provided in com4FlowPy config >> "
                "'minAddedFriction' has to be <= 'maxAddedFriction'"
            )
            log.error(_msg)
            raise ValueError(_msg)

        if _module == 'forestDetrainment':          
            # velLimit is simple -- just hast to be >= 0 m/s
            if _velLimDt < 0:
                _msg = (
                    "Invalid forest Parameters ('velThForDetrain') provided in com4FlowPy config >> "
                    "please provide velocity limits that are >= 0 m/s."
                )
                log.error(_msg)
                raise ValueError(_msg)
            
            # minAF has to be smaller max AF ...
            if _minDET > _maxDET:
                _msg = (
                    "Invalid forest Parameters provided in com4FlowPy config >> "
                    "'minDetrainmentFor' has to be <= 'maxDetrainmentFor'"
                )
                log.error(_msg)
                raise ValueError(_msg)
            
            # added detrainment has to be between 0 and 1
            if _minDET < 0 or _maxDET > 1:
                _msg = (
                    "invalid 'forestDetrainment' parameters >> "
                    "min and max values should be provided in the range [0, ..., 1]"  
                )
                log.error(_msg)
                raise ValueError(_msg)

    elif _module == "forestFrictionLayer":
        fsiMask = (fsi > 0)
        # check if FSI is provided and in correct scaling (0,...,1)

        # check if velocity limit is non-negative
        if _velLimFr < 0:
            _msg = (
                "Invalid forest Parameters ('velThForFriction') provided in com4FlowPy config >> "
                "please provide velocity limits that are >= 0 m/s."
            )
            log.error(_msg)
            raise ValueError(_msg)
                        
        if forestParameters["fFrLayerType"] == 'absolute':
            isValid = validateInputArray(fsiMask, fsi, 90)
            if not isValid:
                _msg = (
                    f"Invalid forest Parameters ('forestFrictionLayer') found in  {modelPaths['forestPath']} >> "
                    "Please provide 'absolute' 'forestFrictionLayer' in a range between 0 and 90 for all forest cells."
                )
                log.error(_msg)
                raise ValueError(_msg)

        elif forestParameters["fFrLayerType"] == 'relative':
            _dAlphaForMax = np.max(fsi[fsiMask])
            if (_alphaMax + _dAlphaForMax) >= 90 or (_alphaMax + _dAlphaForMax) < 0:
                _msg = (
                    f"Invalid forest Parameters ('forestFrictionLayer') found in  {modelPaths['forestPath']} >> "
                    "please provide 'relative' 'forestFrictionLayer' values so that absolute forest friction values "
                    "stay within a range between 0 and 90 for all forest cells."
                )
                log.error(_msg)
                raise ValueError(_msg)
        else:
            _msg = (
                "unknown value for 'forestFrictionLayerType' provided >> "
                "please specify either 'absolute' or 'relative'"               
            )
            log.error(_msg)
            raise ValueError(_msg)
    
    else:
        _msg = (
            f"invalid 'forestModule' defined in config ({_module}) >> "
            "valid choices ['forestFriction','forestDetrainment','forestFrictionLayer']"
        )
        log.error(_msg)
        raise ValueError(_msg)

    log.info("forest related input parameters are complete and within valid ranges.")
    log.info(24*"=")
