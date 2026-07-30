#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Calculation functions (raster level)
"""

import sys
import numpy as np
import logging
import pathlib
import gc
import psutil
import time
import pickle
from itertools import zip_longest

from multiprocessing import Pool

from avaframe.com4FlowPy.flowClass import Cell
from avaframe.com4FlowPy.flowPath import Path

log = logging.getLogger(__name__)

def get_start_idx(dem, release, relIdArray=None, calcThalweg=False):
    """Sort Release Pixels by altitude and return the result as lists for the
    Rows and Columns, starting with the highest altitude
    If releaseIds are provided, sort by release Id to ensure that all cells belonging to a
    segmented PRA are computed after each other.

    Parameters
    -----------
    dem: numpy array
        Digital Elevation Model to gain information about altitude
    release: numpy array
        The release layer, release pixels need int value > 0
    relIdArray: numpy array
        release Ids
    calcThalweg: bool
        flag if thalweg is computed

    Returns
    -----------
    row_list: list
        Row indices of release pixels sorted by altitude
    col_list: list
        Column indices of release pixels sorted by altitude
    """
    row_list, col_list = np.where(release > 0)  # Gives back the indices of the release areas
    if len(row_list) > 0:
        altitude_list = []
        relIdList = []
        for i in range(len(row_list)):
            altitude_list.append(dem[row_list[i], col_list[i]])
            if relIdArray is not None and calcThalweg:
                relIdList.append(relIdArray[row_list[i], col_list[i]])

        # sort this list by releaseId
        if relIdArray is not None and calcThalweg:
            relIdList, altitude_list, row_list, col_list = list(
                zip(*sorted(zip(relIdList, altitude_list, row_list, col_list), reverse=True))
            )
        else:
            # Sort this lists by altitude
            altitude_list, row_list, col_list = list(
                zip(*sorted(zip(altitude_list, row_list, col_list), reverse=True))
            )

    return row_list, col_list


def split_release(release, pieces, relIdArray, calcThalweg):
    """Split the release layer in several tiles. The area is determined by
    the number of release pixels in it, so that every tile has the same amount
    of release pixels in it.

    In this version the split is performed along a flattened 2D-array to ensure
    a more even splitting of release pixels than just along the x-Axis ...

    NOTE: TO DO: Ideally a 'greedy' algorithm would let idle CPU cores 'snatch' any
    un-processed release cell until all releaseCells are handled -- this would
    ensure that the total workload is distributed evenly along all CPUs (which
    becomes an important factor for bigger model areas) !!!

    The release tiles have still the size of the original layer, so no split
    for the DEM is needed.

    If thalweg is computed, and release Ids are provided, cells belonging to one
    release Ids are not divided into separate release_lists (for different chunks),
    so the thalweg can be computed for one segmented PRA.

    Parameters
    -----------
    release: np.array
        a binary 0|1 array with release pixels designated by '1'
    pieces:  int
        number of chunck in which the release layer should be split
    relIdArray: numpy array
        release Ids of segmented relase areas
    calcThalweg: bool
        flag if thalweg is calculated

    Returns
    -----------
    release_list:   list
        contains the tiles(arrays) [array0, array1, ..]
    """

    if calcThalweg and relIdArray is not None:
        # release split for thalweg computation

        uniqueIds, counts = np.unique(relIdArray[release == 1], return_counts=True)

        pieces = np.minimum(pieces, len(uniqueIds))

        idCount = list(zip(uniqueIds, counts))
        idCount.sort(key=lambda x: x[1], reverse=True)

        # prepare lists for ids and number of cells
        _numberCells = np.zeros(pieces, dtype=int)
        _ids = [[] for _ in range(pieces)]

        # add relId to this chunk that has less cells yet
        for id, count in idCount:
            idx = np.argmin(_numberCells)
            _ids[idx].append(id)
            _numberCells[idx] += count

        # write release cells
        release_list = []
        for idsChunk in _ids:
            release_piece = np.zeros_like(release, dtype=release.dtype)
            if idsChunk:
                id_piece = np.isin(relIdArray, idsChunk)
                release_piece[id_piece] = release[id_piece]
            release_list.append(release_piece)

    else:
        # Flatten the array and compute the cumulative sum
        flat_release = release.flatten()
        cumulative_sum = np.cumsum(flat_release)

        total_sum = cumulative_sum[-1]
        sum_per_split = total_sum / pieces

        release_list = []
        start_index = 0

        for i in range(1, pieces):
            # Find the split point in the flattened array
            split_index = np.searchsorted(cumulative_sum, sum_per_split * i)

            # Create a new array for this split
            split_flat = np.zeros_like(flat_release)
            split_flat[start_index:split_index] = flat_release[start_index:split_index]

            # Reshape the flat array back to 2D and add to the list
            release_list.append(split_flat.reshape(release.shape))

            start_index = split_index

        # Handle the last piece
        split_flat = np.zeros_like(flat_release)
        split_flat[start_index:] = flat_release[start_index:]
        release_list.append(split_flat.reshape(release.shape))

    return release_list


def run(optTuple):
    """This is a wrapper around calculation() for performing model runs for a single tile of the model domain
    using multiprocessing across multiple CPUs and for saving results for processed tile to temp folder
    (modelPaths["tempFolder"])

    Parameters
    -----------
    optTuple: tuple
        with all necessary model information:

        - optTuple[0] (int) - i index of the processed tile (for loading correct data for tiles)
        - optTuple[1] (int) - j index of the processed tile (for loading correct data for tiles)
        - optTuple[2] (dict) - containing modelParameters
        - optTuple[3] (dict) - containing modelPaths
        - optTuple[4] (dict) - containing rasterAttributes
        - optTuple[5] (dict) - containing forestParameters
        - optTuple[6] (dict) - containing MPOptions

    """

    log = logging.getLogger(__name__)

    # Flow-Py parameters
    alpha = float(optTuple[2]["alpha"])
    exp = float(optTuple[2]["exp"])
    flux_threshold = float(optTuple[2]["flux_threshold"])
    max_z_delta = float(optTuple[2]["max_z"])
    infraBool = optTuple[2]["infraBool"]
    forestBool = optTuple[2]["forestBool"]
    forestInteraction = optTuple[2]["forestInteraction"]
    varUmaxBool = optTuple[2]["varUmaxBool"]
    varAlphaBool = optTuple[2]["varAlphaBool"]
    varExponentBool = optTuple[2]["varExponentBool"]
    fluxDistOldVersionBool = optTuple[2]["fluxDistOldVersionBool"]
    relIdBool = optTuple[2]["outputRelIdBool"]
    previewMode = optTuple[2]["previewMode"]
    calcGeneration = optTuple[2]["calcGeneration"]
    calcThalweg = optTuple[2]["calcThalweg"]
    if calcThalweg:
        thalwegParameters = {
            "thalwegDir": optTuple[3]["thalwegDir"],
            "thalwegCenterOf": optTuple[2]["thalwegCenterOf"],
            "thalwegVariables": optTuple[2]["thalwegVariables"],
            "calcRelID": optTuple[2]["thalwegReleaseArea"],
            "thalwegSaveRam": optTuple[2]["thalwegSaveRam"],
            "videoRelId": optTuple[2]["videoRelId"],
            "videoDataVariable": optTuple[2]["videoDataVariable"],
        }
    else:
        thalwegParameters = {
            "thalwegSaveRam": False,
        }

    # Temp-Dir (all input files are located here and results are written back in here)
    tempDir = optTuple[3]["tempDir"]

    # List of output layers
    outputs = optTuple[3]["outputFileList"]

    # raster-layer Attributes
    rasterAttributes = optTuple[4]
    cellsize = float(optTuple[4]["cellsize"])
    nodata = float(optTuple[4]["nodata"])

    MPOptions = optTuple[6]  # CPU, Multiprocessing options ...

    dem = np.load(tempDir / ("dem_%s_%s.npy" % (optTuple[0], optTuple[1])))
    release = np.load(tempDir / ("init_%s_%s.npy" % (optTuple[0], optTuple[1])))
    extentTile = np.load(tempDir / ("ext_%s_%s" % (optTuple[0], optTuple[1])), allow_pickle=True)
    rasterAttributes["extentTile"] = extentTile
    if infraBool:
        infra = np.load(tempDir / ("infra_%s_%s.npy" % (optTuple[0], optTuple[1])))
    else:
        infra = None

    # if forestBool == 'True'
    # --> load forestFile
    # --> read parametersOfForestExtension
    # NOTE-TODO: this is a quick work-around to simply include forest information - should probably be handled more
    # elegantly/explicitly AND error handling should probably be included
    if forestBool:
        forestArray = np.load(tempDir / ("forest_%s_%s.npy" % (optTuple[0], optTuple[1])))
        forestParams = optTuple[5]
        forestParams["forestInteraction"] = forestInteraction
    else:
        forestParams = None
        forestArray = None

    if varUmaxBool:
        varUmaxArray = np.load(tempDir / ("varUmax_%s_%s.npy" % (optTuple[0], optTuple[1])))
        if optTuple[2]["varUmaxType"].lower() == "umax":
            varUmaxArray[varUmaxArray > 0] = varUmaxArray[varUmaxArray > 0] ** 2 / 2 / 9.81
        elif optTuple[2]["varUmaxType"].lower() != "zdeltalim":
            log.error("PLease provide the type of the uMax Limit: 'uMax' (in m/s) or zDeltaMax (in m)!")
    else:
        varUmaxArray = None

    if varAlphaBool:
        varAlphaArray = np.load(tempDir / ("varAlpha_%s_%s.npy" % (optTuple[0], optTuple[1])))
    else:
        varAlphaArray = None

    if varExponentBool:
        varExponentArray = np.load(tempDir / ("varExponent_%s_%s.npy" % (optTuple[0], optTuple[1])))
    else:
        varExponentArray = None

    if relIdBool:
        relIdArray = np.load(tempDir / ("relId_%s_%s.npy" % (optTuple[0], optTuple[1])))
    else:
        relIdArray = None

    varParams = {
        "varUmaxBool": varUmaxBool,
        "varUmaxArray": varUmaxArray,
        "varAlphaBool": varAlphaBool,
        "varAlphaArray": varAlphaArray,
        "varExponentBool": varExponentBool,
        "varExponentArray": varExponentArray,
    }
    relOutputParams = {
        "relIdBool": relIdBool,
        "relIdArray": relIdArray,
    }

    # convert release areas to binary (0: no release areas, 1: release areas)
    # every positive value >0 is interpreted as release area
    release[release < 0] = 0
    release[release == nodata] = 0  # added in case nodata is non-negative
    release[np.isnan(release)] = 0
    release[release > 0] = 1

    nRel = np.sum(release)
    log.info("Number of release cells: %i" % nRel)

    nProcesses, nChunks = calculateMultiProcessingOptions(
        nRel,
        MPOptions["nCPU"],
        procPerCPU=MPOptions["procPerCPU"],
        maxChunks=MPOptions["maxChunks"],
        chunkSize=MPOptions["chunkSize"],
    )

    release_list = split_release(release, nChunks, relIdArray, calcThalweg)

    # select compute engine: "numba" JIT kernel, else the default Python (Cell) path.
    # numba does not (yet) implement infra/back-calculation, previewMode or the
    # release-id (relIdPolygon/relIdCount) outputs, so fall back to the Python
    # engine when any of those are active.
    engine = optTuple[2].get("engine", "python")
    calcFunc = calculation
    if engine == "numba":
        if infraBool or previewMode or relIdBool:
            log.warning("engine=numba does not support infra/previewMode/relId outputs — "
                        "falling back to the Python engine for this run")
        else:
            try:
                from avaframe.com4FlowPy.flowCoreNumba import calculationNumba
                calcFunc = calculationNumba
                log.info("Using numba compute engine")
            except ImportError:
                log.warning("engine=numba requested but 'numba' is not installed — "
                            "falling back to the Python engine")
    else:
        log.info("Using python compute engine")

    log.info(
        "Multiprocessing starts, used Cores/Processes/Chunks: %i/%i/%i"
        % (MPOptions["nCPU"], nProcesses, nChunks)
    )

    with Pool(processes=nProcesses) as pool:
        results = pool.map(
            calcFunc,
            [
                [  # TODO: write in dicts:
                    dem,
                    infra,
                    release_sub,
                    alpha,
                    exp,
                    flux_threshold,
                    max_z_delta,
                    rasterAttributes,
                    infraBool,
                    forestBool,
                    varParams,
                    fluxDistOldVersionBool,
                    previewMode,
                    forestArray,
                    forestParams,
                    outputs,
                    relOutputParams,
                    calcGeneration,
                    calcThalweg,
                    thalwegParameters,
                ]
                for release_sub in release_list
            ],
        )
        pool.close()
        pool.join()

    # TODO - NOTE:
    # Move this part into a separate function
    # initializing arrays for storing the results from the multiprocessing step
    zDeltaArray = np.zeros_like(dem, dtype=np.float32)
    fluxArray = np.ones_like(dem, dtype=np.float32) * -9999
    countArray = np.zeros_like(dem, dtype=np.int32)
    zDeltaSumArray = np.zeros_like(dem, dtype=np.float32)
    routFluxSumArray = np.zeros_like(dem, dtype=np.float32)
    depFluxSumArray = np.zeros_like(dem, dtype=np.float32)
    if infraBool:
        backcalc = np.ones_like(dem, dtype=np.int32) * -9999
    fpTravelAngleMaxArray = np.ones_like(dem, dtype=np.float32) * -9999
    fpTravelAngleMinArray = np.ones_like(dem, dtype=np.float32) * -9999
    slTravelAngleArray = np.ones_like(dem, dtype=np.float32) * -9999
    travelLengthMaxArray = np.ones_like(dem, dtype=np.float32) * -9999
    travelLengthMinArray = np.ones_like(dem, dtype=np.float32) * -9999
    if forestInteraction:
        forestIntArray = np.ones_like(dem, dtype=np.float32) * -9999
    if relOutputParams["relIdBool"]:
        processedStartCellIdDict = {}

    zDeltaList = []
    fluxList = []
    ccList = []
    zDeltaSumList = []
    routFluxSumList = []
    depFluxSumList = []
    if infraBool:
        backcalcList = []
    fpTravelAngleMaxList = []
    fpTravelAngleMinList = []
    slTravelAngleList = []
    travelLengthMaxList = []
    travelLengthMinList = []
    if relOutputParams["relIdBool"]:
        processedStartCellIdList = []
    if forestInteraction:
        forestIntList = []

    for i in range(len(results)):
        res = results[i]
        res = list(res)
        zDeltaList.append(res[0])
        fluxList.append(res[1])
        ccList.append(res[2])
        zDeltaSumList.append(res[3])
        if infraBool:
            backcalcList.append(res[4])
        fpTravelAngleMaxList.append(res[5])
        slTravelAngleList.append(res[6])
        travelLengthMaxList.append(res[7])
        travelLengthMinList.append(res[8])
        fpTravelAngleMinList.append(res[9])
        routFluxSumList.append(res[10])
        depFluxSumList.append(res[11])
        if relOutputParams["relIdBool"]:
            processedStartCellIdList.append(res[12])
        if forestInteraction:
            forestIntList.append(res[13])

    logging.info("Calculation finished, getting results.")
    for i in range(len(zDeltaList)):
        zDeltaArray = np.maximum(zDeltaArray, zDeltaList[i])
        fluxArray = np.maximum(fluxArray, fluxList[i])
        countArray += ccList[i]
        zDeltaSumArray += zDeltaSumList[i]
        routFluxSumArray += routFluxSumList[i]
        depFluxSumArray += depFluxSumList[i]
        if infraBool:
            backcalc = np.maximum(backcalc, backcalcList[i])
        if "fpTravelAngleMax" in outputs or "fpTravelAngle" in outputs:
            fpTravelAngleMaxArray = np.maximum(fpTravelAngleMaxArray, fpTravelAngleMaxList[i])
        if "fpTravelAngleMin" in outputs:
            fpTravelAngleMinArray = np.where(
                (fpTravelAngleMinArray >= 0) & (fpTravelAngleMinList[i] >= 0),
                np.minimum(fpTravelAngleMinArray, fpTravelAngleMinList[i]),
                np.maximum(fpTravelAngleMinArray, fpTravelAngleMinList[i]),
            )
        slTravelAngleArray = np.maximum(slTravelAngleArray, slTravelAngleList[i])
        if "travelLengthMax" in outputs or "travelLength" in outputs:
            travelLengthMaxArray = np.maximum(travelLengthMaxArray, travelLengthMaxList[i])
        if "travelLengthMin" in outputs:
            travelLengthMinArray = np.where(
                (travelLengthMinArray >= 0) & (travelLengthMinList[i] >= 0),
                np.minimum(travelLengthMinArray, travelLengthMinList[i]),
                np.maximum(travelLengthMinArray, travelLengthMinList[i]),
            )
        if forestInteraction:
            forestIntArray = np.where(
                (forestIntArray >= 0) & (forestIntList[i] >= 0),
                np.minimum(forestIntArray, forestIntList[i]),
                np.maximum(forestIntArray, forestIntList[i]),
            )
        if "relIdPolygon" in outputs or "relIdCount" in outputs:
            for key in processedStartCellIdList[i]:
                if key in processedStartCellIdDict:
                    ids = np.append(processedStartCellIdList[i][key], processedStartCellIdDict[key])
                    processedStartCellIdDict[key] = np.unique(ids)
                else:
                    processedStartCellIdDict[key] = processedStartCellIdList[i][key]

    if relOutputParams["relIdBool"]:
        saveDict = open(
            tempDir / ("res_startCellIdDict_%s_%s.pickle" % (optTuple[0], optTuple[1])),
            "wb",
        )
        pickle.dump(processedStartCellIdDict, saveDict)
        saveDict.close()
        del processedStartCellIdDict

    # Save Calculated tiles
    np.save(tempDir / ("res_z_delta_%s_%s" % (optTuple[0], optTuple[1])), zDeltaArray)
    np.save(tempDir / ("res_z_delta_sum_%s_%s" % (optTuple[0], optTuple[1])), zDeltaSumArray)
    np.save(
        tempDir / ("res_rout_flux_sum_%s_%s" % (optTuple[0], optTuple[1])),
        routFluxSumArray,
    )
    np.save(
        tempDir / ("res_dep_flux_sum_%s_%s" % (optTuple[0], optTuple[1])),
        depFluxSumArray,
    )
    np.save(tempDir / ("res_flux_%s_%s" % (optTuple[0], optTuple[1])), fluxArray)
    np.save(tempDir / ("res_count_%s_%s" % (optTuple[0], optTuple[1])), countArray)
    np.save(
        tempDir / ("res_fp_max_%s_%s" % (optTuple[0], optTuple[1])),
        fpTravelAngleMaxArray,
    )
    np.save(
        tempDir / ("res_fp_min_%s_%s" % (optTuple[0], optTuple[1])),
        fpTravelAngleMinArray,
    )
    np.save(tempDir / ("res_sl_%s_%s" % (optTuple[0], optTuple[1])), slTravelAngleArray)
    np.save(
        tempDir / ("res_travel_length_max_%s_%s" % (optTuple[0], optTuple[1])),
        travelLengthMaxArray,
    )
    np.save(
        tempDir / ("res_travel_length_min_%s_%s" % (optTuple[0], optTuple[1])),
        travelLengthMinArray,
    )
    if infraBool:
        np.save(tempDir / ("res_backcalc_%s_%s" % (optTuple[0], optTuple[1])), backcalc)
    if forestInteraction:
        np.save(
            tempDir / ("res_forestInt_%s_%s" % (optTuple[0], optTuple[1])),
            forestIntArray,
        )


def calculation(args):
    """This is the core function where all the data handling and calculation is
    done.

    Parameters
    -----------
    args: list
        contains the following model input data:

        - args[0] (np.array) - The digital elevation model
        - args[1] (np.array) - The infrastructure layer
        - args[2] (np.array) - cells with value 1 are PRAs
        - args[3] (float) - alpha angle
        - args[4] (float) - exponent
        - args[5] (float) - threshold of minimum flux
        - args[6] (float) - maximum of zDelta
        - args[7] (float) - raster attributes
        - args[8] (bool) -  flag for calculation with/without infrastructure
        - args[9] (bool) - flag for calculation with/without forest
        - args[10] (dict) - contains flags and numpy arrays for variable input parameters (Alpha, exp, uMax)
        - args[11] (bool) - flag for computing flux distribution with old version
        - args[12] (bool) - flag for previewMode / fast Calculation

        - args[13] (numpy array) - contains forest information (None if forestBool=False)
        - args[14] (dict) - contains parameters for forest interaction models (None if forestBool=False)
        - args[15] (list) - output names
        - args[16] (dict) - contains flags and rasters for release - information outputs
        - args[17] (bool) - flag for computing each generation
        - args[18] (bool) - flag for computing thalweg
        - args[19] (dict) - thalweg parameters

    Returns
    -----------
    zDeltaArray: numpy array
        the maximum of kinetic velocity height (zDelta) in every raster cell
    fluxArray: numpy array
        the maximum of flux in every cell
    countArray: numpy array
        the number of hits (GMF paths) in every cell
    zDeltaSumArray: numpy array
        the maximum of zDelta in every cell per path and the sum over the paths
    backcalc: numpy array
          Array with back calculation results
    fpTravelAngleArray: numpy array
        maximum of flow-path travel-angle in every cell
    slTravelAngleArray: numpy array
        maximum of sl travel-angle in every cell
    travelLengthArray: numpy array
        maximum of travel length in every cell
    routFluxSumArray:numpy array
        sum of routing flux in every cell
    depFluxSumArray:numpy array
        sum of deposition flux in every cell
    forestIntArray: numpy array
        minimum of the count a forested cell is hit (only returned if args[18]["forestInteraction"]==True)

    """
    log = logging.getLogger(__name__)

    # helper function for backTracking, a bit slower than inline but improves
    # readability by avoiding repetitions
    def updateInfraDirGraph(row, col, parentRow=None, parentCol=None):
        if (row, col) not in pathTopology:
            # if the current node is not a key in the dir-graph
            # it is added here along with it's infrastructure value
            pathTopology[(row, col)] = []
            infraValues[(row, col)] = max(0, infraArr[row, col])
        # adding the child node as a child to the parent cell in the dir-graph
        # if a parent is provided
        if parentRow and parentCol:
            pathTopology[(parentRow, parentCol)].append((row, col))

    # check if there's enough RAM available (default value set to 5%)
    # if not, wait for 30 secs and check again
    # should prevent the occurence of broken pipe errors or similar issues related
    # to RAM overflow
    handleMemoryAvailability()

    dem = args[0]
    infra = args[1]
    release = args[2]
    alpha = args[3]
    exp = args[4]
    flux_threshold = args[5]
    max_z_delta = args[6]
    rasterAttributes = args[7]
    cellsize = rasterAttributes["cellsize"]
    nodata = rasterAttributes["nodata"]
    infraBool = args[8]
    forestBool = args[9]
    varUmaxBool = args[10]["varUmaxBool"]
    varUmaxArray = args[10]["varUmaxArray"]
    varAlphaBool = args[10]["varAlphaBool"]
    varAlphaArray = args[10]["varAlphaArray"]
    varExponentBool = args[10]["varExponentBool"]
    varExponentArray = args[10]["varExponentArray"]
    fluxDistOldVersionBool = args[11]
    previewMode = args[12]
    outputs = args[15]
    relIdArray = args[16]["relIdArray"]
    relIdBool = args[16]["relIdBool"]
    calcGeneration = args[17]
    calcThalweg = args[18]
    thalwegParameters = args[19]

    if forestBool:
        forestArray = args[13]
        forestParams = args[14]
        forestInteraction = forestParams["forestInteraction"]
    else:
        forestInteraction = False
        forestArray = None
        forestParams = None
    if infraBool:
        # initialize infrastructure array
        # TODO: check if this can be simplified
        infraArr = infra  # infrastructure array (input file)

    zDeltaArray = np.zeros_like(dem, dtype=np.float32)
    zDeltaSumArray = np.zeros_like(dem, dtype=np.float32)
    zDeltaPathList = []
    routFluxSumArray = np.zeros_like(dem, dtype=np.float32)
    depFluxSumArray = np.zeros_like(dem, dtype=np.float32)
    fluxArray = np.ones_like(dem, dtype=np.float32) * -9999
    countArray = np.zeros_like(dem, dtype=np.int32)
    slTravelAngleArray = np.ones_like(dem, dtype=np.float32) * -9999  # sl = Straight Line

    if "fpTravelAngleMax" in outputs or "fpTravelAngle" in outputs:
        fpTravelAngleMaxArray = np.ones_like(dem, dtype=np.float32) * -9999  # fp = Flow Path
    else:
        fpTravelAngleMaxArray = None

    if "fpTravelAngleMin" in outputs:
        fpTravelAngleMinArray = np.ones_like(dem, dtype=np.float32) * -9999  # fp = Flow Path
    else:
        fpTravelAngleMinArray = None

    if "travelLengthMin" in outputs:
        travelLengthMinArray = np.ones_like(dem, dtype=np.float32) * -9999
    else:
        travelLengthMinArray = None

    if "travelLengthMax" in outputs or "travelLength" in outputs:
        travelLengthMaxArray = np.ones_like(dem, dtype=np.float32) * -9999
    else:
        travelLengthMaxArray = None

    if infraBool:
        backcalc = np.ones_like(dem, dtype=np.int32) * -9999
    else:
        backcalc = None

    if forestInteraction:
        forestIntArray = np.ones_like(dem, dtype=np.float32) * -9999
    else:
        forestIntArray = None
    if relIdBool:
        startCellIdDict = {}
    else:
        startCellIdDict = None

    # Core
    # NOTE-TODO: row_list, col_list are tuples - rethink variable naming
    row_list, col_list = get_start_idx(dem, release, relIdArray, calcThalweg)

    generationListRelId = []
    colListRelId = []
    rowListRelId = []
    fluxListRelId = []
    zdeltaListRelId = []
    travelLengthMaxListRelId = []
    startcell_idx = 0
    if calcThalweg:
        timeThalweg = 0.0
        nextRowIdx = row_list[0]
        nextColIdx = col_list[0]
    while startcell_idx < len(row_list):
        if infraBool:
            # if infraBool - here we initialize a directed graph structure
            pathTopology = {}  # topology of path as directed graph
            infraValues = {}  # values

        processedCells = {}  # dictionary of cells that have been processed already
        zDeltaPathArray = np.zeros_like(dem, dtype=np.float32)

        row_idx = row_list[startcell_idx]
        col_idx = col_list[startcell_idx]
        dem_ng = dem[row_idx - 1 : row_idx + 2, col_idx - 1 : col_idx + 2]  # neighbourhood DEM
        if varUmaxBool and varUmaxArray is not None:
            if varUmaxArray[row_idx, col_idx] > 0 and varUmaxArray[row_idx, col_idx] <= 8848:
                max_z_delta = varUmaxArray[row_idx, col_idx]
            else:
                max_z_delta = args[6]

        if varAlphaBool and varAlphaArray is not None:
            if varAlphaArray[row_idx, col_idx] > 0 and varAlphaArray[row_idx, col_idx] <= 90:
                alpha = varAlphaArray[row_idx, col_idx]
            else:
                alpha = args[3]

        if varExponentBool and varExponentArray is not None:
            if varExponentArray[row_idx, col_idx] > 0:
                exp = varExponentArray[row_idx, col_idx]
            else:
                exp = args[4]

        if (nodata in dem_ng) or np.size(dem_ng) < 9:
            startcell_idx += 1
            continue

        if relIdBool:
            startcellId = relIdArray[row_idx, col_idx]
        else:
            startcellId = None

        startcell = Cell(
            row_idx,
            col_idx,
            dem_ng,
            cellsize,
            1,
            0,
            None,
            alpha,
            exp,
            flux_threshold,
            max_z_delta,
            startcell=True,
            fluxDistOldVersionBool=fluxDistOldVersionBool,
            FSI=forestArray[row_idx, col_idx] if isinstance(forestArray, np.ndarray) else None,
            forestParams=forestParams,
        )

        # dictionary of all the cells that have been processed and the number of times the cell has been visited
        processedCells[(startcell.rowindex, startcell.colindex)] = 1

        # list of flowClass.Cell() Objects that is contains the "path" for each release-cell
        if calcGeneration:
            cellList = [startcell]  # list of parents for current iteration
            genList = [cellList]  # list of all cells (which are calculated), organised in generations
            childList = []  # list of childs of the current iteration
            childIndex = {}
            if thalwegParameters["thalwegSaveRam"]:
                colThalwegLists = []
                rowThalwegLists = []
                fluxThalwegLists = []
                zdeltaThalwegLists = []
                travelLengthMaxThalwegLists = []

            for gen, cellList in enumerate(genList):
                if thalwegParameters["thalwegSaveRam"]:
                    colThalwegGen = []
                    rowThalwegGen = []
                    fluxThalwegGen = []
                    zdeltaThalwegGen = []
                    travelLengthMaxThalwegGen = []
                for idx, cell in enumerate(cellList):
                    if relIdBool:
                        if (cell.rowindex, cell.colindex) in startCellIdDict:
                            startcellIdList = np.append(
                                startCellIdDict[(cell.rowindex, cell.colindex)],
                                startcellId,
                            )
                            startCellIdDict[(cell.rowindex, cell.colindex)] = np.unique(startcellIdList)
                        else:
                            startCellIdDict[(cell.rowindex, cell.colindex)] = np.array([startcellId])

                    # calculate flux, z_delta from current cell (cell) to child-cells
                    # lenght of row, col, flux, and z_delta vectors correspond to
                    # number of child cells (successors) to currently processed cell
                    row, col, flux, z_delta = cell.calc_distribution()

                    if len(row) > 0:
                        # mass, row, col  = list(zip(*sorted(zip( mass, row, col), reverse=False)))
                        z_delta, flux, row, col = list(
                            zip(*sorted(zip(z_delta, flux, row, col), reverse=False))
                        )
                        # Sort this lists by elh, to start with the highest cell

                    if infraBool:
                        # if the current cell is not already in the dir-graph, then we add it here
                        updateInfraDirGraph(cell.rowindex, cell.colindex)

                    newRow = []
                    newCol = []
                    newFlux = []
                    newZDelta = []

                    for r, c, f, zd in zip(row, col, flux, z_delta):
                        key = (r, c)

                        if key in childIndex:
                            child = childList[childIndex[key]]
                            child.add_os(f)
                            child.add_parent(cell)

                            if infraBool:
                                updateInfraDirGraph(r, c, cell.rowindex, cell.colindex)

                            if zd > child.z_delta:
                                child.z_delta = zd

                        else:
                            newRow.append(r)
                            newCol.append(c)
                            newFlux.append(f)
                            newZDelta.append(zd)

                    row = newRow
                    col = newCol
                    flux = newFlux
                    z_delta = newZDelta


                    # TODO: we could put this checking part in an extra function, if we can move updateInfraDirGraph
                    """
                    # I substitute this with checking for the cell in the dict (part above)
                    for i in range(len(childList)):  # Check if Cell already exists in childList
                        k = 0
                        while k < len(row):
                            if row[k] == childList[i].rowindex and col[k] == childList[i].colindex:
                                childList[i].add_os(flux[k])
                                childList[i].add_parent(cell)

                                if infraBool:
                                    updateInfraDirGraph(row[k], col[k], cell.rowindex, cell.colindex)

                                if z_delta[k] > childList[i].z_delta:
                                    childList[i].z_delta = z_delta[k]
                                row = np.delete(row, k)
                                col = np.delete(col, k)
                                flux = np.delete(flux, k)
                                z_delta = np.delete(z_delta, k)
                            else:
                                k += 1
                    """

                    for k in range(len(row)):
                        dem_ng = dem[row[k] - 1 : row[k] + 2, col[k] - 1 : col[k] + 2]  # neighbourhood DEM

                        # This bit handles edge cases and noData-values in the DEM!! this is an important piece of code, since
                        # no-data handling is expected (by some users/applications) to behave like here:
                        # i.e. if nodata in the 3x3 neighbourhood --> no calculation
                        if (nodata in dem_ng) or np.size(dem_ng) < 9:
                            continue
                        if infraBool:
                            updateInfraDirGraph(row[k], col[k], cell.rowindex, cell.colindex)

                        # if the current child cell is already in processedCells
                        # just add +1 to the visit-counter, else add it to the
                        # processedCells dictionary with visit-count = 1
                        if (row[k], col[k]) in processedCells:
                            processedCells[(row[k], col[k])] += 1
                        else:
                            processedCells[(row[k], col[k])] = 1

                        childIndex[(row[k], col[k])] = len(
                            childList)  # important that it's before childlist.append(...)
                        childList.append(
                            Cell(
                                row[k],
                                col[k],
                                dem_ng,
                                cellsize,
                                flux[k],
                                z_delta[k],
                                cell,
                                alpha,
                                exp,
                                flux_threshold,
                                max_z_delta,
                                startcell,
                                FSI=(
                                    forestArray[row[k], col[k]]
                                    if isinstance(forestArray, np.ndarray)
                                    else None
                                ),
                                forestParams=forestParams,
                            )
                        )

                    # TODO: writing arrays in a separate function?
                    routFluxSumArray[cell.rowindex, cell.colindex] += cell.flux
                    depFluxSumArray[cell.rowindex, cell.colindex] += cell.fluxDep
                    zDeltaArray[cell.rowindex, cell.colindex] = max(
                        zDeltaArray[cell.rowindex, cell.colindex], cell.z_delta
                    )
                    fluxArray[cell.rowindex, cell.colindex] = max(
                        fluxArray[cell.rowindex, cell.colindex], cell.flux
                    )
                    zDeltaPathArray[cell.rowindex, cell.colindex] = max(
                        zDeltaPathArray[cell.rowindex, cell.colindex], cell.z_delta
                    )
                    if "fpTravelAngleMax" in outputs or "fpTravelAngle" in outputs:
                        fpTravelAngleMaxArray[cell.rowindex, cell.colindex] = max(
                            fpTravelAngleMaxArray[cell.rowindex, cell.colindex],
                            cell.max_gamma,
                        )
                    if "fpTravelAngleMin" in outputs:
                        if fpTravelAngleMinArray[cell.rowindex, cell.colindex] >= 0 and cell.max_gamma >= 0:
                            fpTravelAngleMinArray[cell.rowindex, cell.colindex] = min(
                                fpTravelAngleMinArray[cell.rowindex, cell.colindex],
                                cell.max_gamma,
                            )
                        else:
                            fpTravelAngleMinArray[cell.rowindex, cell.colindex] = max(
                                fpTravelAngleMinArray[cell.rowindex, cell.colindex],
                                cell.max_gamma,
                            )
                    slTravelAngleArray[cell.rowindex, cell.colindex] = max(
                        slTravelAngleArray[cell.rowindex, cell.colindex], cell.sl_gamma
                    )
                    if "travelLengthMax" in outputs or "travelLength" in outputs:
                        travelLengthMaxArray[cell.rowindex, cell.colindex] = max(
                            travelLengthMaxArray[cell.rowindex, cell.colindex],
                            cell.min_distance,
                        )
                    if "travelLengthMin" in outputs:
                        if (
                            travelLengthMinArray[cell.rowindex, cell.colindex] >= 0
                            and cell.min_distance >= 0
                        ):
                            travelLengthMinArray[cell.rowindex, cell.colindex] = min(
                                travelLengthMinArray[cell.rowindex, cell.colindex],
                                cell.min_distance,
                            )
                        else:
                            travelLengthMinArray[cell.rowindex, cell.colindex] = max(
                                travelLengthMinArray[cell.rowindex, cell.colindex],
                                cell.min_distance,
                            )

                    # TODO: why does the cell count not work as without generation-computation?
                    if processedCells[(cell.rowindex, cell.colindex)] == 1:
                        countArray[cell.rowindex, cell.colindex] += int(1)
                    elif (
                        processedCells[(cell.rowindex, cell.colindex)] > 1
                        and countArray[cell.rowindex, cell.colindex] <= 0
                    ):
                        countArray[cell.rowindex, cell.colindex] += int(1)

                    if forestInteraction:
                        if forestIntArray[cell.rowindex, cell.colindex] >= 0 and cell.forestIntCount >= 0:
                            forestIntArray[cell.rowindex, cell.colindex] = min(
                                forestIntArray[cell.rowindex, cell.colindex],
                                cell.forestIntCount,
                            )
                        else:
                            forestIntArray[cell.rowindex, cell.colindex] = max(
                                forestIntArray[cell.rowindex, cell.colindex],
                                cell.forestIntCount,
                            )
                    if thalwegParameters["thalwegSaveRam"]:
                        colThalwegGen.append(cell.colindex)
                        rowThalwegGen.append(cell.rowindex)
                        fluxThalwegGen.append(cell.flux)
                        zdeltaThalwegGen.append(cell.z_delta)
                        travelLengthMaxThalwegGen.append(cell.min_distance)

                if len(childList) > 0:
                    cellList = childList
                    genList.append(cellList)
                    childList = []
                    childIndex = {}

                    if thalwegParameters["thalwegSaveRam"]:
                        colThalwegLists.append(colThalwegGen)
                        rowThalwegLists.append(rowThalwegGen)
                        fluxThalwegLists.append(fluxThalwegGen)
                        zdeltaThalwegLists.append(zdeltaThalwegGen)
                        travelLengthMaxThalwegLists.append(travelLengthMaxThalwegGen)

                        # empty last generation in genList to save RAM
                        if gen > 1:
                            genList[gen - 1] = []

            if calcThalweg and thalwegParameters["calcRelID"]:
                if thalwegParameters["thalwegSaveRam"]:
                    colListRelId = [
                        (colThisCell or []) + (colBefore or [])
                        for colThisCell, colBefore in zip_longest(colListRelId, colThalwegLists)
                    ]
                    rowListRelId = [
                        (rowThisCell or []) + (rowBefore or [])
                        for rowThisCell, rowBefore in zip_longest(rowListRelId, rowThalwegLists)
                    ]
                    fluxListRelId = [
                        (fluxThisCell or []) + (fluxBefore or [])
                        for fluxThisCell, fluxBefore in zip_longest(fluxListRelId, fluxThalwegLists)
                    ]
                    zdeltaListRelId = [
                        (zdeltaThisCell or []) + (zdeltaBefore or [])
                        for zdeltaThisCell, zdeltaBefore in zip_longest(zdeltaListRelId, zdeltaThalwegLists)
                    ]
                    travelLengthMaxListRelId = [
                        (travelLengthMaxThisCell or []) + (travelLengthMaxBefore or [])
                        for travelLengthMaxThisCell, travelLengthMaxBefore in zip_longest(
                            travelLengthMaxListRelId, travelLengthMaxThalwegLists
                        )
                    ]
                else:
                    # zip the generationLists within one release Id
                    generationListRelId = [
                        (generationThisCell or []) + (generationBefore or [])
                        for generationThisCell, generationBefore in zip_longest(generationListRelId, genList)
                    ]

                # check if the next startcell has the same startcellId
                if startcell_idx + 1 < len(row_list):
                    nextRowIdx = row_list[startcell_idx + 1]
                    nextColIdx = col_list[startcell_idx + 1]
                    lastStartcell = False
                else:
                    # if this was the last startcell, we also want to compute the thalweg!
                    lastStartcell = True
                if startcellId != relIdArray[nextRowIdx, nextColIdx] or lastStartcell:
                    log.info(f"Finished computing PRA with ID {startcellId}. Start computing its thalweg!")
                    timeThawlegStart = time.time()
                    if thalwegParameters["thalwegSaveRam"]:
                        listsRelId = {
                            "row": rowListRelId,
                            "col": colListRelId,
                            "flux": fluxListRelId,
                            "zdelta": zdeltaListRelId,
                            "travelLengthMax": travelLengthMaxListRelId,
                        }
                        path = Path(
                            dem,
                            row_list[startcell_idx],
                            col_list[startcell_idx],
                            None,
                            rasterAttributes,
                            countArray,
                            startcellId,
                            listsRelId,
                            cellList[0],
                        )
                        path.calcAndSaveThalwegData(thalwegParameters)
                        del path
                        colListRelId = []
                        rowListRelId = []
                        fluxListRelId = []
                        zdeltaListRelId = []
                        travelLengthMaxListRelId = []
                        listsRelId = {}

                    else:
                        if str(int(startcellId)) in list(thalwegParameters["videoRelId"].split("|")):
                            saveGenerationVideoData(generationListRelId,
                                                    startcellId,
                                                    dem,
                                                    rasterAttributes,
                                                    outDir=thalwegParameters["thalwegDir"] / "videoData",
                                                    variable=thalwegParameters["videoDataVariable"])
                        path = Path(
                            dem,
                            row_list[startcell_idx],
                            col_list[startcell_idx],
                            generationListRelId,
                            rasterAttributes,
                            countArray,
                            startcellId,
                        )
                        path.calcAndSaveThalwegData(thalwegParameters)
                        del path
                        generationListRelId = []
                    timeThalwegEnd = time.time()
                    timeThalweg += timeThalwegEnd - timeThawlegStart
                    log.info(
                        f"Finished computing thalweg of PRA with ID {startcellId}, it took {np.round(timeThalwegEnd - timeThawlegStart, 1)} s."
                    )

            elif calcThalweg:
                path = Path(
                    dem,
                    row_list[startcell_idx],
                    col_list[startcell_idx],
                    genList,
                    rasterAttributes,
                    countArray,
                )
                path.calcAndSaveThalwegData(thalwegParameters)

        else:
            cellList = []
            cellList.append(startcell)

            if infraBool:
                # adding start-cell as "root-node" to directed graph of the modeled process path
                updateInfraDirGraph(startcell.rowindex, startcell.colindex)

            for idx, cell in enumerate(cellList):
                if relIdBool:
                    if (cell.rowindex, cell.colindex) in startCellIdDict:
                        startcellIdList = np.append(
                            startCellIdDict[(cell.rowindex, cell.colindex)], startcellId
                        )
                        startCellIdDict[(cell.rowindex, cell.colindex)] = np.unique(startcellIdList)
                    else:
                        startCellIdDict[(cell.rowindex, cell.colindex)] = np.array([startcellId])

                row, col, flux, z_delta = cell.calc_distribution()

                if len(flux) > 0:
                    # mass, row, col  = list(zip(*sorted(zip( mass, row, col), reverse=False)))
                    z_delta, flux, row, col = list(zip(*sorted(zip(z_delta, flux, row, col), reverse=False)))
                    # Sort this lists by elh, to start with the highest cell

                if infraBool:
                    # if the current cell is not already in the dir-graph, then we add it here
                    updateInfraDirGraph(cell.rowindex, cell.colindex)

                # check if cell already exists
                for i in range(idx, len(cellList)):  # Check if Cell already exists
                    k = 0
                    while k < len(row):
                        if row[k] == cellList[i].rowindex and col[k] == cellList[i].colindex:
                            cellList[i].add_os(flux[k])
                            cellList[i].add_parent(cell)

                            if infraBool:
                                updateInfraDirGraph(row[k], col[k], cell.rowindex, cell.colindex)

                            if z_delta[k] > cellList[i].z_delta:
                                cellList[i].z_delta = z_delta[k]
                            row = np.delete(row, k)
                            col = np.delete(col, k)
                            flux = np.delete(flux, k)
                            z_delta = np.delete(z_delta, k)
                        else:
                            k += 1
                for k in range(len(row)):
                    dem_ng = dem[row[k] - 1 : row[k] + 2, col[k] - 1 : col[k] + 2]  # neighbourhood DEM

                    # This bit handles edge cases and noData-values in the DEM!! this is an important piece of
                    # code, sinceno-data handling is expected (by some users/applications) to behave like here:
                    # i.e. if nodata in the 3x3 neighbourhood --> no calculation
                    if (nodata in dem_ng) or np.size(dem_ng) < 9:
                        continue

                    if infraBool:
                        updateInfraDirGraph(row[k], col[k], cell.rowindex, cell.colindex)

                    # if the current child cell is already in processedCells
                    # just add +1 to the visit-counter, else add it to the
                    # processedCells dictionary with visit-count = 1
                    if (row[k], col[k]) in processedCells:
                        processedCells[(row[k], col[k])] += 1
                    else:
                        processedCells[(row[k], col[k])] = 1

                    cellList.append(
                        Cell(
                            row[k],
                            col[k],
                            dem_ng,
                            cellsize,
                            flux[k],
                            z_delta[k],
                            cell,
                            alpha,
                            exp,
                            flux_threshold,
                            max_z_delta,
                            startcell,
                            FSI=forestArray[row[k], col[k]] if isinstance(forestArray, np.ndarray) else None,
                            forestParams=forestParams,
                        )
                    )
                zDeltaArray[cell.rowindex, cell.colindex] = max(
                    zDeltaArray[cell.rowindex, cell.colindex], cell.z_delta
                )
                fluxArray[cell.rowindex, cell.colindex] = max(
                    fluxArray[cell.rowindex, cell.colindex], cell.flux
                )
                routFluxSumArray[cell.rowindex, cell.colindex] += cell.flux
                depFluxSumArray[cell.rowindex, cell.colindex] += cell.fluxDep
                zDeltaPathArray[cell.rowindex, cell.colindex] = max(
                    zDeltaPathArray[cell.rowindex, cell.colindex], cell.z_delta
                )
                if "fpTravelAngleMax" in outputs or "fpTravelAngle" in outputs:
                    fpTravelAngleMaxArray[cell.rowindex, cell.colindex] = max(
                        fpTravelAngleMaxArray[cell.rowindex, cell.colindex],
                        cell.max_gamma,
                    )
                if "fpTravelAngleMin" in outputs:
                    if fpTravelAngleMinArray[cell.rowindex, cell.colindex] >= 0 and cell.max_gamma >= 0:
                        fpTravelAngleMinArray[cell.rowindex, cell.colindex] = min(
                            fpTravelAngleMinArray[cell.rowindex, cell.colindex],
                            cell.max_gamma,
                        )
                    else:
                        fpTravelAngleMinArray[cell.rowindex, cell.colindex] = max(
                            fpTravelAngleMinArray[cell.rowindex, cell.colindex],
                            cell.max_gamma,
                        )
                slTravelAngleArray[cell.rowindex, cell.colindex] = max(
                    slTravelAngleArray[cell.rowindex, cell.colindex], cell.sl_gamma
                )

                if "travelLengthMax" in outputs or "travelLength" in outputs:
                    travelLengthMaxArray[cell.rowindex, cell.colindex] = max(
                        travelLengthMaxArray[cell.rowindex, cell.colindex],
                        cell.min_distance,
                    )
                if "travelLengthMin" in outputs:
                    if travelLengthMinArray[cell.rowindex, cell.colindex] >= 0 and cell.min_distance >= 0:
                        travelLengthMinArray[cell.rowindex, cell.colindex] = min(
                            travelLengthMinArray[cell.rowindex, cell.colindex],
                            cell.min_distance,
                        )
                    else:
                        travelLengthMinArray[cell.rowindex, cell.colindex] = max(
                            travelLengthMinArray[cell.rowindex, cell.colindex],
                            cell.min_distance,
                        )

                if processedCells[(cell.rowindex, cell.colindex)] == 1:
                    countArray[cell.rowindex, cell.colindex] += int(1)

                if forestInteraction:
                    if forestIntArray[cell.rowindex, cell.colindex] >= 0 and cell.forestIntCount >= 0:
                        forestIntArray[cell.rowindex, cell.colindex] = min(
                            forestIntArray[cell.rowindex, cell.colindex],
                            cell.forestIntCount,
                        )
                    else:
                        forestIntArray[cell.rowindex, cell.colindex] = max(
                            forestIntArray[cell.rowindex, cell.colindex],
                            cell.forestIntCount,
                        )

        if infraBool:
            # if 'infraBool' is True - i.e. calculation is performed with infrastructure information
            # then we perform the back-tracking of the stored directed graph (topology and node values)

            updatedInfraValues = backTracking(
                pathTopology, infraValues
            )  # actual "back-tracking" for current process-path

            for key, val in updatedInfraValues.items():
                backcalc[key[0], key[1]] = max(
                    backcalc[key[0], key[1]], val
                )  # writing max-values to back-tracking array

            del pathTopology, infraValues, updatedInfraValues
            gc.collect()

        if previewMode:
            # if the 'previewMode' is On/'True', then we check here if the current modeled process zones already
            # includes other release Cells (i.e. if release cells are "hit from above")
            # if this is the case, then we exclude the affected release cell(s) from further processing and update
            # the row_list, col_list variables containing the release cells that should be processed
            release[zDeltaArray > 0] = 0
            row_list, col_list = get_start_idx(dem, release, relIdArray, calcThalweg)

        if "zDeltaSum" in outputs:
            zDeltaPathList.append(zDeltaPathArray)
        del processedCells, zDeltaPathArray

        startcell_idx += 1

    if "zDeltaSum" in outputs:
        for zDeltaPathArray in zDeltaPathList:
            zDeltaSumArray += zDeltaPathArray

    gc.collect()
    return (
        zDeltaArray,
        fluxArray,
        countArray,
        zDeltaSumArray,
        backcalc,
        fpTravelAngleMaxArray,
        slTravelAngleArray,
        travelLengthMaxArray,
        travelLengthMinArray,
        fpTravelAngleMinArray,
        routFluxSumArray,
        depFluxSumArray,
        startCellIdDict,
        forestIntArray,
    )


def enoughMemoryAvailable(limit=0.05):
    """simple function to monitor memory(RAM) availability during parallel processing
    of calculation() inside run(). utilizing psutil

    Parameters
    -----------
    limit: float
        available RAM memory limit (between 0 and 1) - default at 0.05 (i.e. 5%)

    Returns
    -----------
    bool
        'True' if more than the defined memory-limit is still available;
        'False' if less than the defined memory-limit is available
    """

    log = logging.getLogger(__name__)
    availableMemory = psutil.virtual_memory().available / psutil.virtual_memory().total

    if availableMemory >= limit:
        # log.info('RAM availability o.k. -- %.2f %% of %.2f GiB'%
        # (availableMemory*100,psutil.virtual_memory().total/(1024.**3)))
        return True
    else:
        log.info(
            "RAM availability at limit -- %.2f %% of %.2f GiB - maybe recheck multiProcessing/Tiling settings"
            % (availableMemory * 100, psutil.virtual_memory().total / (1024.0**3))
        )
        return False


def calculateMultiProcessingOptions(nRel, nCPU, procPerCPU=1, maxChunks=500, chunkSize=50):
    """compute required options for multiprocessing of calulation() function inside run() and accompanied splitting of
    release cells into chunks in split_release().append
    The general idea is to make good use of available CPU resources to speed up calculations while not getting into
    trouble with RAM issues ...

    NOTE: this is still a quick'n'dirty hack, it might make sense to have a more sophisticated approach for optimization
    of CPU and RAM resource usage during multiprocessing depending on e.g.:
        - size of the numpy arrays that are processed (depending on tileSize and rasterResolution)
        - density of release areas in the tile
        - total available RAM and CPUs on the machine
        - (other com4FlowPy Parameterization)

    Parameters
    -----------
    nRel: int
        number of release Pixels inside the tile (i.e. all cells/pixels with values >=1 in 'release')
    nCPU: int
        number of available CPUs (as defined in the .ini files)
    procPerCPU: int
        number of processes to be spawned per CPU (default = 1) - might be set higher for increased performance
    maxChunks: int
        hard limit to the maximum number of chunks that is used --> a larger number of chunks will very
        probably increase performance in terms of maximising CPU workload (especially with large numbers of
        nCPU) but also cause higher RAM consumption (in the current multiprocessing implementation)
    chunkSize: int
        default number of release pixels per chunk in cases where the chunk-size is not constrained by
        nCPU*procPerCPU or maxChunks

    Returns
    -----------
    nChunks: int
        the number of chunks into which the release layer/array is split for multiprocessing
    nProcesses: int
        the number of processes used in Pool.map() inside run()
    """
    nProcesses = int(nCPU * procPerCPU)
    # check if release is empty - if so, there's no reason to split
    if nRel == 0:
        nChunks = 1
    # if the number of release cells is smaller/equal than/to the number of processes
    # each single release cell is assigned to a different process
    elif nRel <= nProcesses:
        nChunks = nRel
    # if there are more release cells than number of Processes available (this is the main case!)
    # then either divide release cells equally to the available processes - however limit the size of single chunks
    # to chunkSize if possible ...
    else:
        _nChunks = max(nProcesses, int(nRel / chunkSize))
        nChunks = min(_nChunks, maxChunks)

    return int(nProcesses), int(nChunks)


def handleMemoryAvailability(recheckInterval=30):
    """function is called at the start of each subProcess for parallel processing to check if enough memory is available
    and handle the situation if not

    NOTE: currently only time.sleep() is called to delay the subprocess for a defined time and then re-check
    memory availability.
    other possible options:
        - log message and abort model run?
    NOTE: The implementation with time.sleep() can cause an "infinite loop" in time-sleep if for some
    reason memory is not freed after a sensible amount of time.altzone
    Memory consumption is dependend on tile-sizes and number of Chunks/tile ...

    Parameters
    -----------
    recheckInterval: int
        delay time (in seconds) for the process after which memory availability is re-checked
    """
    while not enoughMemoryAvailable():
        time.sleep(recheckInterval)


def backTracking(topologyDict, infraValDict):
    """
    peform the back-tracking of infrastructure values across the dir-graph
    that is constructed if 'infra' option is set to 'True'

    Parameters
    -----------
    topologyDict : dict
        dictionary containing the topology of the modeled process path
        where parent nodes (colindex, rowindex) serve as keys and children of the
        respective parent node are stored as list items for the respective key
    infraValDict : dict
        dictionay containing information if a node is an infrastructure cell
        (value at key 'node' > 0) or not (value at key 'node' == 0)

    Returns
    -----------
    infraValDict : dict
        dictionary with updated values "back-tracked" from the infrastructure
        cells to the start-cell along the modeled path topology
    """
    # sort valDict (so we start traversing from highest infrastructure cells first)
    # this makes the algorithm more efficient
    valDictSorted = {k: v for k, v in sorted(infraValDict.items(), key=lambda item: item[1], reverse=True)}
    # reverse the graph topology, so "parents" become "children"
    reverseGraph = reverseTopology(topologyDict)

    # helper function to recursively traverse the reverseGraph and
    # propagate the infraValues "upslope"
    def propagateInfraVal(node, infraValToPropagate, visited):
        # if a node has been visited already --> no need to process again
        if node in visited:
            return
        # if the current propagation value is larger or equal to the one in
        # the processed node --> no need to look further
        elif (infraValDict.get(node, 0) >= infraValToPropagate) and (bool(visited)):
            return
        # in all other cases update the value of the current node and add node
        # to the set of visited nodes
        infraValDict[node] = max(infraValDict[node], infraValToPropagate)
        visited.add(node)

        for parentNode in reverseGraph.get(node, []):
            propagateInfraVal(parentNode, infraValToPropagate, visited)

    for node, val in valDictSorted.items():
        if val > 0:
            propagateInfraVal(node, val, set())

    return infraValDict


def reverseTopology(topologyDict):
    """
    reverse graph topology
    i.e. directions of graph edges connecting the nodes in the dir graph

    Parameters
    -----------
    topologyDict : dict
        dictionary containing the topology of the modeled process path
        where parent nodes (colindex, rowindex) serve as keys and children of the
        respective parent node are stored as list items for the respective key

    Returns
    -----------
    reverseGraph : dict
        dir-graph with reversed edges in same dictionary format as orignial input
    """
    reverseGraph = {}

    for parentNode, childNodes in topologyDict.items():
        childSet = set(childNodes)
        for child in childSet:
            if child not in reverseGraph:
                reverseGraph[child] = []
            reverseGraph[child].append(parentNode)

    return reverseGraph


def saveGenerationVideoData(genList, relId, dem, rasterAttributes,
                            outDir, variable="z_delta"):
    """
    Saves, per generation, both a cumulative raster snapshot (history) and
    a generation-only raster snapshot (just the cells of that generation),
    plus the center-of-flux (thalweg) position, for a given relId.
    Before saving, both the raster stacks and the row
    coordinates are flipped along the row axis (upside down), so the saved
    ``.npz`` data is already in the same up/down orientation as the
    other com4FlowPy output rasters.


    Parameters
    -----------
    genList: list
        list of cell lists (one list per generation), as built in calculation()
    relId: int
        release ID for which the data is saved
    dem: np.array
        DEM of the tile (used for the array shape)
    rasterAttributes: dict
        contains, among other things, "extentTile" for converting tile
        coordinates to full-DEM coordinates
    outDir: pathlib.Path
        target directory
    variable: str
        "z_delta" or "flux" (attribute name of the Cell class)

    Returns
    ---------
    outFile: pathlib.Path
        Path to the saved `video data file.
    """
    ((startY, _), (startX, _)) = rasterAttributes["extentTile"]
    nGen = len(genList)
    nRows = dem.shape[0]

    frameStackHistory = np.zeros((nGen, *dem.shape), dtype=np.float32)  # cumulative
    frameStackCurrent = np.zeros((nGen, *dem.shape), dtype=np.float32)  # this generation only
    rowCoF = np.zeros(nGen, dtype=np.float32)
    colCoF = np.zeros(nGen, dtype=np.float32)
    rowCoE = np.zeros(nGen, dtype=np.float32)
    colCoE = np.zeros(nGen, dtype=np.float32)

    runningMax = np.zeros_like(dem, dtype=np.float32)

    for gen, cellList in enumerate(genList):
        if len(cellList) == 0:
            frameStackHistory[gen] = runningMax
            # the thalweg location is not moved compared to the generation before
            rowCoF[gen] = rowCoF[gen - 1] if gen > 0 else np.nan
            colCoF[gen] = colCoF[gen - 1] if gen > 0 else np.nan
            rowCoE[gen] = rowCoE[gen - 1] if gen > 0 else np.nan
            colCoE[gen] = colCoE[gen - 1] if gen > 0 else np.nan
            continue

        rowArr = []
        colArr = []
        valArr = []
        fluxWeights = []
        energyWeights = []

        for cell in cellList:
            rowArr.append(cell.rowindex)
            colArr.append(cell.colindex)
            valArr.append(getattr(cell, variable))
            fluxWeights.append(cell.flux)
            energyWeights.append(cell.flowEnergy)

        rowArr = np.asarray(rowArr)
        colArr = np.asarray(colArr)
        valArr = np.asarray(valArr)
        fluxWeights = np.asarray(fluxWeights)
        energyWeights = np.asarray(energyWeights)

        # generation-only snapshot (not cumulative)
        currentArr = np.zeros_like(dem, dtype=np.float32)
        currentArr[rowArr, colArr] = valArr
        frameStackCurrent[gen] = currentArr

        # cumulative snapshot (history, including current generation)
        runningMax[rowArr, colArr] = np.maximum(runningMax[rowArr, colArr], valArr)
        frameStackHistory[gen] = runningMax

        # center of flux (weighted average position, weighted by flux)
        if fluxWeights.sum() > 0:
            rowCoF[gen] = np.average(rowArr, weights=fluxWeights)
            colCoF[gen] = np.average(colArr, weights=fluxWeights)
        else:
            rowCoF[gen] = np.average(rowArr)
            colCoF[gen] = np.average(colArr)

        # center of energy
        if energyWeights.sum() > 0:
            rowCoE[gen] = np.average(rowArr, weights=energyWeights)
            colCoE[gen] = np.average(colArr, weights=energyWeights)
        else:
            rowCoE[gen] = np.average(rowArr)
            colCoE[gen] = np.average(colArr)

    # --- flip everything from the simulation's internal (upside-down)
    # row orientation to the standard output orientation, BEFORE applying
    # the tile-to-full-DEM offset ---------------------------------------
    frameStackHistory = np.flip(frameStackHistory, axis=1)
    frameStackCurrent = np.flip(frameStackCurrent, axis=1)

    rowCoF = (nRows - 1) - rowCoF
    rowCoE = (nRows - 1) - rowCoE

    rowCoFFull = rowCoF + startY
    colCoFFull = colCoF + startX
    rowCoEFull = rowCoE + startY
    colCoEFull = colCoE + startX

    outDir = pathlib.Path(outDir)
    outDir.mkdir(parents=True, exist_ok=True)
    outFile = outDir / f"videoData_{variable}_{int(relId)}.npz"
    np.savez_compressed(outFile,
                        framesHistory=frameStackHistory,
                        framesCurrent=frameStackCurrent,
                        rowCoF=rowCoFFull, colCoF=colCoFFull,
                        rowCoE=rowCoEFull, colCoE=colCoEFull, )
    log.info(f"Video data for relId {int(relId)} saved: {outFile}")
    return outFile
