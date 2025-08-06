#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Calculation functions (raster level)
"""

import sys
import numpy as np
import logging
import os
import platform
import gc
import psutil
import time

if os.name == "nt":
    from multiprocessing.pool import Pool as Pool
elif platform.system() == "Darwin":
    from multiprocessing.pool import ThreadPool as Pool
else:
    from multiprocessing import Pool

from avaframe.com4FlowPy.flowClass import Cell


def get_start_idx(dem, release):
    """Sort Release Pixels by altitude and return the result as lists for the
    Rows and Columns, starting with the highest altitude

    Parameters
    -----------
    dem: numpy array
        Digital Elevation Model to gain information about altitude
    release: numpy array
        The release layer, release pixels need int value > 0

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
        for i in range(len(row_list)):
            altitude_list.append(dem[row_list[i], col_list[i]])
        altitude_list, row_list, col_list = list(zip(*sorted(zip(altitude_list, row_list, col_list), reverse=True)))
        # Sort this lists by altitude
    return row_list, col_list


def split_release(release, pieces):
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

    Parameters
    -----------
    release: np.array
        a binary 0|1 array with release pixels designated by '1'
    pieces:  int
        number of chunck in which the release layer should be split

    Returns
    -----------
    release_list:   list
        contains the tiles(arrays) [array0, array1, ..]
    """

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
    previewMode = optTuple[2]["previewMode"]

    # Temp-Dir (all input files are located here and results are written back in here)
    tempDir = optTuple[3]["tempDir"]

    # List of output layers
    outputs = optTuple[3]["outputFileList"]

    # raster-layer Attributes
    cellsize = float(optTuple[4]["cellsize"])
    nodata = float(optTuple[4]["nodata"])

    MPOptions = optTuple[6]  # CPU, Multiprocessing options ...

    dem = np.load(tempDir / ("dem_%s_%s.npy" % (optTuple[0], optTuple[1])))
    release = np.load(tempDir / ("init_%s_%s.npy" % (optTuple[0], optTuple[1])))
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
        if optTuple[2]["varUmaxType"].lower() == 'umax':
            varUmaxArray[varUmaxArray > 0] = varUmaxArray[varUmaxArray > 0] ** 2 / 2 / 9.81
        elif optTuple[2]["varUmaxType"].lower() != 'zdeltalim':
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

    varParams = {
        'varUmaxBool': varUmaxBool,
        'varUmaxArray': varUmaxArray,
        'varAlphaBool': varAlphaBool,
        'varAlphaArray': varAlphaArray,
        'varExponentBool': varExponentBool,
        'varExponentArray': varExponentArray,
    }

    # convert release areas to binary (0: no release areas, 1: release areas)
    # every positive value >0 is interpreted as release area
    release[release < 0] = 0
    release[release == nodata] = 0 # added in case nodata is non-negative
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

    release_list = split_release(release, nChunks)
    log.info("Multiprocessing starts, used Cores/Processes/Chunks: %i/%i/%i" % (MPOptions["nCPU"], nProcesses, nChunks))

    with Pool(processes=nProcesses) as pool:
        results = pool.map(
            calculation,
            [
                [  # TODO: write in dicts:
                    dem,
                    infra,
                    release_sub,
                    alpha,
                    exp,
                    flux_threshold,
                    max_z_delta,
                    nodata,
                    cellsize,
                    infraBool,
                    forestBool,
                    varParams,
                    fluxDistOldVersionBool,
                    previewMode,
                    forestArray,
                    forestParams,
                    outputs,
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
        if forestInteraction:
            forestIntList.append(res[12])

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
            forestIntArray = np.where((forestIntArray >= 0) & (forestIntList[i] >= 0),
                                    np.minimum(forestIntArray, forestIntList[i]),
                                    np.maximum(forestIntArray, forestIntList[i]))

    # Save Calculated tiles
    np.save(tempDir / ("res_z_delta_%s_%s" % (optTuple[0], optTuple[1])), zDeltaArray)
    np.save(tempDir / ("res_z_delta_sum_%s_%s" % (optTuple[0], optTuple[1])), zDeltaSumArray)
    np.save(tempDir / ("res_rout_flux_sum_%s_%s" % (optTuple[0], optTuple[1])), routFluxSumArray)
    np.save(tempDir / ("res_dep_flux_sum_%s_%s" % (optTuple[0], optTuple[1])), depFluxSumArray)
    np.save(tempDir / ("res_flux_%s_%s" % (optTuple[0], optTuple[1])), fluxArray)
    np.save(tempDir / ("res_count_%s_%s" % (optTuple[0], optTuple[1])), countArray)
    np.save(tempDir / ("res_fp_max_%s_%s" % (optTuple[0], optTuple[1])), fpTravelAngleMaxArray)
    np.save(tempDir / ("res_fp_min_%s_%s" % (optTuple[0], optTuple[1])), fpTravelAngleMinArray)
    np.save(tempDir / ("res_sl_%s_%s" % (optTuple[0], optTuple[1])), slTravelAngleArray)
    np.save(tempDir / ("res_travel_length_max_%s_%s" % (optTuple[0], optTuple[1])), travelLengthMaxArray)
    np.save(tempDir / ("res_travel_length_min_%s_%s" % (optTuple[0], optTuple[1])), travelLengthMinArray)
    if infraBool:
        np.save(tempDir / ("res_backcalc_%s_%s" % (optTuple[0], optTuple[1])), backcalc)
    if forestInteraction:
        np.save(tempDir / ("res_forestInt_%s_%s" % (optTuple[0], optTuple[1])), forestIntArray)


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
        - args[7] (float) - nodata values of rasters
        - args[8] (float) - cellsize of rasters
        - args[9] (bool) -  flag for calculation with/without infrastructure
        - args[10] (bool) - flag for calculation with/without forest
        - args[11] (dict) - contains flags and numpy arrays for variable input parameters (Alpha, exp, uMax)
        - args[12] (bool) - flag for computing flux distribution with old version
        - args[13] (bool) - flag for previewMode / fast Calculation

        - args[14] (numpy array) - contains forest information (None if forestBool=False)
        - args[15] (dict) - contains parameters for forest interaction models (None if forestBool=False)
        - args[16] (list) - output names

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
    nodata = args[7]
    cellsize = args[8]
    infraBool = args[9]
    forestBool = args[10]
    varUmaxBool = args[11]['varUmaxBool']
    varUmaxArray = args[11]['varUmaxArray']
    varAlphaBool = args[11]['varAlphaBool']
    varAlphaArray = args[11]['varAlphaArray']
    varExponentBool = args[11]['varExponentBool']
    varExponentArray = args[11]['varExponentArray']
    fluxDistOldVersionBool = args[12]
    previewMode = args[13]
    outputs = args[16]

    if forestBool:
        forestArray = args[14]
        forestParams = args[15]
        forestInteraction = forestParams["forestInteraction"]
    else:
        forestInteraction = False
        forestArray = None
        forestParams = None

    zDeltaArray = np.zeros_like(dem, dtype=np.float32)
    zDeltaSumArray = np.zeros_like(dem, dtype=np.float32)
    zDeltaPathList = []
    routFluxSumArray = np.zeros_like(dem, dtype=np.float32)
    depFluxSumArray = np.zeros_like(dem, dtype=np.float32)
    fluxArray = np.ones_like(dem, dtype=np.float32) * -9999
    countArray = np.zeros_like(dem, dtype=np.int32)

    fpTravelAngleMaxArray = np.ones_like(dem, dtype=np.float32) * -9999  # fp = Flow Path
    fpTravelAngleMinArray = np.ones_like(dem, dtype=np.float32) * -9999  # fp = Flow Path
    slTravelAngleArray = np.ones_like(dem, dtype=np.float32) * -9999  # sl = Straight Line

    travelLengthMinArray = np.ones_like(dem, dtype=np.float32) * -9999
    travelLengthMaxArray = np.ones_like(dem, dtype=np.float32) * -9999

    if infraBool:
        backcalc = np.ones_like(dem, dtype=np.int32) * -9999
    else:
        backcalc = None

    if infraBool:
        # initialize infrastructure array
        # TODO: check if this can be simplified
        infraArr = infra  # infrastructure array (input file)

    if forestInteraction:
        forestIntArray = np.ones_like(dem, dtype=np.float32) * -9999

    # Core
    # NOTE-TODO: row_list, col_list are tuples - rethink variable naming
    row_list, col_list = get_start_idx(dem, release)

    startcell_idx = 0
    while startcell_idx < len(row_list):

        if infraBool:
            # if infraBool - here we initialize a directed graph structure
            pathTopology = {} # topology of path as directed graph
            infraValues = {}   # values

        processedCells = {}  # dictionary of cells that have been processed already
        zDeltaPathArray = np.zeros_like(dem, dtype=np.float32)
        cell_list = []
        row_idx = row_list[startcell_idx]
        col_idx = col_list[startcell_idx]
        dem_ng = dem[row_idx - 1: row_idx + 2, col_idx - 1: col_idx + 2]  # neighbourhood DEM
        if varUmaxBool and varUmaxArray is not None:
            if varUmaxArray[row_idx, col_idx] > 0 and varUmaxArray[row_idx, col_idx] <= 8848:
                max_z_delta = varUmaxArray[row_idx, col_idx]
        if varAlphaBool and varAlphaArray is not None:
            if varAlphaArray[row_idx, col_idx] > 0 and varAlphaArray[row_idx, col_idx] <= 90:
                alpha = varAlphaArray[row_idx, col_idx]
        if varExponentBool and varExponentArray is not None:
            if varExponentArray[row_idx, col_idx] > 0:
                exp = varExponentArray[row_idx, col_idx]

        if (nodata in dem_ng) or np.size(dem_ng) < 9:
            startcell_idx += 1
            continue

        startcell = Cell(
            row_idx, col_idx,
            dem_ng, cellsize,
            1, 0, None,
            alpha, exp, flux_threshold, max_z_delta,
            startcell=True, fluxDistOldVersionBool=fluxDistOldVersionBool,
            FSI=forestArray[row_idx, col_idx] if isinstance(forestArray, np.ndarray) else None,
            forestParams=forestParams,
        )

        # dictionary of all the cells that have been processed and the number of times the cell has been visited
        processedCells[(startcell.rowindex, startcell.colindex)] = 1
        # list of flowClass.Cell() Objects that is contains the "path" for each release-cell
        cell_list.append(startcell)

        if infraBool:
            # adding start-cell as "root-node" to directed graph of the modeled process path
            updateInfraDirGraph(startcell.rowindex, startcell.colindex)

        for idx, cell in enumerate(cell_list):

            # calculate flux, z_delta from current cell (cell) to child-cells
            # lenght of row, col, flux, and z_delta vectors correspond to
            # number of child cells (successors) to currently processed cell
            row, col, flux, z_delta = cell.calc_distribution()

            if len(flux) > 0: #i.e. if there are child cells
                # Sort this lists by z_delta, to start with the highest cell
                z_delta, flux, row, col = list(zip(*sorted(zip(z_delta, flux, row, col), reverse=False)))

            if infraBool:
                # if the current cell is not already in the dir-graph, then we add it here
                updateInfraDirGraph(cell.rowindex, cell.colindex)                

            # check if child cells already exist
            for i in range(idx, len(cell_list)):
                k = 0
                while k < len(row):
                    if row[k] == cell_list[i].rowindex and col[k] == cell_list[i].colindex:
                        cell_list[i].add_os(flux[k])
                        cell_list[i].add_parent(cell)

                        if infraBool:
                            updateInfraDirGraph(row[k], col[k], cell.rowindex, cell.colindex)

                        if z_delta[k] > cell_list[i].z_delta:
                            cell_list[i].z_delta = z_delta[k]
                        row = np.delete(row, k)
                        col = np.delete(col, k)
                        flux = np.delete(flux, k)
                        z_delta = np.delete(z_delta, k)
                    else:
                        k += 1

            for k in range(len(row)):
                dem_ng = dem[row[k] - 1: row[k] + 2, col[k] - 1: col[k] + 2]  # neighbourhood DEM

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

                cell_list.append(Cell(
                            row[k], col[k],
                            dem_ng, cellsize,
                            flux[k], z_delta[k],
                            cell,
                            alpha, exp, flux_threshold, max_z_delta,
                            startcell, fluxDistOldVersionBool=fluxDistOldVersionBool,
                            FSI=forestArray[row[k], col[k]] if isinstance(forestArray, np.ndarray) else None,
                            forestParams=forestParams,
                                     ))

            zDeltaArray[cell.rowindex, cell.colindex] = max(zDeltaArray[cell.rowindex, cell.colindex], cell.z_delta)
            fluxArray[cell.rowindex, cell.colindex] = max(fluxArray[cell.rowindex, cell.colindex], cell.flux)
            routFluxSumArray[cell.rowindex, cell.colindex] += cell.flux
            depFluxSumArray[cell.rowindex, cell.colindex] += cell.fluxDep
            zDeltaPathArray[cell.rowindex, cell.colindex] = max(zDeltaPathArray[cell.rowindex, cell.colindex], cell.z_delta)
            if "fpTravelAngleMax" in outputs or "fpTravelAngle" in outputs:
                fpTravelAngleMaxArray[cell.rowindex, cell.colindex] = max(
                    fpTravelAngleMaxArray[cell.rowindex, cell.colindex], cell.max_gamma
                )
            if "fpTravelAngleMin" in outputs:
                if fpTravelAngleMinArray[cell.rowindex, cell.colindex] >= 0 and cell.max_gamma >= 0:
                    fpTravelAngleMinArray[cell.rowindex, cell.colindex] = min(
                        fpTravelAngleMinArray[cell.rowindex, cell.colindex], cell.max_gamma
                    )
                else:
                    fpTravelAngleMinArray[cell.rowindex, cell.colindex] = max(
                        fpTravelAngleMinArray[cell.rowindex, cell.colindex], cell.max_gamma
                    )
            slTravelAngleArray[cell.rowindex, cell.colindex] = max(slTravelAngleArray[cell.rowindex, cell.colindex],
                                                                   cell.sl_gamma)
            if "travelLengthMax" in outputs or "travelLength" in outputs:
                travelLengthMaxArray[cell.rowindex, cell.colindex] = max(
                    travelLengthMaxArray[cell.rowindex, cell.colindex], cell.min_distance
                )
            if "travelLengthMin" in outputs:
                if travelLengthMinArray[cell.rowindex, cell.colindex] >= 0 and cell.min_distance >= 0:
                    travelLengthMinArray[cell.rowindex, cell.colindex] = min(
                        travelLengthMinArray[cell.rowindex, cell.colindex], cell.min_distance
                    )
                else:
                    travelLengthMinArray[cell.rowindex, cell.colindex] = max(
                        travelLengthMinArray[cell.rowindex, cell.colindex], cell.min_distance
                    )
            if processedCells[(cell.rowindex, cell.colindex)] == 1:
                countArray[cell.rowindex, cell.colindex] += int(1)

            if forestInteraction:
                if forestIntArray[cell.rowindex, cell.colindex] >= 0 and cell.forestIntCount >= 0:
                    forestIntArray[cell.rowindex, cell.colindex] = min(forestIntArray[cell.rowindex, cell.colindex],
                                                                       cell.forestIntCount)
                else:
                    forestIntArray[cell.rowindex, cell.colindex] = max(forestIntArray[cell.rowindex, cell.colindex],
                                                                       cell.forestIntCount)

        if infraBool:
            # if 'infraBool' is True - i.e. calculation is performed with infrastructure information
            # then we perform the back-tracking of the stored directed graph (topology and node values)

            updatedInfraValues = backTracking(pathTopology, infraValues) # actual "back-tracking" for current process-path

            for key, val in updatedInfraValues.items():
                backcalc[key[0], key[1]] = max(backcalc[key[0], key[1]], val) # writing max-values to back-tracking array

            del pathTopology, infraValues, updatedInfraValues
            gc.collect()

        if previewMode:
            # if the 'previewMode' is On/'True', then we check here if the current modeled process zones already
            # includes other release Cells (i.e. if release cells are "hit from above")
            # if this is the case, then we exclude the affected release cell(s) from further processing and update
            # the row_list, col_list variables containing the release cells that should be processed
            release[zDeltaArray > 0] = 0
            row_list, col_list = get_start_idx(dem, release)

        zDeltaPathList.append(zDeltaPathArray)
        del cell_list, processedCells, zDeltaPathArray

        startcell_idx += 1

    for zDeltaPathArray in zDeltaPathList:
        zDeltaSumArray += zDeltaPathArray

    gc.collect()

    if forestInteraction:
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
            forestIntArray,
        )
    else:
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
    valDictSorted = {k: v for k, v in sorted(infraValDict.items(), key= lambda item: item[1], reverse=True)}
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
    '''
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
    '''
    reverseGraph = {}

    for parentNode, childNodes in topologyDict.items():
        childSet = set(childNodes)
        for child in childSet:
            if child not in reverseGraph:
                reverseGraph[child] = []
            reverseGraph[child].append(parentNode)

    return reverseGraph
