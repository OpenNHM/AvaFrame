#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Functions to handle the raster tiles.
 Tiling is intended to manage processing of large(r) computational domains.
"""

import logging
import pickle
import gc
import numpy as np
import avaframe.in2Trans.rasterUtils as IOf
import shapely
import geopandas as gpd

# create local logger
log = logging.getLogger(__name__)


def tileRaster(fNameIn, fNameOut, dirName, xDim, yDim, U, isInit=False):
    """
    divides a raster into tiles and saves the tiles

    Parameters
    -----------
    fNameIn : pathlib path
        path to raster that is tiled
    fNameOut: str
        name of saved raster file
    dirName: pathlib path
        path to folder, where tiled raster is saved (temp - folder)
    xDim: int
        size of one tile in x dimension (number of raster columns)
    yDim: int
        size of one tile in y dimension (number of raster rows)
    U: int
        size of tile overlapping (number of raster cells)
    isInit: bool
        if isInit is True, edges are assigned to -9999 (default: False)
    """

    # if not os.path.exists(dirName):
    #    os.makedirs(dirName)

    # largeRaster, largeHeader = iof.f_readASC(fNameIn, dType='float')
    largeData = IOf.readRaster(fNameIn, noDataToNan=False)
    largeRaster = largeData["rasterData"]
    # einlesen des Rasters und der Header

    i, j, imax, jmax = 0, 0, 0, 0
    sX, sY, eX, eY = 0, 0, 0, 0
    # starte mit den tiles in der NW-Ecke sX,eX = cols; sY,eY = rows;
    # cs=largeHeader['cellsize']
    # xllc = largeHeader['xllcorner']
    # yllc = largeHeader['yllcorner']

    nrows, ncols = largeRaster.shape[0], largeRaster.shape[1]
    pickle.dump((nrows, ncols), open(dirName / "extentLarge", "wb"))

    # print ncols, nrows

    I, J, IMAX, JMAX = 0, 0, 0, 0

    while eY < nrows:
        eY = sY + yDim
        while eX < ncols:
            eX = sX + xDim

            # rangeRowsCols = ((sY,eY),(sX,eX))
            # pickle.dump(rangeRowsCols, open("%s/ext_%i_%i"%(dirName,i,j),"wb"))

            # headerTile = {}
            # headerTile['ncols'] = eX-sX
            # headerTile['nrows'] = eY-sY
            # headerTile['xllcorner'] = xllc + sX*cs
            # headerTile['yllcorner'] = yllc + nrows*cs - eY*cs
            # headerTile['cellsize'] = cs
            # headerTile['noDataValue'] = largeHeader['noDataValue']

            # pickle.dump( headerTile, open( "temp/header%d_%d.p"%(i,j), "wb" ) )
            # np.save("%s/%s_%i_%i"%(dirName,fNameOut, i, j), largeRaster[sY:eY,sX:eX])
            # pickle.dump(, open( "temp/header_large.p"%(fNameOut, i,j), "wb" ) )
            # log.info("saved %s - TileNr.: %i_%i"%(fNameOut,i,j))

            sX = eX - 2 * U
            JMAX = max(J, JMAX)
            J += 1
        sX, J, eX = 0, 0, 0
        sY = eY - 2 * U
        IMAX = max(I, IMAX)
        I += 1

    sX, sY, eX, eY = 0, 0, 0, 0

    if isInit is False:
        while eY < nrows:
            eY = sY + yDim
            while eX < ncols:
                eX = sX + xDim
                rangeRowsCols = ((sY, eY), (sX, eX))
                pickle.dump(rangeRowsCols, open(dirName / ("ext_%i_%i" % (i, j)), "wb"))

                # headerTile = {}
                # headerTile['ncols'] = eX-sX
                # headerTile['nrows'] = eY-sY
                # headerTile['xllcorner'] = xllc + sX*cs
                # headerTile['yllcorner'] = yllc + nrows*cs - eY*cs
                # headerTile['cellsize'] = cs
                # headerTile['noDataValue'] = largeHeader['noDataValue']

                # pickle.dump( headerTile,
                # open( "temp/header%d_%d.p"%(i,j), "wb" ) )
                np.save(dirName / ("%s_%i_%i" % (fNameOut, i, j)), largeRaster[sY:eY, sX:eX])
                # pickle.dump(,
                # open( "temp/header_large.p"%(fNameOut, i,j), "wb" ) )
                log.info("saved %s - TileNr.: %i_%i", fNameOut, i, j)

                sX = eX - 2 * U
                jmax = max(j, jmax)
                j += 1
            sX, j, eX = 0, 0, 0
            sY = eY - 2 * U
            imax = max(i, imax)
            i += 1
    else:
        while eY < nrows:
            eY = sY + yDim
            while eX < ncols:
                eX = sX + xDim

                rangeRowsCols = ((sY, eY), (sX, eX))
                pickle.dump(rangeRowsCols, open(dirName / ("ext_%i_%i" % (i, j)), "wb"))

                # headerTile = {}
                # headerTile['ncols'] = eX-sX
                # headerTile['nrows'] = eY-sY
                # headerTile['xllcorner'] = xllc + sX*cs
                # headerTile['yllcorner'] = yllc + nrows*cs - eY*cs
                # headerTile['cellsize'] = cs
                # headerTile['noDataValue'] = largeHeader['noDataValue']
                # pickle.dump(headerTile,
                # open( "temp/hd_%s%_d_%d.p"%(fNameOut, i,j), "wb" ) )

                initRas = largeRaster[sY:eY, sX:eX].copy()
                # shapeX = np.shape(initRas)[1]
                # shapeY = np.shape(initRas)[0]
                if j != JMAX:
                    initRas[:, -U:] = -9999  # Rand im Osten
                if i != 0:
                    initRas[0:U, :] = -9999  # Rand im Norden
                if j != 0:
                    initRas[:, 0:U] = -9999  # Rand im Westen
                if i != IMAX:
                    initRas[-U:, :] = -9999  # Rand im Sueden

                # log.info("%i_%i"%(shapeX-U, shapeX))
                np.save(dirName / ("%s_%i_%i" % (fNameOut, i, j)), initRas)
                del initRas
                # pickle.dump(,
                # open( "temp/header_large.p"%(fNameOut, i,j), "wb" ) )
                log.info("saved %s - TileNr.: %i_%i", fNameOut, i, j)

                sX = eX - 2 * U
                jmax = max(j, jmax)
                j += 1
            sX, j, eX = 0, 0, 0
            sY = eY - 2 * U
            imax = max(i, imax)
            i += 1

    pickle.dump((imax, jmax), open(dirName / "nTiles", "wb"))
    log.info("finished tiling %s: nTiles=%s" % (fNameOut, (imax + 1) * (jmax + 1)))
    log.info("----------------------------")
    # del largeRaster, largeHeader
    del largeRaster
    gc.collect()
    # return largeRaster


def tileRasterWithIndices(fNameIn, fNameOut, dirName, exList, eyList, U, isInit=False):
    """
    divides a raster into tiles and saves the tiles
    the tile size is determined by the indices of the tile ends and can vary within the tiles.

    Parameters
    -----------
    fNameIn : str
        path to raster that is tiled
    fNameOut: str
        name of saved raster file
    dirName: str
        path to folder, where tiled raster is saved (temp - folder)
    exList: list
        contains end indices of tiles in x dimension (number of raster columns)
    eyList: list
        contains end indices of tiles in y dimension (number of raster columns)
    U: int
        size of tile overlapping (number of raster cells)
    isInit: bool
        if isInit is True, edges are assigned to -9999 (default: False)
    """

    log.info("tile rasters using ends!")

    # largeRaster, largeHeader = iof.f_readASC(fNameIn, dType='float')
    largeData = IOf.readRaster(fNameIn, noDataToNan=False)
    largeRaster = largeData["rasterData"]

    i, j, imax, jmax = 0, 0, 0, 0
    sX, sY, eX, eY = 0, 0, 0, 0

    JMAX = len(exList) - 1
    IMAX = len(eyList) - 1

    if isInit is False:
        for eY in eyList:
            for eX in exList:
                rangeRowsCols = ((sY, eY), (sX, eX))
                pickle.dump(rangeRowsCols, open(dirName / ("ext_%i_%i" % (i, j)), "wb"))

                np.save(dirName / ("%s_%i_%i" % (fNameOut, i, j)), largeRaster[sY:eY, sX:eX])
                log.info("saved %s - TileNr.: %i_%i", fNameOut, i, j)

                sX = eX - 2 * U
                jmax = max(j, jmax)
                j += 1
            sX, j, eX = 0, 0, 0
            sY = eY - 2 * U
            imax = max(i, imax)
            i += 1
    else:
        for eY in eyList:
            for eX in exList:
                rangeRowsCols = ((sY, eY), (sX, eX))
                pickle.dump(rangeRowsCols, open(dirName / ("ext_%i_%i" % (i, j)), "wb"))

                initRas = largeRaster[sY:eY, sX:eX].copy()
                if j != JMAX:
                    initRas[:, -U:] = -9999  # Rand im Osten
                if i != 0:
                    initRas[0:U, :] = -9999  # Rand im Norden
                if j != 0:
                    initRas[:, 0:U] = -9999  # Rand im Westen
                if i != IMAX:
                    initRas[-U:, :] = -9999  # Rand im Sueden

                np.save(dirName / ("%s_%i_%i" % (fNameOut, i, j)), initRas)
                del initRas
                log.info("saved %s - TileNr.: %i_%i", fNameOut, i, j)

                sX = eX - 2 * U
                jmax = max(j, jmax)
                j += 1
            sX, j, eX = 0, 0, 0
            sY = eY - 2 * U
            imax = max(i, imax)
            i += 1

    pickle.dump((imax, jmax), open(dirName / "nTiles", "wb"))
    log.info("finished tiling %s: nTiles=%s" % (fNameOut, (imax + 1) * (jmax + 1)))
    log.info("----------------------------")
    del largeRaster
    gc.collect()


def getTileEnds(dirName, xDim, yDim, U, relIdRaster):
    """
    computes the indices of tiling ends for a raster.
    The tiles can not split a release cells with the same release ID.

    Parameters
    -----------
    dirName: str
        path to folder, where tiled raster is saved (temp - folder)
    xDim: int
        minimum size of one tile in x dimension (number of raster columns)
    yDim: int
        minimum size of one tile in y dimension (number of raster rows)
    U: int
        size of tile overlapping (number of raster cells)

    Returns
    -------------
    exList: list
        contains end x - indices of the tiles
    eyList: list
        contains end y - indices of the tiles
    """
    log.info("get tile ends")
    i, j, imax, jmax = 0, 0, 0, 0
    sX, sY, eX, eY = 0, 0, 0, 0

    nrows, ncols = relIdRaster.shape[0], relIdRaster.shape[1]
    pickle.dump((nrows, ncols), open(dirName / "extentLarge", "wb"))

    I, J, IMAX, JMAX = 0, 0, 0, 0

    # compute the maximum or tiles in x and y direction are possible
    while eY < nrows:
        eY = sY + yDim
        while eX < ncols:
            eX = sX + xDim

            sX = eX - 2 * U
            JMAX = max(J, JMAX)
            J += 1
        sX, J, eX = 0, 0, 0
        sY = eY - 2 * U
        IMAX = max(I, IMAX)
        I += 1

    sX, sY, eX, eY = 0, 0, 0, 0
    exList = []
    eyList = []
    i, j, imax, jmax = 0, 0, 0, 0

    while eY < nrows:
        eY = sY + yDim
        shiftCountY = 0
        # iterate as long as necessary that the y - end of the tile does not cut a PRA
        while True:
            # TODO: loop stops if ey = nrows??
            if i == IMAX:
                # the border tile
                eyList.append(eY)
                break

            relIdSY = 0
            relIdEY = eY - U
            if j != 0:
                relIdSY = sY + U

            mask = np.zeros_like(relIdRaster)
            mask[relIdSY:relIdEY, :] = 1
            idsIn, idsOut = getMaskedRasters(mask, relIdRaster)

            if np.any(np.isin(idsIn, idsOut)):
                shiftCountY += 1
                eY += 1

                # relIdEY is relIdSY in the next iteration, so we only need to search for the end indices
            else:
                eyList.append(eY)
                break
        if shiftCountY > 0:
            log.info(
                f"tiling would divide a release area, tiling size (in y direction) changed by {shiftCountY} cells."
            )

        sY = eY - 2 * U
        imax = max(i, imax)
        i += 1

    while eX < ncols:
        eX = sX + xDim

        shiftCountX = 0

        while True:
            # iterate as long as necessary that the x - end of the tile does not cut a PRA
            if j == JMAX:
                # the border tile
                exList.append(eX)
                break
            relIdSX = 0
            relIdEX = eX - U
            if j != 0:
                relIdSX = sX + U

            mask = np.zeros_like(relIdRaster)
            mask[:, relIdSX:relIdEX] = 1
            idsIn, idsOut = getMaskedRasters(mask, relIdRaster)

            if np.any(np.isin(idsIn, idsOut)):
                shiftCountX += 1
                eX += 1
            else:
                exList.append(eX)
                break

        if shiftCountX > 0:
            log.info(
                f"tiling would divide a release area, tiling size (in x direction) changed by {shiftCountX} cells."
            )
        sX = eX - 2 * U
        jmax = max(j, jmax)
        j += 1

    pickle.dump((imax, jmax), open(dirName / "nTiles", "wb"))
    log.info("nTiles=%s" % ((imax + 1) * (jmax + 1)))
    log.info("----------------------------")
    gc.collect()
    return exList, eyList


def getMaskedRasters(mask, raster):
    """
    get unique values of a raster within and outside a mask

    Parameters
    -----------
    mask : numpy array
        mask that contains 1 (inside) and 0 (outside)
    raster : numpy array
        raster values

    Returns
    -------------
    idsIn: numpy array
        unique values of raster inside mask
    idsOut: numpy array
        unique values of raster outside mask

    """

    relIdInside = np.where(mask == 1, raster, -9999)
    relIdOutside = np.where(mask == 0, raster, -9999)

    idsIn = np.unique(relIdInside)
    idsOut = np.unique(relIdOutside)
    idsIn = idsIn[idsIn > 0]
    idsOut = idsOut[idsOut > 0]
    return idsIn, idsOut


def mergeRaster(inDirPath, fName, method="max"):
    """
    Merges the results for each tile to one array using the
    method provided through the function parameters

    Parameters
    ----------
    inDirPath: pathlib path
        Path to the temporary files, that are results for each tile
    fName : str
        file name of the parameter which should be merged from tile-results
    method : str
        method, how the tiles should be merged (default: max)
        method 'min' calculates the minimum of input raster tiles,
        if the minimum is < 0, then 0 is used
        method 'sum' calculates the sum of the raster tiles

    Returns
    -------
    mergedRas : numpy array
        merged raster
    """

    # os.chdir(inDirPath)

    extL = pickle.load(open(inDirPath / "extentLarge", "rb"))
    # print extL
    nTiles = pickle.load(open(inDirPath / "nTiles", "rb"))

    if method in ["max", "min"]:
        mergedRas = np.full((extL[0], extL[1]), np.nan)
    elif method == "sum":
        mergedRas = np.zeros((extL[0], extL[1]))
    else:
        message = f"Invalid function argument <method = '{method}'>. Supported values for 'method': ['min', 'max', 'sum']"
        log.error(message)
        raise ValueError(message)

    for i in range(nTiles[0] + 1):
        for j in range(nTiles[1] + 1):
            smallRas = np.load(inDirPath / ("%s_%i_%i.npy" % (fName, i, j)))
            # print smallRas
            pos = pickle.load(open(inDirPath / ("ext_%i_%i" % (i, j)), "rb"))
            # print pos

            if method == "max":
                mergedRas[pos[0][0] : pos[0][1], pos[1][0] : pos[1][1]] = np.fmax(
                    mergedRas[pos[0][0] : pos[0][1], pos[1][0] : pos[1][1]], smallRas
                )
            elif method == "min":
                mergedRas[pos[0][0] : pos[0][1], pos[1][0] : pos[1][1]] = np.where(
                    (mergedRas[pos[0][0] : pos[0][1], pos[1][0] : pos[1][1]] >= 0) & (smallRas >= 0),
                    np.fmin(
                        mergedRas[pos[0][0] : pos[0][1], pos[1][0] : pos[1][1]],
                        smallRas,
                    ),
                    np.fmax(
                        mergedRas[pos[0][0] : pos[0][1], pos[1][0] : pos[1][1]],
                        smallRas,
                    ),
                )
            if method == "sum":
                mergedRas[pos[0][0] : pos[0][1], pos[1][0] : pos[1][1]] = np.add(
                    mergedRas[pos[0][0] : pos[0][1], pos[1][0] : pos[1][1]], smallRas
                )
            del smallRas
            log.info("appended result %s_%i_%i", fName, i, j)
    return mergedRas


def mergeDict(inDirPath, fName):
    """
    Merges the dictionary-results for each tile to one dictionary

    Parameters
    ----------
    inDirPath: pathlib path
        Path to the temporary files, that are results for each tile
    fName : str
        file name of the parameter which should be merged from tile-results

    Returns
    -------
    mergedDict: dict
        contains all
    """
    nTiles = pickle.load(open(inDirPath / "nTiles", "rb"))
    mergedDict = {}

    for i in range(nTiles[0] + 1):
        for j in range(nTiles[1] + 1):
            pos = pickle.load(open(inDirPath / ("ext_%i_%i" % (i, j)), "rb"))
            with open(inDirPath / ("%s_%i_%i.pickle" % (fName, i, j)), "rb") as file:
                smallDict = pickle.load(file)
                if bool(smallDict):
                    for cellindSmall in smallDict:
                        cellind = (
                            cellindSmall[0] + pos[0][0],
                            cellindSmall[1] + pos[1][0],
                        )
                        if cellind in mergedDict:
                            mergedDict[cellind] = np.append(smallDict[cellindSmall], mergedDict[cellind])
                        else:
                            mergedDict[cellind] = smallDict[cellindSmall]
                        mergedDict[cellind] = np.unique(mergedDict[cellind])
                    log.info("appended result %s_%i_%i", fName, i, j)
    return mergedDict


def mergeDictToRaster(inDirPath, fName):
    """
    Merges the dictionary-results for each tile to one array using
    the length of the array assigned to each cell

    Parameters
    ----------
    inDirPath: str
        Path to the temporary files, that are results for each tile
    fName : str
        file name of the parameter which should be merged from tile-results

    Returns
    -------
    mergedRas : numpy array
        merged raster
    """
    extL = pickle.load(open(inDirPath / "extentLarge", "rb"))
    mergedRas = np.zeros((extL[0], extL[1]))

    mergedDict = mergeDict(inDirPath, fName)
    for cellind in mergedDict:
        mergedRas[cellind] = len(np.unique(mergedDict[cellind]))
    del mergedDict
    return mergedRas


def mergeDictToPolygon(inDirPath, fName, demHeader):
    """
    Merges the dictionary-results for each tile to polygons for every path per PRA ID

    Parameters
    ----------
    inDirPath: str
        Path to the temporary files, that are results for each tile
    fName : str
        file name of the parameter which should be merged from tile-results
    demHeader: dict
        header of DEM raster

    Returns
    -------
    gdfPathPolygons: GeoDataFrame
        polygons per path for every PRA ID
    """
    # get path polygons for every PRA ID
    mergedDict = mergeDict(inDirPath, fName)
    cellsize = demHeader["cellsize"]
    pathPolygons = {}
    buffer = 1e-10

    for (row, col), praIds in mergedDict.items():
        # get a polygon around every cell contained in mergedDict
        xmin = col * cellsize + demHeader["xllcenter"] - cellsize / 2
        ymin = row * cellsize + demHeader["yllcenter"] - cellsize / 2
        xmax = xmin + cellsize
        ymax = ymin + cellsize
        cellsPoly = shapely.geometry.box(xmin, ymin, xmax, ymax)

        # reorder the dictionary: keys: PRA ID, values: list of polygons around every cell
        for pid in praIds:
            if pid not in pathPolygons:
                pathPolygons[pid] = []
            pathPolygons[pid].append(cellsPoly)
    del mergedDict

    for pid, polys in pathPolygons.items():
        # merge all polygons belonging to a PRA ID
        unionPoly = shapely.union_all(polys)
        bufferedPoly = unionPoly.buffer(buffer, cap_style="flat").buffer(-buffer, cap_style="square")
        bufferedPoly = shapely.make_valid(bufferedPoly)
        cleanedPoly = shapely.set_precision(bufferedPoly, 1e-3)
        cleanedPoly = shapely.remove_repeated_points(cleanedPoly, tolerance=1e-3)
        pathPolygons[pid] = cleanedPoly

    gdfPathPolygons = gpd.GeoDataFrame(
        {"PRA_id": list(pathPolygons.keys())},
        geometry=list(pathPolygons.values()),
        crs=demHeader["crs"],
    )
    del pathPolygons
    return gdfPathPolygons
