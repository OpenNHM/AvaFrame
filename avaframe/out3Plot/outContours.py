"""
    Plotting and saving contour line plots

"""

import os
import logging
import math
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib
from matplotlib import cm
from cmcrameri import cm as cmapCrameri
import matplotlib as mpl

# Local imports
import avaframe.out3Plot.plotUtils as pU
import avaframe.in3Utils.fileHandlerUtils as fU
from avaframe.out3Plot import statsPlots as sPlot
from avaframe.in3Utils import cfgUtils
from avaframe.in2Trans import rasterUtils as iOf
from avaframe.in2Trans import shpConversion as shpConv
from avaframe.in3Utils import geoTrans as gT


# create local logger
log = logging.getLogger(__name__)


def readShpLines(shpData, layerName='name'):
    """ derive a dict with x, y coordinates for each contourline according to given contour lines
        dependent on contLine name format here used resType_unit_level (for example: pft_m_1.0)

        Parameters
        -----------
        shpData: dict
            shp file data extracted with SHP2Array
        layerName: str
            name that is used to name contourline and to identify lines in shpData

        Returns
        --------
        contourDict: dict
            contour line dictionary with name and in dict x, y coordinates

    """

    contourDict = {}

    for index, name in enumerate(shpData['layerName']):
        contLine = name.split('_')[2]
        contourDict[name] = {}
        startIndex = int(shpData['Start'][index])
        endIndex = int(startIndex + shpData['Length'][index])
        contourDict[name]['x'] = shpData['x'][startIndex:endIndex]
        contourDict[name]['y'] = shpData['y'][startIndex:endIndex]
        contourDict[name]['z'] = shpData['z'][startIndex:endIndex]
        contourDict[name]['contourLevel'] = float(contLine)

    return contourDict


def createRasterContourDict(inFile, levels):
    """ create a dict with contour line coordinates for an ascii file

        Parameters
        -----------
        inFile: pathlib path
            path to ascii file
        levels: list
            list of levels for contour lines

        Returns
        --------
        contourDict: dict
            dictionary with keys: name of contourlines (format: resType_unit_level)
                x: x coordinates
                y: y coordinates
                contourlevel: float of contour line value
    """

    contourDict = {}
    simData = iOf.readRaster(inFile)
    xGrid, yGrid,_ ,_ = gT.makeCoordGridFromHeader(simData['header'])
    for level in levels:
        contourDictXY = pU.fetchContourCoords(xGrid, yGrid, simData['rasterData'], level)
        contourDict['pft_m_' + str(level)] = contourDictXY
        contourDict['pft_m_' + str(level)]['contourLevel'] = level

    return contourDict


def plotContoursFromDict(contourDictRef, contourDictSim, pathDict, levels, multiplePlots=True):
    """ plot contour lines of two contourLine dicts only plot lines that are available within ref
        and save to file

        Parameters
        -----------
        contourDict: dict
            dictionary with contourline coordinates for reference
        contourDict: dict
            dictionary with contourline coordinates for simulation
        pathDict: dict
            dictionary with info on avaDir, outDir, title, parameter, unit
        levels: list
            list of contour levels
        multiplePlots: bool
            if True one plot for each contour level

    """

    # if more than one level initialize colors
    if len(levels) > 1:
        norm = mpl.colors.Normalize(vmin=min(levels), vmax=max(levels))
        cmap = mpl.cm.ScalarMappable(norm=norm, cmap=cmapCrameri.hawaii)
        cmap.set_array([])

    # setup figure if only one plot
    if multiplePlots is False:
        fig = plt.figure(figsize=(pU.figW, pU.figH))
        ax1 = fig.add_subplot(111)
        ax1.set_title(pathDict['title'])
        ax1.set_ylabel('x')
        ax1.set_xlabel('y')

    countLine = 0
    for name in contourDictRef:
        contLine = contourDictRef[name]['contourLevel']
        for nameSim in contourDictSim:
            if contLine == contourDictSim[nameSim]['contourLevel']:
                if multiplePlots:
                    fig = plt.figure(figsize=(pU.figW, pU.figH))
                    ax1 = fig.add_subplot(111)
                    ax1.set_title(pathDict['title'] + ' level: %.2f' % contLine)
                    ax1.set_ylabel('x')
                    ax1.set_xlabel('y')
                    ax1.plot(contourDictRef[name]['x'], contourDictRef[name]['y'], 'k.', label='ref')
                    for key in contourDictSim[nameSim]:
                        if '_0' in key and 'line' in key:
                            ax1.plot(contourDictSim[nameSim][key]['x'], contourDictSim[nameSim][key]['y'], 'r-', label='sim')
                        elif 'line' in key:
                            ax1.plot(contourDictSim[nameSim][key]['x'], contourDictSim[nameSim][key]['y'], 'r-')
                    ax1.legend()
                    pU.putAvaNameOnPlot(ax1, pathDict['avaDir'])
                    nameStr = name.replace('/', '-')
                    outFileName = 'contourLine_%s' % nameStr
                    pU.saveAndOrPlot(pathDict, outFileName, fig)
                else:
                    ax1.plot(contourDictRef[name]['x'], contourDictRef[name]['y'], 'b.')
                    ax1.plot(contourDictSim[nameSim]['x'], contourDictSim[nameSim]['y'], marker='.',
                        linestyle='', c=cmap.to_rgba(contLine))
                    ax1.annotate('-- ref', xy=(0.05, 0.95), xycoords='axes fraction')

                countLine = countLine + 1

    if multiplePlots is False:
        cbar = ax1.figure.colorbar(cmap)
        cbar.outline.set_visible(False)
        cbar.ax.set_title('(' + pathDict['unit'] + ')', pad=10)
        cbar.set_label(pathDict['parameter'])
        pU.putAvaNameOnPlot(ax1, pathDict['avaDir'])
        fU.makeADir(pathDict['pathResult'])
        outFileName = 'contourLines_allLevels'
        pU.saveAndOrPlot(pathDict, outFileName, fig)
