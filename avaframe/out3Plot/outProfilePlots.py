"""
Functions to plot 2D avalanche simulation results as well as comparison plots
between two datasets of identical shape.

"""

import matplotlib.pyplot as plt
import numpy as np
import pathlib
import logging
from cmcrameri import cm as cmapCrameri
import matplotlib as mpl
import geopandas as gpd

# Local imports
import avaframe.in3Utils.geoTrans as gT
from avaframe.in3Utils import fileHandlerUtils as fU
import avaframe.out3Plot.plotUtils as pU
import avaframe.in1Data.getInput as gI
import avaframe.in2Trans.transformFields as tF
import avaframe.in2Trans.rasterUtils as IOf

# create local logger
# change log level in calling module to DEBUG to see log messages
log = logging.getLogger(__name__)


def plotAlongProfile(avaDir, peakFilesDF, resType='pft', cInd='names'):
    """ create a four panel plot with pft values along a chosen profile and also on top of DEM
        for thickness and depth values
        peak files read from avaDir/Outputs/modName/peakFiles, unless inputDir is provided

        Parameters
        -------------
        avaDir: pathlib path or str
            path to avalanche directory
        peakFilesDF: pandas dataFrame
            dataFrame with one line per result file
        resType: str
            name of result variable
        cInd: str
            name of column in resultFiles dataframe to be used in legend and for colorcoding

    """

    # load DEM
    demDict = gI.readDEM(avaDir)

    # read profile
    profileDir = pathlib.Path(avaDir, 'Inputs', 'LINES')
    profileFileList = list(profileDir.glob("*plotProfile.shp"))
    if profileFileList == []:
        message = 'No *plotProfile.shp file found'
        log.error(message)
        raise FileNotFoundError(message)

    # read profile info
    profileDF = gpd.read_file(profileFileList[0])
    profileLine = {'x': profileDF.get_coordinates()['x'].to_numpy(), 'y': profileDF.get_coordinates()['y'].to_numpy()}

    # resample line using cell size of DEM info
    profileLine, _ = gT.prepareLine(demDict, profileLine, demDict['header']['cellsize'], Point=None, k=1, s=None)

    # get elevation of DEM along profile
    demAlitudePoints, _ = gT.projectOnRaster(demDict, profileLine, interp="bilinear", inData="rasterData",
                                             outData="z")

    # only use result files for resType
    pftPeakDF = peakFilesDF[peakFilesDF['resType'] == resType]

    # setup figure to show DEM and location of profile
    fig1, ax1 = plt.subplots(ncols=1, nrows=1, figsize=(pU.figW, pU.figH), layout='constrained')

    # add DEM hillshade with contour lines
    extentCellCenters, extentCellCorners = pU.createExtentMinMax(demDict['rasterData'], demDict['header'], originLLCenter=True)
    _, _ = pU.addHillShadeContours(ax1, demDict['rasterData'], demDict['header']['cellsize'], extentCellCenters)

    # add profile line
    ax1.plot(profileLine['x'], profileLine['y'], color='orange', linestyle='-', zorder=10, label='profile line')
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.legend()

    # initialize profile figure
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(pU.figW * 2, pU.figH * 1.5), layout='constrained')
    # fetch colormap for all result files
    colorDict, markerDict = fetchCategoricalColors(varParStrValues=pftPeakDF[cInd].to_list(), style='line')

    # loop over all result files
    for index, row in pftPeakDF.iterrows():

        # load pft field
        peakRaster = IOf.readRaster(row["files"])
        # get pft values along transect
        peakThicknessPoints, _ = gT.projectOnRaster(peakRaster, profileLine, interp="bilinear", inData="rasterData", outData="pft")

        # convert thickness to depth
        depthDict, thicknessRasterNew, slopeAngleField = tF.convertDepthThickness(peakRaster, demDict, typeOfInput='thickness')
        # get depth values along profile
        peakDepthPoints, _ = gT.projectOnRaster(depthDict, profileLine, interp="bilinear", inData="rasterData",
                                           outData="pfd")

        # plot profiles of thickness and depth
        ax[1, 0].plot(profileLine['s'], peakThicknessPoints['pft'], color=colorDict[row[cInd]],
                      linestyle=markerDict[row[cInd]], label=row[cInd], zorder=10)
        ax[0, 0].plot(profileLine['s'], peakDepthPoints['pfd'], color=colorDict[row[cInd]],
                      linestyle=markerDict[row[cInd]], label=[row[cInd]], zorder=10)

        # plot thickness and depth values on top of DEM profile
        ax[0, 1].plot(profileLine['s'], demAlitudePoints['z'] + peakDepthPoints['pfd'], color=colorDict[row[cInd]],
                      linestyle=markerDict[row[cInd]], zorder=10)
        ax[1, 1].plot(profileLine['s'], demAlitudePoints['z'] + peakThicknessPoints['pft'], color=colorDict[row[cInd]],
                      linestyle=markerDict[row[cInd]], zorder=10)

    # add DEM elevation profile
    ax[1, 1].plot(profileLine['s'], demAlitudePoints['z'], color='k', linestyle='-', zorder=11, linewidth=1.4)
    ax[0, 1].plot(profileLine['s'], demAlitudePoints['z'], color='k', linestyle='-', linewidth=1.4, zorder=11,
                  label='dem elevation\nalong given profile')

    # add labels
    ax[1,0].set_xlabel('distance along given profile (m)')
    ax[1, 1].set_xlabel('distance along given profile (m)')
    ax[0, 0].set_ylabel('peak flow depth (pfd) (m)')
    ax[1, 0].set_ylabel('peak flow thickness (pft) (m)')
    ax[1, 1].set_ylabel('DEM elevation (+ pft) (m)')
    ax[0, 1].set_ylabel('DEM elevation (+ pfd) (m)')

    # add title to panels
    ax[0, 0].set_title('pfd along given profile (m)', fontsize=11)
    ax[1, 0].set_title('pft along given profile (m)', fontsize=11)
    ax[0, 1].set_title('pfd on top of DEM for given profile (m)', fontsize=11)
    ax[1, 1].set_title('pft on top of DEM for given profile (m)', fontsize=11)

    # add legend
    ax[0,0].legend()
    ax[0,1].legend()

    # save and or plot
    outDir = pathlib.Path(avaDir, 'Outputs', 'out3Plot', 'profilePlots')
    fU.makeADir(outDir)
    plotName1 = 'profileLocation'
    pU.saveAndOrPlot({"pathResult": outDir}, plotName1, fig1)
    plotName2 = 'pftvspfdAlonProfileDEM'
    pU.saveAndOrPlot({"pathResult": outDir}, plotName2, fig)


def fetchCategoricalColors(varParStrValues, style='line'):
    """ create dicts with colors and line-/ markerstyles for a list of different inputs
        uses two different colormaps to make individual inputs more distinguishable in the plots

        Parameters
        -----------
        varParStrValues: list
            list of input parameters
        style: str
            type of style: marker or line

        Returns
        ---------
        colordict: dict
            dictionary with key: input parameter name and item: color
        lineDict: dict
            dictionary with key: input parameter name and item: marker/linestyle

    """

    norm3 = mpl.colors.Normalize(vmin=0, vmax=1)

    # setup colormap
    cmapUsed = cmapCrameri.hawaii
    cmapValues3 = np.linspace(0, 1, len(varParStrValues))
    # create cmap object
    cmap3 = mpl.cm.ScalarMappable(norm=norm3, cmap=cmapUsed)
    cmap3.set_array([])
    # add a second cmap to get varying values for lines
    cmap34 = mpl.cm.ScalarMappable(norm=norm3, cmap=cmapCrameri.lapaz)
    cmap34.set_array([])

    countInd2 = 0
    cmapDict = {}
    lineDict = {}
    for indV, varPar1 in enumerate(varParStrValues):
        if (indV) % 2:
            linestyle = '-'
            cmapVal = cmapValues3[varParStrValues.index(varPar1)]
            cmapVal0 = cmap34.to_rgba(cmapVal)
            markerstyle = '+'
        else:
            linestyle = '-'
            cmapVal = cmapValues3[varParStrValues.index(varPar1)]
            cmapVal0 = cmap3.to_rgba(cmapVal)
            countInd2 = countInd2 + 1
            markerstyle = '+'
        if countInd2 % 2:
            linestyle = 'dashdot'
            markerstyle = 'x'
        cmapDict[varPar1] = cmapVal0
        if style == 'line':
            lineDict[varPar1] = linestyle
        else:
            lineDict[varPar1] = markerstyle

    return cmapDict, lineDict