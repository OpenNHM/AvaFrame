"""
    Plotting and saving AIMEC results

    This file is part of Avaframe.
"""

import os
import logging
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib
from matplotlib import cm
from cmcrameri import cm as cmapCrameri
import matplotlib as mpl
import matplotlib.patheffects as pe
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import pathlib

# Local imports
import avaframe.out3Plot.plotUtils as pU
from avaframe.out3Plot import statsPlots as sPlot
from avaframe.in3Utils import cfgUtils

# create local logger
log = logging.getLogger(__name__)


def visuTransfo(rasterTransfo, inputData, cfgSetup, pathDict):
    """
    Plot and save the domain transformation figure
    The left subplot shows the reference result raster with the outline of the
    new domain. The second one shows this same data in the (s,l) coordinate
    system define by the outline in the first plot.

    Parameters
    ----------
    rasterTransfo: dict
        domain transformation information
    inputData : dict
        inputData dictionary:
            slRaster: numpy array with (s,l) raster
            xyRaster: numpy array with (x,y) raster
            headerXY: header corresponding to xyRaster
    cfgSetup : configparser
        configparser with ana3AIMEC settings defined in ana3AIMECCfg.ini
        'runoutResType'
    pathDict : dict
        dictionary with path to data to analyze

    """
    ####################################
    # Get input data
    runoutResType = cfgSetup['runoutResType']
    displayRunoutResType = pathDict.get('displayRunoutResType', runoutResType)
    unit = pU.cfgPlotUtils['unit' + runoutResType]
    # read paths
    projectName = pathDict['projectName']
    # read rasterdata
    slRaster = inputData['slRaster']
    xyRaster = inputData['xyRaster']
    headerXY = inputData['xyHeader']
    cellsize = headerXY['cellsize']
    n = headerXY['nrows']
    m = headerXY['ncols']
    xllc = headerXY['xllcenter']
    yllc = headerXY['yllcenter']
    # read avaPath with scale
    xPath = rasterTransfo['x']
    yPath = rasterTransfo['y']
    # read domain boundarries with scale
    cellSizeSL = rasterTransfo['cellSizeSL']
    DBXl = rasterTransfo['DBXl']*cellSizeSL
    DBXr = rasterTransfo['DBXr']*cellSizeSL
    DBYl = rasterTransfo['DBYl']*cellSizeSL
    DBYr = rasterTransfo['DBYr']*cellSizeSL

    ############################################
    # prepare for plot
    x = np.arange(m)*cellsize + xllc
    y = np.arange(n)*cellsize + yllc
    indStartOfRunout = rasterTransfo['indStartOfRunout']
    xx = rasterTransfo['x'][indStartOfRunout]
    yy = rasterTransfo['y'][indStartOfRunout]

    l = rasterTransfo['l']
    s = rasterTransfo['s']

    maskedArray = np.ma.masked_where(xyRaster == 0, xyRaster)
    maskedArraySL = np.ma.masked_where(slRaster == 0, slRaster)

    cmap, _, ticks, norm = pU.makeColorMap(pU.colorMaps[runoutResType], np.nanmin(
        maskedArray), np.nanmax(maskedArray), continuous=pU.contCmap)
    cmap.set_under(color='w')

    labelRunout = rasterTransfo['labelRunout']

    ############################################
    # Figure: Raster transformation
    fig = plt.figure(figsize=(pU.figW*3, pU.figH))
    gs = gridspec.GridSpec(1, 3)
    ax1 = fig.add_subplot(gs[0, 0])

    ref0, im = pU.NonUnifIm(ax1, x, y, maskedArray, 'x [m]', 'y [m]',
                            extent=[x.min(), x.max(), y.min(), y.max()],
                            cmap=cmap, norm=norm)
    plt.plot(xx, yy, 'r*', markersize=8, label=labelRunout, zorder=10)
    plt.plot(xPath, yPath, 'k--', label='flow path')
    plt.plot(DBXl, DBYl, 'k-', label='domain')
    plt.plot(DBXr, DBYr, 'k-')
    plt.plot([DBXl, DBXr], [DBYl, DBYr], 'k-')
    if cfgSetup.getboolean('defineRunoutArea'):
        plt.plot(rasterTransfo['projSplitPoint']['x'], rasterTransfo['projSplitPoint']['y'], 'k*', label='split point')

    ax1.set_title('XY Domain')
    ax1.legend(loc='best')
    ax1.set_aspect('equal')
    pU.putAvaNameOnPlot(ax1, pathDict['projectName'])

    ax2 = fig.add_subplot(gs[0, 1:])

    maskedArraySLTransposed = np.transpose(maskedArraySL)

    ref0, im = pU.NonUnifIm(ax2, s, l, maskedArraySLTransposed, '$S_{XY}$ (thalweg) [m]', '$L_{XY}$ (thalweg) [m]',
                            extent=[s.min(), s.max(), l.min(), l.max()],
                            cmap=cmap, norm=norm)
    ax2.axvline(x=s[indStartOfRunout], color='k', linestyle='--',
                label=labelRunout)

    ax2.set_title('SL Domain' + '\n' + 'Black = out of raster')
    ax2.legend(loc=4)
    if not np.any(xyRaster):
        ax2.text(
            0.5,
            0.5,
            "reference only 0 values for %s" % displayRunoutResType,
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax2.transAxes,
        )
    else:
        pU.addColorBar(im, ax2, ticks, unit, title=displayRunoutResType)

    outFileName = '_'.join([projectName, 'DomainTransformation'])
    pU.saveAndOrPlot(pathDict, outFileName, fig)


def visuRunoutComp(rasterTransfo, resAnalysisDF, cfgSetup, pathDict):
    """
    Plot and save the Peak Fields distribution (max mean per cross section)
    after coordinate transformation

    Parameters
    ----------
    rasterTransfo: dict
        domain transformation information
    resAnalysis: dict
        results from Aimec analysis (for ppr, pft and pfv):
            PPRCrossMax: numpy array with max peak field along path for each file to analyse
            PPRCrossMean: numpy array with mean peak field along path for each file to analyse
    cfgSetup : configparser
        configparser with ana3AIMEC settings defined in ana3AIMECCfg.ini
        'runoutResType' and 'thresholdValue'
    pathDict : dict
        dictionary with path to data to analyze
    """
    ####################################
    # Get input data
    runoutResType = cfgSetup['runoutResType']
    thresholdValue = cfgSetup['thresholdValue']
    # read paths
    projectName = pathDict['projectName']
    refSimRowHash = pathDict['refSimRowHash']
    simRowHash = pathDict['simRowHash']
    # read data
    s = rasterTransfo['s']
    l = rasterTransfo['l']

    ############################################
    # prepare for plot
    title = ['Pressure ', 'Flow Thickness ', 'Flow Velocity ']
    unit = ['$ppr_{CrossMax}$ [kPa]', '$pft_{CrossMax}$ [m]', '$pfv_{CrossMax}$ $[ms^{-1}]$']
    peakList = ['ppr', 'pft', 'pfv']

    ############################################
    # Figure: Pressure thickness speed

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(pU.figW, pU.figH*3))
    fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, hspace=0.3)

    for ax, peak, titleVal, unitVal in zip(axes.flatten(), peakList, title, unit):
        ax.plot(s, resAnalysisDF.loc[refSimRowHash, peak + 'CrossMax'], '--k', label='Max Reference')
        ax.plot(s, resAnalysisDF.loc[refSimRowHash, peak + 'CrossMean'], '-k', label='Mean Reference')
        ax.plot(s, resAnalysisDF.loc[simRowHash, peak + 'CrossMax'], '--b', label='Max Simulation')
        ax.plot(s, resAnalysisDF.loc[simRowHash, peak + 'CrossMean'], '-b', label='Mean Simulation')

        ax.set_title(titleVal + 'distribution along thalweg')
        ax.legend(loc='best')
        ax.set_xlabel('$S_{XY}$ (thalweg) [m]')
        ax.set_xlim([s.min(), s.max()])
        ax.set_ylim(auto=True)
        ax.set_ylabel(unitVal)
    pU.putAvaNameOnPlot(ax, projectName)

    outFileName = '_'.join([projectName, runoutResType, str(thresholdValue).replace('.', 'p'), 'slComparison'])
    outFilePath = pU.saveAndOrPlot(pathDict, outFileName, fig)

    return outFilePath


def visuRunoutStat(rasterTransfo, inputsDF, resAnalysisDF, newRasters, cfgSetup, pathDict):
    """
    Panel1 reference peak field with runout points of all sims and distribution of runout SXY
    Panel 2 crossMax values of peak field along thalweg for all sims
    Panel 3 mean, median and envelope of cross max values for all sims

    option to colorcode according to first item of varParList
    if str there is also the option to use a categorical colormap - then colors are indicated in legend not as colorbar


    Parameters
    ----------
    rasterTransfo: dict
        domain transformation information
    inputsDF: dataFrame
        aimec inputs DF
    resAnalysis: dataFrame
        results from Aimec analysis:
            numpy array with max peak field (of the 'runoutResType') along path for each file to analyse
            runout for 'runoutResType'
    newRasters: dict
        dictionary with new (s, l) raster for the 'runoutResType'
    cfgSetup : configparser
        configparser with ana3AIMEC settings defined in ana3AIMECCfg.ini
        'percentile', 'runoutResType' and 'thresholdValue'
    pathDict : dict
        dictionary with path to data to analyze
    """

    ####################################
    # Get input data
    varParList = cfgSetup['varParList'].split('|')
    paraVar = varParList[0]
    firstVar = defineVariableType(paraVar, resAnalysisDF, pathDict)

    percentile = cfgSetup.getfloat('percentile')
    runoutResType = cfgSetup['runoutResType']
    # layer-resolved name for data access (e.g. ppr_l2); falls back to base name for single-layer
    resolvedRunoutResType = pathDict.get('runoutResType', runoutResType)
    displayRunoutResType = pathDict.get('displayRunoutResType', runoutResType)
    thresholdValue = cfgSetup['thresholdValue']
    unit = pU.cfgPlotUtils['unit' + runoutResType]
    name = pU.cfgPlotUtils['name' + runoutResType]
    # read paths
    projectName = pathDict['projectName']
    # read data
    s = rasterTransfo['s']
    l = rasterTransfo['l']
    indStartOfRunout = rasterTransfo['indStartOfRunout']
    rasterdataPres = newRasters['newRefRaster' + resolvedRunoutResType.upper()]
    runout = resAnalysisDF['sRunout'].to_numpy()
    crossValue = 'Cross' + cfgSetup['runoutCrossType'].upper()[0] + cfgSetup['runoutCrossType'].lower()[1:]
    pprCrossMax = np.stack(resAnalysisDF[resolvedRunoutResType.lower() + crossValue].to_numpy())
    ############################################
    # compute mean, median and percenti. of peak field cross max values and mask array with threshold
    pMean = np.mean(pprCrossMax, axis=0)
    pMedian = np.median(pprCrossMax, axis=0)
    pPercentile = np.percentile(pprCrossMax, [percentile/2, 50, 100-percentile/2], axis=0)
    maskedArray = np.ma.masked_where(rasterdataPres <= float(thresholdValue), rasterdataPres)

    # transpose array for plot
    maskedArrayTransposed = np.transpose(maskedArray)

    if np.all(np.isnan(runout)):
        log.warning(
            "No %s values exceeding threshold of %.2f found to analyze"
            % (displayRunoutResType, float(thresholdValue))
        )
        outFilePath = ""
    else:

        # get plots limits
        indXMin = max(0, indStartOfRunout - 5)
        xMin = s[indXMin]
        xMax = np.nanmax(runout) + 25
        # check if for reference array runout point has been found, do not constrain data
        if len(np.nonzero(np.any(maskedArray[indStartOfRunout:, :] > 0, axis=0))[0]) == 0:
            indYMin = 0
        else:
            indYMin = max(
                0, np.min(np.nonzero(np.any(maskedArray[indStartOfRunout:, :] > 0, axis=0))[0]) - 5
            )
        yMin = l[indYMin]
        if len(np.nonzero(np.any(maskedArray[indStartOfRunout:, :] > 0, axis=0))[0]) == 0:
            indYMax = maskedArray.shape[1] - 1
        else:
            indYMax = min(
                np.max(np.nonzero(np.any(maskedArray[indStartOfRunout:, :] > 0, axis=0))[0]) + 5, len(l) - 1
            )
        yMax = l[indYMax]

        # get colormap for raster plot of peak field
        cmap, _, ticks, norm = pU.makeColorMap(
            pU.colorMaps[runoutResType],
            np.nanmin(maskedArrayTransposed[indYMin:indYMax, indXMin:]),
            np.nanmax(maskedArrayTransposed[indYMin:indYMax, indXMin:]),
            continuous=pU.contCmap,
        )
        cmap.set_bad("w", 1.0)

        # Get colors for colorcoding runout points and crossMax values along thalweg for all sims
        unitSC = cfgSetup["unit"]
        nSamples = np.size(runout)
        colorFlag = False
        colorSuffix = ""
        if "colorParameter" in pathDict:
            if pathDict["colorParameter"] is False:
                colorFlag = False
            elif isinstance(firstVar, str):
                # if str then check for parameter values and create colormap that varies between 0, 1 with number of unique
                # values as steps
                values = inputsDF[varParList[0]].to_list()
                minVal = 0
                maxVal = 1
                colorFlag = True
                cmapSCVals = np.linspace(0, 1, nSamples)
            else:
                colorFlag, colorSuffix, minVal, maxVal, values, cmapSCVals = defineColorcodingValues(
                    inputsDF, paraVar, nSamples
                )

        if not colorFlag:
            values = None
            minVal = 0
            maxVal = 1

        # create colormap and setup ticks and itemsList
        cmapSC, colorSC, ticksSC, normSC, unitSC, itemsList, displayColorBar = pU.getColors4Scatter(
            values, nSamples, unitSC
        )

        # if parameter type for colorcoding is str and categorical colormap is chosen
        if cfgSetup.getboolean("varParCategorical"):
            norm3 = mpl.colors.Normalize(vmin=minVal, vmax=maxVal)
            # map string varParList value to 0, 1 colorbar values
            if isinstance(firstVar, str):
                # fetch amount of different str parameter values
                varParStrValues = list(set(resAnalysisDF[varParList[0]].tolist()))
                # setup colormap
                cmapUsed = cmapCrameri.glasgowS
                cmapValues3 = np.linspace(0, 1, len(varParStrValues))
                # create cmap object
                cmap3 = mpl.cm.ScalarMappable(norm=norm3, cmap=cmapUsed)
                cmap3.set_array([])
            else:
                message = "Categorical color scheme only valid if ordering parameter of type str"
                log.error(message)
                raise AttributeError(message)

        ############################################
        # Figure: Analysis runout
        fig, (ax1, ax3, ax2) = plt.subplots(nrows=3, ncols=1, figsize=(pU.figW * 2, pU.figH * 3.5))

        ax1.axvline(x=np.max(runout), color="k", linestyle="-.", label="runout max %.0f m" % np.max(runout))
        ax1.axvline(
            x=np.average(runout), color="k", linestyle="-", label="runout mean %.0f m" % np.mean(runout)
        )
        ax1.axvline(x=np.min(runout), color="k", linestyle=":", label="runout min %.0f m" % np.min(runout))

        ax1.axvline(x=s[indStartOfRunout], color="k", linestyle="--", label=rasterTransfo["labelRunout"])
        # label=('start of runout area: '+ r'$\beta_{%.1f °}$' % (rasterTransfo['startOfRunoutAreaAngle'])))
        ref5, im = pU.NonUnifIm(
            ax1,
            s,
            l,
            maskedArrayTransposed,
            "$S_{XY}$ (thalweg) [m]",
            "$L_{XY}$ (thalweg) [m]",
            extent=[xMin, xMax, yMin, yMax],
            cmap=cmap,
            norm=norm,
        )

        if cfgSetup.getboolean("varParCategorical"):
            for simRowHash, resAnalysisRow in resAnalysisDF.iterrows():
                cmapVal1 = cmapValues3[varParStrValues.index(resAnalysisRow[varParList[0]])]
                sc = ax1.plot(
                    resAnalysisRow["sRunout"],
                    resAnalysisRow["lRunout"],
                    marker="o",
                    c=cmap3.to_rgba(cmapVal1),
                    label=resAnalysisRow[varParList[0]],
                )
            # add legend for categorical values and move outside of panel
            ax1.legend(loc="center left", bbox_to_anchor=(1.2, 0.5))
        else:
            sc = ax1.scatter(
                resAnalysisDF["sRunout"],
                resAnalysisDF["lRunout"],
                c=colorSC,
                cmap=cmapSC,
                norm=normSC,
                marker=pU.markers[0],
                label=(
                    "runout points (%s<%.1f%s)" % (displayRunoutResType, cfgSetup.getfloat("thresholdValue"), unit)
                ),
            )
            if displayColorBar:
                pU.addColorBar(sc, ax1, ticksSC, unitSC, title=paraVar, pad=0.08, tickLabelsList=itemsList)
            ax1.legend(loc="upper left")

        # set panel axis labels
        ax1.set_ylim([yMin, yMax])
        ax1.set_xlim([xMin, xMax])
        ax1.set_title(
            "%s field (reference) for (%s>%.1f%s)"
            % (name, displayRunoutResType, cfgSetup.getfloat("thresholdValue"), unit)
        )
        ax1.set_aspect("equal")
        pU.putAvaNameOnPlot(ax1, projectName)

        # add colorbar for peak field
        if not np.any(rasterdataPres):
            ax1.text(
                0.5,
                0.5,
                "reference only 0 values for %s" % displayRunoutResType,
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax1.transAxes,
            )
        else:
            pU.addColorBar(im, ax1, ticks, unit)

        # add third panel for statistical measures of distribution of cross max values
        ax2.fill_between(
            s,
            pPercentile[2],
            pPercentile[0],
            facecolor=[0.8, 0.8, 0.8],
            alpha=0.5,
            label=("[%.2f, %.2f]%% interval" % (percentile / 2, 100 - percentile / 2)),
        )
        matplotlib.patches.Patch(alpha=0.5, color=[0.8, 0.8, 0.8])
        ax2.plot(s, pMedian, color="r", label="median")
        ax2.plot(s, pMean, color="b", label="mean")

        ax2.set_title("%s distribution along thalweg for all sims" % name)
        ax2.legend(loc="upper right")
        ax2.set_xlabel("$S_{xy}$ (thalweg) [m]")
        ax2.set_xlim([s.min(), s.max()])
        ax2.set_ylim(auto=True)
        ax2.set_ylabel("$%s_{%s}$ [%s]" % (displayRunoutResType, crossValue, unit))

        # add middle panel with cross max values along s
        # loop over all sims and compute colorbar value and add line plot
        countSim = 1
        for simRowHash, resAnalysisRow in resAnalysisDF.iterrows():
            if colorFlag and (isinstance(firstVar, str) == False):
                cmapVal = cmapSCVals[values.index(resAnalysisRow[varParList[0] + colorSuffix])]
            elif colorFlag and isinstance(firstVar, str):
                cmapVal = cmapSCVals[values.index(resAnalysisRow[varParList[0]])]
            else:
                cmapVal = countSim / nSamples
            if resAnalysisRow["simName"] == pathDict["refSimName"]:
                ax3.plot(
                    s,
                    resAnalysisRow[resolvedRunoutResType.lower() + crossValue],
                    c="k",
                    label="reference",
                    zorder=nSamples + 1,
                )
            else:
                if cfgSetup.getboolean("varParCategorical"):
                    cmapVal = cmapValues3[varParStrValues.index(resAnalysisRow[varParList[0]])]
                    ax3.plot(
                        s,
                        resAnalysisRow[resolvedRunoutResType.lower() + crossValue],
                        c=cmap3.to_rgba(cmapVal),
                        label=resAnalysisRow[varParList[0]],
                    )
                else:
                    ax3.plot(s, resAnalysisRow[resolvedRunoutResType.lower() + crossValue], c=cmapSC(cmapVal))
            countSim = countSim + 1

        # add colorbar
        if colorFlag and (cfgSetup.getboolean("varParCategorical") is False):
            cmapSC2 = ScalarMappable(norm=Normalize(minVal, maxVal), cmap=cmapSC)
            cbar = ax3.figure.colorbar(cmapSC2, ax=ax3)
            cbar.outline.set_visible(False)
            if cfgSetup["unit"] != "":
                cbar.ax.set_title("[" + cfgSetup["unit"] + "]", pad=10)
            cbar.set_label(paraVar)
            if isinstance(firstVar, str):
                cbar.set_ticks(ticks=np.linspace(0, 1, len(itemsList)), labels=itemsList)
        # add labels title
        ax3.set_title("%s along thalweg for all sims" % name)
        ax3.legend(loc="upper right")
        ax3.set_xlabel("$S_{xy}$ (thalweg) [m]")
        ax3.set_xlim([s.min(), s.max()])
        ax3.set_ylim(auto=True)
        ax3.set_ylabel("$%s_{%s}$ [%s]" % (displayRunoutResType, crossValue, unit))

        outFileName = "_".join(
            [projectName, runoutResType, str(thresholdValue).replace(".", "p"), "slComparisonStat"]
        )

        outFilePath = pU.saveAndOrPlot(pathDict, outFileName, fig)
    return outFilePath


def visuMass(resAnalysisDF, pathDict, simRowHash, refSimRowHash, timeMass):
    """
    Plot and save the results from mass analysis

    Parameters
    ----------
    resAnalysis: dict
        mass results from Aimec analysis:
            entMassFlowArray: entrained mass array corresponding to the time array for each simulation to analyse
            totalMassArray: entrained mass array corresponding to the time array for each simulation to analyse
            entMass: final entrained for each simulation
            finalMass: final mass for each simulation
            time: time array
    pathDict : dict
        dictionary with path to data to analyze
    """
    ####################################
    # Get input data
    # read paths
    projectName = pathDict['projectName']
    # read data
    entMassRef = resAnalysisDF.loc[refSimRowHash, 'entMass']
    finalMassRef = resAnalysisDF.loc[refSimRowHash, 'finalMass']
    entMass = resAnalysisDF.loc[simRowHash, 'entMass']
    finalMass = resAnalysisDF.loc[simRowHash, 'finalMass']
    refSimName = resAnalysisDF.loc[refSimRowHash, 'simName']
    simName = resAnalysisDF.loc[simRowHash, 'simName']
    ############################################
    # prepare for plot
    Title = ['Entrained Mass Flow', 'Total Mass']
    Unit = ['Entrained Mass Flow [$kg.s{^-1}$]', 'Total Mass [kg]']
    fieldList = ['entMassFlowArray', 'totalMassArray']
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(pU.figW*2, pU.figH))
    fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, hspace=0.3)

    for ax, field, title, unit in zip(axes.flatten(), fieldList, Title, Unit):
        refArray = resAnalysisDF.loc[refSimRowHash, field]
        simArray = resAnalysisDF.loc[simRowHash, field]
        if refSimRowHash != simRowHash:
            ax.plot(timeMass, simArray, '-b', label='Simulation: %s ' % simName)
        ax.plot(timeMass, refArray, '-k', label='Reference: %s ' % refSimName)

        ax.set_title(title + ' function of time')
        ax.legend(loc=1)
        ax.set_xlabel('t [s]')
        ax.set_ylabel(unit)

    if refSimRowHash != simRowHash:
        ax2 = axes.flatten()[1].twinx()
        ax2.spines['right'].set_color('r')
        ax2.tick_params(axis='y', colors='r')
        # after loop this refers o the totalMassArray field as final item in fieldList
        ax2.plot(timeMass, (simArray-refArray) / refArray*100, 'r--', label='total mass')

        if np.any(entMass):
            axes.flatten()[1].text(timeMass[-1]/4, (np.nanmin(refArray) + np.nanmax(refArray))/2,
                                   'Entrained Mass Difference : %.2f kg \n Relative to total mass : %.2f %% ' %
                                   ((entMassRef-entMass), (entMassRef-entMass)/finalMassRef*100),
                                   bbox=dict(boxstyle="square", ec='white', fc='white'),
                                   horizontalalignment='left', verticalalignment='bottom')

        ax2.set_ylabel('Total Mass Difference [%]', color='r')

    outFileName = '_'.join([projectName, str(simName), 'massAnalysis'])
    pU.putAvaNameOnPlot(ax, pathDict['projectName'])
    outFilePath = pU.saveAndOrPlot(pathDict, outFileName, fig)

    return outFilePath


def visuSimple(cfgSetup, rasterTransfo, resAnalysisDF, newRasters, pathDict):
    """
    Plot and save the Peak Pressure Peak Flow thickness and Peak speed
    fields after coord transfo

    Parameters
    ----------
    rasterTransfo: dict
        domain transformation information
    resAnalysis: dict
        results from Aimec analysis:
            numpy array with the 'runout' for each simulation and the 'thresholdValue'
    newRasters: dict
        dictionary with new (s, l) raster for ppr, pft and pft
    pathDict : dict
        dictionary with path to data to analyze
    """
    ####################################
    # Get input data
    # read paths
    projectName = pathDict['projectName']
    refSimRowHash = pathDict['refSimRowHash']
    # read data
    runoutResType = cfgSetup['runoutResType']
    thresholdValue = cfgSetup['thresholdValue']
    s = rasterTransfo['s']
    l = rasterTransfo['l']
    indStartOfRunout = rasterTransfo['indStartOfRunout']
    rasterdataPres = newRasters['newRefRasterPPR']
    rasterdataThickness = newRasters['newRefRasterPFT']
    rasterdataSpeed = newRasters['newRefRasterPFV']
    runout = resAnalysisDF.loc[refSimRowHash, 'sRunout']

    ############################################
    # prepare for plot
    Cmap = [pU.cmapPres, pU.cmapThickness, pU.cmapSpeed]
    Title = ['peak pressure', 'peak flow thickness', 'peak flow velocity']
    Unit = [pU.cfgPlotUtils['unitppr'], pU.cfgPlotUtils['unitpft'], pU.cfgPlotUtils['unitpfv']]
    Data = np.array(([None] * 3))
    Data[0] = rasterdataPres
    Data[1] = rasterdataThickness
    Data[2] = rasterdataSpeed

    ############################################
    # Figure: Pressure thickness speed

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(pU.figW*2, pU.figH*3))
    fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, hspace=0.3)

    for ax, cmap, data, title, unit in zip(axes.flatten(), Cmap, Data, Title, Unit):
        maskedArray = np.ma.masked_where(data == 0, data)
        maskedArrayTranspose = np.transpose(maskedArray)
        cmap, _, ticks, norm = pU.makeColorMap(cmap, np.nanmin(maskedArray), np.nanmax(maskedArray),
                                               continuous=pU.contCmap)
        cmap.set_bad('w', 1.)
        ax.axvline(x=runout, color='k', linestyle='-',
                   label='runout (%s < %.1f %s)' % (runoutResType, cfgSetup.getfloat('thresholdValue'),
                                                    pU.cfgPlotUtils['unit' + runoutResType]))

        ax.axvline(x=s[indStartOfRunout], color='k', linestyle='--', label=rasterTransfo['labelRunout'])
        ref3, im = pU.NonUnifIm(ax, s, l, maskedArrayTranspose, '$S_{XY}$ (thalweg) [m]', '$L_{XY}$ (thalweg) [m]',
                                extent=[s.min(), s.max(), l.min(), l.max()],
                                cmap=cmap, norm=norm)

        ax.set_title(title)
        ax.legend(loc=4)
        pU.addColorBar(im, ax, ticks, unit)
        pU.putAvaNameOnPlot(ax, pathDict['projectName'])

    outFileName = '_'.join([projectName, runoutResType, str((thresholdValue)).replace('.', 'p'), 'referenceFields'])

    pU.saveAndOrPlot(pathDict, outFileName, fig)


def visuComparison(rasterTransfo, inputs, pathDict):
    """
    Plot and save the comparison between current simulation and Reference
    in the runout area

    Parameters
    ----------
    rasterTransfo: dict
        domain transformation information
    inputs: dict
        input data for plot:
            'refData' and 'compData' arrays
            'refRasterMask' and 'compRasterMask' arrays
            'thresholdArray'
            'i' id of the simulation
            'runoutResType' result type analyzed
            'diffLim'
    pathDict : dict
        dictionary with path to data to analyze
    """
    ####################################
    # Get input data
    # read paths
    projectName = pathDict['projectName']
    # read data
    s = rasterTransfo['s']
    l = rasterTransfo['l']
    indStartOfRunout = rasterTransfo['indStartOfRunout']
    refData = inputs['refData']
    compData = inputs['compData']
    refRasterMask = inputs['refRasterMask']
    compRasterMask = inputs['compRasterMask']
    simName = inputs['simName']
    refSimName = pathDict['refSimName']
    runoutResType = inputs['runoutResType']
    # base name without layer suffix for cosmetic lookups (units, names, colorMaps)
    baseRunoutResType = inputs.get('baseRunoutResType', runoutResType)
    displayRunoutResType = inputs.get('displayRunoutResType', runoutResType)
    unit = pU.cfgPlotUtils['unit' + baseRunoutResType]
    name = pU.cfgPlotUtils['name' + baseRunoutResType]
    thresholdArray = inputs['thresholdArray']
    thresholdValue = thresholdArray[-1]

    if not np.any(refData):
        log.warning(
            "ContourComparisonToReference plot not generated as only 0 values for reference simulation: %s"
            % refSimName
        )
        outFilePath = None
    else:
        cmapTF, _, ticks, normTF = pU.makeColorMap(
            pU.colorMaps[baseRunoutResType], np.nanmin((refData)), np.nanmax((refData)), continuous=pU.contCmap
        )
        cmapTF.set_bad(color="w")

        cmapTF.set_under(color="b")
        cmapTF.set_over(color="r")
        cmapTF.set_bad(alpha=0)
        dataTF = compRasterMask - refRasterMask
        dataTF = np.ma.masked_where(dataTF == 0.0, dataTF)

        xLimRef = s[np.max(np.nonzero(np.any(refData > 0, axis=1))[0])] + 20
        xLim = (
            s[
                max(
                    np.max(np.nonzero(np.any(refData > 0, axis=1))[0]),
                    np.max(np.nonzero(np.any(compData > 0, axis=1))[0]),
                )
            ]
            + 20
        )

        # define figure extent
        xExtent = xLim - s[indStartOfRunout]
        ratio = xExtent / np.nanmax(rasterTransfo["l"])
        if ratio > 1:
            figHM = 2 * (6 / 4)
            figWM = ratio * 2
            ncolLegend = 4
        else:
            figHM = ((1 + (1 - ratio)) * 2) * (6 / 4)
            figWM = 2
            ncolLegend = 2
        ############################################
        # Figure: Raster comparison
        # , constrained_layout=True)
        fig = plt.figure(figsize=(pU.figW * figWM, pU.figH * figHM), layout="tight")
        gs = gridspec.GridSpec(4, 2, hspace=1.5)

        compData = compData[indStartOfRunout:, :]
        refData = refData[indStartOfRunout:, :]
        dataDiff = compData - refData
        dataDiff = np.where((refData == 0) & (compData == 0), np.nan, dataDiff)
        dataDiffPlot = np.where(
            (refData < thresholdArray[-1]) & (compData < thresholdArray[-1]), np.nan, dataDiff
        )
        dataDiffPlot = dataDiffPlot[~np.isnan(dataDiffPlot)]

        if dataDiffPlot.size:
            # only add the second axis if one of the two avalanches reached the runout area
            indDiff = np.abs(dataDiffPlot) > 0

            if indDiff.any():
                # only plot hist and CDF if there is a difference in the data
                ax2 = fig.add_subplot(gs[0:5, 1])
                ax1 = fig.add_subplot(gs[0:5, 0])
            else:
                ax2 = fig.add_subplot(gs[:, 1])
                ax1 = fig.add_subplot(gs[:, 0])

            cmap = pU.cmapdiv
            cmap.set_bad(color="w")
            elev_max = inputs["diffLim"]
            ref0, im3 = pU.NonUnifIm(
                ax2,
                s[indStartOfRunout:],
                l,
                np.transpose(dataDiff),
                "$S_{XY}$ (thalweg) [m]",
                "$L_{XY}$ (thalweg) [m]",
                extent=[s[indStartOfRunout], xLim, l.min(), l.max()],
                cmap=cmap,
            )
            im3.set_clim(vmin=-elev_max, vmax=elev_max)

            # print contour lines only if the threshold is reached
            S, L = np.meshgrid(s[indStartOfRunout:], l)
            colorsP = pU.colorMaps["pft"]["colors"][1:]
            if (np.where(refData > thresholdArray[-1], True, False)).any():
                contourRef = ax2.contour(
                    S, L, np.transpose(refData), levels=thresholdArray[:-1], linewidths=2, colors=colorsP
                )
                # generate corresponding labels
                labels = [str(level) for level in thresholdArray[:-1]]
                # add legend associated to the contour plot
                handles, _ = contourRef.legend_elements()
                ax2.legend(
                    title=displayRunoutResType + " contour lines [" + unit + "]",
                    handles=handles,
                    labels=labels,
                    ncol=ncolLegend,
                    loc="lower left",
                )
            else:
                log.warning("Reference %s did not reach the runout area!" % refSimName)
                ax2.text(
                    (s[indStartOfRunout] + xLim) / 2,
                    0,
                    "Reference %s did not reach the runout area!" % refSimName,
                    fontsize=24,
                    color="red",
                    bbox=dict(facecolor="none", edgecolor="red", boxstyle="round,pad=1"),
                    ha="center",
                    va="center",
                )
            if (np.where(compData > thresholdArray[-1], True, False)).any():
                contourComp = ax2.contour(
                    S,
                    L,
                    np.transpose(compData),
                    levels=thresholdArray[:-1],
                    linewidths=2,
                    colors=colorsP,
                    linestyles="dashed",
                )
            else:
                log.warning("Simulation %s did not reach the runout area!" % simName)
                ax2.text(
                    (s[indStartOfRunout] + xLim) / 2,
                    0,
                    "Simulation %s did not reach the runout area!" % simName,
                    fontsize=24,
                    color="red",
                    bbox=dict(facecolor="none", edgecolor="red", boxstyle="round,pad=1"),
                    ha="center",
                    va="center",
                )

            # if compRasterMask[indStartOfRunout:, :].any() or refRasterMask[indStartOfRunout:, :].any():
            if dataTF.any():
                ref1, im1 = pU.NonUnifIm(
                    ax1,
                    s,
                    l,
                    np.transpose(dataTF),
                    "$S_{XY}$ (thalweg) [m]",
                    "$L_{XY}$ (thalweg) [m]",
                    extent=[s[indStartOfRunout], xLim, l.min(), l.max()],
                    cmap=cmapTF,
                )
                im1.set_clim(vmin=-0.5, vmax=0.5)
                ax1.set_xlim([s[indStartOfRunout], xLim])
            else:
                ax1.text(
                    0.5,
                    0.5,
                    "No difference in runout area",
                    fontsize=18,
                    color="red",
                    bbox=dict(facecolor="none", edgecolor="red", boxstyle="round,pad=1"),
                    ha="center",
                    va="center",
                )
            if pathDict["compType"][0] == "comModules":
                namePrint = "refMod:" + pathDict["compType"][1] + "\n" + "compMod:" + pathDict["compType"][2]
                pU.putAvaNameOnPlot(ax1, namePrint)
            else:
                namePrint = "ref:" + str(refSimName) + "\n" + "sim:" + str(simName)
                pU.putAvaNameOnPlot(ax1, namePrint)

            ax1.set_title(
                "%s difference (sim - reference) in runout area" % displayRunoutResType
                + "\n"
                + "blue = FN, red = FP"
            )

            if indDiff.any():
                # only plot hist and CDF if there is a difference in the data
                ax3 = fig.add_subplot(gs[3, 0])
                ax4 = fig.add_subplot(gs[3, 1])
                # there is data to compare in the runout area
                _ = sPlot.plotHistCDFDiff(
                    dataDiffPlot,
                    ax4,
                    ax3,
                    insert="False",
                    title=[
                        "%s diff histogram" % displayRunoutResType,
                        "%s diff CDF (95%% and 99%% centiles)" % displayRunoutResType,
                    ],
                )

            ax2.set_xlim([s[indStartOfRunout], xLim])
            divider = make_axes_locatable(ax2)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            pU.addColorBar(
                im3, ax2, None, None, title=displayRunoutResType + (" [%s]" % unit), extend="both", cax=cax
            )
            ax2.set_aspect("equal")
            ax1.set_aspect("equal")
        else:
            # if no avalanche reached the runout area print a warning on the second plot
            ax2 = plt.subplot2grid((3, 3), (0, 0), rowspan=2, colspan=3)
            log.warning("No data in runout area")
            ax2.text(
                0.5,
                0.5,
                "No difference in runout area",
                fontsize=24,
                color="red",
                bbox=dict(facecolor="none", edgecolor="red", boxstyle="round,pad=1"),
                ha="center",
                va="center",
            )

        if pathDict["compType"][0] == "comModules":
            ax2.set_title(
                "%s difference and contour lines" % displayRunoutResType
                + "\n"
                + "refMod = full, compMod = dashed line"
            )
        else:
            ax2.set_title(
                "%s difference and contour lines" % displayRunoutResType + "\n" + "ref = full, sim = dashed line"
            )

        # fig.subplots_adjust(hspace=0.13, wspace=0.3)
        outFileName = "_".join(
            [
                projectName,
                runoutResType,
                str(thresholdValue).replace(".", "p"),
                str(simName),
                "ContourComparisonToReference",
            ]
        )
        outFilePath = pU.saveAndOrPlot(pathDict, outFileName, fig)

    return outFilePath


def resultWrite(pathDict, cfg, rasterTransfo, resAnalysisDF):
    """
    This function writes the main Aimec results to a file (outputFile)
    in pathDict
    """
    ####################################
    # Get input data
    projectName = pathDict['projectName']
    pathResult = pathDict['pathResult']
    pathName = pathDict['pathName']
    refSimRowHash = pathDict['refSimRowHash']
    demName = os.path.basename(pathDict['demSource'])
    # dataName = [os.path.basename(name) for name in pathDict['ppr']]
    cfgSetup = cfg['AIMECSETUP']
    cfgFlags = cfg['FLAGS']
    flagMass = cfgFlags.getboolean('flagMass')
    domainWidth = cfgSetup['domainWidth']
    thresholdValue = cfgSetup['thresholdValue']
    runoutResType = cfgSetup['runoutResType']
    unit = pU.cfgPlotUtils['unit' + runoutResType]
    name = pU.cfgPlotUtils['name' + runoutResType]

    startOfRunoutAreaAngle = rasterTransfo['startOfRunoutAreaAngle']
    areaSum = resAnalysisDF.loc[refSimRowHash, 'TP'] + resAnalysisDF.loc[refSimRowHash, 'FN']
    if areaSum == 0:
        log.warning('Reference did not reach the runout area. Not normalizing area indicators')
    else:
        resAnalysisDF['TP'] = resAnalysisDF['TP'] / areaSum
        resAnalysisDF['FN'] = resAnalysisDF['FN'] / areaSum
        resAnalysisDF['FP'] = resAnalysisDF['FP'] / areaSum
        resAnalysisDF['TN'] = resAnalysisDF['TN'] / areaSum
    # do some statistics
    forStats = ['xRunout', 'yRunout', 'sRunout', 'elevRel', 'deltaH', 'maxpprCrossMax', 'maxpftCrossMax',
                'maxpfvCrossMax', 'TP', 'FN', 'FP', 'TN']
    if flagMass:
        forStats = forStats + ['relMass', 'entMass', 'finalMass', 'relativMassDiff', 'growthIndex', 'growthGrad']
    forStats = list(set(forStats) & set(resAnalysisDF.columns))
    # compute some statistics
    resAnalysisStatsDF = resAnalysisDF[list(forStats)].describe(percentiles=None)

    header = ''.join(['projectName: ', projectName, '\n',
                      'path: ', pathName, '\n',
                      'dhm: ', demName, '\n',
                      'domain_width: ', str(domainWidth), ' m\n',
                      'runout based on ', name, ' results\n',
                      name, ' limit: ', str(thresholdValue), unit, '\n',
                      'start of runout area Angle (SROA angle): ', str(round(startOfRunoutAreaAngle, 2)), ' °\n'])

    outFileName = '_'.join(['Results', projectName, str(runoutResType), 'lim', str(thresholdValue), 'w',
                           str(domainWidth)]) + 'resAnalysisDF'
    outname = os.path.join(pathResult, outFileName)
    outFileNameStats = '_'.join(['Results', projectName, str(runoutResType), 'lim', str(thresholdValue), 'w',
                                str(domainWidth)]) + 'stats.csv'
    outnameStats = os.path.join(pathResult, outFileNameStats)

    # check if folder exists / create
    if not os.path.exists(os.path.dirname(outname)):
        os.makedirs(os.path.dirname(outname))
    resAnalysisDF.to_csv(outname + '.csv')
    resAnalysisDF.to_pickle(outname + '.pkl')

    if not os.path.exists(os.path.dirname(outnameStats)):
        os.makedirs(os.path.dirname(outnameStats))
    resAnalysisStatsDF.to_csv(outnameStats)

    log.info('File written: %s' % outname)
    log.info('File written: %s' % outnameStats)


def resultVisu(cfgSetup, inputsDF, pathDict, cfgFlags, rasterTransfo, resAnalysisDF):
    """
    plot the normalized area difference between reference and other simulations
    """
    ####################################
    # Get input data
    runoutResType = cfgSetup['runoutResType']
    displayRunoutResType = pathDict.get('displayRunoutResType', runoutResType)
    varParList = cfgSetup['varParList'].split('|')
    unit = cfgSetup['unit']
    paraVar = cfgSetup['varParList'].split('|')[0]
    name = pU.cfgPlotUtils['name' + runoutResType]
    nSim = len(resAnalysisDF.index)
    refSimRowHash = pathDict['refSimRowHash']
    thresholdValue = cfgSetup['thresholdValue']

    runout = resAnalysisDF['sRunout'].to_numpy()
    runoutRef = resAnalysisDF.loc[refSimRowHash, 'sRunout']
    areaSum = resAnalysisDF.loc[refSimRowHash, 'TP'] + resAnalysisDF.loc[refSimRowHash, 'FN']
    if areaSum == 0:
        log.warning('Reference did not reach the runout area. Not normalizing area indicators')
        areaSum = 1
    rTP = resAnalysisDF['TP'].to_numpy() / areaSum
    rFP = resAnalysisDF['FP'].to_numpy() / areaSum
    rTPRef = resAnalysisDF.loc[refSimRowHash, 'TP'] / areaSum
    rFPRef = resAnalysisDF.loc[refSimRowHash, 'FP'] / areaSum

    # If more than 100 files are provided, add a density plot
    plotDensity = 0
    if (nSim > cfgFlags.getfloat('nDensityPlot')):
        plotDensity = 1
    # Get colors for scatter
    unitSC = cfgSetup['unit']
    nSamples = np.size(runout)
    firstVar = defineVariableType(paraVar, resAnalysisDF, pathDict)
    if 'colorParameter' in pathDict:
        if pathDict['colorParameter'] is False:
            values = None
        elif not isinstance(firstVar, str):
            _, _, _, _, values, _ = defineColorcodingValues(resAnalysisDF, paraVar, nSamples)
        else:
            values = None
    else:
        values = None
    cmapSC, colorSC, ticksSC, normSC, unitSC, itemsList, displayColorBar = pU.getColors4Scatter(values, nSamples,
                                                                                                unitSC)

    ############################################
    # Final result diagram - roc-plots

    fig = plt.figure(figsize=(pU.figW, pU.figH))
    ax1 = fig.add_subplot(111)
    ax1.set_title('Normalized difference compared to reference')
    ax1.set_ylabel('True positive rate')
    ax1.set_xlabel('False positive rate')
    pU.putAvaNameOnPlot(ax1, pathDict['projectName'])
    if plotDensity:  # estimate 2D histogram --> create pcolormesh
        nbins = 100
        H, xedges, yedges = np.histogram2d(rFP, rTP, bins=nbins)
        H = np.flipud(np.rot90(H))
        Hmasked = np.ma.masked_where(H == 0, H)
        dataDensity = plt.pcolormesh(xedges, yedges, Hmasked, cmap=cm.cool)
        cbar = plt.colorbar(dataDensity, orientation='horizontal')
        cbar.ax.set_ylabel('hit rate density')
    else:
        sc = ax1.scatter(rFP, rTP, c=colorSC, cmap=cmapSC, norm=normSC, marker=pU.markers[0],
                         label=('runout points (%s<%.1f%s)' % (displayRunoutResType, cfgSetup.getfloat('thresholdValue'),
                                                                  pU.cfgPlotUtils['unit' + runoutResType])))

        if displayColorBar:
            pU.addColorBar(sc, ax1, ticksSC, unitSC, title=paraVar, pad=0.08, tickLabelsList=itemsList)

        ax1.plot(rFPRef, rTPRef, color='g', label='Reference', marker='+', markersize=2*pU.ms, linestyle='None')
        ax1.legend(loc=4)

    plt.xlim([-0.03, max(1, max(rFP)+0.03)])
    plt.ylim([-0.03, 1.03])
    plt.grid('on')

    outFileName = '_'.join([pathDict['projectName'], runoutResType, str(thresholdValue).replace('.', 'p'), 'ROC'])

    pU.saveAndOrPlot(pathDict, outFileName, fig)

    return


def fetchContourLines(rasterTransfo, inputs, level, contourDict):
    """ fetch contour line of transformed field data

        Parameters
        -----------
        rasterTransfo: dict
            raster transformation dictionary
        inputs: dict
            input data here needed the transformed field data and simName
        level: float
            the contour level
        contourDict: dict
            dictionary with x, y coordinates for each sim contour line

        Returns
        ---------
        contourDict: dict
            updat. dictionary with name of contour line: x, y coordinates for each sim contour line -> added info
            of current simulation
    """

    # fetch input data
    s = rasterTransfo['s']
    l = rasterTransfo['l']
    compData = inputs['compData']
    simName = inputs['simName']

    gridL, gridS = np.meshgrid(l, s)

    # create contour lines and extract coordinates and write to dict
    contourDictXY = pU.fetchContourCoords(gridL, gridS, compData, level)
    # add to dict
    contourDict[simName] = contourDictXY

    return contourDict


def plotContoursTransformed(contourDict, pathDict, rasterTransfo, cfgSetup, inputsDF):
    """ plot contour lines of all transformed fields
        colorcode contour lines with first parameter in varParList if not type string of value

        Parameters
        -----------
        contourDict: dict
            dictionary with contourline coordinates
        pathDict: dict
            dictionary with info on project name
        rasterTransfo: dict
            raster transformation dictionary
        cfgSetup: configparser
            configuration settings for AIMECSETUP
        inputsDF: pandas DataFrame
            dataframe with one row per simulation and information on parameter (if no configuration file found,
            derived from simName) and available files

    """
    # fetch raster coordinate data
    s = rasterTransfo['s']
    l = rasterTransfo['l']
    indStartOfRunout = rasterTransfo['indStartOfRunout']
    unit = pU.cfgPlotUtils['unit' + cfgSetup['runoutResType']]
    displayRunoutResType = pathDict.get('displayRunoutResType', cfgSetup['runoutResType'])
    colorOrdering = False
    nSamples = inputsDF.shape[0]

    if pathDict['colorParameter']:
        # fetch parameter info for sims
        simDF = inputsDF
        varParList = cfgSetup['varParList'].split('|')
        paraVar = varParList[0]
        firstVar = defineVariableType(paraVar, inputsDF, pathDict)
        if not isinstance(firstVar, str):
            colorOrdering, colorSuffix, minVal, maxVal, _, _ = defineColorcodingValues(
                inputsDF, paraVar, nSamples
            )

    if colorOrdering is False:
        minVal = 0
        maxVal = 1

    # define figure extent
    # first find max SXY value of contour lines to define max extent on SXY
    xMax = 0
    for index, simName in enumerate(contourDict):
        for key in contourDict[simName]:
            if np.amax(contourDict[simName][key]["y"]) > xMax:
                xMax = np.amax(contourDict[simName][key]["y"])
    sMax = np.where(s >= (xMax - 1.e-8))[0]
    # compute the ratio of LXY to SXY to get figure width and height
    xExtent = s[sMax[0]] - s[indStartOfRunout]
    ratio = (xExtent / np.nanmax(rasterTransfo['l']))
    if ratio > 1:
        figHM = 1
        figWM = ratio
    else:
        figHM = ((1 + (1 - ratio)))
        figWM = 1

    # intialize fig - if start of runout area at 0 thalweg - only one panel in plot
    if indStartOfRunout == 0:
        fig = plt.figure(figsize=(pU.figW * figWM, pU.figH * figHM))
        gs = gridspec.GridSpec(1, 2)
    else:
        fig = plt.figure(figsize=(pU.figW * (2*figWM), pU.figH * figHM))
        gs = gridspec.GridSpec(1, 3)

    # PANEL 1 show contour lines of all sims in contourDict for thresholdValue runoutResType
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax1.set_title('%s %s %s contour lines' % (displayRunoutResType, cfgSetup['thresholdValue'],
        unit))
    ax1.set_xlabel('$S_{XY}$ (thalweg) [m]')
    ax1.set_ylabel('$L_{XY}$ (thalweg) [m]')
    pU.putAvaNameOnPlot(ax1, pathDict['projectName'])

    norm = mpl.colors.Normalize(vmin=minVal, vmax=maxVal)
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=cmapCrameri.hawaii)
    cmap.set_array([])
    # limit extent to avalanche extent
    for index, simName in enumerate(contourDict):
        if colorOrdering:
            cmapVal = simDF.loc[simDF["simName"] == simName, paraVar + colorSuffix].values[0]
        else:
            cmapVal = index / len(contourDict)
        for key in contourDict[simName]:
            if simName == pathDict['refSimName']:
                addLinePlot(contourDict[simName][key], 'k', 'reference', ax1, key, zorder=len(contourDict),
                            linestyle='solid', alpha=0.7)
            else:
                ax1.plot(contourDict[simName][key]['y'], contourDict[simName][key]['x'], c=cmap.to_rgba(cmapVal))

    # tolerance needed because of rounding and saving issues
    sMax = np.where(s >= (xMax-1.e-8))[0]
    ax1.set_xlim([s[0], s[sMax[0]]])
    if colorOrdering:
        cbar = ax1.figure.colorbar(cmap, ax=ax1)
        cbar.outline.set_visible(False)
        if cfgSetup['unit'] != '':
            cbar.ax.set_title('[' + cfgSetup['unit'] + ']', pad=10)
        cbar.set_label(paraVar)
    else:
        # TODO: remove?
        log.warning('ordering for contour line plot not available for parameter of type string')

    # add indication for runout area
    ax1.axvline(s[indStartOfRunout], color='gray', linestyle='--',
                label=rasterTransfo['labelRunout'])
    # label=('start of runout area: '+ r'$\beta_{%.1f °}$' % (rasterTransfo['startOfRunoutAreaAngle'])))
    ax1.legend(loc='upper left')

    if indStartOfRunout != 0:
        # PANEL 2 zoom into runout area
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.set_title('zoom into runout area, equal axes')
        ax2.set_xlabel('$S_{XY}$ (thalweg) [m]')
        ax2.set_ylabel('$L_{XY}$ (thalweg) [m]')
        pU.putAvaNameOnPlot(ax1, pathDict['projectName'])
        for index, simName in enumerate(contourDict):
            if colorOrdering:
                cmapVal = simDF.loc[simDF["simName"] == simName, paraVar + colorSuffix].values[0]
            else:
                cmapVal = index / len(contourDict)
            for key in contourDict[simName]:
                if simName == pathDict['refSimName']:
                    # define label for reference sim, if info on what it is based is available
                    if pathDict['valRef'] == '':
                        labelReference = 'reference'
                    else:
                        labelReference ='reference: %s = %s' % (cfgSetup['varParList'].split('|')[0], pathDict['valRef'])
                    # add contour lines
                    addLinePlot(contourDict[simName][key], 'k', labelReference, ax2, key, zorder=len(contourDict),
                                linestyle='solid', alpha=0.7)
                else:
                    ax2.plot(contourDict[simName][key]['y'], contourDict[simName][key]['x'], c=cmap.to_rgba(cmapVal))

        ax2.set_xlim([s[indStartOfRunout], s[sMax[0]]])
        ax2.legend()
        ax2.set_aspect('equal')

    # save and or plot fig
    outFileName = pathDict['projectName'] + '_ContourLinesAll'
    pU.saveAndOrPlot(pathDict, outFileName, fig)


def addLinePlot(contourDict, colorStr, labelStr, ax, key, zorder='', linestyle='solid', alpha=1.):
    """ add a contour line with label only for line that has  _0 in name

        Parameters
        -----------
        contourDict: dict
            dict with x, y info of contourline coordinates
        colorStr: str
            color of line
        labelStr: str
            name of legend entry for line
        ax: matplotlib axes object
            axes where to add the lines too
        key: str
            name of the contour line
        zorder: int
            order of line on plot
        linestyle: str
            style of line plot: dashed, solid, dotted,..
        alpha: float
            between 0 and 1 to define how opaque the line
    """

    if key.split('.')[1] == '0_1':
        ax.plot(contourDict['y'], contourDict['x'], c=colorStr, label=labelStr,
            zorder=zorder, linestyle=linestyle, alpha=alpha)
    else:
        ax.plot(contourDict['y'], contourDict['x'], c=colorStr, zorder=zorder, linestyle=linestyle, alpha=alpha)


def plotThalwegAltitude(pathDict, rasterTransfo, pfvCrossMax, simName, velocityThreshold):
    """ create thalweg-altitude plot
        the thalweg z profile and the mpfv²/2g (velocity-altitude) colorcoded using the mpfv values
        the mpfv values come from the cross max values along the thalweg coordinate system (aimec)
        plot is saved to avaDir/Outputs/out3Plot/thalwegAltSimname

        Parameters
        ------------
        avaDir: pathlib path
            path to avalanche directory
        rasterTransfo: dict
            dict with info on transformation from cartesian to thalweg coordinate system
        pfvCrossMax: pandas dataFrame series
            cross profile max values of peak flow velocity transformed into thalweg coordinate system
        simName: str
            simulation name
    """

    # initialize figure
    fig = plt.figure(figsize=(pU.figW, pU.figH))
    ax1 = fig.add_subplot(1, 1, 1)

    # add thalweg-altitude plot to axes
    ax1 = addThalwegAltitude(ax1, rasterTransfo, pfvCrossMax, velocityThreshold)

    # save and or plot
    plotName = ('thalwegAlt%s' % (simName))
    plotPath = pU.saveAndOrPlot(pathDict, plotName, fig)


def addThalwegAltitude(ax1, rasterTransfo, pfvCrossMax, velocityThreshold, zMaxM=np.nan):
    """ add thalweg-altitude plot to axes
        the thalweg z profile and the mpfv²/2g (velocity-altitude) colorcoded using the mpfv values
        the mpfv values come from the cross max values along the thalweg coordinate system (aimec)

        Parameters
        ------------
        rasterTransfo: dict
            dict with info on transformation from cartesian to thalweg coordinate system
        pfvCrossMax: pandas dataFrame series
            cross profile max values of peak flow velocity transformed into thalweg coordinate system
        zMaxM: float
            optional - value to define max limit of y -axis
    """

    # compute velocity-Altitude-Field
    g = pU.gravityAcc
    pfvCM = pfvCrossMax.to_numpy()[0]
    velAltField = rasterTransfo['z'] + (pfvCM ** 2.) / (2. * g)

    # add scatter plot of velocity-Altitude field colocoded with max peak flow velocity
    scat = ax1.scatter(rasterTransfo['s'], velAltField, marker='s', cmap=pU.cmapRangeTime, s=8 * pU.ms, c=pfvCM)
    ax1.plot(rasterTransfo['s'], rasterTransfo['z'], '-y',
             path_effects=[pe.Stroke(linewidth=2, foreground='k'), pe.Normal()], zorder=20, linewidth=1, markersize=0.8,
             label='Thalweg altitude')

    # get indices
    indVelStart, indVelZero = getIndicesVel(pfvCM, velocityThreshold)

    # add colorbar
    cbar2 = ax1.figure.colorbar(scat, ax=ax1, use_gridspec=True)
    cbar2.ax.set_title('[m/s]', pad=10)
    cbar2.ax.set_ylabel('Max peak flow velocity (mpfv)')

    # draw the horizontal and vertical lines for angle computation
    ax1.vlines(x=rasterTransfo['s'][indVelStart], ymin=velAltField[indVelZero], ymax=velAltField[indVelStart],
               color='grey', linestyle='--')
    ax1.hlines(y=rasterTransfo['z'][indVelZero], xmin=rasterTransfo['s'][indVelStart],
               xmax=rasterTransfo['s'][indVelZero],
               color='grey', linestyle='--')
    # compute alpha angle based on pfvCM field
    deltaz = rasterTransfo['z'][indVelStart] - rasterTransfo['z'][indVelZero]
    deltas = rasterTransfo['s'][indVelZero] - rasterTransfo['s'][indVelStart]
    alpha = np.rad2deg(np.arctan(deltaz / deltas))

    # add textbox with angles, delta values
    textString = (r'$\Delta z$' + ('=%s m\n' % str(round(deltaz, 1))) + (r'$\Delta s_{xy}$=%s ' %
                  str(round(deltas, 1))) + 'm\n' + r'$\alpha$=' + str(round(alpha, 2)) +
                  '° (pfv > %.1f $ms{-1}$)' % velocityThreshold)
    ax1.text(0.98, 0.9, textString, horizontalalignment='right',
             verticalalignment='top', fontsize=10, transform=ax1.transAxes, multialignment='left')
    X = [rasterTransfo['s'][indVelStart], rasterTransfo['s'][indVelZero]]
    Y = [rasterTransfo['z'][indVelStart], rasterTransfo['z'][indVelZero]]
    ax1.plot(X, Y, color='grey', linestyle='--', linewidth=0.8)

    # Labels
    ax1.set_xlabel('$s_{xy}$ [m]', fontsize=20)
    ax1.set_ylabel('$z + mpfv²/2g$ [m]', fontsize=20)

    # limit axes extent to where there is data with buffer
    sExt = rasterTransfo['s'][indVelZero] * 0.1
    zMax = np.nanmax([zMaxM, np.nanmax(velAltField)])
    zExt = (zMax - velAltField[indVelZero]) * 0.1
    ax1.set_xlim([rasterTransfo['s'][0] - sExt, rasterTransfo['s'][indVelZero] + sExt])
    ax1.set_ylim([velAltField[indVelZero] - zExt, zMax + zExt])
    ax1.set_title('Thalweg-Altitude')

    return ax1


def plotMaxValuesComp(pathDict, resultsDF, name1, name2, hue=None):
    """plot result type name1 vs name 2 and colorcode scenarios using hue
       add reference sim value with label

        Parameters
        -----------
        resultsDF: pandas dataFrame
            dataframe with one row per simulation with info on parameters and aimec analysis results
        name1: str
            name of result type one
        name2: str
            name of result type two
        hue: str
            name of parameter used for colorcoding
    """

    # get dataFrame values for reference
    valDF = resultsDF[resultsDF['simName'] == pathDict['refSimName']]

    # define units for available analysis parameters
    availableoptions = [
        "pfvFieldMax",
        "pfvFieldMin",
        "pfvFieldMean",
        "maxpfvCrossMax",
        "pftFieldMax",
        "pftFieldMin",
        "pftFieldMean",
        "maxpftCrossMax",
        "pfdFieldMax",
        "pfdFieldMin",
        "pfdFieldMean",
        "maxpfdCrossMax",
        "pprFieldMax",
        "pprFieldMin",
        "pprFieldMean",
        "maxpprCrossMax",
        "sRunout",
        "deltaSXY",
        "zRelease",
        "zRunout",
        "deltaZ",
        "relMass",
        "finalMass",
        "entMass",
        "runoutAngle",
    ]
    units = [
        "ms-1",
        "ms-1",
        "ms-1",
        "ms-1",
        "m",
        "m",
        "m",
        "m",
        "m",
        "m",
        "m",
        "m",
        "kPa",
        "kPa",
        "kPa",
        "kPa",
        "m",
        "m",
        "m",
        "m",
        "m",
        "kg",
        "kg",
        "kg",
        "°",
    ]

    if name1 not in availableoptions:
        message = 'compResType1: %s not in available options' % name1
        log.error(message)
        raise ValueError(message)
    elif name2 not in availableoptions:
        message = 'compResType2: %s not in available options' % name2
        log.error(message)
        raise ValueError(message)

    labelRef = 'reference'
    if hue == '':
        hue = None
    elif hue not in resultsDF.columns.values.tolist():
        message = 'Scenario name for comparison plot not valid name %s not using for colorcoding' % hue
        log.warning(message)
    elif isinstance(resultsDF[hue].iloc[0], str) == False and np.isnan(resultsDF[hue].to_numpy()).any():
        message = 'Not all sims have a value for %s - no scenario used for plot' % hue
        log.warning(message)
        hue = None
    else:
        labelRef = 'reference (%s)' % valDF[hue].values[0]

    name1Index = availableoptions.index(name1)
    name2Index = availableoptions.index(name2)
    fig = sns.jointplot(data=resultsDF, x=name1, y=name2, hue=hue)
    fig.ax_joint.scatter(valDF[name1], valDF[name2], color='k', label=labelRef)
    fig.ax_joint.legend()

    # add label names with units
    fig.ax_joint.set_xlabel('%s [%s]' % (availableoptions[name1Index], units[name1Index]))
    fig.ax_joint.set_ylabel('%s [%s]' % (availableoptions[name2Index], units[name2Index]))
    pU.putAvaNameOnPlot(fig.ax_joint, pathDict['avalancheDir'])

    outFileName = pathDict['projectName'] + ('_%svs%s' % (name1, name2))
    pU.saveAndOrPlot(pathDict, outFileName, fig.figure)


def plotVelThAlongThalweg(pathDict, rasterTransfo, pftCrossMax, pfvCrossMax, cfgPlots, simName):
    """ plot the velocity and thickness cross max values along the thalweg, with pft x10
        only plot every barInt value

        Parameters
        -----------
        pathDict: dict
            info on avalancheDir
        rasterTransfo: dict
            info on domain transformation
        pftCrossMax: numpy nd array
            peak flow thickness max values along cross profiles of thalweg coordinate system
        pfvCrossMax: numpy nd array
            peak flow velocity max values along cross profiles of thalweg coordinate system
        cfgPlots: configparser object
            used: barInterval: width to compute the values that should be used for plotting
            velocityTreshold: threshold for computation of alpha angle where to identify stop of
            avalanche
        simName: str
            simulation name

    """

    indVelStart, indVelZero = getIndicesVel(pfvCrossMax, cfgPlots.getfloat('velocityThreshold'))
    indStartOfRunout = rasterTransfo['indStartOfRunout']

    # compute alpha angle based on pfvCM field
    deltaz = rasterTransfo['z'][indVelStart] - rasterTransfo['z'][indVelZero]
    deltas = rasterTransfo['s'][indVelZero] - rasterTransfo['s'][indVelStart]
    alpha = np.rad2deg(np.arctan(deltaz / deltas))
    deltazBeta = rasterTransfo['z'][indVelStart] - rasterTransfo['z'][indStartOfRunout]
    deltasBeta = rasterTransfo['s'][indStartOfRunout] - rasterTransfo['s'][indVelStart]
    beta = np.rad2deg(np.arctan(deltazBeta /  deltasBeta))
    # compute average of every barInt values to take from arrays for plotting
    barInterval = cfgPlots.getfloat('barInterval')
    barInt = int(np.floor(len(rasterTransfo['s']) / (rasterTransfo['s'][-1] / barInterval)))
    barIntStart = int(np.round(np.floor(barInt*0.5)))
    # need to append nans to make array long enough to reshape in barInt intervals
    fullInt = len(pftCrossMax) / barInt
    nansAdd = int(np.round((np.ceil(fullInt) - fullInt) * barInt))
    pftCrossMaxNans = np.concatenate((pftCrossMax, np.zeros(nansAdd)*np.nan))
    pfvCrossMaxNans = np.concatenate((pfvCrossMax, np.zeros(nansAdd) * np.nan))
    # compute average values for barInt intervals
    pftCrossMaxInt = np.average(pftCrossMaxNans.reshape(-1, barInt), axis=1)
    pfvCrossMaxInt = np.average(pfvCrossMaxNans.reshape(-1, barInt), axis=1)
    # check if arrays are of same length still - if not remove last value where nans are included
    if len(rasterTransfo['s'][barIntStart::barInt]) != len(pfvCrossMaxInt):
        pfvCrossMaxInt = pfvCrossMaxInt[:-1]
        pftCrossMaxInt = pftCrossMaxInt[:-1]
    # get thalweg coordinates
    sXY = rasterTransfo['s']
    z = rasterTransfo['z']
    # setup a colorbar for pfv cross max values
    pfvColors = [val / np.nanmax(pfvCrossMaxInt) if val != 0. else 0.0 for val in pfvCrossMaxInt]
    # ind max pfvCrossMax and max pftCrossMax
    indMPFV = np.nanargmax(pfvCrossMax)
    indMPFT = np.nanargmax(pftCrossMax)

    # initialize figure
    fig = plt.figure(figsize=(pU.figW*2, pU.figH))
    ax1 = fig.add_subplot(1, 1, 1)
    ax2 = ax1.twinx()

    # add scatter plot of velocity-Altitude field colorcoded with max peak flow velocity
    ax1.bar(rasterTransfo['s'][barIntStart::barInt], pftCrossMaxInt * 10. + z[barIntStart::barInt], width=40.,
            color=cmapCrameri.batlow.reversed()(pfvColors))
    ax1.bar(rasterTransfo['s'][barIntStart::barInt], rasterTransfo['z'][barIntStart::barInt], width=40., color='white')
    ax2.plot(rasterTransfo['s'], rasterTransfo['z'], '-y', zorder=100,
             path_effects=[pe.Stroke(linewidth=2, foreground='k'), pe.Normal()], linewidth=1, markersize=0.8,
             label='thalweg')
    # get indices where velocity is first bigger than 0 (start of velocity >0) and where velocity is again back to zero
    ax2.set_ylim([np.nanmin(z)-np.nanmax(z)*0.05, np.nanmax(z)+np.nanmax(z)*0.01])
    ax1.set_ylim([np.nanmin(z) - np.nanmax(z) * 0.05, np.nanmax(z) + np.nanmax(z) * 0.01])

    # draw the horizontal and vertical lines for angle computation
    ax2.vlines(x=sXY[indVelStart], ymin=z[indVelZero], ymax=z[indVelStart],
               color='silver', linestyle='--', linewidth=1.5, label=(r'$\Delta z = %.1f$$m$' % deltaz))
    ax2.hlines(y=z[indVelZero], xmin=sXY[indVelStart],
               xmax=sXY[indVelZero],
               color='grey', linestyle='--', linewidth=1.5, label=(r'$\Delta s_{xy} = %.1f$$m$' % deltas))

    X = [rasterTransfo['s'][indVelStart], rasterTransfo['s'][indVelZero]]
    Y = [rasterTransfo['z'][indVelStart], rasterTransfo['z'][indVelZero]]
    ax2.plot(X, Y, color='lightgrey', linestyle='--', linewidth=1.5,
             label= (r'$\alpha$=%.1f° (pfv>%s%s)' % (alpha, cfgPlots['velocityThreshold'], pU.cfgPlotUtils['unitpfv'])))

    ax2.plot(rasterTransfo['s'][indMPFV], rasterTransfo['z'][indMPFV], color='darkred', marker='.', linestyle='',
             label=(r'$maxpfv$ = %.1f$ms^{-1}$' % pfvCrossMax[indMPFV]), zorder=200)
    ax2.plot(rasterTransfo['s'][indMPFT], rasterTransfo['z'][indMPFT], color='lightcoral', marker='.', linestyle='',
             label=(r'$maxpft$ = %.1f$m$' % pftCrossMax[indMPFT]), zorder=201)

    if np.isnan(rasterTransfo['startOfRunoutAreaAngle']):
        labelBeta = ('runout area')
    else:
        labelBeta = ('runout area ' +(r'$\beta_{%.1f °}$' % rasterTransfo['startOfRunoutAreaAngle']) +
                      '\n' + r'$\beta_{angle}$=%.1f° (pfv>%s%s)' % (beta, cfgPlots['velocityThreshold'], pU.cfgPlotUtils['unitpfv']))
    ax2.fill_between(sXY[indStartOfRunout:], np.nanmin(z), np.nanmax(z),
                     facecolor=[.9, .9, .9], alpha=0.4,
                     label=labelBeta)

    # add colorbar for pft bars using pfv for colors
    sm2 = ScalarMappable(cmap=cmapCrameri.batlow.reversed(),
                        norm=plt.Normalize(np.nanmin(pfvCrossMax[barIntStart::barInt]),
                                           np.nanmax(pfvCrossMax[barIntStart::barInt])))
    sm2.set_array([])
    cax = ax1.inset_axes([1.04, 0.0, 0.025, 0.99])
    cbar2 = plt.colorbar(sm2, shrink=0.5, ax=ax1, cax=cax)
    cbar2.set_label(r'pfv [m/s]', rotation=270, labelpad=25)

    # add bar for scale reference
    if len(np.where(rasterTransfo['z'] > 0)[0]) > 1:
        asb = AnchoredSizeBar(ax1.transData,
                              40,
                              r"5 m",
                              loc='lower left',
                              color='grey',
                              pad=0.5, borderpad=0.5, sep=5,
                              size_vertical=50,
                              frameon=False)
        ax1.add_artist(asb)

    # set axes labels and title
    ax1.set_yticks([])
    ax1.set_yticklabels([])
    ax2.yaxis.tick_left()
    ax2.yaxis.set_label_position("left")
    ax2.xaxis.set_label_position("bottom")
    ax2.set_ylabel('altitude [m] (pft x 10)')
    ax1.set_xlabel('$S_{xy}$ [m]')
    plt.title('Max peak flow thickness (bars) along thalweg colored with max peak flow velocity')
    leg = plt.legend()
    textLeg = leg.get_texts()[6]
    textLeg.set_fontsize(8)

    # save and or plot
    pU.putAvaNameOnPlot(ax2, simName, date=True, color='grey', fontsize=8)
    outFileName = pathDict['projectName'] + ('_%s_thalwegAltitude' % (simName))
    pU.saveAndOrPlot(pathDict, outFileName, fig)


def getIndicesVel(pfvCM, velocityThreshold):
    """create indices of peak flow velocity cross max vector first above tresholdValue and first
        below threshold again

        Parameters
        -----------
        pfvCM: numpy array
            peak flow velocity cross max values along thalweg
        velocityThreshold: float
            threhshold value

        Returns
        --------
        indVelStart: int
            index where pfvCM first exceeds threshold
        indVelZero: int
            index where pfvCM first smaller than threshold but only further downstream than indVelStart

    """

    # get indices where velocity is first bigger than velocityThreshold (start of velocity > velocityThreshold)
    # and where velocity is again back to < velocityThreshold
    sIndex = np.nonzero(pfvCM > velocityThreshold)[0]
    if len(sIndex) == 0:
        message = 'No peak flow velocity max along thalweg found exceeding: %.2f ms-1' % velocityThreshold
        log.error(message)
        raise AssertionError(message)

    indVelStart = min(sIndex)
    indVelZero = max(sIndex)

    return indVelStart, indVelZero


def referenceLinePlot(refLineRasterXY, dem, refLine, rasterTransfo, refRasterSL, refLineSL, pathDict):
    """ Create a plot with two panels, showing reference line and corresponding raster in
    panel 1) x, y coordinates and 2) s, l coordinates

        Parameters
        ------------
        refLineRasterXY: numpy nd array
            raster created from referenceLine (which cells are affected by line)
        dem: dict
            header and raster data of dem corresponding to simulations (for extent of plots)
        refLine: dict
            info of reference line read from shp file
        rasterTransfo: dict
            aimec transformation information
        refRasterSL: numpy nd array
            reference line raster (affected cells) in s, l coordinate system
        refLineSL: dict
            information about referenceLine in s, l coordinates
        pathDict: dict
            info on aimec paths, where to save figures
    """

    # create extent for imshow plot of rasterXY
    extentCellCenters, extentCellCorners = pU.createExtentMinMax(dem['rasterData'], dem['header'], originLLCenter=True)

    # fetch info on domain transformation
    # read avaPath with scale
    xPath = rasterTransfo['x']
    yPath = rasterTransfo['y']
    # read domain boundarries with scale
    cellSizeSL = rasterTransfo['cellSizeSL']
    DBXl = rasterTransfo['DBXl']*cellSizeSL
    DBXr = rasterTransfo['DBXr']*cellSizeSL
    DBYl = rasterTransfo['DBYl']*cellSizeSL
    DBYr = rasterTransfo['DBYr']*cellSizeSL

    fig = plt.figure(figsize=(pU.figW*2, pU.figH))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    # first panel XY coordinates
    ax1.set_title('Raster XY reference %s' % refLineSL['type'])
    ax1.imshow(refLineRasterXY, cmap='Reds', origin='lower', extent=extentCellCorners, vmin=0.1)
    if refLineSL['type'] == 'poly':
        ax1.plot(refLineSL['x'], refLineSL['y'], 'r', linewidth=4, label='runout line')
    else:
        ax1.plot(refLine['x'], refLine['y'], 'r', linewidth=4, label='original %s' % refLineSL['type'])
    ax1.plot(refLineSL['x'], refLineSL['y'], 'b+', label='available in SL')
    ax1.plot(xPath, yPath, 'k--', label='thalweg')
    ax1.plot(DBXl, DBYl, 'k-', label='SL domain')
    ax1.plot(DBXr, DBYr, 'k-')
    ax1.plot([DBXl, DBXr], [DBYl, DBYr], 'k-')
    ax1.set_xlabel('x [m]')
    ax2.set_ylabel(['y [m]'])
    ax1.legend()

    # second panel SL coordinates
    ax2.set_title('Raster SL reference %s' % refLineSL['type'])
    pU.NonUnifIm(ax2, rasterTransfo['l'], rasterTransfo['s'], refRasterSL, '$L_{XY}$ (thalweg) [m]', '$S_{XY}$ (thalweg) [m]',
                     extent=[rasterTransfo['l'].min(), rasterTransfo['l'].max(), rasterTransfo['s'].min(), rasterTransfo['s'].max()], aspect='equal')
    if refLineSL['type'] == 'point':
        ax2.plot(refLineSL['l'], refLineSL['s'], 'b', linestyle='', marker='*', markersize=4, label='point in SL')
    else:
        ax2.plot(refLineSL['l'], refLineSL['s'], 'b', label='line in SL')
    ax2.legend()
    ax2.yaxis.set_inverted(True)  #

    # save figure
    outFileName = pathDict['projectName'] + '_' + 'referenceLineTransfo_%s' % refLineSL['name']
    pU.saveAndOrPlot(pathDict, outFileName, fig)


def compareRunoutLines(cfgSetup, refDataTransformed, newRaster, runoutLine, rasterTransfo, resAnalysisRow, pathDict):
    """ plot runout line of current simulation and raster as well as runout lines
    derived from reference data

        Parameters
        ------------
        cfgSetup: configparser object
            configuration settings for AIMEC
        refDataTransformed: dict
            info of all reference data sets found including derived runoutlines, points
        newRaster: dict
            raster of simulation
        runoutLine: dict
            runoutLine derived for current simulation
        rasterTransfo: dict
            information on raster transformation into s,l coordinates
        resAnalysisRow: pandas data frame series
            row of current simulation of resAnalysisDF (info on sim config, results, analysis)
        pathDict: dict
            dictionary with info on aimec paths

    """

    maskedArray = np.ma.masked_where(newRaster == 0, newRaster)
    cmapTF, _, ticks, normTF = pU.makeColorMap(pU.colorMaps[cfgSetup['runoutResType']], np.nanmin(maskedArray) ,
                                               np.nanmax(maskedArray), continuous=pU.contCmap)

    fig = plt.figure(figsize=(pU.figW, pU.figH))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_title('Runout lines')

    pU.NonUnifIm(ax1, rasterTransfo['l'], rasterTransfo['s'], maskedArray,
                '$L_{XY}$ (thalweg) [m]', '$S_{XY}$ (thalweg) [m]',
                extent=[rasterTransfo['l'].min(), rasterTransfo['l'].max(),
                 rasterTransfo['s'].min(), rasterTransfo['s'].max()], cmap=cmapTF,
                 aspect='equal')

    ax1.plot(runoutLine['l'], runoutLine['s'], 'k', label='simulation')

    # add runout lines from reference data
    colors = {'line': 'lightcoral', 'point': 'gold','poly': 'purple'}
    for ind, key in enumerate(refDataTransformed):
        ax1.plot(refDataTransformed[key]['l'], refDataTransformed[key]['s'],
                 marker='+', linestyle='-', c=colors[refDataTransformed[key]['type']], label=key)

    # add legend
    ax1.legend()

    # save figure
    outFileName = pathDict['projectName'] + '_runoutLines_%s' % resAnalysisRow['simName']
    pU.saveAndOrPlot(pathDict, outFileName, fig)


def plotRunoutLineComparisonToReference(cfgSetup, refLine, runoutLine, pathDict, simName, runoutLineNoPoints, refLineNoPoints, RMSE, diffNoNans):
    """ compare runout line from current simulation to runout line derived from reference data

        Parameters
        ------------
        refLine: dict
            dictionary with info on runout line derived from reference data set (line or polygon)
        runoutLine: dict
            dictionary with info on runout line derived from current simulation based on runoutResType and threshold
        pathDict: dict
            dictionary with info on where to save plot
        simName: str
            name of current simulation
        runoutLineNoPoints: str
            number of points where no runout line was found where points were found for refLine
            format of str: nPointsFound/numberOfAllPointsRefLine
        refLineNoPoints: str
            number of points where no reference line was found where points were found for runoutLine
            format of str: nPointsFound/numberOfAllPointsRunoutLine
        RMSE: float
            root mean squared error between s location of runout line from sim or from reference data set
            only for l coordinates where for sim and ref runout line coordinate were found
        diffNoNans: numpy array
            difference values runoutLine sim s location - refLine s location
            only for l coordinates where for sim and ref runout line coordinate were found
    """

    colors = {'line': 'lightcoral', 'point': 'gold','poly': 'purple'}
    unitResType = pU.cfgPlotUtils['unit' + cfgSetup['runoutResType']]
    displayRunoutResType = pathDict.get('displayRunoutResType', cfgSetup['runoutResType'])

    # create figure
    fig = plt.figure(figsize=(pU.figW*1.5, pU.figH))
    gs = gridspec.GridSpec(4, 2, hspace=1.5)

    # add panel one with lines and differences across thalweg Lxy
    ax1 = fig.add_subplot(gs[0:3, 0:2])
    ax1.set_title('Runout line (sim %s > %s %s) difference (from reference %s)' %
                  (displayRunoutResType, cfgSetup['thresholdValue'], unitResType, refLine['type']))
    ax2 = ax1.twinx()
    ax1.bar(refLine['l'], runoutLine['s']-refLine['s'], width=5, zorder=1)
    ax2.plot(runoutLine['l'], runoutLine['s'], label='sim', c='k', zorder=3)
    ax2.plot(refLine['l'], refLine['s'], label=('reference %s' % refLine['type']), c=colors[refLine['type']],
             zorder=2)

    # add differences as bars to panel
    boxInfo = ax2.text(0.01, 0.01, ('points not found in sim: %s \npoints not found in ref: %s \nRMSE: %.2f m' %
             (runoutLineNoPoints, refLineNoPoints, RMSE)), horizontalalignment='left',
             verticalalignment='bottom', transform=ax2.transAxes)
    boxInfo.set_bbox({'facecolor':'white', 'alpha': 0.5})
    ax2.set_ylabel('S_xy along thalweg [m]')
    ax1.set_ylabel('runout difference [m]')
    ax1.set_xlabel('L_xy across thalweg [m]')
    ax2.legend()

    # add panels for hist and CDF
    ax3 = fig.add_subplot(gs[3, 0])
    ax4 = fig.add_subplot(gs[3, 1])
    ax3.hist(diffNoNans)
    _ = sPlot.plotHistCDFDiff(diffNoNans, ax4, ax3, insert='False', title=['diff histogram',
                                                                             'diff CDF (95% and 99% centiles)'])

    # save figure
    outFileName = pathDict['projectName'] + '_runoutLinesComparison_%s_%s' % (simName, refLine['type'])
    pU.saveAndOrPlot(pathDict, outFileName, fig)


def boxScalarMeasures(pathDict, resultsDF, name='', orderList=None):
    """ create a boxplot for distribution of scalar measures deltaSXY, runoutAngle, max field values of pfv and pft

        Parameters
        -----------
        pathDict: dict
            dictionary with info on where to save plot
        resultsDF: pandas Dataframe
            one row per simulation with info on simulation results, aimec analysis and model configuration
        name: str
            name extension for plot name
        orderList: lst
            list of strings to order the appearance on x axis

    """

    # setup boxplot of resultsDF
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(pU.figW * 2.4, pU.figH * 1.))
    sns.boxplot(resultsDF, x='testCase', y='sRunout', ax=ax[0, 0], order=orderList)
    sns.boxplot(resultsDF, x='testCase', y='runoutAngle', ax=ax[0, 1], order=orderList)
    sns.boxplot(resultsDF, x='testCase', y='pfvFieldMax', ax=ax[1, 0], order=orderList)
    sns.boxplot(resultsDF, x='testCase', y='pftFieldMax', ax=ax[1, 1], order=orderList)

    # add single data points next to boxplot of distribution
    for ind2, resVal2 in resultsDF.iterrows():
        ax[0, 0].plot(resVal2['testCasePoints'], resVal2['sRunout'], 'b*')
        ax[0, 1].plot(resVal2['testCasePoints'], resVal2['runoutAngle'], 'b*')
        ax[1, 0].plot(resVal2['testCasePoints'], resVal2['pfvFieldMax'], 'b*')
        ax[1, 1].plot(resVal2['testCasePoints'], resVal2['pftFieldMax'], 'b*')

    # remove x labels
    ax[0, 0].set_xlabel('')
    ax[0, 1].set_xlabel('')
    ax[1, 0].set_xlabel('')
    ax[1, 1].set_xlabel('')

    # add y labels
    ax[1, 0].set_ylabel('max pfv [ms-1]', fontsize=14)
    ax[1, 1].set_ylabel('max pft [m]',  fontsize=14)
    ax[0, 0].set_ylabel('sRunout [m]',  fontsize=14)
    ax[0, 1].set_ylabel('runout angle [°]',  fontsize=14)

    # set tick size
    ax[1, 1].tick_params(axis='both', which='major', labelsize=14)
    ax[1, 1].tick_params(axis='both', which='minor', labelsize=14)
    ax[1, 0].tick_params(axis='both', which='major', labelsize=14)
    ax[1, 0].tick_params(axis='both', which='minor', labelsize=14)
    ax[0, 0].tick_params(axis='both', which='major', labelsize=14)
    ax[0, 0].tick_params(axis='both', which='minor', labelsize=14)
    ax[0, 1].tick_params(axis='both', which='major', labelsize=14)
    ax[0, 1].tick_params(axis='both', which='minor', labelsize=14)

    # only add ticks for each test case not for also points
    ax[0,0].set_xticks(orderList[::2])
    ax[0, 1].set_xticks(orderList[::2])
    ax[1, 0].set_xticks(orderList[::2])
    ax[1, 1].set_xticks(orderList[::2])

    plt.show()
    outFileName = 'distributionAimecScalarValues_%s' % (name)
    pU.saveAndOrPlot(pathDict, outFileName, fig)


def defineVariableType(paraVar, dataDF, pathDict):
    """define type of parameter (str or not str)
    only empty strings do not account for type str

    Parameters
    -----------
    paraVar: str
        parameter name
    dataDF: pandas Dataframe
        one row per simulation with info on simulation results, aimec analysis and model configuration
    pathDict: dict
        key colorParameter

    Returns
    --------
    firstVar: str, float, None
        variable to be checked for type
    """
    if "colorParameter" in pathDict:
        if pathDict["colorParameter"] is False:
            firstVar = None
        else:
            # get first sim value of paraVar to decide if str or float or bool
            varList = dataDF[paraVar].tolist()
            indicatorList = [True if isinstance(item, str) and not item == "" else False for item in varList]
            if np.asarray(indicatorList).any():
                firstVar = ""
            else:
                firstVar = 1.0

    return firstVar


def defineColorcodingValues(dataDF, paraVar, nSamples):
    """define list of values that are used for colorcoding of simulation results
    as thickness values are potentially read from shp file - check for e.g. relTh0 instead of relTh

    Parameters
    ------------
    dataDF: pandas Dataframe
        one row per simulation with info on simulation results, aimec analysis and model configuration
    paraVar: str
        parameter name
    nSamples: int
        number of simulation results to be analyzed

    Returns
    --------
    colorFlag: bool
        True if colorCoding shall be applied
    minVal, maxVal: float
        min and max values of values list
    values: list
        list of values used for colorcoding
    """

    colorFlag = False
    minVal = 0
    maxVal = 1
    cmapSCVals = None
    colorSuffix = ""

    values = sorted(dataDF[paraVar].to_list())
    # check if no thickness value provided (because read from file)
    if (
        paraVar in ["relTh", "entTh", "secondaryRelTh"]
        and np.asarray([True if item == "" else False for item in values]).all()
    ):
        # check if read from shapefile and therefore e.g. relTh0 is available
        if paraVar + "0" in dataDF.columns:
            values = sorted(dataDF[paraVar + "0"].to_list())
            values = [np.nan if item == "" else item for item in values]
            colorSuffix = "0"
    # check if a value is provided for every simulation
    if not np.isnan(values).any():
        minVal = np.nanmin(values)
        maxVal = np.nanmax(values)
        colorFlag = True
        cmapSCVals = np.linspace(0, 1, nSamples)

    if not colorFlag:
        values = None

    return colorFlag, colorSuffix, minVal, maxVal, values, cmapSCVals


def plotOverlap(rasterTransfo, newData, name, avaDir):
    """plot where s,l coordinate system has overlaps and where this coincides with data to be analysed

    Parameters
    -----------
    rasterTransfo: dict
        dictionary with all the information about the coordinate transformation
    newData: numpy nd array
        current data field to be analysed and already transformed in s,l coordinate system
    """

    l = rasterTransfo["l"]
    s = rasterTransfo["s"]

    cmap1, _, ticks1, norm1 = pU.makeColorMap(
        pU.colorMaps["prob"],
        np.nanmin(newData),
        np.nanmax(newData),
        continuous=pU.contCmap,
    )

    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(pU.figW * 2, pU.figH * 3))

    ax[0].set_title("Data field")
    ref0, im = pU.NonUnifIm(
        ax[0],
        s,
        l,
        np.transpose(np.where((newData > 0), newData, np.nan)),
        "$S_{XY}$ (thalweg) [m]",
        "$L_{XY}$ (thalweg) [m]",
        extent=[s.min(), s.max(), l.min(), l.max()],
    )
    ax[1].set_title("Overlap in coordinate system")
    ref1, im1 = pU.NonUnifIm(
        ax[1],
        s,
        l,
        np.transpose(np.where(rasterTransfo["intersectionPoints"], 1, np.nan)),
        "$S_{XY}$ (thalweg) [m]",
        "$L_{XY}$ (thalweg) [m]",
        extent=[s.min(), s.max(), l.min(), l.max()],
    )

    ax[2].set_title("Overlap coinciding with analysed data")
    fieldOverlap = np.where(rasterTransfo["intersectionPoints"], newData, np.nan)
    ref2, im2 = pU.NonUnifIm(
        ax[2],
        s,
        l,
        np.transpose(np.where(fieldOverlap > 0, newData, np.nan)),
        "$S_{XY}$ (thalweg) [m]",
        "$L_{XY}$ (thalweg) [m]",
        extent=[s.min(), s.max(), l.min(), l.max()],
    )

    outFileName = "ErrorPlot_domainOverlap%s" % (name.stem)
    pU.saveAndOrPlot({"pathResult": pathlib.Path(avaDir, "Outputs", "ana3AIMEC")}, outFileName, fig)
