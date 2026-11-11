""" functions for plotting ama event data and geometry info """

from shapely import wkb, LineString, Point, length
import shapely as sp
from shapely.ops import split
import numpy as np
import pandas as pd
import geopandas
import pathlib
import seaborn as sns
import configparser
import matplotlib.pyplot as plt

# local imports
from avaframe.in3Utils import cfgUtils
from avaframe.in3Utils import logUtils
import avaframe.in3Utils.fileHandlerUtils as fU
import avaframe.in3Utils.geoTrans as gT
import avaframe.out3Plot.plotUtils as pU


def plotPathAngle(dbData, cfgMain, nameEvent, tickOrigin='orig-transit'):
    """ make a two panel plot of avalanche thalweg, release point, runout point,
        origin, transit and deposition point
        and xy distances and computed angles

        Parameters
        -----------
        dbData: pandas dataframe
            dataframe with geometry info of events
        cfgMain: configparser object
            config settings here used: avalancheDir, projstr,
        nameEvent: str
            name of xy distance lines and angles
        tickOrigin: str
            origin for panel 2 Sxy distance origin

    """

    # fetch setup info
    avalancheDir = pathlib.Path(cfgMain['MAIN']['avalancheDir'])
    projstr = cfgMain['MAIN']['projstr']

    # loop over all events in dbData
    for index, row in dbData.iterrows():

        fig = plt.figure(figsize=(pU.figW*3, pU.figH))
        fig.suptitle('%s (%s) analysis in %s' % (row['path_name'], row['event_id'], projstr))

        # panel 1: plot map view of path, release, event point and snapped points
        ax1 = fig.add_subplot(121)
        ax1.set_title('event on thalweg in xy')
        ax1.plot(row['geom_path_ln3d_%s_resampled' % projstr].xy[0], row['geom_path_ln3d_%s_resampled' % projstr].xy[1],
            'b', linestyle='dashed', label='thalweg ($S_{xy}$)')
        ax1.plot(row['%s_Line' % nameEvent].xy[0], row['%s_Line' % nameEvent].xy[1], 'b-',
            label=nameEvent)
        ax1.plot(row['geom_origin_pt3d_%s' % projstr].x, row['geom_origin_pt3d_%s' % projstr].y,
            '*', color='lightgray', markersize=12, label='origin pt')
        ax1.plot(row['geom_transit_pt3d_%s' % projstr].x, row['geom_transit_pt3d_%s' % projstr].y,
            '*', color='silver', markersize=12, label='transit pt')
        ax1.plot(row['geom_runout_pt3d_%s' % projstr].x, row['geom_runout_pt3d_%s' % projstr].y,
            '*', color='gray', markersize=12, label='deposition pt')
        ax1.plot(row['geom_rel_event_pt3d_%s' % projstr].x, row['geom_rel_event_pt3d_%s' % projstr].y,
            'c+', markersize=20, label='release pt')
        ax1.plot(row['geom_rel_event_pt3d_%s_snapped' % projstr].x, row['geom_rel_event_pt3d_%s_snapped' % projstr].y,
            'c*', markersize=14, label='snapped')
        ax1.plot(row['geom_event_pt3d_%s' % projstr].x, row['geom_event_pt3d_%s' % projstr].y,
            'r+', markersize=20, label='runout pt')
        ax1.plot(row['geom_event_pt3d_%s_snapped' % projstr].x, row['geom_event_pt3d_%s_snapped' % projstr].y,
            'r*', markersize=14, label='snapped')
        ax1.set_xlabel('x (m)')
        ax1.set_ylabel('y (m)')
        ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax1.set_aspect('equal')

        # panel 2 plot xy distance and elevation drop and add angle info
        ax21 = fig.add_subplot(122)
        ax21.set_title('runout length in $S_{xy}$: %.1f (m), altitude drop: %.1f (m)' % (row['%s_Distance' % nameEvent], row['%s_LineAltDrop' % nameEvent]))

        z2coor = np.asarray([coord[2] for coord in row['geom_path_ln3d_%s' % projstr].coords])
        ax21.plot(row['%s_PathDist' % nameEvent], z2coor, '--', color='lightgray', label='thalweg')

        # origin  to deposition
        nameEvent = 'orig-depo'
        zcoor = [coord[2] for coord in row['%s_Line' % nameEvent].coords]
        ax21.plot(np.asarray(row['%s_LineStart' % nameEvent])+row['%s_LineDist' % nameEvent], zcoor,
            '-', color='lightgray', linewidth=4, alpha=1, label=nameEvent)
        ax21.plot([row['%s_LineStart' % nameEvent]+row['%s_LineDist' % nameEvent][0],
            row['%s_LineStart' % nameEvent]+row['%s_LineDist' % nameEvent][-1]],
            [row['%s_Line' % nameEvent].coords[0][2],row['%s_Line' % nameEvent].coords[-1][2]],
            '--', color='lightgray', linewidth=4, alpha=1., label='beta line [%.1f°]' % row['%s_Angle' % nameEvent])

        # origin point - transit point
        nameEvent = 'orig-transit'
        zcoor = [coord[2] for coord in row['%s_Line' % nameEvent].coords]
        ax21.plot(np.asarray(row['%s_LineStart' % nameEvent])+row['%s_LineDist' % nameEvent], zcoor,
            '-', color='gray', linewidth=2.5, alpha=1., label=nameEvent)
        ax21.plot([row['%s_LineStart' % nameEvent]+row['%s_LineDist' % nameEvent][0],
            row['%s_LineStart' % nameEvent]+row['%s_LineDist' % nameEvent][-1]],
            [row['%s_Line' % nameEvent].coords[0][2],
            row['%s_Line' % nameEvent].coords[-1][2]], '--', color='gray', linewidth=3, alpha=1,
            label='theta line [%.1f°]' % row['%s_Angle' % nameEvent])

        # travel length (release event)
        nameEvent = 'rel-runout'
        zcoor = [coord[2] for coord in row['%s_Line' % nameEvent].coords]
        ax21.plot(np.asarray(row['%s_LineStart' % nameEvent])+row['%s_LineDist' % nameEvent], zcoor, 'r-',
            label=nameEvent)
        ax21.plot([row['%s_LineStart' % nameEvent]+row['%s_LineDist' % nameEvent][0],row['%s_LineStart' % nameEvent]+row['%s_LineDist' % nameEvent][-1]],
            [row['%s_Line' % nameEvent].coords[0][2],row['%s_Line' % nameEvent].coords[-1][2]], 'r--',
            label='alpha line [%.1f°]' % row['%s_Angle' % nameEvent])

        # add vertical horizontal helper lines for release and runout point
        xRel = row['%s_LineStart' % nameEvent]+row['%s_LineDist' % nameEvent][0]
        ax21.plot([xRel]*100, np.linspace(np.amin(z2coor), zcoor[0], 100), 'r', linestyle='dotted',
            alpha=0.5)
        xRun = row['%s_LineStart' % nameEvent]+row['%s_LineDist' % nameEvent][-1]
        ax21.plot([xRun]*100, np.linspace(np.amin(z2coor), zcoor[-1], 100), 'r', linestyle='dotted',
            alpha=0.5)
        xPath = row['%s_PathDist' % nameEvent]
        ax21.plot(np.linspace(xPath[0], xRel, 100), [zcoor[0]]*100, 'r', linestyle='dotted',
            alpha=0.5)
        ax21.plot(np.linspace(xPath[0], xRun, 100), [zcoor[-1]]*100, 'r', linestyle='dotted',
            alpha=0.5)

        # set tick Origin for x-axis in Sxy plot
        axTicks = ax21.get_xticks()
        tickSpacing = axTicks[1]- axTicks[0]
        xTicks = axTicks + row['%s_LineStart' % tickOrigin]
        ticksFromOrig = xTicks - row['%s_LineStart' % tickOrigin]
        ticksFromOrigStr = [('%.0f' % tF) for tF in ticksFromOrig]
        ax21.set_xticks(xTicks[1:-1])
        ax21.set_xticklabels(ticksFromOrigStr[1:-1])

        ax21.set_xlabel('$S_{xy}$ (m)')
        ax21.set_ylabel('altitude (m)')
        ax21.set_aspect('equal')
        ax21.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        # save figure
        outFile = ('%s_%s_analysis' % (row['path_name'], row['event_id']))
        plotPath = pU.saveAndOrPlot({'pathResult': avalancheDir}, outFile, fig)


def plotHist(dbData, name, cfgMain, colorcode=''):
    """ create a histogram of name column of dbData

        Parameters
        -----------
        dbData: pandas dataframe
            dataframe with geometry info of events and name_Line and name_Angle
        name: str
            name of Line and Angle to plot
        cfgMain: configparser
            configuration settings
        colorcode: str
            name of column to colorcode data

    """

    # fetch setup info
    avalancheDir = pathlib.Path(cfgMain['MAIN']['avalancheDir'])
    projstr = cfgMain['MAIN']['projstr']
    nameAngle = cfgMain['PLOT']['name_' + name]

    fig = plt.figure(figsize=(pU.figW, pU.figH))
    ax3 = plt.subplot(111)

    if colorcode == '':
        sns.histplot(data=dbData, x='%s_Angle' % name)
    else:
        sns.histplot(data=dbData, x='%s_Angle' % name, hue=colorcode)

    ax3.set_title('Histogram of %s angle' % nameAngle)
    ax3.text(0.05, 0.05, ('%s events \nmean %s angle: %.1f°' %
        (len(dbData['%s_Angle' % name]), nameAngle, np.nanmean(dbData['%s_Angle' % name]))),
        transform=ax3.transAxes, fontsize=12, horizontalalignment='left',
        verticalalignment='bottom', alpha=0.8, bbox=dict(boxstyle="round", ec=(1., 1., 1.),
        fc=(1., 1., 1.), alpha=0.5))
    ax3.set_xlabel('angle (°)')

    # save figure
    outFile = ('histogram_%s_angle' % (nameAngle))
    plotPath = pU.saveAndOrPlot({'pathResult': avalancheDir}, outFile, fig)


def plotBoxPlot(dbData, colList, outDir, namePlot, renameCols=[]):
    """ create a boxplot of all columns of dbData

        Parameters
        -----------
        dbData: pandas dataframe
            dataframe with geometry info
        colList: list
            names of columns in dbData to plot
        outDir: pathlib path or str
            path to folder where plot shall be saved to
        namePlot: str
            name of plot to be added to boxplot

    """

    fig = plt.figure(figsize=(pU.figW, pU.figH))
    ax3 = plt.subplot(111)
    sn = sns.boxplot(data=dbData[colList])
    ax3.set_title('Distribution of %s for %d events' % (namePlot, len(dbData[colList[0]])))
    ax3.set_ylabel(namePlot)
    if renameCols != []:
        sn.set_xticklabels(renameCols,rotation=90)
    else:
        sn.set_xticklabels(sn.get_xticklabels(),rotation=90)

    # save figure
    outFile = ('boxplot_%s' % namePlot)
    plotPath = pU.saveAndOrPlot({'pathResult': outDir}, outFile, fig)
