"""
    plot statistics of simulations

"""

# load python modules
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pathlib
import re

# local imports
import avaframe.out3Plot.plotUtils as pU
import avaframe.in3Utils.fileHandlerUtils as fU
import avaframe.in2Trans.rasterUtils as IOf
from avaframe.in3Utils import cfgHandling
import avaframe.in1Data.getInput as gI
import avaframe.com1DFA.deriveParameterSet as dP
import avaframe.out3Plot.plotUtils as pU
import avaframe.in3Utils.geoTrans as gT


# create local logger
log = logging.getLogger(__name__)


def plotValuesScatter(
    peakValues, resType1, resType2, cfg, avalancheDir, statsMeasure="max", flagShow=False, layer=""
):
    """Produce scatter plot of statistical measures (eg. max values) (resType1 und resType2),
    for one set of simulations or multiple

    Parameters
    -----------
    peakDictList: dict
        peakValues dictionary that contain max values of peak parameters and parameter variation info
    resType1: str
        result parameter 1, 'ppr', 'pft', 'pfv'
    resType2: str
        result parameter 1, 'ppr', 'pft', 'pfv'
    cfg: dict
        configuration, for now contains output location and varPar: parameter that is varied
        to perfom a set of simulations
    statsMeasure: str
        statistical measure for plotting, options: max, mean, min, std
    flagShow: bool
        if True show plot
    layer: str
        layer info for multilayer results
    """

    varPar = cfg['varPar']
    # extract values from dictionaries
    varVal = []
    values1 = []
    values2 = []
    for key in peakValues:
        values1.append(peakValues[key][resType1][statsMeasure])
        values2.append(peakValues[key][resType2][statsMeasure])
        varVal.append(peakValues[key]['varPar'])

    log.info('Number of simulations is: %d' % (len(varVal)))

    # Get name and units for resTypes and parameter variation to annotate plots
    baseResType1 = re.sub(r"_l\d+$", "", resType1)
    baseResType2 = re.sub(r"_l\d+$", "", resType2)
    # if multilayer add layer info to label names
    if layer != "":
        suffix = " (%s)" % layer
    else:
        suffix = ""
    name1 = pU.cfgPlotUtils["name%s" % baseResType1] + suffix
    name2 = pU.cfgPlotUtils["name%s" % baseResType2] + suffix
    unit1 = pU.cfgPlotUtils["unit%s" % baseResType1]
    unit2 = pU.cfgPlotUtils["unit%s" % baseResType2]
    nameVar = cfg['varParName']
    unitVar = cfg['varParUnit']

    # if the varied parameter used for colorcoding is a string
    if isinstance(varVal[0], str):
        itemsList, ticksList, varValV = pU.getColorbarTicksForStrings(varVal)
    else:
        varValV = np.array(varVal)
        itemsList = ''

    # load variation colormap
    cmap, _, ticks, norm = pU.makeColorMap(pU.cmapVar, np.amin(varValV), np.amax(varValV), continuous=True)
    if isinstance(varVal[0], str):
        ticks = ticksList

    fig, ax = plt.subplots()
    plt.title('%s vs. %s' % (name1, name2))
    # set range and steps of colormap
    cc = varValV
    sc = ax.scatter(values1, values2, c=cc, cmap=cmap)
    ax.set_xlabel('%s %s [%s]' % (statsMeasure, name1, unit1))
    ax.set_ylabel('%s %s [%s]' % (statsMeasure, name2, unit2))
    pU.addColorBar(sc, ax, ticks, unitVar, nameVar, tickLabelsList=itemsList)
    pU.putAvaNameOnPlot(ax, avalancheDir)

    outDir = cfg['outDir']
    plotName = 'Scatter_%s_vs_%s_dist%s_%s' % (resType1, resType2, cfg['distType'], cfg['scenario'])
    # save and or show figure
    plotPath = pU.saveAndOrPlot({'pathResult': outDir}, plotName, fig)


def plotValuesScatterHist(
    peakValues,
    resType1,
    resType2,
    cfg,
    avalancheDir,
    statsMeasure="max",
    flagShow=False,
    flagHue=False,
    layer="",
):
    """Produce scatter and marginal kde plot of max values, for one set of simulations or multiple

    Parameters
    -----------
    peakValues: dict
        peakValues dictionary that contain max values of peak parameters and parameter variation info
    resType1: str
        result parameter 1, 'ppr', 'pft', 'pfv'
    resType2: str
        result parameter 1, 'ppr', 'pft', 'pfv'
    cfg: dict
        configuration, for now contains output location and varPar: parameter that is varied
        to perfom a set of simulations
    statsMeasure: str
        statistical measure for plotting, options: max, mean, min, std
    flagShow: bool
        if True show plot
    layer: str
        layer info for multilayer results

    """

    varPar = cfg['varPar']
    # extract values from dictionaries
    varVal = []
    values1 = []
    values2 = []
    scenario = []
    # if multilayer add layer info to label names
    for key in peakValues:
        values1.append(peakValues[key][resType1][statsMeasure])
        values2.append(peakValues[key][resType2][statsMeasure])
        varVal.append(peakValues[key]['varPar'])
        if 'scenario' in peakValues[key] and flagHue:
            scenario.append(peakValues[key]['scenario'])

    log.info('Number of simulations is: %d' % (len(varVal)))

    # Get name and units for resTypes and parameter variation to annotate plots
    baseResType1 = re.sub(r"_l\d+$", "", resType1)
    baseResType2 = re.sub(r"_l\d+$", "", resType2)
    if layer != "":
        suffix = " (%s)" % layer
    else:
        suffix = ""
    name1 = pU.cfgPlotUtils["name%s" % baseResType1] + suffix
    name2 = pU.cfgPlotUtils["name%s" % baseResType2] + suffix
    unit1 = pU.cfgPlotUtils["unit%s" % baseResType1]
    unit2 = pU.cfgPlotUtils["unit%s" % baseResType2]
    nameVar = cfg['varParName']
    unitVar = cfg['varParUnit']
    varValV = np.array(varVal)
    title1 = statsMeasure + ' ' + name1+' [' + unit1 + ']'
    title2 = statsMeasure + ' ' + name2+' [' + unit2 + ']'

    if flagHue:
        # create pandas data frame reqiured for seaborn jointplot
        maxVals = {name1: values1, name2: values2, 'scenario': scenario}
        maxData = pd.DataFrame(maxVals)
        fig1 = sns.jointplot(data=maxData, x=name1, y=name2, hue="scenario")
    else:
        fig1 = sns.jointplot(x=values1, y=values2)

    # add title and text box
    fig1.ax_joint.set_xlabel(title1)
    fig1.ax_joint.set_ylabel(title2)
    pU.putAvaNameOnPlot(fig1.ax_joint, avalancheDir)

    # save fig
    outDir = cfg['outDir']
    plotName = 'Scatterkde_%s_vs_%s_dist%s_%s' % (resType1, resType2, cfg['distType'], cfg['scenario'])
    # save and or show figure
    plotPath = pU.saveAndOrPlot({'pathResult': outDir}, plotName, fig1.figure)


def plotHistCDFDiff(dataDiffPlot, ax1, ax2, insert='True', title=['', '']):
    """ Produce histogram and CDF plot of the raster difference of two simulations

    Parameters
    -----------
    dataDiffPlot: 2D numpy array
        raster of the difference of the two simulations
    ax1: axes
        axes for the histogram plot
    ax2: axes
        axes for the CDF plot
    insert: boolean
        true if the plots are in inserted axes (size of the lables is then smaller)
    title: list
        if not inserts, title for the plots

    """
    # Difference between datasets
    diffMax = np.nanmax(dataDiffPlot)
    diffMin = np.nanmin(dataDiffPlot)

    sortedDiffPlot = np.sort(np.abs(dataDiffPlot))
    nSample = np.size(sortedDiffPlot)
    hist = np.array(range(nSample))/float(nSample)
    ticks = []
    centile1 = float(pU.cfg['centile1'])
    centile2 = float(pU.cfg['centile2'])
    for val in [centile1, centile2]:
        ind = np.searchsorted(hist, val)
        ind = min(ind, np.size(hist)-1)
        ax1.plot(sortedDiffPlot, hist)
        ax1.hlines(hist[ind], 0, sortedDiffPlot[ind], linestyles='--', linewidths=0.5)
        ax1.vlines(sortedDiffPlot[ind], 0, hist[ind], linestyles='--', linewidths=0.5)
        ticks.append(sortedDiffPlot[ind])

    ax2.set_xlim([-sortedDiffPlot[ind], sortedDiffPlot[ind]])
    width = diffMax - diffMin
    stepsInterval = int(pU.cfg['steps2Centile2'])
    stepWidth = 2*sortedDiffPlot[ind]/stepsInterval    # stepsInterval bins in the [-2sigma,+2sigma] interval
    if stepWidth > 0 and width > 0:
        bins = int(width/stepWidth)
    else:
        bins = 1

    # reduce bins to a sensible size
    if bins > 1000:
        bins = 1000

    ax2.hist(dataDiffPlot, bins=bins, density=True, histtype="stepfilled")
    ax2.get_yaxis().set_ticks([])
    if insert:
        ax2.tick_params(axis='x', which='major', labelsize=8, rotation=45)
        ax2.set_title(title[0])

    ticks.append(np.floor(np.nanmax(np.abs(dataDiffPlot))))
    ax1.set_xticks(ticks)
    if insert:
        ax1.tick_params(axis='both', which='major', labelsize=8, rotation=45)
        ax1.set_title(title[1])

    return ticks


def plotProbMap(avaDir, inDir, cfgFull, demPlot=False):
    """ plot probability maps including contour lines

        Parameters
        ----------
        avaDir: str
            path to avalanche directory
        inDir: str
            path to datasets that shall be plotted
        cfgFull: configParser object
            configuration settings for probAna
            keys used: name, cmapType, levels, unit
        demPlot: bool
            if True plot dem in background with contourlines for elevation that is found in avaDir/Inputs

    """

    # configuration settings
    cfg = cfgFull['PLOT']

    avaName = pathlib.PurePath(avaDir).name

    # fetch probabiltiy map datasets in inDir
    dataFiles = list(inDir.glob('*.asc')) + list(inDir.glob("*.tif"))
    if dataFiles == []:
        message = 'No probability map dataset found in: %s' % (inDir)
        log.error(message)
        raise FileNotFoundError(message)
        log.info('Probability maps found for generating plots: %d' % len(dataFiles))

    # load levels and define colors
    levels = fU.splitIniValueToArraySteps(cfg['levels'])
    multLabel = False
    if len(levels) > 2:
        multLabel = True
        cmapType = pU.cmapGreys1
        colorBackGround = 'seashell'
    else:
        cmapType = pU.colorMaps[cfg['cmapType']]
        colorsP = ['black', 'white']
        colorBackGround = 'gainsboro'
        if len(levels) == 1 and levels[0] > 0.5:
            colorsP = ['white']
    unit = cfg['unit']

    # set colors for contourlines according to levels
    labels = []
    for lev in levels:
        labels.append('%.0f %%' % (lev*100))

    # loop over all datasets and create plots
    for data in dataFiles:

        raster = IOf.readRaster(data, noDataToNan=True)
        dataPlot = raster['rasterData']
        header = IOf.readRasterHeader(data)
        cellSize = header['cellsize']

        # load correspoding DEM check for matching cellsize
        demInputs = gI.getDEMPath(avaDir)
        cfgDEM = {'GENERAL': {'avalancheDir': avaDir, 'meshCellSize': header['cellsize'],
            'meshCellSizeThreshold': cfgFull['PLOT']['meshCellSizeThreshold']}}
        pathDem = dP.checkRasterMeshSize(cfgDEM, demInputs, onlySearch=True)
        if pathDem != '':
            demFile = pathlib.Path(cfgDEM['GENERAL']['avalancheDir'], 'Inputs', pathDem)
            demData = IOf.readRaster(demFile, noDataToNan=True)
            demField = demData['rasterData']
        else:
            log.warning('No matching DEM file found for cellSize of %s - skipping DEM plot and zoom plot' % header['cellsize'])
            demPlot = False

        # Set dimensions of plots
        ny = dataPlot.shape[0]
        nx = dataPlot.shape[1]
        Ly = ny*cellSize
        Lx = nx*cellSize

        # constrain plot to where there is data
        rowsMin, rowsMax, colsMin, colsMax, dataConstrained = pU.constrainPlotsToData(dataPlot, cellSize,
            extentOption=False, constrainedData=True)
        dataPlot = np.ma.masked_where(dataConstrained == 0.0, dataConstrained)

        # create figure and add title
        fig = plt.figure(figsize=(pU.figW*2, pU.figH))

        layer = cfgFull['GENERAL'].get('layer', '')
        peakVarLabel = '%s (%s)' % (cfgFull['GENERAL']['peakVar'], layer) if layer else cfgFull['GENERAL']['peakVar']
        fullTitle = '%s %s based on %s $>$ %s %s' % (avaName,
                                                     cfg['name'],
                                                     peakVarLabel,
                                                     cfgFull['GENERAL']['peakLim'],
                                                     cfgFull['GENERAL']['unit'])
        suptitle = fig.suptitle(fullTitle, fontsize=14, color='0.5')
        ax1 = fig.add_subplot(121)

        # set extent in meters using cellSize and llcenter location
        extentCellCenters, extentCellCorners, rowsMinPlot, rowsMaxPlot, colsMinPlot, colsMaxPlot = pU.createExtent(rowsMin, rowsMax, colsMin, colsMax, header)

        # for now continuous color map is desired
        cmap, _, ticks, norm = pU.makeColorMap(cmapType, np.nanmin(dataPlot), np.nanmax(dataPlot),
            continuous=True)

        if demPlot:
            # also constrain DEM to data constrained
            demConstrained = demField[rowsMin:rowsMax+1, colsMin:colsMax+1]
            # add DEM hillshade with contour lines
            pU.addHillShadeContours(ax1, demConstrained, cellSize, extentCellCenters)
            dataPlot = np.ma.masked_where(dataConstrained == 0.0, dataConstrained)
            cmap.set_bad(alpha=0)
        else:
            cmap.set_bad(colorBackGround)

        # add data plot
        pU.checkExtentFormat(extentCellCorners, cellSize, dataPlot, plotName='probMap left panel')
        im1 = ax1.imshow(dataPlot, cmap=cmap, extent=extentCellCorners, origin='lower', aspect='equal', norm=norm,
            zorder=3)

        # create meshgrid for contour plot also constrained to where there is data
        xx = np.linspace(colsMinPlot, colsMaxPlot, dataPlot.shape[1])
        yy = np.linspace(rowsMinPlot, rowsMaxPlot, dataPlot.shape[0])
        X, Y = np.meshgrid(xx, yy)
        pU.checkMeshgridInputs(xx, yy, cellSize, dataPlot, plotName="probMap (right) zoom panel")

        # add contourlines for levels
        if multLabel:
            CS = ax1.contour(X, Y, dataPlot, levels=levels, cmap=pU.cmapT.reversed(), linewidths=1, zorder=4)
        else:
            CS = ax1.contour(X, Y, dataPlot, levels=levels, colors=colorsP, linewidths=1, zorder=4)

        pU.addColorBar(im1, ax1, ticks, unit)
        title = str('%s' % cfg['name'])
        ax1.set_xlabel('x [m]')
        ax1.set_ylabel('y [m]')

        # add zoom plot of runout area
        ax2 = fig.add_subplot(122)

        if demPlot:

            # determine zoom in runout area
            dataCut, xOrigin, yOrigin = pU.constrainToMinElevation(avaDir, raster['rasterData'], cfg, cellSize,
                extentOption=True, providedDEM=demData)

            # constrain to where there is data
            rowsMinPlot, rowsMaxPlot, colsMinPlot, colsMaxPlot, dataCutConstrained = pU.constrainPlotsToData(dataCut,
                cellSize, extentOption=True, constrainedData=True, buffer=cfg.getfloat('constrainBuffer'))
            dataCutConstrained = np.ma.masked_where(dataCutConstrained == 0.0, dataCutConstrained)

            # set extent of zoom Plot, x0, x1, y0, y1 are cell center coordinates
            x0 = xOrigin + colsMinPlot + header['xllcenter']
            x1 = xOrigin + colsMaxPlot + header['xllcenter']
            y0 = yOrigin + rowsMinPlot + header['yllcenter']
            y1 = yOrigin + rowsMaxPlot + header['yllcenter']
            # create extent for imshow plot - origin at -0.5*cellSize not at cell center
            # for x1, y1 +0.5cellSize to get outer edge of cell (upper right corner)
            extentDEMPlot = [x0-0.5*cellSize, x1+0.5*cellSize, y0-0.5*cellSize, y1+0.5*cellSize]

            # add plot
            pU.checkExtentFormat(extentDEMPlot, cellSize, dataCutConstrained, plotName='probMap (right) zoom panel')
            im2 = ax2.imshow(dataCutConstrained, cmap=cmap, extent=extentDEMPlot,
                             origin='lower', norm=norm, aspect='equal')

            # create meshgrid for contour plot also constrained to where there is data
            # use cell center coordinates here
            xx2 = np.linspace(x0, x1, dataCutConstrained.shape[1])
            yy2 = np.linspace(y0, y1, dataCutConstrained.shape[0])
            X2, Y2 = np.meshgrid(xx2, yy2)
            pU.checkMeshgridInputs(xx2, yy2, cellSize, dataCutConstrained, plotName="probMap (right) zoom panel")

            # add contourlines for levels
            if multLabel:
                CS2 = ax2.contour(X2, Y2, dataCutConstrained, levels=levels,
                                  cmap=pU.cmapT.reversed(), linewidths=1)
            else:
                CS2 = ax2.contour(X2, Y2, dataCutConstrained, levels=levels,
                                  colors=colorsP, linewidths=1)

            # Get the handles for the legend elements
            handles, _ = CS2.legend_elements()

            ax2.set_xlabel('x [m]')
            ax2.set_ylabel('y [m]')

            plt.legend(handles, labels, facecolor='black', framealpha=0.04)
            pU.addColorBar(im2, ax2, ticks, unit)

        outDir = inDir / 'plots'
        outFileName = 'probMap_' + data.stem
        pathDict = {'pathResult': outDir}
        pU.saveAndOrPlot(pathDict, outFileName, fig)


def resultHistPlot(cfg, dataDF, xName='', scenario=None, stat='count', parametersDict=''):
    """ create a histogram of values and optional colorcode using scenario name
        and option to filter simulations using parametersDict

        Parameters
        -----------
        cfg: configparser object
            configuration info here used outDir
        dataDF: dataFrame
            dataFrame with info on simulation results and configuration one line per simulation
            (e.g. aimec resAnalysisDF)
        xName: str
            column name for x axis
        scenario: str
            column name used to colorcode values
        stat: str
            statistical measure to show (percent, probability, density, count, frequency), default count
        parametersDict: dict
            optional - dictionary filter criteria, parameter name and list of values


        Returns
        --------
        plotPath: pathlib path
            path to figure

    """

    # filter DF with parametersDict
    if parametersDict != '':
        simNameList = cfgHandling.filterSims(cfg['avalancheDir'], parametersDict, specDir='', simDF=dataDF)
        dataDF = dataDF[dataDF['simName'].isin(simNameList)]

    # initialize figure
    fig, ax = plt.subplots()

    # create histogram
    bars = sns.histplot(data=dataDF, x=xName, hue=scenario, stat=stat, ax=ax)

    # create second y axis for ecdf
    ax2 = ax.twinx()
    cdf = sns.ecdfplot(data=dataDF, x=xName, hue=scenario, stat='count', ax=ax2)

    ax.set_ylabel('histogram ' + stat)
    ax2.set_ylabel('ecdf count')

    outFileName = '%s_' % xName + 'histogram'
    plotPath = pU.saveAndOrPlot({'pathResult': cfg['outDir']}, outFileName, fig)

    return plotPath


def plotDistFromDF(cfg, dataDF, name1, name2, scenario=None, parametersDict='', type=''):
    """ create a dist plot from dataframe for name1 on x axis and name2 on y axis, optionally
        colorcoded with scenario name and filtered with parametersDict

        Parameters
        -----------
        cfg: configparser object
            configuration settings here outDir
        dataDF: dataframe
            dataframe with one line per simulation and info on model parameters and results
        name1: str
            column name of dataDF to use for plot x axis
        name2: str
            column name of dataDF to use for plot y axis
        scenario: str
            optional name of column used to colorcode points in plots for type=scatter
            or kde for type=dist
        parametersDict: dict
            optional - dictionary filter criteria
        type: str
            optional - type of plot dist or scatter

        Returns
        --------
        plotPath: pathlib path
            path to figure

    """

    # filter DF using parametersDict
    if parametersDict != '':
        simNameList = cfgHandling.filterSims(cfg['avalancheDir'], parametersDict, specDir='', simDF=dataDF)
        dataDF = dataDF[dataDF['simName'].isin(simNameList)]

    # # create figure
    if scenario !='':
        if type == 'scatter':
            fig, ax = plt.subplots()
            ax = sns.scatterplot(data=dataDF[dataDF['simName'].isin(simNameList)], x=name1, y=name2, hue=scenario)
        else:
            dist = sns.jointplot(data=dataDF[dataDF['simName'].isin(simNameList)], x=name1, y=name2, hue=scenario)
            ax = dist.ax_joint
            fig = dist.fig
    else:
        dist = sns.jointplot(data=dataDF[dataDF['simName'].isin(simNameList)], x=name1, y=name2)
        ax = dist.ax_joint
        fig = dist.fig

    # put ava name on plot and save figure
    pU.putAvaNameOnPlot(ax, cfg['avalancheDir'])
    outFileName = '%s_vs_%s_' % (name1, name2) + 'distplot'
    plotPath = pU.saveAndOrPlot({'pathResult': cfg['outDir']}, outFileName, fig)

    return plotPath


def plotSample(paramValuesD, outDir, releaseScenario=''):
    """ plot the parameter sample only if two parameters are varied

        Parameters
        -----------
        paramValuesD: dict
            dictionary with parameter names and sets of values
        outDir: pathlib path
            path where to save the plot
    """

    # Figure
    fig, ax = plt.subplots(figsize=(pU.figW, pU.figH))
    plt.title('Parameter sample')
    plt.plot(paramValuesD['values'][:,0], paramValuesD['values'][:,1], 'b*')
    plt.xlabel(paramValuesD['names'][0])
    plt.ylabel(paramValuesD['names'][1])

    # put ava name on plot and save figure
    outFileName = 'ParameterSample_%s' % releaseScenario
    plotPath = pU.saveAndOrPlot({'pathResult': outDir}, outFileName, fig)


def plotThSample(simDF, name1, thName, outDir):
    """ plot the parameter sample if generated for th thickness values read from shp file
        one plot for each release Scenario using simDF

        Parameters
        ------------
        simDF: pandas DataFrame
            dataframe with one row per sim and info on sim configuration
        name1: str
            name of parameter that is varied for x-axis
        thName: str
            name of the thickness parameter (relTh, entTh, secondaryRelTh)
        outDir: pathlib path
            path to folder where to save plot

    """

    # fetch all release scenarios
    releaseScenario = simDF['releaseScenario'].values
    relSc = list(set(releaseScenario))

    # create one plot per release scenario
    for count, rel in enumerate(relSc):
        simDFPlot = simDF[simDF['releaseScenario'] == rel]
        # create names of release features
        thIds = (simDFPlot[thName + 'Id'].iloc[0]).split('|')
        thFeatures = [thName + id for id in thIds]

        # check if empty strings - possible if multiple scenarios and not all have same thId features
        # replace empty string with nans to create numpy array of type float and not string
        # required for the twinx to work properly
        simDF2 = simDFPlot.copy()
        for thF in thFeatures:
            simDF2[thF] = simDFPlot[thF].replace('', np.nan).astype(float)

        # setup figure
        fig, ax = plt.subplots(figsize=(pU.figW*1.5, pU.figH))
        fig.suptitle('Parameter sample of %s vs %s for %s' % (name1, thName, rel))
        # scatterplot for first th feature vs name1
        dist = sns.scatterplot(data=simDF2, x=name1, y= thFeatures[0], ax=ax, hue='releaseScenario')
        # position of twin axes
        distPosition = 1.

        # loop over all the other th features and add axes for them
        for count1, thFeat in enumerate(thFeatures[1:]):
            axNew = ax.twinx()
            axNew.spines["right"].set_position(("axes", distPosition))
            dist = sns.scatterplot(data=simDF2, x=name1, y=thFeat, ax=axNew, hue='releaseScenario')
            distPosition = distPosition + 0.25

        plt.show()
        # put ava name on plot and save figure
        outFileName = 'ParameterSample_%s' % rel
        plotPath = pU.saveAndOrPlot({'pathResult': outDir}, outFileName, fig)


def plotThSampleFromVals(paramValuesD, outDir):
    """ plot the parameter sample if generated for thickness values read from shp file
        one plot for each release Scenario - hence paramValuesD should be for one release scenario only

        Parameters
        ------------
        paramValuesD: dict
            dict with info on parameter variation and initial values of parameters
        outDir: pathlib path
            path to folder for saving plot

    """

    # if only two parameters are varied but one is read from shp then create a sample plot
    # fetch parameters that are varied and thickness feature names
    name1 = list(set(paramValuesD['varParNamesInitial']).symmetric_difference(set(paramValuesD['thVariationBasedOnFromShp'])))[0]
    thName = paramValuesD['thVariationBasedOnFromShp'][0]
    thFeatures = [th for th in paramValuesD['names'] if thName in th]
    name1Index = paramValuesD['names'].index(name1)

    # setup figure
    fig, ax = plt.subplots(figsize=(pU.figW*1.5, pU.figH))
    fig.suptitle('Parameter sample of %s vs %s' % (name1, thName))
    # scatterplot for first th feature vs name1
    thIndex = paramValuesD['names'].index(thFeatures[0])
    plt.plot(paramValuesD['values'][:, name1Index], paramValuesD['values'][:, thIndex], 'o')
    ax.set_ylabel('%s' % thFeatures[0])
    # position of twin axes
    distPosition = 1.

    # loop over all the other th features and add axes for them
    for count1, thFeat in enumerate(thFeatures[1:]):
        axNew = ax.twinx()
        axNew.spines["right"].set_position(("axes", distPosition))
        axNew.set_ylabel('%s' % thFeat)
        thIndex = paramValuesD['names'].index(thFeat)
        plt.plot(paramValuesD['values'][:, name1Index], paramValuesD['values'][:, thIndex], 'o')
        distPosition = distPosition + 0.25

    # put ava name on plot and save figure
    outFileName = 'ParameterSample_%s' % paramValuesD['releaseScenario']
    plotPath = pU.saveAndOrPlot({'pathResult': outDir}, outFileName, fig)
