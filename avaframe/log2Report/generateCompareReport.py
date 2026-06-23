"""
    Generate a markdown report for data provided in dictionary with format to compare two simulations
"""

# Load modules
import os
import glob
import logging
import numpy as np
import shutil
import pathlib
from avaframe.in3Utils import fileHandlerUtils as fU

# create local logger
# change log level in calling module to DEBUG to see log messages
log = logging.getLogger(__name__)


def copyQuickPlots(avaName, testName, outDir, plotListRep, rel=''):
    """ copy the quick plots to report directory """

    if not isinstance(outDir, pathlib.PurePath):
        outDir = pathlib.Path(outDir)
    plotPaths = {}
    for key in plotListRep:
        plotName = outDir / ('%s_%s_%s.png' % (testName, rel, key))
        shutil.copy2(plotListRep[key], plotName)
        log.debug('Copied: %s to %s' % (plotListRep[key], plotName))
        plotPaths[key] = plotName

    return plotPaths


def copyAimecPlots(plotFiles, testName, outDir, plotPaths, rel=''):
    """ copy the aimec plots to report directory """

    if not isinstance(outDir, pathlib.PurePath):
        outDir = pathlib.Path(outDir)
    for pDict in plotFiles:
        for key in pDict:
            pDict2 = pDict[key]
            for key2 in pDict2:
                if isinstance(pDict2[key2], pathlib.PurePath):
                    name = pathlib.Path(pDict2[key2]).name
                    plotName = outDir / ('%s_%s_%s' % (testName, rel, name))
                    shutil.copy2(pDict2[key2], plotName)
                    log.info('Copied: %s to %s' % (pDict2[key2], plotName))
                    plotPaths[key] = plotName

    return plotPaths


def makeLists(simDict, benchDict):
    """ Create a list for table creation """

    parameterList = []
    valuesSim = []
    valuesBench = []
    for key in simDict:
        if key != 'type':
            if simDict[key] == '' and benchDict.get(key) is None:
                log.debug('this parameter is not used: %s' % key)
            else:
                parameterList.append(key)
                valuesSim.append(simDict[key])
                if key in benchDict:
                    valuesBench.append(benchDict[key])
                else:
                    valuesBench.append('non existent')

    return parameterList, valuesSim, valuesBench


def writeCompareReport(reportFile, reportD, benchD, avaName, cfgRep):
    """ Write a report in markdown format of the comparison of simulations to reference simulation results;
        report is saved to location of reportFile


        Parameters
        ----------
        reportFile : str
            path to report file
        reportD : dict
            report dictionary with simulation info
        benchD : str
            dictionary with simulation info of refernce simulation
        avaName : str
            name of avalanche
        cfgRep : dict
            dictionary with configuration info
    """

    # Set limit to produce warning for differences in peak fields
    diffLim = cfgRep['GENERAL'].getfloat('diffLim')
    perDiff = cfgRep['GENERAL'].getfloat('perDiff')

    # Start writing markdown style report
    with open(reportFile, 'a') as pfile:

        # HEADER BLOCK
        # Simulation name
        pfile.write('## *%s* \n' % avaName)
        simName = reportD['simName']['name']
        pfile.write('### Simulation name: *%s* \n' % reportD['simName']['name'])
        if benchD['simName']['name'] != simName:
            textString = '<span style="color:red"> Reference simulation name is different: \
                         %s  </span>' % benchD['simName']['name']
            pfile.write('#### %s \n' % textString)
        pfile.write(' \n')

        # TEST Info
        pfile.write('#### Test Info \n')
        pfile.write(' \n')
        for value in benchD['Test Info']:
            if value != 'type':
                # pfile.write('##### Topic:  %s \n' % value)
                pfile.write('%s \n' % (benchD['Test Info'][value]))
                pfile.write(' \n')

        # Create lists to write tables
        parameterList, valuesSim, valuesBench = makeLists(reportD['Simulation Parameters'], benchD['Simulation Parameters'])

        # PARAMETER BLOCK
        # Table listing all the simulation parameters
        pfile.write('#### Simulation Parameters \n')
        pfile.write(' \n')
        pfile.write('| Parameter | Reference | Simulation | \n')
        pfile.write('| --------- | --------- | ---------- | \n')
        countValue = 0
        for value in parameterList:
            # if reference and simulation parameter value do not match
            # mark red
            if valuesBench[countValue] != valuesSim[countValue]:
                valueRed = '<span style="color:red"> %s </span>' % valuesSim[countValue]
                pfile.write('| %s | %s | %s | \n' % (value, valuesBench[countValue], valueRed))
                countValue = countValue + 1
            else:
                pfile.write('| %s | %s | %s | \n' % (value, valuesBench[countValue], valuesSim[countValue]))
                countValue = countValue + 1

        # Add time needed for simulation to table
        pfile.write(' \n')
        pfile.write(' \n')

        # Create lists to write tables for aimec results
        parameterList, valuesSim, valuesBench = makeLists(reportD['Aimec analysis'], benchD['Aimec analysis'])

        # PARAMETER BLOCK
        # Table listing all the simulation parameters
        pfile.write('#### Aimec Analysis \n')
        pfile.write(' \n')
        pfile.write('| Characteristics | Reference | Simulation | \n')
        pfile.write('| --------- | --------- | ---------- | \n')
        countValue = 0
        for value in parameterList:
            # if reference and simulation parameter value do not match
            # mark red
            if float(valuesBench[countValue]) == 0.0 and float(valuesSim[countValue]) == 0.0:
                diffPercent = 0.0
            elif float(valuesBench[countValue]) == 0.0 or float(valuesSim[countValue]) == 0.0:
                diffPercent = 'undefined'
            else:
                diffPercent = ((float(valuesBench[countValue]) - float(valuesSim[countValue])) /
                              float(valuesBench[countValue]))
            if diffPercent == 'undefined' or abs(diffPercent) < perDiff:
                pfile.write('| %s | %.1f | %.1f | \n' % (value, valuesBench[countValue], valuesSim[countValue]))
                countValue = countValue + 1
            else:
                valueRed = '<span style="color:red"> %.1f </span>' % valuesSim[countValue]
                pfile.write('| %s | %.1f | %s | \n' % (value, valuesBench[countValue], valueRed))
                countValue = countValue + 1

        pfile.write(' \n')
        pfile.write(' \n')

        # IMAGE BLOCK
        pfile.write('#### Comparison Plots \n')
        pfile.write(' \n')
        for value in reportD['Simulation Results']:
            if value != 'type':
                pfile.write('##### Figure:   %s \n' % value)
                pfile.write(' \n')
                if value == 'ppr' or value == 'pft' or value == 'pfv':
                    maxDiff = max(abs(float(reportD['Simulation Difference'][value][2])),
                                  abs(float(reportD['Simulation Difference'][value][0])))
                    maxVal = float(reportD['Simulation Stats'][value][0])
                    if maxDiff < (-1.0*diffLim*maxVal) or maxDiff > (diffLim*maxVal):
                        textString1 = ('<span style="color:red"> Warning absolute difference \
                                      exceeds the tolerance of %.0f percent of %s-max value (%.2f) \
                                      </span>' % (100.*diffLim, value, maxVal))
                        textString2 = ('<span style="color:red"> Difference is: Max = %0.2f, \
                                      Mean = %.02f and Min = %.02f </span>' %
                                      (reportD['Simulation Difference'][value][0],
                                      reportD['Simulation Difference'][value][1],
                                      reportD['Simulation Difference'][value][2]))
                        pfile.write(' %s \n' % textString1)
                        pfile.write(' %s \n' % textString2)
                        pfile.write(' \n')
                pfile.write('![%s](%s) \n' % (value, reportD['Simulation Results'][value].name))
                pfile.write(' \n')
                pfile.write(' \n')

        pfile.write(' \n')
        pfile.write(' \n')

    log.info('Standard tests performed - Report saved to: %s ' % reportFile)


def collectComparisonWarnings(reportD, benchD, cfgRep):
    """ Collect the comparison warnings that writeCompareReport flags in red.

        Returns a structured list of warnings (one dict per flagged item) instead
        of writing markdown. Mirrors the flag logic of writeCompareReport so the
        summary stays consistent with the rendered report.

        Parameters
        ----------
        reportD : dict
            report dictionary with simulation info
        benchD : dict
            dictionary with reference simulation info
        cfgRep : dict
            dictionary with configuration info (uses GENERAL.diffLim and
            GENERAL.perDiff)

        Returns
        -------
        warnings : list[dict]
            each dict has 'category' and 'field'; see categories below
    """

    diffLim = cfgRep['GENERAL'].getfloat('diffLim')
    perDiff = cfgRep['GENERAL'].getfloat('perDiff')
    warnings = []

    # Parameter mismatches: skip metadata keys that are expected to differ
    paramIgnore = ['type', 'Program version']
    simParams = reportD['Simulation Parameters']
    benchParams = benchD['Simulation Parameters']
    for key in simParams:
        if key in paramIgnore:
            continue
        refVal = benchParams.get(key, 'non existent')
        if simParams[key] == '' and benchParams.get(key) is None:
            continue
        if refVal != simParams[key]:
            warnings.append(
                {'category': 'parameter', 'field': key, 'ref': refVal, 'sim': simParams[key]})

    # Aimec percentage differences
    aimecIgnore = ['type']
    simAimec = reportD['Aimec analysis']
    benchAimec = benchD['Aimec analysis']
    for key in simAimec:
        if key in aimecIgnore:
            continue
        refVal = float(benchAimec[key])
        simVal = float(simAimec[key])
        if refVal == 0.0 and simVal == 0.0:
            diffPercent = 0.0
        elif refVal == 0.0 or simVal == 0.0:
            diffPercent = 'undefined'
        else:
            diffPercent = (refVal - simVal) / refVal
        if diffPercent != 'undefined' and abs(diffPercent) >= perDiff:
            warnings.append(
                {'category': 'aimec', 'field': key, 'ref': refVal, 'sim': simVal,
                 'diffPercent': diffPercent})

    # Plot absolute differences (ppr/pft/pfv)
    simDiff = reportD['Simulation Difference']
    simStats = reportD['Simulation Stats']
    for var in ['ppr', 'pft', 'pfv']:
        if var not in simDiff:
            continue
        maxDiff, meanDiff, minDiff = (float(v) for v in simDiff[var])
        maxVal = float(simStats[var][0])
        # largest absolute deviation exceeds the tolerance
        if max(abs(maxDiff), abs(minDiff)) > diffLim * maxVal:
            warnings.append(
                {'category': 'plot', 'field': var,
                 'maxDiff': maxDiff, 'meanDiff': meanDiff, 'minDiff': minDiff, 'maxVal': maxVal})

    return warnings
