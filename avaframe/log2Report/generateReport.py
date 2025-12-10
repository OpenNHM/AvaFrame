"""
    Generate a markdown report for data provided in dictionary

"""

# Load modules
import logging
import pathlib
import shutil
import pandas as pd
from tabulate import tabulate
from datetime import datetime

# create local logger
# change log level in calling module to DEBUG to see log messages
log = logging.getLogger(__name__)


def copyPlots2ReportDir(reportDir, plotDict):
    """ copy the plots to report directory
    The plots are in a dictionary:
    plotDict = {'plot1': PurePath to plot1, 'plot2': PurePath to plot2...}

    Parameters
    -----------
    reportDir: pathlib path
        path to directory where to copy the plots to
    plotDict: dict
        dict of the location of the plots to copy
    """

    if not isinstance(reportDir, pathlib.PurePath):
        reportDir = pathlib.Path(reportDir)
    for key in plotDict:
        if key != 'type':
            if isinstance(plotDict[key], pathlib.PurePath):
                plotPath = plotDict[key]
                plotName = reportDir / plotPath.name
                shutil.copy2(plotPath, plotName)
                log.debug('Copied: %s to %s' % (plotPath, plotName))


def addLineBlock(titleString, reportDKey, pfile, italicFont=False, onlyFirstLine=False):
    """ add lineblock to report

        Parameters
        -----------
        titleString: str
            string that shall be added
        reportDKey: dict
            dictionary with info for string
        pfile: file
        italicFont: bool
            if True write value in italic
        onlyFirstLine: bool
            if first item in reportDKey is added to pfile - break
    """

    for value in reportDKey:
        if value != 'type':
            if italicFont:
                pfile.write('%s *%s* \n' % (titleString, reportDKey[value]))
            else:
                pfile.write('%s %s \n' % (titleString, reportDKey[value]))
            pfile.write(' \n')
            if onlyFirstLine:
                break


def writeReportFile(reportD, pfile):
    """ Create markdown report with blocks, tables, list according to type key

        Parameters
        ----------
        reportD : dict
            report dictionary
        pfile : file object
            file where markdown format report is written to

    """

    # Loop through keys and perform action according to value found in type
    for key in reportD:
        subKey = 'type'
        # HEADER BLOCK
        # Title
        if reportD[key][subKey] == 'title':
            addLineBlock('#', reportD[key], pfile, onlyFirstLine=True)

        # Avalanche name
        if reportD[key][subKey] == 'avaName':
            addLineBlock('### Avalanche directory:', reportD[key], pfile, italicFont=True)

        # Simulation name
        if reportD[key][subKey] == 'simName':
            addLineBlock('### Simulation name:', reportD[key], pfile, italicFont=True)

        # Time info
        if reportD[key][subKey] == 'time':
            addLineBlock('### Date:', reportD[key], pfile, onlyFirstLine=True)

        # PARAMETER BLOCK
        # Table listing all the key : value pairs in rows
        if reportD[key][subKey] == 'list':
            pfile.write('### %s \n' % key)
            pfile.write(' \n')
            pfile.write('| Parameters | Values | \n')
            pfile.write('| ---------- | ------ | \n')
            for value in reportD[key]:
                if value != 'type':
                    pfile.write('| %s | %s | \n' % (value, reportD[key][value]))
            pfile.write(' \n')
        # Table listing the key : value pairs in columns
        if reportD[key][subKey] == 'columns':
            pfile.write('### %s \n' % key)
            pfile.write(' \n')
            for value in reportD[key]:
                if value != 'type':
                    pfile.write('| %s ' % value)
            pfile.write('| \n')
            for value in reportD[key]:
                if value != 'type':
                    pfile.write('| ----------')
            pfile.write('| \n')
            for value in reportD[key]:
                if value != 'type':
                    pfile.write('| %s ' % reportD[key][value])
            pfile.write('| \n')
            pfile.write(' \n')
        # Multiline, multicolumn table from pandas Dataframe
        if reportD[key][subKey] == 'pandasDF':
            pfile.write('### %s \n' % key)
            pfile.write(' \n')
            df = reportD[key]['dataDF']
            # only keep necessary columns
            colNames = list(reportD[key]['column names'].keys())
            df = df[colNames]
            # format floats
            dataType = df.dtypes
            for col in colNames:
                if 'float' in str(dataType[col]):
                    df[col] = df[col].map(lambda x: '{0:.4f}'.format(x))
            # rename columns
            df = df.rename(columns=reportD[key]['column names'])
            df = df.transpose()
            nCols = len(reportD[key]['column names'])
            df.to_markdown(buf=pfile, tablefmt="github", headers=["<!-- -->"]*nCols)
            pfile.write(' \n')
            pfile.write(' \n')

        # IMAGE BLOCK
        if reportD[key][subKey] == 'image':
            pfile.write('### %s \n' % key)
            pfile.write(' \n')
            for value in reportD[key]:
                if value != 'type' and isinstance(reportD[key][value], pathlib.Path):
                    pfile.write('##### Figure:   %s \n' % value)
                    pfile.write(' \n')
                    pfile.write('![%s](%s) \n' % (value, reportD[key][value].name))
                    pfile.write(' \n')
                    pfile.write(' \n')

        # TEXT BLOCK
        if reportD[key][subKey] == 'text':
            pfile.write('### %s \n' % key)
            pfile.write(' \n')
            for value in reportD[key]:
                if value != 'type':
                    pfile.write('##### Topic:  %s \n' % value)
                    pfile.write(' \n')
                    pfile.write('%s \n' % (reportD[key][value]))
                    pfile.write(' \n')


def writeReport(outDir, reportDictList, reportOneFile, plotDict='', standaloneReport=False,
                reportName='fullSimulationReport'):
    """ Write a report in markdown format for simulations, saved to outDir

        Parameters
        ----------
        outDir : str
            path to output directory
        reportDictList : list
            list of report dictionaries from simulations
        reportOneFile : boolean
            True to write all info in the same file
        plotDict : dict
            optional dictionary with info on plots that shall be included in report
        standaloneReport: bool
            if True copy plots to reportDir
        reportName: str
            report file name, fullSimulationReport is the default value

    """

    if reportOneFile:
        now = datetime.now()
        dtString = now.strftime("%Y%m%d_%Hh%Mm%Ss")
        # Start writing markdown style report
        reportPath = pathlib.Path(outDir, (reportName + '_' + dtString + '.md'))
        with open(reportPath, 'w') as pfile:

            # Loop through all simulations
            for reportD in reportDictList:

                if plotDict != '' and ('simName' in reportD) and (reportD['simName']['name'] in plotDict):
                    # add plot info to general report Dict
                    reportD['Simulation Results'] = plotDict[reportD['simName']['name']]
                    reportD['Simulation Results'].update({'type': 'image'})

                # copy plots to reportDir
                if standaloneReport:
                    for key in reportD:
                        subKey = 'type'
                        if reportD[key][subKey] == 'image':
                            copyPlots2ReportDir(outDir, reportD[key])
                # Write report file
                writeReportFile(reportD, pfile)

    else:

        # Loop through all simulations
        for reportD in reportDictList:

            if plotDict != '' and (reportD['simName']['name'] in plotDict):
                # add plot info to general report Dict
                reportD['Simulation Results'] = plotDict[reportD['simName']['name']]
                reportD['Simulation Results'].update({'type': 'image'})

            # Start writing markdown style report
            reportPath = pathlib.Path(outDir, ('%s.md' % reportD['simName']['name']))
            with open(reportPath, 'w') as pfile:

                # Write report file
                writeReportFile(reportD, pfile)


def checkAndCleanReportDictOnWinIssue872(reportDictList):
    """ Check report dict for inconsistencies in simName. Sometimes the simName is not
     properly set in the dict report. This is a dirty fix!! (see issue 872). S

        Parameters
        ----------
        reportDictList : list
            of report dictionaries from simulations

        Returns
        -------
        reportDictList : list
            checked and cleaned reportDict
    """

    for k, listItem in enumerate(reportDictList):
        simName = listItem['simName']['name']
        if '_AF_' in simName:
            nameParts = simName.split('_AF_')
        else:
            nameParts = simName.split('_')

        # This is the proper simName
        simNameClean = nameParts[0]

        if listItem['Simulation Parameters']['Release Area Scenario'] != simNameClean:
            reportDictList[k]['Simulation Parameters']['Release Area Scenario'] = simNameClean
            log.warning("Ran into windows QGis simName report problem, set release area scenario to simName.")

        if listItem['Release Area']['Release area scenario'] != simNameClean:
            reportDictList[k]['Release Area']['Release area scenario'] = simNameClean
            log.warning("Ran into windows QGis simName report problem, set release area scenario to simName.")

    return reportDictList
