"""
    Pytest for generate Report

    This file is part of Avaframe.

 """

#  Load modules
import numpy as np
import os
from avaframe.log2Report import generateReport as gR

import pytest
import configparser
import pathlib


def test_addLineBlock(tmp_path):
    """ test adding a line block """

    # setup required input
    reportDKey = {'type': 'class', 'name': 'TestName'}
    testReport = pathlib.Path(tmp_path, 'testReport.md')
    with open(testReport, 'w') as pfile:

        # call function to be tested
        gR.addLineBlock('##Hallo: ', reportDKey, pfile)
        pfile.close()

    reportFile = open(testReport, 'r')
    lines = reportFile.readlines()
    lineVals = []
    for line in lines:
        lineVals.append(line)

    assert lineVals[0] == '##Hallo:  TestName \n'


def test_writeReport(tmp_path):
    """ Test generation markdown report file """

    # Initialise input dictionary
    plot0 = pathlib.Path('release1HS2_entres_dfa_0.750_pft.png')
    testDict = {'title info': {'name': 'This is my report title', 'type': 'title',
                'testNo': 'this should not appear in the report if working fine!'},
        'avaName': {'type': 'avaName', 'name': 'data/avaTest'},
        'time': {'type': 'time', 'time': '1/01/2021 12:46:56'},
        'simName': {'type': 'simName', 'name': 'thisIsMySimulation1'},
        'Testing Parameters': {'release area': 'release1HS2', 'Mu': '0.155', 'type': 'list'},
        'Release area': {'release area scenario': 'my test release area', 'type': 'columns'},
        'Special Input': {'Additional snow-covered area': 'Area 1', 'type': 'columns'},
        'Additional Info': {'type': 'text',
                            'Report Info': 'This simulations were performed with a developement version.'},
        'Fancy plots': {'type': 'image',
                        'Peak Pressure Field of my test': plot0}}

    # make a list of input dictionaries
    reportDictList = [testDict]

    # initialise test directory
    reportOneFile = True
    plot1 = pathlib.Path('testPath', 'pft.png')
    plotDict = {'thisIsMySimulation1': {'pft': plot1},
                'simulationNumber2': {'pft': plot1}}
    # Call function to be tested
    gR.writeReport(tmp_path, reportDictList, reportOneFile, plotDict=plotDict, standaloneReport=False,
                   reportName='fullSimulationReport')

    # Load simulation report
    reportFile1P = pathlib.Path(tmp_path)
    reportFile1 = list(reportFile1P.glob('fullSimulationReport*.md'))[0]
    with open(reportFile1, 'r') as pfile:
        lines = pfile.readlines()
        lineVals = []
        for line in lines:
            lineVals.append(line)

#    print(lineVals)

    # Test
    assert lineVals[0] == '# This is my report title \n'
    assert lineVals[2] == '### Simulation directory: *data/avaTest* \n'
    assert lineVals[4] == '### Date: 1/01/2021 12:46:56 \n'
    assert lineVals[11] == '| ---------- | ------ | \n'
    assert lineVals[12] == '| release area | release1HS2 | \n'
    assert lineVals[23] == '| Additional snow-covered area | \n'
    assert lineVals[24] == '| ----------| \n'
    assert lineVals[-3] == '![pft](pft.png) \n'
    assert lineVals[37] == '![Peak Pressure Field of my test](release1HS2_entres_dfa_0.750_pft.png) \n'

    # call function to be tested
    # make a list of input dictionaries
    testDict2 = testDict
    testDict2['simName']['name'] = 'simulationNumber2'
    reportDictList = [testDict, testDict]
    reportOneFile = False
    gR.writeReport(tmp_path, reportDictList, reportOneFile, plotDict=plotDict, standaloneReport=False,
                   reportName='simulationNumber2')

    # Load simulation report
    reportFile2 = pathlib.Path(tmp_path, 'simulationNumber2.md')
    with open(reportFile2, 'r') as pfile:
        lines = pfile.readlines()
        lineVals2 = []
        for line in lines:
            lineVals2.append(line)
#        print('lineVals2', lineVals2)

        assert lineVals2[0] == '# This is my report title \n'
        assert lineVals2[2] == '### Simulation directory: *data/avaTest* \n'
        assert lineVals2[4] == '### Date: 1/01/2021 12:46:56 \n'
        assert lineVals2[11] == '| ---------- | ------ | \n'
        assert lineVals2[12] == '| release area | release1HS2 | \n'
        assert lineVals2[23] == '| Additional snow-covered area | \n'
        assert lineVals2[24] == '| ----------| \n'
        assert lineVals2[-3] == '![pft](pft.png) \n'
        assert lineVals2[37] == '![Peak Pressure Field of my test](release1HS2_entres_dfa_0.750_pft.png) \n'


def test_copyPlots2ReportDir(tmp_path):
    """ Test generation markdown report file """

    # Initialise input dictionary
    dirPath = pathlib.Path(__file__).parents[0]
    pathPlot = dirPath / 'data' / 'release1HX_ent_dfa_3f8d2d9327_pft.png'
    plotDict = {'pft': pathPlot, 'notAPlot': 1}
    outDir = pathlib.Path(tmp_path)
    # Call function to be tested
    gR.copyPlots2ReportDir(outDir, plotDict)
    assert (outDir / 'release1HX_ent_dfa_3f8d2d9327_pft.png').is_file()
