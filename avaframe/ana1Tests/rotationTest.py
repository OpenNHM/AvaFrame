"""
Rotation test

This module runs a DFA simulation for different grid alignments
and compares the results
"""
import logging
import shutil
from datetime import datetime

# Local imports
# import config and init tools
from avaframe.in3Utils import fileHandlerUtils as fU
from avaframe.in3Utils import cfgUtils
from avaframe.log2Report import generateReport as gR
from avaframe.in3Utils import geoTrans as gT
import avaframe.in2Trans.rasterUtils as IOf
from avaframe.version import getVersion
# import analysis modules
from avaframe.ana1Tests import energyLineTest
# import plotting module
import avaframe.out3Plot.plotUtils as pU
# create local logger
log = logging.getLogger(__name__)


def mainRotationTest(avalancheDir, energyLineTestCfg, com1DFACfg, dem, simDF, resTypeList, flagMass, refSimRowHash,
                     comModule):
    """This is the core function of the Rotation Test module

    This module runs the energy line test and rotates the simulation results for the aimec analysis

    Parameters
    ----------
    avalancheDir : pathlib path
        path to avalanche directory
    energyLineTestCfg : configParser
        energy line test configuration object
    com1DFACfg : configParser
        com1DFA configuration object
    dem : dict
        dem dictionary
    simDF : dataFrame
        DFA simulation dataFrame
    resTypeList : list
        list of result types available
    flagMass : boolean
        should the aimec mass analysis be done
    refSimRowHash : str
        row index of the reference simulation in the simDF
    comModule : str
        computation module used for the DFA simulation

    Returns
    -------
    simDF : dataFrame
        DFA simulation dataFrame updated with the energy line test results
    flagMass : boolean
        should the aimec mass analysis be done (switched to true if it is an entrainment simulation)
    """
    # read energy line config. ToDo should we only read the DFAPath config and remove the energy line one?
    inDir = avalancheDir / 'Outputs' / 'com1DFA'
    outDir = avalancheDir / 'Outputs' / 'com1DFARotated'
    fU.makeADir(outDir)
    # get reference angle
    simName = simDF.loc[refSimRowHash, 'simName']
    relName = cfgUtils.parseSimName(simName)["releaseName"]
    thetaRef = float(relName[3:])
    for rowSimHash in simDF.index:
        # rotate results to be able to proceed to the aimec analysis
        simName = simDF.loc[rowSimHash, 'simName']
        # activate mass analysis if it is an entrainment simulation
        # (note this can be forced to be true with the flagMass)
        if 'ent' in simDF.loc[rowSimHash, 'simType']:
            flagMass = True
        # copy mass file
        massFile = list(inDir.glob('mass_*' + simName + '*.txt'))
        shutil.copy(massFile[0], outDir)
        # rotate simulation results (will be the input for the AIMEC analysis)
        rotateDFAResults(avalancheDir, simDF, rowSimHash, resTypeList, thetaRef, comModule)

        # make energy line analysis
        resultEnergyTest, savePath = energyLineTest.mainEnergyLineTest(avalancheDir, energyLineTestCfg, com1DFACfg,
                                                                       simName, dem)
        simDF.loc[rowSimHash, 'zEnd'] = resultEnergyTest['zEnd']
        simDF.loc[rowSimHash, 'sEnd'] = resultEnergyTest['sEnd']
        simDF.loc[rowSimHash, 'runoutAngle'] = resultEnergyTest['runoutAngle']
        simDF.loc[rowSimHash, 'energyLinePlotPath'] = savePath
    # now compare energy line results of rotated sims to the reference sim
    for rowSimHash in simDF.index:
        simDF.loc[rowSimHash, 'zDiff'] = simDF.loc[rowSimHash, 'zEnd'] - simDF.loc[refSimRowHash, 'zEnd']
        simDF.loc[rowSimHash, 'sDiff'] = simDF.loc[rowSimHash, 'sEnd'] - simDF.loc[refSimRowHash, 'sEnd']
        simDF.loc[rowSimHash, 'runoutAngleDiff'] = simDF.loc[rowSimHash, 'runoutAngle'] - simDF.loc[refSimRowHash,
                                                                                                    'runoutAngle']

    return simDF, flagMass


def rotateDFAResults(avalancheDir, simDF, rowSimHash, resTypeList, thetaRef, comModule):
    """Rotate the DFA results

    Parameters
    ----------
    avalancheDir : pathlib path
        path to avalanche directory
    simDF : dataFrame
        DFA simulation dataFrame
    rowSimHash : str
        row index (in the simDF dataframe) of the simulation to analyze
    resTypeList : list
        list of result types available
    thetaRef : float
        angle of the path of the reference simulation
    comModule : str
        computation module used for the DFA simulation

    Returns
    -------
    saves the rotated rasters to avalancheDir / 'Outputs' / comModule + 'Rotated' / 'peakFiles'
    """
    log.debug('Rotating simulation: %s' % rowSimHash)
    simName = simDF.loc[rowSimHash, 'simName']
    relName = cfgUtils.parseSimName(simName)["releaseName"]
    theta = float(relName[3:])
    simDF.loc[rowSimHash, 'relAngle'] = theta
    thetaRot = theta - thetaRef
    for resType in resTypeList:
        fName = simDF.loc[rowSimHash, resType]
        rasterDict = IOf.readRaster(fName)
        rotatedRaster = gT.rotateRaster(rasterDict, thetaRot, deg=True)
        comRotated = comModule + 'Rotated'
        outDir = avalancheDir / 'Outputs' / comRotated / 'peakFiles'
        fU.makeADir(outDir)
        outFileName = outDir / fName.stem
        IOf.writeResultToRaster(rotatedRaster['header'], rotatedRaster['rasterData'], outFileName, flip=False)


def initializeRotationTestReport(avalancheDir, resTypeList, comModule, refSimName, flagMass):
    """Initialize dictionary that is used for markdown report generation for rotation test

    Parameters
    ----------
    avalancheDir : pathlib path
        path to avalanche directory
    resTypeList : list
        list of result types available
    comModule : str
        computation module name
    refSimName : str
        name of the reference simulation in the simDF
    flagMass : boolean
        Was a mass analysis conducted?

    Returns
    -------
    reportRotationTest : dict
        report dictionary
    """
    dateTimeInfo = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    reportRotationTest = {'headerLine': {'type': 'title', 'title': 'Rotation Test for DFA Simulation'},
                          'avaName': {'type': 'avaName', 'name': str(avalancheDir)},
                          'time': {'type': 'time', 'time': dateTimeInfo},
                          'Simulation Parameters': {
                          'type': 'list',
                          'Program version': getVersion(),
                          'DFA module': comModule,
                          'Reference simulation': refSimName},
                          'Rotation test input simulations': {
                          'type': 'pandasDF', 'column names': {'simName': 'Simulation name',
                                                               'releaseScenario': 'Release name',
                                                               'relAngle': 'Angle °'}},
                          'Rotation test Energy line result table': {
                          'type': 'pandasDF', 'column names': {'simName': 'Simulation name', 'relAngle': 'Angle °',
                                                               'sDiff': 's Diff [m]',
                                                               'zDiff': 'z Diff [m]',
                                                               'runoutAngleDiff': 'angle Diff [°]'}},
                          'Rotation test Energy line figures': {'type': 'image'},
                          'Rotation test AIMEC result table': {
                          'type': 'pandasDF', 'column names': {'simName': 'Simulation name', 'relAngle': 'Angle °',
                                                               'sRunout': 'Runout [m]',
                                                               'TP': 'True positive area [-]',
                                                               'FP': 'False positive area [-]',
                                                               'FN': 'False negative area [-]'}},
                          'Rotation test AIMEC figures': {'type': 'image'}}
    if flagMass:
        reportRotationTest['Rotation test AIMEC result table']['column names']['relMass'] = 'release mass [kg]'
        reportRotationTest['Rotation test AIMEC result table']['column names']['finalMass'] = 'final mass [kg]'
        reportRotationTest['Rotation test AIMEC result table']['column names']['entMass'] = 'entrained mass [kg]'
    resTypeList = list(set(resTypeList).intersection(['ppr', 'pfv', 'pft']))
    for resType in resTypeList:
        colName = 'max%sCrossMax' % resType
        colLabel = 'Maximum %s' % resType
        unit = pU.cfgPlotUtils['unit' + resType]
        colLabel = '%s [%s]' % (colLabel, unit)
        reportRotationTest['Rotation test AIMEC result table']['column names'][colName] = colLabel
    return reportRotationTest


def buildRotationTestReport(avalancheDir, reportRotationTest, simDF, resAnalysisDF, aimecPlotDict, flagMass):
    """Write the rotation test report

    Parameters
    ----------
    avalancheDir : pathlib path
        path to avalanche directory
    reportRotationTest : dict
        report dictionary
    simDF : dataFrame
        DFA simulation dataFrame
    resAnalysisDF : dataFrame
        aimec results dataFrame
    aimecPlotDict : dict
        aimec plots dictionary
    flagMass : boolean
        Was a mass analysis conducted?

    Returns
    -------
    generates the markDown report
    """
    outDir = avalancheDir / 'Outputs' / 'ana1Tests' / 'rotationTest'
    fU.makeADir(outDir)
    energyLinePlotsDF = simDF.set_index('simName')['energyLinePlotPath']
    energyLinePlotsDict = energyLinePlotsDF.to_dict()
    inPlotsDict = {'energyLinePlots': energyLinePlotsDict}
    inPlotsDict.update(aimecPlotDict['slCompPlot'])
    inPlotsDict.update(aimecPlotDict['areasPlot'])
    if flagMass:
        inPlotsDict.update(aimecPlotDict['massAnalysisPlot'])

    simDF = simDF.sort_values(by=['relAngle'], ascending=True)
    resAnalysisDF = resAnalysisDF.sort_values(by=['relAngle'], ascending=True)

    reportRotationTest['Rotation test input simulations']['dataDF'] = simDF
    reportRotationTest['Rotation test Energy line result table']['dataDF'] = simDF
    reportRotationTest['Rotation test Energy line figures'].update(inPlotsDict['energyLinePlots'])
    reportRotationTest['Rotation test AIMEC result table']['dataDF'] = resAnalysisDF
    reportRotationTest['Rotation test AIMEC figures'].update(
        inPlotsDict['Aimec comparison of mean and max values along path'])
    reportRotationTest['Rotation test AIMEC figures'].update(inPlotsDict['Aimec area analysis'])

    if flagMass:
        reportRotationTest['Rotation test AIMEC figures'].update(inPlotsDict['Aimec mass analysis'])
    # add energy line and aimec results table to report
    gR.writeReport(outDir, [reportRotationTest], True, plotDict='', standaloneReport=True,
                   reportName='rotationTestReport')
