"""
    Run script for running the variations tests
"""

# Load modules
import time
import pathlib

# Local imports
from avaframe.com1DFA import com1DFA
from avaframe.ana1Tests import testUtilities as tU
from avaframe.log2Report import generateReport as gR
from avaframe.log2Report import generateCompareReport
from avaframe.ana3AIMEC import ana3AIMEC, dfa2Aimec, aimecTools
from avaframe.out3Plot import outQuickPlot
from avaframe.out1Peak import outPlotAllPeak as oP
from avaframe.in3Utils import fileHandlerUtils as fU
from avaframe.in3Utils import initializeProject as initProj
from avaframe.in3Utils import cfgUtils
from avaframe.in3Utils import cfgHandling
from avaframe.in3Utils import logUtils
from benchmarks import simParametersVar

# +++++++++REQUIRED+++++++++++++
# Which result types for comparison plots
outputVariable = ['ppr', 'pft', 'pfv']
# aimec settings
aimecResType = 'ppr'
aimecThresholdValue = '1'
aimecDiffLim = '5'
aimecContourLevels = '1|3|5|10'
aimecFlagMass = 'False'
aimecComModules = 'benchmarkReference|com1DFA'
# ++++++++++++++++++++++++++++++


# TODO: needs to be updated and benchmarks too


# log file name; leave empty to use default runLog.log
logName = 'runVariationsTestsCom1DFA'

# Load settings from general configuration file
cfgMain = cfgUtils.getGeneralConfig()

# load all benchmark info as dictionaries from description files
testDictList = tU.readAllBenchmarkDesDicts(info=False)

# filter benchmarks for a tag
type = 'TAGS'
valuesList = ['varParTest']
testList = tU.filterBenchmarks(testDictList, type, valuesList, condition='and')

# Set directory for full standard test report
outDir = pathlib.Path.cwd() / 'tests' / 'reportsVariationsCom1DFA'
fU.makeADir(outDir)

# Start writing markdown style report for standard tests
reportFile = outDir / 'variationTestsReportCom1DFA.md'
with open(reportFile, 'w') as pfile:
    # Write header
    pfile.write('# Variation Tests Report \n')
    pfile.write('## Compare com1DFA simulations to benchmark results \n')

log = logUtils.initiateLogger(outDir, logName)
log.info('The following benchmark tests will be fetched ')
for test in testList:
    log.info('%s' % test['NAME'])

for test in testList:

    # Start logging
    avaDir = test['AVADIR']

    # Fetch benchmark test info
    benchDict = simParametersVar.fetchBenchParameters(test['NAME'])
    simNameRef = test['simNameRef']
    simNameRefTest = simNameRef.replace('ref', 'dfa')
    refDir = pathlib.Path('..', 'benchmarks', test['NAME'])
    simType = benchDict['simType']
    rel = benchDict['Simulation Parameters']['Release Area Scenario']

    # Clean input directory(ies) of old work and output files
    initProj.cleanSingleAvaDir(avaDir)

    # Load input parameters from configuration file for standard tests
    # write config to log file
    avaName = pathlib.Path(avaDir).name
    standardCfg = refDir / ('%sVarPar_com1DFACfg.ini' % test['AVANAME'])
    modName = 'com1DFA'

    # Set timing
    startTime = time.time()
    # call com1DFA run
    dem, plotDict, reportDictList, simDF = com1DFA.com1DFAMain(cfgMain, cfgInfo=standardCfg)
    endTime = time.time()
    timeNeeded = endTime - startTime
    log.info(('Took %s seconds to calculate.' % (timeNeeded)))

    # Fetch correct reportDict according to simType and release area scenario
    # read all simulation configuration files and return dataFrame and write to csv
    parametersDict = {'simTypeActual': simType, 'releaseScenario': rel}
    simNameComp = cfgHandling.filterSims(avaDir, parametersDict)
    if len(simNameComp) > 1:
        log.error('more than one matching simulation found for criteria! ')
    else:
        simNameComp = simNameComp[0]

    # find report dictionary corresponding to comparison simulation
    reportD = {}
    for dict in reportDictList:
        if simNameComp in dict['simName']['name']:
            reportD = dict
    if reportD == {}:
        message = 'No matching simulation found for reference simulation: %s' % simNameRef
        log.error(message)
        raise ValueError(message)
    log.info('Reference simulation %s and comparison simulation %s ' % (simNameRef, simNameComp))

    # set result files directory
    compDir = pathlib.Path(avaDir, 'Outputs', modName, 'peakFiles')

    # Add info on run time
    reportD['runTime'] = timeNeeded

    # +++++++Aimec analysis
    # load configuration
    aimecCfg = refDir / ('%s_AIMECCfg.ini' % test['AVANAME'])
    if aimecCfg.is_file():
        cfgAimec = cfgUtils.getModuleConfig(ana3AIMEC, avaDir, fileOverride=aimecCfg)
    else:
        cfgAimec = cfgUtils.getDefaultModuleConfig(ana3AIMEC)
    cfgAimec['AIMECSETUP']['resType'] = aimecResType
    cfgAimec['AIMECSETUP']['thresholdValue'] = aimecThresholdValue
    cfgAimec['AIMECSETUP']['diffLim'] = aimecDiffLim
    cfgAimec['AIMECSETUP']['contourLevels'] = aimecContourLevels
    cfgAimec['FLAGS']['flagMass'] = aimecFlagMass
    cfgAimec['AIMECSETUP']['comModules'] = aimecComModules
    cfgAimec['AIMECSETUP']['testName'] = test['NAME']

    # Setup input from com1DFA and reference
    pathDict = []
    inputsDF, pathDict = dfa2Aimec.dfaBench2Aimec(avaDir, cfgAimec, simNameRef, simNameComp)
    pathDict['refSimulation'] = inputsDF.index[0]
    log.info('reference file comes from: %s' % pathDict['refSimulation'])

    # Extract input file locations
    pathDict = aimecTools.readAIMECinputs(avaDir, pathDict, cfgAimec['AIMECSETUP'].getboolean('defineRunoutArea'),
                                          dirName=simNameComp)

    # perform analysis
    rasterTransfo, resAnalysisDF, aimecPlotDict = ana3AIMEC.mainAIMEC(pathDict, inputsDF, cfgAimec)

    # add aimec results to report dictionary
    reportD, benchDict = ana3AIMEC.aimecRes2ReportDict(resAnalysisDF, reportD, benchDict,
        pathDict['refSimulation'])
    # +++++++++++Aimec analysis

    # Create plots for report
    # Load input parameters from configuration file
    cfgRep = cfgUtils.getModuleConfig(generateCompareReport, avaDir)

    plotListRep = {}
    reportD['Simulation Difference'] = {}
    reportD['Simulation Stats'] = {}

    # Plot data comparison for all output variables defined in suffix
    for var in outputVariable:
        plotDict = outQuickPlot.quickPlotBench(avaDir, simNameRef, simNameComp, refDir, compDir, cfgMain, var)
        for plot in plotDict['plots']:
            plotListRep.update({var: plot})
            reportD['Simulation Difference'].update({var: plotDict['difference']})
            reportD['Simulation Stats'].update({var: plotDict['stats']})

    # copy files to report directory
    plotPaths = generateCompareReport.copyQuickPlots(avaName, test['NAME'], outDir, plotListRep)
    aimecPlots = [aimecPlotDict['slCompPlot'], aimecPlotDict['areasPlot']]
    plotPaths = generateCompareReport.copyAimecPlots(aimecPlots, test['NAME'], outDir, plotPaths)

    # add plot info to general report Dict
    reportD['Simulation Results'] = plotPaths

    # write report
    generateCompareReport.writeCompareReport(reportFile, reportD, benchDict, avaName, cfgRep)
