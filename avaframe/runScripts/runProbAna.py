"""
    Run script for performing an avalanche simulation with parameter variation and performing a probability analysis with the simulation results
"""

# Load modules
import time
import glob
import pathlib

# Local imports
from avaframe.com1DFA import com1DFA
from avaframe.out3Plot import statsPlots as sP
from avaframe.out3Plot import outQuickPlot as oP
from avaframe.ana4Stats import probAna
from avaframe.in3Utils import initializeProject as initProj
from avaframe.in3Utils import cfgUtils
from avaframe.in3Utils import logUtils
import avaframe.in3Utils.fileHandlerUtils as fU


# log file name; leave empty to use default runLog.log
logName = 'runCom1DFAandProbAna'

# Load general configuration filee
cfgMain = cfgUtils.getGeneralConfig()

# Define avalanche directories for tests
avalancheDirectories = ['data/avaHockeyChannel']

# run simulations sequentially
for avaDir in avalancheDirectories:

    # Start logging
    log = logUtils.initiateLogger(avaDir, logName)
    log.info('MAIN SCRIPT')
    log.info('Current avalanche: %s', avaDir)
    cfgMain['MAIN']['avalancheDir'] = avaDir

    # Load input parameters from configuration file
    # write config to log file
    avaDir = pathlib.Path(avaDir)
    avaName = avaDir.name
    probAnaCfg = pathlib.Path('..', 'benchmarks', '%sStatsTest' % avaName, '%sProbAna_probAnaCfg.ini' % avaName)
    # Clean input directory(ies) of old work and output files
    initProj.cleanSingleAvaDir(avaDir)

    # Load configuration file for probabilistic run and analysis
    cfgProb = cfgUtils.getModuleConfig(probAna, avaDir, fileOverride=probAnaCfg)

    # create configuration files for com1DFA simulations including parameter
    # variation - defined in the probabilistic config
    # prob4AnaCfg.ini or its local copy
    cfgFiles, cfgPath = probAna.createComModConfig(cfgProb, avaDir, com1DFA)

    # perform com1DFA simulations
    outDir = pathlib.Path(avaDir, 'Outputs')
    dem, plotDict, reportDictList, simDF = com1DFA.com1DFAMain(cfgMain, cfgInfo=cfgPath)

    # check if sampling strategy is from full sample - then only one configuration is possible
    probabilityConfigurations = probAna.fetchProbConfigs(cfgProb['PROBRUN'])

    # perform pobability analysis
    for probConf in probabilityConfigurations:

        # filter simulations according to probability configurations
        cfgProb['FILTER'] = probabilityConfigurations[probConf]
        log.info('Perform proba analysis for configuration: %s' % probConf)
        # provide optional filter criteria for simulations
        parametersDict = fU.getFilterDict(cfgProb, 'FILTER')

        # perform probability analysis
        anaPerformed, contourDict = probAna.probAnalysis(avaDir,
                                                         cfgProb,
                                                         'com1DFA',
                                                         parametersDict=parametersDict,
                                                         probConf=probConf
                                                         )
        if anaPerformed is False:
            log.warning('No files found for configuration: %s' % probConf)


        # make a plot of the contours
        inputDir = pathlib.Path(avaDir, 'Outputs', 'ana4Stats')
        outName = '%s_prob_%s_%s_lim%s' % (str(avaDir.stem),
                                           probConf,
                                           cfgProb['GENERAL']['peakVar'],
                                           cfgProb['GENERAL']['peakLim'])
        pathDict = {'pathResult': str(inputDir / 'plots'),
                    'avaDir': str(avaDir),
                    'plotScenario': outName
                    }
        oP.plotContours(contourDict,
                        cfgProb['GENERAL']['peakVar'],
                        cfgProb['GENERAL']['peakLim'], pathDict,
                        layer=cfgProb['GENERAL'].get('layer', ''))

    # plot probability maps
    sP.plotProbMap(avaDir, inputDir, cfgProb, demPlot=True)
