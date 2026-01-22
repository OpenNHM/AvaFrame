"""
    Run script for computing statistics of results including the runout derived from aimec
"""

# Load modules
import os
import time
import glob
import configparser
import shutil
import pathlib
import numpy as np

# Local imports
from avaframe.out3Plot import statsPlots as sPlot
from avaframe.com1DFA import com1DFA
import avaframe.ana4Stats.probAna as probAna
from avaframe.ana4Stats import getStats
from avaframe.ana3AIMEC import ana3AIMEC, dfa2Aimec, aimecTools
from avaframe.in3Utils import initializeProject as initProj
from avaframe.in3Utils import fileHandlerUtils as fU
from avaframe.log2Report import generateReport as gR
from avaframe.out1Peak import outPlotAllPeak as oP
from avaframe.out3Plot import plotUtils
from avaframe.in3Utils import cfgUtils
from avaframe.in3Utils import logUtils


# log file name; leave empty to use default runLog.log
logName = 'runGetStats'
modName = 'com1DFA'

# Load general configuration filee
cfgMain = cfgUtils.getGeneralConfig()
flagShow = cfgMain['FLAGS'].getboolean('showPlot')

avaDir = 'data/avaHockeyChannel'
cfgMain['MAIN']['avalancheDir'] = avaDir
cfgStats = cfgUtils.getModuleConfig(getStats)
cfg = cfgStats['GENERAL']
# Clean input directory(ies) of old work and output files
initProj.cleanSingleAvaDir(avaDir)

# set output directory, first ava in list
avaDir = pathlib.Path(avaDir)
outDir = avaDir / 'Outputs' / 'ana4Stats'
cfgStats['GENERAL']['outDir'] = str(outDir)
# Specify where you want the results to be stored
fU.makeADir(outDir)

# Start logging
log = logUtils.initiateLogger(outDir, logName)

# --- run Com1DFA -----
avaName = avaDir.name
avaNameTest = avaName + 'StatsTest'
statsSimCfg = pathlib.Path('..', 'benchmarks', avaNameTest, '%sStats_com1DFACfg.ini' % (avaName))

# perform com1DFA simulations
outDir = pathlib.Path(avaDir, 'Outputs')
dem, plotDict, reportDictList, simDF = com1DFA.com1DFAMain(cfgMain, cfgInfo=statsSimCfg)

if cfg.getboolean('aimec'):

    initProj.cleanModuleFiles(avaDir, ana3AIMEC)
    # run aimec
    statsAimecCfg = pathlib.Path('..', 'benchmarks', avaNameTest, '%sStats_ana3AIMECCfg.ini' % (avaName))
    cfgAIMEC = cfgUtils.getModuleConfig(ana3AIMEC, fileOverride=statsAimecCfg)
    cfgAimecSetup = cfgAIMEC['AIMECSETUP']

    # Setup input from com1DFA
    pathDict = dfa2Aimec.mainDfa2Aimec(avaDir, cfgAimecSetup['anaMod'], cfgAIMEC)

    # TODO: define referenceFile
    pathDict['numSim'] = len(pathDict['ppr'])
    # define reference simulation
    pathDict = aimecTools.fetchReferenceSimNo(pathDict, cfgAIMEC)

    # Extract input file locations
    pathDict = aimecTools.readAIMECinputs(avaDir, pathDict, cfgAimecSetup.getboolean('defineRunoutArea'),
                                          dirName=cfgAimecSetup['anaMod'])

    # Run AIMEC postprocessing
    ana3AIMEC.mainAIMEC(pathDict, cfgAIMEC)


# ----- determine max values of peak fields
# set directory of peak files
inputDir = avaDir / 'Outputs' / modName / 'peakFiles'

# provide optional filter criteria for simulations
parametersDict = fU.getFilterDict(cfgStats, 'FILTER')

# get statisical measure of simulations
peakValues = getStats.extractMaxValues(inputDir, avaDir, cfgStats['GENERAL']['varPar'],
                      restrictType=cfgStats['GENERAL']['restrictType'], nameScenario=cfgStats['GENERAL']['nameScenario'],
                      parametersDict=parametersDict)

# log to screen
for key in peakValues:
    print('peakValues:', key, peakValues[key])

# ++++++++++++++ Plot max values +++++++++++++++++
sPlot.plotValuesScatter(peakValues, 'pft', 'pfv', cfgStats['GENERAL'], avaDir,
                        statsMeasure='max', flagShow=flagShow)
sPlot.plotValuesScatterHist(peakValues, 'pft', 'pfv', cfgStats['GENERAL'], avaDir,
                            statsMeasure='max', flagShow=flagShow, flagHue=True)

log.info('Plots have been saved to: %s' % outDir)
