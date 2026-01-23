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
import pandas as pd
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
from avaframe.runScripts import runAna3AIMEC as rAimec


# log file name; leave empty to use default runLog.log
logName = 'runStatsPlots'
modName = 'com1DFA'

# Load general configuration filee
cfgMain = cfgUtils.getGeneralConfig()
flagShow = cfgMain['FLAGS'].getboolean('showPlot')

avaDir = cfgMain['MAIN']['avalancheDir']
cfgStats = cfgUtils.getModuleConfig(getStats, avaDir)
cfg = cfgStats['GENERAL']

# set output directory, first ava in list
outDir = os.path.join(avaDir, 'Outputs', 'ana4Stats')
cfgStats['GENERAL']['outDir'] = outDir
cfgStats['GENERAL']['avalancheDir'] = avaDir

# Specify where you want the results to be stored
fU.makeADir(outDir)

# Start logging
log = logUtils.initiateLogger(outDir, logName)

# requires aimec output resAnalysisDF files if not already create set aimec flag to True
if cfgStats['GENERAL'].getboolean('aimec'):
    # clean all existing aimec files first
    initProj.cleanModuleFiles(avaDir, ana3AIMEC)
    # fetch config for aimec
    cfgAIMEC = cfgUtils.getModuleConfig(ana3AIMEC, avaDir)
    # run aimec
    pathDict, rasterTransfo, resAnalysisDF, plotDict = rAimec.runAna3AIMEC(avaDir, cfgAIMEC)

# load results from aimec
pathToCsv = pathlib.Path(avaDir, 'Outputs', 'ana3AIMEC', modName)
resAnalysisDFFiles = list(pathToCsv.glob('*resAnalysisDF.csv'))
if len(resAnalysisDFFiles) > 1:
    log.warning('Multiple resAnalysisDF files found, taking only first one: %s' % resAnalysisDFFiles[0])
elif len(resAnalysisDFFiles) == 0:
    message = 'No resAnalysisDF file found'
    log.error(message)
    raise FileNotFoundError

# load dataframe
resAnalysisDF = pd.read_csv(resAnalysisDFFiles[0])

# filter simulations
parametersDict = fU.getFilterDict(cfgStats, 'FILTER')

# ------- runout histogram plot
sPlot.resultHistPlot(cfgStats['GENERAL'], resAnalysisDF, xName='sRunout', scenario=None,
    stat='probability', parametersDict=parametersDict)

# -------- max result values scatter plot
sPlot.plotDistFromDF(cfgStats['GENERAL'], resAnalysisDF, name1='pftFieldMax',
    name2='pfvFieldMax', scenario=None, parametersDict=parametersDict, type='')
