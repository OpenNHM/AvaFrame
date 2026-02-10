"""
    Run script for running the dam break problem on a sloping bed and compare to simulation results
"""

# Load modules
import numpy as np
import pathlib
import pandas as pd

# Local imports
from avaframe.com1DFA import com1DFA
from avaframe.in3Utils import fileHandlerUtils as fU
from avaframe.ana1Tests import damBreak
from avaframe.in3Utils import cfgUtils
from avaframe.in3Utils import logUtils
import avaframe.out3Plot.outAna1Plots as outAna1Plots



# log file name; leave empty to use default runLog.log
logName = 'runAnalyzeDamBreakProblem'

# Load settings from general configuration file
cfgMain = cfgUtils.getGeneralConfig()
avalancheDir = 'data/avaDamBreak'
cfgMain['MAIN']['avalancheDir'] = avalancheDir

# Start logging
log = logUtils.initiateLogger(avalancheDir, logName)
log.info('MAIN SCRIPT')
log.info('Current avalanche: %s', avalancheDir)

# create output directory for test result plots
outDirTest = pathlib.Path(avalancheDir, 'Outputs', 'ana1Tests')
fU.makeADir(outDirTest)

# Load configuration
damBreakCfg = pathlib.Path(avalancheDir, 'Inputs', 'damBreak_com1DFACfg.ini')
cfg = cfgUtils.getModuleConfig(com1DFA, fileOverride=damBreakCfg)
cfgGen = cfg['GENERAL']

# Load flow thickness from analytical solution
solDam = damBreak.damBreakSol(avalancheDir, cfgMain, cfg, outDirTest)

# create dataFrame of results
simDF, _ = cfgUtils.readAllConfigurationInfo(avalancheDir)


# if the analysis already exists and you only want to replot uncomment this (and put the good result name)
# pathToResults = pathlib.Path(avalancheDir, 'Outputs', 'ana1Tests', 'results5.p')
# if pathToResults.is_file():
#     simDF = pd.read_pickle(pathToResults)
simDF = damBreak.postProcessDamBreak(avalancheDir, cfgMain, cfg, simDF, solDam, outDirTest)

# make convergence plot (if you add the fiting lines, make sure only the coloredBy and sizedBy parameters are varied)
fig1, ax1 = outAna1Plots.plotErrorConvergence(simDF, outDirTest, cfg['DAMBREAK'], 'nPart', 'vhErrorL2',
                          'aPPK', 'nPPK0', logScale=True, fit=False)

# make convergence plot
outAna1Plots.plotTimeCPULog(simDF, outDirTest, cfg['DAMBREAK'], 'nPart', 'aPPK', 'nPPK0')

# do some filtering for the presentation plot
simDF = simDF[simDF['sphKernelRadius']==3]

# make convergence plot (if you add the fiting lines, make sure only the coloredBy and sizedBy parameters are varied)
# same as plotErrorConvergence but adds the points corresponding to different coloredBy values one after the others
# and saves itermediate plots
fig1, ax1 = outAna1Plots.plotPresentation(simDF, outDirTest, cfg['DAMBREAK'], 'nPart', 'hErrorL2',
                          'aPPK', 'nPPK0', logScale=True, fit=True)
