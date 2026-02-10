"""
    Run script for running python DFA kernel
"""
import pathlib
import time
import numpy as np

# Local imports
import avaframe.in3Utils.initializeProject as initProj
import avaframe.in3Utils.fileHandlerUtils as fU
from avaframe.in1Data import getInput as gI
from avaframe.com1DFA import com1DFA, particleTools
import avaframe.ana1Tests.FPtest as FPtest

# from avaframe.DFAkernel.setParam import *
from avaframe.in3Utils import cfgUtils
from avaframe.in3Utils import logUtils

# +++++++++SETUP CONFIGURATION++++++++++++++++++++++++
# log file name; leave empty to use default runLog.log
logName = 'runFlatPlaneTest'

# Load avalanche directory from general configuration file
cfgMain = cfgUtils.getGeneralConfig()
avalancheDir = 'data/avaFPtest'
cfgMain['MAIN']['avalancheDir'] = avalancheDir

# Clean input directory(ies) of old work and output files
initProj.cleanModuleFiles(avalancheDir, com1DFA)

# Start logging
log = logUtils.initiateLogger(avalancheDir, logName)
log.info('MAIN SCRIPT')
log.info('Current avalanche: %s', avalancheDir)

# Load configuration
FPCfg = pathlib.Path(avalancheDir, 'Inputs', 'FlatPlane_com1DFACfg.ini')
cfg = cfgUtils.getModuleConfig(com1DFA, fileOverride=FPCfg)
cfgGen = cfg['GENERAL']
cfgFP = cfg['FPSOL']

# for timing the sims
startTime = time.time()
# create output directory for test result plots
outDirTest = pathlib.Path(avalancheDir, 'Outputs', 'ana1Tests')
fU.makeADir(outDirTest)

# Define release thickness distribution
demFile = gI.getDEMPath(avalancheDir)
relDict = FPtest.getReleaseThickness(avalancheDir, cfg, demFile)
relTh = relDict['relTh']

# call com1DFA to perform simulation - provide configuration file and release thickness function
dem, plotDict, reportDictList, simDF = com1DFA.com1DFAMain(cfgMain, cfgInfo=FPCfg)
Particles, Tsave = particleTools.readPartFromPickle(avalancheDir, simName='', flagAvaDir=True, comModule='com1DFA')
relDict['dem'] = dem

# +++++++++POSTPROCESS++++++++++++++++++++++++
# option for user interaction
if cfgFP.getboolean('flagInteraction'):
    showPlot = True
    value = input("give time step to plot (float in s):\n")
else:
    showPlot = cfgMain['FLAGS'].getboolean('showPlot')
    value = cfgFP.getfloat('dtSol')
try:
    value = float(value)
except ValueError:
    value = 'n'
while isinstance(value, float):
    ind_t = min(np.searchsorted(Tsave, value), len(Tsave)-1)

    # fetch fields for desired time step
    timeStep = Tsave[ind_t]
    fields, fieldHeader, timeList = com1DFA.readFields(avalancheDir, ['FT'], simName='',
        flagAvaDir=True, comModule='com1DFA', timeStep=timeStep)

    # get particle parameters
    comSol = FPtest.postProcessFPcom1DFA(cfgGen, Particles[ind_t], fields[0], ind_t, relDict)
    comSol['outDirTest'] = outDirTest
    comSol['showPlot'] = showPlot
    comSol['Tsave'] = Tsave[ind_t]

    # make plot
    # TODO interaction
    FPtest.plotProfilesFPtest(cfg, ind_t, relDict, comSol)

    # option for user interaction
    if cfgFP.getboolean('flagInteraction'):
        value = input("give time step to plot (float in s):\n")
        try:
            value = float(value)
        except ValueError:
            value = 'n'
    else:
        value = 'n'
