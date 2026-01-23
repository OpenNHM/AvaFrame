"""
    Run script for running python DFA kernel
"""

import pathlib

# Local imports
from avaframe.com1DFA import com1DFA
from avaframe.com1DFA import particleTools
from avaframe.in3Utils import cfgUtils
from avaframe.in3Utils import logUtils
import avaframe.out3Plot.outCom1DFA as outCom1DFA
from avaframe.in1Data import getInput as gI
import avaframe.in2Trans.rasterUtils as IOf
import avaframe.com1DFA.particleTools as particleTools


""" run com1DFA module """

# +++++++++SETUP CONFIGURATION++++++++++++++++++++++++
# log file name; leave empty to use default runLog.log
logName = 'runCom1DFA'

# Load avalanche directory from general configuration file
cfgMain = cfgUtils.getGeneralConfig()
avalancheDir = cfgMain['MAIN']['avalancheDir']

# Start logging
log = logUtils.initiateLogger(avalancheDir, logName)
log.info('MAIN SCRIPT')
log.info('Current avalanche: %s', avalancheDir)
modName = 'com1DFA'
cfg = cfgUtils.getModuleConfig(com1DFA, avalancheDir)
# Create output and work directories
outputDir = pathlib.Path(avalancheDir, 'Outputs', modName)
inDirPart = outputDir / 'particles'
particlesList, timeStepInfo = particleTools.readPartFromPickle(inDirPart)
demFile = gI.getDEMPath(avalancheDir)
demOri = IOf.readRaster(demFile, noDataToNan=True)
cfgGen = cfg['GENERAL']
methodMeshNormal = cfg.getfloat('GENERAL', 'methodMeshNormal')
thresholdPointInPoly = cfgGen.getfloat('thresholdPointInPoly')
# -----------------------
# Initialize mesh
log.debug('Initializing Mesh')
demOri, dem = com1DFA.initializeMesh(cfgGen, demOri, methodMeshNormal)
outCom1DFA.plotParticles(particlesList, cfg, dem)
