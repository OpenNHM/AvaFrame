"""
Run a combination of the DFA kernel (com1DFA) to get an alphaBeta path
to run alphaBeta model (com2AB) to get the alpha angle
and run the DFA kernel again
"""
import time

# Local imports
# import config and init tools
import avaframe.in3Utils.initializeProject as initProj
from avaframe.in3Utils import cfgUtils
from avaframe.in3Utils import logUtils
from avaframe.com3Hybrid import com3Hybrid


# Time the whole routine
startTime = time.time()

# log file name; leave empty to use default runLog.log
logName = 'runcom3Hybrid'

# Load avalanche directory from general configuration file
cfgMain = cfgUtils.getGeneralConfig()
avalancheDir = cfgMain['MAIN']['avalancheDir']

# Start logging
log = logUtils.initiateLogger(avalancheDir, logName)
log.info('MAIN SCRIPT')
log.info('Current avalanche: %s', avalancheDir)

# ----------------
# Clean input directory(ies) of old work and output files
initProj.cleanSingleAvaDir(avalancheDir, deleteOutput=True)

# Load configuration for hybrid model
cfgHybrid = cfgUtils.getModuleConfig(com3Hybrid, avalancheDir)
com3Hybrid.maincom3Hybrid(cfgMain, cfgHybrid)
