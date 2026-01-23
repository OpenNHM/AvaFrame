"""Run script for module template"""

# Local imports
from avaframe.tmp1Ex import tmp1Ex
from avaframe.in3Utils import cfgUtils
from avaframe.in3Utils import logUtils

# log file name; leave empty to use default runLog.log
logName = 'runTmp1Ex'

# Load avalanche directory from general configuration file
cfgMain = cfgUtils.getGeneralConfig()
avalancheDir = cfgMain['MAIN']['avalancheDir']

# Start logging
log = logUtils.initiateLogger(avalancheDir, logName)
log.info('MAIN SCRIPT')
log.info('Current avalanche: %s', avalancheDir)

# Load all input Parameters from config file
# get the configuration of an already imported module
# Write config to log file
cfg = cfgUtils.getModuleConfig(tmp1Ex, avalancheDir)

# Different ways to call functions
tmp1Ex.tmp1ExMain(cfg)
