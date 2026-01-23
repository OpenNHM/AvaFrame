"""
    Run script for creating configuration files with parameter variation for com4FlowPy
    Define settings in ana4Stats/probAnaCfg.ini or your local copy - local_probAnaCfg.ini
"""

# Load modules
import pathlib

# Local imports
from avaframe.com4FlowPy import com4FlowPy
from avaframe.ana4Stats import probAna
from avaframe.in3Utils import initializeProject as initProj
from avaframe.in3Utils import cfgUtils
from avaframe.in3Utils import logUtils

# log file name; leave empty to use default runLog.log
logName = 'runAna4ProbAna'

# Load general configuration filee
cfgMain = cfgUtils.getGeneralConfig()
avalancheDir = cfgMain['MAIN']['avalancheDir']
avalancheDir = pathlib.Path(avalancheDir)

# Start logging
log = logUtils.initiateLogger(avalancheDir, logName)
log.info('MAIN SCRIPT')
log.info('Current avalanche: %s', avalancheDir)

# Clean input directory(ies) of old work files
initProj.cleanSingleAvaDir(avalancheDir, deleteOutput=False)

# Load configuration file for probabilistic run and analysis
cfgProb = cfgUtils.getModuleConfig(probAna, avalancheDir)

# create configuration files for com1DFA simulations including parameter
# variation - defined in the probabilistic config
# prob4AnaCfg.ini or its local copy
cfgFiles, cfgPath = probAna.createComModConfig(cfgProb, avalancheDir, com4FlowPy)



# loop over all cfgFiles
for index, cfgF in enumerate(cfgFiles):
    # read configuration
    cfgFromFile = cfgUtils.getModuleConfig(com4FlowPy, avalancheDir, fileOverride=cfgF, toPrint=False)

    log.info('exp: %s, alpha: %s' % (cfgFromFile['GENERAL']['exp'], cfgFromFile['GENERAL']['alpha']))