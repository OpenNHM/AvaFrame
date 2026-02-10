"""
    Run ana3AIMEC
"""
# Load modules
import logging
# Local imports
from avaframe.ana3AIMEC import dfa2Aimec, ana3AIMEC, aimecTools
from avaframe.in3Utils import initializeProject as iP
from avaframe.in3Utils import cfgUtils
from avaframe.in3Utils import logUtils


# create local logger
log = logging.getLogger(__name__)

# -----------Required settings-----------------
# log file name; leave empty to use default runLog.log
logName = 'runAna3AIMEC'

# ---------------------------------------------


def runAna3AIMEC(avalancheDir, cfg, inputDir='', demFileName=''):
    """ run script for AIMEC analysis
    proceeds to AIMEC analysis and produces plots and reports
    """

    rasterTransfo, resAnalysisDF, plotDict, _, pathDict = ana3AIMEC.fullAimecAnalysis(avalancheDir, cfg,
                                                                                      inputDir=inputDir, demFileName=demFileName)

    return pathDict, rasterTransfo, resAnalysisDF, plotDict


if __name__ == '__main__':
    # Load avalanche directory from general configuration file
    cfgMain = cfgUtils.getGeneralConfig()
    avalancheDir = cfgMain['MAIN']['avalancheDir']

    # Start logging
    log = logUtils.initiateLogger(avalancheDir, logName)
    log.info('MAIN SCRIPT')
    log.info('Current avalanche: %s', avalancheDir)

    # Load all input Parameters from config file
    # get the configuration of an already imported module
    # write config to log file
    cfg = cfgUtils.getModuleConfig(ana3AIMEC, avalancheDir)

    iP.cleanModuleFiles(avalancheDir, ana3AIMEC)

    # write configuration to file
    cfgUtils.writeCfgFile(avalancheDir, ana3AIMEC, cfg)
    runAna3AIMEC(avalancheDir, cfg)
