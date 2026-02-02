"""
    2D regional thalweg plot
"""
import logging

from avaframe.in3Utils import cfgUtils
from avaframe.in3Utils import logUtils
from avaframe.ana5Utils import regionalThalweg2DPlot

log = logging.getLogger(__name__)
logName = 'runRegionalThalweg2DPlot'


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
    cfg = cfgUtils.getModuleConfig(regionalThalweg2DPlot)


    regionalThalweg2DPlot.regionalThalweg2DPlotMain(avalancheDir, cfg)
