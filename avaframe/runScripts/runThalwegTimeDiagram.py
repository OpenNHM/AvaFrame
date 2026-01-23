"""
    Run script for creatint thalweg-time diagrams
"""

import matplotlib.pyplot as plt
from cmcrameri import cm
import pathlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np

# Local imports
import avaframe.in3Utils.fileHandlerUtils as fU
from avaframe.in3Utils import cfgUtils
from avaframe.in3Utils import logUtils
import avaframe.ana5Utils.distanceTimeAnalysis as dtAna
import avaframe.in1Data.getInput as gI
import avaframe.in2Trans.rasterUtils as IOf
import avaframe.out3Plot.outDistanceTimeAnalysis as dtAnaPlots

#+++++++++++++USER INPUT++++++++
# if True use preprocessedData
preProcessedData = False
#++++++++++++++++++++++++++

# log file name; leave empty to use default runLog.log
logName = 'runRangeTimeDiagram'

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
cfgRangeTime = cfgUtils.getModuleConfig(dtAna, avalancheDir)
cfgRangeTime['GENERAL']['avalancheDir'] = avalancheDir

if preProcessedData:
    # fetch preprocesses mtiInfo dicts
    mtiInfoDicts = dtAna.importMTIData(avalancheDir, 'com1DFA', inputDir='', simHash='')

    # loop over all mtiInfo dicts found
    for mtiInfo in mtiInfoDicts:
        # log file path of info dict
        log.info('Process mtiInfo: %s' % mtiInfo['name'].stem)

        # read configuration from mtiInfo and convert to configparser
        cfgRangeTimeRead = cfgUtils.convertDictToConfigParser(mtiInfo['configurationSettings'])
        # set section GENEREAL but keep section PLOTS from module ini file to be adjustable
        cfgRangeTime['GENERAL'] = cfgRangeTimeRead['GENERAL']

        # TODO: not so nice, logged settings twice now but required here as overwrites other settings
        log.info('Settings used for analysis')
        for section in cfgRangeTime.sections():
            for key in cfgRangeTime.items(section):
                log.info('section: %s - values %s' % (section, key))

        # produce radar field of view plot if type is range time
        if mtiInfo['type'] != 'thalwegTime':
            log.warning('Found %s - but type is thalweg-time cannot produce simulated thalweg-time diagram' % mtiInfo['name'].stem)
        else:
            # create plot
            dtAnaPlots.plotRangeTime(mtiInfo, cfgRangeTime)

else:

    # fetch all flow parameter result fields
    configDir = pathlib.Path(avalancheDir, 'Outputs', 'com1DFA', 'configurationFiles')
    if (configDir.is_dir() is False) or (len(list(configDir.glob('*.ini'))) == 0):
        fU.fileNotFoundMessage(('No configuration files found in %s - consider first running avalanche simulations' %
                                configDir))

    # fetch info on available simulations
    simDF = cfgUtils.createConfigurationInfo(avalancheDir, standardCfg='', writeCSV=False, specDir='')

    # loop over all sims found in simDF
    for index in simDF.index:

        # fetch simulation dem
        # get dem dictionary - already read DEM with correct mesh cell size
        dem = gI.initializeDEM(avalancheDir, demPath=simDF['DEM'].loc[index])
        dem['originalHeader'] = dem['header'].copy()

        # create mtiInfo dict
        cfgRangeTime, mtiInfo = dtAna.createThalwegTimeInfoFromSimResults(avalancheDir,
            cfgRangeTime, 'com1DFA', index, simDF, dem)

        # create plot
        dtAnaPlots.plotRangeTime(mtiInfo, cfgRangeTime)
