"""
    Run script for creating a range-time diagram from avalanche simulations
"""

import pathlib
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
comModule = 'com1DFA'
#++++++++++++++++++++++++++++++
# IF comModule != com1DFA
# required simulation results: FV or FT fields for desired time steps (e.g every second) in avalancheDir/Outputs/comModule/peakFiles/timeSteps
# in addition, peak files (pft or pfv) for each simulation are required in avalancheDir/Outputs/comModule/peakFiles
# file name format A_B_C_D_E., where:
# A - releaseAreaScenario: refers to the name of the release shape file
# B - simulationID: needs to be unique for the respective simulation
# C - simType: refers to null (no entrainment, no resistance), ent (with entrainment), res (with resistance), entres (with entrainment and resistance)
# D - modelType: can be any descriptive string of the employed model (here dfa for dense flow avalanche)
# E - result type: is pft (peak flow thickness) and pfv (peak flow velocity)

# if not com1DFA was used to create FV or FT fields, provide here a path to the DEM file that has been used to perform to the simulations
# path needs to be relative to avalancheDir/Inputs
demPath = ""
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

    log.info('Distance -time analysis performed with preprocessed data')
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
        if mtiInfo['type'] != 'rangeTime':
            log.warning('Found %s - but type is thalweg-time cannot produce simulated range-time diagram' % mtiInfo['name'].stem)
        else:
            # create plot of range distance already masked with radar field of view
            dtAnaPlots.radarFieldOfViewPlot(mtiInfo['radarFov'], mtiInfo['rangeMasked'], cfgRangeTime,
                mtiInfo['rangeGates'], mtiInfo['dem'])
            # create plot
            dtAnaPlots.plotRangeTime(mtiInfo, cfgRangeTime)
else:

    # fetch all flow parameter result fields
    configDir = pathlib.Path(avalancheDir, 'Outputs', comModule, 'configurationFiles')
    if (configDir.is_dir() is False) or ( len(list(configDir.glob('*.ini'))) == 0):
        log.warning('No configuration files found in %s' % configDir)
        # fetch info on available simulations
        simDF, resTypeListAll = fU.makeSimFromResDF(avalancheDir, comModule=comModule, inputDir="", simName="")
    else:
        simDF = cfgUtils.createConfigurationInfo(avalancheDir, standardCfg='', writeCSV=False, specDir='')

    for index, simDFrow in simDF.iterrows():

        log.info('Perform range-time diagram for simulation: %s' % index)

        # fetch simulation dem
        # get dem dictionary - already read DEM with correct mesh cell size
        if comModule == 'com1DFA':
            dem = gI.initializeDEM(avalancheDir, demPath=simDF['DEM'].loc[index])
        else:
            dem = gI.initializeDEM(avalancheDir, demPath=demPath)
        dem['originalHeader'] = dem['header'].copy()

        # add simHash info
        if comModule == 'com1DFA':
            cfgRangeTime['GENERAL']['simHash'] = index
        else:
            cfgRangeTime['GENERAL']['simHash'] = simDFrow['simHash']

        # setup required data
        mtiInfo = dtAna.setupRangeTimeDiagram(dem, cfgRangeTime)
        # create plot of range distance already masked with radar field of view
        dtAnaPlots.radarFieldOfViewPlot(mtiInfo['radarFov'], mtiInfo['rangeMasked'], cfgRangeTime,
            mtiInfo['rangeGates'], dem)
        np.savetxt('rangeMasked.txt', mtiInfo['rangeMasked'])
        mtiInfo['plotTitle'] = 'range-time diagram %s' % index

        # fetch all flow parameter result fields
        flowFieldsDir = pathlib.Path(avalancheDir, 'Outputs', 'com1DFA', 'peakFiles', 'timeSteps')
        simNameSuffix = simDFrow['simName'] + '_' + cfgRangeTime['GENERAL']['rangeTimeResType']
        flowFields = fU.fetchFlowFields(flowFieldsDir, suffix=simNameSuffix)

        # check if simulation results are available
        if len(flowFields) == 0:
            fU.fileNotFoundMessage(('No flow variable results found in %s - consider first running avalanche simulations' %
                flowFieldsDir))

        for flowField in flowFields:

            # read flow field data
            flowFieldDict = IOf.readRaster(flowField)
            flowF = flowFieldDict['rasterData']

            # extract avalanche front distance to radar and average values of range gates for mti plot
            mtiInfo = dtAna.extractFrontAndMeanValuesRadar(cfgRangeTime, flowF, mtiInfo)
            timeStep, _ = dtAna.fetchTimeStepFromName(flowField)
            mtiInfo['timeList'].append(timeStep[0])

        # create plot
        dtAnaPlots.plotRangeTime(mtiInfo, cfgRangeTime )
