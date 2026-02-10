"""
    Run ana3AIMEC

    This file is part of Avaframe.

"""

# Load modules
import time

# Local imports
from avaframe.ana3AIMEC import dfa2Aimec, ana3AIMEC, aimecTools
from avaframe.in3Utils import initializeProject as iP
from avaframe.in3Utils import cfgUtils
from avaframe.in3Utils import logUtils


def runAna3AIMECCompMods(avalancheDir=''):
    """ run script for AIMEC module comparison analysis
    reads the avalancheDir from the configuration file or the one given in input
    proceeds to AIMEC analysis and produces plots and reports
    """
    # -----------Required settings-----------------
    # log file name; leave empty to use default runLog.log
    logName = 'runAna3AIMECCompMods'
    # ---------------------------------------------
    # Load avalanche directory from general configuration file or the one provided in inputs
    cfgMain = cfgUtils.getGeneralConfig()
    if avalancheDir != '':
        cfgMain['MAIN']['avalancheDir'] = avalancheDir
    else:
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

    # Setup input from com1DFA
    cfgSetup = cfg['AIMECSETUP']
    comModules = cfgSetup['comModules'].split('|')
    referenceSimName = cfgSetup['referenceSimName']
    inputsDF, pathDict = dfa2Aimec.dfaBench2Aimec(avalancheDir, cfg, simNameRef=referenceSimName)
    pathDict['valRef'] = ''

    if referenceSimName != '':
        log.info('Reference simulation is based on provided simName: \'%s\' and corresponds to simulation %s'
                 % (referenceSimName, pathDict['refSimName']))
    else:
        log.info(('Reference simulation is the only simulation in the \'%s\' module folder')
                 % (comModules[0]))
    # Extract input file locations
    cfgSetup = cfg['AIMECSETUP']
    comModules = cfgSetup['comModules'].split('|')
    pathDict = aimecTools.readAIMECinputs(avalancheDir, pathDict, cfgSetup.getboolean('defineRunoutArea'),
                                          dirName=(comModules[0] + '_' + comModules[1]))

    startTime = time.time()
    
    log.info("Running ana3AIMEC model on test case DEM \n %s \n with profile \n %s ",
             pathDict['demSource'], pathDict['profileLayer'])
    # Run AIMEC postprocessing
    rasterTransfo, resAnalysisDF, plotDict, _ = ana3AIMEC.mainAIMEC(pathDict, inputsDF, cfg)

    endTime = time.time()

    log.info(('Took %s seconds to calculate.' % (endTime - startTime)))


if __name__ == '__main__':
    runAna3AIMECCompMods()
