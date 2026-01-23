"""
    Create an avalanche path from DFA simulation results and create a split point
    Configuration should be specified in DFAPathGenerationCfg.ini
    or the local version of it.
    It is possible to generate a path from particles or from fields.
    From particles, you need to save the particles dictionaries at
    multiple time steps (first, some in the middle and last)
    From fields, you need the FT, FM at multiple time steps
"""
# import general python modules
import pathlib
import time
import argparse

# local imports
from avaframe.in3Utils import cfgUtils
from avaframe.in3Utils import logUtils
from avaframe.in3Utils import initializeProject as initProj
from avaframe.ana5Utils import DFAPathGeneration

def runAna5DFAPathGeneration(avalancheDir='', runDFAModule=''):
    """
    Run DFA path generation in the default configuration with only an avalanche directory as input.

    Parameters
    ----------
    avalancheDir: str
        path to avalanche directory (setup e.g. with init scripts)
    runDFAModule: bool (optional)
        whether to run the DFA module (True) or load existing results (False); overrides setting in ini

    Returns
    -------
    massAvgPath: pathlib
        file path to the mass-averaged path result saved as a shapefile
    splitPoint: pathlib
        file path to the split point result saved as a shapefile

    """

    # Time the whole routine
    startTime = time.time()

    # log file name; leave empty to use default runLog.log
    logName = 'runAna5DFAPathGeneration'

    # Load avalanche directory from general configuration file
    # More information about the configuration can be found here
    # on the Configuration page in the documentation
    cfgMain = cfgUtils.getGeneralConfig()
    if avalancheDir != '':
        cfgMain['MAIN']['avalancheDir'] = avalancheDir
    else:
        avalancheDir = cfgMain['MAIN']['avalancheDir']

    # Start logging
    log = logUtils.initiateLogger(avalancheDir, logName)
    log.info("MAIN SCRIPT")
    log.info('Current avalanche: %s', avalancheDir)

    # load config for path generation (from DFAPathGenerationCfg.ini or its local)
    cfgDFAPath = cfgUtils.getModuleConfig(DFAPathGeneration, avalancheDir)

    # Clean DFAPath output in avalanche directory
    # If you just created the directory this one should be clean but if you
    # already did some calculations you might want to clean it::
    initProj.cleanModuleFiles(avalancheDir, DFAPathGeneration, deleteOutput=True)

    # Run or load DFA results depending on runDFAModule bool in command call
    if args.runDFA:
        runDFAModule = args.runDFA.lower() == 'true'
    else:
        runDFAModule = cfgDFAPath['GENERAL'].getboolean('runDFAModule')

    # Call main function to generate path and splitPoint
    avaPath, splitPoint = DFAPathGeneration.generatePathAndSplitpoint(avalancheDir, cfgDFAPath, cfgMain, runDFAModule)

    log.info("DFA path generation completed")

    # Print time needed
    endTime = time.time()
    log.info('Took %.1f seconds to calculate.' % (endTime - startTime))

    return avaPath,splitPoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run computeDFAPath workflow')
    parser.add_argument('avadir', metavar='avalancheDir', type=str, nargs='?', default='',
                        help='the avalanche directory')
    parser.add_argument('-runDFA', type=str, choices=['true', 'false'], default='',
                        help='add to override ini setting to run DFA module')

    args = parser.parse_args()
    runAna5DFAPathGeneration(str(args.avadir), str(args.runDFA))
