"""
    Run script for running python com8MoTPSA kernel
"""
# Load modules
# importing general python modules
import time
import argparse
import pathlib
import sys

# Local imports
import avaframe.in3Utils.initializeProject as initProj
from avaframe.com8MoTPSA import com8MoTPSA
from avaframe.in3Utils import cfgUtils
from avaframe.in3Utils import logUtils
from avaframe.in3Utils import fileHandlerUtils as fU


def runCom8MoTPSA(avalancheDir=''):
    """ Run com8MoTPSA in the default configuration with only an
    avalanche directory as input

    Parameters
    ----------
    avalancheDir: str
        path to avalanche directory (setup eg. with init scipts)

    Returns
    -------
    peakFilesDF: pandas dataframe
        dataframe with info about com8MoTPSA peak file locations
    """

    # Time the whole routine
    startTime = time.time()

    # log file name; leave empty to use default runLog.log
    logName = 'runCom8MoTPSA'

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
    log.info('MAIN SCRIPT')
    log.info('Current avalanche: %s', avalancheDir)

    # ----------------
    # Clean input directory(ies) of old work and output files
    # If you just created the ``avalancheDir`` this one should be clean but if you
    # already did some calculations you might want to clean it::
    initProj.cleanSingleAvaDir(avalancheDir, deleteOutput=False)

    # Get module config
    cfgCom8MoTPSA = cfgUtils.getModuleConfig(com8MoTPSA, avalancheDir, toPrint=False)

    # ----------------
    # Run psa
    com8MoTPSA.com8MoTPSAMain(cfgMain, cfgInfo=cfgCom8MoTPSA)

    # # Get peakfiles to return to QGIS
    # avaDir = pathlib.Path(avalancheDir)
    # inputDir = avaDir / 'Outputs' / 'com8MoTPSA' / 'peakFiles'
    # peakFilesDF = fU.makeSimDF(inputDir, avaDir=avaDir)

    # Print time needed
    endTime = time.time()
    log.info('Took %6.1f seconds to calculate.' % (endTime - startTime))

    # return peakFilesDF
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run com8MoTPSA workflow')
    parser.add_argument('avadir', metavar='avalancheDir', type=str, nargs='?', default='',
                        help='the avalanche directory')

    args = parser.parse_args()
    runCom8MoTPSA(str(args.avadir))
