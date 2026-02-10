"""
    Run script for creating a release area info file
"""

import pathlib
import argparse

# Local imports
from avaframe.com1DFA import com1DFA
from avaframe.in3Utils import cfgUtils
from avaframe.in3Utils import logUtils
import avaframe.in1Data.getInput as gI
from avaframe.in3Utils import initializeProject as initProj


def runRelInfo(avalancheDir=''):
    """ Run a com1DFA release area analysis

    Parameters
    ----------
    avalancheDir: str
        path to avalanche directory (setup eg. with init scipts)

    Returns
    -------
    """

    logName = 'runReleaseInfo'

    # Load general configuration file
    cfgMain = cfgUtils.getGeneralConfig()

    # Load avalanche directory from general configuration file,
    # if not provided as input argument
    if avalancheDir != '':
        cfgMain['MAIN']['avalancheDir'] = avalancheDir
    else:
        avalancheDir = cfgMain['MAIN']['avalancheDir']

    avalancheDir = pathlib.Path(avalancheDir)

    # Start logging
    log = logUtils.initiateLogger(avalancheDir, logName)
    log.info('MAIN SCRIPT')
    log.info('Current avalanche: %s', avalancheDir)

    # Clean input directory(ies) of old work files
    initProj.cleanSingleAvaDir(avalancheDir, deleteOutput=False)

    # Load configuration file for com1DFA module
    cfg = cfgUtils.getModuleConfig(com1DFA, avalancheDir)

    # fetch input data and create release info dataframe and csv file
    relDFDict = gI.createReleaseStats(avalancheDir, cfg)

    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run in1RelInfo workflow')
    parser.add_argument('avadir', metavar='a', type=str, nargs='?', default='',
                        help='the avalanche directory')

    args = parser.parse_args()
    runRelInfo(str(args.avadir))
