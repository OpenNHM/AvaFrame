"""
    Run script for running the operational workflow
"""

# Load modules
# importing general python modules
import time
import logging
import argparse
import pathlib
import pandas

# Import the avaframe modules you want to use. This list will
# grow as you use more avaframe modules. You can refer to the different
# computational modules documentation to know which imports are required

# first import the always useful tools
from avaframe.in3Utils import cfgUtils
from avaframe.in3Utils import logUtils
from avaframe.in3Utils import initializeProject as initProj
from avaframe.in3Utils import fileHandlerUtils as fU
from avaframe.log2Report import generateReport as gR

# then depending on which computational module you want to use
from avaframe.com1DFA import com1DFA
from avaframe.com2AB import com2AB
from avaframe.out3Plot import outAB


def runOperational(avalancheDir=''):
    """ Run com1DFA and com2AB in the default configuration with only an
    avalanche directory as input

    Parameters
    ----------
    avalancheDir: str
        path to avalanche directory (setup eg. with init scipts)

    Returns
    -------
    abShpFile: str
        path to com2AB results as shapefile
    peakFilesDF: pandas dataframe
        dataframe with info about com1DFA peak file locations
    """
    # Time the whole routine
    startTime = time.time()

    # log file name; leave empty to use default runLog.log
    logName = 'runOperational'

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

    # ----------------
    # Run dense flow
    _, plotDict, reportDictList, _ = com1DFA.com1DFAMain(cfgMain)

    # Get peakfiles to return to QGIS
    avaDir = pathlib.Path(avalancheDir)
    inputDir = avaDir / 'Outputs' / 'com1DFA' / 'peakFiles'
    peakFilesDF = fU.makeSimDF(inputDir, avaDir=avaDir)

    # Check if profile and splitpoint exist, and only run if available
    try:
        # See if reading AB inputs throws errors
        com2AB.readABinputs(avalancheDir)

        # if not: Run Alpha Beta
        cfgAB = cfgUtils.getModuleConfig(com2AB, avalancheDir)
        pathDict, dem, splitPoint, eqParams, resAB = com2AB.com2ABMain(cfgAB, avalancheDir)
        abShpFile = outAB.writeABtoSHP(pathDict, resAB)

        # Generate report info for com2AB
        reportDictList, _, _ = outAB.writeABpostOut(pathDict, dem, splitPoint, eqParams, resAB, cfgMain, reportDictList)

    except AssertionError as e:
        log.info(e)
        log.info('No Alpha Beta info, skipping')
        abShpFile = None

    # ----------------
    # Collect results/plots/report  to a single directory
    # make simple plots (com1DFA, com2AB)

    # Set directory for report
    reportDir = avaDir / 'Outputs' / 'reports'
    fU.makeADir(reportDir)
    # write report and copy plots to report dir
    gR.writeReport(reportDir, reportDictList, cfgMain['FLAGS'].getboolean('reportOneFile'), plotDict=plotDict,
                   standaloneReport=True)

    # Print time needed
    endTime = time.time()
    log.info('Took %6.1f seconds to calculate.' % (endTime - startTime))

    return str(abShpFile), peakFilesDF


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run Operational workflow')
    parser.add_argument('avadir', metavar='a', type=str, nargs='?', default='',
                        help='the avalanche directory')

    args = parser.parse_args()
    runOperational(str(args.avadir))
