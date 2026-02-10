"""
    Run script for module com2AB
"""

# importing general python modules
import time
import argparse
import pathlib

# Import the avaframe modules you want to use. This list will
# grow as you use more avaframe modules. You can refer to the different
# computational modules documentation to know which imports are required

# first import the always useful tools
from avaframe.in3Utils import cfgUtils
from avaframe.in3Utils import logUtils
from avaframe.in3Utils import initializeProject as initProj

# then depending on which computational module you want to use
from avaframe.com2AB import com2AB
from avaframe.out3Plot import outAB
from avaframe.in3Utils import fileHandlerUtils as fU
from avaframe.log2Report import generateReport as gR


def runCom2AB(avalancheDir='', smallAva=False):
    """ Run com2AB in the default configuration with only an
    avalanche directory as input

    Parameters
    ----------
    avalancheDir: str
        path to avalanche directory (setup eg. with init scipts)
    smallAva: bool
        use small avalanche setting; overrides ini setting

    Returns
    -------
    abShpFile: str
        path to com2AB results as shapefile
    """
    # Time the whole routine
    startTime = time.time()

    # log file name; leave empty to use default runLog.log
    logName = 'runCom2AB'

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

    # Call the main com2AB functions
    cfgAB = cfgUtils.getModuleConfig(com2AB, avalancheDir)

    # Override smallAva from ini file if needed
    if smallAva:
        cfgAB['ABSETUP']['smallAva'] = 'True'
        log.info('Small avalanche setting override from runCom2AB (forced to true)')

    pathDict, dem, splitPoint, eqParams, resAB = com2AB.com2ABMain(cfgAB, avalancheDir)
    abShpFile = outAB.writeABtoSHP(pathDict, resAB, cfgAB['ABSETUP']['smallAva'])

    # Analyse/ plot/ write results #
    reportDictList = []
    reportDictList, plotFile, writeFile = outAB.writeABpostOut(pathDict, dem, splitPoint, 
                                                               eqParams, resAB,
                                                               cfgMain, reportDictList)

    log.info('Plotted to: %s' % [str(plotFileName) for plotFileName in plotFile])
    log.info('Data written: %s' % [str(writeFileName) for writeFileName in writeFile])

    # Set directory for report
    avaDir = pathlib.Path(avalancheDir)
    reportDir = avaDir / 'Outputs' / 'com2AB' / 'report'
    fU.makeADir(reportDir)
    # write report and copy plots to report dir
    gR.writeReport(reportDir, reportDictList, cfgMain['FLAGS'].getboolean('reportOneFile'), plotDict='',
                   standaloneReport=True)

    # Print time needed
    endTime = time.time()
    log.info('Took %6.1f seconds to calculate.' % (endTime - startTime))

    return str(abShpFile)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run com2AB workflow')
    parser.add_argument('avadir', metavar='a', type=str, nargs='?', default='',
                        help='the avalanche directory')
    parser.add_argument("--small_ava", action="store_true",
                        help='activate small avalanche calibration')

    args = parser.parse_args()
    runCom2AB(str(args.avadir), args.small_ava)