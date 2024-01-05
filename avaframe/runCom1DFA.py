"""
    Run script for running python com1DFA kernel
"""
# Load modules
# importing general python modules
import time
import argparse
import pathlib
import sys


# Local imports
import avaframe.in3Utils.initializeProject as initProj
from avaframe.com1DFA import com1DFA
from avaframe.in3Utils import cfgUtils
from avaframe.in3Utils import logUtils
from avaframe.in3Utils import fileHandlerUtils as fU

def runCom1DFA(avalancheDir="", configurationDir="", calibration=""):
    """Run com1DFA in the default configuration with only an
    avalanche directory as input and the (optional) friction calibration
    size

    Parameters
    ----------
    avalancheDir: str
        path to avalanche directory (setup eg. with init scipts)
    calibration: str
        which friction calibration to use: large, medium, small, auto, ini

    Returns
    -------
    peakFilesDF: pandas dataframe
        dataframe with info about com1DFA peak file locations
    """

    # Time the whole routine
    startTime = time.time()

    # log file name; leave empty to use default runLog.log
    logName = "runCom1DFA"

    # Load avalanche directory from general configuration file
    # More information about the configuration can be found here
    # on the Configuration page in the documentation
    cfgMain = cfgUtils.getGeneralConfig()
    if avalancheDir != "":
        cfgMain["MAIN"]["avalancheDir"] = avalancheDir
    else:
        avalancheDir = cfgMain["MAIN"]["avalancheDir"]
    if configurationDir != "":
        cfgMain["MAIN"]["configurationDir"] = configurationDir
    else:
        configurationDir = cfgMain["MAIN"]["configurationDir"]

    # Start logging
    log = logUtils.initiateLogger(avalancheDir, logName)
    log.info("MAIN SCRIPT")
    log.info("Current avalanche: %s", avalancheDir)

    # ----------------
    # Clean input directory(ies) of old work and output files
    # If you just created the ``avalancheDir`` this one should be clean but if you
    # already did some calculations you might want to clean it::
    initProj.cleanSingleAvaDir(avalancheDir, deleteOutput=False)

    # Set friction model according to cmd argument
    cfgCom1DFA = cfgUtils.getModuleConfig(com1DFA, configDir=configurationDir, toPrint=False)

    if calibration.lower() == "small":
        cfgCom1DFA["GENERAL"]["frictModel"] = "samosATSmall"
    elif calibration.lower() == "medium":
        cfgCom1DFA["GENERAL"]["frictModel"] = "samosATMedium"
    elif calibration.lower() == "large":
        cfgCom1DFA["GENERAL"]["frictModel"] = "samosAT"
    elif calibration.lower() == "auto":
        cfgCom1DFA["GENERAL"]["frictModel"] = "samosATAuto"
    else:
        log.info("no friction calibration override given - using ini")

    # ----------------
    # Run dense flow
    dem, plotDict, reportDictList, simDF = com1DFA.com1DFAMain(cfgMain, cfgInfo=cfgCom1DFA)

    # Get peakfiles to return to QGIS
    avaDir = pathlib.Path(avalancheDir)
    inputDir = avaDir / "Outputs" / "com1DFA" / "peakFiles"
    peakFilesDF = fU.makeSimDF(inputDir, avaDir=avaDir)

    # Print time needed
    endTime = time.time()
    log.info("Took %6.1f seconds to calculate." % (endTime - startTime))

    return peakFilesDF


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run com1DFA workflow")
    parser.add_argument(
        "avadir", metavar="avalancheDir", type=str, nargs="?", default="", help="the avalanche directory"
    )
    parser.add_argument(
        "configdir", metavar="configurationDir", type=str, nargs="?", default="",
        help="the configuration file directory"
    )
    parser.add_argument(
        "-fc",
        "--friction_calibration",
        choices=["auto", "large", "medium", "small", "ini"],
        type=str,
        default="ini",
        help="friction calibration override, possible values are large, medium , small, auto and ini."
        + "Overrides default AND local configs",
    )

    args = parser.parse_args()
    runCom1DFA(str(args.avadir), str(args.configdir),str(args.friction_calibration))
