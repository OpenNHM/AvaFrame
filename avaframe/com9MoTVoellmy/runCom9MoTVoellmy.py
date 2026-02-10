"""
Run script for running python com9MoTVoellmy kernel
"""

# Load modules
# importing general python modules
import time
import argparse

# Local imports
import avaframe.in3Utils.initializeProject as initProj
from avaframe.com9MoTVoellmy import com9MoTVoellmy
from avaframe.in3Utils import cfgUtils
from avaframe.in3Utils import logUtils


def runCom9MoTVoellmy(avalancheDir="", simType=""):
    """Run com9MoTVoellmy in the default configuration with only an
    avalanche directory as input

    Parameters
    ----------
    avalancheDir: str
        path to avalanche directory (setup eg. with init scipts)
    simType: str
        simulation type override (null, ent, res, entres, available)

    Returns
    -------
    peakFilesDF: pandas dataframe
        dataframe with info about com9MoTVoellmy peak file locations
    """

    # Time the whole routine
    startTime = time.time()

    # log file name; leave empty to use default runLog.log
    logName = "runCom9MoTVoellmy"

    # Load avalanche directory from general configuration file
    # More information about the configuration can be found here
    # on the Configuration page in the documentation
    cfgMain = cfgUtils.getGeneralConfig()
    if avalancheDir != "":
        cfgMain["MAIN"]["avalancheDir"] = avalancheDir
    else:
        avalancheDir = cfgMain["MAIN"]["avalancheDir"]

    # Start logging
    log = logUtils.initiateLogger(avalancheDir, logName)
    log.info("MAIN SCRIPT")
    log.info("Current avalanche: %s", avalancheDir)

    # ----------------
    # Clean input directory(ies) of old work and output files
    # If you just created the ``avalancheDir`` this one should be clean but if you
    # already did some calculations you might want to clean it::
    initProj.cleanSingleAvaDir(avalancheDir, deleteOutput=False)

    # Get module config
    cfgCom9MoTVoellmy = cfgUtils.getModuleConfig(com9MoTVoellmy, avalancheDir, toPrint=False)

    # Override simTypeList if provided via command line
    if simType != "":
        cfgCom9MoTVoellmy["GENERAL"]["simTypeList"] = simType
        log.info("Overriding simTypeList with: %s", simType)
    else:
        log.info("No simType override given - using ini")

    # ----------------
    com9MoTVoellmy.com9MoTVoellmyMain(cfgMain, cfgInfo=cfgCom9MoTVoellmy)

    # # Get peakfiles to return to QGIS
    # avaDir = pathlib.Path(avalancheDir)
    # inputDir = avaDir / 'Outputs' / 'com8MoTPSA' / 'peakFiles'
    # peakFilesDF = fU.makeSimDF(inputDir, avaDir=avaDir)

    # Print time needed
    endTime = time.time()
    log.info("Took %6.1f seconds to calculate." % (endTime - startTime))

    # return peakFilesDF
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run com9MoTVoellmy workflow")
    parser.add_argument(
        "avadir", metavar="avalancheDir", type=str, nargs="?", default="", help="the avalanche directory"
    )
    parser.add_argument(
        "-st",
        "--sim_type",
        choices=["null", "ent", "res", "entres", "available"],
        type=str,
        default="",
        help="simulation type override, possible values are null, ent, res, entres, available"
        + "Overrides default AND local configs",
    )

    args = parser.parse_args()
    runCom9MoTVoellmy(str(args.avadir), str(args.sim_type))
