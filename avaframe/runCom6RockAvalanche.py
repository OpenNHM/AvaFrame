"""
Run the rock avalanche setup of com1DFA
"""

import pathlib
import time
import argparse

# Local imports
# import config and init tools
from avaframe.in3Utils import cfgUtils
from avaframe.in3Utils import logUtils
import avaframe.in3Utils.initializeProject as initProj
from avaframe.in3Utils import fileHandlerUtils as fU

# import computation modules
from avaframe.com6RockAvalanche import com6RockAvalanche
from avaframe.in3Utils import spatialVoellmyInputs


def runCom6RockAvalanche(avalancheDir="", calibration="voellmy"):
    """Run com1DFA with rock avalanche parameters

    Parameters
    ----------
    avalancheDir: str
        path to avalanche directory (setup e.g. with init scripts)
    calibration: str
        friction model: voellmy (default) or spatialVoellmy

    Returns
    -------
    peakFilesDF: pandas dataframe
        with info about com1DFA peak file locations
    """
    # Time the whole routine
    startTime = time.time()

    # log file name; leave empty to use default runLog.log
    logName = "runCom6RockAvalanche"

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
    # Clean input directory(ies) of old work files
    initProj.cleanSingleAvaDir(avalancheDir, deleteOutput=False)

    # pathlib version of avalanche dir
    avaDir = pathlib.Path(avalancheDir)

    # load rock avalanche config
    rockAvalancheCfg = cfgUtils.getModuleConfig(com6RockAvalanche, avalancheDir)

    # override friction model if spatialVoellmy calibration is requested
    if calibration == "spatialVoellmy":
        rockAvalancheCfg["com1DFA_com1DFA_override"]["frictModel"] = "spatialVoellmy"

        muRasters = list((avaDir / "Inputs" / "RASTERS").glob("*_mu.*"))
        xiRasters = list((avaDir / "Inputs" / "RASTERS").glob("*_xi.*"))
        spatialShps = list(
            (avaDir / "Inputs" / "POLYGONS").glob("*_spatialVoellmy.shp")
        )

        rastersExist = bool(muRasters and xiRasters)
        shpExists = bool(spatialShps)

        if rastersExist and shpExists:
            raise RuntimeError(
                "spatialVoellmy friction model: both rasters in Inputs/RASTERS/"
                " and *_spatialVoellmy.shp in Inputs/POLYGONS/ found"
                " - ambiguous input"
            )
        elif shpExists:
            spatialVoellmyCfg = cfgUtils.getModuleConfig(spatialVoellmyInputs)
            # set default fill values from rock avalanche Voellmy defaults
            overrideParams = rockAvalancheCfg["com1DFA_com1DFA_override"]
            spatialVoellmyCfg["DEFAULTS"]["default_mu"] = overrideParams["muvoellmy"]
            spatialVoellmyCfg["DEFAULTS"]["default_xi"] = overrideParams["xsivoellmy"]
            spatialVoellmyInputs.generateMuXiRasters(avaDir, spatialVoellmyCfg)
        elif rastersExist:
            log.info("spatialVoellmy: using existing mu/xi rasters from Inputs/RASTERS/")
        else:
            raise FileNotFoundError(
                "spatialVoellmy friction model: no *_mu and *_xi rasters found in"
                " Inputs/RASTERS/ and no *_spatialVoellmy.shp found in"
                " Inputs/POLYGONS/"
            )

    # perform com1DFA simulation with rock avalanche settings
    _, plotDict, reportDictList, _ = com6RockAvalanche.com6RockAvalancheMain(cfgMain, rockAvalancheCfg)

    # Get peakfiles to return to QGIS
    inputDir = avaDir / "Outputs" / "com1DFA" / "peakFiles"
    peakFilesDF = fU.makeSimDF(inputDir, avaDir=avaDir)

    # Print time needed
    endTime = time.time()
    log.info("Took %6.1f seconds to calculate." % (endTime - startTime))

    return peakFilesDF


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run rock avalanche workflow")
    parser.add_argument(
        "avadir", metavar="avadir", type=str, nargs="?", default="", help="the avalanche directory"
    )
    parser.add_argument(
        "-fc",
        "--friction_calibration",
        choices=["voellmy", "spatialVoellmy"],
        type=str,
        default="voellmy",
        help="friction model: voellmy (default) or spatialVoellmy",
    )

    args = parser.parse_args()
    runCom6RockAvalanche(str(args.avadir), str(args.friction_calibration))
