"""
Run script for generating spatial Voellmy friction raster inputs.
"""

import argparse
import pathlib
import time

from avaframe.in3Utils import cfgUtils
from avaframe.in3Utils import logUtils
from avaframe.in3Utils import initializeProject as initProj
from avaframe.in3Utils import spatialVoellmyInputs


def runSpatialVoellmyInputs(avaDir=""):
    """Run generation of mu and xi rasters from shapefile.

    Parameters
    ----------
    avaDir : str
        Path to the avalanche directory. If empty, read from
        avaframeCfg.ini.
    """
    startTime = time.time()
    cfgMain = cfgUtils.getGeneralConfig()
    if avaDir:
        avaDir = pathlib.Path(avaDir)
    else:
        avaDir = pathlib.Path(cfgMain["MAIN"]["avalancheDir"])

    logName = "runSpatialVoellmyInputs"
    log = logUtils.initiateLogger(avaDir, logName)
    log.info("MAIN SCRIPT")
    log.info("Avalanche directory: %s", avaDir)

    initProj.cleanSingleAvaDir(avaDir, deleteOutput=False)

    cfg = cfgUtils.getModuleConfig(spatialVoellmyInputs)
    spatialVoellmyInputs.generateMuXiRasters(avaDir, cfg)

    endTime = time.time()
    log.info("Took %6.1f seconds to calculate.", endTime - startTime)
    log.info("Workflow completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate spatial Voellmy mu/xi rasters from shapefiles")
    parser.add_argument(
        "avaDir",
        nargs="?",
        default="",
        help="Path to avalanche directory",
    )
    args = parser.parse_args()
    runSpatialVoellmyInputs(str(args.avaDir))
