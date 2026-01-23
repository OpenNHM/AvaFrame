"""
Run script for running simulations in parallel based on input from a regional folder containing
multiple avaFolders
"""

import time
import pathlib
import argparse

from avaframe.com7Regional import com7Regional as com7
from avaframe.com7Regional import splitInputs as sI
from avaframe.in3Utils import cfgUtils
from avaframe.in3Utils import logUtils


def runCom7Regional(avalancheDir="", splitInputs=False, runComputations=False):
    """Run regional avalanche simulations in parallel.

    Parameters
    ----------
    avalancheDir : str, optional
        Path to the main avalanche directory. If not provided, uses the path from general configuration.
    splitInputs : bool, optional
        Whether to run the split inputs preprocessor before running regional simulations. Default is False.
    runComputations : bool, optional
        Whether to run the main regional computations. Default is False.

    Returns
    -------
    allPeakFilesDir : pathlib.Path or None
        Path to the directory containing the collected peak files from all sub-avalanche directories, if enabled
    allTimeStepsDir : pathlib.Path or None
        Path to the directory containing consolidated time step files, if enabled
    mergedRastersDir : pathlib.Path or None
        Path to the directory containing merged output rasters, if enabled
    """

    # Time the whole routine
    startTime = time.time()

    # Load the avalanche directory from command input or the general configuration file
    cfgMain = cfgUtils.getGeneralConfig()
    if avalancheDir != "":
        cfgMain["MAIN"]["avalancheDir"] = avalancheDir
    else:
        avalancheDir = cfgMain["MAIN"]["avalancheDir"]

    # Start logging
    log = logUtils.initiateLogger(str(avalancheDir), logName="runCom7Regional")
    log.info("MAIN SCRIPT")

    # Run split inputs preprocessing if requested
    if splitInputs:
        log.info("Running split inputs preprocessor...")
        splitStartTime = time.time()

        # Define the output directory for split inputs
        outputDir = pathlib.Path(avalancheDir) / "com7Regional"

        # Load split inputs module configuration
        splitCfg = cfgUtils.getModuleConfig(sI, avalancheDir)

        # Run splitting process
        sI.splitInputsMain(pathlib.Path(avalancheDir), outputDir, splitCfg, cfgMain)

        splitEndTime = time.time()
        log.info(f"Split inputs preprocessing completed after {splitEndTime - splitStartTime:.1f} seconds.")

    # Initialize return values
    allPeakFilesDir, mergedRastersDir = None, None

    # Run main computations if requested
    if runComputations:
        # Load module configuration
        cfg = cfgUtils.getModuleConfig(com7, avalancheDir, fileOverride="", toPrint=False, onlyDefault=False)

        # Call main function
        allPeakFilesDir, mergedRastersDir = com7.com7RegionalMain(cfgMain, cfg)

    # Print time needed
    endTime = time.time()
    log.info("Regional process finished after %.1f seconds." % (endTime - startTime))

    return allPeakFilesDir, mergedRastersDir


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run regional workflow")
    parser.add_argument(
        "avadir", metavar="avalancheDir", type=str, nargs="?", default="", help="the avalanche directory"
    )
    parser.add_argument(
        "--split-inputs",
        action="store_true",
        help="run split inputs preprocessor before regional simulations",
    )
    parser.add_argument(
        "--run-computations",
        action="store_true",
        help="run the main regional computations",
    )

    args = parser.parse_args()
    runCom7Regional(str(args.avadir), splitInputs=args.split_inputs, runComputations=args.run_computations)
