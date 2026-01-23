"""Module for handling regional avalanche simulations."""

import pathlib
import shutil
import logging
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

import avaframe.in3Utils.initializeProject as initProj
from avaframe.com1DFA import com1DFA
from avaframe.in3Utils import cfgUtils, cfgHandling
from avaframe.in3Utils import logUtils
from avaframe.in2Trans import rasterUtils
from avaframe.in3Utils import fileHandlerUtils as fU

from rasterio.merge import merge

from avaframe.in3Utils.fileHandlerUtils import findAvaDirsBasedOnInputsDir

# create local logger
log = logging.getLogger(__name__)


def com7RegionalMain(cfgMain, cfg):
    """Run com7Regional with given configuration.

    This function processes multiple avalanche directories in parallel, running simulations
    for each directory.

    Parameters
    ----------
    cfgMain : configparser.ConfigParser
        Main avaframe configuration settings
    cfg : configparser.ConfigParser
        Regional configuration settings with potential overrides

    Returns
    -------
    allPeakFilesDir : pathlib.Path or None
        Path to the directory containing all peak files, if copyPeakFiles is True
    mergedRastersDir : pathlib.Path or None
        Path to the directory containing merged rasters, if mergeOutput is True

    Notes
    -----
    The function expects the following directory structure:
    avalancheDir/
    └── regionalDir/
        ├── avalanche1/
        ├── avalanche2/
        └── ...

    Where:
    - avalancheDir: Main directory specified in cfgMain
    - regionalDir: Subdirectory specified in cfg['GENERAL']['regionalDir']
    """
    # Define the regional directory in relation to the avalanche directory
    regionalDirFromCfg = str(cfg["GENERAL"]["regionalDir"])
    regionalDir = pathlib.Path(cfgMain["MAIN"]["avalancheDir"]) / regionalDirFromCfg

    # List valid avalanche directories within the regional directory
    avaDirs = findAvaDirsBasedOnInputsDir(regionalDir)

    # Get total number of simulations
    log.info(f"Getting total number of simulations to perform...")
    with logUtils.silentLogger():
        totalSims = getTotalNumberOfSims(avaDirs, cfgMain, cfg)
    log.info(f"Found {totalSims} (new) simulations to perform across {len(avaDirs)} directories")

    # Get number of processes based on number of avaDirs
    nProcesses = cfgUtils.getNumberOfProcesses(cfgMain, len(avaDirs))

    # Set nCPU for com1 to 1 to avoid nested parallelization
    cfgMain["MAIN"]["nCPU"] = "1"

    # Track progress and results
    completed = 0
    nSuccesses = 0

    # Process avalanche directories within the regional folder concurrently
    with ProcessPoolExecutor(max_workers=nProcesses) as executor:
        # Submit each avalanche directory to the executor
        futures = {
            executor.submit(processAvaDirCom1Regional, cfgMain, cfg, avaDir): avaDir for avaDir in avaDirs
        }
        # Log results as each future completes
        for future in as_completed(futures):
            avaDir = futures[future]
            try:
                resultDir, status = future.result()
                completed += 1

                if status == "Success":
                    nSuccesses += 1

                log.info(
                    f"{status} in directory: {resultDir.relative_to(pathlib.Path(regionalDir))} "
                    f"- Overall progress: {completed}/{len(avaDirs)}"
                )
            except Exception as e:
                log.error(f"Error processing {avaDir}: {e}")

    log.info(f"Processing complete. Success in {nSuccesses} out of {len(avaDirs)} directories.")

    # Copy/move peak files if configured
    allPeakFilesDir = None
    if cfg["GENERAL"].getboolean("copyPeakFiles"):
        allPeakFilesDir = moveOrCopyPeakFiles(cfg, regionalDir)

    # Merge output rasters if configured
    mergedRastersDir = None
    if cfg["GENERAL"].getboolean("mergeOutput"):
        mergedRastersDir = mergeOutputRasters(cfg, regionalDir)

    return allPeakFilesDir, mergedRastersDir


def getTotalNumberOfSims(avaDirs, cfgMain, cfgCom7):
    """Get total number of simulations across all avalanche directories.

    Parameters
    ----------
    avaDirs : list
        List of avalanche directories
    cfgMain : configparser.ConfigParser
        Main configuration
    cfgCom7 : configparser.ConfigParser
        Regional configuration with potential overrides

    Returns
    -------
    int
        Total number of simulations
    """
    totalSims = 0
    for avaDir in avaDirs:
        # Create copies of configs to avoid modifying originals
        cfgMainCopy = cfgUtils.convertDictToConfigParser(cfgUtils.convertConfigParserToDict(cfgMain))
        cfgMainCopy["MAIN"]["avalancheDir"] = str(avaDir)

        # Get com1DFA config with regional overrides (same as in processAvaDirCom1Regional)
        cfgCom1DFA = cfgUtils.getModuleConfig(
            com1DFA,
            str(avaDir),
            fileOverride="",
            toPrint=False,
            onlyDefault=cfgCom7["com1DFA_com1DFA_override"].getboolean("defaultConfig"),
        )
        cfgCom1DFA, _ = cfgHandling.applyCfgOverride(cfgCom1DFA, cfgCom7, com1DFA, addModValues=False)

        # Get simulations for this directory
        try:
            simDict, _, _, _ = com1DFA.com1DFAPreprocess(cfgMainCopy, "cfgFromObject", cfgCom1DFA)
            totalSims += len(simDict)
        except Exception as e:
            log.warning(f"Could not get simulations for {avaDir}: {e}")
            continue

    return totalSims


def processAvaDirCom1Regional(cfgMain, cfgCom7, avalancheDir):
    """Run com1DFA simulation in a specific avalanche directory with regional settings.

    Note: This function calls com1DFA within each avalanche directory within the input directory.
    If wanted it may be used as a template to call another operation within each directory, such as com2AB, ana5Utils, etc.

    Parameters
    ----------
    cfgMain : configparser.ConfigParser
        Main configuration settings
    cfgCom7 : configparser.ConfigParser
        Regional configuration settings with potential overrides
    avalancheDir : pathlib.Path or str
        Path to the avalanche directory to process

    Returns
    -------
    avalancheDir : pathlib.Path or str
        Path to the avalanche directory that was processed
    status : str
        Status of the simulation, "Success" if completed
    """
    # Initialize log for each process
    log = logUtils.initiateLogger(avalancheDir, logName="runCom1DFA")
    log.info("COM1DFA PROCESS CALLED BY COM7REGIONAL RUN")
    log.info("Current avalanche: %s", avalancheDir)

    # Update cfgMain setting to reflect the current avalancheDir
    cfgMain["MAIN"]["avalancheDir"] = str(avalancheDir)

    # Clean input directory of old work and output files from module
    initProj.cleanModuleFiles(avalancheDir, com1DFA, deleteOutput=True)

    # Create com1DFA configuration for the current avalanche directory and override with regional settings
    cfgCom1DFA = cfgUtils.getModuleConfig(
        com1DFA,
        str(avalancheDir),
        fileOverride="",
        toPrint=False,
        onlyDefault=cfgCom7["com1DFA_com1DFA_override"].getboolean("defaultConfig"),
    )
    cfgCom1DFA, cfgCom7 = cfgHandling.applyCfgOverride(cfgCom1DFA, cfgCom7, com1DFA, addModValues=False)

    # Run com1DFA in the current avalanche directory
    com1DFA.com1DFAMain(cfgMain, cfgInfo=cfgCom1DFA)

    return avalancheDir, "Success"


def moveOrCopyPeakFiles(cfg, avalancheDir):
    """Collects peak files from multiple sub-avalanche directories.

    Creates directory allPeakFiles: Contains peak files from all avalanche directories

    Parameters
    ----------
    cfg : configparser.ConfigParser
        Configuration containing GENERAL settings:
        - copyPeakFiles: If True, copy/move files; if False, do nothing
        - moveInsteadOfCopy: If True, move files instead of copying
    avalancheDir : pathlib.Path or str
        Base directory where allPeakFiles will be created

    Returns
    -------
    allPeakFilesDir : pathlib.Path or None
        Path to the created allPeakFiles directory or None if copyPeakFiles is False
    """
    if not cfg["GENERAL"].getboolean("copyPeakFiles"):
        log.info("copyPeakFiles is False - no files will be copied or moved")
        return None, None

    # Get avalanche directories
    # with logUtils.silentLogger():
    avaDirs = findAvaDirsBasedOnInputsDir(avalancheDir)
    if not avaDirs:
        log.warning("No avalanche directories found to copy/move files from")
        return None, None

    # Set up outdirs
    allPeakFilesDir = pathlib.Path(avalancheDir, "allPeakFiles")
    for dirPath in [allPeakFilesDir]:
        if dirPath.exists():
            shutil.rmtree(str(dirPath))
        fU.makeADir(dirPath)

    # Get operation type
    operation = shutil.move if cfg["GENERAL"].getboolean("moveInsteadOfCopy") else shutil.copy2
    operationType = "Moving" if operation == shutil.move else "Copying"

    # Process each avalanche directory
    for avaDir in avaDirs:
        log.info(f"{operationType} files from: {avaDir.name}")

        # Process peak files
        peakFiles = list(avaDir.glob("Outputs/**/peakFiles/*.*"))
        for peakFile in peakFiles:
            targetPath = allPeakFilesDir / peakFile.name
            operation(str(peakFile), str(targetPath))

    return allPeakFilesDir


def getRasterBounds(rasterFiles):
    """Get the union bounds and validate cell sizes of multiple rasters.

    Parameters
    ----------
    rasterFiles : list
        List of paths to raster files

    Returns
    -------
    bounds : dict
        Dictionary containing xMin, yMin, xMax, yMax of the union bounds
    cellSize : float
        Cell size of the rasters

    Raises
    ------
    ValueError
        If cell sizes of rasters differ
    """
    # Read first raster to get cellSize and initialize bounds
    firstRaster = rasterUtils.readRaster(rasterFiles[0])
    cellSize = float(firstRaster["header"]["cellsize"])
    bounds = {
        "xMin": float("inf"),
        "yMin": float("inf"),
        "xMax": float("-inf"),
        "yMax": float("-inf"),
    }

    # Find bounds and validate cell sizes
    for rasterFile in rasterFiles:
        raster = rasterUtils.readRaster(rasterFile)
        header = raster["header"]

        if float(header["cellsize"]) != cellSize:
            raise ValueError(f"Different cell sizes found: {cellSize} vs {header['cellsize']}")

        # Update bounds
        bounds["xMin"] = min(bounds["xMin"], float(header["xllcenter"]))
        bounds["yMin"] = min(bounds["yMin"], float(header["yllcenter"]))
        bounds["xMax"] = max(
            bounds["xMax"],
            float(header["xllcenter"]) + float(header["ncols"]) * cellSize,
        )
        bounds["yMax"] = max(
            bounds["yMax"],
            float(header["yllcenter"]) + float(header["nrows"]) * cellSize,
        )

    return bounds, cellSize


def mergeRasters(rasterFiles, bounds, mergeMethod="max"):
    """Merge multiple rasters into a single raster.

    Parameters
    ----------
    rasterFiles : list
        List of paths to raster files
    bounds : dict
        Dictionary containing xMin, yMin, xMax, yMax of the union bounds
    mergeMethod : str, optional
        Method to use for merging overlapping cells. Options:
        - 'max': maximum value (default)
        - 'min': minimum value
        - 'sum': sum of values
        - 'count': number of overlapping valid results per cell

    Returns
    -------
    mergedHeader : dict
        Header dictionary containing ncols, nrows, xllcenter, yllcenter, cellsize, nodata_value
    mergedData : numpy.ndarray
        2D array containing the merged raster data
    """

    # Merge data with rasterio
    # If something other than min/max is wanted, it is possible to provide a custom function to merge
    mergedData, outputTransform = merge(rasterFiles, method=mergeMethod, masked=True)

    mergedData = np.squeeze(mergedData)

    # Calculate dimensions for merged raster; helps checking if merged raster is correct
    nCols = int((bounds["xMax"] - bounds["xMin"]) / outputTransform[0])
    nRows = int((bounds["yMax"] - bounds["yMin"]) / outputTransform[0])
    #
    # # Create merged raster header
    exampleRaster = rasterUtils.readRaster(rasterFiles[0])
    mergedHeader = exampleRaster["header"]
    mergedHeader["ncols"] = nCols
    mergedHeader["nrows"] = nRows
    mergedHeader["xllcenter"] = float(bounds["xMin"])
    mergedHeader["yllcenter"] = float(bounds["yMin"])
    mergedHeader["transform"] = outputTransform

    return mergedHeader, mergedData


def mergeOutputRasters(cfg, avalancheDir):
    """Merge output rasters (peakFiles) from all avalanche simulations.

    Parameters
    ----------
    cfg : configparser.ConfigParser
        Configuration containing settings:
        - GENERAL.mergeOutput: If True, merge rasters
        - GENERAL.mergeTypes: Types of rasters to merge (e.g., 'ppr|pfv|pft')
        - GENERAL.mergeMethods: Methods to use for merging (e.g., 'max')
    avalancheDir : pathlib.Path or str
        Base directory where merged files will be saved

    Returns
    -------
    mergedRastersDir : pathlib.Path or None
        Path to the directory containing merged rasters or None if mergeOutput is False
    """
    if not cfg["GENERAL"].getboolean("mergeOutput", False):
        log.info("mergeOutput is False - no rasters will be merged")
        return None

    # Get all avalanche directories
    # with logUtils.silentLogger():
    avaDirs = findAvaDirsBasedOnInputsDir(avalancheDir)
    if not avaDirs:
        log.warning("No avalanche directories found to merge")
        return None

    # Set up merged rasters directory
    mergedRastersDir = pathlib.Path(avalancheDir, "mergedRasters")
    if mergedRastersDir.exists():
        shutil.rmtree(str(mergedRastersDir))
    mergedRastersDir.mkdir(parents=True, exist_ok=True)

    # Get types to merge
    mergeTypes = cfg["GENERAL"].get("mergeTypes").split("|")
    log.info(f"Merging raster types: {mergeTypes}")

    # Get merge methods
    mergeMethods = cfg["GENERAL"].get("mergeMethods", "max").lower().split("|")
    log.info(f"Using merge methods: {mergeMethods}")

    # Validate merge methods
    validMethods = {"max", "min", "mean", "sum", "count"}
    invalidMethods = set(mergeMethods) - validMethods
    if invalidMethods:
        raise ValueError(f"Invalid merge methods: {invalidMethods}. Valid options are: {validMethods}")

    # Process each raster type
    for rasterType in mergeTypes:
        # Find all files of this type across all avalanche directories
        rasterFiles = []
        for avaDir in avaDirs:
            peakFilesDir = avaDir / "Outputs" / "com1DFA" / "peakFiles"
            if peakFilesDir.is_dir():
                rasterFiles.extend(list(peakFilesDir.glob(f"*_{rasterType}.*")))

        if not rasterFiles:
            log.warning(f"No {rasterType} rasters found to merge")
            continue

        log.info(f"Merging {len(rasterFiles)} {rasterType} rasters")

        # Get bounds and validate cell sizes
        bounds, cellSize = getRasterBounds(rasterFiles)

        # Merge and save rasters
        for mergeMethod in mergeMethods:
            mergedHeader, mergedData = mergeRasters(rasterFiles, bounds, mergeMethod=mergeMethod)
            outputPath = mergedRastersDir / f"merged_{rasterType}_{mergeMethod}"
            rasterUtils.writeResultToRaster(mergedHeader, mergedData, outputPath, flip=False)
            log.info(f"Saved merged {rasterType} raster (method: {mergeMethod}) to: {outputPath}")

    return mergedRastersDir
