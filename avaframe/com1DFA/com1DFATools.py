"""
Tools specific to the com1DFA computational kernel
"""

import configparser

# Load modules
import logging
import math
import pathlib
import numpy as np

from deepdiff import DeepDiff

# local imports
import avaframe.com1DFA.deriveParameterSet as dP
import avaframe.in3Utils.initialiseDirs as inDirs
from avaframe.com1DFA import com1DFA
from avaframe.in1Data import getInput as gI
from avaframe.in3Utils import cfgUtils
from avaframe.in2Trans import rasterUtils as IOf


# create local logger
# change log level in calling module to DEBUG to see log messages
log = logging.getLogger(__name__)


def getPartInitMethod(cfg, csz, relThForPart):
    """Get particle initialization parameters

    Get the massPerPart and nPPK corresponding to the desired initialization method

    Parameters
    ----------
    cfg: configparser
        configuration for DFA simulation
    csz: float
        cell size
    relThForPart: float
        relTh value

    Returns
    -------
    massPerPart : float
        mass per particle desired
    nPPK : int
        number of particles per kernel radius desired
    """
    rho = cfg.getfloat("rho")
    massPerParticleDeterminationMethod = cfg["massPerParticleDeterminationMethod"]
    nPPK = 0
    # derive mass per particle to define number of particles per cell:
    if massPerParticleDeterminationMethod == "MPPDIR":
        massPerPart = cfg.getfloat("massPerPart")
        log.debug("Number of particles defined by: mass per particle %s" % cfg["massPerPart"])
    elif massPerParticleDeterminationMethod == "MPPDH":
        deltaTh = cfg.getfloat("deltaTh")
        ds = min(csz, cfg.getfloat("sphKernelRadius"))
        massPerPart = rho * ds * ds * deltaTh
        log.debug("Number of particles defined by: release thickness per particle: %s" % cfg["deltaTh"])
        log.debug("mass per particle is %.2f" % massPerPart)
    elif massPerParticleDeterminationMethod == "MPPKR":
        sphKernelRadius = cfg.getfloat("sphKernelRadius")
        cszMin = min(csz, sphKernelRadius)
        nPPK0 = cfg.getfloat("nPPK0")
        sphKR0 = cfg.getfloat("sphKR0")
        aPPK = cfg.getfloat("aPPK")
        nPPK = round(nPPK0 * (cszMin / sphKR0) ** aPPK)
        massPerPart = rho * math.pi * cszMin * cszMin * relThForPart / nPPK

    return massPerPart, nPPK


def setFrictTypeIndicator(simCfg):
    """Sets the friction type indicator for the simname
    Default is L, otherwise M for samosATMedium, S for samosATSmall

    Parameters
    -----------
    simCfg: dict
        simulation configuration

    Returns
    --------
    frictTypeIdentifier: str
        L if samosAT,  S for samosATSmall, M for samosATMedium

    """

    frictTypeIdentifier = "L"

    if simCfg["GENERAL"]["frictModel"].lower() == "samosatsmall":
        frictTypeIdentifier = "S"
    if simCfg["GENERAL"]["frictModel"].lower() == "samosatmedium":
        frictTypeIdentifier = "M"

    return frictTypeIdentifier


def compareSimCfgToDefaultCfgCom1DFA(simCfg, module=com1DFA):
    """Compares the given simulation configuration (as dict) to the default
    com1DFA configuration. Disregards values like avalancheDir that are expected to
    change. Returns True if it is the default + an identifier string: D = Default and
    C = Changed

    Parameters
    -----------
    simCfg: dict
        simulation configuration
    module: module
        module to be used for task (optional)

    Returns
    --------
    defaultIdentifierString: str
        D if default and C if changed

    """

    # extract the name of the module
    modName = module.__name__.split(".")[-1]

    log.info("Comparing simCfg to default cfg")

    defaultIdentifierString = "D"

    # Get default cfg and convert to dict for comparison
    defCfgObject = cfgUtils.getDefaultModuleConfig(com1DFA, toPrint=False)
    defCfg = cfgUtils.convertConfigParserToDict(defCfgObject)

    # Which changes to ignore (in the case of com1DFA). These are expected
    # to change...
    excludeItems = [
        "root['GENERAL']['avalancheDir']",
        "root['GENERAL']['secRelArea']",
        "root['GENERAL']['simTypeList']",
        "root['INPUT']['releaseScenario']",
    ]

    # If entrainment is requested, and it is set in shapefile, check if it contains the default entrainment thickness
    # in ALL features of the shapefile
    if simCfg["GENERAL"]["simTypeList"] == "ent" and simCfg["GENERAL"]["entThFromFile"] == "True":
        defaultEntTh = defCfg["GENERAL"]["entThIfMissingInShp"]

        # this try handles raster instead of shapefile
        try:
            if not all([x == defaultEntTh for x in simCfg["INPUT"]["entThThickness"].split("|")]):
                defaultIdentifierString = "C"
                log.info("Non-default entrainment value(s) used: %s" % simCfg["INPUT"]["entThThickness"])
        except KeyError:
            defaultIdentifierString = "D"

    # Entrainment might not be set in shpfile, but still the default from
    # ini file is used. This is still default D and not changed C
    if simCfg["GENERAL"]["entThFromFile"] == "False":
        # check if entTh is the default entTh if no other entTh is set
        if simCfg["GENERAL"]["entTh"] == defCfg["GENERAL"]["entThIfMissingInShp"]:
            excludeItems.append("root['GENERAL']['entThFromFile']")
            excludeItems.append("root['GENERAL']['entTh']")

    # sphKernelSize is set during runtime, make sure it is not reported
    # as changed if default is set to meshCellSize
    if modName == "com1DFA":
        if defCfg["GENERAL"]["sphKernelRadius"] == "meshCellSize":
            if simCfg["GENERAL"]["sphKernelRadius"] == simCfg["GENERAL"]["meshCellSize"]:
                excludeItems.append("root['GENERAL']['sphKernelRadius']")

    # do the diff and analyse
    # this is the deepdiff > 8.0 version
    # TODO: remove this again in the future when deepdiff > 8.0 is wider established
    try:
        diff = DeepDiff(defCfg, simCfg, exclude_paths=excludeItems, threshold_to_diff_deeper=0)
    # for older deepdiff versions which don't know threshold_to_diff_deeper
    except ValueError:
        diff = DeepDiff(defCfg, simCfg, exclude_paths=excludeItems)

    # Sometimes (after variation split) the type changes, so check if it is the default or something else
    # If it is, check the type_changes and convert to values_change if necessary
    if "type_changes" in diff:
        diff = _treatTypeChangeDiff(diff)

    valuesChanged = dict()
    # This needs to be checked AFTER check for type_changes
    if "values_changed" in diff:
        defaultIdentifierString = "C"
        for key, val in diff["values_changed"].items():
            valuesChanged[_cleanDiffKey(key)] = val

        log.info("Comparing to default cfg, values changed:")
        log.info(valuesChanged)

    else:
        valuesChanged = None

    if "dictionary_item_added" in diff:
        log.debug("Comparing to default cfg, added items:")
        log.debug(diff["dictionary_item_added"])

    return defaultIdentifierString, valuesChanged


def _treatTypeChangeDiff(diff):
    """Internal function to convert type changes in the result of DeepDiff to actual changes

    Parameters
    ----------
    diff:
        DeepDiff result to analyse

    Returns
    -------
    diff:
        DeepDiff result with changed values
    """

    allChangedValues = dict()

    for key, val in diff["type_changes"].items():
        # The type(x)(y) applies the type of x on y and gives the value of y
        if val["old_value"] == type(val["old_value"])(val["new_value"]):
            continue
        else:
            del val["old_type"]
            del val["new_type"]
            val["new_value"] = type(val["old_value"])(val["new_value"])
            allChangedValues[key] = val

    if allChangedValues:
        diff["values_changed"] = allChangedValues
    return diff


def _cleanDiffKey(keyString):
    """small helper function to change format of key for DeepDiff results

    Parameters
    ----------
    keyString : str
        string to change

    Returns
    -------
    cleanString: str
        cleaned string
    """

    keyStr = keyString.replace("root", "")
    keyStr = keyStr.replace("']['", "->")
    keyStr = keyStr.replace("['", "")
    keyStr = keyStr.replace("']", "")

    return keyStr


def createSimDictFromCfgs(cfgMain, cfgPath, module=com1DFA):
    """From multiple cfg files create a simDict with one item for each simulation to perform
    within these cfg files still parameter variations are allowed

    Parameters
    ------------
    cfgMain: configparser object
        main configuration of AvaFrame
    cfgPath: pathlib Path or str
        path to directory with cfg files
    module: module
        module to be used for task (optional)

    Returns
    --------
    simDict: dict
        dictionary with info on simHash, releaseScenario, release area file path,
        simType and contains full configuration configparser object for simulation run
    """

    # fetch avaDir
    avalancheDir = cfgMain["MAIN"]["avalancheDir"]

    # fetch input data and create work and output directories
    # TODO: so for now remeshed dir is cleaned before a run
    inputSimFilesAll, outDir, simDFExisting, simNameExisting = initializeInputs(avalancheDir, True, module)

    # save dem file path as it is deleted from input sim files dict once it is set in the config
    demFile = inputSimFilesAll["demFile"]

    # fetch all cfg files in configuration directory
    cfgDir = pathlib.Path(cfgPath)
    cfgFilesAll = list(cfgDir.glob("*.ini"))
    if len(cfgFilesAll) == 0:
        message = "No configuration file found to create simulation runs in: %s" % str(cfgDir)
        log.error(message)
        raise FileNotFoundError(message)
    else:
        log.info("Found %d configuration files" % len(cfgFilesAll))

    # initialize simDict
    simDictAll = {}

    # loop over all cfgFiles and create simDict
    for index, cfgFile in enumerate(cfgFilesAll):
        # read configuration
        cfgFromFile = cfgUtils.getModuleConfig(module, fileOverride=cfgFile, toPrint=False)

        # create dictionary with one key for each simulation that shall be performed
        # NOTE: sims that are added don't need to be added to the simNameExisting list as
        # if new identical sims are added the simDict entry is just updated and not a duplicate one added
        simDict = dP.createSimDict(avalancheDir, module, cfgFromFile, inputSimFilesAll, simNameExisting)
        simDictAll.update(simDict)

        # reset dem file
        inputSimFilesAll["demFile"] = demFile

    return simDictAll, inputSimFilesAll, simDFExisting, outDir


def initializeInputs(avalancheDir, cleanRemeshedRasters, module=com1DFA):
    """Create work and output directories, fetch input files and thickness info
    If cleanRemeshedRasters is true, the remesh folder contents will be deleted at the beginning

    Parameters
    -----------
    avalancheDir: pathlib path
        to avalanche directory
    cleanRemeshedRasters: bool
        flag if the remesh directory shall be cleaned
    module: module
        module to be used for task (optional)

    Returns
    --------
    inputSimFiles: dict
        dictionary with input files info
    outDir: str
        path to store outputs
    """

    # extract the name of the module
    modName = module.__name__.split(".")[-1]

    # Create output and work directories
    _, outDir = inDirs.initialiseRunDirs(avalancheDir, modName, cleanRemeshedRasters)

    # first fetch info on already existing simulations in Outputs
    # if need to reproduce exactly the hash - need to be strings with exactly the same number of digits!!
    # searchCfgFiles=True enables to search for preformed sims if run has been interrupted
    simDFExisting, simNameExisting = cfgUtils.readConfigurationInfoFromDone(
        avalancheDir,
        specDir="",
    )

    # fetch input data - dem, release-, entrainment- and resistance areas (and secondary release areas)
    inputSimFilesAll = gI.getInputDataCom1DFA(avalancheDir)

    # get thickness of release and entrainment areas (and secondary release areas) -if thFromShp = True
    inputSimFilesAll = gI.getThicknessInputSimFiles(inputSimFilesAll)

    return inputSimFilesAll, outDir, simDFExisting, simNameExisting


def checkCfgInfoType(cfgInfo):
    """check if cfgInfo is a configparser object, a file or a directory

    Parameters
    ------------
    cfgInfo: configparser object, str or pathlib path

    Returns
    ---------
    typeCfgInfo: str
        name of type of cfgInfo
    """

    if cfgInfo == "":
        typeCfgInfo = "cfgFromDefault"

    elif isinstance(cfgInfo, (pathlib.Path, str)):
        # if path is provided check if file or directory
        cfgInfoPath = pathlib.Path(cfgInfo)
        if cfgInfoPath.is_dir():
            typeCfgInfo = "cfgFromDir"
            log.info("----- CFG override from directory is used -----")
        elif cfgInfo.is_file():
            typeCfgInfo = "cfgFromFile"
            log.info("----- CFG override from file is used ----")

    elif isinstance(cfgInfo, configparser.ConfigParser):
        typeCfgInfo = "cfgFromObject"
        log.info("---- CFG override object is used ----")

    else:
        message = (
            "cfgInfo is not of valid format, needs to be a path to a cfg file, \
            directory, configparser object or an empty str, cfgInfo is: %s"
            % cfgInfo
        )
        log.error(message)
        raise AssertionError(message)

    return typeCfgInfo


def updateResCoeffFields(fields, cfg):
    """update fields of cRes and detK, coefficients of resistance parameter and detrainment parameter
    according to the thresholds of FV and FT
    if FV OR FT below min thresholds -> apply detrainment in that area
    if FV AND FT above min thresholds AND below max thresholds -> apply cResH (additional friction) in that area
    if FV OR FT above max thresholds -> no impact of resistance area

    Parameters
    ------------
    fields: dict
        dictionary with cResRasterOrig, cResRaster and detRasterOrig, detRaster fields
    cfg: configparser object
        configuration of com1DFA, thresholds

    Returns
    ------------
    fields: dict
        updated cRes and detK fields, required to write raster with header
    """

    # fetch cRes and detK raster and thresholds for FV and FT
    cResOrig = fields["cResRasterOrig"].copy()
    detOrig = fields["detRasterOrig"].copy()
    vMin = cfg.getfloat("forestVMin")
    thMin = cfg.getfloat("forestThMin")
    vMax = cfg.getfloat("forestVMax")
    thMax = cfg.getfloat("forestThMax")

    # create new rasters using FV, FT and thresholds to mask
    detRasterInt = np.where(((fields["FV"] <= vMin) | (fields["FT"] <= thMin)), detOrig, 0.0)
    detRaster = np.where(((fields["FV"] > vMax) | (fields["FT"] > thMax)), 0, detRasterInt)
    cResRasterInt = np.where(((fields["FV"] > vMin) & (fields["FT"] > thMin)), cResOrig, 0.0)
    cResRaster = np.where(((fields["FV"] > vMax) | (fields["FT"] > thMax)), 0, cResRasterInt)

    if len(np.where((detRaster > 0) & (cResRaster > 0))[0]) > 0:
        message = "Detrainment and increased friction within same cell!"
        log.error(message)
        raise AssertionError(message)

    # if max thresholds are exceeded: forest destroyed remove forest
    lTh = len(np.where((fields["FV"] > vMax) | (fields["FT"] > thMax))[0])
    if lTh > 0:
        cResOrig = np.where(((fields["FV"] > vMax) | (fields["FT"] > thMax)), 0, cResOrig)
        detOrig = np.where(((fields["FV"] > vMax) | (fields["FT"] > thMax)), 0, detOrig)
        fields["cResRasterOrig"] = cResOrig
        fields["detRasterOrig"] = detOrig
        log.debug(
            "Resistance area removed %d cells because FV or FT exceeded %.2f ms-1, %.2f m"
            % (lTh, vMax, thMax)
        )

    # update fields dictionary
    fields["cResRaster"] = cResRaster
    fields["detRaster"] = detRaster

    return fields


def chooseDemPlot(dem, adaptedDemBackground=False):
    """
    choose the DEM (rasterData) that is used for report plots

    Parameters
    -----------
    adaptedDemBackground: bool
        if True the adapted DEM is used as background in the plots

    Returns
    -------
    demPlot: dict
        contains the rasterdata that should be used in report plots as background
    """

    demPlot = dem.copy()
    # choose if the input DEM or the adapted DEM is used as background
    if adaptedDemBackground:
        demPlot["rasterData"] = dem["rasterData"]
        log.info("The resulting adapted DEM is used in the plots.")
    else:
        demPlot["rasterData"] = dem["originalRasterData"]
        log.info("The original DEM is used in the plots.")
    return demPlot
