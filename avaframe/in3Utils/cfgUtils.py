"""
Utilities for handling configuration files

"""

import configparser
import logging
import pathlib
import hashlib
import json
import pandas as pd
import re
import math
import multiprocessing
from deepmerge import always_merger
from copy import deepcopy
from deepdiff import DeepDiff
from pprint import pformat
import numpy as np

# Local imports
import avaframe as avaf
from avaframe.in3Utils import logUtils
from avaframe.in3Utils import fileHandlerUtils as fU


log = logging.getLogger(__name__)


def getGeneralConfig(nameFile=""):
    """Returns the general configuration for avaframe
    returns a configParser object

    Parameters
    ----------
    nameFile: pathlib path
        optional full path to file, if empty use avaframeCfg from folder one level up
    """

    # get path of module
    modPath = pathlib.Path(avaf.__file__).resolve().parent

    if isinstance(nameFile, pathlib.Path):
        localFile = nameFile.parents[0] / ("local_" + nameFile.name)
        defaultFile = nameFile
    else:
        localFile = modPath / "local_avaframeCfg.ini"
        defaultFile = modPath / "avaframeCfg.ini"

    if localFile.is_file():
        iniFile = localFile
        iniFile = [defaultFile, localFile]
        compare = True
    elif defaultFile.is_file():
        iniFile = defaultFile
        compare = False
    else:
        raise FileNotFoundError("None of the provided cfg files exist ")

    # Finally read it
    cfg, _ = readCompareConfig(iniFile, "General", compare)

    return cfg


def getModuleConfig(module, fileOverride="", modInfo=False, toPrint=True, onlyDefault=False):
    """Returns the configuration for a given module
    returns a configParser object

    module object: module : the calling function provides the already imported
           module eg.:
           from avaframe.com2AB import com2AB
           leads to getModuleConfig(com2AB)
           whereas
           from avaframe.com2AB import com2AB as c2
           leads to getModuleConfig(c2)
           OR: pathlib Path to module (python file)

    Str: fileOverride : allows for a completely different file location. However note:
        missing values from the default cfg will always be added!

    modInfo: bool
        true if dictionary with info on differences to standard config
    onlyDefault: bool
        if True, only use the default configuration

    Order is as follows:
    fileOverride -> local_MODULECfg.ini -> MODULECfg.ini

    """

    if isinstance(onlyDefault, bool) == False:
        message = "OnlyDefault parameter is not a boolean but %s" % type(onlyDefault)
        log.error(message)
        raise TypeError(message)

    if isinstance(module, pathlib.Path):
        modPath = module.parent
        # get filename of module
        modName = module.stem
    else:
        modPath, modName = getModPathName(module)

    localFile = modPath / ("local_" + modName + "Cfg.ini")
    defaultFile = modPath / (modName + "Cfg.ini")

    log.debug("localFile: %s", localFile)
    log.debug("defaultFile: %s", defaultFile)

    # Decide which one to take
    if fileOverride:
        fileOverride = fU.checkPathlib(fileOverride)
        if fileOverride.is_file():
            iniFile = [defaultFile, fileOverride]
            compare = True
        else:
            raise FileNotFoundError("Provided fileOverride does not exist: " + str(fileOverride))

    elif localFile.is_file() and not onlyDefault:
        iniFile = localFile
        iniFile = [defaultFile, localFile]
        compare = True
    elif defaultFile.is_file():
        iniFile = defaultFile
        compare = False
    else:
        raise FileNotFoundError("None of the provided cfg files exist ")

    # Finally read it
    cfg, modDict = readCompareConfig(iniFile, modName, compare, toPrint)
    if modInfo:
        return cfg, modDict

    return cfg


def getDefaultModuleConfig(module, toPrint=True):
    """Returns the default configuration for a given module
    returns a configParser object

    module object: module : the calling function provides the already imported
           module eg.:
           from avaframe.com2AB import com2AB
           leads to getModuleConfig(com2AB)
           whereas
           from avaframe.com2AB import com2AB as c2
           leads to getModuleConfig(c2)

    """

    # get path to the module and its name
    modPath, modName = getModPathName(module)

    defaultFile = modPath / (modName + "Cfg.ini")

    log.info("Getting the default config for %s", modName)
    log.debug("defaultFile: %s", defaultFile)

    # Finally read it
    cfg, _ = readCompareConfig(defaultFile, modName, compare=False, toPrint=toPrint)

    return cfg


def readCompareConfig(iniFile, modName, compare, toPrint=True):
    """Read and optionally compare configuration files (if a local and default are both provided)
    and inform user of the eventual differences. Take the default as reference.

    Parameters
    ----------
    iniFile: path to config file
        Only one path if compare=False
    compare: boolean
        True if two paths are provided and a comparison is needed
    toPrint: boolean
        True (default) to print configuration to terminal. Differences to default
        will ALWAYS be printed

    Returns
    -------
    Output: ConfigParser object
        contains combined config
    modDict: dict
        dictionary containing only differences from default
    """

    if compare:
        log.info("Reading config from: %s and %s" % (iniFile[0], iniFile[1]))
        # initialize configparser object to read
        defCfg = configparser.ConfigParser()
        defCfg.optionxform = str
        locCfg = configparser.ConfigParser()
        locCfg.optionxform = str
        # read default and local parser files
        defCfg.read(iniFile[0])
        locCfg.read(iniFile[1])
        log.debug("Writing cfg for: %s", modName)
        # compare to default config and get modification dictionary and config
        modDict, modCfg = compareTwoConfigs(defCfg, locCfg, toPrint=toPrint)

    else:
        log.info("Reading config from: %s", iniFile)
        # initialize our final configparser object
        modCfg = configparser.ConfigParser()
        modCfg.optionxform = str
        # Finally read it
        modCfg.read(iniFile)
        modDict = {}
        # Write config to log file
        if toPrint:
            logUtils.writeCfg2Log(modCfg, modName)

    return modCfg, modDict


def _splitDeepDiffValuesChangedItem(inKey, inVal):
    """splits one item of a deepdiff result into section, key, old value, new value

    Parameters
    -----------
    inKey: str
        key of a deepdiff changed_values item
    inVal: dict
        value of a deepdiff changed_values item

    Returns
    --------
    section: str
        section name of changed item
    key: str
        key name of changed item
    oldVal: str
        old value
    newVal: str
        new value
    """
    splitKey = re.findall(r"\[\s*['\"]([^'\"]+)['\"]\s*\]", inKey)
    section = splitKey[0]
    key = splitKey[1]

    return section, key, inVal["old_value"], inVal["new_value"]


def compareTwoConfigs(defCfg, locCfg, toPrint=False):
    """compare locCfg to defCfg and return a cfg object and modification dict
    Values are merged from locCfg to defCfg:
    - parameters already in defCfg get the value from locCfg
    - additional values in locCfg get added in the resulting Cfg

    Parameters
    -----------
    defCfg: configparser object
        default configuration
    locCfg: configuration object
        configuration that is compared to defCfg
    toPrint: bool
        flag if config shall be printed to log

    Returns
    --------
    modInfo: dict
        dictionary containing only differences from default
    cfg: configParser object
        contains combined config

    """

    log.info("Comparing two configs")

    # initialize modInfo and printOutInfo
    modInfo = dict()

    # Switch to dict
    defCfgD = convertConfigParserToDict(defCfg)
    locCfgD = convertConfigParserToDict(locCfg)

    # Get the difference info
    # this is the deepdiff > 8.0 version
    # TODO: remove this again in the future when deepdiff > 8.0 is wider
    # established
    try:
        cfgDiff = DeepDiff(defCfgD, locCfgD, threshold_to_diff_deeper=0)
    # for older deepdiff versions which don't know threshold_to_diff_deeper
    except ValueError:
        cfgDiff = DeepDiff(defCfgD, locCfgD)

    # Combine them, different keys are just added, for the same keys, the
    # local (right) value is used
    modCfgD = deepcopy(defCfgD)
    always_merger.merge(modCfgD, locCfgD)

    # Convert to ConfigParser
    modCfg = convertDictToConfigParser(modCfgD)
    modCfg.optionxform = str

    # Merge is done, from here on down it is only printout and modInfo creation

    # If toPrint is set, print full configuration:
    if toPrint:
        for line in pformat(modCfgD, sort_dicts=False).split("\n"):
            log.info(line)

    # Generate modInfo dictionary for output
    if "values_changed" in cfgDiff:
        for key, value in cfgDiff["values_changed"].items():
            section, itemKey, defValue, locValue = _splitDeepDiffValuesChangedItem(key, value)

            if section not in modInfo:
                modInfo[section] = {}

            modString = [locValue, defValue]
            modInfo[section][itemKey] = modString

    # Log changes
    log.info("COMPARING TO DEFAULT, THESE CHANGES HAPPENED:")
    for line in cfgDiff.pretty().split("\n"):
        log.info(line.replace("root", ""))

    return modInfo, modCfg


def writeCfgFile(avaDir, module, cfg, fileName="", filePath=""):
    """Save configuration used to text file in Outputs/moduleName/configurationFiles/modName.ini
    or optional to filePath and with fileName

    Parameters
    -----------
    avaDir: str
        path to avalanche directory
    module:
        module
    cfg: configparser object
        configuration settings
    fileName: str
        name of saved configuration file - optional
    filePath: str or pathlib path
        path where file should be saved to except file name - optional

    """

    # get filename of module
    name = pathlib.Path(module.__file__).name
    modName = name.split(".")[0]

    # set outputs
    if filePath == "":
        outDir = pathlib.Path(avaDir, "Outputs", modName, "configurationFiles")
        fU.makeADir(outDir)
    else:
        if filePath.is_dir():
            outDir = pathlib.Path(filePath)
        else:
            message = "%s is not a valid location for saving cfg file" % str(filePath)
            log.error(message)
            raise NotADirectoryError(message)

    # set path to file
    if fileName == "":
        fileName = modName
    pathToFile = pathlib.Path(outDir, "%s.ini" % (fileName))

    # write file
    with open(pathToFile, "w") as conf:
        cfg.write(conf)

    return pathToFile


def readCfgFile(avaDir, module="", fileName=""):
    """Read configuration from ini file, if module is provided, module configuration is read from Ouputs,
    if fileName is provided configuration is read from fileName

    Parameters
    -----------
    avaDir: str
        path to avalanche directory
    module:
        module
    fileName: str
        path to file that should be read - optional

    Returns
    --------
    cfg: configParser object
        configuration that is from file

    """

    # define file that should be read
    if fileName != "":
        inFile = fileName
    elif module != "":
        # get module name
        name = pathlib.Path(module.__file__).name
        modName = name.split(".")[0]
        # set input file
        inFile = pathlib.Path(avaDir, "Outputs", "%s_settings.ini" % (modName))
    else:
        log.error("Please provide either a module or a fileName to read configuration from file")
        raise NameError

    # read configParser object from input file, case sensitive
    cfg = configparser.ConfigParser()
    cfg.optionxform = str
    cfg.read(inFile)
    cfg.optionxform = str

    return cfg


def cfgHash(cfg, typeDict=False):
    """UID hash of a config. Given a configParser object cfg,
    or a dictionary - then typeDict=True, returns a uid hash

    Parameters
    ----------
    cfg: configParser object
    typeDict : dict
        dictionary

    Returns:
    --------
    uid: str
       uid hash
    """

    uidHash = hashlib.shake_256()

    if typeDict:
        cfgDict = cfg
    else:
        cfgDict = convertConfigParserToDict(cfg)

    jsonDict = json.dumps(cfgDict, sort_keys=True, ensure_ascii=True)
    encoded = jsonDict.encode()

    uidHash.update(encoded)
    uid = uidHash.hexdigest(5)

    return uid


def convertConfigParserToDict(cfg):
    """create dictionary from configparser object"""

    cfgDict = {}
    for section in cfg.sections():
        cfgDict[section] = {}
        for key, val in cfg.items(section):
            cfgDict[section][key] = val

    return cfgDict


def convertDictToConfigParser(cfgDict):
    """create configParser object from dict"""

    cfg = configparser.ConfigParser()
    cfg.optionxform = str
    for section in cfgDict:
        cfg[section] = cfgDict[section]

    return cfg


def writeDictToJson(inDict, outFilePath):
    """write a dictionary to a json file"""

    jsonDict = json.dumps(inDict, sort_keys=True, ensure_ascii=True)
    f = open(outFilePath, "w")
    f.write(jsonDict)
    f.close()


def createConfigurationInfo(
    avaDir,
    comModule="com1DFA",
    standardCfg="",
    writeCSV=False,
    specDir="",
    simNameList=[],
):
    """Read configurations from all simulations configuration ini files from directory

    Parameters
    -----------
    avaDir: str
        path to avalanche directory
    standardCfg: dict
        standard configuration for module - option
    writeCSV: bool
        True if configuration dataFrame shall be written to csv file
    specDir: str
        path to a directory where simulation configuration files can be found - optional
    simNameList: list
        if non-empty list only use cfgFiles that are included within simNameList

    Returns
    --------
    simDF: pandas DataFrame
        DF with all the simulation configurations
    """

    # collect all configuration files for this module from directory
    if specDir != "":
        inDir = pathlib.Path(specDir, "configurationFiles")
    else:
        inDir = pathlib.Path(avaDir, "Outputs", comModule, "configurationFiles")
    configFiles = list(inDir.glob("*.ini"))

    if not inDir.is_dir():
        message = "configuration file directory not found: %s" % (inDir)
        log.error(message)
        raise NotADirectoryError(message)
    elif configFiles == []:
        message = "No configuration file found in: %s" % (inDir)
        log.error(message)
        raise FileNotFoundError(message)

    # if a simNameList is provided only look for the files with matching simName
    if simNameList != []:
        configFiles = [cfgF for cfgF in configFiles if cfgF.stem in simNameList]

    if len(configFiles) == 0:
        simDF = None
    else:
        # create configparser object, convert to json object, write to dataFrame
        # append all dataFrames
        simDF = ""
        for cFile in configFiles:
            if "sourceConfiguration" not in str(cFile):
                simName = pathlib.Path(cFile).stem
                if "_AF_" in simName:
                    nameParts = simName.split("_AF_")
                    infoParts = nameParts[1].split("_")

                else:
                    nameParts = simName.split("_")
                    infoParts = nameParts[1:]
                simHash = infoParts[0]
                cfgObject = readCfgFile(avaDir, fileName=cFile)
                simDF = appendCgf2DF(simHash, simName, cfgObject, simDF)

        # convert numeric parameters to numerics
        simDF = convertDF2numerics(simDF)

        # add default configuration
        if standardCfg != "":
            # read default configuration of this module
            simDF = appendCgf2DF("current standard", "current standard", standardCfg, simDF)

        # if writeCSV, write dataFrame to csv file
        if writeCSV:
            writeAllConfigurationInfo(avaDir, simDF, specDir=specDir)

    return simDF


def appendCgf2DF(simHash, simName, cfgObject, simDF):
    """append simulation configuration to the simulation dataframe
    append all sections to the dataframe

    Parameters
    -----------
    simHash: str
        hash of the simulation to append
    simName: str
        name of the simulation
    cfgObject: configParser
        configuration coresponding to the simulation
    simDF: pandas dataFrame
        configuration dataframe

    Returns
    --------
    simDF: pandas DataFrame
        DFappended with the new simulation configuration
    """
    indexItem = [simHash]
    cfgDict = convertConfigParserToDict(cfgObject)
    simItemDFList = []
    for section in cfgDict:
        simItemDFSection = pd.DataFrame(data=cfgDict[section], index=indexItem)
        simItemDFList.append(simItemDFSection)
    simItemDF = pd.concat(simItemDFList, axis=1)
    simItemDF = simItemDF.assign(simName=simName)

    # check for duplicates: if yes, rename them by adding Dupl1 to the duplicate name
    if simItemDF.columns.duplicated().any():
        renameDuplicates(simItemDF)

    if isinstance(simDF, str):
        simDF = simItemDF
    else:
        simDF = pd.concat([simDF, simItemDF], axis=0)
    return simDF


def renameDuplicates(df):
    """
    Rename duplicate column names in the given DataFrame. This ensures all column names in the DataFrame
    are unique by adding a suffix 'DuplX' where X is the occurrence number, starting
    from 1 for the first duplicate.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame whose column names need to be checked for duplicates.

    Returns
    -------
    bool
        Returns True to indicate the renaming of duplicate column names was successful.
    """
    seen = {}
    new_cols = []

    for col in df.columns:
        if col not in seen:
            seen[col] = 0
            new_cols.append(col)
        else:
            seen[col] += 1
            new_cols.append(f"{col}_{seen[col]}")

    df.columns = new_cols
    return True


def appendTcpu2DF(simHash, tCPU, tCPUDF):
    """append Tcpu dictionary to the dataframe

    Parameters
    -----------
    simHash: str
        hash of the simulation corresponding to the tCPU dict to append
    tCPU: dict
        cpu time dict of the simulation
    tCPUDF: pandas dataFrame
        tCPU dataframe

    Returns
    --------
    simDF: pandas DataFrame
        DFappended with the new simulation configuration
    """
    indexItem = [simHash]
    tCPUItemDF = pd.DataFrame(data=tCPU, index=indexItem)
    if isinstance(tCPUDF, str):
        tCPUDF = tCPUItemDF
    else:
        tCPUDF = pd.concat([tCPUDF, tCPUItemDF], axis=0)
    return tCPUDF


def convertDF2numerics(simDF):
    """convert a string DF to a numerical one

    Parameters
    -----------
    simDF: pandas dataFrame
        dataframe

    Returns
    --------
    simDF: pandas DataFrame
    """

    for name, values in simDF.items():
        simDFTest = simDF[name].str.replace(".", "", regex=False)
        # allow for - sign too
        simDFTest = simDFTest.replace("-", "", regex=False)
        # check for str(np.nan) as these cannot be converted to numerics by pd.to_numeric
        # but as friction model parameters are set to nans this is required here
        if simDFTest.str.match("nan").any():
            simDF = setStrnanToNan(simDF, simDFTest, name)
        # also include columns where nan is in first row - so check for any row
        if simDFTest.str.isdigit().any() and (name != "tSteps"):
            # problem here is that it finds even if not present in | although not in ini
            simDFTest = simDF[name].str.replace("|", "§", regex=False)
            if simDFTest.str.contains("§").any() == False:
                simDF[name] = pd.to_numeric(simDF[name])
                log.debug("Converted to numeric %s" % name)
        else:
            log.debug("Not converted to numeric: %s" % name)

    return simDF


def setStrnanToNan(simDF, simDFTest, name):
    """set pandas element to np.nan if it is a string nan

    Parameters
    -----------
    simDF: pandas dataFrame
        dataframe
    simDFTest: pandas series
        series of sim DF column named name
        replaced "." with " "
    name: str
        name of pandas dataframe column

    Returns
    --------
    simDF: pandas dataframe
        updated pandas dataframe with np.nan values where string nan was
    """

    nanIndex = simDFTest.str.match("nan", flags=re.IGNORECASE)
    simIndex = simDF.index.values
    # loop over each row and use simDF.at to avoid copy vs view warning
    for index, nanInd in enumerate(nanIndex):
        if nanInd:
            simDF.at[simIndex[index], name] = np.nan
            log.info("%s for index: %s set to numpy nan" % (name, index))
    return simDF


def readConfigurationInfoFromDone(avaDir, specDir="", latest=False):
    """Check avaName/Outputs/com1DFA/configurationFilesDone and pass
    names of all files found in this directory and create corresponding simDF
    this is useful if e.g. no allConfigurations.csv has
    been written but already some simulations have been performed as a txt file is saved in
    avaName/Outputs/com1DFA/configurationFiles after the respective simulation has been run
    whereas the allConfigurations file is written at the end of a call to com1DFAMain that can
    include several individual sims
    if latest=True only look for latest simulations in avaName/Outputs/com1DFA/configurationFilesLatest

    Parameters
    -----------
    avaDir: str
        path to avalanche directory
    specDir: str
        path to a directory where simulation configuration files directory called configurationFiles can be found - optional
    latest: bool
        if True check for files found in avaName/Outputs/com1DFA/configurationFilesLatest

    Returns
    --------
    simDF: pandas DataFrame
        DF with all the simulation configurations
    simDFName: array
        simName column of the dataframe
    """

    # collect all configuration files for this module from directory
    if specDir != "":
        inDir = pathlib.Path(specDir, "configurationFiles")
    else:
        inDir = pathlib.Path(avaDir, "Outputs", "com1DFA", "configurationFiles")

    # search inDir/configurationFilesDone or inDir/configurationFilesLatest (depending on latest flag) for already existing sims
    if latest:
        configDir = inDir / "configurationFilesLatest"
    else:
        configDir = inDir / "configurationFilesDone"

    existingSims = list(configDir.glob("*.ini"))

    simNameExisting = []
    for fName in existingSims:
        simNameExisting.append(fName.stem)
    if list((inDir / "configurationFilesDone").glob("*.ini")) == []:
        log.info("No existing simulations in Outputs found")
        simDF = None
    else:
        # create simDF (dataFrame with one row per simulation of configuration files found in configDir)
        simDF = createConfigurationInfo(
            avaDir,
            comModule="com1DFA",
            standardCfg="",
            writeCSV=False,
            specDir=specDir,
            simNameList=simNameExisting,
        )

    # check for allConfigurationsInfo to find computation info and add to info fetched from ini files
    if latest == False and isinstance(simDF, pd.DataFrame):
        # check if in allConfigurationsInfo also info for existing sims
        simDFALL, _ = readAllConfigurationInfo(avaDir, specDir="", configCsvName="allConfigurations")
        if isinstance(simDFALL, pd.DataFrame):
            simDF = (
                simDF.reset_index()
                .merge(
                    simDFALL[
                        [
                            "nPart",
                            "timeLoop",
                            "timeForce",
                            "timeForceSPH",
                            "timePos",
                            "timeNeigh",
                            "timeField",
                            "nSave",
                            "nIter",
                            "simName",
                        ]
                    ],
                    how="left",
                    on="simName",
                )
                .set_index("index")
            )

    return simDF, simNameExisting


def readAllConfigurationInfo(avaDir, specDir="", configCsvName="allConfigurations"):
    """Read allConfigurations.csv file as dataFrame from directory

    Parameters
    -----------
    avaDir: str
        path to avalanche directory
    specDir: str
        path to a directory where simulation configuration files can be found - optional
    configCsvName: str
        name of configuration csv file

    Returns
    --------
    simDF: pandas DataFrame
        DF with all the simulation configurations
    simDFName: array
        simName column of the dataframe
    """

    # collect all configuration files for this module from directory
    if specDir != "":
        inDir = pathlib.Path(specDir, "configurationFiles")
    else:
        inDir = pathlib.Path(avaDir, "Outputs", "com1DFA", "configurationFiles")
    configFiles = inDir / ("%s.csv" % configCsvName)

    if configFiles.is_file():
        with open(configFiles, "rb") as file:
            simDF = pd.read_csv(file, index_col=0, keep_default_na=False)
        simDFName = simDF["simName"].to_numpy()
    else:
        simDF = None
        simDFName = []

    return simDF, simDFName


def writeAllConfigurationInfo(avaDir, simDF, specDir="", csvName="allConfigurations.csv"):
    """Write cfg configuration to allConfigurations.csv

    Parameters
    -----------
    avaDir: str
        path to avalanche directory
    simDF: pandas dataFrame
        daaframe of the configuration
    specDir: str
        path to a directory where simulation configuration shal be saved - optional
    csvName: str
        name of csv file in which to save to - optional

    Returns
    --------
    configFiles: pathlib Path
        path where the configuration dataframe was saved
    """

    # collect all configuration files for this module from directory
    if specDir != "":
        inDir = pathlib.Path(specDir, "configurationFiles")
    else:
        inDir = pathlib.Path(avaDir, "Outputs", "com1DFA", "configurationFiles")
    configFiles = inDir / csvName

    simDF.to_csv(configFiles)

    return configFiles


def convertToCfgList(parameterList):
    """convert a list into a string where individual list items are separated by |

    Parameters
    -----------
    parameterList: list
        list of parameter values

    Returns
    ---------
    parameterString: str
        str with parameter values separated by |
    """

    if len(parameterList) == 0:
        parameterString = ""
    else:
        parameterString = parameterList[0]
        for item in parameterList[1:]:
            parameterString = parameterString + "|" + item

    return parameterString


def getNumberOfProcesses(cfgMain, nSims):
    """Determine how many CPU cores to take for parallel tasks

    Parameters
    -----------
    cfgMain: configuration object
        the main avaframe configuration
    nSims: integer
        number of simulations that need to be calculated


    Returns
    ---------
    nCPU: int
        number of cores to take
    """

    maxCPU = multiprocessing.cpu_count()

    if cfgMain["MAIN"]["nCPU"] == "auto":
        cpuPerc = float(cfgMain["MAIN"]["CPUPercent"]) / 100.0
        nCPU = math.floor(maxCPU * cpuPerc)
    else:
        nCPU = cfgMain["MAIN"].getint("nCPU")

    # if number of sims is lower than nCPU
    nCPU = min(nCPU, nSims)

    log.info("Number of tasks to perform: %s " % nSims)
    log.info("Taking %s cpu cores out of maximum of %s cores." % (nCPU, maxCPU))

    return nCPU


def getModPathName(module):
    """get the path and name of a module from imported module

    Parameters
    ------------
    module: imported module

    Returns
    --------
    modPath: pathlib path
        path to directory where module is located
    modName: str
        name of module

    """

    # get path of module
    modPath = pathlib.Path(module.__file__).resolve().parent

    # get filename of module
    modName = str(pathlib.Path(module.__file__).stem)

    return modPath, modName


def cfgToRcf(cfg, fileName):
    """Convert configuration object to RCF format file (used by NGI MoT).

    Takes a ConfigParser object and writes its contents to a file in rcf format,
    excluding certain sections and formatting others according to RCF requirements.

    Parameters
    ----------
    cfg : configparser.ConfigParser
        Configuration object containing sections and their key-value pairs
    fileName : str or pathlib.Path
        Path to the output file where the RCF format will be written
    """
    with open(fileName, "w") as f:
        for section in cfg.sections():
            if section in ("FOREST_EFFECTS", "ENTRAINMENT"):
                pass
            elif section in ("GENERAL", "INPUT"):
                continue
            else:
                f.write(f"# {section.replace('_', ' ')}\n")
                f.write("#\n")
            for key, value in cfg.items(section):
                # key = key.replace('_', ' ')
                key = key.strip()
                f.write(f"{key:<40}{value}\n")
            f.write("#\n")


def getGeneralResType(resType):
    """get general resType for resType with layer info and check if in available options

    Parameters
    --------------
    resType: string
        result type including layer info: e.g. pprL1

    Returns
    ---------
    resTypeGeneral: string
        general result type so for pprL1 this is ppr

    """

    generalResType = resType.split("L")[0]

    availableResTypes = [
        "ppr",
        "pft",
        "pfd",
        "pfv",
        "pta",
        "FT",
        "FD",
        "FV",
        "P",
        "FM",
        "Vx",
        "Vy",
        "Vz",
        "TA",
        "dmDet",
        "sfcChange",
        "demAdapted",
    ]

    if generalResType not in availableResTypes:
        log.error(
            "General result type %s is not available; derived from resType %s" % (generalResType, resType)
        )

    return generalResType
