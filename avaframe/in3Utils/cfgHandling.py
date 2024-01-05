#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Utilities for working with cfg info
"""

import logging
import numpy as np
import pathlib
import pandas as pd
import configparser

# Local imports
from avaframe.in3Utils import cfgUtils
import avaframe.in3Utils.fileHandlerUtils as fU

log = logging.getLogger(__name__)


def insertIntoSimName(name, keys, values, index):
    """Add keys and values to name, in between parts of name split by index

    Parameters
    -----------
    name: str
        name to extend
    keys: list
        list with keys
    values: list
        list with values
    index: str
        used to split name

    Returns
    --------
    newName: string
        containing newName, with keys and values inserted after index
    """

    # Split according to index
    splitName = name.split(index + "_")
    newPart = "_"

    # Loop through keys
    for key, value in zip(keys, values):
        newPart = newPart + str(key) + "_" + str(value) + "_"

    # Put newname back together
    try:
        newName = splitName[0] + str(index) + newPart + splitName[1]
    except IndexError:
        log.info(splitName)
        msg = "Some part is missing. SOMENAME_simHash_XXX is expected"
        log.error(msg)
        raise IndexError(msg)

    return newName


def addInfoToSimName(avalancheDir, csvString=""):
    """Add parameterName and value to simNames of simulation dataframe

    E.g used as helper routine for renaming layernames in qgis

    Parameters
    -----------
    avalancheDir: str
        path to avalanche directory
    csvString:
        comma separated list with parameter names, as found in com1DFA ini file
        eg. 'mu,tau0,tEnd'

    Returns
    --------
    simDF: dataframe
        containing index, the parameters and the old and new name
    """

    # read the allConfiigurationInfo
    simDF, _ = cfgUtils.readAllConfigurationInfo(avalancheDir)

    vars = csvString.split(",")

    for var in vars:
        # get the newName for every row by applying insertIntoSimName on each row
        simDF["newName"] = simDF.apply(
            lambda row: insertIntoSimName(row["simName"], vars, row[vars], row.name), axis=1
        )

    vars.append("simName")
    vars.append("newName")

    return simDF[vars]


def filterSims(avalancheDir, parametersDict, specDir="", simDF=""):
    """Filter simulations using a list of parameters and a pandas dataFrame of simulation configurations
    if ~ is used as a prefix for a parameter - it is filtered according to values that do NOT match the value
    provided with the ~Parameter

    Parameters
    -----------
    avalancheDir: str
        path to avalanche directory
    parametersDict: dict
        dictionary with parameter and parameter values for filtering
    specDir: str
        path to a directory where simulation configuration files can be found - optional
    simDF: pandas DataFrame
        optional - if simDF already available

    Returns
    --------
    simNameList: list
        list of simNames that match filtering criteria
    """

    if isinstance(simDF, pd.DataFrame) is False:
        # load dataFrame for all configurations
        simDF = cfgUtils.createConfigurationInfo(
            avalancheDir, standardCfg="", writeCSV=False, specDir=specDir
        )

    # filter simulations all conditions in the parametersDict have to be met
    if parametersDict != "":
        for key, value in parametersDict.items():
            # first check if values are valid
            if value == "" or value == []:
                log.debug(
                    "Parameter %s is not used for filtering as no valid value is provided: %s" % (key, value)
                )
                # required as np.float64 is False for np.float64 != []
            else:
                # convert values to list
                if not isinstance(value, (list, np.ndarray)):
                    value = [value]
                # remove non matching simulations from simDF
                if key in ["relTh", "entTh", "secondaryRelTh", "~relTh", "~entTh", "~secondaryRelTh"]:
                    simDF = filterCom1DFAThicknessValues(key, value, simDF)
                else:
                    simDF = removeSimsNotMatching(simDF, key, value)

    # list of simNames after filtering
    simNameList = simDF["simName"].tolist()
    return simNameList


def removeSimsNotMatching(simDF, key, value):
    """remove simulations from simDF that do not match filtering critera

    Parameters
    -----------
    simDF: pandas dataframe
        dataframe with one row per simulation and info on its characteristics, parameters used,..
    key: str
        name of parameter that shall be used for filtering
    value: list
        list of parameter values used for filtering

    Returns
    ---------
    simDF: pandas dataframe
        updated dataframe with only those simulations that match filtering criteria
    """

    # check if negation in filtering criteria
    notIn = False
    if "~" in key:
        # only add simulations that do not match the value of ~key
        key = key.replace("~", "")
        notIn = True

    # only keep simulations in simDF that match filtering criteria
    if isinstance(value[0], str):
        if "<" in value[0]:
            simDF = simDF[simDF[key] < float(value[0].split("<")[1])]
        elif ">" in value[0]:
            simDF = simDF[simDF[key] > float(value[0].split(">")[1])]
        else:
            if notIn:
                simDF = simDF[~simDF[key].isin(value)]
            else:
                simDF = simDF[simDF[key].isin(value)]
    else:
        # if float comparison allow for tolerance
        filterMask = np.isclose(simDF[key].values.reshape(-1, 1), value, atol=1.0e-7, rtol=1.0e-8).any(
            axis=1
        )
        if notIn:
            simDF = simDF[~filterMask]
        else:
            simDF = simDF[filterMask]

    return simDF


def orderSimulations(varParList, ascendingOrder, simDF):
    """Order simulations datadframe using a list of parameters and a flag if in ascending or descending order

    Parameters
    -----------
    varParList: str or list
        simulation configuration parameters for ordering simulations
    ascendingOrder: bool
        True if simulations shall be ordered in ascending order regarding varPar
    simDF: pandas dataFrame
        dataFrame of simulations (one line per simultaion with fileName, ... and values for parameters in
        varParList)

    Returns
    --------
    simDF: pandas dataFrame
        sorted dataFrame of simulation results (fileName, ... and values for parameters in varParList)
    """
    # make sure that parameters used for ordering are provided as list
    if isinstance(varParList, str):
        varParList = [varParList]
    # sort according to varParList and ascendingOrder flag
    # also check that key exists
    try:
        simDF = simDF.sort_values(by=varParList, ascending=ascendingOrder)
    except KeyError as e:
        message = "Choose a valid parameter for sorting the simulations. '%s' is not valid." % e.args[0]
        log.error(message)
        raise KeyError(message)
    return varParList, simDF


def fetchAndOrderSimFiles(avalancheDir, inputDir, varParList, ascendingOrder, specDir="", resFiles=False):
    """Filter simulations results using a list of parameters and a flag if in ascending or descending order

    Parameters
    -----------
    avalancheDir: str
        path to avalanche directory
    inputDir: str
        path to simulation results
    varParList: str or list
        simulation configuration parameters for ordering simulations
    ascendingOrder: bool
        True if simulations shall be ordered in ascending order regarding varPar
    specDir: str
        path to a directory where simulation configuration files can be found - optional

    Returns
    --------
    dataDF: pandas dataFrame
        dataFrame of simulation results (fileName, ... and values for parameters in varParList)
    """

    # load dataFrame for all configurations
    simDF = cfgUtils.createConfigurationInfo(avalancheDir, specDir=specDir)

    if resFiles:
        # create dataframe for simulation results in inputDir
        dataDF = fU.makeSimDF(inputDir)
        if isinstance(varParList, str):
            varParList = [varParList]
        # append 'simName' for merging of dataframes according to simNames
        columnNames = ["simName"] + varParList
        # merge varParList parameters as columns to dataDF for matching simNames
        dataDFNew = dataDF.merge(simDF[columnNames], left_on="simName", right_on="simName")
    else:
        dataDFNew = simDF

    varParList, dataDFNew = orderSimulations(varParList, ascendingOrder, dataDFNew)

    return dataDFNew


def orderSimFiles(avalancheDir, inputDir, varParList, ascendingOrder, specDir="", resFiles=False):
    """Filter simulations results using a list of parameters and a flag if in ascending or descending order

    Parameters
    -----------
    avalancheDir: str
        path to avalanche directory
    inputDir: str
        path to simulation results
    varParList: str or list
        simulation configuration parameters for ordering simulations
    ascendingOrder: bool
        True if simulations shall be ordered in ascending order regarding varPar
    specDir: str
        path to a directory where simulation configuration files can be found - optional

    Returns
    --------
    dataDF: pandas dataFrame
        dataFrame of simulation results (fileName, ... and values for parameters in varParList)
    """

    # load dataFrame for all configurations
    simDF = cfgUtils.createConfigurationInfo(avalancheDir, specDir=specDir)

    # make sure that parameters used for ordering are provided as list
    if isinstance(varParList, str):
        varParList = [varParList]

    if resFiles:
        # create dataframe for simulation results in inputDir
        dataDF = fU.makeSimDF(inputDir)
        # append 'simName' for merging of dataframes according to simNames
        columnNames = ["simName"] + varParList
        # merge varParList parameters as columns to dataDF for matching simNames
        dataDFNew = dataDF.merge(simDF[columnNames], left_on="simName", right_on="simName")
    else:
        dataDFNew = simDF

    # sort according to varParList and ascendingOrder flag
    dataDFNew = dataDFNew.sort_values(by=varParList, ascending=ascendingOrder)

    return dataDFNew


def filterCom1DFAThicknessValues(key, value, simDF):
    """thickness settings different if read from shpfile - requires more complex filtering
    if read from shp - thickness values are provided per feature!!
    for example relTh = '' but relTh0 = 1 is appended for feature with id 0, relTh1 for feature
    with id 1, etc.

    Parameters
    -----------
    key: str
        name of parameter
    value: list
        list of values used for filtering
    simDF: pandas dataframe
        configuration info for each simulation

    Returns
    --------
    simDF: pandas data frame
        updated dataframe
    """

    # check if filter for values that do NOT match criteria
    notIn = False
    if "~" in key:
        key = key.split("~")[1]
        notIn = True

    # create required parameters for searching
    thFlag = key + "FromFile"
    thId = key + "Id"
    thThickness = key + "Thickness"
    thPercentVariation = key + "PercentVariation"

    # append identifier if simulation matches thickness filter criteria
    simDF = pd.concat([simDF, pd.DataFrame({"toBeAdded": False}, index=simDF.index)], axis=1).copy()

    # initialize list for thickness parameter names (according to thickness configuration -
    # e.g. multiple features)
    allThNames = []
    # loop over simDF and set identifier if filter criteria are matched
    for simHash, simDFrow in simDF.iterrows():
        if simDFrow[thFlag] == "True":
            # inititialise thickness ids and thickness parameter names if thickness read from shp
            thIdList = str(simDFrow[thId]).split("|")
            thNames = [(key + id) for id in thIdList]
            allThNames = allThNames + thNames
            log.warning(
                "Filtering applied for %s - multiple features found as %s was read \
                from shp file - only simulations where all features match %s will be added"
                % (key, key, value)
            )
        else:
            # if thickness read from ini add thickness parameter name
            thIdList = [0]
            thNames = [key]
            allThNames = allThNames + [key]
        # check if filter criteria are met by thickness parameters for the sim in simDFrow
        for val in value:
            validationString = fetchValidationString(val, thIdList, thNames, simDFrow)
            # if we set new column value to True
            if validationString:
                simDF.loc[simHash, "toBeAdded"] = True

    # get a list with all thickness parameters included in search
    allThNames = list(set(allThNames))
    if notIn:
        # return all sims that do not match filter criteria
        simDF = simDF[simDF["toBeAdded"] == False]
    else:
        # return all sims that do match filter criteria
        simDF = simDF[simDF["toBeAdded"] == True]

    log.info("simulations for %s found with values: %s" % (key, simDF[allThNames]))

    return simDF


def fetchValidationString(val, thIdList, thNames, simDFrow):
    """create a validation string to be checked if simDFrow matches filtering criteria (given by val)

    Parameters
    -----------
    val: str, float
        value to be checked
    thIdList: list
        list with thickness feature ids
    thNames: list
        list with thickness feature names
    simDFrow: pandas dataframe row
        parameters of simulation

    Returns
    --------
    validationString: bool
        bool if simulation given by simDFrow matches filtering criteria
    """

    if isinstance(val, str):
        if "<" in val:
            validationString = (simDFrow[thNames].values < [float(val.split("<")[1])] * len(thIdList)).all()
        elif ">" in val:
            validationString = (simDFrow[thNames].values > [float(val.split(">")[1])] * len(thIdList)).all()
    else:
        validationString = (simDFrow[thNames].values == [val] * len(thIdList)).all()

    return validationString


def applyCfgOverride(cfgToOverride, cfgWithOverrideParameters, module, addModValues=False):
    """override configuration parameter values with the values provided in cfgWithOverrideParameters[modName_override]
    if addModValues True update the cfgWithOverrideParameters with the values for all parameters that are not
    provided in the override parameters

    Parameters
    ----------
    cfgToOverride: configparer object
        configuration of module of interest
    cfgWithOverrideParameters: configparser object
        full configuration settings containing a section modName_override with parameter values
        that should be overriden in the cfgToOverride
    module
        module of the cfgToOverride configuration
        OR pathlib path to module
    addModValues: bool
        if True add all parameters from cfgToOverride module to cfgWithOverrideParameters override
        section

    Returns
    --------
    cfgToOverride: configparser object
        updated configuration of module
    cfgWithOverrideParameters: configparser object
        updated configuration of module
    """

    # get filename of module
    if isinstance(module, pathlib.Path):
        modP = (module.parent).stem
        # get filename of module
        modName = module.stem
    else:
        # get path of module
        modP = (pathlib.Path(module.__file__).resolve().parent).stem
        # get filename of module
        modName = str(pathlib.Path(module.__file__).stem)

    # create list with parameters that become overridden
    overrideParameters = cfgWithOverrideParameters["%s_%s_override" % (modP, modName)]
    overrideKeys = [item for item in overrideParameters]
    overrideKeys.remove("defaultConfig")
    message = "duplicate parameter names appearing in override section"
    errorDuplicateListEntry(overrideKeys, message)

    # loop through sections of the configuration of the module
    foundKeys = []
    for section in cfgToOverride.sections():
        for key in overrideKeys:
            if cfgToOverride.has_option(section, key):
                cfgToOverride.set(section, key, overrideParameters[key])
                log.info(
                    "Override %s parameter: %s in section: %s with %s"
                    % (modName, key, section, str(overrideParameters[key]))
                )
                foundKeys.append(key)
    if addModValues:
        for section in cfgToOverride.sections():
            for key in cfgToOverride[section]:
                if key not in overrideKeys:
                    # if no override value is provided add actual configuration parameter to override section
                    # useful for reproduction if onlyDefault = False and modName config was read from local
                    cfgWithOverrideParameters["%s_%s_override" % (modP, modName)][key] = cfgToOverride[
                        section
                    ][key]
                    log.debug("Added %s: %s to override parameters " % (key, cfgToOverride[section][key]))

    # log warning if parameter in override was not found in modName configuration
    notOverride = set(foundKeys).symmetric_difference(set(overrideKeys))
    for item in notOverride:
        if item != "defaultConfig":
            log.warning(
                "Additional Key ['%s'] in section %s_%s_override is ignored." % (item, modP, modName)
            )

    # if an override key has been found in multiple sections - throw error
    message = (
        "duplicate parameter name appearing in sections of module config where override should be applied"
    )
    errorDuplicateListEntry(foundKeys, message)

    return cfgToOverride, cfgWithOverrideParameters


def errorDuplicateListEntry(listKeys, message):
    """check if duplicate entries appear in a list and raise Assertion error using
    message

    Parameters
    -----------
    listKeys: list
        list with keys
    message: str
        message of error
    """

    if len(listKeys) != len(list(set(listKeys))):
        log.error(message)
        raise AssertionError


def rewriteLocalCfgs(cfgFull, avalancheDir, localCfgPath=''):
    """fetch all override sections in cfgFull and write a local_NAMEOVERRIDE.ini configuration file for the
    available sections - naming is collection_module_override
    if no localCfgPath is provided, default saved to avalancheDir/Inputs/configurationOverrides
    where package refers to e.g. ana1Tests, ana3AIMEC, etc-
    and module to e.g. energyLineTest.py, ana3AIMEC.py so all python files inside the packages that
    have a nameCfg.ini file too

     Parameters
     -----------
     cfgFull: configparser
        configuration with override sections for modules
    avalancheDir: pathlib path or str
        path to avalanche directory
    localCfgPath: pathlib Path
        optional - path to directory to store local_ cfg ini file to
        if not provided - local_ cfg ini file is saved to avalanche directory

    Returns
    --------
    locFilePath: pathlib Path
        path to directory where local configuration files are written to derived from override sections

    """

    # if a path is provided - save local cfg ini file there
    pathProvided = False
    if localCfgPath != '':
        if pathlib.Path(localCfgPath).is_dir() is False:
            message1 = 'Provided path for local cfg files is not a directory: %s' % localCfgPath
            log.error(message1)
            raise NotADirectoryError(message1)
        else:
            pathProvided = True

    # Get all override sections
    cfgSections = cfgFull.sections()
    overrideSections = [sec for sec in cfgSections if "_override" in sec]

    # Check if all override sections have 3 parts separated by underscore
    if any(len((match := sec).split("_")) != 3 for sec in overrideSections):
        message = (
            "Override section needs to provide moduleName_fileName_override; provided: %s invalid format"
            % match
        )
        log.error(message)
        raise AssertionError(message)

    # Go through sections
    for section in overrideSections:
        modName = section.split("_")[0]
        cfgName = section.split("_")[1]
        thisFilePath = pathlib.Path(cfgUtils.__file__).resolve().parents[1]
        modPath = thisFilePath / modName
        cfgNamePath = modPath / cfgName
        locFilePath = modPath

        cfgModule = cfgUtils.getModuleConfig(
            cfgNamePath,
            '',
            fileOverride="",
            modInfo=False,
            toPrint=False,
            onlyDefault=cfgFull[section].getboolean("defaultConfig"),
        )
        cfgModule, cfgFull = applyCfgOverride(cfgModule, cfgFull, cfgNamePath, addModValues=False)

        overrideParameters = cfgFull["%s_%s_override" % (modName, cfgName)]
        overrideKeys = [item for item in overrideParameters]

        # remove items that are not in Override
        cfgModule = _removeCfgItemsNotInOverride(cfgModule, overrideKeys)

        # fetch directory to save local cfg ini file
        if pathProvided:
            locFilePath = pathlib.Path(localCfgPath)
        else:
            # if not provided save to default location
            locFilePath = pathlib.Path(avalancheDir, 'Inputs', 'configurationOverrides')
            fU.makeADir(locFilePath)

        cfgF = pathlib.Path(locFilePath, ("local_%sCfg.ini" % (cfgName)))
        if cfgF.is_file():
            warningText = "%s already exists - overwriting file here %s!" % (cfgF.name, cfgF)
            log.warning(warningText)
        with open(cfgF, "w") as configfile:
            cfgModule.write(configfile)

        log.info("%s CONFIGURATION wrote to %s" % (cfgName, str(cfgF)))

    return locFilePath


def _removeCfgItemsNotInOverride(cfgModule, overrideKeys):
    """ remove options of cfgModule if not part of overrideKeys
        in order to just have override parameters in new local cfg ini file

        Parameters
        ------------
        cfgModule: configparser object
            configuration of module
        overrideKeys: list
            list of options of configparser object that have been in override section and should be kept in cfgModule

        Returns
        ---------
        cfgModule: configparser object
            updated configuration - only override parameters left
    """

    for sec in cfgModule.sections():
        for item in cfgModule[sec]:
            if item not in overrideKeys:
                cfgModule.remove_option(sec, item)

    return cfgModule
