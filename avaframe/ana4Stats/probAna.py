"""

This is a simple function for computing a probability map of all peak files of one parameter that
exceed a particular threshold

"""

import numpy as np
import logging
import pathlib
from scipy.stats import qmc
from SALib.sample import morris
import pickle
from deepdiff import grep

import avaframe.out3Plot.plotUtils as pU
from avaframe.in3Utils import cfgUtils
from avaframe.in3Utils import cfgHandling
from avaframe.in3Utils import fileHandlerUtils as fU
import avaframe.in2Trans.rasterUtils as IOf
import avaframe.in1Data.computeFromDistribution as cP
import avaframe.com1DFA.deriveParameterSet as dP
from avaframe.in3Utils import geoTrans as gT
from avaframe.out3Plot import statsPlots as sP
from avaframe.in1Data import getInput as gI


# create local logger
# change log level in calling module to DEBUG to see log messages
log = logging.getLogger(__name__)


def createComModConfig(cfgProb, avaDir, modName):
    """create configuration file for performing sims with modName com module

    Parameters
    -----------
    cfgProb: configParser object
        configuration settings
    avaDir: pathlib path
        path to avalanche directory
    modName: module
        computational module

    Returns
    -------
    cfgFiles: list
        list of paths to newly generated configuration files for com module inlcuding
        parameter variations
    """

    # setup where configuration file is saved
    modNameString = str(pathlib.Path(modName.__file__).stem)
    outDir = avaDir / "Work" / ("%sConfigFiles" % modNameString)
    fU.makeADir(outDir)

    # check variation settings
    variationsDict = makeDictFromVars(cfgProb["PROBRUN"])

    if cfgProb["PROBRUN"].getint("samplingStrategy") == 2:
        log.info(
            "Probability run performed by varying one parameter at a time - local approach."
        )
        cfgFiles = cfgFilesLocalApproach(variationsDict, cfgProb, modName, outDir)
    else:
        log.info("Probability run perfromed drawing parameter set from full sample.")
        cfgFiles = cfgFilesGlobalApproach(avaDir, cfgProb, modName, outDir)

    return cfgFiles, outDir


def cfgFilesGlobalApproach(avaDir, cfgProb, modName, outDir):
    """create configuration files with all parameter variations - drawn from full sample
    for performing sims with modName comModule

    Parameters
    -----------
    cfgProb: configParser object
        configuration settings
    avaDir: pathlib path
        path to avalanche directory
    modName: module
        computational module

    Returns
    -------
    cfgFiles: list
        list of paths to newly generated configuration files for com module inlcuding parameter
        variations
    """

    # create sample of all parameter variations
    paramValuesDList = createSampleFromConfig(avaDir, cfgProb, modName)

    # create plot of parameter sample if variation of two parameters
    for paramValuesD in paramValuesDList:
        if "releaseScenario" in paramValuesD.keys():
            releaseScenario = paramValuesD["releaseScenario"]
        else:
            releaseScenario = ""
        plotDir = avaDir / "Outputs" / "ana4Stats" / "plots"
        if len(paramValuesD["names"]) == 2:
            sP.plotSample(paramValuesD, plotDir, releaseScenario=releaseScenario)
        elif "varParNamesInitial" in paramValuesD.keys():
            if len(paramValuesD["varParNamesInitial"]) == 2:
                sP.plotThSampleFromVals(paramValuesD, plotDir)
        else:
            log.debug(
                "More or less than two parameters have been varied - no plot of sample available"
            )

    # write cfg files one for each parameter set drawn from full sample
    cfgFiles = createCfgFiles(paramValuesDList, modName, cfgProb, cfgPath=outDir)

    return cfgFiles


def cfgFilesLocalApproach(variationsDict, cfgProb, modName, outDir):
    """create configuration file for performing sims with modName com module

    Parameters
    -----------
    variationsDict: dict
        dictionary with for each varName, varVariation, varSteps, and type of variation
    cfgProb: configParser object
        configuration settings
    modName: module
        computational module

    Returns
    -------
    cfgFiles: dict
        dictionary of paths to newly generated configuration files for com module for all parameters

    """

    cfgFiles = []
    for varName in variationsDict:
        # define configuration files
        # get filename of module
        modNameString = str(pathlib.Path(modName.__file__).stem)
        cfgFile = outDir / ("probRun%sCfg%s.ini" % (modNameString, varName))

        # use cfgFile, local com module settings or default settings if local not available
        modCfg = fetchStartCfg(modName, cfgProb)
        modCfg = updateCfgRange(modCfg, cfgProb, varName, variationsDict[varName])

        with open(cfgFile, "w") as configfile:
            modCfg.write(configfile)
        # append cfgFiles to list
        cfgFiles.append(cfgFile)

    return cfgFiles


def updateCfgRange(cfg, cfgProb, varName, varDict):
    """update cfg with a range for parameters in cfgProb

    Parameters
    -----------
    cfg: configparser object
        configuration object to update
    cfgProb: configParser object
        configparser object with info on update
    varName: str
        name of parameter used for variation
    varDict: dict
        dictionary with variationValue and numberOfSteps for varName

    Returns
    --------
    cfg: configParser
        updated configuration object

    """

    # set reference values of parameters - override values in com module configurations
    varParList = cfgProb["PROBRUN"]["varParList"].split("|")
    # also for the other parameters that are varied subsequently
    # first check if no parameter variation in provided for these parameters in the com module ini
    # if so - error
    _, _ = checkParameterSettings(cfg, varParList)

    # this is now done for parameter VARNAME from inputs
    # get range, steps and reference value of parameter to perform variations
    valVariation = varDict["variationValue"]
    valSteps = varDict["numberOfSteps"]
    valVal = cfg["GENERAL"][varName]
    variationType = varDict["variationType"]

    if variationType.lower() == "normaldistribution":
        # get computeFromDistribution configuration and apply override
        cfgDist = cfgUtils.getModuleConfig(
            cP,
            fileOverride="",
            modInfo=False,
            toPrint=False,
            onlyDefault=cfgProb["in1Data_computeFromDistribution_override"].getboolean(
                "defaultConfig"
            ),
        )
        cfgDist, cfgProb = cfgHandling.applyCfgOverride(
            cfgDist, cfgProb, cP, addModValues=False
        )

    # set variation in configuration
    if varName in ["relTh", "entTh", "secondaryRelTh"]:
        # if variation using normal distribution
        if variationType.lower() == "normaldistribution":
            parName = varName + "DistVariation"
            if valVariation == "":
                valVariation = "-"
            parValue = (
                variationType
                + "$"
                + valSteps
                + "$"
                + valVariation
                + "$"
                + cfgDist["GENERAL"]["minMaxInterval"]
                + "$"
                + cfgDist["GENERAL"]["buildType"]
                + "$"
                + cfgDist["GENERAL"]["support"]
            )
        # if variation using percent
        elif variationType.lower() == "percent":
            parName = varName + "PercentVariation"
            parValue = valVariation + "$" + valSteps
        # if variation using absolute range
        elif variationType.lower() == "range":
            parName = varName + "RangeVariation"
            parValue = valVariation + "$" + valSteps
            if "ci" in valVariation:
                message = (
                    "Variation Type: range - variationValue is %s not a valid option - only \
                 scalar value allowed or consider variationType rangefromci"
                    % valVariation
                )
                log.error(message)
                raise AssertionError(message)
        elif variationType.lower() == "rangefromci":
            parName = varName + "RangeFromCiVariation"
            parValue = valVariation + "$" + valSteps
        else:
            message = (
                "Variation Type: %s - not a valid option, options are: percent, range, \
                normaldistribution, rangefromci"
                % variationType
            )
            log.error(message)
            raise AssertionError(message)
        # write parameter variation for varName in config file
        cfg["GENERAL"][parName] = parValue
    else:
        # set variation
        if variationType.lower() == "normaldistribution":
            cfgDist = {
                "sampleSize": valSteps,
                "mean": valVal,
                "buildType": cfgProb["in1Data_computeFromDistribution_override"][
                    "buildType"
                ],
                "buildValue": valVariation,
                "minMaxInterval": cfgDist["GENERAL"]["minMaxInterval"],
                "support": cfgDist["GENERAL"]["support"],
            }
            _, valValues, _, _ = cP.extractNormalDist(cfgDist)
            cfg["GENERAL"][varName] = dP.writeToCfgLine(valValues)
        elif variationType.lower() == "percent":
            cfg["GENERAL"][varName] = "%s$%s$%s" % (valVal, valVariation, valSteps)
            valValues = fU.splitIniValueToArraySteps(cfg["GENERAL"][varName])

        elif variationType.lower() == "range":
            if "-" in valVariation or "+" in valVariation:
                valStart = str(float(valVal) + float(valVariation))
                valStop = float(valVal)
            else:
                valStart = str(float(valVal) - float(valVariation))
                valStop = str(float(valVal) + float(valVariation))
            cfg["GENERAL"][varName] = "%s:%s:%s" % (valStart, valStop, valSteps)
            valValues = np.linspace(float(valStart), float(valStop), int(valSteps))
        else:
            message = (
                "Variation Type: %s - not a valid option, options are: percent, range, \
                normaldistribution, rangefromci"
                % variationType
            )
            log.error(message)
            raise AssertionError(message)

    # add a scenario Name to VISUALISATION
    cfg["VISUALISATION"]["scenario"] = varName

    return cfg


def checkParameterSettings(cfg, varParList):
    """check if parameter settings in comMod configuration do not include variation for parameters to be varied

    Parameters
    -----------
    cfg: configparser object
        configuration settings
    varParList: list
        list of parameters (names) that shall be varied

    """

    # set a list of all thickness parameters that are set to be read from shp file
    thReadFromShp = []

    for varPar in varParList:
        # Check if valid parameter exists in any section and check for duplicates
        _ = checkIfParameterInConfig(cfg, varPar)

        # Fetch section where parameter was found
        section = fetchParameterSection(cfg, varPar)

        if any(chars in cfg[section][varPar] for chars in ["|", "$", ":"]):
            message = "Only one reference value is allowed for %s: but %s is given" % (
                varPar,
                cfg[section][varPar],
            )
            log.error(message)
            raise AssertionError(message)
        elif varPar in ["entTh", "relTh", "secondaryRelTh"]:
            thFromFile = varPar + "FromFile"
            # check if reference settings have already variation of varPar
            _ = checkForNumberOfReferenceValues(cfg["GENERAL"], varPar)
            # check if th read from shp file
            if cfg["GENERAL"].getboolean(thFromFile):
                thReadFromShp.append(varPar)

    return True, thReadFromShp


def checkIfParameterInConfig(cfg, varPar):
    """
    Checks the existence and uniqueness of a parameter within a configuration object.

    This function searches for a specified parameter within a configuration object, ensuring the
    parameter exists and is not duplicated across sections. If the parameter is found multiple
    times or not found at all, an error is logged. If the parameter is found in only one section,
    that section is returned.

    Parameters
    ----------
    cfg : configparser object
        The configuration object in which to search for the specified parameter.
    varPar : str
        The name of the parameter to locate within the configuration.

    Returns
    -------
    True if the parameter is found and unique.
    """

    # search for parameter in cfg object
    matches = cfg | grep(varPar, verbose_level=2)

    # Check if matches is empty
    if not matches.get("matched_paths"):
        message = "'%s' is not a valid parameter" % varPar
        log.error(message)
        raise AssertionError(message)

    # Exact key match (case-insensitive)
    exactKeyMatch = {
        path: val
        for path, val in matches["matched_paths"].items()
        if path.lower().endswith(f"['{varPar.lower()}']")
    }

    # Check for duplicates
    if len(exactKeyMatch) != 1:
        message = (
            "Parameter '%s' does not uniquely match a single configuration parameter."
            % varPar
        )
        log.error(message)
        raise AssertionError(message)

    return True


def fetchParameterSection(cfg, parameter):
    """Fetch the section name that contains the specified parameter in a configuration file.

    Parameters
    ----------
    cfg : configparser object
        Configuration settings
    parameter : str
        Name of the parameter to find

    Returns
    -------
    str
        Name of the section containing the parameter or None if the parameter is not found.
    """

    for section in cfg.sections():
        if parameter in cfg[section]:
            return section

    return None


def checkForNumberOfReferenceValues(cfgGen, varPar):
    """check if in reference configuration no variation option of varPar is set
    if set - throw error

    Parameters
    -----------
    cfgGen: configparser object
        reference configuration settings
    varPar: str
        name of parameter to be checked

    """

    thPV = varPar + "PercentVariation"
    thRV = varPar + "RangeVariation"
    thDV = varPar + "DistVariation"
    thRCiV = varPar + "RangeFromCiVariation"

    # check if variation is set
    if (
        cfgGen[thPV] != ""
        or cfgGen[thRV] != ""
        or cfgGen[thDV] != ""
        or cfgGen[thRCiV] != ""
    ):
        message = (
            "Only one reference value is allowed for %s: but %s %s, %s %s, %s %s, %s %s is given"
            % (
                varPar,
                thPV,
                cfgGen[thPV],
                thRV,
                cfgGen[thRV],
                thDV,
                cfgGen[thDV],
                thRCiV,
                cfgGen[thRCiV],
            )
        )
        log.error(message)
        raise AssertionError(message)

    return True


def probAnalysis(
    avaDir, cfg, modName, parametersDict="", inputDir="", probConf="", simDFActual=""
):
    """Compute probability map of a given set of simulation result exceeding a particular threshold and save to outDir

    Parameters
    ----------
    avaDir: str
        path to avalanche directory
    cfg : dict
        configuration read from ini file of probAna function
    modName
        name of computational module that was used to run the simulations - to locate results files and filtering options
    parametersDict: dict
        dictionary with simulation parameters to filter simulations - only available if modName=com1DFA
    inputDir : str
        optional - path to directory where data that should be analysed can be found in
        a subfolder called peakFiles and configurationFiles, required if not in module results
    probConf : str
        name of probability configuration
    simDFActual: pandas dataFrame
        dataframe of simulation configurations that shall be used for prob analysis

    """

    avaDir = pathlib.Path(avaDir)

    # set output directory
    outDir = avaDir / "Outputs" / "ana4Stats"
    fU.makeADir(outDir)

    # fetch all result files and filter simulations according to parametersDict
    if modName.lower() in ["com1dfa", "com5snowslide", "com6rockavalanche"]:
        simNameList = cfgHandling.filterSims(
            avaDir, parametersDict, specDir=inputDir, simDF=simDFActual
        )
        filtering = True
    else:
        simNameList = []
        filtering = False
        log.info("No filtering available for this comMod: %s" % modName)

    # initialize flag if analysis has been performed or e.g. no matching files found
    analysisPerformed = False
    if simNameList == [] and filtering:
        # no matching sims found for filtering criteria
        log.warning("No matching simulations found for filtering criteria")
        return analysisPerformed

    # if matching sims found - perform analysis
    if inputDir == "":
        inputDir = avaDir / "Outputs" / modName / "peakFiles"
        peakFilesDF = fU.makeSimDF(inputDir, avaDir=avaDir)
    else:
        inputDirPF = inputDir / "peakFiles"
        peakFilesDF = fU.makeSimDF(inputDirPF, avaDir=avaDir)

    if len(peakFilesDF) == 0:
        message = "No peak files found in %s" % str(inputDir)
        log.error(message)

        raise FileNotFoundError(message)

    # get header info from peak files - this should be the same for all peakFiles
    header = IOf.readRasterHeader(peakFilesDF["files"][0])
    refData = IOf.readRaster(peakFilesDF["files"][0])
    nRows = header["nrows"]
    nCols = header["ncols"]

    # Initialise array for computations
    probSum = np.zeros((nRows, nCols))
    count = 0
    contourDict = {}

    # Loop through peakFiles and compute probability
    for m in range(len(peakFilesDF["names"])):
        # only take simulations that match filter criteria from parametersDict
        if (peakFilesDF["simName"][m] in simNameList) or filtering == False:
            # Load peak field for desired peak field parameter
            if peakFilesDF["resType"][m] == cfg["GENERAL"]["peakVar"]:
                # Load data
                fileName = peakFilesDF["files"][m]
                dataLim = np.zeros((nRows, nCols))
                fileData = IOf.readRaster(fileName)

                # check if extent is the same as first loaded dataset
                # if not - remesh and print warning
                if (
                    fileData["header"]["nrows"] != nRows
                    or fileData["header"]["ncols"] != nCols
                ):
                    log.warning(
                        "datasets used to create probMap do not match in extent - remeshing: %s to cellSize %s"
                        % (fileName, header["cellsize"])
                    )
                    dataRead, _ = gT.resizeData(fileData, refData)
                else:
                    dataRead = fileData["rasterData"]
                data = np.flipud(dataRead)

                # fetch contourline info
                xGrid, yGrid, _, _ = gT.makeCoordGridFromHeader(refData["header"])
                contourDictXY = pU.fetchContourCoords(
                    xGrid,
                    yGrid,
                    fileData["rasterData"],
                    float(cfg["GENERAL"]["peakLim"]),
                )
                contourDict[fileName.stem] = contourDictXY

                log.info(
                    "File Name: %s , simulation parameter %s "
                    % (fileName, cfg["GENERAL"]["peakVar"])
                )

                # Check if peak values exceed desired threshold
                dataLim[data > float(cfg["GENERAL"]["peakLim"])] = 1.0
                probSum = probSum + dataLim
                count = count + 1

    # Create probability map ranging from 0-1
    probMap = probSum / count
    unit = pU.cfgPlotUtils["unit%s" % cfg["GENERAL"]["peakVar"]]
    log.info(
        "probability analysis performed for peak parameter: %s and a peak value "
        "threshold of: %s %s"
        % (cfg["GENERAL"]["peakVar"], cfg["GENERAL"]["peakLim"], unit)
    )
    log.info("%s peak fields added to analysis" % count)

    # Save to raster file
    avaName = avaDir.name
    outFileName = "%s_prob_%s_%s_lim%s" % (
        avaName,
        probConf,
        cfg["GENERAL"]["peakVar"],
        cfg["GENERAL"]["peakLim"],
    )
    outFile = outDir / outFileName
    IOf.writeResultToRaster(header, probMap, outFile)
    log.info("Prob result written to %s" % outFile)
    analysisPerformed = True

    return analysisPerformed, contourDict


def makeDictFromVars(cfg):
    """create a dictionary with info on parameter variation for all parameter in
    varParList

    Parameters
    -----------
    cfg: configparser object
        configuration settings, here varParList, variationValue, numberOfSteps

    Returns
    --------
    variationsDict: dict
        dictionary with for each varName, varVariation, varSteps, and type of variation

    """

    varParList = cfg["varParList"].split("|")
    varValues = cfg["variationValue"].split("|")
    varSteps = cfg["numberOfSteps"].split("|")
    varTypes = cfg["variationType"].split("|")

    # check if value is provided for each parameter
    if cfg.getint("samplingStrategy") == 1:
        lengthsPar = "varParType"
    elif cfg.getint("samplingStrategy") == 2:
        lengthsPar = "numberOfSteps"
    else:
        message = "Chosen sampling strategy not valid: options are 1 or 2"
        log.error(message)
        raise AssertionError(message)

    if (
        len(varParList)
        == len(varValues)
        == len(cfg[lengthsPar].split("|"))
        == len(varTypes)
    ) is False:
        message = (
            "For every parameter in varParList a variationValue, %s and variationType needs to be provided"
            % lengthsPar
        )
        log.error(message)
        raise AssertionError(message)

    # check if correct values provided for rangefromci
    rangeFromCi = [idx for idx, v in enumerate(varTypes) if v.lower() == "rangefromci"]
    varValuesRCi = np.asarray(varValues)[rangeFromCi]
    ciCheck = [False for ci in varValuesRCi if ci != "ci95"]

    if len(ciCheck) > 0:
        message = "If rangefromci is chosen as variation type, ci95 is required as variationValue"
        log.error(message)
        raise AssertionError(message)

    variationsDict = {}
    if cfg.getint("samplingStrategy") == 2:
        for idx, val in enumerate(varParList):
            variationsDict[val] = {
                "variationValue": varValues[idx],
                "numberOfSteps": varSteps[idx],
                "variationType": varTypes[idx],
            }

    return variationsDict


def fetchThicknessInfo(avaDir):
    """Fetch input data for avaDir and thickness info

    Parameters
    ------------
    avaDir: pathlib path or str
        path to avalanche directory

    Returns
    -----------
    inputSimFilesAll: dict
        dictionary with info on available input data (release areas, entrainment, and thickness info)
    """

    # fetch input data - dem, release-, entrainment- and resistance areas (and secondary release areas)
    inputSimFilesAll = gI.getInputDataCom1DFA(avaDir)

    # get thickness of release and entrainment areas (and secondary release areas) -if thFromShp = True
    inputSimFilesAll = gI.getThicknessInputSimFiles(inputSimFilesAll)

    return inputSimFilesAll


def createSampleFromConfig(avaDir, cfgProb, comMod):
    """Create a sample of parameters for a desired parameter variation,
    and draw nSample sets of parameter values
    if thickness values read from shp for comMod, convert sample values for these


    Parameters
    ------------
    avaDir: pathlib path
        path to avalanche directory
    cfgProb: configparser object
        configuration settings for parameter variation
    comMod: computational module
        module to perform then sims for parameter variation

    Returns
    --------
    paramValuesDList: list
        list of paramValuesD (multiple if multiple release area scenarios)

        - names: list, list of parameter names (that are varied)
        - values: numpy nd array, as many rows as sets of parameter values and as many rows as parameters
        - typeList: list, list of types of parameters (float, ...)
        - thFromIni: str, str of parameter names where the base value is read from shape

    """

    # read initial configuration
    cfgStart = fetchStartCfg(comMod, cfgProb)

    # fetch parameter names for parameter variation and variation value and variation type
    varParList = cfgProb["PROBRUN"]["varParList"].split("|")
    valVariationValue = cfgProb["PROBRUN"]["variationValue"].split("|")
    varType = cfgProb["PROBRUN"]["variationType"].split("|")
    # check if thickness parameters are actually read from shp file
    _, thReadFromShp = checkParameterSettings(cfgStart, varParList)

    modNameString = str(pathlib.Path(comMod.__file__).stem)
    if modNameString.lower() in [
        "com1dfa",
        "com5snowslide",
        "com6rockavalanche",
        "com8motpsa",
    ]:
        # check if thickness parameters are actually read from shp file
        _, thReadFromShp = checkParameterSettings(cfgStart, varParList)
    else:
        thReadFromShp = []

    # create sets of parameters values for parameter variation
    if len(thReadFromShp) > 0:
        paramValuesDList = createSampleWithVariationForThParameters(
            avaDir,
            cfgProb,
            cfgStart,
            varParList,
            valVariationValue,
            varType,
            thReadFromShp,
        )
    else:
        paramValuesD = createSampleWithVariationStandardParameters(
            cfgProb, cfgStart, varParList, valVariationValue, varType
        )
        paramValuesDList = [paramValuesD]

    # save dictionary to pickle file
    outDir = pathlib.Path(avaDir, "Outputs", "ana4Stats")
    fU.makeADir(outDir)
    with open(outDir / "paramValuesD.pickle", "wb") as fi:
        pickle.dump(paramValuesDList[0], fi)

    return paramValuesDList


def createSampleWithVariationStandardParameters(
    cfgProb, cfgStart, varParList, valVariationValue, varType
):
    """create a sample for a parameter variation using latin hypercube sampling

    Parameters
    ------------
    cfgProb: configparser object
        configuration settings for parameter variation
    cfgStart: configparser object
        configuration settings for comMod without variation values
    varParList: list
        list of parameters that shall be varied
    valVariationValue: list
        list if value used for variation
    varType: list
        list of type of variation for each parameter (percent, range, rangefromci)

    Returns
    --------
    paramValuesD: dict
        dictionary used to pass parameter variation values

        - names: list, list of parameter names (that are varied)
        - values: numpy nd array, as many rows as sets of parameter values and as many rows as parameters
        - typeList: list, list of types of parameters (float, ...)
        - thFromIni: str, str of parameter names where the base value is read from shape

    """

    # initialze lower and upper bounds required to get a sample for the parameter values
    lowerBounds = []
    upperBounds = []
    for idx, varPar in enumerate(varParList):
        section = fetchParameterSection(cfgStart, varPar)
        varVal = cfgStart[section].getfloat(varPar)
        if varType[idx].lower() == "percent":
            lB = varVal - varVal * (float(valVariationValue[idx]) / 100.0)
            uB = varVal + varVal * (float(valVariationValue[idx]) / 100.0)
        elif varType[idx].lower() == "range":
            lB = varVal - float(valVariationValue[idx])
            uB = varVal + float(valVariationValue[idx])
        else:
            message = "Variation method: %s not a valid option" % varType[idx]
            log.error(message)
            raise AssertionError(message)
        # update bounds
        lowerBounds.append(lB)
        upperBounds.append(uB)

    # create a sample of parameter values using scipy latin hypercube sampling
    sample = createSample(cfgProb, varParList)
    sampleWBounds = qmc.scale(sample, lowerBounds, upperBounds)

    # create dictionary with all the info
    paramValuesD = {
        "names": varParList,
        "values": sampleWBounds,
        "typeList": cfgProb["PROBRUN"]["varParType"].split("|"),
        "thFromIni": "",
    }

    return paramValuesD


def createSampleWithVariationForThParameters(
    avaDir, cfgProb, cfgStart, varParList, valVariationValue, varType, thReadFromShp
):
    """Create a sample of parameters for a desired parameter variation,
    and fetch thickness values from shp file and perform variation for each feature within
    shapefile but treating the features of one shapefile as not-independent

    paramsValuesD dict in output list contains



    Parameters
    ------------
    cfgProb: configparser object
        configuration settings for parameter variation
    cfgStart: configparser object
        configuration settings for comMod without variation values
    varParList: list
        list of parameters that shall be varied
    valVariationValue: list
        list if value used for variation
    varType: list
        list of type of variation for each parameter (percent, range, rangefromci)

    Returns
    --------
    paramValuesDList: list
        list of paramValuesD (multiple if multiple release area scenarios)

        - names: list, list of parameter names (that are varied)
        - values: numpy nd array, as many rows as sets of parameter values and as many rows as parameters
        - typeList: list, list of types of parameters (float, ...)
        - thFromIni: str, str of parameter names where the base value is read from shape

    """

    # fetch input files and corresponding thickness info
    inputSimFiles = fetchThicknessInfo(avaDir)
    fileTypeText = {".asc": "raster", ".tif": "raster", ".shp": "shapefile"}
    for thFType in thReadFromShp:
        if inputSimFiles["entResInfo"][thFType + "FileType"] != ".shp":
            message = "%s file type: %s not a valid option for desired %s variation" % (
                thFType,
                fileTypeText[inputSimFiles["entResInfo"][thFType + "FileType"]],
                thFType,
            )
            log.error(message)
            raise AssertionError(message)

    paramValuesDList = []
    for iRel, relF in enumerate(inputSimFiles["relFiles"]):
        paramValuesD = {}
        # create lower and upper bounds for all thickness parameters - taking into account all features
        fullListOfParameters = []
        parentParameterId = []
        staParameter = []
        thValues = np.asarray([])
        ciValues = np.asarray([])
        for idx1, varPar in enumerate(varParList):
            if varPar in thReadFromShp:
                ciRequired = varType[idx1].lower() == "rangefromci"
                thV, ciV, thFeatureNames = fetchThThicknessLists(
                    varPar, inputSimFiles, relF, ciRequired=ciRequired
                )
                # add to list all the parameter names
                fullListOfParameters = fullListOfParameters + thFeatureNames
                parentParameterId = parentParameterId + [
                    varParList.index(varPar)
                ] * len(thFeatureNames)
                thValues = np.append(thValues, thV)
                ciValues = np.append(ciValues, ciV)
            else:
                parentParameterId.append(varParList.index(varPar))
                fullListOfParameters.append(varPar)
                staParameter.append(varPar)
                thValues = np.append(thValues, np.asarray([None]))
                ciValues = np.append(ciValues, np.asarray([None]))

        # initialize lower and upper bounds required to get a sample for the parameter values
        # numpy arrays required to do masking as lists don't work for a list indices
        varValList = np.asarray(
            [
                (
                    cfgStart[fetchParameterSection(cfgStart, varPar)].getfloat(varPar)
                    if varPar in staParameter
                    else thValues[idx]
                )
                for idx, varPar in enumerate(fullListOfParameters)
            ]
        )
        fullValVar = np.asarray(
            [
                float(valVariationValue[i])
                if valVariationValue[i] != "ci95"
                else np.nan
                for i in parentParameterId
            ]
        )
        fullVarType = np.asarray([varType[i].lower() for i in parentParameterId])
        lowerBounds = np.asarray([None] * len(fullListOfParameters))
        upperBounds = np.asarray([None] * len(fullListOfParameters))

        # set lower and upper bounds depending on varType (percent, range, rangefromci)
        lowerBounds[fullVarType == "percent"] = varValList[
            fullVarType == "percent"
        ] - varValList[fullVarType == "percent"] * (
            fullValVar[fullVarType == "percent"] / 100.0
        )
        upperBounds[fullVarType == "percent"] = varValList[
            fullVarType == "percent"
        ] + varValList[fullVarType == "percent"] * (
            fullValVar[fullVarType == "percent"] / 100.0
        )

        lowerBounds[fullVarType == "range"] = (
            varValList[fullVarType == "range"] - fullValVar[fullVarType == "range"]
        )
        upperBounds[fullVarType == "range"] = (
            varValList[fullVarType == "range"] + fullValVar[fullVarType == "range"]
        )

        lowerBounds[fullVarType == "rangefromci"] = (
            varValList[fullVarType == "rangefromci"]
            - ciValues[fullVarType == "rangefromci"]
        )
        upperBounds[fullVarType == "rangefromci"] = (
            varValList[fullVarType == "rangefromci"]
            + ciValues[fullVarType == "rangefromci"]
        )

        # create a sample of parameter values using scipy latin hypercube or morris sampling
        sample = createSample(cfgProb, varParList)

        # create a full sample including those thickness values for the potentially multiple features
        # however, the thickness values for one parameter (relTh or entTh or secondaryRelTh) should not
        # be independent for the different features within one parameter
        if cfgProb["PROBRUN"]["sampleMethod"] == "morris":
            fullSample = np.zeros(
                (
                    int(cfgProb["PROBRUN"]["nSample"]) * (len(varParList) + 1),
                    len(fullListOfParameters),
                )
            )
        else:
            fullSample = np.zeros(
                (int(cfgProb["PROBRUN"]["nSample"]), len(fullListOfParameters))
            )

        for idx, varPar in enumerate(fullListOfParameters):
            lB = [0] * len(varParList)
            uB = [1] * len(varParList)
            lB[parentParameterId[idx]] = lowerBounds[idx]
            uB[parentParameterId[idx]] = upperBounds[idx]
            parSample = qmc.scale(sample, lB, uB)
            fullSample[:, idx] = parSample[:, parentParameterId[idx]]

        # create dictionary with all the info
        thFromIni = cfgUtils.convertToCfgList(
            list(set(varParList).symmetric_difference(set(staParameter)))
        )
        paramValuesD = {
            "names": fullListOfParameters,
            "values": fullSample,
            "typeList": cfgProb["PROBRUN"]["varParType"].split("|"),
            "thFromIni": thFromIni,
            "thVariationBasedOnFromShp": thReadFromShp,
            "varParNamesInitial": varParList,
            "releaseScenario": relF.stem,
        }

        paramValuesDList.append(paramValuesD)

    return paramValuesDList


def createSample(cfgProb, varParList):
    """create a sample of parameters

    Parameters
    -----------
    cfgProb: configparser object
        configuration settings
    varParList: list
        list of parameters used for creating a sample

    Returns
    --------
    sample: scipy object
        sample object of given dimension that can be adjusted to desired bounds
    """

    # random generator initialized with seed
    sampleSeed = cfgProb["PROBRUN"].getint("sampleSeed")
    randomGen = np.random.default_rng(sampleSeed)
    nTrajectories = cfgProb["PROBRUN"].getint("nSample")

    # create a sample of parameter values using salib morris sampling
    if cfgProb["PROBRUN"]["sampleMethod"].lower() == "morris":
        param_ranges = {
            "num_vars": len(varParList),
            "names": varParList,
            "bounds": [[0, 1]] * len(varParList),
        }
        sample = morris.sample(
            param_ranges,
            N=nTrajectories,  # number of trajectories
            num_levels=6,  # how many discrete values per parameter
            seed=sampleSeed,
        )

    # create a sample of parameter values using scipy latin hypercube sampling
    elif cfgProb["PROBRUN"]["sampleMethod"].lower() == "latin":
        sampler = qmc.LatinHypercube(d=len(varParList), seed=randomGen)
        sample = sampler.random(n=int(cfgProb["PROBRUN"]["nSample"]))
        log.info("Parameter sample created using latin hypercube sampling")
    else:
        message = (
            "Sampling method: %s not a valid option"
            % cfgProb["PROBRUN"]["sampleMethod"]
        )
        log.error(message)
        raise AssertionError(message)

    return sample


def fetchThThicknessLists(varPar, inputSimFiles, releaseFile, ciRequired=False):
    """fetch the desired thickness shp file info on thickness, id and ci values
    of all available features in shp file

    Parameters
    -----------
    varPar: str
        name of thickness parameter
    inputSimFiles: dict
        dictionary with info in input data
    ciRequired: bool
        if True throw error if ci Values not provided

    Returns
    --------
    thicknessFeatureNames: list
        list of names of thickness features
    thValues: list
        list of thickness values for all features
    ciValues: list
        list of ci values for all feature
    """

    if varPar == "relTh":
        thFile = [
            inputSimFiles["relFiles"][idx]
            for idx, relF in enumerate(inputSimFiles["relFiles"])
            if relF == releaseFile
        ][0]
    elif varPar == "entTh":
        thFile = inputSimFiles["entFile"]
    elif varPar == "secondaryRelTh":
        thFile = inputSimFiles["secondaryReleaseFile"]

    infoDict = inputSimFiles[thFile.stem]
    thicknessFeatureNames = [varPar + str(id) for id in infoDict["id"]]
    thValues = [float(th) for th in infoDict["thickness"]]
    ciValues = [float(ci) if ci != "None" else np.nan for ci in infoDict["ci95"]]

    if np.nan in ciValues and ciRequired:
        msg = "ci95 values required in shape file but not provided for %s" % varPar
        log.error(msg)
        raise AssertionError(msg)

    return thValues, ciValues, thicknessFeatureNames


def createCfgFiles(paramValuesDList, comMod, cfg, cfgPath=""):
    """create all config files required to run com Module from parameter variations using paramValues

    Parameters
    -----------
    paramValuesDList: list
        list of dictionaries with parameter names and values (array of all sets of parameter values,
         one row per value set)
        multiple dictionaries if multiple release area scenarios and thFromShp
    comMod: com module
        computational module
    cfg: configparser object
        configuration settings
    cfgPath: str
        path where cfg files should be saved to

    Returns
    --------
    cfgFiles: list
        list of cfg file paths for comMod including the updated values of the parameters to vary

    """

    # get filename of module
    modName = str(pathlib.Path(comMod.__file__).stem)

    # create one cfgFile with one line of the parameter values from the full parameter variation
    cfgFiles = []
    countS = 0
    for paramValuesD in paramValuesDList:
        # read initial configuration
        cfgStart = fetchStartCfg(comMod, cfg)
        for count1, pVal in enumerate(paramValuesD["values"]):
            for index, par in enumerate(paramValuesD["names"]):
                section = fetchParameterSection(cfgStart, par)
                # If parameter not found in any section, add it to 'GENERAL'.
                if section is not None:
                    cfgStart[section][par] = str(pVal[index])
                else:
                    cfgStart["GENERAL"][par] = str(pVal[index])
            if modName.lower() in ["com1dfa", "com5snowslide", "com6rockavalanche"]:
                cfgStart["VISUALISATION"]["scenario"] = str(count1)
                cfgStart["INPUT"]["thFromIni"] = paramValuesD["thFromIni"]
                if "releaseScenario" in paramValuesD.keys():
                    cfgStart["INPUT"]["releaseScenario"] = paramValuesD[
                        "releaseScenario"
                    ]
            cfgF = pathlib.Path(cfgPath, ("%d_%sCfg.ini" % (countS, modName)))
            with open(cfgF, "w") as configfile:
                cfgStart.write(configfile)
            # append file path to list of cfg files
            cfgFiles.append(cfgF)
            countS = countS + 1

    return cfgFiles


def fetchStartCfg(comMod, cfgProb):
    """fetch start configuration of comMod
    if onlyDefault use default comModCfg.ini and if false check if there is a local_comModCfg.ini

    Parameters
    -----------
    comMod: computational module
        module where configuration is read from
    cfgProb: configparser object
        configuration settings of probAna with collection_comMod_override section

    Returns
    --------
    cfgStart: configparser object
        configuration object of comMod
    """
    # get filename of module
    modName = str(pathlib.Path(comMod.__file__).stem)
    modP = (pathlib.Path(comMod.__file__).resolve().parent).stem

    # fetch comMod config
    cfgStart = cfgUtils.getModuleConfig(
        comMod,
        fileOverride="",
        toPrint=False,
        onlyDefault=cfgProb["%s_%s_override" % (modP, modName)].getboolean(
            "defaultConfig"
        ),
    )

    # override with parameters set in in the cfgProb comMod_override section
    cfgStart, cfgProb = cfgHandling.applyCfgOverride(
        cfgStart, cfgProb, comMod, addModValues=False
    )

    return cfgStart


def fetchProbConfigs(cfg):
    """fetch configurations of prob run in order to filter simulations
    e.g. to create probability maps for different scenarios

    Parameters
    -----------
    cfg: configparser object
        configuration setting, here used: samplingStrategy, varParList

    Returns
    --------
    probConfigs: dict
        dictionary with one key per config and a dict per key with parameter and value
    """

    probConfigs = {"includeAll": {}}

    if cfg.getint("samplingStrategy") == 2:
        for par in cfg["varParList"].split("|"):
            probConfigs["include" + par] = {"scenario": par}
        log.info(
            "Probability maps are created for full parameter variation and for %s separately"
            % cfg["varParList"]
        )
    else:
        log.info("Probability map is created for full parameter variation")

    return probConfigs
