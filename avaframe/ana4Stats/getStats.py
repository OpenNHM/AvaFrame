"""

Function to determine statistics of datasets

"""

import numpy as np
import logging
import pathlib
from matplotlib import pyplot as plt
import pandas as pd

from avaframe.in3Utils import fileHandlerUtils as fU
import avaframe.in2Trans.rasterUtils as IOf
from avaframe.in3Utils import cfgUtils
from avaframe.in3Utils import cfgHandling


# create local logger
# change log level in calling module to DEBUG to see log messages
log = logging.getLogger(__name__)


def extractMaxValues(
    inputDir,
    avaDir,
    varPar,
    restrictType="",
    nameScenario="",
    parametersDict="",
    layer="",
    modName="com1DFA",
):
    """ Extract max values of result parameters and save to dictionary

        - optionally restrict data of peak fields by defining which result parameter with restrictType,
          provide nameScenario and a parametersDict to filter simulations

        Parameters
        -----------
        inputDir: str
            path to directoy where peak files can be found
        avaDir: str
            path to avalanche directoy
        varPar: str
            parameter that has been varied when performing simulations (for example relTh)
        restrictType: str
            optional -result type of result parameters that should be used to mask result fields (eg. ppr, pft, ..)
        nameScenario: str
            optional -parameter that shall be used for color coding of simulation results
            in plots (for example releaseScenario)
        parametersDict: dict
            optional -dictionary with parameter and parameter values to filter simulations

        Returns
        --------
        peakValues: dict
            dictionary that contains max values for all result parameters for
            each simulation
        """

    # filter simulation results using parametersDict
    simNameList = cfgHandling.filterSims(avaDir, parametersDict, modName=modName)
    # load dataFrame of all simulation configurations
    simDF = cfgUtils.createConfigurationInfo(avaDir, standardCfg="", comModule=modName)

    # load peakFiles of all simulations and generate dataframe
    peakFilesDF = fU.makeSimDF(inputDir, avaDir=avaDir)
    hasMultiLayer = any(peakFilesDF["layer"] != "")
    if hasMultiLayer and layer == "":
        message = (
            "Multi-layer results detected but no layer specified. "
            "Set 'layer' in getStatsCfg.ini (e.g. layer = L1)"
        )
        log.error(message)
        raise ValueError(message)
    elif not hasMultiLayer and layer != "":
        message = "No multi-layer results detected but layer is specified. "
        log.error(message)
        raise ValueError(message)

    nSims = len(peakFilesDF['simName'])
    # initialize peakValues dictionary
    peakValues = {}
    for sName in simDF['simName'].tolist():
        peakValues[sName] = {}

    # Loop through result field files and compute statistical measures
    for m in range(nSims):
        simName = peakFilesDF['simName'][m]
        # skip simulations not in the filtered list
        if simName not in simNameList:
            peakValues.pop(simName, None)
            continue

        # Load data
        fileName = peakFilesDF['files'][m]
        dataFull = IOf.readRaster(fileName, noDataToNan=True)

        # if restrictType, set result field values to nan if restrictType result field equals 0
        if restrictType != "":
            peakFilesSimName = peakFilesDF[peakFilesDF['simName'] == simName]
            # mask for resultType
            mask = peakFilesSimName["resType"] == restrictType
            # mask for layer
            if layer:
                mask &= peakFilesSimName["layer"] == layer
            fileNameRestrict1 = peakFilesSimName[mask]
            if len(fileNameRestrict1) == 0:
                message = "No %s file found for %s to restrict current result file %s with." % (
                    restrictType,
                    simName,
                    fileName.name,
                )
                log.error(message)
                raise ValueError(message)
            fileNameRestrict = fileNameRestrict1["files"].to_list()[0]
            dataRestrict = IOf.readRaster(fileNameRestrict, noDataToNan=True)
            log.info("%s file restricted using %s." % (fileName.name, fileNameRestrict.name))
            data = np.where((dataRestrict['rasterData'] == 0.0), np.nan, dataFull['rasterData'])
        else:
            data = dataFull['rasterData']

        # compute max, mean, min and standard deviation of result field
        max = np.nanmax(data)
        min = np.nanmin(data)
        mean = np.nanmean(data)
        std = np.nanstd(data)
        statVals = {'max': max, 'min': min, 'mean': mean, 'std': std}
        # use layer-suffixed key for multi-layer results (e.g. ppr_l1)
        resType = peakFilesDF['resType'][m]
        if peakFilesDF['layer'][m]:
            resType = "%s_%s" % (resType, peakFilesDF['layer'][m].lower())
        peakValues[simName].update({resType: statVals})

        # fetch varPar value and nameScenario
        simRow = simDF[simDF['simName'] == simName]
        varParVal = simRow[varPar].iloc[0]

        # if varParVal is empty or nan check if thickness value
        # if so - read from shp thickness value for each feature need to be found
        if (varParVal == '' or pd.isna(varParVal)) and varPar in ['relTh', 'entTh', 'secondaryRelTh']:
            # fetch id of thickness features
            varParId = varPar + 'Id'
            varParIds = str(simRow[varParId].iloc[0])
            varParIdList = varParIds.split('|')
            # create feature thickness parameter names
            varParFirstName = varPar + varParIdList[0]
            # chose value from first feature found!
            varParVal = simRow[varParFirstName].iloc[0]
            if len(varParIdList) > 1:
                log.warning('%s value - there are multiple features - %s value used '
                            'from first feature only!' % (varPar, varPar))
        # add varPar value to dictionary
        peakValues[simName].update({'varPar': varParVal})
        if nameScenario != '':
            nameScenarioVal = simRow[nameScenario]
            peakValues[simName].update({'scenario': nameScenarioVal.iloc[0]})
            log.info('Simulation parameter %s= %s for resType: %s and name %s' %
                    (varPar, str(varParVal), peakFilesDF['resType'][m], nameScenarioVal.iloc[0]))
        else:
            log.info('Simulation parameter %s= %s for resType: %s' %
                    (varPar, str(varParVal), peakFilesDF['resType'][m]))

    return peakValues, hasMultiLayer
