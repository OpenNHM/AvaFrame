"""
    Helper function to export required data from com1DFA to be used by Aimec.
"""

# Load modules
import logging
import pathlib
import pandas as pd

# local modules
from avaframe.in3Utils import fileHandlerUtils as fU

# create local logger
# change log level in calling module to DEBUG to see log messages
log = logging.getLogger(__name__)


def getMBInfo(avaDir, inputsDF, comMod, simName=''):
    """ Get mass balance info for com1DFA module

    The mass info should be located in the mass_XXX.txt file (XXX being the simulation name)
    The file should contain 3 columns (time, current, entrained) and the coresponding values for each time step

    Parameters
    -----------
    avaDir: str
        path to avalanche directory
    inputsDF: dataFrame
        simulation dataframe
    comMod: str
        computational module name that has been used to produce simulation results
    simName: str
        part or full name of the simulation name (if '', look at all sims)

    Returns
    --------
    inputsDF: dataFrame
        simulation dataframe with mass path information
    """
    # Get info from mass log file
    if simName != '':
        mbFile = pathlib.Path(avaDir, 'Outputs', comMod, 'mass_%s.txt' % simName)
        fU.checkIfFileExists(mbFile, fileType='mass')
        simRowHash = inputsDF[inputsDF['simName'] == simName].index[0]
        inputsDF.loc[simRowHash, 'massBal'] = mbFile
        log.debug('Added to inputsDF[massBal] %s' % (mbFile))

    else:
        dir = pathlib.Path(avaDir, 'Outputs', comMod)
        mbFiles = list(dir.glob('mass*.txt'))
        if len(mbFiles) == 0:
            message = 'No mass log file found in directory %s' % (str(dir))
            log.error(message)
            raise FileNotFoundError(message)
        mbNames = sorted(set(mbFiles), key=lambda s: (str(s).split("_")[1], str(s).split("_")[2], str(s).split("_")[4]))

        for mFile in mbNames:
            name = mFile.stem
            nameParts = name.split('_')
            simName = ('_'.join(nameParts[1:]))
            simRowHash = inputsDF[inputsDF['simName'] == simName].index[0]
            inputsDF.loc[simRowHash, 'massBal'] = mFile
            log.debug('Added to inputsDF[massBal] %s' % (mFile))
    return inputsDF


def getRefMB(testName, inputsDF, simName):
    """ Get mass balance info for benchmark simulation

    Parameters
    -----------
    testName: str
        benchmark test name
    inputsDF: dataFrame
        simulation dataframe
    simName: str
        full name of the simulation

    Returns
    --------
    inputsDF: dataFrame
        simulation dataframe with mass path information
    """
    # Get info from mass log file
    mbFile = pathlib.Path('..', 'benchmarks', testName, 'mass_%s.txt' % simName)
    simRowHash = inputsDF[inputsDF['simName'] == simName].index[0]
    inputsDF.loc[simRowHash, 'massBal'] = mbFile
    log.debug('Added to inputsDF[massBal] %s' % (mbFile))

    return inputsDF


def dfaBench2Aimec(avaDir, cfg, simNameRef='', simNameComp=''):
    """ Exports the required data from the computational modules to be used by Aimec

        Parameters
        -----------
        avaDir: str
            path to avalanche directory
        cfg: confiParser object
            configuration settings for aimec
        simNameRef: str
            name of reference simulation results (or part of the name)
        simNameComp: str
            name of comparison simulation results (or part of the name)

        Returns
        --------
        inputsDF: dataFrame
            simulation dataframe with optionally path to mass file
        pathDict: dict
            dictionary with paths to simulation results

    """

    cfgSetup = cfg['AIMECSETUP']
    comModules = cfgSetup['comModules'].split('|')
    # get directories where simulation results can be found for both modules
    inputDirRef, inputDirComp, pathDict = getCompDirs(avaDir, cfgSetup)

    # Load all infos on reference simulations
    refData, resTypeRefList = fU.makeSimFromResDF(avaDir, None, inputDir=inputDirRef, simName=simNameRef)
    if len(refData.index) == 0:
        message = ('Found no simulation matching the reference criterion %s, '
                   'there should be one' % simNameRef)
        log.error(message)
        raise ValueError(message)
    elif len(refData.index) > 1:
        message = ('Found multiple simulations matching the reference criterion %s,'
                   'there should be only one reference' % simNameRef)
        log.error(message)
        raise ValueError(message)
    else:
        colorParameter = False
        refSimRowHash = refData.index[0]
        refSimName = refData.loc[refSimRowHash, 'simName']
    # all went fine through the fetchReferenceSimNo function meaning that we found the reference and it is unique
    # now we make sure we only keep this reference in the dataframe
    refData = refData.loc[refSimRowHash, :].to_frame().transpose()

    # Load all infos on comparison module simulations
    compData, resTypeCompList = fU.makeSimFromResDF(avaDir, None, inputDir=inputDirComp, simName=simNameComp)

    resTypeList = list(set(resTypeRefList).intersection(resTypeCompList))
    pathDict['resTypeList'] = resTypeList
    pathDict['colorParameter'] = colorParameter
    pathDict['refSimRowHash'] = refSimRowHash
    pathDict['refSimName'] = refSimName
    # set pathDict ValRef to empty as reference sim is not chosen on a value
    pathDict['valRef'] = ''
    # if desired set path to mass log files
    if cfg['FLAGS'].getboolean('flagMass'):
        refData = getMassInfoInDF(avaDir, refData, comModules[0], sim=refSimName, testName=cfgSetup['testName'])
    # at least one simulation is needed in the comparison dataFrame
    if len(compData.index) == 0:
        message = ('Did not find the comparison simulation in %s with name %s'
                   % (inputDirRef, simNameRef))
        log.error(message)
        raise ValueError(message)
    # all simulations should have a different name in the comparison dataframe
    compDataCount = (compData['simName'].value_counts() > 1).to_list()
    # this should never actually happen
    if True in compDataCount:
        message = ('Multiple rows of the comparison dataFrame have the same simulation name. This is not allowed')
        log.error(message)
        raise ValueError(message)
    # if desired set path to mass log files
    if cfg['FLAGS'].getboolean('flagMass'):
        compData = getMassInfoInDF(avaDir, compData, comModules[1], sim='', testName=cfgSetup['testName'])
    # build input dataFrame
    inputsDF = pd.concat([refData, compData], ignore_index=False)
    # all simulations should have a different name in the comparison dataframe
    inputsDFCount = (inputsDF['simName'].value_counts() > 1).to_list()
    # this can happen, the difference between simualtion is based on the row hash so this is no problem
    if True in inputsDFCount:
        message = ('Multiple rows of the dataFrame have the same simulation name.')
        log.warning(message)

    return inputsDF, pathDict


def getMassInfoInDF(avaDir, inputsDF, comMod, sim='', testName=''):
    """ Fetch mass information depending on the computational module
    So far, a com1DFA style mass file is expected, either located in the benchmark directory
    or in the Outputs/comMod one

    Parameters
    -----------
    avaDir: str
        path to avalanche directory
    inputsDF: dataFrame
        simulation dataframe
    comMod: str
        computational module name that has been used to produce simulation results
    simName: str
        part or full name of the simulation name (if '', look at all sims)
    testName: str
        if the mass is fetched from benchmarks, provide the name of the test

    Returns
    --------
    inputsDF: dataFrame
        simulation dataframe with path to mass file

"""
    if comMod == 'benchmarkReference':
        inputsDF = getRefMB(testName, inputsDF, sim)
    else:
        try:
            inputsDF = getMBInfo(avaDir, inputsDF, comMod, simName=sim)
        except FileNotFoundError as e:
            message = ('Make sure the mass files are available and that a method is implemented to analyze mass'
                       'for module %s' % (comMod))
            log.error(message)
            message = str(e) + '. ' + message
            raise FileNotFoundError(message)
    return inputsDF


def getCompDirs(avaDir, cfgSetup):
    """ Determine dictionaries where simulation results can be found

        Parameters
        ----------
        avaDir: str
            path to avalanche directory
        cfgSetup: configParser object
            configuration settings for aimec

        Returns
        --------
        inputDirRef: str
            path to reference simulation results
        inputDirComp: str
            path to comparison simulation results
        pathDict: dict
            dictionary with paths for aimec
        refModule: str
            name of reference module
    """

    # look for matching simulations
    comModules = cfgSetup['comModules'].split('|')
    refModule = comModules[0]
    compModule = comModules[1]
    log.info('Reference data is from module: %s' % refModule)
    log.info('Comparison data is from module: %s' % compModule)

    # Lead all infos on refernce simulations
    if refModule == 'benchmarkReference':
        testName = cfgSetup['testName']
        inputDirRef = pathlib.Path('..', 'benchmarks', testName)
    else:
        inputDirRef = pathlib.Path(avaDir, 'Outputs', refModule, 'peakFiles')

    inputDirComp = pathlib.Path(avaDir, 'Outputs', compModule, 'peakFiles')
    pathDict = {}
    pathDict['compType'] = ['comModules', refModule, compModule]
    # info about colourmap
    pathDict['contCmap'] = True

    return inputDirRef, inputDirComp, pathDict


def mainDfa2Aimec(avaDir, comModule, cfg, inputDir=''):
    """ Fetch available raster results path from comModule to be used by Aimec

        Parameters
        -----------
        avaDir: str
            path to avalanche directory
        comModule: str
            computational module name that has been used to produce simulation results
        cfg: configParser object
            configuration for aimec
        inputDir: str or pathlib path
            optional- directory where peak files are located, if '',
            avaDir/Outputs/comMod/peakFiles is set

        Returns
        --------
        inputsDF: dataFrame
            with a line for each simulation available and the corresponding simulation results: ppr, pft, pfv
    """

    # create data frame that lists all available simulations and path to their result type result files
    inputsDF, resTypeList, resTypeAndLayerTypeList, layerTypesList = fU.makeSimFromResDF(
        avaDir, comModule, inputDir=inputDir
    )

    # check if mass analysis shall be performed
    if cfg['FLAGS'].getboolean('flagMass'):
        # Add path to mb info file to dataframe
        inputsDF = getMassInfoInDF(avaDir, inputsDF, comModule, sim='', testName='')

    return inputsDF, resTypeList, resTypeAndLayerTypeList, layerTypesList
