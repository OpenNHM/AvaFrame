'''
    Functions to initialize project, i.e create folder structure

    This file is part of Avaframe.
'''

import os
import logging
import shutil
import pathlib

import avaframe.in3Utils.fileHandlerUtils as fU

log = logging.getLogger(__name__)


def _checkForFolderAndDelete(baseDir, folderName):
    '''Helper function ONLY USE WITH CHECKED INPUT'''
    fPath = os.path.join(baseDir, folderName)
    try:
        shutil.rmtree(fPath)
    except FileNotFoundError:
        log.debug("No %s folder found.", folderName)


def _checkAvaDirVariable(avaDir):
    '''Check for empty or nonString avaDir variable'''
    # check for empty avaDir name, abort if empty
    if not avaDir:
        log.warning("avaDir variable is undefined, returning. ")
        return 'avaDir is empty'

    # make sure avaDir is a string
    isString = isinstance(avaDir, str)
    isPurePath = isinstance(avaDir, pathlib.PurePath)
    if not ((isString) or (isPurePath)):
        log.warning("avaDir is not a string or a PurePath, returning")
        return 'avaDir is NOT a string or PurePath'

    return 'SUCCESS'


def cleanModuleFiles(avaDir, module, alternativeName='', deleteOutput=True):
    '''Cleans all generated files from the provided module in the outputs
    folder and work folder

    Parameters
    ----------
    avaDir : path/string
        Avalanche directory path
    module : module object
        The module do delete.
        e.g. from avaframe.com2AB import com2AB
        leads to cleanModuleFiles(com2AB)
        whereas
        from avaframe.com2AB import com2AB as c2
        leads to cleanModuleFiles(c2)
        Boolean to be able to avoid deletion of Outputs (true by default)
    '''

    # check for empty or non string variable
    result = _checkAvaDirVariable(avaDir)
    if 'SUCCESS' not in result:
        return result

    # get filename of module
    # TODO: remove this option when com1DFAPy replaces com1DFA
    if alternativeName:
        modName = alternativeName
    else:
        modName = os.path.basename(module.__file__)
        modName = os.path.splitext(modName)[0]

    # output dir
    outDir = os.path.join(avaDir, 'Outputs')
    workDir = os.path.join(avaDir, 'Work')

    if deleteOutput:
        log.info("Cleaning module %s in folder: %s ", modName, outDir)
        _checkForFolderAndDelete(outDir, modName)
    _checkForFolderAndDelete(workDir, modName)

    return 'SUCCESS'


def cleanSingleAvaDir(avaDir, deleteOutput=True):
    ''' Clean a single avalanche directory of the work and output directories

    Parameters
    ----------
    avaDir : path/string
        Avalanche directory path
    deleteOutput : boolean
        If True (default), directory output and the log files are deleted 
    '''

    # check for empty or non string variable
    result = _checkAvaDirVariable(avaDir)
    if 'SUCCESS' not in result:
        return result

    # Info to user
    log.debug("Cleaning folder: %s ", avaDir)

    # Try to remove OUTPUTS folder, only pass FileNotFoundError, i.e. folder
    # does not exist
    if deleteOutput:
        _checkForFolderAndDelete(avaDir, 'Outputs')

        # check for *.log files, go to remove only if exists
        dirToClean = pathlib.Path(avaDir)
        logFiles = list(dirToClean.glob('*.log'))

        for fi in logFiles:
            os.remove(fi)

    # Try to remove Work folder, only pass FileNotFoundError, i.e. folder
    # does not exist
    _checkForFolderAndDelete(avaDir, 'Work')

    log.debug("Directory is squeaky clean")
    return 'Cleaned directory'


def cleanRemeshedDir(avaDir):
    """ clean remeshedRasters folder in avaDir Inputs

        Parameters
        ------------
        avaDir: str or pathlib patch
            path to avalanche directory
    """

    avaDirInputs = pathlib.Path(avaDir, 'Inputs')
    avaDirInputsString = str(avaDirInputs)

    # check for empty or non string variable
    result = _checkAvaDirVariable(avaDirInputsString)
    if 'SUCCESS' not in result:
        return result

    # clean directory
    folderName = 'remeshedRasters'
    _checkForFolderAndDelete(avaDirInputsString, folderName)
    log.info('Cleaned %s/%s directory' % (avaDirInputsString,folderName))

    return 'SUCCESS'


def cleanLatestConfigurationsDirAndCreate(avaDir, modName):
    """ clean configurationFilesLatest folder in avaDir/modName/Outputs/

        Parameters
        ------------
        avaDir: str or pathlib patch
            path to avalanche directory
        modName: str
            name of module
    """

    avaDirOutputs = pathlib.Path(avaDir, 'Outputs', modName, 'configurationFiles')
    avaDirOutputsString = str(avaDirOutputs)

    # check for empty or non string variable
    result = _checkAvaDirVariable(avaDirOutputsString)
    if 'SUCCESS' not in result:
        return result

    # clean directory
    folderName = 'configurationFilesLatest'
    _checkForFolderAndDelete(avaDirOutputsString, folderName)
    log.info('Cleaned %s/%s directory' % (avaDirOutputsString,folderName))

    # create new dir configurationFilesLatest
    latestConfigDir = avaDirOutputs / folderName
    fU.makeADir(latestConfigDir)

    return 'SUCCESS'

def initializeFolderStruct(pathAvaName, removeExisting=False):
    ''' Initialize the standard folder structure. If removeExisting is true,
    deletes any existing folders! BEWARE!

    Parameters
    ----------
    pathAvaName: str
        string with base avalanche path

    removeExisting: boolean, optional, default: False
        remove existing directory, use this to completely clean out the
        directory
    '''

    avaName = os.path.basename(os.path.normpath(pathAvaName))

    log.info("Running initialization sequence for avalanche %s", avaName)

    if os.path.exists(pathAvaName):
        if removeExisting:
            log.warning('Will delete %s !', pathAvaName)
            # remove and remake
            shutil.rmtree(pathAvaName)
            os.makedirs(pathAvaName)
        else:
            log.warning("Folder %s already exists. Continuing", avaName)

    else:
        try:
            os.makedirs(pathAvaName)
        except IOError as e:
            log.error('I/O error({0}): {1}'.format(e.errno, e.strerror))
            return

    createFolderStruct(pathAvaName)

    log.info("Done initializing avalanche %s", avaName)


def createFolderStruct(pathAvaName):
    '''creates the folder structure with avalanche base path'''

    Inputs = checkMakeDir(pathAvaName, 'Inputs')

    inputsSubDirs = ['CFGs', 'RES', 'REL', 'SECREL', 'ENT',
                     'POINTS', 'LINES', 'POLYGONS', 'RELTH', "RASTERS"]

    for cuDir in inputsSubDirs:
        checkMakeDir(Inputs, cuDir)

    checkMakeDir(pathAvaName, 'Outputs')

    checkMakeDir(pathAvaName, 'Work')


def checkMakeDir(base, dirName):
    '''combines base and dir, checks whether it exists, if not creates it
    '''

    pathName = os.path.join(base, dirName)
    if not os.path.exists(pathName):
        os.makedirs(pathName)
        log.debug('Creating folder: %s: ', pathName)
    else:
        log.info('Folder exists: %s, Skipping', pathName)
    return pathName
