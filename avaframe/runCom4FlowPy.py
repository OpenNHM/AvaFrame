"""
Run script for module com4FlowPy
"""

# Load modules
import pathlib
import os
import sys
from datetime import datetime
import logging
import json

# Local imports
import avaframe.in3Utils.initializeProject as initProj
from avaframe.com4FlowPy import com4FlowPy
from avaframe.in3Utils import cfgUtils
from avaframe.in3Utils import logUtils
import avaframe.in3Utils.initialiseDirs as inDirs

# Local imports (avaFrame API)
from avaframe.in1Data import getInput as gI
import avaframe.in3Utils.initialiseDirs as inDirs
from avaframe.in3Utils import fileHandlerUtils as fU
import avaframe.in2Trans.shpConversion as shpConv
import avaframe.in2Trans.rasterUtils as IOf
import avaframe.in3Utils.geoTrans as gT


def main(avalancheDir=''):
    """this is a wrapper around com4FlowPy.py that handles the following tasks:
    * reading inputs from (local_)avaframeCfg.ini and (local_)com4FlowPyCfg.ini
    * constructing cfgPath and cfgSetup dictionaries for passing to com4FlowPy.com4FlowPyMain()
    * handling creation of working directories ...

    NOTE-TODO:
        * This function needs clean-up!
    """
    # log file name; leave empty to use default runLog.log
    logName = "runcom4FlowPy"
    # Read main Config and com4FlowPy config from files
    cfgMain = cfgUtils.getGeneralConfig()
    cfg = cfgUtils.getModuleConfig(com4FlowPy)

    # check and handle outputFiles list provided in (local_)com4FlowPyCfg.ini
    cfg["PATHS"]["outputFiles"] = checkOutputFilesFormat(cfg["PATHS"]["outputFiles"])

    cfgSetup = cfg["GENERAL"]
    cfgFlags = cfg["FLAGS"]
    cfgCustomPaths = cfg["PATHS"]

    # if customPaths == False --> use AvaFrame Folder structure
    if cfgCustomPaths["useCustomPaths"] == "False":
        # if "useCustomPaths" == False, we also use the AvaDir Info for the
        # creation of the simulaiton uid

        if avalancheDir != '':
            cfgMain['MAIN']['avalancheDir'] = avalancheDir
        else:
            # Load avalanche directory from general configuration file
            avalancheDir = cfgMain["MAIN"]["avalancheDir"]
        cfg['GENERAL']['avaDir'] = cfgMain["MAIN"]["avalancheDir"]
        uid = cfgUtils.cfgHash(cfg)
        # Clean input directory of old work and output files from module
        initProj.cleanModuleFiles(avalancheDir, com4FlowPy, deleteOutput=False)

        # Start logging
        log = logUtils.initiateLogger(avalancheDir, logName+'_'+uid)
        log.info("==================================")
        log.info("MAIN SCRIPT")
        log.info("Current avalanche: %s", avalancheDir)
        log.info("==================================")

        # Extract input file locations
        cfgPath = readFlowPyinputs(avalancheDir, cfg, log)

        # use all input data from AvaFrame structure except DEM
        if cfgCustomPaths["useCustomPathDEM"] == "True":
            cfgPath["demPath"] = pathlib.Path(cfgCustomPaths["demPath"])

        # IMPORTANT!! - this is a quick'n'dirty hack to set nSims to 9999, so that
        # min(nCPU,nSims) in cfgUtils.getNumberOfProcesses returns nCPU
        cfgSetup["cpuCount"] = str(cfgUtils.getNumberOfProcesses(cfgMain, 9999))
        cfgPath["customDirs"] = cfgCustomPaths["useCustomPaths"]

        # Create result directory
        # NOTE-TODO: maybe move into separate function as well ...
        timeString = datetime.now().strftime("%Y%m%d_%H%M%S")

        cfgPath["resDir"] = cfgPath["outDir"] / "peakFiles" / "res_{}".format(uid)  # (timeString)
        # check if simulation with same uid already has results folder
        if os.path.isdir(cfgPath["resDir"]):
            log.info("folder with same name already exists - aborting")
            log.info("simulation results folder with same .ini parameters already exists: simulation {}".format(uid))
            sys.exit(1)
        else:
            fU.makeADir(cfgPath["resDir"])
            cfgPath["tempDir"] = cfgPath["workDir"] / "temp"
            fU.makeADir(cfgPath["tempDir"])

        # writing config to .json file
        successToJSON = writeCfgJSON(cfg, uid, cfgPath['outDir'])

        if successToJSON is True:
            log.info('wrote config to {}/{}.json'.format(cfgPath['outDir'], uid))
        else:
            log.info('could not write  config to {}/{}.json'.format(cfgPath['outDir'], uid))
            log.error("Exception occurred: %s", str(successToJSON), exc_info=True)

        cfgPath["deleteTemp"] = "False"

        cfgPath["uid"] = uid
        cfgPath["timeString"] = timeString
        cfgPath["outputFiles"] = cfgCustomPaths["outputFiles"]
        cfgPath["outputNoDataValue"] = cfgCustomPaths["outputNoDataValue"]

        com4FlowPy.com4FlowPyMain(cfgPath, cfgSetup)

    # if customPaths == True --> check
    elif cfgCustomPaths["useCustomPaths"] == "True":
        # if "useCustomPaths" == True, we don't need the AvaDir Info for the
        # creation of the simulation uid
        uid = cfgUtils.cfgHash(cfg)
        cfgPath = {}

        # Handling Custom directory creation
        workDir = pathlib.Path(cfgCustomPaths["workDir"])

        if not os.path.isdir(workDir):
            try:
                os.makedirs(workDir)
            except Exception as e:
                return e

        log = logUtils.initiateLogger(workDir, logName+'_'+uid)

        timeString = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            os.makedirs(workDir / "res_{}".format(uid))  # (time_string))
            res_dir = workDir / "res_{}".format(uid)   # (time_string)
        except FileExistsError:
            log.info("simulation results folder with same .ini parameters already exists: simulation {}".format(uid))
            sys.exit(1)
        try:
            os.makedirs(workDir / res_dir / "temp")
            temp_dir = workDir / res_dir / "temp"
        except FileExistsError:
            log.info("temp folder for simualtion {} already exists - aborting".format(uid))
            sys.exit(1)

        # writing config to .json file
        successToJSON = writeCfgJSON(cfg, uid, workDir)

        if successToJSON is True:
            log.info('wrote config to {}/{}.json'.format(workDir, uid))
        else:
            log.info('could not write  config to {}/{}.json'.format(workDir, uid))
            log.error("Exception occurred: %s", str(successToJSON), exc_info=True)

        cfgPath["workDir"] = pathlib.Path(workDir)
        cfgPath["outDir"] = pathlib.Path(res_dir)
        cfgPath["resDir"] = cfgPath["outDir"]
        cfgPath["tempDir"] = pathlib.Path(temp_dir)
        cfgPath["demPath"] = pathlib.Path(cfgCustomPaths["demPath"])
        cfgPath["releasePath"] = pathlib.Path(cfgCustomPaths["releasePath"])
        cfgPath["infraPath"] = pathlib.Path(cfgCustomPaths["infraPath"])
        cfgPath["forestPath"] = pathlib.Path(cfgCustomPaths["forestPath"])
        cfgPath["varUmaxPath"] = pathlib.Path(cfgCustomPaths["varUmaxPath"])
        cfgPath["varAlphaPath"] = pathlib.Path(cfgCustomPaths["varAlphaPath"])
        cfgPath["varExponentPath"] = pathlib.Path(cfgCustomPaths["varExponentPath"])
        cfgPath["deleteTemp"] = cfgCustomPaths["deleteTempFolder"]
        cfgPath["outputFileFormat"] = cfgCustomPaths["outputFileFormat"]
        cfgPath["outputFiles"] = cfgCustomPaths["outputFiles"]
        cfgPath["outputNoDataValue"] = cfgCustomPaths.getfloat("outputNoDataValue")

        cfgSetup["cpuCount"] = str(cfgUtils.getNumberOfProcesses(cfgMain, 9999))
        cfgPath["customDirs"] = cfgCustomPaths["useCustomPaths"]

        cfgPath["uid"] = uid
        cfgPath["timeString"] = timeString

        com4FlowPy.com4FlowPyMain(cfgPath, cfgSetup)

    else:
        print(
            "INPUT SETTINGS incorrect - please check (local_)avaframeCfg.ini and (local_)com4FlowPyCfg.ini"
        )
        sys.exit(1)


def readFlowPyinputs(avalancheDir, cfgFlowPy, log):
    """function is used to read necessary flowPy Inputs
    function is only called from '../runCom4FlowPy.py',
    which in turn creates the necessary 'cfgPath' dictionary, that is passed to
    the com4FlowPyMain() function in this file...
    TODO-20240418: check this function - this is still from MTo Flow-Py --> avaFrame port

    Parameters
    -----------
    avalancheDir: str
        directory to avalanche - folder
    cfgFlowPy: configParser object
        Configparser object containing FlowPy model parameters (from (local_)com4FlowPyCfg.ini)
    log: logging object
        initiated logger

    Returns
    -----------
    cfgPath: dict
        contains paths to input data
    """

    cfgPath = {}
    avalancheDir = pathlib.Path(avalancheDir)

    inputDir = avalancheDir / "Inputs"
    relFile, available = gI.getAndCheckInputFiles(inputDir, "REL", "Release Area", fileExt="shp")
    
    if available == "No":
        relFile, available = gI.getAndCheckInputFiles(inputDir, "REL", "Release Area", fileExt="raster")
    if available == "No":
        message = f"There is no release area file in supported format provided in {avalancheDir}/REL"
        log.error(message)
        raise AssertionError(message)
    log.info("Release area file is: %s" % relFile)
    cfgPath["releasePath"] = relFile
    
    # TODO: also use the getAndCheckInputFiles to get the paths for the following files?
    # read infra area
    if cfgFlowPy.getboolean("GENERAL", "infra") is True:
        infraPath, available = gI.getAndCheckInputFiles(inputDir, "INFRA", "Infra", fileExt="raster")
        if available == "No":
            message = f"There is no infra file in supported format provided in {avalancheDir}/INFRA"
            log.error(message)
            raise AssertionError(message)
        log.info("Infrastructure area file is: %s" % infraPath)
    else:
        infraPath = ""
    cfgPath["infraPath"] = infraPath

    # read uMax Limit Raster
    if cfgFlowPy.getboolean("GENERAL", "variableUmaxLim") is True:
        varUmaxPath, available = gI.getAndCheckInputFiles(inputDir, "UMAX", "Umax", fileExt="raster")
        if available == "No":
            message = f"There is no variable UMAX file in supported format provided in {avalancheDir}/UMAX"
            log.error(message)
            raise AssertionError(message)
        log.info("uMax Limit file is: %s" % varUmaxPath)
    else:
        varUmaxPath = ""
    cfgPath["varUmaxPath"] = varUmaxPath

    # read variable Alpha Angle Raster

    if cfgFlowPy.getboolean("GENERAL", "variableAlpha") is True:
        varAlphaPath, available = gI.getAndCheckInputFiles(inputDir, "ALPHA", "alpha", fileExt="raster")
        if available == "No":
            message = f"There is no variable ALPHA file in supported format provided in {avalancheDir}/ALPHA"
            log.error(message)
            raise AssertionError(message)
        log.info("variable Alpha file is: %s" % varAlphaPath)
    else:
        varAlphaPath = ""
    cfgPath["varAlphaPath"] = varAlphaPath

    # read variable Exponent Raster

    if cfgFlowPy.getboolean("GENERAL", "variableExponent") is True:
        varExponentPath, available = gI.getAndCheckInputFiles(inputDir, "EXP", "exp", fileExt="raster")
        if available == "No":
            message = f"There is no variable EXPONENT file in supported format provided in {avalancheDir}/EXP"
            log.error(message)
            raise AssertionError(message)
        log.info("variable Exponent file is: %s" % varExponentPath)
    else:
        varExponentPath = ""
    cfgPath["varExponentPath"] = varExponentPath

    # check if forest should be used (assumed to be in the RES - 'RESISTANCE' directory)
    # TODO: should we also allow forest as shp - file?

    if cfgFlowPy.getboolean("GENERAL", "forest") is True:
        forestPath, available = gI.getAndCheckInputFiles(inputDir, "RES", "forest", fileExt="raster")
        if available == "No":
            message = f"There is no forest file in supported format provided in {avalancheDir}/RES"
            log.error(message)
            raise AssertionError(message)
        log.info("Forest file is: %s" % forestPath)
    else:
        forestPath = ""
    cfgPath["forestPath"] = forestPath

    # read DEM
    if cfgFlowPy.getboolean("PATHS", "useCustomPathDEM") is False:
        demPath = gI.getDEMPath(avalancheDir)

        log.info("DEM file is: %s" % demPath)
        cfgPath["demPath"] = demPath

    # make output path
    workDir, outDir = inDirs.initialiseRunDirs(avalancheDir, "com4FlowPy", False)
    cfgPath["outDir"] = outDir
    cfgPath["workDir"] = workDir

    cfgPath["outputFileFormat"] = cfgFlowPy["PATHS"]["outputFileFormat"]

    return cfgPath


def checkOutputFilesFormat(strOutputFiles):
    """check if outputFiles option is provided in proper format, else return
    default string in right format

    Parameters
    ---------------
    strOutputFiles: str
        outputFiles string from (local_)com4FlowPy.ini

    Returns
    ---------------
    strOutputFiles: str
        returns the input if the format is ok, else default value string is returned
    """

    try:
        setA = set(strOutputFiles.split('|'))
        setB = set(['zDelta', 'cellCounts', 'fpTravelAngle', 'travelLength', 'fpTravelAngleMax',
                    'fpTravelAngleMin', 'travelLengthMax', 'travelLengthMin',
                    'slTravelAngle', 'flux', 'zDeltaSum', 'routFluxSum', 'depFluxSum'])
        # if there is at least 1 correct outputfile defined, we use the string provided in the .ini file
        if (setA & setB):
            return strOutputFiles
        else:
            raise ValueError('outputFiles defined in .ini have wrong format - using default settings')
    except ValueError:
        # else we return the default options
        return 'zDelta|cellCounts|travelLength|fpTravelAngle'


def writeCfgJSON(cfg, uid, workDir):
    """
    writes a JSON file containing all the input parameters from the configFile
    using the same uid as the simulation results

    Parameters
    ---------------
    cfg: configParser object
        all the model configs are in here
    uid: str
        UID created based on the cfg object
    workDir: str
        workDirectory (place to write the .json file to)

    Returns
    ---------------
    success: bool/Exception
        True if file is written successfully, else Exception

    """

    cfgDict = cfgUtils.convertConfigParserToDict(cfg)

    try:
        with open(workDir / "{}.json".format(uid), 'w') as outfile:
            jsonDict = json.dumps(cfgDict, sort_keys=True, ensure_ascii=True)
            outfile.write(jsonDict)

        return True

    except Exception as e:
        return e


if __name__ == "__main__":
    main()
