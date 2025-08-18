"""
Directory and file handling helper functions
"""

# Load modules
import os
import glob
import logging
import pathlib
import numpy as np
import pandas as pd
import shutil

# Local imports
import avaframe.in2Trans.rasterUtils as IOf

# create local logger
# change log level in calling module to DEBUG to see log messages
log = logging.getLogger(__name__)


def makeADir(dirName):
    """Check if a directory exists, if not create directory

    Parameters
    ----------
    dirName : str
        path of directory that should be made
    """

    # If directory already exists - Delete directory first is default
    if os.path.isdir(dirName):
        log.debug("Be careful directory %s already existed - data saved on top of it" % (dirName))
    else:
        # due parallel computing it might be problematic if exist_ok is not set to True
        os.makedirs(dirName, exist_ok=True)
    log.debug("Directory: %s created" % dirName)


def checkPathlib(checkPath):
    """check if pathlib.PurePath if not convert to

    Parameters
    ----------
    checkPath: str or pathlib path
    path to be checked

    Returns
    -------
    checkPath: pathlib path
        pathlib path version of checkPath
    """

    if not isinstance(checkPath, pathlib.PurePath):
        checkPath = pathlib.Path(checkPath)

    return checkPath


def readLogFile(logName, cfg=""):
    """Read experiment log file and make dictionary that contains general info on all simulations

    Parameters
    ----------
    logName : str
        path to log file
    cfg : dict
        optional - configuration read from com1DFA simulation

    Returns
    -------
    logDict : dict
        dictionary with number of simulation (noSim), name of simulation (simName),
        parameter variation, full name
    """

    # Read log file
    logFile = open(logName, "r")
    log.debug("Take com1DFA full experiment log")

    # Parameter variation
    if cfg != "":
        varPar = cfg["varPar"]
    else:
        varPar = "Mu"

    # Save info to dictionary, add all result parameters that are saved in com1DFA Outputs
    logDict = {"noSim": [], "simName": [], varPar: [], "fullName": []}

    lines = logFile.readlines()[1:]
    countSims = 1
    for line in lines:
        vals = line.strip().split()
        logDict["noSim"].append(countSims)
        logDict["simName"].append(vals[1])
        logDict[varPar].append(float(vals[2]))
        logDict["fullName"].append(vals[1] + "_" + "%.5f" % float(vals[2]))
        countSims = countSims + 1

    logFile.close()

    return logDict


def extractParameterInfo(avaDir, simName, reportD):
    """Extract info about simulation parameters from the log file

    Parameters
    ----------
    avaDir : str
        path to avalanche
    simName : str
        name of the simulation
    reportD: dict
        report dictionary

    Returns
    -------
    parameterDict : dict
        dictionary listing name of parameter and value; release mass, final time step and current mast
    reportD: dict
        upated report dictionary with info on simulation
    """
    # Initialise parameter dictionary
    parameterDict = {}

    # Read log file
    fileName = pathlib.Path(avaDir, "Outputs", "com1DFAOrig", "start%s.log" % (simName))
    logDict = extractLogInfo(fileName)
    stopCrit = logDict["stopCrit"]

    # Set values in dictionary
    parameterDict["release mass [kg]"] = logDict["mass"][0]
    parameterDict["final time step [s]"] = logDict["time"][-1]
    parameterDict["current mass [kg]"] = logDict["mass"][-1]
    if stopCrit != "":
        parameterDict["stop criterion"] = logDict["stopCrit"]
    else:
        parameterDict["stop criterion"] = "end time reached: %.2f s" % logDict["time"][-1]
    parameterDict["Avalanche run time [s]"] = "%.2f" % logDict["stopTime"]
    parameterDict["CPU time [s]"] = logDict["stopTime"]

    reportD["Simulation Parameters"].update({"Stop criterion": parameterDict["stop criterion"]})
    reportD["Simulation Parameters"].update({"Avalanche run time [s]": parameterDict["final time step [s]"]})
    reportD["Simulation Parameters"].update(
        {"Initial mass [kg]": ("%.2f" % parameterDict["release mass [kg]"])}
    )
    reportD["Simulation Parameters"].update(
        {"Final mass [kg]": ("%.2f" % parameterDict["current mass [kg]"])}
    )

    return parameterDict, reportD


def extractLogInfo(fileName):
    """read log file and extract info on time, mass stop criterion

    Parameters
    -----------
    fileName: str or pathlib path
        path to log file

    Returns
    --------
    logDict: dict
        dictionary with arrays for mass entrained mass time step
        and info on simulation run and stop criterion"""

    time = []
    mass = []
    entrMass = []
    stopCrit = ""
    indRun = [0]
    countMass = 0

    with open(fileName, "r") as file:
        for line in file:
            if "computing time step" in line:
                ltime = line.split()[3]
                timeNum = ltime.split("...")[0]
                time.append(float(timeNum))
            elif "entrained DFA mass" in line:
                entrMass.append(float(line.split()[3]))
            elif "total DFA mass" in line:
                mass.append(float(line.split()[3]))
                countMass = countMass + 1
            elif "Kinetische Energie" in line:
                stopCrit = "kinetic energy %.2f of peak KE" % (float(line.split()[3]))
            elif "terminated" in line:
                stopTime = float(line.split()[5])
                indRun.append(countMass)

    logDict = {
        "time": np.asarray(time),
        "mass": np.asarray(mass),
        "stopTime": stopTime,
        "entrMass": np.asarray(entrMass),
        "indRun": indRun,
        "stopCrit": stopCrit,
    }

    return logDict


def checkIfFileExists(filePath, fileType=""):
    """test if file exists if not throw error

    Parameters
    -----------
    filePath: pathlib path
        path to file
    fileType: str
        string for error message which kind of file is not found

    """
    if not isinstance(filePath, pathlib.PurePath):
        filePath = pathlib.Path(filePath)
    if not filePath.is_file():
        message = "No %s file found called: %s" % (fileType, str(filePath))
        log.error(message)
        raise FileNotFoundError(message)


def checkCommonSims(logName, localLogName):
    """Check which files are common between local and full ExpLog"""

    if os.path.isfile(localLogName) is False:
        localLogName = logName

    # Read log files and extract info
    logDict = readLogFile(logName)
    logDictLocal = readLogFile

    # Identify common simulations
    setSim = set(logDictLocal["simName"])
    indSims = [i for i, item in enumerate(logDict["simName"]) if item in setSim]

    log.info("Common simulations are: %d" % indSims)

    return indSims


def getFilterDict(cfg, section):
    """Create parametersDict from ini file, for filtering simulations

    Parameters
    -----------
    cfg: configParser object
        configuration with information on filtering criteria
    section: str
        section of cfg where filtering criteria can be found

    Returns
    ---------
    parametersDict : dict
        dictionary with parameter and parameter values for filtering simulation results

    """

    parametersDict = {}
    if cfg.has_section(section):
        for key, value in cfg.items(section):
            if value == "":
                parametersDict.pop(key, None)
            elif ":" in value or "|" in value:
                locValue = splitIniValueToArraySteps(value)
                parametersDict[key] = locValue
                log.info("Filter simulations that match %s: %s" % (key, locValue))
            elif ">" in value:
                parametersDict[key] = [value]
                log.info("Filter simulations where %s is greater than %s" % (key, value))
            elif "<" in value:
                parametersDict[key] = [value]
                log.info("Filter simulations where %s is smaller than %s" % (key, value))
            elif isinstance(value, str):
                testValue = value.replace(".", "")
                if testValue.isdigit():
                    parametersDict[key] = [float(value)]
                else:
                    parametersDict[key] = [value]
                log.info("Filter simulations that match %s: %s" % (key, value))
    else:
        log.warning("No section %s in configuration file found - cannot create dict for filtering" % section)

    return parametersDict


def splitIniValueToArraySteps(cfgValues, returnList=False):
    """read values in ini file and return numpy array or list if the items are strings;
    values can either be separated by | or provided in start:end:numberOfSteps format
    if separated by : or $ also optional add one additional value using &
    if format of refVal$percent$steps is used - an array is created with +- percent of refVal in nsteps

    Parameters
    ----------
    cfgValues : str
        values of parameter to be read from ini file
    returnList: bool
        if True force to return values as list

    Returns
    --------
    items : 1D numpy array or list
        values as 1D numpy array or list (in the case of strings)
    """

    if ":" in cfgValues:
        if "&" in cfgValues:
            itemsInputBig = cfgValues.split("&")
            itemsInput = itemsInputBig[0].split(":")
        else:
            itemsInput = cfgValues.split(":")
        items = np.linspace(float(itemsInput[0]), float(itemsInput[1]), int(itemsInput[2]))
        if "&" in cfgValues:
            items = np.append(items, float(itemsInputBig[1]))
    elif cfgValues == "":
        items = []
    elif "$" in cfgValues:
        if "&" in cfgValues:
            itemsPBig = cfgValues.split("&")
            itemsP = itemsPBig[0].split("$")
        else:
            itemsP = cfgValues.split("$")
        itemsPRange = (float(itemsP[1]) / 100.0) * float(itemsP[0])
        # check if pos or neg or full var
        if "-" in itemsP[1] or "+" in itemsP[1]:
            items = np.linspace(float(itemsP[0]), float(itemsP[0]) + itemsPRange, int(itemsP[2]))
        else:
            items = np.linspace(
                float(itemsP[0]) - itemsPRange, float(itemsP[0]) + itemsPRange, int(itemsP[2])
            )
        if "&" in cfgValues:
            items = np.append(items, float(itemsPBig[1]))
    else:
        itemsL = cfgValues.split("|")
        if returnList:
            items = itemsL
        else:
            flagFloat = False
            flagString = False
            for its in itemsL:
                if its.upper().isupper() and ".e" not in its and ".E" not in its:
                    flagString = True
                if "." in its:
                    flagFloat = True
            if flagString:
                items = itemsL
            elif flagFloat:
                items = np.array(itemsL, dtype=float)
            else:
                items = np.array(itemsL, dtype=int)

    return items


def splitTimeValueToArrayInterval(cfgValues, endTime):
    """read save time step info from ini file and return numpy array of values

    values can either be separated by | or provided in start:interval format

    Parameters
    ----------
    cfgValues: str
        time steps info
    endTime: float
        end time

    Returns
    --------
    items : 1D numpy array
        sorted time step values as 1D numpy array
    """

    if ":" in cfgValues:
        itemsInput = cfgValues.split(":")
        if float(endTime - float(itemsInput[0])) < float(itemsInput[1]):
            items = np.array([float(itemsInput[0]), endTime])
        else:
            items = np.arange(float(itemsInput[0]), endTime, float(itemsInput[1]))
    elif cfgValues == "":
        items = np.array([2 * endTime])
    else:
        itemsL = cfgValues.split("|")
        items = np.array(itemsL, dtype=float)
        items = np.sort(items)

    # make sure that 0 is not in the array (initial time step is any ways saved)
    if items[0] == 0:
        items = np.delete(items, 0)
    # make sure the array is not empty
    # ToDo : make it work without this arbitrary 2*timeEnd
    if items.size == 0:
        items = np.array([2 * endTime])

    return items


def exportcom1DFAOrigOutput(avaDir, cfg="", addTSteps=False):
    """Export the simulation results from com1DFA output to desired location

    Parameters
    ----------
    avaDir: str
        path to avalanche directory
    cfg : dict
        configuration read from ini file that has been used for the com1DFAOrig simulation
    addTSteps : bool
        if True: first and last time step of flow thickness are exported

    """

    # Initialise directories
    inputDir = pathlib.Path(avaDir, "Work", "com1DFAOrig")
    outDir = pathlib.Path(avaDir, "Outputs", "com1DFAOrig")
    outDirPF = outDir / "peakFiles"
    outDirRep = outDir / "reports"
    makeADir(outDir)
    makeADir(outDirPF)
    makeADir(outDirRep)

    # Read log file information
    logName = inputDir / "ExpLog.txt"
    if cfg != "":
        logDict = readLogFile(logName, cfg)
        varPar = cfg["varPar"]
    else:
        logDict = readLogFile(logName)
        varPar = "Mu"

    # Get number of values
    sNo = len(logDict["noSim"])

    # Path to com1DFA results
    resPath = inputDir / ("FullOutput_%s_" % varPar)

    if addTSteps is True:
        timeStepDir = outDirPF / "timeSteps"
        makeADir(timeStepDir)

    # Export peak files and reports
    for k in range(sNo):
        pathFrom = pathlib.Path(
            "%s%.05f" % (resPath, logDict[varPar][k]),
            logDict["simName"][k],
            "raster",
            "%s_pfd.asc" % logDict["simName"][k],
        )
        pathTo = outDirPF / ("%s_%.05f_pft.asc" % (logDict["simName"][k], logDict[varPar][k]))
        shutil.copy(pathFrom, pathTo)
        if addTSteps is True:
            pathFrom = pathlib.Path(
                "%s%.05f" % (resPath, logDict[varPar][k]),
                logDict["simName"][k],
                "raster",
                "%s_fd.asc" % logDict["simName"][k],
            )
            pathTo = (
                outDirPF
                / "timeSteps"
                / ("%s_%.05f_tLast_ft.asc" % (logDict["simName"][k], logDict[varPar][k]))
            )
            shutil.copy(pathFrom, pathTo)
        pathFrom = pathlib.Path(
            "%s%.05f" % (resPath, logDict[varPar][k]),
            logDict["simName"][k],
            "raster",
            "%s_ppr.asc" % logDict["simName"][k],
        )
        pathTo = outDirPF / ("%s_%.05f_ppr.asc" % (logDict["simName"][k], logDict[varPar][k]))
        shutil.copy(pathFrom, pathTo)
        pathFrom = pathlib.Path(
            "%s%.05f" % (resPath, logDict[varPar][k]),
            logDict["simName"][k],
            "raster",
            "%s_pv.asc" % logDict["simName"][k],
        )
        pathTo = outDirPF / ("%s_%.05f_pfv.asc" % (logDict["simName"][k], logDict[varPar][k]))
        shutil.copy(pathFrom, pathTo)
        pathFrom = pathlib.Path("%s%.05f" % (resPath, logDict[varPar][k]), "%s.html" % logDict["simName"][k])
        pathTo = outDirRep / ("%s_%.05f.html" % (logDict["simName"][k], logDict[varPar][k]))
        shutil.copy(pathFrom, pathTo)

    if addTSteps is True:

        # Export peak files and reports
        for k in range(sNo):
            pathFrom = pathlib.Path(
                "%s%.05f" % (resPath, logDict[varPar][k]), "%s_tFirst_fd.txt" % logDict["simName"][k]
            )
            pathTo = (
                outDirPF
                / "timeSteps"
                / ("%s_%.05f_tFirst_fd.asc" % (logDict["simName"][k], logDict[varPar][k]))
            )
            shutil.copy(pathFrom, pathTo)

    # Export ExpLog to Outputs/com1DFA
    shutil.copy2(pathlib.Path("%s" % inputDir, "ExpLog.txt"), outDir)


def fetchFlowFields(flowFieldsDir, suffix=""):
    """fetch paths to all desired flow fields within folder

    Parameters
    ------------
    flowFieldsDir: str or pathlib path
        path to flow field ascii files
    suffix: str
        suffix in flow field name to be searched for

    Returns
    --------
    flowFields: list
        list of pathlib paths to flow fields

    """

    # check if pathlib path
    if isinstance(flowFieldsDir, pathlib.PurePath):
        flowFieldsDir = pathlib.Path(flowFieldsDir)

    if suffix == "":
        searchString = "*.asc"
    else:
        searchString = "*%s*.asc" % suffix
    flowFields = list(flowFieldsDir.glob(searchString))

    return flowFields


def fileNotFoundMessage(messageName):
    """throw error if file not found with message and path

    Parameters
    -----------
    messageName: str
        error message

    """

    log.error(messageName)
    raise FileNotFoundError(messageName)


# ToDo Maybe try to use makeSimFromResDF instead of makeSimDF
def makeSimDF(inputDir, avaDir="", simID="simID"):
    """Create a  dataFrame that contains all info on simulations

    this can then be used to filter simulations for example

    Parameters
    ----------
    inputDir : str
        path to directory of simulation results
    avaDir : str
        optional - path to avalanche directory
    simID : str
        optional - simulation identification, depending on the computational module:
        com1DFA: simHash
        com1DFAOrig: Mu or parameter that has been used in parameter variation

    Returns
    -------
    dataDF : dataFrame
        dataframe with full file path, file name, release area scenario, simulation type (null, entres, etc.),
        model type (dfa, ref, etc.), simID, result type (ppr, pft, etc.), simulation name,
        cell size and optional name of avalanche, optional time step
    """

    # Load input datasets from input directory
    if isinstance(inputDir, pathlib.Path) is False:
        inputDir = pathlib.Path(inputDir)

    datafiles = list(inputDir.glob("*.asc")) + list(inputDir.glob("*.tif"))

    # Sort datafiles by name
    datafiles = sorted(datafiles)

    # Set name of avalanche if avaDir is given
    # Make dictionary of input data info
    data = {
        "files": [],
        "names": [],
        "resType": [],
        "simType": [],
        "isDefault": [],
        "frictCalib": [],
        "simName": [],
        "modelType": [],
        "releaseArea": [],
        "cellSize": [],
        simID: [],
        "timeStep": [],
    }

    # Set name of avalanche if avaDir is given
    if avaDir != "":
        avaDir = pathlib.Path(avaDir)
        avaName = avaDir.name
        data.update({"avaName": []})

    for m in range(len(datafiles)):
        data["files"].append(datafiles[m])
        name = datafiles[m].stem
        data["names"].append(name)
        if "_AF_" in name:
            nameParts = name.split("_AF_")
            fNamePart = nameParts[0] + "_AF"
            relNameSim = nameParts[0]
            infoParts = nameParts[1].split("_")

        else:
            nameParts = name.split("_")
            fNamePart = nameParts[0]
            relNameSim = nameParts[0]
            infoParts = nameParts[1:]

        data["releaseArea"].append(relNameSim)
        data[simID].append(infoParts[0])

        indiStr = ["_C_", "_D_"]
        if any(x in name for x in indiStr):
            data["isDefault"].append(infoParts[1])
            # now check for friction calibration info
            frictIndi = ["_S_", "_M_", "_L_"]
            if any(x in name for x in frictIndi):
                data["frictCalib"].append(infoParts[2])
                j = 1  # j indicates whether there's an additional info
            else:
                data["frictCalib"].append(None)
                j = 0

            data["simType"].append(infoParts[2 + j])
            data["modelType"].append(infoParts[3 + j])
            data["resType"].append(infoParts[4 + j])
            data["simName"].append(fNamePart + "_" + ("_".join(infoParts[0 : (4 + j)])))

            header = IOf.readRasterHeader(datafiles[m])
            data["cellSize"].append(header["cellsize"])
            if len(infoParts) == (6 + j):
                data["timeStep"].append(infoParts[5 + j])
            else:
                data["timeStep"].append("")

        # If it still is an 'old' simname
        # This can be removed at one point
        else:
            data["isDefault"].append(None)
            data["frictCalib"].append(None)
            data["simType"].append(infoParts[1])
            data["modelType"].append(infoParts[2])
            data["resType"].append(infoParts[3])
            data["simName"].append(fNamePart + "_" + ("_".join(infoParts[0:3])))

            header = IOf.readRasterHeader(datafiles[m])
            data["cellSize"].append(header["cellsize"])
            if len(infoParts) == 5:
                data["timeStep"].append(infoParts[4])
            else:
                data["timeStep"].append("")

        # Set name of avalanche if avaDir is given
        if avaDir != "":
            data["avaName"].append(avaName)

    dataDF = pd.DataFrame.from_dict(data)

    return dataDF


def makeSimFromResDF(avaDir, comModule, inputDir="", simName=""):
    """Create a  dataFrame that contains all info on simulations in output/comModule/peakFiles

    One line for each simulation - so all peakfiles that belong to one simulation are listed in one line
    that corresponds to that simulation

    Parameters
    ----------
    avaDir : str
        path to avalanche directory
    comModule : str
        module used to create the results
    inputDir : str
        optional - path to directory of simulation results
    simName : str
        optional - key phrase to be found in the simulation result name

    Returns
    -------
    dataDF : dataFrame
        dataframe with for each simulation, the full file path, file name, release area scenario,
        simulation type (null, entres, etc.), model type (dfa, ref, etc.), simID,
        path to result files (ppr, pft, etc.), simulation name,
        cell size and optional name of avalanche, optional time step
    resTypeListAll: list
        list of res types available for all simulations
    """

    # get path to folder containing the raster files
    if inputDir == "":
        inputDir = pathlib.Path(avaDir, "Outputs", comModule, "peakFiles")
    if isinstance(inputDir, pathlib.Path) is False:
        inputDir = pathlib.Path(inputDir)
    if inputDir.is_dir() is False:
        message = "Input directory %s does not exist - check anaMod" % inputDir
        log.error(message)
        raise FileNotFoundError(message)

    # Load input datasets from input directory
    if simName != "":
        ascName = "*" + simName + "*.asc"
        tifName = "*" + simName + "*.tif"

    else:
        ascName = "*.asc"
        tifName = "*.tif"

    datafiles = list(inputDir.glob(ascName)) + list(inputDir.glob(tifName))

    # build the result data frame
    resTypeListFromFiles = list(set([file.stem.split("_")[-1] for file in datafiles]))
    columnsList = [
        "simName",
        "releaseArea",
        "simHash",
        "simModified",
        "simType",
        "modelType",
        "cellSize",
    ] + resTypeListFromFiles
    dataDF = pd.DataFrame(columns=columnsList)
    resTypeListOne = []

    for file in datafiles:
        name = file.stem
        if "_AF_" in name:
            nameParts = name.split("_AF_")
            fNamePart = nameParts[0] + "_AF"
            relNameSim = nameParts[0]
            infoParts = nameParts[1].split("_")
            resType = infoParts[-1]

        else:
            nameParts = name.split("_")
            fNamePart = nameParts[0]
            relNameSim = nameParts[0]
            infoParts = nameParts[1:]
            resType = infoParts[-1]
        simName = fNamePart + "_" + ("_".join(infoParts[0:-1]))
        # add line in the DF if the simulation does not exist yet
        if simName not in dataDF.simName.values:
            newLine = pd.DataFrame([[simName]], columns=["simName"], index=[simName])
            dataDF = pd.concat([dataDF, newLine], ignore_index=False)
            dataDF.loc[simName, "releaseArea"] = relNameSim
            dataDF.loc[simName, "simHash"] = infoParts[0]
            # TODO: remove once all simNames are updated to include C or D as simModified
            if len(infoParts) == 6:  # this is the _C_M_ etc variant
                dataDF.loc[simName, "simModified"] = infoParts[1]
                dataDF.loc[simName, "simType"] = infoParts[3]
                dataDF.loc[simName, "modelType"] = infoParts[4]
            elif len(infoParts) == 5:
                dataDF.loc[simName, "simModified"] = infoParts[1]
                dataDF.loc[simName, "simType"] = infoParts[2]
                dataDF.loc[simName, "modelType"] = infoParts[3]
            elif len(infoParts) == 4:
                dataDF.loc[simName, "simModified"] = "not specified"
                dataDF.loc[simName, "simType"] = infoParts[1]
                dataDF.loc[simName, "modelType"] = infoParts[2]
            else:
                message = "simName format not recognized for simName: %s" % simName
                log.error(message)
                raise AssertionError(message)

            # add info about the cell size
            header = IOf.readRasterHeader(file)
            dataDF.loc[simName, "cellSize"] = header["cellsize"]
        # add full path to resType
        dataDF.loc[simName, resType] = pathlib.Path(file)
        # list all res types found
        if resType not in resTypeListOne:
            resTypeListOne.append(resType)

    # add a hash for each line of the DF and use as index - required for identifcation
    hash = pd.util.hash_pandas_object(dataDF)
    # reset the index using the dataframe hash
    dataDF = dataDF.set_index(hash)
    # now find res types available for all simulations
    resTypeListAll = []
    for resType in resTypeListOne:
        if not dataDF[resType].isnull().values.any():
            resTypeListAll.append(resType)

    return dataDF, resTypeListAll


def findAvaDirsBasedOnInputsDir(Dir):
    """Find all valid avalanche directories within a given directory.

    A directory is considered a valid avalanche directory if it contains an "Inputs" folder.

    Parameters
    ----------
    Dir : pathlib.Path or str
        Path to the directory to search in

    Returns
    -------
    avaDirs : list
        List of pathlib.Path objects pointing to valid avalanche directories
    """
    avaDirs = [pathlib.Path(p).parent for p in pathlib.Path(Dir).glob("*/Inputs")]
    log.info(f"Found a total of '{len(avaDirs)}' avalanche directories in: {Dir}:")
    for avaDir in avaDirs:
        log.info(f"'{avaDir.name}'")

    return avaDirs
