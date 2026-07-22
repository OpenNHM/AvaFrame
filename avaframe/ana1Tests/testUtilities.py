"""
    Functions for handling benchmark tests data, filtering, sorting
"""

# Load modules
import pathlib
import logging
import json
import numpy as np

# Local imports
from avaframe.in3Utils import cfgUtils
from avaframe.in3Utils import logUtils
import avaframe.com1DFA.com1DFA as com1DFA


# create local logger
# change log level in calling module to DEBUG to see log messages
log = logging.getLogger(__name__)


def createDesDictTemplate():
    """create an empty dictionary with all required test info keys"""

    desDict = {"TAGS": [], "DESCRIPTION": "", "TYPE": [], "FILES": [], "AVANAME": ""}

    return desDict


def writeDesDicttoJson(desDict, testName, outDir):
    """create a json file with all required test info from desdict

    Parameters
    -----------
    desDict: dict
        description dictionary of test
    testName: str
        name of benchmark test
    outDir: str or pathlib Path
        path where json file is saved

    Returns
    --------
    fileName: pathlib path
        path to json file
    """

    fileName = pathlib.Path(outDir, "%s_desDict.json" % testName)
    with open(fileName, 'w') as outfile:
        json.dump(desDict, outfile, indent=2)

    return fileName


def readDesDictFromJson(fileName):
    """read dict from json file

    Parameters
    -----------
    fileName: str or pathlib path
        path to json file

    Returns
    --------
    desDict: dict
        dictionary read from json file
    """

    with open(fileName) as f:
        desDict = json.load(f)

    return desDict


def readAllBenchmarkDesDicts(info=False, inDir=""):
    """get descritption dicts for all benchmark tests and add test name as key

    Parameters
    ----------
    info: bool
        True if info shall be printed to log
    inDir: pathLib object
        path to benchmarks directory

    Returns
    --------
    testDictList: list
        list of test info dictionaries
    """

    if inDir == "":
        inDir = pathlib.Path("..", "benchmarks")

    testDirs = list(inDir.glob("ava*"))
    testDirsCom4 = list(inDir.glob("com4*"))
    testDirs.extend(testDirsCom4)
    testDictList = []

    for testDir in testDirs:
        desDictFile = list(testDir.glob("*desDict.json"))
        if desDictFile != []:
            testName = testDir.name
            desDict = readDesDictFromJson(desDictFile[0])
            desDict.update({"NAME": testName})
            if info:
                log.info("%s: Tags: %s" % (desDict["NAME"], desDict["TAGS"]))
                log.debug("%s: Full Info: %s" % (testDir, desDict))

            testDictList.append(desDict)
        else:
            testName = testDir.name
            log.debug("%s: No test description file provided! Test skipped" % testName)

    return testDictList


def filterBenchmarks(testDictList, filterType, valuesList, condition="or"):
    """filter benchmarks according to characteristic

    Parameters
    -----------
    testDictList: list
        list of benchmark dictionaries
    filterType: str
        key which is used for filtering (options: TAGS, TYPE)
    valuesList: list
        list of values used for filtering
    condition: str
        if multiple values given in valuesList - if 'or' then all test dictionaries
        are returned that have either of the values, if 'and' then only those are
        returned that feature both values

    Returns
    ---------
    testList: list
        list of all the benchmark dicionaries that meet filter criterion

    """

    testList = []
    flag = False

    # convert valueslist to list if not already list
    if (isinstance(valuesList, list) is False) and (isinstance(valuesList, np.ndarray) is False):
        valuesList = [valuesList]

    for testDict in testDictList:
        if filterType == "AVANAME":
            if testDict["AVANAME"] in valuesList:
                testList.append(testDict)
        elif filterType == "NAME":
            if testDict["NAME"] in valuesList:
                testList.append(testDict)
        else:
            # convert testDict[filterType] to list if not already list
            if isinstance(testDict[filterType], list) is False:
                testDict[filterType] = [testDict[filterType]]
            if condition == "or":
                if any(values in testDict[filterType] for values in valuesList):
                    testList.append(testDict)
            elif condition == "and":
                if all(values in testDict[filterType] for values in valuesList):
                    testList.append(testDict)
            else:
                flag = True

    if flag:
        log.warning("Not a valid condition - please chose and or or as condition")

    return testList


def getTestAvaDirs(testList):
    """get a list of avalanche directories of tests

    Parameters
    -----------
    testList: list
        list of test dictionaries

    Returns
    --------
    avaDirs: list
        list of paths to avalanche directories
    """

    avaDirs = []
    for tests in testList:
        avaDir = pathlib.Path("data", tests["AVANAME"])
        avaDirs.append(str(avaDir))

    return avaDirs


def fetchBenchmarkResults(testName, resTypes=[], refDir=""):
    """Fetch a list of all paths to result files in a benchmark test,
    default all result file paths, if resType list provided only for resTypes in list

    Parameters
    -----------
    refDir: str
        path to benchmark test directory
    resTypes: list
        list of all resTypes that shall be returned

    Returns
    -------
    refFiles: list
        list of paths to all result files found for benchmark test and resType
    """

    if refDir == "":
        refDir = pathlib.Path("..", "benchmarks", testName)

    if resTypes != []:
        refFiles = []
        for resType in resTypes:
            refFiles.extend(
                list(refDir.glob("*" + resType + ".asc")) + list(refDir.glob("*" + resType + ".tif"))
            )
    else:
        refFiles = list(refDir.glob("*.asc")) + list(refDir.glob("*.tif"))

    for refFile in refFiles:
        log.info("Benchmarktest fetched result file: %s" % refFile)

    return refFiles
