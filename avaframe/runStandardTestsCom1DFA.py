"""
Run script for running the standard tests with com1DFA
in this test all the available tests tagged standardTest are performed
"""

# Load modules
import time
import pathlib
import logging
import logging.handlers
import multiprocessing
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

# Local imports
from avaframe.com1DFA import com1DFA
from avaframe.ana1Tests import testUtilities as tU
from avaframe.log2Report import generateCompareReport
from avaframe.ana3AIMEC import ana3AIMEC, dfa2Aimec, aimecTools
from avaframe.out3Plot import outQuickPlot
from avaframe.in3Utils import fileHandlerUtils as fU
from avaframe.in3Utils import initializeProject as initProj
from avaframe.in3Utils import cfgUtils
from avaframe.in3Utils import cfgHandling
from avaframe.in3Utils import logUtils
from avaframe.com5SnowSlide import com5SnowSlide


def runSingleTest(
    test,
    cfgMainDict,
    outputVariable,
    aimecDiffLim,
    aimecContourLevels,
    aimecFlagMass,
    aimecComModules,
    outDir,
    tmpTestsDir,
    logQueue,
):
    """
    Run a single standard test in a worker process.

    Parameters
    ----------
    test : dict
        Test dictionary containing test configuration
    cfgMainDict : dict
        Main configuration dictionary (copy for this worker)
    outputVariable : list
        List of output variables for comparison plots
    aimecDiffLim : str
        AIMEC difference limit
    aimecContourLevels : str
        AIMEC contour levels
    aimecFlagMass : str
        AIMEC mass flag
    aimecComModules : str
        AIMEC comparison modules
    outDir : pathlib.Path
        Output directory for reports
    tmpTestsDir : pathlib.Path
        Temporary directory for test isolation
    logQueue : multiprocessing.Queue
        Queue for thread-safe logging

    Returns
    -------
    tuple
        (testName, reportD, benchDict, avaName, cfgRep)
    """
    # Configure logger to use queue handler
    queueHandler = logging.handlers.QueueHandler(logQueue)
    rootLogger = logging.getLogger()
    rootLogger.addHandler(queueHandler)
    rootLogger.setLevel(logging.INFO)
    log = logging.getLogger(__name__)

    # Reconstruct cfgMain from dictionary
    cfgMain = cfgUtils.getGeneralConfig()
    for section in cfgMainDict:
        for key, value in cfgMainDict[section].items():
            cfgMain[section][key] = value

    # Get original avaDir and create temporary isolated copy
    avaDirOriginal = pathlib.Path(test["AVADIR"])
    avaName = avaDirOriginal.name

    # Create unique temp directory for this test to avoid conflicts when multiple tests share same avaDir
    import os

    testTempDir = tmpTestsDir / f"{avaName}_{test['NAME']}_{os.getpid()}"
    testTempDir.mkdir(parents=True, exist_ok=True)

    # Copy only Inputs directory to temp location (Outputs and Work will be created fresh)
    avaDirInputsOriginal = avaDirOriginal / "Inputs"
    avaDirInputsTemp = testTempDir / "Inputs"
    shutil.copytree(avaDirInputsOriginal, avaDirInputsTemp)

    # Use temp directory as avaDir for this test
    avaDir = str(testTempDir)
    cfgMain["MAIN"]["avalancheDir"] = avaDir

    log.info("=" * 80)
    log.info(f"Starting test: {test['NAME']}")
    log.info("=" * 80)

    # Fetch benchmark test info
    benchDict = test
    simNameRef = test["simNameRef"]
    refDir = pathlib.Path("..", "benchmarks", test["NAME"])
    simType = benchDict["simType"]
    rel = benchDict["Simulation Parameters"]["Release Area Scenario"]

    # Load input parameters from configuration file for standard tests
    standardCfg = refDir / ("%s_com1DFACfg.ini" % test["AVANAME"])
    modName = "com1DFA"

    # if snowGlide test
    if "snowglide" in test["NAME"].lower():
        snowSlideCfgFile = refDir / ("%s_com5SnowGlideCfg.ini" % test["AVANAME"])
        # load snow slide tool config
        snowSlideCfg = cfgUtils.getModuleConfig(com5SnowSlide, fileOverride=snowSlideCfgFile)
        # ++++++++++ set configurations for com1DFA and override ++++++++++++
        # get comDFA configuration and update with snow slide parameter set
        standardCfg = cfgUtils.getModuleConfig(
            com1DFA,
            fileOverride="",
            modInfo=False,
            toPrint=False,
            onlyDefault=snowSlideCfg["com1DFA_com1DFA_override"].getboolean("defaultConfig"),
        )
        standardCfg, snowSlideCfg = cfgHandling.applyCfgOverride(
            standardCfg, snowSlideCfg, com1DFA, addModValues=False
        )

    # Set timing
    startTime = time.time()
    # call com1DFA run
    dem, plotDict, reportDictList, simDF = com1DFA.com1DFAMain(cfgMain, cfgInfo=standardCfg)
    endTime = time.time()
    timeNeeded = endTime - startTime
    log.info(("Took %s seconds to calculate." % (timeNeeded)))

    # Fetch correct reportDict according to simType and release area scenario
    # read all simulation configuration files and return dataFrame and write to csv
    parametersDict = {"simTypeActual": simType, "releaseScenario": rel}
    simNameComp = cfgHandling.filterSims(avaDir, parametersDict)
    if len(simNameComp) > 1:
        log.error("more than one matching simulation found for criteria! ")
    else:
        simNameComp = simNameComp[0]

    # find report dictionary corresponding to comparison simulation
    reportD = {}
    for dict in reportDictList:
        if simNameComp in dict["simName"]["name"]:
            reportD = dict
    if reportD == {}:
        message = "No matching simulation found for reference simulation: %s" % simNameRef
        log.error(message)
        raise ValueError(message)
    log.info("Reference simulation %s and comparison simulation %s " % (simNameRef, simNameComp))

    # set result files directory
    compDir = pathlib.Path(avaDir, "Outputs", modName, "peakFiles")

    # Add info on run time
    reportD["runTime"] = timeNeeded

    # +++++++Aimec analysis
    # load configuration
    aimecCfg = refDir / ("%s_AIMECCfg.ini" % test["AVANAME"])
    if aimecCfg.is_file():
        cfgAimec = cfgUtils.getModuleConfig(ana3AIMEC, aimecCfg)
    else:
        cfgAimec = cfgUtils.getDefaultModuleConfig(ana3AIMEC)

    cfgAimec["AIMECSETUP"]["diffLim"] = aimecDiffLim
    cfgAimec["AIMECSETUP"]["contourLevels"] = aimecContourLevels
    cfgAimec["FLAGS"]["flagMass"] = aimecFlagMass
    cfgAimec["AIMECSETUP"]["comModules"] = aimecComModules
    cfgAimec["AIMECSETUP"]["testName"] = test["NAME"]

    # Setup input from com1DFA and reference
    pathDict = []
    inputsDF, pathDict = dfa2Aimec.dfaBench2Aimec(avaDir, cfgAimec, simNameRef, simNameComp)
    log.info("reference file comes from: %s" % pathDict["refSimName"])

    # Extract input file locations
    pathDict = aimecTools.readAIMECinputs(
        avaDir,
        pathDict,
        cfgAimec["AIMECSETUP"].getboolean("defineRunoutArea"),
        dirName=reportD["simName"]["name"],
    )

    # perform analysis
    rasterTransfo, resAnalysisDF, aimecPlotDict, _ = ana3AIMEC.mainAIMEC(pathDict, inputsDF, cfgAimec)

    # add aimec results to report dictionary
    reportD, benchDict = ana3AIMEC.aimecRes2ReportDict(resAnalysisDF, reportD, benchDict, pathDict)
    # +++++++++++Aimec analysis

    # Create plots for report
    # Load input parameters from configuration file
    cfgRep = cfgUtils.getModuleConfig(generateCompareReport)

    plotListRep = {}
    reportD["Simulation Difference"] = {}
    reportD["Simulation Stats"] = {}

    # Plot data comparison for all output variables defined in outputVariable
    for var in outputVariable:
        plotDict = outQuickPlot.quickPlotBench(
            avaDir, simNameRef, simNameComp, refDir, compDir, cfgMain, var
        )
        for plot in plotDict["plots"]:
            plotListRep.update({var: plot})
            reportD["Simulation Difference"].update({var: plotDict["difference"]})
            reportD["Simulation Stats"].update({var: plotDict["stats"]})

    # copy files to report directory
    plotPaths = generateCompareReport.copyQuickPlots(avaName, test["NAME"], outDir, plotListRep)
    aimecPlots = [aimecPlotDict["slCompPlot"], aimecPlotDict["areasPlot"]]
    plotPaths = generateCompareReport.copyAimecPlots(aimecPlots, test["NAME"], outDir, plotPaths)

    # add plot info to general report Dict
    reportD["Simulation Results"] = plotPaths

    # Return all data needed for report writing
    return (test["NAME"], reportD, benchDict, avaName, cfgRep)


# +++++++++REQUIRED+++++++++++++
# Which result types for comparison plots
outputVariable = ["ppr", "pft", "pfv"]
# aimec settings that are not used from default aimecCfg or aimecCfg in benchmark folders
aimecDiffLim = "5"
aimecContourLevels = "1|3|5|10"
aimecFlagMass = "False"
aimecComModules = "benchmarkReference|com1DFA"
# ++++++++++++++++++++++++++++++

# log file name; leave empty to use default runLog.log
logName = "runStandardTestsCom1DFA"

# Load settings from general configuration file
cfgMain = cfgUtils.getGeneralConfig()

# load all benchmark info as dictionaries from description files
testDictList = tU.readAllBenchmarkDesDicts(info=False)

# filter benchmarks for tag standardTest
# filterType = "TAGS"
# valuesList = ["resistance"]
filterType = "TAGS"
valuesList = ["standardTest", "standardTestSnowGlide"]
# filterType = "NAME"
# valuesList = ["avaInclinedPlaneEntresTest"]

testList = tU.filterBenchmarks(testDictList, filterType, valuesList, condition="or")

# Clean temporary test directory used for parallel execution
tmpTestsDir = pathlib.Path(__file__).parent / "data" / "tmpStdTests"
if tmpTestsDir.exists():
    shutil.rmtree(tmpTestsDir)
tmpTestsDir.mkdir(parents=True, exist_ok=True)

# Set directory for full standard test report
outDir = pathlib.Path.cwd() / "tests" / "reportsCom1DFA"
fU.makeADir(outDir)

# Start writing markdown style report for standard tests
reportFile = outDir / "standardTestsReportCom1DFA.md"
with open(reportFile, "w") as pfile:

    # Write header
    pfile.write("# Standard Tests Report \n")
    pfile.write("## Compare com1DFA simulations to benchmark results \n")

log = logUtils.initiateLogger(outDir, logName)
log.info("The following benchmark tests will be fetched ")
for test in testList:
    log.info("%s" % test["NAME"])

# Set up queue-based logging for parallel execution
logQueue = multiprocessing.Manager().Queue()
queueListener = logging.handlers.QueueListener(logQueue, *log.handlers)
queueListener.start()

# Convert cfgMain to dictionary for pickling
cfgMainDict = {section: dict(cfgMain[section]) for section in cfgMain.sections()}

# Get number of CPU cores for parallel execution
nCPU = cfgUtils.getNumberOfProcesses(cfgMain, len(testList))

# Run tests in parallel using ProcessPoolExecutor
results = []
failedTests = []
with ProcessPoolExecutor(max_workers=nCPU) as executor:
    # Submit all tests
    futures = {}
    for test in testList:
        future = executor.submit(
            runSingleTest,
            test,
            cfgMainDict,
            outputVariable,
            aimecDiffLim,
            aimecContourLevels,
            aimecFlagMass,
            aimecComModules,
            outDir,
            tmpTestsDir,
            logQueue,
        )
        futures[future] = test["NAME"]

    # Collect results as they complete
    for future in as_completed(futures):
        testName = futures[future]
        try:
            result = future.result()
            results.append(result)
            log.info("Completed test: %s" % testName)
        except Exception as exc:
            log.error("Test %s generated an exception: %s" % (testName, exc))
            failedTests.append({"name": testName, "error": str(exc)})

# Stop the queue listener
queueListener.stop()

# Write reports sequentially in completion order
for testName, reportD, benchDict, avaName, cfgRep in results:
    generateCompareReport.writeCompareReport(reportFile, reportD, benchDict, avaName, cfgRep)

# Collect CPU time data from results
cpuTimeName = []
cpuTimeBench = []
cpuTimeSim = []
for testName, reportD, benchDict, avaName, cfgRep in results:
    cpuTimeName.append(testName)
    cpuTimeBench.append(benchDict["Simulation Parameters"]["Computation time [s]"])
    cpuTimeSim.append(reportD["Simulation Parameters"]["Computation time [s]"])

# Display CPU time in log if we have any successful results
if results:
    # Get version info from first successful test
    _, reportD, benchDict, _, _ = results[0]
    log.info(
        "CPU performance comparison between benchmark results (version : %s) and curent branch (version : %s)"
        % (
            benchDict["Simulation Parameters"]["Program version"],
            reportD["Simulation Parameters"]["Program version"],
        )
    )
    log.info(("{:<30}" * 3).format("Test Name", "cpu time Benchmark [s]", "cpu time curent version [s]"))
    for name, cpuBench, cpuSim in zip(cpuTimeName, cpuTimeBench, cpuTimeSim):
        log.info(("{:<30s}" * 3).format(name, cpuBench, cpuSim))

# Display summary of test results
log.info("=" * 80)
log.info("TEST SUMMARY")
log.info("=" * 80)
log.info("Total tests: %d" % len(testList))
log.info("Successful: %d" % len(results))
log.info("Failed: %d" % len(failedTests))

if failedTests:
    log.info("Failed tests:")
    for failure in failedTests:
        log.info("  - %s: %s" % (failure["name"], failure["error"]))
else:
    log.info("All tests completed successfully!")
