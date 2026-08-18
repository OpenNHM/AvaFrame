"""
    Run script for running the standard tests with com4FlowPy
    in this test all the available tests tagged standardTest are performed
"""

# Load modules
import time
import pathlib
import numpy as np
from datetime import datetime
import tempfile
import os

# Local imports
import avaframe as avaf
from avaframe.com4FlowPy import com4FlowPy
from avaframe.runCom4FlowPy import readFlowPyinputs
from avaframe.ana1Tests import testUtilities as tU
from avaframe.in3Utils import fileHandlerUtils as fU
from avaframe.in3Utils import initializeProject as initProj
from avaframe.in3Utils import cfgUtils
from avaframe.in3Utils import logUtils
import avaframe.in2Trans.rasterUtils as rasterUtils

def _checkNumbaInstalled() -> bool:
    try:
        import numba
        return True
    except ImportError:
        return False


def compareRasters(path, pathRef):
    """
    compare two rasters and compute the difference between them

    Parameters
    ----------
    path: string or pathlib.Path
        path to raster file
    pathRef: string or pathlib.Path
        path to reference raster file

    Returns
    -------
    diff: np.array
        difference of the rasters in every rastercell
    equal: boolean
        True if the rasters are equal
    closePercentage: float
        the proportion of cells that match closely between both rasters, out of all cells that were actually processed
    """
    rasterDict = rasterUtils.readRaster(path, noDataToNan=False)
    raster = rasterDict["rasterData"]
    rasterRefDict = rasterUtils.readRaster(pathRef, noDataToNan=False)
    rasterRef = rasterRefDict["rasterData"]
    # difference of both rasters
    diff = rasterRef - raster

    equal = np.array_equal(rasterRef, raster)

    closeArray = np.isclose(raster, rasterRef, rtol=1e-04, equal_nan=True)
    mask = np.logical_or(raster > 0, rasterRef > 0)
    num_close = np.count_nonzero(closeArray[mask])
    total = rasterRef[mask].size
    closePercent = num_close / total

    return diff, equal, closePercent

def main():

    # avaframe directory
    _avaframeDir  = pathlib.Path(avaf.__file__).parents[0]
    _benchmarkDir = pathlib.Path(_avaframeDir, '..', 'benchmarks')

    # Which result types for comparison plots
    # outputVariable = ['fpTravelAngleMax', 'zDelta', 'flux', 'cellCounts']

    # log file name; leave empty to use default runLog.log
    logName = 'runStandardTestsCom4FlowPy'

    # Load settings from general configuration file
    cfgMain = cfgUtils.getGeneralConfig()

    # load all benchmark info as dictionaries from description files
    testDictList = tU.readAllBenchmarkDesDicts(info=False, inDir = _benchmarkDir)

    # filter benchmarks for tag standardTest
    filterType = 'TAGS'

    valuesList = ['standardTest', 'com4FlowPy']
    # looking for 'com4FlowPy' and 'standardTest' in TAGS list
    testListAll = tU.filterBenchmarks(testDictList, filterType, valuesList, condition='and')

    valuesList = ['standardTest', 'com4FlowPy', 'numba']
    testListNumba = tU.filterBenchmarks(testDictList, filterType, valuesList, condition='and')

    if _checkNumbaInstalled():
        testList = testListAll
    else:
        testList = [item for item in testListAll if item not in testListNumba]

    # Set directory for full standard test report
    outDir = _avaframeDir / 'tests' / 'reportsCom4FlowPy'
    fU.makeADir(outDir)

    # Start writing markdown style report for standard tests
    reportFile = outDir / 'standardTestsReportCom4FlowPy.md'

    _startDate = datetime.now()
    with open(reportFile, 'w') as pfile:
        # Write header
        pfile.write('# Standard Tests Report \n\n')
        pfile.write('Comparing __com4FlowPy__ simulations to selected benchmark results \n\n')
        pfile.write('* * * \n')
        if _checkNumbaInstalled():
            pfile.write('`numba` __found__: running all tests for python and numba engines &check;')
        else:
            pfile.write('`numba` __NOT found__:  skipping tests for numba engine &cross;')
        pfile.write('\n* * * \n')
        
    log = logUtils.initiateLogger(outDir, logName)
    log.info('The following benchmark tests will be fetched ')

    with open(reportFile, 'a') as pfile:
        pfile.write("__tests fetched__:\n")
        for test in testList:
            pfile.write(f"- {test['NAME']}\n")
            log.info('%s' % test['NAME'])
        pfile.write('\n')
        pfile.write(f'__tests started__ : {_startDate}\n\n')
        pfile.write('* * * \n')

    # create a temporary directory, where the outputs of all standard Tests are stored
    # clean-up is automatic - this way we don't pollute the avaframe/data/ directory
    with tempfile.TemporaryDirectory(prefix="avaframe_stdTests_") as tempDir:
        
        tmpTestsDir = pathlib.Path(tempDir)

        # run Standard Tests sequentially
        for i, test in enumerate(testList):

            with open(reportFile, 'a') as pfile:
                pfile.write("\n")
                pfile.write(f"### Test {i+1}: *{test['NAME']}*\n")
                for descLine in str(test['DESCRIPTION']).split('\n'):
                    pfile.write(f"{descLine}{' ' * 2}\n")
                if 'REFERENCE' in test and isinstance(test['REFERENCE'], str):
                    pfile.write(f"__Reference__: {test['REFERENCE']}{' ' * 2}\n")
                
                testAvaFVersion = avaf.version.getVersion()
                pfile.write(f"__tested AvaFrame Version__: {testAvaFVersion}{' ' * 2}\n")

                if 'BENCHMARKED_AVAFRAME_VERSION' in test and isinstance(test['BENCHMARKED_AVAFRAME_VERSION'], str):
                    benchAvaFVersion = test['BENCHMARKED_AVAFRAME_VERSION']
                    if benchAvaFVersion == testAvaFVersion:
                        pfile.write(f"__benchmarked AvaFrame::com4FlowPy version__: {benchAvaFVersion}{' ' * 2}\n")
                    else:
                        _str = "__benchmarked AvaFrame::com4FlowPy version__: "
                        _str += f"<span style=\"color:red\"> {benchAvaFVersion} </span>{' ' * 2}\n"
                        pfile.write(_str)


                pfile.write("\n")
                pfile.write("|Model Output|Result of comparison|status\n")
                pfile.write("|----:|:-----:|:---:|\n")

            # define avaDir relative to _avaframeDir to allow execution of this script from different locations
            # not just AvaFrame/avaframe directory
            avaDir = str( _avaframeDir / pathlib.Path(test['AVADIR']) )
            cfgMain['MAIN']['avalancheDir'] = avaDir

            # Fetch benchmark test info
            refDir = pathlib.Path(_avaframeDir, '..', 'benchmarks', test['NAME'])

            # Clean input directory(ies) of old work and output files
            initProj.cleanSingleAvaDir(_avaframeDir / avaDir, deleteOutput=False)

            # Load input parameters from configuration file for standard tests
            benchmarkCfg = refDir / ('%s' % test['INI'])
            modName = 'com4FlowPy'
            cfg = cfgUtils.getModuleConfig(com4FlowPy, fileOverride=benchmarkCfg)
            cfgGen = cfg["GENERAL"]
            cfgGen["cpuCount"] = str(cfgUtils.getNumberOfProcesses(cfgMain, 9999))
            cfgGen["overwriteResults"] = "reRunAndOverwrite"

            avalancheDir = cfgMain["MAIN"]["avalancheDir"]
            cfgPath = readFlowPyinputs(avalancheDir, cfg, log)

            # for the temporary output folder we cannot use the full path, but just the relative part of the path
            # following the _avaframeDir path
            avaDirTempOutPut = test['AVADIR']
            # compDir = output location of outputs generated for each test within the temporary directory
            compDir = tmpTestsDir / pathlib.Path(avaDirTempOutPut)
            
            cfgPath["customDirs"] = False
            cfgPath["resDir"] = compDir
            fU.makeADir(cfgPath["resDir"])
            cfgPath["thalwegDir"] = cfgPath["resDir"] / "thalwegData"
            cfgPath["tempDir"] = cfgPath["workDir"] / "temp"
            fU.makeADir(cfgPath["tempDir"])
            cfgPath["deleteTemp"] = "False"
            cfgPath["outputFiles"] = cfg["PATHS"]["outputFiles"]
            cfgPath["outputNoDataValue"] = cfg["PATHS"].getfloat("outputNoDataValue")
            cfgPath["useCompression"] = cfg["PATHS"].getboolean("useCompression")
            cfgPath["uid"] = cfgUtils.cfgHash(cfg)
            cfgPath["timeString"] = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Set timing
            startTime = time.time()
            # call com4FlowPy run
            com4FlowPy.com4FlowPyMain(cfgPath, cfgGen)
            endTime = time.time()
            timeNeeded = endTime - startTime
            log.info(('Took %s seconds to calculate.' % (timeNeeded)))

            # files that are compared - must be defined in the .json inside the testDir
            # every test can have its own set of output files / variables that are compared
            outputVariable = test['FILES'] 

            for variable in outputVariable:

                for file in os.listdir(refDir):
                    if file.endswith('%s.tif' % variable):
                        pathRasterRef = refDir / file
                        break
                    else:
                        continue
                if os.path.isfile(pathRasterRef) is False:
                    raise FileExistsError("in %s does not exist a file for variable %s" %(refDir, variable))
                
                for file in os.listdir(compDir):
                    if file.endswith('%s.tif' % variable) and ( str(file).split('_')[1] == cfgPath["uid"] ):
                        pathRaster = compDir / file
                        break
                    else:
                        continue

                if os.path.isfile(pathRaster) is False:
                    raise FileExistsError("in %s does not exist a file for variable %s" %(compDir, variable))
                diff, eq, close = compareRasters(pathRaster, pathRasterRef)

                if eq and np.sum(abs(diff[diff != 0])) == 0:
                    message = f"|__{variable}__| rasters are equal |&check;\n"
                    _logMsg = f"{variable} - rasters are equal"
                else:
                    message = f"|{variable}| rasters are __NOT(!)__ equal - {np.round(close, 4) * 100}%"
                    message += f" of the affected area is close (relative tolerance: 10^-4)|&cross;\n"

                    _logMsg = f"{variable} - rasters are NOT equal - {np.round(close, 4) * 100}% is close"

                log.info(f"{test['NAME']}: {_logMsg}")

                with open(reportFile, 'a') as pfile:
                    pfile.write(message)

            _endDate = datetime.now()
            with open(reportFile, 'a') as pfile:
                pfile.write('\n * * * \n')
        
    with open(reportFile, 'a') as pfile:
        pfile.write(f"__test(s) finished @__: {_endDate}\n")
        pfile.write(f"__timeDelta__: {_endDate-_startDate}")

if __name__ == "__main__":
    main()