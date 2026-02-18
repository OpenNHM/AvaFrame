"""
Run script for computing statistics of results including the runout derived from aimec
"""

# Load modules
import pathlib


# Local imports
from avaframe.out3Plot import statsPlots as sPlot
from avaframe.ana4Stats import getStats
from avaframe.in3Utils import fileHandlerUtils as fU
from avaframe.in3Utils import cfgUtils
from avaframe.in3Utils import logUtils


# log file name; leave empty to use default runLog.log
logName = "runGetStats"

# ++++++Set comModule name+++++++++
modName = "com1DFA"

# Load general configuration filee
cfgMain = cfgUtils.getGeneralConfig()
flagShow = cfgMain["FLAGS"].getboolean("showPlot")

avaDir = cfgMain["MAIN"]["avalancheDir"]
cfgStats = cfgUtils.getModuleConfig(getStats, avaDir)
cfg = cfgStats["GENERAL"]

# set output directory, first ava in list
avaDir = pathlib.Path(avaDir)
outDir = avaDir / "Outputs" / "ana4Stats"
cfgStats["GENERAL"]["outDir"] = str(outDir)
# Specify where you want the results to be stored
fU.makeADir(outDir)

# Start logging
log = logUtils.initiateLogger(outDir, logName)

# ----- determine max values of peak fields
# set directory of peak files
inputDir = avaDir / "Outputs" / modName / "peakFiles"

# provide optional filter criteria for simulations
parametersDict = fU.getFilterDict(cfgStats, "FILTER")

# get statistical measure of simulations
peakValues, hasMultiLayer = getStats.extractMaxValues(
    inputDir,
    avaDir,
    cfgStats["GENERAL"]["varPar"],
    restrictType=cfgStats["GENERAL"]["restrictType"],
    nameScenario=cfgStats["GENERAL"]["nameScenario"],
    parametersDict=parametersDict,
    layer=cfgStats["GENERAL"].get("layer", ""),
    modName=modName,
)


# ++++++++++++++ Plot max values +++++++++++++++++
# create resType + layer for fetching the results if multilayer
if hasMultiLayer:
    resType1 = cfgStats["GENERAL"]["resType1"] + "_" + cfgStats["GENERAL"]["layer"].lower()
    resType2 = cfgStats["GENERAL"]["resType2"] + "_" + cfgStats["GENERAL"]["layer"].lower()
else:
    resType1 = cfgStats["GENERAL"]["resType1"]
    resType2 = cfgStats["GENERAL"]["resType2"]

sPlot.plotValuesScatter(
    peakValues,
    resType1,
    resType2,
    cfgStats["GENERAL"],
    avaDir,
    statsMeasure="max",
    flagShow=flagShow,
    layer=cfgStats["GENERAL"]["layer"],
)
sPlot.plotValuesScatterHist(
    peakValues,
    resType1,
    resType2,
    cfgStats["GENERAL"],
    avaDir,
    statsMeasure="max",
    flagShow=flagShow,
    flagHue=True,
    layer=cfgStats["GENERAL"]["layer"],
)

log.info("Plots have been saved to: %s" % outDir)
