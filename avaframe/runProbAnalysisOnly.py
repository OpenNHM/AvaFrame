"""
    Run script for performing a probability analysis with the simulation results found in avalancheDir
    Define settings in ana4Stats/probAnaCfg.ini or your local copy - local_probAnaCfg.ini
"""

import argparse

# Load modules
import pathlib

from avaframe.ana4Stats import probAna

# Local imports
from avaframe.com1DFA import com1DFA
from avaframe.in3Utils import cfgUtils
from avaframe.in3Utils import initializeProject as initProj
from avaframe.in3Utils import logUtils
from avaframe.out3Plot import outQuickPlot as oP
from avaframe.out3Plot import statsPlots as sP


def runProbAna(avalancheDir="", modName=""):
    """Run a com1DFA probability analysis with parameters and only an
    avalanche directory as input

    Parameters
    ----------
    avalancheDir: str
        path to avalanche directory (setup e.g. with init scripts)
    modName: str
        name of computational module that has been used to produce sim results
         - to locate results files and filtering options

    Returns
    -------
    """

    # log file name; leave empty to use default runLog.log
    logName = "runProbAnalysisOnly"

    # Load general configuration file
    cfgMain = cfgUtils.getGeneralConfig()

    # Load avalanche directory from general configuration file,
    # if not provided as input argument
    if avalancheDir != "":
        cfgMain["MAIN"]["avalancheDir"] = avalancheDir
    else:
        avalancheDir = cfgMain["MAIN"]["avalancheDir"]

    avalancheDir = pathlib.Path(avalancheDir)

    # Start logging
    log = logUtils.initiateLogger(avalancheDir, logName)
    log.info("MAIN SCRIPT")
    log.info("Current avalanche: %s", avalancheDir)

    # Clean input directory(ies) of old work files
    initProj.cleanSingleAvaDir(avalancheDir, deleteOutput=False)

    # Load configuration file for probabilistic run and analysis
    cfgProb = cfgUtils.getModuleConfig(probAna, avalancheDir)
    if modName == "":
        modName = cfgProb["GENERAL"]["modName"]

    # perform probability analysis
    anaPerformed, contourDict = probAna.probAnalysis(
        avalancheDir, cfgProb, modName, parametersDict="", probConf="", simDFActual=""
    )
    if anaPerformed is False:
        log.warning("No files found")

    # make a plot of the contours
    inputDir = pathlib.Path(avalancheDir, "Outputs", "ana4Stats")
    outName = "%s_prob_%s_lim%s" % (
        str(avalancheDir.stem),
        cfgProb["GENERAL"]["peakVar"],
        cfgProb["GENERAL"]["peakLim"],
    )
    pathDict = {"pathResult": str(inputDir / "plots"), "avaDir": str(avalancheDir), "plotScenario": outName}
    oP.plotContours(
        contourDict, cfgProb["GENERAL"]["peakVar"], cfgProb["GENERAL"]["peakLim"], pathDict, addLegend=False,
        layer=cfgProb["GENERAL"].get("layer", "")
    )

    # plot probability maps
    sP.plotProbMap(avalancheDir, inputDir, cfgProb, demPlot=True)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run probAna workflow")
    parser.add_argument(
        "avadir", metavar="a", type=str, nargs="?", default="", help="the avalanche directory"
    )

    parser.add_argument(
        "modName", metavar="modN", type=str, nargs="?", default="", help="name of computational module"
    )

    args = parser.parse_args()
    runProbAna(str(args.avadir), str(args.modName))
