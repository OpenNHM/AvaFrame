"""
    Run script for making a contour line plot
"""
# Load modules
# importing general python modules
import pathlib

# Local imports
from avaframe.in3Utils import cfgUtils
from avaframe.out3Plot import outQuickPlot as oQ
import avaframe.runScripts.runPlotContoursFromAsc as rCon
from avaframe.in3Utils import cfgHandling


def plotContoursFromAsc(cfg, avalancheDir):

    # call contour line plot
    oQ.plotAllContours(
        avaDir,
        "com1DFA",
        cfg["GENERAL"]["resType"],
        cfg["GENERAL"]["level"],
        specDir=cfg["GENERAL"]["specDir"],
    )


if __name__ == "__main__":
    # Load configuration for runPlotContour
    cfg = cfgUtils.getModuleConfig(rCon, '', fileOverride="", modInfo=False, toPrint=False, onlyDefault=False)

    # fetch input directory
    cfgMain = cfgUtils.getGeneralConfig()
    avaDir = cfgMain["MAIN"]["avalancheDir"]

    cfgHandling.rewriteLocalCfgs(cfg, avaDir)
    plotContoursFromAsc(cfg, avaDir)

