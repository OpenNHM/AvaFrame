"""
    Pytest for similarity solution test
    This file is part of Avaframe.
 """

#  Load modules
import pathlib
import shutil

# Local imports
from avaframe.in3Utils import cfgUtils
from avaframe.in3Utils import fileHandlerUtils as fU
import avaframe.com1DFA.com1DFA as com1DFA
from avaframe.in1Data import getInput as gI
import avaframe.ana1Tests.simiSolTest as simiSolTest
import avaframe.out3Plot.outAna1Plots as outAna1Plots


def test_mainCompareSimSolCom1DFA(tmp_path):
    dirname = pathlib.Path(__file__).parents[0]
    simiSolCfg = dirname / ".." / "data" / "avaSimilaritySol" / "Inputs" / "simiSol_com1DFACfg.ini"
    sourceDir = dirname / ".." / "data" / "avaSimilaritySol" / "Inputs"
    destDir = tmp_path / "avaSimilaritySol" / "Inputs"
    avalancheDir = tmp_path / "avaSimilaritySol"

    shutil.copytree(sourceDir, destDir)

    outDirTest = avalancheDir / "Outputs" / "ana1Tests"
    fU.makeADir(outDirTest)

    cfgMain = cfgUtils.getGeneralConfig()
    cfgMain['MAIN']['avalancheDir'] = str(avalancheDir)
    cfg = cfgUtils.getModuleConfig(com1DFA, fileOverride=simiSolCfg)
    # adjust settings for faster computation times
    cfg["GENERAL"]["cMax"] = "0.04"
    cfg["GENERAL"]["sphKernelRadius"] = "8"
    cfg["GENERAL"]["nPPK0"] = "15|20"
    cfg["GENERAL"]["aPPK"] = "-0.5|-1"
    cfg["GENERAL"]["meshCellSize"] = "8"

    # write updated cfg to file (needed for mainSimilaritySol later)
    cfgFile = pathlib.Path(destDir, "%s.ini" % ("simiSolUpdated_com1DFACfg"))
    with open(cfgFile, "w") as conf:
        cfg.write(conf)

    # Define release thickness distribution
    demFile = gI.getDEMPath(avalancheDir)
    relDict = simiSolTest.getReleaseThickness(avalancheDir, cfg, demFile)
    # call com1DFA to perform simulations
    _, _, _, simDF = com1DFA.com1DFAMain(cfgMain, cfgInfo=cfg)

    simDF, _ = cfgUtils.readAllConfigurationInfo(avalancheDir)
    solSimi = simiSolTest.mainSimilaritySol(cfgFile)

    simDF = simiSolTest.postProcessSimiSol(
        avalancheDir, cfgMain, cfg, simDF, solSimi, outDirTest
    )

    # make convergence plot
    fig1, ax1 = outAna1Plots.plotErrorConvergence(
        simDF,
        outDirTest,
        cfg["SIMISOL"],
        "nPart",
        "hErrorL2",
        "aPPK",
        "cMax",
        logScale=True,
    )

    outAna1Plots.plotTimeCPULog(
        simDF, outDirTest, cfg["SIMISOL"], "nPart", "aPPK", "nPPK0"
    )

    simDF = simDF[simDF["nPPK0"] == 15]
    fig1, ax1 = outAna1Plots.plotPresentation(
        simDF,
        outDirTest,
        cfg["SIMISOL"],
        "nPart",
        "hErrorL2",
        "aPPK",
        "nPPK0",
        logScale=True,
        fit=True,
    )

    simDF = simDF[simDF["aPPK"] == -0.5]
    cfg["SIMISOL"]["plotErrorTime"] = "True"
    cfg["SIMISOL"]["plotSequence"] = "True"
    cfg["SIMISOL"]["onlyLast"] = "False"

    simDF = simiSolTest.postProcessSimiSol(
        avalancheDir, cfgMain, cfg, simDF, solSimi, outDirTest
    )
