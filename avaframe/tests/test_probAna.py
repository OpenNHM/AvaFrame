"""
Pytest for module ana4Stats

This file is part of Avaframe.

"""

#  Load modules
import numpy as np
import os
from avaframe.com1DFA import com1DFA
import avaframe.in3Utils.fileHandlerUtils as fU
from avaframe.in3Utils import cfgUtils
from avaframe.ana4Stats import probAna as pA
import pytest
import configparser
import shutil
import pathlib
from scipy.stats import qmc


def test_probAnalysis(tmp_path):
    """test probAna function to compute mask for parameter exceeding threshold"""

    # set input directory
    avaName = "avaParabola"
    avaTestDir = "avaParabolaStatsTest"
    dirPath = pathlib.Path(__file__).parents[0]
    avaDir = dirPath / ".." / ".." / "benchmarks" / avaTestDir
    avaDirtmp = pathlib.Path(tmp_path, avaName)
    inputDir = pathlib.Path(tmp_path, avaName)
    inputDir1 = avaDir
    shutil.copytree(inputDir1, inputDir)

    # set configurations
    testCfg = os.path.join(inputDir, "%sProbAna_com1DFACfg.ini" % avaName)
    cfgMain = cfgUtils.getModuleConfig(com1DFA, fileOverride=testCfg)

    # Initialise input in correct format
    cfg = configparser.ConfigParser()
    cfg["GENERAL"] = {"peakLim": 1.0, "peakVar": "ppr"}
    cfg["FILTER"] = {}

    # provide optional filter criteria for simulations
    parametersDict = fU.getFilterDict(cfg, "FILTER")

    # call function to test
    pA.probAnalysis(avaDirtmp, cfg, "com1DFA", parametersDict=parametersDict, inputDir="")
    outputPath = os.path.join(avaDirtmp, "Outputs", "ana4Stats", "avaParabola_prob__ppr_lim1.0.asc")
    # print(outputPath)
    probTest = np.loadtxt(outputPath, skiprows=6)

    # Load reference solution
    probSol = np.loadtxt(os.path.join(inputDir1, "avaParabola_prob__ppr_lim1.0.txt"), skiprows=6)

    # Compare result to reference solution
    testRes = np.allclose(probTest, probSol, atol=1.0e-6)

    # Test
    assert testRes is True

    # call function to test
    testInputDir = avaDir / "Outputs" / "com1DFA"
    avaDirtmp2 = pathlib.Path(tmp_path, "avaTest")
    avaDirtmp2.mkdir()
    pA.probAnalysis(avaDirtmp2, cfg, "com1DFA", parametersDict="", inputDir=testInputDir)
    probTest2 = np.loadtxt(
        os.path.join(avaDirtmp2, "Outputs", "ana4Stats", "avaTest_prob__ppr_lim1.0.asc"),
        skiprows=6,
    )

    # Compare result to reference solution
    testRes2 = np.allclose(probTest2, probSol, atol=1.0e-6)

    # Test
    assert testRes2 is True

    # set input directory
    avaName2 = "avaTest2"
    avaTestDir2 = "avaProbTest"
    dirPath2 = pathlib.Path(__file__).parents[0]
    avaDir2 = dirPath / "data" / avaTestDir2
    inputDir2 = pathlib.Path(tmp_path, avaName2)
    shutil.copytree(avaDir2, inputDir2)

    cfg2 = configparser.ConfigParser()
    cfg2.optionxform = str
    cfg2["GENERAL"] = {"peakVar": "ppr", "unit": "kPa", "peakLim": "1.0"}
    cfg2["PLOT"] = {
        "name": "probability map",
        "cmapType": "prob",
        "levels": "0.95",
        "unit": "fraction",
        "zoomBuffer": 250.0,
        "contrainBuffer": 10.0,
        "meshCellSizeThreshold": 0.001,
    }
    modName = "com1DFA"

    # call function to be tested
    analysisPerformed, contourDict = pA.probAnalysis(
        inputDir2,
        cfg2,
        modName,
        parametersDict="",
        inputDir="",
        probConf="",
        simDFActual="",
    )

    assert analysisPerformed is True


def test_createComModConfig(tmp_path):
    """test creatig a config file"""

    # set input directory
    avaName = "testCom1DFA2"
    dirPath = pathlib.Path(__file__).parents[0]
    inputDir = dirPath / "data" / avaName
    avaDir = pathlib.Path(tmp_path, avaName)
    shutil.copytree(inputDir, avaDir)

    cfgProb = configparser.ConfigParser()
    cfgProb.optionxform = str
    cfgProb["PROBRUN"] = {
        "varParList": "musamosat|relTh",
        "variationType": "percent|percent",
        "variationValue": "60|50",
        "numberOfSteps": "2|3",
        "defaultSetup": "True",
        "samplingStrategy": "2",
        "varParType": "float|float",
    }
    cfgProb["sampling_override"] = {"defaultConfig": "True"}
    cfgProb["com1DFA_com1DFA_override"] = {
        "defaultConfig": "True",
        "frictModel": "samosAT",
    }
    # call function to be tested
    cfgFiles, outDir = pA.createComModConfig(cfgProb, avaDir, com1DFA)

    #    print('cfgFiles', cfgFiles)

    # load cfg from file
    cfgMu = configparser.ConfigParser()
    cfgRelTh = configparser.ConfigParser()

    cfgMu.read(cfgFiles[0])
    cfgRelTh.read(cfgFiles[1])

    #    print(cfgMu['GENERAL']['musamosat'], cfgMu['GENERAL']['relTh'], cfgRelTh['GENERAL']['musamosat'],
    #           cfgRelTh['GENERAL']['relTh'])

    assert cfgMu["GENERAL"]["musamosat"] == "0.155$60$2"
    assert cfgMu["GENERAL"]["relTh"] == ""
    assert cfgRelTh["GENERAL"]["musamosat"] == "0.155"
    assert cfgRelTh["GENERAL"]["relTh"] == ""
    assert cfgRelTh["GENERAL"]["relThFromFile"] == "True"
    assert cfgRelTh["GENERAL"]["relThPercentVariation"] == "50$3"
    assert cfgRelTh["GENERAL"]["relThFromFile"] == "True"
    assert cfgMu["GENERAL"]["relThFromFile"] == "True"
    assert cfgRelTh.has_option("GENERAL", "addStandardConfig") is False

    cfgProb["com1DFA_com1DFA_override"] = {
        "defaultConfig": "True",
        "relThFromFile": False,
        "relTh": 2.0,
        "musamosat": 0.155,
        "frictModel": "samosAT",
    }
    # call function to be tested
    cfgFiles, outDir = pA.createComModConfig(cfgProb, avaDir, com1DFA)

    # load cfg from file
    cfgMu = configparser.ConfigParser()
    cfgRelTh = configparser.ConfigParser()

    cfgMu.read(cfgFiles[0])
    cfgRelTh.read(cfgFiles[1])

    #    print(cfgMu['GENERAL']['musamosat'], cfgMu['GENERAL']['relTh'], cfgRelTh['GENERAL']['musamosat'],
    #           cfgRelTh['GENERAL']['relTh'])

    assert cfgMu["GENERAL"]["musamosat"] == "0.155$60$2"
    assert np.isclose(cfgMu["GENERAL"].getfloat("relTh"), 2.0)
    assert cfgRelTh["GENERAL"]["musamosat"] == "0.155"
    assert np.isclose(cfgRelTh["GENERAL"].getfloat("relTh"), 2.0)
    assert cfgRelTh["GENERAL"]["relThFromFile"] == "False"
    assert cfgRelTh["GENERAL"]["relThPercentVariation"] == "50$3"
    assert cfgRelTh["GENERAL"]["relThFromFile"] == "False"
    assert cfgMu["GENERAL"]["relThFromFile"] == "False"

    cfgProb = configparser.ConfigParser()
    cfgProb.optionxform = str
    cfgProb["PROBRUN"] = {
        "varParList": "musamosat|relTh",
        "variationType": "percent|percent",
        "variationValue": "60|50",
        "numberOfSteps": "2|3",
        "defaultSetup": "True",
        "samplingStrategy": "1",
        "varParType": "float|float",
        "nSample": "40",
        "sampleSeed": "12345",
        "sampleMethod": "latin",
    }
    cfgProb["com1DFA_com1DFA_override"] = {
        "defaultConfig": "True",
        "relThFromFile": False,
        "relTh": 2.0,
        "musamosat": 0.155,
        "frictModel": "samosAT",
    }

    # call function to be tested
    cfgFiles, outDir = pA.createComModConfig(cfgProb, avaDir, com1DFA)

    #    print('cfgFiles', cfgFiles)

    for cfgF in cfgFiles:
        cfgTest = configparser.ConfigParser()
        cfgTest.read(cfgF)
        #        print('cfgTest', cfgTest['GENERAL']['relThFromFile'], cfgTest['GENERAL']['relTh'],
        #             cfgTest['GENERAL']['relThPercentVariation'], cfgTest['GENERAL']['musamosat'])

        assert cfgTest["GENERAL"]["relThFromFile"] == "False"
        assert cfgTest["GENERAL"].getfloat("relTh") <= 3.0
        assert cfgTest["GENERAL"].getfloat("relTh") >= 1.0
        assert cfgTest["GENERAL"].getfloat("musamosat") <= 0.248
        assert cfgTest["GENERAL"].getfloat("musamosat") >= 0.062

    cfgTest = configparser.ConfigParser()
    cfgTest.read(cfgFiles[0])
    assert cfgTest["GENERAL"]["relTh"] == "2.2719559079879"
    assert len(cfgFiles) == 40

    cfgProb = configparser.ConfigParser()
    cfgProb.optionxform = str
    cfgProb["PROBRUN"] = {
        "varParList": "musamosat|relTh",
        "variationType": "percent|percent",
        "variationValue": "60|50",
        "numberOfSteps": "2|3",
        "defaultSetup": "True",
        "samplingStrategy": "1",
        "varParType": "float|float",
        "nSample": "40",
        "sampleSeed": "12345",
        "sampleMethod": "latin",
    }
    cfgProb["com1DFA_com1DFA_override"] = {
        "defaultConfig": "True",
        "frictModel": "samosAT",
    }

    # call function to be tested
    cfgFiles2, outDir = pA.createComModConfig(cfgProb, avaDir, com1DFA)

    #    print('cfgFiles', cfgFiles2)

    cfgTest2 = configparser.ConfigParser()
    cfgTest2.read(cfgFiles2[0])
    cfgTest21 = configparser.ConfigParser()
    cfgTest21.read(cfgFiles2[40])
    #    print('cfgTest', cfgTest['GENERAL']['relThFromFile'], cfgTest['GENERAL']['relTh'],
    #         cfgTest['GENERAL']['relThPercentVariation'], cfgTest['GENERAL']['musamosat'])

    assert cfgTest2["GENERAL"]["relThFromFile"] == "True"
    assert cfgTest2["GENERAL"]["relTh"] == ""
    assert cfgTest2["GENERAL"]["relThPercentVariation"] == ""
    assert cfgTest2["INPUT"]["releaseScenario"] == "relParabola"
    assert cfgTest2.has_option("GENERAL", "relTh0")
    assert len(cfgFiles2) == 80
    assert cfgTest21["GENERAL"]["relThFromFile"] == "True"
    assert cfgTest21["GENERAL"]["relTh"] == ""
    assert cfgTest21["GENERAL"]["relThPercentVariation"] == ""
    assert cfgTest21["GENERAL"]["musamosat"] == cfgTest2["GENERAL"]["musamosat"]
    assert cfgTest21["INPUT"]["releaseScenario"] == "relParabolaTwo"
    assert cfgTest21.has_option("GENERAL", "relTh0")
    assert len(cfgFiles2) == 80

    cfgProb = configparser.ConfigParser()
    cfgProb.optionxform = str
    cfgProb["PROBRUN"] = {
        "varParList": "musamosat|relTh",
        "variationType": "range|range",
        "variationValue": "0.2|1.2",
        "numberOfSteps": "2|3",
        "samplingStrategy": "2",
        "varParType": "float|float|int",
    }
    cfgProb["com1DFA_com1DFA_override"] = {
        "defaultConfig": "True",
        "frictModel": "samosAT",
    }

    # call function to be tested
    cfgFiles3, outDir = pA.createComModConfig(cfgProb, avaDir, com1DFA)

    #    print('cfgFiles', cfgFiles3)

    cfgTest3 = configparser.ConfigParser()
    cfgTest3.read(cfgFiles3[1])

    assert cfgTest3["GENERAL"]["relThFromFile"] == "True"
    assert cfgTest3["GENERAL"]["relTh"] == ""
    assert cfgTest3["GENERAL"]["relThRangeVariation"] == "1.2$3"
    assert len(cfgFiles3) == 2

    cfgProb = configparser.ConfigParser()
    cfgProb.optionxform = str
    cfgProb["PROBRUN"] = {
        "varParList": "musamosat|relTh|entTh",
        "variationType": "percent|rangefromci|range",
        "variationValue": "60|ci95|0.15",
        "numberOfSteps": "2|3|4",
        "samplingStrategy": "1",
        "varParType": "float|float|float",
        "nSample": "30",
        "sampleSeed": "12345",
        "sampleMethod": "latin",
    }
    cfgProb["com1DFA_com1DFA_override"] = {
        "defaultConfig": "True",
        "frictModel": "samosAT",
    }

    # call function to be tested
    cfgFiles4, outDir = pA.createComModConfig(cfgProb, avaDir, com1DFA)

    #    print('cfgFiles', cfgFiles)

    cfgTest4 = configparser.ConfigParser()
    cfgTest4.read(cfgFiles4[0])
    #    print('cfgTest', cfgTest4['GENERAL']['relThFromFile'], cfgTest4['GENERAL']['relTh'],
    #         cfgTest4['GENERAL']['relThPercentVariation'], cfgTest4['GENERAL']['musamosat'])

    assert cfgTest4["GENERAL"]["relThFromFile"] == "True"
    assert cfgTest4["GENERAL"]["relTh"] == ""
    assert cfgTest4["GENERAL"]["relThRangeVariation"] == ""
    assert cfgTest4["GENERAL"]["relTh0"] != ""
    assert cfgTest4["GENERAL"]["relTh1"] != ""
    assert cfgTest4["GENERAL"]["entTh0"] != ""
    assert cfgTest4["INPUT"]["releaseScenario"] == "relParabola"
    assert len(cfgFiles4) == 60


def test_updateCfgRange():
    """test updating cfg values"""

    # setup inputs
    cfg = configparser.ConfigParser()
    cfg["PROBRUN"] = {
        "varParList": "musamosat|relTh",
        "variationType": "percent|percent",
        "variationValue": "60|50",
        "numberOfSteps": "2|2",
        "defaultSetup": "True",
        "varParType": "float",
        "samplingStrategy": "2",
    }

    com1DFACfg = cfgUtils.getDefaultModuleConfig(com1DFA)
    varName = "musamosat"

    varDict = pA.makeDictFromVars(cfg["PROBRUN"])
    # call function
    cfgNew = pA.updateCfgRange(com1DFACfg, cfg, varName, varDict[varName])

    assert cfgNew["GENERAL"]["musamosat"] == "0.155$60$2"
    assert cfgNew["GENERAL"]["relTh"] == ""
    assert cfgNew["GENERAL"]["relThFromFile"] == "True"
    assert cfgNew["GENERAL"]["relThPercentVariation"] == ""

    com1DFACfg = cfgUtils.getDefaultModuleConfig(com1DFA)
    varName = "relTh"

    # call function
    cfgNew = pA.updateCfgRange(com1DFACfg, cfg, varName, varDict[varName])

    assert cfgNew["GENERAL"]["musamosat"] == "0.155"
    assert cfgNew["GENERAL"]["relTh"] == ""
    assert cfgNew["GENERAL"]["relThFromFile"] == "True"
    assert cfgNew["GENERAL"]["relThPercentVariation"] == "50$2"

    com1DFACfg = cfgUtils.getDefaultModuleConfig(com1DFA)
    varName = "musamosat"
    varDict = {}
    varDict = pA.makeDictFromVars(cfg["PROBRUN"])

    # call function
    cfgNew = pA.updateCfgRange(com1DFACfg, cfg, varName, varDict[varName])

    assert cfgNew["GENERAL"]["musamosat"] == "0.155$60$2"
    assert cfgNew["GENERAL"]["relTh"] == ""
    assert cfgNew["GENERAL"]["relThFromFile"] == "True"
    assert cfgNew["GENERAL"]["relThPercentVariation"] == ""

    com1DFACfg = cfgUtils.getDefaultModuleConfig(com1DFA)
    varName = "relTh"

    # call function
    cfgNew = pA.updateCfgRange(com1DFACfg, cfg, varName, varDict[varName])

    assert cfgNew["GENERAL"]["musamosat"] == "0.155"
    assert cfgNew["GENERAL"]["relTh"] == ""
    assert cfgNew["GENERAL"]["relThFromFile"] == "True"
    assert cfgNew["GENERAL"]["relThPercentVariation"] == "50$2"

    # setup inputs
    cfg = configparser.ConfigParser()
    cfg["PROBRUN"] = {
        "varParList": "musamosat|relTh",
        "variationType": "range|range",
        "variationValue": "0.1|0.5",
        "numberOfSteps": "5|12",
        "varParType": "float|float",
        "defaultSetup": "True",
        "samplingStrategy": "2",
    }

    com1DFACfg = cfgUtils.getDefaultModuleConfig(com1DFA)
    varName = "musamosat"

    varDict = pA.makeDictFromVars(cfg["PROBRUN"])
    # call function
    cfgNew = pA.updateCfgRange(com1DFACfg, cfg, varName, varDict[varName])

    assert np.isclose(float(cfgNew["GENERAL"]["musamosat"].split(":")[0]), 0.055, rtol=1.0e-5)
    assert np.isclose(float(cfgNew["GENERAL"]["musamosat"].split(":")[1]), 0.255, rtol=1.0e-5)
    assert cfgNew["GENERAL"]["musamosat"].split(":")[2] == "5"
    assert cfgNew["GENERAL"]["relTh"] == ""
    assert cfgNew["GENERAL"]["relThFromFile"] == "True"
    assert cfgNew["GENERAL"]["relThPercentVariation"] == ""

    com1DFACfg = cfgUtils.getDefaultModuleConfig(com1DFA)
    varName = "relTh"
    # call function
    cfgNew = pA.updateCfgRange(com1DFACfg, cfg, varName, varDict[varName])

    assert cfgNew["GENERAL"]["musamosat"] == "0.155"
    assert cfgNew["GENERAL"]["relTh"] == ""
    assert cfgNew["GENERAL"]["relThFromFile"] == "True"
    assert cfgNew["GENERAL"]["relThRangeVariation"] == "0.5$12"

    # setup inputs
    cfg = configparser.ConfigParser()
    cfg.optionxform = str
    cfg["PROBRUN"] = {
        "varParList": "musamosat|relTh",
        "variationType": "normaldistribution|normaldistribution",
        "variationValue": "0.1|0.3",
        "numberOfSteps": "3|12",
        "defaultSetup": "True",
        "varParType": "float|float",
        "samplingStrategy": "2",
    }
    cfg["in1Data_computeFromDistribution_override"] = {
        "buildType": "ci95",
        "minMaxInterval": "95",
        "defaultConfig": "True",
    }

    com1DFACfg = cfgUtils.getDefaultModuleConfig(com1DFA)
    varName = "musamosat"

    varDict = pA.makeDictFromVars(cfg["PROBRUN"])
    # call function
    cfgNew = pA.updateCfgRange(com1DFACfg, cfg, varName, varDict[varName])

    assert np.isclose(float(cfgNew["GENERAL"]["musamosat"].split("|")[0]), 0.055, rtol=1.0e-3)
    assert np.isclose(float(cfgNew["GENERAL"]["musamosat"].split("|")[1]), 0.155, rtol=1.0e-3)
    assert np.isclose(float(cfgNew["GENERAL"]["musamosat"].split("|")[2]), 0.255, rtol=1.0e-3)
    assert cfgNew["GENERAL"]["relTh"] == ""
    assert cfgNew["GENERAL"]["relThFromFile"] == "True"
    assert cfgNew["GENERAL"]["relThDistVariation"] == ""

    com1DFACfg = cfgUtils.getDefaultModuleConfig(com1DFA)
    varName = "relTh"

    varDict = pA.makeDictFromVars(cfg["PROBRUN"])
    # call function
    cfgNew = pA.updateCfgRange(com1DFACfg, cfg, varName, varDict[varName])

    #    print('value', cfgNew['GENERAL']['relThPercentVariation'])

    assert cfgNew["GENERAL"]["musamosat"] == "0.155"
    assert cfgNew["GENERAL"]["relTh"] == ""
    assert cfgNew["GENERAL"]["relThFromFile"] == "True"
    assert cfgNew["GENERAL"]["relThDistVariation"] == "normaldistribution$12$0.3$95$ci95$10000"

    # setup inputs
    cfg = configparser.ConfigParser()
    cfg["PROBRUN"] = {
        "varParList": "musamosat|relTh",
        "variationType": "percent|rangefromci",
        "variationValue": "60|ci95",
        "numberOfSteps": "2|2",
        "defaultSetup": "True",
        "varParType": "float|float",
        "samplingStrategy": "2",
    }

    com1DFACfg = cfgUtils.getDefaultModuleConfig(com1DFA)
    varName = "relTh"

    varDict = pA.makeDictFromVars(cfg["PROBRUN"])
    # call function
    cfgNew = pA.updateCfgRange(com1DFACfg, cfg, varName, varDict[varName])

    assert cfgNew["GENERAL"]["musamosat"] == "0.155"
    assert cfgNew["GENERAL"]["relTh"] == ""
    assert cfgNew["GENERAL"]["relThFromFile"] == "True"
    assert cfgNew["GENERAL"]["relThPercentVariation"] == ""
    assert cfgNew["GENERAL"]["relThRangeFromCiVariation"] == "ci95$2"

    # setup inputs
    cfg = configparser.ConfigParser()
    cfg["PROBRUN"] = {
        "varParList": "musamosat|relTh",
        "variationType": "percent|range",
        "variationValue": "60|0.25",
        "numberOfSteps": "2|8",
        "defaultSetup": "True",
        "varParType": "float|float",
        "samplingStrategy": "2",
    }

    com1DFACfg = cfgUtils.getDefaultModuleConfig(com1DFA)
    varName = "relTh"

    varDict = pA.makeDictFromVars(cfg["PROBRUN"])
    # call function
    cfgNew = pA.updateCfgRange(com1DFACfg, cfg, varName, varDict[varName])

    #    print('cfgNEW', cfgNew['GENERAL']['relThRangeVariation'])

    assert cfgNew["GENERAL"]["musamosat"] == "0.155"
    assert cfgNew["GENERAL"]["relTh"] == ""
    assert cfgNew["GENERAL"]["relThFromFile"] == "True"
    assert cfgNew["GENERAL"]["relThPercentVariation"] == ""
    assert cfgNew["GENERAL"]["relThRangeVariation"] == "0.25$8"


def test_makeDictFromVars():
    """test creating a dict from parameter variation info"""

    cfgProb = configparser.ConfigParser()
    cfgProb.optionxform = str
    cfgProb["PROBRUN"] = {
        "varParList": "musamosat|relTh|entTh",
        "variationType": "percent|rangeFromCi|range",
        "variationValue": "60|ci95|0.15",
        "numberOfSteps": "2|3|4",
        "samplingStrategy": "2",
        "varParType": "float|float|float",
        "nSample": "30",
        "sampleSeed": "12345",
        "sampleMethod": "latin",
    }

    # call function to be tested
    variationD = pA.makeDictFromVars(cfgProb["PROBRUN"])

    assert len(variationD) == 3
    assert len(variationD["relTh"]) == 3
    assert variationD["relTh"]["numberOfSteps"] == "3"

    cfgProb["PROBRUN"]["variationType"] = "percent|range"

    message = "For every parameter in varParList a variationValue, numberOfSteps and variationType needs to be provided"
    with pytest.raises(AssertionError) as e:
        assert pA.makeDictFromVars(cfgProb["PROBRUN"])
    assert message in str(e.value)

    cfgProb["PROBRUN"]["variationType"] = "percent|rangefromci|range"
    cfgProb["PROBRUN"]["variationValue"] = "60|50|0.15"

    message = "If rangefromci is chosen as variation type, ci95 is required as variationValue"
    with pytest.raises(AssertionError) as e:
        assert pA.makeDictFromVars(cfgProb["PROBRUN"])
    assert message in str(e.value)


def test_createSampleWithVariationStandardParameters():
    """test creation of sample for standard parameters"""

    cfgProb = configparser.ConfigParser()
    cfgProb.optionxform = str
    cfgProb["PROBRUN"] = {
        "varParList": "musamosat|relTh|entTh",
        "variationType": "percent|range|range",
        "variationValue": "60|0.25|0.15",
        "numberOfSteps": "2|3|4",
        "samplingStrategy": "1",
        "varParType": "float|float|float",
        "nSample": "30",
        "sampleSeed": "12345",
        "sampleMethod": "latin",
    }

    varParList = cfgProb["PROBRUN"]["varParList"].split("|")
    valVariationValue = cfgProb["PROBRUN"]["variationValue"].split("|")
    varType = cfgProb["PROBRUN"]["variationType"].split("|")

    # read initial configuration
    cfgStart = cfgUtils.getDefaultModuleConfig(com1DFA, toPrint=False)
    cfgStart["GENERAL"]["relThFromFile"] = "False"
    cfgStart["GENERAL"]["relTh"] = "1.5"
    cfgStart["GENERAL"]["entTh"] = "0.4"
    paramValuesD = pA.createSampleWithVariationStandardParameters(
        cfgProb, cfgStart, varParList, valVariationValue, varType
    )

    assert len(paramValuesD["values"]) == 30
    assert paramValuesD["names"] == ["musamosat", "relTh", "entTh"]
    assert paramValuesD["thFromIni"] == ""
    assert np.amax(paramValuesD["values"][:, 1]) <= 1.75
    assert np.amin(paramValuesD["values"][:, 1]) >= 1.25
    assert np.amax(paramValuesD["values"][:, 0]) <= 0.248
    assert np.amin(paramValuesD["values"][:, 0]) >= 0.062
    assert np.amax(paramValuesD["values"][:, 2]) <= 0.55
    assert np.amin(paramValuesD["values"][:, 2]) >= 0.25


def test_createSampleWithVariationForThParameters(tmp_path):
    """test if thickness parameters are also included"""

    # set input directory
    avaName = "testCom1DFA2"
    dirPath = pathlib.Path(__file__).parents[0]
    inputDir = dirPath / "data" / avaName
    avaDir = pathlib.Path(tmp_path, avaName)
    shutil.copytree(inputDir, avaDir)

    cfgProb = configparser.ConfigParser()
    cfgProb.optionxform = str
    cfgProb["PROBRUN"] = {
        "varParList": "musamosat|relTh|entTh",
        "variationType": "percent|rangefromci|range",
        "variationValue": "60|ci95|0.15",
        "numberOfSteps": "2|3|4",
        "samplingStrategy": "1",
        "varParType": "float|float|float",
        "nSample": "30",
        "sampleSeed": "12345",
        "sampleMethod": "latin",
    }

    varParList = cfgProb["PROBRUN"]["varParList"].split("|")
    valVariationValue = cfgProb["PROBRUN"]["variationValue"].split("|")
    varType = cfgProb["PROBRUN"]["variationType"].split("|")

    # read initial configuration
    cfgStart = cfgUtils.getDefaultModuleConfig(com1DFA, toPrint=False)

    thReadFromShp = ["relTh", "entTh"]

    paramValuesDList = pA.createSampleWithVariationForThParameters(
        avaDir, cfgProb, cfgStart, varParList, valVariationValue, varType, thReadFromShp
    )
    paramValuesD = paramValuesDList[0]

    assert len(paramValuesD["values"]) == 30
    assert paramValuesD["names"] == ["musamosat", "relTh0", "relTh1", "entTh0"]
    assert len(paramValuesD["thFromIni"].split("|")) == 2
    assert "relTh" in paramValuesD["thFromIni"]
    assert "entTh" in paramValuesD["thFromIni"]
    assert np.amax(paramValuesD["values"][:, 1]) <= 1.75
    assert np.amin(paramValuesD["values"][:, 1]) >= 1.25
    assert np.amax(paramValuesD["values"][:, 2]) <= 1.4
    assert np.amin(paramValuesD["values"][:, 2]) >= 1.0
    assert np.amax(paramValuesD["values"][:, 0]) <= 0.248
    assert np.amin(paramValuesD["values"][:, 0]) >= 0.062
    assert np.amax(paramValuesD["values"][:, 3]) <= 0.45
    assert np.amin(paramValuesD["values"][:, 3]) >= 0.15

    cfgProb["PROBRUN"] = {
        "varParList": "musamosat|relTh|entTh",
        "variationType": "percent|rangefromci|rangefromci",
        "variationValue": "60|ci95|ci95",
        "numberOfSteps": "2|3|4",
        "samplingStrategy": "1",
        "varParType": "float|float|float",
        "nSample": "30",
        "sampleSeed": "12345",
        "sampleMethod": "latin",
    }

    varParList = cfgProb["PROBRUN"]["varParList"].split("|")
    valVariationValue = cfgProb["PROBRUN"]["variationValue"].split("|")
    varType = cfgProb["PROBRUN"]["variationType"].split("|")

    # read initial configuration
    cfgStart = cfgUtils.getDefaultModuleConfig(com1DFA, toPrint=False)

    thReadFromShp = ["relTh", "entTh"]

    paramValuesDList = pA.createSampleWithVariationForThParameters(
        avaDir, cfgProb, cfgStart, varParList, valVariationValue, varType, thReadFromShp
    )
    paramValuesD = paramValuesDList[0]
    assert len(paramValuesD["values"]) == 30
    assert paramValuesD["names"] == ["musamosat", "relTh0", "relTh1", "entTh0"]
    assert len(paramValuesD["thFromIni"].split("|")) == 2
    assert "relTh" in paramValuesD["thFromIni"]
    assert "entTh" in paramValuesD["thFromIni"]
    assert np.amax(paramValuesD["values"][:, 1]) <= 1.75
    assert np.amin(paramValuesD["values"][:, 1]) >= 1.25
    assert np.amax(paramValuesD["values"][:, 2]) <= 1.4
    assert np.amin(paramValuesD["values"][:, 2]) >= 1.0
    assert np.amax(paramValuesD["values"][:, 0]) <= 0.248
    assert np.amin(paramValuesD["values"][:, 0]) >= 0.062
    assert np.amax(paramValuesD["values"][:, 3]) <= 0.4
    assert np.amin(paramValuesD["values"][:, 3]) >= 0.1

    cfgProb["PROBRUN"] = {
        "varParList": "musamosat|relTh|entTh",
        "variationType": "percent|rangefromci|percent",
        "variationValue": "60|ci95|5",
        "numberOfSteps": "2|3|4",
        "samplingStrategy": "1",
        "varParType": "float|float|float",
        "nSample": "30",
        "sampleSeed": "12345",
        "sampleMethod": "latin",
    }

    varParList = cfgProb["PROBRUN"]["varParList"].split("|")
    valVariationValue = cfgProb["PROBRUN"]["variationValue"].split("|")
    varType = cfgProb["PROBRUN"]["variationType"].split("|")

    # read initial configuration
    cfgStart = cfgUtils.getDefaultModuleConfig(com1DFA, toPrint=False)
    cfgStart["GENERAL"]["entTh"] = "0.5"
    cfgStart["GENERAL"]["entThFromFile"] = "False"
    thReadFromShp = ["relTh"]

    paramValuesDList = pA.createSampleWithVariationForThParameters(
        avaDir, cfgProb, cfgStart, varParList, valVariationValue, varType, thReadFromShp
    )
    paramValuesD = paramValuesDList[0]
    assert len(paramValuesD["values"]) == 30
    assert paramValuesD["names"] == ["musamosat", "relTh0", "relTh1", "entTh"]
    assert len(paramValuesD["thFromIni"].split("|")) == 1
    assert "relTh" in paramValuesD["thFromIni"]
    assert np.amax(paramValuesD["values"][:, 1]) <= 1.75
    assert np.amin(paramValuesD["values"][:, 1]) >= 1.25
    assert np.amax(paramValuesD["values"][:, 2]) <= 1.4
    assert np.amin(paramValuesD["values"][:, 2]) >= 1.0
    assert np.amax(paramValuesD["values"][:, 0]) <= 0.248
    assert np.amin(paramValuesD["values"][:, 0]) >= 0.062
    assert np.amax(paramValuesD["values"][:, 3]) <= 0.525
    assert np.amin(paramValuesD["values"][:, 3]) >= 0.475


def test_createCfgFiles(tmp_path):
    """test writing of cfg files"""

    paramValuesD = {
        "names": ["relTh", "musamosat"],
        "values": np.asarray([[1.2, 0.1], [1.4, 0.12], [1.6, 0.14]]),
        "thFromIni": "",
    }

    cfgProb = configparser.ConfigParser()
    cfgProb.optionxform = str
    cfgProb["PROBRUN"] = {}
    cfgProb["com1DFA_com1DFA_override"] = {
        "defaultConfig": True,
        "relTh": "",
        "musamosat": 0.2,
        "thFromIni": False,
    }

    cfgFiles = pA.createCfgFiles([paramValuesD], com1DFA, cfgProb, cfgPath="")

    #    print(cfgFiles)

    cfgTest1 = configparser.ConfigParser()
    cfgTest1.read(cfgFiles[0])
    cfgTest2 = configparser.ConfigParser()
    cfgTest2.read(cfgFiles[1])
    cfgTest3 = configparser.ConfigParser()
    cfgTest3.read(cfgFiles[2])

    assert len(cfgFiles) == 3
    assert cfgTest1["GENERAL"].getfloat("relTh") == 1.2
    assert cfgTest1["GENERAL"].getfloat("musamosat") == 0.1
    assert cfgTest1["INPUT"]["thFromIni"] == ""
    assert cfgTest1["VISUALISATION"]["scenario"] == "0"
    assert cfgTest2["GENERAL"].getfloat("relTh") == 1.4
    assert cfgTest2["GENERAL"].getfloat("musamosat") == 0.12
    assert cfgTest2["INPUT"]["thFromIni"] == ""
    assert cfgTest2["VISUALISATION"]["scenario"] == "1"
    assert cfgTest3["GENERAL"].getfloat("relTh") == 1.6
    assert cfgTest3["GENERAL"].getfloat("musamosat") == 0.14
    assert cfgTest3["INPUT"]["thFromIni"] == ""
    assert cfgTest3["VISUALISATION"]["scenario"] == "2"

    cfgProb["PROBRUN"] = {}
    cfgProb["com1DFA_override"] = {"defaultConfig": True}


def test_fetchStartCfg(tmp_path):
    """test fetching starting cfg"""

    cfg = configparser.ConfigParser()
    cfg.optionxform = str
    cfg["com1DFA_com1DFA_override"] = {"defaultConfig": True, "musamosat": 0.2}

    cfgStart = pA.fetchStartCfg(com1DFA, cfg)

    assert cfgStart.has_option("GENERAL", "entTh") is True
    assert cfgStart["GENERAL"].getfloat("musamosat") == 0.2

    cfg["com1DFA_com1DFA_override"] = {"defaultConfig": True}
    cfgStart = pA.fetchStartCfg(com1DFA, cfg)

    assert cfgStart.has_option("GENERAL", "entTh") is True
    assert cfgStart["GENERAL"].getfloat("musamosat") == 0.155


def test_fetchProbConfigs():
    """test creation of probability configurations"""

    cfg = configparser.ConfigParser()
    cfg.optionxform = str
    cfg["PROBRUN"] = {"samplingStrategy": "2", "varParList": "relTh|musamosat"}

    probConfigs = pA.fetchProbConfigs(cfg["PROBRUN"])

    assert len(probConfigs) == 3
    assert "includemusamosat" in probConfigs.keys()
    assert "includerelTh" in probConfigs.keys()

    cfg["PROBRUN"] = {"samplingStrategy": "1", "varParList": "relTh|musamosat"}

    probConfigs = pA.fetchProbConfigs(cfg["PROBRUN"])

    assert len(probConfigs) == 1
    assert "includeAll" in probConfigs.keys()


def test_fetchThicknessInfo(tmp_path):
    """test fetching info on thickness settings if readFromShp"""
    avaName = "testCom1DFA2"
    dirPath = pathlib.Path(__file__).parents[0]
    inputDir = dirPath / "data" / avaName
    avaDir = pathlib.Path(tmp_path, avaName)
    shutil.copytree(inputDir, avaDir)

    inputSimFiles = pA.fetchThicknessInfo(avaDir)

    assert inputSimFiles["releaseScenarioList"] == ["relParabola", "relParabolaTwo"]
    assert inputSimFiles["relParabola"]["thickness"][0] == "1.5"
    assert inputSimFiles["relParabola"]["thickness"][1] == "1.2"
    assert inputSimFiles["relParabola"]["id"][0] == "0"
    assert inputSimFiles["relParabola"]["id"][1] == "1"
    assert inputSimFiles["relParabola"]["ci95"][0] == "0.25"
    assert inputSimFiles["relParabola"]["ci95"][1] == "0.2"


def test_fetchThicknessList(tmp_path):
    """test fetching the thickness info"""

    avaName = "testCom1DFA2"
    dirPath = pathlib.Path(__file__).parents[0]
    inputDir = dirPath / "data" / avaName
    avaDir = pathlib.Path(tmp_path, avaName)
    shutil.copytree(inputDir, avaDir)

    # fetch input files and corresponding thickness info
    inputSimFiles = pA.fetchThicknessInfo(avaDir)

    thValues, ciValues, thicknessFeatureNames = pA.fetchThThicknessLists(
        "relTh", inputSimFiles, inputSimFiles["relFiles"][0], ciRequired=True
    )

    assert thValues[0] == 1.5
    assert thValues[1] == 1.2
    assert ciValues[0] == 0.25
    assert ciValues[1] == 0.2
    assert thicknessFeatureNames[0] == "relTh0"
    assert thicknessFeatureNames[1] == "relTh1"


def test_cfgFilesGlobalApproach(tmp_path):
    """test global approach to fetch sample and create cfg files"""

    # set input directory
    avaName = "testCom1DFA2"
    dirPath = pathlib.Path(__file__).parents[0]
    inputDir = dirPath / "data" / avaName
    avaDir = pathlib.Path(tmp_path, avaName)
    shutil.copytree(inputDir, avaDir)
    outDir = avaDir / "Outputs"

    cfgProb = configparser.ConfigParser()
    cfgProb.optionxform = str
    cfgProb["PROBRUN"] = {
        "varParList": "musamosat|relTh",
        "variationType": "percent|percent",
        "variationValue": "60|50",
        "numberOfSteps": "2|3",
        "defaultSetup": "True",
        "samplingStrategy": "1",
        "varParType": "float|float",
        "nSample": "40",
        "sampleSeed": "12345",
        "sampleMethod": "latin",
    }
    cfgProb["com1DFA_com1DFA_override"] = {
        "defaultConfig": "True",
        "relThFromFile": False,
        "relTh": 2.0,
        "musamosat": 0.155,
        "frictModel": "samosAT",
    }

    # call function to be tested
    cfgFiles = pA.cfgFilesGlobalApproach(avaDir, cfgProb, com1DFA, outDir)

    #    print('cfgFiles', cfgFiles)

    cfgTest = configparser.ConfigParser()
    cfgTest.read(cfgFiles[0])
    #    print('cfgTest', cfgTest['GENERAL']['relThFromFile'], cfgTest['GENERAL']['relTh'],
    #         cfgTest['GENERAL']['relThPercentVariation'], cfgTest['GENERAL']['musamosat'])

    assert cfgTest["GENERAL"]["relThFromFile"] == "False"
    assert cfgTest["GENERAL"]["relTh"] == "2.2719559079879"
    assert len(cfgFiles) == 40

    cfgProb = configparser.ConfigParser()
    cfgProb.optionxform = str
    cfgProb["PROBRUN"] = {
        "varParList": "musamosat|relTh",
        "variationType": "percent|percent",
        "variationValue": "60|50",
        "numberOfSteps": "2|3",
        "defaultSetup": "True",
        "samplingStrategy": "1",
        "varParType": "float|float",
        "nSample": "40",
        "sampleSeed": "12345",
        "sampleMethod": "latin",
    }
    cfgProb["com1DFA_com1DFA_override"] = {"defaultConfig": "True"}

    # call function to be tested
    cfgFiles2 = pA.cfgFilesGlobalApproach(avaDir, cfgProb, com1DFA, outDir)

    #    print('cfgFiles', cfgFiles)

    cfgTest1 = configparser.ConfigParser()
    cfgTest1.read(cfgFiles2[0])
    #    print('cfgTest', cfgTest1['GENERAL']['relThFromFile'], cfgTest1['GENERAL']['relTh'],
    #         cfgTest1['GENERAL']['relThPercentVariation'], cfgTest1['GENERAL']['musamosat'])

    assert cfgTest1["GENERAL"]["relThFromFile"] == "True"
    assert cfgTest1["GENERAL"]["relTh"] == ""
    assert cfgTest1["GENERAL"]["relTh0"] != ""
    assert cfgTest1["INPUT"]["releaseScenario"] == "relParabola"
    assert len(cfgFiles2) == 80


def test_cfgFilesLocalApproach(tmp_path):
    """test creating cfg files from one at a time var"""

    # set input directory
    avaName = "testCom1DFA2"
    dirPath = pathlib.Path(__file__).parents[0]
    inputDir = dirPath / "data" / avaName
    avaDir = pathlib.Path(tmp_path, avaName)
    shutil.copytree(inputDir, avaDir)
    outDir = avaDir

    cfgProb = configparser.ConfigParser()
    cfgProb.optionxform = str
    cfgProb["PROBRUN"] = {
        "varParList": "musamosat|relTh",
        "variationType": "percent|percent",
        "variationValue": "60|50",
        "numberOfSteps": "2|3",
        "defaultSetup": "True",
        "samplingStrategy": "2",
        "varParType": "float|float",
    }
    cfgProb["sampling_override"] = {"defaultConfig": "True"}
    cfgProb["com1DFA_com1DFA_override"] = {"defaultConfig": "True"}

    # check variation settings
    variationsDict = pA.makeDictFromVars(cfgProb["PROBRUN"])

    # call function to be tested
    cfgFiles = pA.cfgFilesLocalApproach(variationsDict, cfgProb, com1DFA, outDir)

    #    print('cfgFiles', cfgFiles)

    # load cfg from file
    cfgMu = configparser.ConfigParser()
    cfgRelTh = configparser.ConfigParser()

    cfgMu.read(cfgFiles[0])
    cfgRelTh.read(cfgFiles[1])

    #    print(cfgMu['GENERAL']['musamosat'], cfgMu['GENERAL']['relTh'], cfgRelTh['GENERAL']['musamosat'],
    #           cfgRelTh['GENERAL']['relTh'])

    assert cfgMu["GENERAL"]["musamosat"] == "0.155$60$2"
    assert cfgMu["GENERAL"]["relTh"] == ""
    assert cfgRelTh["GENERAL"]["musamosat"] == "0.155"
    assert cfgRelTh["GENERAL"]["relTh"] == ""
    assert cfgRelTh["GENERAL"]["relThFromFile"] == "True"
    assert cfgRelTh["GENERAL"]["relThPercentVariation"] == "50$3"
    assert cfgRelTh["GENERAL"]["relThFromFile"] == "True"
    assert cfgMu["GENERAL"]["relThFromFile"] == "True"

    cfgProb = configparser.ConfigParser()
    cfgProb.optionxform = str
    cfgProb["PROBRUN"] = {
        "varParList": "musamosat|relTh",
        "variationType": "percent|range",
        "variationValue": "60|0.5",
        "numberOfSteps": "2|3",
        "samplingStrategy": "2",
        "varParType": "float|float",
    }
    cfgProb["sampling_override"] = {"defaultConfig": "True"}
    cfgProb["com1DFA_com1DFA_override"] = {
        "defaultConfig": "True",
        "relThFromFile": False,
        "relTh": 2.0,
        "musamosat": 0.155,
        "frictModel": "samosAT",
    }

    # check variation settings
    variationsDict = pA.makeDictFromVars(cfgProb["PROBRUN"])

    # call function to be tested
    cfgFiles = pA.cfgFilesLocalApproach(variationsDict, cfgProb, com1DFA, outDir)

    #    print('cfgFiles', cfgFiles)

    # load cfg from file
    cfgMu = configparser.ConfigParser()
    cfgRelTh = configparser.ConfigParser()

    cfgMu.read(cfgFiles[0])
    cfgRelTh.read(cfgFiles[1])

    #    print(cfgMu['GENERAL']['musamosat'], cfgMu['GENERAL']['relTh'], cfgRelTh['GENERAL']['musamosat'],
    #           cfgRelTh['GENERAL']['relTh'])

    assert cfgMu["GENERAL"]["musamosat"] == "0.155$60$2"
    assert np.isclose(cfgMu["GENERAL"].getfloat("relTh"), 2.0)
    assert cfgRelTh["GENERAL"]["musamosat"] == "0.155"
    assert np.isclose(cfgRelTh["GENERAL"].getfloat("relTh"), 2.0)
    assert cfgRelTh["GENERAL"]["relThFromFile"] == "False"
    assert cfgRelTh["GENERAL"]["relThRangeVariation"] == "0.5$3"
    assert cfgRelTh["GENERAL"]["relThFromFile"] == "False"
    assert cfgMu["GENERAL"]["relThFromFile"] == "False"


def test_checkParameterSettings():
    """test if parameter settings are valid"""

    cfg = configparser.ConfigParser()
    cfg.optionxform = str
    cfg["GENERAL"] = {
        "relTh": "",
        "musamosat": "0.155",
        "relThFromFile": "True",
        "relThPercentVariation": "",
        "relThRangeVariation": "",
        "relThRangeFromCiVariation": "",
        "relThDistVariation": "",
    }
    varParList = ["relTh", "musamosat"]

    check, thReadFromShp = pA.checkParameterSettings(cfg, varParList)

    assert check
    assert len(thReadFromShp) == 1
    assert thReadFromShp[0] == "relTh"

    cfg["GENERAL"]["musamosat"] = "0.1:0.2:10"
    message = "Only one reference value is allowed for %s: but %s is given" % (
        "musamosat",
        "0.1:0.2:10",
    )
    with pytest.raises(AssertionError) as e:
        assert pA.checkParameterSettings(cfg, varParList)
    assert message in str(e.value)

    cfg["GENERAL"] = {
        "relTh": "",
        "musamosat": "0.155",
        "relThFromFile": "True",
        "relThPercentVariation": "",
        "relThRangeVariation": "5$4",
        "relThRangeFromCiVariation": "",
        "relThDistVariation": "",
    }

    with pytest.raises(AssertionError) as e:
        assert pA.checkParameterSettings(cfg, varParList)
    assert "Only one reference value is allowed for relTh" in str(e.value)
    assert "relThRangeVariation" in str(e.value)


def test_checkForNumberOfReferenceValues():
    """check if reference (base) value already has a variation set for thickness parameters"""

    cfg = configparser.ConfigParser()
    cfg.optionxform = str
    cfg["GENERAL"] = {
        "relTh": "",
        "musamosat": "0.155",
        "relThFromFile": "True",
        "relThPercentVariation": "",
        "relThRangeVariation": "",
        "relThRangeFromCiVariation": "",
        "relThDistVariation": "",
    }

    # call function to be tested
    checkIs = pA.checkForNumberOfReferenceValues(cfg["GENERAL"], "relTh")

    assert checkIs is True

    cfg["GENERAL"]["relThPercentVariation"] = "50$3"

    with pytest.raises(AssertionError) as e:
        assert pA.checkForNumberOfReferenceValues(cfg["GENERAL"], "relTh")
    assert "Only one reference value is allowed for relTh" in str(e.value)


def test_createSample_latin():
    """Test Latin Hypercube sampling method"""

    # Create test configuration
    testConfig = configparser.ConfigParser()
    testConfig["PROBRUN"] = {"nSample": "50", "sampleSeed": "12345", "sampleMethod": "latin"}

    testParList = ["param1", "param2", "param3"]

    from avaframe.ana4Stats.probAna import createSample

    resultSample = createSample(testConfig, testParList)

    # Check sample properties
    assert isinstance(resultSample, np.ndarray)
    assert resultSample.shape == (50, 3)  # nSample x number of parameters
    assert np.all(resultSample >= 0) and np.all(resultSample <= 1)  # Values should be in [0,1]

    # Check Latin Hypercube properties
    for colIdx in range(resultSample.shape[1]):
        # Check if values are well distributed (no clustering)
        binCounts = np.histogram(resultSample[:, colIdx], bins=10)[0]
        assert np.all(binCounts > 0)  # Each bin should have at least one sample


def test_createSample_morris():
    """Test Morris sampling method"""

    # Create test configuration
    testConfig = configparser.ConfigParser()
    testConfig["PROBRUN"] = {"nSample": "4", "sampleSeed": "12345", "sampleMethod": "morris"}

    testParList = ["param1", "param2"]

    from avaframe.ana4Stats.probAna import createSample

    resultSample = createSample(testConfig, testParList)

    # Check sample properties for Morris method
    expectedSampleSize = 4 * (len(testParList) + 1)  # Morris method formula
    assert resultSample.shape == (expectedSampleSize, 2)
    assert np.all(resultSample >= 0) and np.all(resultSample <= 1)


def test_createSample_reproducibility():
    """Test reproducibility with same seed"""

    # Create test configuration
    testConfig = configparser.ConfigParser()
    testConfig["PROBRUN"] = {"nSample": "30", "sampleSeed": "54321", "sampleMethod": "latin"}

    testParList = ["param1", "param2"]

    from avaframe.ana4Stats.probAna import createSample

    # Generate two samples with same seed
    firstSample = createSample(testConfig, testParList)
    secondSample = createSample(testConfig, testParList)

    # Check if samples are identical
    np.testing.assert_array_equal(firstSample, secondSample)


def test_createSample_different_seeds():
    """Test different seeds produce different samples"""

    # Create test configurations with different seeds
    testConfig1 = configparser.ConfigParser()
    testConfig1["PROBRUN"] = {"nSample": "30", "sampleSeed": "12345", "sampleMethod": "latin"}

    testConfig2 = configparser.ConfigParser()
    testConfig2["PROBRUN"] = {"nSample": "30", "sampleSeed": "54321", "sampleMethod": "latin"}

    testParList = ["param1", "param2"]

    from avaframe.ana4Stats.probAna import createSample

    # Generate samples with different seeds
    firstSample = createSample(testConfig1, testParList)
    secondSample = createSample(testConfig2, testParList)

    # Check if samples are different
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(firstSample, secondSample)


def test_createSample_invalid_method():
    """Test handling of invalid sampling method"""

    # Create test configuration with invalid method
    testConfig = configparser.ConfigParser()
    testConfig["PROBRUN"] = {"nSample": "30", "sampleSeed": "12345", "sampleMethod": "invalid_method"}

    testParList = ["param1", "param2"]

    from avaframe.ana4Stats.probAna import createSample

    # Check if appropriate error is raised
    with pytest.raises(AssertionError):
        createSample(testConfig, testParList)


def _writeSmallAsc(filePath, nRows, nCols, data):
    """Helper to write a minimal .asc raster file for testing"""
    header = (
        "ncols %d\n"
        "nrows %d\n"
        "xllcenter 0.0\n"
        "yllcenter 0.0\n"
        "cellsize 5.0\n"
        "nodata_value -9999\n" % (nCols, nRows)
    )
    with open(filePath, "w") as f:
        f.write(header)
        for row in data:
            f.write(" ".join(str(v) for v in row) + "\n")


def test_probAnalysis_singleLayer_unchanged(tmp_path):
    """Single-layer results with empty layer config work as before"""

    # Create minimal directory structure
    avaDir = tmp_path / "avaTest"
    peakDir = avaDir / "Outputs" / "testMod" / "peakFiles"
    peakDir.mkdir(parents=True)
    outDir = avaDir / "Outputs" / "ana4Stats"
    outDir.mkdir(parents=True)

    # Create two single-layer ppr files with values > threshold(1.0) in different cells
    nRows, nCols = 3, 3
    data1 = [[0, 0, 0], [0, 5.0, 0], [0, 0, 0]]
    data2 = [[0, 0, 0], [0, 5.0, 0], [0, 0, 2.0]]

    _writeSmallAsc(peakDir / "rel1_abc123_com1_C_null_dfa_ppr.asc", nRows, nCols, data1)
    _writeSmallAsc(peakDir / "rel1_def456_com1_C_null_dfa_ppr.asc", nRows, nCols, data2)

    cfg = configparser.ConfigParser()
    cfg["GENERAL"] = {"peakLim": "1.0", "peakVar": "ppr", "layer": ""}

    # Use a non-filtering modName so all files are included
    analysisPerformed, contourDict = pA.probAnalysis(avaDir, cfg, "testMod")

    assert analysisPerformed is True
    # Both sims have center cell > threshold, so probability = 1.0 there
    # Only sim2 has bottom-right > threshold, so probability = 0.5 there
    probFile = outDir / "avaTest_prob__ppr_lim1.0.asc"
    assert probFile.exists()
    probData = np.loadtxt(probFile, skiprows=6)
    assert probData.shape == (nRows, nCols)
    # Center cell: both sims exceed threshold → 1.0
    assert np.isclose(probData[1, 1], 1.0)
    # Bottom-right: only 1 of 2 sims exceeds → 0.5
    assert np.isclose(probData[2, 2], 0.5)


def test_probAnalysis_multiLayer_withLayer(tmp_path):
    """Multi-layer results with layer set filters to only the specified layer"""

    avaDir = tmp_path / "avaTest"
    peakDir = avaDir / "Outputs" / "com8MoTPSA" / "peakFiles"
    peakDir.mkdir(parents=True)
    outDir = avaDir / "Outputs" / "ana4Stats"
    outDir.mkdir(parents=True)

    nRows, nCols = 3, 3
    # L1 files: high values everywhere
    dataL1 = [[5.0, 5.0, 5.0], [5.0, 5.0, 5.0], [5.0, 5.0, 5.0]]
    # L2 files: only center cell has high value
    dataL2 = [[0, 0, 0], [0, 3.0, 0], [0, 0, 0]]

    _writeSmallAsc(peakDir / "rel1_abc123_com8_C_null_psa_L1_ppr.asc", nRows, nCols, dataL1)
    _writeSmallAsc(peakDir / "rel1_abc123_com8_C_null_psa_L2_ppr.asc", nRows, nCols, dataL2)
    _writeSmallAsc(peakDir / "rel1_def456_com8_C_null_psa_L1_ppr.asc", nRows, nCols, dataL1)
    _writeSmallAsc(peakDir / "rel1_def456_com8_C_null_psa_L2_ppr.asc", nRows, nCols, dataL2)

    cfg = configparser.ConfigParser()
    cfg["GENERAL"] = {"peakLim": "1.0", "peakVar": "ppr", "layer": "L2"}

    analysisPerformed, contourDict = pA.probAnalysis(
        avaDir, cfg, "testMod", inputDir=avaDir / "Outputs" / "com8MoTPSA"
    )

    assert analysisPerformed is True
    # Only L2 files should be included: both have center=3.0 > 1.0, rest=0
    probFile = outDir / "avaTest_prob__ppr_L2_lim1.0.asc"
    assert probFile.exists()
    probData = np.loadtxt(probFile, skiprows=6)
    # Center cell: both L2 sims exceed → 1.0
    assert np.isclose(probData[1, 1], 1.0)
    # Corner cells: no L2 sim exceeds → 0.0
    assert np.isclose(probData[0, 0], 0.0)


def test_probAnalysis_multiLayer_noLayer_errors(tmp_path):
    """Multi-layer results without layer config raises an error"""

    avaDir = tmp_path / "avaTest"
    peakDir = avaDir / "Outputs" / "com8MoTPSA" / "peakFiles"
    peakDir.mkdir(parents=True)
    outDir = avaDir / "Outputs" / "ana4Stats"
    outDir.mkdir(parents=True)

    nRows, nCols = 3, 3
    data = [[0, 0, 0], [0, 5.0, 0], [0, 0, 0]]

    _writeSmallAsc(peakDir / "rel1_abc123_com8_C_null_psa_L1_ppr.asc", nRows, nCols, data)
    _writeSmallAsc(peakDir / "rel1_abc123_com8_C_null_psa_L2_ppr.asc", nRows, nCols, data)

    cfg = configparser.ConfigParser()
    cfg["GENERAL"] = {"peakLim": "1.0", "peakVar": "ppr", "layer": ""}

    with pytest.raises(ValueError, match="Multi-layer results detected"):
        pA.probAnalysis(
            avaDir, cfg, "testMod", inputDir=avaDir / "Outputs" / "com8MoTPSA"
        )
