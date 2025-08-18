"""Tests for dfa2Aimec"""

import configparser

import numpy as np
import pandas as pd
import pytest
import pathlib

import rasterio.crs

import avaframe.in2Trans.rasterUtils as IOf

# Local imports
import avaframe.com1DFA.deriveParameterSet as dP
import avaframe.in3Utils.fileHandlerUtils as fU
from avaframe.com1DFA.deriveParameterSet import setRangeFromCiVariation


def test_getVariationDict(caplog):
    """test creating a variation dictionary"""

    # setup required input
    avaDir = "test/avaTest"
    cfg = configparser.ConfigParser()
    cfg.optionxform = str
    cfg["GENERAL"] = {
        "simTypeList": "null|ent",
        "howMeBlue": "10:20:4",
        "modelType": "dfa",
        "resType": "ppr|pft|pfv|particles|FT",
        "tSteps": "0:1",
        "initPartDistType": "random",
        "initialiseParticlesFromFile": "False",
        "particleFile": "",
        "seed": "12345",
        "rho": "300|400",
        "rhoEnt": "100",
        "relTh": "1.",
        "secRelArea": "True",
        "secondaryRelTh": "0.5",
        "dt": "0.05",
        "tEnd": "400",
    }
    cfg["INPUT"] = {"releaseScenario": "relTest"}
    modDict = {
        "GENERAL": {
            "simTypeList": ["null|ent", "available"],
            "resType": ["ppr|pft|pfv", "ppr|pft|pfv|particles|FT"],
            "tSteps": ["0:1", "1"],
            "rho": ["300|400", "200"],
            "secRelArea": ["True", "False"],
        },
        "TEST": {"test": ["test1", ""]},
    }

    # call function to be tested
    variations = dP.getVariationDict(avaDir, cfg, modDict)

    #    print("variations", variations)

    # see if a parameter that was locally added to the GENERAL cfg and has a variation is filtered out:
    assert ("Parameter howMeBlue: has a variation, seems to be added") in caplog.text

    variationsTest = {
        "simTypeList": ["null", "ent"],
        "rho": np.asarray([300, 400]),
        "releaseScenario": ["relTest"],
    }

    assert len(variations.keys()) == len(variationsTest.keys())
    assert variations["simTypeList"][0] == "null"
    assert variations["simTypeList"][1] == "ent"
    assert np.array_equal(variations["rho"], np.asarray([300, 400]))

    cfg["GENERAL"]["relThFromFile"] = "True"
    cfg["GENERAL"]["relThFromFile"] = "False"
    cfg["GENERAL"]["relTh"] = ""
    cfg["GENERAL"]["relThPercentVariation"] = "40$5"

    modDict["GENERAL"]["relThPercentVariation"] = "40$5"

    # call function to be tested
    variations2 = dP.getVariationDict(avaDir, cfg, modDict)

    #    print("variations2", variations2)

    variationsTest2 = {
        "simTypeList": ["null", "ent"],
        "rho": np.asarray([300, 400]),
        "relThPercentVariation": np.linspace(0.6, 1.4, 5),
        "releaseScenario": ["relTest"],
    }

    assert len(variations2.keys()) == len(variationsTest2.keys())
    assert variations2["simTypeList"][0] == "null"
    assert variations2["simTypeList"][1] == "ent"
    assert np.array_equal(variations2["relThPercentVariation"], variationsTest2["relThPercentVariation"])
    assert np.array_equal(variations2["rho"], np.asarray([300, 400]))


def test_validateVarDict():
    """test if variation dict has only valid parameters"""

    # setup required input
    variationDict = {
        "simTypeList": ["null", "ent"],
        "rho": np.asarray([300, 400]),
        "relThAA": np.asarray([1.0, 2.0]),
        "secRelAreA": ["False", "True"],
        "rhoEnt": "200.",
    }
    standardCfg = configparser.ConfigParser()
    standardCfg.optionxform = str
    standardCfg["GENERAL"] = {
        "simTypeList": "available",
        "modelType": "dfa",
        "resType": "ppr|pft|pfv",
        "tSteps": "0:1",
        "initPartDistType": "random",
        "initialiseParticlesFromFile": "False",
        "particleFile": "",
        "seed": "12345",
        "rho": "300|400",
        "rhoEnt": "100",
        "relTh": "1.",
        "secRelArea": "True",
        "secondaryRelTh": "0.5",
        "dt": "0.05",
        "tEnd": "400",
    }

    # call function to be tested
    variationDictTest = dP.validateVarDict(variationDict, standardCfg)

    #    print("variationDictTest", variationDictTest)

    assert len(variationDictTest.keys()) == 3
    assert variationDictTest["simTypeList"][0] == "null"
    assert variationDictTest["simTypeList"][1] == "ent"
    assert variationDictTest["rhoEnt"] == ["200."]
    assert np.array_equal(variationDictTest["rho"], np.asarray([300, 400]))
    assert "relThAA" not in variationDictTest.keys()

    variationDict = {
        "simTypeList": ["null", "ent"],
        "rho": np.asarray([300, 400]),
        "relThAA": np.asarray([1.0, 2.0]),
        "secRelAreA": ["False", "True"],
        "rhoEnt": "100:200:2&400",
    }
    # call function to be tested
    variationDictTest = dP.validateVarDict(variationDict, standardCfg)

    #    print("variationDictTest", variationDictTest)

    assert len(variationDictTest.keys()) == 3
    assert variationDictTest["simTypeList"][0] == "null"
    assert variationDictTest["simTypeList"][1] == "ent"
    assert variationDictTest["rhoEnt"][0] == 100.0
    assert variationDictTest["rhoEnt"][1] == 200.0
    assert variationDictTest["rhoEnt"][2] == 400.0
    assert len(variationDictTest["rhoEnt"]) == 3
    assert np.array_equal(variationDictTest["rho"], np.asarray([300, 400]))
    assert "relThAA" not in variationDictTest.keys()


def test_checkResType():
    """test checking if desired result type is in availble result types"""

    # setup required input
    fullCfg = configparser.ConfigParser()
    fullCfg["GENERAL"] = {"resType": "pft|ppr|pfv|particles|test1|test2"}
    section = "GENERAL"
    key = "resType"
    value = "pft|ppr|pfv|particles|test1|test2"

    # call function to be tested
    fullCfg = dP.checkResType(fullCfg, section, key, value)

    #    print("fullCfg", fullCfg)

    assert fullCfg["GENERAL"]["resType"] == "pft|ppr|pfv|particles"


def test_getThicknessValue():
    """test setting of thickness values"""

    inputSimFiles = {"release1HS": {"thickness": ["1.2", "1.4"]}}
    inputSimFiles["release1HS"]["id"] = ["0", "1"]
    inputSimFiles["release1HS"]["ci95"] = ["None", "None"]

    thType = "relTh"
    cfg = configparser.ConfigParser()
    cfg["GENERAL"] = {
        "relThFromFile": "False",
        "relTh": "",
        "relThFromFile": "True",
        "relThPercentVariation": "40$3",
        "relThDistVariation": "",
    }
    cfg["INPUT"] = {"releaseScenario": "release1HS"}

    cfg = dP.getThicknessValue(cfg, inputSimFiles, "release1HS", thType)

    assert cfg["INPUT"]["release1HS_relThId"] == "0|1"
    assert cfg["INPUT"]["release1HS_relThThickness"] == "1.2|1.4"
    assert cfg["GENERAL"]["relThPercentVariation"] == "40$3"

    inputSimFiles = {"release1HS": {"thickness": ["1.2", "None"]}}
    inputSimFiles["release1HS"]["id"] = ["0", "1"]
    inputSimFiles["release1HS"]["ci95"] = ["None", "None"]

    thType = "relTh"
    cfg = configparser.ConfigParser()
    cfg["GENERAL"] = {
        "relThFromFile": "False",
        "relTh": "",
        "relThFromFile": "True",
        "relThPercentVariation": "40$3",
        "relThDistVariation": "",
    }
    cfg["INPUT"] = {"releaseScenario": "release1HS"}

    with pytest.raises(AssertionError) as e:
        assert dP.getThicknessValue(cfg, inputSimFiles, "release1HS", thType)
    assert (
        str(e.value)
        == "Not all features in shape file have a thickness value - check shape file attributes: %s"
        % "release1HS"
    )

    inputSimFiles = {"release1HS": {"thickness": ["1.2", "None"]}}
    inputSimFiles["release1HS"]["id"] = ["0", "1"]
    inputSimFiles["release1HS"]["ci95"] = ["None", "None"]

    thType = "relTh"
    cfg = configparser.ConfigParser()
    cfg["GENERAL"] = {
        "relThFromFile": "False",
        "relTh": "40$2",
        "relThFromFile": "False",
        "relThPercentVariation": "",
        "relThDistVariation": "",
    }
    cfg["INPUT"] = {"releaseScenario": "release1HS"}

    with pytest.raises(AssertionError) as e:
        assert dP.getThicknessValue(cfg, inputSimFiles, "release1HS", thType)
    assert (
        str(e.value)
        == "Format of relTh value in ini file is not correct - for variation from ini use refValue$percent$numberOfSteps"
    )

    inputSimFiles = {"release1HS": {"thickness": ["1.2", "None"]}}
    inputSimFiles["release1HS"]["id"] = ["0", "1"]
    inputSimFiles["release1HS"]["ci95"] = ["None", "None"]

    thType = "relTh"
    cfg = configparser.ConfigParser()
    cfg["GENERAL"] = {
        "relThFromFile": "False",
        "relTh": "1.0",
        "relThFromFile": "False",
        "relThPercentVariation": "",
        "relThDistVariation": "",
    }
    cfg["INPUT"] = {"releaseScenario": "release1HS"}

    cfg = dP.getThicknessValue(cfg, inputSimFiles, "release1HS", thType)

    assert cfg.has_option("INPUT", "relThId") is False
    assert cfg.has_option("INPUT", "relThThickness") is False
    assert cfg["GENERAL"]["relThPercentVariation"] == ""
    assert cfg["GENERAL"]["relTh"] == "1.0"

    inputSimFiles = {"entrainment1": {"thickness": ["0.5", "0.6"]}}
    inputSimFiles["entrainment1"]["id"] = ["0", "1"]
    inputSimFiles["entrainment1"]["ci95"] = ["None", "None"]

    thType = "entTh"
    cfg = configparser.ConfigParser()
    cfg["GENERAL"] = {
        "entTh": "",
        "entThFromFile": "True",
        "entThPercentVariation": "",
        "entThDistVariation": "",
        "entThIfMissingInShp": "0.3",
    }
    cfg["INPUT"] = {"releaseScenario": ""}

    cfg = dP.getThicknessValue(cfg, inputSimFiles, "entrainment1", thType)

    assert cfg["INPUT"]["entThThickness"] == "0.5|0.6"
    assert cfg["INPUT"]["entThId"] == "0|1"
    assert cfg["INPUT"]["entThCi95"] == "None|None"
    assert cfg["GENERAL"]["entThPercentVariation"] == ""
    assert cfg["GENERAL"]["entTh"] == ""
    assert cfg["GENERAL"]["entThIfMissingInShp"] == "0.3"

    inputSimFiles = {"entrainment1": {"thickness": ["None", "None"]}}
    inputSimFiles["entrainment1"]["id"] = ["0", "1"]
    inputSimFiles["entrainment1"]["ci95"] = ["None", "None"]

    thType = "entTh"
    cfg = configparser.ConfigParser()
    cfg["GENERAL"] = {
        "entTh": "",
        "entThFromFile": "True",
        "entThPercentVariation": "",
        "entThDistVariation": "",
        "entThIfMissingInShp": "0.3",
    }
    cfg["INPUT"] = {"releaseScenario": ""}

    cfg = dP.getThicknessValue(cfg, inputSimFiles, "entrainment1", thType)

    assert cfg.has_option("INPUT", "entThId") is False
    assert cfg.has_option("INPUT", "entThThickness") is False
    assert cfg["GENERAL"]["entThPercentVariation"] == ""
    assert cfg["GENERAL"]["entTh"] == "0.3"
    assert cfg["GENERAL"]["entThIfMissingInShp"] == "0.3"


def test_checkThicknessSettings():
    """test checking thickness settings function"""

    # setup required inputs
    cfg = configparser.ConfigParser()
    cfg["GENERAL"] = {
        "entThFromFile": "True",
        "entTh": "",
        "entThPercentVariation": "",
        "entThRangeVariation": "",
        "entThRangeFromCiVariation": "",
    }

    thName = "entTh"
    inputSimFiles = {"entResInfo": {"flagEnt": "Yes", "entThFileType": ".shp"}}

    thicknessSettingsCorrect = dP.checkThicknessSettings(cfg, thName, inputSimFiles)

    assert thicknessSettingsCorrect

    cfg["GENERAL"]["entTh"] = "0.3"

    with pytest.raises(AssertionError) as e:
        assert dP.checkThicknessSettings(cfg, thName, inputSimFiles)
    assert str(e.value) == "If %s is set to True - it is not allowed to set a value for %s" % (
        "entThFromFile",
        "entTh",
    )

    cfg["GENERAL"]["entThFromFile"] = "False"
    cfg["GENERAL"]["entTh"] = ""

    with pytest.raises(AssertionError) as e:
        assert dP.checkThicknessSettings(cfg, thName, inputSimFiles)
    assert str(
        e.value
    ) == "If %s is set to False and Entrainment area defined by a shapefile - it is required to set a value for %s" % (
        "entThFromFile",
        "entTh",
    )

    cfg["GENERAL"]["entThFromFile"] = ""

    with pytest.raises(AssertionError) as e:
        assert dP.checkThicknessSettings(cfg, thName, inputSimFiles)
    assert str(e.value) == "Check %s - needs to be True or False" % "entThFromFile"

    cfg["GENERAL"]["relThFromFile"] = "False"
    cfg["GENERAL"]["relTh"] = "1.0"
    inputSimFiles = {"entResInfo": {"flagRel": "Yes", "relThFileType": ".asc"}}

    with pytest.raises(AssertionError) as e:
        assert dP.checkThicknessSettings(cfg, "relTh", inputSimFiles)
    assert str(e.value) == (
        "If Release area file is not a shapefile - it is not allowed to set a value for relTh"
    )

    # setup required inputs
    cfg = configparser.ConfigParser()
    cfg["GENERAL"] = {
        "relThFromFile": "False",
        "relTh": "",
        "relThPercentVariation": "",
        "relThRangeVariation": "",
        "relThRangeFromCiVariation": "",
    }

    thName = "relTh"

    thicknessSettingsCorrect = dP.checkThicknessSettings(cfg, thName, inputSimFiles)

    assert thicknessSettingsCorrect

    cfg["GENERAL"]["relThRangeVariation"] = "50$4"

    with pytest.raises(AssertionError) as e:
        assert dP.checkThicknessSettings(cfg, "relTh", inputSimFiles)
    assert "Release area read from raster" in str(e.value) and "variation not allowed" in str(e.value)

    # setup required inputs
    cfg = configparser.ConfigParser()
    cfg["GENERAL"] = {
        "relThFromFile": "True",
        "relTh": "",
        "relThPercentVariation": "",
        "relThRangeVariation": "50$4",
        "relThRangeFromCiVariation": "50$1",
    }
    inputSimFiles = {"entResInfo": {"flagRel": "Yes", "relThFileType": ".shp"}}

    thName = "relTh"
    with pytest.raises(AssertionError) as e:
        assert dP.checkThicknessSettings(cfg, "relTh", inputSimFiles)
    assert "Only one variation type is allowed - check" in str(e.value)


def test_appendShpThickness():
    """test appending thickness values"""

    # setup required inputs
    cfg = configparser.ConfigParser()
    cfg["GENERAL"] = {
        "secRelArea": "False",
        "simTypeActual": "null",
        "relThFromFile": "True",
        "relTh": "",
        "relThPercentVariation": "",
        "relThRangeVariation": "",
        "entThRangeFromCiVariation": "",
        "relThRangeFromCiVariation": "",
        "relThDistVariation": "",
    }
    cfg["INPUT"] = {"relThThickness": "1.2|1.4", "relThId": "0|1", "releaseScenario": "release1HS"}

    # call function to be tested
    cfg = dP.appendThicknessToCfg(cfg)

    assert cfg["GENERAL"]["relTh0"] == "1.2"
    assert cfg["GENERAL"]["relTh1"] == "1.4"

    # call function to be tested with different inputs again
    cfg = configparser.ConfigParser()
    cfg["GENERAL"] = {
        "secRelArea": "False",
        "simTypeActual": "null",
        "relThFromFile": "True",
        "relTh": "",
        "relThPercentVariation": "40$3",
        "relThRangeVariation": "",
        "entThRangeFromCiVariation": "",
        "relThRangeFromCiVariation": "",
        "relThDistVariation": "",
    }
    cfg["INPUT"] = {"relThThickness": "1.2|1.4", "relThId": "0|1", "releaseScenario": "release1HS"}
    cfg = dP.appendThicknessToCfg(cfg)

    assert cfg.has_option("GENERAL", "relTh0") is False
    assert cfg.has_option("GENERAL", "relTh1") is False


def test_setThicknessValueFromVariation():
    """test setting thickness from variation"""

    # setup required inputs
    key = "relThPercentVariation"
    cfg = configparser.ConfigParser()
    cfg["GENERAL"] = {
        "secRelArea": "False",
        "relThFromFile": "True",
        "relTh": "",
        "relThPercentVariation": "40$3",
        "relThDistVariation": "",
    }
    cfg["INPUT"] = {
        "relThThickness": "1.2|1.4",
        "relThId": "0|1",
        "releaseScenario": "release1HS",
        "relThCi95": "None|None",
    }

    data = {"relThPercentVariation": [0.6, 1.0, 1.4], "relTh": ["", "", ""], "ind": [0, 1, 2]}
    simDF = pd.DataFrame.from_dict(data)

    for row in simDF.itertuples():
        if row._asdict()["ind"] == 0:
            cfg = dP.setThicknessValueFromVariation(key, cfg, "null", row)

            assert cfg["GENERAL"]["relThPercentVariation"] == "-40.0$1"
            assert cfg["GENERAL"]["relTh0"] == "0.72"
            assert cfg["GENERAL"]["relTh1"] == "0.84"
        elif row._asdict()["ind"] == 1:
            cfg = dP.setThicknessValueFromVariation(key, cfg, "null", row)

            assert cfg["GENERAL"]["relThPercentVariation"] == ""
            assert cfg["GENERAL"]["relTh0"] == "1.2"
            assert cfg["GENERAL"]["relTh1"] == "1.4"

        elif row._asdict()["ind"] == 2:
            cfg = dP.setThicknessValueFromVariation(key, cfg, "null", row)

            assert cfg["GENERAL"]["relThPercentVariation"] == "+39.99999999999999$1"
            assert cfg["GENERAL"]["relTh0"] == "1.68"
            assert cfg["GENERAL"]["relTh1"] == "1.9599999999999997"

    cfg["GENERAL"] = {
        "secRelArea": "False",
        "relThFromFile": "False",
        "relTh": "1.",
        "relThPercentVariation": "40$3",
        "relThDistVariation": "",
    }
    cfg["INPUT"] = {"releaseScenario": "release1HS"}

    data = {"relThPercentVariation": [0.6, 1.0, 1.4], "relTh": ["1.", "1.", "1."], "ind": [0, 1, 2]}
    simDF = pd.DataFrame.from_dict(data)

    for row in simDF.itertuples():
        if row._asdict()["ind"] == 0:
            cfg["GENERAL"]["relTh"] = "1."
            cfg = dP.setThicknessValueFromVariation(key, cfg, "null", row)

            assert cfg["GENERAL"]["relThPercentVariation"] == ""
            assert cfg["GENERAL"]["relTh"] == "0.6"

        elif row._asdict()["ind"] == 1:
            cfg["GENERAL"]["relTh"] = "1."
            cfg = dP.setThicknessValueFromVariation(key, cfg, "null", row)

            assert cfg["GENERAL"]["relThPercentVariation"] == ""
            assert cfg["GENERAL"]["relTh"] == "1.0"

        elif row._asdict()["ind"] == 2:
            cfg["GENERAL"]["relTh"] = "1."
            cfg = dP.setThicknessValueFromVariation(key, cfg, "null", row)

            assert cfg["GENERAL"]["relThPercentVariation"] == ""
            assert cfg["GENERAL"]["relTh"] == "1.4"


def test_splitVariationToArraySteps():
    """testing splitting variation to array steps"""

    # setup required inputs
    value = "40$5"
    key = "relThPercentVariation"

    fullCfg = configparser.ConfigParser()

    itemsArray = dP.splitVariationToArraySteps(value, key, fullCfg)

    assert np.array_equal(itemsArray, np.linspace(0.6, 1.4, 5))

    value = "40$4"
    itemsTest = np.linspace(0.6, 1.4, 4)
    itemsArray = dP.splitVariationToArraySteps(value, key, fullCfg)
    #    print("itemsTest", itemsTest)
    #    print("itemsArray", itemsArray)

    assert np.array_equal(itemsArray, itemsTest)

    value = "2$5"
    key = "relThRangeVariation"
    itemsArray = dP.splitVariationToArraySteps(value, key, fullCfg)

    assert np.array_equal(itemsArray, np.linspace(-2, 2, 5))

    value = "+2$5"
    key = "relThRangeVariation"
    itemsArray = dP.splitVariationToArraySteps(value, key, fullCfg)

    assert np.array_equal(itemsArray, np.linspace(0, 2, 5))

    value = "-2$5"
    key = "relThRangeVariation"
    itemsArray = dP.splitVariationToArraySteps(value, key, fullCfg)

    assert np.array_equal(itemsArray, np.linspace(-2, 0, 5))

    value = "normaldistribution$3$0.1$95$ci95$10000"
    key = "relThDistVariation"
    itemsArray = dP.splitVariationToArraySteps(value, key, fullCfg)

    assert np.array_equal(
        itemsArray,
        [
            "0$normaldistribution$3$0.1$95$ci95$10000",
            "1$normaldistribution$3$0.1$95$ci95$10000",
            "2$normaldistribution$3$0.1$95$ci95$10000",
        ],
    )

    value = "4$normaldistribution$3$0.1$95$ci95$10000"
    key = "relThDistVariation"
    itemsArray = dP.splitVariationToArraySteps(value, key, fullCfg)

    assert np.array_equal(itemsArray, ["4$normaldistribution$3$0.1$95$ci95$10000"])


def test_checkExtentAndCellSize(tmp_path):
    """test if inputFile has close enough extent to DEM file and if so resize if required"""

    # setup required inputs
    testDir = pathlib.Path(tmp_path, "test")
    cfg = configparser.ConfigParser()
    cfg["GENERAL"] = {"resizeThreshold": 3.0, "meshCellSize": 1.0, "meshCellSizeThreshold": 0.0001}
    cfg["GENERAL"]["avalancheDir"] = str(testDir)
    cfg["EXPORTS"] = {"useCompression": "True"}
    inDir = testDir / "Inputs"
    inDirR = inDir / "RASTERS"
    fU.makeADir(inDirR)

    demField = np.ones((4, 5))
    dem = {
        "header": {
            "nrows": 4,
            "ncols": 5,
            "xllcenter": 1,
            "yllcenter": 5,
            "cellsize": 1,
            "nodata_value": -9999,
            "driver": "AAIGrid",
        },
        "rasterData": demField,
    }

    dem["header"]["transform"] = IOf.transformFromASCHeader(dem["header"])
    dem["header"]["crs"] = rasterio.crs.CRS()

    inputFile = inDirR / "inputFile.asc"
    headerInput = {
        "nrows": 4,
        "ncols": 5,
        "xllcenter": 1.2,
        "yllcenter": 4.3,
        "cellsize": 1,
        "nodata_value": -9999,
        "driver": "AAIGrid",
    }
    headerInput["transform"] = IOf.transformFromASCHeader(headerInput)
    headerInput["crs"] = rasterio.crs.CRS()

    inField = np.ones((4, 5))
    inField[2, 2] = 10.0
    IOf.writeResultToRaster(headerInput, inField, inputFile.parent / inputFile.stem, flip=False)

    testFile, outFile, remeshedFlag = dP.checkExtentAndCellSize(cfg, inputFile, dem, "mu")

    newRaster = IOf.readRaster((inDir / testFile))

    assert "remeshedmu1.00" in testFile
    assert newRaster["header"]["nrows"] == 4
    assert newRaster["rasterData"].shape[1] == 5
    assert newRaster["header"]["xllcenter"] == 1.0
    assert newRaster["header"]["yllcenter"] == 5.0
    assert remeshedFlag == "Yes"
    assert outFile.name == testFile.split("/")[1]

    inputFile2 = inDirR / "inputFile1.asc"
    headerInput2 = {
        "nrows": 4,
        "ncols": 5,
        "xllcenter": 1.0,
        "yllcenter": 5.0,
        "cellsize": 1,
        "nodata_value": -9999,
        "driver": "AAIGrid",
    }
    headerInput2["transform"] = IOf.transformFromASCHeader(headerInput2)
    headerInput2["crs"] = rasterio.crs.CRS()
    inField2 = np.ones((4, 5))
    inField2[2, 2] = 10.0
    IOf.writeResultToRaster(headerInput2, inField2, inputFile2.parent / inputFile2.stem, flip=False)

    testFile2, outFile2, remeshedFlag = dP.checkExtentAndCellSize(cfg, inputFile2, dem, "mu")
    # print("test 2", testFile2)
    newRaster2 = IOf.readRaster((inDir / testFile2))

    assert "remeshedmu1.00" not in testFile2
    assert "RASTERS" in testFile2
    assert newRaster2["header"]["nrows"] == 4
    assert newRaster2["rasterData"].shape[1] == 5
    assert newRaster2["header"]["xllcenter"] == 1.0
    assert newRaster2["header"]["yllcenter"] == 5.0

    inputFile2 = inDirR / "inputFile1.asc"
    headerInput2 = {
        "nrows": 5,
        "ncols": 5,
        "xllcenter": 10.0,
        "yllcenter": 5.0,
        "cellsize": 1,
        "nodata_value": -9999,
        "driver": "AAIGrid",
    }
    headerInput2["transform"] = IOf.transformFromASCHeader(headerInput2)
    headerInput2["crs"] = rasterio.crs.CRS()
    inField2 = np.ones((5, 5))
    inField2[2, 2] = 10.0
    IOf.writeResultToRaster(headerInput2, inField2, inputFile2.parent / inputFile2.stem, flip=False)

    with pytest.raises(AssertionError) as e:
        assert dP.checkExtentAndCellSize(cfg, inputFile2, dem, "mu")
    assert "Lower left center coordinates of DEM and " in str(e.value)


# Produced by AI (test):


def test_setRangeFromCiVariation_ciValue_none_raises_error():
    with pytest.raises(AssertionError) as exc_info:
        setRangeFromCiVariation(None, "chi95$0$1", "thValue", "None")
    assert "ci95 values required" in str(exc_info.value)


def test_setRangeFromCiVariation_valid_variation_factor():
    # Test case 1: varValStep=2, allSteps=5, ciValue=10
    variationValue = setRangeFromCiVariation(None, "2$middle$5", "ignore", "10")
    expected_values = np.linspace(-10, 10, 5)
    assert variationValue == expected_values[2]

    # Test case 2: varValStep=0, allSteps=3, ciValue=3
    variationValue = setRangeFromCiVariation(None, "0$x$3", "ignore", "3")
    expected_values = np.linspace(-3, 3, 3)
    assert variationValue == expected_values[0]

    # Test case 3: varValStep=4, allSteps=5, ciValue=5
    variationValue = setRangeFromCiVariation(None, "4$y$5", "ignore", "5")
    expected_values = np.linspace(-5, 5, 5)
    assert variationValue == expected_values[4]


def test_setRangeFromCiVariation_edge_cases():
    # All steps = 1, varValStep=0
    variationValue = setRangeFromCiVariation(None, "0$z$1", "ignore", "8")
    assert variationValue == -8.0

    # varValStep at midpoint
    variationValue = setRangeFromCiVariation(None, "1$mid$3", "ignore", "6")
    expected = np.linspace(-6, 6, 3)[1]
    assert variationValue == expected
