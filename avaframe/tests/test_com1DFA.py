"""
Pytest for module com1DFA
"""

import configparser
import copy
import logging
import pathlib
import pickle
import shutil
import rasterio
import subprocess

#  Load modules
import numpy as np
import pytest

import avaframe.in2Trans.rasterUtils as IOf
import avaframe.in3Utils.fileHandlerUtils as fU
import avaframe.in3Utils.initializeProject as initProj
from avaframe.com1DFA import com1DFA
from avaframe.in2Trans.rasterUtils import transformFromASCHeader
from avaframe.in3Utils import cfgUtils
import avaframe.in3Utils.geoTrans as geoTrans
import avaframe.com1DFA.DFAtools as DFAtls


def test_prepareInputData(tmp_path):
    """test preparing input data"""

    # setup requuired input data
    inputSimFiles = {"entResInfo": {"flagEnt": "Yes", "flagRes": "No", "flagSecondaryRelease": "No"}}
    dirName = pathlib.Path(__file__).parents[0]
    avaDir = dirName / ".." / "data" / "avaAlr"
    relFile = avaDir / "Inputs" / "REL" / "relAlr.shp"
    inputSimFiles["releaseScenario"] = relFile
    inputSimFiles["demFile"] = avaDir / "Inputs" / "avaAlr.tif"
    inputSimFiles["entFile"] = avaDir / "Inputs" / "ENT" / "entAlr.shp"
    inputSimFiles["relThFile"] = ""
    inputSimFiles["entThFile"] = ""
    inputSimFiles["muFile"] = None
    inputSimFiles["xiFile"] = None
    inputSimFiles["kFile"] = None
    inputSimFiles["tauCFile"] = None
    cfg = configparser.ConfigParser()
    cfg["GENERAL"] = {
        "secRelArea": "False",
        "simTypeActual": "ent",
        "avalancheDir": str(avaDir),
    }
    cfg["GENERAL"]["relThFromFile"] = "False"
    cfg["INPUT"] = {"DEM": "avaAlr.tif"}
    cfg["INPUT"]["relThFile"] = ""
    cfg["INPUT"]["entThFile"] = ""

    # call function to be tested
    demOri, inputSimLines = com1DFA.prepareInputData(inputSimFiles, cfg)

    assert demOri["header"]["ncols"] == 417
    assert demOri["header"]["nrows"] == 915
    assert inputSimLines["releaseLine"]["thickness"] == ["1.0"]
    assert inputSimLines["releaseLine"]["Start"] == np.asarray([0.0])
    assert inputSimLines["releaseLine"]["Length"] == np.asarray([33.0])
    assert inputSimLines["releaseLine"]["Name"] == ["AlR"]
    assert inputSimLines["releaseLine"]["initializedFrom"] == "shapefile"
    assert inputSimLines["entLine"]["thickness"] == ["0.3"]
    assert inputSimLines["entLine"]["Start"] == np.asarray([0.0])
    assert inputSimLines["entLine"]["Length"] == np.asarray([48.0])
    assert inputSimLines["entLine"]["Name"] == ["entAlr"]
    assert inputSimLines["resLine"] is None
    assert inputSimLines["entrainmentArea"] == "entAlr.shp"
    assert inputSimLines["entLine"]["initializedFrom"] == "shapefile"

    # call function to be tested
    inputSimFiles = {"entResInfo": {"flagEnt": "No", "flagRes": "Yes", "flagSecondaryRelease": "No"}}
    dirName = pathlib.Path(__file__).parents[0]
    avaDir = dirName / ".." / "data" / "avaParabola"
    relFile = avaDir / "Inputs" / "REL" / "release1PF.shp"
    inputSimFiles["releaseScenario"] = relFile
    inputSimFiles["demFile"] = avaDir / "Inputs" / "DEM_PF_Topo.asc"
    inputSimFiles["resFile"] = avaDir / "Inputs" / "RES" / "resistance1PF.shp"
    inputSimFiles["relThFile"] = None
    inputSimFiles["muFile"] = None
    inputSimFiles["xiFile"] = None
    inputSimFiles["kFile"] = None
    inputSimFiles["tauCFile"] = None
    inputSimFiles["entResInfo"]["resFileType"] = ".shp"
    cfg["GENERAL"]["simTypeActual"] = "res"
    cfg["GENERAL"]["avalancheDir"] = str(avaDir)
    cfg["GENERAL"]["relThFromFile"] = "False"
    cfg["INPUT"] = {"DEM": "DEM_PF_Topo.asc"}
    cfg["INPUT"]["relThFile"] = ""

    demOri, inputSimLines = com1DFA.prepareInputData(inputSimFiles, cfg)

    #    print("inputSimLines", inputSimLines)

    assert inputSimLines["entLine"] is None
    assert inputSimLines["resLine"]["Start"] == np.asarray([0.0])
    assert inputSimLines["resLine"]["Length"] == np.asarray([5.0])
    assert inputSimLines["resLine"]["Name"] == [""]
    assert inputSimLines["resLine"]["initializedFrom"] == "shapefile"

    # call function to be tested
    inputSimFiles = {"entResInfo": {"flagEnt": "No", "flagRes": "Yes", "flagSecondaryRelease": "No"}}
    dirName = pathlib.Path(__file__).parents[0]
    avaDir = dirName / ".." / "data" / "avaParabola"
    relFile = avaDir / "Inputs" / "REL" / "release1PF.shp"
    inputSimFiles["releaseScenario"] = relFile
    inputSimFiles["demFile"] = avaDir / "Inputs" / "DEM_PF_Topo.asc"
    inputSimFiles["resFile"] = avaDir / "Inputs" / "RES" / "resistance1PF.shp"
    inputSimFiles["entResInfo"]["resFileType"] = ".shp"
    inputSimFiles["relThFile"] = dirName / "data" / "relThFieldTestFile.asc"
    inputSimFiles["muFile"] = None
    inputSimFiles["xiFile"] = None
    inputSimFiles["kFile"] = None
    inputSimFiles["tauCFile"] = None
    cfg["GENERAL"]["simTypeActual"] = "res"
    cfg["GENERAL"]["relThFromFile"] = "False"
    cfg["INPUT"]["relThFile"] = ""
    demOri, inputSimLines = com1DFA.prepareInputData(inputSimFiles, cfg)

    assert demOri["header"]["ncols"] == 1001
    assert demOri["header"]["nrows"] == 401
    assert inputSimLines["releaseLine"]["thickness"] == ["1.0"]
    assert inputSimLines["entLine"] is None
    assert inputSimLines["resLine"]["Start"] == np.asarray([0.0])
    assert inputSimLines["resLine"]["Length"] == np.asarray([5.0])
    assert inputSimLines["resLine"]["Name"] == [""]
    assert inputSimLines["relThField"] == ""

    # call function to be tested
    inputSimFiles = {"entResInfo": {"flagEnt": "No", "flagRes": "Yes", "flagSecondaryRelease": "No"}}
    dirName = pathlib.Path(__file__).parents[0]
    avaDir = dirName / ".." / "data" / "avaParabola"
    relFile = avaDir / "Inputs" / "REL" / "release1PF.shp"
    inputSimFiles["releaseScenario"] = relFile
    inputSimFiles["demFile"] = avaDir / "Inputs" / "DEM_PF_Topo.asc"
    inputSimFiles["resFile"] = avaDir / "Inputs" / "RES" / "resistance1PF.shp"
    inputSimFiles["entResInfo"]["resFileType"] = ".shp"
    inputSimFiles["relThFile"] = dirName / "data" / "relThFieldTestFile.asc"
    inputSimFiles["muFile"] = None
    inputSimFiles["xiFile"] = None
    inputSimFiles["kFile"] = None
    inputSimFiles["tauCFile"] = None
    cfg["GENERAL"]["simTypeActual"] = "res"
    cfg["GENERAL"]["relThFromFile"] = "True"
    cfg["INPUT"]["relThFile"] = str(dirName / "data" / "relThFieldTestFile.asc")
    demOri, inputSimLines = com1DFA.prepareInputData(inputSimFiles, cfg)

    #    print("inputSimLines", inputSimLines)

    assert inputSimLines["entLine"] is None
    assert inputSimLines["resLine"]["Start"] == np.asarray([0.0])
    assert inputSimLines["resLine"]["Length"] == np.asarray([5.0])
    assert inputSimLines["resLine"]["Name"] == [""]
    assert inputSimLines["relThField"].shape[0] == 401
    assert inputSimLines["relThField"].shape[1] == 1001
    assert inputSimLines["releaseLine"]["initializedFrom"] == "raster"
    assert inputSimLines["releaseLine"]["Name"] == "from raster"
    assert inputSimLines["releaseLine"]["thickness"] == "from raster"
    assert inputSimLines["releaseLine"]["file"] == dirName / "data" / "relThFieldTestFile.asc"
    assert inputSimLines["releaseLine"]["type"] == "Release from raster"

    # call function to be tested
    inputSimFiles = {"entResInfo": {"flagEnt": "No", "flagRes": "Yes", "flagSecondaryRelease": "No"}}
    dirName = pathlib.Path(__file__).parents[0]
    avaDir = dirName / ".." / "data" / "avaParabola"
    relFile = avaDir / "Inputs" / "REL" / "release1PF.shp"
    inputSimFiles["releaseScenario"] = relFile
    inputSimFiles["demFile"] = avaDir / "Inputs" / "DEM_PF_Topo.asc"
    inputSimFiles["resFile"] = avaDir / "Inputs" / "RES" / "resistance1PF.shp"
    inputSimFiles["entResInfo"]["resFileType"] = ".shp"
    inputSimFiles["muFile"] = None
    inputSimFiles["xiFile"] = None
    inputSimFiles["kFile"] = None
    inputSimFiles["tauCFile"] = None
    testField = np.zeros((10, 10))
    testFile = pathlib.Path(tmp_path, "testFile2")

    testHeader = {
        "ncols": 10,
        "nrows": 10,
        "cellsize": 5,
        "xllcenter": 0.0,
        "yllcenter": 0.0,
        "nodata_value": 0.0,
        "driver": "AAIGrid",
    }
    transform = rasterio.transform.from_origin(0 - 5 / 2, (0 - 5 / 2) + 10 * 5, 5, 5)
    crs = rasterio.crs.CRS()
    testHeader["transform"] = transform
    testHeader["crs"] = crs
    IOf.writeResultToRaster(testHeader, testField, testFile, flip=True)
    inputSimFiles["relThFile"] = str(testFile) + ".asc"
    cfg["GENERAL"]["simTypeActual"] = "res"
    cfg["GENERAL"]["relThFromFile"] = "True"
    cfg["INPUT"]["relThFile"] = str(testFile) + ".asc"

    # with pytest.raises(AssertionError) as e:
    #     assert com1DFA.prepareInputData(inputSimFiles, cfg)
    # assert str(e.value) == (
    #     "Release thickness field read from %s does not match the number of rows and columns of the dem"
    #     % inputSimFiles["relThFile"]
    # )

    # setup required input data
    inputSimFiles = {"entResInfo": {"flagEnt": "No", "flagRes": "No", "flagSecondaryRelease": "No"}}
    dirName = pathlib.Path(__file__).parents[0]
    avaDir = dirName / "data" / "avaTestRelTh"
    relFile = avaDir / "Inputs" / "REL" / "rel1.shp"
    inputSimFiles["releaseScenario"] = relFile
    inputSimFiles["demFile"] = avaDir / "Inputs" / "testDEM.asc"
    inputSimFiles["relThFile"] = avaDir / "Inputs" / "REL" / "testRel2.asc"
    inputSimFiles["muFile"] = None
    inputSimFiles["xiFile"] = None
    inputSimFiles["kFile"] = None
    inputSimFiles["tauCFile"] = None
    cfg = configparser.ConfigParser()
    cfg["GENERAL"] = {
        "secRelArea": "False",
        "simTypeActual": "null",
        "avalancheDir": str(avaDir),
        "relThFromFile": "True",
    }
    cfg["INPUT"] = {"DEM": "testDEM.asc"}
    cfg["INPUT"]["relThFile"] = str(inputSimFiles["relThFile"])

    demOri, inputSimLines = com1DFA.prepareInputData(inputSimFiles, cfg)

    print("inputSimLines----------", inputSimLines)

    assert inputSimLines["entLine"] is None
    assert inputSimLines["resLine"] == None
    assert inputSimLines["relThField"].shape[0] == 22
    assert inputSimLines["relThField"].shape[1] == 20
    assert np.amax(inputSimLines["relThField"]) == 2.0
    assert np.isclose(np.mean(inputSimLines["relThField"]), 0.0590909)
    assert np.amin(inputSimLines["relThField"]) == 0.0
    assert demOri["header"]["ncols"] == 20
    assert demOri["header"]["nrows"] == 22
    assert inputSimLines["releaseLine"]["thickness"] == "from raster"
    assert "Start" not in inputSimLines["releaseLine"]
    assert inputSimLines["releaseLine"]["Name"] == "from raster"
    assert "ci95" not in inputSimLines["releaseLine"]
    assert inputSimLines["releaseLine"]["initializedFrom"] == "raster"

    # setup requuired input data
    inputSimFiles = {"entResInfo": {"flagEnt": "No", "flagRes": "No", "flagSecondaryRelease": "Yes"}}
    dirName = pathlib.Path(__file__).parents[0]
    avaDir = dirName / "data" / "avaTestRelTh"
    relFile = avaDir / "Inputs" / "REL" / "rel1.shp"
    secrelFile = avaDir / "Inputs" / "SECREL" / "testSecRel2.asc"
    inputSimFiles["releaseScenario"] = relFile
    inputSimFiles["secondaryRelScenario"] = secrelFile
    inputSimFiles["demFile"] = avaDir / "Inputs" / "testDEM.asc"
    inputSimFiles["relThFile"] = None
    inputSimFiles["secondaryRelThFile"] = avaDir / "Inputs" / "SECREL" / "testSecRel2.asc"
    inputSimFiles["muFile"] = None
    inputSimFiles["xiFile"] = None
    inputSimFiles["kFile"] = None
    inputSimFiles["tauCFile"] = None
    cfg = configparser.ConfigParser()
    cfg["GENERAL"] = {
        "secRelArea": "True",
        "simTypeActual": "null",
        "avalancheDir": str(avaDir),
        "relThFromFile": "True",
        "relTh": "",
    }
    cfg["INPUT"] = {"DEM": "testDEM.asc"}
    cfg["INPUT"]["relThFile"] = ""
    cfg["INPUT"]["secondaryRelThFile"] = str(inputSimFiles["secondaryRelThFile"])

    demOri, inputSimLines = com1DFA.prepareInputData(inputSimFiles, cfg)

    #    print("inputSimLines", inputSimLines)

    assert inputSimLines["entLine"] is None
    assert inputSimLines["resLine"] == None
    assert inputSimLines["relThField"] == ""
    assert demOri["header"]["ncols"] == 20
    assert demOri["header"]["nrows"] == 22
    assert inputSimLines["releaseLine"]["thickness"] == ["1.5", "0.7"]
    assert np.array_equal(inputSimLines["releaseLine"]["Start"], np.asarray([0.0, 9.0]))
    assert np.array_equal(inputSimLines["releaseLine"]["Length"], np.asarray([9.0, 5.0]))
    assert inputSimLines["releaseLine"]["Name"] == ["releaseNew1", "releaseNew2"]
    assert inputSimLines["releaseLine"]["ci95"] == ["0.4", "0.1"]
    assert inputSimLines["secondaryReleaseLine"]["Name"] == "from raster"
    assert inputSimLines["secondaryReleaseLine"]["thickness"] == "from raster"
    assert inputSimLines["secondaryReleaseLine"]["initializedFrom"] == "raster"
    assert inputSimLines["secondaryReleaseLine"]["type"] == "Secondary release from raster"
    assert inputSimLines["releaseLine"]["type"] == "Release"
    assert inputSimLines["releaseLine"]["initializedFrom"] == "shapefile"

    # setup requuired input data
    inputSimFiles = {"entResInfo": {"flagEnt": "No", "flagRes": "No", "flagSecondaryRelease": "No"}}
    dirName = pathlib.Path(__file__).parents[0]
    avaDir = dirName / "data" / "avaTestRelTh"
    relFile = avaDir / "Inputs" / "REL" / "testRel2.asc"
    inputSimFiles["releaseScenario"] = relFile
    inputSimFiles["demFile"] = avaDir / "Inputs" / "testDEM.asc"
    inputSimFiles["relThFile"] = avaDir / "Inputs" / "REL" / "testRel2.asc"
    inputSimFiles["muFile"] = None
    inputSimFiles["xiFile"] = None
    inputSimFiles["kFile"] = None
    inputSimFiles["tauCFile"] = None
    cfg = configparser.ConfigParser()
    cfg["GENERAL"] = {
        "secRelArea": "False",
        "simTypeActual": "null",
        "avalancheDir": str(avaDir),
        "relThFromFile": "True",
    }
    cfg["INPUT"] = {"DEM": "testDEM.asc"}
    cfg["INPUT"]["relThFile"] = str(inputSimFiles["relThFile"])

    # with pytest.raises(AssertionError) as e:
    #     assert com1DFA.prepareInputData(inputSimFiles, cfg)
    # assert str(e.value) == (
    #     "Release thickness field contains nans - not allowed no release thickness must be set to 0"
    # )

    testDir = pathlib.Path(__file__).parents[0]
    avaDir = pathlib.Path(tmp_path, "avaTestHoles")
    avaDir = dirName / ".." / "data" / "avaAlr"

    relFile = testDir / "data" / "testForHoles" / "relAlr2.shp"
    inputSimFiles["releaseScenario"] = relFile
    inputSimFiles["relThFile"] = ""
    inputSimFiles["muFile"] = None
    inputSimFiles["xiFile"] = None
    inputSimFiles["kFile"] = None
    inputSimFiles["tauCFile"] = None
    cfg = configparser.ConfigParser()
    cfg["GENERAL"] = {
        "secRelArea": "False",
        "simTypeActual": "null",
        "avalancheDir": str(avaDir),
    }
    cfg["INPUT"] = {"DEM": "avaAlr.tif"}
    cfg["INPUT"]["relThFile"] = ""

    with pytest.raises(AssertionError) as e:
        assert com1DFA.prepareInputData(inputSimFiles, cfg)
    assert "One or more release features in relAlr2.shp have holes - check error plots in" in str(e.value)


def test_prepareReleaseEntrainment(tmp_path):
    """test preparing release areas"""

    # setup required inputs
    cfg = configparser.ConfigParser()
    cfg["GENERAL"] = {
        "secRelArea": "True",
        "relThFromFile": "False",
        "secondaryRelThFromFile": "True",
        "relTh": "1.32",
        "secondaryRelTh0": "1.789",
        "secondaryRelThPercentVariation": "0.7",
        "simTypeActual": "null",
    }
    cfg["INPUT"] = {
        "secondaryRelThThickness": "1.2523",
        "secondaryRelThId": "0",
        "thFromIni": "",
        "relThFile": "",
        "secondaryRelThFile": "",
    }

    inputSimLines = {}
    inputSimLines["entResInfo"] = {"flagSecondaryRelease": "Yes", "flagEnt": "No"}
    inputSimLines["releaseLine"] = {
        "thickness": ["None", "None"],
        "type": "Release",
        "id": ["0", "1"],
        "initializedFrom": "shapefile",
    }
    inputSimLines["relThField"] = ""
    inputSimLines["secondaryReleaseLine"] = {
        "thickness": ["1.2523"],
        "type": "Secondary release",
        "id": ["0"],
        "initializedFrom": "shapefile",
    }
    rel = pathlib.Path(tmp_path, "release1PF_test.shp")

    # call function to be tested
    relName, inputSimLines, badName = com1DFA.prepareReleaseEntrainment(cfg, rel, inputSimLines)

    assert relName == "release1PF_test"
    assert inputSimLines["entResInfo"]["flagSecondaryRelease"] == "Yes"
    assert inputSimLines["releaseLine"]["thickness"] == [1.32, 1.32]
    assert inputSimLines["secondaryReleaseLine"]["thickness"] == [1.789]
    assert inputSimLines["releaseLine"]["thicknessSource"] == ["ini file", "ini file"]
    assert inputSimLines["secondaryReleaseLine"]["thicknessSource"] == ["shp file"]
    assert badName is True

    # setup required inputs
    cfg["GENERAL"]["secondaryRelThFromFile"] = "False"
    cfg["GENERAL"]["relThFromFile"] = "True"
    cfg["GENERAL"]["secondaryRelTh"] = "2.5"
    cfg["GENERAL"]["relTh"] = ""
    cfg["GENERAL"]["secondaryRelThPercentVariation"] = ""
    cfg["GENERAL"]["relThPercentVariation"] = ""
    cfg["INPUT"] = {"relThThickness": "1.78|4.328", "relThId": "0|1", "thFromIni": ""}
    cfg["GENERAL"]["relTh0"] = "1.78"
    cfg["GENERAL"]["relTh1"] = "4.328"
    cfg["INPUT"]["relThFile"] = ""
    cfg["INPUT"]["secondaryRelThFile"] = ""

    inputSimLines = {}
    inputSimLines["entResInfo"] = {"flagSecondaryRelease": "Yes", "flagEnt": "No"}
    inputSimLines["releaseLine"] = {
        "thickness": ["1.78", "4.328"],
        "type": "release",
        "id": ["0", "1"],
        "initializedFrom": "shapefile",
    }
    inputSimLines["relThFile"] = ""
    inputSimLines["secondaryReleaseLine"] = {
        "thickness": ["None"],
        "type": "Secondary release",
        "id": ["0"],
        "initializedFrom": "shapefile",
    }
    inputSimLines["secondaryRelThFile"] = ""
    inputSimLines["relThField"] = ""
    rel = pathlib.Path(tmp_path, "release1PF_test.shp")

    # call function to be tested
    relName2, inputSimLines2, badName2 = com1DFA.prepareReleaseEntrainment(cfg, rel, inputSimLines)

    assert relName2 == "release1PF_test"
    assert inputSimLines2["entResInfo"]["flagSecondaryRelease"] == "Yes"
    assert inputSimLines2["releaseLine"]["thickness"] == [1.78, 4.328]
    assert inputSimLines2["secondaryReleaseLine"]["thickness"] == [2.5]
    assert inputSimLines2["secondaryReleaseLine"]["thicknessSource"] == ["ini file"]
    assert inputSimLines2["releaseLine"]["thicknessSource"] == ["shp file", "shp file"]
    assert badName2 is True

    # setup required inputs
    cfg["GENERAL"]["secondaryRelThFromFile"] = "True"
    cfg["GENERAL"]["relThFromFile"] = "True"
    cfg["GENERAL"]["secondaryRelTh"] = ""
    cfg["INPUT"]["secondaryRelThThickness"] = "2.7"
    cfg["INPUT"]["secondaryRelThId"] = "0"
    cfg["GENERAL"]["secondaryRelTh0"] = "2.7"
    cfg["GENERAL"]["relTh"] = ""
    cfg["GENERAL"]["relTh0"] = "0.5"
    cfg["GENERAL"]["relTh1"] = "1."
    cfg["GENERAL"]["secondaryRelThPercentVariation"] = ""
    cfg["GENERAL"]["relThPercentVariation"] = "0.5"
    cfg["INPUT"]["relThThickness"] = "1|2"
    cfg["INPUT"]["relThId"] = "0|1"

    inputSimLines = {}
    inputSimLines["entResInfo"] = {"flagSecondaryRelease": "Yes", "flagEnt": "No"}
    inputSimLines["releaseLine"] = {
        "thickness": ["1.", "2."],
        "type": "release",
        "id": ["0", "1"],
        "initializedFrom": "shapefile",
    }
    inputSimLines["relThFile"] = ""
    inputSimLines["secondaryReleaseLine"] = {
        "thickness": ["2.7"],
        "type": "Secondary release",
        "id": ["0"],
        "initializedFrom": "shapefile",
    }
    inputSimLines["secondaryRelThFile"] = ""
    rel = pathlib.Path(tmp_path, "release1PF_test.shp")

    # call function to be tested
    relName2, inputSimLines2, badName2 = com1DFA.prepareReleaseEntrainment(cfg, rel, inputSimLines)

    # print(
    #      "Test",
    #      cfg["GENERAL"]["secondaryRelTh"],
    #      cfg["GENERAL"]["secondaryRelThFromFile"],
    #      cfg["GENERAL"]["secondaryRelTh0"],
    #  )
    #    print("inputSimLines", inputSimLines2)
    assert relName2 == "release1PF_test"
    assert inputSimLines2["entResInfo"]["flagSecondaryRelease"] == "Yes"
    assert inputSimLines2["releaseLine"]["thickness"] == [0.5, 1.0]
    assert inputSimLines2["secondaryReleaseLine"]["thickness"] == [2.7]
    assert inputSimLines2["secondaryReleaseLine"]["thicknessSource"] == ["shp file"]
    assert inputSimLines2["releaseLine"]["thicknessSource"] == ["shp file", "shp file"]
    assert badName2 is True

    # call function to be tested
    cfg["GENERAL"]["secRelArea"] = "False"
    relName3, inputSimLines3, badName3 = com1DFA.prepareReleaseEntrainment(cfg, rel, inputSimLines)

    assert relName3 == "release1PF_test"
    assert inputSimLines3["entResInfo"]["flagSecondaryRelease"] == "No"
    assert inputSimLines3["releaseLine"]["thickness"] == [0.5, 1.0]
    assert inputSimLines3["releaseLine"]["thicknessSource"] == ["shp file", "shp file"]
    assert inputSimLines3["secondaryReleaseLine"] is None
    assert badName3 is True

    # setup required inputs
    inputSimLines = {}
    inputSimLines["entResInfo"] = {"flagSecondaryRelease": "No", "flagEnt": "No"}
    inputSimLines["releaseLine"] = {
        "thickness": ["1.78", "4.328"],
        "type": "release",
        "id": ["0", "1"],
        "initializedFrom": "shapefile",
    }
    inputSimLines["relThFile"] = ""
    rel = pathlib.Path(tmp_path, "release1PF_test.shp")
    cfg["GENERAL"]["relThFromFile"] = "False"
    cfg["GENERAL"]["relTh"] = "1.32"

    # call function to test
    relName4, inputSimLines4, badName4 = com1DFA.prepareReleaseEntrainment(cfg, rel, inputSimLines)

    assert relName4 == "release1PF_test"
    assert inputSimLines4["entResInfo"]["flagSecondaryRelease"] == "No"
    assert inputSimLines4["releaseLine"]["thickness"] == [1.32, 1.32]
    assert inputSimLines4["secondaryReleaseLine"] is None
    assert inputSimLines4["releaseLine"]["thicknessSource"] == ["ini file", "ini file"]

    # call function to test
    cfg["GENERAL"] = {
        "secRelArea": "False",
        "relThFromFile": "False",
        "entThFromFile": "True",
        "relTh": "1.32",
        "secondaryRelTh": "2.5",
        "entTh0": "0.4",
        "entTh1": "0.3",
        "entTh": "",
        "simTypeActual": "ent",
        "entThPercentVariation": "1.5",
    }
    cfg["INPUT"] = {
        "relThFile": "",
        "entThFile": "",
        "releaseScenario": "test",
        "thFromIni": "",
    }
    inputSimLines = {}
    inputSimLines["entResInfo"] = {"flagSecondaryRelease": "No", "flagEnt": "Yes"}
    inputSimLines["releaseLine"] = {
        "thickness": ["None", "None"],
        "type": "Release",
        "id": ["0", "1"],
        "initializedFrom": "shapefile",
    }
    inputSimLines["relThFile"] = ""
    inputSimLines["entThFile"] = ""
    inputSimLines["entLine"] = {
        "thickness": ["1.20", "0.9"],
        "type": "Entrainment",
        "id": ["0", "1"],
        "initializedFrom": "shapefile",
    }
    relName5, inputSimLines5, badName5 = com1DFA.prepareReleaseEntrainment(cfg, rel, inputSimLines)

    assert relName5 == "release1PF_test"
    assert inputSimLines5["entResInfo"]["flagSecondaryRelease"] == "No"
    assert inputSimLines5["releaseLine"]["thickness"] == [1.32, 1.32]
    assert inputSimLines5["entLine"]["thickness"] == [0.4, 0.3]
    assert inputSimLines5["secondaryReleaseLine"] is None
    assert inputSimLines5["entLine"]["thicknessSource"] == ["shp file", "shp file"]
    assert inputSimLines5["releaseLine"]["thicknessSource"] == ["ini file", "ini file"]


def test_setThickness():
    """test setting thickness to line dicts"""

    # setup required input
    cfg = configparser.ConfigParser()
    cfg["GENERAL"] = {
        "entThFromFile": "False",
        "entTh": "1.0",
        "entThPercentVariation": "",
    }
    cfg["INPUT"] = {"thFromIni": ""}

    lineTh = {
        "Name": ["testRel", "test2"],
        "Start": np.asarray([0.0, 5]),
        "Length": np.asarray([5, 5]),
        "thickness": ["None", "None"],
        "x": np.asarray([0, 10.0, 10.0, 0.0, 0.0, 20.0, 26.0, 26.0, 20.0, 20.0]),
        "y": np.asarray([0.0, 0.0, 10.0, 10.0, 0.0, 21.0, 21.0, 27.0, 27.0, 21.0]),
        "type": "Entrainment",
        "id": ["0", "1"],
    }

    typeTh = "entTh"

    # call function to be tested
    lineTh = com1DFA.setThickness(cfg, lineTh, typeTh)

    assert lineTh["thickness"] == [1.0, 1.0]
    assert lineTh["thicknessSource"] == ["ini file", "ini file"]
    assert np.array_equal(lineTh["x"], np.asarray([0, 10.0, 10.0, 0.0, 0.0, 20.0, 26.0, 26.0, 20.0, 20.0]))

    # call function to be tested
    lineTh = {
        "Name": ["testRel", "test2"],
        "Start": np.asarray([0.0, 5]),
        "Length": np.asarray([5, 5]),
        "thickness": ["None", "0.7"],
        "x": np.asarray([0, 10.0, 10.0, 0.0, 0.0, 20.0, 26.0, 26.0, 20.0, 20.0]),
        "id": ["0", "1"],
        "y": np.asarray([0.0, 0.0, 10.0, 10.0, 0.0, 21.0, 21.0, 27.0, 27.0, 21.0]),
        "type": "Entrainment",
    }
    lineTh = com1DFA.setThickness(cfg, lineTh, typeTh)

    assert lineTh["thickness"] == [1.0, 1.0]
    assert lineTh["thicknessSource"] == ["ini file", "ini file"]
    assert np.array_equal(lineTh["x"], np.asarray([0, 10.0, 10.0, 0.0, 0.0, 20.0, 26.0, 26.0, 20.0, 20.0]))

    # call function to be tested
    cfg["GENERAL"]["entThFromFile"] = "True"
    cfg["GENERAL"]["entTh0"] = "1.0"
    cfg["GENERAL"]["entTh1"] = "0.7"
    cfg["GENERAL"]["entThPercentVariation"] = "0.5"
    cfg["INPUT"]["entThId"] = "0|1"
    cfg["INPUT"]["entThThickness"] = "2|1.4"

    lineTh = {
        "Name": ["testRel", "test2"],
        "Start": np.asarray([0.0, 5]),
        "Length": np.asarray([5, 5]),
        "thickness": ["1.0", "0.7"],
        "x": np.asarray([0, 10.0, 10.0, 0.0, 0.0, 20.0, 26.0, 26.0, 20.0, 20.0]),
        "id": ["0", "1"],
        "y": np.asarray([0.0, 0.0, 10.0, 10.0, 0.0, 21.0, 21.0, 27.0, 27.0, 21.0]),
        "type": "Entrainment",
    }
    lineTh = com1DFA.setThickness(cfg, lineTh, typeTh)

    assert lineTh["thickness"] == [1.0, 0.7]
    assert lineTh["thicknessSource"] == ["shp file", "shp file"]
    assert np.array_equal(lineTh["x"], np.asarray([0, 10.0, 10.0, 0.0, 0.0, 20.0, 26.0, 26.0, 20.0, 20.0]))

    # call function to be tested
    cfg["GENERAL"]["entThFromFile"] = "True"
    cfg["GENERAL"]["entTh0"] = "1.2"
    cfg["GENERAL"]["entTh1"] = "0.7"
    lineTh = {
        "Name": ["testRel", "test2"],
        "Start": np.asarray([0.0, 5]),
        "Length": np.asarray([5, 5]),
        "thickness": ["1.2", "0.7"],
        "id": ["0", "1"],
        "x": np.asarray([0, 10.0, 10.0, 0.0, 0.0, 20.0, 26.0, 26.0, 20.0, 20.0]),
        "y": np.asarray([0.0, 0.0, 10.0, 10.0, 0.0, 21.0, 21.0, 27.0, 27.0, 21.0]),
        "type": "Entrainment",
    }
    lineTh = com1DFA.setThickness(cfg, lineTh, typeTh)

    assert lineTh["thickness"] == [1.2, 0.7]
    assert lineTh["thicknessSource"] == ["shp file", "shp file"]
    assert np.array_equal(lineTh["x"], np.asarray([0, 10.0, 10.0, 0.0, 0.0, 20.0, 26.0, 26.0, 20.0, 20.0]))


def test_createReportDict():
    """test creating a report dictionary"""

    # setup required input
    avaDir = "data/avaTest"
    logName = "testName"
    relName = "relTest"
    inputSimLines = {
        "entrainmentArea": "entTest",
        "resistanceArea": "resTest",
        "releaseLine": {"Name": "relTestFeature", "thickness": ["1.45"]},
        "entLine": {"Name": ["entTest"], "thickness": ["1.1"]},
    }
    reportAreaInfo = {
        "entrainment": "Yes",
        "resistance": "Yes",
        "Release area info": {"Projected Area [m2]": "m2"},
        "secRelArea": "No",
    }
    cfg = configparser.ConfigParser()
    cfg["GENERAL"] = {
        "musamosat": "0.15500",
        "tau0samosat": "0",
        "Rs0samosat": "0.222",
        "kappasamosat": "0.43",
        "Rsamosat": "0.05",
        "Bsamosat": "4.13",
        "rho": "200.",
        "frictModel": "samosAT",
        "entTh": "0.3",
        "rhoEnt": "100.0",
    }

    # call function to be tested
    reportST = com1DFA.createReportDict(avaDir, logName, relName, inputSimLines, cfg, reportAreaInfo)

    assert "Simulation Parameters" in reportST
    assert "Program version" in reportST["Simulation Parameters"]
    assert reportST["avaName"]["name"] == avaDir
    assert reportST["simName"]["name"] == logName
    assert reportST["Simulation Parameters"]["Release Area Scenario"] == relName
    assert reportST["Simulation Parameters"]["Entrainment"] == "Yes"
    assert reportST["Simulation Parameters"]["Resistance"] == "Yes"
    assert reportST["Friction model"]["mu"] == "0.15500"
    assert reportST["Simulation Parameters"]["Density [kgm-3]"] == "200."
    assert reportST["Simulation Parameters"]["Friction model"] == "samosAT"
    assert reportST["Release Area"]["Release area scenario"] == relName
    assert reportST["Release Area"]["Release Area"] == "relTestFeature"
    assert reportST["Release Area"]["Release thickness [m]"] == ["1.45"]
    assert reportST["Entrainment area"]["Entrainment area scenario"] == "entTest"
    assert "Projected Area [m2]" in reportST["Release Area"]


def test_reportAddTimeMassInfo():
    """test adding mass and time info to report dict"""

    # setup required input
    reportDict = {"Simulation Parameters": {"testItem": 1.0}}
    tcpuDFA = 400.0
    infoDict = {
        "initial mass": 400000.2345,
        "final mass": 400000.8345,
        "entrained mass": 0.8,
        "entrained volume": 0.2,
        "stopInfo": {"Stop criterion": "0.1 percent of PKE"},
    }

    # call function to be tested
    reportDict = com1DFA.reportAddTimeMassInfo(reportDict, tcpuDFA, infoDict)

    assert reportDict["Simulation Parameters"]["testItem"] == 1.0
    assert reportDict["Simulation Parameters"]["Initial mass [kg]"] == "400000.23"
    assert reportDict["Simulation Parameters"]["Final mass [kg]"] == "400000.83"
    assert reportDict["Simulation Parameters"]["Entrained mass [kg]"] == "0.80"
    assert reportDict["Simulation Parameters"]["Entrained volume [m3]"] == "0.20"
    assert reportDict["Simulation Parameters"]["Stop criterion"] == "0.1 percent of PKE"


def test_initializeMassEnt():
    """test initializing entrainment area"""

    # setup required input
    nrows = 110
    ncols = 150
    demHeader = {}
    demHeader["xllcenter"] = 0.0
    demHeader["yllcenter"] = 0.0
    demHeader["cellsize"] = 1.0
    demHeader["nodata_value"] = -9999
    demHeader["nrows"] = nrows
    demHeader["ncols"] = ncols
    dem = {"header": demHeader}
    dem["rasterData"] = np.ones((nrows, ncols))
    dem["originalHeader"] = dem["header"]
    dem["header"]["xllcenter"] = 0.0
    dem["header"]["yllcenter"] = 0.0

    cfg = configparser.ConfigParser()
    cfg["GENERAL"] = {
        "rhoEnt": "200.0",
        "entTempRef": "-10.",
        "cpIce": "2050.",
        "TIni": "-10.",
    }
    cfg["EXPORTS"] = {"exportRasters": False}

    simTypeActual = "entres"
    dirName = pathlib.Path(__file__).parents[0]
    fileName = dirName / "testEnt.shp"
    entLine = {
        "fileName": fileName,
        "Name": ["testEnt"],
        "Start": np.asarray([0.0]),
        "Length": np.asarray([5]),
        "x": np.asarray([0, 10.0, 10.0, 0.0, 0.0]),
        "y": np.asarray([0.0, 0.0, 10.0, 10.0, 0.0]),
        "type": "Entrainment",
        "thickness": [1.0],
        "thicknessSource": ["ini File"],
        "initializedFrom": "shapefile",
    }
    reportAreaInfo = {
        "entrainment": "",
    }
    thresholdPointInPoly = 0.001

    # call function to be tested
    entrMassRaster, entrEnthRaster, reportAreaInfo = com1DFA.initializeMassEnt(
        dem,
        simTypeActual,
        entLine,
        reportAreaInfo,
        thresholdPointInPoly,
        cfg,
    )
    testData = np.zeros((nrows, ncols))
    testData[0:11, 0:11] = 1.0 * 200.0
    testEnt = np.zeros((nrows, ncols))
    testEnt[0:11, 0:11] = -10.0 * 2050.0
    #    print("data", testData)
    #    print("ent", entrMassRaster, entLine)

    assert np.array_equal(entrMassRaster, testData)
    assert np.array_equal(entrEnthRaster, testEnt)
    assert np.sum(entrMassRaster) == 24200.0
    assert entrMassRaster.shape[0] == nrows
    assert reportAreaInfo["entrainment"] == "Yes"

    # call function to be tested
    simTypeActual = "res"
    entrMassRaster, entrEnthRaster, reportAreaInfo = com1DFA.initializeMassEnt(
        dem,
        simTypeActual,
        entLine,
        reportAreaInfo,
        thresholdPointInPoly,
        cfg,
    )

    assert np.array_equal(entrMassRaster, np.zeros((nrows, ncols)))
    assert reportAreaInfo["entrainment"] == "No"


def test_initializeResistance():
    """test initializing resistance area"""

    # setup required input
    cfg = configparser.ConfigParser()
    cfg["GENERAL"] = {
        "cResH": 0.003,
        "ResistanceModel": "default",
        "detK": 10,
        "detrainment": False,
        "forestVMin": 6.0,
        "forestThMin": 0.6,
        "forestVMax": 40.0,
        "forestThMax": 10.0,
    }

    nrows = 11
    ncols = 15
    demHeader = {}
    demHeader["nrows"] = nrows
    demHeader["ncols"] = ncols
    demHeader["xllcenter"] = 0.0
    demHeader["yllcenter"] = 0.0
    demHeader["cellsize"] = 1.0
    demHeader["nodata_value"] = -9999
    dem = {"header": demHeader}
    dem["rasterData"] = np.ones((nrows, ncols))

    simTypeActual = "entres"
    resLine = {
        "fileName": "resTest",
        "Start": np.asarray([0]),
        "Length": np.asarray([5]),
        "Name": ["resTestFeat"],
        "type": "resistance",
        "x": np.asarray([0, 10.0, 10.0, 0.0, 0.0]),
        "y": np.asarray([0.0, 0.0, 10.0, 10.0, 0.0]),
        "initializedFrom": "shapefile",
    }
    reportAreaInfo = {"entrainment": "Yes", "resistance": "No"}
    thresholdPointInPoly = 0.01

    # call function to be tested
    dem["originalHeader"] = dem["header"]
    dem["header"]["xllcenter"] = 0.0
    dem["header"]["yllcenter"] = 0.0
    cResRaster, detRaster, reportAreaInfo = com1DFA.initializeResistance(
        cfg["GENERAL"],
        dem,
        simTypeActual,
        resLine,
        reportAreaInfo,
        thresholdPointInPoly,
    )
    testArray = np.zeros((nrows, ncols))
    testArray[0:11, 0:11] = 0.003

    #    print("cResRaster", cResRaster)
    #    print("reportAreaInfo", reportAreaInfo)

    assert np.array_equal(cResRaster, testArray)
    assert np.array_equal(detRaster, np.zeros((nrows, ncols)))
    assert np.sum(detRaster) == 0.0
    assert np.sum(cResRaster) == 0.363
    assert reportAreaInfo["resistance"] == "Yes"
    assert reportAreaInfo["detrainment"] == "No"

    cfg["GENERAL"] = {
        "cResH": 0.003,
        "ResistanceModel": "default",
        "detK": 10.0,
        "detrainment": True,
        "forestVMin": 6.0,
        "forestThMin": 0.6,
        "forestVMax": 40.0,
        "forestThMax": 10.0,
    }
    cResRaster, detRaster, reportAreaInfo = com1DFA.initializeResistance(
        cfg["GENERAL"],
        dem,
        simTypeActual,
        resLine,
        reportAreaInfo,
        thresholdPointInPoly,
    )
    detTestArray = np.zeros((nrows, ncols))
    detTestArray[0:11, 0:11] = 10.0

    assert np.array_equal(cResRaster, testArray)
    assert np.array_equal(detRaster, detTestArray)
    assert np.sum(detRaster) == 1210.0
    assert np.sum(cResRaster) == 0.363
    assert reportAreaInfo["resistance"] == "Yes"
    assert reportAreaInfo["detrainment"] == "Yes"

    cfg["GENERAL"] = {
        "cResH": 0.003,
        "ResistanceModel": "cRes",
        "detK": 10.0,
        "detrainment": True,
        "forestVMin": 6.0,
        "forestThMin": 0.6,
        "forestVMax": 40.0,
        "forestThMax": 10.0,
    }

    with pytest.raises(AssertionError) as e:
        assert com1DFA.initializeResistance(
            cfg["GENERAL"],
            dem,
            simTypeActual,
            resLine,
            reportAreaInfo,
            thresholdPointInPoly,
        )
    assert "Resistance model cres not a valid option" in str(e.value)


def test_setDEMOriginToZero():
    """test if origin is set to zero"""

    # setup required input
    tHeader = {}
    tHeader["xllcenter"] = 10.0
    tHeader["yllcenter"] = 4.0
    dem = {"header": tHeader}

    # call function to be tested
    demTest = com1DFA.setDEMoriginToZero(dem)

    assert demTest["header"]["xllcenter"] == 0.0
    assert demTest["header"]["yllcenter"] == 0.0


def test_initializeMesh():
    """test mesh initialization"""

    # setup required input
    demHeader = {}
    demHeader["xllcenter"] = 101.23
    demHeader["yllcenter"] = 24.54
    demHeader["cellsize"] = 1.0
    demHeader["nodata_value"] = -9999
    demHeader["nrows"] = 5
    demHeader["ncols"] = 5

    # define plane with constant slope of 45°
    demData = np.asarray(
        [
            [1.0, 2.0, 3.0, 4.0, np.nan],
            [1.0, 2.0, 3.0, 4.0, np.nan],
            [1.0, 2.0, 3.0, 4.0, np.nan],
            [1.0, 2.0, 3.0, 4.0, np.nan],
            [1.0, 2.0, 3.0, 4.0, np.nan],
        ]
    )

    demOri = {"header": demHeader, "rasterData": demData}
    cfg = configparser.ConfigParser()
    cfg["GENERAL"] = {
        "sphKernelRadius": "0.5",
        "meshCellSizeThreshold": "0.0001",
        "meshCellSize": "1.",
    }
    num = 1

    # setup testResults
    demNewHeader = {}
    demNewHeader["xllcenter"] = 0.0
    demNewHeader["yllcenter"] = 0.0
    demNewHeader["cellsize"] = 1.0
    demNewHeader["nodata_value"] = -9999
    demNewHeader["nrows"] = 5
    demNewHeader["ncols"] = 5
    demTest = {"header": demNewHeader}
    demTest["originalHeader"] = demTest["header"]
    demTest["outOfDEM"] = np.asarray(
        [
            [False, False, False, False, True],
            [False, False, False, False, True],
            [False, False, False, False, True],
            [False, False, False, False, True],
            [False, False, False, False, True],
        ]
    )
    # normal vector of plane
    demTest["Nx"] = (
        np.asarray(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        - 25.0
    )
    demTest["Ny"] = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    demTest["Nz"] = (
        np.asarray(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        + 25.0
    )
    # setup neighbour grid
    headerNeighbourGrid = {}
    headerNeighbourGrid["cellsize"] = 0.5
    headerNeighbourGrid["ncols"] = 10
    headerNeighbourGrid["nrows"] = 10
    headerNeighbourGrid["xllcenter"] = 0
    headerNeighbourGrid["yllcenter"] = 0
    demTest["headerNeighbourGrid"] = headerNeighbourGrid
    areaCell = 1 / np.cos(np.deg2rad(45))
    demTest["areaRaster"] = np.zeros((5, 5)) + areaCell
    demTest["rasterData"] = demData

    # call function to be tested
    dem = com1DFA.initializeMesh(cfg["GENERAL"], demOri, num)

    assert dem["header"]["xllcenter"] == demTest["header"]["xllcenter"]
    assert dem["header"]["yllcenter"] == demTest["header"]["yllcenter"]
    assert dem["header"]["ncols"] == demTest["header"]["ncols"]
    assert dem["header"]["nrows"] == demTest["header"]["nrows"]
    assert dem["header"]["cellsize"] == demTest["header"]["cellsize"]
    assert dem["header"]["yllcenter"] == demTest["header"]["yllcenter"]
    assert np.array_equal(dem["rasterData"][0:4, 0:4], demTest["rasterData"][0:4, 0:4])
    assert np.all(np.isnan(dem["rasterData"][0:5, 4]))
    assert abs(dem["Nx"][2, 2]) == abs(dem["Nz"][2, 2])
    assert np.isclose(dem["areaRaster"][2, 2], demTest["areaRaster"][2, 2])
    assert dem["headerNeighbourGrid"]["xllcenter"] == demTest["headerNeighbourGrid"]["xllcenter"]
    assert dem["headerNeighbourGrid"]["yllcenter"] == demTest["headerNeighbourGrid"]["yllcenter"]
    assert dem["headerNeighbourGrid"]["ncols"] == demTest["headerNeighbourGrid"]["ncols"]
    assert dem["headerNeighbourGrid"]["nrows"] == demTest["headerNeighbourGrid"]["nrows"]
    assert dem["headerNeighbourGrid"]["cellsize"] == demTest["headerNeighbourGrid"]["cellsize"]
    assert dem["headerNeighbourGrid"]["yllcenter"] == demTest["headerNeighbourGrid"]["yllcenter"]


def test_getSimTypeList():
    """test create list of simTypes"""

    # setup required input
    standardCfg = configparser.ConfigParser()
    standardCfg["GENERAL"] = {"secRelArea": "False"}
    simTypeList = ["ent", "res", "null", "available", "entres"]
    inputSimFiles = {"entResInfo": {"flagEnt": "Yes", "flagRes": "Yes", "flagSecondaryRelease": "No"}}

    # call function to be tested
    standardCfg, simTypeList = com1DFA.getSimTypeList(standardCfg, simTypeList, inputSimFiles)

    # setup test result
    simTypeListTest = ["ent", "null", "res", "entres"]

    assert set(simTypeListTest).issubset(simTypeList)
    assert "available" not in simTypeList

    # call function to be tested
    simTypeList = ["ent", "null", "available"]
    inputSimFiles["entResInfo"]["flagRes"] = "No"
    standardCfg2, simTypeList2 = com1DFA.getSimTypeList(standardCfg, simTypeList, inputSimFiles)

    # setup test result
    simTypeListTest2 = ["ent", "null"]

    assert set(simTypeListTest2).issubset(simTypeList2)
    assert "available" not in simTypeList2
    assert "entres" not in simTypeList2
    assert "res" not in simTypeList2

    # call function to be tested
    simTypeList = ["res", "null", "available"]
    inputSimFiles["entResInfo"]["flagEnt"] = "No"
    inputSimFiles["entResInfo"]["flagRes"] = "Yes"
    standardCfg3, simTypeList3 = com1DFA.getSimTypeList(standardCfg, simTypeList, inputSimFiles)

    # setup test result
    simTypeListTest3 = ["res", "null"]

    assert set(simTypeListTest3).issubset(simTypeList3)
    assert "available" not in simTypeList3
    assert "entres" not in simTypeList3
    assert "ent" not in simTypeList3

    # call function to be tested
    simTypeList = ["ent", "null", "available", "entres", "res"]
    inputSimFiles["entResInfo"]["flagEnt"] = "Yes"
    inputSimFiles["entResInfo"]["flagRes"] = "No"
    with pytest.raises(FileNotFoundError) as e:
        assert com1DFA.getSimTypeList(standardCfg, simTypeList, inputSimFiles)
    assert str(e.value) == "No resistance file found"

    # call function to be tested
    inputSimFiles["entResInfo"]["flagEnt"] = "No"
    inputSimFiles["entResInfo"]["flagRes"] = "Yes"
    with pytest.raises(FileNotFoundError) as e:
        assert com1DFA.getSimTypeList(standardCfg, simTypeList, inputSimFiles)
    assert str(e.value) == "No entrainment file found"


def test_appendFieldsParticles():
    """test if correct fields and particles list are created for export"""

    # setup required input
    fieldsListIn = [{"ppr": np.zeros((3, 3)), "pfv": np.zeros((3, 3))}]
    particlesListIn = [
        {
            "x": np.asarray([0.0, 4.0, 0.0]),
            "y": np.asarray([0.0, 4.0, 0.0]),
            "m": np.asarray([0.0, 4.0, 0.0]),
        }
    ]
    particles = {
        "x": np.asarray([0.0, 5.0, 0.0]),
        "y": np.asarray([0.0, 5.0, 0.0]),
        "m": np.asarray([0.0, 4.0, 0.0]),
    }
    fields = {
        "ppr": np.ones((3, 3)),
        "pft": np.ones((3, 3)),
        "pfv": np.ones((3, 3)),
        "FT": np.ones((3, 3)),
    }
    resTypes = ["ppr", "pfv", "pft", "particles"]

    # call function to be tested
    fieldsList, particlesList = com1DFA.appendFieldsParticles(
        fieldsListIn, particlesListIn, particles, fields, resTypes
    )
    #    print("fieldsList", fieldsList[1])

    assert np.array_equal(fieldsList[1]["ppr"], np.ones((3, 3)))
    assert np.array_equal(fieldsList[1]["pfv"], np.ones((3, 3)))
    assert np.array_equal(fieldsList[1]["pft"], np.ones((3, 3)))
    assert resTypes[0:3] == list(fieldsList[1].keys())
    assert len(fieldsList) == 2
    assert np.array_equal(particlesList[1]["x"], particles["x"])
    assert np.array_equal(particlesList[1]["y"], particles["y"])
    assert np.array_equal(particlesList[1]["m"], particles["m"])
    assert ["x", "y", "m"] == list(particlesList[1].keys())
    assert fieldsList[1].get("FT") is None


def test_releaseSecRelArea():
    """test if secondary release area is triggered"""

    # setup required input
    cfg = configparser.ConfigParser()
    cfg["GENERAL"] = {
        "rho": "200.",
        "gravAcc": "9.81",
        "massPerParticleDeterminationMethod": "MPPDH",
        "interpOption": "2",
        "sphKernelRadius": "1",
        "deltaTh": "0.25",
        "seed": "12345",
        "initPartDistType": "uniform",
        "thresholdPointInPoly": "0.001",
        "avalancheDir": "data/avaTest",
        "entTempRef": "-10.",
        "cpIce": "2050.",
        "TIni": "-10.",
    }
    demHeader = {}
    demHeader["cellsize"] = 1
    demHeader["ncols"] = 12
    demHeader["nrows"] = 12
    demHeader["xllcenter"] = 1.0
    demHeader["yllcenter"] = 1.0
    demRaster = np.ones((demHeader["nrows"], demHeader["ncols"]))
    areaRaster = np.ones((demHeader["nrows"], demHeader["ncols"]))
    dem = {"header": demHeader, "rasterData": demRaster, "areaRaster": areaRaster}
    dem["originalHeader"] = dem["header"].copy()
    dem["header"]["xllcenter"] = 0.0
    dem["header"]["yllcenter"] = 0.0
    #    print("dem", dem)
    secRelRaster2 = np.zeros((demHeader["nrows"], demHeader["ncols"]))
    secRelRaster2[6:8, 7] = 1.0
    secRelRaster3 = np.zeros((demHeader["nrows"], demHeader["ncols"]))
    secRelRaster3[9, 9] = 0.5
    secRelRaster1 = np.zeros((demHeader["nrows"], demHeader["ncols"]))
    secRelRaster1[1, 1] = 0.5
    secondaryReleaseInfo = {
        "x": np.asarray(
            [
                1.5,
                2.5,
                2.5,
                1.5,
                1.5,
                7.4,
                8.5,
                8.5,
                7.4,
                7.4,
                9.5,
                10.5,
                10.5,
                9.5,
                9.5,
            ]
        ),
        "y": np.asarray(
            [
                1.5,
                1.5,
                2.5,
                2.5,
                1.5,
                7.4,
                7.4,
                8.5,
                8.5,
                7.4,
                9.5,
                9.5,
                10.5,
                10.5,
                9.5,
            ]
        ),
        "z": np.asarray(
            [
                1.5,
                1.5,
                2.5,
                2.5,
                1.5,
                7.4,
                7.4,
                8.5,
                8.5,
                7.4,
                9.5,
                9.5,
                10.5,
                10.5,
                9.5,
            ]
        ),
        "Start": np.asarray([0, 5, 10]),
        "Length": np.asarray([5, 5, 5]),
        "Name": ["secRel1", "secRel2", "secRel3"],
        "thickness": [0.5, 1.0, 0.5],
        "rasterData": [secRelRaster1, secRelRaster2, secRelRaster3],
        "initializedFrom": "shapefile",
    }
    secondaryReleaseInfo["header"] = demHeader
    secondaryReleaseInfo["header"]["xllcenter"] = dem["originalHeader"]["xllcenter"]
    secondaryReleaseInfo["header"]["yllcenter"] = dem["originalHeader"]["yllcenter"]
    secondaryReleaseInfo2 = copy.deepcopy(secondaryReleaseInfo)
    particlesIn = {"secondaryReleaseInfo": secondaryReleaseInfo}
    particlesIn["x"] = np.asarray([6.0, 7.0])
    particlesIn["y"] = np.asarray([6.0, 7.0])
    particlesIn["z"] = np.asarray([1.0, 2.0])
    particlesIn["m"] = np.asarray([1250.0, 1250.0])
    particlesIn["mTot"] = np.sum(particlesIn["m"])
    particlesIn["t"] = 1.0
    particlesIn["nPart"] = 2.0
    particlesIn["totalEnthalpy"] = np.asarray([6.0, 7.0])
    fieldsFT = np.zeros((demHeader["nrows"], demHeader["ncols"]))
    fieldsFT[7:9, 7:9] = 1.0
    fields = {"FT": fieldsFT}
    zPartArray0 = np.asarray([2.0, 3.0])
    reportAreaInfo = {"secRelArea": {"features released at time [s]": []}}

    # call function to be tested
    particles, zPartArray0New, reportAreaInfo = com1DFA.releaseSecRelArea(
        cfg["GENERAL"], particlesIn, fields, dem, zPartArray0, reportAreaInfo
    )
    #    print("particles IN pytest 1", particles)

    # call function to be tested test 2
    particlesIn2 = {"secondaryReleaseInfo": secondaryReleaseInfo2}
    particlesIn2["x"] = np.asarray([6.0, 7.0, 9.1])
    particlesIn2["y"] = np.asarray([6.0, 7.0, 9.1])
    particlesIn2["z"] = np.asarray([6.0, 7.0, 9.1])
    particlesIn2["m"] = np.asarray([1250.0, 1250.0, 1250.0])
    particlesIn2["mTot"] = np.sum(particlesIn2["m"])
    particlesIn2["t"] = 1.0
    particlesIn2["nPart"] = 3
    fieldsFT2 = np.zeros((demHeader["nrows"], demHeader["ncols"]))
    fieldsFT2[7:9, 7:9] = 1.0
    fieldsFT2[9, 9] = 0.4
    fields2 = {"FT": fieldsFT2}
    zPartArray0 = np.asarray([1.0, 2.0, 3])

    particles2, zPartArray0New2, reportAreaInfo = com1DFA.releaseSecRelArea(
        cfg["GENERAL"], particlesIn2, fields2, dem, zPartArray0, reportAreaInfo
    )

    pEnt = -10.0 * 2050 + 9.81 * 1.0

    #    print("particles IN pytest socond", particles2)
    assert particles["nPart"] == 6
    assert np.array_equal(particles["x"], np.asarray([6.0, 7.0, 6.75, 7.25, 6.75, 7.25]))
    assert np.array_equal(particles["totalEnthalpy"], np.asarray([6.0, 7.0, pEnt, pEnt, pEnt, pEnt]))
    assert np.array_equal(particles["y"], np.asarray([6.0, 7.0, 6.75, 6.75, 7.25, 7.25]))
    assert np.array_equal(zPartArray0New, np.asarray([2, 3, 1.0, 1.0, 1.0, 1.0]))
    assert np.array_equal(particles["m"], np.asarray([1250.0, 1250.0, 50.0, 50.0, 50.0, 50.0]))
    assert particles["mTot"] == 2700.0
    assert particles2["nPart"] == 11
    assert np.array_equal(
        particles2["x"],
        np.asarray([6.0, 7.0, 9.1, 6.75, 7.25, 6.75, 7.25, 8.75, 9.25, 8.75, 9.25]),
    )
    assert np.array_equal(
        particles2["y"],
        np.asarray([6.0, 7.0, 9.1, 6.75, 6.75, 7.25, 7.25, 8.75, 8.75, 9.25, 9.25]),
    )
    assert np.array_equal(zPartArray0New2, np.asarray([1, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1]))
    assert np.array_equal(
        particles2["m"],
        np.asarray([1250.0, 1250.0, 1250.0, 50.0, 50.0, 50.0, 50.0, 25.0, 25.0, 25.0, 25.0]),
    )
    assert particles2["mTot"] == 4050.0


def test_getRelThFromPart():
    """test fetching max value of release thickness used"""

    # setup required input
    cfg = configparser.ConfigParser()
    cfg["GENERAL"] = {"relThFromFile": "True", "relTh": ""}
    inputSimLines = {"releaseLine": {"thickness": ["1.2", "1.5"], "id": ["0", "1"]}}
    relThField = ""

    # call function to be tested
    relThFromPart = com1DFA.getRelThFromPart(cfg["GENERAL"], inputSimLines["releaseLine"], relThField, "rel")

    assert relThFromPart == 1.5

    cfg["GENERAL"]["relThFromFile"] = "False"
    cfg["GENERAL"]["relTh"] = "2.0"
    # call function to be tested
    relThFromPart = com1DFA.getRelThFromPart(cfg["GENERAL"], inputSimLines["releaseLine"], relThField, "rel")

    assert relThFromPart == 2.0

    cfg["GENERAL"]["relThFromFile"] = "False"
    cfg["GENERAL"]["relTh"] = ""
    relThField = np.zeros((10, 10))
    relThField[0:10, 1] = 10.0
    # call function to be tested
    relThFromPart = com1DFA.getRelThFromPart(cfg["GENERAL"], inputSimLines["releaseLine"], relThField, "rel")

    assert relThFromPart == 10.0


def test_initializeParticles():
    """test initialising particles"""

    # setup required input
    cfg = configparser.ConfigParser()
    cfg["REPORT"] = {"plotFields": "ppr|pft|pfv"}
    cfg["GENERAL"] = {
        "resType": "ppr|pft|pfv",
        "rho": "200.",
        "gravAcc": "9.81",
        "massPerParticleDeterminationMethod": "MPPDH",
        "interpOption": "2",
        "sphKernelRadius": "1",
        "deltaTh": "0.25",
        "seed": "12345",
        "initPartDistType": "uniform",
        "thresholdPointInPoly": "0.001",
        "avalancheDir": "data/avaTest",
        "entTempRef": "-10.",
        "cpIce": "2050.",
        "TIni": "-10.",
        "rhoEnt": "200.",
    }
    demHeader = {}
    demHeader["cellsize"] = 1
    demHeader["ncols"] = 12
    demHeader["nrows"] = 12
    demHeader["xllcenter"] = 0.0
    demHeader["yllcenter"] = 0.0
    headerNeighbourGrid = copy.deepcopy(demHeader)
    demRaster = np.ones((demHeader["nrows"], demHeader["ncols"]))
    areaRaster = np.ones((demHeader["nrows"], demHeader["ncols"]))
    dem = {
        "header": demHeader,
        "rasterData": demRaster,
        "areaRaster": areaRaster,
        "headerNeighbourGrid": headerNeighbourGrid,
    }
    dem["originalHeader"] = dem["header"].copy()
    dem["originalHeader"]["xllcenter"] = 1.0
    dem["originalHeader"]["yllcenter"] = 1.0

    relRaster = np.zeros((12, 12))
    relRaster[5:9, 5:9] = 1.0
    releaseLine = {
        "x": np.asarray([6.9, 8.5, 8.5, 6.9, 6.9]),
        "y": np.asarray([6.9, 6.9, 8.5, 8.5, 6.9]),
        "Start": np.asarray([0]),
        "Length": np.asarray([5]),
        "Name": [""],
        "thickness": [1.0],
        "rasterData": relRaster,
    }

    releaseLine["header"] = demHeader
    releaseLine["header"]["xllcenter"] = dem["originalHeader"]["xllcenter"]
    releaseLine["header"]["yllcenter"] = dem["originalHeader"]["yllcenter"]

    dictKeys = [
        "nPart",
        "x",
        "y",
        "trajectoryLengthXY",
        "trajectoryLengthXYCor",
        "trajectoryLengthXYZ",
        "z",
        "m",
        "dmDet",
        "massPerPart",
        "nPPK",
        "mTot",
        "h",
        "ux",
        "uy",
        "uz",
        "uAcc",
        "stoppCriteria",
        "kineticEne",
        "trajectoryAngle",
        "potentialEne",
        "peakKinEne",
        "peakMassFlowing",
        "simName",
        "xllcenter",
        "yllcenter",
        "ID",
        "nID",
        "parentID",
        "t",
        "inCellDEM",
        "indXDEM",
        "indYDEM",
        "indPartInCell",
        "partInCell",
        "secondaryReleaseInfo",
        "iterate",
        "idFixed",
        "peakForceSPH",
        "forceSPHIni",
        "totalEnthalpy",
        "velocityMag",
        "nExitedParticles",
        "tPlot",
        "dmEnt",
        "stoppedParticles",
    ]

    # call function to be tested
    particles = com1DFA.initializeParticles(cfg["GENERAL"], releaseLine, dem)
    particles, fields = com1DFA.initializeFields(cfg, dem, particles, releaseLine)
    particles["iterate"] = True
    particles["secondaryReleaseInfo"] = {"flagSecondaryRelease": "No"}
    # check keys
    missing = set(dictKeys) - particles.keys()
    # if len(missing) > 0:
    #    print("there is an missing key in particles: ", set(dictKeys) - particles.keys())
    extra = particles.keys() - set(dictKeys)
    # if len(extra) > 0:
    #    print("there is an extra key in particles: ", particles.keys() - set(dictKeys))

    # are we missing any keys?
    assert all(key in dictKeys for key in particles)

    # do we have too any keys?
    assert all(key in particles for key in dictKeys)

    assert particles["nPart"] == 9
    assert np.array_equal(
        particles["x"],
        np.asarray([6.25, 6.75, 7.25, 6.25, 6.25, 6.75, 7.25, 6.75, 7.25]),
    )
    assert np.array_equal(
        particles["y"],
        np.asarray([6.25, 6.25, 6.25, 6.75, 7.25, 6.75, 6.75, 7.25, 7.25]),
    )
    assert np.array_equal(
        particles["m"],
        np.asarray([50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0]),
    )
    assert particles["mTot"] == 450.0
    assert np.sum(particles["ux"]) == 0.0

    cfg["GENERAL"]["massPerParticleDeterminationMethod"] = "MPPDIR"
    cfg["GENERAL"].update({"massPerPart": "60."})
    particles = com1DFA.initializeParticles(cfg["GENERAL"], releaseLine, dem)
    particles, fields = com1DFA.initializeFields(cfg, dem, particles, releaseLine)
    particles["iterate"] = True
    particles["secondaryReleaseInfo"] = {"flagSecondaryRelease": "No"}
    # check keys
    # are we missing any keys?
    missing = set(dictKeys) - particles.keys()
    # if len(missing) > 0:
    #    print("there is an missing key in particles: ", set(dictKeys) - particles.keys())
    assert all(key in dictKeys for key in particles)

    # do we have too any keys?
    extra = particles.keys() - set(dictKeys)
    # if len(extra) > 0:
    #    print("there is an extra key in particles: ", particles.keys() - set(dictKeys))
    assert all(key in particles for key in dictKeys)

    #    print("particles", particles)

    assert particles["nPart"] == 9
    assert np.array_equal(
        particles["x"],
        np.asarray([6.25, 6.75, 7.25, 6.25, 6.25, 6.75, 7.25, 6.75, 7.25]),
    )
    assert np.array_equal(
        particles["y"],
        np.asarray([6.25, 6.25, 6.25, 6.75, 7.25, 6.75, 6.75, 7.25, 7.25]),
    )
    assert np.array_equal(
        particles["m"],
        np.asarray([50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0]),
    )
    assert particles["mTot"] == 450.0
    assert np.sum(particles["ux"]) == 0.0

    cfg["GENERAL"]["massPerParticleDeterminationMethod"] = "MPPKR"
    cfg["GENERAL"].update({"nPPK0": "5"})
    cfg["GENERAL"].update({"sphKR0": "1"})
    cfg["GENERAL"].update({"aPPK": "-1"})
    cfg["GENERAL"].update({"relTh": "1."})
    particles = com1DFA.initializeParticles(cfg["GENERAL"], releaseLine, dem)
    particles, fields = com1DFA.initializeFields(cfg, dem, particles, releaseLine)
    particles["iterate"] = True
    particles["secondaryReleaseInfo"] = {"flagSecondaryRelease": "No"}
    # check keys
    # are we missing any keys?
    missing = set(dictKeys) - particles.keys()
    # if len(missing) > 0:
    #    print("there is an missing key in particles: ", set(dictKeys) - particles.keys())
    assert all(key in dictKeys for key in particles)

    # do we have too any keys?
    extra = particles.keys() - set(dictKeys)
    # if len(extra) > 0:
    #    print("there is an extra key in particles: ", particles.keys() - set(dictKeys))
    assert all(key in particles for key in dictKeys)

    #    print("particles", particles)

    assert particles["nPart"] == 9
    assert np.array_equal(
        particles["x"],
        np.asarray([6.25, 6.75, 7.25, 6.25, 6.25, 6.75, 7.25, 6.75, 7.25]),
    )
    assert np.array_equal(
        particles["y"],
        np.asarray([6.25, 6.25, 6.25, 6.75, 7.25, 6.75, 6.75, 7.25, 7.25]),
    )
    assert np.array_equal(
        particles["m"],
        np.asarray([50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0]),
    )
    assert particles["mTot"] == 450.0
    assert np.sum(particles["ux"]) == 0.0


def test_writeMBFile(tmp_path):
    """test writing of mass balance info to file"""

    # setup required input
    infoDict = {"timeStep": np.asarray([0, 1, 2, 3, 4])}
    infoDict["massEntrained"] = np.asarray([0, 0, 10, 20, 30])
    infoDict["massDetrained"] = np.asarray([0, 0, 0, 0, 0])
    infoDict["massDetrainedTotal"] = np.asarray([0, 0, 0, 0, 0])
    infoDict["massTotal"] = np.asarray([60.0, 60.0, 70.0, 90.0, 120.0])
    infoDict["pfvTimeMax"] = np.asarray([0, 0, 0, 0, 0])
    infoDict["massStopped"] = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0])
    avaName = "data/avaTest"
    avaDir = pathlib.Path(tmp_path, avaName)
    logName = "simTestName"

    # call function to be tested
    com1DFA.writeMBFile(infoDict, avaDir, logName)

    mbFilePath = avaDir / "Outputs" / "com1DFA" / "mass_simTestName.txt"
    mbInfo = np.loadtxt(mbFilePath, delimiter=",", skiprows=1)

    #    print("mbInfo", mbInfo)

    assert np.array_equal(mbInfo[:, 0], infoDict["timeStep"])
    assert np.array_equal(mbInfo[:, 2], infoDict["massEntrained"])
    assert np.array_equal(mbInfo[:, 3], infoDict["massDetrained"])
    assert np.array_equal(mbInfo[:, 4], infoDict["massDetrainedTotal"])
    assert np.array_equal(mbInfo[:, 1], infoDict["massTotal"])
    assert np.array_equal(mbInfo[:, 5], infoDict["massStopped"])
    assert mbInfo.shape[0] == 5
    assert mbInfo.shape[1] == 6

    infoDict["massEntrained"] = np.asarray([0, 0, 0, 0, 0])
    infoDict["massDetrained"] = np.asarray([0, 10, 0, 30, 0])
    infoDict["massTotal"] = np.asarray([60.0, 50.0, 50.0, 20.0, 20.0])
    infoDict["massStopped"] = np.asarray([10.0, 10.0, 10.0, 50.0, 0.0])
    infoDict["massDetrainedTotal"] = np.asarray([0, 10, 10, 40, 40])

    com1DFA.writeMBFile(infoDict, avaDir, logName)
    mbFilePath = avaDir / "Outputs" / "com1DFA" / "mass_simTestName.txt"
    mbInfo = np.loadtxt(mbFilePath, delimiter=",", skiprows=1)

    assert np.array_equal(mbInfo[:, 0], infoDict["timeStep"])
    assert np.array_equal(mbInfo[:, 2], infoDict["massEntrained"])
    assert np.array_equal(mbInfo[:, 3], infoDict["massDetrained"])
    assert np.array_equal(mbInfo[:, 4], infoDict["massDetrainedTotal"])
    assert np.array_equal(mbInfo[:, 1], infoDict["massTotal"])
    assert np.array_equal(mbInfo[:, 5], infoDict["massStopped"])
    assert mbInfo.shape[0] == 5
    assert mbInfo.shape[1] == 6

    infoDict["massEntrained"] = np.asarray([0, 20, 0, 0, 10])
    infoDict["massDetrained"] = np.asarray([0, 10, 0, 30, 0])
    infoDict["massDetrainedTotal"] = np.asarray([0, 10, 10, 40, 40])
    infoDict["massTotal"] = np.asarray([60.0, 70.0, 70.0, 40.0, 50.0])
    infoDict["massStopped"] = np.asarray([0, 10, 0, 30, 0])

    com1DFA.writeMBFile(infoDict, avaDir, logName)
    mbFilePath = avaDir / "Outputs" / "com1DFA" / "mass_simTestName.txt"
    mbInfo = np.loadtxt(mbFilePath, delimiter=",", skiprows=1)

    assert np.array_equal(mbInfo[:, 0], infoDict["timeStep"])
    assert np.array_equal(mbInfo[:, 2], infoDict["massEntrained"])
    assert np.array_equal(mbInfo[:, 3], infoDict["massDetrained"])
    assert np.array_equal(mbInfo[:, 4], infoDict["massDetrainedTotal"])
    assert np.array_equal(mbInfo[:, 1], infoDict["massTotal"])
    assert np.array_equal(mbInfo[:, 5], infoDict["massStopped"])
    assert mbInfo.shape[0] == 5
    assert mbInfo.shape[1] == 6


def test_savePartToPickle(tmp_path):
    """test saving particles info to pickle"""

    # setup required input
    particles1 = {
        "x": np.asarray([1.0, 2.0, 3.0]),
        "y": np.asarray([1.0, 4.0, 5.0]),
        "m": np.asarray([10.0, 11.0, 11.0]),
        "t": 0.0,
    }
    particles2 = {
        "x": np.asarray([10.0, 20.0, 30.0]),
        "y": np.asarray([10.0, 40.0, 50.0]),
        "m": np.asarray([100.0, 110.0, 110.0]),
        "t": 2.0,
    }
    dictList = [particles1, particles2]
    outDir = pathlib.Path(tmp_path, "particles")
    outDir.mkdir()
    logName = "simNameTest"

    # call function to be tested
    com1DFA.savePartToPickle(dictList, outDir, logName)

    # read pickle
    picklePath1 = outDir / "particles_simNameTest_0000.0000.pickle"
    picklePath2 = outDir / "particles_simNameTest_0002.0000.pickle"
    particlesRead1 = pickle.load(open(picklePath1, "rb"))
    particlesRead2 = pickle.load(open(picklePath2, "rb"))

    #    print("particklesRead1", particlesRead1)
    #    print("particklesRead2", particlesRead2)

    assert np.array_equal(particlesRead1["x"], particles1["x"])
    assert np.array_equal(particlesRead1["y"], particles1["y"])
    assert np.array_equal(particlesRead1["m"], particles1["m"])
    assert particlesRead1["t"] == 0.0
    assert np.array_equal(particlesRead2["x"], particles2["x"])
    assert np.array_equal(particlesRead2["y"], particles2["y"])
    assert np.array_equal(particlesRead2["m"], particles2["m"])
    assert particlesRead2["t"] == 2.0

    # call function to be tested
    logName = "simNameTest3"
    com1DFA.savePartToPickle(particles1, outDir, logName)

    # read pickle
    picklePath3 = outDir / "particles_simNameTest3_0000.0000.pickle"
    particlesRead3 = pickle.load(open(picklePath3, "rb"))

    #    print("particklesRead3", particlesRead3)
    #    print("particklesRead2", particlesRead2)

    assert np.array_equal(particlesRead3["x"], particles1["x"])
    assert np.array_equal(particlesRead3["y"], particles1["y"])
    assert np.array_equal(particlesRead3["m"], particles1["m"])
    assert particlesRead3["t"] == 0.0


def test_exportFields(tmp_path):
    """test exporting fields to ascii files"""

    # setup required input
    cfg = configparser.ConfigParser()
    cfg["GENERAL"] = {"resType": "ppr|pft|FT"}
    cfg["REPORT"] = {"plotFields": "ppr|pft|pfv|pke"}
    cfg["EXPORTS"] = {"useCompression": "True"}
    Tsave = [0, 10, 15, 25, 40]
    demHeader = {}
    demHeader["cellsize"] = 1
    demHeader["ncols"] = 10
    demHeader["nrows"] = 10
    demHeader["xllcenter"] = 0
    demHeader["yllcenter"] = 0
    demHeader["nodata_value"] = -9999
    demHeader["driver"] = "AAIGrid"

    transform = transformFromASCHeader(demHeader)

    demHeader["transform"] = transform
    demHeader["crs"] = rasterio.crs.CRS()

    areaRaster = np.ones((5, 5))
    dem = {"originalHeader": demHeader, "areaRaster": areaRaster}
    outDir = pathlib.Path(tmp_path, "testDir")
    outDir.mkdir()
    logName = "simNameTest"
    FT = np.zeros((5, 5))
    pke = np.zeros((5, 5))
    pft = np.zeros((5, 5))
    pfv = np.zeros((5, 5))
    ppr = np.zeros((5, 5))
    fields1 = {
        "ppr": ppr + 1,
        "pft": pft + 1,
        "pfv": pfv + 1,
        "FT": FT + 1,
        "pke": pke + 1,
    }
    fields2 = {
        "ppr": ppr + 2,
        "pft": pft + 2,
        "pfv": pfv + 2,
        "FT": FT + 2,
        "pke": pke + 2,
    }
    fields3 = {
        "ppr": ppr + 4,
        "pft": pft + 4,
        "pfv": pfv + 4,
        "FT": FT + 4,
        "pke": pke + 4,
    }
    fields4 = {
        "ppr": ppr + 5,
        "pft": pft + 5,
        "pfv": pfv + 5,
        "FT": FT + 5,
        "pke": pke + 5,
    }
    fields5 = {
        "ppr": ppr + 6,
        "pft": pft + 6,
        "pfv": pfv + 6,
        "FT": FT + 6,
        "pke": pke + 6,
    }
    fieldsList = [fields1, fields2, fields3, fields4, fields5]

    # call function to be tested
    com1DFA.exportFields(cfg, 10.00, fields2, dem, outDir, logName, TSave="intermediate")
    com1DFA.exportFields(cfg, 40.00, fields5, dem, outDir, logName, TSave="final")

    # read fields
    fieldDir = outDir / "peakFiles"
    fieldDirTSteps = outDir / "peakFiles" / "timeSteps"
    fieldFiles = list(fieldDirTSteps.glob("*.asc"))
    fieldsListTest = []

    for f in fieldFiles:
        fieldsListTest.append(f.name)

    field1 = fieldDir / "simNameTest_ppr.asc"
    field2 = fieldDirTSteps / "simNameTest_pft_t10.00.asc"
    fieldFinal = np.loadtxt(field1, skiprows=6)
    field10 = np.loadtxt(field2, skiprows=6)
    pprFinal = ppr + 0.006
    pftt10 = pft + 2

    #    print("field1", fieldFinal)
    #    print("pprFinal", pprFinal)
    #    print("fields", fieldsListTest)

    assert np.array_equal(fieldFinal, pprFinal)
    assert np.array_equal(field10, pftt10)
    assert len(fieldsListTest) == 8

    # call function to be tested
    outDir2 = pathlib.Path(tmp_path, "testDir2")
    outDir2.mkdir()
    cfg["GENERAL"]["resType"] = ""
    cfg["REPORT"] = {"plotFields": "ppr|pft|pfv"}

    com1DFA.exportFields(cfg, 0.00, fields1, dem, outDir2, logName, TSave="initial")
    com1DFA.exportFields(cfg, 10.00, fields2, dem, outDir2, logName, TSave="intermediate")
    com1DFA.exportFields(cfg, 15.00, fields3, dem, outDir2, logName, TSave="intermediate")
    com1DFA.exportFields(cfg, 25.00, fields4, dem, outDir2, logName, TSave="intermediate")
    com1DFA.exportFields(cfg, 40.00, fields5, dem, outDir2, logName, TSave="final")

    # read fields
    fieldDir = outDir2 / "peakFiles"
    fieldDirTSteps = outDir2 / "peakFiles" / "timeSteps"
    fieldFiles = list(fieldDirTSteps.glob("*.asc"))
    fieldFiles3 = list(fieldDir.glob("*.asc"))
    fieldsListTest2 = []
    fieldsListTest3 = []
    for f in fieldFiles:
        fieldsListTest2.append(f.name)
    #    print("fields file", fieldFiles)

    for f in fieldFiles3:
        fieldsListTest3.append(f.name)

    assert len(fieldsListTest2) == 6
    assert len(fieldsListTest3) == 3


def test_initializeFields():
    """test initializing fieldgetSimTypeLists"""

    # setup required inputs
    demHeader = {"nrows": 11, "ncols": 12, "cellsize": 1}
    areaRaster = np.ones((11, 12))
    dem = {
        "header": demHeader,
        "headerNeighbourGrid": demHeader,
        "areaRaster": areaRaster,
    }
    particles = {
        "x": np.asarray([1.0, 2.0, 3.0]),
        "y": np.asarray([1.0, 2.0, 3.0]),
        "nPart": 3,
        "ux": np.asarray([0.0, 0.0, 0.0]),
        "uy": np.asarray([0.0, 0.0, 0.0]),
        "uz": np.asarray([0.0, 0.0, 0.0]),
        "m": np.asarray([10.0, 10.0, 10.0]),
        "dmDet": np.asarray([0.0, 0.0, 0.0]),
        "dmEnt": np.asarray([0.0, 0.0, 0.0]),
        "trajectoryAngle": np.asarray([0.0, 0.0, 0.0]),
        "stoppedParticles": {"m": np.empty(0), "x": np.empty(0), "y": np.empty(0)},
    }
    cfg = configparser.ConfigParser()
    cfg["REPORT"] = {"plotFields": "ppr|pft|pfv"}
    cfg["GENERAL"] = {
        "rho": "200.",
        "interpOption": "2",
        "resType": "ppr|pft|pfv",
        "rhoEnt": 200,
    }

    dem["originalHeader"] = dem["header"]
    dem["header"]["xllcenter"] = 0.0
    dem["header"]["yllcenter"] = 0.0

    # call function to be tested
    particles, fields = com1DFA.initializeFields(cfg, dem, particles, "")

    #    print("particles", particles)
    #    print("fields", fields)
    #    print("compute KE", fields["computeKE"])
    #    print("compute TA", fields["computeTA"])
    #    print("compute P", fields["computeP"])

    assert len(fields) == 23
    assert fields["computeTA"] is False
    assert fields["computeKE"] is False
    assert fields["computeP"]
    assert np.sum(fields["pfv"]) == 0.0
    assert np.sum(fields["pta"]) == 0.0
    assert np.sum(fields["ppr"]) == 0.0
    assert np.sum(fields["FV"]) == 0.0
    assert np.sum(fields["P"]) == 0.0
    assert np.sum(fields["TA"]) == 0.0
    assert np.sum(fields["Vx"]) == 0.0
    assert np.sum(fields["Vy"]) == 0.0
    assert np.sum(fields["Vz"]) == 0.0
    assert np.sum(fields["pft"]) != 0.0
    assert np.sum(fields["FT"]) != 0.0
    assert np.sum(fields["FM"]) != 0.0
    assert np.sum(fields["dmDet"]) == 0.0
    assert np.sum(fields["sfcChange"]) == 0.0
    assert np.sum(fields["demAdapted"]) == 0.0
    assert np.sum(fields["FTDet"]) == 0.0
    assert np.sum(fields["FTStop"]) == 0.0
    assert np.sum(fields["FTEnt"]) == 0.0

    cfg["REPORT"] = {"plotFields": "pft|pfv"}
    cfg["GENERAL"] = {
        "resType": "pke|pta|pft|pfv",
        "rho": "200.",
        "interpOption": "2",
        "rhoEnt": 200,
    }
    # call function to be tested
    particles, fields = com1DFA.initializeFields(cfg, dem, particles, "")
    assert len(fields) == 23
    assert fields["computeTA"]
    assert fields["computeKE"]
    assert fields["computeP"] is False


def test_prepareVarSimDict(tmp_path, caplog):
    """test prepare variation sim dictionary"""

    # setup required input
    standardCfg = configparser.ConfigParser()
    standardCfg.optionxform = str
    standardCfg["GENERAL"] = {
        "simTypeList": "entres|null",
        "modelType": "dfa",
        "simTypeActual": "entres",
        "secRelArea": "False",
        "relThFromFile": "False",
        "relThFromFile": "False",
        "entThFromFile": "True",
        "entThPercentVariation": "",
        "relThPercentVariation": "",
        "entThRangeVariation": "",
        "relThRangeVariation": "",
        "entThDistVariation": "",
        "relThDistVariation": "",
        "entThRangeFromCiVariation": "",
        "relThRangeFromCiVariation": "",
        "meshCellSize": "5.",
        "meshCellSizeThreshold": "0.001",
        "sphKernelRadius": "meshCellSize",
        "frictModel": "samosAT",
        "musamosat": "0.155",
        "tau0samosat": "0",
        "Rs0samosat": "0.222",
        "kappasamosat": "0.43",
        "Rsamosat": "0.05",
        "Bsamosat": "4.13",
        "muvoellmy": "4000.",
        "xsivoellmy": "4000.",
        "dam": "True",
        "explicitFriction": 0,
    }
    standardCfg["INPUT"] = {
        "entThThickness": "1.",
        "entThId": "0",
        "entThCi95": "None",
        "releaseScenario": "",
        "relThFile": "",
    }

    testDir = pathlib.Path(__file__).parents[0]
    inputDir = testDir / ".." / "data" / "avaAlr" / "Inputs"
    avaDirInputs = pathlib.Path(tmp_path, "avaTestNew", "Inputs")
    avaDir = pathlib.Path(tmp_path, "avaTestNew")
    shutil.copytree(inputDir, avaDirInputs)
    avaDEM = avaDir / "Inputs" / "avaAlr.tif"

    standardCfg["INPUT"]["DEM"] = "avaAlr.tif"
    standardCfg["GENERAL"]["avalancheDir"] = str(avaDir)
    relPath = pathlib.Path(avaDir, "Inputs", "REL", "relAlr.shp")
    inputSimFiles = {
        "relFiles": [relPath],
        "entResInfo": {
            "flagEnt": "Yes",
            "flagRes": "Yes",
            "entThFileType": ".shp",
            "relThFileType": ".shp",
            "resFileType": ".shp",
            "secondaryRelThFileType": None,
        },
        "demFile": avaDEM,
        "damFile": None,
        "entFile": pathlib.Path(avaDir, "Inputs", "ENT", "entAlr.shp"),
        "resFile": pathlib.Path(avaDir, "Inputs", "ENT", "entAlr.shp"),
    }
    variationDict = {"rho": np.asarray([200.0, 150.0]), "releaseScenario": ["relAlr"]}

    # call function to be tested
    simDict = com1DFA.prepareVarSimDict(standardCfg, inputSimFiles, variationDict)

    testCfg = configparser.ConfigParser()
    testCfg.optionxform = str
    testCfg["GENERAL"] = {
        "simTypeList": "entres",
        "modelType": "dfa",
        "simTypeActual": "entres",
        "secRelArea": "False",
        "relThFromFile": "False",
        "relThFromFile": "False",
        "entThFromFile": "True",
        "entThPercentVariation": "",
        "relThPercentVariation": "",
        "rho": "200.0",
        "entTh0": "1.0",
        "entThRangeVariation": "",
        "relThRangeVariation": "",
        "entThDistVariation": "",
        "relThDistVariation": "",
        "entThRangeFromCiVariation": "",
        "relThRangeFromCiVariation": "",
        "meshCellSize": "5.",
        "meshCellSizeThreshold": "0.001",
        "sphKernelRadius": "5.",
        "frictModel": "samosAT",
        "musamosat": "0.155",
        "tau0samosat": "0",
        "Rs0samosat": "0.222",
        "kappasamosat": "0.43",
        "Rsamosat": "0.05",
        "Bsamosat": "4.13",
        "muvoellmy": "4000.",
        "xsivoellmy": "4000.",
        "dam": "True",
        "explicitFriction": 0,
    }

    testCfg["INPUT"] = {
        "entThThickness": "1.",
        "entThId": "0",
        "entThCi95": "None",
        "releaseScenario": "relAlr",
    }
    testCfg["INPUT"]["DEM"] = "avaAlr.tif"
    testCfg["INPUT"]["relThFile"] = ""
    testCfg["INPUT"]["entrainmentScenario"] = str(pathlib.Path("ENT", "entAlr.shp"))
    testCfg["INPUT"]["resistanceScenario"] = str(pathlib.Path("RES", "entAlr.shp"))
    testCfg["GENERAL"]["avalancheDir"] = str(avaDir)

    simHash = cfgUtils.cfgHash(testCfg)
    simName1 = "relAlr_" + simHash + "_C_L_entres_dfa"
    testDict = {
        simName1: {
            "simHash": simHash,
            "releaseScenario": "relAlr",
            "simType": "entres",
            "relFile": relPath,
            "cfgSim": testCfg,
        }
    }

    for key in testDict[simName1]:
        #        print(simDict)
        #        print(simDict[simName1][key])
        assert simDict[simName1][key] == testDict[simName1][key]

    for section in testCfg.sections():
        for key in testCfg[section]:
            assert simDict[simName1]["cfgSim"][section][key] == testCfg[section][key]

    # call function to be tested
    # relPath = pathlib.Path('test', 'relTest_extended.shp')
    inputSimFiles = {
        "relFiles": [relPath],
        "entResInfo": {
            "flagEnt": "Yes",
            "flagRes": "Yes",
            "entThFileType": ".shp",
            "relThFileType": ".shp",
            "resFileType": ".shp",
            "secondaryRelThFileType": None,
        },
        "demFile": avaDEM,
        "damFile": relPath,
        "entFile": pathlib.Path(avaDir, "Inputs", "ENT", "entAlr.shp"),
        "resFile": pathlib.Path(avaDir, "Inputs", "RES", "entAlr.shp"),
    }
    variationDict = {
        "rho": np.asarray([200.0, 150.0]),
        "simTypeList": ["entres", "ent"],
        "releaseScenario": ["relAlr"],
    }

    simDict2 = com1DFA.prepareVarSimDict(standardCfg, inputSimFiles, variationDict)

    inputSimFiles = {
        "relFiles": [relPath],
        "entResInfo": {
            "flagEnt": "Yes",
            "flagRes": "Yes",
            "entThFileType": ".shp",
            "relThFileType": ".shp",
            "resFileType": ".shp",
            "secondaryRelThFileType": None,
        },
        "demFile": avaDEM,
        "damFile": relPath,
        "entFile": pathlib.Path(avaDir, "Inputs", "ENT", "entAlr.shp"),
        "resFile": pathlib.Path(avaDir, "Inputs", "ENT", "entAlr.shp"),
    }
    testCfg2 = configparser.ConfigParser()
    testCfg2.optionxform = str
    testCfg2["GENERAL"] = {
        "simTypeList": "entres",
        "modelType": "dfa",
        "simTypeActual": "entres",
        "secRelArea": "False",
        "relThFromFile": "False",
        "entThFromFile": "True",
        "relThFromFile": "False",
        "entThPercentVariation": "",
        "relThPercentVariation": "",
        "entThRangeFromCiVariation": "",
        "relThRangeFromCiVariation": "",
        "rho": "150.0",
        "entTh0": "1.0",
        "entThRangeVariation": "",
        "relThRangeVariation": "",
        "entThDistVariation": "",
        "relThDistVariation": "",
        "meshCellSize": "5.",
        "meshCellSizeThreshold": "0.001",
        "sphKernelRadius": "5.",
        "frictModel": "samosAT",
        "musamosat": "0.155",
        "tau0samosat": "0",
        "Rs0samosat": "0.222",
        "kappasamosat": "0.43",
        "Rsamosat": "0.05",
        "Bsamosat": "4.13",
        "muvoellmy": "4000.",
        "xsivoellmy": "4000.",
        "dam": "True",
        "explicitFriction": 0,
    }
    testCfg2["INPUT"] = {
        "entThThickness": "1.",
        "entThId": "0",
        "entThCi95": "None",
        "releaseScenario": "relAlr",
        "DAM": str(pathlib.Path("DAM", relPath.name)),
    }
    testCfg2["INPUT"]["DEM"] = "avaAlr.tif"
    testCfg2["INPUT"]["relThFile"] = ""
    testCfg2["INPUT"]["entrainmentScenario"] = str(pathlib.Path("ENT", "entAlr.shp"))
    testCfg2["INPUT"]["resistanceScenario"] = str(pathlib.Path("RES", "entAlr.shp"))
    testCfg2["GENERAL"]["avalancheDir"] = str(avaDir)
    simHash2 = cfgUtils.cfgHash(testCfg2)
    simName2 = "relAlr_" + simHash2 + "_C_L_entres_dfa"
    testDict2 = {
        simName2: {
            "simHash": simHash2,
            "releaseScenario": "relAlr",
            "simType": "entres",
            "relFile": relPath,
            "cfgSim": testCfg2,
        }
    }

    #    print(simDict2)
    #    print(testDict2)
    for key in testDict2[simName2]:
        assert simDict2[simName2][key] == testDict2[simName2][key]

    for section in testCfg2.sections():
        for key in testCfg2[section]:
            #            print("section", section, "key", key)
            assert simDict2[simName2]["cfgSim"][section][key] == testCfg2[section][key]

    # What if a simulation already exists
    with caplog.at_level(logging.WARNING):
        simDict2 = com1DFA.prepareVarSimDict(
            standardCfg, inputSimFiles, variationDict, simNameExisting=[simName2]
        )
    assert ("Simulation %s already exists, not repeating it" % simName2) in caplog.text
    assert simName2 not in simDict2


def test_initializeSimulation(tmp_path):
    """test initializing a simulation"""

    outDir = pathlib.Path(tmp_path, "Outputs")
    testDir = pathlib.Path(__file__).parents[0]
    inputDir = testDir / "data" / "avaTestInputs"
    avaDir = pathlib.Path(tmp_path, "avaTest1")
    shutil.copytree(inputDir, avaDir)

    # setup required input
    cfg = configparser.ConfigParser()
    cfg["REPORT"] = {"plotFields": "ppr|pft|pfv"}
    cfg["GENERAL"] = {
        "methodMeshNormal": "1",
        "thresholdPointInPoly": "0.001",
        "useRelThFromIni": "False",
        "resType": "ppr|pft|pfv",
        "relTh": "1.0",
        "useEntThFromIni": "False",
        "meshCellSizeThreshold": "0.0001",
        "meshCellSize": "1.",
        "simTypeActual": "ent",
        "rhoEnt": "100.",
        "entTh": "0.3",
        "rho": "200.",
        "gravAcc": "9.81",
        "massPerParticleDeterminationMethod": "MPPDH",
        "interpOption": "2",
        "sphKernelRadius": "1",
        "deltaTh": "0.25",
        "seed": "12345",
        "initPartDistType": "uniform",
        "thresholdPointInPoly": "0.001",
        "avalancheDir": "data/avaTest",
        "cRes": "0.003",
        "initialiseParticlesFromFile": "False",
        "entTempRef": "-10.",
        "cpIce": "2050.",
        "TIni": "-10.",
        "ResistanceModel": "cRes",
    }
    cfg["EXPORTS"] = {"exportRasters": "False"}
    # setup dem input
    demHeader = {}
    demHeader["xllcenter"] = 1.0
    demHeader["yllcenter"] = 2.0
    demHeader["cellsize"] = 1.0
    demHeader["nodata_value"] = -9999
    demHeader["nrows"] = 12
    demHeader["ncols"] = 12
    demData = np.ones((12, 12))
    demOri = {"header": demHeader, "rasterData": demData}

    # setup release line, entrainment line
    relFileTest = avaDir / "REL" / "relAlr.shp"
    releaseLine = {
        "x": np.asarray([6.9, 8.5, 8.5, 6.9, 6.9]),
        "y": np.asarray([7.9, 7.9, 9.5, 9.5, 7.9]),
        "Start": np.asarray([0]),
        "Length": np.asarray([5]),
        "Name": [""],
        "thickness": [1.0],
        "thicknessSource": ["ini File"],
        "type": "release",
        "file": relFileTest,
        "initializedFrom": "shapefile",
    }
    entLine = {
        "fileName": (avaDir / "ENT" / "entAlr.shp"),
        "Name": ["testEnt"],
        "Start": np.asarray([0.0]),
        "thickness": [0.3, 0.3],
        "thicknessSource": ["shp file", "shp file"],
        "Length": np.asarray([5]),
        "x": np.asarray([4, 5.0, 5.0, 4.0, 4.0]),
        "type": "entrainment",
        "y": np.asarray([4.0, 4.0, 5.0, 5.0, 4.0]),
        "initializedFrom": "shapefile",
    }

    inputSimLines = {
        "releaseLine": releaseLine,
        "entResInfo": {"flagSecondaryRelease": "No"},
        "entLine": entLine,
        "secondaryReleaseLine": None,
        "resLine": "",
        "relThFile": "",
        "entThFile": "",
        "relThField": "",
        "damLine": None,
        "muFile": None,
        "xiFile": None,
    }
    # set release thickness read from file or not
    logName = "simLog"

    # call function to be tested
    particles, fields, dem, reportAreaInfo = com1DFA.initializeSimulation(
        cfg, outDir, demOri, inputSimLines, logName
    )

    #    print("particles", particles)
    #    print("fields", fields)
    #    print("dem", dem)
    #    print("reportAreaInfo", reportAreaInfo)

    pEnt = -10.0 * 2050.0 + 9.81 * 1.0
    assert np.array_equal(
        particles["y"],
        np.asarray([6.25, 6.25, 6.25, 6.75, 7.25, 6.75, 6.75, 7.25, 7.25]),
    )
    assert np.sum(fields["pfv"]) == 0.0
    assert np.sum(fields["pft"]) != 0.0
    assert dem["header"]["xllcenter"] == 0.0
    assert dem["header"]["yllcenter"] == 0.0
    assert dem["originalHeader"]["xllcenter"] == 1.0
    assert dem["originalHeader"]["yllcenter"] == 2.0
    assert particles["nPart"] == 9
    assert np.array_equal(
        particles["totalEnthalpy"],
        np.asarray([pEnt, pEnt, pEnt, pEnt, pEnt, pEnt, pEnt, pEnt, pEnt]),
    )
    assert np.array_equal(
        particles["x"],
        np.asarray([6.25, 6.75, 7.25, 6.25, 6.25, 6.75, 7.25, 6.75, 7.25]),
    )
    assert np.array_equal(
        particles["m"],
        np.asarray([50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0]),
    )
    assert particles["mTot"] == 450.0
    assert np.sum(particles["ux"]) == 0.0
    assert reportAreaInfo["Release area info"]["Projected Area [m2]"] == "4.00"
    assert reportAreaInfo["entrainment"] == "Yes"
    assert reportAreaInfo["resistance"] == "No"

    # call function to be tested
    inputSimLines["entResInfo"]["flagSecondaryRelease"] = "Yes"
    inputSimLines["secondaryReleaseLine"] = {
        "x": np.asarray([1.5, 2.5, 2.5, 1.5, 1.5]),
        "y": np.asarray([2.5, 2.5, 3.5, 3.5, 2.5]),
        "Start": np.asarray([0]),
        "Length": np.asarray([5]),
        "type": "Secondary release",
        "fileName": (avaDir / "SECREL" / "ec1.shp"),
        "Name": ["secRel1"],
        "thickness": [0.5],
        "thicknessSource": ["ini File"],
        "muFile": None,
        "xiFile": None,
        "initializedFrom": "shapefile",
    }
    relThField = np.zeros((12, 12))
    relThField[2:4, 8:10] = 0.5
    inputSimLines["releaseLine"] = {
        "Name": "fromRaster",
        "thickness": "fromRaster",
        "thicknessSource": "from raster",
        "type": "Release read from raster",
        "file": relFileTest,
        "initializedFrom": "raster",
        "rasterData": relThField,
    }

    cfg["GENERAL"]["relThFromFile"] = "False"
    cfg["GENERAL"]["relTh"] = ""
    cfg["GENERAL"]["relThFromFile"] = "True"
    inputSimLines["relThField"] = relThField

    particles2, fields2, dem2, reportAreaInfo2 = com1DFA.initializeSimulation(
        cfg, outDir, demOri, inputSimLines, logName
    )

    #    print("secRel", particles2["secondaryReleaseInfo"])
    # print("particles", particles2)
    #    print("fields", fields2["pft"])

    assert np.sum(fields2["pfv"]) == 0.0
    assert np.sum(fields2["pft"]) != np.sum(fields["pft"])
    assert dem2["header"]["xllcenter"] == 0.0
    assert dem2["header"]["yllcenter"] == 0.0
    assert dem2["originalHeader"]["xllcenter"] == 1.0
    assert dem2["originalHeader"]["yllcenter"] == 2.0
    assert particles2["nPart"] == 16
    assert particles2["mTot"] == 400.0
    assert np.sum(particles["ux"]) == 0.0
    assert reportAreaInfo["Release area info"]["Projected Area [m2]"] == "4.00"
    assert reportAreaInfo["entrainment"] == "Yes"
    assert reportAreaInfo["resistance"] == "No"
    assert np.sum(particles2["secondaryReleaseInfo"]["rasterData"]) == 4.5

    # test if dam is found
    # setup required input
    cfg = configparser.ConfigParser()
    cfg["REPORT"] = {"plotFields": "ppr|pft|pfv"}
    cfg["GENERAL"] = {
        "methodMeshNormal": "1",
        "thresholdPointInPoly": "0.001",
        "useRelThFromIni": "False",
        "resType": "ppr|pft|pfv",
        "relTh": "1.0",
        "useEntThFromIni": "False",
        "meshCellSizeThreshold": "0.0001",
        "meshCellSize": "1.",
        "simTypeActual": "null",
        "rhoEnt": "100.",
        "entTh": "0.3",
        "rho": "200.",
        "gravAcc": "9.81",
        "massPerParticleDeterminationMethod": "MPPDH",
        "interpOption": "2",
        "sphKernelRadius": "1",
        "deltaTh": "0.25",
        "seed": "12345",
        "initPartDistType": "uniform",
        "thresholdPointInPoly": "0.001",
        "avalancheDir": "data/avaTest",
        "cRes": "0.003",
        "initialiseParticlesFromFile": "False",
        "entTempRef": "-10.",
        "cpIce": "2050.",
        "TIni": "-10.",
        "ResistanceModel": "cRes",
        "restitutionCoefficient": 1,
        "nIterDam": 1,
    }
    cfg["EXPORTS"] = {"exportRasters": "False"}
    releaseLine = {
        "x": np.asarray([6.9, 8.5, 8.5, 6.9, 6.9]),
        "y": np.asarray([7.9, 7.9, 9.5, 9.5, 7.9]),
        "Start": np.asarray([0]),
        "Length": np.asarray([5]),
        "Name": [""],
        "thickness": [1.0],
        "thicknessSource": ["ini File"],
        "type": "release",
        "file": relFileTest,
        "initializedFrom": "shapefile",
    }
    inputSimLines = {
        "releaseLine": releaseLine,
        "entResInfo": {"flagSecondaryRelease": "No"},
        "entLine": None,
        "secondaryReleaseLine": None,
        "resLine": "",
        "relThFile": "",
        "damLine": None,
        "muFile": None,
        "xiFile": None,
        "relThField": "",
    }
    inputSimLines["damLine"] = {
        "fileName": [avaDir / "DAM" / "damLine.shp"],
        "Name": [""],
        "thickness": ["None"],
        "slope": 60.0,
        "Start": np.asarray([0.0]),
        "Length": np.asarray([2.0]),
        "x": np.asarray([5.0, 7.0]),
        "y": np.asarray([4.0, 6.0]),
        "z": np.asarray([0.0, 0.0]),
        "id": ["0"],
        "ci95": ["None"],
        "layerName": [None],
        "nParts": [[0, 2]],
        "nFeatures": 1,
        "type": "Dam",
    }

    particles3, fields3, dem3, reportAreaInfo3 = com1DFA.initializeSimulation(
        cfg, outDir, demOri, inputSimLines, logName
    )

    print("dam", inputSimLines["damLine"])

    assert reportAreaInfo3["dam"] == "Yes"
    assert "xCrown" in inputSimLines["damLine"]
    assert "height" in inputSimLines["damLine"]


def test_runCom1DFA(tmp_path, caplog):
    """Check that runCom1DFA produces the good outputs"""
    testDir = pathlib.Path(__file__).parents[0]
    inputDir = testDir / "data" / "testCom1DFA"
    avaDir = pathlib.Path(tmp_path, "testCom1DFA")
    shutil.copytree(inputDir, avaDir)
    cfgFile = avaDir / "test_com1DFACfg.ini"
    cfgMain = configparser.ConfigParser()
    cfgMain["MAIN"] = {"avalancheDir": str(avaDir), "nCPU": "auto", "CPUPercent": "90"}
    cfgMain["FLAGS"] = {
        "showPlot": "False",
        "savePlot": "True",
        "ReportDir": "True",
        "reportOneFile": "True",
        "debugPlot": "False",
    }
    modCfg, modInfo = cfgUtils.getModuleConfig(com1DFA, fileOverride=cfgFile, modInfo=True)

    dem, plotDict, reportDictList, simDF = com1DFA.com1DFAMain(cfgMain, cfgInfo=cfgFile)

    print("DONE")

    dictKeys = [
        "nPart",
        "x",
        "y",
        "trajectoryLengthXY",
        "trajectoryLengthXYCor",
        "trajectoryLengthXYZ",
        "z",
        "m",
        "dt",
        "massPerPart",
        "nPPK",
        "mTot",
        "h",
        "ux",
        "uy",
        "uz",
        "uAcc",
        "stoppCriteria",
        "kineticEne",
        "trajectoryAngle",
        "potentialEne",
        "peakKinEne",
        "peakMassFlowing",
        "simName",
        "xllcenter",
        "yllcenter",
        "ID",
        "nID",
        "parentID",
        "t",
        "inCellDEM",
        "indXDEM",
        "indYDEM",
        "indPartInCell",
        "partInCell",
        "secondaryReleaseInfo",
        "iterate",
        "velocityMag",
        "massEntrained",
        "idFixed",
        "peakForceSPH",
        "forceSPHIni",
        "gEff",
        "curvAcc",
        "totalEnthalpy",
        "nExitedParticles",
        "dmDet",
        "massDetrained",
        "tPlot",
        "dmEnt",
        "massStopped",
        "stoppedParticles",
    ]

    # read one particles dictionary
    inDir = avaDir / "Outputs" / "com1DFA" / "particles"
    PartDicts = sorted(list(inDir.glob("*.pickle")))
    particlesList = []
    timeStepInfo = []
    for particles in PartDicts:
        particles = pickle.load(open(particles, "rb"))
        particlesList.append(particles)
        timeStepInfo.append(particles["t"])

    # are we missing any keys?
    missing = set(dictKeys) - particlesList[-1].keys()
    # if len(missing) > 0:
    #    print("there is an missing key in particles: ", set(dictKeys) - particlesList[-1].keys())
    assert all(key in particlesList[-1] for key in dictKeys)

    # do we have too any keys?
    extra = particlesList[-1].keys() - set(dictKeys)
    # if len(extra) > 0:
    #     print("there is an extra key in particles: ", particlesList[-1].keys() - set(dictKeys))
    assert all(key in dictKeys for key in particlesList[-1])

    assert len(particlesList) == 12

    #    print(simDF["simName"])
    outDir = avaDir / "Outputs" / "com1DFA"
    for ext in ["ppr", "pft", "pfv"]:
        assert (outDir / "peakFiles" / ("%s_%s.asc" % (simDF["simName"].iloc[0], ext))).is_file()
        assert (outDir / "peakFiles" / ("%s_%s.asc" % (simDF["simName"].iloc[1], ext))).is_file()

    assert (outDir / "configurationFiles" / ("%s.ini" % (simDF["simName"].iloc[0]))).is_file()
    assert (outDir / "configurationFiles" / ("%s.ini" % (simDF["simName"].iloc[1]))).is_file()
    assert (outDir / "configurationFiles" / ("allConfigurations.csv")).is_file()

    initProj.cleanModuleFiles(avaDir, com1DFA, deleteOutput=False)
    with caplog.at_level(logging.WARNING):
        dem, plotDict, reportDictList, simDF = com1DFA.com1DFAMain(cfgMain, cfgInfo=cfgFile)
    assert "There is no simulation to be performed" in caplog.text


def test_runOrLoadCom1DFA(tmp_path, caplog):
    testDir = pathlib.Path(__file__).parents[0]
    avalancheDir = testDir / ".." / ".." / "benchmarks" / "avaNoAva"
    cfgMain = configparser.ConfigParser()
    with pytest.raises(FileExistsError) as e:
        dem, simDF, resTypeList = com1DFA.runOrLoadCom1DFA(
            avalancheDir, cfgMain, runDFAModule=False, cfgFile=""
        )
    assert ("Did not find any com1DFA simulations in") in str(e.value)

    testDir = pathlib.Path(__file__).parents[0]
    avalancheDir = testDir / ".." / ".." / "benchmarks" / "avaHockeyChannelPytest"
    cfgMain = configparser.ConfigParser()
    dem, simDF, resTypeList = com1DFA.runOrLoadCom1DFA(avalancheDir, cfgMain, runDFAModule=False, cfgFile="")
    #    print(simDF.index)
    #    print(simDF.columns)
    assert "pft" in resTypeList
    assert "pfv" in resTypeList
    assert "ppr" in resTypeList
    assert "release1HS_0dcd58fc86_ent_dfa" in simDF["simName"].to_list()
    assert "release2HS_3d519adab0_ent_dfa" in simDF["simName"].to_list()


def test_fetchRelVolume(tmp_path):
    testDir = pathlib.Path(__file__).parents[0]
    inputDir = testDir / "data" / "avaTestRel"
    avaDir = pathlib.Path(tmp_path, "avaTest1")
    shutil.copytree(inputDir, avaDir)

    # get path to release shp file
    rel1 = avaDir / "rel1.shp"

    # create DEM
    dem = {
        "header": {
            "xllcenter": 0.0,
            "yllcenter": 0.0,
            "cellsize": 1.0,
            "nrows": 10,
            "ncols": 20,
            "nodata_value": -9999,
            "driver": "AAIGrid",
        }
    }

    transform = transformFromASCHeader(dem["header"])
    dem["header"]["transform"] = transform
    dem["header"]["crs"] = rasterio.crs.CRS.from_epsg(31287)

    dem["rasterData"] = np.ones((10, 20))
    demPath = pathlib.Path(avaDir, "Inputs", "testDem.asc")
    fU.makeADir(pathlib.Path(avaDir, "Inputs"))
    IOf.writeResultToRaster(dem["header"], dem["rasterData"], demPath.parent / demPath.stem, flip=False)

    # subprocess.run(["cat", demPath])
    # write relThField
    relThF = {
        "header": {
            "xllcenter": 0.0,
            "yllcenter": 0.0,
            "cellsize": 1.0,
            "nrows": 10,
            "ncols": 20,
            "nodata_value": -9999,
            "driver": "AAIGrid",
        }
    }
    transform = transformFromASCHeader(relThF["header"])
    relThF["header"]["transform"] = transform
    relThF["header"]["crs"] = rasterio.crs.CRS.from_epsg(31287)
    relThF["rasterData"] = np.zeros((10, 20))
    for k in range(10):
        relThF["rasterData"][k, :] = k * 1
    relThField1 = pathlib.Path(avaDir, "Inputs", "RELTH", "relThField1.asc")
    fU.makeADir(pathlib.Path(avaDir, "Inputs", "RELTH"))
    IOf.writeResultToRaster(
        relThF["header"],
        relThF["rasterData"],
        relThField1.parent / relThField1.stem,
        flip=False,
    )

    cfg = {}
    # relTh read from shp
    cfg["GENERAL"] = {
        "methodMeshNormal": 1,
        "thresholdPointInPoly": 0.001,
        "avalancheDir": avaDir,
        "relTh": "",
        "relThFromFile": True,
        "relTh0": 2.0,
        "relTh1": 4.0,
        "secRelArea": False,
    }
    cfg["INPUT"] = {
        "relThFile": "",
        "relThId": "0|1",
        "relThThickness": "2.|4.",
        "thFromIni": "",
    }

    relVolume = com1DFA.fetchRelVolume(rel1, cfg, demPath, None)

    assert relVolume == 34.0

    cfg = {}
    # relTh read from cfg
    cfg["GENERAL"] = {
        "methodMeshNormal": 1,
        "thresholdPointInPoly": 0.001,
        "avalancheDir": avaDir,
        "relTh": 5.0,
        "relThFromFile": False,
        "secRelArea": False,
    }
    cfg["INPUT"] = {"relThFile": "", "thFromIni": True}

    relVolume = com1DFA.fetchRelVolume(rel1, cfg, demPath, None)

    # call function
    assert relVolume == 65.0

    cfg = {}
    # relTh read from relThField
    cfg["GENERAL"] = {
        "methodMeshNormal": 1,
        "thresholdPointInPoly": 0.001,
        "avalancheDir": avaDir,
        "relTh": "",
        "relThFromFile": False,
        "secRelArea": False,
    }
    cfg["INPUT"] = {"relThFile": "RELTH/relThField1.asc", "thFromIni": False}

    # call function
    relVolume = com1DFA.fetchRelVolume(rel1, cfg, demPath, None)

    assert relVolume == 900.0


def test_adaptDEM():
    cfg = configparser.ConfigParser()
    cfg["GENERAL"] = {
        "methodMeshNormal": 1,
        "adaptSfcStopped": 0,
        "adaptSfcDetrainment": 0,
        "adaptSfcEntrainment": 0,
    }

    header = {
        "nrows": 5,
        "ncols": 5,
        "cellsize": 5,
    }

    data = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
        ]
    )
    dem = {
        "header": header,
        "rasterData": data,
        "originalHeader": {},
        "originalRasterData": data,
        "headerNeighbourGrid": {},
        "damLine": {},
    }

    fields = {
        "FTDet": np.zeros_like(data),
        "FTStop": np.zeros_like(data),
        "FTEnt": np.zeros_like(data),
        "demAdapted": data,
        "sfcChangeTotal": np.zeros_like(data),
        "sfcChange": np.zeros_like(data),
    }

    dem = geoTrans.getNormalMesh(dem, num=cfg["GENERAL"].getfloat("methodMeshNormal"))
    dem = DFAtls.getAreaMesh(dem, cfg["GENERAL"].getfloat("methodMeshNormal"))

    _, _, NzNormed = DFAtls.normalize(dem["Nx"].copy(), dem["Ny"].copy(), dem["Nz"].copy())

    demInput = dem.copy()
    fieldsInput = fields.copy()

    demAdapted, fieldsAdapted = com1DFA.adaptDEM(demInput, fieldsInput, cfg["GENERAL"])
    for key in demAdapted.keys():
        assert np.all(demAdapted[key] == dem[key])
    for key in fieldsAdapted.keys():
        assert np.all(fieldsAdapted[key] == fields[key])

    cfg = configparser.ConfigParser()
    cfg["GENERAL"] = {
        "methodMeshNormal": 1,
        "adaptSfcStopped": 1,
        "adaptSfcDetrainment": 1,
        "adaptSfcEntrainment": 1,
    }

    # all rasters for depth changes are zero
    demAdapted, fieldsAdapted = com1DFA.adaptDEM(demInput, fieldsInput, cfg["GENERAL"])
    for key in demAdapted.keys():
        assert np.all(demAdapted[key] == dem[key])
    for key in fieldsAdapted.keys():
        assert np.all(fieldsAdapted[key] == fields[key])

    fields["FTDet"] += 1
    fieldsInput = fields.copy()
    demInput = dem.copy()

    demAdapted, fieldsAdapted = com1DFA.adaptDEM(demInput, fieldsInput, cfg["GENERAL"])

    for key in demAdapted.keys():
        if key == "rasterData":
            assert np.all(demAdapted[key] == dem[key] + 1 / NzNormed)
        elif key == "header":
            assert np.all(demAdapted[key] == dem[key])

    for key in fieldsAdapted.keys():
        if key == "demAdapted":
            assert np.all(fieldsAdapted[key] == fields[key] + 1 / NzNormed)
        elif key == "sfcChange":
            assert np.all(fieldsAdapted[key] == fields["FTDet"] / NzNormed)
        elif key == "sfcChangeTotal":
            assert np.all(fieldsAdapted[key] == fields["FTDet"] / NzNormed)
        else:
            assert np.all(fieldsAdapted[key] == fields[key])

    cfg = configparser.ConfigParser()
    cfg["GENERAL"] = {
        "methodMeshNormal": 1,
        "adaptSfcStopped": 1,
        "adaptSfcDetrainment": 0,
        "adaptSfcEntrainment": 1,
    }

    fieldsInput = fields.copy()
    demInput = dem.copy()
    demAdapted, fieldsAdapted = com1DFA.adaptDEM(demInput, fieldsInput, cfg["GENERAL"])

    for key in demAdapted.keys():
        assert np.all(demAdapted[key] == dem[key])
    for key in fieldsAdapted.keys():
        assert np.all(fieldsAdapted[key] == fields[key])

    cfg = configparser.ConfigParser()
    cfg["GENERAL"] = {
        "methodMeshNormal": 1,
        "adaptSfcStopped": 1,
        "adaptSfcDetrainment": 1,
        "adaptSfcEntrainment": 1,
    }

    fields["FTEnt"] -= 1
    fieldsInput = fields.copy()
    demInput = dem.copy()

    demAdapted, fieldsAdapted = com1DFA.adaptDEM(demInput, fieldsInput, cfg["GENERAL"])

    for key in demAdapted.keys():
        assert np.all(demAdapted[key] == dem[key])
    for key in fieldsAdapted.keys():
        assert np.all(fieldsAdapted[key] == fields[key])

    fields["FTEnt"] = np.zeros_like(fields["FTDet"])
    fields["FTDet"] = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=float,
    )
    fieldsInput = fields.copy()
    demInput = dem.copy()
    demAdapted, fieldsAdapted = com1DFA.adaptDEM(demInput, fieldsInput, cfg["GENERAL"])

    assert np.all(
        demAdapted["rasterData"]
        == np.array(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [1.0, 2.0, 3.0, 4.0, 5.0] + 1 / NzNormed[2],
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [1.0, 2.0, 3.0, 4.0, 5.0],
            ]
        )
    )
    assert np.any(demAdapted["Nx"] != dem["Nx"])
    assert np.any(demAdapted["Ny"] != dem["Ny"])
    assert np.all(demAdapted["Nz"] == dem["Nz"])
    assert np.any(dem["areaRaster"] != demAdapted["areaRaster"])
    assert np.all(fieldsAdapted["sfcChange"] == fields["FTDet"] / NzNormed)
    assert np.all(fieldsAdapted["sfcChangeTotal"] == fields["FTDet"] / NzNormed)
