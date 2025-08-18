"""
Pytest for module com1DFATools
"""

#  Load modules
import math
import pytest
import configparser
import pathlib
import shutil
import numpy as np
import rasterio

from avaframe.com1DFA import com1DFATools
from avaframe.in2Trans.rasterUtils import transformFromASCHeader
import avaframe.in3Utils.fileHandlerUtils as fU


def test_getPartInitMethod(capfd):
    cfg = configparser.ConfigParser()
    csz = 1
    relThForPart = 2
    cfg["GENERAL"] = {
        "rho": "3",
        "massPerPart": "10",
        "deltaTh": "0.25",
        "sphKernelRadius": "1",
        "nPPK0": "5",
        "aPPK": "-1",
        "sphKR0": "5",
        "massPerParticleDeterminationMethod": "MPPDIR",
    }
    massPerPart, nPPK = com1DFATools.getPartInitMethod(cfg["GENERAL"], csz, relThForPart)
    assert massPerPart == 10
    assert nPPK == 0

    cfg["GENERAL"]["massPerParticleDeterminationMethod"] = "MPPDH"
    massPerPart, nPPK = com1DFATools.getPartInitMethod(cfg["GENERAL"], csz, relThForPart)
    assert massPerPart == 0.75
    assert nPPK == 0

    cfg["GENERAL"]["massPerParticleDeterminationMethod"] = "MPPKR"
    massPerPart, nPPK = com1DFATools.getPartInitMethod(cfg["GENERAL"], csz, relThForPart)

    assert massPerPart == pytest.approx(math.pi * 6 / 25, abs=1e-6)
    assert nPPK == 25


def test_createSimDictFromCfgs(tmp_path):
    """test creating a simDict from multiple cfg files"""

    dirPath = pathlib.Path(__file__).parents[0]
    testPath = dirPath / "data" / "com1DFAConfigsTest"
    inputDir = dirPath / "data" / "testCom1DFA2"
    avaDir = pathlib.Path(tmp_path, "testCom1DFA")
    shutil.copytree(inputDir, avaDir)
    cfgMain = configparser.ConfigParser()
    cfgMain["MAIN"] = {"avalancheDir": avaDir}

    with pytest.raises(FileNotFoundError) as e:
        # call function to be tested
        simDict, inputSimFiles, simDFExisting, outDir = com1DFATools.createSimDictFromCfgs(cfgMain, testPath)
    assert ("No configuration file found to create simulation runs in:") in str(e.value)

    testPath = dirPath / "data" / "com1DFAConfigs"
    avaDir = pathlib.Path(tmp_path, "testCom1DFA2")
    shutil.copytree(inputDir, avaDir)
    cfgMain["MAIN"] = {"avalancheDir": avaDir}
    # call function to be tested
    simDict, inputSimFiles, simDFExisting, outDir = com1DFATools.createSimDictFromCfgs(cfgMain, testPath)

    # for sim in simDict.keys():
    #        print('simDict: ', sim, simDict[sim])
    #    print('inputSimFiles', inputSimFiles)
    #    print('simDFExisting', simDFExisting)
    #    print('outDir', outDir)

    assert simDFExisting == None
    assert len(simDict) == 16


def test_updateResCoeffFields(tmp_path):
    """test updating fields of cResH and detK for resistance"""

    outDir = pathlib.Path(tmp_path, "Outputs", "com1DFA", "reports")
    fU.makeADir(outDir)
    cfg = configparser.ConfigParser()
    cfg["GENERAL"] = {
        "forestVMin": 6.0,
        "forestThMin": 0.6,
        "forestVMax": 40.0,
        "forestThMax": 10.0,
        "avalancheDir": pathlib.Path(tmp_path),
    }

    cResRasterOrig = np.ones((4, 10))
    detRasterOrig = np.ones((4, 10)) + 2
    FV = np.zeros((4, 10))
    FT = np.zeros((4, 10))

    fields = {
        "cResRasterOrig": cResRasterOrig,
        "detRasterOrig": detRasterOrig,
        "FV": FV,
        "FT": FT,
    }
    dem = {
        "originalHeader": {
            "ncols": 10,
            "nrows": 4,
            "xllcenter": 1000.00,
            "yllcenter": -5000.00,
            "cellsize": 5,
            "nodata_value": -9999.00,
            "driver": "AAIGrid",
        }
    }
    transform = transformFromASCHeader(dem["originalHeader"])
    dem["originalHeader"]["transform"] = transform
    dem["originalHeader"]["crs"] = rasterio.crs.CRS()

    fields = com1DFATools.updateResCoeffFields(fields, cfg["GENERAL"])

    assert fields["cResRaster"].any() == False

    FV[0, 4] = 7
    FT[0, 4] = 0.7
    FT[2, 2] = 0.7
    FV[1, 3] = 41
    FT[1, 1] = 11
    fields = {
        "cResRasterOrig": cResRasterOrig,
        "detRasterOrig": detRasterOrig,
        "FV": FV,
        "FT": FT,
    }

    fields = com1DFATools.updateResCoeffFields(fields, cfg["GENERAL"])

    assert fields["cResRaster"].any() == True
    assert fields["cResRaster"][0, 4] == 1
    assert fields["detRaster"][0, 4] == 0
    assert fields["cResRaster"][2, 2] == 0
    assert fields["detRaster"][2, 2] == 3
    assert fields["cResRaster"][1, 3] == 0
    assert fields["detRaster"][1, 3] == 0
    assert fields["cResRaster"][1, 1] == 0
    assert fields["detRaster"][1, 1] == 0
    assert fields["cResRaster"][0, 0] == 0
    assert fields["detRaster"][0, 0] == 3
    assert fields["cResRaster"][2, 3] == 0
    assert fields["detRaster"][2, 3] == 3
