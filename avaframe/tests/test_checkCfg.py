"""
Pytest for module com1DFA
"""

#  Load modules
import numpy as np
import logging
import pytest
import configparser
import pathlib


from avaframe.com1DFA import checkCfg


def test_checkCfgConsistency():
    """test check if sphOption and viscOption settings are consistent"""

    # setup requuired input data
    cfg = configparser.ConfigParser()
    cfg["GENERAL"] = {"viscOption": 2, "sphOption": 2, "methodMeshNormal": 1}

    # call function to be tested
    testFlag = checkCfg.checkCfgConsistency(cfg)

    assert testFlag == True

    cfg["GENERAL"] = {"viscOption": 2, "sphOption": 1, "methodMeshNormal": 1}

    # call function to be tested
    with pytest.raises(AssertionError) as e:
        assert checkCfg.checkCfgConsistency(cfg)
    assert "If viscOption is set to 2" in str(e.value)

    cfg["GENERAL"] = {"viscOption": 2, "sphOption": 2, "methodMeshNormal": 2}

    # call function to be tested
    with pytest.raises(AssertionError) as e:
        assert checkCfg.checkCfgConsistency(cfg)
    assert "The methodMeshNormal set in the ini file does NOT have" in str(e.value)


def test_checkCellSizeKernelRadius():
    """test check if sphOption and viscOption settings are consistent"""

    # setup requuired input data
    cfg = {"GENERAL": {"sphKernelRadius": "5", "meshCellSize": "10"}}

    # call function to be tested
    cfg = checkCfg.checkCellSizeKernelRadius(cfg)

    assert cfg["GENERAL"]["sphKernelRadius"] == "5"

    # setup requuired input data
    cfg = {"GENERAL": {"sphKernelRadius": "meshCellSize", "meshCellSize": "10"}}

    # call function to be tested
    cfg = checkCfg.checkCellSizeKernelRadius(cfg)

    assert cfg["GENERAL"]["sphKernelRadius"] == "10"


def test_checkCfgFrictionModel():
    """test if friction model parameters are set to nan if not used"""

    cfg = {
        "GENERAL": {
            "frictModel": "samosAT",
            "musamosat": "0.155",
            "tau0samosat": "0",
            "Rs0samosat": "0.222",
            "kappasamosat": "0.43",
            "Rsamosat": "0.05",
            "Bsamosat": "4.13",
            "muvoellmy": "4000.",
            "xsivoellmy": "4000.",
            "explicitFriction": "0",
        }
    }
    inputSimFiles = {}

    cfg = checkCfg.checkCfgFrictionModel(cfg, inputSimFiles)

    assert cfg["GENERAL"]["frictModel"] == "samosAT"
    assert float(cfg["GENERAL"]["musamosat"]) == 0.155
    assert float(cfg["GENERAL"]["tau0samosat"]) == 0.0
    assert float(cfg["GENERAL"]["kappasamosat"]) == 0.43
    assert float(cfg["GENERAL"]["Rsamosat"]) == 0.05
    assert float(cfg["GENERAL"]["Bsamosat"]) == 4.13
    assert float(cfg["GENERAL"]["muvoellmy"]) == 4000.0
    assert float(cfg["GENERAL"]["xsivoellmy"]) == 4000.0

    cfg = {
        "GENERAL": {
            "frictModel": "samosAT",
            "musamosat": "0.155",
            "tau0samosat": "0",
            "Rs0samosat": "0.222",
            "kappasamosat": "0.43",
            "Rsamosat": "0.05",
            "Bsamosat": "9.13",
            "muvoellmy": "4000.",
            "xsivoellmy": "nan",
            "explicitFriction": "0",
        }
    }

    cfg = checkCfg.checkCfgFrictionModel(cfg, inputSimFiles)

    assert cfg["GENERAL"]["frictModel"] == "samosAT"
    assert float(cfg["GENERAL"]["musamosat"]) == 0.155
    assert float(cfg["GENERAL"]["tau0samosat"]) == 0
    assert float(cfg["GENERAL"]["kappasamosat"]) == 0.43
    assert float(cfg["GENERAL"]["Rsamosat"]) == 0.05
    assert float(cfg["GENERAL"]["Bsamosat"]) == 9.13
    assert float(cfg["GENERAL"]["muvoellmy"]) == 4000.0
    assert np.isnan(float(cfg["GENERAL"]["xsivoellmy"]))

    cfg = {
        "GENERAL": {
            "frictModel": "samosAT",
            "musamosat": "0.155",
            "tau0samosat": "0",
            "Rs0samosat": "0.222",
            "kappasamosat": "0.43",
            "Rsamosat": "0.05",
            "Bsamosat": "nan",
            "muvoellmy": "4000.",
            "xsivoellmy": "nan",
            "explicitFriction": "0",
        }
    }

    # call function to be tested
    with pytest.raises(ValueError) as e:
        assert checkCfg.checkCfgFrictionModel(cfg, inputSimFiles)
    assert "Friction model used samosAT, but Bsamosat is nan - not valid" in str(e.value)

    cfg = {
        "GENERAL": {
            "frictModel": "samosAT",
            "musamosat": "0.155",
            "tau0samosat": "0",
            "Rs0samosat": "0.222",
            "kappasamosat": "0.43",
            "Rsamosat": "0.05",
            "Bsamosat": "test",
            "muvoellmy": "4000.",
            "xsivoellmy": "nan",
            "explicitFriction": "0",
        }
    }

    # call function to be tested
    with pytest.raises(ValueError) as e:
        assert checkCfg.checkCfgFrictionModel(cfg, inputSimFiles)
    assert "Friction model used samosAT, but test is not of valid" in str(e.value)

    cfg = {
        "GENERAL": {
            "frictModel": "samosATAuto",
            "musamosat": "0.155",
            "tau0samosat": "0",
            "Rs0samosat": "0.222",
            "kappasamosat": "0.43",
            "Rsamosat": "0.05",
            "Bsamosat": "9.13",
            "muvoellmy": "4000.",
            "xsivoellmy": "nan",
            "musamosatsmall": "0.22",
            "tau0samosatsmall": "0",
            "Rs0samosatsmall": "0.222",
            "kappasamosatsmall": "0.43",
            "Rsamosatsmall": "0.05",
            "Bsamosatsmall": "4.13",
            "musamosatmedium": "0.17",
            "tau0samosatmedium": "0",
            "Rs0samosatmedium": "0.222",
            "kappasamosatmedium": "0.43",
            "Rsamosatmedium": "0.05",
            "Bsamosatmedium": "4.13",
            "volClassSmall": "25000.",
            "volClassMedium": "60000.",
            "meshCellSize": "5",
            "explicitFriction": "1",
        }
    }

    cfg = checkCfg.checkCfgFrictionModel(cfg, inputSimFiles, relVolume=24999.0)

    assert cfg["GENERAL"]["frictModel"] == "samosATSmall"
    assert float(cfg["GENERAL"]["musamosatsmall"]) == 0.22
    assert float(cfg["GENERAL"]["tau0samosatsmall"]) == 0
    assert float(cfg["GENERAL"]["Rs0samosatsmall"]) == 0.222
    assert float(cfg["GENERAL"]["kappasamosatsmall"]) == 0.43
    assert float(cfg["GENERAL"]["Rsamosatsmall"]) == 0.05
    assert float(cfg["GENERAL"]["Bsamosatsmall"]) == 4.13
    assert float(cfg["GENERAL"]["muvoellmy"]) == 4000.0
    assert np.isnan(float(cfg["GENERAL"]["xsivoellmy"]))

    cfg = {
        "GENERAL": {
            "frictModel": "samosATAuto",
            "musamosat": "0.155",
            "tau0samosat": "0",
            "Rs0samosat": "0.222",
            "kappasamosat": "0.43",
            "Rsamosat": "0.05",
            "Bsamosat": "9.13",
            "muvoellmy": "4000.",
            "xsivoellmy": "nan",
            "musamosatsmall": "0.22",
            "tau0samosatsmall": "0",
            "Rs0samosatsmall": "0.222",
            "kappasamosatsmall": "0.43",
            "Rsamosatsmall": "0.05",
            "Bsamosatsmall": "4.13",
            "musamosatmedium": "0.17",
            "tau0samosatmedium": "0.9",
            "Rs0samosatmedium": "1.222",
            "kappasamosatmedium": "2.43",
            "Rsamosatmedium": "0.75",
            "Bsamosatmedium": "4.23",
            "volClassSmall": "25000.",
            "volClassMedium": "60000.",
            "meshCellSize": "5",
            "explicitFriction": "1",
        }
    }

    cfg = checkCfg.checkCfgFrictionModel(cfg, inputSimFiles, relVolume=26999.0)

    assert cfg["GENERAL"]["frictModel"] == "samosATMedium"
    assert float(cfg["GENERAL"]["musamosatmedium"]) == 0.17
    assert float(cfg["GENERAL"]["tau0samosatmedium"]) == 0.9
    assert float(cfg["GENERAL"]["Rs0samosatmedium"]) == 1.222
    assert float(cfg["GENERAL"]["kappasamosatmedium"]) == 2.43
    assert float(cfg["GENERAL"]["Rsamosatmedium"]) == 0.75
    assert float(cfg["GENERAL"]["Bsamosatmedium"]) == 4.23
    assert float(cfg["GENERAL"]["muvoellmy"]) == 4000.0
    assert np.isnan(float(cfg["GENERAL"]["xsivoellmy"]))

    cfg = {
        "GENERAL": {
            "frictModel": "samosATAuto",
            "musamosat": "0.155",
            "tau0samosat": "0.8",
            "Rs0samosat": "0.2227",
            "kappasamosat": "1.43",
            "Rsamosat": "1.05",
            "Bsamosat": "9.13",
            "muvoellmy": "4000.",
            "xsivoellmy": "nan",
            "musamosatsmall": "0.22",
            "tau0samosatsmall": "0",
            "Rs0samosatsmall": "0.222",
            "kappasamosatsmall": "0.43",
            "Rsamosatsmall": "0.05",
            "Bsamosatsmall": "4.13",
            "musamosatmedium": "0.17",
            "tau0samosatmedium": "0",
            "Rs0samosatmedium": "0.222",
            "kappasamosatmedium": "0.43",
            "Rsamosatmedium": "0.05",
            "Bsamosatmedium": "4.13",
            "volClassSmall": "25000.",
            "volClassMedium": "60000.",
            "meshCellSize": "5",
            "explicitFriction": "1",
        }
    }

    cfg = checkCfg.checkCfgFrictionModel(cfg, inputSimFiles, relVolume=74999.0)

    assert cfg["GENERAL"]["frictModel"] == "samosAT"
    assert float(cfg["GENERAL"]["musamosat"]) == 0.155
    assert float(cfg["GENERAL"]["tau0samosat"]) == 0.8
    assert float(cfg["GENERAL"]["Rs0samosat"]) == 0.2227
    assert float(cfg["GENERAL"]["kappasamosat"]) == 1.43
    assert float(cfg["GENERAL"]["Rsamosat"]) == 1.05
    assert float(cfg["GENERAL"]["Bsamosat"]) == 9.13
    assert float(cfg["GENERAL"]["muvoellmy"]) == 4000.0
    assert np.isnan(float(cfg["GENERAL"]["xsivoellmy"]))


def _getMuICfg(**overrides):
    """mu(I) friction model configuration with the default parameter values"""

    cfg = {
        "GENERAL": {
            "frictModel": "muI",
            "rho": "200",
            "muIMuS": "0.3640",
            "muIMu2": "0.6249",
            "muII0": "0.3",
            "muIGrainSize": "0.0009",
            "muIGrainDensity": "917",
            "muIShearFactor": "2.5",
            "explicitFriction": "0",
        }
    }
    cfg["GENERAL"].update({key: str(value) for key, value in overrides.items()})
    return cfg


def test_checkCfgFrictionModelMuI(caplog):
    """test mu(I) parameter checks: valid defaults, errors and warnings"""

    inputSimFiles = {}

    # default parameters: pass without warning
    with caplog.at_level(logging.WARNING):
        cfg = checkCfg.checkCfgFrictionModel(_getMuICfg(), inputSimFiles)
    assert cfg["GENERAL"]["frictModel"] == "muI"
    assert caplog.records == []

    # nan values are caught by the generic parameter check
    cfg = _getMuICfg(muIGrainSize="nan")
    with pytest.raises(ValueError) as e:
        checkCfg.checkCfgFrictionModel(cfg, inputSimFiles)
    assert "Friction model used muI, but muIGrainSize is nan - not valid" in str(e.value)

    # every parameter must be positive
    for name in ["muIMuS", "muIMu2", "muII0", "muIGrainSize", "muIGrainDensity", "muIShearFactor"]:
        for value in ["0", "-1"]:
            with pytest.raises(ValueError) as e:
                checkCfg.checkCfgFrictionModel(_getMuICfg(**{name: value}), inputSimFiles)
            assert "%s must be larger than 0" % name in str(e.value)

    # mu_2 must be larger than mu_s
    for muS, mu2 in [("0.6", "0.6"), ("0.7", "0.6")]:
        with pytest.raises(ValueError) as e:
            checkCfg.checkCfgFrictionModel(_getMuICfg(muIMuS=muS, muIMu2=mu2), inputSimFiles)
        assert "muIMu2 must be larger than muIMuS" in str(e.value)

    # grain density below the flow density: probably bulk density entered as grain density
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        checkCfg.checkCfgFrictionModel(_getMuICfg(muIGrainDensity="150"), inputSimFiles)
    assert "muIGrainDensity" in caplog.text
    assert "bulk density" in caplog.text

    # non-default shear rate closure
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        checkCfg.checkCfgFrictionModel(_getMuICfg(muIShearFactor="3"), inputSimFiles)
    assert "muIShearFactor" in caplog.text
    assert "2.5" in caplog.text
