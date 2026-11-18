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


def _getHysteresisCfg(**overrides):
    """hysteresis friction model configuration with the default parameter values"""

    cfg = {
        "GENERAL": {
            "frictModel": "hysteresis",
            "hysteresisMu1": "0.5867",
            "hysteresisMu2": "0.9391",
            "hysteresisMu3": "0.6976",
            "hysteresisL": "0.00056",
            "hysteresisBeta": "0.63",
            "hysteresisBetaStar": "0.47",
            "hysteresisGamma": "0.40",
            "hysteresisViscosity": "0",
            "hysteresisNu": "0.0014",
            "explicitFriction": "1",
            "adaptSfcStopped": "0",
        }
    }
    cfg["GENERAL"].update({key: str(value) for key, value in overrides.items()})
    return cfg


def test_checkCfgFrictionModelHysteresis(caplog):
    """test hysteresis parameter checks: valid defaults, errors and warnings"""

    inputSimFiles = {}

    # default parameters: pass without warning
    with caplog.at_level(logging.WARNING):
        cfg = checkCfg.checkCfgFrictionModel(_getHysteresisCfg(), inputSimFiles)
    assert cfg["GENERAL"]["frictModel"] == "hysteresis"
    assert caplog.records == []

    # nan values are caught by the generic parameter check
    with pytest.raises(ValueError) as e:
        checkCfg.checkCfgFrictionModel(_getHysteresisCfg(hysteresisL="nan"), inputSimFiles)
    assert "Friction model used hysteresis, but hysteresisL is nan - not valid" in str(e.value)

    # positive parameters
    for name in ["hysteresisMu1", "hysteresisL", "hysteresisBeta", "hysteresisBetaStar"]:
        for value in ["0", "-1"]:
            with pytest.raises(ValueError) as e:
                checkCfg.checkCfgFrictionModel(_getHysteresisCfg(**{name: value}), inputSimFiles)
            assert "%s must be larger than 0" % name in str(e.value)

    # gamma may be zero (Pouliquen and Forterre 2002) but not negative
    checkCfg.checkCfgFrictionModel(_getHysteresisCfg(hysteresisGamma="0"), inputSimFiles)
    with pytest.raises(ValueError) as e:
        checkCfg.checkCfgFrictionModel(_getHysteresisCfg(hysteresisGamma="-0.1"), inputSimFiles)
    assert "hysteresisGamma must not be negative" in str(e.value)

    # mu2 and mu3 must be larger than mu1
    for name in ["hysteresisMu2", "hysteresisMu3"]:
        for value in ["0.5867", "0.5"]:
            with pytest.raises(ValueError) as e:
                checkCfg.checkCfgFrictionModel(_getHysteresisCfg(**{name: value}), inputSimFiles)
            assert "%s must be larger than hysteresisMu1" % name in str(e.value)

    # the yield condition requires the explicit friction method
    with pytest.raises(ValueError) as e:
        checkCfg.checkCfgFrictionModel(_getHysteresisCfg(explicitFriction="0"), inputSimFiles)
    assert "requires explicitFriction = 1" in str(e.value)

    # static friction below the dynamic friction at betaStar for some thicknesses: the minimum of
    # muStart - muDyn is (mu3 - mu1) - 0.0284 for the default mu1, mu2, beta, betaStar and gamma
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        checkCfg.checkCfgFrictionModel(_getHysteresisCfg(hysteresisMu3="0.60"), inputSimFiles)
    assert "intermediate regime" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        checkCfg.checkCfgFrictionModel(_getHysteresisCfg(hysteresisMu3="0.62"), inputSimFiles)
    assert caplog.records == []

    # stopped particles are removed with adaptSfcStopped
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        checkCfg.checkCfgFrictionModel(_getHysteresisCfg(adaptSfcStopped="1"), inputSimFiles)
    assert "adaptSfcStopped" in caplog.text

    # viscous term: valid options, no negative coefficient, warning if it has no effect
    for option in ["1", "2"]:
        checkCfg.checkCfgFrictionModel(_getHysteresisCfg(hysteresisViscosity=option), inputSimFiles)
    with pytest.raises(ValueError) as e:
        checkCfg.checkCfgFrictionModel(_getHysteresisCfg(hysteresisViscosity="3"), inputSimFiles)
    assert "hysteresisViscosity is 3, must be 0, 1 or 2" in str(e.value)
    with pytest.raises(ValueError) as e:
        checkCfg.checkCfgFrictionModel(_getHysteresisCfg(hysteresisNu="-0.001"), inputSimFiles)
    assert "hysteresisNu must not be negative" in str(e.value)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        checkCfg.checkCfgFrictionModel(_getHysteresisCfg(hysteresisViscosity="1", hysteresisNu="0"), inputSimFiles)
    assert "viscous term has no effect" in caplog.text


def _getEntMethodCfg(**overrides):
    """valid static layer settings, single settings replaced by overrides"""
    cfg = {
        "GENERAL": {
            "frictModel": "hysteresis",
            "entMethod": "staticParticles",
            "entPartDistType": "uniform",
            "entrainableDeposition": "False",
            "snowSlide": "0",
            "viscOption": "1",
            "adaptSfcStopped": "0",
            "adaptSfcEntrainment": "0",
        }
    }
    cfg["GENERAL"].update({key: str(value) for key, value in overrides.items()})
    return cfg


def test_checkCfgEntrainmentMethod(caplog):
    """test entMethod checks: rate is always valid, staticParticles requires hysteresis and compatible settings"""
    # rate (default) is valid with any friction model, a missing entMethod means rate
    checkCfg.checkCfgEntrainmentMethod({"GENERAL": {"frictModel": "samosAT"}})
    checkCfg.checkCfgEntrainmentMethod(_getEntMethodCfg(entMethod="rate", frictModel="samosAT", entrainableDeposition="True"))

    # valid static layer settings: pass without warning
    with caplog.at_level(logging.WARNING):
        cfg = checkCfg.checkCfgEntrainmentMethod(_getEntMethodCfg())
    assert cfg["GENERAL"]["entMethod"] == "staticParticles"
    assert caplog.records == []

    with pytest.raises(ValueError) as e:
        checkCfg.checkCfgEntrainmentMethod(_getEntMethodCfg(entMethod="static"))
    assert "must be either rate or staticParticles" in str(e.value)

    with pytest.raises(ValueError) as e:
        checkCfg.checkCfgEntrainmentMethod(_getEntMethodCfg(frictModel="samosAT"))
    assert "requires frictModel = hysteresis" in str(e.value)

    with pytest.raises(ValueError) as e:
        checkCfg.checkCfgEntrainmentMethod(_getEntMethodCfg(entrainableDeposition="True"))
    assert "not compatible with entrainableDeposition" in str(e.value)

    with pytest.raises(ValueError) as e:
        checkCfg.checkCfgEntrainmentMethod(_getEntMethodCfg(snowSlide="1"))
    assert "not compatible with snowSlide" in str(e.value)

    with pytest.raises(ValueError) as e:
        checkCfg.checkCfgEntrainmentMethod(_getEntMethodCfg(entPartDistType="grid"))
    assert "entPartDistType is grid" in str(e.value)

    # without subgrid mixing the flow can pass over the layer
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        checkCfg.checkCfgEntrainmentMethod(_getEntMethodCfg(viscOption="0"))
    assert "subgrid mixing" in caplog.text

    # adaptive surface: stopped particles would remove the layer, entrainment adaptation has no effect
    with pytest.raises(ValueError) as e:
        checkCfg.checkCfgEntrainmentMethod(_getEntMethodCfg(adaptSfcStopped="1"))
    assert "not compatible with adaptSfcStopped" in str(e.value)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        checkCfg.checkCfgEntrainmentMethod(_getEntMethodCfg(adaptSfcEntrainment="1"))
    assert "adaptSfcEntrainment = 1 has no effect" in caplog.text
    # entLayerInFields must be a boolean
    checkCfg.checkCfgEntrainmentMethod(_getEntMethodCfg(entLayerInFields="True"))
    with pytest.raises(ValueError) as e:
        checkCfg.checkCfgEntrainmentMethod(_getEntMethodCfg(entLayerInFields="maybe"))
    assert "entLayerInFields is maybe" in str(e.value)
