"""Tests for module cfgHandling"""


import pathlib
import pytest
import configparser
import logging

from avaframe.in3Utils import cfgUtils
from avaframe.in3Utils import cfgHandling
from avaframe.tests import test_logUtils
from avaframe.com1DFA import com1DFA

log = logging.getLogger(__name__)


def test_addInfoToSimName():
    """Test for addInfoToSimname"""
    avaTestDir = "avaParabolaStatsTest"
    dirPath = pathlib.Path(__file__).parents[0]
    avaDir = dirPath / ".." / ".." / "benchmarks" / avaTestDir
    renameDF = cfgHandling.addInfoToSimName(avaDir, "mu")
    assert renameDF.loc["c4f3a000c3"]["newName"] == "release1PF_c4f3a000c3_mu_0.155_null_dfa"
    renameDF = cfgHandling.addInfoToSimName(avaDir, "mu,tEnd")
    assert renameDF.loc["c4f3a000c3"]["newName"] == "release1PF_c4f3a000c3_mu_0.155_tEnd_400_null_dfa"


def test_orderSimFiles():
    """test generating order of simulation results"""

    avaTestDir = "avaHockeyChannelPytest"
    dirPath = pathlib.Path(__file__).parents[0]
    avaDir = dirPath / ".." / ".." / "benchmarks" / avaTestDir

    varParList = "releaseScenario"

    simDF = cfgUtils.createConfigurationInfo(avaDir, specDir="")

    varParList, simDF = cfgHandling.orderSimulations(varParList, True, simDF)

    assert simDF["simName"].iloc[0] == "release1HS_0dcd58fc86_ent_dfa"

    varParList = "releaseSenario"
    message = "Choose a valid parameter for sorting the simulations. 'releaseSenario' is not valid."
    with pytest.raises(KeyError) as e:
        assert cfgHandling.orderSimulations(varParList, True, simDF)
    assert message in str(e.value)


def test_fetchAndOrderSimFiles():
    """test generating order of simulation results"""

    avaTestDir = "avaHockeyChannelPytest"
    dirPath = pathlib.Path(__file__).parents[0]
    avaDir = dirPath / ".." / ".." / "benchmarks" / avaTestDir
    inputDir = avaDir / "Outputs" / "com1DFA" / "peakFiles"

    varParList = "releaseScenario"

    simDF = cfgHandling.fetchAndOrderSimFiles(avaDir, inputDir, varParList, True, specDir="", resFiles=True)

    assert simDF["simName"][0] == "release1HS_0dcd58fc86_ent_dfa"

    varParList = "releaseSenario"
    message = "Choose a valid parameter for sorting the simulations. 'releaseSenario' is not valid."
    with pytest.raises(KeyError) as e:
        simDF = cfgHandling.fetchAndOrderSimFiles(
            avaDir, "inputDir", varParList, True, specDir="", resFiles=False
        )
    assert message in str(e.value)


def test_filterSims(tmp_path):
    """test filtering of simulations using configuration files"""

    avaTestDir = "avaHockeyChannelPytest"
    dirPath = pathlib.Path(__file__).parents[0]
    avaDir = dirPath / ".." / ".." / "benchmarks" / avaTestDir
    avaDir2 = dirPath / "data" / "avaFilterTest" / "com1DFA"

    parametersDict = {"releaseScenario": "release1HS"}

    simNames = cfgHandling.filterSims(avaDir, parametersDict, specDir="")

    testRel = False
    testRel2 = False
    if "release1HS" in simNames[0]:
        testRel = True
    if "release2HS" in simNames[0]:
        testRel2 = True

    assert testRel
    assert testRel2 is False
    assert len(simNames) == 1

    parametersDict = {"relTh": 1.0}
    simNames2 = cfgHandling.filterSims(avaDir, parametersDict, specDir="")
    simNames2 = sorted(simNames2)
    assert len(simNames2) == 2
    assert simNames2[0] == "release1HS_0dcd58fc86_ent_dfa"
    assert simNames2[1] == "release2HS_3d519adab0_ent_dfa"

    parametersDict = {"~relTh": 1.0}
    simNames3 = cfgHandling.filterSims(avaDir, parametersDict, specDir="")

    assert len(simNames3) == 0

    parametersDict = {"~releaseScenario": "release1HS"}
    simNames4 = cfgHandling.filterSims(avaDir, parametersDict, specDir="")

    assert len(simNames4) == 1
    assert simNames4[0] == "release2HS_3d519adab0_ent_dfa"

    parametersDict = {"relTh": 1.0}

    simNames = cfgHandling.filterSims(avaDir2, parametersDict, specDir=avaDir2)
    simNames = sorted(simNames)

    assert len(simNames) == 2
    assert simNames == ["relGar_6f35cbd808_null_dfa", "relGar_b9b17dd019_ent_dfa"]

    parametersDict = {"relTh": [1.0, 1.5]}

    simNames = cfgHandling.filterSims(avaDir2, parametersDict, specDir=avaDir2)
    simNames = sorted(simNames)

    assert len(simNames) == 4
    assert simNames == [
        "relGar_1022880a70_null_dfa",
        "relGar_6f35cbd808_null_dfa",
        "relGar_b9b17dd019_ent_dfa",
        "relGar_c4337e50ac_null_dfa",
    ]

    parametersDict = {"relTh": ">1.0"}

    simNames = cfgHandling.filterSims(avaDir2, parametersDict, specDir=avaDir2)
    simNames = sorted(simNames)
#    print("SIMAMES", simNames)

    assert len(simNames) == 4
    assert simNames == [
        "relGar_1022880a70_null_dfa",
        "relGar_789ce37489_ent_dfa",
        "relGar_9b75355a9a_ent_dfa",
        "relGar_c4337e50ac_null_dfa",
    ]

    parametersDict = {"relTh": "<1.0"}

    simNames = cfgHandling.filterSims(avaDir2, parametersDict, specDir=avaDir2)
    simNames = sorted(simNames)

    assert len(simNames) == 1
    assert simNames == ["relGar_d5a270b689_ent_dfa"]


def test_applyCfgOverride(caplog):
    """test overriding cfg parameters in a cfg object from another cfg with an override section"""

    cfgWithOverrideParameters = configparser.ConfigParser()
    cfgWithOverrideParameters.optionxform = str
    cfgWithOverrideParameters["GENERAL"] = {"testp1": 1.0, "testp2": "testValue"}
    cfgWithOverrideParameters["com1DFA_com1DFA_override"] = {
        "defaultConfig": True,
        "mu": 0.7,
        "tStep": 100.0,
        "notParameter": 1.0,
    }

    cfgToOverride = configparser.ConfigParser()
    cfgToOverride.optionxform = str
    cfgToOverride["GENERAL"] = {"mu": 1.0, "tEnd": 400.0, "relTh": 1.0}
    cfgToOverride["DFA"] = {"tStep": 0.3, "dt1": 6.0, "flowF": 3.0}

    cfgToOverride, cfgWithOverrideParameters = cfgHandling.applyCfgOverride(
        cfgToOverride, cfgWithOverrideParameters, com1DFA, addModValues=False
    )

    assert cfgToOverride["GENERAL"]["mu"] == "0.7"
    assert cfgToOverride["GENERAL"]["tEnd"] == "400.0"
    assert cfgToOverride["GENERAL"]["relTh"] == "1.0"
    assert cfgToOverride["DFA"]["tStep"] == "100.0"
    assert cfgToOverride["DFA"]["dt1"] == "6.0"
    assert cfgToOverride["DFA"]["flowF"] == "3.0"
    assert cfgWithOverrideParameters["GENERAL"]["testp1"] == "1.0"
    assert cfgWithOverrideParameters["GENERAL"]["testp2"] == "testValue"
    assert cfgWithOverrideParameters["com1DFA_com1DFA_override"]["defaultConfig"] == "True"
    assert cfgWithOverrideParameters["com1DFA_com1DFA_override"]["mu"] == "0.7"
    assert cfgWithOverrideParameters["com1DFA_com1DFA_override"]["tStep"] == "100.0"
    assert cfgWithOverrideParameters["com1DFA_com1DFA_override"]["notParameter"] == "1.0"
    assert len(cfgWithOverrideParameters.items("GENERAL")) == 2
    assert len(cfgWithOverrideParameters.items("com1DFA_com1DFA_override")) == 4
    assert len(cfgToOverride.items("GENERAL")) == 3
    assert len(cfgToOverride.items("GENERAL")) == 3

    cfgWithOverrideParameters = configparser.ConfigParser()
    cfgWithOverrideParameters.optionxform = str
    cfgWithOverrideParameters["GENERAL"] = {"testp1": 1.0, "testp2": "testValue"}
    cfgWithOverrideParameters["com1DFA_com1DFA_override"] = {
        "defaultConfig": True,
        "mu": 0.7,
        "tStep": 100.0,
        "notParameter": 1.0,
    }

    cfgToOverride = configparser.ConfigParser()
    cfgToOverride.optionxform = str
    cfgToOverride["GENERAL"] = {"mu": 1.0, "tEnd": 400.0, "relTh": 1.0}
    cfgToOverride["DFA"] = {"tStep": 0.3, "dt1": 6.0, "flowF": 3.0}

    with caplog.at_level(logging.WARNING):
        cfgToOverride, cfgWithOverrideParameters = cfgHandling.applyCfgOverride(
            cfgToOverride, cfgWithOverrideParameters, com1DFA, addModValues=True
        )
    assert (
        "Additional Key ['%s'] in section com1DFA_%s_override is ignored." % ("notParameter", "com1DFA")
    ) in caplog.text

    assert cfgToOverride["GENERAL"]["mu"] == "0.7"
    assert cfgToOverride["GENERAL"]["tEnd"] == "400.0"
    assert cfgToOverride["GENERAL"]["relTh"] == "1.0"
    assert cfgToOverride["DFA"]["tStep"] == "100.0"
    assert cfgToOverride["DFA"]["dt1"] == "6.0"
    assert cfgToOverride["DFA"]["flowF"] == "3.0"
    assert cfgWithOverrideParameters["GENERAL"]["testp1"] == "1.0"
    assert cfgWithOverrideParameters["GENERAL"]["testp2"] == "testValue"
    assert cfgWithOverrideParameters["com1DFA_com1DFA_override"]["defaultConfig"] == "True"
    assert cfgWithOverrideParameters["com1DFA_com1DFA_override"]["mu"] == "0.7"
    assert cfgWithOverrideParameters["com1DFA_com1DFA_override"]["tStep"] == "100.0"
    assert cfgWithOverrideParameters["com1DFA_com1DFA_override"]["notParameter"] == "1.0"
    assert cfgWithOverrideParameters["com1DFA_com1DFA_override"]["tEnd"] == "400.0"
    assert cfgWithOverrideParameters["com1DFA_com1DFA_override"]["dt1"] == "6.0"
    assert cfgWithOverrideParameters["com1DFA_com1DFA_override"]["flowF"] == "3.0"
    assert cfgWithOverrideParameters["com1DFA_com1DFA_override"]["relTh"] == "1.0"
    assert len(cfgWithOverrideParameters.items("GENERAL")) == 2
    assert len(cfgWithOverrideParameters.items("com1DFA_com1DFA_override")) == 8
    assert len(cfgToOverride.items("GENERAL")) == 3
    assert len(cfgToOverride.items("GENERAL")) == 3


