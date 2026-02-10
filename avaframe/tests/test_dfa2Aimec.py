''' Tests for dfa2Aimec '''

import pathlib
import pytest
import configparser
import numpy as np
import pandas as pd

# Local imports
import avaframe.ana3AIMEC.dfa2Aimec as dfa2Aimec


def test_mainDfa2Aimec(tmp_path):

    # Initialise inputs
    dirPath = pathlib.Path(__file__).parents[0]
    avaTestName = 'avaHockeyChannelPytest'
    testPath = dirPath / '..' / '..' / 'benchmarks' / avaTestName
    pathData = testPath / 'Outputs' / 'com1DFA' / 'peakFiles'
    cfg = configparser.ConfigParser()
    cfg['AIMECSETUP'] = {'varParList': 'releaseScenario', 'ascendingOrder': 'True'}
    cfg['FLAGS'] = {'flagMass': 'True'}
    inputDF, resTypeList = dfa2Aimec.mainDfa2Aimec(testPath, 'com1DFA', cfg)
#    print('path', dirPath)
    # get path dictionary for test
    pathDTest = {}
    pathDTest['ppr'] = [pathData / 'release1HS_0dcd58fc86_ent_dfa_ppr.asc',
                        pathData / 'release2HS_3d519adab0_ent_dfa_ppr.asc']
    pathDTest['pft'] = [pathData / 'release1HS_0dcd58fc86_ent_dfa_pft.asc',
                        pathData / 'release2HS_3d519adab0_ent_dfa_pft.asc']
    pathDTest['pfv'] = [pathData / 'release1HS_0dcd58fc86_ent_dfa_pfv.asc',
                        pathData / 'release2HS_3d519adab0_ent_dfa_pfv.asc']
    pathDTest['massBal'] = [testPath / 'Outputs' / 'com1DFA' / 'mass_release1HS_0dcd58fc86_ent_dfa.txt',
                            testPath / 'Outputs' / 'com1DFA' / 'mass_release2HS_3d519adab0_ent_dfa.txt']
#    print(inputDF['ppr'].to_list())
#    print(pathDTest['ppr'])
    diff = set(inputDF['ppr'].to_list()) ^ set(pathDTest['ppr'])
    assert not diff
    diff = set(inputDF['pft'].to_list()) ^ set(pathDTest['pft'])
    assert not diff
    diff = set(inputDF['pfv'].to_list()) ^ set(pathDTest['pfv'])
    assert not diff
    diff = set(inputDF['massBal'].to_list()) ^ set(pathDTest['massBal'])
    assert not diff


def test_dfaComp2Aimec(tmp_path):

    # Initialise inputs
    dirPath = pathlib.Path(__file__).parents[0]
    testPath = dirPath / '..' / '..' / 'benchmarks' / 'avaHelixChannelPytest'
    pathDataRef = testPath / 'Outputs' / 'com1DFARef' / 'peakFiles'
    pathDataComp = testPath / 'Outputs' / 'com1DFAComp' / 'peakFiles'
    cfg = configparser.ConfigParser()
    cfg['AIMECSETUP'] = {'comModules': 'com1DFARef|com1DFAComp', 'testName': '', 'referenceSimValue': '',
                         'referenceSimName': '', 'varParList': '',
                         'runoutResType': 'ppr', 'resTypes': 'ppr|pft|pfv', 'runoutLayer': ''}
    cfg['FLAGS'] = {'flagMass': 'True'}
    inputDF, pathDict = dfa2Aimec.dfaBench2Aimec(testPath, cfg, 'release1HX', 'release1HX')

    # get path dictionary for test
    massNameRef = 'mass_release1HX_25476e950f_ent_dfa.txt'
    massNameComp = 'mass_release1HX_0a452280a5_ent_dfa.txt'
    pathDTest = {}
    pathDTest['ppr'] = [pathDataRef / 'release1HX_25476e950f_ent_dfa_ppr.asc',
                        pathDataComp / 'release1HX_0a452280a5_ent_dfa_ppr.asc']
    pathDTest['pft'] = [pathDataRef / 'release1HX_25476e950f_ent_dfa_pft.asc',
                        pathDataComp / 'release1HX_0a452280a5_ent_dfa_pft.asc']
    pathDTest['pfv'] = [pathDataRef / 'release1HX_25476e950f_ent_dfa_pfv.asc',
                        pathDataComp / 'release1HX_0a452280a5_ent_dfa_pfv.asc']
    pathDTest['massBal'] = [testPath / 'Outputs' / 'com1DFARef' / massNameRef,
                            testPath / 'Outputs' / 'com1DFAComp' / massNameComp]
#    print(inputDF['ppr'].to_list())
#    print(pathDTest['ppr'])
    diff = set(inputDF['ppr'].to_list()) ^ set(pathDTest['ppr'])
    assert not diff
    diff = set(inputDF['pft'].to_list()) ^ set(pathDTest['pft'])
    assert not diff
    diff = set(inputDF['pfv'].to_list()) ^ set(pathDTest['pfv'])
    assert not diff
    diff = set(inputDF['massBal'].to_list()) ^ set(pathDTest['massBal'])
    assert not diff

    with pytest.raises(ValueError) as e:
        assert dfa2Aimec.dfaBench2Aimec(testPath, cfg, 'release2HX', 'release1HX')
#        print(e)
    assert ('Found no simulation matching the reference criterion release2HX, there should be one') in str(e.value)


def test_getRefMB():
    """ test get reference mass balance info """

    # setup required input
    avaTestName = 'avaHockeyChannelPytest'
    simName = 'release1HS_ent_dfa_3d519adab0'
    d = {'simName': [simName]}
    inputDF = pd.DataFrame(data=d, index=[0])

    # call function to be tested
    inputDF = dfa2Aimec.getRefMB(avaTestName, inputDF, simName)

#    print('pathDict', inputDF)

    assert 'mass_release1HS_ent_dfa_3d519adab0' in str(inputDF['massBal'].to_list()[0])


def test_getMBInfo():
    """ test get mass balance info """

    # setup required input
    dirPath = pathlib.Path(__file__).parents[0]
    avaTestName = 'avaHockeyChannelPytest'
    avaDir = dirPath / '..' / '..' / 'benchmarks' / avaTestName
    pathDict = {'ppr': 'test', 'massBal': []}
    comMod = 'com1DFA'
    simName = ''
    inputsDF = pd.DataFrame()
    # call function to be tested and check for correct error if file does not exist
    with pytest.raises(FileNotFoundError) as e:
        assert dfa2Aimec.getMBInfo(avaDir, inputsDF, comMod, simName='testName')
    assert 'mass_testName.txt' in str(e.value)

    with pytest.raises(FileNotFoundError) as e:
        comModTest = 'com1DFATest'
        assert dfa2Aimec.getMBInfo(avaDir, inputsDF, comModTest, simName='')
    assert 'avaHockeyChannelPytest/Outputs/com1DFATest' in str(e.value)

    # call fucntion to be tested
    inputsDF = pd.DataFrame(data={'simName': ['release1HS_0dcd58fc86_ent_dfa', 'release2HS_3d519adab0_ent_dfa']},
                            index=[0, 1])
    inputsDF = dfa2Aimec.getMBInfo(avaDir, inputsDF, comMod, simName=simName)

#    print('pathDict', pathDict)
    assert 'mass_release1HS_0dcd58fc86_ent_dfa' in str(inputsDF['massBal'][0])
    assert 'mass_release2HS_3d519adab0_ent_dfa' in str(inputsDF['massBal'][1])

    # call fucntion to be tested
    simName = 'release1HS_0dcd58fc86_ent_dfa'
    inputsDF = pd.DataFrame(data={'simName': ['release1HS_0dcd58fc86_ent_dfa']},
                            index=[0])
    inputsDF = dfa2Aimec.getMBInfo(avaDir, inputsDF, comMod, simName=simName)

#    print('pathDict2', inputsDF)
    assert 'mass_release1HS_0dcd58fc86_ent_dfa' in str(inputsDF['massBal'][0])
    assert 'mass_release2HS_3d519adab0_ent_dfa' not in str(inputsDF['massBal'][0])


def test_dfaBench2Aimec():
    """ test export data used for aimec """

    # setup required input
    cfg = configparser.ConfigParser()
    cfg['AIMECSETUP'] = {'comModules': 'com1DFARef|com1DFAComp', 'testName': '', 'referenceSimValue': '',
                         'referenceSimName': '', 'varParList': '',
                         'runoutResType': 'ppr', 'resTypes': 'ppr|pft|pfv', 'runoutLayer': ''}
    cfg['FLAGS'] = {'flagMass': 'True'}
    dirPath = pathlib.Path(__file__).parents[0]
    avaTestName = 'avaHelixChannelPytest'
    avaDir = dirPath / '..' / '..' / 'benchmarks' / avaTestName
    simNameRef = 'release1HX_25476e950f_ent_dfa'
    simNameComp = 'release1HX_0a452280a5_ent_dfa'

    # call function to be tested
    inputsDF, pathDict = dfa2Aimec.dfaBench2Aimec(avaDir, cfg, simNameRef, simNameComp)

    assert 'mass_release1HX_0a452280a5_ent_dfa' in str(inputsDF['massBal'].iloc[1])
    assert 'release1HX_0a452280a5_ent_dfa_pft' in str(inputsDF['pft'].iloc[1])
    assert 'release1HX_0a452280a5_ent_dfa_ppr' in str(inputsDF['ppr'].iloc[1])
    assert 'release1HX_0a452280a5_ent_dfa_pfv' in str(inputsDF['pfv'].iloc[1])
    assert 'mass_release1HX_25476e950f_ent_dfa' in str(inputsDF['massBal'].iloc[0])
    assert 'release1HX_25476e950f_ent_dfa_pft' in str(inputsDF['pft'].iloc[0])
    assert 'release1HX_25476e950f_ent_dfa_ppr' in str(inputsDF['ppr'].iloc[0])
    assert 'release1HX_25476e950f_ent_dfa_pfv' in str(inputsDF['pfv'].iloc[0])


def test_getCompDirs():
    """ test get comparison dirs """

    # setup required input
    dirPath = pathlib.Path(__file__).parents[0]
    avaTestName = 'avaHockeyChannelPytest'
    avaDir = dirPath / '..' / '..' / 'benchmarks' / avaTestName
    cfg = configparser.ConfigParser()
    cfg['AIMECSETUP'] = {'comModules': 'com1DFAOrig|com1DFA'}
    pathDict = {'ppr': []}

    # call function to be tested
    inputDirRef, inputDirComp, pathDict = dfa2Aimec.getCompDirs(avaDir, cfg['AIMECSETUP'])

#    print('inputDirRef', inputDirRef)
#    print('inputDirComp', inputDirComp)
#    print('pathDict', pathDict)
#    print('refModule', pathDict['compType'][1])

    assert str(inputDirRef) == str(avaDir / 'Outputs' / 'com1DFAOrig' / 'peakFiles')
    assert str(inputDirComp) == str(avaDir / 'Outputs' / 'com1DFA' / 'peakFiles')
    assert pathDict['compType'][1] == 'com1DFAOrig'
    assert pathDict['compType'] == ['comModules', 'com1DFAOrig', 'com1DFA']
    assert pathDict['contCmap'] is True

    # call function to be tested
    cfg['AIMECSETUP'] = {'comModules': 'benchmarkReference|com1DFA'}
    pathDict = {'ppr': []}
    cfg['AIMECSETUP']['testName'] = 'avaHockeyChannelEntTest'
    inputDirRef, inputDirComp, pathDict = dfa2Aimec.getCompDirs(avaDir, cfg['AIMECSETUP'])

    testPath = dirPath / '..' / '..' / 'benchmarks' / 'avaHockeyChannelEntTest'

    assert str(inputDirRef) == '../benchmarks/avaHockeyChannelEntTest'
    assert str(inputDirComp) == str(avaDir / 'Outputs' / 'com1DFA' / 'peakFiles')
    assert pathDict['compType'][1] == 'benchmarkReference'
    assert pathDict['compType'] == ['comModules', 'benchmarkReference', 'com1DFA']
    assert pathDict['contCmap'] is True
