"""
    Pytest for module com4FlowPy
"""

#  Load modules
import numpy as np
import pytest
from numpy.ma.core import ones_like

from avaframe.com4FlowPy import flowClass
import avaframe.com4FlowPy.flowCore as flowCore

def test_add_os():
    cell = flowClass.Cell(1,1,
                          np.array([[10,10,10], [10,10,10], [10,10,10]]), 10,
                          1,0, None,
                          20, 8, 3e-4, 270,
                          startcell=True)
    cell.add_os(0.2)
    assert cell.flux == 1.2

def test_reverseTopology():
    '''
    testing flowCore.reverseTopology() for different
    examples of dir graphs
    '''

    testGraph = {0: [1,2,3],
                      1: [2,4],
                      2: [4,5],
                      3: [2],
                      4: [],
                      5: [6],
                      6: []}

    testGraphReverse = {0: [],
                        1: [0],
                        2: [0,1,3],
                        3: [0],
                        4: [2,1],
                        5: [2],
                        6: [5]}

    reverseGraphCalc = flowCore.reverseTopology(testGraph)

    for key, item in reverseGraphCalc.items():
        assert key in testGraphReverse.keys()
        setTestChildren = set(item)
        setCalcChildren = set(testGraphReverse[key])
        assert setTestChildren == setCalcChildren

    testGraph = {0:[1,2,3],
                 1:[4],
                 2:[5,6],
                 3:[7,8],
                 4:[9],
                 5:[9,10],
                 6:[10],
                 7:[11],
                 8:[],
                 9:[],
                 10:[12],
                 11:[],
                 12:[]}

    testGraphReverse = {0:[],
                        1:[0],
                        2:[0],
                        3:[0],
                        4:[1],
                        5:[2],
                        6:[2],
                        7:[3],
                        8:[3],
                        9:[4,5],
                        10:[5,6],
                        11:[7],
                        12:[10]}

    reverseGraphCalc = flowCore.reverseTopology(testGraph)

    for key, item in reverseGraphCalc.items():
        assert key in testGraphReverse.keys()
        setTestChildren = set(item)
        setCalcChildren = set(testGraphReverse[key])
        assert setTestChildren == setCalcChildren


def test_backTracking():
    '''
    testing flowCore.backTracking() for different
    examples of dir graphs - basic graphs are the same as
    in test_reverseTopology() with added 'infra' values
    as valueDicts
    '''
    
    testGraph = {0: [1,2,3],
                 1: [2,4],
                 2: [4,5],
                 3: [2],
                 4: [],
                 5: [6],
                 6: []}
    
    testValsIn = {0: 0,
                  1: 0,
                  2: 0,
                  3: 0,
                  4: 3,
                  5: 0,
                  6: 2}
    
    testValsBT = {0: 3,
                  1: 3,
                  2: 3,
                  3: 3,
                  4: 3,
                  5: 2,
                  6: 2}
    
    calcValsBT = flowCore.backTracking(testGraph, testValsIn)

    for key,item in calcValsBT.items():
        assert calcValsBT[key] == testValsBT[key]
    
    testGraph = {0:[1,2,3],
                 1:[4],
                 2:[5,6],
                 3:[7,8],
                 4:[9],
                 5:[9,10],
                 6:[10],
                 7:[11],
                 8:[],
                 9:[],
                 10:[12],
                 11:[],
                 12:[]}
    
    testValsIn = {0:0,
                  1:0,
                  2:0,
                  3:0,
                  4:0,
                  5:0,
                  6:0,
                  7:0,
                  8:0,
                  9:1,
                  10:0,
                  11:3,
                  12:2}
    
    testValsBT = {0:3,
                  1:1,
                  2:2,
                  3:3,
                  4:1,
                  5:2,
                  6:2,
                  7:3,
                  8:0,
                  9:1,
                  10:2,
                  11:3,
                  12:2}
    
    calcValsBT = flowCore.backTracking(testGraph, testValsIn)

    for key,item in calcValsBT.items():
        assert calcValsBT[key] == testValsBT[key]


def test_calculation():
    dem = np.array(
        [[40, 40, 40, 40, 40], [30, 30, 30, 30, 30], [20, 20, 20, 20, 20], [10, 10, 10, 10, 10], [0, 0, 0, 0, 0]])
    infra = None
    pra = np.array([[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
    alpha = 10
    exp = 99
    fluxTh = 0.001
    zDeltaMax = 8000
    nodata = -9999
    cellsize = 10
    infraBool = False
    forestBool = False
    variableParameters = {
        "varUmaxBool": False,
        "varUmaxArray": None,
        "varAlphaBool": False,
        "varAlphaArray": None,
        "varExponentBool": False,
        "varExponentArray": None,
    }
    fluxDistOldVersionBool = False
    previewMode = False
    outputs = ['travelLengthMin', 'flux']
    forestArray = None
    forestParams = None
    args = [dem, infra, pra, alpha, exp, fluxTh, zDeltaMax, nodata, cellsize, infraBool, forestBool, variableParameters,
            fluxDistOldVersionBool, previewMode, forestArray, forestParams, outputs]

    flux = ones_like(dem) * -9999.
    flux[[1, 2, 3], [2, 2, 2]] = 1
    depFluxSum = np.zeros_like(dem)
    routFluxSum = np.where(flux == 1, 1., 0.)
    travelLengthMin = ones_like(dem) * -9999.
    travelLengthMin[1, 2] = 0
    travelLengthMin[2, 2] = np.sqrt(cellsize ** 2)
    travelLengthMin[3, 2] = 2 * np.sqrt(cellsize ** 2)
    results = flowCore.calculation(args)

    assert len(results) == 12
    assert np.all(results[1] == flux)
    assert np.all(results[10] == routFluxSum)
    assert np.all(results[11] == depFluxSum)
    assert np.all(results[8] == travelLengthMin)
    assert np.all(results[7] == np.ones_like(flux) * -9999)


if __name__=='__main__':
    test_add_os()
    test_reverseTopology()
    test_backTracking()
    test_calculation()
