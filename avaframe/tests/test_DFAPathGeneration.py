"""Tests for module DFAPathGeneration"""
import numpy as np
import math
import configparser
import pytest

# Local imports
import avaframe.ana5Utils.DFAPathGeneration as DFAPathGeneration
import avaframe.in3Utils.geoTrans as gT
import avaframe.in3Utils.fileHandlerUtils as fU


def test_appendAverageStd():
    values = np.array([1, 2, 3, 4, 5])
    weights = np.array([2, 1, 2, 1, 2])
    average, std = DFAPathGeneration.weightedAvgAndStd(values, weights)
    assert average == 3
    assert std == 1.5
    proList = ['x', 'y', 'u2', 'ekin']
    particles = {'x': values, 'y': values, 'u2': 2*values, 'ekin': 10*2*values}
    avaProfile = {'x': np.empty((0, 1)), 'y': np.empty((0, 1)), 'xstd': np.empty((0, 1)), 'ystd': np.empty((0, 1)),
                  'u2': np.empty((0, 1)), 'u2std': np.empty((0, 1)), 'ekin': np.empty((0, 1)),
                  'ekinstd': np.empty((0, 1)), 'totEKin': np.empty((0, 1))}
    avaProfile = DFAPathGeneration.appendAverageStd(proList, avaProfile, particles, weights)
#    print(avaProfile)
    assert avaProfile['x'] == 3
    assert avaProfile['xstd'] == 1.5
    assert avaProfile['y'] == 3
    assert avaProfile['ystd'] == 1.5
    assert avaProfile['u2'] == 6
    assert avaProfile['u2std'] == 3
    assert avaProfile['ekin'] == 60
    assert avaProfile['ekinstd'] == 30


def test_getDFAPathFromPart():
    values = np.array([1, 2, 3, 4, 5])
    weights = np.array([2, 1, 2, 1, 2])
    average, std = DFAPathGeneration.weightedAvgAndStd(values, weights)
    assert average == 3
    assert std == 1.5
    particlesList = [{'nPart': 5, 'm': weights, 'x': values, 'y': values, 'z': values,
                      'trajectoryLengthXY': values, 'trajectoryLengthXYCor': values,
                      'ux': values, 'uy': values, 'uz': values}]
    avaProfile = DFAPathGeneration.getMassAvgPathFromPart(particlesList, addVelocityInfo=False)
#    print(avaProfile)
    for prop in ['x', 'y', 'z', 's', 'sCor']:
        assert avaProfile[prop] == 3
        assert avaProfile[prop + 'std'] == 1.5

    avaProfile = DFAPathGeneration.getMassAvgPathFromPart(particlesList, addVelocityInfo=True)
#    print(avaProfile)
    for prop in ['x', 'y', 'z', 's', 'sCor']:
        assert avaProfile[prop] == 3
        assert avaProfile[prop + 'std'] == 1.5
    assert avaProfile['u2'] == 33.75
    assert avaProfile['u2std'] == pytest.approx(27.52612396, abs=1e-6)
    assert avaProfile['ekin'] == pytest.approx(30, abs=1e-6)
    assert avaProfile['ekinstd'] == pytest.approx(27.69927797, abs=1e-6)


def test_extendDFAPath():
    """"""
    # setup required inputs
    cfg = configparser.ConfigParser()
    cfg['PATH'] = {'nCellsResample': '1', 'extTopOption': '1', 'nCellsMinExtend': '1',
                   'nCellsMaxExtend': '2', 'factBottomExt': 0.2,
                   'maxIterationExtBot': 10, 'nBottomExtPrecision': 10,
                   'uInterval': '1000'}

    # TODO if k=3 for spline needs at least 4 pointsin path
    avaProfile = {'x': np.array([1, 2, 3, 8]), 'y': np.array([1, 2, 3, 8]), 'z': np.array([40, 30, 20, 0])
                  }#'s': np.array([0, np.sqrt(4), np.sqrt(8), np.sqrt(98)])}
    particlesIni = {'x': np.array([0.7, 0.69]),
                    'y': np.array([1, 2])}
    dem = {'header': {'xllcenter': 0, 'yllcenter': 0, 'cellsize': 2, 'nrows': 10, 'ncols': 11},
           'rasterData': np.array([[50, 40, 30, 20, 10, 0, 0, 0, 0, 0, 0],
                                   [50, 40, 30, 20, 10, 0, 0, 0, 0, 0, 0],
                                   [50, 40, 30, 20, 10, 0, 0, 0, 0, 0, 0],
                                   [50, 40, 30, 20, 10, 0, 0, 0, 0, 0, 0],
                                   [50, 40, 30, 20, 10, 0, 0, 0, 0, 0, 0],
                                   [50, 40, 30, 20, 10, 0, 0, 0, 0, 0, 0],
                                   [50, 40, 30, 20, 10, 0, 0, 0, 0, 0, 0],
                                   [50, 40, 30, 20, 10, 0, 0, 0, 0, 0, 0],
                                   [50, 40, 30, 20, 10, 0, 0, 0, 0, 0, 0],
                                   [50, 40, 30, 20, 10, 0, 0, 0, 0, 0, 0]])}

    particlesIni, _ = gT.projectOnRaster(dem, particlesIni, interp='bilinear')

    # using the longest runout method
    avaProfileExt = DFAPathGeneration.extendDFAPath(cfg['PATH'], avaProfile, dem, particlesIni)
#    print(avaProfileExt)
    atol = 1e-10
    assert avaProfileExt['x'][0] == 0.7
    assert avaProfileExt['x'][-1] == pytest.approx(9.44242641, abs=1e-6)
    assert avaProfileExt['y'][0] == 1.
    assert avaProfileExt['y'][-1] == pytest.approx(9.44242641, abs=1e-6)
    assert avaProfileExt['z'][0] == pytest.approx(46.5, abs=1e-6)
    assert avaProfileExt['z'][-1] == pytest.approx(2.78786797, abs=1e-6)

    # now use the highest point method
    cfg = configparser.ConfigParser()
    cfg['PATH'] = {'nCellsResample': '5', 'extTopOption': '0', 'nCellsMinExtend': '1',
                   'nCellsMaxExtend': '2', 'factBottomExt': 0.2, 'maxIterationExtBot': 10, 'nBottomExtPrecision': 10}
    avaProfile = {'x': np.array([10, 20, 30]), 'y': np.array([10, 20, 30]), 'z': np.array([40, 30, 20])}
    particlesIni = {'x': np.array([7., 6.9]),
                    'y': np.array([10, 20])}
    dem['header']['cellsize'] = 10.

    particlesIni, _ = gT.projectOnRaster(dem, particlesIni, interp='bilinear')


    avaProfileExt = DFAPathGeneration.extendDFAPath(cfg['PATH'], avaProfile, dem, particlesIni)
#    print(avaProfileExt)
    atol = 1e-10
    assert np.allclose(avaProfileExt['x'][0], 6.9, atol=atol)
    assert avaProfileExt['x'][-1] == pytest.approx(30., abs=1e-6)
    assert np.allclose(avaProfileExt['y'][0], 20.0, atol=atol)
    assert avaProfileExt['y'][-1] == pytest.approx(30., abs=1e-6)
    assert avaProfileExt['z'][0] == pytest.approx(43.09999999999, abs=1e-6)
    assert avaProfileExt['z'][-1] == pytest.approx(20., abs=1e-6)

    # now If we extend too 1
    cfg = configparser.ConfigParser()
    cfg['PATH'] = {'nCellsResample': '5', 'extTopOption': '0', 'nCellsMinExtend': '2',
                   'nCellsMaxExtend': '30', 'factBottomExt': 1, 'maxIterationExtBot': 10, 'nBottomExtPrecision': 1}
    avaProfile = {'x': np.array([10, 20, 30, 70]), 'y': np.array([10, 20, 30, 70]), 'z': np.array([40, 30, 20, 0])}

    avaProfileExt = DFAPathGeneration.extendDFAPath(cfg['PATH'], avaProfile, dem, particlesIni)
#    print(avaProfileExt)
    atol = 1e-10
    assert np.allclose(avaProfileExt['x'][0], 6.9, atol=atol)
    assert avaProfileExt['x'][-1] == pytest.approx(87.48555023, abs=1e-6)
    assert np.allclose(avaProfileExt['y'][0], 20.0, atol=atol)
    assert avaProfileExt['y'][-1] == pytest.approx(86.19110116, abs=1e-6)
    assert avaProfileExt['z'][0] == pytest.approx(43.09999999999, abs=1e-6)
    assert avaProfileExt['z'][-1] == pytest.approx(0, abs=1e-6)


def test_findFlowFront():
    """"""
    # plane sloping towards increasing column index, flat runout zone from column 5 on
    demRaster = np.tile(np.array([50., 40., 30., 20., 10., 0., 0., 0., 0., 0., 0.]), (10, 1))
    fieldPFT = np.zeros((10, 11))
    # flow tongue along row 5, thicker towards the front
    fieldPFT[5, 1:9] = np.array([0.5, 0.5, 0.5, 0.5, 1., 2., 3., 5.])
    frontRow, frontCol = DFAPathGeneration.findFlowFront(fieldPFT, demRaster, 0.01, 0.05)
    # front band is the flat zone (columns 5-8), ft-weighted centroid at column 78/11
    assert frontRow == 5
    assert frontCol == 7

    # flat deposit: the band covers the whole footprint, ft-weighted centroid at column 83/13
    frontRow, frontCol = DFAPathGeneration.findFlowFront(fieldPFT, np.ones((10, 11)), 0.01, 0.05)
    assert (frontRow, frontCol) == (5, 6)

    # two disjoint lobes at the same elevation: the centroid falls between them and is
    # snapped to the nearest cell of the front band
    twoLobes = np.zeros((10, 11))
    twoLobes[5, 1] = 1.
    twoLobes[5, 9] = 1.
    frontRow, frontCol = DFAPathGeneration.findFlowFront(twoLobes, np.ones((10, 11)), 0.01, 0.05)
    assert (frontRow, frontCol) in [(5, 1), (5, 9)]

    # no flow above threshold
    frontRow, frontCol = DFAPathGeneration.findFlowFront(np.zeros((10, 11)), demRaster, 0.01, 0.05)
    assert frontRow is None
    assert frontCol is None


def test_leastCostPath():
    """"""
    # flat dem with a flow channel along the top row and the two outer columns;
    # the penalty on the distance (in meters) from the flow keeps the path inside
    # the channel instead of cutting straight through the no-flow area
    csz = 5.
    demRaster = np.zeros((7, 7))
    fieldPFT = np.zeros((7, 7))
    fieldPFT[0, :] = 1.
    fieldPFT[:, 0] = 1.
    fieldPFT[:, 6] = 1.
    cellPath = DFAPathGeneration.leastCostPath((3, 0), (3, 6), fieldPFT, demRaster, csz, 0.01, 10., 1.)
    assert cellPath[0] == (3, 0)
    assert cellPath[-1] == (3, 6)
    assert all(fieldPFT[row, col] > 0 for row, col in cellPath)

    # a goal behind a nodata barrier is unreachable
    demBarrier = np.zeros((7, 7))
    demBarrier[:, 3] = np.nan
    cellPath = DFAPathGeneration.leastCostPath((3, 0), (3, 6), fieldPFT, demBarrier, csz, 0.01, 10., 1.)
    assert cellPath == []


def test_extendProfileToFront():
    """"""
    # setup required inputs
    cfg = configparser.ConfigParser()
    cfg['PATH'] = {'nCellsResample': '1', 'extTopOption': '0', 'extBottomOption': '1',
                   'nCellsMinExtend': '1', 'nCellsMaxExtend': '20', 'factBottomExt': 0.2,
                   'maxIterationExtBot': 10, 'nBottomExtPrecision': 10,
                   'ftThreshold': 0.01, 'lowFrontFraction': 0.05,
                   'upSlopePenalty': 10., 'flowDistPenalty': 5.}

    dem = {'header': {'xllcenter': 0, 'yllcenter': 0, 'cellsize': 2, 'nrows': 10, 'ncols': 11},
           'rasterData': np.tile(np.array([50., 40., 30., 20., 10., 0., 0., 0., 0., 0., 0.]), (10, 1))}
    # flow tongue along the profile, reaching the flat runout zone (front cell (5, 7))
    fieldPFT = np.zeros((10, 11))
    fieldPFT[5, 1:9] = np.array([0.5, 0.5, 0.5, 0.5, 1., 2., 3., 5.])

    avaProfile = {'x': np.array([2, 4, 5, 6]), 'y': np.array([10, 10, 10, 10]),
                  'z': np.array([40, 30, 25, 20])}
    particlesIni = {'x': np.array([1., 0.9]), 'y': np.array([10., 10.])}
    particlesIni, _ = gT.projectOnRaster(dem, particlesIni, interp='bilinear')

    avaProfileExt = DFAPathGeneration.extendDFAPath(cfg['PATH'], avaProfile, dem, particlesIni,
                                                    fieldPFT=fieldPFT)
    # the extension descends along the tongue and ends on the deposit front
    assert avaProfileExt['x'][-1] == pytest.approx(14., abs=1e-6)
    assert avaProfileExt['y'][-1] == pytest.approx(10., abs=1e-6)
    assert avaProfileExt['z'][-1] == pytest.approx(0., abs=1e-6)
    assert np.all(np.diff(avaProfileExt['s']) > 0)

    # if the path already ends on the front cell, the straight-line extension takes over so
    # that the profile is always extended at the bottom (resamplePath relies on it)
    avaProfileEnd = {'x': np.array([8, 10, 12, 14]), 'y': np.array([10, 10, 10, 10]),
                     'z': np.array([10., 0., 0., 0.])}
    avaProfileExt = DFAPathGeneration.extendDFAPath(cfg['PATH'], avaProfileEnd, dem, particlesIni,
                                                    fieldPFT=fieldPFT)
    assert avaProfileExt['x'][-1] > 14.
    assert np.all(np.isfinite(avaProfileExt['z']))
    assert np.all(np.diff(avaProfileExt['s']) > 0)

    # without a peak flow thickness field, option 1 falls back to the straight-line extension
    avaProfileNoField = {'x': np.array([2, 4, 5, 6]), 'y': np.array([10, 10, 10, 10]),
                         'z': np.array([40, 30, 25, 20])}
    avaProfileFallback = DFAPathGeneration.extendDFAPath(cfg['PATH'], avaProfileNoField, dem,
                                                         particlesIni)
    cfg['PATH']['extBottomOption'] = '0'
    avaProfileOpt0 = {'x': np.array([2, 4, 5, 6]), 'y': np.array([10, 10, 10, 10]),
                      'z': np.array([40, 30, 25, 20])}
    avaProfileOpt0 = DFAPathGeneration.extendDFAPath(cfg['PATH'], avaProfileOpt0, dem, particlesIni,
                                                     fieldPFT=fieldPFT)
    assert np.allclose(avaProfileFallback['x'], avaProfileOpt0['x'])
    assert np.allclose(avaProfileFallback['y'], avaProfileOpt0['y'])


def test_readPeakFT(tmp_path):
    """"""
    peakDir = tmp_path / 'Outputs' / 'com1DFA' / 'peakFiles'
    peakDir.mkdir(parents=True)
    content = ('ncols 3\nnrows 2\nxllcenter 0.\nyllcenter 0.\ncellsize 5.\nNODATA_value -9999\n'
               '1. 2. 3.\n4. 5. 6.\n')
    (peakDir / 'relA_0123456789_C_M_null_dfa_pft.asc').write_text(content)
    # the peak files are parsed once and the dataframe is passed to readPeakFT
    peakFilesDF = fU.makeSimDF(peakDir, avaDir=tmp_path)
    # the simulation is found by its hash (the index of the configuration dataframe)
    fieldPFT = DFAPathGeneration.readPeakFT(peakFilesDF, '0123456789')
    assert fieldPFT.shape == (2, 3)
    # and by its full simulation name
    fieldPFT = DFAPathGeneration.readPeakFT(peakFilesDF, 'relA_0123456789_C_M_null_dfa')
    assert fieldPFT.shape == (2, 3)
    # no pft available for the simulation: returns None
    assert DFAPathGeneration.readPeakFT(peakFilesDF, 'someOtherSim') is None


def test_resamplePath():
    """"""
    # setup required inputs
    cfg = configparser.ConfigParser()
    cfg['PATH'] = {'nCellsResample': '1', 'uInterval': '1000'}
    avaProfile = {'x': np.array([5, 15, 20, 25, 30, 35]), 'y': np.array([5, 15, 20, 25, 30, 35]),
                  'z': np.array([40, 30, 20, 10, 0, 0]),
                  's': np.array([0, math.sqrt(200), math.sqrt(450), math.sqrt(800), math.sqrt(1250), math.sqrt(1800)]),
                  'indStartMassAverage': 1, 'indEndMassAverage': 4}
    dem = {'header': {'xllcenter': 0, 'yllcenter': 0, 'cellsize': 5, 'nrows': 10, 'ncols': 11},
           'rasterData': np.array([[50, 40, 30, 20, 10, 0, 0, 0, 0, 0, 0],
                                   [50, 40, 30, 20, 10, 0, 0, 0, 0, 0, 0],
                                   [50, 40, 30, 20, 10, 0, 0, 0, 0, 0, 0],
                                   [50, 40, 30, 20, 10, 0, 0, 0, 0, 0, 0],
                                   [50, 40, 30, 20, 10, 0, 0, 0, 0, 0, 0],
                                   [50, 40, 30, 20, 10, 0, 0, 0, 0, 0, 0],
                                   [50, 40, 30, 20, 10, 0, 0, 0, 0, 0, 0],
                                   [50, 40, 30, 20, 10, 0, 0, 0, 0, 0, 0],
                                   [50, 40, 30, 20, 10, 0, 0, 0, 0, 0, 0],
                                   [50, 40, 30, 20, 10, 0, 0, 0, 0, 0, 0]])}

    # using the longest runout method
    avaProfile = DFAPathGeneration.resamplePath(cfg['PATH'], dem, avaProfile)
#    print(avaProfile)
    assert avaProfile['indStartMassAverage'] == 3
    assert avaProfile['indEndMassAverage'] == 7


def test_getParabolicFit():
    """"""
    # setup required inputs
    cfg = configparser.ConfigParser()
    cfg['PATH'] = {'fitOption': '0', 'nCellsSlope': '2'}
    avaProfile = {'x': np.array([0, 10, 20, 30, 40, 50, 60, 70, 80]),
                  'y': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
                  'z': np.array([50, 40, 30, 20, 10, 0, 0, 0, 0]),
                  's': np.array([0, 10, 20, 30, 40, 50, 60, 70, 80]),
                  'indStartMassAverage': 1, 'indEndMassAverage': 8}
    dem = {'header': {'cellsize': 5}}

    # using the distance minimization method
    parabolicFit = DFAPathGeneration.getParabolicFit(cfg['PATH'], avaProfile, dem)
    zPara = parabolicFit['a']*avaProfile['s']*avaProfile['s']+parabolicFit['b']*avaProfile['s']+parabolicFit['c']
#    print(parabolicFit)
#    print(zPara)
    slope = 2*parabolicFit['a']*avaProfile['s']+parabolicFit['b']
#    print(slope)
    assert zPara[0] == 50
    assert zPara[-1] == 0
    assert slope[-1] != 0

    # using the bottom matching dlope method
    cfg['PATH'] = {'fitOption': '1', 'nCellsSlope': '2', 'slopeSplitPoint': '30', 'dsMin': '5'}
    parabolicFit = DFAPathGeneration.getParabolicFit(cfg['PATH'], avaProfile, dem)
    zPara = parabolicFit['a']*avaProfile['s']*avaProfile['s']+parabolicFit['b']*avaProfile['s']+parabolicFit['c']
#    print(parabolicFit)
#    print(zPara)
    slope = 2*parabolicFit['a']*avaProfile['s']+parabolicFit['b']
    angle = np.rad2deg(np.arctan(slope))
    assert zPara[0] == 50
    assert zPara[-1] == 0
    assert slope[-1] == 0

    splitPoint = DFAPathGeneration.getSplitPoint(cfg['PATH'], avaProfile, parabolicFit)
#    print(splitPoint)
#    print(angle)
    assert splitPoint['s'] == 50


def test_getSplitPoint_noPointFound():
    """Test getSplitPoint when no point meets slope criteria - should return top point"""
    cfg = configparser.ConfigParser()
    cfg['PATH'] = {'slopeSplitPoint': '5', 'dsMin': '5'}  # Very low slope requirement

    # Create profile with steep slope everywhere
    avaProfile = {
        'x': np.array([0, 10, 20, 30]),
        'y': np.array([0, 10, 20, 30]),
        'z': np.array([50, 30, 10, 0]),  # Steep slope throughout
        's': np.array([0, 14.14, 28.28, 42.43]),
        'indStartMassAverage': 0,
        'indEndMassAverage': 3
    }
    # Parabolic fit with steep slope at bottom
    parabolicFit = {'a': 0.01, 'b': -2, 'c': 50}

    splitPoint = DFAPathGeneration.getSplitPoint(cfg['PATH'], avaProfile, parabolicFit)

    # Should return top point when no split point found
    assert splitPoint.get('isTopSplitPoint', False) is True
    assert splitPoint['x'] == avaProfile['x'][0]
    assert splitPoint['y'] == avaProfile['y'][0]
    assert splitPoint['z'] == avaProfile['z'][0]


def test_getMassAvgPathFromFields():
    """Test computing mass-averaged path from field data"""
    # Create simple 5x5 field with flow in the middle
    fieldsList = [{
        'FT': np.array([[0, 0, 0, 0, 0],
                        [0, 1, 2, 1, 0],
                        [0, 2, 3, 2, 0],
                        [0, 1, 2, 1, 0],
                        [0, 0, 0, 0, 0]]),  # Flow thickness
        'FM': np.array([[0, 0, 0, 0, 0],
                        [0, 5, 10, 5, 0],
                        [0, 10, 15, 10, 0],
                        [0, 5, 10, 5, 0],
                        [0, 0, 0, 0, 0]]),  # Flow mass
        'FV': np.array([[0, 0, 0, 0, 0],
                        [0, 2, 3, 2, 0],
                        [0, 3, 4, 3, 0],
                        [0, 2, 3, 2, 0],
                        [0, 0, 0, 0, 0]])   # Flow velocity
    }]

    fieldHeader = {
        'ncols': 5,
        'nrows': 5,
        'xllcenter': 100,
        'yllcenter': 200,
        'cellsize': 5
    }

    dem = {
        'rasterData': np.array([[50, 50, 50, 50, 50],
                                [40, 40, 40, 40, 40],
                                [30, 30, 30, 30, 30],
                                [20, 20, 20, 20, 20],
                                [10, 10, 10, 10, 10]])
    }

    result = DFAPathGeneration.getMassAvgPathFromFields(fieldsList, fieldHeader, dem)

    # Verify structure
    assert 'x' in result
    assert 'y' in result
    assert 'z' in result
    assert 's' in result
    assert 'xstd' in result
    assert 'ystd' in result
    assert 'zstd' in result

    # Verify velocity info is included
    assert 'u2' in result
    assert 'ekin' in result
    assert 'u2std' in result
    assert 'ekinstd' in result
    assert 'totEKin' in result

    # Should have one time step
    assert len(result['x']) == 1
    assert len(result['y']) == 1
    assert len(result['z']) == 1

    # Coordinates should be relative to origin (xllcenter and yllcenter subtracted)
    # Mass-weighted average should be close to center
    assert result['x'][0] > 0  # Relative to xllcenter
    assert result['y'][0] > 0  # Relative to yllcenter


def test_getMassAvgPathFromFields_noVelocity():
    """Test getMassAvgPathFromFields without velocity data"""
    # Create simple field without velocity info
    fieldsList = [{
        'FT': np.array([[0, 1, 0],
                        [0, 2, 0],
                        [0, 0, 0]]),
        'FM': np.array([[0, 5, 0],
                        [0, 10, 0],
                        [0, 0, 0]])
        # No FV field
    }]

    fieldHeader = {
        'ncols': 3,
        'nrows': 3,
        'xllcenter': 0,
        'yllcenter': 0,
        'cellsize': 10
    }

    dem = {
        'rasterData': np.array([[30, 30, 30],
                                [20, 20, 20],
                                [10, 10, 10]])
    }

    result = DFAPathGeneration.getMassAvgPathFromFields(fieldsList, fieldHeader, dem)

    # Verify basic structure
    assert 'x' in result
    assert 'y' in result
    assert 'z' in result
    assert 's' in result

    # Velocity info should NOT be present
    assert 'u2' not in result
    assert 'ekin' not in result
    assert 'totEKin' not in result

    # Should have one time step
    assert len(result['x']) == 1
