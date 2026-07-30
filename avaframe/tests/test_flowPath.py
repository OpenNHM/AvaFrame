import numpy as np
import pytest
import pickle
import logging
from types import SimpleNamespace

# adjust this import to match your actual module path
from avaframe.com4FlowPy.flowPath import Path  # noqa: F401  (placeholder alias, see note below)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def makeCell(z_delta=0.0, flux=0.0, min_distance=0.0, altitude=0.0,
             rowindex=0, colindex=0, max_gamma=0.0, flowEnergy=0.0,
             fluxDep=0.0, alpha=25.0, exp=8, max_z_delta=100.0):
    """ build a minimal fake cell object with the attributes Path expects """
    return SimpleNamespace(
        z_delta=z_delta, flux=flux, min_distance=min_distance, altitude=altitude,
        rowindex=rowindex, colindex=colindex, max_gamma=max_gamma,
        flowEnergy=flowEnergy, fluxDep=fluxDep, alpha=alpha, exp=exp,
        max_z_delta=max_z_delta,
    )


def makeRasterAttributes(cellsize=10.0, nrows=5, ncols=5, extentTile=((0, 5), (0, 5))):
    return {"cellsize": cellsize, "nrows": nrows, "ncols": ncols, "extentTile": extentTile}


def test_Path_init_withGenList():
    """ test Path init when genList is provided (normal, non-RAM-saving branch) """
    dem = np.zeros((5, 5))
    countArray = np.zeros((5, 5))
    countArray[1, 1] = 3
    countArray[2, 2] = 5

    cell1 = makeCell(alpha=27.0, exp=8, max_z_delta=150.0)
    cell2 = makeCell(alpha=27.0, exp=8, max_z_delta=150.0)
    genList = [[cell1, cell2], [cell1]]

    rasterAttributes = makeRasterAttributes()

    p = Path(dem, 2, 3, genList, rasterAttributes, countArray, relId=7)

    assert p.cellsize == 10.0
    assert p.nrows == 5
    assert p.startcellRow == 2
    assert p.startcellCol == 3
    assert p.relId == 7
    assert p.alpha == 27.0
    assert p.exp == 8
    assert p.maxZDelta == 150.0
    assert p.numberGen == 2

    # pathRaster: countArray values kept where >0, else nan
    assert p.pathRaster[1, 1] == 3
    assert p.pathRaster[2, 2] == 5
    assert np.isnan(p.pathRaster[0, 0])

    # output arrays initialized to zero, correct shape/dtype
    for arrName in ["zDeltaArray", "flowEnergyArray", "fluxArray", "routFluxSumArray", "depFluxSumArray"]:
        arr = getattr(p, arrName)
        assert arr.shape == dem.shape
        assert arr.dtype == np.float32
        np.testing.assert_array_equal(arr, np.zeros_like(dem, dtype=np.float32))


def test_Path_init_withoutGenList():
    """ test Path init when genList is None (RAM-saving branch, uses listsRelId/exampleCell) """
    dem = np.zeros((5, 5))
    countArray = np.zeros((5, 5))
    rasterAttributes = makeRasterAttributes()

    listsRelId = {
        "row": [1, 2, 3],
        "col": [1, 2, 3],
        "flux": [0.1, 0.2, 0.3],
        "zdelta": [10.0, 20.0, 30.0],
        "travelLengthMax": [5.0, 10.0, 15.0],
    }
    exampleCell = makeCell(alpha=30.0, exp=9, max_z_delta=200.0)

    p = Path(dem, 0, 0, None, rasterAttributes, countArray, relId=1,
             listsRelId=listsRelId, exampleCell=exampleCell)

    assert p.rowList == listsRelId["row"]
    assert p.colList == listsRelId["col"]
    assert p.fluxList == listsRelId["flux"]
    assert p.zdeltaList == listsRelId["zdelta"]
    assert p.travelLengthList == listsRelId["travelLengthMax"]
    assert p.alpha == 30.0
    assert p.exp == 9
    assert p.maxZDelta == 200.0
    assert p.numberGen == 3


# ---------------------------------------------------------------------------
# getListFromCellList
# ---------------------------------------------------------------------------

def _minimalPath():
    """ build a minimal, cheaply-constructed Path instance for method-level tests """
    dem = np.zeros((4, 4))
    countArray = np.zeros((4, 4))
    cell = makeCell()
    genList = [[cell]]
    rasterAttributes = makeRasterAttributes(nrows=4, ncols=4, extentTile=((2, 6), (3, 7)))
    return Path(dem, 0, 0, genList, rasterAttributes, countArray, relId=1)


def test_getListFromCellList_zDelta():
    """ test that variable 'zDelta' maps to cell.z_delta """
    dummyPath = _minimalPath()
    cells = [makeCell(z_delta=1.0, flux=2.0, min_distance=3.0, altitude=4.0,
                      rowindex=5, colindex=6, max_gamma=7.0, flowEnergy=8.0),
             makeCell(z_delta=10.0, flux=20.0, min_distance=30.0, altitude=40.0,
                      rowindex=50, colindex=60, max_gamma=70.0, flowEnergy=80.0)]

    result = dummyPath.getListFromCellList(cells, "zDelta")
    assert result == [1.0, 10.0]

    result = dummyPath.getListFromCellList(cells, "flux")
    assert result == [2.0, 20.0]

    result = dummyPath.getListFromCellList(cells, "travelLength")
    assert result == [3.0, 30.0]

    result = dummyPath.getListFromCellList(cells, "row")
    assert result == [5, 50]

    result = dummyPath.getListFromCellList(cells, "col")
    assert result == [6, 60]


# ---------------------------------------------------------------------------
# getGenerationList
# ---------------------------------------------------------------------------

def test_getGenerationList_allGenerations():
    """ test that getGenerationList (no generation arg) returns nested per-generation lists """
    dem = np.zeros((4, 4))
    countArray = np.zeros((4, 4))
    gen0 = [makeCell(flux=1.0), makeCell(flux=2.0)]
    gen1 = [makeCell(flux=3.0)]
    genList = [gen0, gen1]
    rasterAttributes = makeRasterAttributes(nrows=4, ncols=4)

    p = Path(dem, 0, 0, genList, rasterAttributes, countArray, relId=1)

    result = p.getGenerationList("flux")

    assert result == [[1.0, 2.0], [3.0]]


def test_getGenerationList_singleGeneration():
    """ test that getGenerationList with a specific generation index returns a flat list """
    dem = np.zeros((4, 4))
    countArray = np.zeros((4, 4))
    gen0 = [makeCell(flux=1.0), makeCell(flux=2.0)]
    gen1 = [makeCell(flux=3.0)]
    genList = [gen0, gen1]
    rasterAttributes = makeRasterAttributes(nrows=4, ncols=4)

    p = Path(dem, 0, 0, genList, rasterAttributes, countArray, relId=1)

    result = p.getGenerationList("flux", generation=1)

    assert result == [3.0]


# ---------------------------------------------------------------------------
# getPathArrays
# ---------------------------------------------------------------------------

def test_getPathArrays():
    """ test that getPathArrays fills max/sum arrays correctly from the genList cells """
    dem = np.zeros((4, 4))
    countArray = np.zeros((4, 4))
    rasterAttributes = makeRasterAttributes(nrows=4, ncols=4)

    # same (row, col) hit twice, across two generations -> max/sum behavior should differ
    cellA = makeCell(rowindex=1, colindex=1, z_delta=5.0, flowEnergy=50.0, flux=0.5, fluxDep=0.1)
    cellB = makeCell(rowindex=1, colindex=1, z_delta=8.0, flowEnergy=20.0, flux=0.3, fluxDep=0.2)
    cellC = makeCell(rowindex=2, colindex=3, z_delta=1.0, flowEnergy=1.0, flux=1.0, fluxDep=1.0)
    genList = [[cellA], [cellB, cellC]]

    p = Path(dem, 0, 0, genList, rasterAttributes, countArray, relId=1)
    p.getPathArrays()

    zDeltaArrayRef = np.zeros((4, 4))
    zDeltaArrayRef[1, 1] = 8.0
    zDeltaArrayRef[2, 3] = 1.0

    # max of z_delta at (1,1) across both hits
    assert p.zDeltaArray[1, 1] == 8.0
    # max of flowEnergy at (1,1)
    assert p.flowEnergyArray[1, 1] == 50.0
    # max of flux at (1,1)
    assert p.fluxArray[1, 1] == pytest.approx(0.5)
    # sums accumulate across hits
    assert p.routFluxSumArray[1, 1] == pytest.approx(0.5 + 0.3)
    assert p.depFluxSumArray[1, 1] == pytest.approx(0.1 + 0.2)

    # cell hit only once
    assert p.zDeltaArray[2, 3] == 1.0
    assert p.routFluxSumArray[2, 3] == pytest.approx(1.0)

    # untouched cell stays zero
    assert p.zDeltaArray[0, 0] == 0.0
    assert p.routFluxSumArray[0, 0] == 0.0

    assert np.all(p.zDeltaArray == zDeltaArrayRef)


# ---------------------------------------------------------------------------
# calcThalwegCenterof
# ---------------------------------------------------------------------------

def test_calcThalwegCenterof_weightedAverage():
    """ test weighted average computation when weights are non-zero """
    p = _minimalPath()
    p.numberGen = 2

    variable = [[1.0, 3.0], [10.0]]
    variableCo = [[1.0, 1.0], [5.0]]  # generation 0: equal weights -> avg 2.0; generation 1: single value

    variableSum, coVar = p.calcThalwegCenterof(variable, variableCo)

    np.testing.assert_allclose(variableSum, [4.0, 10.0])
    np.testing.assert_allclose(coVar, [2.0, 10.0])


def test_calcThalwegCenterof_zeroWeightFallback():
    """ test that a generation with all-zero weights falls back to a plain average """
    p = _minimalPath()
    p.numberGen = 1

    variable = [[2.0, 4.0, 6.0]]
    variableCo = [[0.0, 0.0, 0.0]]  # sum of weights is 0 -> fallback branch

    variableSum, coVar = p.calcThalwegCenterof(variable, variableCo)

    assert variableSum[0] == pytest.approx(12.0)
    assert coVar[0] == pytest.approx(4.0)  # plain average of [2,4,6]


# ---------------------------------------------------------------------------
# getCenterofs
# ---------------------------------------------------------------------------

def test_getCenterofs_setsWeightedAttributes_col():
    """ test that getCenterofs computes and sets e.g. colCoF from the generation data """
    dem = np.zeros((4, 4))
    countArray = np.zeros((4, 4))
    rasterAttributes = makeRasterAttributes(nrows=4, ncols=4)
    gen0 = [makeCell(rowindex=1, colindex=4, flux=1.0), makeCell(rowindex=3, colindex=12, flux=3.0)]
    gen1 = [makeCell(rowindex=5, colindex=5, flux=5.0)]
    genList = [gen0, gen1]
    p = Path(dem, 0, 0, genList, rasterAttributes, countArray, relId=1)
    variables = ["col"]
    p.getCenterofs(variables, ["CoF"])
    # col weighted by flux -> weighted avg of [1,3] w/ weights [1,3] = (1*1+3*3)/4 = 2.5; gen1 single value 5
    assert hasattr(p, "colCoF")
    np.testing.assert_allclose(p.colCoF, [10.0, 5.0])

    variables = ["flux"]
    p.getCenterofs(variables, ["CoF"])

    # flux weighted by itself -> weighted avg of [1,3] w/ weights [1,3] = (1*1+3*3)/4 = 2.5; gen1 single value 5
    assert hasattr(p, "fluxCoF")
    np.testing.assert_allclose(p.fluxCoF, [2.5, 5.0])

    variables = ["row"]
    p.getCenterofs(variables, ["CoF"])
    # row weighted by flux -> weighted avg of [1,3] w/ weights [1,3] = (1*1+3*3)/4 = 2.5; gen1 single value 5
    assert hasattr(p, "rowCoF")
    np.testing.assert_allclose(p.rowCoF, [2.5, 5.0])


def test_getCenterofs_skipsExcludedVariables():
    """ test that variables in the exclusion list are skipped (no attribute is set) """
    dem = np.zeros((4, 4))
    countArray = np.zeros((4, 4))
    rasterAttributes = makeRasterAttributes(nrows=4, ncols=4)
    genList = [[makeCell(flux=1.0)]]

    p = Path(dem, 0, 0, genList, rasterAttributes, countArray, relId=1)

    variables = ["x", "zDeltaArray", "y"]
    p.getCenterofs(variables, ["CoF"])

    assert not hasattr(p, "xCoF")
    assert not hasattr(p, "zDeltaArrayCoF")
    assert hasattr(p, "colCoF")
    assert hasattr(p, "rowCoF")


def test_getCenterofs_expandsDepFluxSumAndFluxSum():
    """ test that 'depFluxSum'/'fluxSum' trigger appending 'depFlux'/'flux' to the variables list """
    dem = np.zeros((4, 4))
    countArray = np.zeros((4, 4))
    rasterAttributes = makeRasterAttributes(nrows=4, ncols=4)
    genList = [[makeCell(flux=2.0, fluxDep=1.0)]]

    p = Path(dem, 0, 0, genList, rasterAttributes, countArray, relId=1)

    variables = ["fluxSum"]
    p.getCenterofs(variables, ["CoF"])

    # "flux" should have been appended and processed
    assert "flux" in variables
    assert hasattr(p, "fluxCoF")


# ---------------------------------------------------------------------------
# correctIndicesTile
# ---------------------------------------------------------------------------

def test_correctIndicesTile():
    """ test that row/col indices are correctly offset by the tile's extent """
    p = _minimalPath()
    p.rasterAttributes["extentTile"] = ((100, 200), (50, 150))

    row = np.array([0, 1, 2])
    col = np.array([0, 5, 10])

    rowLarge, colLarge = p.correctIndicesTile(row, col)

    np.testing.assert_array_equal(rowLarge, row + 100)
    np.testing.assert_array_equal(colLarge, col + 50)


# ---------------------------------------------------------------------------
# saveDict
# ---------------------------------------------------------------------------

def test_saveDict_withRelId(tmp_path):
    """ test that saveDict writes a pickle file named by relId and with correct content """
    p = _minimalPath()
    p.alpha = 27.456
    p.exp = 8
    p.maxZDelta = 123.456
    p.numberGen = 4
    p.relId = 42

    # attributes needed for variable "x" and "s" with centerOf "CoF"
    p.xCoF = np.array([1.0, 2.0, 3.0])
    p.travelLengthCoF = np.array([0.0, 10.0, 20.0])

    p.saveDict(tmp_path, ["CoF"], ["x", "s"])

    outFile = tmp_path / "thalwegData_CoF_42.pickle"
    assert outFile.is_file()

    with open(outFile, "rb") as f:
        data = pickle.load(f)

    assert data["alpha"] == pytest.approx(27.5, abs=0.05)  # rounded to 1 decimal
    assert data["exponent"] == 8
    assert data["zDeltaMax"] == pytest.approx(123.5, abs=0.05)
    assert data["numberGen"] == 4
    np.testing.assert_allclose(data["x"], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(data["s"], [0.0, 10.0, 20.0])


def test_saveDict_withoutRelId(tmp_path):
    """ test that saveDict falls back to startcellRow/startcellCol naming when relId is None """
    p = _minimalPath()
    p.relId = None
    p.startcellRow = 7
    p.startcellCol = 9
    p.xCoE = np.array([1.0])

    p.saveDict(tmp_path, ["CoE"], ["x"])

    outFile = tmp_path / "thalwegData_CoE_7_9.pickle"
    assert outFile.is_file()


# ---------------------------------------------------------------------------
# calcAndSaveThalwegData
# ---------------------------------------------------------------------------

def test_calcAndSaveThalwegData_invalidCoRaises(tmp_path):
    """ test that an invalid thalweg 'centerOf' parameter raises ValueError """
    p = _minimalPath()

    thalwegParameters = {
        "thalwegDir": tmp_path,
        "thalwegSaveRam": False,
        "thalwegCenterOf": "['notValid']",
        "thalwegVariables": "['x']",
    }

    with pytest.raises(ValueError):
        p.calcAndSaveThalwegData(thalwegParameters)


def test_calcAndSaveThalwegData_saveRamBranch(tmp_path):
    """ test the thalwegSaveRam=True branch: computes CoF attributes from the RAM-saving lists
    and writes output via saveDict, using the real gT.indicesToCoords for coordinate conversion.
    """
    dem = np.zeros((10, 10))
    countArray = np.zeros((10, 10))
    header = {"cellsize": 10.0, "xllcenter": 0.0, "yllcenter": 0.0}
    rasterAttributes = {
        "cellsize": header["cellsize"],
        "xllcenter": header["xllcenter"],
        "yllcenter": header["yllcenter"],
        "nrows": 10,
        "ncols": 10,
        "extentTile": ((0, 10), (0, 10)),
    }

    listsRelId = {
        "row": [1, 2, 3],
        "col": [1, 2, 3],
        "flux": [1.0, 1.0, 1.0],
        "zdelta": [10.0, 20.0, 30.0],
        "travelLengthMax": [0.0, 5.0, 10.0],
    }
    exampleCell = makeCell(alpha=25.0, exp=8, max_z_delta=100.0)

    p = Path(dem, 0, 0, None, rasterAttributes, countArray, relId=3,
             listsRelId=listsRelId, exampleCell=exampleCell)

    thalwegParameters = {
        "thalwegDir": tmp_path,
        "thalwegSaveRam": True,
    }

    p.calcAndSaveThalwegData(thalwegParameters)

    assert hasattr(p, "colCoF")
    assert hasattr(p, "rowCoF")
    assert hasattr(p, "zdeltaCoF")
    assert hasattr(p, "travelLengthCoF")
    assert hasattr(p, "xCoF")
    assert hasattr(p, "yCoF")

    # since row/col/flux are all equal-weighted with equal flux ([1,2,3] weighted equally by [1,1,1]),
    # the weighted center-of-flux is just the mean: row = col = 2
    assert p.colCoF == pytest.approx(2.0)
    assert p.rowCoF == pytest.approx(2.0)

    # tile offset is (0,0) here, so rowLarge/colLarge == rowCoF/colCoF
    # x = xllcorner + col*cellsize = -5.0 + 2*10.0 = 15.0 (xllcenter=0, cellsize=10 -> xllcorner=-5)
    assert p.xCoF == pytest.approx(15.0)
    assert p.yCoF == pytest.approx(15.0)

    # genList should have been emptied to save RAM
    assert p.genList == []

    outFile = tmp_path / "thalwegData_CoF_3.pickle"
    assert outFile.is_file()


def test_getGenerationList_threeGenerations_allGenerations():
    """ test getGenerationList (no generation arg) with a 3-generation genList,
    including a generation that reuses cell1 from generation 0
    """
    dem = np.zeros((5, 5))
    countArray = np.zeros((5, 5))

    cell1 = makeCell(z_delta=1.0, flux=1.0)
    cell2 = makeCell(z_delta=2.0, flux=2.0)
    cell3 = makeCell(z_delta=3.0, flux=3.0)
    cell4 = makeCell(z_delta=4.0, flux=4.0)
    cell5 = makeCell(z_delta=5.0, flux=5.0)

    genList = [[cell1, cell2], [cell3, cell4, cell5], [cell1]]
    rasterAttributes = makeRasterAttributes(nrows=5, ncols=5)

    p = Path(dem, 0, 0, genList, rasterAttributes, countArray, relId=1)

    result = p.getGenerationList("zDelta")

    assert result == [[1.0, 2.0], [3.0, 4.0, 5.0], [1.0]]


def test_getGenerationList_threeGenerations_singleGeneration():
    """ test getGenerationList with an explicit generation index into a 3-generation genList """
    dem = np.zeros((5, 5))
    countArray = np.zeros((5, 5))

    cell1 = makeCell(flux=1.0)
    cell2 = makeCell(flux=2.0)
    cell3 = makeCell(flux=3.0)
    cell4 = makeCell(flux=4.0)
    cell5 = makeCell(flux=5.0)

    genList = [[cell1, cell2], [cell3, cell4, cell5], [cell1]]
    rasterAttributes = makeRasterAttributes(nrows=5, ncols=5)

    p = Path(dem, 0, 0, genList, rasterAttributes, countArray, relId=1)

    # generation 0: 2 cells
    assert p.getGenerationList("flux", generation=0) == [1.0, 2.0]
    # generation 1: 3 cells
    assert p.getGenerationList("flux", generation=1) == [3.0, 4.0, 5.0]
    # generation 2: reuses cell1 -> 1 cell
    assert p.getGenerationList("flux", generation=2) == [1.0]


def test_getPathArrays_threeGenerations_repeatedCell():
    """ test getPathArrays with a 3-generation genList where cell1 (row=0, col=0) appears
    in both generation 0 and generation 2, verifying max/sum behavior across the repeat
    """
    dem = np.zeros((5, 5))
    countArray = np.zeros((5, 5))

    cell1 = makeCell(rowindex=0, colindex=0, z_delta=5.0, flowEnergy=50.0, flux=1.0, fluxDep=0.5)
    cell2 = makeCell(rowindex=1, colindex=1, z_delta=2.0, flowEnergy=20.0, flux=2.0, fluxDep=0.2)
    cell3 = makeCell(rowindex=2, colindex=2, z_delta=3.0, flowEnergy=30.0, flux=3.0, fluxDep=0.3)
    cell4 = makeCell(rowindex=3, colindex=3, z_delta=4.0, flowEnergy=40.0, flux=4.0, fluxDep=0.4)
    cell5 = makeCell(rowindex=4, colindex=4, z_delta=1.0, flowEnergy=10.0, flux=0.5, fluxDep=0.1)

    genList = [[cell1, cell2], [cell3, cell4, cell5], [cell1]]
    rasterAttributes = makeRasterAttributes(nrows=5, ncols=5)

    p = Path(dem, 0, 0, genList, rasterAttributes, countArray, relId=1)
    p.getPathArrays()

    # cell1 is hit twice (gen 0 and gen 2) at (0,0), with identical values both times
    assert p.zDeltaArray[0, 0] == 5.0
    assert p.flowEnergyArray[0, 0] == 50.0
    assert p.fluxArray[0, 0] == pytest.approx(1.0)
    # sums accumulate over both hits
    assert p.routFluxSumArray[0, 0] == pytest.approx(1.0 + 1.0)
    assert p.depFluxSumArray[0, 0] == pytest.approx(0.5 + 0.5)

    # cells hit only once, spread across generations
    assert p.zDeltaArray[1, 1] == 2.0
    assert p.zDeltaArray[2, 2] == 3.0
    assert p.zDeltaArray[3, 3] == 4.0
    assert p.zDeltaArray[4, 4] == 1.0
    assert p.routFluxSumArray[4, 4] == pytest.approx(0.5)


def test_calcThalwegCenterof_threeGenerations():
    """ test calcThalwegCenterof directly using values shaped like a 3-generation genList
    (generation sizes 2, 3, 1) with flux as the weighting variable
    """
    p = _minimalPath()
    p.numberGen = 3

    # zDelta values per generation, matching [[cell1,cell2],[cell3,cell4,cell5],[cell1]]
    zDeltaGen = [[1.0, 2.0], [3.0, 4.0, 5.0], [1.0]]
    fluxGen = [[1.0, 2.0], [3.0, 4.0, 5.0], [1.0]]

    variableSum, coVar = p.calcThalwegCenterof(zDeltaGen, fluxGen)

    # gen 0: sum=3.0, weighted avg = (1*1+2*2)/(1+2) = 5/3
    # gen 1: sum=12.0, weighted avg = (3*3+4*4+5*5)/(3+4+5) = 50/12
    # gen 2: sum=1.0, weighted avg = 1.0 (single value)
    np.testing.assert_allclose(variableSum, [3.0, 12.0, 1.0])
    np.testing.assert_allclose(coVar, [5.0 / 3.0, 50.0 / 12.0, 1.0])
