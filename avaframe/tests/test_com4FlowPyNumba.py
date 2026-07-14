"""
Pytest for the com4FlowPy numba compute engine (flowCoreNumba).

Verifies that the numba engine (engine = numba) reproduces the reference Python
(Cell-based) engine, flowCore.calculation(), bit-for-bit on consistently-sloped
synthetic terrain (no flat-terrain routing ties, which are numerically
knife-edge in both engines).
"""
import numpy as np
import pytest

import avaframe.com4FlowPy.flowCore as flowCore

# the whole module is skipped if the optional 'numba' dependency is absent
pytest.importorskip("numba")
from avaframe.com4FlowPy import flowCoreNumba  # noqa: E402


def _make_dem(ny=30, nx=30):
    """A consistently down-sloping DEM (downhill in +y) with gentle cross-valley
    curvature and strictly varying values, so flow routing has no flat-terrain
    ties (which are numerically knife-edge in both engines)."""
    yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float64)
    return 500.0 - 7.0 * yy + 0.15 * (xx - nx / 2.0) ** 2 + 0.013 * xx


def _default_var():
    return {
        "varUmaxBool": False, "varUmaxArray": None,
        "varAlphaBool": False, "varAlphaArray": None,
        "varExponentBool": False, "varExponentArray": None,
    }


def _args(dem, pra, *, alpha=25.0, forestBool=False, forestParams=None,
          forestArray=None, varParams=None, outputs=None):
    if varParams is None:
        varParams = _default_var()
    if outputs is None:
        outputs = ["zDelta", "flux", "cellCounts", "zDeltaSum", "fpTravelAngleMax",
                   "fpTravelAngleMin", "slTravelAngle", "travelLengthMax",
                   "travelLengthMin", "routFluxSum", "depFluxSum"]
    relOutputParams = {"relIdBool": False, "relIdArray": None,
                       "relVolBool": False, "relVolArray": None}
    return [dem, None, pra, alpha, 8, 3e-4, 270, -9999, 10.0, False, forestBool,
            varParams, False, False, forestArray, forestParams, outputs, relOutputParams]


# comparable result indices (4=backcalc and 12=relId are None on this path;
# 13=forestInteraction is compared separately when active)
_IDX = {0: "zDelta", 1: "flux", 2: "cellCounts", 3: "zDeltaSum",
        5: "fpTravelAngleMax", 6: "slTravelAngle", 7: "travelLengthMax",
        8: "travelLengthMin", 9: "fpTravelAngleMin", 10: "routFluxSum",
        11: "depFluxSum"}


def _compare(py, nb):
    assert len(nb) == len(py) == 14
    for i, name in _IDX.items():
        a = np.asarray(py[i], dtype=np.float64)
        b = np.asarray(nb[i], dtype=np.float64)
        assert np.array_equal(a, b), \
            f"{name}: {int((a != b).sum())} cells differ, max|Δ|={np.abs(a - b).max()}"


def _pra(dem):
    pra = np.zeros_like(dem, dtype=np.int32)
    pra[1, dem.shape[1] // 2] = 1  # single release near the top of the slope
    return pra


def test_numba_matches_python_noforest():
    dem = _make_dem()
    args = _args(dem, _pra(dem))
    _compare(flowCore.calculation(args), flowCoreNumba.calculationNumba(args))


def test_numba_matches_python_forestFriction():
    dem = _make_dem()
    forest = np.zeros_like(dem)
    forest[10:20, :] = 0.6
    fp = {"forestModule": "forestFriction", "maxAddedFriction": 20.0,
          "minAddedFriction": 2.0, "velThForFriction": 30.0, "maxDetrainment": 0.0,
          "minDetrainment": 0.0, "velThForDetrain": 0.0, "fFrLayerType": "absolute",
          "skipForestDist": 0.0, "forestInteraction": True}
    args = _args(dem, _pra(dem), forestBool=True, forestParams=fp, forestArray=forest)
    py = flowCore.calculation(args)
    nb = flowCoreNumba.calculationNumba(args)
    _compare(py, nb)
    assert np.array_equal(np.asarray(py[13], float), np.asarray(nb[13], float)), \
        "forestInteraction array mismatch"


def test_numba_matches_python_forestDetrainment():
    dem = _make_dem()
    forest = np.zeros_like(dem)
    forest[8:22, :] = 0.5
    fp = {"forestModule": "forestDetrainment", "maxAddedFriction": 52.0,
          "minAddedFriction": 5.0, "velThForFriction": 270.0, "maxDetrainment": 0.003,
          "minDetrainment": 0.00001, "velThForDetrain": 270.0, "fFrLayerType": "absolute",
          "skipForestDist": 0.0, "forestInteraction": True}
    args = _args(dem, _pra(dem), forestBool=True, forestParams=fp, forestArray=forest)
    py = flowCore.calculation(args)
    nb = flowCoreNumba.calculationNumba(args)
    _compare(py, nb)
    assert np.array_equal(np.asarray(py[13], float), np.asarray(nb[13], float))


def test_numba_matches_python_variableAlphaUmax():
    dem = _make_dem()
    var = _default_var()
    var["varAlphaBool"] = True
    var["varAlphaArray"] = np.full_like(dem, 22.0)   # per-cell alpha (deg)
    var["varUmaxBool"] = True
    var["varUmaxArray"] = np.full_like(dem, 150.0)   # per-cell zDeltaLim (m)
    args = _args(dem, _pra(dem), varParams=var)
    _compare(flowCore.calculation(args), flowCoreNumba.calculationNumba(args))


if __name__ == "__main__":
    test_numba_matches_python_noforest()
    test_numba_matches_python_forestFriction()
    test_numba_matches_python_forestDetrainment()
    test_numba_matches_python_variableAlphaUmax()
    print("all numba-engine equivalence tests passed")
