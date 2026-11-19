''' Tests for module ana3AIMEC plots '''
import pandas as pd
import numpy as np
import pathlib
import configparser
import pytest

# Local imports
import avaframe.out3Plot.outAIMEC as oA
import avaframe.ana3AIMEC.aimecTools as aT

def test_getIndicesVel():
    """ test if indices of a  vector exceeding a threshold are found """

    # setup required input
    a = np.zeros(10)
    a[2:8] = 10.
    a[2] = 2.
    a[5] = 1.2
    velocityThreshold = 1.

    # call function
    indStart, indStop = oA.getIndicesVel(a, velocityThreshold)

    assert indStart == 2
    assert indStop == 7

    a = np.zeros(10) * np.nan
    a[2:8] = 10.
    a[2] = 2.
    a[5] = 1.2
    velocityThreshold = 1.

    # call function
    indStart, indStop = oA.getIndicesVel(a, velocityThreshold)

    assert indStart == 2
    assert indStop == 7

    a = np.zeros(10)
    a[2:] = 10.
    a[2] = 0.9
    velocityThreshold = 1.

    # call function
    indStart, indStop = oA.getIndicesVel(a, velocityThreshold)

    assert indStart == 3
    assert indStop == 9

    a = np.zeros(10)
    a[2:] = 0.4
    a[2] = 0.9
    velocityThreshold = 1.

    with pytest.raises(AssertionError) as e:
        assert oA.getIndicesVel(a, velocityThreshold)
    assert 'No peak flow velocity max along thalweg found exceeding' in str(e.value)


def test_plotSfcChangeAlongThalweg(tmp_path):
    """plot of the surface change along the thalweg is saved"""

    s = np.arange(0.0, 100.0, 5.0)
    l = np.linspace(-20.0, 20.0, 9)
    sGrid, lGrid = np.meshgrid(s, l, indexing="ij")
    # erosion in the track, deposition in the runout, first cross section out of the raster
    sfc = np.where(sGrid < 60.0, -0.5, 1.5) * (np.abs(lGrid) <= 10.0)
    sfc[0, :] = np.nan
    runout = np.where(np.abs(lGrid) <= 10.0, 5.0, 0.0)
    rasterTransfo = {
        "s": s,
        "l": l,
        "cellSizeSL": 5.0,
        "indStartOfRunout": 12,
        "startOfRunoutAreaAngle": 10.0,
        "labelRunout": "start of runout area",
        "rasterArea": np.full(sGrid.shape, 25.0),
    }
    cfg = configparser.ConfigParser()
    cfg["AIMECSETUP"] = {"runoutResType": "ppr", "thresholdValue": "1", "minValueField": "1.e-6"}
    resAnalysisDF = pd.DataFrame({"simName": ["simTest"]}, index=["hashA"])
    resAnalysisDF = aT.analyzeSfcChange("hashA", rasterTransfo, sfc, "sfcChangeCum", resAnalysisDF)
    pathDict = {"projectName": "avaTest", "pathResult": tmp_path}
    contourDict = oA.fetchContourLines(rasterTransfo, {"compData": runout, "simName": "simTest"}, 1.0, {})

    plotPath = oA.plotSfcChangeAlongThalweg(
        cfg["AIMECSETUP"],
        pathDict,
        rasterTransfo,
        sfc,
        runout,
        contourDict,
        resAnalysisDF.loc["hashA"],
        "sfcChangeCum",
    )

    assert plotPath == tmp_path / "avaTest_simTest_sfcChangeCumAlongThalweg.png"
    assert plotPath.is_file()
