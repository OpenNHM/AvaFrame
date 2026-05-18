""" Tests for module outParticlesAnalysis  """

import numpy as np
import numpy.ma as ma
import pandas as pd
import pathlib
import configparser
import matplotlib.pyplot as plt
import pytest

# Local imports
import avaframe.out3Plot.outParticlesAnalysis as oA
import avaframe.in2Trans.rasterUtils as IOf
from avaframe.in3Utils import cfgUtils


def test_velocityEnvelope():
    """test computing envelope of particlesTimeArrays"""

    # setup required input
    particlesTimeArrays = {"t": [1.0, 2.0, 3.0, 4.0, 5.0]}
    velMag = np.zeros((5, 10))
    velMag[1:, 1:] = 1.0
    velMag[1:, 1] = 5.0
    velMag[2, 3] = 7
    trajectoryLengthXYZ = np.zeros((5, 10))
    trajectoryLengthXYZ[1, :] = 2.0
    trajectoryLengthXYZ[2, :] = 3.0
    trajectoryLengthXYZ[3, :] = 5.0
    trajectoryLengthXYZ[4, :] = 7.0

    uAcc = np.zeros((5, 10))
    uAcc[1:, 1:] = 4.0
    particlesTimeArrays["velocityMag"] = velMag
    particlesTimeArrays["trajectoryLengthXYZ"] = trajectoryLengthXYZ
    particlesTimeArrays["uAcc"] = uAcc

    # call function to be tested
    dictVelEnvelope = oA.velocityEnvelope(particlesTimeArrays)

    # TODO check what if a particle has been merged or removed what happens to array
    b = 19.0 / 10.0
    assert np.array_equal(np.asarray(dictVelEnvelope["Time"]), np.asarray([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert np.array_equal(dictVelEnvelope["Velocity"], velMag)
    assert np.array_equal(dictVelEnvelope["Acc"], uAcc)
    assert np.array_equal(dictVelEnvelope["Mean"], np.asarray([0, 1.3, b, 1.3, 1.3]))
    assert np.array_equal(dictVelEnvelope["Max"], np.asarray([0, 5.0, 7.0, 5.0, 5.0]))
    assert np.array_equal(dictVelEnvelope["Min"], np.asarray([0.0, 0.0, 0.0, 0.0, 0.0]))
    assert np.array_equal(dictVelEnvelope["Median"], np.asarray([0.0, 1.0, 1.0, 1.0, 1.0]))
    assert np.array_equal(dictVelEnvelope["SxyzMax"], np.asarray([0.0, 2.0, 3.0, 5.0, 7.0]))
    assert np.array_equal(dictVelEnvelope["SxyzMax"], np.asarray([0.0, 2.0, 3.0, 5.0, 7.0]))


def test_velocityEnvelopeThalweg():
    """test computing envelope  along the thalweg of particlesTimeArrays"""

    # setup required input
    particlesTimeArrays = {"t": [1.0, 2.0, 3.0, 4.0, 5.0]}
    velMag = np.zeros((3, 5))
    velMag[1:, 1:] = 1.0
    velMag[1:, 1] = 5.0
    velMag[2, 3] = 7
    trajectoryLengthXYZ = np.zeros((3, 5))
    trajectoryLengthXYZ[1, :] = np.asarray([0.0, 2.0, 3.0, 4.0, 5.0])
    trajectoryLengthXYZ[2, :] = np.asarray([0.0, 3.0, 4.0, 6.0, 7.0])

    sAimec = np.zeros((3, 5))
    sAimec[1, :] = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
    sAimec[2, :] = np.asarray([0.0, 4.0, 5.0, 7.0, 8.0])

    elevation = np.zeros((3, 5))
    elevation[1, :] = np.asarray([0.0, 10.0, 20.0, 30.0, 40.0])
    elevation[2, :] = np.asarray([0.0, 20.0, 30.0, 40.0, 50.0])

    uAcc = np.zeros((3, 5))
    uAcc[1:, 1:] = 4.0
    particlesTimeArrays["velocityMag"] = velMag
    particlesTimeArrays["trajectoryLengthXYZ"] = trajectoryLengthXYZ
    particlesTimeArrays["uAcc"] = uAcc
    particlesTimeArrays["sBetaPoint"] = 4.0
    particlesTimeArrays["sAimec"] = sAimec
    particlesTimeArrays["z"] = elevation

    # call function to be tested
    dictVelAltThalweg = oA.velocityEnvelopeThalweg(particlesTimeArrays)

    velMagMax = np.asarray([0.0, 5.0, 1.0, 1.0, 5.0, 1.0, 7.0, 1.0])
    velMagMin = np.asarray([0.0, 5.0, 1.0, 1.0, 1.0, 1.0, 7.0, 1.0])
    velMagMean = np.asarray([0.0, 5.0, 1.0, 1.0, 3.0, 1.0, 7.0, 1.0])
    sxyzMax = np.asarray([0.0, 2.0, 3.0, 4.0, 5.0, 4.0, 6.0, 7.0])
    sxyzMin = np.asarray([0.0, 2.0, 3.0, 4.0, 3.0, 4.0, 6.0, 7.0])

    assert np.array_equal(dictVelAltThalweg["maxVelocity"], velMagMax)
    assert np.array_equal(dictVelAltThalweg["minVelocity"], velMagMin)
    assert np.array_equal(dictVelAltThalweg["meanVelocity"], velMagMean)
    assert np.array_equal(dictVelAltThalweg["medianVelocity"], velMagMean)
    assert np.array_equal(dictVelAltThalweg["maxSxyz"], sxyzMax)
    assert np.array_equal(dictVelAltThalweg["minSxyz"], sxyzMin)


def test_checkSavingTimeStepParticles():
    """check if time step exceeds limitValue"""

    # setup input
    timeStepInfo = [0.0, 1.999, 2.0, 4.0, 4.5]
    limitValue = 2.0

    oA.checkSavingTimeStepParticles(timeStepInfo, limitValue=limitValue)

    timeStepInfo = [0.0, 2.0001, 2.4, 4.0, 4.5]

    with pytest.raises(AssertionError) as e:
        oA.checkSavingTimeStepParticles(timeStepInfo, limitValue=limitValue)
    assert (
        "aving time step of simulation particle Info exceeds two seconds - this can lead to errors in analysis"
        in str(e.value)
    )
