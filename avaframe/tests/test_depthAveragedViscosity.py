"""
Pytest for module depthAveragedViscosity (viscous term of the hysteresis friction model)

The velocity fields are polynomials of degree <= 2 on planes, for which the central differences on the interior grid
nodes are exact (only for the variable thickness test the differences are approximate).
"""

import configparser
import logging
import math

import numpy as np
import pytest

from avaframe.com1DFA import depthAveragedViscosity as dAV


def _getCfg(option="1", nu="0.002"):
    cfg = configparser.ConfigParser()
    cfg["GENERAL"] = {
        "hysteresisViscosity": option,
        "hysteresisNu": nu,
        "hysteresisMu1": "0.5867",
        "hysteresisMu2": "0.9391",
        "hysteresisMu3": "0.6976",
        "hysteresisL": "0.00056",
        "hysteresisBeta": "0.63",
        "gravAcc": "9.81",
        "depMin": "1.0e-6",
    }
    return cfg["GENERAL"]


def _getPlane(thetaDeg, nrows=11, ncols=13, csz=0.5):
    """plane descending in x direction with slope angle theta, node coordinates and downslope unit vector"""
    theta = math.radians(thetaDeg)
    dem = {
        "header": {"cellsize": csz, "nrows": nrows, "ncols": ncols},
        "Nx": np.full((nrows, ncols), math.sin(theta)),
        "Ny": np.zeros((nrows, ncols)),
        "Nz": np.full((nrows, ncols), math.cos(theta)),
    }
    x, y = np.meshgrid(np.arange(ncols) * csz, np.arange(nrows) * csz)
    # distance along the slope and downslope unit vector
    s = x / math.cos(theta)
    eS = (math.cos(theta), 0.0, -math.sin(theta))
    return dem, x, y, s, eS


def _getFields(velocity, h):
    return {"Vx": velocity[0], "Vy": velocity[1], "Vz": velocity[2], "FT": h}


def _interior(a):
    return a[2:-2, 2:-2]


@pytest.mark.parametrize("thetaDeg", [0.0, 35.0])
def test_viscousAccelerationRigidMotion(thetaDeg):
    """no viscous acceleration for a uniform velocity, and for a rigid rotation on a flat plane"""

    cfgGen = _getCfg()
    dem, x, y, s, eS = _getPlane(thetaDeg)
    h = np.full(np.shape(x), 0.01)
    velocity = [np.full(np.shape(x), 3.0 * e) for e in eS]
    acc, _ = dAV.computeViscousAcceleration(_getFields(velocity, h), dem, cfgGen)
    for k in range(3):
        assert np.allclose(acc[k], 0.0, atol=1e-14)

    if thetaDeg == 0.0:
        omega = 0.7
        velocity = [-omega * y, omega * x, np.zeros(np.shape(x))]
        acc, _ = dAV.computeViscousAcceleration(_getFields(velocity, h), dem, cfgGen)
        for k in range(3):
            assert np.allclose(acc[k], 0.0, atol=1e-12)


@pytest.mark.parametrize("thetaDeg", [0.0, 20.0, 35.0])
def test_viscousAccelerationShearAndExtension(thetaDeg):
    """constant thickness: downslope extension U(s) = c s² gives a = nu h^(1/2) U'' e_s,
    cross-slope shear W(y) = c y² gives a = nu h^(1/2) W'' / 2 e_s (D has half of the shear rate)"""

    cfgGen = _getCfg(nu="0.002")
    nu = 0.002
    c = 0.3
    hValue = 0.04
    dem, x, y, s, eS = _getPlane(thetaDeg)
    h = np.full(np.shape(x), hValue)

    # extension along the slope
    velocity = [c * s**2 * e for e in eS]
    acc, kappaMax = dAV.computeViscousAcceleration(_getFields(velocity, h), dem, cfgGen)
    for k in range(3):
        assert np.allclose(_interior(acc[k]), nu * math.sqrt(hValue) * 2.0 * c * eS[k], rtol=1e-10, atol=1e-14)
    assert kappaMax == pytest.approx(nu * math.sqrt(hValue))

    # shear across the slope
    velocity = [c * y**2 * e for e in eS]
    acc, _ = dAV.computeViscousAcceleration(_getFields(velocity, h), dem, cfgGen)
    for k in range(3):
        assert np.allclose(_interior(acc[k]), nu * math.sqrt(hValue) * c * eS[k], rtol=1e-10, atol=1e-14)


def test_viscousAccelerationVariableThickness():
    """thickness h(x) and linear velocity U(x) on a flat plane: a = (1/h) d/dx(nu h^(3/2) U')"""

    cfgGen = _getCfg(nu="0.002")
    nu = 0.002
    dem, x, y, s, eS = _getPlane(0.0, nrows=7, ncols=201, csz=0.01)
    h = 0.01 + 0.02 * x
    uPrime = 0.5
    velocity = [uPrime * x, np.zeros(np.shape(x)), np.zeros(np.shape(x))]
    acc, _ = dAV.computeViscousAcceleration(_getFields(velocity, h), dem, cfgGen)
    expected = nu * uPrime * 1.5 * np.sqrt(h) * 0.02 / h
    assert np.allclose(_interior(acc[0]), _interior(expected), rtol=1e-4)
    assert np.allclose(acc[1], 0.0) and np.allclose(acc[2], 0.0)


def test_viscousAccelerationNoFlow():
    """no acceleration where the flow thickness is zero"""

    cfgGen = _getCfg()
    dem, x, y, s, eS = _getPlane(0.0)
    h = np.where(x < 3.0, 0.02, 0.0)
    velocity = [0.4 * x**2, np.zeros(np.shape(x)), np.zeros(np.shape(x))]
    acc, _ = dAV.computeViscousAcceleration(_getFields(velocity, h), dem, cfgGen)
    assert np.all(acc[0][h == 0] == 0.0)
    assert np.all(np.isfinite(acc[0]))


def test_viscousAccelerationStaticLayerFields():
    """with a static entrainment layer left out of the result fields, the fields including it (FTAll, VxAll, ...)
    are used"""

    cfgGen = _getCfg()
    dem, x, y, s, eS = _getPlane(0.0)
    h = 0.01 + 0.002 * x
    velocity = [0.4 * x**2, 0.1 * x, np.zeros(np.shape(x))]
    accAll, kappaAll = dAV.computeViscousAcceleration(_getFields(velocity, h), dem, cfgGen)
    fields = _getFields([0.5 * v for v in velocity], 0.5 * h)
    fields.update({"FTAll": h, "VxAll": velocity[0], "VyAll": velocity[1], "VzAll": velocity[2]})
    acc, kappa = dAV.computeViscousAcceleration(fields, dem, cfgGen)
    for k in range(3):
        assert np.array_equal(acc[k], accAll[k])
    assert kappa == kappaAll


def test_getViscosityCoefficient():
    """option 1: constant, option 2: Edwards et al. (2021, eq. 3.6 and table 2) with the local slope, limited"""

    thetas = np.radians(np.array([[33.5, 35.5, 30.0, 45.0]]))
    nz = np.cos(thetas)
    nu = dAV.getViscosityCoefficient(_getCfg(option="1", nu="0.001"), nz)
    assert np.all(nu == 0.001)

    nu = dAV.getViscosityCoefficient(_getCfg(option="2", nu="0.0014"), nz)
    # table 2: 1.4e-3 (33.5 deg) and 0.7e-3 (35.5 deg) m^(3/2)/s, rounded
    assert nu[0, 0] == pytest.approx(1.4e-3, abs=0.05e-3)
    assert nu[0, 1] == pytest.approx(0.7e-3, abs=0.05e-3)
    # below atan(mu1): limit, above atan(mu2): 0
    assert nu[0, 2] == 0.0014
    assert nu[0, 3] == 0.0


def test_addViscousForce(caplog):
    """force = particle mass x acceleration interpolated to the particles, time step warning issued once"""

    cfgGen = _getCfg(nu="0.002")
    nu = 0.002
    c = 0.3
    hValue = 0.04
    dem, x, y, s, eS = _getPlane(35.0)
    fields = _getFields([c * s**2 * e for e in eS], np.full(np.shape(x), hValue))
    particles = {
        "x": np.array([2.3, 3.1]),
        "y": np.array([2.2, 2.9]),
        "m": np.array([2.0, 0.5]),
        "dt": 1.0e-3,
    }
    force = {key: np.ones(2) for key in ["forceX", "forceY", "forceZ"]}
    with caplog.at_level(logging.WARNING):
        force = dAV.addViscousForce(cfgGen, particles, force, fields, dem)
    assert caplog.records == []
    for k, key in enumerate(["forceX", "forceY", "forceZ"]):
        expected = 1.0 + particles["m"] * nu * math.sqrt(hValue) * 2.0 * c * eS[k]
        assert np.allclose(force[key], expected, rtol=1e-10, atol=1e-14)

    # time step above the explicit limit cellsize² / (4 nu h^(1/2)) = 156 s: warning only once
    particles["dt"] = 200.0
    with caplog.at_level(logging.WARNING):
        dAV.addViscousForce(cfgGen, particles, force, fields, dem)
        dAV.addViscousForce(cfgGen, particles, force, fields, dem)
    assert caplog.text.count("explicit limit") == 1
