"""Tests for module com1DFAtools"""
import numpy as np
import matplotlib.pyplot as plt
import configparser
import pytest

# Local imports
import avaframe.com1DFA.DFAToolsCython as DFAfunC
import avaframe.in3Utils.geoTrans as gT


def test_normalizeC(capfd):
    '''test DFAfunctions tools
    norm, norm2, normalize, crossProd and scalProd'''
    x = 1.
    y = 1.
    z = 1.
    norme = DFAfunC.norm(x, y, z)
    norme2 = DFAfunC.norm2(x, y, z)
    xn, yn, zn = DFAfunC.normalize(x, y, z)
#    print(xn, yn, zn)
    atol = 1e-10
    assert norme == np.sqrt(3.)
    assert norme2 == 3.
    assert np.sqrt(xn*xn + yn*yn + zn*zn) == pytest.approx(1., rel=atol)
    assert xn == 1/np.sqrt(3.)
    assert yn == 1/np.sqrt(3.)
    assert zn == 1/np.sqrt(3.)

    x = 0.
    y = 0.
    z = 1e-18
    xn, yn, zn = DFAfunC.normalize(x, y, z)
    assert np.sqrt(xn*xn + yn*yn + zn*zn) == pytest.approx(1, rel=atol)

    x = 0.
    y = 0.
    z = 0.
    xn, yn, zn = DFAfunC.normalize(x, y, z)
    assert np.sqrt(xn*xn + yn*yn + zn*zn) == pytest.approx(0, rel=atol)

    x = 1.
    y = 2.
    z = 3.
    xn, yn, zn = DFAfunC.normalize(x, y, z)
    assert np.sqrt(xn*xn + yn*yn + zn*zn) == pytest.approx(1., rel=atol)

    x = 1.
    y = 0.
    z = 1.
    xn, yn, zn = DFAfunC.normalize(x, y, z)
    assert np.sqrt(xn*xn + yn*yn + zn*zn) == pytest.approx(1, rel=atol)
    assert xn == pytest.approx(1/np.sqrt(2.), rel=atol)
    assert yn == pytest.approx(0, rel=atol)
    assert zn == pytest.approx(1/np.sqrt(2.), rel=atol)

    x = 1.
    y = 2.
    z = 3.
    x1 = 4.
    y1 = 5.
    z1 = 6.
    xn, yn, zn = DFAfunC.crossProd(x, y, z, x1, y1, z1)
    assert xn == -3
    assert yn == 6
    assert zn == -3

    x = 1.
    y = 2.
    z = 3.
    x1 = 4.
    y1 = 5.
    z1 = 6.
    scal = DFAfunC.scalProd(x, y, z, x1, y1, z1)
    assert scal == 32


def test_getWeightsC(capfd):
    '''getWeights'''
    ncols = 4
    nrows = 3
    X = np.array([0., 1, 2.5, 4., 4., 4.9])
    Y = np.array([0., 0., 2.5, 2.5, 3., 4.9])
    F00 = np.array([[1., 1., 0., 0., 0., 0.],
                    [1./4, 1./4, 1./4, 1./4, 1./4, 1./4],
                    [1., 0.8, 1./4, 0.09999999999999998, 0.07999999999999999, 0.0003999999999999963]])
    F10 = np.array([[0., 0., 0., 0., 0., 0.],
                    [1./4, 1./4, 1./4, 1./4, 1./4, 1./4],
                    [0., 0.2, 1./4, 0.4, 0.32, 0.01959999999999991]])
    F01 = np.array([[0., 0., 0., 0., 0., 0.],
                    [1./4, 1./4, 1./4, 1./4, 1./4, 1./4],
                    [0., 0., 1./4, 0.09999999999999998, 0.11999999999999997, 0.01959999999999991]])
    F11 = np.array([[0., 0., 1., 1., 1., 1.],
                    [1./4, 1./4, 1./4, 1./4, 1./4, 1./4],
                    [0., 0., 1./4, 0.4, 0.48, 0.9604000000000001]])
    csz = 5.
    atol = 1e-10
    # option 0: nearest, 1, unifom, 2 bilinear
    for interpOption in range(3):
        FF00 = F00[interpOption]
        FF10 = F10[interpOption]
        FF01 = F01[interpOption]
        FF11 = F11[interpOption]
        for x, y, ff00, ff10, ff01, ff11 in zip(X, Y, FF00, FF10, FF01, FF11):
            Lx0, Ly0, iCell, f00, f10, f01, f11 = DFAfunC.getCellAndWeights(x, y, ncols, nrows,
                                                                            csz, interpOption)
#            print(Lx0, Ly0)
#            print(f00, f10, f01, f11)
            assert Lx0 == 0.
            assert Ly0 == 0.
            assert iCell == 0.
            assert ff00 == pytest.approx(f00, rel=atol)
            assert ff10 == pytest.approx(f10, rel=atol)
            assert ff01 == pytest.approx(f01, rel=atol)
            assert ff11 == pytest.approx(f11, rel=atol)

    # same as before but shift 2 in x dir and 1 in y dir
    X = X + 2*csz
    Y = Y + csz
    for interpOption in range(3):
        FF00 = F00[interpOption]
        FF10 = F10[interpOption]
        FF01 = F01[interpOption]
        FF11 = F11[interpOption]
        for x, y, ff00, ff10, ff01, ff11 in zip(X, Y, FF00, FF10, FF01, FF11):
            Lx0, Ly0, iCell, f00, f10, f01, f11 = DFAfunC.getCellAndWeights(x, y, ncols, nrows,
                                                                            csz, interpOption)
#            print(Lx0, Ly0, iCell)
#            print(f00, f10, f01, f11)
            assert Lx0 == 2.
            assert Ly0 == 1.
            assert iCell == 6.
            assert ff00 == pytest.approx(f00, rel=atol)
            assert ff10 == pytest.approx(f10, rel=atol)
            assert ff01 == pytest.approx(f01, rel=atol)
            assert ff11 == pytest.approx(f11, rel=atol)

    ncols = 4
    nrows = 3
    csz = 1.
    X = np.array([0., 1, -0.5, 0, 0, 1, 3, 3, 3.1, 2])
    Y = np.array([0., -0.5, 1, 1, 2, 0, 0, 2, 1, 2.1])
    cellInd = np.array([0, -1, -1, 4, -1, 1, -1, -1, -1, -1])
    for x, y, res in zip(X, Y, cellInd):
        iCell = DFAfunC.getCells(x, y, ncols, nrows, csz)
#        print(iCell)
        assert iCell == res


def test_getScalVect(capfd):
    '''getScalar and getVector'''
    ncols = 100
    nrows = 10
    csz = 5
    interpOption = 2
    X = np.linspace(0, csz*(ncols-1), ncols)
    Y = np.linspace(0, csz*(nrows-1), nrows)
    XX, YY = np.meshgrid(X, Y)
    ZZ = -2*XX + 1000 - YY + 500

    xpExpected = [0, 12.5, 12.5, 52, 95, 494]
    ypExpected = [0, 0, 12.5, 41, 37, 44]
    for xpExp, ypExp in zip(xpExpected, ypExpected):
        Lx0, Ly0, iCell, w0, w1, w2, w3 = DFAfunC.getCellAndWeights(xpExp, ypExp, ncols, nrows,
                                                                    csz, interpOption)
        zScalRes = DFAfunC.getScalar(Lx0, Ly0, w0, w1, w2, w3, ZZ)
        xRes, yRes, zRes = DFAfunC.getVector(Lx0, Ly0, w0, w1, w2, w3, ZZ,
                                             2*ZZ, 3*ZZ)
        res = -2*xpExp + 1000-ypExp + 500
#        print(xRes, res)
        atol = 1e-10
        assert zScalRes == pytest.approx(res, rel=atol)
        assert xRes == pytest.approx(res, rel=atol)
        assert yRes == pytest.approx(2*res, rel=atol)
        assert zRes == pytest.approx(3*res, rel=atol)

    zRes = DFAfunC.projOnRaster(np.array(xpExpected), np.array(ypExpected), ZZ, csz, ncols, nrows,
                                interpOption)
    res = -2*np.array(xpExpected) + 1000-np.array(ypExpected) + 500
    atol = 1e-10
    assert np.allclose(zRes, res, atol=atol)


def test_reprojectionC(capfd):
    '''test reprojection'''
    ncols = 100
    nrows = 10
    csz = 5
    header = {}
    header['ncols'] = ncols
    header['nrows'] = nrows
    header['cellsize'] = csz
    dem = {}
    dem['header'] = header
    x0 = 500
    z0 = 1000
    X = np.linspace(0, csz*(ncols-1), ncols)
    Y = np.linspace(0, csz*(nrows-1), nrows)
    XX, YY = np.meshgrid(X, Y)
    ZZ = z0/(x0*x0) * (XX-x0)*(XX-x0)
    dem['rasterData'] = ZZ
    num = 1
    interpOption = 2
    threshold = 0.001

    # build normals corresponding to dem
    dem = gT.getNormalMesh(dem, num=num)
    Nx = dem['Nx']
    Ny = dem['Ny']
    Nz = dem['Nz']

    # expected normal projection
    xpExpected = 242
    ypExpected = 25
    Lx0, Ly0, iCell, w0, w1, w2, w3 = DFAfunC.getCellAndWeights(xpExpected, ypExpected, ncols,
                                                                nrows, csz, interpOption)
    zpExpected = DFAfunC.getScalar(Lx0, Ly0, w0, w1, w2, w3, ZZ)
    nx, ny, nz = DFAfunC.getVector(Lx0, Ly0, w0, w1, w2, w3, Nx, Ny, Nz)
    nx, ny, nz = DFAfunC.normalize(nx, ny, nz)
    # previous position
    xPrev = 230
    yPrev = 25
    Lx0, Ly0, iCell, w0, w1, w2, w3 = DFAfunC.getCellAndWeights(xPrev, yPrev, ncols, nrows,
                                                                csz, interpOption)
    zPrev = DFAfunC.getScalar(Lx0, Ly0, w0, w1, w2, w3, ZZ)

    # make a point above the parabola at dist d:
    d = 15

#    print(xpExpected, zpExpected)
    x1 = xpExpected + d*nx
    y1 = ypExpected + d*ny
    z1 = zpExpected + d*nz
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(X, ZZ[5, :], 'ok-')
    ax.plot(x1, z1, 'ro')
    ax.plot(xpExpected, zpExpected, 'go')
    ax.set_aspect('equal', 'box')

    for reprojectionIterations in range(10):
        xpn, ypn, iCell, Lx0, Ly0, w0, w1, w2, w3 = DFAfunC.samosProjectionIteratrive(x1, y1, z1,
                                                                                      ZZ, Nx, Ny, Nz, csz, ncols, nrows, interpOption, reprojectionIterations)

        zpn = DFAfunC.getScalar(Lx0, Ly0, w0, w1, w2, w3, ZZ)
#        print(xpn, zpn)
        ax.plot(xpn, zpn, 'bo')
    ax.set_aspect('equal', 'box')
    # plt.show()
    plt.close(fig)

    atol = 1e-2

    assert xpn == pytest.approx(xpExpected, rel=atol)
    assert ypn == pytest.approx(ypExpected, rel=atol)
    assert zpn == pytest.approx(zpExpected, rel=atol)

    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(X, ZZ[5, :], 'ok-')
    ax.plot(x1, z1, 'ro')
    ax.plot(xpExpected, zpExpected, 'go')

    for reprojectionIterations in range(10):
        xpn, ypn, iCell, Lx0, Ly0, w0, w1, w2, w3 = DFAfunC.normalProjectionIteratrive(x1, y1, z1,
                                                                                       ZZ, Nx, Ny, Nz, csz, ncols, nrows, interpOption, reprojectionIterations, threshold)
        zpn = DFAfunC.getScalar(Lx0, Ly0, w0, w1, w2, w3, ZZ)
#        print(xpn, zpn)
        ax.plot(xpn, zpn, 'bo')

    ax.plot([xpExpected, x1], [zpExpected, z1], 'b')
    ax.set_aspect('equal', 'box')
    # plt.show()
    plt.close(fig)

    atol = 1e-2

    dist = DFAfunC.norm(xpn-xpExpected, ypn-ypExpected, zpn-zpExpected)
    assert dist <= csz*threshold
    assert xpn == pytest.approx(xpExpected, rel=atol)
    assert ypn == pytest.approx(ypExpected, rel=atol)
    assert zpn == pytest.approx(zpExpected, rel=atol)

    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(X, ZZ[5, :], 'ok-')
    ax.plot(x1, z1, 'ro')
    ax.plot(xPrev, zPrev, 'mo')
    ax.plot(xpExpected, zpExpected, 'go')

    for reprojectionIterations in range(10):
        xpn, ypn, zpn, iCell, Lx0, Ly0, w0, w1, w2, w3 = DFAfunC.distConservProjectionIteratrive(
            xPrev, yPrev, zPrev, ZZ, Nx, Ny, Nz, x1, y1, z1, csz, ncols, nrows, interpOption,
            reprojectionIterations, threshold)
#        print(xpn, zpn)
        ax.plot(xpn, zpn, 'bo')

    ax.plot([xpExpected, x1], [zpExpected, z1], 'b')
    dist0 = DFAfunC.norm(x1-xPrev, y1-yPrev, z1-zPrev)
    circ = plt.Circle([xPrev, zPrev], dist0, fill=True)
    ax.add_patch(circ)
    ax.set_aspect('equal', 'box')
    # plt.show()
    plt.close(fig)

    atol = 1e-2

    dist0 = DFAfunC.norm(x1-xPrev, y1-yPrev, z1-zPrev)
    dist = DFAfunC.norm(xpn-xPrev, ypn-yPrev, zpn-zPrev)
    assert abs(dist-dist0) <= threshold*(dist0 + csz)


def test_distConservStoppedParticle(capfd):
    """distConservProjectionIteratrive handles distn=0 inside loop without NaN.

    prev is on a flat surface, current is above the surface at the same (x,y).
    Initial distn is non-zero (current is above surface), so the loop enters.
    After orthogonal reprojection, the particle lands on the surface and distn
    becomes zero, triggering the division by zero in the distance conservation step."""
    ncols = 20
    nrows = 20
    csz = 5
    header = {}
    header['ncols'] = ncols
    header['nrows'] = nrows
    header['cellsize'] = csz
    dem = {}
    dem['header'] = header
    X = np.linspace(0, csz * (ncols - 1), ncols)
    Y = np.linspace(0, csz * (nrows - 1), nrows)
    XX, YY = np.meshgrid(X, Y)
    # Flat surface so normal is straight up, no lateral shift
    ZZ = np.zeros_like(XX)
    dem['rasterData'] = ZZ
    num = 1
    interpOption = 2
    dem = gT.getNormalMesh(dem, num=num)
    Nx = dem['Nx']
    Ny = dem['Ny']
    Nz = dem['Nz']

    xPrev = 50.0
    yPrev = 50.0
    zPrev = 0.0  # on flat surface
    xCur = xPrev  # same x, y — no lateral movement
    yCur = yPrev
    zCur = 20.0  # above surface
    threshold = 0.001

    xNew, yNew, zNew, iCell, Lx0, Ly0, w0, w1, w2, w3 = DFAfunC.distConservProjectionIteratrive(
        xPrev, yPrev, zPrev, ZZ, Nx, Ny, Nz,
        xCur, yCur, zCur,
        csz, ncols, nrows, interpOption, 10, threshold)

    # Must not be NaN or inf (would be if division by zero is unguarded)
    assert not np.isnan(xNew)
    assert not np.isnan(yNew)
    assert not np.isnan(zNew)
    assert np.isfinite(xNew)
    assert np.isfinite(yNew)
    assert np.isfinite(zNew)


def test_SamosATfric(capfd):
    """ Test the account4FrictionForce function"""
    tau0 = 0
    Rs0 = 0.222
    kappa = 0.43
    R = 0.05
    B = 4.13
    rho = 0.1
    mu = 0.155
    uMag = 10
    sigmaB = 10
    h = 1
    tau = DFAfunC.SamosATfric(rho, tau0, Rs0, mu, kappa, B, R, uMag, sigmaB, h)
#    print(tau)
    assert tau == 1.9128193823277053
    tau0 = 2
    tau = DFAfunC.SamosATfric(rho, tau0, Rs0, mu, kappa, B, R, uMag, sigmaB, h)
#    print(tau)
    assert tau == 3.9128193823277053
