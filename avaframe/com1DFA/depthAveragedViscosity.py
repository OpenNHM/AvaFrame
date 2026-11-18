"""
Depth-averaged viscous term of the mu(I) rheology for the hysteresis friction model

The momentum balance of Gray and Edwards (2014) and Edwards et al. (2021, eq. 3.2) contains the term
div(nu h^(3/2) D), with the depth-averaged strain rate D = (grad(u) + grad(u)^T) / 2. Per unit mass this is the
acceleration (1/h) div(nu h^(3/2) D). It is computed on the grid nodes from the mass averaged velocity (Vx, Vy, Vz)
and the flow thickness (FT) with surface derivatives on the topography, and interpolated to the particles.
"""

import logging
import math

import numpy as np

import avaframe.com1DFA.DFAtools as DFAtls
import avaframe.in3Utils.geoTrans as geoTrans

# create local logger
log = logging.getLogger(__name__)


def getViscosityCoefficient(cfgGen, nz):
    """viscosity coefficient nu [m^(3/2)/s] on the grid nodes

    hysteresisViscosity = 1: constant nu = hysteresisNu
    hysteresisViscosity = 2: nu of Edwards et al. (2021, eq. 3.6) with the local slope angle zeta,
    nu = 2/9 * L sqrt(g) / beta * sin(zeta) / sqrt(cos(zeta)) * (mu2 - tan(zeta)) / (tan(zeta) - mu1),
    limited to hysteresisNu (nu tends to infinity for tan(zeta) -> mu1 and is undefined below) and 0 for
    tan(zeta) >= mu2

    Parameters
    -----------
    cfgGen: configparser section
        configuration settings, here used: hysteresisViscosity, hysteresisNu, gravAcc and the hysteresis parameters
    nz: 2D numpy array
        z component of the unit normal vector of the topography (cos(zeta))

    Returns
    --------
    nu: 2D numpy array
        viscosity coefficient on the grid nodes
    """
    option = cfgGen.getint("hysteresisViscosity")
    nuMax = cfgGen.getfloat("hysteresisNu")
    if option == 1:
        return np.full(np.shape(nz), nuMax)

    mu1 = cfgGen.getfloat("hysteresisMu1")
    mu2 = cfgGen.getfloat("hysteresisMu2")
    cosZeta = nz
    sinZeta = np.sqrt(np.maximum(1.0 - nz * nz, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        tanZeta = sinZeta / cosZeta
        nu = (
            2.0
            / 9.0
            * cfgGen.getfloat("hysteresisL")
            * math.sqrt(cfgGen.getfloat("gravAcc"))
            / cfgGen.getfloat("hysteresisBeta")
            * sinZeta
            / np.sqrt(cosZeta)
            * (mu2 - tanZeta)
            / (tanZeta - mu1)
        )
    nu = np.where(tanZeta <= mu1, nuMax, np.where(tanZeta >= mu2, 0.0, np.minimum(nu, nuMax)))
    return np.nan_to_num(nu, nan=0.0)


def gradXY(a, csz):
    """derivatives of a grid array with respect to x (columns) and y (rows)

    Central differences on the interior nodes, one-sided differences on the border nodes (numpy.gradient).

    Parameters
    -----------
    a: 2D numpy array
        values on the grid nodes
    csz: float
        cell size [m]

    Returns
    --------
    dadx: 2D numpy array
        derivative with respect to x (along the rows, from column to column)
    dady: 2D numpy array
        derivative with respect to y (along the columns, from row to row)
    """
    return np.gradient(a, csz, axis=1), np.gradient(a, csz, axis=0)


def computeViscousAcceleration(fields, dem, cfgGen):
    """acceleration (1/h) div(nu h^(3/2) D) of the depth-averaged viscous term on the grid nodes

    Derivatives on the topography use the tangential projection P = I - n n: the surface gradient of the velocity is
    G = J P with J_kj = dV_k / dx_j (j = x, y; central differences on the grid), its tangential part L = P G,
    D = (L + L^T) / 2 and the surface divergence of T = nu h^(3/2) D is f_k = sum_l sum_j P_jl dT_kl / dx_j. The
    tangential part of f divided by h is the acceleration. Nodes with h <= depMin get no acceleration.

    Parameters
    -----------
    fields: dict
        fields dictionary with Vx, Vy, Vz (mass averaged velocity) and FT (flow thickness) on the grid nodes
    dem: dict
        dem dictionary with header (cellsize) and the normal vectors Nx, Ny, Nz on the grid nodes
    cfgGen: configparser section
        configuration settings, here used: depMin and see getViscosityCoefficient

    Returns
    --------
    acc: list of three 2D numpy arrays
        x, y and z component of the acceleration [m/s²] on the grid nodes
    kappaMax: float
        largest effective kinematic viscosity nu h^(1/2) [m²/s] (for the time step criterion)
    """
    csz = dem["header"]["cellsize"]
    # fields including the static entrainment layer particles (FTAll, VxAll, ... only exist if the layer particles at
    # rest are left out of the result fields)
    h = fields.get("FTAll", fields["FT"])
    vel = [fields.get("VxAll", fields["Vx"]), fields.get("VyAll", fields["Vy"]), fields.get("VzAll", fields["Vz"])]
    nx, ny, nz = DFAtls.normalize(dem["Nx"].copy(), dem["Ny"].copy(), dem["Nz"].copy())
    n = [nx, ny, nz]
    # tangential projection P = I - n n (symmetric)
    proj = [[(1.0 if i == j else 0.0) - n[i] * n[j] for j in range(3)] for i in range(3)]

    # surface gradient of the velocity G_kl = sum_j dV_k/dx_j P_jl (no z derivative)
    dVel = [gradXY(vel[k], csz) for k in range(3)]
    grad = [[dVel[k][0] * proj[0][l] + dVel[k][1] * proj[1][l] for l in range(3)] for k in range(3)]
    # tangential part L = P G
    velGrad = [[sum(proj[k][m] * grad[m][l] for m in range(3)) for l in range(3)] for k in range(3)]

    nu = getViscosityCoefficient(cfgGen, nz)
    hPos = np.maximum(h, 0.0)
    coef = nu * hPos**1.5
    stress = [[coef * 0.5 * (velGrad[k][l] + velGrad[l][k]) for l in range(3)] for k in range(3)]

    # surface divergence f_k = sum_l sum_j P_jl dT_kl/dx_j
    div = []
    for k in range(3):
        divK = np.zeros(np.shape(h))
        for l in range(3):
            dTdx, dTdy = gradXY(stress[k][l], csz)
            divK = divK + proj[0][l] * dTdx + proj[1][l] * dTdy
        div.append(divK)
    # tangential part, per unit mass
    depMin = cfgGen.getfloat("depMin")
    flowing = h > depMin
    acc = []
    for k in range(3):
        divTang = sum(proj[k][m] * div[m] for m in range(3))
        acc.append(np.where(flowing, divTang / np.where(flowing, h, 1.0), 0.0))

    kappaMax = float(np.max(nu * np.sqrt(hPos))) if np.size(h) > 0 else 0.0
    return acc, kappaMax


def addViscousForce(cfgGen, particles, force, fields, dem):
    """add the force of the depth-averaged viscous term to the particle forces (forceX, forceY, forceZ)

    The force is part of the driving force, so it also enters the yield condition of particles at rest.
    A warning is issued once per simulation if the time step exceeds the explicit diffusion limit
    cellsize² / (4 nu h^(1/2)).

    Parameters
    -----------
    cfgGen: configparser section
        configuration settings, see computeViscousAcceleration
    particles: dict
        particles dictionary (x, y relative to the dem origin, m, dt)
    force: dict
        force dictionary with forceX, forceY, forceZ
    fields: dict
        fields dictionary, see computeViscousAcceleration
    dem: dict
        dem dictionary, see computeViscousAcceleration

    Returns
    --------
    force: dict
        force dictionary with the viscous force added to forceX, forceY, forceZ
    """
    # TODO: is this viscosity necessary?

    acc, kappaMax = computeViscousAcceleration(fields, dem, cfgGen)
    csz = dem["header"]["cellsize"]
    if kappaMax > 0 and particles["dt"] > csz * csz / (4.0 * kappaMax) and not fields.get("viscosityWarned", False):
        log.warning(
            "Time step %.3g s is larger than the explicit limit %.3g s of the depth-averaged viscous term "
            "(cellsize² / (4 nu h^(1/2)), nu h^(1/2) = %.3g m²/s): reduce dt or hysteresisNu"
            % (particles["dt"], csz * csz / (4.0 * kappaMax), kappaMax)
        )
        fields["viscosityWarned"] = True

    for k, key in enumerate(["forceX", "forceY", "forceZ"]):
        accPart, _ = geoTrans.projectOnGrid(particles["x"], particles["y"], acc[k], csz=csz, interp="bilinear")
        force[key] = force[key] + particles["m"] * np.nan_to_num(accPart, nan=0.0)
    return force
