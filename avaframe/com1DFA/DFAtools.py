"""
    Basic tools for getting grid normals, area and working with vectors.
"""

# Load modules
import logging
import numpy as np

# Local imports
import avaframe.in3Utils.geoTrans as geoTrans


# create local logger
# change log level in calling module to DEBUG to see log messages
log = logging.getLogger(__name__)


def getAreaMesh(dem, num):
    """ Get area of grid cells.

        Parameters
        ----------
        dem dict updated with:
            Nx: 2D numpy array
                x component of the normal vector field on grid points
            Ny: 2D numpy array
                y component of the normal vector field on grid points
            Nz: 2D numpy array
                z component of the normal vector field on grid points
            header : dict
                dem header (cellsize)
        num: int
            chose between 4, 6 or 8 (using then 4, 6 or 8 triangles) or
            1 to use the simple cross product method (with the diagonals)

        Returns
        -------
        dem dict updated with:
            areaRaster: 2D numpy array
                real area of grid cells
    """
    csz = dem['header']['cellsize']
    Nx = dem['Nx']
    Ny = dem['Ny']
    Nz = dem['Nz']
    # see documentation and issue 202
    if num == 1:
        A = norm(Nx, Ny, Nz)
    else:
        _, _, NzNormed = normalize(Nx, Ny, Nz)
        A = csz * csz / NzNormed
    dem['areaRaster'] = A
    return dem


##############################################################################
# ###################### Vectorial functions #################################
##############################################################################


def norm(x, y, z):
    """ Compute the Euclidean norm of the vector (x, y, z).

    (x, y, z) can be numpy arrays.

    Parameters
    ----------
        x: numpy array
            x component of the vector
        y: numpy array
            y component of the vector
        z: numpy array
            z component of the vector

    Returns
    -------
        norme: numpy array
            norm of the vector
    """
    norme = np.sqrt(x*x + y*y + z*z)
    return norme


def norm2(x, y, z):
    """ Compute the square of the Euclidean norm of the vector (x, y, z).

    (x, y, z) can be numpy arrays.

    Parameters
    ----------
        x: numpy array
            x component of the vector
        y: numpy array
            y component of the vector
        z: numpy array
            z component of the vector

    Returns
    -------
        norme2: numpy array
            square of the norm of the vector
    """
    norme2 = (x*x + y*y + z*z)
    return norme2


def normalize(x, y, z):
    """ Normalize vector (x, y, z) for the Euclidean norm.

    (x, y, z) can be np arrays.

    Parameters
    ----------
        x: numpy array
            x component of the vector
        y: numpy array
            y component of the vector
        z: numpy array
            z component of the vector

    Returns
    -------
        x: numpy array
            x component of the normalized vector
        y: numpy array
            y component of the normalized vector
        z: numpy array
            z component of the normalized vector
    """
    norme = norm(x, y, z)
    # elementwise division by norme
    factor = np.divide(
        1.0,
        norme,
        out=np.zeros_like(norme, dtype=float),
        where=norme > 0,
    )

    x = x * factor
    y = y * factor
    z = z * factor

    return x, y, z


def crossProd(ux, uy, uz, vx, vy, vz):
    """ Compute cross product of vector u = (ux, uy, uz) and v = (vx, vy, vz).
    """
    wx = uy*vz - uz*vy
    wy = uz*vx - ux*vz
    wz = ux*vy - uy*vx

    return wx, wy, wz


def scalProd(ux, uy, uz, vx, vy, vz):
    """ Compute scalar product of vector u = (ux, uy, uz) and v = (vx, vy, vz).
    """
    scal = ux*vx + uy*vy + uz*vz

    return scal
