#!python
# cython: boundscheck=False, wraparound=False, cdivision=True
"""
    function related to SPH calculations in com1DFA
    to build: go to repository containing this file and run:
    python setup.py build_ext --inplace
"""

# Load modules
import copy
import logging
import math
import cython
import numpy as np
cimport numpy as np
from libc cimport math as math

# create local logger
# change log level in calling module to DEBUG to see log messages
log = logging.getLogger(__name__)


def pointsToRasterC(double[:] xArray, double[:] yArray, double[:] zArray, Z0, double csz=1, double xllc=0, double yllc=0):
    """ Interpolate from unstructured points to grid

    Interpolate unstructured points on a structured grid using bilinear interpolation
    The (x, y) points have to be on the extend of the DEM!! (unexpected behaviours or errors if not)

    Parameters
      ----------
      xArray: 1D numpy array
          x coordinate of the points
      yArray: 1D numpy array
          y coordinate of the points
      zArray: 1D numpy array
          quantity to interpolate associated to (x, y) points
      Z0: 2D numpy array
          initial ratser for interpolated result
      csz : float
          cellsize
      xllc : float
          x coord of the lower left center
      yllc : float
          y coord of the lower left center

      Returns
      -------
      Z1: 2D numpy array
          interpolated results
    """
    n, m = np.shape(Z0)
    cdef int nrow = int(n)
    cdef int ncol = int(m)
    cdef int Lx0, Ly0, Lx1, Ly1
    cdef double Lx, Ly, x, y, z
    cdef double[:] zRaster = Z0.flatten()
    cdef int nPart = len(xArray)
    cdef int k, ic

    for k in range(nPart):
      x = xArray[k]
      y = yArray[k]
      z = zArray[k]
      # find coordinates in normalized ref (origin (0,0) and cellsize 1)
      Lx = (x - xllc) / csz
      Ly = (y - yllc) / csz

      # find coordinates of the 4 nearest cells on the raster
      Lx0 = <int>math.floor(Lx)
      Ly0 = <int>math.floor(Ly)
      Lx1 = Lx0 + 1
      Ly1 = Ly0 + 1
      # prepare for bilinear interpolation
      dx = Lx - Lx0
      dy = Ly - Ly0

      # add the component of the points value to the 4 neighbour grid points
      # start with the lower left
      f11 = z*(1-dx)*(1-dy)
      ic = Lx0 + ncol * Ly0
      zRaster[ic] = zRaster[ic] + f11
      # lower right
      f21 = z*dx*(1-dy)
      ic = Lx1 + ncol * Ly0
      zRaster[ic] = zRaster[ic] + f21
      # uper left
      f12 = z*(1-dx)*dy
      ic = Lx0 + ncol * Ly1
      zRaster[ic] = zRaster[ic] + f12
      # and uper right
      f22 = z*dx*dy
      ic = Lx1 + ncol * Ly1
      zRaster[ic] = zRaster[ic] + f22

    Z1 = np.reshape(np.asarray(zRaster), (np.shape(Z0)))

    return Z1


cpdef double norm(double x, double y, double z):
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
  return math.sqrt(x*x + y*y + z*z)

cpdef double norm2(double x, double y, double z):
  """ Compute the Euclidean norm2 (square of the norm) of the vector (x, y, z).

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
  return x*x + y*y + z*z


cpdef (double, double, double) normalize(double x, double y, double z):
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
      xn: numpy array
          x component of the normalized vector
      yn: numpy array
          y component of the normalized vector
      zn: numpy array
          z component of the normalized vector
  """
  cdef double norme
  cdef double xn = 0.0
  cdef double yn = 0.0
  cdef double zn = 0.0

  norme = norm(x, y, z)
  if norme>0:
    xn = x / norme
    yn = y / norme
    zn = z / norme
  return xn, yn, zn


cpdef (double, double, double) crossProd(double ux, double uy, double uz, double vx, double vy, double vz):
  """ Compute cross product of vector u = (ux, uy, uz) and v = (vx, vy, vz).
  """
  cdef double wx = uy * vz - uz * vy
  cdef double wy = uz * vx - ux * vz
  cdef double wz = ux * vy - uy * vx
  return wx, wy, wz


cpdef double scalProd(double ux, double uy, double uz, double vx, double vy, double vz):
  """ Compute scalar product of vector u = (ux, uy, uz) and v = (vx, vy, vz).
  """
  return ux*vx + uy*vy + uz*vz


cpdef (int) getCells(double x, double y, int ncols, int nrows, double csz):
  """ Locate point on grid (find the index of the grid cell containing the point).

  Parameters
  ----------
      x: float
          location in the x location of desired interpolation
      y: float
          location in the y location of desired interpolation
      ncols: int
          number of columns
      nrows: int
          number of rows
      csz: float
          cellsize of the grid

  Returns
  -------
      iCell: int
          index of the nearest lower left cell
  """
  cdef double Lx, Ly
  cdef int Lx0, Ly0
  cdef double xllc = 0.
  cdef double yllc = 0.
  cdef int iCell

  # find coordinates in normalized ref (origin (0,0) and cellsize 1)
  Lx = (x - xllc) / csz
  Ly = (y - yllc) / csz
  # find coordinates of the lower left cell
  Lx0 = <int>math.floor(Lx)
  Ly0 = <int>math.floor(Ly)
  iCell = Ly0*ncols + Lx0
  if (Lx0<0) | (Ly0<0) | (Lx0+1>=ncols) | (Ly0+1>=nrows):
    # check whether we are in the domain or not
    return -1

  return iCell


cpdef (double, double, double, double) getWeights(double x, double y, int iCell, double csz, int ncols, int interpOption):
  """ Get weight for interpolation from grid to single point location

  3 Options available : 0: nearest neighbour interpolation
                        1: equal weights interpolation
                        2: bilinear interpolation

  Parameters
  ----------
    x: float
        location in the x location of desiered interpolation
    y: float
        location in the y location of desiered interpolation
    iCell: int
        index of the nearest lower left cell
    csz: float
        cellsize of the grid
    ncols: int
      number of columns
    interpOption: int
        0: nearest neighbour interpolation
        1: equal weights interpolation
        2: bilinear interpolation

  Returns
  -------
      w00, w10, w01, w11: floats
          corresponding weights
  """
  cdef double w[4]
  cdef double Lx, Ly, dx, dy
  cdef int Lx0 = iCell % ncols
  cdef int Ly0 = iCell / ncols
  cdef double xllc = 0.
  cdef double yllc = 0.

  # find coordinates in normalized ref (origin (0,0) and cellsize 1)
  Lx = (x - xllc) / csz
  Ly = (y - yllc) / csz

  # prepare for bilinear interpolation
  dx = Lx - Lx0
  dy = Ly - Ly0

  if interpOption == 0:
    dx = 1.*math.round(dx)
    dy = 1.*math.round(dy)
  elif interpOption == 1:
    dx = 1./2.
    dy = 1./2.

  # lower left
  w[0] = (1-dx)*(1-dy)
  # lower right
  w[1] = dx*(1-dy)
  # uper left
  w[2] = (1-dx)*dy
  # and uper right
  w[3] = dx*dy

  return w[0], w[1], w[2], w[3]


cpdef (int, int, int, double, double, double, double) getCellAndWeights(double x, double y, int ncols, int nrows, double csz, int interpOption):
  """ Get cell and weight for interpolation from grid to single point location

  3 Options available : 0: nearest neighbour interpolation
                        1: equal weights interpolation
                        2: bilinear interpolation

  Parameters
  ----------
    x: float
        location in the x location of desiered interpolation
    y: float
        location in the y location of desiered interpolation
    ncols: int
        number of columns
    nrows: int
        number of rows
    csz: float
        cellsize of the grid
    interpOption: int
        0: nearest neighbour interpolation
        1: equal weights interpolation
        2: bilinear interpolation

  Returns
  -------
    Lx0: int
      colomn of the nearest lower left cell
    Ly0: int
      row of the nearest lower left cell
    w: float[4]
      corresponding weights
  """
  cdef int Lx0, Ly0, iCell
  cdef double w[4]
  iCell = getCells(x, y, ncols, nrows, csz)
  w[0], w[1], w[2], w[3] = getWeights(x, y, iCell, csz, ncols, interpOption)
  Lx0 = iCell % ncols
  Ly0 = iCell / ncols
  return Lx0, Ly0, iCell, w[0], w[1], w[2], w[3]


cpdef (double, double, double, double) reprojectVelocity(double uxNew, double uyNew, double uzNew, double nxNew,
                                                         double nyNew, double nzNew, double velMagMin, int keep):
  """ Reproject velocity vector on the topography
  Parameters
  ----------
    uxNew: float
      x component of the velocity vector
    uyNew: float
      y component of the velocity vector
    uzNew: float
      z component of the velocity vector
    nxNew: float
      x component of the normal vector
    nyNew: float
      y component of the normal vector
    nzNew: float
      z component of the normal vector
    velMagMin: float
      velocity magnitude
    keep: int
    if 1 conserve velocity magnitude after the reprojection (does not work if the velocity is normal to the surface)
  """
  cdef double uMag, uN, uMagNew
  # velocity magnitude
  uMag = norm(uxNew, uyNew, uzNew)
  # normal component of the velocity
  uN = scalProd(uxNew, uyNew, uzNew, nxNew, nyNew, nzNew)
  # remove normal component of the velocity
  uxNew = uxNew - uN * nxNew
  uyNew = uyNew - uN * nyNew
  uzNew = uzNew - uN * nzNew
  # velocity magnitude new
  uMagNew = norm(uxNew, uyNew, uzNew)
  if uMag > 0.0 and keep == 1:
    # ensure that velocitity magnitude stays the same also after reprojection onto terrain
    uxNew = uxNew * uMag / (uMagNew + velMagMin)
    uyNew = uyNew * uMag / (uMagNew + velMagMin)
    uzNew = uzNew * uMag / (uMagNew + velMagMin)
  return uxNew, uyNew, uzNew, uMag


cpdef (double, double, int, int, int, double, double, double, double) normalProjectionIteratrive(
  double xOld, double yOld, double zOld, double[:,:] ZDEM, double[:,:] nxArray, double[:,:] nyArray,
  double[:,:] nzArray, double csz, int ncols, int nrows, int interpOption,
  int reprojectionIterations, double threshold):
  """ Find the orthogonal projection of a point on a mesh

  Iterative method to find the projection of a point on a surface defined by its mesh

  Parameters
  ----------
      xOld: float
          x coordinate of the point to project
      yOld: float
          y coordinate of the point to project
      zOld: float
          z coordinate of the point to project
      ZDEM: 2D array
          z component of the DEM field at the grid nodes
      Nx: 2D array
          x component of the normal vector field at the grid nodes
      Ny: 2D array
          y component of the normal vector field at the grid nodes
      Nz: 2D array
          z component of the normal vector field at the grid nodes
      csz: float
          cellsize of the grid
      interpOption: int
          -0: nearest neighbour interpolation
          -1: equal weights interpolation
          -2: bilinear interpolation
      reprojectionIterations: int
          maximum number or iterations
      threshold: double
          stop criterion for reprojection, stops when the distance between two iterations
          is smaller than threshold * csz

    Returns
    -------
    xNew: float
        x coordinate of the projected point
    yNew: float
        y coordinate of the projected point
    iCell: int
        index of the nearest lower left cell
    Lx0: int
        colomn of the nearest lower left cell
    Ly0: int
        row of the nearest lower left cell
    w: float[4]
        corresponding weights
    """
  cdef double zTemp, zn, xNew, yNew, zNew, dist, xPrev, yPrev, zPrev
  cdef double nx, ny, nz
  cdef int Lxi, Lyi, Lxj, Lyj
  cdef int Lx0, Ly0, iCell
  cdef double w[4]
  xNew = xOld
  yNew = yOld
  zNew = zOld
  iCell = getCells(xNew, yNew, ncols, nrows, csz)
  if iCell < 0:
    # if not on the DEM exit with iCell=-1
    return xNew, yNew, iCell, -1, -1, 0, 0, 0, 0
  w[0], w[1], w[2], w[3] = getWeights(xNew, yNew, iCell, csz, ncols, interpOption)
  Lx0 = iCell % ncols
  Ly0 = iCell / ncols
  dist = csz
  # iterate
  while reprojectionIterations > 0 and dist > threshold*csz:
    reprojectionIterations = reprojectionIterations - 1
    xPrev = xNew
    yPrev = yNew
    zPrev = zNew
    # vertical projection of the point
    zTemp = getScalar(Lx0, Ly0, w[0], w[1], w[2], w[3], ZDEM)
    # normal vector at this vertical projection location
    # if we take the normal at the new particle position)
    nx, ny, nz = getVector(Lx0, Ly0, w[0], w[1], w[2], w[3], nxArray, nyArray, nzArray)
    nx, ny, nz = normalize(nx, ny, nz)
    # What I think is better for normal projection
    # Normal component of the vector between the initial and projected point
    zn = (xNew-xOld) * nx + (yNew-yOld) * ny + (zTemp-zOld) * nz
    # correct position with the normal part
    xNew = xOld + zn * nx
    yNew = yOld + zn * ny
    zNew = zOld + zn * nz
    # get cell
    iCell = getCells(xNew, yNew, ncols, nrows, csz)
    if iCell < 0:
      # if not on the DEM exit with iCell=-1
      return xNew, yNew, iCell, -1, -1, 0, 0, 0, 0
    w[0], w[1], w[2], w[3] = getWeights(xNew, yNew, iCell, csz, ncols, interpOption)
    Lx0 = iCell % ncols
    Ly0 = iCell / ncols

    dist = norm(xNew-xPrev, yNew-yPrev, zNew-zPrev)

  return xNew, yNew, iCell, Lx0, Ly0, w[0], w[1], w[2], w[3]


cpdef (double, double, int, int, int, double, double, double, double) samosProjectionIteratrive(
  double xOld, double yOld, double zOld, double[:,:] ZDEM, double[:,:] nxArray, double[:,:] nyArray,
  double[:,:] nzArray, double csz, int ncols, int nrows, int interpOption, int reprojectionIterations):
  """ Find the projection of a point on a mesh (comes from samos)

  Iterative method to find the projection of a point on a surface defined by its mesh (samos way)

  Parameters
  ----------
      xOld: float
          x coordinate of the point to project
      yOld: float
          y coordinate of the point to project
      zOld: float
          z coordinate of the point to project
      ZDEM: 2D array
          z component of the DEM field at the grid nodes
      Nx: 2D array
          x component of the normal vector field at the grid nodes
      Ny: 2D array
          y component of the normal vector field at the grid nodes
      Nz: 2D array
          z component of the normal vector field at the grid nodes
      csz: float
          cellsize of the grid
      interpOption: int
          -0: nearest neighbour interpolation
          -1: equal weights interpolation
          -2: bilinear interpolation
      reprojectionIterations: int
          maximum number or iterations

    Returns
    -------
    xNew: float
        x coordinate of the projected point
    yNew: float
        y coordinate of the projected point
    iCell: int
        index of the nearest lower left cell
    Lx0: int
        colomn of the nearest lower left cell
    Ly0: int
        row of the nearest lower left cell
    w: float[4]
        corresponding weights
    """
  cdef double zTemp, zn, xNew, yNew, zNew
  cdef double nx, ny, nz
  cdef int Lxi, Lyi, Lxj, Lyj
  cdef int iCell, Lx0, Ly0
  cdef double w[4]
  xNew = xOld
  yNew = yOld
  zNew = zOld
  iCell = getCells(xNew, yNew, ncols, nrows, csz)
  if iCell < 0:
    # if not on the DEM exit with iCell=-1
    return xNew, yNew, iCell, -1, -1, 0, 0, 0, 0
  w[0], w[1], w[2], w[3] = getWeights(xNew, yNew, iCell, csz, ncols, interpOption)
  # get the cell location of the point
  Lx0 = iCell % ncols
  Ly0 = iCell / ncols
  Lxi = Lx0
  Lyi = Ly0
  # iterate
  while reprojectionIterations > 0:
    reprojectionIterations = reprojectionIterations - 1
    # vertical projection of the point
    zTemp = getScalar(Lx0, Ly0, w[0], w[1], w[2], w[3], ZDEM)
    # normal vector at this vertical projection location
    # if we take the normal at the new particle position)
    # nx, ny, nz = getVector(Lxy[0], Lxy[1], Lxy[2], Lxy[3], w[0], w[1], w[2], w[3], Nx, Ny, Nz)
    # What Peter does: get the normal of the cell center (but still not exactly giving the same results)
    nx, ny, nz = getVector(Lx0, Ly0, 0.25, 0.25, 0.25, 0.25, nxArray, nyArray, nzArray)
    nx, ny, nz = normalize(nx, ny, nz)
    zn = (xNew-xNew) * nx + (yNew-yNew) * ny + (zTemp-zNew) * nz
    # correct position with the normal part
    xNew = xNew + zn * nx
    yNew = yNew + zn * ny
    zNew = zNew + zn * nz
    # get cell
    iCell = getCells(xNew, yNew, ncols, nrows, csz)
    if iCell < 0:
      # if not on the DEM exit with iCell=-1
      return xNew, yNew, iCell, -1, -1, 0, 0, 0, 0
    w[0], w[1], w[2], w[3] = getWeights(xNew, yNew, iCell, csz, ncols, interpOption)
    Lx0 = iCell % ncols
    Ly0 = iCell / ncols
    Lxj = Lx0
    Lyj = Ly0

    # are we in the same cell?
    if Lxi==Lxj and Lyi==Lyj:
      return xNew, yNew, iCell, Lx0, Ly0, w[0], w[1], w[2], w[3]
    Lxj = Lxi
    Lyj = Lyi
  return xNew, yNew, iCell, Lx0, Ly0, w[0], w[1], w[2], w[3]


cpdef (double, double, double, int, int, int, double, double, double, double) distConservProjectionIteratrive(
  double xPrev, double yPrev, double zPrev, double[:,:] ZDEM, double[:,:] nxArray, double[:,:] nyArray,
  double[:,:] nzArray, double xOld, double yOld, double zOld, double csz, int ncols, int nrows, int interpOption,
  int reprojectionIterations, double threshold):
  """ Find the projection of a point on a mesh conserving the distance
  with the previous time step position

  Iterative method to find the projection of a point on a surface defined by its mesh (conserve distance)

  Parameters
  ----------
      xPrev: float
          x coordinate of the point at the previous time step
      yPrev: float
          y coordinate of the point at the previous time step
      zPrev: float
          z coordinate of the point at the previous time step
      ZDEM: 2D array
          z component of the DEM field at the grid nodes
      Nx: 2D array
          x component of the normal vector field at the grid nodes
      Ny: 2D array
          y component of the normal vector field at the grid nodes
      Nz: 2D array
          z component of the normal vector field at the grid nodes
      xOld: float
          x coordinate of the point to project
      yOld: float
          y coordinate of the point to project
      zOld: float
          z coordinate of the point to project
      csz: float
          cellsize of the grid
      interpOption: int
          -0: nearest neighbour interpolation
          -1: equal weights interpolation
          -2: bilinear interpolation
      reprojectionIterations: int
          maximum number or iterations
      threshold: double
          stop criterion for reprojection, stops when the error on the distance after projection
          is smaller than threshold * (dist + csz)

    Returns
    -------
    xNew: float
        x coordinate of the projected point
    yNew: float
        y coordinate of the projected point
    zNew: float
        z coordinate of the projected point
    iCell: int
        index of the nearest lower left cell
    Lx0: int
        colomn of the nearest lower left cell
    Ly0: int
        row of the nearest lower left cell
    w: float[4]
        corresponding weights
    """
  cdef double zTemp, zn, dist, distn, xNew, yNew, zNew
  cdef double nx, ny, nz
  cdef int Lx0, Ly0, iCell
  cdef double w[4]
  xNew = xOld
  yNew = yOld
  zNew = zOld
  # distance traveled by the point within the time step
  dist = norm(xNew-xPrev, yNew-yPrev, zNew-zPrev)
  # vertical projection of the point on the DEM
  iCell = getCells(xNew, yNew, ncols, nrows, csz)
  if iCell < 0:
    # if not on the DEM exit with iCell=-1
    return xNew, yNew, zNew, iCell, -1, -1, 0, 0, 0, 0
  w[0], w[1], w[2], w[3] = getWeights(xNew, yNew, iCell, csz, ncols, interpOption)
  Lx0 = iCell % ncols
  Ly0 = iCell / ncols

  zTemp = getScalar(Lx0, Ly0, w[0], w[1], w[2], w[3], ZDEM)
  # measure distance between the projected point and the previous time step position
  distn = norm(xNew-xPrev, yNew-yPrev, zTemp-zPrev)
  # while iterate and error too big (aiming to conserve the distance dist during the reprojection)
  while reprojectionIterations > 0 and abs(distn-dist) > threshold*(dist + csz):
    reprojectionIterations = reprojectionIterations - 1
    # first step: orthogonal reprojection
    # get normal vector
    nx, ny, nz = getVector(Lx0, Ly0, w[0], w[1], w[2], w[3], nxArray, nyArray, nzArray)
    nx, ny, nz = normalize(nx, ny, nz)
    zn = (xNew-xNew) * nx + (yNew-yNew) * ny + (zTemp-zNew) * nz
    # correct position with the normal part
    xNew = xNew + zn * nx
    yNew = yNew + zn * ny
    zNew = zNew + zn * nz

    # second step: conserve the distance dist (correction ov the position)
    # measure distance between this new point and the previous time step position
    distn = norm(xNew-xPrev, yNew-yPrev, zNew-zPrev)
    # guard against stopped particle (distn == 0)
    if distn == 0:
      break
    # adjust position on the Xprev, Xnew line
    xNew = xPrev + (xNew-xPrev) * dist / distn
    yNew = yPrev + (yNew-yPrev) * dist / distn
    zNew = zPrev + (zNew-zPrev) * dist / distn

    # third step
    # vertical projection of the point on the DEM
    iCell = getCells(xNew, yNew, ncols, nrows, csz)
    if iCell < 0:
      # if not on the DEM exit with iCell=-1
      return xNew, yNew, zNew, iCell, -1, -1, 0, 0, 0, 0
    w[0], w[1], w[2], w[3] = getWeights(xNew, yNew, iCell, csz, ncols, interpOption)
    Lx0 = iCell % ncols
    Ly0 = iCell / ncols
    zTemp = getScalar(Lx0, Ly0, w[0], w[1], w[2], w[3], ZDEM)
    # measure distance between this new point and the previous time step position
    distn = norm(xNew-xPrev, yNew-yPrev, zTemp-zPrev)

  return xNew, yNew, zTemp, iCell, Lx0, Ly0, w[0], w[1], w[2], w[3]


cpdef double[:] projOnRaster(double[:] xArray, double[:] yArray, double[:, :] vArray, double csz, int ncols,
                 int nrows, int interpOption):
  """ Interpolate vector field from grid to points

  Parameters
  ----------
    xArray: 1D float array
        x coordinate of the point at which to interpolate
    yArray: 1D float array
        y coordinate of the point at which to interpolate
    vArray: 2D float array
        raster values of the field to interpolate
    csz: float
        raster cell size
    ncols: int
        raster number of columns
    nrows: int
        raster number of rows

  Returns
  -------
      v: 1D float array
          interpolated scalar at position (xArray, yArray)
  """
  cdef int N = xArray.shape[0]
  cdef double x, y
  cdef int Lx0, Ly0, iCell
  cdef double w[4]
  cdef int k
  cdef double[:] v = np.zeros(N)
  for k in range(N):
    x = xArray[k]
    y = yArray[k]

    Lx0, Ly0, iCell, w[0], w[1], w[2], w[3] = getCellAndWeights(x, y, ncols, nrows, csz, interpOption)

    v[k] = getScalar(Lx0, Ly0, w[0], w[1], w[2], w[3], vArray)

  return v


cpdef double getScalar(int Lx0, int Ly0, double w0, double w1, double w2, double w3, double[:, :] V):
  """ Interpolate scalar field from grid to single point location

  Originaly created to get the normal vector at location (x,y) given the
  normal vector field on the grid. Grid has its origin in (0,0).
  Can be used to interpolate any vector field.
  Interpolation using a bilinear interpolation

  Parameters
  ----------
    Lx0: int
        colomn of the nearest lower left cell
    Ly0: int
        row of the nearest lower left cell
    w: float[4]
        corresponding weights
    V: 2D numpy array
        scalar field at the grid nodes

  Returns
  -------
      v: float
          interpolated scalar at position (x, y)
  """
  cdef double v = (V[Ly0, Lx0]*w0 +
                   V[Ly0, Lx0+1]*w1 +
                   V[Ly0+1, Lx0]*w2 +
                   V[Ly0+1, Lx0+1]*w3)


  return v


cpdef (double, double, double) getVector(
  int Lx0, int Ly0, double w0, double w1, double w2, double w3,
  double[:, :] Nx, double[:, :] Ny, double[:, :] Nz):
  """ Interpolate vector field from grid to single point location

  Originally created to get the normal vector at location (x,y) given the
  normal vector field on the grid. Grid has its origin in (0,0).
  Can be used to interpolate any vector field.

  Parameters
  ----------
      Lx0: int
          colomn of the nearest lower left cell
      Ly0: int
          row of the nearest lower left cell
      w: float[4]
          corresponding weights
          location in the y location of desiered interpolation
      Nx: 2D numpy array
          x component of the vector field at the grid nodes
      Ny: 2D numpy array
          y component of the vector field at the grid nodes
      Nz: 2D numpy array
          z component of the vector field at the grid nodes

  Returns
  -------
      nx: float
          x component of the interpolated vector field at position (x, y)
      ny: float
          y component of the interpolated vector field at position (x, y)
      nz: float
          z component of the interpolated vector field at position (x, y)
  """
  cdef double nx = (Nx[Ly0, Lx0]*w0 +
                   Nx[Ly0, Lx0+1]*w1 +
                   Nx[Ly0+1, Lx0]*w2 +
                   Nx[Ly0+1, Lx0+1]*w3)
  cdef double ny = (Ny[Ly0, Lx0]*w0 +
                   Ny[Ly0, Lx0+1]*w1 +
                   Ny[Ly0+1, Lx0]*w2 +
                   Ny[Ly0+1, Lx0+1]*w3)
  cdef double nz = (Nz[Ly0, Lx0]*w0 +
                   Nz[Ly0, Lx0+1]*w1 +
                   Nz[Ly0+1, Lx0]*w2 +
                   Nz[Ly0+1, Lx0+1]*w3)
  return nx, ny, nz


cpdef double SamosATfric(double rho, double tau0, double Rs0, double mu, double kappa, double B, double R,
                         double v, double p, double h):
  """ Get tau (basal friction stress) for samos friction type

  Parameters
  ----------
      rho: float
          density
      tau0: float

      Rs0: float

      mu: float
          coulomb friction coefficient
      kappa: float

      B: float

      R: float

      v: float
          flow velocity
      p: float
          basal normal stress
      h: float
          floa thickness

  Returns
  -------
      tau: float
          basal shear stress
  """
  cdef double Rs = rho * v * v / (p + 0.001)
  cdef double div = h / R
  if div < 1.0:
    div = 1.0
  div = math.log(div) / kappa + B
  cdef double tau = tau0 + p * mu * (1.0 + Rs0 / (Rs0 + Rs)) + rho * v * v / (div * div)
  return tau
