#!python
# cython: boundscheck=False, wraparound=False, cdivision=True
"""
    function related to SPH calculations in com1DFA
    to build: go to repository containing this file and run:
    python setup.py build_ext --inplace
"""

cpdef double norm(double , double , double)

cpdef double norm2(double , double , double)

cpdef (double, double, double) normalize(double , double , double)

cpdef (double, double, double) crossProd(double , double , double, double , double , double)

cpdef double scalProd(double , double , double, double , double , double)

cpdef (int) getCells(double, double, int, int, double)

cpdef (double, double, double, double) getWeights(double, double, int, double, int, int)

cpdef (int, int, int, double, double, double, double) getCellAndWeights(double, double, int, int, double, int)

cpdef (double, double, double, double) reprojectVelocity(double , double , double, double,
                                                         double , double , double, int)

cpdef (double, double, int, int, int, double, double, double, double) normalProjectionIteratrive(
  double , double , double, double[:,:], double[:,:], double[:,:],
  double[:,:], double, int, int, int,
  int, double)

cpdef (double, double, int, int, int, double, double, double, double) samosProjectionIteratrive(
  double , double , double, double[:,:], double[:,:], double[:,:],
  double[:,:], double csz, int, int, int, int)

cpdef (double, double, double, int, int, int, double, double, double, double) distConservProjectionIteratrive(
  double , double , double, double[:,:], double[:,:], double[:,:],
  double[:,:], double , double , double, double, int, int, int,
  int, double)

cpdef double[:] projOnRaster(double[:], double[:], double[:, :], double, int,
                 int, int)

cpdef double getScalar(int, int, double , double , double, double, double[:, :])

cpdef (double, double, double) getVector(
  int, int, double , double , double, double,
  double[:, :], double[:, :], double[:, :])

cpdef double SamosATfric(double , double , double, double , double , double, double , double , double, double)

cpdef double hysteresisMu(double, double, double, double, double, double, double, double, double)
