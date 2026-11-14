#!python
# cython: boundscheck=False, wraparound=False, cdivision=True
"""
    function related to SPH calculations in com1DFA
    to build: go to repository containing this file and run:
    python setup.py build_ext --inplace
"""

cpdef (double, double) computeEntMassAndForce(double, double, double, double, double, double, double)

cpdef (double, double, double) computeEntMassAndForceTjem(double, double, double, double, double, double, double)

cpdef double computeDetMass(double, double, double, double)

cpdef double computeResForce(double, double, double, double, int, int)

cdef (double, double, double) addArtificialViscosity(double, double, double, double, double, double, double, double,
                                                     int, int, double, double, double, double,
                                                     double[:, :], double[:, :], double[:, :],  double, double, double)

cpdef (double, double, double, double) account4FrictionForce(double, double, double, double, double, double, double,
                                                             int)

cpdef double computePressure(double, double)
