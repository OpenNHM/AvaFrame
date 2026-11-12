""" Flat plane test

"""

# imports
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
import pathlib

# local imports
import avaframe.com1DFA.DFAtools as DFAtls
import avaframe.com1DFA.DFAfunctionsCython as DFAfunC
import avaframe.com1DFA.DFAToolsCython as DFAtlsC
import avaframe.in2Trans.rasterUtils as IOf
import avaframe.out3Plot.plotUtils as pU
import avaframe.in3Utils.fileHandlerUtils as fU


# create local logger
# change log level in calling module to DEBUG to see log messages
log = logging.getLogger(__name__)


def getReleaseThickness(avaDir, cfg, demFile):
    """ define release thickness for Flat Plane solution test """

    # Read dem
    demOri = IOf.readRaster(demFile)
    nrows = demOri['header']['nrows']
    ncols = demOri['header']['ncols']
    xllc = demOri['header']['xllcenter']
    yllc = demOri['header']['yllcenter']
    csz = demOri['header']['cellsize']

    # define release thickness distribution
    cfgFP = cfg['FPSOL']
    H0 = float(cfgFP['H0'])
    deltaX = float(cfgFP['deltaX'])
    slope = float(cfgFP['slope'])
    x = np.linspace(0, ncols-1, ncols)*csz+xllc
    y = np.linspace(0, nrows-1, nrows)*csz+yllc
    X, Y = np.meshgrid(x, y)
    r = np.sqrt((X*X)+(Y*Y))
    relTh = H0 - (r-deltaX)*slope
    relTh = np.where(relTh < 0, 0, relTh)
    relTh = np.where(relTh > H0, H0, relTh)

    relDict = {'relTh': relTh, 'demOri': demOri, 'X': X, 'Y': Y}

    # save release thickness field to file
    relThPath = pathlib.Path(avaDir, 'Inputs', 'RELTH')
    fU.makeADir(relThPath)
    relThFile = relThPath / 'releaseThickness.asc'
    IOf.writeResultToRaster(demOri['header'], relTh, relThFile, flip=True)

    return relDict


def postProcessFPcom1DFA(cfgGen, particles, fields, ind_t, relDict):
    """ get fields and particles dictionaries for given time step """

    demOri = relDict['demOri']
    nrows = demOri['header']['nrows']
    ncols = demOri['header']['ncols']
    xllc = demOri['header']['xllcenter']
    yllc = demOri['header']['yllcenter']
    csz = demOri['header']['cellsize']
    dem = relDict['dem']

    x = particles['x']
    y = particles['y']
    ux = particles['ux']
    uy = particles['uy']
    m = particles['m']
    h = particles['h']
    force2 = {}
    particles, force2 = DFAfunC.computeForceSPHC(cfgGen, particles, force2, dem,
        cfgGen.getint('sphOption'), gradient=1)
    gradNorm = DFAtls.norm(force2['forceSPHX'], force2['forceSPHY'], force2['forceSPHZ'])
    x1, y1, z1, = DFAtls.normalize(x+xllc, y+yllc, 0*x)
    uMag = DFAtls.norm(ux, uy, 0)
    v = DFAtls.scalProd(ux, uy, 0, x1, y1, z1)
    grad = DFAtls.scalProd(force2['forceSPHX'], force2['forceSPHY'], force2['forceSPHZ'], x1, y1, z1)
    Grad = np.zeros((nrows, ncols))
    MassBilinear = np.zeros((nrows, ncols))
    MassBilinear = DFAtlsC.pointsToRasterC(x, y, m, MassBilinear, csz=5)
    Grad = DFAtlsC.pointsToRasterC(x, y, m*gradNorm, Grad, csz=5)
    indMass = np.where(MassBilinear > 0)
    Grad[indMass] = Grad[indMass]/MassBilinear[indMass]
    x = particles['x']+xllc
    y = particles['y']+yllc
    r = np.sqrt(x*x + y*y)
    com1DFASol = {'x': x, 'y': y, 'r': r, 'h': h, 'v': v,
                    'gradNorm': gradNorm,
                    'grad': grad, 'Grad': Grad, 'uMag': uMag, 'fields': fields}

    return com1DFASol


def plotProfilesFPtest(cfg, ind_time, relDict, comSol):
    """ Plot flow thickness and gradient for FlatPlane simulation results

        Parameters
        -----------
        cfg: configparser
        ind_time: int
            time index for simiSol
        relDict: dict
            dictionary of release area info
        comSol: dict
            dictionary of simulation results and info (particles, fields, indices, time step)

    """
    cfgGen = cfg['GENERAL']
    mu = cfgGen.getfloat('mu' + cfgGen['frictModel'].lower())
    cfgFP = cfg['FPSOL']
    H0 = float(cfgFP['H0'])
    deltaX = float(cfgFP['deltaX'])
    slope = float(cfgFP['slope'])

    # get info from dem
    demOri = relDict['demOri']
    relTh = relDict['relTh']
    ncols = demOri['header']['ncols']
    nrows = demOri['header']['nrows']
    xllc = demOri['header']['xllcenter']
    yllc = demOri['header']['yllcenter']
    csz = demOri['header']['cellsize']

    # com1DFA results
    fields = comSol['fields']
    x = comSol['x']
    y = comSol['y']
    r = comSol['r']
    h = comSol['h']
    v = comSol['v']
    gradNorm = comSol['gradNorm']
    v = comSol['v']
    outDirTest = comSol['outDirTest']
    showPlot = comSol['showPlot']
    Tsave = comSol['Tsave']

    fig = plt.figure(figsize=(pU.figW, pU.figH))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.plot(np.linspace(xllc, xllc+(ncols-1)*csz, ncols), fields['FT'][100,:], '--b', label='field flow thickness')
    ax1.plot(r, h, color='b', marker='.', linestyle='None', label='particle flow thickness')

    # ax2.plot(r, grad, color='b', marker='.', linestyle='None')
    ax2.plot(r, gradNorm, color='k', marker='o', linestyle='None', label='SPH gradient used')
    # ax2.plot(r, v, color='b', marker='.', linestyle='None')

    ax1.plot(np.linspace(xllc, xllc+(ncols-1)*csz, ncols), relTh[100, :], '--k')
    ax1.plot(r, H0-mu*(r-deltaX), '-k', label='initial expected flow thickness')
    ax1.set_xlabel('r in (m)')
    ax1.set_title('flow thickness, t=%.2f s' % (Tsave))

    ax2.plot(r, mu*np.ones(np.shape(r)), '-k', label='friction threashold')
    ax2.set_xlabel('r in (m)')
    ax2.set_title('Gradient of the flow thickness')
    ax1.legend()
    ax2.legend()

    fig.savefig(os.path.join(outDirTest, 'radialCutSol.%s' % (pU.outputFormat)))

    if showPlot:
        plt.show()
    else:
        plt.close(fig)
