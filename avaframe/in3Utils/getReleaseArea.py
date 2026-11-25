"""
    Get release area corner coordinates for a rectangle of an area of approx. 100 000 m2

    This file is part of Avaframe.
"""

# Load modules
import numpy as np
import matplotlib.pyplot as plt
from avaframe.in3Utils import generateTopo as gT
import shapefile
import logging
import shutil
import pathlib

# create local logger
# change log level in calling module to DEBUG to see log messages
log = logging.getLogger(__name__)


def makexyPoints(x1, x2, y1, cfgR):
    """ Make xy Points of release area from given extent and start and end points """

    lenp = int(cfgR['GENERAL']['lenP'])

    # define corner coordinates
    xyPoints = np.zeros((lenp, 2))
    xyPoints[0, 0] = x1 + x2          # xlr
    xyPoints[0, 1] = -0.5 * y1        # ylr
    xyPoints[1, 0] = x1 + x2          # xur
    xyPoints[1, 1] = 0.5 * y1         # yur
    xyPoints[2, 0] = x1               # xul
    xyPoints[2, 1] = 0.5 * y1         # yul
    xyPoints[3, 0] = x1               # xll
    xyPoints[3, 1] = -0.5 * y1        # yll

    return xyPoints


def getCornersFP(cfgR):

    # Terrain parameters
    hr = cfgR['GENERAL'].getfloat('hr')
    vol = cfgR['GENERAL'].getfloat('vol')
    dh = cfgR['GENERAL'].getfloat('dh')
    xStart = cfgR['GENERAL'].getfloat('xStart')

    # Compute release area ---------------------------------------------------------------------------------------
    xr = cfgR['FP'].getfloat('xExtent')
    yr = vol / (xr * dh)

    log.info('volume is: %f' % (yr*xr*dh))

    # make corner coordinates of release area
    xyPoints = makexyPoints(xStart, xr, yr, cfgR)

    return xyPoints


def getCornersIP(cfgR, cfgT):

    # Terrain parameters
    hr = cfgR['GENERAL'].getfloat('hr')
    vol = cfgR['GENERAL'].getfloat('vol')
    dh = cfgR['GENERAL'].getfloat('dh')
    xStart = cfgR['GENERAL'].getfloat('xStart')
    meanAlpha = cfgT['TOPO'].getfloat('meanAlpha')

    # Compute release area ---------------------------------------------------------------------------------------
    xr = hr / np.sin(np.radians(meanAlpha))        # along valley extent of release area
    xrp = hr / np.tan(np.radians(meanAlpha))       # projected to the flat plane
    yr = vol / (xr * dh)

    log.info('volume is: %f, xr: %f, xrp: %f' % (yr*xr*dh, xr, xrp))

    # make corner coordinates of release area
    xyPoints = makexyPoints(xStart, xrp, yr, cfgR)

    return xyPoints


def getCornersHS(cfgR, cfgT):

    # Terrain parameters
    hr = cfgR['GENERAL'].getfloat('hr')
    vol = cfgR['GENERAL'].getfloat('vol')
    dh = cfgR['GENERAL'].getfloat('dh')
    lenp = int(cfgR['GENERAL']['lenP'])
    alphaStop = cfgR['HS'].getfloat('alphaStop')

    # get A, B from HS
    [A, B, fLen] = gT.getParabolaParams(cfgT)

    # Compute release area ---------------------------------------------------------------------------------------
    # along valley margin of release area at alpha_stop° point
    xStop = (np.tan(np.radians(-alphaStop)) - B) / (2. * A)
    xr = hr / np.sin(np.radians(alphaStop))
    xrp = hr / np.tan(np.radians(alphaStop))
    yr = vol / (xr * dh)

    xStart = xStop - xrp
    # make corner coordinates of release area
    xyPoints = makexyPoints(xStart, xrp, yr, cfgR)

    return xyPoints


def correctOrigin(xyPoints, cfgT):
    """ Move the points so that the correct origin is set """

    # determine number of rows and columns to define domain
    dx = cfgT['TOPO'].getfloat('dx')
    xEnd = cfgT['TOPO'].getfloat('xEnd')
    yEnd = cfgT['TOPO'].getfloat('yEnd')

    # Compute coordinate grid
    xv = np.arange(0, xEnd+dx, dx)
    yv = np.arange(-0.5 * yEnd, 0.5 * (yEnd+dx), dx)

    xl = cfgT['DEMDATA'].getfloat('xl')
    yl = cfgT['DEMDATA'].getfloat('yl')

    xv = xv + xl
    yv = yv + yl + 0.5 * yEnd

    xyPoints[:, 0] = xyPoints[:, 0] + xl
    xyPoints[:, 1] = xyPoints[:, 1] + yl + 0.5 * yEnd

    # list all points
    for m in range(len(xyPoints)):
        log.info('Point %d: x %f y %f' % (m, xyPoints[m, 0], xyPoints[m, 1]))

    return xv, yv, xyPoints


def writeReleaseArea(xyPoints, demType, cfgR, outDir):
    """ Write topography information to file """

    # add enpoint=startpoint to close polygon
    xyPoints = np.append(xyPoints, [xyPoints[0,:]], axis=0)

    lenp = len(xyPoints)
    relNo = int(cfgR['FILE']['relNo'])
    relH = cfgR['GENERAL'].getfloat('dh')
    relName = cfgR['FILE']['relName']

    # Add vertical coordinate
    z = np.zeros((lenp, 1))
    p_mat = np.asarray(np.append(xyPoints, z, axis=1))

    # Save elevation data to .asc file and add header lines
    releaseFile = outDir / ('release%d%s.nxyz' % (relNo, demType))
    with open(releaseFile, 'w') as f:
        f.write('name=%s\n' % (relName))
        f.write('d0=%.2f\n' % (relH))
        f.write('rho=None\n')
        f.write('%d\n' % (lenp))
        np.savetxt(f, p_mat, fmt='%f')

    # Log info here
    log.info('Release Area written to: %s/release_%d%s as .nxyz and .shp' % (outDir, relNo, demType))
    if cfgR.getboolean('GENERAL', 'outputtxt'):
        shutil.copyfile((outDir / ('release%d%s.nxyz' % (relNo, demType))),
                        (outDir / ('release%d%s.txt' % (relNo, demType))))

    # Make list of Points
    xy = []
    for m in range(len(xyPoints)):
        xy.append([xyPoints[m, 0], xyPoints[m, 1]])

    # Write shp file - include endpoint =startpoint for polygon!
    releaseFileName = outDir / ('release%d%s' % (relNo, demType))
    w = shapefile.Writer(str(releaseFileName))
    w.poly([[xy[0], xy[1], xy[2], xy[3], xy[4]]])
    w.field('ID', 'C', '40')
    w.field('Name', 'C', '40')
    w.field('thickness', 'N', decimal=4)
    w.record('1', 'Rel_Example', relH)
    w.close()


def getReleaseArea(cfgT, cfgR, avalancheDir):
    """ Main function to compute release areas """

    # Which DEM type
    demType = cfgT['TOPO']['demType']

    log.info('DEM type is set to: %s' % demType)

    # Set Output directory
    outDir = pathlib.Path(avalancheDir, 'Inputs', 'REL')
    if outDir.is_dir():
        log.info('The new release area is saved to %s' % (outDir))
    else:
        log.error('Required folder structure: NameOfAvalanche/Inputs missing! \
                    Run runInitializeProject first!')

    flagCont = False
    # Get release area
    if demType == 'FP':
        xyPoints = getCornersFP(cfgR)
        flagCont = True

    elif demType == 'IP':
        xyPoints = getCornersIP(cfgR, cfgT)
        flagCont = True

    elif demType == 'PF':
        xyPoints = getCornersHS(cfgR, cfgT)
        flagCont = True

    elif demType == 'HS':
        xyPoints = getCornersIP(cfgR, cfgT)
        flagCont = True

    elif demType == 'HX':
        log.info('no release area available for demType: %s' % (demType))

    elif demType == 'BL':
        log.warning('no release area available for demType: %s' % (demType))

    if flagCont:

        # Move to correct correctOrigin
        [xv, yv, xyPoints] = correctOrigin(xyPoints, cfgT)

        # Write release area
        writeReleaseArea(xyPoints, demType, cfgR, outDir)

    return xv, yv, xyPoints
