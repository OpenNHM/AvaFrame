"""
Energy line test
This module runs a DFA simulation extracts the center of mass path
and compares it to the analytic geometric/alpha line solution
"""
import pathlib
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import FormatStrFormatter
import matplotlib.patheffects as pe

# Local imports
# import computation modules
from avaframe.com1DFA import com1DFA

# import analysis modules
import avaframe.ana5Utils.DFAPathGeneration as DFAPath

# import plotting tools
import avaframe.out3Plot.plotUtils as pU
import avaframe.out3Plot.outCom1DFA as outCom1DFA

# create local logger
log = logging.getLogger(__name__)


def mainEnergyLineTest(avalancheDir, energyLineTestCfg, com1DFACfg, simName, dem):
    """This is the core function of the energyLineTest module
    This module extracts the center of mass path from a DFA simulation
    and compares it to he analytic geometric/alpha line solution
    """
    log.info('Energy line test for simulation: %s' % simName)
    pathFromPart = energyLineTestCfg['energyLineTest'].getboolean('pathFromPart')
    avaProfileMass, particlesIni = DFAPath.generateMassAveragePath(avalancheDir, pathFromPart, simName, dem,
                                                                   addVelocityInfo=True)

    # read pfv for plot
    fieldsList, fieldHeader, timeList = com1DFA.readFields(avalancheDir, ['pfv'], simName=simName, flagAvaDir=True,
                                                           comModule='com1DFA')
    # make analysis and generate plots
    resultEnergyTest, savePath = generateCom1DFAEnergyPlot(avalancheDir, energyLineTestCfg, com1DFACfg, avaProfileMass,
                                                           dem, fieldsList, simName)
    return resultEnergyTest, savePath


def generateCom1DFAEnergyPlot(avalancheDir, energyLineTestCfg, com1DFACfg, avaProfileMass, dem, fieldsList, simName):
    """ Make energy test analysis and plot results

    Parameters
    -----------
    avalancheDir: pathlib
        avalanche directory pathlib path
    energyLineTestCfg: configParser
        energy line test config
    com1DFACfg: configParser
        com1DFA  config
    avaProfileMass: dict
        particle mass averaged properties
    dem: dict
        com1DFA simulation dictionary
    fieldsList: list
        field dictionary list
    simName: str
        simulation name

    """
    plotScor = energyLineTestCfg['energyLineTest'].getboolean('plotScor')
    # read physical parameters from DFA configuration
    g = com1DFACfg['GENERAL'].getfloat('gravAcc')
    mu = com1DFACfg['GENERAL'].getfloat('mu' + com1DFACfg['GENERAL']['frictModel'].lower())
    alphaRad = np.arctan(mu)
    alphaDeg = np.rad2deg(alphaRad)
    csz = dem['header']['cellsize']
    # compute simulation run out angle
    runOutAngleRad, runOutAngleDeg = getRunOutAngle(avaProfileMass)
    # extend path profile and find intersection between the alpha line and the profile
    slopeExt, sIntersection, zIntersection, coefExt = getAlphaProfileIntersection(energyLineTestCfg, avaProfileMass,
                                                                                  mu, csz)
    # compute errors on runout and veloctity altitude
    zEne, u2Path, sGeomL, zGeomL, resultEnergyTest = getEnergyInfo(avaProfileMass, g, mu, sIntersection, zIntersection,
                                                                   runOutAngleDeg, alphaDeg)
    z0 = avaProfileMass['z'][0]
    # create figures and plots
    fig = plt.figure(figsize=(pU.figW*3, pU.figH*2))
    cmap, _, ticks, norm = pU.makeColorMap(pU.colorMaps['pfv'], np.amin(u2Path/(2*g)), np.amax(u2Path/(2*g)),
                                           continuous=pU.contCmap)
    cmap.set_under(color='w')
    # make the bird view plot
    ax1 = plt.subplot2grid((2, 2), (1, 0))
    rowsMin, rowsMax, colsMin, colsMax = pU.constrainPlotsToData(fieldsList[-1]['pfv'], 5, extentOption=True,
                                                                 constrainedData=False, buffer='')
    ax1, extent, cbar0, cs1 = outCom1DFA.addResult2Plot(ax1, dem['header'], fieldsList[-1]['pfv'], 'pfv')
    cbar0.ax.set_ylabel('peak flow velocity')

    # add DEM hillshade with contour lines
    ax1 = outCom1DFA.addDem2Plot(ax1, dem, what='hillshade', extent=extent)
    ax1.plot(avaProfileMass['x'], avaProfileMass['y'], '-y.', zorder=20, label='Center of mass path',
             lw=1, path_effects=[pe.Stroke(linewidth=2, foreground='k'), pe.Normal()])
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.axis('equal')
    ax1.set_ylim([rowsMin, rowsMax])
    ax1.set_xlim([colsMin, colsMax])
    l1 = ax1.legend(loc='upper right')
    l1.set_zorder(40)
    pU.putAvaNameOnPlot(ax1, avalancheDir)

    # make profile plot, zoomed out
    ax2 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    # plot mass averaged center of mass
    ax2.plot(avaProfileMass['s'], avaProfileMass['z'], '-y.', label='Center of mass altitude',
             lw=1, path_effects=[pe.Stroke(linewidth=2, foreground='k'), pe.Normal()])
    if plotScor:
        ax2.plot(avaProfileMass['sCor'], avaProfileMass['z'], '--k.', label='Center of mass altitude (corrected s)')
    # extend this curve towards the bottom using a linear regression on the last x points
    ax2.plot(avaProfileMass['s'][-1]*np.array([1, 1+coefExt]), avaProfileMass['z'][-1] +
             slopeExt*avaProfileMass['s'][-1]*np.array([0, coefExt]), ':k',
             label='Center of mass altitude extrapolation')

    # add center of mass velocity points and runout line
    ax2.plot(avaProfileMass['s'][[0, -1]], zEne[[0, -1]], '-r', label='com1dfa energy line (%.2f°)' % runOutAngleDeg)
    scat = ax2.scatter(avaProfileMass['s'], zEne, marker='s', cmap=cmap, s=8*pU.ms, c=u2Path/(2*g),
                       label='Center of mass velocity altitude')
    cbar2 = ax2.figure.colorbar(scat, ax=ax2, use_gridspec=True)
    cbar2.ax.set_title('(' + pU.cfgPlotUtils['unitFT'] + ')', pad=10)
    cbar2.ax.set_ylabel('Center of mass velocity altitude')

    # add alpha line
    ax2.plot(sGeomL, zGeomL, 'b-', label=r'$\alpha$ line (%.2f°)' % alphaDeg)
    if energyLineTestCfg['energyLineTest'].getboolean('shiftedAlphaLine'):
        ax2.plot(avaProfileMass['s'], avaProfileMass['z'][-1] - (avaProfileMass['s']-avaProfileMass['s'][-1]) * mu,
                 'b-.', label=r'shifted $\alpha$ line (%.2f°)' % alphaDeg)
    zLim = ax2.get_ylim()
    sLim = ax2.get_xlim()
    zMin = zLim[0]
    ax2.vlines(x=avaProfileMass['s'][-1], ymin=zMin, ymax=avaProfileMass['z'][-1],
               color='r', linestyle='--')
    ax2.vlines(x=sIntersection, color='b', ymin=zMin, ymax=zIntersection, linestyle='--')

    ax2.hlines(y=avaProfileMass['z'][-1], xmin=0, xmax=avaProfileMass['s'][-1],
               color='r', linestyle='--')
    ax2.hlines(y=zIntersection, color='b', xmin=0, xmax=sIntersection, linestyle='--')

    ax2.set_xlabel('s (m)')
    ax2.set_ylabel('z (m)')
    ax2.set_xlim(sLim)
    ax2.set_ylim(zLim)
    ax2.legend(loc='upper right')
    ax2.set_title('Energy line test')

    # make profile plot, zoomed in
    ax3 = plt.subplot2grid((2, 2), (1, 1))
    # plot mass averaged center of mass
    ax3.plot(avaProfileMass['s'], avaProfileMass['z']-z0, '-y.', label='Center of mass altitude',
             lw=1, path_effects=[pe.Stroke(linewidth=2, foreground='k'), pe.Normal()])
    if plotScor:
        ax3.plot(avaProfileMass['sCor'], avaProfileMass['z'], '--k.', label='Center of mass altitude (corrected s)')
    # extend this curve towards the bottom using a linear gegression on the last x points
    ax3.plot(avaProfileMass['s'][-1]*np.array([1, 1+coefExt]), avaProfileMass['z'][-1] +
             slopeExt*avaProfileMass['s'][-1]*np.array([0, coefExt])-z0, ':k',
             label='Center of mass altitude extrapolation')

    # add center of mass velocity points and runout line
    ax3.plot(avaProfileMass['s'][[0, -1]], zEne[[0, -1]]-z0, '-r',
             label='com1dfa energy line (%.2f°)' % runOutAngleDeg)
    scat = ax3.scatter(avaProfileMass['s'], zEne-z0, marker='s', cmap=cmap, s=8*pU.ms, c=u2Path/(2*g),
                       label='Center of mass velocity altitude extrapolation')
    cbar3 = ax3.figure.colorbar(scat, ax=ax3, use_gridspec=True)
    cbar3.ax.set_title('(' + pU.cfgPlotUtils['unitFT'] + ')', pad=10)
    cbar3.ax.set_ylabel('Center of mass velocity altitude')

    # add alpha line
    ax3.plot(sGeomL, zGeomL-z0, 'b-', label=r'$\alpha$ line (%.2f°)' % alphaDeg)
    if energyLineTestCfg['energyLineTest'].getboolean('shiftedAlphaLine'):
        ax3.plot(avaProfileMass['s'], avaProfileMass['z'][-1]-z0 - (avaProfileMass['s']-avaProfileMass['s'][-1]) * mu,
                 'b-.', label=r'shifted $\alpha$ line (%.2f°)' % alphaDeg)

    # add horizontal line at the final mass averaged position
    # compute plot limits
    runOutSError = resultEnergyTest['runOutSError']
    runOutZError = resultEnergyTest['runOutZError']
    runOutAngleError = resultEnergyTest['runOutAngleError']
    rmseVelocityElevation = resultEnergyTest['rmseVelocityElevation']
    errorS = abs(runOutSError)
    errorZ = abs(runOutZError)
    sMin = min(avaProfileMass['s'][-1], sIntersection) - max(errorS, 0)
    sMax = max(avaProfileMass['s'][-1], sIntersection) + max(errorS, 0)
    zMin = avaProfileMass['z'][-1] + min(slopeExt*(sMax - avaProfileMass['s'][-1]), 0 - 2*errorZ)-z0
    zMax = avaProfileMass['z'][0] - sMin*np.tan(min(runOutAngleRad, alphaRad))-z0
    if avaProfileMass['z'][-1] == zIntersection:
        zMin = zMin - (zMax-zMin)*0.1
    if errorZ < 1e-3:
        zMin = zMin - (zMax-zMin)*0.1
    ax3.vlines(x=avaProfileMass['s'][-1], ymin=zMin, ymax=avaProfileMass['z'][-1]-z0,
               color='r', linestyle='--')
    ax3.vlines(x=sIntersection, color='b', ymin=zMin, ymax=zIntersection-z0, linestyle='--')

    ax3.hlines(y=avaProfileMass['z'][-1]-z0, xmin=sMin, xmax=avaProfileMass['s'][-1],
               color='r', linestyle='--')
    ax3.hlines(y=zIntersection-z0, color='b', xmin=sMin, xmax=sIntersection, linestyle='--')

    ax3.set_xlabel('s (m)')
    ax3.set_ylabel('$\Delta z$ (m)')
    ax3.yaxis.set_label_coords(0, 0.9)
    ax3.set_xlim([sMin, sMax])
    ax3.set_ylim([zMin, zMax])
    ax3.tick_params(axis='x', which='major', rotation=45)
    ax3.tick_params(axis='y', which='major', rotation=45)
    ax3.set_xticks([avaProfileMass['s'][-1], sIntersection])
    ax3.set_yticks([avaProfileMass['z'][-1]-z0, zIntersection-z0])
    ax3.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax3.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    text = ('Runout s diff : %.4e m \nRunout z diff : %.4e m \nRunout angle diff : %.4e° \nvelocity height rmse : %.4e m \n(energy line - alpha line)' %
            (runOutSError, runOutZError, runOutAngleError, rmseVelocityElevation))
    text_box = AnchoredText(text, frameon=False, loc=1, pad=0.5,
                            prop=dict(fontsize=pU.fs))
    log.info(text)
    plt.setp(text_box.patch, facecolor='white', alpha=0.5)
    ax3.add_artist(text_box)
    outFileName = '_'.join([simName, 'EnergyTest'])
    outDir = pathlib.Path(avalancheDir, 'Outputs', 'ana1Tests', 'energyLineTest')
    plt.tight_layout()
    savePath = pU.saveAndOrPlot({'pathResult': outDir}, outFileName, fig)

    return resultEnergyTest, savePath


def getRunOutAngle(avaProfileMass, indStart=0, indEnd=-1):
    """Compute the center of mass runout angle

    Parameters
    -----------
    avaProfileMass: dict
        particle mass average properties
    indStart: int
        index of the start of the mass averaged path (to enable e.g. to discard the top extension).
        0 by default - so full mass averaged path from top
    indEnd: int
        index of the start of the mass averaged pass (to discard the bottom extension). -1 by default

    Returns
    --------
    runOutAngleRad: float
        Center of Mass runout angle in radians
    runOutAngleDeg: float
        Center of Mass runout angle in degrees
    """
    # compute horizontal and vertical traveled distance
    deltaS = avaProfileMass['s'][indEnd] - avaProfileMass['s'][indStart]
    deltaZ = avaProfileMass['z'][indEnd] - avaProfileMass['z'][indStart]
    # extract simulation runout
    runOutAngleRad = np.arctan(np.abs(deltaZ/deltaS))
    runOutAngleDeg = np.rad2deg(runOutAngleRad)
    return runOutAngleRad, runOutAngleDeg


def getAlphaProfileIntersection(energyLineTestCfg, avaProfileMass, mu, csz):
    """Extend the  profile path and compute the intersection
    between the theoretical energy line and the path profile
    The profile is extended by a line. The line slope is computed
    from the slope of the regression on the last points of the profile

    Parameters
    -----------
    energyLineTestCfg: configParser
        energy test config
    avaProfileMass: dict
        particle mass average properties
    mu: float
        friction coefficient
    csz: float
        dem cell size

    Returns
    --------
    slopeExt: float
        slope of the extrapolation line
    sIntersection: float
        s coord of the intersection between the line of slope mu and the
        mass average path profile
    zIntersection: float
        z coord of the intersection between the line of slope mu and the
        mass average path profile
    coefExt: float
        coefficient saying how long the path was extended to find the intersection

    """
    # get slope of the last profile points to extend the profile
    nCellsExtrapolation = energyLineTestCfg['energyLineTest'].getint('nCellsExtrapolation')
    idxExtra = np.nanmin(np.argwhere(avaProfileMass['s'][-1] - avaProfileMass['s'] < nCellsExtrapolation * csz))
    p1 = np.polyfit(avaProfileMass['s'][idxExtra:], avaProfileMass['z'][idxExtra:], 1)
    slopeExt = p1[0]
    # First check if the intersection is on the not extended profile
    s = avaProfileMass['s']
    z = avaProfileMass['z']
    alphaLine = z[0] - s * mu
    # find the intersection segment
    idx = np.argwhere(np.diff(np.sign(z - alphaLine))).flatten()
    idx = idx[-1]
    coefExt = 0
    # did not find the intersection, look further
    while s[idx] == 0 and coefExt < 4:
        s = np.append(avaProfileMass['s'], (1+coefExt)*avaProfileMass['s'][-1])
        z = np.append(avaProfileMass['z'], avaProfileMass['z'][-1] + coefExt*slopeExt*avaProfileMass['s'][-1])
        # find intersection between alpha line and profile
        alphaLine = z[0] - s * mu
        # find the intersection segment
        idx = np.argwhere(np.diff(np.sign(z - alphaLine))).flatten()
        idx = idx[-1]
        coefExt = coefExt + 1
    # find the exact intersection point
    s0 = s[idx]
    s1 = s[idx+1]
    zP0 = z[idx]
    zP1 = z[idx+1]
    zA0 = alphaLine[idx]
    zA1 = alphaLine[idx+1]
    sIntersection = s0 + (s1-s0)*(zA0-zP0)/((zP1-zP0)-(zA1-zA0))
    zIntersection = zP0 + (sIntersection-s0) * (zP1-zP0) / (s1-s0)
    if coefExt > 0:
        s[-1] = sIntersection
        z[-1] = zIntersection
    coefExt = np.max(s[-1]/avaProfileMass['s'][-1]-1, 0)
    return slopeExt, sIntersection, zIntersection, coefExt


def getEnergyInfo(avaProfileMass, g, mu, sIntersection, zIntersection, runOutAngleDeg, alphaDeg):
    """Compute energy dots and errors

    Parameters
    -----------
    avaProfileMass: dict
        particle mass average properties
    g: float
        gravity
    mu: float
        friction coefficient
    sIntersection: float
        s coord of the intersection between the line of slope mu and the
        mass average path profile
    zIntersection: float
        z coord of the intersection between the line of slope mu and the
        mass average path profile
    runOutAngleRad: float
        center of mass runout angle in radians
    runOutAngleDeg: float
        center of mass runout angle in degrees

    Returns
    --------
    zEne: numpy 1D array
        energy height of the particle averaged time steps
    u2Path: numpy 1D array
        kinetic energy of the particle averaged time steps
    sGeomL: 2 element list
        s coord (start and end) of the run out angle line
    zGeomL: 2 element list
        z coord (start and end) of the run out angle line
    resultEnergyTest: dict
        zEnd, sEnd, runoutAngle as well as rmseVelocityElevation, runOutSError, runOutZError, runOutAngleError
        between theoretical solution and simulation result

    """
    # read mass average quantities
    u2Path = avaProfileMass['u2']
    # compute mass average velocity elevation
    # extract energy altitude
    zEne = avaProfileMass['z'] + u2Path/(2*g)

    # Create the alpha line
    GK = sIntersection * mu
    z_end = avaProfileMass['z'][0] - GK
    zEneTarget = avaProfileMass['z'][0] - avaProfileMass['s'] * mu
    sGeomL = [avaProfileMass['s'][0], sIntersection]
    zGeomL = [avaProfileMass['z'][0], z_end]
    # compute errors
    # rmse of the energy height
    rmseVelocityElevation = np.sqrt(((zEne - zEneTarget) ** 2).mean())
    # error on s runout
    runOutSError = avaProfileMass['s'][-1] - sIntersection
    # error on z runout
    runOutZError = avaProfileMass['z'][-1] - zIntersection
    # error on angle runout
    runOutAngleError = runOutAngleDeg - alphaDeg
    resultEnergyTest = {'zEnd': avaProfileMass['z'][-1], 'sEnd': avaProfileMass['s'][-1], 'runoutAngle': runOutAngleDeg,
                        'rmseVelocityElevation': rmseVelocityElevation, 'runOutSError': runOutSError,
                        'runOutZError': runOutZError, 'runOutAngleError': runOutAngleError}
    return zEne, u2Path, sGeomL, zGeomL, resultEnergyTest
