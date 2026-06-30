import numpy as np
import pathlib
import copy
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import avaframe.com1DFA.DFAtools as DFAtls


import avaframe.out3Plot.plotUtils as pU
import avaframe.com1DFA.DFAfunctionsCython as DFAfunC
import avaframe.out3Plot.outCom1DFA as outCom1DFA


def plotBufferRelease(inputSimLines, xBuffered, yBuffered):
    """ plot release lines with added bufferLine """

    plt.plot(inputSimLines['releaseLine']['x'], inputSimLines['releaseLine']['y'], 'g')
    plt.plot(xBuffered, yBuffered, 'b')
    plt.title('Buffered release polygon')
    plt.show()
    plt.show()


def plotBondsSnowSlideFinal(cfg, particles, dem, inputSimLines=''):
    """With snowSlide option on, plot the bonds between particles as well as the particles properties
    """
    fig, (ax1) = plt.subplots(ncols=1)
    ax1.set_aspect('equal')

    if cfg['GENERAL'].getint('snowSlide') == 1:
        points = np.zeros((particles['nPart'], 2))
        points[:, 0] = particles['x'] - 0*dem['originalHeader']['xllcenter']
        points[:, 1] = particles['y'] - 0*dem['originalHeader']['yllcenter']
        edges = DFAfunC.plotBondC(particles)
        lc = LineCollection(points[edges])
        plt.gca().add_collection(lc)
    # ax1.plot(particles['x'], particles['y'], '.b')
    particles['v'] = DFAtls.norm(particles['ux'], particles['uy'], particles['uz'])
    ax1, cb = outCom1DFA.addParticles2Plot(particles, ax1, dem, whatS='m', whatC='h', colBarResType='FT')
    if inputSimLines != '':
        if inputSimLines['resLine'] is not None:
            NameRel = inputSimLines['resLine']['Name']
            StartRel = inputSimLines['resLine']['Start']
            LengthRel = inputSimLines['resLine']['Length']
            for i in range(len(NameRel)):
                start = StartRel[i]
                end = start + LengthRel[i]
                avapath = {}
                avapath['x'] = inputSimLines['resLine']['x'][int(start):int(end)] - dem['originalHeader']['xllcenter']
                avapath['y'] = inputSimLines['resLine']['y'][int(start):int(end)] - dem['originalHeader']['yllcenter']
                plt.plot(avapath['x'], avapath['y'], 'g')
    fig.legend()
    plt.title('Bonds between particles with snowSlide option activated')
    plt.show()
    # plt.pause(1)
    plt.close()


def plotPartIni(particles, dem):
    header = dem['header']
    x = np.arange(header['ncols']) * header['cellsize']
    y = np.arange(header['nrows']) * header['cellsize']
    fig, ax = plt.subplots(figsize=(pU.figW, pU.figH))
    cmap = copy.copy(mpl.cm.get_cmap("Greys"))
    ref0, im = pU.NonUnifIm(ax, x, y, dem['areaRaster'], 'x [m]', 'y [m]',
                            extent=[x.min(), x.max(), y.min(), y.max()],
                            cmap=cmap, norm=None)

    ax.plot(particles['x'], particles['y'], 'or', linestyle='None')
    pU.addColorBar(im, ax, None, 'm²')
    plt.show()


def plotAreaDebug(header, avapath, Raster):
    ncols = header['ncols']
    nrows = header['nrows']
    cellsize = header['cellsize']
    x = np.arange(ncols) * cellsize
    y = np.arange(nrows) * cellsize
    fig, ax = plt.subplots(figsize=(pU.figW, pU.figH))
    ax.set_title('Release area')
    cmap = copy.copy(mpl.cm.get_cmap("Greys"))
    ref0, im = pU.NonUnifIm(ax, x, y, Raster, 'x [m]', 'y [m]',
                            extent=[x.min(), x.max(), y.min(), y.max()],
                            cmap=cmap, norm=None)
    ax.plot(avapath['x'] * cellsize, avapath['y'] * cellsize, 'r', label='release polyline')
    plt.legend()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cax)
    plt.show()


def plotRemovePart(xCoord0, yCoord0, header, X, Y, Mask, mask):
    x = np.arange(header['ncols']) * header['cellsize']
    y = np.arange(header['nrows']) * header['cellsize']
    fig, ax = plt.subplots(figsize=(pU.figW, pU.figH))
    ax.set_title('Release area')
    cmap = copy.copy(mpl.cm.get_cmap("Greys"))
    ref0, im = pU.NonUnifIm(ax, x, y, Mask, 'x [m]', 'y [m]',
                            extent=[x.min(), x.max(), y.min(), y.max()],
                            cmap=cmap, norm=None)
    ax.plot(xCoord0 * header['cellsize'], yCoord0 * header['cellsize'], 'r', label='release polyline')
    ax.plot(X[mask] * header['cellsize'], Y[mask] * header['cellsize'], '.b')
    plt.legend()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cax)
    plt.show()


def plotPartAfterRemove(points, xCoord0, yCoord0, mask):
    fig, ax = plt.subplots(figsize=(pU.figW, pU.figH))
    ax.set_title('Release area')
    ax.plot(xCoord0, yCoord0, 'r', label='release polyline')
    ax.plot(points['x'], points['y'], '.b')
    ax.plot(points['x'][mask], points['y'][mask], '.g')
    plt.legend()
    plt.show()


def analysisPlots(particlesList, fieldsList, cfg, demOri, dem, outDir):
    """ create analysis plots during simulation run """

    cfgGen = cfg['GENERAL']
    partRef = particlesList[0]
    Z0 = partRef['z'][0]
    rho = cfgGen.getfloat('rho')
    gravAcc = cfgGen.getfloat('gravAcc')
    mu = cfgGen.getfloat('mu')
    repeat = True
    while repeat:
        fig, ax = plt.subplots(figsize=(pU.figW, pU.figH))
        T = np.array([0])
        Z = np.array([0])
        U = np.array([0])
        S = np.array([0])
        for part, field in zip(particlesList, fieldsList):
            T = np.append(T, part['t'])
            S = np.append(S, part['s'][0])
            Z = np.append(Z, part['z'][0])
            U = np.append(U, DFAtls.norm(part['ux'][0], part['uy'][0], part['uz'][0]))
            fig, ax = plotPosition(
                fig, ax, part, demOri, dem['Nz'], pU.cmapDEM2, '', plotPart=True)
            fig.savefig(pathlib.Path(outDir, 'particlest%f.%s' % (part['t'], pU.outputFormat)))

        fig, ax = plotPosition(
                fig, ax, part, demOri, dem['Nz'], pU.cmapDEM2, '', plotPart=True, last=True)
        fig.savefig(pathlib.Path(outDir, 'particlesFinal.%s' % (pU.outputFormat)))
        value = input("[y] to repeat:\n")
        if value != 'y':
            repeat = False

    fieldEnd = fieldsList[-1]
    partEnd = particlesList[-1]
    fig1, ax1 = plt.subplots(figsize=(pU.figW, pU.figH))
    fig2, ax2 = plt.subplots(figsize=(pU.figW, pU.figH))
    fig3, ax3 = plt.subplots(figsize=(pU.figW, pU.figH))
    fig1, ax1 = plotPosition(
        fig1, ax1, partEnd, demOri, fieldEnd['FT'], pU.cmapPres, 'm', plotPart=False)
    fig2, ax2 = plotPosition(
        fig2, ax2, partEnd, demOri, fieldEnd['FV'], pU.cmapPres, 'm/s', plotPart=False)
    fig3, ax3 = plotPosition(
        fig3, ax3, partEnd, demOri, fieldEnd['P']/1000, pU.cmapPres, 'kPa', plotPart=False)
    plt.show()


def plotPosition(fig, ax, particles, dem, data, Cmap, unit, plotPart=False, last=False):
    header = dem['header']
    ncols = header.ncols
    nrows = header.nrows
    xllc = header.xllcenter
    yllc = header.yllcenter
    csz = header.cellsize
    xgrid = np.linspace(xllc, xllc+(ncols-1)*csz, ncols)
    ygrid = np.linspace(yllc, yllc+(nrows-1)*csz, nrows)
    PointsX, PointsY = np.meshgrid(xgrid, ygrid)
    X = PointsX[0, :]
    Y = PointsY[:, 0]
    Z = dem['rasterData']
    x = particles['x'] + xllc
    y = particles['y'] + yllc
    xx = np.arange(ncols) * csz + xllc
    yy = np.arange(nrows) * csz + yllc
    try:
        # Get the images on an axis
        cb = ax.images[-1].colorbar
        if cb:
            cb.remove()
    except IndexError:
        pass

    ax.clear()
    ax.set_title('t=%.2f s' % particles['t'])
    cmap, _, ticks, norm = pU.makeColorMap(Cmap, np.nanmin(data), np.nanmax(data), continuous=True)
    cmap.set_under(color='w')
    ref0, im = pU.NonUnifIm(ax, xx, yy, data, 'x [m]', 'y [m]',
                         extent=[x.min(), x.max(), y.min(), y.max()],
                         cmap=cmap, norm=norm)

    Cp1 = ax.contour(X, Y, Z, levels=10, colors='k')
    pU.addColorBar(im, ax, ticks, unit)
    if plotPart:
        # ax.plot(x, y, '.b', linestyle='None', markersize=1)
        # ax.plot(x[NPPC == 1], y[NPPC == 1], '.c', linestyle='None', markersize=1)
        # ax.plot(x[NPPC == 4], y[NPPC == 4], '.b', linestyle='None', markersize=1)
        # ax.plot(x[NPPC == 9], y[NPPC == 9], '.r', linestyle='None', markersize=1)
        # ax.plot(x[NPPC == 16], y[NPPC == 16], '.m', linestyle='None', markersize=1)
        # load variation colormap
        variable = particles['h']
        cmap, _, ticks, norm = pU.makeColorMap(pU.cmapThickness, np.nanmin(data), np.amax(variable), continuous=True)
        # set range and steps of colormap
        cc = variable
        sc = ax.scatter(x, y, c=cc, cmap=cmap, marker='.')

        if last:
            pU.addColorBar(sc, ax, ticks, 'm', 'Flow Thickness')

    plt.pause(0.1)
    return fig, ax


def plotContours(fig, ax, t, header, data, Cmap, unit, last=False):
    ncols = header['ncols']
    nrows = header['nrows']
    xllc = header['xllcenter']
    yllc = header['yllcenter']
    csz = header['cellsize']
    xgrid = np.linspace(xllc, xllc+(ncols-1)*csz, ncols)
    ygrid = np.linspace(yllc, yllc+(nrows-1)*csz, nrows)
    PointsX, PointsY = np.meshgrid(xgrid, ygrid)
    X = PointsX[0, :]
    Y = PointsY[:, 0]
    try:
        # Get the images on an axis
        cb = ax.images[-1].colorbar
        if cb:
            cb.remove()
    except IndexError:
        pass

    ax.clear()
    ax.set_title('t=%.2f s' % t)
    cmap, _, ticks, norm = pU.makeColorMap(Cmap, np.nanmin(data), np.nanmax(data), continuous=True)
    cmap.set_under(color='w')

    CS = ax.contour(X, Y, data, levels=8, origin='lower', cmap=cmap,
                    linewidths=2)
    lev = CS.levels

    if last:
        # pU.addColorBar(im, ax, ticks, unit, 'Flow Thickness')
        CB = fig.colorbar(CS)
        ax.clabel(CS, inline=1, fontsize=8)
    return fig, ax, cmap, lev


def plotPathExtTop(profile, particlesIni, xFirst, yFirst, zFirst, dz1):
    """Plot the extended path towards the top of the release"""
    # get highest particle
    indHighest = np.argmax(particlesIni['z'])
    xHighest = particlesIni['x'][indHighest]
    yHighest = particlesIni['y'][indHighest]
    zHighest = particlesIni['z'][indHighest]
    cmap, _, ticks, norm = pU.makeColorMap(pU.cmapThickness, np.nanmin(dz1), np.nanmax(dz1), continuous=True)
    fig, ax = plt.subplots(figsize=(pU.figW, pU.figH))
    ax.set_title('Extend path towards the top')
    ax.tricontour(particlesIni['x'], particlesIni['y'], dz1, levels=14, linewidths=0.5, colors='k')
    # cntr2 = ax.tricontourf(particlesIni['x'], particlesIni['y'], dz1, levels=14, cmap=cmap, norm=norm)
    sc = ax.scatter(particlesIni['x'], particlesIni['y'], c=dz1, cmap=cmap, norm=norm, label='particles at t=0s')
    ax.plot(xHighest, yHighest, '.r', label='highest particle at t=0s')
    ax.plot(profile['x'][1:], profile['y'][1:], '.k', label='mass averaged path')
    ax.plot(xFirst, yFirst, '.b', markersize=10, label='top point of the mass averaged path')
    ax.plot(profile['x'][0], profile['y'][0], '.g', label='point leading to longest runout')
    ax.plot(profile['x'][0:2], profile['y'][0:2], 'k--', label='extended path')
    pU.addColorBar(sc, ax, ticks, 'm', title='energy height')
    plt.legend()

    fig1, ax1 = plt.subplots(figsize=(pU.figW, pU.figH))
    ax1.set_title('Extend path towards the top')
    ax1.plot(particlesIni['x'], particlesIni['z'], '.c', label='particles at t=0s')
    ax1.plot(xHighest, zHighest, '.r', label='highest particle at t=0s')
    ax1.plot(profile['x'][1:], profile['z'][1:], '.k', label='mass averaged path')
    ax1.plot(xFirst, zFirst, '.b', markersize=10, label='top point of the mass averaged path')
    ax1.plot(profile['x'][0], profile['z'][0], '.g', label='point leading to longest runout')
    ax1.plot(profile['x'][0:2], profile['z'][0:2], 'k--', label='extended path')
    plt.legend()
    plt.show()


def plotPathExtBot(profile, xInterest, yInterest, zInterest, xLast, yLast):
    """Plot the extended path towards the bottom of the avalanche"""
    fig, ax = plt.subplots(figsize=(pU.figW, pU.figH))
    ax.set_title('Extend path towards the bottom')
    ax.plot(profile['x'][:-1], profile['y'][:-1], '.k', label='mass averaged path')
    ax.plot(xInterest, yInterest, '.m', markersize=10, label='points considered to find drection')
    ax.plot(xLast, yLast, '.b', markersize=10, label='bottom point of the mass averaged path')
    ax.plot(profile['x'][0], profile['y'][0], '.g',
            label='point in the extention direction at distance \n 0.2 x path length from the bottom point')
    ax.plot(profile['x'][-2:], profile['y'][-2:], 'k--', label='extended path')
    plt.legend()
    plt.show()


def plotSlopeAngle(s, angle, idsBetaPoint):
    """plot slope angle along a profile, add beta info"""
    plt.figure(figsize=(10, 6))
    plt.plot(s, angle, '.k')
    plt.plot(s[idsBetaPoint], angle[idsBetaPoint], 'or')
    plt.axhline(y=10, color='0.8',
                linewidth=1, linestyle='-.', label='Beta angle line')
    plt.show()
    plt.close()


def plotFindAngle(avaProfile, angleProf, parabolicProfile, anglePara, s0, sEnd, splitPoint, indSplitPoint):
    """helper plot for the getSplitPoint, findAngleProfile and prepareAngleProfile functions
    Plots the slope angle and elevation function of s"""
    plt.figure(figsize=(10, 6))
    plt.plot(parabolicProfile['s'], anglePara, '.k')
    plt.plot(avaProfile['s'] - s0, angleProf, '.b')
    # plt.plot(s[ids10Point], anglePara[indSplitPoint], 'or')
    plt.axhline(y=10, color='0.8',
                linewidth=1, linestyle='-.', label='10° line')
    plt.axhline(y=20, color='0.8',
                linewidth=1, linestyle='-.', label='10° line')
    plt.axhline(y=15, color='0.8',
                linewidth=1, linestyle='-.', label='10° line')
    plt.axvline(x=0, color='0.8',
                linewidth=1, linestyle='-.', label='Start')
    plt.axvline(x=sEnd-s0, color='0.8',
                linewidth=1, linestyle='-.', label='End')
    if splitPoint != '':
        plt.plot(parabolicProfile['s'][indSplitPoint], anglePara[indSplitPoint], '.r')
        plt.plot(avaProfile['s'][indSplitPoint] - s0, angleProf[indSplitPoint], '.r')

    plt.figure(figsize=(10, 6))
    plt.plot(parabolicProfile['s'], parabolicProfile['z'], '.k')
    plt.plot(avaProfile['s'] - s0, avaProfile['z'], '.b')
    plt.axvline(x=0, color='0.8',
                linewidth=1, linestyle='-.', label='Start')
    plt.axvline(x=sEnd-s0, color='0.8',
                linewidth=1, linestyle='-.', label='End')
    if splitPoint != '':
        plt.plot(parabolicProfile['s'][indSplitPoint], parabolicProfile['z'][indSplitPoint], '.r')
        plt.plot(avaProfile['s'][indSplitPoint] - s0, avaProfile['z'][indSplitPoint], '.r')
    plt.show()
    plt.close()


def plotProfile(s, z, idsBetaPoint):
    """plot profile, add beta info"""
    plt.figure(figsize=(10, 6))
    plt.plot(s, z)
    plt.axvline(x=s[idsBetaPoint], color='0.8',
                linewidth=1, linestyle='-.')
    plt.show()
    plt.close()

def plotVolumeRelease(releaseLine, relThField, releaseLineField):
    """ create a plot of the release line raster, the relThField for release thickness,
        releaseLineField - combination of relThField and release line raster mask """

    fig = plt.figure()
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    im0 = ax1.imshow(releaseLine['rasterData'])
    im1 = ax2.imshow(relThField)
    im2 = ax3.imshow(releaseLineField)
    fig.colorbar(im0 , ax=ax1)
    fig.colorbar(im1, ax=ax2)
    fig.colorbar(im2, ax=ax3)
    plt.show()


def plotParticlesRelease(particles, relRaster, releaseLine, dem, cfg, xyParticlesAll):
    """plot the release raster and release polygon outline and the particles that were placed and the ones that
    are finally used for the computation after removing the ones that are outside the release polygon
    - used in particleInitialisation to visualize initialisation process in com1DFA/initializeParticles

    Parameters
    ------------
    particles: dict
        final particles dict after initialization
    relRaster: numpy ndarray
        release raster
    releaseLine: dict
        release polygon info dict
    dem: dict
        dem info dict
    cfg: configparser object
        simulation configuration settings, requires rho
    xyParticlesAll: dict
        particles dict during initialization before removing those outside the release polygon

    """

    # compute volume of particles
    volParticles = particles["mTot"] / cfg.getfloat("rho")

    extentCellCenters, extentCellCorners = pU.createExtentMinMax(
        relRaster, dem["originalHeader"], originLLCenter=True
    )

    # figure
    fig, ax = plt.subplots(nrows=1, ncols=1)
    # Minor ticks
    ax.set_xticks(
        np.arange(extentCellCorners[0], extentCellCorners[1], dem["originalHeader"]["cellsize"]), minor=True
    )
    ax.set_yticks(
        np.arange(extentCellCorners[2], extentCellCorners[3], dem["originalHeader"]["cellsize"]), minor=True
    )
    # Gridlines based on minor ticks
    ax.grid(which="minor", color="w", linestyle="-", linewidth=2)
    ax.imshow(relRaster, extent=extentCellCorners, origin="lower")
    # plot all particles before removing the ones outside of release polygon
    ax.plot(
        xyParticlesAll["x"] + dem["originalHeader"]["xllcenter"],
        xyParticlesAll["y"] + dem["originalHeader"]["yllcenter"],
        "+g",
    )
    # only particles that have not been removed
    ax.plot(
        particles["x"] + dem["originalHeader"]["xllcenter"],
        particles["y"] + dem["originalHeader"]["yllcenter"],
        "*r",
    )
    ax.plot(releaseLine["x"], releaseLine["y"], "-b")
    ax.set_title("mass/rho: %.2fm3" % (volParticles))
    plt.show()
