import pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import FormatStrFormatter

# Local imports
from avaframe.in3Utils import geoTrans
import avaframe.out3Plot.plotUtils as pU
import avaframe.out3Plot.outCom1DFA as outCom1DFA
from avaframe.com1DFA import com1DFA
from avaframe.ana1Tests import energyLineTest


def hybridProfilePlot(avalancheDir, resultsHybrid):
    """Update profile plot with result of curent iteration"""
    fig = plt.figure(figsize=(3 * pU.figW, 2 * pU.figH))
    ax = plt.subplot(111)
    nIter = len(resultsHybrid.keys())
    i = 0
    cmap, _, ticks, norm = pU.makeColorMap(pU.cmapAvaframeCont, 0, nIter, continuous=pU.contCmap)
    for key, dict in resultsHybrid.items():
        avaProfileMassExt = dict["path"]
        avaProfileMassExt = geoTrans.computeS(avaProfileMassExt)
        alpha = dict["alpha"]
        sBetaPoint = dict["sBetaPoint"]
        col = cmap(norm(i))
        # Plot the whole profile with beta, alpha ... points and lines
        ax.plot(
            avaProfileMassExt["s"],
            avaProfileMassExt["z"],
            linestyle="-",
            color=col,
            label="profile (iteration %d)" % i,
        )
        ax.axvline(
            x=sBetaPoint, color=col, linestyle=":", linewidth=1, label="Beta point (iteration %d)" % i
        )
        s = avaProfileMassExt["s"][[0, -1]]
        z = avaProfileMassExt["z"][0] - s * np.tan(np.deg2rad(alpha))
        ax.plot(s, z, color=col, linestyle="--", label="AlphaLine (iteration %d)" % i)
        i = i + 1

    titleText = (
        r"Profiles extracted from the DFA simulations with corresponding $\alpha-\beta$ model results"
    )
    ax.set_title(titleText)
    ax.set_xlabel("projectd length s [m]")
    ax.set_ylabel("Height [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(linestyle=":", color="0.9")
    ax.legend(frameon=False)
    title = "com3HybProfPlot"
    l = ax.legend(loc="lower left")
    l.set_zorder(40)
    pU.putAvaNameOnPlot(ax, avalancheDir)
    path = pathlib.Path(avalancheDir, "Outputs", "com3Hybrid")
    pU.saveAndOrPlot({"pathResult": path}, title, fig)


def hybridPathPlot(avalancheDir, dem, resultsHybrid, fields, particles, muArray):
    """Update path plot with result of curent iteration"""
    headerOri = dem["originalHeader"]
    xllcOri = headerOri["xllcenter"]
    yllcOri = headerOri["yllcenter"]
    fig = plt.figure(figsize=(3 * pU.figW, 2 * pU.figH))
    ax = plt.subplot(111)
    nIter = len(resultsHybrid.keys())
    i = 0
    cmap, _, ticks, norm = pU.makeColorMap(pU.cmapAvaframeCont, 0, nIter, continuous=pU.contCmap)
    for key, dict in resultsHybrid.items():
        mu = muArray[i]
        avaProfileMassExt = dict["path"]
        xAB = dict["xAB"]
        yAB = dict["yAB"]
        col = cmap(norm(i))
        ax.plot(
            avaProfileMassExt["x"],
            avaProfileMassExt["y"],
            color=col,
            label="Center of mass path iteration %d ($\mu$ = %.2f )" % (i, mu),
            zorder=20,
        )
        ax.plot(
            xAB - xllcOri,
            yAB - yllcOri,
            "X",
            color=col,
            markersize=8,
            label=r"com2AB $\alpha$ point iteration %d" % i,
            zorder=20,
        )
        i = i + 1
    titleText = "Avalanche path for each iteration with peak travel angle field \n and particles flow thickness in final time step"
    ax.set_title(titleText)
    ax.set_ylabel("x [m]")
    ax.set_ylabel("y [m]")
    ax, extent, cb, CS = outCom1DFA.addResult2Plot(ax, dem["header"], fields["pta"], "pta")
    cb.ax.set_ylabel(pU.cfgPlotUtils["namepta"])
    ax = outCom1DFA.addDem2Plot(ax, dem, what="slope", extent=extent)
    ax, cb2 = outCom1DFA.addParticles2Plot(particles, ax, dem, whatS="h")
    cb2.ax.set_ylabel("particle " + pU.cfgPlotUtils["nameFT"])
    ax.set_ylim(extent[2:])
    ax.set_xlim(extent[:2])
    title = "com3HybRasterPlot"
    l = ax.legend(loc="lower left")
    l.set_zorder(40)
    pU.putAvaNameOnPlot(ax, avalancheDir)
    path = pathlib.Path(avalancheDir, "Outputs", "com3Hybrid")
    pU.saveAndOrPlot({"pathResult": path}, title, fig)


def generateCom1DFAPathPlot(avalancheDir, cfgPath, avaProfileMass, dem, parabolicFit, splitPoint, simName):
    """Make energy test analysis and plot results

    Parameters
    -----------
    avalancheDir: pathlib
        avalanche directory pathlib path
    cfgPath: configParser
        path finding config
    avaProfileMass: dict
        particle mass averaged properties
    dem: dict
        com1DFA simulation dictionary
    fieldsList: list
        field dictionary list
    simName: str
        simulation name
    """
    # read field
    fieldsList, fieldHeader, timeList = com1DFA.readFields(
        avalancheDir, ["pta"], simName=simName, flagAvaDir=True, comModule="com1DFA"
    )
    # compute simulation run out angle
    indStart = avaProfileMass["indStartMassAverage"]
    indEnd = avaProfileMass["indEndMassAverage"]
    runOutAngleRad, runOutAngleDeg = energyLineTest.getRunOutAngle(
        avaProfileMass, indStart=indStart, indEnd=indEnd
    )
    s0 = avaProfileMass["s"][indStart]
    avaProfileMass["s"] = avaProfileMass["s"] - s0
    # get parabola
    sPara = np.linspace(avaProfileMass["s"][0], avaProfileMass["s"][-1], 500)
    zPara = parabolicFit["a"] * sPara * sPara + parabolicFit["b"] * sPara + parabolicFit["c"]
    parabolicProfile = {"s": sPara, "z": zPara}
    # get angles of profiles
    anglePara, tmpPara, dsPara = geoTrans.prepareAngleProfile(
        cfgPath.getfloat("slopeSplitPoint"), parabolicProfile, raiseWarning=False
    )
    angleProf, tmpProf, dsProf = geoTrans.prepareAngleProfile(
        cfgPath.getfloat("slopeSplitPoint"), avaProfileMass, raiseWarning=False
    )

    # Create figures and plots
    fig = plt.figure(figsize=(pU.figW * 2, pU.figH * 1.5))

    # make the top-down view plot
    ax1 = plt.subplot2grid((2, 2), (1, 0), colspan=1)
    ax1 = avalancheThalwegPlot(ax1, fieldsList[-1]["pta"], dem, avaProfileMass, splitPoint, avalancheDir)

    # plot angle of profile and parabola
    ax2 = plt.subplot2grid((2, 2), (1, 1), colspan=1)
    # add path
    ax2.plot(sPara, anglePara, "k", lw=1, label="_parabolic fit")
    ax2.plot(
        avaProfileMass["s"][: indStart + 1],
        angleProf[: indStart + 1],
        "y.",
        label="_top extension",
        lw=2,
        path_effects=[pe.Stroke(linewidth=3, foreground="b"), pe.Normal()],
    )
    ax2.plot(
        avaProfileMass["s"][indEnd:],
        angleProf[indEnd:],
        "y.",
        label="_bottom extension",
        lw=2,
        path_effects=[pe.Stroke(linewidth=3, foreground="g"), pe.Normal()],
    )
    ax2.plot(
        avaProfileMass["s"][indStart : indEnd + 1],
        angleProf[indStart : indEnd + 1],
        "y.",
        label="_Center of mass path slope",
        lw=2,
        path_effects=[pe.Stroke(linewidth=3, foreground="k"), pe.Normal()],
    )
    minY, _ = ax2.get_ylim()
    minX, _ = ax2.get_xlim()
    if not splitPoint.get("isTopSplitPoint", False):  # if not a top split point
        ax2.axvline(x=splitPoint["s"], color="r", linewidth=1, linestyle="-.", label="_Split point")
        ax2.text(splitPoint["s"], minY, "%.2f m" % (splitPoint["s"]), color="r", ha="right", va="bottom")
    ax2.axhline(
        y=cfgPath.getfloat("slopeSplitPoint"),
        color="r",
        linewidth=1,
        linestyle="-.",
        label="_Split point angle (%.0f°)" % cfgPath.getfloat("slopeSplitPoint"),
    )
    ax2.text(
        minX,
        cfgPath.getfloat("slopeSplitPoint"),
        "%.0f°" % (cfgPath.getfloat("slopeSplitPoint")),
        color="r",
        ha="left",
        va="bottom",
    )
    ax2.axhline(y=10, color="0.8", linewidth=1, linestyle="-.", label="_Beta angle (10°)")
    ax2.text(minX, 10, "%.0f°" % (10), color="0.8", ha="left", va="bottom")
    ax2.set_xlabel("$s_{xy}$ [m]")
    ax2.set_ylabel("slope angle [°]")
    ax2.set_title("Avalanche thalweg profile slope")

    # make profile plot, zoomed out
    ax3 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    # plot mass averaged center of mass
    ax3.plot(
        avaProfileMass["s"][: indStart + 1],
        avaProfileMass["z"][: indStart + 1],
        "-y.",
        label="top extension",
        lw=1,
        path_effects=[pe.Stroke(linewidth=3, foreground="b"), pe.Normal()],
    )
    ax3.plot(
        avaProfileMass["s"][indEnd:],
        avaProfileMass["z"][indEnd:],
        "-y.",
        label="bottom extension",
        lw=1,
        path_effects=[pe.Stroke(linewidth=3, foreground="g"), pe.Normal()],
    )
    ax3.plot(
        avaProfileMass["s"][indStart : indEnd + 1],
        avaProfileMass["z"][indStart : indEnd + 1],
        "-y.",
        label="Center of mass path / profile / angle",
        lw=1,
        path_effects=[pe.Stroke(linewidth=3, foreground="k"), pe.Normal()],
    )
    ax3.plot(sPara, zPara, "-k", label="Parabolic fit")
    if not splitPoint.get("isTopSplitPoint", False):  # if not a top split point
        ax3.axvline(x=splitPoint["s"], color="r", linewidth=1, linestyle="-.", label="Split point")
        ax3.axhline(y=splitPoint["z"], color="r", linewidth=1, linestyle="-.", label="_Split point")
        minY, _ = ax3.get_ylim()
        minX, _ = ax3.get_xlim()
        ax3.text(splitPoint["s"], minY, "%.2f m" % (splitPoint["s"]), color="r", ha="left", va="bottom")
        ax3.text(minX, splitPoint["z"], "%.2f m" % (splitPoint["z"]), color="r", ha="left", va="bottom")
    ax3.set_xlabel("$s_{xy}$ [m]")
    ax3.set_ylabel("z [m]")
    ax3.legend()
    ax3.set_title("Avalanche thalweg profile")

    outFileName = "_".join([simName, "DFAPath"])
    outDir = pathlib.Path(avalancheDir, "Outputs", "ana5Utils", "DFAPath")
    plt.tight_layout()
    outPath = pU.saveAndOrPlot({"pathResult": outDir}, outFileName, fig)
    return outPath


def avalancheThalwegPlot(ax, fieldRaster, dem, avaProfileMass, splitPoint=None, avalancheDir="", cmapHS=None):
    """
    plots the location of the thalweg (including extensions to top and bottom)
    on a raster Field and a hillshade

    Parameters
    ----------
    ax: matplotlib.axis
        axis on which the thalweg is plotted
    fieldRaster: numpy array
        raster that is plotted
    dem: dict
        dictionary containing DEM header and data
    avaProfileMass: dict
        contains x and y coordinates of thalweg,
        and the indices of start and end of the averaged profile
    splitPoint: dict or None
        contains x and y coordinates of splitpoints that are plotted
    avalancheDir: str
        name of the avalanche directory to add optionally on the plot
    cmapHS: matplotlib.colormap
        optionally defined colormap for the plotted hillshade
    Returns
    -------
    ax: matplotlib.axis
        axis containing the plotted thalweg with the raster and hillshade
    """

    indStart = avaProfileMass["indStartMassAverage"]
    indEnd = avaProfileMass["indEndMassAverage"]
    rowsMin, rowsMax, colsMin, colsMax = pU.constrainPlotsToData(
        fieldRaster, dem["header"]["cellsize"], extentOption=True, constrainedData=False, buffer=""
    )
    ax, extent, cbar0, cs1 = outCom1DFA.addResult2Plot(ax, dem["header"], fieldRaster, "pta")
    cbar0.ax.set_ylabel("peak travel angle")
    # add DEM hillshade with contour lines
    ax = outCom1DFA.addDem2Plot(ax, dem, what="hillshade", extent=extent, cmap=cmapHS)
    # add path
    ax.plot(
        avaProfileMass["x"][: indStart + 1],
        avaProfileMass["y"][: indStart + 1],
        "-y.",
        zorder=20,
        label="top extension",
        lw=2,
        path_effects=[pe.Stroke(linewidth=3, foreground="b"), pe.Normal()],
    )
    ax.plot(
        avaProfileMass["x"][indEnd:],
        avaProfileMass["y"][indEnd:],
        "-y.",
        zorder=20,
        label="bottom extension",
        lw=2,
        path_effects=[pe.Stroke(linewidth=3, foreground="g"), pe.Normal()],
    )
    ax.plot(
        avaProfileMass["x"][indStart : indEnd + 1],
        avaProfileMass["y"][indStart : indEnd + 1],
        "-y.",
        zorder=20,
        label="Center of mass path",
        lw=2,
        path_effects=[pe.Stroke(linewidth=3, foreground="k"), pe.Normal()],
    )
    if splitPoint is not None:
        if not splitPoint.get("isTopSplitPoint", False):  # if not a top split point
            ax.plot(splitPoint["x"], splitPoint["y"], "P", color="r", label="_Split point", zorder=20)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.set_ylim([rowsMin, rowsMax])
    ax.set_xlim([colsMin, colsMax])
    ax.set_title("Avalanche thalweg")
    pU.putAvaNameOnPlot(ax, avalancheDir)
    return ax
