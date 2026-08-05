""" functions to create a GIF for com4FlowPys generation data"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from avaframe.runScripts.runComputeDist import outFile
from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection
import logging
import os
import pathlib
from matplotlib.colors import LightSource
from PIL import Image
from cmcrameri import cm as cmapCrameri

import avaframe.in2Trans.rasterUtils as rasterUtils
from avaframe.in3Utils import cfgUtils
import avaframe.in1Data.getInput as getInput
import avaframe.in3Utils.fileHandlerUtils as fU
import avaframe.in3Utils.geoTrans as gT
import pickle

import avaframe.out3Plot.outCom4Gif as outCom4Gif

log = logging.getLogger(__name__)



def addHillshadeSimple(dem, cellSize, cfgHS):
    """
    Computes a hillshade array from the DEM

    Parameters
    -----------
    dem: numpy array
        DEM raster data
    cellSize: float
        cellsize of DEM
    cfgHS: configparser object
        configuration for hillshade
    """
    lightSource = LightSource(azdeg=cfgHS.getfloat("azimuth"), altdeg=cfgHS.getfloat("altitude"))
    hillshade = lightSource.hillshade(dem, vert_exag=cfgHS.getfloat("vertExag"), dx=cellSize, dy=cellSize)
    return hillshade


def searchResFolder(avalanchedir, module="com4FlowPy"):
    """
    search for a com4Flowpy result folder, return the respective simhash if only one exists

    Parameters
    ----------
    avalanchedir: str or pathlib.Path
        path to avalanche project
    module: str
        module name (default: com4FlowPy)

    Returns
    -------
    oneResFolder: bool
        True if exactly one result folder exists, otherwise False
    simHash: str
        the simhash of the simulation, if one result folder exists
    """
    dir = avalanchedir / "Outputs" / module / "peakFiles"

    resFolders = []
    for filename in os.listdir(dir):
        if filename.startswith("res_"):
            resFolders.append(filename)
    if len(resFolders) == 0:
        message = f"No results {module} folder found in {dir}"
        log.error(message)
        raise FileNotFoundError(message)
    elif len(resFolders) > 1:
        oneResFolder = False
        simHash = ""
    elif len(resFolders) == 1:
        oneResFolder = True
        simHash = str(resFolders[0]).split("res_", 1)[1]
    return oneResFolder, simHash


def removeContour(contourHolder):
    """
    Removes the previously drawn contour artist (if any), so a new one
    can be drawn for the current frame without accumulating old contours.

    Parameters
    -----------
    contourHolder: dict
        mutable dict with key "artist", holding the current contour artist
        (or None). Passed by reference so state persists across calls.
    """
    if contourHolder["artist"] is not None:
        try:
            contourHolder["artist"].remove()
        except AttributeError:
            for coll in contourHolder["artist"].collections:
                coll.remove()
        contourHolder["artist"] = None
    return contourHolder


def figureToRgbArray(fig):
    """
    Render the current state of a matplotlib figure to an RGB numpy array.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to render (must already have been drawn/updated).

    Returns
    -------
    numpy.ndarray
        RGB image array of shape (height, width, 3), dtype uint8.
    """
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return buf[..., :3].copy()


def buildGlobalPalette(frameArrays, colors=256, sampleEveryFrame=5, sampleEveryPixel=8):
    """
    Build a single shared color palette from a subsample of frames and pixels.

    Using one global palette (instead of letting PIL quantize each GIF
    frame independently) prevents visible color flicker between frames,
    since every frame is then mapped onto the exact same set of colors.

    Both frames and pixels within each frame are subsampled before
    building the palette, since using all pixels of many full-resolution
    frames can require more contiguous memory than PIL can allocate for
    a single quantization image.

    Parameters
    ----------
    frameArrays : list of numpy.ndarray
        List of RGB frame arrays, each of shape (height, width, 3).
    colors : int, optional
        Number of palette colors (GIF hard limit is 256). Default 256.
    sampleEveryFrame : int, optional
        Use only every Nth frame to build the palette. Default 5.
    sampleEveryPixel : int, optional
        Use only every Nth pixel (in both height and width) within each
        sampled frame. Default 8.

    Returns
    -------
    paletteImg: PIL.Image.Image
        A palette-mode image whose palette should be reused for all
        frames via ``Image.quantize(palette=...)``.
    """
    sampledFrames = frameArrays[::max(1, sampleEveryFrame)]

    # subsample pixels within each frame too, so the total number of
    # pixels stays small regardless of frame resolution or frame count
    pixelSamples = [
        frame[::sampleEveryPixel, ::sampleEveryPixel, :].reshape(-1, 3)
        for frame in sampledFrames
    ]
    stacked = np.concatenate(pixelSamples, axis=0)

    # reshape into a reasonably square-ish image instead of a single row,
    # which avoids extremely wide (1, N) images that some PIL/C backends
    # fail to allocate contiguous memory for
    nPixels = stacked.shape[0]
    width = int(np.ceil(np.sqrt(nPixels)))
    height = int(np.ceil(nPixels / width))
    padded = np.zeros((height * width, 3), dtype=np.uint8)
    padded[:nPixels] = stacked

    paletteSourceImg = Image.fromarray(padded.reshape(height, width, 3), mode="RGB")
    paletteImg = paletteSourceImg.quantize(colors=colors, method=Image.MEDIANCUT)
    return paletteImg


def loadExtendedProfile(extendedProfilePicklePath):
    """
    Load the extended thalweg profile (avaProfileMass) from a pickle file.

    Parameters
    ----------
    extendedProfilePicklePath : str or pathlib.Path
        Path to the pickle file containing the extended profile dict,
        with keys "x", "y", "indStartMassAverage", "indEndMassAverage"
        (same structure as used in avalancheThalwegPlot).

    Returns
    -------
    dict
        The loaded avaProfileMass dictionary.
    """
    with open(extendedProfilePicklePath, "rb") as handle:
        avaProfileMass = pickle.load(handle)
    return avaProfileMass


def coordsToRowCol(x, y, rasterHeader):
    """
    Convert real-world x/y coordinates to raster row/col array indices.

    Assumes the standard AvaFrame raster convention: row 0 corresponds to
    the northernmost (top) row, col 0 to the westernmost (left) column,
    matching how ``imshow`` displays the raster arrays elsewhere in this
    module (no ``extent`` set, i.e. plotted in pixel/array index space).

    Parameters
    ----------
    x, y : numpy.ndarray or float
        Real-world coordinates.
    rasterHeader : dict
        Raster header as returned by ``rasterUtils.readRaster``, must
        contain "xllcenter", "yllcenter", "cellsize" and "nrows".

    Returns
    -------
    col, row : numpy.ndarray or float
        Corresponding array indices (float, not rounded, so subpixel
        positions along the profile are preserved for smooth plotting).
    """
    cellSize = rasterHeader["cellsize"]
    col = (x - rasterHeader["xllcenter"]) / cellSize
    row = rasterHeader["nrows"] - 1 - (y - rasterHeader["yllcenter"]) / cellSize
    return col, row


def addThalwegExtension(ax, avaProfileMass, rasterHeader):
    """
    Plot the top and bottom thalweg extensions plus the center-of-mass
    path onto an existing axes, in the same row/col pixel coordinate
    system used by the generation video's imshow layers.

    Mirrors the styling used in avaframe.out3Plot.outCom1DFA's
    avalancheThalwegPlot (colored outline via path_effects), so the
    final GIF frame looks consistent with AvaFrame's standard thalweg
    plots.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on (same axes as the animated raster layers).
    avaProfileMass : dict
        Extended profile data with keys "x", "y",
        "indStartMassAverage", "indEndMassAverage".
    rasterHeader : dict
        Raster header used to convert avaProfileMass["x"]/["y"] into
        row/col indices via coordsToRowCol.

    Returns
    -------
    None
    """
    indStart = avaProfileMass["indStartMassAverage"]
    indEnd = avaProfileMass["indEndMassAverage"]

    colProfile, rowProfile = coordsToRowCol(
        avaProfileMass["x"], avaProfileMass["y"], rasterHeader
    )

    topLine, = ax.plot(
        colProfile[: indStart + 1],
        rowProfile[: indStart + 1],
        "-b.",
        # color="blue",
        zorder=20,
        label="top extension",
        lw=2.5,
        path_effects=[pe.Stroke(linewidth=2, foreground="b"), pe.Normal()],
    )
    bottomLine, = ax.plot(
        colProfile[indEnd:],
        rowProfile[indEnd:],
        "-g.",
        # color="green",
        zorder=20,
        label="bottom extension",
        lw=2.5,
        path_effects=[pe.Stroke(linewidth=2, foreground="g"), pe.Normal()],
    )
    centerLine, = ax.plot(
        colProfile[indStart: indEnd + 1],
        rowProfile[indStart: indEnd + 1],
        "-k.",
        # color="black",
        zorder=20,
        label="center of flux path",
        lw=2.5,
        path_effects=[pe.Stroke(linewidth=2, foreground="k"), pe.Normal()],
    )
    return topLine, bottomLine, centerLine


def getProfileData(dem, profilePickleData):
    """
    Compute travel distance (s), elevation (z) and get energy-height
    (zDelta) values per generation.

    Parameters
    ----------
    dem : dict
        DEM dictionary containing header and the raster data
    profilePickleData : dict
        Thalweg data loaded from the pickle file, must contain the x, y and zDelta values
        per generation/iteration, (in real-world coordinates.)

    Returns
    -------
    profile: dict
        contains numpy arrays with:
        x: x-coordinates of thalweg
        y: y-coordinates of thalweg
        zDelta: zDelta values for each thalweg
        s: Cumulative travel distance along the (x, y) trajectory, one
           value per generation, starting at 0.
        z: DEM elevation at each (x, y) position, one value per generation,
           obtained via bilinear interpolation on the DEM.
    """
    x = np.asarray(profilePickleData["x"], dtype=np.float64)
    y = np.asarray(profilePickleData["y"], dtype=np.float64)
    zDelta = np.asarray(profilePickleData["zDelta"], dtype=np.float32)

    # cumulative travel distance: s[0] = 0, then running sum of
    # step-wise Euclidean distances between consecutive (x, y) points
    dx = np.diff(x)
    dy = np.diff(y)
    stepDist = np.sqrt(dx ** 2 + dy ** 2)
    s = np.concatenate([[0.0], np.cumsum(stepDist)]).astype(np.float32)

    # project (x, y) onto the DEM to get elevation z, via bilinear interpolation
    points = {"x": x, "y": y}
    points, ioob = gT.projectOnRaster(dem, points, interp="bilinear")
    if ioob > 0:
        log.warning(f"{ioob} thalweg points were out of bounds of the DEM during elevation projection")
    z = points["z"].astype(np.float32)

    profile = {"x": x, "y": y, "z": z, "s": s, "zDelta": zDelta}

    return profile


def setup2DProfileAxis(ax2, sFull, zFull, zDeltaFull, sharedCmap, norm):
    """
    Set up a 2D profile axis (elevation and z + zDelta vs.
    travel distance).

    Parameters
    ----------
    ax2 : matplotlib.axes.Axes
        Axes to set up.
    sFull, zFull, zDeltaFull : numpy.ndarray
        travel distance, elevation and zDelta arrays that are plotted.

    Returns
    -------
    zLine, zVelLine : matplotlib.lines.Line2D
        Line artists for the elevation profile and the z + zDelta
        ("velocity altitude") profile, to be updated per frame.
    currentProfilePoint : matplotlib.lines.Line2D
        Marker artist for the current generation's position on the
        profile.
    """
    # set general axis limits
    validMask = ~np.isnan(zFull)
    sMargin = 0.05 * (np.nanmax(sFull) - np.nanmin(sFull) + 1e-6)
    zMin = np.nanmin(zFull[validMask])
    zMax = np.nanmax((zFull + zDeltaFull)[validMask])
    zMargin = 0.05 * (zMax - zMin + 1e-6)

    ax2.set_xlim(np.nanmin(sFull) - sMargin, np.nanmax(sFull) + sMargin)
    ax2.set_ylim(zMin - zMargin, zMax + zMargin)
    ax2.set_xlabel("Travel distance (horizontally projected) [m]")
    ax2.set_ylabel("Elevation [m]")
    # ax2.set_title("Thalweg elevation and energy line")

    ax2.hlines(zFull[validMask][-1], 0, sFull[-1],
               colors="grey", linestyles="dotted", linewidths=1,
               label="Runout length")
    ax2.vlines(0, zFull[validMask][-1], zFull[validMask][0],
               colors="grey", linestyles="dashed", linewidths=1,
               label="Elevation drop")
    (zLine,) = ax2.plot([], [], color="black", lw=2, label="Elevation z")
    zVelDummy = ax2.plot([], [], color="blue", lw=2, label="Energy line height (+ z) \n(color indicates velocity)")

    segments = np.empty((0, 2, 2))

    (currentProfilePoint,) = ax2.plot([], [], "o", color="k", markersize=8,
                                      markeredgecolor="white", markeredgewidth=1.5,
                                      zorder=20)
    ax2.legend(loc="upper right")
    zVelLine = LineCollection(
        segments,
        cmap=sharedCmap,
        norm=norm,
        linewidth=3,
    )
    ax2.add_collection(zVelLine)

    return zLine, zVelLine, currentProfilePoint


def makeGenerationVideo(cfg=None, avalancheDir="", ax=None):
    """
    Create an animated GIF of com4FlowPy generation data for one release ID.
    The animation is saved as a GIF file to disk

    For a single release ID, renders one frame per generation
    (or per group of generations, see ``gensPerFrame``) showing:

    - a static hillshade of the DEM as background,
    - the cumulative raster values of all previous generations
    - the raster values of the current generation only and outlined with a contour line,
    - the center-of-flux or center-of-energy trajectory ("thalweg")
      accumulated up to the current generation, plus a marker at the
      current position.

    Parameters
    ----------
    cfg : configparser.ConfigParser, optional
        Full configuration object for the GIF module.
    avalancheDir : str or pathlib.Path, optional
        Path to the avalanche project directory.
    """
    if cfg is None:
        cfg = cfgUtils.getModuleConfig(outCom4Gif)
    cfgHS = cfg["HILLSHADE"]
    cfgGen = cfg["GENERAL"]
    cfgPath = cfg["PATH"]

    showProfilePanel = cfgGen.getboolean("show2DProfilePanel", fallback=True)

    if avalancheDir == "":
        # Load avalanche directory from general configuration file
        cfgMain = cfgUtils.getGeneralConfig()
        avalancheDir = cfgMain["MAIN"]["avalancheDir"]
    avalancheDir = pathlib.Path(avalancheDir)

    if cfgPath.get("demPath", fallback="") == "" or cfgPath.get("demPath", fallback="") is None:
        demFile = getInput.getDEMPath(avalancheDir)
    else:
        demFile = cfgPath["demPath"]

    resFolderUnique, simhash = searchResFolder(avalancheDir, module="com4FlowPy")

    if resFolderUnique:
        resHash = simhash
    else:
        resHash = cfgGen.get("simhash")
        if resHash == "" or resHash is None:
            message = "Please provide a valid simhash to simulation results."
            log.error(message)
            raise ValueError(message)

    outDir = avalancheDir / "Outputs" / "com4FlowPy"
    thalwegDataDir = outDir / "peakFiles" / f"res_{resHash}" / "thalwegData"
    videoDataDir = thalwegDataDir / "videoData"

    videoDataVariable = cfgGen.get("videoDataVariable")
    relId = cfgGen.get("relId")
    centerOf = cfgGen.get("centerOf")

    if videoDataVariable == "velocity":
        videoDataVariableFile = "z_delta"
    else:
        videoDataVariableFile = videoDataVariable

    npzFile = videoDataDir / f"videoData_{videoDataVariableFile}_{str(relId)}.npz"

    if showProfilePanel:
        outFile2D = "2D"
    else:
        outFile2D = ""

    if cfgPath.get("outVideoPath", fallback="") == "" or cfgPath.get("outVideoPath", fallback="") is None:
        videoOutputDir = outDir / "reports"
        outFile = videoOutputDir / f"videoData{outFile2D}_{resHash}_{videoDataVariable}_{str(relId)}_{centerOf}.gif"
    else:
        outFile = cfgPath["outVideoPath"]

    fU.makeADir(videoOutputDir)

    clabelDict = {"flux": "flux", "z_delta": "energy line height zDelta [m]", "min_distance": "travel length [m]",
                  "max_gamma": "tavel angle [°]", "velocity": "velocity [m/s]", }

    fps = cfgGen.getint("fps")
    gensPerFrame = cfgGen.getint("gensPerFrame")

    data = np.load(npzFile)
    frameStackHistory = data["framesHistory"]
    frameStackCurrent = data["framesCurrent"]

    if cfgGen.get("videoDataVariable") == "velocity":
        # compute Zdelta to velocity
        for gen, (histArr, currArr) in enumerate(zip(frameStackHistory, frameStackCurrent)):
            frameStackCurrent[gen] = (currArr * 2 * 9.81) ** 0.5
            frameStackHistory[gen] = (histArr * 2 * 9.81) ** 0.5

    demDict = rasterUtils.readRaster(demFile)

    if centerOf.lower() == "coe":
        coRow = data["rowCoE"]
        coCol = data["colCoE"]
        centerOfPickle = "CoE"
    else:
        coRow = data["rowCoF"]
        coCol = data["colCoF"]
        centerOfPickle = "CoF"

    if gensPerFrame > 1:
        nFramesOut = int(np.ceil(len(frameStackHistory) / gensPerFrame))
        frameStackHistory = frameStackHistory[::gensPerFrame][:nFramesOut]
        frameStackCurrent = frameStackCurrent[::gensPerFrame][:nFramesOut]
        coRow = coRow[::gensPerFrame][:nFramesOut]
        coCol = coCol[::gensPerFrame][:nFramesOut]

    if showProfilePanel:
        # read thalweg data and also use subset (generation per frame)
        profilePicklePath = thalwegDataDir / f"thalwegData_{centerOfPickle}_{relId}.pickle"
        thalwegData = np.load(profilePicklePath, allow_pickle=True)

        profile = getProfileData(
            demDict, thalwegData
        )

        if gensPerFrame > 1:
            sProfile = profile["s"][::gensPerFrame][:nFramesOut]
            zProfile = profile["z"][::gensPerFrame][:nFramesOut]
            zDeltaProfile = profile["zDelta"][::gensPerFrame][:nFramesOut]

    dem = np.flipud(demDict["rasterData"])
    cellSize = demDict["header"]["cellsize"]
    hillshade = addHillshadeSimple(dem, cellSize, cfgHS)

    # fixed color range, computed once from the GLOBAL max across all frames
    vmin = 0
    vmax = np.nanmax(frameStackHistory)
    sharedCmap = cmapCrameri.batlow.reversed()

    norm = mcolors.Normalize(
        vmin=0,
        vmax=vmax,
        clip=True
    )

    thalwegColor = cfgGen.get("thalwegColor")
    thalwegWidth = cfgGen.getfloat("thalwegWidth")
    pointColor = cfgGen.get("pointColor")
    pointSize = cfgGen.getfloat("pointSize")
    outlineColor = cfgGen.get("outlineColor")
    outlineWidth = cfgGen.getfloat("outlineWidth")
    historyAlpha = cfgGen.getfloat("historyAlpha")
    currentAlpha = cfgGen.getfloat("currentAlpha")
    historyPointsColor = cfgGen.get("historyPointsColor")
    historyPointsAlpha = cfgGen.getfloat("historyPointsAlpha")
    historyPointsSize = cfgGen.getfloat("historyPointsSize")

    if showProfilePanel:
        fig = plt.figure(figsize=(14, 10))

        gs = GridSpec(
            1, 2,
            width_ratios=[1.0, 1.8],
            wspace=0.5
        )

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])

    else:
        fig, ax1 = plt.subplots(figsize=(8, 10))

    # mapping panel
    ax1.imshow(hillshade, cmap="gray", vmin=0, vmax=1)
    imHistory = ax1.imshow(np.where(frameStackHistory[0] > 0, frameStackHistory[0], np.nan),
                           cmap=sharedCmap, norm=norm, alpha=historyAlpha)
    imCurrent = ax1.imshow(np.where(frameStackCurrent[0] > 0, frameStackCurrent[0], np.nan),
                           cmap=sharedCmap, norm=norm, alpha=currentAlpha)

    imHistory.set_clim(vmin, vmax)
    imCurrent.set_clim(vmin, vmax)

    thalwegLine, = ax1.plot([], [], "-", color=thalwegColor, linewidth=thalwegWidth,
                            alpha=0.8, label="Thalweg (path so far)")
    historyPoints = ax1.scatter([], [], s=historyPointsSize, color=historyPointsColor,
                                alpha=historyPointsAlpha, zorder=15,
                                label="Center of flux (history)")
    currentPoint, = ax1.plot([], [], "o", color=pointColor, markersize=pointSize,
                             markeredgecolor="white", markeredgewidth=1.5,
                             label="Center of flux (current)")

    titleText = ax1.set_title("")
    ax1.legend(loc="lower left")
    ax1.set_axis_off()

    fig.canvas.draw()  # triggers layout computation

    # get proper position for colorbar
    p1 = ax1.get_position()
    if showProfilePanel:
        ax1.set_aspect("auto")
        p2 = ax2.get_position()

        ax1.set_position([
            p1.x0,
            p2.y0,
            p1.width,
            p2.height
        ])
        p1 = ax1.get_position()
        gap = p2.x0 - p1.x1
    else:
        gap = 0.1

    cax_width = 0.015
    cax = fig.add_axes([
        p1.x1 + 0.08 * gap,
        p1.y0,
        cax_width,
        p1.height * 0.95,
    ])

    gradient = np.linspace(vmin, vmax, 256).reshape(-1, 1)
    cax.imshow(gradient, aspect="auto", cmap=sharedCmap, origin="lower",
               extent=[0, 1, vmin, vmax])
    cax.set_xticks([])
    cax.yaxis.tick_right()
    cax.yaxis.set_label_position("right")
    cax.set_ylabel(clabelDict[cfgGen.get("videoDataVariable")])

    if showProfilePanel:
        # --- right panel: growing z / zDelta profile ------------------------
        zLine, zVelLine, currentProfilePoint = setup2DProfileAxis(ax2, sProfile, zProfile, zDeltaProfile, sharedCmap,
                                                                  norm)

    contourHolder = {"artist": None}


    def update(gen):
        history = frameStackHistory[gen]
        current = frameStackCurrent[gen]

        imHistory.set_data(np.where(history > 0, history, np.nan))
        imCurrent.set_data(np.where(current > 0, current, np.nan))
        imHistory.set_clim(vmin, vmax)
        imCurrent.set_clim(vmin, vmax)

        removeContour(contourHolder)
        mask = (current > 0).astype(np.float32)
        if mask.max() > 0:
            contourHolder["artist"] = ax1.contour(mask, levels=[0.5],
                                                  colors=outlineColor,
                                                  linewidths=outlineWidth)

        validMask = ~np.isnan(coRow[: gen + 1])
        thalwegLine.set_data(coCol[: gen + 1][validMask], coRow[: gen + 1][validMask])

        historyMask = ~np.isnan(coRow[:gen])
        historyOffsets = np.column_stack([coCol[:gen][historyMask], coRow[:gen][historyMask]])
        historyPoints.set_offsets(historyOffsets)

        currentPoint.set_data([coCol[gen]], [coRow[gen]])
        titleText.set_text(f"Generation {gen * gensPerFrame}")

        artists = (imHistory, imCurrent, thalwegLine, historyPoints, currentPoint, titleText)

        if showProfilePanel:
            zLine.set_data(sProfile[: gen + 1], zProfile[: gen + 1])

            x = sProfile[:gen + 1]
            y = zProfile[:gen + 1] + zDeltaProfile[:gen + 1]

            points = np.column_stack([x, y]).reshape(-1, 1, 2)

            segments = np.concatenate(
                [points[:-1], points[1:]],
                axis=1,
            )

            zVelLine.set_segments(segments)

            if videoDataVariable == "velocity":
                values = np.sqrt(
                    2 * 9.81 * zDeltaProfile[:gen]
                )
            else:
                values = zDeltaProfile[:gen]

            zVelLine.set_array(values)

            currentProfilePoint.set_data([sProfile[gen]], [zProfile[gen] + zDeltaProfile[gen]])
            artists += (zLine, zVelLine, currentProfilePoint)

        return artists

    frameArrays = []
    for gen in range(len(frameStackHistory)):
        update(gen)
        frameArrays.append(figureToRgbArray(fig))

    # --- final extra frame: last generation state + thalweg extension ---
    if cfgGen.getboolean("showExtendedThalweg"):
        if cfgPath.get("extendedProfilePicklePath", "") == "":
            extendedProfilePicklePath = thalwegDataDir / f"extended_thalwegData_{centerOfPickle}_{relId}.pickle"
        else:
            extendedProfilePicklePath = cfgPath.get("extendedProfilePicklePath")
        avaProfileMass = loadExtendedProfile(extendedProfilePicklePath)
        topLine, bottomLine, centerLine = addThalwegExtension(ax1, avaProfileMass, demDict["header"])
        ax1.legend(
            handles=[topLine, bottomLine, centerLine],
            loc="lower left")
        fig.canvas.draw()
        frameArrays.append(figureToRgbArray(fig))  # just ONE extra frame now
    # -----------------------------------------------------------------

    plt.close(fig)

    basePalette = buildGlobalPalette(frameArrays)
    pilFrames = []
    for arr in frameArrays:
        img = Image.fromarray(arr, mode="RGB")
        imgQuantized = img.quantize(palette=basePalette, dither=Image.NONE)
        pilFrames.append(imgQuantized)

    normalDuration = int(1000 / fps)
    finalFrameDuration = cfgGen.getint("finalFrameDurationMs", fallback=3000)  # ms, e.g. 3 seconds

    if cfgGen.getboolean("showExtendedThalweg"):
        # last frame (the extension frame) gets a longer duration
        durations = [normalDuration] * (len(pilFrames) - 1) + [finalFrameDuration]
    else:
        durations = [normalDuration] * len(pilFrames)

    pilFrames[0].save(
        outFile,
        save_all=True,
        append_images=pilFrames[1:],
        duration=durations,
        loop=0,  # play once, do not loop; use loop=0 for infinite looping
    )

    print(f"Video saved: {outFile}")
