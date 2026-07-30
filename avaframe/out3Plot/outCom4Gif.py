""" functions to create a GIF for com4FlowPys generation data"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import logging
import os
import pathlib
from matplotlib.colors import LightSource
from PIL import Image

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
        print(str(resFolders[0]).split("res_", 1))
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


def buildGlobalPalette(frameArrays, colors=256, sampleEvery=1):
    """
    Build a single shared color palette from a sample of frames.

    Using one global palette (instead of letting PIL quantize each GIF
    frame independently) prevents visible color flicker between frames,
    since every frame is then mapped onto the exact same set of colors.

    Parameters
    ----------
    frameArrays : list of numpy.ndarray
        List of RGB frame arrays, each of shape (height, width, 3).
    colors : int, optional
        Number of palette colors (GIF hard limit is 256). Default 256.
    sampleEvery : int, optional
        Use only every Nth frame to build the palette (speeds this up
        for long animations; the color content across frames from this
        kind of plot is similar enough that subsampling is fine).

    Returns
    -------
    paletteImg: PIL.Image.Image
        A palette-mode image whose palette should be reused for all
        frames via ``Image.quantize(palette=...)``.
    """
    sampledFrames = frameArrays[::max(1, sampleEvery)]
    stacked = np.concatenate([f.reshape(-1, 3) for f in sampledFrames], axis=0)
    # PIL needs a 2D "image" to quantize from
    paletteSourceImg = Image.fromarray(stacked.reshape(1, -1, 3).astype(np.uint8), mode="RGB")
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

    ax.plot(
        colProfile[: indStart + 1],
        rowProfile[: indStart + 1],
        "-b.",
        # color="blue",
        zorder=20,
        label="top extension",
        lw=2.5,
        path_effects=[pe.Stroke(linewidth=2, foreground="b"), pe.Normal()],
    )
    ax.plot(
        colProfile[indEnd:],
        rowProfile[indEnd:],
        "-g.",
        # color="green",
        zorder=20,
        label="bottom extension",
        lw=2.5,
        path_effects=[pe.Stroke(linewidth=2, foreground="g"), pe.Normal()],
    )
    ax.plot(
        colProfile[indStart: indEnd + 1],
        rowProfile[indStart: indEnd + 1],
        "-k.",
        # color="black",
        zorder=20,
        label="center of flux path",
        lw=2.5,
        path_effects=[pe.Stroke(linewidth=2, foreground="k"), pe.Normal()],
    )


def makeGenerationVideo(cfg=None, avalancheDir=""):
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

    if cfgPath.get("outVideoPath", fallback="") == "" or cfgPath.get("outVideoPath", fallback="") is None:
        videoOutputDir = outDir / "reports"
        outFile = videoOutputDir / f"videoData_{resHash}_{videoDataVariable}_{str(relId)}_{centerOf}.gif"
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

    if centerOf.lower() == "coe":
        coRow = data["rowCoE"]
        coCol = data["colCoE"]
    else:
        coRow = data["rowCoF"]
        coCol = data["colCoF"]

    demDict = rasterUtils.readRaster(demFile)
    dem = np.flipud(demDict["rasterData"])
    cellSize = demDict["header"]["cellsize"]
    hillshade = addHillshadeSimple(dem, cellSize,
                                   cfgHS)

    if gensPerFrame > 1:
        nFramesOut = int(np.ceil(len(frameStackHistory) / gensPerFrame))
        frameStackHistory = frameStackHistory[::gensPerFrame][:nFramesOut]
        frameStackCurrent = frameStackCurrent[::gensPerFrame][:nFramesOut]
        coRow = coRow[::gensPerFrame][:nFramesOut]
        coCol = coCol[::gensPerFrame][:nFramesOut]

    # fixed color range, computed once from the GLOBAL max across all frames
    vmin = 0
    vmax = np.nanmax(frameStackHistory)

    sharedCmap = plt.get_cmap(cfgGen.get("cmap"))

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

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(hillshade, cmap="gray", vmin=0, vmax=1)

    imHistory = ax.imshow(np.where(frameStackHistory[0] > 0, frameStackHistory[0], np.nan),
                          cmap=sharedCmap, norm=norm, alpha=historyAlpha)
    imCurrent = ax.imshow(np.where(frameStackCurrent[0] > 0, frameStackCurrent[0], np.nan),
                          cmap=sharedCmap, norm=norm, alpha=currentAlpha)

    thalwegLine, = ax.plot([], [], "-", color=thalwegColor, linewidth=thalwegWidth,
                           alpha=0.8, label="Thalweg")
    historyPoints = ax.scatter([], [], s=historyPointsSize, color=historyPointsColor,
                               alpha=historyPointsAlpha, zorder=15,
                               label="Center of flux (previous)")
    currentPoint, = ax.plot([], [], "o", color=pointColor, markersize=pointSize,
                            markeredgecolor="white", markeredgewidth=1.5,
                            label="Center of flux (current generation)")
    imHistory.set_clim(vmin, vmax)
    imCurrent.set_clim(vmin, vmax)

    titleText = ax.set_title("")
    ax.legend(loc="upper right")
    ax.set_axis_off()

    # --- manual, fully static colorbar -------------------------------
    # built as its own small axes with a fixed gradient image; never
    # touched again after this point, so it is guaranteed not to change
    fig.subplots_adjust(right=0.85)
    cax = fig.add_axes([0.88, 0.15, 0.03, 0.7])  # [left, bottom, width, height]
    gradient = np.linspace(vmin, vmax, 256).reshape(-1, 1)
    cax.imshow(gradient, aspect="auto", cmap=sharedCmap,
               origin="lower", extent=[0, 1, vmin, vmax])
    cax.set_xticks([])
    cax.yaxis.tick_right()
    cax.yaxis.set_label_position("right")

    cax.set_ylabel(clabelDict[cfgGen.get("videoDataVariable")])
    # --------------------------------------------------------------
    contourHolder = {"artist": None}
    frameArrays = []

    def update(gen):
        """
        Update all animated artists for a single animation frame (generation).

        Parameters
        ----------
        gen : int
            Index of the current animation frame, corresponding to the
            (possibly subsampled) generation index.

        Returns
        -------
        imHistory: matplotlib.artist.Artist
            updated history imshow raster
        imCurrent: matplotlib.artist.Artist
            updated current imshow raster
        thalwegLine: matplotlib.artist.Artist
            updated thalweg line array
        currentPoint: matplotlib.artist.Artist
            updated current point
        titleText: matplotlib.artist.Artist
            updated title text
        """
        history = frameStackHistory[gen]
        current = frameStackCurrent[gen]

        imHistory.set_data(np.where(history > 0, history, np.nan))
        imCurrent.set_data(np.where(current > 0, current, np.nan))

        imHistory.set_clim(vmin, vmax)
        imCurrent.set_clim(vmin, vmax)

        removeContour(contourHolder)
        mask = (current > 0).astype(np.float32)
        if mask.max() > 0:
            contourHolder["artist"] = ax.contour(mask, levels=[0.5],
                                                 colors=outlineColor,
                                                 linewidths=outlineWidth)

        validMask = ~np.isnan(coRow[: gen + 1])
        thalwegLine.set_data(coCol[: gen + 1][validMask], coRow[: gen + 1][validMask])

        historyMask = ~np.isnan(coRow[:gen])
        historyOffsets = np.column_stack([coCol[:gen][historyMask], coRow[:gen][historyMask]])
        historyPoints.set_offsets(historyOffsets)

        currentPoint.set_data([coCol[gen]], [coRow[gen]])
        titleText.set_text(f"Generation {gen * gensPerFrame}")
        return imHistory, imCurrent, thalwegLine, currentPoint, titleText

    for gen in range(len(frameStackHistory)):
        update(gen)
        frameArrays.append(figureToRgbArray(fig))

    # --- final extra frame: last generation state + thalweg extension ---
    if cfgGen.getboolean("showExtendedThalweg"):
        if cfgPath.get("extendedProfilePicklePath", "") == "":
            if centerOf.lower() == "coe":
                centerOfExt = "CoE"
            else:
                centerOfExt = "CoF"
            extendedProfilePicklePath = thalwegDataDir / f"extended_thalwegData_{centerOfExt}_{relId}.pickle"
        else:
            extendedProfilePicklePath = cfgPath.get("extendedProfilePicklePath")
        avaProfileMass = loadExtendedProfile(extendedProfilePicklePath)
        addThalwegExtension(ax, avaProfileMass, demDict["header"])
        ax.legend(loc="upper right")
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
        loop=1,  # play once, do not loop; use loop=0 for infinite looping
    )

    print(f"Video saved: {outFile}")
