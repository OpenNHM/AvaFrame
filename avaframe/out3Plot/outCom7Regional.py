import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle, Patch
from shapely import MultiPolygon

from avaframe.in2Trans import rasterUtils
from avaframe.out3Plot import plotUtils as pU


def createReportPlot(dirListGrouped, inputDEM, outputDir, groupExtents, groupFeatures, reportType):
    """Create a visual report showing the DEM extent with either basic or optional inputs.

    Parameters
    ----------
    dirListGrouped : list
        List of dictionaries containing dirName and list of geometries
    inputDEM : pathlib.Path
        Path to input DEM file
    outputDir : pathlib.Path
        Path to output directory where the report will be saved
    groupExtents : dict
        Dictionary with dirName as key and (xMin, xMax, yMin, yMax) as value,
        containing the DEM clipping extents for each group
    groupFeatures : dict
        Dictionary containing clipped features for each group and type
    reportType : str
        Type of report to create, either 'basic' or 'optional'
        - 'basic': Shows DEM extent and release areas only
        - 'optional': Shows DEM extent with entrainment and resistance areas

    Returns
    -------
    reportPath: pathlib.Path
        Path to the generated report image
    """
    # Set up figure
    plt.figure(figsize=(10, 8))
    ax = plt.subplot(1, 1, 1)

    # Read and plot DEM
    demData = rasterUtils.readRaster(inputDEM)
    header = demData["header"]
    cellSize = header["cellsize"]
    xMin = header["xllcenter"]
    yMin = header["yllcenter"]
    xMax = xMin + cellSize * header["ncols"]
    yMax = yMin + cellSize * header["nrows"]
    im = ax.imshow(
        demData["rasterData"],
        extent=[xMin, xMax, yMin, yMax],
        cmap=pU.cmapDEM.reversed(),
        alpha=1,
        origin="lower",
        zorder=1,
    )

    # Custom color scheme for groups
    colors = [
        "#0EF8EA",
        "#FFA500",
        "#C71585",
        "#00FF00",
        "#FF4500",
        "#800080",
        "#ADFF2F",
        "#FF6347",
        "#8A2BE2",
        "#FFFF00",
        "#FF0000",
    ]

    # Plot groups
    for idx, group in enumerate(dirListGrouped):
        dirName = group["dirName"]
        color = colors[idx % len(colors)]
        # Plot DEM extent using groupExtents
        if dirName in groupExtents:
            xMin, xMax, yMin, yMax = groupExtents[dirName]
            rect = Rectangle(
                (xMin, yMin),
                xMax - xMin,
                yMax - yMin,
                fill=False,
                linestyle="--",
                color=color,
            )
            ax.add_patch(rect)

            if reportType == "basic":
                # Plot release areas
                for geom in group["geometries"]:
                    for x, y in getExteriorCoords(geom):
                        plt.fill(x, y, alpha=1.0, color=color)
            else:
                # Plot optional features
                if dirName in groupFeatures:
                    for geom in groupFeatures[dirName].get("ENT", []):
                        for x, y in getExteriorCoords(geom):
                            plt.fill(x, y, alpha=0.3, color=color, edgecolor="none")
                    for geom in groupFeatures[dirName].get("RES", []):
                        for x, y in getExteriorCoords(geom):
                            plt.fill(
                                x,
                                y,
                                alpha=0.5,
                                color=color,
                                hatch="xxxx",
                                fill=False,
                                edgecolor=color,
                                linewidth=0.5,
                            )

            # Place group label using groupExtents
            plt.text(
                xMin,
                yMax,
                dirName,
                color=color,
                fontsize=8,
                transform=mpl.transforms.offset_copy(ax.transData, x=1, y=-7, units="points", fig=ax.figure),
            )

    # Create legend
    mapElements = [
        Rectangle(
            (0, 0),
            1,
            1,
            fill=False,
            linestyle="--",
            color="black",
            label="Clipped DEM Extent",
        )
    ]
    if reportType == "basic":
        mapElements.append(Patch(facecolor="black", alpha=1.0, label="Release Areas"))
    else:  # optional
        mapElements.extend(
            [
                Patch(facecolor="black", alpha=0.3, label="Entrainment Areas"),
                Patch(
                    facecolor="none",
                    alpha=0.3,
                    hatch="xxxx",
                    label="Resistance Areas",
                    edgecolor="black",
                    linewidth=0.5,
                ),
            ]
        )

    plt.legend(handles=mapElements, title="Legend", loc="center left", bbox_to_anchor=(1, 0.5))

    # Add DEM colorbar
    cax = ax.inset_axes([1.015, 0, 0.375, 0.02])  # [x, y, width, height]
    plt.colorbar(im, cax=cax, orientation="horizontal", label="Elevation (m)")

    # Format plot; add title and labels
    ax.set_aspect("equal")
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
    ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useOffset=False))
    regionalDir = inputDEM.parent.parent.name
    reportName = "Basic" if reportType == "basic" else "Optional"
    plt.title(f"Split Inputs Report - {reportName} - {regionalDir}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True, linestyle="--", alpha=0.3)

    # Set axis limits based on DEM extents with a small margin
    xMins = [ext[0] for ext in groupExtents.values()]
    xMaxs = [ext[1] for ext in groupExtents.values()]
    yMins = [ext[2] for ext in groupExtents.values()]
    yMaxs = [ext[3] for ext in groupExtents.values()]
    xMin, xMax = min(xMins), max(xMaxs)
    yMin, yMax = min(yMins), max(yMaxs)
    margin = 0.01
    dx = (xMax - xMin) * margin
    dy = (yMax - yMin) * margin
    ax.set_xlim(xMin - dx, xMax + dx)
    ax.set_ylim(yMin - dy, yMax + dy)

    reportPath = outputDir / f"splitInputs_visualReport_{reportType}.png"
    plt.savefig(reportPath, dpi=200, bbox_inches="tight")
    plt.close()

    return reportPath


def getExteriorCoords(geom):
    """Get the exterior coordinates of a shapely geometry to handle both single and multi-polygon geometries.

    Parameters
    ----------
    geom : shapely.geometry
        The shapely geometry to get the exterior coordinates from.

    Returns
    -------
    list
        A list of tuples containing the x and y coordinates of the geometry exterior.
    """
    if isinstance(geom, MultiPolygon):
        return [poly.exterior.xy for poly in geom.geoms]
    else:
        return [geom.exterior.xy]
