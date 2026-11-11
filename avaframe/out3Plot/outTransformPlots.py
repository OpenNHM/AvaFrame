"""
Functions to create plots for visualizing transformation steps

"""

import matplotlib.pyplot as plt
import logging
import seaborn as sns

sns.set_theme()
import pathlib

# Local imports
import avaframe.out3Plot.plotUtils as pU

# create local logger
# change log level in calling module to DEBUG to see log messages
log = logging.getLogger(__name__)


def plotDepthToThickness(depth, thickness, slopeAngleField, profileAxis, profileIndex, outDir, pName):
    """plot depth, thickness, diff depth to thickness and profile

    Parameters
    -----------
    depth: numpy nd array
        depth field
    thickness: numpy nd array
        thickness field (slope normal)
    slopeAngleField: numpy nd array
        slope angle values from DEM
    profileAxis: str
        name of axis (x or y)
    profileIndex: int
        index for profile of fields
    outDir: pathlib Path
        path to directory to save plot
    pName: str
        name of plot
    """

    # if profile indices are None choose approx. center of fields
    if profileIndex is None:
        if profileAxis == "x":
            profileIndex = int(depth.shape[0] * 0.5)
        else:
            profileIndex = int(depth.shape[1] * 0.5)

    if profileAxis == "x":
        depthProfile = depth[profileIndex, :]
        thicknessProfile = thickness[profileIndex, :]
        slopeProfile = slopeAngleField[profileIndex, :]
    else:
        depthProfile = depth[:, profileIndex]
        thicknessProfile = thickness[:, profileIndex]
        slopeProfile = slopeAngleField[:, profileIndex]

    # setup figure
    fig1, ax = plt.subplots(ncols=2, nrows=2, figsize=(pU.figW * 3, pU.figH * 2))
    # plot fields
    im1 = ax[0, 0].imshow(depth, label="depth")
    im2 = ax[0, 1].imshow(thickness, label="thickness")
    im3 = ax[1, 0].imshow(depth - thickness, label="depth - thickness")
    ax[0, 0].set_title("Depth")
    ax[0, 1].set_title("Thickness")
    ax[1, 0].set_title("Difference depth-thickness")
    fig1.colorbar(im1, ax=ax[0, 0])
    fig1.colorbar(im2, ax=ax[0, 1])
    fig1.colorbar(im3, ax=ax[1, 0])

    # plot profile
    ax[1, 1].plot(depthProfile, label="depth")
    ax[1, 1].plot(thicknessProfile, label="thickness")

    # add slope angle profile
    ax3 = ax[1, 1].twinx()
    ax3.plot(
        slopeProfile,
        color="grey",
        linestyle="--",
        label="slope angle",
    )
    ax[1, 1].set_xlabel("x")
    ax3.set_ylabel("slope angle (°)")
    ax[1, 1].set_ylabel("flow thickness (m)")
    ax[1, 1].legend()

    # save and or show figure
    pU.saveAndOrPlot({"pathResult": outDir}, pName, fig1)


def plotFetchedValues(avalancheDir, dataDF, xName, yName, plotName, scenario=None):
    """plot values of dataDF

    Parameters
    -----------
    dataDF: pandas dataFrame
        dataframe with simulation info and pointValues
    xName: str
        name of column used for x axis
    yName: str
        name of column used for y axis
    plotName: str
        name of plot
    scenario: str
        name of column used for coloring
    """

    if dataDF[xName].dtype.name == "object" or dataDF[yName].dtype.name == "object":
        g = sns.catplot(data=dataDF, x=xName, y=yName, hue=scenario)
    else:
        g = sns.jointplot(data=dataDF, x=xName, y=yName, hue=scenario)
    outDir = pathlib.Path(avalancheDir, "Outputs", "out1Peak")

    pU.saveAndOrPlot({"pathResult": outDir}, plotName, g.fig)
