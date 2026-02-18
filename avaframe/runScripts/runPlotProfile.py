"""
    Run script for plotting a profile along the middle of the y-axis for peak files within an avalancheDir,
    all peakfields need to have same spatial extent 
"""
# Load modules
# importing general python modules
import pathlib
import re
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


# Local imports
from avaframe.in3Utils import cfgUtils
from avaframe.in3Utils import logUtils
from avaframe.in3Utils import fileHandlerUtils as fU
from avaframe.out3Plot import plotUtils as pU
import avaframe.in2Trans.rasterUtils as IOf

################USER Input#############
varPar = "relTh0"
resType = "pfv"
layer = ""
modName = "com1DFA"
############################################################

# Load avalanche directory from general configuration file
cfgMain = cfgUtils.getGeneralConfig()
avalancheDir = cfgMain["MAIN"]["avalancheDir"]
outDir = pathlib.Path(avalancheDir, 'Outputs', '../out3Plot')

# Start logging
logName = "plotProfiles_%s_%s" % (resType, varPar)
# Start logging
log = logUtils.initiateLogger(avalancheDir, logName)
log.info("MAIN SCRIPT")
log.info("Current avalanche: %s", avalancheDir)

# load dataFrame for all configurations
simDF = cfgUtils.createConfigurationInfo(avalancheDir, comModule=modName)

# create data frame that lists all available simulations and path to their result type result files
inputsDF, resTypeList = fU.makeSimFromResDF(avalancheDir, comModule=modName)
# merge  parameters as columns to dataDF for matching simNames
dataDF = inputsDF.merge(simDF, left_on="simName", right_on="simName")

# check for multilayer results
# fetch all result files
peakFilesDF = fU.makeSimDF(pathlib.Path(avalancheDir, "Outputs", modName, "peakFiles"), avaDir=avalancheDir)
hasMultiLayer = any(peakFilesDF["layer"] != "")
if hasMultiLayer and layer == "":
    message = (
        "Multi-layer results detected but no layer specified. "
        "Set 'layer' in the beginning of the script (e.g. layer = L1)"
    )
    log.error(message)
    raise ValueError(message)
elif hasMultiLayer and layer != "":
    resTypeFile = resType + "_" + layer.lower()
else:
    resTypeFile = resType

# fetch unique values of varPar
values = sorted(set(dataDF[varPar].to_list()))
log.info('%s values used for colorcoding are: %s' % (varPar, values))
minVal = np.nanmin(values)
maxVal = np.nanmax(values)
cmapSCVals = np.linspace(0, 1, len(values))
# create colormap and setup ticks and itemsList
unit = pU.cfgPlotUtils["unit" + resType]
cmapSC, colorSC, ticksSC, normSC, unitSC, itemsList, displayColorBar = pU.getColors4Scatter(values, len(values),
                                                                                                unit)
# create figure
fig = plt.figure(figsize=(pU.figW * 2, pU.figH * 1.5))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
testField = IOf.readRaster(dataDF[resTypeFile].iloc[0])
nrowsHalf = int(testField['rasterData'].shape[0] * 0.5)
log.info('Dimension of raster: %d, %d' % (testField['rasterData'].shape[0], testField['rasterData'].shape[1]))
log.info('Profile plotted for row %d' % nrowsHalf)
for index, row in dataDF.iterrows():
    # read field
    field = IOf.readRaster(row[resTypeFile])
    fieldData = field['rasterData']
    cmapVal = cmapSCVals[values.index(row[varPar])]
    ax1.plot(fieldData[nrowsHalf, :],  c=cmapSC(cmapVal))

ax1.set_xlabel('x axis')
ax1.set_ylabel('%s' % resType)
cmapSC2 = ScalarMappable(norm=Normalize(minVal, maxVal), cmap=cmapSC)
cbar = ax1.figure.colorbar(cmapSC2, ax=ax1)
cbar.outline.set_visible(False)
cbar.ax.set_title('[' + unit + ']', pad=10)
cbar.set_label(varPar)
ax2.imshow(testField['rasterData'])
ax2.hlines(nrowsHalf, 0, testField['rasterData'].shape[1], "white", label="profile location")
ax2.legend()
pU.saveAndOrPlot({"pathResult": outDir}, ('profile_%s_%s' % (resType, varPar)), fig)
