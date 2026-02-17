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
simDF = cfgUtils.createConfigurationInfo(avalancheDir)

# create data frame that lists all available simulations and path to their result type result files
inputsDF, resTypeList = fU.makeSimFromResDF(avalancheDir, "com1DFA")
# merge  parameters as columns to dataDF for matching simNames
dataDF = inputsDF.merge(simDF, left_on="simName", right_on="simName")

# fetch unique values of varPar
values = sorted(set(dataDF[varPar].to_list()))
log.info('%s values used for colorcoding are: %s' % (varPar, values))
minVal = np.nanmin(values)
maxVal = np.nanmax(values)
cmapSCVals = np.linspace(0, 1, len(values))
# create colormap and setup ticks and itemsList
# strip layer suffix for unit lookup (e.g. pfv_l1 -> pfv)
baseResType = re.sub(r"_l\d+$", "", resType)
unit = pU.cfgPlotUtils['unit' + baseResType]
cmapSC, colorSC, ticksSC, normSC, unitSC, itemsList, displayColorBar = pU.getColors4Scatter(values, len(values),
                                                                                                unit)
# create figure
fig = plt.figure(figsize=(pU.figW * 2, pU.figH * 1.5))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
testField = IOf.readRaster(dataDF[resType].iloc[0])
nrowsHalf = int(testField['rasterData'].shape[0] * 0.5)
log.info('Dimension of raster: %d, %d' % (testField['rasterData'].shape[0], testField['rasterData'].shape[1]))
log.info('Profile plotted for row %d' % nrowsHalf)
for index, row in dataDF.iterrows():
    # read field
    field = IOf.readRaster(row[resType])
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

