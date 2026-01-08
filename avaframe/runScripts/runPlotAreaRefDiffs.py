"""
Run script for plotting a comparison of simulation result to reference polygon
"""

# Load modules
# importing general python modules
import pathlib
import numpy as np

# Local imports
from avaframe.in3Utils import cfgUtils
from avaframe.in3Utils import logUtils
from avaframe.in3Utils import fileHandlerUtils as fU
import avaframe.in2Trans.rasterUtils as IOf
import avaframe.out1Peak.outPlotAllPeakDiffs as oPD
import avaframe.in1Data.getInput as gI
import avaframe.in2Trans.shpConversion as shpConv
import avaframe.in3Utils.geoTrans as gT
import avaframe.com1DFA.DFAtools as DFAtls


################USER Input#############
resType = "ppr"
thresholdValueSimulation = 0.9
modName = "com1DFA"
############################################################

# Load avalanche directory from general configuration file
cfgMain = cfgUtils.getGeneralConfig()
avalancheDir = cfgMain["MAIN"]["avalancheDir"]
outDir = pathlib.Path(avalancheDir, "Outputs", "out1Peak")
fU.makeADir(outDir)

# Start logging
logName = "plotAreaDiff_%s" % (resType)
# Start logging
log = logUtils.initiateLogger(avalancheDir, logName)
log.info("MAIN SCRIPT")
log.info("Current avalanche: %s", avalancheDir)


# initialize DEM from avalancheDir (used to perform simulations)
# TODO: if meshCellSize was changed - use actual simulation DEM
dem = gI.readDEM(avalancheDir)
# get normal vector of the grid mesh
dem = gT.getNormalMesh(dem, num=1)
# get real Area
dem = DFAtls.getAreaMesh(dem, 1)
dem["originalHeader"] = dem["header"]

# read reference data set
inDir = pathlib.Path(avalancheDir, "Inputs")
referenceFile, availableFile, _ = gI.getAndCheckInputFiles(
    inDir, "REFDATA", "POLY", fileExt="shp", fileSuffix="POLY"
)
# convert polygon to raster with value 1 inside polygon and 0 outside the polygon
referenceLine = shpConv.readLine(referenceFile, "reference", dem)
referenceLine = gT.prepareArea(referenceLine, dem, np.sqrt(2), combine=True, checkOverlap=False)

# if available zoom into area provided by crop shp file in Inputs/CROPSHAPE
cropFile, cropInfo, _ = gI.getAndCheckInputFiles(
    inDir, "POLYGONS", "cropFile", fileExt="shp", fileSuffix="_cropshape"
)
if cropInfo:
    cropLine = shpConv.readLine(cropFile, "cropFile", dem)
    cropLine = gT.prepareArea(cropLine, dem, np.sqrt(2), combine=True, checkOverlap=False)

if modName in ["com1DFA", "com5SnowSlide", "com6RockAvalanche", "com8MoTPSA", "com9MoTVoellmy"]:
    # load dataFrame for all configurations of simulations in avalancheDir
    simDF = cfgUtils.createConfigurationInfo(avalancheDir)
    # create data frame that lists all available simulations and path to their result type result files
    inputsDF, resTypeList = fU.makeSimFromResDF(avalancheDir, "com1DFA")
    # merge  parameters as columns to dataDF for matching simNames
    dataDF = inputsDF.merge(simDF, left_on="simName", right_on="simName")

    ## loop over all simulations and load desired resType
    for index, row in dataDF.iterrows():
        simFile = row[resType]
        simData = IOf.readRaster(simFile)

    # compute referenceMask and simulationMask and true positive, false positive and false neg. arrays
    # here thresholdValueReference is set to 0.9 as when converting the polygon to a raster,
    # values inside polygon are set to 1 and outside to 0
    refMask, compMask, indicatorDict = oPD.computeAreaDiff(
        referenceLine["rasterData"],
        simData["rasterData"],
        0.9,
        thresholdValueSimulation,
        dem,
        cropToArea=cropLine["rasterData"],
    )

    # plot differences
    oPD.plotAreaDiff(
        referenceLine["rasterData"],
        refMask,
        simData["rasterData"],
        compMask,
        resType,
        simData["header"],
        thresholdValueSimulation,
        outDir,
        indicatorDict,
        row["simName"],
        cropFile=cropFile,
    )
else:
    # load all result files
    resultDir = pathlib.Path(avalancheDir, "Outputs", modName, "peakFiles")
    peakFilesList = list(resultDir.glob("*_%s.tif" % resType)) + list(resultDir.glob("*_%s.asc" % resType))
    for pF in peakFilesList:
        simData = IOf.readRaster(pF)
        simName = pF.stem

        # compute referenceMask and simulationMask and true positive, false positive and false neg. arrays
        refMask, compMask, indicatorDict = oPD.computeAreaDiff(
            referenceLine["rasterData"],
            simData["rasterData"],
            0.9,
            thresholdValueSimulation,
            dem,
            cropToArea=cropLine["rasterData"],
        )

        # plot differences
        oPD.plotAreaDiff(
            referenceLine["rasterData"],
            refMask,
            simData["rasterData"],
            compMask,
            resType,
            simData["header"],
            thresholdValueSimulation,
            outDir,
            indicatorDict,
            simName,
            cropFile=cropFile,
        )
