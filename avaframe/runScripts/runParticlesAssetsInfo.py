"""
Run creating a raster with numerical particle trajectories colorcoded with assets classes, highest overrides lower classes
"""

import pathlib
import time
import configparser

# Local imports
import avaframe.in1Data.getInput as gI
from avaframe.in3Utils import cfgUtils
import avaframe.out3Plot.outParticlesAnalysis as oP
import avaframe.com1DFA.particleTools as pT
import avaframe.in2Trans.rasterUtils as rU
import avaframe.in3Utils.fileHandlerUtils as fU
from avaframe.in3Utils import logUtils

# +++++++++REQUIRED+++++++++++++
# set the code for parts of particle trajectories that do not affect any assets class
# if not provided, default -1 is used
noAssetsClass = -1
# if particle locations are saved e.g. every second, the resulting assets raster might
# show gaps as particles travelled further within this time step, to avoid these gaps
# option to perform interpolation - can lead to errors if particle locations too spaced out
# recommendation is particle saving time step every second
interpolateParticlesTrajectoriesFlag = True
# if interpolateParticlesTrajectoriesFlag the mesh cellsize x cellSizeFactor will give desired
# distance of interpolated points
cellSizeFactor = 0.5
resizeThreshold = 3
meshCellSizeThreshold = 0.001
useCompression = True # only active with .tif outputFileType, no .asc compression
remeshInterpMethod = "nearest"
# ++++++++++++++++++++++++++++++

# +++++++++OPTIONAL+++++++++++++
# choose whether outputFileType = '.tif', '.asc' or 'default'
# 'default' resolves to the file type of the utilized input DEM
outputFileType = '.tif'
# ++++++++++++++++++++++++++++++

# load avalanche directory
cfgMain = cfgUtils.getGeneralConfig()
avalancheDir = cfgMain["MAIN"]["avalancheDir"]
outDir = pathlib.Path(avalancheDir, "Outputs", "out3Plot", "particleAnalysis")
fU.makeADir(outDir)
cfg = configparser.ConfigParser()
cfg["GENERAL"] = {
    "avalancheDir": avalancheDir,
    "meshCellSizeThreshold": meshCellSizeThreshold,
    "resizeThreshold": resizeThreshold,
    "remeshInterpMethod": remeshInterpMethod,
}
cfg["EXPORTS"] = {"useCompression": useCompression}

# log file name; leave empty to use default runLog.log
logName = "runParticlesAssetsInfo"
# Start logging
log = logUtils.initiateLogger(avalancheDir, logName)
log.info("MAIN SCRIPT")
log.info("Current avalanche: %s", avalancheDir)

# create data frame that lists all available simulations
inputsDF, resTypeList = fU.makeSimFromResDF(avalancheDir, "com1DFA")
# load dataFrame for all configurations
configurationDF = cfgUtils.createConfigurationInfo(avalancheDir, comModule="com1DFA")
# Merge inputsDF with the configurationDF. Make sure to keep the indexing from inputs and to merge on 'simName'
inputsDF = inputsDF.reset_index().merge(configurationDF, on=["simName", "modelType"]).set_index("index")

# loop over all sims found
for index, row in inputsDF.iterrows():
    startTime = time.time()
    # fetch simName
    simName = row["simName"]
    dem = rU.readRaster(pathlib.Path(avalancheDir, "Inputs", row["DEM"]))
    log.info("Find particle trajectories for simulation: %s" % (simName))
    # add info on meshCellSize
    cfg["GENERAL"]["meshCellSize"] = str(dem["header"]["cellsize"])

    # fetch info on infrastructure
    uniqueAssets, assets, assetsValues = gI.preprocessAssets(avalancheDir, dem, cfg)

    # read particles saved from com1DFA simulation
    Particles, timeStepInfo = pT.readPartFromPickle(
        pathlib.Path(avalancheDir), simName=simName, flagAvaDir=True, comModule="com1DFA"
    )

    # Error if time step is larger than 2 seconds
    oP.checkSavingTimeStepParticles(timeStepInfo)

    # create time series of particles arrays
    particlesTimeArrays = pT.reshapeParticlesDicts(
        Particles, ["ID", "indXDEM", "indYDEM", "x", "y", "z", "inCellDEM"]
    )

    if interpolateParticlesTrajectoriesFlag:
        # interpolate particle trajectories +
        pLong = pT.interpolateParticlesTrajectories(dem, particlesTimeArrays, cellSizeFactor)
        particleTimeInfo = pLong.copy()
    else:
        particleTimeInfo = particlesTimeArrays.copy()

    # derive info on which particles interacted with infrastructure
    particleAssets, particleTimeInfo = pT.createAssetsRasterFromParticleLocations(
        particleTimeInfo,
        dem,
        uniqueAssets,
        assetsValues,
        noAssetsClass,
    )

    if outputFileType in ['.asc', '.tif']:
        _extMap = {".asc": "AAIGrid", ".tif": "GTiff"}
        dem["header"]["driver"] = _extMap[outputFileType]
    elif outputFileType != 'default':
        msg = f"{outputFileType} is not a valid option for 'outputFileType' --> use any of ['default','.asc','.tif']"
        raise ValueError(msg)

    # export raster
    rU.writeResultToRaster(
        dem["header"], particleAssets, (outDir / ("particleAssetsInfo_%s" % simName)), flip=True
    )
    

    # create plot
    plotName = "particleAssetsInfo_%s" % simName
    if interpolateParticlesTrajectoriesFlag:
        plotTitle = "Particle trajectories (interpolated) color-coded with asset classes"
    else:
        plotTitle = "Particle trajectories color-coded with asset classes"
    _ = oP.plotParticlesAssets(dem, assets, particleAssets, outDir, plotName, plotTitle)

    timeNeeded = "%.2f" % (time.time() - startTime)
    log.info("computation took: %s s " % timeNeeded)
