"""
Run script for plotting a comparison of simulation result to reference polygon
"""

# Load modules
# importing general python modules
import pathlib
import numpy as np
import pickle
import re

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
from avaframe.in3Utils import cfgHandling


# TODO: create a function for this not located in runScript
def runPlotAreaRefDiffs(
    thresholdValueSimulation, modName, allResults, resType, layer="", simName="", alpha=1, beta=1
):
    """compute the area indicators comparing a simulation result to a reference polygon

    Parameters:
    -------------
    thresholdValueSimulation: float
        threshold value of simulation result used to compute area indicators
    modName: str
        name of the computational module used to perform simulation
    allResults: dict
        dictionary containing area indicators for each simulation result
    resType: str
        type of simulation result (pft, ppr, etc.)
    layer: str
        optional - name of the result layer if multi-layer computational module (l1, l2, etc.)
    simName: str
        optional - simName
    alpha, beta: float
        weighting factors used to compute area indicators

    Returns:
    ---------
    allResults: dict
        dictionary containing area indicators for each simulation result

    """

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

    # TODO: this should also work for com9
    if modName in ["com1DFA", "com5SnowSlide", "com6RockAvalanche", "com8MoTPSA"]:
        # create resType to find matching result files
        if layer != "":
            resTypeAnalysis = resType.lower() + "_" + layer.lower()
        else:
            resTypeAnalysis = resType.lower()

        # load dataFrame for all configurations of simulations in avalancheDir
        simDF = cfgUtils.createConfigurationInfo(avalancheDir, comModule=modName)

        # create data frame that lists all available simulations and path to their result type result files
        inputsDF, resTypeList = fU.makeSimFromResDF(avalancheDir, modName, simName=simName)
        # merge  parameters as columns to dataDF for matching simNames
        dataDF = inputsDF.merge(simDF, left_on="simName", right_on="simName")

        ## loop over all simulations and load desired resType
        for index, row in dataDF.iterrows():
            simFile = row[resTypeAnalysis]
            simData = IOf.readRaster(simFile)
            simName = simFile.stem

            # compute areal indicators and create plot
            allResults = oPD.mainAreaDiffAndPlot(
                referenceLine,
                simData,
                cropLine,
                cropFile,
                dem,
                thresholdValueSimulation,
                outDir,
                simName,
                alpha,
                beta,
                allResults,
                resType,
            )

    else:
        # load all result files
        resultDir = pathlib.Path(avalancheDir, "Outputs", modName, "peakFiles")
        # filter for resType and layer if available
        if layer == "":
            peakFilesList = list(resultDir.glob("*_%s.tif" % resType)) + list(
                resultDir.glob("*_%s.asc" % resType)
            )
        else:
            peakFilesList = list(resultDir.glob("*_%s_%s.tif" % (layer, resType))) + list(
                resultDir.glob("*_%s_%s.asc" % (layer, resType))
            )

        # filter all simulations for simName if provided
        if simName != "":
            peakFilesList = [pf for pf in peakFilesList if simName in pf.stem]

        for pF in peakFilesList:
            simData = IOf.readRaster(pF)
            simName = pF.stem

            # compute areal indicators and create plot
            allResults = oPD.mainAreaDiffAndPlot(
                referenceLine,
                simData,
                cropLine,
                cropFile,
                dem,
                thresholdValueSimulation,
                outDir,
                simName,
                alpha,
                beta,
                allResults,
                resType,
            )

    return allResults


def createArealIndicatorPickle(allResults, outDir):

    # Save summary of TP/FP/FN indicators as pickle
    rows = []
    for entry in allResults:
        cleanName = re.sub(r"_ppr$", "", entry["sim_name"])
        indicators = entry["indicator_dict"]
        row = {"simName": cleanName}

        for key, subdict in indicators.items():
            shortKey = (
                key.replace("truePositive", "TP_SimRef")
                .replace("falsePositive", "FP_SimRef")
                .replace("falseNegative", "FN_SimRef")
            )
            row["%s_cells" % shortKey] = subdict.get("nCells", None)
            row["%s_area" % shortKey] = subdict.get("areaSum", None)

        rows.append(row)

    with open(outDir / "arealIndicators.pkl", "wb") as f:
        pickle.dump(rows, f)


if __name__ == "__main__":
    ################USER Input#############
    resType = "ppr"
    thresholdValueSimulation = 1
    modName = "com8MoTPSA"

    runPlotAreaRefDiffs(resType, thresholdValueSimulation, modName)
