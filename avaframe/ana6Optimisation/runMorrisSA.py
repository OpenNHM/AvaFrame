"""
Run script for the Morris sensitivity analysis. For usage read README_ana6.md.
"""
import sys
import numpy as np
import pandas as pd
import pathlib
from datetime import datetime
from SALib.analyze import morris as morris_analyze

from avaframe.in3Utils import cfgUtils
from avaframe.in3Utils import fileHandlerUtils as fU
from avaframe.ana3AIMEC import ana3AIMEC
import avaframe.out3Plot.outAna6Plots as saveResults
import optimisationUtils

# Get module config
module = sys.modules[__name__]
cfgMorrisSA = cfgUtils.getModuleConfig(module, toPrint=False)

# Load avalanche directory from general configuration file
cfgMain = cfgUtils.getGeneralConfig()
avalancheDir = cfgMain['MAIN']['avalancheDir']
avaName = pathlib.Path(avalancheDir).name

# Calculate Areal indicators and AIMEC and save the results in Outputs/ana3AIMEC and Outputs/out1Peak
optimisationUtils.calcArealIndicatorsAndAimec(cfgMorrisSA, avalancheDir, ana3AIMEC)

# Load variation data with bounds from pickle file
inDir = pathlib.Path(avalancheDir, 'Outputs', "ana4Stats")
paramValuesD = pd.read_pickle(inDir / "paramValuesD.pickle")
varParList = paramValuesD['names']
#
# Read and merge results from parameter sets (simulation data), areal indicators and AIMEC
finalDF = optimisationUtils.buildFinalDF(avalancheDir, varParList, cfgMorrisSA)

# Use only morris samples
morrisDF = finalDF[finalDF['sampleMethods'] == 'morris'].copy(deep=True)
# Set order as index
morrisDF.set_index('order', inplace=True)
# Order df based on order (which is the index)
morrisDF.sort_index(inplace=True)

# Define input for SA
problem = {
    'num_vars': len(varParList),
    'names': varParList,
    'bounds': paramValuesD['bounds']
}
samples = np.vstack(morrisDF['parameterSet'].values).astype(float)
Y = morrisDF['optimisationVariable'].values

# Perform SA
Si = morris_analyze.analyze(
    problem,
    samples,
    Y,
    conf_level=float(cfgMorrisSA['MORRIS SA']['conf_level']),
    num_levels=int(cfgMorrisSA['MORRIS SA']['num_levels'])
)

# Rank Parameters
SiData = {
    "Parameter": Si['names'],
    "mu_star": Si['mu_star'],
    "sigma": Si['sigma'],
    "mu_star_conf": Si['mu_star_conf']}

# Convert to dataframe
SiDF = pd.DataFrame(SiData)

# Create folder for saving the results of the analysis, if not already existing
resultDir = cfgMorrisSA['GENERAL']['resultDir']
comModuleName = cfgMorrisSA['GENERAL']['modName']
outDir = pathlib.Path(avalancheDir, resultDir, comModuleName)
fU.makeADir(outDir)

saveResults.barplotSA(SiDF, avaName, outDir)
saveResults.scatterplotSA(SiDF, avaName, outDir)
saveResults.scatterplotUncertaintySA(SiDF, avaName, outDir)

# Sort SA results
SiDFSort = SiDF.sort_values("mu_star", ascending=False).reset_index(drop=True)
# Append bounds to SiDFSort
paramBounds = dict(zip(problem["names"], problem["bounds"]))
SiDFSort["bounds"] = SiDFSort["Parameter"].map(paramBounds)
# Save as Pickle for Optimization
SiDFSort.to_pickle(outDir / f"{avaName}_sortedSAResultsWithBounds.pkl")

# Create df with parameters and the loss function for summary statistics
paramLossDF, paramLossScaledDF = optimisationUtils.createDFParameterLoss(morrisDF, SiDFSort['Parameter'])
N = int(cfgMorrisSA['MORRIS SA']['N'])
paramLossSubsetDF = paramLossDF.sort_values(by='Loss', ascending=True)[:N]
# Save mean values of best input parameters as csv
date = datetime.now().strftime("%Y%m%d")
csvPath = f"{outDir}/{avaName}_MorrisBEST{N}Simulations_{date}.csv"
paramLossSubsetDF.describe().to_csv(csvPath)
