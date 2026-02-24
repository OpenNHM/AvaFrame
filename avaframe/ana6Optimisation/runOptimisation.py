"""
Run script for the optimization process. For usage read README_ana6.md.
"""
import sys
import pathlib
import pickle

from avaframe.in3Utils import cfgUtils
from avaframe.in3Utils import fileHandlerUtils as fU
from avaframe.ana3AIMEC import ana3AIMEC
import avaframe.out3Plot.outAna6Plots as saveResults
import optimisationUtils

# Get module config
module = sys.modules[__name__]
cfgOpt = cfgUtils.getModuleConfig(module, toPrint=False)

# Load avalanche directory from general configuration file
cfgMain = cfgUtils.getGeneralConfig()
avalancheDir = cfgMain['MAIN']['avalancheDir']
avaName = pathlib.Path(avalancheDir).name

# Create folder for saving the results of the analysis, if not already existing
resultDirOpt = cfgOpt['GENERAL']['resultDir']
comModuleName = cfgOpt['GENERAL']['modName']
outDir = pathlib.Path(avalancheDir, resultDirOpt, comModuleName)
fU.makeADir(outDir)

# Get config from morris for path to morris results
cfgDir = 'runMorrisSA.ini'
cfgMorrisSA = cfgUtils.getModuleConfig(pathlib.Path(cfgDir), toPrint=False)
resultDirMorris = cfgMorrisSA['GENERAL']['resultDir']
inDir = pathlib.Path(avalancheDir, resultDirMorris, comModuleName)

# Load variation parameters and their bounds
paramBounds, paramSelected = optimisationUtils.loadVariationData(cfgOpt, inDir, avalancheDir)

# Calculate Areal indicators and AIMEC and save the results in Outputs/ana3AIMEC and Outputs/out1Peak
optimisationUtils.calcArealIndicatorsAndAimec(cfgOpt, avalancheDir, ana3AIMEC)

# Read and merge results from parameter sets (simulation data), areal indicators and AIMEC
finalDF = optimisationUtils.buildFinalDF(avalancheDir, paramSelected, cfgOpt)

# ----------------------------------------------------------------------------------------------------------------------
optimisationType = cfgOpt['OPTIMISATION']['optType']
if optimisationType == 'nonseq':
    csv_path = outDir / f"{avaName}_BestOrCurrentSimulation_NonSeq.csv"
    # Save sim with currently best y
    saveResults.saveBestorCurrentModelrun(finalDF, paramSelected, csv_path=csv_path)

    # Create df with most important parameters and the loss function
    emulatorDF, emulatorScaledDF = optimisationUtils.createDFParameterLoss(finalDF, paramSelected)

    # Create surrogate
    X, y, gp_pipe = optimisationUtils.fitSurrogate(emulatorDF, cfgOpt)  # X,y are features of emulatorDF

    # K fold cross validation
    optimisationUtils.KFoldCV(X, y, gp_pipe, cfgOpt, outDir, avaName, "Gaussian Process Matern Kernel")

    # Fit final pipline
    gp_pipe.fit(X, y)

    # Optimize non-sequential (only use pipe once to find best param)
    topNStat = optimisationUtils.optimiseNonSeq(gp_pipe, cfgOpt, paramBounds)

    # Run com8 with mean parameters of best N surrogate evaluations
    simNameMean = optimisationUtils.runCom8MoTPSA(avalancheDir, topNStat['TopNBest']['mean_params'], cfgMain,
                                                  optimisationType='nonSeq')
    # Run com8 with parameters of best surrogate evaluations
    simNameBest = optimisationUtils.runCom8MoTPSA(avalancheDir, topNStat['Best']['params'], cfgMain,
                                                  optimisationType='nonSeq')
    # SimName could be None if sim is already available, if so get the name from finalDF
    if simNameMean is None:
        simNameMean = optimisationUtils.findSimName(finalDF, topNStat["TopNBest"]["mean_params"], atol=1e-6)
    if simNameBest is None:
        simNameBest = optimisationUtils.findSimName(finalDF, topNStat["Best"]["params"], atol=1e-6)

    # Calculate Areal indicators and AIMEC and save the results in Outputs/ana3AIMEC and Outputs/out1Peak
    optimisationUtils.calcArealIndicatorsAndAimec(cfgOpt, avalancheDir, ana3AIMEC)

    # Read and merge results from parameter sets (simulation data), areal indicators and AIMEC
    finalDF = optimisationUtils.buildFinalDF(avalancheDir, paramSelected, cfgOpt)

    # Create image of table
    saveResults.saveTopCandidates(finalDF, paramSelected, cfgOpt, topNStat,
                                  out_path=outDir / f"{avaName}_StatisticsBestParameterValues_NonSeqAnalysis.png",
                                  title=f"{avaName}, NonSeq-Analysis: Best Surrogate vs Best Model Run",
                                  simNameMean=simNameMean, simNameBest=simNameBest)

    # Save latest real sim
    saveResults.saveBestorCurrentModelrun(finalDF, paramSelected, simName=simNameMean, csv_path=csv_path)

# ----------------------------------------------------------------------------------------------------------------------
elif optimisationType == 'seq':
    csv_path = outDir / f"{avaName}_BestOrCurrentSimulation_BO.csv"
    # Save sim with currently best y
    saveResults.saveBestorCurrentModelrun(finalDF, paramSelected, csv_path=csv_path)

    eiThreshold = float(cfgOpt['OPTIMISATION']['eiThreshold'])
    nGoodSims = float(cfgOpt['OPTIMISATION']['numberOfGoodSimulations'])
    countGoodSims = 0
    bo_max_iterations = int(cfgOpt['OPTIMISATION']['bo_max_iterations'])

    for i in range(bo_max_iterations):
        # Create df with most important parameters and the loss function
        emulatorDF, emulatorScaledDF = optimisationUtils.createDFParameterLoss(finalDF, paramSelected)
        # Train surrogate
        X, y, gp_pipe = optimisationUtils.fitSurrogate(emulatorDF, cfgOpt)  # X,y are features of emulatorDF

        # K fold cross validation
        optimisationUtils.KFoldCV(X, y, gp_pipe, cfgOpt, outDir, avaName, "Gaussian Process Matern Kernel")

        # Fit final pipline
        gp_pipe.fit(X, y)

        # Get next input parameters with EI
        xBest, xBestDict, ei, lcb = optimisationUtils.EINextPoint(gp_pipe, y, paramBounds, cfgOpt)

        # Run com8 with best x
        simName = optimisationUtils.runCom8MoTPSA(avalancheDir, xBestDict, cfgMain, i, optimisationType='seq')

        # Calculate Areal indicators and AIMEC and save the results in Outputs/ana3AIMEC and Outputs/out1Peak
        optimisationUtils.calcArealIndicatorsAndAimec(cfgOpt, avalancheDir, ana3AIMEC)

        # Read and merge results from parameter sets (simulation data), areal indicators and AIMEC
        finalDF = optimisationUtils.buildFinalDF(avalancheDir, paramSelected, cfgOpt)

        # Save latest sim
        saveResults.saveBestorCurrentModelrun(finalDF, paramSelected, ei, lcb, simName,
                                              csv_path=csv_path)
        # If ei is smaller than threshold, the simulation is counted as 'good', if number of good simulations is
        # reached, optimization stops
        if ei < eiThreshold:
            countGoodSims = countGoodSims + 1
            if countGoodSims >= nGoodSims:
                break

    saveResults.saveTopCandidates(finalDF, paramSelected, cfgOpt,
                                  out_path=outDir / f"{avaName}_StatisticsBestParameterValues_BOAnalysis.png",
                                  title=f"{avaName}, Seq-Analysis: Best Model Runs")

    # Save BO plots
    n_top_samples = int(cfgOpt['OPTIMISATION']['n_model_top'])
    emulatorDF, emulatorScaledDF = optimisationUtils.createDFParameterLoss(finalDF, paramSelected)

    saveResults.BOConvergencePlot(finalDF, avaName, outDir)
    saveResults.BOBoxplot(emulatorDF, avaName, outDir, N=n_top_samples)
    saveResults.BOBoxplotNormalised(emulatorDF, paramBounds, avaName, outDir, N=n_top_samples)

# Save pickle file of finalDF
with open(outDir / f"{avaName}_finalDF.pickle", "wb") as fi:
    pickle.dump(finalDF, fi)
