import sys
import pathlib
import pickle
import pandas as pd

from avaframe.in3Utils import cfgUtils, logUtils
from avaframe.in3Utils import fileHandlerUtils as fU
from avaframe.in3Utils import initializeProject as initProj
import avaframe.out3Plot.outAna6Plots as saveResults
from avaframe.com8MoTPSA import com8MoTPSA
from avaframe.ana6Optimisation import optimisationUtils


def runOptimisation():
    """
    Run script for the optimization process. For usage read README_ana6.md.
    """
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
    cfgDir = 'runMorrisSA.py'
    cfgMorrisSA = cfgUtils.getModuleConfig(pathlib.Path(cfgDir), toPrint=False)
    resultDirMorris = cfgMorrisSA['GENERAL']['resultDir']
    inDir = pathlib.Path(avalancheDir, resultDirMorris, comModuleName)

    # Start logging
    logName = 'runOptimisation'
    log = logUtils.initiateLogger(avalancheDir, logName)
    log.info('MAIN SCRIPT')
    log.info('Current avalanche: %s', avalancheDir)

    # Load variation parameters and their bounds
    paramBounds, paramSelected = optimisationUtils.loadVariationData(cfgOpt, avalancheDir, inDir)

    # Calculate Areal indicators and AIMEC and save the results in Outputs/ana3AIMEC and Outputs/out1Peak
    optimisationUtils.calcArealIndicatorsAndAimec(cfgOpt, avalancheDir)

    # Read and merge results from parameter sets (simulation data), areal indicators and AIMEC
    finalDF = optimisationUtils.buildFinalDF(avalancheDir, paramSelected, cfgOpt)

    # Check if there is simulation in finalDF
    nAvailable = len(finalDF)
    if nAvailable == 0:
        message = "No simulations found in finalDF."
        log.error(message)
        raise RuntimeError(message)
    # ------------------------------------------------------------------------------------------------------------------
    # Two surrogate-based optimisation strategies are available:
    # nonseq: Non-sequential optimisation uses a fixed set of simulations to train the surrogate and evaluate the loss.
    # seq:    Sequential (Bayesian) optimisation uses an initial set of samples to train the surrogate and iteratively
    #         adds new simulations using the Expected Improvement acquisition function.
    optimisationType = cfgOpt['OPTIMISATION']['optType'].lower()
    if optimisationType == 'nonseq':
        csv_path = outDir / f"{avaName}_BestOrCurrentSimulation_NonSeq.csv"
        # Save sim with currently best y
        saveResults.saveBestOrSpecificSimulation(finalDF, paramSelected, csv_path=csv_path)

        # Create df with most important parameters and the loss function
        emulatorDF, emulatorScaledDF = optimisationUtils.createDFParameterLoss(finalDF, paramSelected)

        # Create surrogate
        X, y, gp_pipe = optimisationUtils.fitSurrogate(emulatorDF, cfgOpt)  # X,y are features of emulatorDF

        # K fold cross validation
        optimisationUtils.KFoldCrossValidation(X, y, gp_pipe, cfgOpt, outDir, avaName, "Gaussian Process Matern Kernel")

        # Fit final pipline
        gp_pipe.fit(X, y)

        # Optimize non-sequential (only use pipe once to find best param)
        topNStat = optimisationUtils.optimiseNonSeq(gp_pipe, cfgOpt, paramBounds)
        paramSets = [
            topNStat["TopNBest"]["mean_params"],
            topNStat["Best"]["params"],
        ]
        # Generate and write config files with mean parameter values of best N surrogate evaluations and with best
        # parameter values of surrogate
        # Clean input directory(ies) of old work files
        initProj.cleanSingleAvaDir(avalancheDir, deleteOutput=False)

        cfgFiles, cfgPath = optimisationUtils.writeCfgFiles(avalancheDir, paramSets, optimisationType, comModuleName)

        # Perform com8MoTPSA simulations
        com8MoTPSA.com8MoTPSAMain(cfgMain, cfgInfo=cfgPath)

        # Calculate Areal indicators and AIMEC and save the results in Outputs/ana3AIMEC and Outputs/out1Peak
        optimisationUtils.calcArealIndicatorsAndAimec(cfgOpt, avalancheDir)

        # Read and merge results from parameter sets (simulation data), areal indicators and AIMEC
        finalDF = optimisationUtils.buildFinalDF(avalancheDir, paramSelected, cfgOpt)

        simNameMean = optimisationUtils.findSimName(finalDF, topNStat["TopNBest"]["mean_params"], atol=1e-6)
        simNameBest = optimisationUtils.findSimName(finalDF, topNStat["Best"]["params"], atol=1e-6)

        # Create image of table
        saveResults.saveTopCandidates(finalDF, paramSelected, cfgOpt, topNStat,
                                      out_path=outDir / f"{avaName}_StatisticsBestParameterValues_NonSeqAnalysis.png",
                                      title=f"{avaName}, NonSeq-Analysis: Best Surrogate vs Best Model Run",
                                      simNameMean=simNameMean, simNameBest=simNameBest)

        # Save latest real sim
        saveResults.saveBestOrSpecificSimulation(finalDF, paramSelected, simName=simNameMean, csv_path=csv_path)
        saveResults.BOConvergencePlot(finalDF, avaName, outDir, cfgOpt, method_name='NonSeq')

    # ------------------------------------------------------------------------------------------------------------------
    elif optimisationType == 'seq':
        csv_path = outDir / f"{avaName}_BestOrCurrentSimulation_BO.csv"
        # Save sim with currently best y
        saveResults.saveBestOrSpecificSimulation(finalDF, paramSelected, csv_path=csv_path)

        eiThreshold = cfgOpt.getfloat('OPTIMISATION', 'eiThreshold')
        nGoodSims = cfgOpt.getfloat('OPTIMISATION', 'numberOfGoodSimulations')
        countGoodSims = 0
        bo_max_iterations = cfgOpt.getint('OPTIMISATION', 'bo_max_iterations')

        # Before the loop: start counter based on existing BO ("seq") simulations
        # Important to avoid same order number across different BO runs
        seq_orders = pd.to_numeric(finalDF.loc[finalDF["sampleMethods"] == "seq", "order"], errors="coerce")
        start_counter = int(seq_orders.max() + 1) if seq_orders.notna().any() else 0

        for i in range(bo_max_iterations):
            # Create df with most important parameters and the loss function
            emulatorDF, emulatorScaledDF = optimisationUtils.createDFParameterLoss(finalDF, paramSelected)
            # Train surrogate
            X, y, gp_pipe = optimisationUtils.fitSurrogate(emulatorDF, cfgOpt)  # X,y are features of emulatorDF

            # K fold cross validation
            optimisationUtils.KFoldCrossValidation(X, y, gp_pipe, cfgOpt, outDir, avaName,
                                                   "Gaussian Process Matern Kernel")

            # Fit final pipline
            gp_pipe.fit(X, y)

            # Get next input parameters with EI
            xBest, xBestDict, ei, lcb = optimisationUtils.EINextPoint(gp_pipe, y, paramBounds, cfgOpt)

            # Clean input directory(ies) of old work files
            initProj.cleanSingleAvaDir(avalancheDir, deleteOutput=False)

            # Generate and write config file
            cfgFiles, cfgPath = optimisationUtils.writeCfgFiles(avalancheDir, xBestDict, optimisationType,
                                                                comModuleName, counter=start_counter + i)

            # Perform com8MoTPSA simulation
            com8MoTPSA.com8MoTPSAMain(cfgMain, cfgInfo=cfgPath)

            # Calculate Areal indicators and AIMEC and save the results in Outputs/ana3AIMEC and Outputs/out1Peak
            optimisationUtils.calcArealIndicatorsAndAimec(cfgOpt, avalancheDir)

            # Read and merge results from parameter sets (simulation data), areal indicators and AIMEC
            finalDF = optimisationUtils.buildFinalDF(avalancheDir, paramSelected, cfgOpt)

            # Save latest sim
            simName = optimisationUtils.findSimName(finalDF, xBestDict, atol=1e-6)
            saveResults.saveBestOrSpecificSimulation(finalDF, paramSelected, ei, lcb, simName,
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
        n_top_samples = cfgOpt.getint('OPTIMISATION', 'n_model_top')
        emulatorDF, emulatorScaledDF = optimisationUtils.createDFParameterLoss(finalDF, paramSelected)

        saveResults.BOConvergencePlot(finalDF, avaName, outDir, cfgOpt)
        # Standard boxplot
        saveResults.BOBoxplot(emulatorDF, avaName, outDir, N=n_top_samples)
        # y-scaled boxplot
        saveResults.BOBoxplot(emulatorDF, avaName, outDir, N=n_top_samples, paramBounds=paramBounds, yScaled=True)
        # Normalized boxplot
        saveResults.BOBoxplot(emulatorDF, avaName, outDir, N=n_top_samples, paramBounds=paramBounds, normalised=True)
        # Boxplot where Loss = 1, y-scaled
        saveResults.BoxplotNoPSC(emulatorDF, paramBounds, avaName, outDir)

    else:
        message = f"Unknown optimisation type '{optimisationType}'. Expected 'nonseq' or 'seq'."
        log.error(message)
        raise ValueError(message)

    # Save pickle file of finalDF
    with open(outDir / f"{avaName}_finalDF.pickle", "wb") as fi:
        pickle.dump(finalDF, fi)

    # Save comparison boxplots for the top N simulations and No-PSC simulations of two avalanche paths.
    # Required are the corresponding finalDF.pickle files.
    saveComparisonBoxplot = cfgOpt['OPTIMISATION']['saveComparisonBoxplot']
    n_top_samples = cfgOpt.getint('OPTIMISATION', 'n_model_top')
    if saveComparisonBoxplot:
        saveResults.plotComparisonBoxplots(
            outDir=outDir,
            avaName1="avaFleisskar",
            avaName2="avaWolfsgrube",
            N=n_top_samples,
            paramBounds=paramBounds,
            yScaled=True,
        )


if __name__ == '__main__':
    runOptimisation()
