import numpy as np
import pathlib
import pickle
import logging
import pandas as pd
import configparser
import os
from scipy.stats import norm, qmc
from datetime import datetime
import re

from sklearn.model_selection import KFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

from avaframe.in3Utils import cfgUtils
from avaframe.in3Utils import fileHandlerUtils as fU
from avaframe.com8MoTPSA import com8MoTPSA
from avaframe.com1DFA import com1DFA
from avaframe.ana3AIMEC import ana3AIMEC
from avaframe.ana4Stats import probAna
from avaframe.runScripts.runPlotAreaRefDiffs import runPlotAreaRefDiffs, createArealIndicatorPickle
import avaframe.out3Plot.outAna6Plots as saveResults
from avaframe.in3Utils import cfgHandling

# create local logger
log = logging.getLogger("avaframe")


def calcArealIndicatorsAndAimec(cfgOpt, avalancheDir):
    """
    Calculate areal indicators between reference polygon and simulation and AIMEC analysis and save data in ana3AIMEC
    and out1Peak.
    Parameters
    ----------
    cfgOpt : configparser.ConfigParser
        Global configuration object. Must contain section "GENERAL"
        with keys:
            - resType
            - thresholdValueSimulation
            - modName
    avalancheDir : str
        Path to the avalanche directory.
    """

    # ToDo: to reduce comp. cost: run AIMEC and calcArealIndicators only for new simulations
    # Areal indicators
    resType = cfgOpt["GENERAL"]["resType"]
    thresholdValueSimulation = cfgOpt.getfloat("GENERAL", "thresholdValueSimulation")
    modName = cfgOpt["GENERAL"]["modName"]
    allResults = []
    allResults = runPlotAreaRefDiffs(
        thresholdValueSimulation, modName, allResults, resType, layer="", alpha=1, beta=1
    )

    # AIMEC
    cfgAIMEC = cfgUtils.getModuleConfig(ana3AIMEC)
    rasterTransfo, resAnalysisDF, plotDict, _, pathDict = ana3AIMEC.fullAimecAnalysis(avalancheDir, cfgAIMEC)


def calcArealIndicatorsAndAimecOneAtATime(cfgOpt, avalancheDir, aimecInfo, simName, allResults):
    """
    Calculate areal indicators between reference polygon and simulation and AIMEC analysis and save data in ana3AIMEC
    and out1Peak.

    Parameters
    ----------
    cfgOpt : configparser.ConfigParser
        Global configuration object. Must contain section "GENERAL"
        with keys:
            - resType
            - thresholdValueSimulation
            - modName
    avalancheDir : str
        Path to the avalanche directory.

    Returns
    -------
    allResults: list
        list with areal indicators for current simulation

    """

    # Areal indicators
    # TODO: make layer information more prominent - handle differently?
    resType = cfgOpt["GENERAL"]["resType"]
    thresholdValueSimulation = cfgOpt.getfloat("GENERAL", "thresholdValueSimulation")
    modName = cfgOpt["GENERAL"]["modName"]
    allResults = runPlotAreaRefDiffs(
        thresholdValueSimulation,
        modName,
        allResults,
        resType,
        layer=cfgOpt["GENERAL"]["layer"],
        simName=simName,
        alpha=cfgOpt["LOSS_PARAMETERS"].getfloat("tverskyAlpha"),
        beta=cfgOpt["LOSS_PARAMETERS"].getfloat("tverskyBeta"),
    )

    # AIMEC runoutline comparison
    resAnalysisDF = ana3AIMEC.addSimToResAnalysisDFForRunoutComparison(
        avalancheDir, simName, modName, aimecInfo
    )
    # add runoutLineDiff_RMSE values to dataframe
    resAnalysisDFFull = aimecInfo["resAnalysisDFFull"]
    allAnalysis = [resAnalysisDFFull, resAnalysisDF]
    resAnalysisDFFull = pd.concat(allAnalysis)
    aimecInfo["resAnalysisDFFull"] = resAnalysisDFFull
    # save resAnalysisFillDF to file
    resAnalysisDFFullPath = pathlib.Path(aimecInfo["pathDict"]["pathResult"], "resAnalysisDFFull.csv")
    resAnalysisDFFull.to_csv(resAnalysisDFFullPath)

    return aimecInfo, allResults


def readParamSetDF(inDir, varParList):
    """
    Read parameter sets from .ini files in a directory and build a DataFrame.

    Parameters
    ----------
    inDir : str or pathlib.Path
        Path to directory containing .ini files.
    varParList : list of str
        List of parameter names to extract values from each .ini file.

    Returns
    -------
    paramSetDF : pandas.DataFrame
        DataFrame with simName, parameterSet and order as columns
    """

    # List to hold all parameters sets
    paramSet = []
    order = []
    sampleMethods = []
    filenames = []

    # Loop over all files in the folder
    for filename in os.listdir(inDir):
        if filename.endswith('.ini') and 'sourceConfiguration' not in filename:
            filepath = os.path.join(inDir, filename)

            # Load the .ini file
            config = configparser.ConfigParser()
            config.read(filepath)

            # Set defaults per file
            index = np.nan
            sampleMethod = np.nan

            if 'VISUALISATION' in config.sections():
                # config is inifile
                index = config['VISUALISATION']['scenario']
                if 'sampleMethod' in config['VISUALISATION']:
                    sampleMethod = config['VISUALISATION']['sampleMethod']
                else:
                    sampleMethod = np.nan

            row = []  # row contains 1 row
            for param in varParList:
                section = probAna.fetchParameterSection(config, param)
                value = config[section][param]
                value = float(value)
                row.append(value)

            order.append(index)
            sampleMethods.append(sampleMethod)
            paramSet.append(row)  # rows contains all rows
            filenames.append(os.path.splitext(filename)[0])

    # Convert to pandas DF
    paramSetDF = pd.DataFrame({
        'simName': filenames,
        'parameterSet': paramSet,  # [row for row in paramSet], # Wrap each row as a list
        'order': pd.to_numeric(order),  # convert to int
        'sampleMethods': sampleMethods
    })
    return paramSetDF


def readArealIndicators(inDir):
    """
    Read areal indicator results from a pickle file and convert to a DataFrame.

    Parameters
    ----------
    inDir : str or pathlib.Path
        Path to pickle file containing indicator results.

    Returns
    -------
    indicatorsDF : pandas.DataFrame
        DataFrame with simName and areal indicators,
    """
    with open(inDir, "rb") as f:
        all_results = pickle.load(f)

    indicatorsDF = pd.DataFrame(all_results)
    return indicatorsDF


def addLossMetrics(df, referenceDF, cfg):
    """
    Compute evaluation metrics (recall, precision, F1, Tversky score) and a combined weighted Loss (or optimisation)
    variable from a given DataFrame.

    The metrics are based on area (number of pixel would also be possible). Invalid values
    (division by zero) are replaced with 0.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with at least the columns
        ``TP_SimRef_area``, ``FP_SimRef_area``, ``FN_SimRef_area``.
    referenceDF: pandas.Dataframe
        Dataframe with information of the reference in AIMEC e.g. reference_sRunout of the polygon
    cfg: configparser.ConfigParser
        Config parser that contains values of the loss function, either runOptimisationCfg.ini or runMorrisSA.ini file

    Returns
    -------
    df : pandas.DataFrame
        Same DataFrame with additional columns:
        - ``recall`` : float
        - ``precision`` : float
        - ``f1_score`` : float
        - ``tversky_score`` : float
        - ``optimisationVariable`` : float
    """
    # Decide if loss function is based on ncells or area
    basedOn = '_area'
    TP = df[f"TP_SimRef{basedOn}"]
    FP = df[f"FP_SimRef{basedOn}"]
    FN = df[f"FN_SimRef{basedOn}"]

    # Recall = TP / (TP + FN)
    denomRecall = TP + FN
    df["recall"] = np.where(denomRecall != 0, TP / denomRecall, 0.0)
    # Precision = TP / (TP + FP)
    denomPrecision = TP + FP
    df["precision"] = np.where(denomPrecision != 0, TP / denomPrecision, 0.0)
    # F1 Score
    denomF1 = df['precision'] + df['recall']
    df["f1_score"] = np.where(denomF1 != 0, 2.0 * df['precision'] * df['recall'] / denomF1, 0.0)

    # Tversky score = TP / (TP + alpha * FP + beta * FN), gives penalty to overshoot --> alpha
    alpha = cfg.getfloat('LOSS_PARAMETERS', 'tverskyAlpha')
    beta = cfg.getfloat('LOSS_PARAMETERS', 'tverskyBeta')

    denomTversky = TP + alpha * FP + beta * FN
    df["tversky_score"] = np.where(denomTversky != 0, TP / denomTversky, 0.0)
    # Subtract 1 to ensure that 0 are good values and 1 bad
    df['1-tversky'] = 1 - df["tversky_score"]

    # Runout
    # RMSE divided with Runout length of Reference
    sRunoutRef = referenceDF['reference_sRunout'].values
    runoutRMSE = df['runoutLineDiff_poly_RMSE']
    df['runoutRMSENormalised'] = runoutRMSE / sRunoutRef  # 0 is good, 1 is bad

    # Weights
    wTversky = cfg.getfloat('LOSS_PARAMETERS', 'weightTversky')
    wRunout = cfg.getfloat('LOSS_PARAMETERS', 'weightRunout')

    df['optimisationVariable'] = (
            wRunout * df['runoutRMSENormalised'].fillna(1) + wTversky * df['1-tversky'].fillna(1))
    return df


def buildFinalDF(avalancheDir, varParList, cfg, modName):
    """
    Build the final merged DataFrame for a given avalanche.

    Combines parameter sets, AIMEC results, and areal indicators into one DataFrame,
    then computes evaluation metrics 'via addLossMetrics'. The resulting DataFrame contains one row per simulation.
    If simulations are available for two layers (e.g. L1 and L2), only the layer specified in the configuration files
    is considered. Simulations from the selected layer are kept, while simulations from the other layer are excluded
    from the final DataFrame. The analysis layer must be defined in the corresponding .ini configuration files. Further
    details are provided in README_ana6.md.

    Parameters
    ----------
    avalancheDir : str
        Path of avalanche directory
    varParList: list of str
        List of parameter names that are varied
    cfg: configparser.ConfigParser
        Config parser that contains values of the loss function, either runOptimisationCfg.ini or runMorrisSA.ini file

    Returns
    -------
    finalDF : pandas.DataFrame
        Final DataFrame containing:
        - `simName`
        - `parameterSet`
        - `order`:  used later for visualisation. Morris, Latin hypercube and sequential samples are assigned an order
                    index. For each sampling method, the index starts at 0 and increases in increments of 1 up to the
                    total number of samples of that method.
        - Areal indicator columns
        - Evaluation metrics (recall, precision, f1_score, tversky_score, optimisationVariable)
    """
    # old code----------------
    # Folder where ini files from simulations are
    # inDir = pathlib.Path(avalancheDir, ("Outputs/%s/configurationFiles" % modName))
    # Read parameterSetDF
    # paramSetDF = readParamSetDF(inDir, varParList)
    # -------------------------
    # TODO: consider using cfgHandling/createInfoDF instead
    paramSetDF = cfgHandling.createInfoDF(
        avalancheDir, modName, ["simName", "scenario", "sampleMethod"], varParList
    )

    # old code ------------------------
    # Get Dataframe from AIMEC analysis
    # cfgAIMEC = cfgUtils.getModuleConfig(ana3AIMEC)
    # runoutResType = cfgAIMEC["AIMECSETUP"]["runoutResType"]
    # thresholdValue = cfgAIMEC['AIMECSETUP']['thresholdValue']
    # domainWidth = cfgAIMEC['AIMECSETUP']['domainWidth']
    # -----------------------------------

    AIMECPath = avalancheDir + ("/Outputs/ana3AIMEC/%s/" % modName)
    # AIMECFileName = (
    #    f"Results_{avaName}_{runoutResType}_lim_{thresholdValue}_w_{domainWidth}resAnalysisDF.csv"
    # )
    AIMECFileName = "resAnalysisDFFull.csv"

    df_aimec = pd.read_csv(AIMECPath + AIMECFileName)
    # Get data of the reference in AIMEC
    referenceDF = pd.read_csv(f"{avalancheDir}/Outputs/ana3AIMEC/{modName}/referenceDF.csv")

    # Read areal indicators
    arealIndicatorDir = pathlib.Path(avalancheDir, 'Outputs', 'out1Peak', 'arealIndicators.pkl')
    indicatorsDF = readArealIndicators(arealIndicatorDir)

    # Remove layer suffix _L1 or _L2 from simName for merging, the layer is provided by cfg['GENERAL']['layer'].
    layer = cfg['GENERAL']['layer']
    indicatorsDF["simName"] = indicatorsDF["simName"].str.replace(fr"_{layer}$", "", regex=True)

    # Merge df's
    df_merged = pd.merge(paramSetDF, df_aimec, on='simName', how='inner')
    df_merged = df_merged.merge(indicatorsDF, on="simName", how="left")

    # Add optimisation variables
    finalDF = addLossMetrics(df_merged, referenceDF, cfg)
    return finalDF


def createDFParameterLoss(df, paramSelected):
    """
    Create DataFrames linking selected parameters with the loss function.
    The meaning of selected depends on the chosen scenario. If Morris sensitivity analysis is not run beforehand, all
    varied input parameters from probAnaCfg.ini are included. If Morris sensitivity analysis is run beforehand, only the topN
    highest-ranked parameters are selected based on the Morris results.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame that contains the finalDF including the input parameters and their values as well as the loss function.
    paramSelected : list of str
        List of parameters to include in the output DataFrames. This determines which parameters will be used for
        optimisation and depends on the scenario.

    Returns
    -------
    paramLossDF : pandas.DataFrame
        DataFrame with one column per selected parameter and an additional
        ``Loss`` column with the raw values of ``optimisationVariable``.
    paramLossDFScaled : pandas.DataFrame
        Same as ``paramLossDF`` but with the selected parameters normalised
        to the range [0, 1] using min–max scaling.
    """
    paramLossDF = df[paramSelected].copy()
    paramLossDFScaled = (paramLossDF - paramLossDF.min()) / (paramLossDF.max() - paramLossDF.min())  # normalise
    paramLossDF['Loss'] = df[
        'optimisationVariable']
    paramLossDFScaled['Loss'] = df[
        'optimisationVariable']
    return paramLossDF, paramLossDFScaled


def fitSurrogate(df, cfgOpt):
    """
    Prepare data and initialize surrogate models for loss prediction.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing input parameters as columns and a
        column named 'Loss' as target variable.
    cfgOpt: configparser.ConfigParser
        Config parser of the runOptimisationCfg.ini file.

    Returns
    -------
    X : numpy.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : numpy.ndarray
        Target vector of shape (n_samples,).
    gp_pipe : sklearn.pipeline.Pipeline
        Pipeline consisting of feature standardization and a
        Gaussian Process regressor with Matern kernel.
    """
    # Prepare X, y
    y_col = 'Loss'
    X = df.drop(columns=[y_col]).to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float).reshape(-1)
    n_features = X.shape[1]

    # GP Surrogate settings: ConstantKernel * MaternKernel + WhiteKernel
    matern_nu = cfgOpt.getfloat('GP_SURROGATE', 'matern_nu')
    constant_value = cfgOpt.getfloat('GP_SURROGATE', 'constant_value')
    constant_value_bounds = tuple(float(v) for v in cfgOpt['GP_SURROGATE']['constant_bounds'].split('|'))
    length_scale_initial = cfgOpt.getfloat('GP_SURROGATE', 'length_scale_initial')
    length_scale_bounds = tuple(float(v) for v in cfgOpt['GP_SURROGATE']['length_scale_bounds'].split('|'))
    white_noise_level = cfgOpt.getfloat('GP_SURROGATE', 'white_noise_level')
    white_noise_bounds = tuple(float(v) for v in cfgOpt['GP_SURROGATE']['white_noise_bounds'].split('|'))

    gp_alpha = cfgOpt.getfloat('GP_SURROGATE', 'gp_alpha')
    normalize_y = cfgOpt.getboolean('GP_SURROGATE', 'normalize_y')
    n_restarts_optimizer = cfgOpt.getint('GP_SURROGATE', 'n_restarts_optimizer')
    random_state = cfgOpt.getint('GP_SURROGATE', 'random_state')

    kernel = (
            ConstantKernel(constant_value=constant_value,
                           constant_value_bounds=constant_value_bounds)
            * Matern(length_scale=np.full(n_features, length_scale_initial),
                     length_scale_bounds=length_scale_bounds,
                     nu=matern_nu)
            + WhiteKernel(noise_level=white_noise_level, noise_level_bounds=white_noise_bounds)
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=gp_alpha,
        normalize_y=normalize_y,
        n_restarts_optimizer=n_restarts_optimizer,
        random_state=random_state,
    )

    # Pipelines (feature scaling + model)
    gp_pipe = Pipeline([("x_scaler", StandardScaler()), ("model", gp)])
    return X, y, gp_pipe


def KFoldCrossValidation(X, y, pipe, cfgOpt, outDir, avaName, pipeName):
    """
    Perform k-fold cross-validation for a regression pipeline.

    Parameters
    ----------
    X : numpy.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : numpy.ndarray
        Target vector of shape (n_samples,).
    pipe : sklearn.pipeline.Pipeline
        Regression pipeline to be evaluated.
    cfgOpt: configparser.ConfigParser
        Config parser of the runOptimisationCfg.ini file.
    outDir : pathlib.Path
        File path where the generated image will be saved.
    avaName : str
        Name of the avalanche. Used for naming the output figure.
    pipeName : str
        Name of the pipeline, used for formatted console output.

    Returns
    -------
    scores : dict
        Dictionary containing cross-validation results as returned
        by sklearn.model_selection.cross_validate.
    """
    # Get optType
    optType = cfgOpt['OPTIMISATION']['optType']
    # For losses, sklearn uses "neg_*" because higher-is-better internally
    rmse_scorer = "neg_root_mean_squared_error"
    mae_scorer = "neg_mean_absolute_error"
    r2_scorer = "r2"
    k = cfgOpt.getint('OPTIMISATION', 'k')
    cv = KFold(n_splits=k, shuffle=True, random_state=0)

    scores = cross_validate(
        pipe, X, y, cv=cv,
        scoring={"rmse": rmse_scorer, "mae": mae_scorer, "r2": r2_scorer},
        return_train_score=True,
        error_score="raise"  # fail fast if something else is wrong
    )

    # NOTE: rmse/mae were returned as NEGATIVE numbers because the higher is better internally
    neg_cols = ["test_rmse", "test_mae", "train_rmse", "train_mae"]
    for c in neg_cols:
        scores[c] = -scores[c]
    # get smoothness parameter nu
    matern_nu = cfgOpt.getfloat('GP_SURROGATE', 'matern_nu')
    row = {
        "experiment_name": pipeName,
        "n_samples": X.shape[0],
        "kernel": f"Matern {matern_nu}",
        "noise_model": "WhiteKernel",
    }

    for split in ("test", "train"):
        for m in ("rmse", "mae", "r2"):
            arr = scores[f"{split}_{m}"]
            row[f"{split} {m} mean"] = arr.mean()
            row[f"{split} {m} std"] = arr.std()

    # Log results of cross validation
    log.info(f"\n{pipeName}, {k}-fold CV:")
    for split in ("test", "train"):
        log.info(f"  {split.capitalize()} metrics:")
        for m in ("rmse", "mae", "r2"):
            arr = scores[f"{split}_{m}"]
            log.info(f"    {m.upper():<4}: {arr.mean():.4g} ± {arr.std():.4g}")

    df = pd.DataFrame([row])
    # Include date, format: YYYYMMDD
    date = datetime.now().strftime("%Y%m%d")
    base = os.path.join(outDir, f"{avaName}_{k}FoldCV_{optType}")
    df.to_csv(base + f'_{date}.csv', mode="a", header=not os.path.exists(base + f'_{date}.csv'),
              index=False)  # checks if header exists

    # Save results as image
    saveResults.saveKFoldCVPrintImage(scores, pipeName, k, base + f'_Matern_{matern_nu}_Kernel_{date}.png')
    return scores


def optimiseNonSeq(pipe, cfgOpt, paramBounds):
    """
    Perform non-sequential surrogate-based optimization using
    Latin Hypercube sampling.

    Parameters
    ----------
    pipe : sklearn.pipeline.Pipeline
        Trained surrogate model pipeline used to predict loss
        and uncertainty.
    cfgOpt : configparser.ConfigParser
        Config parser of the runOptimisationCfg.ini file.
    paramBounds : dict
        Dictionary mapping parameter names to (min, max) bounds.

    Returns
    -------
    topNStat : pandas.DataFrame
        DataFrame containing statistics of the top N surrogate
        candidates with the lowest predicted loss.
    """
    paramSelected = list(paramBounds.keys())
    bounds = np.array(list(paramBounds.values()), dtype=float)  # shape (d,2)
    d = bounds.shape[0]

    # Create LH samples
    seed = cfgOpt.getint('OPTIMISATION', 'seed')
    n_lhs = cfgOpt.getint('OPTIMISATION', 'n_lhs')
    sampler = qmc.LatinHypercube(d=d, seed=seed)
    sample = sampler.random(n=n_lhs)
    X0 = qmc.scale(sample, bounds[:, 0], bounds[:, 1])

    # Prediction of the loss with GP-Model
    mu, sigma = pipe.predict(X0, return_std=True)
    # Convert X0 to pandas df for analyze function
    df_candidates = pd.DataFrame(X0, columns=paramSelected)
    n_top_samples = cfgOpt.getint('OPTIMISATION', 'n_surrogate_top')
    topNStat, _ = analyzeTopCandidates(df_candidates, mu, sigma, paramSelected, N=n_top_samples)
    return topNStat


def analyzeTopCandidates(df_candidates, mu, sigma, param_cols, N):
    """
    Analyze the top N surrogate candidates.

    Parameters
    ----------
    df_candidates : pandas.DataFrame
        Candidate parameter sets.
    mu : numpy.ndarray
        Predicted loss values.
    sigma : numpy.ndarray
        Predicted uncertainty values.
    param_cols : list of str
        Parameter column names.
    N : int
        Number of best candidates to analyze.

    Returns
    -------
    stats : dict
        Summary statistics for the top N and best candidate.
    topNData : pandas.DataFrame
        Top N candidates with predicted mu and sigma.
    """

    # Top N points
    idx_topN = np.argsort(mu)[:N]
    topNData = df_candidates.iloc[idx_topN].copy()
    topNData["mu"] = mu[idx_topN]
    topNData["sigma"] = sigma[idx_topN]

    mean_params = topNData[param_cols].mean()
    std_params = topNData[param_cols].std()
    mean_mu = topNData["mu"].mean()
    std_mu = topNData["mu"].std()
    mean_sigma = topNData["sigma"].mean()
    std_sigma = topNData["sigma"].std()

    log.info(f"Surrogate Top {N} candidates: mean ± std")

    for p in param_cols:
        m, s = mean_params[p], std_params[p]
        perc = (s / m * 100) if m != 0 else np.nan
        log.info(f"  {p:30s}: {m:.6f} ± {s:.6f} ({perc:.1f}%%)")

    perc_mu = (std_mu / mean_mu * 100) if mean_mu != 0 else np.nan
    perc_sigma = (std_sigma / mean_sigma * 100) if mean_sigma != 0 else np.nan

    log.info(f"mu:    {mean_mu:.4f} ± {std_mu:.4f} ({perc_mu:.1f}%%)")
    log.info(f"sigma: {mean_sigma:.4f} ± {std_sigma:.4f} ({perc_sigma:.1f}%%)")

    # Best single point
    idx_best = np.argmin(mu)
    best_params = df_candidates.iloc[idx_best].copy()
    best_loss = mu[idx_best]
    best_sigma = sigma[idx_best]

    log.info("Best single parameter combination from Surrogate:")

    for p in param_cols:
        log.info(f"  {p:30s}: {best_params[p]:.4f}")

    log.info(f"mu:    {best_loss:.4f}")
    log.info(f"sigma: {best_sigma:.4f}")

    return {
        "TopNBest": {
            "mean_params": mean_params,
            "std_params": std_params,
            "mean_mu": mean_mu,
            "std_mu": std_mu,
            "mean_sigma": mean_sigma,
            "std_sigma": std_sigma,
        },
        "Best": {
            "params": best_params,
            "mu": best_loss,
            "sigma": best_sigma,
        }
    }, topNData


def expectedImprovement(mu, sigma, f_best, xi):
    """
    Compute the Expected Improvement (EI) acquisition function
    for minimization problems.
    Formula taken from:
    https://ekamperi.github.io/machine%20learning/2021/06/11/acquisition-functions.html

    Parameters
    ----------
    mu : numpy.ndarray
        Predicted mean values of the surrogate model.
    sigma : numpy.ndarray
        Predicted standard deviations of the surrogate model.
    f_best : float
        Best observed objective value so far.
    xi : float
        Exploration parameter controlling exploitation–exploration
        trade-off.

    Returns
    -------
    ei : numpy.ndarray
        Expected Improvement values for each candidate.
    """
    sigma = np.maximum(sigma, 1e-12)  # numeric safety
    imp = f_best - mu - xi  # minimization, that's why the sign is different, xi for finetunig exploitation

    Z = imp / sigma
    ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma <= 1e-12] = 0.0  # set EI to zero where sigma is 1e-12
    return ei


def lowerConfidenceBound(mu, sigma, k=0.0):
    """
    Compute the Lower Confidence Bound (LCB) acquisition function.
    Formula taken from:
    https://ekamperi.github.io/machine%20learning/2021/06/11/acquisition-functions.html

    Parameters
    ----------
    mu : numpy.ndarray
        Predicted mean values of the surrogate model.
    sigma : numpy.ndarray
        Predicted standard deviations of the surrogate model.
    k : float, optional
        Exploration parameter controlling the trade-off between
        exploitation and exploration.

    Returns
    -------
    lcb : numpy.ndarray
        Lower Confidence Bound values for each candidate.
    """
    return -mu + k * sigma


def EINextPoint(pipe, y, paramBounds, cfgOpt):
    """
    Propose the next evaluation point using Expected Improvement (EI) or lower confidence bound (LCB).

    This function generates a Latin Hypercube sample (LHS) of candidate points within the
    parameter bounds, evaluates a surrogate model (``pipe``) on those candidates, and
    selects the candidate that maximises EI or LCB for a *minimisation*
    problem.

    Parameters
    ----------
    pipe : sklearn.pipeline.Pipeline
        Trained surrogate model pipeline used to predict loss
        and uncertainty.
    y : array-like
        Observed objective values (loss) from previous evaluations.
    paramBounds : dict
        Dictionary mapping parameter names to (lower, upper) bounds.
    cfgOpt : configparser.ConfigParser
        Config parser of the runOptimisationCfg.ini file.

    Returns
    -------
    xBest : numpy.ndarray
        Vector of the values of the selected parameters that maximises EI among the LHS candidates.
    xBestDict : dict
        Mapping of parameter name to selected value for ``xBest``.
    best_ei : float
        Maximum EI value among the candidate set.
    best_lcb : float
        Maximum LCB value among the candidate set (computed for reference/diagnostics).
    """
    paramSelected = list(paramBounds.keys())
    bounds = np.array(list(paramBounds.values()), dtype=float)  # shape (d,2)
    d = bounds.shape[0]
    f_best = np.nanmin(y)

    # Create LH samples
    seed = cfgOpt.getint('OPTIMISATION', 'seed')
    n_lhs = cfgOpt.getint('OPTIMISATION', 'n_lhs')
    sampler = qmc.LatinHypercube(d=d, seed=seed)
    sample = sampler.random(n=n_lhs)
    X0 = qmc.scale(sample, bounds[:, 0], bounds[:, 1])

    # Predict with pipe
    mu, sigma = pipe.predict(X0, return_std=True)

    log.info("Prediction statistics:")
    log.info("  mu    mean=%.6f  max=%.6f", mu.mean(), mu.max())
    log.info("  sigma mean=%.6f  max=%.6f", sigma.mean(), sigma.max())

    # EI or LCB for minimization
    xi = cfgOpt.getfloat('OPTIMISATION', 'xi')
    ei = expectedImprovement(mu, sigma, f_best, xi)
    lcb = lowerConfidenceBound(mu, sigma)

    xBest = X0[np.argmax(ei)].copy()
    # xBest = X0[np.argmax(lcb)].copy()
    xBestDict = {feat: float(val) for feat, val in zip(paramSelected, xBest)}

    return xBest, xBestDict, np.max(ei), np.max(lcb)


def writeCfgFiles(avalancheDir, paramSets, optimisationType, comModuleName, cfgOpt, counter=None):
    """
    Generate and write configuration files for a given computation module
    based on optimisation results.

    Two configuration files are created:
    1. Using the mean parameter values of the top-N surrogate evaluations.
    2. Using the single best surrogate parameter set.

    Parameters
    ----------
    avalancheDir : str or pathlib.Path
        Path to the avalanche directory.
    paramSets : dict or list of dict
        Parameter set(s) to write to config files
    optimisationType : str
        Optimisation mode ('nonseq' or 'seq'), stored in the config file
        for traceability.
    comModuleName : str
        Name of the computation module (e.g. "com8MoTPSA").
        Used for naming the configuration directory and files.
    counter : int or None, optional
        If provided, use this value as starting index for scenario/file naming.
        Useful when calling this function inside an outer optimisation loop.

    Returns
    -------
    cfgFiles : list of pathlib.Path
        Paths to the written configuration files.
    cfgPath : pathlib.Path
        Directory where configuration files were stored.
        """
    # Allow single dict as input
    if isinstance(paramSets, dict):
        paramSets = [paramSets]

    cfgFiles = []
    avaDir = pathlib.Path(avalancheDir)
    # Directory where generated configuration files will be saved
    cfgPath = avaDir / "Work" / f"{comModuleName}ConfigFiles"
    fU.makeADir(cfgPath)

    # If counter is given, start indexing from there; otherwise start at 0
    start = int(counter) if counter is not None else 0

    for i, xBestDict in enumerate(paramSets):
        idx = start + i  # index used for scenario + filename

        if comModuleName == "com8MoTPSA":
            # Load a fresh module configuration template
            # cfgComModule = cfgUtils.getModuleConfig(com8MoTPSA, toPrint=False)
            cfgComModule = probAna.fetchStartCfg(com8MoTPSA, cfgOpt)
        elif comModuleName == "com1DFA":
            # Load a fresh module configuration template
            # cfgComModule = cfgUtils.getModuleConfig(com1DFA, toPrint=False)
            cfgComModule = probAna.fetchStartCfg(com1DFA, cfgOpt)
        else:
            message = "Functionality not yet implemented for %s" % comModuleName
            log.error(message)
            raise AttributeError(message)

        # Overwrite parameters in their corresponding sections
        for param, val in xBestDict.items():
            section = probAna.fetchParameterSection(cfgComModule, param)
            cfgComModule[section][param] = str(val)

        # Assign unique scenario ID and optimisation type for later identification
        cfgComModule["VISUALISATION"]["scenario"] = str(idx)
        cfgComModule["VISUALISATION"]["sampleMethod"] = optimisationType

        # Write configuration file to disk
        cfgF = cfgPath / f"{idx}_{comModuleName}Cfg.ini"
        with open(cfgF, "w") as configfile:
            cfgComModule.write(configfile)

        cfgFiles.append(cfgF)

    return cfgFiles, cfgPath


def findSimName(finalDF, paramValue, atol=1e-6):
    """
    Return the simName in finalDF whose parameter columns match
    the given paramValue within a numerical tolerance.

    Parameters
    ----------
    finalDF : pandas.DataFrame
        Must contain column simName and parameter columns given in paramValue.
    paramValue : dict or iterable of (name, value)
        Parameter values to match.
    atol : float, optional
        Absolute tolerance for float comparison (default: 1e-6).

    Returns
    -------
    str
        The first matching simName.
    """
    mask = np.ones(len(finalDF), dtype=bool)
    for col, val in dict(paramValue).items():
        s = pd.to_numeric(finalDF[col], errors="coerce")
        mask &= np.isclose(s, float(val), atol=atol, rtol=0)

    matches = finalDF.loc[mask, "simName"]
    return matches.iloc[0]


def loadVariationData(cfgOpt, avaDir, outDir=None):
    """
    Load parameter bounds and selected parameters for optimisation.

    The used parameters and their bounds are not defined directly in this
    function. Instead, they are obtained from results of previous simulation runs.

    Two execution modes are supported and controlled via cfgOpt['PARAM_BOUNDS']['scenario']:

    Scenario 1 (Manual definition):
        - No prior Morris screening.
        - Parameter names and corresponding bounds are loaded from a previously saved pickle file 'paramValuesD.pickle'
          generated by ``runAna4ProbAnaCom8MoTPSA.py`` or  ``runAna4ProbAna.py`` when using com1DFA. The
          parameter variation is therefore not defined within this function, but is determined by the configuration
          specified in ``probAnaCfg.ini``. The ``probAnaCfg.ini`` file contains the settings used to generate the
          initial sample set, including parameter ranges and variation rules.

    This is the standard scenario, as Latin Hypercube Sampling provides good coverage of the parameter space, which is
    important for training the surrogate model.

    Scenario 2 (Morris pre-run):
        - A Morris sensitivity analysis has already been executed.
        - Ranked parameters and their bounds are loaded from 'sa_parameters_bounds.pkl' and morris samples are directly
          used for optimisation.
        - The top-N most influential parameters are selected for optimisation.

    This option is mainly intended for experimental use. Morris samples are designed for sensitivity analysis and
    parameter ranking, but they do not provide an optimal coverage of the parameter space for surrogate-based optimisation.

    Parameters
    ----------
    cfgOpt: configparser.ConfigParser
        Configuration object containing the section 'PARAM_BOUNDS'.
    outDir: pathlib.Path
        Directory containing the Morris output file ('sa_parameters_bounds.pkl').
    avaDir: str
        Directory of the avalanche.

    Returns
    -------
    paramBounds : dict
        Dictionary mapping parameter names to (min, max) tuples.
    paramSelected : list
        List of selected parameter names used for optimisation.
    """
    # Read scenario flag
    scenario = cfgOpt.getint('PARAM_BOUNDS', 'scenario')

    avaName = pathlib.Path(avaDir).name

    # Scenario 1: Morris is not run prior
    if scenario == 1:
        # Load variation data with bounds from pickle file
        inDir2 = pathlib.Path(avaDir, 'Outputs', "ana4Stats")
        paramValuesD = pd.read_pickle(inDir2 / "paramValuesD.pickle")
        paramSelected = paramValuesD['names']
        paramBounds = {
            name: (float(bounds[0]), float(bounds[1]))
            for name, bounds in zip(paramValuesD["names"], paramValuesD["bounds"])
        }
        paramSets = paramValuesD["values"]
    # Scenario 2: Morris run prior
    elif scenario == 2:
        # Load SA data and define how much parameters should be optimized (variation bounds included)
        SiDFSort = pd.read_pickle(outDir / f"{avaName}_sortedSAResultsWithBounds.pkl")
        topN = cfgOpt.getint('PARAM_BOUNDS', 'topN')
        n_available = len(SiDFSort['Parameter'])
        # Check if enough input parameters are available
        if topN > n_available:
            message = (
                f"Invalid number of parameters topN={topN}. Only {n_available} parameters available from Morris sensitivity"
                f" analysis."
            )
            raise ValueError(message)
        paramSelected = list(SiDFSort['Parameter'][:topN])
        paramBounds = dict(zip(paramSelected, SiDFSort['bounds']))
        paramSets = None

    else:
        message = f"Unknown scenario '{scenario}' for variation data. Expected 1 or 2."
        log.error(message)
        raise ValueError(message)

    return paramBounds, paramSelected, paramSets


def loadMorrisConvergenceData(cfgMorrisSA, avalancheDir, avaName):
    """
    Load Morris sensitivity analysis results for convergence plotting.

    Parameters
    ----------
    cfgMorrisSA : configparser.ConfigParser
        Morris configuration object containing section
        'MORRIS_CONVERGENCE' and 'GENERAL'.
    avalancheDir : str or pathlib.Path
        Root avalanche directory.
    avaName : str
        Avalanche name used in filename construction.

    Returns
    -------
    SA_dfs : list[pandas.DataFrame]
        Loaded Morris result DataFrames in the order defined in the ini file.
    r_vals : list[int]
        Corresponding number of trajectories (r) extracted from folder names.
    outputs : list[str]
        Output folder names as defined in the ini file.
    outDir: pathlib.Path
        Path where results are saved to.
    reference_r : int
        The number of trajectories used as refernce for ordering the parameters.
    """

    section = cfgMorrisSA["MORRIS_CONVERGENCE"]

    outputs = []
    r_vals = []

    # Read outputs in ini order
    for key, value in section.items():
        if key.lower().startswith("outputs"):
            outputs.append(value)

            match = re.search(r"R(\d+)", value)
            if match:
                r_vals.append(int(match.group(1)))
            else:
                raise ValueError(
                    f"Could not extract r-value from folder name '{value}'"
                )

    # Build module path
    resultDir = cfgMorrisSA["GENERAL"]["resultDir"]  # e.g. Outputs/ana6MorrisSA
    moduleDir = pathlib.Path(resultDir).name  # e.g. ana6MorrisSA
    comModuleName = cfgMorrisSA["GENERAL"]["modName"]

    filename = f"{avaName}_sortedSAResultsWithBounds.pkl"

    # Load DataFrames
    selectedOutput = cfgMorrisSA['MORRIS_CONVERGENCE']['referenceOutput']

    # Get reference trajectory number, import for determining of the order
    match = re.search(r"R(\d+)", selectedOutput)
    if match:
        reference_r = int(match.group(1))
    else:
        raise ValueError(
            f"Could not extract r-value from referenceOutput '{selectedOutput}'"
        )

    selectedDir = None
    SA_dfs = []

    for out_name in outputs:
        outDir = pathlib.Path(avalancheDir, out_name, moduleDir, comModuleName)
        if out_name == selectedOutput:
            selectedDir = outDir
        with open(outDir / filename, "rb") as fi:
            SA_dfs.append(pickle.load(fi))

    return SA_dfs, r_vals, outputs, selectedDir, reference_r
