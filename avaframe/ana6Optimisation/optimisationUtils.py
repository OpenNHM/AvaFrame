import numpy as np
import pathlib
import pickle
import pandas as pd
import configparser
import os
import time
from scipy.stats import norm, qmc
from datetime import datetime
import re

from sklearn.model_selection import KFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

from avaframe.in3Utils import cfgUtils, initializeProject, logUtils
from avaframe.com8MoTPSA import com8MoTPSA
from avaframe.ana4Stats import probAna
from avaframe.runScripts.runPlotAreaRefDiffs import runPlotAreaRefDiffs
import avaframe.out3Plot.outAna6Plots as saveResults


def calcArealIndicatorsAndAimec(cfgOpt, avalancheDir, ana3AIMEC):
    """
    Calculate areal indicators between reference polygon and simulation and AIMEC analysis and save data in ana3AIMEC and out1Peak.

    Parameters
    ----------
    cfgOpt : configparser.ConfigParser
        Global configuration object. Must contain section "GENERAL"
        with keys:
            - resType
            - thresholdValueSimulation
            - modName
    avalancheDir : str
        Directory containing the directory of the reference avalanche
    ana3AIMEC : module
        AIMEC analysis module providing `fullAimecAnalysis()`.

    """
    # ToDo: to reduce comp. cost: run AIMEC and calcArealIndicators only for new simulations
    # Areal indicators
    resType = cfgOpt["GENERAL"]["resType"]
    thresholdValueSimulation = float(cfgOpt["GENERAL"]["thresholdValueSimulation"])
    modName = cfgOpt["GENERAL"]["modName"]
    runPlotAreaRefDiffs(resType, thresholdValueSimulation, modName)

    # AIMEC
    cfgAIMEC = cfgUtils.getModuleConfig(ana3AIMEC)
    rasterTransfo, resAnalysisDF, plotDict, _, pathDict = ana3AIMEC.fullAimecAnalysis(avalancheDir, cfgAIMEC)


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

            if 'VISUALISATION' in config.sections():
                # config is inifile
                index = config['VISUALISATION']['scenario']

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


def addLossMetrics(df, referenceDF, cfgOpt):
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
    cfgOpt: configparser.ConfigParser
        Config parser of the runOptimisationCfg.ini file

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
    alpha = float(cfgOpt['LOSS_PARAMETERS']['tverskyAlpha'])
    beta = float(cfgOpt['LOSS_PARAMETERS']['tverskyBeta'])

    denomTversky = TP + alpha * FP + beta * FN
    df["tversky_score"] = np.where(denomTversky != 0, TP / denomTversky, 0.0)
    # Subtract 1 to ensure that 0 are good values and 1 bad
    tverskyModified = 1 - df["tversky_score"]

    # Runout
    # RMSE divided with Runout length of Reference
    sRunoutRef = referenceDF['reference_sRunout'].values
    runoutRMSE = df['runoutLineDiff_poly_RMSE']
    runoutRMSENormalised = runoutRMSE / sRunoutRef  # 0 is good, 1 is bad

    # Weights
    wTversky = float(cfgOpt['LOSS_PARAMETERS']['weightTversky'])
    wRunout = float(cfgOpt['LOSS_PARAMETERS']['weightRunout'])

    df['optimisationVariable'] = (wRunout * runoutRMSENormalised.fillna(1) + wTversky * tverskyModified.fillna(1))
    return df


def buildFinalDF(avalancheDir, varParList, cfgOpt):
    """
    Build the final merged DataFrame for a given avalanche.

    Combines parameter sets, AIMEC results, and areal indicators into one DataFrame,
    then computes evaluation metrics 'via addLossMetrics'.

    Parameters
    ----------
    avalancheDir : str
        Path of avalanche directory
    varParList: list of str
        List of parameter names that are varied
    cfgOpt: configparser.ConfigParser
        Config parser of the runOptimisationCfg.ini file

    Returns
    -------
    finalDF : pandas.DataFrame
        Final DataFrame containing:
        - ``simName``
        - ``parameterSet``
        - ``order``
        - Areal indicator columns
        - Evaluation metrics (recall, precision, f1_score, tversky_score, optimisationVariable)
    """

    avaName = avalancheDir.split('/')[-1]

    # Folder where ini files from simulations are
    inDir = pathlib.Path(avalancheDir, 'Outputs/com8MoTPSA/configurationFiles')
    # Read parameterSetDF
    paramSetDF = readParamSetDF(inDir, varParList)

    # Dataframe from AIMEC
    df_aimec = pd.read_csv(
        avalancheDir + '/Outputs/ana3AIMEC/com8MoTPSA/Results_' + avaName + '_ppr_lim_1_w_600resAnalysisDF.csv')
    # Get data of the reference in AIMEC
    referenceDF = pd.read_csv(f"{avalancheDir}/Outputs/ana3AIMEC/com8MoTPSA/referenceDF.csv")

    # Read areal indicators
    arealIndicatorDir = pathlib.Path(avalancheDir, 'Outputs', 'out1Peak', 'arealIndicators.pkl')
    indicatorsDF = readArealIndicators(arealIndicatorDir)

    # Merge df's
    df_merged = pd.merge(paramSetDF, df_aimec, on='simName', how='inner')
    df_merged = df_merged.merge(indicatorsDF, on="simName", how="left")

    # Add optimisation variables
    finalDF = addLossMetrics(df_merged, referenceDF, cfgOpt)
    return finalDF


def createDFParameterLoss(df, paramSelected):
    """
    Create DataFrames linking selected parameters with the loss function.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame that contains the selected parameters and their values as well as the loss function.
    paramSelected : list of str
        Subset of parameters to include in the output DataFrames.

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

    # GP kernel with Matern-Covariance
    matern_nu = float(cfgOpt['OPTIMISATION']['matern_nu'])
    kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))  # Output variance tells how strong Y varies
            * Matern(length_scale=np.ones(n_features),
                     length_scale_bounds=(1e-2, 1e2),  # in z (variance) space
                     nu=matern_nu)
            + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e-1))
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-8,
        normalize_y=True,
        n_restarts_optimizer=10,
        random_state=0,
    )

    # Pipelines (feature scaling + model)
    gp_pipe = Pipeline([("x_scaler", StandardScaler()), ("model", gp)])
    return X, y, gp_pipe


def KFoldCV(X, y, pipe, cfgOpt, outDir, avaName, pipeName):
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
    k = int(cfgOpt['OPTIMISATION']['k'])
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
    matern_nu = float(cfgOpt['OPTIMISATION']['matern_nu'])
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

    print(f"\n{pipeName}, {k}-fold CV:")
    for split in ("test", "train"):
        print(f"  {split.capitalize()} metrics:")
        for m in ("rmse", "mae", "r2"):
            arr = scores[f"{split}_{m}"]
            print(f"    {m.upper():<4}: {arr.mean():.4g} ± {arr.std():.4g}")

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
    seed = int(cfgOpt['OPTIMISATION']['seed'])
    n_lhs = int(cfgOpt['OPTIMISATION']['n_lhs'])
    sampler = qmc.LatinHypercube(d=d, seed=seed)
    sample = sampler.random(n=n_lhs)
    X0 = qmc.scale(sample, bounds[:, 0], bounds[:, 1])

    # Prediction of the loss with GP-Model
    mu, sigma = pipe.predict(X0, return_std=True)
    # Convert X0 to pandas df for analyze function
    df_candidates = pd.DataFrame(X0, columns=paramSelected)
    n_top_samples = int(cfgOpt['OPTIMISATION']['n_surrogate_top'])
    topNStat, _ = analyzeTopCandidates(df_candidates, mu, sigma, paramSelected, N=n_top_samples)
    return topNStat


def analyzeTopCandidates(df_candidates, mu, sigma, param_cols, N=5):
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
    N : int, optional
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

    print(f"\n🔍 Mittelwerte ± Std (Top {N}):")
    for p in param_cols:
        m, s = mean_params[p], std_params[p]
        perc = (s / m * 100) if m != 0 else np.nan
        print(f"  {p:30s}: {m:.6f} ± {s:.6f} ({perc:.1f}%)")
    perc_mu = (std_mu / mean_mu * 100) if mean_mu != 0 else np.nan
    perc_sigma = (std_sigma / mean_sigma * 100) if mean_sigma != 0 else np.nan
    print(f"📉 mu:    {mean_mu:.4f} ± {std_mu:.4f} ({perc_mu:.1f}%)")
    print(f"📊 sigma: {mean_sigma:.4f} ± {std_sigma:.4f} ({perc_sigma:.1f}%)")

    # Best single point
    idx_best = np.argmin(mu)
    best_params = df_candidates.iloc[idx_best].copy()
    best_loss = mu[idx_best]
    best_sigma = sigma[idx_best]

    print("\n🔍 Best single Parameter combination:")
    for p in param_cols:
        print(f"  {p:30s}: {best_params[p]:.4f}")
    print(f"📉 mu:    {best_loss:.4f}")
    print(f"📊 sigma: {best_sigma:.4f}")

    return {
        f"TopNBest": {
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


def expectedImprovement(mu, sigma, f_best, xi=0):
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
    xi : float, optional
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
    seed = int(cfgOpt['OPTIMISATION']['seed'])
    n_lhs = int(cfgOpt['OPTIMISATION']['n_lhs'])
    sampler = qmc.LatinHypercube(d=d, seed=seed)
    sample = sampler.random(n=n_lhs)
    X0 = qmc.scale(sample, bounds[:, 0], bounds[:, 1])

    # Predict with pipe
    mu, sigma = pipe.predict(X0, return_std=True)

    print(mu.mean(), mu.max(), sigma.mean(), sigma.max())

    # EI or LCB for minimization
    ei = expectedImprovement(mu, sigma, f_best)
    lcb = lowerConfidenceBound(mu, sigma)

    xBest = X0[np.argmax(ei)].copy()
    # xBest = X0[np.argmax(lcb)].copy()
    xBestDict = {feat: float(val) for feat, val in zip(paramSelected, xBest)}

    return xBest, xBestDict, np.max(ei), np.max(lcb)


def runCom8MoTPSA(avalancheDir, xBestDict, cfgMain, i=0, optimisationType=None):
    """
    Based on the default runCom8MoTPSA function in com8MoTPSA/runCom8MoTPSA.py file, but overrides parameter values in
    the module configuration using the values provided in ``xBestDict``. It also assigns a unique visualisation scenario
    identifier and records the sampling method for traceability.

    Parameters
    ----------
    avalancheDir: str
        Path to avalanche directory.
    xBestDict : dict
        Mapping of parameter name to selected value for ``xBest``.
    cfgMain : configparser.ConfigParser
        General configuration for avaframe.
    i : int, optional
        Counter for identifying the number of iterations.
    optimisationType: str, optional
        Name of the optimisation type, sequential or non-sequential.

    Returns
    ---------
    simName: str
        Name of the simulation.
    """
    # Time the whole routine
    startTime = time.time()

    # log file name; leave empty to use default runLog.log
    logName = 'runCom8MoTPSA'

    # Start logging
    log = logUtils.initiateLogger(avalancheDir, logName)
    log.info('MAIN SCRIPT')
    log.info('Current avalanche: %s', avalancheDir)
    # ----------------
    # Clean input directory(ies) of old work and output files
    # If you just created the ``avalancheDir`` this one should be clean but if you
    # already did some calculations you might want to clean it::
    initializeProject.cleanSingleAvaDir(avalancheDir, deleteOutput=False)
    # Get module config
    cfgCom8MoTPSA = cfgUtils.getModuleConfig(com8MoTPSA, toPrint=False)

    # overwrite com8MoTPSACfg with xBest values
    for param, val in xBestDict.items():
        # print(param, val)
        section = probAna.fetchParameterSection(cfgCom8MoTPSA, param)
        cfgCom8MoTPSA[section][param] = str(val)
    # give visualisation unique scenario for identifying later
    cfgCom8MoTPSA['VISUALISATION']['scenario'] = str(i)
    if optimisationType == 'nonSeq':
        cfgCom8MoTPSA["VISUALISATION"]["sampleMethod"] = 'nonSeq'
    else:
        cfgCom8MoTPSA["VISUALISATION"]["sampleMethod"] = 'EI/LCB'

    # ----------------
    # Run psa
    simName = com8MoTPSA.com8MoTPSAMain(cfgMain, cfgInfo=cfgCom8MoTPSA, returnSimName=True)
    # Print time needed
    endTime = time.time()
    log.info('Took %6.1f seconds to calculate.' % (endTime - startTime))

    return simName


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


def loadVariationData(cfgOpt, outDir, avaDir):
    """
    Load parameter bounds and selected parameters for optimisation. Two execution modes are supported, controlled via
    cfgOpt['PARAM_BOUNDS']['scenario']:

    Scenario 1 (Morris pre-run):
        - A Morris sensitivity analysis has already been executed.
        - Ranked parameters and their bounds are loaded from
          'sa_parameters_bounds.pkl'.
        - The top-N most influential parameters are selected for optimisation.

    Scenario 2 (Manual definition):
        - No prior Morris screening.
        - Parameter names and corresponding bounds are loaded from a previously saved pickle file generated by
         ``runAna4ProbAnaCom8MoTPSA.py``. The parameter variation is therefore not defined within this function, but is
          determined by the configuration specified in ``probAnaCfg.ini``. The ``probAnaCfg.ini`` file contains the
          settings used to generate the initial sample set, including parameter ranges and variation rules.


    Parameters
    ----------
    cfgOpt: configparser.ConfigParser
        Configuration object containing the section 'PARAM_BOUNDS'.
    outDir: pathlib.Path
        Directory containing the Morris output file
        ('sa_parameters_bounds.pkl').
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
    scenario = int(cfgOpt['PARAM_BOUNDS']['scenario'])

    avaName = avaDir.split('/')[-1]

    # Scenario 1: Morris run prior
    if scenario == 1:
        # Load SA data and define how much parameters should be optimized (variation bounds included)
        SiDFSort = pd.read_pickle(outDir / f"{avaName}_sortedSAResultsWithBounds.pkl")
        topN = int(cfgOpt['PARAM_BOUNDS']['topN'])
        paramSelected = list(SiDFSort['Parameter'][:topN])
        paramBounds = dict(zip(paramSelected, SiDFSort['bounds']))

    # Scenario 2: Morris is not run prior
    else:
        # Load variation data with bounds from pickle file
        inDir2 = pathlib.Path(avaDir, 'Outputs', "ana4Stats")
        paramValuesD = pd.read_pickle(inDir2 / "paramValuesD.pickle")
        paramSelected = paramValuesD['names']
        paramBounds = {
            name: (float(bounds[0]), float(bounds[1]))
            for name, bounds in zip(paramValuesD["names"], paramValuesD["bounds"])
        }
    return paramBounds, paramSelected


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
