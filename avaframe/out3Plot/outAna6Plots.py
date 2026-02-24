import numpy as np
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
from adjustText import adjust_text
from datetime import datetime


def barplotSA(SiDF, avaName, outDir):
    """
    Create a bar plot of Morris sensitivity results.

    Bars show μ* as percentage of the total sensitivity, with bar width
    proportional to σ. A vertical dashed line is drawn where the cumulative μ*
    first reaches 80%.

    Parameters
    ----------
    SiDF : pandas.DataFrame
        DataFrame with at least the following columns:
        - ``Parameter`` : str, parameter names
        - ``mu_star`` : float, mean absolute elementary effect
        - ``mu_star_conf`` : float, optional, confidence interval of μ*
        - ``sigma`` : float, standard deviation of elementary effects
    avaName : str
        Avalanche name, used in the saved filename.
    outDir : str or pathlib.Path
        Directory where the plot is saved.
    """

    # 1) Sort by mu_star (descending)
    df = SiDF.sort_values("mu_star", ascending=False).reset_index(drop=True)

    # 2) Normalize μ* so the percentages sum to 100
    total_mu = df["mu_star"].sum()
    mu_pct = 100 * df["mu_star"] / total_mu

    # scale the confidence interval to the same percentage units
    if "mu_star_conf" in df:
        mu_pct_conf = 100 * df["mu_star_conf"] / total_mu
    else:
        mu_pct_conf = None

    # 3) Map σ to bar widths
    wmin, wmax = 0.3, 0.9
    rng = df["sigma"].max() - df["sigma"].min()
    sigma_norm = (df["sigma"] - df["sigma"].min()) / (rng if rng else 1)
    bar_widths = wmin + (wmax - wmin) * sigma_norm

    # 4) Plot
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(
        x, mu_pct, width=bar_widths, edgecolor="black",
        capsize=3 if mu_pct_conf is not None else 0
    )

    ax.set_xticks(x, df["Parameter"], rotation=60, ha="right")
    ax.set_ylabel("μ* (% of total)")
    ax.set_title("Sensitivity: μ* (percent of total) with σ mapped to bar width")

    # Show value labels
    for xi, yi in zip(x, mu_pct):
        ax.text(xi, yi, f"{yi:.1f}%", ha="center", va="bottom", fontsize=9)

    # 7) Save figure
    # Include date, format: YYYYMMDD
    date = datetime.now().strftime("%Y%m%d")
    figName = f"{outDir}/{avaName}_paramRanking_{date}.png"
    plt.savefig(figName, dpi=300, bbox_inches="tight")


def scatterplotSA(SiDF, avaName, outDir):
    """
    Create a scatter plot of Morris sensitivity results (μ* vs σ).

    Parameters
    ----------
    SiDF : pandas.DataFrame
        DataFrame with at least the following columns:
        - ``Parameter`` : str, parameter names
        - ``mu_star`` : float, mean absolute elementary effect
        - ``sigma`` : float, standard deviation of elementary effects
    avaName : str
        Avalanche name, used in the saved filename.
    outDir : str or pathlib.Path
        Directory where the plot is saved.
    """

    # Scatter Plot
    plt.figure(figsize=(12, 7))
    plt.scatter(SiDF['mu_star'], SiDF['sigma'], color='blue')

    # Annotate name to the points
    for i, txt in enumerate(SiDF['Parameter']):
        plt.annotate(txt, (SiDF['mu_star'][i], SiDF['sigma'][i]), fontsize=9, xytext=(5, 5), textcoords='offset points')

    # Label and title
    plt.xlabel('mu_star (Einflussstärke)', fontsize=12)
    plt.ylabel('sigma (Nichtlinearität / Interaktionen)', fontsize=12)
    plt.title('Morris Sensitivitätsanalyse: mu_star vs sigma', fontsize=14)
    plt.grid(True)

    # Save figure
    # Include date, format: YYYYMMDD
    date = datetime.now().strftime("%Y%m%d")
    figName = f"{outDir}/{avaName}_Scatterplot_{date}.png"
    plt.savefig(figName, dpi=300, bbox_inches="tight")


def scatterplotUncertaintySA(SiDF, avaName, outDir):
    """
    Create a scatter plot of Morris sensitivity results with uncertainty.
    Plots μ* vs σ with horizontal error bars given by ``mu_star_conf``.

    Parameters
    ----------
    SiDF : pandas.DataFrame
        DataFrame with at least the following columns:
        - ``Parameter`` : str, parameter names
        - ``mu_star`` : float, mean absolute elementary effect
        - ``mu_star_conf`` : float, confidence interval of μ*
        - ``sigma`` : float, standard deviation of elementary effects
    avaName : str
        Avalanche name, used in the saved filename.
    outDir : str or pathlib.Path
        Directory where the plot is saved.
    """
    # Plot with error bars
    plt.figure(figsize=(12, 7))
    plt.errorbar(
        SiDF['mu_star'], SiDF['sigma'],
        xerr=SiDF['mu_star_conf'],
        fmt='o', color='blue', ecolor='gray', elinewidth=1.5, capsize=4
    )

    # Annotations with adjustText
    texts = [plt.text(SiDF['mu_star'][i], SiDF['sigma'][i], SiDF['Parameter'][i], fontsize=9) for i in range(len(SiDF))]
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

    # Axes and layout
    plt.xlabel('mu_star (Einflussstärke)', fontsize=12)
    plt.ylabel('sigma (Nichtlinearität / Interaktionen)', fontsize=12)
    plt.title('Morris Sensitivitätsanalyse: mu_star vs sigma (mit Unsicherheit)', fontsize=14)

    # Save figure
    # Include date, format: YYYYMMDD
    date = datetime.now().strftime("%Y%m%d")
    figName = f"{outDir}/{avaName}_ScatterplotUncertainty_{date}.png"
    plt.savefig(figName, dpi=300, bbox_inches="tight")


def BOConvergencePlot(finalDF, avaName, outDir):
    """
    This function visualises the evolution of the optimisation variable
    (loss) across different sampling phases:

        1. Latin hypercube sampling
        2. Bayesian optimisation (EI/LCB)
        3. Optional non-sequential surrogate-based sampling

    Parameters
    ----------
    finalDF : pandas.DataFrame
        DataFrame containing simulation results.

    avaName : str
        Name of the avalanche. Used for naming the output figure.

    outDir : pathlib.Path
        Directory where the convergence plot will be saved.

    """
    # Color palette
    c_best = '#4e79a7'
    c_bo = '#59a14f'
    c_lhs = '#76b7b2'
    c_nonseq = '#e15759'

    df = finalDF.dropna(
        subset=["sampleMethods", "order", "optimisationVariable"]
    ).copy()

    # Split by sampling method
    latin = df[df["sampleMethods"] == "latin"].sort_values("order")
    bo = df[df["sampleMethods"] == "EI/LCB"].sort_values("order")
    nonseq = df[df["sampleMethods"] == "nonSeq"].sort_values("order")

    # Iteration axis
    current_offset = 0

    if not latin.empty:
        latin["iter"] = latin["order"]
        current_offset = latin["iter"].max() + 1

    if not bo.empty:
        bo["iter"] = bo["order"] + current_offset
        current_offset = bo["iter"].max() + 1

    if not nonseq.empty:
        nonseq = nonseq.sort_index().copy()
        nonseq["iter"] = np.arange(current_offset, current_offset + len(nonseq))
        current_offset = nonseq["iter"].max() + 1

    # Combine for best-so-far
    all_parts = [latin, bo]
    if not nonseq.empty:
        all_parts.append(nonseq)

    all_df = pd.concat(all_parts).sort_values("iter")
    all_df["best_so_far"] = all_df["optimisationVariable"].cummin()

    # Plot
    fig, ax = plt.subplots(figsize=(9, 5))

    if not latin.empty:
        ax.scatter(
            latin["iter"],
            latin["optimisationVariable"],
            s=35,
            alpha=0.7,
            color=c_lhs,
            label="Latin hypercube"
        )

    if not bo.empty:
        ax.scatter(
            bo["iter"],
            bo["optimisationVariable"],
            s=40,
            alpha=0.85,
            color=c_bo,
            label="Bayesian optimisation (EI/LCB)"
        )

    if not nonseq.empty:
        ax.scatter(
            nonseq["iter"],
            nonseq["optimisationVariable"],
            s=40,
            alpha=0.85,
            color=c_nonseq,
            label="Non-sequential sampling"
        )

    ax.plot(
        all_df["iter"],
        all_df["best_so_far"],
        linewidth=1.5,
        color=c_best,
        label="Best-so-far"
    )

    # Add separator lines between sampling phases (if applicable)
    if not latin.empty:
        ax.axvline(latin["iter"].max() + 0.5, linestyle="--", linewidth=1, color="black")
    if not bo.empty:
        ax.axvline(bo["iter"].max() + 0.5, linestyle="--", linewidth=1, color="black")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Optimisation variable (loss)")
    ax.set_title("Convergence: Latin hypercube → Bayesian optimisation")
    ax.legend(frameon=False, loc='best')

    fig.tight_layout()

    # Save figure
    date = datetime.now().strftime("%Y%m%d")
    figName = f"{outDir}/{avaName}_BOConvergence_{date}.png"
    fig.savefig(figName, dpi=300, bbox_inches="tight")
    plt.close(fig)


def BOBoxplot(paramLossDF, avaName, outDir, N=10):
    """
    Create boxplots of the top-N parameter sets based on loss.

    This function selects the N best-performing simulations (lowest loss)
    and visualises the distribution of their parameter values using boxplots.
    Each parameter is plotted in a separate subplot.

    Parameters
    ----------
    paramLossDF : pandas.DataFrame
        DataFrame containing model parameters and corresponding loss values.

    avaName : str
        Name of the avalanche. Used for naming the output figure.

    outDir : pathlib.Path
        Directory where the boxplot figure will be saved.

    N : int, optional
        Number of best-performing simulations (lowest loss) to include.
        Default is 10.
    """
    df_best = paramLossDF.nsmallest(N, "Loss")
    param_cols = paramLossDF.columns.drop('Loss')

    fig, axes = plt.subplots(2, 4, figsize=(10, 6))
    axes = axes.flatten()
    for ax, col in zip(axes, param_cols):
        ax.boxplot(df_best[col])
        ax.set_xticklabels([col])

    # If less than 8 parameters, remove empty axes
    for ax in axes[len(param_cols):]:
        ax.axis("off")
    plt.tight_layout()

    # Save figure
    # Include date, format: YYYYMMDD
    date = datetime.now().strftime("%Y%m%d")
    figName = f"{outDir}/{avaName}_BOBoxplot_{date}.png"
    plt.savefig(figName, dpi=300, bbox_inches="tight")


def BOBoxplotNormalised(paramLossDF, paramBounds, avaName, outDir, N=10):
    """
    Create normalized boxplots of parameters for the top-N simulations.

    This function selects the N best-performing simulations (lowest loss)
    and visualises the distribution of their parameter values after
    min–max normalization based on predefined parameter bounds.

    Parameters
    ----------
    paramLossDF : pandas.DataFrame
        DataFrame containing model parameters and corresponding loss values.

    paramBounds : dict
        Dictionary mapping parameter names to their (min, max) bounds:
            {parameter_name: (lower_bound, upper_bound)}

        Bounds are used for min–max normalization.

    avaName : str
        Name of the avalanche. Used for naming the output figure.

    outDir : pathlib.Path
        Directory where the normalized boxplot figure will be saved.

    N : int, optional
        Number of best-performing simulations (lowest loss) to include.
        Default is 10.
    """
    df_best = paramLossDF.nsmallest(N, "Loss")
    param_cols = paramLossDF.columns.drop('Loss')

    # normalize using parameter bounds
    data = [
        (df_best[c] - paramBounds[c][0]) / (paramBounds[c][1] - paramBounds[c][0])
        for c in param_cols
    ]

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.boxplot(data)
    ax.set_xticks(range(1, len(param_cols) + 1))
    ax.set_xticklabels(param_cols, rotation=45, ha="right", fontsize=14)
    ax.set_ylabel("Normalized value (min–max)", fontsize=14)
    ax.set_title(f"Normalized parameter distributions (best {N})", fontsize=16)
    ax.tick_params(axis="y", labelsize=12)
    fig.tight_layout()

    # Save figure
    # Include date, format: YYYYMMDD
    date = datetime.now().strftime("%Y%m%d")
    figName = f"{outDir}/{avaName}_BOBoxplotNormalised_{date}.png"
    plt.savefig(figName, dpi=300, bbox_inches="tight")


def saveKFoldCVPrintImage(scores, pipeName, k, out_path):
    """
    Save a summary of K-fold cross-validation results as an image. The output image contains metrics for both
    training and test sets.

    Reported metrics per split:
        - RMSE (Root Mean Squared Error)
        - MAE  (Mean Absolute Error)
        - R²   (Coefficient of determination)

    Parameters
    ----------
    scores : dict
        Dictionary containing cross-validation results. Expected keys:
            - "train_rmse", "train_mae", "train_r2"
            - "test_rmse",  "test_mae",  "test_r2"

    pipeName : str
        Name of the model or pipeline. Included in the figure header.

    k : int
        Number of folds used in cross-validation.

    out_path : str or pathlib.Path
        File path where the generated image will be saved.
    """
    lines = [f"{pipeName}, {k}-fold CV:\n"]
    for split in ("test", "train"):
        lines.append(f"{split.capitalize()} metrics:")
        for m in ("rmse", "mae", "r2"):
            arr = scores[f"{split}_{m}"]
            lines.append(f"  {m.upper():<4}: {arr.mean():.4g} ± {arr.std():.4g}")
        lines.append("")

    text = "\n".join(lines)

    fig = plt.figure(figsize=(6, 4))
    plt.axis("off")
    plt.text(0.01, 0.99, text, va="top", family="monospace")

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def saveBestorCurrentModelrun(finalDF, paramSelected, ei=None, lcb=None, simName=None, csv_path='dummy.csv'):
    """
    Save either the best model run (based on optimisationVariable) or a
    specified simulation (simName) to a CSV file.

    Parameters
    ----------
    finalDF : pandas.DataFrame
        containing all simulation results and optimization metrics.
    paramSelected : list of str
        List of parameter names that should be includedvin the exported output.
    ei : float, optional
        Expected Improvement value (used in Bayesian optimization).
        If None, the column will be created with None.
    lcb : float, optional
        Lower Confidence Bound value (used in Bayesian optimization).
        If None, the column will be created with None.
    simName : str, optional
        If provided, the row corresponding to this simulation name is saved.
        If None, the row with the minimal optimisationVariable is saved.
    csv_path : str or pathlib.Path, optional
        Path to the CSV file.

    Notes
    -----
    Only a subset of relevant columns (including selected parameters)
    is written to the output file.
    """
    # Subset df, save only important entries
    cols_keep = ['simName', 'sampleMethods', 'order', 'Simulation time (s)', 'Minimum time step (s)',
                 'Initial CFL number (-)', 'TP_SimRef_cells', 'TP_SimRef_area', 'FP_SimRef_cells', 'FP_SimRef_area',
                 'FN_SimRef_cells', 'FN_SimRef_area', 'recall', 'precision', 'f1_score', 'tversky_score',
                 'optimisationVariable']
    columns_keep = cols_keep[:6] + [p for p in paramSelected if p not in cols_keep] + cols_keep[6:]
    df = finalDF[columns_keep]

    if simName is not None:
        row = df.loc[df['simName'] == simName].copy()
    else:
        idx = df['optimisationVariable'].idxmin()
        row = df.loc[[idx]].copy()

    # Ensure the optional columns always exist
    if ei is not None:
        row["ei"] = ei
    else:
        row['ei'] = None
    if lcb is not None:
        row["lcb"] = lcb
    else:
        row['lcb'] = None
    # Write to csv
    path = pathlib.Path(csv_path)
    row.to_csv(path, mode="a", index=False, header=not path.exists())


def saveTopCandidates(finalDF, paramSelected, cfgOpt, results_dict=None, out_path="analysisTable.png", title=None,
                      simNameMean=None, simNameBest=None):
    """
    Create result table(s) for surrogate/model top candidates and save as PNG and CSV.

    Behavior
    --------
    - If `results_dict` is provided: creates up to two tables
        1) Surrogate summary (TopNBest mean/std + surrogate single best in "best" column)
           Optionally appends a row "optimisationVariable" where:
             - "mean" is filled from `simNameMean` (lookup in finalDF)
             - "best" is filled from `simNameBest` (lookup in finalDF)
        2) Model summary (Top N + single best), where N is read from cfgOpt['OPTIMISATION']['n_model_top']
    - If `results_dict` is None: creates only the model table (2).

    Outputs
    -------
    - PNG at `out_path`
    - CSV next to the PNG with suffix `_tables.csv` (tidy format with a `table` column)

    Parameters
    ----------
    finalDF : pandas.DataFrame
        Must contain column "optimisationVariable".
        If simNameMean/simNameBest is used, must contain column "simName".
        Must contain parameter columns listed in `paramSelected`.
    paramSelected : list[str]
        Parameter column names to summarize for the model table.
    cfgOpt : configparser.ConfigParser-like
        Needs:
          - cfgOpt['OPTIMISATION']['n_surrogate_top']
          - cfgOpt['OPTIMISATION']['n_model_top']
    results_dict : dict | None, optional
        If provided, expected keys:
          - "TopNBest": dict with "mean_params", "std_params", "mean_mu", "std_mu", "mean_sigma", "std_sigma"
          - "Best": dict with "params", "mu", "sigma"
    out_path : str | pathlib.Path, optional
        Path to save PNG figure.
    title : str | None, optional
        Optional global title for the PNG figure.
    simNameMean : str | None, optional
        Simulation name whose model optimisationVariable is placed into the surrogate table row
        "optimisationVariable" under the "mean" column.
    simNameBest : str | None, optional
        Simulation name whose model optimisationVariable is placed into the surrogate table row
        "optimisationVariable" under the "best" column.

    Returns
    -------
    pathlib.Path
        Path to the saved PNG figure.
    """
    tables = []

    # ==========================================================
    # Surrogate table (optional)
    # ==========================================================
    if results_dict is not None:
        top = results_dict["TopNBest"]

        # --- normalize mean/std params to Series ---
        mean_params = top["mean_params"]
        std_params = top["std_params"]

        # If params are list/tuple of (name,value) pairs, convert to dict first
        if not isinstance(mean_params, dict):
            mean_params = dict(mean_params)
        if not isinstance(std_params, dict):
            std_params = dict(std_params)

        df_top = pd.DataFrame(
            {
                "mean": pd.Series(mean_params, dtype="float64"),
                "std": pd.Series(std_params, dtype="float64"),
            }
        )
        df_top["relStd [%]"] = relstd(df_top["std"], df_top["mean"])

        extra = pd.DataFrame(
            {
                "mean": [top["mean_mu"], top["mean_sigma"]],
                "std": [top["std_mu"], top["std_sigma"]],
            },
            index=["optimisationVariable_Surrogate", "sigma"],
        )
        extra["relStd [%]"] = relstd(extra["std"], extra["mean"])

        df_top = pd.concat([df_top, extra], axis=0)

        # --- create/attach "best" column from surrogate Best ---
        best = results_dict["Best"]
        best_params = best["params"]
        if not isinstance(best_params, dict):
            best_params = dict(best_params)

        best_series = pd.Series(best_params, dtype="float64")
        best_series.loc["optimisationVariable_Surrogate"] = float(best["mu"])
        best_series.loc["sigma"] = float(best["sigma"])
        df_top["best"] = best_series

        # --- add model optimisationVariable row: mean and/or best ---
        def _get_optvar_for_sim(sim_name, label):
            if "simName" not in finalDF.columns:
                raise ValueError(f"{label} set but finalDF has no 'simName' column.")
            sel = finalDF.loc[finalDF["simName"] == sim_name, "optimisationVariable"]
            if sel.empty:
                raise ValueError(f"{label}='{sim_name}' not found in finalDF['simName'].")
            return float(pd.to_numeric(sel.iloc[0], errors="coerce"))

        if simNameMean is not None or simNameBest is not None:
            # Ensure row exists
            if "optimisationVariable" not in df_top.index:
                df_top.loc["optimisationVariable", ["mean", "std", "relStd [%]", "best"]] = [np.nan, np.nan, np.nan,
                                                                                             np.nan]

            if simNameMean is not None:
                df_top.loc["optimisationVariable", ["mean", "std", "relStd [%]"]] = [
                    _get_optvar_for_sim(simNameMean, "simNameMean"),
                    np.nan,
                    np.nan,
                ]
            if simNameBest is not None:
                df_top.loc["optimisationVariable", "best"] = _get_optvar_for_sim(simNameBest, "simNameBest")

        n_surrogate_top = int(cfgOpt["OPTIMISATION"]["n_surrogate_top"])
        mean_tag = f" ({simNameMean})" if simNameMean else ""
        best_tag = f" ({simNameBest})" if simNameBest else ""

        t1 = (
            f"Surrogate: Mean of Top {n_surrogate_top} Best{mean_tag} "
            f"+ Single Best{best_tag}"
        )
        tables.append((fmt_df(df_top), t1))

    # ==========================================================
    # Model table (always)
    # ==========================================================
    if "optimisationVariable" not in finalDF.columns:
        raise ValueError("finalDF must contain column 'optimisationVariable'.")

    best_idx = finalDF["optimisationVariable"].idxmin()

    n_model_top = int(cfgOpt["OPTIMISATION"]["n_model_top"])
    topN = finalDF.nsmallest(n_model_top, "optimisationVariable")

    df_topN = summary_table(topN, paramSelected, best_row=finalDF.loc[best_idx])

    opt_mean = pd.to_numeric(topN["optimisationVariable"], errors="coerce").mean()
    opt_std = pd.to_numeric(topN["optimisationVariable"], errors="coerce").std()

    df_topN.loc["optimisationVariable", ["mean", "std", "relStd [%]", "best"]] = [
        opt_mean,
        opt_std,
        relstd(opt_std, opt_mean),
        pd.to_numeric(finalDF.loc[best_idx, "optimisationVariable"], errors="coerce"),
    ]

    best_sim = finalDF.at[best_idx, "simName"] if "simName" in finalDF.columns else ""
    t2 = f"Model: Mean of Top {n_model_top} Best + Single Best{f' ({best_sim})' if best_sim else ''}"
    tables.append((fmt_df(df_topN), t2))

    # ==========================================================
    # Plot
    # ==========================================================
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig_h = 1.2 + 0.38 * sum(len(df) for df, _ in tables)
    fig, axes = plt.subplots(len(tables), 1, figsize=(10, fig_h))
    axes = [axes] if len(tables) == 1 else axes

    if title:
        fig.suptitle(title, fontsize=14, y=0.99)

    for ax, (df_disp, t) in zip(axes, tables):
        ax.axis("off")
        tbl = ax.table(
            cellText=df_disp.values,
            rowLabels=df_disp.index.tolist(),
            colLabels=df_disp.columns.tolist(),
            loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.2)
        ax.set_title(t, fontsize=12, pad=10)

    plt.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ==========================================================
    # CSV export (tidy)
    # ==========================================================
    csv_path = out_path.with_name(f"{out_path.stem}_tables.csv")
    frames = []
    for df_disp, t in tables:
        tmp = df_disp.copy().reset_index(names=["row"])
        tmp.insert(0, "table", t)
        frames.append(tmp)
    pd.concat(frames, ignore_index=True).to_csv(csv_path, index=False)

    return out_path


def formatSig(x):
    """
    Format numbers for display in tables (significant digits, sci notation for small values).
        - NaN or non-numeric values → ""
        - 0 → "0"
        - |x| < 1e-3 → scientific notation with 2 decimal places
        - |x| < 100 → 2 significant digits
        - |x| ≥ 100 → rounded integer

    Parameters
    ----------
    x : Any
        Value to be formatted. If conversion to float fails or the value
        is NaN, an empty string is returned.

    Returns
    -------
    str
        Formatted string representation suitable for compact table output.
    """
    # --- Number formatting ---
    try:
        x = float(x)
    except (TypeError, ValueError):
        return ""
    if pd.isna(x):
        return ""
    if x == 0.0:
        return "0"
    if abs(x) < 1e-3:
        return f"{x:.2e}"  # small numbers in scientific notation
    elif abs(x) < 100:
        return f"{x:.2g}"  # 2 significant digits
    else:
        return str(int(round(x)))


def fmt_df(df):
    """
    Apply formatSig to common summary columns for prettier display/export.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing summary statistics.

    Returns
    -------
    pandas.DataFrame
        A copy of the input DataFrame with selected columns formatted
        as strings for display purposes.

    """
    g = df.copy()
    for c in ["mean", "std", "relStd [%]", "best"]:
        if c in g.columns:
            g[c] = g[c].map(formatSig)
    return g


def relstd(std, mean):
    """
    Relative std in percent; returns NaN if mean is 0 or NaN.

    Parameters
    ----------
    std : array-like or scalar
        Standard deviation values.

    mean : array-like or scalar
        Mean values corresponding to `std`.

    Returns
    -------
    numpy.ndarray
        Relative standard deviation in percent. Returns NaN where:
            - mean is 0
            - mean is NaN
            - conversion to numeric fails
    """
    mean = pd.to_numeric(mean, errors="coerce")
    std = pd.to_numeric(std, errors="coerce")
    return np.where((mean == 0) | pd.isna(mean), np.nan, std / mean * 100.0)


def summary_table(df, cols, best_row=None):
    """
    Compute summary statistics (mean, standard deviation, relative standard deviation,
    and optional best values) for selected numeric columns.

    The function converts the specified columns to numeric (coercing errors to NaN),
    computes column-wise statistics, and returns them in a compact summary table.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the data to summarise.

    cols : list-like
        List of column names for which summary statistics should be computed.

    best_row : pandas.Series or dict-like, optional
        Row containing reference or "best" parameter values. If provided,
        the values corresponding to `cols` are included in the output under
        the column "best". If None, the "best" column is filled with NaN.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by `cols` containing:
            - "mean"       : Column-wise mean
            - "std"        : Column-wise standard deviation
            - "relStd [%]" : Relative standard deviation in percent
            - "best"       : Reference/best values (if provided)
    """
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    out = pd.DataFrame({"mean": X.mean(), "std": X.std()})
    out["relStd [%]"] = relstd(out["std"], out["mean"])
    if best_row is not None:
        out["best"] = pd.to_numeric(best_row[cols], errors="coerce").values
    else:
        out["best"] = np.nan
    return out


def plotMorrisConvergence(SA_dfs, r_values, reference_r, k=None, outpath=None, title=None):
    """
    Plot convergence of Morris mu_star sensitivities.

    Parameters
    ----------
    SA_dfs : list of pandas.DataFrame
        Morris result DataFrames.
        Must contain columns: 'mu_star', 'Parameter'
    r_values : list[int]
        Number of trajectories corresponding to SA_dfs.
    reference_r : int
        Trajectory count r that defines the reference dataset (parameter ranking/order).
    k : int or None, optional
        Number of top parameters to plot. If None, all parameters are plotted.
    outpath : str or pathlib.Path, optional
        If given, figure is saved to this path.
    title : str, optional
        Plot title.
    Notes
   --------
   - This code was written with AI.

    """

    # --------------------------------------------------
    # Normalize (in-place safe) + ALIGN BY PARAMETER NAME
    # --------------------------------------------------
    SA_dfs_aligned = []
    for df in SA_dfs:
        d = df.copy()
        if "pct" not in d.columns:
            d["pct"] = d["mu_star"] / d["mu_star"].sum() * 100
        # Key fix: align all dataframes by the unique parameter name
        d = d.set_index("Parameter")
        SA_dfs_aligned.append(d)

    if reference_r not in r_values:
        raise ValueError(f"reference_r={reference_r} not found in r_values")

    ref_idx = r_values.index(reference_r)
    SA_ref = SA_dfs_aligned[ref_idx]

    # --------------------------------------------------
    # Select parameters (REFERENCE DEFINES ORDER)
    # --------------------------------------------------
    if k is None:
        top_params = SA_ref.index.tolist()
    else:
        top_params = SA_ref["pct"].nlargest(k).index.tolist()

    labels = top_params  # already parameter names

    # --------------------------------------------------
    # Stack results (REINDEX EACH DF TO REFERENCE ORDER)
    # --------------------------------------------------
    Y = np.column_stack([
        d.reindex(top_params)["pct"].to_numpy()
        for d in SA_dfs_aligned
    ])

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.grid(True, ls="--", lw=0.8, color="gray", alpha=0.6, zorder=0)

    for label, y in zip(labels, Y):
        ax.plot(
            r_values,
            y,
            marker="o",
            lw=1.5,
            label=label
        )

    ax.set_xticks(r_values, [f"r = {r}" for r in r_values])
    ax.set_xlabel("Number of Morris trajectories (r)", fontsize=16)
    ax.set_ylabel("Mu* sensitivity (%)", fontsize=16)

    if title is None:
        title = "Morris sensitivity convergence"
    ax.set_title(title, fontsize=18)

    ax.tick_params(axis="both", labelsize=14)
    ax.legend(fontsize=11)
    fig.tight_layout()

    if outpath is not None:
        fig.savefig(outpath, dpi=300, bbox_inches="tight")
