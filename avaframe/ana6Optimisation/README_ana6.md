# ana6 – Sensitivity Analysis & Optimisation

The `ana6Optimisation` module provides tools for performing Morris sensitivity analysis and parameter optimisation
within the AvaFrame workflow.
It supports input parameter ranking, convergence analysis of sensitivity indices and surrogate-based optimisation
strategies.
The module can be used either sequentially (Morris analysis followed by optimisation) or independently for direct
optimisation.

---

## Module Structure

The module contains the following files:

- `runMorrisSA.py` (configuration: `runMorrisSACfg.ini`)
- `runPlotMorrisConvergence.py` (uses `runMorrisSACfg.ini`)
- `runOptimisation.py` (configuration: `runOptimisationCfg.ini`)
- `optimisationUtils.py`

---

## Workflow

### Working Directory

The above mentioned runScripts must be executed within the directory: `avaframe/ana6Optimisation`

In `avaframeCfg.ini`, the avalanche reference directory (`avalancheDir`) must include the suffix `../`, for example:
`../data/avaParabola`

This ensures correct relative path resolution within the AvaFrame project structure.

---

### Loss Function and Configuration Settings

The optimisation compares avalanche simulations with a reference runout polygon.
Model performance is evaluated using a weighted loss function combining:

- a modified Tversky score *(1 − Tversky)* using weights *alpha* and *beta* and multiplied by weight *weightTversky*
- the normalized RMSE of the runout line between simulation and reference multiplied by weight *weightRunout*

*optimisationVariable = weightRunout * runoutRMSENormalised + weightTversky * (1-tversky)*

The Tversky score is computed from areal indicators (TP, FP, FN) and weighting factors (alpha, beta) within a predefined
cropshape.
The areal indicators are calculated using `runPlotAreaRefDiffs.py`. Settings in `runMorrisSACfg.ini` and
`runOptimisationCfg.ini` in the section `GENERAL`.

The runout line difference is computed using functions of the AIMEC analysis implemented in `ana3AIMEC.py`.
Settings are defined in the `ana3AIMEC_ana3AIMEC_override` section in `runMorrisSACfg.ini` and/or
`runOptimisationCfg.ini`.
Here, a runout line is derived from the simulation result, based on a specified resType and thresholdValue, in the
thalweg-following coordinate system.
This runout line is then compared to the runout line derived from the runout area polygon representing the observation
from the actual event, also transformed into the thalweg-following coordinate system.
Finally, the RMSE of the normalized (divided with reference runout line value) runout line difference between simulation
and observation for all points across the thalweg where for both a runout line value has been found is computed.

In `probAnaCfg.ini`, `runMorrisSACfg.ini` (for Morris SA) and `runOptimisationCfg.ini` (for optimisation),
the parameter runoutLayer or layer defines which avalanche layer is used for the analysis (e.g. L1 or L2).
For now, the selected layer must be specified consistently in all configuration files.
This is important because the entire evaluation workflow (including AIMEC analysis, optimisation,
and Morris sensitivity analysis) is performed using this layer if it is a multi-layer model.

For the optimisation workflow, the following parameters must additionally be set in `ana3AIMEC_ana3AIMEC_override`
section:

- `resTypes = ppr`
- `runoutResType = ppr`
- `runoutLayer = L2` (since `com8MoTPSA` is a multi-layer model)
- `runoutCrossType = Max`
- `thresholdValue = 1`
- `anaMod = com8MoTPSA`
- `flagMass = False` (since `com8MoTPSA` currently does not produce a mass file)
- `includeReference = True`

### Reference Data

To compute these goodness-of-fit metrics and to perform the AIMEC analysis, the following data must be provided in:
`avaframe/data/<avaName>/Inputs`.

The required folder structure is:
Folder:

- **LINES**  
  Contains the AIMEC path as `path_aimec.shp` file.

- **POLYGONS**  
  Contains the cropshape and defines the maximal extent of runout area that is used for calculating areal indicators.
  It also defines the upstream end of computing the areal indicators. The shapefile must have the suffix `_cropshape.shp`.

- **REFDATA**  
  Defines the runout area of the reference event. This shapefile must contain a polygon feature and have the suffix `_POLY.shp`.

- **REL**  
  Defines the release area of the avalanche event.

File:

- **Digital Elevation Model (DEM)**  
  Must be placed directly in the `Inputs` directory and must cover the entire affected area.

More details here in the section `Inputs`: https://docs.avaframe.org/en/latest/moduleCom1DFA.html

___

### Morris Sensitivity Analysis (MorrisSA)

The Morris sensitivity analysis provides a ranking of input parameters based on their influence on the model response.

Before running `runMorrisSA.py`, the following step is required prior:

- Execute `runAna4ProbAnaCom8MoTPSA.py`
- In `probAnaCfg.ini`:
    - Set the sampling method to `'morris'`
    - Define the number of Morris trajectories (`nSample`).
      A prior convergence analysis suggests a minimum of 10 trajectories, while using ~20 trajectories provides more
      robust and stable results.
    - Select the input parameters and define their variation bounds

This step generates the required simulations and stores the sampled parameters and their bounds in a pickle file.

**Afterwards:**

- Run `runMorrisSA.py`
- Configure settings via `runMorrisSACfg.ini`
- The `MORRIS_CONVERGENCE` setting can be ignored for standard sensitivity analysis

**Outputs:**

- Pickle file containing:
    - Ranked input parameters
    - Morris sensitivity indices
    - Parameter bounds
- Visualisation plots of the sensitivity results

---

### Morris Convergence Analysis

The convergence analysis evaluates how the Morris sensitivity indices stabilise with increasing numbers of trajectories.
Its purpose is to determine the minimum number of trajectories that yields robust results.

**Requirements:**

- Run `runAna4ProbAnaCom8MoTPSA` multiple times with different numbers of Morris trajectories
- Rename Output folders afterwards with the following naming convention: OutputsR`<number>`

where `<number>` corresponds to the number of trajectories

This process is computationally expensive, as it requires a large number of simulations.

**Execution:**

- Run `runPlotMorrisConvergence.py`

**Output:**

- Convergence plots of Morris sensitivity indices

---

### Optimisation Strategies

The optimisation process identifies the set of input parameters that yields the best agreement between simulation
results
and a defined reference. “Best” is defined by the objective function implemented in the optimisation routine.
The optimisation problem is formulated as a minimisation of the loss function, where lower values indicate better
agreement between simulation results and the reference data.

Two surrogate-based optimisation strategies are implemented.
In both approaches, a Gaussian Process (GP) surrogate model is used to approximate the loss function.
The surrogate is trained using results from avalanche simulations and provides predictions of the loss
together with an estimate of the prediction uncertainty.

#### Gaussian Process Surrogate Settings

The surrogate model is built with the `GaussianProcessRegressor` from `scikit-learn`. Before training, all input
parameters are standardized with `StandardScaler`. Therefore, the GP works with standardized input values instead of
the original physical units of the avalanche model parameters. A length-scale of 1 corresponds approximately to one
standard deviation of the respective input parameter.

The GP kernel combines a `ConstantKernel`, a `Matern kernel` and a `WhiteKernel`. The `ConstantKernel` controls how
strongly the predicted loss values can vary. The Matern kernel defines how different two parameter sets are
in the standardized input space and how strongly their predicted loss values are expected to differ. It also controls
the smoothness of the predicted loss function. The `WhiteKernel` adds a learnable noise term. This can help the GP
handle small numerical irregularities or non-smooth behaviour in the computed loss values.

The initial kernel values were chosen as neutral starting values. For example, the initial Matern length-scale is set
to 1, which corresponds to one standard deviation in standardized input space. The bounds define the range in which
`scikit-learn` is allowed to adjust the parameters during training. Broad bounds were used to allow the GP to adapt
to different loss function behaviours, while still avoiding extremely small or large values that could lead to unstable
or unrealistic fits.

During training, `scikit-learn` automatically optimizes the `ConstantKernel` value, the `Matern length-scales` and the
`WhiteKernel` noise level within the defined bounds. Therefore, the exact start values of these parameters are less
critical than the chosen bounds and the use of multiple optimizer restarts. The Matern smoothness parameter `nu`,
`alpha=1e-8`, `normalize_y=True` and `n_restarts_optimizer=10` are fixed settings. The `alpha` value is only a very
small numerical stabilization term. It is different from the learnable `WhiteKernel` noise level. The setting
`n_restarts_optimizer=10` repeats the internal hyperparameter optimization from several starting points and was chosen
as a practical compromise between a more robust fit and additional computation time.

#### Surrogate-based Non-sequential Optimisation

In the non-sequential approach, a trained surrogate predicts the loss for a large number of parameter combinations
generated using Latin Hypercube Sampling (LHS).
Parameter sets with the lowest predicted loss values are identified and analysed statistically.
Avalanche simulations are then performed for the best predicted parameter sets as well as for the mean parameter
values derived from the top-performing combinations.

#### Surrogate-based Bayesian (Sequential) Optimisation

In Bayesian optimisation, the GP surrogate model is updated iteratively.
The procedure starts with a small initial set of evaluated avalanche simulations.
Based on these results, the surrogate model is trained and used to guide the selection of new simulation points.
The next parameter set is selected using the Expected Improvement (EI) acquisition function. EI balances two objectives:

- Exploitation – sampling regions where the surrogate predicts low loss values.
- Exploration – sampling regions with high predictive uncertainty.

After each new avalanche simulation, the GP surrogate model is updated and the process is repeated.
The optimisation stops once a stopping criterion is reached (e.g. maximum number of iterations or very small EI values
for several iterations).


---

### Optimisation Workflow

The optimisation can be performed using either non-sequential surrogate-based optimisation or sequential Bayesian
optimisation.
The optimisation strategy can be selected in `runOptimisationCfg.ini` via the parameter `optType`.

Independently of the chosen optimisation strategy, the workflow can be configured in two ways depending on whether
Morris sensitivity analysis is used before:

**Scenario 1: Without using simulation results of prior Morris analysis (recommended):**

- Execute `runAna4ProbAnaCom8MoTPSA.py` to generate some initial samples (for surrogate)
- In `probAnaCfg.ini`:
    - Set the sampling method to `latin`
    - Define the number of model runs (`nSample`), which should scale with the number of input parameters d. A common
      rule of thumb is to use approximately 10·d or more
      samples ([Loeppky et al., 2009](https://doi.org/10.1198/TECH.2009.08040); [Jones et al., 1998](https://doi.org/10.1023/A:1008306431147)).
        - Select the input parameters and define their variation bounds (if prior Morris analysis, can inform which
          parameters should be included)
- Execute `runOptimisation.py` with scenario 1 in `runOptimisationCfg.ini`

This is the standard scenario, as Latin Hypercube Sampling provides good coverage of the parameter space,
which is important for training the surrogate model. Prior Morris analysis can inform which parameters are included in
the
optimization process.

**Scenario 2: With using simulation results of prior Morris analysis:**

- Parameter ranking is available
- Parameter bounds are already defined
- Execute `runOptimisation.py` with scenario 2 in `runOptimisationCfg.ini`

This option is mainly intended for experimental use.
Morris samples are designed for sensitivity analysis and parameter ranking, but they do not provide an optimal coverage
of the parameter space for surrogate-based optimisation.

---

**Outputs:**

- Optimal parameter set
- Visualisation plots of the optimisation results and progress

---

## Notes

- Performing Morris sensitivity analysis before optimisation is recommended to reduce the parameter space, i.e. reduce 
  the number of parameters considered in the optimisation based on ranking of 'most influential' parameters. 
  However, using Morris samples directly for optimisation is not recommended, since they do not provide optimal coverage
  of the input parameter space.
- Convergence analysis significantly increases computational cost.
- All workflows are controlled via `.ini` configuration files.