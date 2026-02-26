# ana6 – Sensitivity Analysis & Optimisation

The `ana6Optimisation` module provides tools for performing Morris sensitivity analysis and parameter optimisation within the AvaFrame workflow. It supports input parameter ranking, convergence analysis of sensitivity indices and surrogate-based optimisation strategies. The module can be used either sequentially (Morris analysis followed by optimisation) or independently for direct optimisation.

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

In `avaframeCfg.ini`, the avalanche reference directory (`avalancheDir`) must include the suffix `../`, for example: `../data/avaFleisskar`

This ensures correct relative path resolution within the AvaFrame project structure.

---

### Loss Function and Configuration Settings

The optimisation compares avalanche simulations with a reference runout polygon. Model performance is evaluated using a weighted loss function combining:

- a modified Tversky score *(1 − Tversky)*
- the normalized RMSE of the runout length between simulation and reference

The Tversky score is computed from areal indicators (TP, FP, FN) within a predefined cropshape. The areal indicators are calculated using `runPlotAreaRefDiffs.py`. Settings in `runMorrisSACfg.ini` and `runOptimisationCfg.ini`.

The normalized runout length is computed using variables of the AIMEC analysis implemented in `ana3AIMEC.py`. Settings are defined in `ana3AIMECCfg.ini`.


In `ana3AIMECCfg.ini`, `probAnaCfg.ini`, `runMorrisSACfg.ini` (for Morris SA) and `runOptimisationCfg.ini` (for optimisation), the parameter runoutLayer or layer defines which avalanche layer is used for the analysis (e.g. L1 or L2). For now, the selected layer must be specified consistently in all configuration files. This is important because the entire evaluation workflow (including AIMEC analysis, optimisation, and Morris sensitivity analysis) is performed using this layer.


For the optimisation workflow, the following parameters must additionally be set in `ana3AIMECCfg.ini`:

- `resTypes = ppr`
- `anaMod = com8MoTPSA`
- `flagMass = False` (since `com8MoTPSA` currently does not produce a mass file)
- `includeReference = True`


---

### Reference Data
To compute these goodness-of-fit metrics and to perform the AIMEC analysis, the following data must be provided in: `avaframe/data/<avaName>/Inputs`.

The required folder structure is:
Folder:
- **LINES**  
  Contains the AIMEC path as `path_aimec.shp` file.

- **POLYGONS**  
  Contains the cropshape and defines the maximal extent of runout area that is used for calculating areal indicators. This shapefile must have the suffix `_cropshape.shp`.

- **REFDATA**  
  Defines the runout area of the reference event. This shapefile must have the suffix `_POLY.shp`.

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
    A prior convergence analysis suggests a minimum of 10 trajectories, while using ~20 trajectories provides more robust and stable results.
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

The convergence analysis evaluates how the Morris sensitivity indices stabilise with increasing numbers of trajectories. Its purpose is to determine the minimum number of trajectories that yields robust results.

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
The optimisation process identifies the set of input parameters that yields the best agreement between simulation results and a defined reference. “Best” is defined by the objective function implemented in the optimisation routine. The optimisation problem is formulated as a minimisation of the loss function, where lower values indicate better agreement between simulation results and the reference data.

Two surrogate-based optimisation strategies are implemented. In both approaches, a Gaussian Process (GP) surrogate model is used to approximate the loss function. The surrogate is trained using results from avalanche simulations and provides predictions of the loss together with an estimate of the prediction uncertainty.

#### Surrogate-based Non-sequential Optimisation

In the non-sequential approach, a trained surrogate predicts the loss for a large number of parameter combinations generated using Latin Hypercube Sampling (LHS). Parameter sets with the lowest predicted loss values are identified and analysed statistically. Avalanche simulations are then performed for the best predicted parameter sets as well as for the mean parameter values derived from the top-performing combinations.

#### Surrogate-based Bayesian (Sequential) Optimisation

In Bayesian optimisation, the GP surrogate model is updated iteratively. The procedure starts with a small initial set of evaluated avalanche simulations. Based on these results, the surrogate model is trained and used to guide the selection of new simulation points. The next parameter set is selected using the Expected Improvement (EI) acquisition function. EI balances two objectives:

- Exploitation – sampling regions where the surrogate predicts low loss values.
- Exploration – sampling regions with high predictive uncertainty.

After each new avalanche simulation, the GP surrogate model is updated and the process is repeated. The optimisation stops once a stopping criterion is reached (e.g. maximum number of iterations or very small EI values for several iterations).


---
### Optimisation Workflow


The optimisation can be performed using either non-sequential surrogate-based optimisation or sequential Bayesian optimisation. The optimisation strategy can be selected in `runOptimisationCfg.ini` via the parameter `optType`.

Independently of the chosen optimisation strategy, the workflow can be configured in two ways depending on whether Morris sensitivity analysis is used before:

**Scenario 1: Without prior Morris analysis (recommended):**
 - Execute `runAna4ProbAnaCom8MoTPSA.py` to generate some initial samples (for surrogate)
 - In `probAnaCfg.ini`:
   - Set the sampling method to `'latin'`
   - Define the number of model runs (`nSample`), which should scale with the number of input parameters d. A common rule of thumb is to use approximately 10·d samples ([Loeppky et al., 2009](https://doi.org/10.1198/TECH.2009.08040); [Jones et al., 1998](https://doi.org/10.1023/A:1008306431147)).
   - Select the input parameters and define their variation bounds
 - Execute `runOptimisation.py` with scenario 1 in `runOptimisationCfg.ini`

This is the standard scenario, as Latin Hypercube Sampling provides good coverage of the parameter space, which is important for training the surrogate model.

**Scenario 2: With prior Morris analysis:**
 - Parameter ranking is available
 - Parameter bounds are already defined
 - Execute `runOptimisation.py` with scenario 2 in `runOptimisationCfg.ini`

This option is mainly intended for experimental use. Morris samples are designed for sensitivity analysis and parameter ranking, but they do not provide an optimal coverage of the parameter space for surrogate-based optimisation.

---

**Outputs:**

- Optimal parameter set
- Visualisation plots of the optimisation results and progress

---

## Notes

- Performing Morris sensitivity analysis before optimisation is recommended to reduce the parameter space. However, using Morris samples directly for optimisation is not recommended, since they do not provide optimal coverage of the input parameter space.
- Convergence analysis significantly increases computational cost.
- All workflows are controlled via `.ini` configuration files.