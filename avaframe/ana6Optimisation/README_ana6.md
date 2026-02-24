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
### Reference Data and Working Directory

All scripts must be executed within the directory: `avaframe/ana6Optimisation`

In `avaframeCfg.ini`, the avalanche reference directory (`avalancheDir`) must include the suffix `../`, for example: `../data/avaFleisskar`

This ensures correct relative path resolution within the AvaFrame project structure.

To compute goodness-of-fit metrics between reference and simulation results and to perform AIMEC analysis, the following reference data must be provided in: `avaframe/data/<avaName>/Inputs`

The required folder structure is:
Folder:
- **LINES**  
  Contains the AIMEC path.

- **POLYGONS**  
  Contains Cropshape and defines the maximal extent of runout area that is used for calculating areal indicators.

- **REFDATA**  
  Defines the runout area of the reference event.

- **REL**  
  Defines the release area of the avalanche event.
  
File:
- **Digital Elevation Model (DEM)**  
  Must be placed directly in the `Inputs` directory and must cover the entire affected area.
    
More Details here: https://docs.avaframe.org/en/latest/moduleCom1DFA.html

___

### Morris Sensitivity Analysis (MorrisSA)
     
The Morris sensitivity analysis provides a ranking of input parameters based on their influence on the model response.

Before running `runMorrisSA.py`, the following step is required prior:

- Execute `runAna4ProbAnaCom8MoTPSA`
- In `probAnaCfg.ini`:
  - Set the sampling method to `'morris'`
  - Define the number of Morris trajectories (`nSample`)
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
- Rename Output folders afterwards with the following naming convention: OutputsR<number>


where `<number>` corresponds to the number of trajectories

This process is computationally expensive, as it requires a large number of simulations.

**Execution:**

- Run `runPlotMorrisConvergence.py`

**Output:**

- Convergence plots of Morris sensitivity indices

---

### Optimisation

The optimisation process identifies the set of input parameters that yields the best agreement between simulation results and a defined reference. "Best" is defined by the objective function implemented in the optimisation routine.

Optimisation can be performed in two ways:

**With prior Morris analysis:**
 - Parameter ranking is available
 - Parameter bounds are already defined
 - Execute `runOpmisiation.py` with scenario 1 in `runOptimisationCfg.ini`

**Without prior Morris analysis:**
 - Execute `runAna4ProbAnaCom8MoTPSA.py` to generate some initial samples (for surrogate)
 - In `probAnaCfg.ini`:
   - Set the sampling method to `'latin'`
   - Define the number of model runs (`nSample`)
   - Select the input parameters and define their variation bounds
 - Execute `runOpmisiation.py` with scenario 2 in `runOptimisationCfg.ini`

**Two optimisation strategies are implemented:**

- Surrogate-based non-sequential optimisation
- Surrogate-based Bayesian (sequential) optimisation

**Outputs:**

- Optimal parameter set
- Visualisation plots of the optimisations results and progress

---

## Notes

- Performing Morris sensitivity analysis before optimisation is recommended to reduce the parameter space.
- Convergence analysis significantly increases computational cost.
- All workflows are controlled via `.ini` configuration files.