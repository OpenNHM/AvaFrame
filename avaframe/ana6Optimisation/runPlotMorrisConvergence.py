"""
Run script for the Morris convergence plot. For usage read README_ana6.md.
"""
import pathlib

import avaframe.out3Plot.outAna6Plots as saveResults
from avaframe.in3Utils import cfgUtils
import optimisationUtils

# Load avalanche directory from general configuration file
cfgMain = cfgUtils.getGeneralConfig()
avalancheDir = cfgMain['MAIN']['avalancheDir']
avaName = pathlib.Path(avalancheDir).name

# Load morris config file
cfgDir = 'runMorrisSA.ini'
cfgMorrisSA = cfgUtils.getModuleConfig(pathlib.Path(cfgDir), toPrint=False)

# Load Morris sensitivity analysis results for convergence plot
SA_dfs, r_vals, outputs, outDir, reference_r = optimisationUtils.loadMorrisConvergenceData(cfgMorrisSA, avalancheDir,
                                                                                           avaName)

# Top 7 parameters
saveResults.plotMorrisConvergence(SA_dfs, r_vals, reference_r, k=7,
                                  outpath=outDir / f'{avaName}_MorrisSAConvergencePlotTop7.png',
                                  title=f"{avaName} Convergence plot for top 7 parameters")
# All parameters
saveResults.plotMorrisConvergence(SA_dfs, r_vals, reference_r, k=None,
                                  outpath=outDir / f'{avaName}_MorrisSAConvergencePlotAll.png',
                                  title=f"{avaName} Convergence plot for all parameters")
