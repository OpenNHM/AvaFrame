"""
    Compare the com1DFA kernel results to the similarity solution
    (to get com1DFA simulations first run runScripts/runSimilaritySol.py)
    This script computes the similarity solution for a gliding avalanche on
    a inclined plane according to similarity solution from :
    Hutter, K., Siegel, M., Savage, S.B. et al.
    Two-dimensional spreading of a granular avalanche down an inclined plane
    Part I. theory. Acta Mechanica 100, 37–68 (1993).
    https://doi.org/10.1007/BF01176861
    and compares it to the DFA kernel com1DFA
"""

import pathlib
import pandas as pd

# Local imports
import avaframe.in3Utils.initializeProject as initProj
import avaframe.in3Utils.fileHandlerUtils as fU
from avaframe.in3Utils import cfgUtils
from avaframe.in3Utils import logUtils
import avaframe.com1DFA.com1DFA as com1DFA
from avaframe.in1Data import getInput as gI
import avaframe.ana1Tests.simiSolTest as simiSolTest
import avaframe.out3Plot.outAna1Plots as outAna1Plots


# +++++++++SETUP CONFIGURATION++++++++++++++++++++++++
# log file name; leave empty to use default runLog.log
logName = 'runSimilarityTest'

# Load general configuration
cfgMain = cfgUtils.getGeneralConfig()
avalancheDir = 'data/avaSimilaritySol'

# Clean avalancheDir directory of old work and output files
initProj.cleanSingleAvaDir(avalancheDir, deleteOutput=False)

# Start logging
log = logUtils.initiateLogger(avalancheDir, logName)
log.info('MAIN SCRIPT')
log.info('Current avalanche: %s', avalancheDir)

# Load configuration for similarity solution test
simiSolCfg = pathlib.Path(avalancheDir, 'Inputs', 'simiSol_com1DFACfg.ini')
cfg = cfgUtils.getModuleConfig(com1DFA, fileOverride=simiSolCfg)

# create output directory for test result plots
outDirTest = pathlib.Path(avalancheDir, 'Outputs', 'ana1Tests')
fU.makeADir(outDirTest)

# Load configuration info of all com1DFA simulations
simDF, _ = cfgUtils.readAllConfigurationInfo(avalancheDir)

# compute the similartiy solution (this corresponds to our reference)
log.info('Computing similarity solution')
solSimi = simiSolTest.mainSimilaritySol(simiSolCfg)

# if the analysis already exists and you only want to replot uncomment this (and put the good result name)
# pathToResults = pathlib.Path(avalancheDir, 'Outputs', 'ana1Tests', 'results5.p')
# if pathToResults.is_file():
#     simDF = pd.read_pickle(pathToResults)
simDF = simiSolTest.postProcessSimiSol(avalancheDir, cfgMain, cfg, simDF, solSimi, outDirTest)


# make convergence plot (if you add the fiting lines, make sure only the coloredBy and sizedBy parameters are varied)
fig1, ax1 = outAna1Plots.plotErrorConvergence(simDF, outDirTest, cfg['SIMISOL'], 'nPart', 'vhErrorL2',
                          'aPPK', 'nPPK0', logScale=True, fit=False)

# make convergence plot
outAna1Plots.plotTimeCPULog(simDF, outDirTest, cfg['SIMISOL'], 'nPart', 'sphKernelRadius', 'nPPK0')

# do some filtering for the presentation plot
simDF = simDF[simDF['sphKernelRadius']==3]

# make convergence plot (if you add the fiting lines, make sure only the coloredBy and sizedBy parameters are varied)
# same as plotErrorConvergence but adds the points corresponding to different coloredBy values one after the others
# and saves itermediate plots
fig1, ax1 = outAna1Plots.plotPresentation(simDF, outDirTest, cfg['SIMISOL'], 'nPart', 'hErrorL2',
                          'aPPK', 'nPPK0', logScale=True, fit=True)

outAna1Plots.plotTimeCPULog(simDF, outDirTest, cfg['SIMISOL'], 'nPart', 'aPPK', 'nPPK0')
