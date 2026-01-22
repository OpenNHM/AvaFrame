"""
    Run com1DFA kernel and compare it tothe similarity solution
    This script computes the similarity solution for a gliding avalanche on
    a inclined plane according to similarity solution from :
    Hutter, K., Siegel, M., Savage, S.B. et al.
    Two-dimensional spreading of a granular avalanche down an inclined plane
    Part I. theory. Acta Mechanica 100, 37–68 (1993).
    https://doi.org/10.1007/BF01176861
    and compares it to the DFA kernel com1DFA
"""

import pathlib
from configupdater import ConfigUpdater

# Local imports
import avaframe.in3Utils.initializeProject as initProj
import avaframe.in3Utils.fileHandlerUtils as fU
from avaframe.in3Utils import cfgUtils
from avaframe.in3Utils import logUtils
import avaframe.com1DFA.com1DFA as com1DFA
from avaframe.in1Data import getInput as gI
import avaframe.ana1Tests.simiSolTest as simiSolTest


# +++++++++SETUP CONFIGURATION++++++++++++++++++++++++
# log file name; leave empty to use default runLog.log
logName = 'runSimilarityTest'

# Load general configuration
cfgMain = cfgUtils.getGeneralConfig()
avalancheDir = 'data/avaSimilaritySol'
cfgMain['MAIN']['avalancheDir'] = avalancheDir

# Clean input directory(ies) of old work and output files
initProj.cleanSingleAvaDir(avalancheDir, deleteOutput=False)

# Start logging
log = logUtils.initiateLogger(avalancheDir, logName)
log.info('MAIN SCRIPT')
log.info('Current avalanche: %s', avalancheDir)

# Load configuration for similarity solution test
simiSolCfg = pathlib.Path(avalancheDir, 'Inputs', 'simiSol_com1DFACfg.ini')

# create output directory for test result plots
outDirTest = pathlib.Path(avalancheDir, 'Outputs', 'ana1Tests')
fU.makeADir(outDirTest)

for sphKernelRadius in [5, 3]:
    updater = ConfigUpdater()
    updater.read(simiSolCfg)
    updater['GENERAL']['sphKernelRadius'].value = sphKernelRadius
    updater['GENERAL']['meshCellSize'].value = sphKernelRadius
    updater.update_file()

    cfg = cfgUtils.getModuleConfig(com1DFA, fileOverride=simiSolCfg)
    # Define release thickness distribution
    demFile = gI.getDEMPath(avalancheDir)
    relDict = simiSolTest.getReleaseThickness(avalancheDir, cfg, demFile)
    # call com1DFA to perform simulations - provide configuration file and release thickness function
    # (may be multiple sims)
    _, _, _, simDF = com1DFA.com1DFAMain(cfgMain, cfgInfo=simiSolCfg)
    # Clean input directory(ies) of old work 
    initProj.cleanSingleAvaDir(avalancheDir, deleteOutput=False)
