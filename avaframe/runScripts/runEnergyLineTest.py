"""
Run a the energy line test
Compare the DFA simulation result to the energy solution
"""
import pathlib

# Local imports
# import config and init tools
from avaframe.in3Utils import initializeProject as iP
from avaframe.in3Utils import cfgUtils
from avaframe.in3Utils import cfgHandling
from avaframe.in3Utils import logUtils
from avaframe.in3Utils import fileHandlerUtils as fU

# import computation modules
from avaframe.com1DFA import com1DFA

# import analysis modules
from avaframe.ana1Tests import energyLineTest


# +++++++++REQUIRED+++++++++++++
# log file name; leave empty to use default runLog.log
logName = 'runEnergyLineTest'
# ++++++++++++++++++++++++++++++

# Load avalanche directory from general configuration file
cfgMain = cfgUtils.getGeneralConfig()
avalancheDir = cfgMain['MAIN']['avalancheDir']

# Start logging
log = logUtils.initiateLogger(avalancheDir, logName)
log.info('MAIN SCRIPT')
log.info('Current avalanche: %s', avalancheDir)

# ----------------
# Clean input directory(ies) of old work and output files
iP.cleanSingleAvaDir(avalancheDir, deleteOutput=False)
workPath = pathlib.Path(avalancheDir, 'Work', 'energyLineTest')
fU.makeADir(workPath)
energyLineTestCfg = cfgUtils.getModuleConfig(energyLineTest, avalancheDir)

# ++++++++++ set configurations for all the used modules and override ++++++++++++
# get comDFA configuration and save to file
com1DFACfg = cfgUtils.getModuleConfig(com1DFA, avalancheDir, toPrint=False,
                                      onlyDefault=energyLineTestCfg['com1DFA_com1DFA_override'].getboolean('defaultConfig'))
com1DFACfg, energyLineTestCfg = cfgHandling.applyCfgOverride(com1DFACfg, energyLineTestCfg, com1DFA, addModValues=False)
com1DFACfgFile = cfgUtils.writeCfgFile(avalancheDir, com1DFA, com1DFACfg, fileName='com1DFA_settings',
                                       filePath=workPath)

# run the com1DFA module or load the results from com1DFA
runDFAModule = energyLineTestCfg['energyLineTest'].getboolean('runDFAModule')
dem, simDF, _ = com1DFA.runOrLoadCom1DFA(avalancheDir, cfgMain, runDFAModule=runDFAModule,
                                         cfgFile=com1DFACfgFile)

# generate mass averaged path profile
for simName in simDF.index:
    # make analysis and generate plots
    resultEnergyTest, savePath = energyLineTest.mainEnergyLineTest(avalancheDir, energyLineTestCfg, com1DFACfg, simName,
                                                                   dem)
