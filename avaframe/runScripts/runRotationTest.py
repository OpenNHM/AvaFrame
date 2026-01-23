"""
Run the rotation test
Analyze the effect of the grid orientation/alignment on DFA simulation results
settings in ana1Tests/rotationTestCfg.ini in aimec to search for reference sim, rel0 is set (data/avaTripleBowl release area)
"""
import pathlib

# Local imports
# import config and init tools
from avaframe.in3Utils import logUtils
from avaframe.in3Utils import cfgUtils
from avaframe.in3Utils import cfgHandling
from avaframe.in3Utils import initializeProject as iP
from avaframe.in3Utils import fileHandlerUtils as fU
# import computation modules
from avaframe.com1DFA import com1DFA

# import analysis modules
from avaframe.runScripts import runAna3AIMEC
from avaframe.ana3AIMEC import ana3AIMEC, aimecTools
from avaframe.ana1Tests import rotationTest
from avaframe.ana1Tests import energyLineTest

# +++++++++REQUIRED+++++++++++++
# log file name; leave empty to use default runLog.log
logName = 'runRotationTest'
# ++++++++++++++++++++++++++++++

# Load avalanche directory from general configuration file
cfgMain = cfgUtils.getGeneralConfig()
avalancheDir = cfgMain['MAIN']['avalancheDir']
avalancheDir = pathlib.Path(avalancheDir)

# Start logging
log = logUtils.initiateLogger(avalancheDir, logName)
log.info('MAIN SCRIPT')
log.info('Current avalanche: %s', avalancheDir)

# ----------------
# Clean input directory(ies) of old work and output files
iP.cleanSingleAvaDir(avalancheDir, deleteOutput=False)
workPath = pathlib.Path(avalancheDir, 'Work', 'energyLineTest')
fU.makeADir(workPath)
rotationTestCfg = cfgUtils.getModuleConfig(rotationTest, avalancheDir)

# ++++++++++ set configurations for all the used modules and override ++++++++++++
# get comDFA configuration and save to file
com1DFACfg = cfgUtils.getModuleConfig(com1DFA, avalancheDir, fileOverride='', modInfo=False, toPrint=False,
                                      onlyDefault=rotationTestCfg['com1DFA_com1DFA_override'].getboolean('defaultConfig'))
com1DFACfg, rotationTestCfg = cfgHandling.applyCfgOverride(com1DFACfg, rotationTestCfg, com1DFA, addModValues=False)
com1DFACfgFile = cfgUtils.writeCfgFile(avalancheDir, com1DFA, com1DFACfg, fileName='com1DFA_settings',
                                       filePath=workPath)
# get energyLine configuration and save to file
energyLineTestCfg = cfgUtils.getModuleConfig(energyLineTest, avalancheDir, fileOverride='', modInfo=False, toPrint=False,
                                             onlyDefault=rotationTestCfg['ana1Tests_energyLineTest_override'].getboolean('defaultConfig'))
energyLineTestCfg, rotationTestCfg = cfgHandling.applyCfgOverride(energyLineTestCfg, rotationTestCfg, energyLineTest,
                                                                  addModValues=False)
# get ana3AIMEC configuration
AIMECCfg = cfgUtils.getModuleConfig(ana3AIMEC, avalancheDir, fileOverride='', modInfo=False, toPrint=False,
                                    onlyDefault=rotationTestCfg['ana3AIMEC_ana3AIMEC_override'].getboolean('defaultConfig'))
AIMECCfg, rotationTestCfg = cfgHandling.applyCfgOverride(AIMECCfg, rotationTestCfg, ana3AIMEC, addModValues=False)
AIMECCfgFile = cfgUtils.writeCfgFile(avalancheDir, ana3AIMEC, AIMECCfg, fileName='ana3AIMEC_settings',
                                     filePath=workPath)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# run the com1DFA module or load the results from com1DFA
runDFAModule = rotationTestCfg['rotationTest'].getboolean('runDFAModule')
dem, simDF, resTypeList = com1DFA.runOrLoadCom1DFA(avalancheDir, cfgMain, runDFAModule=runDFAModule,
                                                   cfgFile=com1DFACfgFile)

# fetch reference simulation based in referenceSimName
refSimRowHash, refSimName = aimecTools.defineRefOnSimName(AIMECCfg['AIMECSETUP']['referenceSimName'], simDF)
# initialize report
comModule = rotationTestCfg['rotationTest']['comModule']
flagMass = AIMECCfg['FLAGS'].getboolean('flagMass')
reportRotationTest = rotationTest.initializeRotationTestReport(avalancheDir, resTypeList, comModule, refSimName,
                                                               flagMass)
# Rotate raster results for aimec analysis and run the energy line test
simDF, flagMass = rotationTest.mainRotationTest(avalancheDir, energyLineTestCfg, com1DFACfg, dem, simDF, resTypeList,
                                                flagMass, refSimRowHash, comModule)

# proceed to aimec analysis on the rotated raster results
iP.cleanModuleFiles(avalancheDir, ana3AIMEC)

pathDict, rasterTransfo, resAnalysisDF, aimecPlotDict = runAna3AIMEC.runAna3AIMEC(avalancheDir, AIMECCfg)
resAnalysisDF = resAnalysisDF.reset_index().merge(simDF[['simName', 'relAngle']], on=['simName']).set_index('index')

# finalize rotation test report
reportRotationTest = rotationTest.buildRotationTestReport(avalancheDir, reportRotationTest, simDF, resAnalysisDF,
                                                          aimecPlotDict, flagMass)
