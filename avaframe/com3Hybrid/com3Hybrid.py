"""
Hybrid model module

This module contains the core function of the hybrid model.
This hybrid model combines the DFA simulation and the alpha beta model
"""
import pathlib
import logging
import numpy as np
import copy

# Local imports
# import config and init tools
import avaframe.in2Trans.shpConversion as shpConv
from avaframe.in3Utils import cfgUtils
from avaframe.in3Utils import cfgHandling
from avaframe.in3Utils import fileHandlerUtils as fU
from avaframe.in1Data import getInput
import avaframe.in3Utils.initializeProject as initProj

# import computation modules
from avaframe.com1DFA import com1DFA, particleTools
import avaframe.ana5Utils.DFAPathGeneration as DFAPath
from avaframe.com2AB import com2AB
from avaframe.com3Hybrid import com3Hybrid

# import analysis tools
from avaframe.out3Plot import outAB

# import plotting tools
from avaframe.out3Plot import outCom3Plots

# create local logger
log = logging.getLogger(__name__)


def maincom3Hybrid(cfgMain, cfgHybrid):
    """This is the core function of the com3Hybrid module

    Here the DFA and AB models are run one after the other to first produce (from com1DFA)
    the avalanche path, then get the friction angle corresponding to the topography (from com2AB)
    and finally run the DFA simultaion one last time with the correct friction angle and get a 3D
    output of the avalanche
    """
    avalancheDir = cfgMain['MAIN']['avalancheDir']
    demOri = getInput.readDEM(avalancheDir)

    # setup outputs folder and work folder
    workPath = pathlib.Path(avalancheDir, 'Work', 'com3Hybrid')
    oPath = pathlib.Path(avalancheDir, 'Outputs', 'com3Hybrid')
    fU.makeADir(workPath)
    fU.makeADir(oPath)

    # ++++++++++ set configurations for all the used modules and override ++++++++++++
    # get comDFA configuration and save to file
    com1DFACfg = cfgUtils.getModuleConfig(com1DFA, avalancheDir, toPrint=False,
        onlyDefault=cfgHybrid['com1DFA_com1DFA_override'].getboolean('defaultConfig'))
    com1DFACfg, cfgHybrid = cfgHandling.applyCfgOverride(com1DFACfg, cfgHybrid, com1DFA, addModValues=False)

    # fetch configuration for DFAPathGeneration
    DFAPathGenerationCfg = cfgUtils.getModuleConfig(DFAPath, avalancheDir, toPrint=False,
        onlyDefault=cfgHybrid['ana5Utils_DFAPathGeneration_override'].getboolean('defaultConfig'))
    DFAPathGenerationCfg, cfgHybrid = cfgHandling.applyCfgOverride(DFAPathGenerationCfg, cfgHybrid, DFAPath,
                                                                   addModValues=False)

    # first create configuration object for com2AB
    com2ABCfg = cfgUtils.getModuleConfig(com2AB, avalancheDir, toPrint=False,
        onlyDefault=cfgHybrid['com1DFA_com1DFA_override'].getboolean('defaultConfig'))
    com2ABCfg, cfgHybrid = cfgHandling.applyCfgOverride(com2ABCfg, cfgHybrid, com2AB, addModValues=False)

    # get initial mu value
    muArray = np.array([cfgHybrid.getfloat('com1DFA_com1DFA_override', 'mucoulomb')])
    alphaArray = np.array([np.degrees(np.arctan(muArray[0]))])

    # prepare for iterating
    nIterMax = cfgHybrid.getint('GENERAL', 'nIterMax')
    iteration = 0
    iterate = True
    resultsHybrid = {}
    while iteration < nIterMax and iterate:
        # create a copy to avoid com1DFAMain modifying the original config
        com1DFACfgCopy = copy.deepcopy(com1DFACfg)
        com1DFACfgCopy['GENERAL']['mucoulomb'] = '%.4f' % muArray[-1]

        log.info('Mu is set to: %f' % muArray[-1])
        # ++++++++++ RUN COM1DFA +++++++++++
        # Run dense flow with coulomb friction
        # Clean input directory of old work and output files from module
        initProj.cleanModuleFiles(avalancheDir, com1DFA, deleteOutput=False)
        dem, _, _, simDF = com1DFA.com1DFAMain(cfgMain, cfgInfo=com1DFACfgCopy)
        simID = simDF.index[0]
        particlesList, timeStepInfo = particleTools.readPartFromPickle(avalancheDir, simName=simID, flagAvaDir=True,
                                                                       comModule='com1DFA')

        # ++++++++++ GENERATE PATH +++++++++++
        # postprocess to extract path and energy line
        avaProfileMass = DFAPath.getMassAvgPathFromPart(particlesList, addVelocityInfo=True)
        # make a copy because extendDFAPathKernel might modify avaProfileMass
        avaProfileMassExt = DFAPath.extendDFAPath(DFAPathGenerationCfg['PATH'], avaProfileMass.copy(), dem,
                                                  particlesList[0])
        avaProfileMassExtOri = copy.deepcopy(avaProfileMassExt)
        avaProfileMassExtOri['x'] = avaProfileMassExtOri['x'] + demOri['header']['xllcenter']
        avaProfileMassExtOri['y'] = avaProfileMassExtOri['y'] + demOri['header']['yllcenter']
        # also save it to work
        pathAB = pathlib.Path(avalancheDir, 'Outputs', 'com3Hybrid', 'pathAB_aimec' + str(iteration))
        name = 'massAvaPath'
        shpConv.writeLine2SHPfile(avaProfileMassExtOri, name, pathAB)

        # ++++++++++ RUN COM2AB +++++++++++
        com2ABCfg['ABSETUP']['path2Line'] = str(pathAB) + '.shp'
        # take the path extracted from the DFA model as input
        pathDict, demAB, splitPoint, eqParams, resAB = com2AB.com2ABMain(com2ABCfg, avalancheDir)
        # make AB plot
        reportDictList = []
        _, plotFile, writeFile = outAB.writeABpostOut(pathDict, demAB, splitPoint, eqParams, resAB, com2ABCfg,
                                                      reportDictList)

        # make custom com3 profile plot
        alpha = resAB[name]['alpha']
        indAlpha = resAB[name]['indAlpha']
        indBetaPoint = resAB[name]['indBetaPoint']
        sBetaPoint = resAB[name]['s'][indBetaPoint]
        xAB = resAB[name]['x'][indAlpha]
        yAB = resAB[name]['y'][indAlpha]
        resultsHybrid['iterration' + str(iteration)] = {'path': avaProfileMassExt, 'alpha': alpha,
                                                        'xAB': xAB, 'yAB': yAB, 'sBetaPoint': sBetaPoint}

        # update alpha (result from AB model on new profile)
        alpha = resAB[name]['alpha']
        log.info('Alpha is %.2f' % alpha)
        # append new alpha and corresponding mu value to the list
        alphaArray = np.append(alphaArray, alpha)
        muArray = np.append(muArray, np.tan(np.radians(alpha)))
        # keep iterating if the change in alpha is too big
        iterate = keepIterating(cfgHybrid, alphaArray)
        iteration = iteration + 1

    # write final com1DFA configuration to file for documentation
    cfgUtils.writeCfgFile(avalancheDir, com1DFA, com1DFACfgCopy, fileName='com1DFA_settings',
                          filePath=workPath)

    # fetch fields for desired time step
    fields, fieldHeader, timeList = com1DFA.readFields(avalancheDir, ['pta'], simName=simID,
                                                       flagAvaDir=True, comModule='com1DFA', timeStep=timeStepInfo[-1])
    outCom3Plots.hybridProfilePlot(avalancheDir, resultsHybrid)
    outCom3Plots.hybridPathPlot(avalancheDir, dem, resultsHybrid, fields[0], particlesList[-1], muArray)

    # write com3HYbrid configuration settings to file for reproduction use
    com3HybridCfgFile = cfgUtils.writeCfgFile(avalancheDir, com3Hybrid, cfgHybrid, fileName='com3Hybrid_settings',
                                              filePath=oPath)

    log.debug('Alpha array is %s' % alphaArray)
    log.debug('mu array is %s' % muArray)


def keepIterating(cfgHybrid, alphaArray):
    """Check the change in alpha between two iterations

    If this change is bigger than the alphaThreshold, keep iterationg and return True
    """
    alphaThreshold = cfgHybrid.getfloat('GENERAL', 'alphaThreshold')
    iterate = np.abs(alphaArray[-1] - alphaArray[-2]) >= alphaThreshold
    return iterate
