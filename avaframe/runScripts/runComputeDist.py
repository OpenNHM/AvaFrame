""" Run script for retrieving a sample from a specified CDF """


# Load modules
import os
import numpy as np

# Local imports
from avaframe.in1Data import computeFromDistribution as cF
from avaframe.out3Plot import in1DataPlots as iPlot
from avaframe.in3Utils import fileHandlerUtils as fU
from avaframe.in3Utils import cfgUtils
from avaframe.in3Utils import logUtils


# log file name; leave empty to use default runLog.log
logName = 'runSampleFromDist'

# Load avalanche directory from general configuration file
cfgAva = cfgUtils.getGeneralConfig()
avalancheDir = cfgAva['MAIN']['avalancheDir']

# set working directory
workingDir = os.path.join(avalancheDir, 'Outputs', 'in1Data')
fU.makeADir(workingDir)

# Load input parameters from configuration file
cfgMain = cfgUtils.getGeneralConfig()
cfg = cfgUtils.getModuleConfig(cF, avalancheDir)
cfgGen = cfg['GENERAL']

# log file name; leave empty to use default runLog.log
logName = 'runSampleFromDist%s' % cfgGen['distType']

# Start logging
log = logUtils.initiateLogger(workingDir, logName)
log.info('MAIN SCRIPT')

# load parameters required to compute specific distribution
a = float(cfgGen['a'])
b = float(cfgGen['b'])
c = float(cfgGen['c'])
steps = int(cfgGen['support'])

# compute the support of the distribution
x = np.linspace(a, c, steps)

# +++++++++++++++Compute desired distribution ++++++++++++
# Derive sample from Pert distribution
# compute parameters
alpha, beta, mu = cF.computeParameters(a, b, c)

# compute PDF and CDF
PDF, CDF, CDFInt = cF.computePert(a, b, c, x, alpha, beta)

# +++++++++++++ Extract samples from distribution ++++++++++
if cfgGen['distType'] == 'Pert':
    # extract sample from pert distribution
    sampleVect = cF.extractFromCDF(CDF, CDFInt, x, cfgGen)
elif cfgGen['distType'] == 'Uni':
    # extract sample from uniform distribution
    CDF, CDFInt, sampleVect = cF.extractUniform(a, c, x, cfgGen)
else:
    log.error('Distribution type not known')

# ++++++++++++++ Print result to log and save results in outFile
for m in range(len(sampleVect)):
    log.info('%.5f' % (sampleVect[m]))
outFile = os.path.join(workingDir, 'sampleFromDist%s.txt' % cfgGen['distType'])
with open(outFile, 'w') as pfile:
    for item in sampleVect:
        pfile.write('%.5f_' % item)
    pfile.close()

# ++++++++++++++++ Make plots of sample distribution
flagShow = cfgMain['FLAGS'].getboolean('showPlot')
# # Plot the results
iPlot.plotDist(workingDir, CDF, a, b, c, cfgGen, flagShow)
iPlot.plotSample(workingDir, sampleVect, cfgGen, flagShow)

# +++++++++++++++ Check if retrieved sample actually fits the desired distriution .....
# retrieve approximation of sample PDF using numpy histogram and cumsum
CDFEmp, xSample = cF.getEmpiricalCDF(sampleVect)
iPlot.plotEmpCDF(workingDir, CDF, CDFEmp, xSample, cfgGen, 'Hist', flagShow)
# retrieve approximation of sample using the ECDF function
ECDF, xSample = cF.getEmpiricalCDFNEW(sampleVect)
iPlot.plotEmpCDF(workingDir, CDF, ECDF, xSample, cfgGen, 'ECDF', flagShow)
# plot sample histogram and PDF of desired distribution
iPlot.plotEmpPDF(workingDir, PDF, sampleVect, cfgGen, flagShow)

# write configuration to file
cfgUtils.writeCfgFile(avalancheDir, cF, cfg, fileName=cfgGen['distType'])
