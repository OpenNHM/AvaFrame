"""
    Run script for plotting a matrix dataset with one profile
"""

# Load modules
import glob
import os

# Local imports
from avaframe.out3Plot import outQuickPlot
from avaframe.in3Utils import cfgUtils
from avaframe.in3Utils import logUtils

# log file name; leave empty to use default runLog.log
logName = 'runQuickPlotOne'

# Load avalanche directory from general configuration file
cfgMain = cfgUtils.getGeneralConfig()
avalancheDir = cfgMain['MAIN']['avalancheDir']

# load configuration for plot generation
cfg = cfgUtils.getModuleConfig(outQuickPlot, avalancheDir)
cfgPlot = cfg['ONEPLOT']

# Start logging
log = logUtils.initiateLogger(avalancheDir, logName)
log.info('MAIN SCRIPT')
log.info('Current avalanche: %s', avalancheDir)

# Set directory where input files is located
if cfgPlot['inputDir'] != '':
    inputDir = cfgPlot['inputDir']
else:
    inputDir = os.path.join(avalancheDir, 'Work', 'simplePlot')

# parameters to be set
location = float(cfgPlot['location'])
resType = cfgPlot['resType']
axis = cfgPlot['axis']

# Load input datasets from input directory
dataFiles = glob.glob(inputDir+os.sep + '*.asc')
dataFiles.extend(glob.glob(inputDir+os.sep + '*.txt'))
dataFiles.extend(glob.glob(inputDir+os.sep + '*.tif'))

for file in dataFiles:
    outQuickPlot.quickPlotOne(inputDir, file, cfgMain, location, axis, resType=resType)
