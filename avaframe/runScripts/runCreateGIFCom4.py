"""
Create a GIF out of com4FlowPy video data results
"""
import time

# Local imports

from avaframe.in3Utils import cfgUtils
from avaframe.in3Utils import logUtils
import avaframe.out3Plot.outCom4Gif as outCom4Gif

# Time the whole routine
startTime = time.time()

# log file name; leave empty to use default runLog.log
logName = 'runCreateGIFCom4'

# Load avalanche directory from general configuration file
cfgMain = cfgUtils.getGeneralConfig()
avalancheDir = cfgMain['MAIN']['avalancheDir']

# Start logging
log = logUtils.initiateLogger(avalancheDir, logName)
log.info('MAIN SCRIPT')
log.info('Current avalanche: %s', avalancheDir)

# TODO: add an option to run FlowPy before (and modify the GIF config to create a video for this simualtion)

# Load configuration for hybrid model
cfg = cfgUtils.getModuleConfig(outCom4Gif)

outCom4Gif.makeGenerationVideo(avalancheDir=avalancheDir, cfg=cfg)
