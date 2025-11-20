
import argparse
import pathlib
import time
from avaframe.in3Utils import cfgUtils
from avaframe.in3Utils import logUtils
import avaframe.in3Utils.initializeProject as initProj
from com6RockAvalanche import variableVoellmyShapeToRaster
from com6RockAvalanche.variableVoellmyShapeToRaster import generateMuXsiRasters

def runMuXsiWorkflow(avadir=''):
    """
    Run the workflow to generate \u03bc and \u03be rasters.

    Parameters
    ----------
    avadir : str
        Path to the avalanche directory containing input and output folders.

    Returns
    -------
    None
    """
    startTime = time.time()
    logName = 'runMuXsi'

    # Load general configuration file
    cfgMain = cfgUtils.getGeneralConfig()
    if avadir:
        cfgMain['MAIN']['avalancheDir'] = avadir
    else:
        avadir = cfgMain['MAIN']['avalancheDir']

    avadir = pathlib.Path(avadir)

    # Start logging
    log = logUtils.initiateLogger(avadir, logName)
    log.info('MAIN SCRIPT')
    log.info('Using avalanche directory: %s', avadir)

    # Clean input directory(ies) of old work files
    initProj.cleanSingleAvaDir(avadir, deleteOutput=False)

    # Load module-specific configuration for Variable Voellmy
    variableVoellmyCfg = cfgUtils.getModuleConfig(variableVoellmyShapeToRaster)

    # Run the raster generation process
    generateMuXsiRasters(avadir, variableVoellmyCfg)

    endTime = time.time()
    log.info("Took %6.1f seconds to calculate.", (endTime - startTime))
    log.info('Workflow completed successfully.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run \u03bc and \u03be raster generation workflow')
    parser.add_argument('avadir', metavar='a', type=str, nargs='?', default='',
                        help='Path to the avalanche directory')

    args = parser.parse_args()
    runMuXsiWorkflow(str(args.avadir))
