"""
    Snow slide implemented in com1DFA
"""

# Load modules
import logging


# Local imports
from avaframe.in3Utils import cfgUtils
from avaframe.in3Utils import cfgHandling
from avaframe.com1DFA import com1DFA

# create local logger
# change log level in calling module to DEBUG to see log messages
log = logging.getLogger(__name__)


def com5SnowSlideMain(cfgMain, snowSlideCfg):
    """Run snow slide tool and adjust parameters to match snow slide settings for com1DFA run,
    result files, reports and plots are saved analog to a standard com1DFA model run

    Parameters
    -----------
    cfgMain: configparser object
        main AvaFrame settings
    snowSlideCfg: configparser object
        configuration settings for snow slide including com1DFA override parameters

    """

    avalancheDir = cfgMain["MAIN"]["avalancheDir"]

    # ++++++++++ set configurations for com1DFA and override ++++++++++++
    # get comDFA configuration and update with snow slide parameter set
    com1DFACfg = cfgUtils.getModuleConfig(
        com1DFA,
        avalancheDir,
        fileOverride="",
        modInfo=False,
        toPrint=False,
        onlyDefault=snowSlideCfg["com1DFA_com1DFA_override"].getboolean("defaultConfig"),
    )
    com1DFACfg, snowSlideCfg = cfgHandling.applyCfgOverride(
        com1DFACfg, snowSlideCfg, com1DFA, addModValues=False
    )

    # run the com1DFA module with snowSlide settings
    dem, plotDict, reportDictList, simDF = com1DFA.com1DFAMain(cfgMain, cfgInfo=com1DFACfg)

    # print info about simulation performed to log
    log.info("Com1DFA run performed with snowSlideTool settings")

    return dem, plotDict, reportDictList, simDF
