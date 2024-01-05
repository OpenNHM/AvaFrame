Configuration
------------------------------

In order to set the configurations required by all the modules within Avaframe, the python module
`configparser <https://docs.python.org/3/library/configparser.html>`_ is used.

This is done in two steps. The first step fetches the main settings::

  from avaframe.in3Utils import cfgUtils
  # Load avalanche directory from general configuration file
  cfgMain = cfgUtils.getGeneralConfig()
  avalancheDir = cfgMain['MAIN']['avalancheDir']

In the second step the specific settings to a given module are imported::

  from avaframe.tmp1Ex import tmp1Ex
  # Load all input Parameters from config file
  # get the configuration of an already imported module
  # Write config to log file
  cfg = cfgUtils.getModuleConfig(tmp1Ex)

The :py:func:`in3Utils.cfgUtils.getModuleConfig` function reads the settings from a configuration file (``tmpEx.ini``
in our example) and writes these settings to the log file. The default settings can be found in the
configuration file provided within each module.

It is possible to modify these settings, there are two options:

* provide the path to your own configuration file when calling
  ``cfgUtils.getModuleConfig(moduleName, path to config file)``

* create a copy of the module configuration file called ``local_`` followed by
  the name of the original configuration file and set the desired values of the
  individual parameters.

* see :ref:`configuration:Override configuration` for additional options to modify configuration

So the order is as follows:

#. if there is a path provided, configuration is read from this file.

#. if there is no path provided, the ``local_...`` configuration file is read if
   it exists.

#. if there is no ``local_...``, the ``getModuleConfig`` function reads the
   settings from the default configuration file with the default settings.


In the configuration file itself, there are multiple options to vary a parameter:

* replace the default parameter value with desired value
* provide a number of parameter values separated by ``|`` (e.g. ``relTh=1.|2.|3.``)
* provide a number of parameter values using ``start:stop:numberOfSteps`` (e.g. ``relTh=1.:3.:3``)) - a
  single value can be added by appending ``&4.0`` for example
  
Override configuration
^^^^^^^^^^^^^^
 
If tools of one module, let's call this module **A** for now, are called from another module **B**, there is the option to include an *collectionName_A_override* section in the
configuration file of module **B**. In this case, the default configuration of module **A** is read and the parameters in the *A_override* section 
in the module **B** configuration file are used to update the configuration settings of module **A**. This has the advantage of gathering all 
the configuration parameters used for one task in one configuration file.
An example of this usage can be found in ``ana1Tests/energyLineTestCfg.ini``.

Additionally, there is the option to write ``local_moduleNameCfg.ini`` configuration files for all modules
where a corresponding *collectionName_moduleName_override* section is provided by using
the function: :py:func:`in3Utils.cfgHandling.rewriteLocalCfgs`. These ``local_moduleNameCfg.ini`` files are
saved to ``avalancheDir/Inputs/configurationsOverrides`` by default if no ``localCfgPath`` is provided as input.
An example of this functionality is provided in :py:mod:`runScripts.runPlotContoursFromAsc.py` with the respective
``runPlotContoursFromAscCfg.ini`` file.

.. Note::
   By using :py:func:`in3Utils.cfgHandling.rewriteLocalCfgs`, already existing ``local_moduleNameCfg.ini`` files in the
   respective directory will be overwritten.
  
