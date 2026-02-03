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

It is possible to modify these settings. The main options are:

* provide the path to your own configuration file using the ``fileOverride`` parameter when calling
  :py:func:`in3Utils.cfgUtils.getModuleConfig`

* create an expert configuration file at ``{avalancheDir}/Inputs/CFGs/{moduleName}Cfg.ini``
  (see :doc:`expertConfiguration` for details)

* create a copy of the module configuration file called ``local_`` followed by
  the name of the original configuration file and set the desired values of the
  individual parameters

* see :ref:`configuration:Override configuration` for additional options to modify configuration

So the order is as follows:

#. if ``batchCfgDir`` is provided, the path to the batch configuration directory is returned
   (this is used for batch processing and returns a ``pathlib.Path`` instead of a ConfigParser object).

#. if ``onlyDefault=True`` is passed to :py:func:`in3Utils.cfgUtils.getModuleConfig`, only
   the default configuration is used (all overrides are skipped).

#. if there is a path provided via the ``fileOverride`` parameter, configuration is read from this file.

#. if the ``avalancheDir`` is provided and ``{avalancheDir}/Inputs/CFGs/{moduleName}Cfg.ini`` exists,
   this expert config is used (see :doc:`expertConfiguration` for details).

#. if there is no expert config, the ``local_...`` configuration file is read if
   it exists.

#. if there is no ``local_...``, the ``getModuleConfig`` function reads the
   settings from the default configuration file with the default settings.

The following flowchart illustrates this priority order:

.. graphviz::

   digraph config_priority {
      rankdir=TB
      node [shape=box, style=rounded]

      start [label="getModuleConfig()", shape=ellipse]
      batch [label="batchCfgDir\nprovided?", shape=diamond]
      batch_ret [label="Return Path\n(batch mode)"]
      only [label="onlyDefault\n= True?", shape=diamond]
      only_ret [label="Return default\nconfig only"]
      file [label="fileOverride\nprovided?", shape=diamond]
      file_load [label="Load fileOverride"]
      expert [label="Expert config\nexists?", shape=diamond]
      expert_load [label="Load expert config\nInputs/CFGs/"]
      local [label="local_* file\nexists?", shape=diamond]
      local_load [label="Load local_* file"]
      default [label="Load default\nmodule config"]
      merge [label="Fill missing values\nfrom default", shape=box]
      done [label="Return ConfigParser", shape=ellipse]

      start -> batch
      batch -> batch_ret [label="yes"]
      batch -> only [label="no"]
      only -> only_ret [label="yes"]
      only -> file [label="no"]
      file -> file_load [label="yes"]
      file -> expert [label="no"]
      expert -> expert_load [label="yes"]
      expert -> local [label="no"]
      local -> local_load [label="yes"]
      local -> default [label="no"]

      file_load -> merge
      expert_load -> merge
      local_load -> merge
      default -> done
      only_ret -> done
      merge -> done
   }


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

  
