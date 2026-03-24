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

The :py:func:`in3Utils.cfgUtils.getModuleConfig` function reads the settings from a configuration file (``tmpExCfg.ini``
in our example) and writes these settings to the log file. The default settings can be found in the
configuration file provided within each module.

It is possible to modify these settings. The main options are:

* provide the path to your own configuration file using the ``fileOverride`` parameter when calling
  :py:func:`in3Utils.cfgUtils.getModuleConfig`

* create an expert configuration file at ``{avalancheDir}/Inputs/CFGs/{moduleName}Cfg.ini``
  (see :doc:`advancedUsage` for details)

* create a copy of the module configuration file called ``local_`` followed by
  the name of the original configuration file and set the desired values of the
  individual parameters. ``local_`` needs to be in the same folder as the original.

* see :ref:`configuration:Override configuration` for additional options to modify configuration

So the order is as follows:

#. if ``batchCfgDir`` is provided, the path to the batch configuration directory is returned
   (this is used for batch processing and returns a ``pathlib.Path`` instead of a ConfigParser object).

#. if ``onlyDefault=True`` is passed to :py:func:`in3Utils.cfgUtils.getModuleConfig`, only
   the default configuration is used (all overrides are skipped).

#. if there is a path provided via the ``fileOverride`` parameter, configuration is read from this file.

#. if the ``avalancheDir`` is provided and ``{avalancheDir}/Inputs/CFGs/{moduleName}Cfg.ini`` exists,
   this expert config is used (see :doc:`advancedUsage` for details).

#. if there is no expert config, the ``local_...`` configuration file is read if
   it exists.

#. if there is no ``local_...``, the ``getModuleConfig`` function reads the
   settings from the default configuration file with the default settings.

The following flowchart illustrates this priority order:

.. graphviz::

   digraph config_priority {
      rankdir=TB
      newrank=true
      graph [ranksep=0.6, nodesep=0.5, splines=ortho, fontname="helvetica", fontsize="20", fontcolor="#0E4160"]
      node [shape=box, style="rounded,filled", fontsize=16, fontcolor="#0E4160", fontname="helvetica", fillcolor="#51ADE5cf", penwidth=1.5]
      edge [fontname="helvetica", fontsize="12", penwidth=1.5]

      // Entry and exit points
      start [label="getModuleConfig()", shape=ellipse, fillcolor="#8BC9F0"]
      done [label="Return ConfigParser", shape=ellipse, fillcolor="#8BC9F0"]

      // Question nodes (left column)
      batch [label="batchCfgDir\nprovided?", shape=box, fillcolor="#5ce0e6", margin="0.3,0.2"]
      only [label="onlyDefault\n= True?", shape=box, fillcolor="#5ce0e6", margin="0.3,0.2"]
      file [label="fileOverride\nprovided?", shape=box, fillcolor="#5ce0e6", margin="0.3,0.2"]
      expert [label="Expert config\nexists?", shape=box, fillcolor="#5ce0e6", margin="0.3,0.2"]
      local [label="local_* file\nexists?", shape=box, fillcolor="#5ce0e6", margin="0.3,0.2"]

      // Action nodes (right column)
      batch_ret [label="Return Path\n(batch mode)", shape=ellipse]
      only_ret [label="Return default\nconfig only"]
      file_load [label="Load fileOverride"]
      expert_load [label="Load expert config\nInputs/CFGs/"]
      local_load [label="Load local_* file"]
      merge [label="Read default config\nFill missing values\nfrom default"]

      // Force horizontal alignment of question-answer pairs
      {rank=same; batch; batch_ret}
      {rank=same; only; only_ret}
      {rank=same; file; file_load}
      {rank=same; expert; expert_load}
      {rank=same; local; local_load}

      // Invisible chain to enforce vertical alignment of questions (left column)
      batch -> only -> file -> expert -> local [style=invis, weight=100]

      // Invisible chain to enforce vertical alignment of actions (right column)
      batch_ret -> only_ret -> file_load -> expert_load -> local_load [style=invis, weight=100]

      // Main flow: entry
      start -> batch

      // Main flow: "no" path (down the left column)
      batch -> only [xlabel="no"]
      only -> file [xlabel="no"]
      file -> expert [xlabel="no"]
      expert -> local [xlabel="no"]
      // Invisible node to route "no" edge from local down then right to merge
      local_merge_route [shape=point, width=0, height=0]
      local -> local_merge_route [dir=none, xlabel="no"]
      local_merge_route -> merge

      // Main flow: "yes" path (to the right column)
      batch -> batch_ret [xlabel="yes"]
      only -> only_ret [xlabel="yes"]
      file -> file_load [xlabel="yes"]
      expert -> expert_load [xlabel="yes"]
      local -> local_load [xlabel="yes"]

      // Convergence to merge and exit
      file_load:e -> merge:ne
      expert_load:se -> merge:ne
      local_load -> merge
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

When module **B** calls tools from module **A**, you can override **A**'s configuration directly in **B**'s config file.
This keeps all settings for a workflow in one place.

**How it works:**

1. Add a section named ``[collectionName_moduleName_override]`` to module **B**'s config file
2. List the parameters you want to override from module **A**
3. When **B** runs, it loads **A**'s default config and applies your overrides (happens in cfgHandling.applyCfgOverride)

**Example:** ``ana1Tests/energyLineTestCfg.ini`` overrides ``com1DFA`` settings:

.. code-block:: ini

   [energyLineTest]
   # energyLineTest's own settings
   runDFAModule = True
   pathFromPart = True

   [com1DFA_com1DFA_override]
   # these override com1DFA's defaults when called from energyLineTest
   defaultConfig = True
   simTypeList = null
   relTh = 1
   frictModel = Coulomb

The ``defaultConfig = True`` parameter ensures the override starts from **A**'s default configuration
(rather than a ``local_`` file if one exists).

  
