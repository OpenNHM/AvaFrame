Expert Configuration
------------------------------

For advanced users who want project-specific configuration that overrides both the default module
configuration and any ``local_`` configuration files, AvaFrame supports an expert configuration directory.

Place your module configuration files in::

  {avalancheDir}/Inputs/CFGs/{moduleName}Cfg.ini

For example, to override com1DFA settings for a specific avalanche project::

  myAvalanche/Inputs/CFGs/com1DFACfg.ini

When this file exists and the ``avalancheDir`` is passed to :py:func:`in3Utils.cfgUtils.getModuleConfig`,
the expert config takes priority over any ``local_com1DFACfg.ini`` file. Values not specified in the
expert config are filled from the default module configuration.

.. Note::
   The ``local_`` file behavior is only active when no expert config exists for the module.
   If an expert config exists, the ``local_`` file is completely ignored for that module.

Example
^^^^^^^

To use custom friction parameters for a specific project:

1. Create the CFGs directory in your avalanche project::

     mkdir -p myAvalanche/Inputs/CFGs

2. Create the module configuration file (e.g., ``com1DFACfg.ini``)::

     [GENERAL]
     frictModel = samosATMedium

3. Run your simulation - the expert config will automatically be used when ``avalancheDir`` is provided.

See also :ref:`configuration:Configuration` for the complete configuration system documentation.
