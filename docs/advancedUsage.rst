Advanced Usage (QGis, Script)
=============================

.. Warning::
   Modifying default parameters moves you outside the validated and calibrated setup.
   Results are no longer covered by the standard validation and you use them at your own risk.

AvaFrame allows overriding the default module configuration with custom settings via expert configuration files.
There are two ways to use expert configuration files:

1. **Via the QGis Connector** — extract a default ini, edit it, and pass it directly to a connector tool
2. **Via the CFGs directory** — place the ini file in the project's ``Inputs/CFGs/`` directory for automatic loading

Both approaches use the same ini file format. The difference is how the file is provided to AvaFrame.


1. Via the QGis Connector
-------------------------

Several connector tools accept an expert configuration file through the *Advanced Parameters* section
in the QGis processing dialog. The workflow is:

1. Use the *Get default module ini* tool (see :ref:`connector:Experimental`) to extract the default
   configuration file for the module you want to customize.

2. Edit the extracted ini file with your desired parameter values.

3. Open the connector tool you want to run (e.g. *Dense Flow (com1)*), expand *Advanced Parameters*,
   and select your edited ini file in the *Expert configuration file* field.

The following connector tools support this:

- Dense Flow (com1)
- Alpha Beta (com2)
- Snow slide (com5)
- Rock Avalanche (com6)
- Scarp (com6)
- MoTVoellmy (com9)


2. Via the CFGs directory
-------------------------

For script-based or repeated use, place expert configuration files directly in the project's input directory.
AvaFrame will automatically pick them up when the ``avalancheDir`` is provided.

Place your module configuration files in::

  {avalancheDir}/Inputs/CFGs/{moduleName}Cfg.ini

For example, to override com1DFA settings for a specific avalanche project::

  myAvalanche/Inputs/CFGs/com1DFACfg.ini

When this file exists and the ``avalancheDir`` is passed to :py:func:`in3Utils.cfgUtils.getModuleConfig`,
the expert config takes priority over any ``local_com1DFACfg.ini`` file. Values not specified in the
expert config are filled from the default module configuration, see :ref:`complexUsage:Configuration`.

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
