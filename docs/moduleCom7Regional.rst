######################################
com7Regional: Regional Modeling
######################################

.. note::
    This module is still under development and may contain bugs or incomplete features.

:py:mod:`com7Regional` is an experimental module targeted at the application of the :ref:`com1DFA <moduleCom1DFA:com1DFA: DFA-Kernel>`
kernel on a regional scale.
In essence, it is a wrapper module that allows for the concurrent execution of com1DFA within multiple avalanche directories,
with the aim of reducing the overall computational load through parallelization. Com1DFA is designed with one single avalanche track as 
the basis for computation. It includes parallel computation of different scenarios, but always on one, reasonably sized DEM. 
Com7Regional allows for the splitting of a setup with multiple avalanche tracks, for example one whole valley, into more managable
setups and then the subsequent results aggregation of scenarios or regions. 

The module therefor provides functions for:

* preparing and splitting input files from a master DEM and master release shapefile
* managing and aggregating outputs after a regional simulation, e.g. merge com1DFA peak files to rasters of maximum / minimum values.

The log output gives information about the processing status of each individual avalanche directory, as well as a summary
of how many simulations were processed successfully.

.. note::
    Experience with :ref:`com1DFA <moduleCom1DFA:com1DFA: DFA-Kernel>` is recommended before using this module.

Regional Input Management
=====================================
:py:mod:`com7Regional.splitInputs` is a module for organizing large amounts of avalanche input data into multiple avalanche directories, based
on the AvaFrame data structure, thus enabling efficient and controlled input management of larger datasets for further processing.
Importantly, the module also provides a simple, automatic method for clipping the input DEM around each release area group,
which then forms the basis for the organization of other input.

Input
-----
The module is currently compatible with the following input file types:

**Required:**
    * Digital elevation model (as .asc or .tif file). *Please note that the maximum possible size of this DEM depends on your
      compute hardware; You might run into problems if it is too big and maxes out your resources*.
    * Release areas: ONE shapefile in ``Inputs/REL`` directory, with additional attributes ``group`` and ``scenario``. Please
      also provide attributes required by com1DFA.

**Optional:**
    * ONE file with entrainment areas (shapefile in ``Inputs/ENT`` directory)
    * ONE file with resistance areas (shapefile in ``Inputs/RES`` directory)


Where the expected input directory structure is::

  avalancheDir/
  └── Inputs/
      ├── REL/
      │   └── *.shp         # release areas (one file)
      ├── ENT/              # entrainment areas (one file; optional)
      │   └── *.shp
      ├── RES/              # resistance areas (one file; optional)
      │   └── *.shp
      └── *.asc or *.tif    # digital elevation model (DEM)

Group and scenario creation
---------------------------
Input data organization is based on two key concepts:

* **Groups**: Collections of avalanche release areas (single polygon features) that are located in the same spatial domain and may be
   wanted to be simulated together. We recommend setting one group per avalanche path/track.

* **Scenarios**: Collections of release area features WITHIN each group, that are simulated together in :ref:`com1DFA <moduleCom1DFA:com1DFA: DFA-Kernel>`, 
  i.e. particles within one scenario can interact.

These are defined through two new attributes in the input release area shapefile attribute table:

* ``group``: expected format: string (e.g. *"group1"*)

* ``scenario``: expected format: comma-separated list of strings, without spaces (e.g. *"small,large"* or *"10y,30y,100y"*)

For example:

.. list-table::
    :header-rows: 1
    :widths: 40 40 40

    * - name
      - group
      - scenario
    * - rel1
      - avaPath1
      - 1,all
    * - rel2
      - avaPath1
      - 2,all
    * - rel3
      - avaPath2
      - 1

In this example, four release scenarios (shapefiles) would be created, in two separate sub-avalanche directories::

  avalancheDir/
  ├── ...
  └── com7Regional/
      ├── avaPath1/
      │   └── REL/
      │       ├── avaPath1_1.shp     - containing [1] feature: rel1
      │       ├── avaPath1_2.shp     - containing [1] feature: rel2
      │       └── avaPath1_all.shp   - containing [2] features: rel1, rel2
      └── avaPath2/
          └── REL/
              └── avaPath2_1.shp     - containing [1] feature: rel3

In the case that scenarios are defined for only some release features within a group, the rest will be grouped together as a single 'NULL' scenario.

In the case that no attributes or values for ``group`` or ``scenario`` are provided, the procedure will create groups based on each release feature 
- so one avalanche directory, including a DEM and release scenario shp file, per release feature (polygon). This may be 
intended to simulate each release feature separately in an automated manner. However, keep in mind, that this approach may result in 
a large amount of duplicated DEM data, in the case that release areas are located in close proximity to each other.

Output
------
Running ``runScripts/runSplitInputs.py`` with valid input data will result in the following output in ``<avalancheDir>/com7Regional``:

1. Individual avalanche directories for each group containing:

  - Clipped DEM file
  - Scenario-specific release area shapefiles
  - Optional: Clipped entrainment and resistance areas

2. Two visual reports (see :numref:`fig-splitInputs-basic` and :numref:`fig-splitInputs-optional`)

3. Scenario report in txt format (see example below)

.. list-table::
   :widths: 50 50

   * - .. _fig-splitInputs-basic:
       .. figure:: /_static/splitInputs_visualReport_basic.png
          :width: 100%
          :alt: Basic visual report

          Example of basic inputs report displaying resulting groups and their extent
     - .. _fig-splitInputs-optional:
       .. figure:: /_static/splitInputs_visualReport_optional.png
          :width: 100%
          :alt: Optional inputs report

          Example of optional inputs report displaying RES and ENT areas for each group

Example Scenario Report:

.. code-block:: text

    SCENARIO REPORT
    ==============
    Generated: 2025-02-04 10:58:56

    Group: group1
    ------------

    Scenario: sce1
    No. of release areas: 2
    - rel1
    - rel2

    No. of entrainment areas: 1
    No. of resistance areas: 3

    Group: group2
    ------------
    ...

Configuration
-------------
Settings are controlled through ``splitInputsCfg.ini``, in which the ``bufferSize`` for the group extent is defined (which is used for DEM, RES, and ENT clipping into
smaller chunks). By default, this value is set to 2500 m. For each group, a bounding box is created from the maximum x-y extent of all release features in the group.
The value for ``bufferSize`` is then added to each direction (+x, -x, +y, -y). This buffer may be adjusted according to the expected maximum runout length of your avalanches -
a larger value will ensure that no simulation will exit its domain, but with the drawback of producing larger output files and longer run times.

Algorithm
---------
The ``splitInputsMain`` function, which is called in ``runScripts/runSplitInputs.py``, performs the following steps:

1. Create central avalanche directory list
2. Set up avalanche directories
3. Split and write release areas to each directory
4. Clip and write DEM to each directory
5. Clip and write optional input to each directory (currently includes RES and ENT)
6. Divide release areas into scenarios
7. Write reports

To Run - Script based
---------------------
1. Prepare inputs in your ``<avalancheDir>/Inputs``
2. If you want to adjust settings, copy ``splitInputsCfg.ini`` to ``local_splitInputsCfg.ini`` and adjust values in there
3. Either set path to avalanche directory in ``avaframeCfg.ini`` (or local version ``local_avaframeCfg.ini``) or
   call command below with the avalanche directory as argument
4. Execute from ``AvaFrame/avaframe`` directory:

.. code-block:: bash

    pixi run python runScripts/runSplitInputs.py

---------------

Running multiple avalanche directories
======================================

Input
-----
A directory structure containing pre-configured avalanche directories (containing an ``Inputs`` folder) is required. For input preparation use
:ref:`moduleCom7Regional:Regional Input Management`, which splits merged input data into standard :ref:`com1DFA <moduleCom1DFA:com1DFA: DFA-Kernel>`
inputs across multiple avalanche directories.

Example of a valid directory structure (as produced by the regional input management above)::

    avalancheDir
    ├── Inputs/       #NOT being used for running; optional
    └── com7Regional/ #This is the default name, can be changed via .ini setup
        ├── sub_avalanche1/
        │   └── Inputs/
        │       ├── REL/*.shp
        │       └── *.asc or *.tif
        ├── sub_avalanche2/
        │   └── Inputs/
        │       ├── REL/*.shp
        │       └── *.asc or *.tif
        └── ...


Output
------
Outputs are organized in two levels:

**1. Merged rasters** and **2. Individual outputs (per sub_avalanche directory)**

Merged rasters
^^^^^^^^^^^^^^
Configure in ``com7RegionalCfg.ini`` (or local):

.. code-block:: ini

    [GENERAL]
    mergeOutput = True
    mergeTypes = pfv # Available options: [ppr|pfv|pft|pta|FT|FV|P|FM|Vx|Vy|Vz|TA]
    mergeMethod = max  # Available options: [max|min|sum|count]

Produces merged rasters of all peakFile results found within the sub-avalanche directories, for each
``mergeTypes`` and ``mergeMethod`` configured, in ``<avalancheDir>/com7Regional/mergedRasters/``. The merged
raster combines output from ALL sub-avalanches.

Creates::

    avalancheDir
    ├── ....
    └── com7Regional/
        ├── sub_avalanche1/
        ...
        └── mergedRasters/ <- this one is created

Individual outputs
^^^^^^^^^^^^^^^^^^
After running com7 with the given module (currently only :ref:`com1DFA <com1DFA>`), the standard output is located
within each of the sub-avalanche directories within e.g. ``<avalancheDir>/com7Regional/<avalancheDir>/Outputs/com1DFA``.
Additionally, com7Regional provides the option of aggregating all output peakFiles and tSteps results into a single directory
for easier management, either through copying or moving the files after an executed run.

Configure in ``com7RegionalCfg.ini`` (or local):

.. code-block:: ini

    [GENERAL]
    copyPeakFiles = True
    moveInsteadOfCopy = False

Creates::

    avalancheDir
    ├── ....
    └── com7Regional/
        ├── sub_avalanche1/
        ...
        └── allPeakFiles/ <- this one is created

Configuration
-------------
Three configuration files are used:

1. Main configuration (``avaframeCfg.ini``)
    - To set nCPUs for handling the amount of avalanche directories processed in parallel
    - Handle plot and other output generation
    
2. com7Regional configuration (``com7RegionalCfg.ini``)
    - Manages output aggregation and merged raster creation
    - Overrides com1DFA parameters if specified

3. com1DFA configuration (``com1DFACfg.ini``)
    - Standard simulation parameters

Processing
----------
Parallelization is handled through the concurrent.futures library, specifically the 
`ProcessPoolExecutor class <https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ProcessPoolExecutor>`_. 
In essence, tasks are executed concurrently within the input regional directory (by default com7Regional), based on the number of currently available CPUs.
So each sub-avalanche directory is one task.
The maximum number of CPUs is set by the ``nCPU`` parameter in ``avaframeCfg.ini``. By default, to avoid nested parallelization, each 
avalanche directory is assigned a single CPU , essentially meaning that any variations (e.g. through different 
scenarios, parameter variations, etc.) within each avalanche directory are handled sequentially. As a consequence, if the number of 
variations is high, and the number of avaDirs to process is lower than ``nCPU``, it may be more efficient to run simulations with the 
standard ``runCom1DFA.py`` instead, to utilize its parallel processing of variations. Alternatively, advanced users may want to adjust the nCPU for 
variations in ``com7Regional.py``.

To Run
------
1. Prepare input directories, we recommend using the regional input management above
2. If you want to adjust settings, copy ``com7RegionalCfg.ini`` to ``local_com7RegionalCfg.ini`` and adjust values in there
3. Set path to avalanche directory in ``avaframeCfg.ini`` (or local version ``local_avaframeCfg.ini``)
   or supply the directory as argument to the command below.
4. Execute from AvaFrame/avaframe directory:

.. code-block:: bash

    pixi run python runCom7Regional.py
