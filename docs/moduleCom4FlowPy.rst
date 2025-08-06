com4FlowPy: Flow-Py
======================

.. Note::
  :py:mod:`com4FlowPy` is now the actively maintained version of Flow-Py, which has been previously hosted on https://github.com/avaframe/FlowPy.
  **The old repository will be archived and is not actively maintained any longer.** We encourage existing users/developers 
  of Flow-Py to switch to :py:mod:`com4FlowPy`.

  THE MODULE IS CURRENTLY UNDER HEAVY DEVELOPMENT!

  **As such the code is not automatically tested or included in the code coverage yet.** Also AvaFrame coding and 
  naming conventions are not (yet) adhered.

  Use at your own risk and if you want to contribute to the modules improvement, you are very welcome to do so!
  We are trying to keep this description up to date with the latest deveopments in the AvaFrame ``master`` branch.

:py:mod:`com4FlowPy` is an open-source tool to compute gravitational mass flow (GMF) run outs and intensities. 
The main objective is to compute the spatial extent of GMF, which consists of the starting, 
transit and runout zones of GMF on the surface of a three dimensional terrain. The resulting 
run out is mainly dependent on the terrain and the location of the starting/release point.
No time-dependent equations are solved in the model. :py:mod:`com4FlowPy` uses a simple angle-of-reach approach in combination
with a raster-based flow-routing routine for solving the routing and stopping of the modeled GMF. 

:py:mod:`com4FlowPy` has been developed for regional scale applications and with a focus on model extendability and 
adaptability. While it has been successfully applied to different regional-scale case studies (see: :ref:`theoryCom4FlowPy:com4FlowPy theory` --> *com4FlowPy applications*), application over larger model
domains can be computationally demanding and computation times are strongly correlated with model parametrization and the
number of modeled release-cells.

The motivational background and concepts behind the model, as well as a list of references can be found under
:ref:`theoryCom4FlowPy:com4FlowPy theory`.


Running the code
----------------

Generate an environment as described in :ref:`developinstall:Advanced Installation (Linux)` or
:ref:`developinstallwin:Advanced Installation (Windows)`. Once you have a working ``pixi shell``, you can run the
model via::

    python runCom4FlowPy.py
     

Configuration
----------------

The model configuratios for :py:mod:`com4FlowPy` can be found in ``avaframe/avaframeCfg.ini`` (general AvaFrame config) and 
``avaframe/com4FlowPy/com4FlowPyCfg.ini`` (module specific config).

In these files,
all model parameters are listed and can be modified. We recommend to create a local copy
of both files and keep the default configurations in ``avaframe/avaframeCfg.ini`` (general AvaFrame config) and 
``avaframe/com4FlowPy/com4FlowPyCfg.ini`` (module specific config) untouched.
For this purpose, inside ``AvaFrame/avaframe/`` run:

  ::

    cp avaframeCfg.ini local_avaframeCfg.ini
    cp com4FlowPy/com4FlowPyCfg.ini com4FlowPy/local_com4FlowPyCfg.ini

and modify the parameter values in the files with ``local_`` prefix. 

in the ``avaframe/(local_)avaframeCfg.ini`` you can set the following general options:

- ``avalancheDir`` ... the avalanche directory (this is only used if ``useCustomPaths = False`` in ``avaframe/com4FlowPy/com4FlowPyCfg.ini``)
- ``nCPU`` ... number of CPU threads that will be used by the model (if *auto*, then ``CPUPercent`` is used), alternatively provide the number of CPU threads (should not exceed the number of actual threads on your machine)
- ``CPUPercent`` ... if ``nCPU = auto`` the number of CPUs used will be calculated based on this percentage.

in the ``avaframe/com4FlowPy/(local_)com4FlowPyCfg.ini`` you can set the following module specific options/parameters:

i) general model parameters:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``alpha``: :math:`\alpha` - travel angle in :math:`^{\circ}` :math:`\ldots` controls the maximum runout along the path
- ``exp``: Exponent controling the concentration of the routing flux and therefore lateral spreading behavior. :math:`exp = 1 \ldots` widespread process paths, :math:`exp \rightarrow \infty \ldots` very confied process paths (single flow direction).
- ``flux_threshold``: minimal flux value that is still processed by the routing algorithm (limits model runtimes by stopping further model caluclation in cells with excessively small *flux* values; thus also influencing process spreading together with ``exp``)
- ``max_z``: :math:`z^{\delta}_{max}\;[\rm{m}]` maximum kinetic energy height limit [m] :math:`\ldots` sets a hard limit to the max. energy line height (see: :cite:`HoJaRuZi_2013`) :math:`\rightarrow` can roughly be interpreted as a limit to maximum process velocities using the conversion :math:`v_{max}=(2 g z_{\delta}^{max})^{(1/2)}` 

ii) additional modules (forest, infrastructure)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``forest``: if set to ``True`` the runout calculation is performed with the *forest module* (a forest layer has to be provided)
- ``infra``: if set to ``True`` the calculation is performend with the *backcalculation module* (an infrastructure layer has to be provided)

if ``infra`` is set to ``True`` the infrastructure layer has to be provided either in ``avalancheDir/INPUTS/INFRA`` (if ``useCustomPaths=False``) or at the defined
``infraPath`` (if ``useCustomPaths=True``). The layer has to be of the same resolution and extent as the other input layers; infrastructure cells have to be coded with values > 0, while 
values <= 0 will be interpreted as non-infrastructure. If infrastructure cells should contain an ordinal ranking (e.g. infrastructure importance), then higher values indicate higher
infrastructure priority.

.. Note::
  - ``previewMode``: if this option is set to ``True`` not all release cells will be processed separately. Instead, release cells that are already contained in a previously modeled process path will not be modeled again.
  Using this option can be useful for preliminary parameter studys since it saves computational time.
  The previewMode will allow a rough approximation of results for output layers like ``zdelta``, ``travelLength``, or ``backCalculation`` with faster model run times.
  However, model results relying on separate processing of all release cells (``cellCounts``, ``fpTravelAngle``, ``zDeltaSum``, ``slTravelAngle``, ``routFluxSum``, ``depFluxSum``) will
  deviate strongly from ''full'' runs (``previewMode=False``) and should be interpreted with caution or simply removed from the output file list in ``(local_)com4FlowPyCfg.ini``.
  

iii) forest module parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``forestModule``: if ``forest=True`` different forest modules ``[ForestFriction, ForestDetrainment, ForestFrictionLayer]`` can be selected.

    - if ``forestModule in {ForestFriction, ForestDetrainment}``: *forest_layer* has to be scaled from 0 (no forest effect) to 1 (optimal forest effect).
    - if ``forestModule = ForestFrictionLayer`` each cell of the provided *forest_layer* has to contain either an ``absolute`` or ``relative`` value for ``alpha``, which will be utilized. 

depending on choice of the ``forestModule`` the following parameters can be set:

**forestModule = `ForestFriction`**:

Friction (i.e. :math:`\alpha`) on forested pixels/raster cells will be increased. The actual value :math:`\Delta_{\alpha}\;[^{\circ}]`, by which the global :math:`\alpha`
will be incremented is calculated as a function of ``maxAddedFrictionFor``, ``minAddedFrictionFor``, ``velThForFriction``, the FSI value of the forested cell (:math:`FSI\in\{0,\ldots,1\}`), and the energy-line height :math:`z^{\delta}` or equivalent velocity :math:`v=(2 g z^{\delta})^{(1/2)}` calculated at the cell.

- ``maxAddedFrictionFor``: max. added friction on a forested pixel expressed as increment to :math:`\alpha` in degrees :math:`[^{\circ}]`
- ``minAddedFrictionFor``: min. added friction on a forested pixel expressed as increment to :math:`\alpha` in degrees :math:`[^{\circ}]`
- ``velThForFriction``: velocity limit in :math:`\frac{\rm{m}}{\rm{s}}` above which added friction on forested pixels is set to ``minAddedFrictionFor``  

**forestModule = `ForestDetrainment`**:

In addition to increased friction also *flux* will be `detrained` on forested raster/cells. The amount of detrained *flux* is calculated in analogy to the added friction as a function of ``maxDetrainmentFor``, ``minDetrainmentFor``, ``velThForDetrain``, FSI and local :math:`z^{\delta}`.

- ``maxAddedFrictionFor``: max. added friction on a forested pixel expressed as increment to :math:`\alpha` in degrees :math:`[^{\circ}]`
- ``minAddedFrictionFor``: min. added friction on a forested pixel expressed as increment to :math:`\alpha` in degrees :math:`[^{\circ}]`
- ``velThForFriction``:  velocity limit in :math:`\frac{\rm{m}}{\rm{s}}` above which added friction on forested pixels is set to ``minAddedFrictionFor``  
- ``maxDetrainmentFor``: max. amount of *flux* that can be `detrained` on a forested cell
- ``minDetrainmentFor``: min. amount of *flux* that can be `detrained` on a forested cell 
- ``velThForDetrain``: velocity limit in :math:`\frac{\rm{m}}{\rm{s}}` above which detrained *flux* on forested pixels is set to ``minDetrainmentFor``

**forestModule = `ForestFrictionLayer`**:

If 'ForestFrictionLayer' is selected, the user-provided *forest_layer* has to contain ``absolute`` or ``relative`` :math:`\alpha` 
in :math:`^{\circ}` on forested cells. In case of  ``absolute``, the provided :math:`\alpha` in the *forest_layer* will
be used; in case of ``relative`` the provided :math:`\alpha` in the *forest_layer* will be added to :math:`\alpha` set in
the general model parameters. In any case a check is performed, that :math:`\alpha` on forested cells has to be equal or
greater than the global :math:`\alpha`.

- ``forestFrictionLayerType``: can be either ``absolute`` or ``relative``

**forestInteraction**:

If ``forest = True`` there is an option to switch on ``forestInteraction``.
The user-provided *forest_layer* is treated binary, which means that forest (``cell.isForest = 1``) is considered when values > 0, no forest is considered when values <= 0 (``cell.isForest = 0``).
If ``forestInteraction = True``, an additional output Layer is computed, which represents the number of forested raster cells a path runs through. In this forest interaction layer, locations (raster cells) of paths are assigned to the number of forested cells previously hit. The output raster layer represents the i) **minimum** forest length within the path (that is from one release cell) and ii) the **minimum** value of overlapping paths. See an application and further description of the forestInteractionLayer in :cite:`SpHeMiFi2024`


iv) variable parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are options to set for each path variable parameters:

- alpha (``variableAlpha = True``), 
- max. zDelta (``variableUmaxLim = True``),
- exponent (``variableExponent = True``). 

When an option is switched on (set ``True``), the user needs to provide a raster file, that contains values for the respective parameter in each grid cell that is assigned to a release cell. The paths are computed with the respective parameters.
If the value of the variable layer in the cell that is assigned to a release cell is not > 0, the default parameters are used as described in i).
When ``variableUmaxLim = True``, the type of the provided parameter is required: ``varUmaxParameter = uMax`` (in m/s) or ``varUmaxParameter = zDeltaMax`` (in m). (A layer containing release cells is still required).


v) tiling and multiprocessing parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the model extent (i.e. number of cells and/or rows in the input layers) is larger than ``tileSize``, then :py:mod:`com4FlowPy` 
will automatically split the input layers into different tiles (these are pickled to ``.npy`` files inside ``\temp`` folder). 
Each (quadratic) tile will then be consecutively calculated using all CPUs as defined by ``nCPU`` in ``avaframeCfg.ini``. The 
``tileOverlap`` option defines by which margins the tiles overlap; in overlapping parts of the model domain the outputs
of the single tiles are combined (maximum, sum - depending on output variable). Note that if a single tile does not cover
the entire process path, the path will be cut off, leading to incorrect results.

The default settings provide reasonable performance on standard machines/model domains - however for special applications (e.g. modeling
over large areas or on HPC hardware, **different raster resolution**) tweaking parameters might improve model performance.

- ``tileSize``: tile size in meters (default = :math:`15\;\rm{km}`)
- ``tileOverlap``: overlap between tiles in meters (default = :math:`5\;\rm{km}`)

These parameters control multiprocessing behavior (each tile is processed in parallel by a number of available CPUs). 
Depending on configuration of available CPU and RAM these settings might be tweaked

- ``procPerCPUCore``: Processes that can be spawned per CPU (default = 1)
- ``chunkSize``: (default = 50) 
- ``maxChunks``: max. number of single work-loads that are spawned for one tile (default = 500 ) - if there are issues with RAM overflow this number should be decreased


Input Files
-------------

in case ``useCustomPaths = False`` in ``avaframe/com4FlowPy/com4FlowPyCfg.ini`` the **Input Data** has to be provided in
the following folder structure inside the ``avalancheDir`` directory inside which is defined in ``avaframe/avaframeCfg.ini``:

::

    NameOfAvalanche/
      Inputs/
        ElevationModel - digital elevation model (.asc)
        REL/      - release area file (can be either .asc, .tif, or .shp) <required>
        RES/      - forest structure information (FSI) (.asc or .tif) <optional>
        INFRA/    - infrastructure layer (.asc or .tif) <optional>
        ALPHA/	  - variable alpha angle layer (.tif) <optional>
        UMAX/	  - variable uMax layer (.tif) <optional>
        EXP/	  - variable exponent layer (.tif) <optional>
      Outputs/
      Work/

if ``useCustomPaths = True`` in ``avaframe/com4FlowPy/com4FlowPyCfg.ini`` then the paths to the input files and working-
directory can be defined inside ``avaframe/com4FlowPy/com4FlowPyCfg.ini`` as follows (:math:`\rightarrow` *this option allows placing model
inputs and working directories/model outputs in different places, which might be desirable for some applications*):

- ``workDir`` :math:`\ldots` working directory (a ``temp/`` folder, model log and model results will be written here)
- ``demPath`` :math:`\ldots` path to input DEM (must be ``.asc`` currently)
- ``releasePath`` :math:`\ldots` path to release area raster (``.asc, .tif``)
- ``infraPath`` :math:`\ldots` path to infrastructure raster (``.asc, .tif``) (required if ``infra = True``)
- ``forestPath`` :math:`\ldots` path to forest (FSI) raster (``.asc, .tif``) (required if ``forest = True``)
- ``varAlphaPath`` :math:`\ldots` path to variable alpha angle raster (``.tif``) (required if ``variableAlpha = True``)
- ``varUmaxPath`` :math:`\ldots` path to variable uMax raster (``.tif``) (required if ``variableUmaxLim = True``)
- ``varExponentPath`` :math:`\ldots` path to variable Exponent raster (``.tif``) (required if ``variableExponent = True``)


**All rasters need the same resolution (we recommend 10x10 meters) and raster extent!!**
In all rasters values < 0 are interpeted as *noData* (standard no data values = -9999).
The locations identified as release areas need values > 0.

if ``useCustomPaths = True`` and ``deleteTempFolder=True`` then the ``temp/`` folder inside the ``workDir`` will be 
deleted after completion of the model run (can be useful for calculation of large model domains).

Output
-------

All outputs are written in *'.tif'* or in *'.asc'* raster format (controlable via the ``outputFileFormat`` option in ``(local_)com4FlowPyCfg.ini``, default is *'.tif'*) in the same resolution and extent as the input raster layers.
You can customize which output rasters are written at the end of the model run by selecting the desired output files through the ``outputFiles`` option in ``(local_)com4FlowPyCfg.ini``.
In the output rasters, the cells, that are not affected by the process, are specified by ``outputNoDataValue`` (default is -9999).

By default the following four output layers are written to disk at the end of the model run:

- ``zdelta``: the maximum z_delta of all paths for every raster cell (geometric measure of process magnitude, can be associated to kinetic energy/velocity)
- ``cellCounts``: number of paths/release cells that route flux through a raster cell
- ``travelLengthMax``: the travel length along the flow path (the maximum value of all paths for every raster cell) - **this output is equivalent to the ``travelLength`` output from previous code versions!**
- ``fpTravelAngleMax``: the gamma angle along the flow path (the maximum value of all paths for every raster cell) - **this output is equivalent to the ``fpTravelAngle``output used in previous versions of the code!!**

In addition these output layers are also available:

- ``flux``: The maximum routing flux of all paths for every raster cell
- ``zDeltaSum``: z_delta summed up over all paths on every raster cell
- ``slTravelAngle``: gamma angle calculated along a straight-line between release cell and current cell
- ``routFluxSum``: routing flux summed up over all paths
- ``depFluxSum``: deposited flux summed up over all paths
- ``travelLengthMin``: the travel length along the flow path (the minimum value of all paths for every raster cell)
- ``fpTravelAngleMin``: the gamma angle along the flow path (the minimum value of all paths for every raster cell)

If ``forestInteraction = True`` this layer will be written automatically (no need to separately define in ``outputFiles``):

- ``forestInteraction``: minimum number of forested raster cells a path runs through

If ``infra = True`` this layer will be written automatically (no need to separately define in ``outputFiles``):

- ``backcalculation``: Parts of modeled process paths upslope of infrastructure cells that are ''hit'' by (a) modeled process(es).

 .. Model Parameterisation
 .. ------------------------
 ..
 .. :py:mod:`com4FlowPy` might be utilized to model a range of different GMFs. Past applications of the model have mainly been
 .. focused on *snow avalanches* and *rockfall*, but also other GMFs can potentially be modelled.
 .. While **we emphasize, that careful adaptation/calibration of model parameters to the specific use case is essential**, we
 .. can try to provide some hints on parameter ranges based on past applications.

 .. a) general model parameters
 .. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 .. - ``alpha``: adaptation based on observations
 .. - ``max_z``: :math:`z_{\delta}^{max}\;[\rm{m}]` might be defined based on observed max. velocities for different GMFs.

 .. b) forest module
 .. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
