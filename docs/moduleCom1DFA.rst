com1DFA: DFA-Kernel
===========================

:py:mod:`com1DFA` is a module for dense flow (snow) avalanche computations (DFA) .
It is a python and cython implementation of the DFA C++ implementation samosAT
(Snow Avalanche Modeling and  Simulation- Advanced Technologies) developed by the Austrian government
in cooperation with the company AVL List GmbH in Graz.
Calculations are based on the thickness integrated governing equations and
solved numerically using the smoothed particle hydrodynamics (sph) method. Please note
the use of *thickness averaged/integrated* instead of *depth averaged/integrated* for clarity and consistency.

Dense flow avalanche simulations can be performed for different release area scenarios, with or without
entrainment and/or resistance areas, and is controlled via a configuration file.
The configration can be modified in order to change any of the default settings and also allows
to perform simulations for varying parameters all at once.

.. Note::
   The configuration provided with com1DFA is well-tested and applied for
   hazard mapping (in Austria). If you change configuration parameters, be aware that
   unwanted/unexpected/spurious side-effects might appear. This is especially
   true if you switch to something far outside the intended range (i.e.
   changing density from snow to something like rock). Furthermore, be aware
   that the parameters are calibrated in connection, so
   changing one might necessitate also changing other connected parameters!

Input
---------

DFA simulations are performed within an avalanche directory, organized with the
folder structure described below.

.. Note::  An avalanche directory can be created by running: :py:mod:`runInitializeProject.py`, which creates the required folder structure:

  ::

    NameOfAvalanche/
      Inputs/
        REL/      - release area scenario
        RES/      - resistance areas
        ENT/      - entrainment areas
        POINTS/   - split points
        LINES/    - avalanche paths
        POLYGONS/ - crop shapes
        SECREL/   - secondary release areas
        RASTERS/  - friction parameter fields
      Outputs/
      Work/


In the directory ``Inputs``, the following files are required. Be aware that ALL inputs have to be provided in the same
projection:

* digital elevation model as raster file with either `ESRI grid format <https://desktop.arcgis.com/en/arcmap/10.3/manage-data/raster-and-images/esri-ascii-raster-format.htm>`_
  or GeoTIFF format. The format of the DEM determines the format of the output files.

* release area scenario as (multi-) polygon shapefile (in Inputs/REL; multiple features are possible)

  - the release area polygon must not contain any "holes" or inner rings
  - the release area name should not contain an underscore, if so '_AF' is added.
  - recommended attributes are *name*, *thickness* (see :ref:`moduleCom1DFA:Release-, entrainment thickness settings`)
    and *ci95* (see :ref:`moduleAna4Stats:probAna - Probability maps`)
  - ALL features within one shapefile are released at the same time (and interact), this is what we refer to as *scenario*
  - if you want to simulate different scenarios with the same features, you have to copy them to separate shapefiles


and the following files are optional. Please note: in the standard configuration (i.e. ``simTypeList = available``) ,
the *null* variant is always run! I.e. if a resistance and/or an entrainment file is given (as described below),
at least two results are generated: the *null* variant and the variant with entrainment and/or resistance.

* one entrainment area (multi-) polygon shapefile (in Inputs/ENT)

  - marks the (multiple) areas where entrainment can occur.
  - attribute *thickness* (see :ref:`moduleCom1DFA:Release-, entrainment thickness settings`)
  - must not contain any "holes" or inner rings


* one resistance area (multi-) polygon shapefile (in Inputs/RES)

  - marks the (multiple) areas where resistance is considered
  - resistance areas must not contain any "holes" or inner rings
  - please consider the information about resistance below :ref:`moduleCom1DFA:Resistance setup`


* one secondary release area (multi-) polygon shapefile (in Inputs/SECREL)

  - can have multiple release areas, each as one feature
  - same setup as the release area scenario (see above)
  - features will release as soon as at least one particle enters its area
  - release area polygons must not contain any "holes" or inner rings

* raster files for the Voellmy friction parameters :math:`\mu` and :math:`\xi` (in Inputs/RASTERS)

  - spatial field of :math:`\mu` and :math:`\xi` values with same extent as DEM
  - file names need to end with ``_mu.*`` and ``_xi.*``
  - only one file per parameter allowed
  - if ``meshCellSize`` is different from simulation ``meshCellSize`` fields will be remeshed
  - only used if ``frictionModel`` is set to ``spatialVoellmy``

* one ``_cropshape.shp`` shape file (in Inputs/POLYGONS)

  - provides a polygon located inside the DEM to define area for report plots of peak fields (bounds of polygon)
  - if not provided peak fields are shown for the extent where peak field values are nonzero




Release-, entrainment thickness settings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. Note::
    Thickness is unambiguous: it is measured normal to the slope.

Release, entrainment and secondary release thickness can be specified in two different ways:

1. Via **shape file**:

  - add an attribute called `thickness` for each feature
  - important: ALL features have to have a single thickness value, which can differ between features
  - for entrainment area only: if the thickness value is missing, the thickness value is
    taken from `entThIfMissingInShp` (default 0.3 m) in the configuration file. If multiple features are
    in the entrainment file the thickness attribute has to be set either for ALL or NONE of the features.
  - for backwards compatibility, the attribute 'd0' also works, but we suggest to use `thickness` in new projects
  - set the flag `THICKNESSFromFile` (i.e. relThFromFile, entThFromFile,
    secondaryRelThFromFile) to True in the configuration file (default is True)
  - a parameter variation can be added with the `THICKNESSPercentVariation`
    parameter in the configuration file in the form of
    ``+-percentage$numberOfSteps``. Provided a `+` a positive variation will be
    performed, if `-` is given, only a negative variation is performed. If no
    sign is given: both directions will be used. Additionally, a variation can be
    added with the `THICKNESSRangeVariation` parameter in the configuration file
    in the form of ``+-range$numberOfSteps``. Provided a `+` a positive variation
    will be performed, if `-` is given, only a negative variation is performed.
    If no sign is given: both directions will be used. Furthermore, there is the
    option to vary the thickness in a range of +- the 95% confidence interval
    value, which is also read from the shape file (requires an attribute called
    ci95). In order to use this variation, set the 'THICKESSRangeFromCiVariation'
    to ``ci95$numberOfSteps``.

2. Via **configuration file (ini)**:

  - set the flag 'THICKNESSFromFile' to False
  - provide your desired thickness value in the respective THICKNESS parameter (i.e. relTh, entTh or secondaryRelth)
  - in addition to the `THICKNESSPercentVariation` and `THICKNESSRangeVariation`
    options (see option 1) and the standard variation options in
    :ref:`configuration:Configuration`, you can also directly set e.g. `relTh =
    1.$50$2`, ``referenceValue$+-percentage$numberOfSteps``, resulting in a
    variation of relTh from 0.5 to 1.5m in two steps.

Only available for release thickness:

3. Via **release thickness file**:
# TODO: HERE IS A MISTAKE, but not sure what is true:
  - set the flag 'relThFromFile' to False
  - set the flag 'relThFromFile' to True
  - save a raster file with info on release thickness as raster file in
    ``Inputs/RELTH`` the number of rows and columns must match the DEM raster
    with desired meshCellSize (recommended)
  - if the cellsize does not match the requested meshCellSize, the file is
    remeshed.


Friction parameters
^^^^^^^^^^^^^^^^^^^

By default the friction parameter set *samosATAuto* is active. This uses the calculated release volume (including
secondary release areas) to determine the parameters used for the samosAT friction model.
See :ref:`samosatfrict` for the limits regarding release volumes.

Resistance setup
^^^^^^^^^^^^^^^^^^^

.. Note::

    In  versions earlier than 1.13, the default resistance setup included information about tree diameter, tree spacing,
    etc. and was dependent on an effective thickness (a function of flow thickness). Please check out previous
    documentation versions for details. This change leads to different results, so please assess whether the new
    default setup is appropriate for your project. We provide a difference report at
    `our homepage <https://avaframe.org/reports/>`_.

By default, the resistance setup includes detrainment depending on the flow thickness (FT) and velocity (FV). Please 
see :ref:`resistance` for more information.


DEM input data
^^^^^^^^^^^^^^^^
Regarding the DEM data: if the DEM in ``Inputs`` is not of cell size 5 meters, it is remeshed to a
cell size of 5 meters. However, it is also possible to specify a desired cell size in the
configuration file (parameter `meshCellSize`). In this case, also consider reading :ref:`FAQ:Can the spatial resolution of simulations performed with com1DFA (dense flow) be changed?`.
If the cell size of the DEM in ``Inputs`` is equal to the desired mesh cell size, the DEM is used without modification. If the cell sizes do not match, several options are available:

    - cleanremeshedRasters = True, directory ``Inputs/remeshedRasters`` is cleaned, and the DEM in Inputs/
      is remeshed to the desired cell size - this is the default setting

    - cleanremeshedRasters = False and a DEM including the name of the DEM in Inputs/ and the desired cell size is found
      in Inputs/remeshedRasters - this DEM is used without modification

    - cleanremeshedRasters = False and no matching DEM is found in Inputs/remeshedRasters - the DEM in Inputs/ is remeshed
      to the desired cell size

If the DEM in Inputs/ is remeshed, it is then saved to ``Inputs/remeshedRasters`` and available for subsequent
simulations.


Dam input
^^^^^^^^^

The com1DFA module provides the option to take the effect of dams into account.
This is done using a ad-hoc method based on particles being reflected/deflected by a dam wall.

The dam is described by the crown line, the slope and the restitution coefficient:

  - crown line as shape file (use the line type and enable the "additional dimensions" option in order
    to specify the z coordinate).
    The z coordinate corresponds to the absolute height (terrain elevation plus dam height).
    The dam is then located on the left side of the dam (when one travels from the first point to the last
    point of the shapefile line).
    The dam shape files live in the ``avaDir/Inputs/DAM/`` directory (only one file is allowed).

  - the ``slope`` of the dam (in degrees °) between the horizontal plane and the wall to be provided in the shape file
    as an attribute (default value is 60° in the provided examples: avaSlide, avaKot and avaBowl)

  - the restitution coefficient (:math:`\alpha_\text{rest}`), a float between 0 (no reflection
    in the normal direction) and 1 (full reflection) to be specified in the ini file (default value is 0)




Model configuration
--------------------
The model configuration is read from a configuration file: ``com1DFA/com1DFACfg.ini``. In this file,
all model parameters are listed and can be modified. We recommend to create a local copy
and keep the default configuration in ``com1DFA/com1DFACfg.ini`` untouched.
For this purpose, in ``AvaFrame/avaframe/`` run:

  ::

    cp com1DFA/com1DFACfg.ini com1DFA/local_com1DFACfg.ini

and modify the parameter values in there. For more information see :ref:`configuration:Configuration`.

It is also possible to perform multiple simulations at once, with varying input parameters.


Output
---------
Using the default configuration, the simulation results are saved to: *Outputs/com1DFA* and include:

* raster files of the peak values for pressure, flow thickness and flow velocity (*Outputs/com1DFA/peakFiles*)
* raster files of the peak values for pressure, flow thickness and flow velocity for the initial time step (*Outputs/com1DFA/peakFiles/timeSteps*)
* markdown report including figures for all simulations (*Outputs/com1DFA/reports*)
    - if a ``_cropshape.shp`` file provided in Inputs/POLYGONS, plots are cropped to the rectangular bounds of the polygon
    - if ``showOnlineBackground = True`` in avaFrameCfg.ini and a suitable ``mapProvider`` is set, peak fields are plotted onto the corresponding map
* mass log files of all simulations (*Outputs/com1DFA*)
* configuration files for all simulations (*Outputs/com1DFA/configurationFiles*)
    - all configuration files that were created for a simulation to be run are stored in (*Outputs/com1DFA/configurationFiles*)
    - one file for each simulation that has actually been performed is saved in (*Outputs/com1DFA/configurationFiles/configurationFilesDone*)
    - one file for each simulation that has actually been performed by the latest call of ``runCom1DFA.py`` is saved in (*Outputs/com1DFA/configurationFiles/configurationFilesLatest*)

    .. Note::
        This kind of storage of configurations from actually performed simulations allows a run that has been terminated
        to be resumed without re-running simulations that have already been performed. For this, just restart the run.

The naming of the output files has the following structure, shown with the example of
*relAlr_ff5f9b78c6_C_L_null_dfa_ppr*:

* *relAlr* - release area name, usually the name of the shapefile
* *ff5f9b78c6* - individual hash of the configuration file used for the simulation. All files related to this simulation
  have the same hash in their name. This allows to identify which files belong to which simulation.
* *C* - indicator of the setup used: D for default setup, C for custom setup, i.e. something was changed in the
  configuration file
* *L* - indicator of the size category used for the friction model: L for large, M for medium, S for small
* *null* - indicator of the run type: null for null variant, ent for entrainment variant, res for resistance variant, etc
* *dfa* - indicator of the simulation type: dfa for dense flow avalanche
* *ppr* - indicator of the result type: ppr for peak pressure, pfv for peak flow velocity, pft for peak flow thickness, etc


Optional outputs

* pickles of particles properties (:ref:`com1DFAAlgorithm:Particle properties`.) for saving time steps if particles are added to the list of resTypes in your local copy of ``com1DFACfg.ini``
* a csv file of specified particle properties for the saving time steps if particles are added to the list of resTypes in your local copy of ``com1DFACfg.ini`` and if in the VISUALISATION section writePartToCsv is set to True

However, in the configuration file, it is possible to change the result parameters and time Steps that shall be exported.
The result types that can be chosen to be exported are (all correspond to fields except the particles):

* ppr - peak pressure
  (:math:`pressure = \mathbf{\rho}  \mathbf{u}²` with :math:`\rho` snow density and :math:`\mathbf{u}` flow velocity)
* pfv - peak flow velocity
* pft - peak flow thickness
* pta - peak travel angle
* FV - flow velocity
* FT - flow thickness
* P - pressure
* FM - flow mass
* Vx, Vy, Vz - velocity x-, y- and z-component
* TA - travel angle
* dmDet - detrained mass
* FTDet - thickness of detrained mass computed based on dmDet / (rho * area of cell)
* sfcChange - flow depth that changed the surface topography due to detrainment, stopping and entrainment
* demAdapted - adapted DEM considering stopping/ detrainment/ entrainment
* particles (:ref:`com1DFAAlgorithm:Particle properties`)

Have a look at the designated subsection Output in ``com1DFA/com1DFACfg.ini``.


Parallel computation
--------------------

If multiple runs of com1DFA are to be executed, these will be calulated in parallel via
multiprocessing. So each task itself is calculated on only one core, but different tasks
are run at the same time.

This happens if you have one of the following (or a combination of them):

* multiple scenarios (multiple input release shapefiles)
* multiple runtypes, i.e null variant and entrainment/resistance variant (e.g.: simTypeList = null|ent)
* some kind of parameter variation (e.g.: relTh = 1.0|1.5|1.7)

The number of CPU cores is controlled in the main ``avaframeCfg.ini`` file. By default a
maximimum of 50 percent of your available cores is being utilized. However you can set
a different number if needed. For sequential execution set nCPU to 1.


To run
--------

* first go to ``AvaFrame/avaframe``
* copy ``avaframeCfg.ini`` to ``local_avaframeCfg.ini`` and set your desired avalanche directory name
* create an avalanche directory with required input files - for this task you can use :ref:`moduleIn3Utils:Initialize Project`
* copy ``com1DFA/com1DFACfg.ini`` to ``com1DFA/local_com1DFACfg.ini`` and if desired change configuration settings
* if you are on a develop installation, make sure you have an updated compilation, see
  :ref:`developinstall:Setup AvaFrame`
* run:
  ::

    python3 runCom1DFA.py


Theory
--------


The governing equations of the dense flow avalanche are derived from the
incompressible mass and momentum balance on a Lagrange control volume
([Zw2000]_ [ZwKlSa2003]_). Assuming the avalanche is much longer and larger
than thick, it is possible to integrate the governing equations over the thickness
of the avalanche and operate some simplifications due to the shape of the avalanche.
This leads, after some calculation steps described in details in Theory
:ref:`theoryCom1DFA:Governing Equations for the Dense Flow Avalanche` to:

.. math::
    \begin{aligned}
    &\frac{\mathrm{d}V(t)}{\mathrm{d}t} = \frac{\mathrm{d}(A_b\overline{h})}{\mathrm{d}t}
    = \frac{\rho_{\text{ent}}}{\rho_0}\,w_f\,h_{\text{ent}}\,\left\Vert \overline{\mathbf{u}}\right\Vert\\
    &\frac{\,\mathrm{d}\overline{u}_i}{\,\mathrm{d}t} =
    g_i + \frac{K_{(i)}}{\overline{\rho}\,A\,\overline{h}}\,\oint\limits_{\partial{A}}\left(\frac{\overline{h}\,\sigma^{(b)}}{2}\right)n_i\,\mathrm{d}l
    -\delta_{i1}\frac{\tau^{(b)}}{\overline{\rho}\,\overline{h}} - C_{\text{res}}\,\overline{\mathbf{u}}^2\,\frac{\overline{u_i}}{\|\overline{\mathbf{u}}\|}
    -\frac{\overline{u_i}}{A\,\overline{h}}\frac{\,\mathrm{d}(A\,\overline{h})}{\,\mathrm{d}t} + \frac{F_i^{\text{ent}}}{\overline{\rho}\,A\,\overline{h}}\\
    &\overline{\sigma}^{(b)}_{33} = \rho\,\left(g_3-\overline{u_1}^2\,\frac{\partial^2{b}}{\partial{x_1^2}}\right)\,\overline{h}
    \end{aligned}


Numerics
---------

Those equations are solved numerically using a **SPH** method (:cite:`LiLi2010,Sa2007`).
**SPH**  is a mesh free method where the basic idea is to divide the avalanche into
small mass particles. The particles interact with each other according to the
equation of motion described in :ref:`moduleCom1DFA:Theory` and the chosen kernel function.
This kernel function describes the domain of influence of a particle (through the smoothing length parameter).
See theory :ref:`theoryCom1DFA:com1DFA DFA-Kernel theory` for further details.
