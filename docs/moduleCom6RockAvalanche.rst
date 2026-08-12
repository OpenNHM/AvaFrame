com6RockAvalanche: Rock Avalanche
=================================

.. Warning:: This is highly experimental and not tested!

The com6RockAvalanche computational module provides an override setting for com1DFA targeting the simulation of rock
avalanches.

Tips
----

* Download the default configuration for each module via
  ``OpenNHM > AvaFrame_Experimental > Get default module ini`` (see :ref:`connector:Experimental`).
* After a run, restart the ``Rock Avalanche (com6)`` tool before rerunning with changed parameters; a fresh instance
  is often more reliable than reusing the old one.
* Parameter meanings can be looked up in the documentation, e.g. ``massPerPart``:
  https://docs.avaframe.org/en/latest/com1DFAAlgorithm.html#initialize-particles

Input
-------

The standard inputs required to perform a simulation run using :py:mod:`com1DFA`
can be found here: :ref:`moduleCom1DFA:Input`.
However there is one main difference: com6RockAvalanche NEEDS a release thickness raster file. This file has to have
the exact same dimensions and resolution as the topography file.
There is a run script to perform a rock avalanche com1DFA run: :py:mod:`runCom6RockAvalanche.py`,
and the configuration settings can be found in ``com6RockAvalanche/com6RockAvalancheCfg.ini``.

The following files are required:

* a DEM from which the release volume has already been removed
  (see :ref:`moduleCom6RockAvalanche:Scarp Calculation`)
* a release thickness raster with the same extent and resolution as the DEM

The following are optional:

* an entrainment polygon shape file (raster support may follow)
* a configuration file that overrides the default settings (see :ref:`connector:Experimental` for how to obtain it)

To run
------

* first go to ``AvaFrame/avaframe``
* copy ``avaframeCfg.ini`` to ``local_avaframeCfg.ini`` and set your desired avalanche directory name
* create an avalanche directory with required input files - for this task you can use
  :ref:`moduleIn3Utils:Initialize Project`
* copy ``com6RockAvalanche/com6RockAvalancheCfg.ini`` to
  ``com6RockAvalanche/local_com6RockAvalancheCfg.ini`` and if desired change configuration settings
* if you are on a develop installation, make sure you have an updated compilation, see
  :ref:`complexUsage:Update AvaFrame`
* optionally, the ``spatialVoellmy`` friction model can be selected with
  ``--friction_calibration spatialVoellmy``. When using the QGis Connector,
  a  shapefile with ``mu`` and ``xi`` attributes
  can be provided and the required rasters will
  be generated automatically. See
  :ref:`moduleIn3Utils:Spatial Voellmy inputs` for details.
* if you are on a develop installation, make sure you have an updated compilation, see :ref:`complexUsage:Update AvaFrame`
* run:
  ::

    pixi run python runCom6RockAvalanche.py


Run via QGis Connector
----------------------

Alternatively, run from QGIS via the OpenNHM connector: open the OpenNHM tool and select ``Rock Avalanche (com6)``.
Choose a suitable DEM and release layer, and optionally an entrainment layer. A path to a configuration file with
additional settings can also be provided.

It is strongly recommended to set ``meshCellSize`` to the cell size of the input rasters
(see :ref:`moduleCom6RockAvalanche:Remeshing`); otherwise remeshing artifacts can occur. For example, the Fluchthorn
dataset uses 10 m.

.. Warning:: Depending on the DEM, release and settings, runs can take very long (up to several hours).

Remeshing
---------

The default configuration remeshes the DEM and release raster to 5 m unless their cell size already equals 5 m. This
occasionally introduces no-data values, which then cause errors.

Either set the mesh size in the configuration file, or reproject the inputs (DEM and release raster) to the target
cell size beforehand. In QGIS use ``Raster -> Projections -> Warp (Reproject)`` and set the output resolution, then
repeat for the release raster. To adjust the configuration instead, add or change::

    # expected mesh size [m]; use the cell size of your raster
    meshCellSize = 10

Entrainment
-----------

A full description of the entrainment model, including formulas, is available here:
https://docs.avaframe.org/en/latest/theoryCom1DFA.html#entrainment
and an introduction on how to use entrainment is available here:
https://docs.avaframe.org/en/latest/moduleCom1DFA.html#input

Entrainment is the uptake of material by the rock avalanche during its flow. Two main processes are distinguished:

* plowing: uptake of material at the rock avalanche front
* erosion: uptake of material at the rock avalanche base

com1DFA uses basal erosion by default; plowing is disabled. The entrainment rate strongly influences rock avalanche
mass and dynamics. Entrainment can be provided as a shape or raster file, but the QGIS connector currently only supports
shapes. Use a (multi-)polygon shape file that:

* defines the areas where entrainment occurs (areas should not overlap)
* carries a ``thickness`` attribute (entrainment material thickness, measured normal to the slope) and contains no
  holes or rings

If no ``thickness`` attribute is present, the default ``entThIfMissingInShp`` from the configuration file is used.
The flag ``THICKNESSFromFile`` (i.e. ``relThFromFile``, ``entThFromFile``, ``secondaryRelThFromFile``) must be set to
True in the configuration file (default is True).

Thickness variations for parameter studies can be defined as follows:

* ``entThPercentVariation``: ``+-percentage$numberOfSteps``; ``+`` gives a positive variation, ``-`` a negative one,
  no sign gives both directions
* ``entThRangeVariation``: ``+-range$numberOfSteps``, same sign convention
* ``entThRangeFromCiVariation``: ``ci95$numberOfSteps``, varies the thickness within +- the 95% confidence interval
  read from a ``ci95`` attribute in the shape file

Entrainment parameters
----------------------

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Parameter
     - Meaning
     - Notes / effect
   * - ``entThFromFile``
     - Read entrainment areas from a shape file (True) or use the global ``entTh`` value (False).
     - If True, expects a file such as ``Inputs/ENT/entrainment.shp``.
   * - ``entThIfMissingInShp``
     - Fallback thickness [m] if the shape file has no thickness attribute.
     - Default 0 m; used for features without a valid value.
   * - ``entThPercentVariation``, ``entThRangeVariation``, ``entThRangeFromCiVariation``, ``entThDistVariation``
     - Options for parameter studies (entrainment thickness variations).
     - E.g. ``entThRangeVariation = 0.5$10`` gives a +-0.5 m variation in 10 steps.
   * - ``entTh``
     - Fixed entrainment thickness [m], only relevant if ``entThFromFile = False``.
     - E.g. ``entTh = 0.2``.
   * - ``entEroEnergy``
     - Erosion energy [J/m2]; controls the energy loss from mass uptake.
     - Higher values lose more energy (and thus speed) during entrainment.
   * - ``entShearResistance``
     - Shear resistance of the entrained material.
     - 0 = no additional resistance.
   * - ``entDefResistance``
     - Deformation resistance of the material.
     - 0 = no deformation resistance.

Common issues
-------------

* Remeshing or other processing errors produce no-data values. Set ``meshCellSize`` to the raster cell size
  (see :ref:`moduleCom6RockAvalanche:Remeshing`).
* Negative values in the release thickness raster cause errors. The
  :ref:`moduleCom6RockAvalanche:Scarp Calculation` step clamps negative thicknesses to 0.

Scarp Calculation
-----------------

* first go to ``AvaFrame/avaframe``
* copy ``avaframeCfg.ini`` to ``local_avaframeCfg.ini`` and set your desired avalanche directory name
* create an avalanche directory - for this task you can use :ref:`moduleIn3Utils:Initialize Project`

Scarp Input
~~~~~~~~~~~

* all input files are automatically read from the set avalancheDir. No file paths need to be specified
* elevation: DEM (ASCII), which serves as the basis for calculating the scarps. Must be in avalancheDir/Inputs.
* geometries: a shapefile containing point geometries. These points represent the centers of the ellipsoids or planes.
  The coordinates (x,y) of these points are used. If the plane method is used, the shape file must contain the
  attributes ``zseed``, ``dipdir_azi`` and ``dipAngle`` as float values. If the ellipsoid method is used, the shape
  file must contain the attributes ``maxdepth``, ``semimajor``, ``semiminor``, ``dipAngle``, ``dipdir_azi``,
  ``rotAngle`` and ``offset`` (see below). The file must be located in avalancheDir/Inputs/POINTS and the file name
  must end with ``_coordinates``. If you are using the QGis Connector, the naming and location of the file is not
  relevant.
* perimeter: a shapefile that defines the spatial extent within which the scarp geometry is applied. It is rasterized
  to a binary mask and used to clip the scarp surface to a predefined area: inside the perimeter the calculated scarp
  elevation replaces the DEM where it is lower, outside the perimeter the original DEM is kept. This allows to limit
  the scarp to a geologically meaningful release area, prevent artificial terrain modification outside the intended
  scarp, and combine multiple scarp elements without affecting the surrounding topography. The file must be located
  in avalancheDir/Inputs/POLYGONS and the file name must end with ``_perimeter``. If you are using the QGis
  Connector, the naming and location of the file is not relevant.

Attribute meanings
~~~~~~~~~~~~~~~~~~

Plane:

* ``zseed``: z coordinate of the plane center (m). Usually the pre-event terrain elevation at the scarp initiation
  point, but any value can be set.
* ``dipdir_azi``: azimuth, the direction the plane faces (degree).
* ``dipAngle``: tilt angle of the plane (degree).

Ellipsoid:

* ``maxdepth``: maximum depth of the untilted ellipsoid (m). E.g. 50 m means the geometric center of the untilted
  ellipsoid lies 50 m below the surface in the vertical direction.
* ``semimajor``: half length of the major axis (m).
* ``semiminor``: half length of the minor axis (m).
* ``dipAngle``: tilt angle of the ellipsoid, i.e. inclination of its x axis (degree).
* ``dipdir_azi``: azimuth, the direction the ellipsoid faces (degree).
* ``rotAngle``: rotation angle of the ellipsoid base (degree).
* ``offset``: offset normal to the DEM slope (m).

Running Scarp
~~~~~~~~~~~~~

From QGIS, open the OpenNHM tool and select ``Scarp (com6)``. Select the input DEM, perimeter shape and coordinate
file, and choose the method. The chosen method must match the attributes created in the coordinate file.

.. Note:: Very large DEMs can lead to long runtimes depending on the machine; keep the input files as small as
   possible, and avoid leaving or interrupting QGIS during the computation.

Alternatively, run from the command line (see :ref:`moduleCom6RockAvalanche:Scarp Config`):

::

    pixi run python runCom6Scarp.py

Scarp parameter sketches
~~~~~~~~~~~~~~~~~~~~~~~~

The following sketches illustrate the correct use of the Scarp parameters.

.. figure:: /_static/com6_ellipsoid_topview.png
    :width: 70%
    :alt: Top view of a rotated but untilted ellipsoid

    Top view of a rotated but untilted ellipsoid (drawn unrotated to avoid distortion). The perimeter boundary does
    not cut anything from the ellipsoid.

.. figure:: /_static/com6_ellipsoid_crosssection.png
    :width: 70%
    :alt: Cross-section of a tilted ellipsoid

    Cross-section of an ellipsoid tilted by 30 degrees but not rotated. The maximum depth refers to the untilted
    ellipsoid, so ``maxdepth = 50 m`` does not necessarily mean a 50 m cut depth; it depends on the tilt angles.

.. figure:: /_static/com6_ellipsoid_offset.png
    :width: 70%
    :alt: Cross-section of an ellipsoid with slope-normal offset

    Cross-section of an ellipsoid with a positive slope-normal offset of the center.

.. figure:: /_static/com6_plane_topview.png
    :width: 70%
    :alt: Top view of the plane method

    Top view of the plane method. P1 and P2 define two points/planes dipping in the same direction. Different dip
    angles combined with the perimeter clip produce a failure body (see next figure). The grey area is the top-view
    failure surface clipped by the perimeter boundary.

.. figure:: /_static/com6_planes_crosssection.png
    :width: 70%
    :alt: Cross-section of the planes

    Cross-section of the planes. The planes defined by P1 and P2 dip in the same direction with different dip
    angles, forming a failure body (orange area).

Scarp Output
~~~~~~~~~~~~

The Scarp step produces a DEM from which the release area has been cut out, plus a raster file with the release
thickness. Both rasters have the same resolution and extent as the original DEM. Negative thicknesses are set to 0 to
avoid errors in the rock avalanche module; the largest (absolute) negative value is logged to indicate how much the
script had to correct.

* elevscarp: Output DGM (ASCII or GeoTIFF), which maps the input DGM minus the calculated scarp. It is saved under
  ``scarpElevation.(asc/tif)`` in ``avalancheDir/Outputs/com6RockAvalanche/scarp``.
* hrelease: File path to the output DGM (ASCII or GeoTIFF), which represents the calculated scarp volumes. It is
  saved under ``scarpHRel.(asc/tif)`` in ``avalancheDir/Outputs/com6RockAvalanche/scarp``.

Scarp Config
~~~~~~~~~~~~

Prepare the config file (scarpCfg.ini):

* copy ``com6RockAvalanche/scarpCfg.ini`` to ``com6RockAvalanche/local_scarpCfg.ini`` and if desired change
  configuration settings
* Input: set ``useShapefiles = True``
* Settings: ``method`` specifies whether the plane or the ellipsoid method is used

If all the data is provided successfully, start the script by running::

    pixi run python runCom6Scarp.py

Scarp common issues
~~~~~~~~~~~~~~~~~~~

* Projections: mismatched projections usually result in an input DEM plus a release file with 0 m thickness
  everywhere. Use one consistent projection for all inputs.
* Attribute names: incorrect attribute names (e.g. wrong case) fail silently. Copy the names from the attribute
  list above.
* Empty attribute fields: every field must contain a value; enter 0 where a field is not applicable.
