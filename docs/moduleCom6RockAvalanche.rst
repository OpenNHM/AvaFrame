com6RockAvalanche: Rock Avalanche
=================================

.. Warning:: This is highly experimental and not tested!

The com6RockAvalanche computational module provides an override setting for com1DFA targeting the simulation of rock
avalanches.

Input
-------

The standard inputs required to perform a simulation run using :py:mod:`com1DFA` 
can be found here: :ref:`moduleCom1DFA:Input`.
However there is one main difference: com6RockAvalanche NEEDS a release thickness raster file. This file has to have
the exact same dimensions as the topography file.
There is a run script to perform a rock avalanche com1DFA run: :py:mod:`runCom6RockAvalanche.py`,
and the configuration settings can be found in ``com6RockAvalanche/com6RockAvalancheCfg.ini``.

To run
------

* first go to ``AvaFrame/avaframe``
* copy ``avaframeCfg.ini`` to ``local_avaframeCfg.ini`` and set your desired avalanche directory name
* create an avalanche directory with required input files - for this task you can use :ref:`moduleIn3Utils:Initialize Project`
* copy ``com6RockAvalanche/com6RockAvalancheCfg.ini`` to ``com6RockAvalanche/local_com6RockAvalancheCfg.ini`` and if desired change configuration settings
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


Scarp Calculation
-----------------


* first go to ``AvaFrame/avaframe``
* copy ``avaframeCfg.ini`` to ``local_avaframeCfg.ini`` and set your desired avalanche directory name
* create an avalanche directory - for this task you can use :ref:`moduleIn3Utils:Initialize Project`

Input
~~~~~

* all input files are automatically read from the set avalancheDir. No file paths need to be specified
* elevation: DEM (ASCII), which serves as the basis for calculating the scarps. Must be in avalancheDir/Inputs.
* geometries: a shapefile containing point geometries. These points represent the centers of the ellipsoids or planes.
  The coordinates (x,y) of these points are used. If the plane method is used, the shape file must contain the
  attributes "zseed", "dipdir" and "dipAngle" as float values. If the ellipsoid method is used, the shape file must
  contain the attributes "maxdepth", "semimajor", "semiminor", "dipAngle", "dipdir", "rotAngle", "offset" (see below).
  The file must be located in avalancheDir/Inputs/POINTS and file name must end with “_coordinates”.
  If you are using the QGis Connector, the naming and location of the file is not relevant.

* perimeter: A shapefile that specifies a boundary area. Must be located in avalancheDir/Inputs/POLYGONS and file name
  must end with “_perimeter”. If you are using the QGis Connector, the naming and location of the file is not relevant.

**Attribute meanings:**

* zseed: defines z coordinate of plane center (m)
* dipdir: direction in which the plane/slope is facing (degree)
* dipAngle: steepness/angle of the slope (degree)

* maxdepth: maximum depth of the ellipsoid (m)
* semimajor: length of the major axis (m)
* semiminor: length of the minor axis (m)
* dipAngle: steepness/angle of the ellipsoid tilt (degree)
* dipdir: direction in which the ellipsoid slope is facing (degree)
* rotAngle: rotation angle of the ellipsoid base (degree)
* offset: offset, normal to the DEM slope (m)

Output
~~~~~~

* elevscarp: Output DGM (ASCII or GeoTIFF), which maps the input DGM minus the calculated scarp. Is saved under
  ``scarpElevation.(asc/tif)`` in ``avalancheDir/Outputs/com6RockAvalanche/scarp``.
* hrelease: File path to the output DGM (ASCII or GeoTIFF), which represents the calculated scarp volumes.Is saved
  under ``scarpHRel.(asc/tif)`` in ``avalancheDir/Outputs/com6RockAvalanche/scarp``.

Config
~~~~~~

Prepare the config file (scarpCfg.ini):

* copy ``com6RockAvalanche/scarpCfg.ini`` to ``com6RockAvalanche/local_scarpCfg.ini`` and if desired change
configuration settings

* Input:
    o set useShapefiles = True
* Settings:
    o method: Here you specify whether the plane or the ellipsoid method should be used

If all the data is provided successfully, start the script by running::

    pixi run python runCom6Scarp.py