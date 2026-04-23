com9MoTVoellmy: NGI MoT-Voellmy
===============================

.. Note:: This module (as well as this documentation) is currently under development and highly experimental!
     The parameter settings are completely untested and unchecked. Do not expect sensible results with the included
     settings!

:py:mod:`com9MoTVoellmy` allows to run the MoT-Voellmy code by Norwegian-Geotechnical-Institute
(`MoT-Voellmy github repository <https://github.com/norwegian-geotechnical-institute/MoT-Voellmy>`_).

More information and background can be found there as
well `MoT-Voellmy Documentation <https://github.com/norwegian-geotechnical-institute/MoT-Voellmy/tree/main/Documentation>`_

**This documentation covers issues related directly to running MoT-Voellmy with the help of AvaFrame**

There are 2 ways of running, please choose below.

MoT Voellmy via QGis
--------------------

Installation on Windows:
^^^^^^^^^^^^^^^^^^^^^^^^

#. Open the OSGeo4WShell that matches your QGis installation
#. Run::

        pip install avaframe==2.0a3

#. Start QGis and go to `Plugins` -> `Manage and install plugins`
#. Choose `All`, search for `AvaFrame` and install/update
#. You find the `AvaFrame` toolset in the `Processing Toolbox`
#. Go to `NGI_experimental` and start `MoTVoellmy (com9)`


Input
^^^^^

For additional info, see also `NGI Documentation <https://github.com/norwegian-geotechnical-institute/MoT-Voellmy/blob/main/Documentation/MoT-Voellmy_use.md#raster-input-data>`_.

Required is

* digital elevation model as raster file with `ESRI grid format <https://desktop.arcgis.com/en/arcmap/10.3/manage-data/raster-and-images/esri-ascii-raster-format.htm>`_.

ONE of the following:

* release area scenario as (multi-) polygon shapefile

  - the release area polygon must not contain any "holes" or inner rings
  - the release area name should not contain an underscore.
  - recommended attributes are *name*, *thickness* (see :ref:`moduleCom1DFA:Release-, entrainment thickness settings`)
  - ALL features within one shapefile are released at the same time (and interact), this is what we refer to as *scenario*
  - if you want to simulate different scenarios with the same features, you have to copy them to separate shapefiles

* release area scenario as raster file

  - same specifications as the DEM (extent, cellsize)

And the following optional raster files, in the same specification as the DEM (extent, cellsize):

* mu / Dry friction coefficient. If this is given, k needs to be set as well.
* k / Turbulent friction coefficient. If this is given, mu needs to be set as well.
* b0 / Erodible snow depth. If this is given, tau_c needs to be set as well.
* tau_c / Snow shear strength. If this is given, b0 needs to be set as well.
* nd / Forest density. If this is given, bhd needs to be set as well.
* bhd / Tree diameter. If this is given, nd needs to be set as well.

The `Destination Folder` should be empty!



Outputs
^^^^^^^

After you ran the simulation, the peak fields will be loaded in QGis and the `Destination Folder` will contain, results,
plots, and configuration files.


MoT Voellmy via the script variant
----------------------------------

To run
^^^^^

* go to ``AvaFrame/avaframe/com9MoTVoellmy``
* copy ``com9MoTVoellmy/com9MoTVoellmyCfg.ini`` to ``com9MoTVoellmy/local_com9MoTVoellmyCfg.ini`` and edit (if not, default values are used)
* make sure all the required inputs are available in the avalanche directory
* enter the path to the desired dataset in ``local_avaframeCfg.ini``
* run::

      pixi run python runCom9MoTVoellmy.py

