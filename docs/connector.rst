OpenNHM QGis Connector
======================

.. Note::
   The AvaFrame QGis Connector is being superseded by the OpenNHM QGis Connector. For more information,
   see `opennhm.org <https://opennhm.org/post/announcementopennhm/>`_.

The OpenNHM QGis Connector allows QGis users to access certain base workflows directly from QGis. The connector
only provides the interface to functions within the AvaFrame python package, and is developed separately, see the
`OpenNHMQGisConnector github repository <https://github.com/OpenNHM/OpenNHMQGisConnector>`_.
It makes use of the QGis processing plugin, which is included in all current QGis releases.

.. Note::
   ALL (!!) data provided to the functions below HAVE to be in the same projection. It does not matter which
   projection it used as long as it is the same one.

Operational
-----------

.. glossary::
   :sorted:

   Dense Flow (com1)
      Runs dense flow avalanche module com1DFA. For more specific info about the inputs (shapefile attributes etc), see
      :ref:`moduleCom1DFA:Input`. You can select multiple shapefiles for release areas, each will be calculated as one scenario.
      Each shapefile can contain multiple polygons. If you provide entrainment/resistance areas, each scenario will be calculated
      once without them (i.e null simulation) and once with them. It is possible to override the default friction
      parameter set *samosATAuto* to another samosAT friction model calibration.

   Alpha Beta (com2)
      Runs the alpha beta calculation via module com2AB. For more specific info about the inputs (shapefile attributes etc), see
      :ref:`moduleCom2AB:Input`.

   Operational Run (com1 and com2)
      Runs com1DFA and additionally com2AB if profile and splitpoint are set.
      Additional info see description of *Dense Flow (com1)* and *Alpha Beta (com2)*.

   Release area stats (in1, com1)
      Returns info for release area statistics in a csv file. See output of QGis processing for the location of
      the file.

   Layer rename
      Renames com1DFA result layers by adding the values of the given variable (from the configuration file).


Experimental
------------

These either require deeper knowledge about the AvaFrame python package, i.e. configuration files etc., or are not yet
fully tested and might produce unwanted results.

.. glossary::
   :sorted:

   Get default module ini
      Extracts the default configuration ini file for the selected AvaFrame module. The file can then be edited
      and supplied as the expert configuration file when running the corresponding simulation tool.

   Load peak files from directory
      Scans an AvaFrame avalanche directory for peakFiles folders and loads the selected result types
      (ppr, pft, pfv, timeInfo) as raster layers with styles applied where available. Timestep files
      are excluded.

   Generate mass average path (ana5, com1)
      Generates the mass average path from a dense flow simulation via module ana5Utils.

   Probability run (ana4, com1)
      Runs probability simulations via module com1DFA. The release shape HAS TO HAVE a ci95 field containing the
      95 percentile confidence interval (same unit as the release thickness). Multiple scenarios can be provided,
      final map includes variations from all scenarios combined. Release thickness and SamosAT friction mu
      are being varied, 40 variations per scenario.

   Probability analysis for directory (ana4)
      Runs the probability analysis on existing com1DFA results from a given avalanche directory without running any
      simulations. Please provide the base avalanche directory (i.e. the one containing Inputs, Outputs, etc).

   Snow slide (com5)
      Runs snow slide simulations via module com1DFA. For more info see com5SnowSlide section in the documentation.
      The resistance layer is meant for building outlines.

   Rock Avalanche (com6)
      Runs rock avalanche simulations via module com6RockAvalanche, based on com1DFA. For more info see
      com6RockAvalanche section in the documentation. Note that a release thickness raster the same size as the
      DEM is needed.

   Scarp (com6)
      Runs scarp calculation via module com6RockAvalanche.

   Regional input splitting (com7)
      Splits input for regional modelling.

   Regional modelling computation (com7)
      Runs multiple avalanches as setup via regional input splitting.

   MoTVoellmy (com9)
      Runs dense flow simulations via module com9MoTVoellmy. All raster files need to have the same extent as
      the DEM.


Admin
-----

.. glossary::
   :sorted:

   GetVersion
      Displays the current version of the AvaFrame python package (NOT the OpenNHM QGis Connector).

   UpdateAvaFrame
      Updates the AvaFrame python package.
