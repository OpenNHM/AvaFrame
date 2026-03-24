Complex Usage (Script)
======================

After the installation is outlined, the next sections describe how to get started. The sections configuration and logging
describe the general methods we use, this is helpful to understand how you can change model parameters and similar. 

.. include:: developinstall.rst

--------------------------

.. include:: developinstallwin.rst

--------------------------

.. include:: gettingstarted.rst

---------------------------

.. include:: configuration.rst

-------------------------

.. include:: logging.rst

-------------------------

Example runscripts
------------------

In :py:mod:`runScripts` we provide ready-to-use scripts for different applications of the modules provided within
AvaFrame.


Derive input data
^^^^^^^^^^^^^^^^^^^
- :py:mod:`runScripts.runComputeDist`
- :py:mod:`runScripts.runD2Th`
- :py:mod:`runScripts.runRenaming`

Create a new project
^^^^^^^^^^^^^^^^^^^^^^
- :py:mod:`runScripts.runInitializeProject`


Generate idealized/generic topography data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- :py:mod:`runScripts.runGenerateTopo`
- :py:mod:`runScripts.runGenProjTopoRelease`
- :py:mod:`runScripts.runRotationTest`
- :py:mod:`runScripts.runEnergyLineTest`


Postprocessing
^^^^^^^^^^^^^^^

- :py:mod:`runScripts.runAna3AIMEC`
- :py:mod:`runScripts.runAna3AIMECCompMods`
- :py:mod:`runScripts.runAimecSummaryPlot`
- :py:mod:`runScripts.runStatsExample`
- :py:mod:`runScripts.runStatsExampleSimple`
- :py:mod:`runScripts.runStatsPlots`
- :py:mod:`runScripts.runProbAna`
- :py:mod:`runScripts.runCreateProbCfgsFlowPy`


Visualisation
^^^^^^^^^^^^^^^
- :py:mod:`runScripts.runQuickPlotSimple`
- :py:mod:`runScripts.runQuickPlotOne`
- :py:mod:`runScripts.runPlotTopo`
- :py:mod:`runScripts.runContourPlot`
- :py:mod:`runScripts.runPlotAlongGivenProfile`
- :py:mod:`runScripts.runPlotAreaRefDiffs`
- :py:mod:`runScripts.runPlotContoursFromAsc`
- :py:mod:`runScripts.runPlotProfile`
- :py:mod:`runScripts.runRangeTimeDiagram`
- :py:mod:`runScripts.runThalwegTimeDiagram`
- :py:mod:`runScripts.runFetchPointValues`
- :py:mod:`runScripts.runParticleAnalysisPlots`
- :py:mod:`runScripts.runExportToCsv`


Testing
^^^^^^^^
- :py:mod:`runScripts.runDamBreak`
- :py:mod:`runScripts.runAnalyzeDamBreak`
- :py:mod:`runScripts.runSimilaritySol`
- :py:mod:`runScripts.runAnalyzeSimilaritySol`
- :py:mod:`runScripts.runTestFP`
- :py:mod:`runScripts.runVariationsTestsCom1DFA`
- :py:mod:`runScripts.runWriteDesDict`

------------------------------

Update AvaFrame
------------------------------

To update go to your avaframe repository [YOURDIR]/Avaframe,  pull the latest changes via::

  git pull

Your next `pixi` call then runs the necessary updates. You might want to run `pixi run rebuild` to recompile the cython
parts.