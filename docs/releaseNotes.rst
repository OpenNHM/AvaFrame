Release Notes
=============

Please refer to `Github releases <https://github.com/OpenNHM/AvaFrame/releases>`_ for current release notes.

1.3 (12. Okt 2022)
------------------

Main change is the change of logic for secondary release areas. This was done to be able
to expose this functions in the AvaFrameConnecter. So it is now possible to include secondary 
release areas in dense flow simulations done via QGis. 

ENHANCEMENTS

- Change logic for secondary release areas:
  - Check for release and secondary release shapefiles
  - If only release available -> just run release
  - If both release and secondary release are available -> run with secondary release 
  UNLESS secRelArea = False 

  - ALL scenarios in release get the secondary release areas!
- Add  rotational energy line test: helps to checks eg. for numerical grid independence
- Update ini file procedure for the energy line test and the rotation test
- Additional statistical plots
- New three panel plot of tt-diagram (plus animation)
- Add variation option for thickness settings and probrun based on normal distribution derived from ci and mean
- Add filtering option to aimec
- Add scenario name to configuration to be used for plotting example #757 
- Add surface parallel coordinate computation to Aimec
- Improve operational installation instructions
- Add german version of operational installation

FIX
- Contour legend bug with matplotlib 3.5.2
- Update installation instructions; fixes #764
- Bugs in deriving variation
- Remeshing issue that lead to standard test differences (originated in commit 419c11f)
- No calls to matplotlib for plotting purposes in com1DFA
- Removes multiple printouts of config during run in, e.g. com1DFA
- CompareConfig always honours the toPrint flag

Contributors:

Code: core team 


1.2 (07. July 2022)
-------------------

Main changes are the automatic split point generation and optional computation of fields inside the 
calculation loop. Furthermore renaming functions used for the QGis AvaFrameconnector are included.

ENHANCEMENTS

- Add function for renaming simulations, i.e. adding info to the simName. Used for AvaFrameConnector
- Split cfgUtils: Utils contains all reading and writing, cfgHandling contains functions that do 
  something with the cfgInfos
- Make computation of ppr, pta, P, TA and pke optional within the calculation loop . Only compute them 
  if they are required as output
- Add automatic split point generation:
  - First run a DFA simulation, either using the runDFAModule flag in runComputeDFAPath.py or yourself 
  (do not forget that you need to saves particles or FD, FM, FV for multiple time steps)

  - What DFA parameters to use is not yet clear. I would use the BenMoussa time stepping and particles 
    parameters, sphOption 2, explicit friction option 1, some artificial viscosity, and maybe activate 
    curvature (with all this and a mu=0.42, I get ok results)

  - Then runComputeDFAPath computes the mass average path, extends it and defines a splitPoint

FIX

- Particle exiting domain was not working properly with the new cython flag (wraparound boundcheck...)
- correct hybrid Path ini file 
- Update fields is too slow #455
- Automatic split point or beta point finding #706
- Add pfe to output files #694
- Make cflTime and sphKernelRadius a common option #668
- Change benchmarks not read particles from file (particles files still exist)
- Add consistency check for com1DFA configuration
- Fix problems with filtering of simulations
- Aimec comparison checks
- Add missing mass files to benchmarks


Contributors:

Code: core team 



1.1 (19. May 2022)
-------------------

The benchmark and thickness release. There are two main changes:

- The benchmarks (i.e. reference results) are updated and originate from com1DFA. 
  Previously these were produced by com1DFAOrig.  
  (Attached zip file contains the standardTest report for the switch)
- All references to *depth* are now switched to *thickness*. This is done to be more consistent
  and precise. It also means result types switch from *pfd / fd* (peak flow depth / flow depth) to 
  *pft / ft* (peak flow thickness / flow thickness). Note that this is a *naming* change; nothing 
  changed regarding the computation!

ENHANCEMENTS

- benchmarks originate from com1DFA
- change all depth variables to thickness
- change the order of simHash within the result names; fixes Move simHash in filename #690
- path finding added; see issue #610. This will be fully introduced in version 1.2, including
  automatic split point generation
  - refactor path computation functions
  - allow computing path from particles or fields (if *from fields*: needs the FM=FlowMass)
  - runscript to compute a path from com1DFA results (requires that one saves some time steps)
- automate the benchmark updating process
- improve energy line plot
- set deleteOutput to False in runOperational; addresses User Feedback (CT) #715. This means
  it is possible to reuse the same directory in the QGis Connector, adding results to existing 
  ones
- add real area to aimec analysis #695
- update hybrid and energy test
- add com3Hybrid documentation #618

FIX

- hybrid model #611
- refactor com2AB for clarity and readability #446
- address savefigFormat TODO in outAB #560
- only one makeDomainTransfo #700

Contributors:

Code: core team 


1.0.1 (20. April 2022)
----------------------

FIX

- #712 , missing init files


1.0 (6. April 2022)
-------------------

ENHANCEMENTS

- adds avaframe version to log
- appends date to logfile name
- update similarity solution plots
- re-add codecov
- add in addition to vary thickness values if read from shp - not just in percent but also in absolute value
- *ana1Test* energy line test
- *documentation* info on visualisation options (Paraview)
- update the pytest github action to version 3.9
- add ana5Hybrid, module that combines statistical module com2AB with the DFA module com1DFA
- new requirement shapely
- add release area info to benchmark ini files
- make AB optional in runOperational (related to QGis AvaFrameConnector)
- updates to ana1Tests 
- hillshade and contours for peak plots
- documentation improvements
- reorder installation and get started documentation
- create distance-time diagrams of ava simulations from a reference point showing the avalanche front and the average values of a chosen result parameter (e.g. flow depth, flow velocity)
- *com1DFA* new flags/system for release thickness and entrainment thickness settings and options
- *com1DFA* add travel angle computation
- *com1DFA* release thickness percent variation option 
- *com1DFA* unique simHash including info on release scenario with correct thickness
- *com1DFA* removed return parameters from com1DFAMain
- *com1DFA* update benchmark ini files 
- *com1DFA* documentation for bottom friction and operator splitting
- *com1DFA* option to redistribute particles after initialisation in order to reduce SPH force
- *com1DFA* Implement Ata Viscosity and an SPH flow thickness computation
- *com1DFA* new splitting/merging of particles
- *com1DFA* enable to initialize particles with a non constant flow thickness
- *com1DFA* remove unmaintained leap frog time stepping scheme 
- *com1DFA* new parameter: cleanDEMremeshed
- *com1DFA* add simulation DEM if remeshed to different cellSize #670
- *com1DFA* check for remeshed DEM, save remeshed DEM #675
- *com1DFA* enable to chose dem asc file for com1DFA #658
- *com1DFA* new parameter: cleanDEMremeshed
- *com1DFA* add simulation DEM if remeshed to different cellSize #670
- *com1DFA* check for remeshed DEM, save remeshed DEM #675
- *com1DFA* enable to chose dem asc file for com1DFA #658
- *ana4Prob* add example for performing a parameter variation run with prob analysis
- *ana4Prob* use default com module setup or specified in local - add variation for prob run
- *ana4Prob* perform analysis using probabilityConfiguration in runScript

FIX

- errors in com2AB documentation
- tcpu field in com1DFA
- ordering of dict for analysisAdd 
- pytest errors related to matplotlib colors and legend
- particle splitting issue
- fix pypi related issues (pypi needs clean version tags)
- quickfix for shapely vs QGis problem with the AvaFrameConnector, see Linux QGis 3.24 crashes on Connector activation QGisAF#9
- move Release-version file for packaged releases
- change naming of log file: fix #689
- (hacky) solution to handle apostrophes in filenames #683
- allow choosing a tau0 in samosAT friction type (so far, tau0 was fixed and equal to 0)
- add tau0 to SamosAT friction #702
- address the wrong logName in runscript
- error running simulations one day after #701
- error on python 3.7 and QGis 3.12 #705
- python3-dev package required. #699

Contributors:

Code: core team, M. v. Busse (UIBK), M. Winkler (UIBK)
Code review tt-diagram: A. Köhler (BFW)


v0.6 (24. September 2021)
-------------------------

ENHANCEMENTS

- installation via pypi (pip install)
- connection to QGis (via plugin manager) 
- function to interpolate data on mesh of different cellSize using splinesp
- testing via pytest extended
- more pathlib usage 
- ASCII header is read as dict
- documentation contains FAQ page
- reworked installation instructions
- cleaner test reports/inis
- github action to deploy to pypi
- switch to codeclimate
- use consistent thickness attributes (shapefiles etc)
- *com1DFA* any resolution is possible now 
- *com1DFA* split the getWeight function in two: first get cell and then get weights. 
- *com1DFA* avoid possibility of segfault because particles exit too quickly the domain.
- *com1DFA* additional particles info: unique identifier for each particle and parent particles
- *com1DFA* central time step calling
- *com1DFA* additional options to set mass per particle directly or via release thickness
- *com1DFA* interpolation option for initialization of Hpart 
- *com1DFA* read entrainment thickness
- *ana3AIMEC* override option for raster cellsize 
- *ana3AIMEC* mass analysis plot even if more than 2 simulations

FIX

- getTimeIndex problem if dtSave < actual dt
- better way to remove particles
- track particles exiting the computation domain
- fix issue save particles
- read aimec grid info from result files and not from dem
- add reasonString to removal of particles
- fix correct module name in AIMEC 
- com2AB write out to shp 

Contributors:

- **Code: core team**


v0.5 (13. July 2021)
--------------------

ENHANCEMENTS

- filtering functions for com1DFA simulations
- flag to disable print at CFG reading
- new colormaps for ppr, pft, pfv
- *com1DFA* option to add friction explicitly using the method described in #273 .
- *com1DFA* Resistance force is  added explicitly.
- *com1DFA* New method to get the release area
- *com2AB* function to write results to shapefile
- *ana3AIMEC* warning for empty runout zone
- *ana3AIMEC* enable simulation ordering/filtering

FIX

- beta angle issue i.e. distance below angle
- correct removal of particles 
- AIMEC produces warning on empty runout area
- adapt quickplot to new naming scheme

Contributors:

- **Code: core team**
- **Colormaps: C.Tollinger**

DOI for this release:

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.5094509.svg
   :target: https://doi.org/10.5281/zenodo.5094509


v0.4.1 (9. June 2021)
---------------------

Minor release to fix issue with zenodo

v0.4 (8. June 2021)
-------------------

The switch release

This is a big release: we switched our dense flow module 'com1DFA' to the python
version. This means that you know get to use the python version as default.
However, the original version is still available in the module com1DFAOrig. The
full documentation for the python com1DFA version as well as updated benchmarks
will be released in the next version.

Module com2AB (AlphaBeta) recieved an update allowing for custom parameters.

Simulation naming and identification also recieved a major change, we introduced
unique ID's for each individual configuration.

Contributors:

- **Code: core team**


v0.3 (26. April 2021)
---------------------

The AIMEC and Windows release

This release brings an AIMEC refactor, plenty of improvements related to the
test cases and Windows capabilities. 3 new idealised/generic test case are 
included: flat plane, inclined slope and pyramid.

Com1DFAPy recieved a lot of advancement as well, e.g. parts of it are converted
to cython to speed up computation times.  

Documentation regarding our testing is included, see more at the
`testing <https://docs.avaframe.org/en/latest/testing.html>`_ page. 

Contributors:

- **Code: core team**

DOI for this release:

.. image:: https://zenodo.org/badge/281922740.svg
   :target: https://zenodo.org/badge/latestdoi/281922740


v0.2 (28. Dezember 2020)
------------------------

The testing release

Version 0.2 includes the first real world avalanches. It provides data for 6
avalanches, including topographies, release areas and benchmark results.
To know more about our data sources, head over to
`our data sources documentation
<https://docs.avaframe.org/en/latest/dataSources.html>`_.
The existing test cases also recieved some updates by including multiple release
areas and multiple scenarios per avalanche.  

This release also is the first to include `API documentation
<https://docs.avaframe.org/en/latest/api.html>`_ for our modules and functions.
However not all functions are included yet.

Contributors:

- **Data: M.Granig, C. Tollinger**
- **Data: Land Tirol**
- **Code: core team**


v0.1 (06 November 2020)
-----------------------

Initial release. 

This release is the result of several months of development.

Several people have contributed to this release, either directly or through code
that was used as reference/basis:

- **Peter Sampl**, code base for com1DFA
- **Jan-Thomas Fischer**, code base AIMEC, code related to com1DFA
- **Michael Neuhauser**, code for helper and transformation utilities, com1DFA
- **Andreas Kofler**, code related to AIMEC and com1DFA 

and the core team:

- **Anna Wirbel**
- **Matthias Tonnel**
- **Felix Oesterle**

