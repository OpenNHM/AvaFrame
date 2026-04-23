######################################
in1Data: Input data utilities
######################################

Get input data
================

:py:mod:`in1Data.getInput` module provides functions to fetch or generate input
data for avalanche simulations within AvaFrame. The main functions are described
in more detail in the following sections. A detailed description of all the
small helper functions is provided in :py:mod:`in1Data.getInput`


getInputCom1DFA
----------------

:py:func:`in1Data.getInput.getInputCom1DFA` fetches the input data from an
avalanche director required to start avalanche simulations with
:py:mod:`com1DFA`. This data consists of a digital elevation model (DEM), a
release scenario shapefiles, optional entrainment and resistance area
shapefiles and is returned as a dictionary.


computeFromDistribution
==========================

:py:mod:`in1Data.computeFromDistribution` is collection of functions that
facilitates retrieving a sample of values that are distributed following a
uniform or Pert distribution. This is useful if you want to perform avalanche
simulations for a range of release thickness values (or any other parameter),
that should be distributed following a specific distribution. Besides returning
a sample of values following the desired distribution, various plots can be
generated detailing the characteristics of the sample. The required input
parameters can be found in the respective configuration file
``in1Data.computeFromDistributionCfg.ini``. Detailed information on the
functions can be found in :py:mod:`in1Data.computeFromDistribution`.

To run
-------

An example of how a parameter set distributed following a uniform or Pert
distribution can be generated, is provided by :py:mod:`runScripts/runComputeDist.py`.

* first go to ``AvaFrame/avaframe``
* copy ``in1Data/computeFromDistributionCfg.ini`` to
  ``in1Data/local_computeFromDistributionCfg.ini`` (if not, the standard settings
  are used)
* adjust path to the desired ``NameOfAvalanche/`` folder in your local copy of
  ``avaframeCfg.ini``
* run::

      pixi run python runScripts/runComputeDist.py
