com3Hybrid: Hybrid modeling
============================

:py:mod:`com3Hybrid` is a computational module that combines the dense flow avalanche (DFA) simulation
model (com1DFA) and the Alpha-Beta statistical one (com2AB),  taking advantage
of their strength and trying to reduce their weaknesses.
The weakness of the DFA simulation lies in its required inputs, among which is the friction parameter :math:`\mu`.
The disadvantage of the statistical model lies in the necessary path, which is used to extract an
avalanche profile and compute a runout angle.
The idea here is to determine the avalanche path in an automated way by:

- first, running a DFA simulation
- computing a mass averaged path from the results
- use the path to compute the runout angle corresponding to this specific avalanche
- using a coulomb friction method, the friction parameter :math:`\mu` is extracted from the runout
- the resulting :math:`\mu` is used in the input for a new DFA simulation.

This iteration process can be repeated multiple times until "convergence", which means until
the :math:`\mu` or :math:`\alpha` value stops varying.

Input
-----

* raster of the DEM (raster file; .asc or .tif)
* a release feature (shapefile) in ``Inputs/REL``
* Split point (shapefile). This requirement is planned to be automated as well.

Outputs
--------

* The avalanche path specific for the input topography
* the :math:`\mu` value (:math:`\alpha` travel angle) specific for the input topography
* results from the DFA simulation for the input topography
* a combined plot showing the results of the DFA simulation and Alpha-Beta model

To run
-------

* go to ``AvaFrame/avaframe``
* copy ``com3Hybrid/com3HybridCfg.ini`` to ``com3Hybrid/local_com3HybridCfg.ini`` and edit (if not, default values are used)
* put your com1DFA and com2AB settings in  ``com3Hybrid/hybridModel_com1DFACfg.ini`` and ``com3Hybrid/hybridModel_com2ABCfg.ini``
  (these files replace the local ``.ini`` files in ``com1DFA`` and ``com2AB``)
* make sure all the required inputs are available in the avalanche directory
* enter the path to the desired dataset in ``local_avaframeCfg.ini``
* run::

      pixi run python runScripts/runCom3Hybrid.py


Procedure
-----------

1. A first com1DFA simulation is run using the ``com3Hybrid/hybridModel_com1DFACfg.ini``
   configuration and the :math:`\mu` provided in ``com3Hybrid/local_com3HybridCfg.ini``.

2. The mass averaged path is computed from the DFA simulation using the :py:mod:`ana5Utils.DFAPathGeneration`.
   From this we obtain the (x,y,z) and (s,z) coordinates of the avalanche profile (the path is extended towards
   the relase and in the runout). This path is saved to ``avalancheDir/Inputs/LINES`` and
   replaces the old path. A copy is also saved to ``avalancheDir/Outputs/com3Hybrid``.

3. com2AB module is run using the ``com3Hybrid/hybridModel_com2ABCfg.ini`` to compute
   the :math:`\mu = \tan{\alpha}` friction parameter corresponding to this specific avalanche and area.

4. The ``com3Hybrid/hybridModel_com1DFACfg.ini`` is updated with the new :math:`\mu = \tan{\alpha}`
   value and com1DFA simulation is run again.

5. Steps 2. and 3. are repeated to get the new avalanche path and :math:`\mu = \tan{\alpha}` value.

This process is repeated as long as the :math:`\alpha = \arctan{\mu}` value varies from more than
``alphaThreshold`` which you can specify in ``com3Hybrid/local_com3HybridCfg.ini``
(this is a value in degrees) and this for a maximum of ``nIterMax`` iterations.


Configuration parameters
---------------------------------

The parameters for path generation (section ``PATH`` of ``com3Hybrid/local_com3HybridCfg.ini``)
are described in :ref:`moduleAna5Utils:Automated path generation`.

The parameters for the DFA simulation are to be specified in ``com3Hybrid/hybridModel_com1DFACfg.ini``.
This file replaces the ``local_com1DFACfg.ini`` that is usually used to run a com1DFA simulation.

The parameters for the :math:`\alpha-\beta` model are to be specified in ``com3Hybrid/hybridModel_com2ABCfg.ini``.
This file replaces the ``local_com2ABCfg.ini`` that is usually used to run a com2AB simulation.
