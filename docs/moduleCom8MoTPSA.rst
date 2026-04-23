com8MoTPSA: NGI MoT-PSA
==========================

.. Note:: This module (as well as this documentation) is currently under development and highly experimental!
     The parameter settings are completely untested and unchecked. Do not expect sensible results with the included
     settings!

:py:mod:`com8MoTPSA` allows to run the MoT-PSA development by NGI (TODO proper cites). It currently does NOT include the
executable needed to run the model. Once NGI open-sources MoT-PSA we will update our installation instructions.

Input
-----
TODO

Outputs
--------
TODO

To run
-------

* go to ``AvaFrame/avaframe/com8MoTPSA``
* copy ``com8MoTPSA/com8MoTPSACfg.ini`` to ``com8MoTPSA/local_com8MoTPSACfg.ini`` and edit (if not, default values are used)
* make sure all the required inputs are available in the avalanche directory
* enter the path to the desired dataset in ``local_avaframeCfg.ini``
* run::

      pixi run python runCom8MoTPSA.py


Theory
------

TODO


Configuration parameters
---------------------------------

TODO