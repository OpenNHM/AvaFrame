com5SnowSlide: Snow slide
=========================

The com5SnowSlide computational module provides the option to add an elastic cohesion force between particles  
based on com1DFA. This can be used for example to simulate small snow slides. 

The cohesion between particles is modeled using elastic bonds with maximum strain rate.
This means that the cohesion force :math:`\mathbf{F}_{lk}` exerted by :math:`l` on :math:`k` is expressed by:

.. math::

  \mathbf{F}_{lk} = -\mathbf{F}_{kl} =  \frac{L-L_0}{L_0} A_{kl} E \mathbf{e_{kl}}

Where :math:`L` and :math:`L_0` are the distances between the particles :math:`k` and :math:`l` at a time t and at rest
(initial time step), :math:`E` is the elastic modulus of the material, :math:`A_{kl}` is the contact area between
particles :math:`k` and :math:`l` and :math:`\mathbf{e_{kl}}` is the unit vector


.. math::

  \epsilon = \frac{L-L_0}{L_0}

represents the strain (:math:`\epsilon` is positive in the case of elongation and negative in compression).
If the strain exceeds a critical value :math:`\epsilon_c` the bond between the particles :math:`k` and :math:`l`
breaks.

The contact area :math:`A_{kl}` between particles :math:`k` and :math:`l` reads:

.. math::

  A_{kl} = h_{kl} d_{kl} = \frac{h_k + h_l}{2} \frac{L}{\sqrt{3}}

:math:`h_{kl}` represents the height of the contact surface and :math:`d_{kl}` represents the
length of the contact surface in the horizontal plane assuming that each particle has 6 neighbours.



Input
-------

The standard inputs required to perform a simulation run using :py:mod:`com1DFA` 
can be found here: :ref:`moduleCom1DFA:Input`.
In contrast to :py:mod:`com1DFA`, in :py:mod:`com5SnowSlide`, resistance areas represent areas of rigid obstacles such
as buildings.
There is a run script to perform a snow slide com1DFA run: :py:mod:`runCom5SnowSlide.py`,
and the configuration settings can be found in ``com5SnowSlide/com5SnowSlideCfg.ini``.
The snow slide-specific parameters are:

  * snowSlide is activated by setting ``snowSlide`` to 1

  * the maximum strain before breaking of the bond ``cohesionMaxStrain``

  * the Young modulus in N/m² ``cohesiveSurfaceTension``

  * resistance areas: extreme values of cResH to represent buildings and detrainment set to False

However, also several other parameters, for example the particle initialization method,
friction model and parameters, are changed compared to the default configuration of :py:mod:`com1DFA` and listed
in the ``com5SnowSlide/com5SnowSlideCfg.ini`` in the com1DFA_override section.

Please note that the provided default setup for snow slide calculation is defined with the following 
premises:

  * slope gradient in the release area from approx. 28° to approx. 50°.

  * a return level for hazard mapping of 150-yr of snow depth as the initial value.

  * movement as an intact unit without turbulence
    
  * height difference from approx. 30 m to max. 60-80 m

  * valid for altitude levels > 500 m and < 1,500 m sea level

  * resistance areas represent buildings 


Initialization of bonds
-------------------------

If the elastic cohesion (``snowSlide`` set to 1) is activated, the bonds between particles are initialized.
The construction of the bonds is done by building a triangular mesh on the particles
(used as a point cloud) and the Delaunay `triangulation function <https://matplotlib.org/stable/api/tri_api.html#matplotlib-tri>`_
from matplotlib.
The length :math:`L_0` of the bonds at rest (initial distance between two bonded particles) is also initialized here.


.. Note:: It is recommended to use the triangular (``triangular``) initialization option (:ref:`com1DFAAlgorithm:Initialize particles`)
          rather than the random or the uniform one in the case where cohesion is activated.


To run
------

* first go to ``AvaFrame/avaframe``
* copy ``avaframeCfg.ini`` to ``local_avaframeCfg.ini`` and set your desired avalanche directory name
* create an avalanche directory with required input files - for this task you can use :ref:`moduleIn3Utils:Initialize Project`
* copy ``com5SnowSlide/com5SnowSlideCfg.ini`` to ``com5SnowSlide/local_com5SnowSlideCfg.ini`` and if desired change configuration settings
* if you are on a develop installation, make sure you have an updated compilation, see :ref:`installation:Update AvaFrame`
* run:
  ::

    python3 runCom5SnowSlide.py
