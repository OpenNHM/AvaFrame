com1DFA Algorithm and workflow
==============================

Algorithm graph
----------------

The following graph describes the Dense Flow Avalanche simulation workflow
(the different nodes are clickable and link to the detailed documentation)

.. graphviz:: com1DFAAlgorithmGraph.dot


Initialization:
-----------------
At the beginning of the simulation, the avalanche folder and the configuration
are read (:ref:`configuration:Configuration`).
Input data is fetched according to the chosen configuration.
Mesh, particles and fields are subsequently initialized.

Initialize Mesh
~~~~~~~~~~~~~~~~~

Read DEM raster file provided in the Input folder (only one DEM raster file allowed; .asc or .tif).
If the DEM cell size is different from the :``meshCellSize`` specified in the configuration
from more than ``meshCellSizeThreshold`` [m] the DEM is remeshed (:py:func:`in3Trans.geoTrans.remesh`).

Prepare DEM for simulation, compute surface normals vector field, cell area (:ref:`DFAnumerics:Mesh`).

This is done in the :py:func:`com1DFA.com1DFA.initializeMesh` function.

Go back to :ref:`com1DFAAlgorithm:Algorithm graph`

Initialize release, entrainment and resistance areas
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Read and check shapefiles according to the configuration (check consistency between
what is required by the configuration file and what is available in the ``Inputs`` folder).
Convert shapefile features (polygons) to rasters (:py:func:`in3Utils.geoTrans.prepareArea`).
Check consistency of rasters according to the following rules:

  - multiple release features in the release and secondary release shapefiles
    are allowed but they should not overlap. If they do, simulation terminates with
    an error message.

  - secondary release and entrainment rasters should no overlap between each other or with the
    main release. If they do, the overlapping part is removed. Order of priority is: main
    release, secondary release, entrainment area.

  - if the release is time dependent, timesteps and respective release thickness and initial velocity
    are read (from csv file) and checked for plausibility in :py:func:`com1DFA.debrisFunctions.checkTimeDepRelease`.

Go back to :ref:`com1DFAAlgorithm:Algorithm graph`

Initialize Dam
~~~~~~~~~~~~~~~~~~~~~

If the dam option is activated (which is the case in the default configuration ``dam`` is ``True``)
AND the dam input shapefile is provided, the dam is initialized and will be take into account for in the DFA computation.

Initialize particles
~~~~~~~~~~~~~~~~~~~~~

Particles are initialized according to the release raster extracted from the
release shapefile and the mass per particle determination method
(``massPerParticleDeterminationMethod``) specified in the configuration.
The mass per particle determination method can be chosen between:

- MPPDIR= mass per particle direct. The ``massPerPart`` value is taken from
  the configuration and is the same for all cells.

- MPPDH= mass per particles through release thickness. The ``massPerPart`` value is computed
  using the release thickness per particle ``deltaTh`` value given in the configuration and the area of
  the release mesh cell: :math:`\mbox{massPerPart} = \rho\times \mbox{cellArea} \times\mbox{deltaTh}`.

- MPPKR= mass per particles through number of particles per kernel radius.
  There is no ``massPerPart`` since it can vary from one cell to another
  depending on the release thickness of the cells. The aim of this method is
  to ensure a constant density of particles within the snow domain (``nPPK``
  particles per kernel radius is the target value). This is related to the
  SPH method used for computing the flow thickness gradient. It requires a
  sufficient number of particles to properly approximate the flow thickness
  gradient. It makes the most sense to combine the MPPKR particle
  initialization method with the `splitOption` 1. In this combination, the
  particles will be merged or split to keep a constant density of particles
  per kernel radius (:ref:`DFAnumerics:Particle splitting and merging`).


.. Note::  If MPPDIR is used, consider adapting the mass per particle value when changing the mesh cell size from the default.
           This is important because, when using MPPDIR, the total number of particles is independent of the cell size. Hence,
           reducing the cell size results in less particles per cell, whereas when using MPPDH,
           the number of particles per cell is fixed (considering the respective release thickness and deltaTh value).
           Reducing the cell size will increase the total number of particles but not the number of
           particles per cell. Finally, using the MPPKR method, the number of particles per cell is independent from
           both cell size and release thickness (``nPPK`` particles per kernel radius is the target value).

The number of particles placed in each release cell ``nPartPerCell`` is computed according to the ``massPerPart`` or ``nPPK`` depending
on the ``massPerParticleDeterminationMethod`` chosen and the area and/or release thickness of the cell.
The number should be an integer meaning that the float is rounded up or down with a probability corresponding to the
decimal part (i.e. 5.7 will be rounded to 6 with a probability of 0.7 and 5 with a probability of 0.3).
This ensures a better match with the desired ``massPerPart`` value.

There are then different ways to place the particles in the cells. This is decided by the
``initPartDistType`` parameter in the configuration file:

    - ``random`` initialization:
      The ``nPartPerCell`` particles are placed randomly within each release cell.
      This initialization method allows steps of ones in the choice of ``nPartPerCell``, which enables to
      avoid large jumps in the mass of the particles or the number of particles per Cell between one release cell to another.
      Due to the random initialization process, some particles cluster can appear. For operational applications,
      this does not seem to have a significant impact since the particles will redistribute in the first
      few time steps due to the pressure gradient. For some specific research applications (e.g. the dam break test),
      clusters might disturb the results. In this case, applying an initialization reprocessing
      can help (see  ``iniStep`` in the configuration file).
      The random number generator is controlled by a random seed which ensures the possibility to
      reproduce the DFA results.

    - ``uniform`` initialization:
      The release (square) cell is divided into (square) subcells (as many as ``nPartPerCell``). The particles
      are placed in the center of each subcell. This method allows only steps of 1, 4, 9, 16, ... due to the
      square subdivision of the cells. This can lead, in some cases, to significant variations in the number of particles
      per cell or in the mass of the particles. This can then lead to spurious numerical artifacts.
      The particles are aligned with the grid, which can also lead to some
      spurious numerical artifacts.

    - ``semirandom`` initialization:
      This method uses the uniform division of the release cell into subcells but the particles are
      placed randomly within each subcell. This method has the same disadvantage as the ``uniform`` method,
      it allows only steps of 1, 4, 9, 16, ... due to the square subdivision of the cells.
      This can lead, in some cases, to significant variations in the number of particles per cell or in the
      mass of the particles. But due to the random position of particles in the subcells
      particles are not aligned anymore with the grid.

    - ``triangular`` initialization:
      The particles are initialized along a regular triangular mesh within the release area.
      This allows a regular distribution of particles with steps of one (like in the case of ``random`` initialization).
      The disadvantages are that the same triangle size is used within the whole release leading in some cases to
      significant differences in the mass of the particles. The same comment applies if multiple release features are used.
      Finally, this method only works for constant release thickness.

      .. Note:: This initialization option is meant to be used when the snow slide option is activated
                (``cohesion`` activated).
                The behavior in the standard case (``cohesion`` deactivated) has not been tested.

If the release is time dependent, particles are initialized in provided timesteps. If an initial velocity is provided,
the particles have this initial velocity (magnitude) in direction of the steepest descent,
it is computed in the following function: :py:func:`com1DFA.DFAfunctionsCython.updateInitialVelocity`.


Particle properties
^^^^^^^^^^^^^^^^^^^^

Other particles properties are also initialized here:
    - ``x``, ``y``, ``z`` - coordinates

    - ``m`` - mass of particle [kg]

    - ``h`` - flow thickness [m]

    - ``ux``, ``uy``, ``uz``, ``velocityMag``- velocity components and magnitude [ms-1]

    - ``uAcc`` - approximation for particle acceleration between each computational time step (simply computed by (velocityMagnitude_t0 - velocityMagnitude_t1) / dt)

    - ``trajectoryLengthXY`` - traveled length of particle accumulated over time in xy plane

    - ``trajectoryLengthXYCor`` - trajectoryLengthXY corrected with the angle difference between the slope and the normal

    - ``trajectoryLengthXYZ`` - traveled length of a particle accumulated over time in xyz

    - ``trajectoryAngle`` - travel angle computed using arctan((z0-z)/trajectoryLengthXY)

    - ``ID, parentID`` - particle IDs and parentID required if splitting, merging

    - ``t``- corresponding time step
    
    - ``dmDet`` - detrained mass of particle [kg]

    - ``dmEnt`` - entrained mass of particle [kg]

    - ``stoppedParticles`` - dictionary with particles (containing x-, y-coordinates, mass and ID) that are stopped (velocity is zero) and deleted from the particles in each time step (only if ``adaptSfcStopped`` is set to ``1`` in the configuration file)

    - additional properties used during computation or summary properties for all particles include: ``nPart``, ``massPerPart``, ``nPPK``, ``mTot``, ``stoppCriteria``, ``kineticEne``, ``potentialEne``, ``peakKinEne``, ``peakMassFlowing``, ``simName``, ``xllcenter``, ``yllcenter``, ``ID``, ``nID``, ``parentID``, ``t``, ``inCellDEM``,  ``indXDEM``, ``indYDEM``, ``indPartInCell``, ``partInCell``, ``secondaryReleaseInfo``,
      ``iterate``, ``idFixed``, ``peakForceSPH``, ``forceSPHIni``, ``totalEnthalpy``, ``velocityMag``, ``nExitedParticles``, ``tPlot``, ``stoppedParticles``, ``massInitialized``, ``massEntrained``,
      ``massDetrained``, ``massStopped``

For more details, see :py:func:`com1DFA.com1DFA.initializeParticles`.

Go back to :ref:`com1DFAAlgorithm:Algorithm graph`

Initialize fields
~~~~~~~~~~~~~~~~~
All fields (mesh values defined as a raster) are initialized. Flow velocity, pressure, peak flow velocity and peak pressures
are set to zero. Flow thickness and peak flow thickness are set according to the initial particle distribution.
See :py:func:`com1DFA.com1DFA.initializeFields`

Go back to :ref:`com1DFAAlgorithm:Algorithm graph`


Time scheme and iterations:
------------------------------
The mass and momentum equations described in :ref:`theoryCom1DFA:Governing Equations for the Dense Flow Avalanche` are solved numerically
in time using an operator splitting method. The different forces involved are sequentially added to update the velocity
(see :ref:`DFAnumerics:Adding forces`).
Position is then updated using a centered Euler scheme.
The time step can either be fixed or dynamically computed using the Courant–Friedrichs–Lewy (CFL) condition
(in the second case one must set ``cflTimeStepping`` to ``True`` and set the desired CFL coefficient).

Go back to :ref:`com1DFAAlgorithm:Algorithm graph`


Compute Forces:
-----------------
This section gives an overview of the different steps to compute the forces acting on the snow particles.
Those forces are separated in several terms: A gravity driving fore (:math:`F_{drive}`), a friction force
(:math:`F_{fric}`), an entrainment force (related to the entrained mass of snow) and an artificial viscous force.
Those forces are computed by the two following functions
:py:func:`com1DFA.DFAfunctionsCython.computeForceC` and :py:func:`com1DFA.DFAfunctionsCython.computeForceSPHC`.

Go back to :ref:`com1DFAAlgorithm:Algorithm graph`

Artificial viscosity
~~~~~~~~~~~~~~~~~~~~~~

This viscous friction force is artificially added to the numerical computation.
The aim of this force is to stabilize the simulation and prevent neighbor particles
to have too significantly different velocities. Physically, this force also makes sense and corresponds
to some second order forces that were neglected (lateral shear stress) as explained in
:ref:`DFAnumerics:Artificial viscosity`.
This force is controlled by the ``subgridMixingFactor`` in the configuration file.
Setting this parameter to 0 deactivates the artificial viscosity term.
The default value (set to 100) does not have any physical foundation yet. Future work
will help defining this parameter in a more physical way. Remember that the artificial viscosity is dependent on the grid cell size.

The velocity is updated immediately after using an explicit/implicit formulation.

Go back to :ref:`com1DFAAlgorithm:Algorithm graph`



Compute friction forces
~~~~~~~~~~~~~~~~~~~~~~~~

The friction force encompasses all forces that oppose the motion of the particles.
One of those forces is the bottom shear force. The other is an optional resistance force.
Both components are added to the :math:`F_{fric}` force term.

Bottom shear force
~~~~~~~~~~~~~~~~~~~~~~~~

This force accounts for the friction between the snow particles and the bottom surface.
The expression of the bottom shear stress depends on the friction model chosen but can be written in the
following general form, :math:`\tau^{(b)}_i = f(\sigma^{(b)},\overline{u},\overline{h},\rho_0,t,\mathbf{x})`.
The friction model is set by the ``frictModel`` value and the corresponding parameters can be set in the configuration file.
More details about the different friction models are given in :ref:`theoryCom1DFA:Friction model`.
Be aware that the normal stress on the bottom surface :math:`\sigma^{(b)}` is composed of the normal component of the
gravity force and the curvature acceleration term as shown in :eq:`sigmab`. It is possible
to deactivate the curvature acceleration component of the shear stress by setting the
``curvAcceleration`` coefficient to 0 in the configuration file.

Go back to :ref:`com1DFAAlgorithm:Algorithm graph`


Added resistance force
~~~~~~~~~~~~~~~~~~~~~~~~

An additional friction force called resistance can be added. This force aims to model the added
resistance due to the specificity of the terrain on which the avalanche evolves, for example
due to forests. To add a resistance force, one must provide a resistance shapefile in the ``Inputs/RES``
folder and switch the ``simType`` to ``res``, ``entres`` or ``available`` to take this resistance area into account.
Then, during the simulation, all particles flowing through this resistance area will undergo an
extra resistance force. More details about how this force is computed and the different parameters chosen
are found in :ref:`Resistance <theoryCom1DFA:Resistance:>`.


Go back to :ref:`com1DFAAlgorithm:Algorithm graph`


Compute driving force
~~~~~~~~~~~~~~~~~~~~~~~~~~~

This force takes into account the gravity force, which is the driving force of the snow motion.
The expression of this force is rather simple, it represents the tangential (tangent to the surface) part of the gravity force
(the normal part of the force is accounted for in the friction term).


Go back to :ref:`com1DFAAlgorithm:Algorithm graph`



Take entrainment into account
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Snow entrainment can be added to the simulation. One must provide an entrainment shapefile
in ``Inputs/ENT`` and set the ``simType`` to ``ent``, ``entres`` or ``available``
(see :ref:`com1DFAAlgorithm:Initialize release, entrainment and resistance areas`).
In the entrainment areas defined by the entrainment shapefile, particles can entrain mass through erosion or plowing.
In both mechanisms, one must account for three things:

    - change of mass due to the entrainment

    - change of momentum - entrained snow was accelerated from rest to the speed of the avalanche

    - loss of momentum due to the plowing or erosion processes -entrained mass bounds with the ground needs to be broken

These three terms are further detailed in :ref:`Entrainment <theoryCom1DFA:Entrainment>`. The parameters
used to compute these processes can be set in the configuration file.

In the numerics, the mass is updated according to the entrainment model in
:py:func:`com1DFA.DFAfunctionsCython.computeEntMassAndForce`. The velocity is updated immediately
after using an implicit formulation.


Go back to :ref:`com1DFAAlgorithm:Algorithm graph`



Take detrainment into account
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Snow detrainment can be added to the simulation. One must provide a resistance shapefile
in ``Inputs/RES`` and set the ``simType`` to ``res``, ``entres`` or ``available``
(see :ref:`com1DFAAlgorithm:Initialize release, entrainment and resistance areas`) and set the flag ``detrainment = True``.

In the areas defined by the resistance shapefile (for example in forests), mass of particles can be detrained causing a change of mass due to the detrainment.

This is further detailed in :ref:`Detrainment <theoryCom1DFA:Detrainment>`. The parameter
used to compute these processes can be set in the configuration file.

In the numerics, the mass is updated according to the detrainment model in
:py:func:`com1DFA.DFAfunctionsCython.computeDetMass`.

Detrainment is only applied when ``ResistanceModel =  default`` and ``detrainment = True``; depending on actual flow thickness and velocity,
either detrainment or the additional resistance force is applied within the resistance area.
See :ref:`theoryCom1DFA:Resistance` for more details.


Go back to :ref:`com1DFAAlgorithm:Algorithm graph`



Adapt surface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The surface (digital elevation model) can be adapted when snow masses change the surface due to

    - :ref:`detrainment <com1DFAAlgorithm:Take detrainment into account>` (if ``adaptSfcDetrainment = 1``),

    - :ref:`entrainment <com1DFAAlgorithm:Take entrainment into account>` (if ``adaptSfcEntrainment = 1``) and/or

    - stopping (velocity of a particle is zero; if ``adaptSfcStopped = 1``).


This is further detailed in :ref:`adaptive surface <theoryCom1DFA:Adaptive surface>`. The parameters
used to compute these processes can be set in the configuration file.

In the numerics, the surface is updated in :py:func:`com1DFA.com1DFA.adaptDEM`.
The surface and corresponding :ref:`DFAnumerics:Cell normals` and
:ref:`DFAnumerics:Cell area` are updated in every time step.

Go back to :ref:`com1DFAAlgorithm:Algorithm graph`



Compute lateral pressure forces
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The lateral pressure forces (:math:`F_{SPH}`) are related to the gradient of the flow thickness (:ref:`DFAnumerics:Forces discretization`). This gradient
is computed using a smoothed particle hydrodynamic method (:ref:`DFAnumerics:SPH gradient`).


Go back to :ref:`com1DFAAlgorithm:Algorithm graph`

Update position
----------------

Driving force, lateral pressure force and friction forces are subsequently used to update the velocity.
Then the particle position is updated using a centered Euler scheme.
These steps are done in :py:func:`com1DFA.DFAfunctionsCython.updatePositionC`.


Take gravity and lateral pressure forces into account
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:math:`F_{drive}` and :math:`F_{SPH}` are summed up and taken into account to update the velocity.
This is done via an explicit method.

Take friction into account
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:math:`F_{fric}` is taken into account to update the velocity.
This is done via an implicit method.

Update particle position
~~~~~~~~~~~~~~~~~~~~~~~~~
The particles position is updated using the new velocity and a centered Euler scheme:

.. math::
  \mathbf{x}_\text{new} = \mathbf{x}_\text{old} + dt 0.5 (\mathbf{u}_\text{old} + \mathbf{u}_\text{new})

Take dam interaction into account
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If the dam option is activated, the interaction between the particles and the dam is taken into account.
During the computation of the DFA simulation, at each time step, if a particle enters one of the dam cells, the dam
algorithm is called. The first step is to check if the particle crosses the dam during the time step.
If not, the particle position and velocity are updated as if there was no dam. If yes, the intersection point between
the particle trajectory and the dam line is computed. The dam properties are interpolated at the intersection point
(dam tangent and normal vectors). The wall tangent and normal vectors are updated taking the flow thickness
into account. The particle position and velocity are updated taking the dam into account.

Let :math:`\mathbf{x_\text{foot}}` and :math:`\mathbf{x_\text{new}}` be the intersection point with
the dam and the new particle position vector if there was no dam. :math:`\mathbf{u_\text{new}}` is the new particle
velocity with no dam. First the position :math:`\mathbf{x_\text{b}}` after elastic bouncing of the particle on the dam is computed:

.. math::

  \mathbf{x_\text{b}} = \mathbf{x_\text{new}} - 2 \left\{\mathbf{n_w^\text{filled}} \cdot (\mathbf{x_\text{new}} - \mathbf{x_\text{foot}})\right\}\mathbf{n_w^\text{filled}}

Which gives the direction :math:`\mathbf{e_\text{b}}`:

.. math::

  \mathbf{e_\text{b}} = \mathbf{x_\text{b}} - \mathbf{x_\text{foot}}

Next, the restitution coefficient is accounted for which leads to the new velocity:


.. math::

  \mathbf{x^\star_\text{new}} = \mathbf{x_\text{foot}} - \alpha_\text{rest} \mathbf{e_\text{b}} =
  \alpha_\text{rest} \mathbf{x_\text{b}} + (1-\alpha_\text{rest}) \mathbf{x_\text{foot}})



The velocity vector :math:`\mathbf{u_\text{new}^\star}` after the dam interaction reads:

.. math::

  \mathbf{u^\star_\text{new}} = (\mathbf{u_\text{new}} - 2  \left\{\mathbf{n_w^\text{filled}} \cdot \mathbf{u_\text{new}}\right\}\mathbf{n_w^\text{filled}})\alpha_\text{rest}

Finally, the new velocity and position are re-projected onto the topography.

Correction step:
~~~~~~~~~~~~~~~~
The particles z coordinate it readjusted so that the particles lie on the surface of the slope.
There are two reasons why the particles might not lie on the surface anymore after updating their position
according to the computed velocities:

  - 1) because of the inaccuracy related to the time and space discretization.
    This can lead to a particle position being slightly above or under the surface.
    We want to correct this inaccuracy and therefore reproject the particle on the surface
    using its x and y coordinates.

  - 2) because of the curvature of the slope and the particle velocity, particles can become
    detached from the ground in - in this case, the particle is located above the
    surface. In the current state, the com1DFA kernel does not allow this.
    If a particle becomes detached, the particle is also reprojected onto the surface
    using its x and y coordinates.

Similarly, the particles velocity is corrected in order to ensure that it lies in the tangent
plane to the surface (the velocity vector magnitude is preserved, only the direction is changed).

The way the particles position is reprojected onto the surface does not allow both the
velocity magnitude and the particle displacement to match perfectly. This is amplified
by highly curved topographies or abrupt changes in slope.


Go back to :ref:`com1DFAAlgorithm:Algorithm graph`

Add secondary release area
----------------------------
If a secondary release area is provided, the flow thickness
field from the previous time step is used to release a potential secondary release area.
To do so, the flow thickness field is compared to the secondary release area rasters. If
they overlap, the secondary release area is triggered and the secondary release particles
are initialized and added to the flowing particles.


Go back to :ref:`com1DFAAlgorithm:Algorithm graph`

Update fields
--------------

This steps are done in :py:func:`com1DFA.DFAfunctionsCython.updateFieldsC`.

The mesh values are updated with the particles properties using
:ref:`particles to mesh interpolation <DFAnumerics:Particles to mesh>` methods.
This is used to compute flow thickness, flow velocity and pressure fields from the particle properties.

Compute pressure
~~~~~~~~~~~~~~~~~
Pressure is computed following:

.. math::

  p = \mathbf{\rho}  \mathbf{u}²,

where :math:`\rho` is the snow density and :math:`\mathbf{u}` is the flow velocity.
This is done in :py:func:`com1DFA.DFAfunctionsCython.computePressure`.


Update particles flow thickness
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The mesh flow thickness is finally used to update the particle flow thickness value
using :ref:`mesh to particle interpolation <DFAnumerics:Mesh to particle>` methods.


Go back to :ref:`com1DFAAlgorithm:Algorithm graph`


Simulation outputs
-------------------

At the end of the simulation, the result fields are exported to raster files (.asc or .tif; depending on file format
of the input DEM) in
:py:func:`com1DFA.com1DFA.exportFields` and the information gathered in an info dictionary is used to
create a markdown report (:py:func:`log2Report.generateReport.writeReport`).
Further details on the available outputs can be found in :ref:`moduleCom1DFA:Output`.

Go back to :ref:`com1DFAAlgorithm:Algorithm graph`
