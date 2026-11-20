com1DFA DFA-Kernel numerics
============================


The numerical method used in com1DFA mixes particle methods and
mesh methods. Mass and momentum are tracked using particles but flow
thickness is tracked using the mesh. The mesh is also used to access topographic information
(surface elevation, normal vector) as well as for displaying results.

Mass :eq:`mass-balance3` and momentum :eq:`momentum-balance6` balance
equations as well as basal normal stress :eq:`sigmab` are computed numerically using a SPH method
(**S**\ moothed **P**\ article **H**\ ydrodynamics) (:cite:`Mo1992`) for the variables
:math:`\overline{\mathbf{u}}=(\overline{u}_1, \overline{u}_2)` and
:math:`\overline{h}` by discretization of the released avalanche volume
in a large number of mass elements. SPH in general, is a mesh-less
numerical method for solving partial differential equations. The SPH
algorithm discretizes the numerical problem within a domain using
particles (:cite:`Sa2007,SaGr2009`), which interact
with each-other in a defined zone of influence. Some of the advantages
of the SPH method are that free surface flows, material boundaries and
moving boundary conditions are considered implicitly. In addition, large
deformations can be modeled due to the fact that the method is not based
on a mesh. From a numerical point of view, the SPH method itself is
relatively robust.


Discretization
----------------

Space discretization
~~~~~~~~~~~~~~~~~~~~~~

The domain is discretized in particles. The following properties are assigned to each particle :math:`p_k`:
a mass :math:`m_{p_k}`, a thickness :math:`{h}_{p_k}`, a density :math:`\rho_{p_k}=\rho_0` and
a velocity :math:`\mathbf{{u}}_{p_k}=({u}_{p_k,1}, {u}_{p_k,2})` (**those
quantities are thickness averaged, note that we dropped the overline from** :eq:`hmean-umean` **for simplicity reasons**).
In the following paragraphs, :math:`i` and :math:`j` indexes refer to the different directions in the NCS,
whereas  :math:`k` and :math:`l` indexes refer to particles.

The quantities velocity, mass and flow thickness are also defined on the fixed mesh. It is possible to navigate
from particle property to mesh property using the interpolation methods described in :ref:`DFAnumerics:Mesh and interpolation`.


Time step
~~~~~~~~~~~~~~~~~~~~~~

A fixed time step can be used or an adaptive time step that depends on the sph kernel radius as well as the particle size.


Mesh and interpolation
-----------------------
Here is a description of the mesh and the interpolation method that is used to
switch from particle to mesh values and the other way around.

Mesh
~~~~~~

For practical reasons, a 2D rectilinear mesh (grid) is used. Indeed the topographic
input information is read from 2D raster files (with :math:`N_{y}` and :math:`N_{x}`
rows and columns) which correspond exactly to a
2D rectilinear mesh. Moreover, as we will see in the following sections,
2D rectilinear meshes are very convenient for interpolations as well as for
particle tracking. The 2D rectilinear mesh is composed of :math:`N_{y}` and
:math:`N_{x}` rows and columns of square cells (of side length :math:`csz`)
and :math:`N_{y}+1` and :math:`N_{x}+1` rows and columns of vertices
as described in :numref:`rasterGrid`. Each cell has a center and four vertices.
The data read from the raster file is assigned to the cell centers. Note that
although this is a 2D mesh, as we use a terrain-following coordinate system to perform
our computations, this 2D mesh is oriented in 3D space and hence the projected side length
corresponds to :math:`csz`, whereas the actual side length and hence also the
:ref:`DFAnumerics:cell area`, depend on the local slope,
expressed by the :ref:`DFAnumerics:Cell normals`.

.. _rasterGrid:

.. figure:: _static/rasterGrid0.png
        :width: 90%

        Rectangular grid

Cell normals
""""""""""""""
There are many different methods available for computing normal vectors
on a 2D rectilinear mesh. Several options are available in com1DFA.

The first one consists in computing the cross product of the diagonal vectors
between four cell centers. This defines the normal vector at the vertices. It is
then possible to interpolate the normal vector at the cell centers from the ones
at the vertices.

The other methods use the plane defined by different adjacent triangles to
a cell center. Each triangle has a normal and the cell center normal is the average
of the triangles normal vectors.

.. _meshNormal:

.. figure:: _static/meshNormal0.png
        :width: 90%

        Grid normal computation

Cell area
"""""""""""
The cell area can be deduced from the grid cellsize and the cell normal.
A cell is a plane (:math:`z = ax+by+c`) of same normal as the cell center:

.. math::
   \mathbf{n} = \frac{1}{\sqrt{1+a^2+b^2}}
   \left|\begin{aligned}
   &-a\\
   &-b\\
   &1
   \end{aligned}
   \right.

Surface integration over the cell extent leads to the area of the cell:

.. math::
   A_{cell} = \iint_{S} \mathrm{d}{S} = \int\limits_{0}^{csz}\int\limits_{0}^{csz}
   \sqrt{1+\frac{\partial z}{\partial x}^2+\frac{\partial z}{\partial y}^2}
   \mathrm{d}{x}\,\mathrm{d}{y} =
   csz^2 \sqrt{1+\frac{\partial z}{\partial x}^2+\frac{\partial z}{\partial y}^2} = \frac{csz^2}{n_z}


Interpolation
~~~~~~~~~~~~~~
In the DFA kernel, mass, flow thickness and flow velocity can be defined at particle
location or on the mesh. We need a method to be able to go from particle properties
to mesh (field) values and from mesh values to particle properties.

Mesh to particle
""""""""""""""""""

On a 2D rectilinear mesh, scalar and vector fields defined at cell centers
can be evaluated anywhere within the mesh using a bilinear interpolation
between mesh cell centers. Evaluating a vector field simply consists in evaluating
the three components as scalar fields.

The bilinear interpolation consists in successive linear interpolations
in both :math:`x` and :math:`y` direction using the four nearest cell centers,
two linear interpolations in the first direction (in our case in the
:math:`y` direction in order to evaluated :math:`f_{0v}` and :math:`f_{1v}`)
followed by a second linear interpolation in the second direction
(:math:`x` in our case to finally evaluate :math:`f_{uv}`) as shown on :numref:`BilinearInterp`:

.. math::
    \begin{aligned}
    f_{0v} = & (1-v)f_{00} + vf_{01}\\
    f_{1v} = & (1-v)f_{10} + vf_{11}
    \end{aligned}

and

.. math::
    \begin{aligned}
    f_{uv} = & (1-u)f_{0v} + uf_{1v}\\
           = & (1-u)(1-v)f_{00} + (1-u)vf_{01} + u(1-v)f_{10} + uvf_{11}\\
                  = & w_{00}f_{00} + w_{01}f_{01} + w_{10}f_{10} + w_{11}f_{11}
    \end{aligned}

the :math:`w` are the bilinear weights. The example given here is for a unit cell.
For no unit cells, the :math:`u` and :math:`v` simply have to be normalized by the
cell size.


.. _BilinearInterp:

.. figure:: _static/BilinearInterp.png
        :width: 90%

        Bilinear interpolation in a unit mesh (cell size is 1).


Particles to mesh
"""""""""""""""""""
Going from particle property to mesh value is also based on bilinear interpolation and
weights but requires a bit more care in order to conserve mass and momentum balance.
Flow thickness and velocity fields are determined on the mesh using, as intermediate step,
mass and momentum fields. First, mass and momentum mesh fields can be evaluated by
summing particles mass and momentum. This can be donne using the bilinear
weights :math:`w` defined in the previous paragraph (here :math:`f` represents
the mass or momentum and :math:`f_{uv}` is the particle value. :math:`f_{nm}`
, :math:`{n, m} \in \{0, 1\} \times \{0, 1\}`, are the cell center values):

.. math::
    \begin{aligned}
    f_{00} = & w_{00}f_{uv}\\
    f_{01} = & w_{01}f_{uv}\\
    f_{10} = & w_{10}f_{uv}\\
    f_{11} = & w_{11}f_{uv}
    \end{aligned}

The contribution of each particle to the different mesh points is summed up to
finally give the mesh value. This method ensures that the total mass and
momentum of the particles is preserved (the mass and momentum on the mesh will
sum up to the same total). Flow thickness and velocity mesh fields can then be deduced
from the mass and momentum fields and the cell area (actual area of each grid cell,
not the projected area).


Neighbor search
------------------

The lateral pressure forces are computed via the SPH flow thickness gradient.
This method is based on particle interactions within a certain neighborhood, meaning that it
is necessary to keep track of all the particles within the neighborhood of each particle.
Computing the gradient of the flow thickness at a particle location, requires to
find all the particles in its surrounding. Considering the number of particles and
their density, computing the gradient ends up in computing a lot of
interactions and represents the most computationally expensive part of the dense
flow avalanche simulation. It is therefore important that the neighbor search is fast and efficient.
:cite:`IhOrSoKoTe2014` describe different rectilinear mesh neighbor search
methods. In com1DFA, the simplest method is used. The idea is to locate each
particle in a cell, this way, it is possible to keep track of the particles
in each cell. To find the neighbors of a particle, one only needs to read the
cell in which the particle is located (dark blue cell in :numref:`neighborSearch`)
, find the direct adjacent cells in all directions (light blue cells) and
simply read all particles within those cells. This is very easily achieved
on rectilinear meshes because locating a particle in a cell is straightforward and
finding the adjacent cells is also easily done.

.. _neighborSearch:

.. figure:: _static/neighborSearch.png
        :width: 90%

        Support mesh for neighbor search:
        if the cell side is bigger than the kernel length :math:`r_{kernel}` (red circle in the picture),
        the neighbors for any particle in any given cell (dark blue square)
        can be found in the direct neighborhood of the cell itself (light blue squares)

.. _partInCell:

.. figure:: _static/partInCell.png
        :width: 90%

        The particles are located in the cells using
        two arrays. indPartInCell of size number of cells + 1
        which keeps track of the number of particles in each cell
        and partInCell of size number of particles + 1 which lists
        the particles contained in the cells.

SPH gradient
--------------
SPH method can be used to solve thickness integrated equations where a 2D
(respectively 3D) equation is reduced to a 1D (respectively 2D) one.
This is used in ocean engineering to solve shallow water equations (SWE)
in open or closed channels for example. In all these applications,
whether it is 1D or 2D SPH, the fluid is most of the time,
assumed to move on a horizontal plane (bed elevation is set to a constant).
In the case of avalanche flow, the "bed" is sloped and irregular.
The aim is to adapt the SPH method to apply it to thickness integrated equations
on a 2D surface living in a 3D world.

Method
~~~~~~~
The SPH method is used to express a quantity (the flow thickness in our case) and
its gradient at a certain particle location as a weighted sum of its neighbors
properties. The principle of the method is well described in :cite:`LiLi2010`.
In the case of thickness integrated equations (for example SWE), a scalar function
:math:`f` and its gradient can be expressed as following:

.. math::
    f_{k} &= \sum\limits_{l}f_{l}A_{l}\,W_{kl}\\
    \mathbf{\nabla}f_{k} &= -\sum\limits_{l}f_{l}A_{l}\,\mathbf{\nabla}W_{kl}
    :label: sph formulation

Which gives for the flow thickness:

.. math::
    \overline{h}_{k} &= \frac{1}{\rho_0}\,\sum\limits_{l}{m_{l}}\,W_{kl}\\
    \mathbf{\nabla}\overline{h}_{k} &= -\frac{1}{\rho_0}\,\sum\limits_{l}{m_{l}}\,\mathbf{\nabla}W_{kl}
    :label: sph formulation for fd

Where :math:`W` represents the SPH-Kernel function.

The computation of its gradient depends on the coordinate system used.

.. _standard-method:

Standard method
""""""""""""""""

Let us start with the computation of the gradient of a scalar function
:math:`f \colon \mathbb{R}^2 \to \mathbb{R}` on a horizontal plane.
Let :math:`P_k=\mathbf{x}_k=(x_{k,1},x_{k,2})` and :math:`Q_l=\mathbf{x}_l=(x_{l,1},x_{l,2})` be two points in :math:`\mathbb{R}^2` defined by
their coordinates in the Cartesian coordinate system :math:`(P_k,\mathbf{e_1},\mathbf{e_2})`. :math:`\mathbf{r}_{kl}=\mathbf{x}_k-\mathbf{x}_l` is the vector going from
:math:`Q_l` to :math:`P_k` and :math:`r_{kl} = \left\Vert \mathbf{r}_{kl}\right\Vert` the length of this vector.
Now consider the kernel function :math:`W`:


.. math::
  \left.
  \begin{aligned}
  W \colon \mathbb{R}^2 \times \mathbb{R}^2 \times \mathbb{R} &\to \mathbb{R}\\
  (P_k, Q_l, r_0) &\mapsto W(P_k, Q_l, r_0)
  \end{aligned}
  \right.\quad, r_0\in\mathbb{R} \mbox{ is the smoothing kernel length}

In the case of the spiky kernel, :math:`W` reads (2D case):

.. math::
   \begin{aligned}
   W_{kl} = &W(\mathbf{x_k},\mathbf{x_l},r_0) = W(\mathbf{x_k}-\mathbf{x_l},r_0) = W(\mathbf{r_{kl}},r_0)\\
   =&\frac{10}{\pi r_0^5}\left\{
   \begin{aligned}
   & (r_0 - \left\Vert \mathbf{r_{kl}}\right\Vert)^3, \quad &0\leq \left\Vert \mathbf{r_{kl}}\right\Vert \leq  r_0\\
   & 0 , & r_0 <\left\Vert \mathbf{r_{kl}}\right\Vert
   \end{aligned}
   \right.
   \end{aligned}
   :label: kernel function


:math:`\left\Vert \mathbf{r_{kl}}\right\Vert= \left\Vert \mathbf{x_{k}}-\mathbf{x_{l}}\right\Vert`
represents the distance between particle :math:`k` and :math:`l` and
:math:`r_0` the smoothing length.

Using the chain rule to express the gradient of :math:`W` in the Cartesian
coordinate system :math:`(x_1,x_2)` leads to:


.. math::
   \mathbf{\nabla}W_{kl} = \frac{\partial W}{\partial r}.\mathbf{\nabla}r,
   \quad r = \left\Vert \mathbf{r} \right\Vert = \sqrt{(x_{k,1}-x_{l,1})^2 + (x_{k,2}-x_{l,2})^2}
   :label: kernel function gradient 1

with,

.. math::
  \frac{\partial W}{\partial r} = -3\frac{10}{\pi r_0^5}\left\{
  \begin{aligned}
  & (r_0 - \left\Vert \mathbf{r_{kl}}\right\Vert)^2, \quad &0\leq \left\Vert \mathbf{r_{kl}}\right\Vert \leq  r_0\\
  & 0 , & r_0 <\left\Vert \mathbf{r_{kl}}\right\Vert
  \end{aligned}
  \right.

and

.. math::
  \frac{\partial r}{\partial x_{k,i}} = \frac{(x_{k,i}-x_{l,i})}{\sqrt{(x_{k,1}-x_{l,1})^2 + (x_{k,2}-x_{l,2})^2}},
  \quad i=\{1,2\}

which leads to the following expression for the gradient:

.. math::
   \mathbf{\nabla}W_{kl} = -3\frac{10}{\pi r_0^5}\left\{
   \begin{aligned}
   & (r_0 - \left\Vert \mathbf{r_{kl}}\right\Vert)^2\frac{\mathbf{r_{kl}}}{r_{kl}}, \quad &0\leq \left\Vert \mathbf{r_{kl}}\right\Vert \leq  r_0\\
   & 0 , & r_0 <\left\Vert \mathbf{r_{kl}}\right\Vert
   \end{aligned}
   \right.
   :label: kernel function gradient

The gradient of :math:`f` is then simply:

.. math::
    \mathbf{\nabla}f_{k} = -\sum\limits_{l}f_{l}A_{l}\,\mathbf{\nabla}W_{kl}
    :label: sph gradient1

2.5D SPH method
""""""""""""""""
We now want to express a function :math:`f` and its gradient on a potentially
curved surface and express this gradient in the 3 dimensional Cartesian
coordinate system :math:`(P_k,\mathbf{e_1},\mathbf{e_2},\mathbf{e_3})`.

Let us consider a smooth surface :math:`\mathcal{S}` and two points
:math:`P_k=\mathbf{x}_k=(x_{k,1},x_{k,2},x_{k,3})` and :math:`Q_l=\mathbf{x}_l=(x_{l,1},x_{l,2},x_{l,3})`
on :math:`\mathcal{S}`. We can define :math:`\mathcal{TP}` the tangent plane
to :math:`\mathcal{S}` in :math:`P_k`. If :math:`\mathbf{u}_k` is the (none zero)
velocity of the particle at :math:`P_k`, it is possible to define the local
orthonormal coordinate system :math:`(P_k,\mathbf{V_1},\mathbf{V_2},\mathbf{V_3}=\mathbf{n})`
with :math:`\mathbf{V_1}=\frac{\mathbf{u}_k}{\left\Vert \mathbf{u}_k\right\Vert}`
and :math:`\mathbf{n}` the normal to :math:`\mathcal{S}` at :math:`P_k`.
Locally, :math:`\mathcal{S}` can be assimilated to :math:`\mathcal{TP}` and
:math:`Q_l` to its projection :math:`Q'_l` on :math:`\mathcal{TP}`.
The vector :math:`\mathbf{r'}_{kl}=\mathbf{x}_k-\mathbf{x'}_l` going from
:math:`Q'_l` to :math:`P_k` lies in :math:`\mathcal{TP}` and can be express
in the plane local basis:

.. math::
  \mathbf{r'}_{kl}=\mathbf{x}_k-\mathbf{x'}_l = v_{kl,1}\mathbf{V_1} + v_{kl,2}\mathbf{V_2}

It is important to define :math:`f` properly and the gradient that will be calculated:

.. math::
  \left.
  \begin{aligned}
  f \colon \mathcal{TP}\subset\mathbb{R}^3 &\to \mathbb{R}\\
  (x_1,x_2,x_3) &\mapsto f(x_1,x_2,x_3) = f(x_1(v_1,v_2),x_2(v_1,v_2)) = \tilde{f}(v_1,v_2)
  \end{aligned}
  \right.

Indeed, since :math:`(x_1,x_2,x_3)` lies in :math:`\mathcal{TP}`, :math:`x_3`
is not independent of :math:`(x_1,x_2)`:

..  .. math::
..   x_3 = \frac{-x_1(\mathbf{e_1}.\mathbf{V_3})-x_2(\mathbf{e_2}.\mathbf{V_3})}{\mathbf{e_3}.\mathbf{V_3}} */

.. math::
  \left.
  \begin{aligned}
  \tilde{f} \colon \mathcal{TP}\subset\mathbb{R}^2 &\to \mathbb{R}\\
  (v_1,v_2) &\mapsto \tilde{f}(v_1,v_2) = \tilde{f}(v_1(x_1,x_2),v_2(x_1,x_2)) = f(x_1,x_2,x_3)
  \end{aligned}
  \right.

The target is the gradient of :math:`\tilde{f}` in terms of the :math:`\mathcal{TP}` variables
:math:`(v_1,v_2)`. Let us call this gradient :math:`\mathbf{\nabla}_\mathcal{TP}`.
It is then possible to apply the :ref:`standard-method` to compute this gradient:


.. math::
   \mathbf{\nabla}_\mathcal{TP}W_{kl} = \frac{\partial W}{\partial r}.\mathbf{\nabla}_\mathcal{TP}r,
   \quad r = \left\Vert \mathbf{r} \right\Vert = \sqrt{v_{kl,1}^2 + v_{kl,2}^2}
   :label: kernel function gradient TP 1

Which leads to:

.. math::
  \mathbf{\nabla}_\mathcal{TP}W_{kl} = -3\frac{10}{\pi r_0^5}\frac{(r_0 - \left\Vert \mathbf{r_{kl}'}\right\Vert)^2}{r_{kl}'}\left\{
  \begin{aligned}
  & v_{kl,1}\mathbf{V_1} + v_{kl,2}\mathbf{V_2}, \quad &0\leq \left\Vert \mathbf{r_{kl}'}\right\Vert \leq  r_0\\
  & 0 , & r_0 <\left\Vert \mathbf{r_{kl}'}\right\Vert
  \end{aligned}
  \right.
  :label: kernel function gradient TP 2


.. math::
  \mathbf{\nabla}_\mathcal{TP}\tilde{f_{k}} = -\sum\limits_{l}\tilde{f_{l}}A_{l}\,\mathbf{\nabla}W_{kl}
  :label: sph gradient2

This gradient can now be expressed in the Cartesian coordinate system.
It is clear that the change of coordinate system was not needed:


.. math::
  \mathbf{\nabla}_\mathcal{TP}W_{kl} = -3\frac{10}{\pi r_0^5}\frac{(r_0 - \left\Vert \mathbf{r_{kl}'}\right\Vert)^2}{r_{kl}'}\left\{
  \begin{aligned}
  & r_{kl,1}\mathbf{e_1} + r_{kl,2}\mathbf{e_2} + r_{kl,3}\mathbf{e_3}, \quad &0\leq \left\Vert \mathbf{r_{kl}'}\right\Vert \leq  r_0\\
  & 0 , & r_0 <\left\Vert \mathbf{r_{kl}'}\right\Vert
  \end{aligned}
  \right.

The advantage of computing the gradient in the local coordinate system is if
the components (in flow direction or in cross flow direction) need to be treated
differently.


.. _2_5DSPH:

.. figure:: _static/2_5DSPH.png
        :width: 90%

        Tangent plane and local coordinate system used to apply the SPH method


Particle splitting and merging
-------------------------------
There are two different approaches treating splitting of particles in com1DFA.
The first one only deals with splitting of particles with too much mass('split only'). The second approach,
"split/merge" approach aims at keeping a stable amount of particles within a given range. This is done in order to
guaranty a sufficient accuracy of the sph flow thickness gradient computation.

Split (**default**)
~~~~~~~~~~~~~~~~~~~~
If the ``splitOption`` is set to 0, particles are split because of snow entrainment. In this case,
particles that entrain snow grow, i.e. their mass increases. At one point the mass of the particles is considered to be
too big and this particle is split in two. The splitting operation happens if the mass of the
particle exceeds a threshold value (:math:`mPart > massPerPart \times thresholdMassSplit`), where ``thresholdMassSplit``
is specified in the configuration file and ``massPerPart`` depends on the chosen ``massPerParticleDeterminationMethod``
as defined here: :ref:`com1DFAAlgorithm:Initialize particles`.
When a particle is split a new child particle is created with the same properties as the parent apart from
mass and position. Both parent and child get half of the parent mass. The parent and child's position are
adjusted: the first / second is placed forward / backward in the direction of the velocity
vector at a distance :math:`distSplitPart \times rPart` of the initial parent position. Particles are considered to
have a circular basal surface :math:`A = \frac{m}{\rho} = \pi r^2`.

Split and merge
~~~~~~~~~~~~~~~
If the ``splitOption`` is set to 1 particles are split or merged in order to keep the particle count
as constant as possible within the kernel radius.
Assessing the number of particles within one kernel radius is done based on the particle area. Particles
are assumed to be cylindrical, i.e the base is a circle. For particle ``k`` we have :math:`A_k = \frac{m_k}{\rho}`. The area
of the support domain of the sph kernel function is :math:`\pi r_0^2`. The aim is to keep ``nPPK`` particles within
the kernel radius. The particles are split if the estimated number of particles per kernel radius :math:`\frac{\pi r_0^2}{A_k}`
falls below a given value of :math:`n_{PPK}^{min} = C_{n_{PPK}}^{min}n_{PPK}`. Particles are split using the same
method as in :ref:`DFAnumerics:Split (**default**)`. Similarly, particles are merged if the estimated
number of particles per kernel radius exceeds a given value :math:`n_{PPK}^{max} = C_{n_{PPK}}^{max}n_{PPK}`.
In this case particles are merged with their closest neighbor. The new position and velocity is the mass
averaged one. The new mass is the sum. Here, two coefficients :math:`C_{n_{PPK}}^{min}` (``cMinNPPK``) and :math:`C_{n_{PPK}}^{max}` (``cMaxNPPK``) were
introduced. A good balance needs to be found for the coefficients so that the particles are not constantly split or
merged but also not too seldom. The split and merge steps happen only once per time step and per particle.

Artificial viscosity
---------------------

Two options are available to add viscosity to stabilize the numerics. The first option
consists in adding artificial viscosity (``viscOption`` = 1). The second option attempts
to adapt the Lax-Friedrich scheme (usually applied to meshes) to the particle method
(``viscOption`` = 2). Finally, ``viscOption`` = 0 deactivates any viscosity force.

SAMOS Artificial viscosity
~~~~~~~~~~~~~~~~~~~~~~~~~~

In :ref:`theoryCom1DFA:Governing Equations for the Dense Flow Avalanche`, the governing
equations for the DFA were derived and all first order or smaller terms where neglected.
Among those terms is the lateral shear stress. This term leads toward
the homogenization of the velocity field. It means that two neighbor elements
of fluid should have similar velocities. The aim behind adding artificial viscosity is to
take this phenomena into account. The following viscosity force is added:

.. math::
    \begin{aligned}
    \mathbf{F_{viscosity}} = &- \frac{1}{2}\rho C_{Lat}\|\mathbf{du}\|^2 A_{Lat}
    \frac{\mathbf{du}}{\|\mathbf{du}\|}\\
    = & - \frac{1}{2}\rho C_{Lat}\|\mathbf{du}\| A_{Lat} \mathbf{du}
    \end{aligned}

Where the velocity difference reads :math:`\mathbf{du} = \mathbf{u} - \mathbf{\bar{u}}`
(:math:`\mathbf{\bar{u}}` is the mesh velocity interpolated at the particle position).
:math:`C_{Lat}` is a coefficient that rules the viscous force. It would be the
equivalent of :math:`C_{Drag}` in the case of the drag force. The :math:`C_{Lat}`
is a numerical parameter that depends on the mesh size. Its value is set to 100
and should be discussed and further tested.

Adding the viscous force
""""""""""""""""""""""""

The viscous force acting on particle :math:`k` reads:

.. math::
  \begin{aligned}
  \mathbf{F_k^{viscosity}} = &-\frac{1}{2}\rho C_{Lat}\|\mathbf{du}_k^{old}\| A_{Lat}
  \mathbf{du}_k^{new}\\
  = &  -\frac{1}{2}\rho C_{Lat}\|\mathbf{u}_k^{old} - \mathbf{\bar{u}}_k^{old}\| A_{Lat}
  (\mathbf{u}_k^{new} - \mathbf{\bar{u}}_k^{old})
  \end{aligned}

Updating the velocity is done in two steps. First adding the explicit term related to the
mean mesh velocity and then the implicit term which leads to:

.. math::
  \mathbf{u}_k^{new} = \frac{\mathbf{u}_k^{old} - C_{vis}\mathbf{\bar{u}}_k^{old}}{1 + C_{vis}}

With :math:`C_{vis} = \frac{1}{2}\rho C_{Lat}\|\mathbf{du}_k^{old}\| A_{Lat}\frac{dt}{m}`


Ata Artificial viscosity
~~~~~~~~~~~~~~~~~~~~~~~~

An upwind method based on Lax-Friedrichs scheme
"""""""""""""""""""""""""""""""""""""""""""""""""""""

Shallow Water Equations are well known for being hyperbolic transport equations.
They have the particularity of carrying discontinuities or shocks which will cause
numerical instabilities.

A decentering in time allows to better capture the discontinuities.
This can be done in the manner of the Lax-Friedrich scheme as described in :cite:`AtSo2005`,
which is formally the same as adding a viscous force. Implementing it for the SPH method,
this viscous force applied on a given particle :math:`k` can be expressed as follows:    

.. math::
    \mathbf{F_k^{viscosity}} = \sum_{l} \frac{m_l}{\rho_l} \Pi_{kl} \mathbf{\nabla}W_{kl}

with :math:`\Pi_{kl} = \lambda_{kl}(\mathbf{u}_l - \mathbf{u}_k) \cdot
\frac{\mathbf{r}_{kl}}{\vert\vert \mathbf{r}_{kl} \vert\vert}`, and
:math:`\mathbf{\nabla}W_{kl}` is the gradient of the kernel function and
is described in :ref:`DFAnumerics:SPH gradient`.

:math:`\mathbf{u}_{kl} = \mathbf{u}_k - \mathbf{u}_l` is the relative velocity
between particle k and l, :math:`\mathbf{r}_{kl} = \mathbf{x}_k - \mathbf{x}_l` is
the vector going from particles :math:`l` to particle :math:`k` and
:math:`\lambda_{kl} = \frac{c_k+c_l}{2}` with :math:`c_k = \sqrt{gh_l}`
the wave speed. The :math:`\lambda_{kl}` is obtained by turning expressions
related to time and spatial discretization parameters into an expression
on maximal speed between both particles in the Lax Friedrich scheme.

Due to the expression of the viscosity force, it makes sense to
compute it at the same place where the SPH pressure force are computed (for this reason, the
``viscOption`` = 2 corresponding to the "Ata" viscosity option is only available
in combination with the ``sphOption`` = 2).


Forces discretization
----------------------

Lateral force
~~~~~~~~~~~~~~

The SPH method is introduced when expressing the flow thickness gradient for each
particle as a weighted sum of its neighbors
(:cite:`LiLi2010,Sa2007`). From now on the :math:`p` for particles in :math:`p_k` is dropped
(same applies for :math:`p_l`).

The lateral pressure forces on each particle are calculated from the compression
forces on the boundary of the particle.
The boundary is approximated as a square with the base side length
:math:`\Delta s = \sqrt{A_p}` and the respective flow height. This leads to
(subscript :math:`|_{.,i}` stands for the component in the :math:`i^{th}`
direction, :math:`i = {1,2}`):

.. math::
    F_{k,i}^{\text{lat}} = K_{(i)}\oint\limits_{\partial{A_{k}}}\left(\int\limits_{b}^{s}\sigma_{33}\,n_i\,\mathrm{d}x_3\right)\mathrm{d}l

From equation :eq:`momentum-balance6`

.. math::
    F_{k,i}^{\text{lat}} = K_{(i)}\,\frac{\Delta{s}}{2}\left((\overline{h}\,\overline{\sigma}^{(b)}_{33})_{x_{i}-
    \frac{\Delta{s}}{2}}-(\overline{h}\,\overline{\sigma}^{(b)}_{33})_{x_{i}+\frac{\Delta{s}}{2}}\right)
    = K_{(i)}\frac{\Delta{s}^2}{2}\,\left.\frac{d\,\overline{h}\,\overline{\sigma}^{(b)}}{d\,x_i}\right\rvert_{k}

The product of the average flow thickness :math:`\overline{h}` and the basal normal pressure :math:`\overline{\sigma}^{(b)}_{33}`
reads (using equation :eq:`sigmab` and dropping the curvature acceleration term):

.. math::
   \overline{h}\,\overline{\sigma}^{(b)} = \overline{h}^2\,\rho_0\,\left(g_3-\overline{u_1}^2\,\frac{\partial^2{b}}{\partial{x_1^2}}\right)
   \approx \overline{h}^2\,\rho_0\,g_3

Which leads to, using the relation :eq:`sph formulation`:

.. math::
    F_{k,i}^{\text{lat}} = K_{(i)}\,\rho_0\,g_3\,A_{k}\,\overline{h}_{k}\,.\,\left.\frac{d\,\overline{h}}{d\,x_i}\right\rvert_{k}
    = -K_{(i)}\,m_{i}\,g_3\,.\,\frac{1}{\rho_0}\,\sum\limits_{l}{m_{l}}\,\left.\frac{d\,W_{kl}}{d\,x_i}\right\rvert_{l}
    :label: lateral force


Bottom friction force
~~~~~~~~~~~~~~~~~~~~~~~
The bottom friction forces on each particle depend on the chose friction model. Using the SamosAT friction model
(using equation :eq:`sigmab` for the expression of :math:`\sigma^{(b)}_{k}`) the formulation of the bottom friction forec
reads:

.. math::
    F_{k,i}^{\text{bot}} = -\frac{\overline{u}_{k,i}}{\|\mathbf{u}_k\|}\,A_{k}\,\tau^{(b)}_{k}
    = -\delta_{k1}\,A_{k}\,\left(\tau_0 + \tan{\delta}\,\left(1+\frac{R_s^0}{R_s^0+R_s}\right)\,\sigma^{(b)}_{k}
     + \frac{\rho_0\,\mathbf{\overline{u}}_{k}^2}{\left(\frac{1}{\kappa}\,\ln\frac{\overline{h}}{R} + B\right)^2}\right)
    :label: bottom force


Added resistance force
~~~~~~~~~~~~~~~~~~~~~~~
The resistance force on each particle reads:

.. math::
    F_{k,i}^{\text{res}}
    = - \rho_0\,A_{k}\,C_{\text{resH}}\,\|\overline{\mathbf{u}}_{k}\|^2\,\frac{\overline{u}_{k,i}}{|\overline{\mathbf{u}}_{k}\|}
    :label: resistance force

Both the bottom friction and resistance force are friction forces. The expression above represent the maximal
friction force that can be added. This maximal force is added if the particles are flowing. If not, the friction force
equals the driving forces. See :cite:`MaVi2003` for more information.

In the default setup (``ResistanceModel =  default``), either detrainment or the extra resistance force is applied within the resistance area, depending on the thickness and speed of the flow.
See :ref:`theoryCom1DFA:Resistance:` for more details.

Entrainment force
~~~~~~~~~~~~~~~~~~~~~~~
The term :math:`- \overline{u_i}\,\rho_0\,\frac{\mathrm{d}(A\,\overline{h})}{\mathrm{d}t}`
related to the entrained mass in :eq:`momentum-balance3` now reads:

.. math::
    - \overline{u}_{k,i}\,\rho_0\,\frac{\mathrm{d}}{\mathrm{d}t}\,\left(A_{k}\,\overline{h}_{k}\right)
    = - \overline{u}_{k,i}\,A^{\text{ent}}_{k}\,q^{\text{ent}}_{k}


The mass of entrained snow for each particle depends on the type of entrainment involved
(plowing or erosion) and reads:

.. math::
    \rho_0\,\frac{\mathrm{d}}{\mathrm{d}t}\,\left(A_{k}\,\overline{h}_{k}\right)
    = \frac{\mathrm{d}\,m_{k}}{\mathrm{d}t}
    = A_{k}^\text{ent}\,q_{k}^{\text{ent}}

with

.. math::
    \begin{aligned}
    A_{k}^{\text{plo}} &= w_f\,h_{k}^{\text{ent}}= \sqrt{\frac{m_{k}}{\rho_0\,\overline{h}_{k}}}\,h_{k}^{\text{ent}}
    \quad &\mbox{and} \quad &q_{k}^{\text{plo}} = \rho_{\text{ent}}\,\left\Vert \overline{\mathbf{u}}_{k}\right\Vert
    \quad &\mbox{for plowing}\\
    A_{k}^{\text{ero}} &= A_{k} = \frac{m_{k}}{\rho_0\,\overline{h}_{k}}
    \quad &\mbox{and} \quad &q_{k}^{\text{ero}} = \frac{\tau_{k}^{(b)}}{e_b}\,\left\Vert \overline{\mathbf{u}}_{k}\right\Vert
    \quad &\mbox{for erosion}\end{aligned}

Finaly, the entrainment force reads:

.. math::
    F_{k,i}^{\text{ent}} = -w_f\,(e_s+\,q_{k}^{\text{ent}}\,e_d)

Adding forces
--------------
The different components are added following an operator splitting method.
This means particle velocities are updated successively with the different forces.


Adding artificial viscosity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If the viscosity option (``viscOption``) is set to 1, artificial viscosity is added first, as described
in :ref:`DFAnumerics:Artificial viscosity` (this is the default option). With ``viscOption`` set to 0, no viscosity is added. Finally, if
``viscOption`` is set to 2, artificial viscosity is added during SPH force computation. (TODO add link to description)

Adding entrainment
~~~~~~~~~~~~~~~~~~~
Entrainment is taken into account by first adding the component representing the loss of momentum due to
acceleration of the entrained mass :math:`- \overline{u}_{k,i}\,A^{\text{ent}}_{k}\,q^{\text{ent}}_{k}`.
Second by adding the force due to the need to break and compact the
entrained mass (:math:`F_{k,i}^{\text{ent}}`) as described in :ref:`DFAnumerics:Entrainment force`.


Adding driving forces
~~~~~~~~~~~~~~~~~~~~~~~~
The driving forces -gravity force and lateral forces- are taken into account next. The velocity is updated explicitly.


Adding friction forces
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Both the bottom friction and resistance forces act against the flow. Two methods are available to add these
forces in com1DFA.

An implicit method:
""""""""""""""""""""

.. math::
  \mathbf{u}_k^{new} = \frac{\mathbf{u}_k^{old}}{1 + \frac{C_{k}^{\text{fric}}\Delta t}{m_k}}

where :math:`F_{k,i}^{\text{fric}} = C_{k}^{\text{fric}} u_{k,i}^{new} = F_{k,i}^{\text{res}} + F_{k,i}^{\text{bot}}`
(the two forces are described in :ref:`DFAnumerics:Bottom friction force` and :ref:`DFAnumerics:Added resistance force`).

This implicit method has a few draw-backs. First the flow does not start properly if the
friction angle :math:`\delta` is too close to the slope angle. Second, the flow never properly stops, even if the
particles physically should, i.e. particles keep oscillating back and force around their end position.


An explicit method:
""""""""""""""""""""

The method based on :cite:`MaVi2003` addresses these two issues.
The idea is that the friction forces only modify the magnitude of velocity and not the direction. This means dissipation,
so the friction force can not become a driving force. Moreover, the friction force magnitude depends on the particle state,
i.e. if it is flowing or at rest.
The friction force is expressed:

.. math::
  \mathbf{F}_k^{\text{fric}} = -\|\mathbf{F}_{k}^{\text{fric}}\| \frac{\mathbf{u}_k}{\|\mathbf{u}_k\|}

with:

.. math::
  \|\mathbf{F}_{k}^{\text{fric}}\| \leq \|\mathbf{F}_{k}^{\text{fric}}\|_{max}

If the velocity of the particle ``k`` reads :math:`\mathbf{u}_k^{old}` after adding the driving forces, adding the
fiction force leads to :

.. math::
  \mathbf{u}_{k} = \mathbf{u}_k^{old} (1 - \frac{\Delta t}{m} \frac{\|\mathbf{F}_{k}^{\text{fric}}\|}{\|\mathbf{u}^{old}_k\|}),
  \quad \|\mathbf{F}_{k}^{\text{fric}}\| = \|\mathbf{F}_{k}^{\text{fric}}\|_{max}


at the condition that  :math:`1 \geq \frac{\Delta t}{m} \frac{\|\mathbf{F}_{k}^{\text{fric}}\|_{max}}{\|\mathbf{u}^{old}_k\|}`.
If on the contrary :math:`1 \leq \frac{\Delta t}{m} \frac{\|\mathbf{F}_{k}^{\text{fric}}\|_{max}}{\|\mathbf{u}^{old}_k\|}`,
the friction would change the velocity direction which is nonphysical. In this case, the particle will stop
before the end of the time step. This allows the particles to start and stop flowing properly.
