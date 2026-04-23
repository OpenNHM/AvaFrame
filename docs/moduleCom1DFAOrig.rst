com1DFAOrig: Original DFA-Kernel
===================================

:py:mod:`com1DFAOrig` is a simulation tool for dense flow (snow) avalanches (DFA). It is
a python wrapper function calling the samos-AT C++ (Snow Avalanche Modeling and  Simulation- Advanced Technologies)
DFA simulation code developed by the Austrian government in cooperation with the company AVL List GmbH in Graz.
This code served as base for the implementation of the python DFA simulation module :py:mod:`com1DFA` presented
in :ref:`moduleCom1DFA:com1DFA: DFA-Kernel`. :py:mod:`com1DFA` module is now the default DFA simulation module in AvaFrame.

The calculation of the DFA is based on the depth integrated governing equations and
solved numerically using the smoothed particle hydrodynamic (sph) method.

Dense flow avalanche simulations can be performed for different release area scenarios, with or without
entrainment and/or resistance areas.
There is the option to vary the internal friction parameter or the release snow thickness.

C++ Executable
------------------

The computation of the :py:mod:`com1DFAOrig` dense flow avalanche module relies on a C++ executable.
The executable (for now 64bit linux and windows) and needed files are
available in this `git repository <https://github.com/avaframe/com1DFA_Exe>`_.
To install, change into your directory [YOURDIR] from the AvaFrame installation above and
clone the repository::

  cd [YOURDIR]
  git clone https://github.com/avaframe/com1DFA_Exe

Rename the executables according to your operating system, i.e. for Linux do::

  mv com1DFA_Exe/com1DFA_x86_64.exe com1DFA_Exe/com1DFA.exe
  mv com1DFA_Exe/SHPConv_linux.exe com1DFA_Exe/SHPConv.exe

for Windows do::

  mv com1DFA_Exe/com1DFA_win64.exe com1DFA_Exe/com1DFA.exe
  mv com1DFA_Exe/SHPConv_win.exe com1DFA_Exe/SHPConv.exe


Go to the ``com1DFAOrig`` directory of the AvaFrame repository from above and copy the configuration file::

  cd AvaFrame/avaframe/com1DFAOrig
  cp com1DFACfg.ini local_com1DFACfg.ini

Open the ``local_com1DFACfg.ini`` file in your preferred text editor and change
the ``com1Exe`` variable to reflect your paths, i.e.::

  com1Exe = [YOURDIR]/com1DFA_Exe/com1DFA.exe -files [YOURDIR]/com1DFA_Exe/files/AK_Attributes


.. Attention::  We suggest to use the full path.

To test go to [YOURDIR], change into the com1DFA_Exe repository and run the
executable::

  cd [YOURDIR]
  cd com1DFA_Exe
  ./com1DFA.exe -files files/AK_Attributes/

The output should start like this::

  Setting config files directory: files/AK_Attributes/	(src/SW_Workspace.cpp:3435)
  ./com1DFA.exe -files files/AK_Attributes/ 	(src/SW_Workspace.cpp:3453)
  =================================================================
  ./com1DFA.exe
  Compiled Oct 19 2020 21:34:15
  ...

Exit by pressing ``q``


Input
---------

The module requires an avalanche directory, that follows a specified folder structure.
This avalanche directory can be created by running :py:mod:`runInitializeProject.py`.
In the directory *Inputs*, the following files are required:

* digital elevation model as .asc file
  -> use `ESRI grid format <https://desktop.arcgis.com/en/arcmap/10.3/manage-data/raster-and-images/esri-ascii-raster-format.htm>`_
* release area scenario as shapefile (in Inputs/REL); multiple are possible
  -> the shapefile name should not contain an underscore, if so '_AF' is added

and the following files are optional:

* entrainment area as shapefile (in *Inputs/ENT*)
* resistance area as shapefile (in *Inputs/RES*)
* secondary release area as shapefile (in Inputs/SECREL)

The simulation settings area defined in the configuration file ``com1DFAOrig/com1DFAOrigCfg.ini``:

* com1Exe - path to com1DFA executable
* flagOut - print full model output
* simTypeList - simulation types that shall be performed (options: null, ent, res, entres, available; if multiple, separate by '|'))
* releaseScenario - name of release area scenario shapefile (with or without extension -shp, if multiple, separate by '|')
* flagVarPar - perform parameter variation
* varPar - parameter to be varied
* varParValues - values for parameter variation


Output
---------
The simulation results are saved to: *Outputs/com1DFAOrig* and include:

* raster files of the peak values for pressure, flow thickness and flow velocity (*Outputs/com1DFAOrig/peakFiles*)
* reports of all simulations (*Outputs/com1DFAOrig/reports*)
* log files of all simulations
* experiment log that lists all simulations


To run
--------

.. Attention:: Please refer to the instructions in :ref:`C++ Executable` on how to get the
               necessary C++ executable and setup the correct paths.

* first go to ``AvaFrame/avaframe``
* create an avalanche directory with required input files - for this task you can use :ref:`moduleIn3Utils:Initialize Project`
* copy ``avaframeCfg.ini`` to ``local_avaframeCfg.ini`` and set your desired avalanche directory name
* run:
  ::

    pixi run python com1DFAOrig/runCom1DFA.py


Theory
--------


The governing equations of the dense flow avalanche are derived from the
incompressible mass and momentum balance on a Lagrange control volume
(:cite:`Zw2000,ZwKlSa2003`). Assuming the avalanche is much longer and larger
than thick, it is possible to integrate the governing equations over the thickness
of the avalanche and operate some simplifications due to the shape of the avalanche.
This leads, after some calculation steps described in details in Theory
:ref:`theoryCom1DFA:Governing Equations for the Dense Flow Avalanche` to:

.. math::
    \begin{aligned}
    &\frac{\mathrm{d}V(t)}{\mathrm{d}t} = \frac{\mathrm{d}(A_b\overline{h})}{\mathrm{d}t}
    = \frac{\rho_{\text{ent}}}{\rho_0}\,w_f\,h_{\text{ent}}\,\left\Vert \overline{\mathbf{u}}\right\Vert\\
    &\frac{\,\mathrm{d}\overline{u}_i}{\,\mathrm{d}t} =
    g_i + \frac{K_{(i)}}{\overline{\rho}\,A\,\overline{h}}\,\oint\limits_{\partial{A}}\left(\frac{\overline{h}\,\sigma^{(b)}}{2}\right)n_i\,\mathrm{d}l
    -\delta_{i1}\frac{\tau^{(b)}}{\overline{\rho}\,\overline{h}} - C_{\text{res}}\,\overline{\mathbf{u}}^2\,\frac{\overline{u_i}}{\|\overline{\mathbf{u}}\|}
    -\frac{\overline{u_i}}{A\,\overline{h}}\frac{\,\mathrm{d}(A\,\overline{h})}{\,\mathrm{d}t} + \frac{F_i^{\text{ent}}}{\overline{\rho}\,A\,\overline{h}}\\
    &\overline{\sigma}^{(b)}_{33} = \rho\,\left(g_3-\overline{u_1}^2\,\frac{\partial^2{b}}{\partial{x_1^2}}\right)\,\overline{h}
    \end{aligned}


Numerics
---------

Those equations are solved numerically using a **SPH** method (:cite:`LiLi2010,Sa2007`).
**SPH**  is a mesh free method where the basic idea is to divide the avalanche into
small mass particles. The particles interact with each other according to the
equation of motion described in :ref:`moduleCom1DFA:Theory` and the chosen kernel function.
This kernel function describes the domain of influence of a particle (through the smoothing length parameter).
See theory :ref:`theoryCom1DFA:Numerics` for further details.
