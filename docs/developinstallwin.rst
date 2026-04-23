Script Installation (Windows)
------------------------------

This is a quick guide on how to install AvaFrame
and the required dependencies on your machine. AvaFrame is developed on **Linux
machines** (Ubuntu/Manjaro/Arch) with recent Python versions > 3.8.
These instructions assume you are familiar with working in a terminal. This
guide is described for **Windows**. For *Linux*, see :ref:`developinstall:Script Installation (Linux)`.

Requirements
^^^^^^^^^^^^

Install `git <https://github.com/git-guides/install-git>`_ and `pixi <https://pixi.sh/latest/#installation>`_.

Install `Microsoft C++ compiler <https://wiki.python.org/moin/WindowsCompilers>`_.
Follow the installation steps for the version corresponding to the installed python version.

.. Note::
   This documentation uses pixi environments. You can either prefix commands with ``pixi run <command>``,
   which runs the command in the pixi environment without requiring you to activate it first, or you
   can run ``pixi shell`` once to start an interactive environment and then run the commands without
   the ``pixi run`` prefix (e.g. ``pixi run python script.py`` is equivalent to ``pixi shell`` then
   ``python script.py``).

Setup AvaFrame
^^^^^^^^^^^^^^

Clone the AvaFrame repository (in a directory of your choice: [YOURDIR]) and change into it::

  cd [YOURDIR]
  git clone https://github.com/OpenNHM/AvaFrame.git
  cd AvaFrame

.. Note::
    If you get some access denied error in Powershell, you might need to run the command

    ``Set-ExecutionPolicy -ExecutionPolicy Unrestricted -Scope CurrentUser``

Compile the cython com1DFA part::

   pixi run build

.. Warning::
   You will have to do this compilation every time something changes in the cython code. We also suggest
   to do this everytime updates from the repositories are pulled. Use ``pixi run rebuild`` to clean
   and rebuild in one step.

   **Before** compilation in Windows, make sure to delete the ``AvaFrame/build`` directory.

This installs avaframe in editable mode, so every time you import avaframe the
current (local) version will be used.

Test it by running::

  pixi run python -c "import avaframe"

If you want to use the lastet stable release instead, run::

  pixi run --environment prod python -c "import avaframe"

If no error comes up, you are good to go.

Head over to :ref:`gettingstarted:First run` for the next steps.

Using QGIS from within an script installation
---------------------------------------------

To run QGIS (this will install the environment on first use)::

  pixi run --environment qgis qgis

Depending on your mode of QGis installation (direct installer, conda, OSGeo4W...) your script AvaFrame installation
might be overruled...

Now you can **install the AvaFrameConnector plugin** via QGIS as per usual (:ref:`standardUsage:Standard Usage (QGis)`).
