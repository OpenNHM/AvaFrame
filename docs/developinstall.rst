Script Installation (Linux)
------------------------------

This is a quick guide on how to install AvaFrame
and the required dependencies on your machine. AvaFrame is developed on **Linux
machines** (Ubuntu/Manjaro/Arch) with recent Python versions > 3.8.
These instructions assume you are familiar with working in a terminal. This
guide is described for **Linux**. For *Windows*, see :ref:`developinstallwin:Script Installation (Windows)`.

Requirements
^^^^^^^^^^^^

Install `git <https://github.com/git-guides/install-git>`_ and `pixi <https://pixi.sh/latest/#installation>`_.
Some operating systems might require the python headers (e.g python-dev on ubuntu) or other supporting
libraries/packages (e.g. Visual Studio on Windows needs the c++ compiler components).

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


Compile the cython com1DFA part. You might also have to install a c-compiler (gcc or similar) through your systems
package manager::

  pixi run build

.. Warning::
   You will have to do this compilation every time something changes in the cython code. We also suggest
   to do this everytime updates from the repositories are pulled. Use ``pixi run rebuild`` to clean
   and rebuild in one step.

All this installs avaframe in editable mode, so every time you import avaframe the
current (local) version will be used.

Test it by running::

  pixi run python -c "import avaframe"

If you want to use the lastet stable release instead, run::

  pixi run --environment prod python -c "import avaframe"

If no error comes up, you are good to go.

To see the current version, you can use::

  pixi list avaframe

Head over to :ref:`gettingstarted:First run` for the next steps.