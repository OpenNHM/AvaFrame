Advanced Installation (Linux)
------------------------------

This is a quick guide on how to install AvaFrame
and the required dependencies on your machine. AvaFrame is developed on **Linux
machines** (Ubuntu/Manjaro/Arch) with recent Python versions > 3.8.
These instructions assume you are familiar with working in a terminal. This
guide is described for **Linux**. For *Windows*, see :ref:`developinstallwin:Advanced Installation (Windows)`.

Requirements
^^^^^^^^^^^^

Install `git <https://github.com/git-guides/install-git>`_ and `pixi <https://pixi.sh/latest/#installation>`_.
Some operating systems might require the python headers (e.g python-dev on ubuntu) or other supporting
libraries/packages (e.g. Visual Studio on Windows needs the c++ compiler components).


Setup AvaFrame
^^^^^^^^^^^^^^

Clone the AvaFrame repository (in a directory of your choice: [YOURDIR]) and change into it::

  cd [YOURDIR]
  git clone https://github.com/OpenNHM/AvaFrame.git
  cd AvaFrame


Run pixi::

  pixi shell

Compile the cython com1DFA part. You might also have to install a c-compiler (gcc or similar) through your systems
package manager::

  python setup.py build_ext --inplace

.. Warning::
   You will have to do this compilation every time something changes in the cython code. We also suggest
   to do this everytime updates from the repositories are pulled.

All this installs avaframe in editable mode, so every time you import avaframe the
current (local) version will be used.

If you want to have the lastet stable release instead, run::

  pixi shell --environment prod

Test it by starting ``python`` and do an ``import avaframe``. If no error comes
up, you are good to go.

To see the current version, you can use::

  pixi list avaframe

Head over to :ref:`gettingstarted:First run` for the next steps.