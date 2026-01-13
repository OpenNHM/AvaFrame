Advanced Installation (Windows)
------------------------------

This is a quick guide on how to install AvaFrame
and the required dependencies on your machine. AvaFrame is developed on **Linux
machines** (Ubuntu/Manjaro/Arch) with recent Python versions > 3.8.
These instructions assume you are familiar with working in a terminal. This
guide is described for **Windows**. For *Linux*, see :ref:`developinstall:Advanced Installation (Linux)`.

Requirements
^^^^^^^^^^^^

Install `git <https://github.com/git-guides/install-git>`_ and `pixi <https://pixi.sh/latest/#installation>`_.

Install `Microsoft C++ compiler <https://wiki.python.org/moin/WindowsCompilers>`_.
Follow the installation steps for the version corresponding to the installed python version.

Setup AvaFrame
^^^^^^^^^^^^^^

Clone the AvaFrame repository (in a directory of your choice: [YOURDIR]) and change into it::

  cd [YOURDIR]
  git clone https://github.com/OpenNHM/AvaFrame.git
  cd AvaFrame



Run pixi::

  pixi shell

.. Note::
    If you get some access denied error in Powershell, you might need to run the command

    ``Set-ExecutionPolicy -ExecutionPolicy Unrestricted -Scope CurrentUser``

Compile the cython com1DFA part::

   python setup.py build_ext --inplace

.. Warning::
   You will have to do this compilation every time something changes in the cython code. We also suggest
   to do this everytime updates from the repositories are pulled.

   **Before** compilation in Windows, make sure to delete ``AvaFrame/build`` directory, in addition to any .pyc, .c, and
   .pycache files in ``AvaFrame/avaframe/com1DFA``

This installs avaframe in editable mode, so every time you import avaframe the
current (local) version will be used.

If you want to have the lastet stable release instead, run::

  pixi shell --environment prod


Test it by starting ``python`` and do an ``import avaframe``. If no error comes
up, you are good to go.

Head over to :ref:`gettingstarted:First run` for the next steps.

Using QGIS from with an advanced installation
^^^^^^^^^^^^^^

Assuming you followed the steps above, you should get a working QGis installation by running (it will take a while)::

  pixi shell --environment qgis


An to run QGIS::

  qgis

Depending on your mode of QGis installation (direct installer, conda, OSGeo4W...) your script AvaFrame installation
might be overruled...

Now you can **install the AvaFrameConnector plugin** via QGIS as per usual (:ref:`installation:Operational Installation`).
