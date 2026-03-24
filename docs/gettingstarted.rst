.. Basic usage
.. -----------

.. .. Note::

   .. These instructions target the development installation. If you are using the standard / operational installation we
   .. provide a recording from the first public `AvaFrame workshop <https://youtu.be/HOFygSkCmp0>`_ . You will find a demo
   .. starting at around the 17:25min mark.  


First run
---------

Follow these steps to run your first simulation (assuming you finished the advanced installation steps; if you come
back later, make sure to run ``pixi shell`` again to activate the environment):

* change into your ``AvaFrame`` directory (replace [YOURDIR]
  with your path from the installation steps)::

    cd [YOURDIR]/AvaFrame/avaframe

* run:
  ::

    python runCom1DFA.py

* a similar output should show up:
  ::

    logUtils - INFO -  Started logging at: 03.11.2020 22:42:04
    logUtils - INFO -  Also logging to: data/avaParabola/runCom1DFA.log
    runCom1DFA - INFO -  MAIN SCRIPT
    runCom1DFA - INFO -  Current avalanche: data/avaParabola
    ...


This will perform a dense flow avalanche simulation using the ``com1DFA`` module.
The results are saved to ``data/avaParabola/Outputs/com1DFA``. For a first
look at the results, got to the folder ``reports``, there you can find
a markdown report of the simulations performed including some plots of the results.

To display markdown files in a nice way use a markdown viewer of your choice.
Some other options are:

* Use the Atom editor with a markdown plugin
* If you have *pandoc* installed use this to convert it to pdf/html
* Some browsers have markdown extensions you can install easily


Workflow example
----------------

The following example should make it easier for you to find your way in AvaFrame and setup your own AvaFrame workflow
after you did the full setup. There is also a directory with examples for different workflows, see more here:
:ref:`complexUsage:Example runScripts`.

Make sure you change to your ``AvaFrame`` directory by::

  cd [YOURDIR]/AvaFrame

Replace ``[YOURDIR]`` with the directory from your installation step.

Initialize project
^^^^^^^^^^^^^^^^^^

To create the folder where the input data lies and where the
output results will be saved, specify the full path to the folder
in the ``local_avaframeCfg.ini`` (which is a copy of
``avaframeCfg.ini`` that you need to create). So::

  cd avaframe
  cp avaframeCfg.ini local_avaframeCfg.ini

and edit ``local_avaframeCfg.ini`` with your favorite text editor and adjust the
variable ``avalancheDir``.

Then run ::

  python runScripts/runInitializeProject.py

This will create a new directory with the input required by AvaFrame structured as described
in :ref:`moduleIn3Utils:Initialize Project`.

Input data
^^^^^^^^^^

Check the input data required by the different modules you want to use and fill
the ``Inputs/`` inside the ``[avalancheDir]`` folder from the initialize step accordingly.

For example the ``com1DFA`` module needs input as described in
:ref:`moduleCom1DFA:Input`. You can also have a look at the default setting for
the module you want to use (for example ``com1DFACfg.ini`` for module com1DFA).
If you want to use different settings, create a ``local_`` copy of the ``.ini``
file and modify the desired parameters.

More information about the configuration can be found here: :ref:`complexUsage:Configuration`

Building your run script
^^^^^^^^^^^^^^^^^^^^^^^^

Create your own workflow by taking the ``runOperational.py`` script as
template.

We suggest you copy it and adjust it to your liking. There are annotations in
the code that should help you to understand the structure.

A lot more examples can be found in the ``runScripts`` directory (see also :ref:`complexUsage:Example runscripts`).

