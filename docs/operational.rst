Manual Installation 
===================

This is the quick start for the manual standard AvaFrame setup with QGis as
frontend. Currently described for **Windows** only (Linux users can use any terminal to go through the same steps below)

Requirements
^^^^^^^^^^^^

The prerequisites are:

* QGis: install from here: `QGis installation <https://qgis.org/en/site/forusers/download.html>`_ (we recommend
  using the latest version)

Setup AvaFrame
^^^^^^^^^^^^^^

#. Open the **OSGeo4W Shell** from your start menu (included in any Windows QGis installation)

   * If you have multiple QGis versions on your system you have to choose the OSGeo4W Shell that matches your QGis version. 
     I.e. if you are using QGis 3.22, make sure you using the corresponding shell!


    .. figure:: _static/OSGeo4WShell.png
            :align: center
            :width: 50%

#. Activate the python 3 environment by putting the following command into the shell. **On newer versions of QGis (>3.18) this is not needed anymore, just skip it**:

    .. code-block:: bash

      py3_env

#. Install AvaFrame by putting the following command into the shell:

    .. code-block::

      python3 -m pip install --user avaframe

#. To avoid a numpy error (see note at bottom of page), update numpy and pandas:

    .. code-block::
     
      python3 -m pip install --user --upgrade numpy pandas


Setup QGis and run
^^^^^^^^^^^^^^^^^^

#. Open QGis from your start menu and go to Plugins -> Manage and Install Plugins

#. Search for `AvaFrameConnector` and install it

#. Access the QGis - Avaframe connector via Toolbox ->  AvaFrame -> AvaFrameConnector

#. Add the described data and run. Results will be loaded after a while
   (depending on the size of your DEM).



Update Avaframe to a new release
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#. Open the **OSGeo4W Shell** from your start menu (included in any Windows QGis installation)

   * If you have multiple QGis versions on your system you have to choose the OSGeo4W Shell that matches your QGis version. 
     I.e. if you are using QGis 3.22, make sure you using the corresponding shell! (see point 1 of the *Setup AvaFrame* above)

#. Activate the python 3 environment by putting the following command into the shell. **On newer versions of QGis (>3.18) this is not needed anymore, just skip it**:

    .. code-block:: bash

      py3_env

#. Update AvaFrame by putting the following command into the shell:

    .. code-block::

      python3 -m pip install -U --user avaframe

#. To avoid a numpy error (see note at bottom of page), update numpy and pandas:

    .. code-block::
     
      python3 -m pip install --user --upgrade numpy pandas


#. Restart/Open QGis from your start menu and go to Plugins -> Manage and Install Plugins

#. Search for `AvaFrameConnector` and check whether it also needs updating


.. Note::
   If you encounter an error similar to this (lowest/last line of the error message; the numbers might differ)::

      ValueError: numpy.ndarray size changed, may indicate binary
      incompatibility. Expected 88 from C header, got 80 from PyObject

   run the following in OSGeo4W shell (the *py3_env* command is not needed on newer versions of QGis, skip it)::

     py3_env
     python3 -m pip install --user --upgrade numpy pandas

   and restart QGis.
