Manuelle Installation
=====================

Dies ist der Schnellstart für die Inbetriebnahme von AvaFrame mit QGis als Frontend. Derzeit nur für Windows beschrieben 
(Linux-Benutzer können ein beliebiges Terminal verwenden, um die gleichen Schritte durchzugehen)

Vorraussetzungen
^^^^^^^^^^^^^^^^

Die Vorraussetzungen sind:

* QGis: `QGis installation <https://qgis.org/de/site/forusers/download.html>`_ (Wir empfehlen die aktuellste Version zu 
  verwenden)

Setup AvaFrame
^^^^^^^^^^^^^^

#. Die **OSGeo4W Shell** aus dem Windows Startmenu aufrufen (ist bei jeder Windows QGis Installation inkludiert)

   * Wenn Sie mehrere QGis Versionen auf Ihrem System haben, müssen Sie die OSGeo4W Shell wählen, die zu Ihrer 
     QGis Version passt. D.h. wenn Sie QGis 3.22 verwenden, stellen Sie sicher, dass Sie die entsprechende Shell verwenden!

    .. figure:: _static/OSGeo4WShell.png
            :align: center
            :width: 50%

#. Aktivieren Sie die Python 3 Umgebung, indem Sie den folgenden Befehl in die Shell eingeben. 
   **Auf neueren Versionen von QGis (>3.18) ist dies nicht mehr nötig**:

    .. code-block:: bash

      py3_env

#. Installieren Sie AvaFrame, indem Sie den folgenden Befehl in die Shell eingeben:

    .. code-block::

      python -m pip install --user avaframe

#. Um einen potentiellen Fehler zu vermeiden (siehe Note unten), numpy und pandas mittels folgendem Befehl updaten:

    .. code-block::
     
      python -m pip install --user --upgrade numpy pandas


Setup QGis 
^^^^^^^^^^

#. Öffnen Sie QGis über Ihr Startmenü und gehen Sie zu Erweiterungen -> Erweiterungen verwalten und installieren

#. Nach `OpenNHMConnector` suchen und installieren

#. Rufen Sie den QGis - OpenNHMConnector über Verarbeitungswerkzeuge -> OpenNHM -> OpenNHMConnector auf

#. Fügen Sie die beschriebenen Daten hinzu und starten Sie den Connector. Die Ergebnisse werden nach einer Weile geladen 
   (abhängig von der Größe Ihres DEM).


Update Avaframe 
^^^^^^^^^^^^^^^

#. Die **OSGeo4W Shell** aus dem Windows Startmenu aufrufen (ist bei jeder Windows QGis Installation inkludiert)

   * Wenn Sie mehrere QGis Versionen auf Ihrem System haben, müssen Sie die OSGeo4W Shell wählen, die zu Ihrer 
     QGis Version passt. D.h. wenn Sie QGis 3.22 verwenden, stellen Sie sicher, dass Sie die entsprechende Shell verwenden!

#. Aktivieren Sie die Python 3 Umgebung, indem Sie den folgenden Befehl in die Shell eingeben. 
   **Auf neueren Versionen von QGis (>3.18) ist dies nicht mehr nötig**:

    .. code-block:: bash

      py3_env

#. Aktualisieren Sie AvaFrame, indem Sie den folgenden Befehl in die Shell eingeben:

    .. code-block::

      python -m pip install -U --user avaframe

#. Um einen potentiellen Fehler zu vermeiden (siehe Note unten), numpy und pandas mittels folgendem Befehl updaten:

    .. code-block::
     
      python -m pip install --user --upgrade numpy pandas

#. Starten Sie QGis neu/öffnen Sie es über Ihr Startmenü und gehen Sie zu Plugins -> Plugins verwalten und installieren

#. Suchen Sie nach OpenNHMConnector und prüfen Sie, ob es aktualisiert werden muss


.. Note::
   Wenn Sie auf einen Fehler wie diesen stoßen (unterste/letzte Zeile der Fehlermeldung; die Zahlen können abweichen)::

     ValueError: numpy.ndarray size changed, may indicate binary
     incompatibility. Expected 88 from C header, got 80 from PyObject

   Führen Sie das Folgende in der OSGeo4W-Shell aus (der py3_env-Befehl wird bei neueren Versionen von QGis 
   nicht benötigt, überspringen Sie ihn)::

     py3_env
     python -m pip install --user --upgrade numpy pandas

   und starten Sie QGis neu.


