Installation 
============

.. Note::
  There are three ways to install and use AvaFrame:

  Standard (QGis)
    If you want the standard workflow, running *com1DFA* (dense flow avalanche) and *com2AB* (alpha beta)
    with standard settings from within *QGis*, head over to :ref:`standardUsage:Standard Usage (QGis)`
    (:ref:`Deutsche Version<standardUsage:Standardinstallation (Deutsch)>`).
    Use this if you want to:

    - use the standard, well tested and calibrated setup for hazard mapping or similar
    - use QGis as frontend
    - have results with an easier setup
    - use the latest stable release


  Advanced (QGis, Script)
    If you want to use project-specific expert configuration files to override default settings,
    head over to :ref:`advancedUsage:Advanced Usage (QGis, Script)`.
    Use this if you want to:

    - use QGis with customized simulation parameters
    - override default module configurations per project
    - use expert configuration files in ``Inputs/CFGs/``

    .. Warning::
       Modifying default parameters moves you outside the validated and calibrated setup.
       Results are no longer covered by the standard validation and you use them at your own risk.


  Complex (Script)
    If you want to us a script-based workflow, or contribute and develop AvaFrame, head over
    to :ref:`complexUsage:Complex Usage (Script)`.
    Use this if you want to:

    - work on the code itself
    - implement new features
    - change/improve existing code
    - have the latest development code. *Warning: might be unstable!*
