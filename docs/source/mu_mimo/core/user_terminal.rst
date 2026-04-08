User Terminal
=============

The :class:`~mu_mimo.core.system.UserTerminal` class is responsible for receiving the data streams from the base station and for properly initializing the combiners in case of non-coordinated beamforming.

.. figure:: /_static/mu-mimo/figures/user_terminal.png
   :width: 60%
   :align: center
   :alt: user terminal block diagram


Documentation
-------------

.. autoclass:: mu_mimo.core.system.UserTerminal()
   :no-index:

   .. autosummary::
      :nosignatures:

      ~UserTerminal.clear_state
      ~UserTerminal.receive_pilots
      ~UserTerminal.transmit_feedback
      ~UserTerminal.receive_feedforward
      ~UserTerminal.receive


See Also
--------

- :doc:`detailed_documentation/user_terminal`
- :doc:`mu_mimo_system`

.. toctree::
   :hidden:
   
   detailed_documentation/user_terminal
