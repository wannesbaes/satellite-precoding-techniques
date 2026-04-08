Base Station
============

The :class:`~mu_mimo.core.system.BaseStation` class is responsible for properly initializing the precoder (and the combiners in case of coordinated beamforming) and for transmitting the data streams to the user terminals.

.. figure:: /_static/mu-mimo/figures/base_station.png
   :width: 60%
   :align: center
   :alt: base station block diagram


Documentation
-------------

.. autoclass:: mu_mimo.core.system.BaseStation()
   :no-index:
   
   .. autosummary::
      :nosignatures:

      ~BaseStation.clear_state
      ~BaseStation.transmit_pilots
      ~BaseStation.receive_feedback
      ~BaseStation.transmit_feedforward
      ~BaseStation.transmit


See Also
--------

- :doc:`detailed_documentation/base_station`
- :doc:`mu_mimo_system`

.. toctree::
   :hidden:
   
   detailed_documentation/base_station