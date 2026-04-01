Datatypes
=========

This page provides an overwiew of the costum datatypes used in the simulation framework.

System Configuration
""""""""""""""""""""

.. autosummary::
   :nosignatures:

   ~mu_mimo.configs.BaseStationConfig
   ~mu_mimo.configs.ChannelConfig
   ~mu_mimo.configs.UserTerminalConfig
   ~mu_mimo.configs.SystemConfig
   ~mu_mimo.configs.SimConfig

.. autosummary::
   :nosignatures:

   ~mu_mimo.configs.setup_sys_configs
   ~mu_mimo.configs.setup_sim_configs

System State
""""""""""""

.. autosummary::
   :nosignatures:

   ~mu_mimo.types.ChannelStateInformation
   ~mu_mimo.types.ChannelState
   ~mu_mimo.types.BaseStationState
   ~mu_mimo.types.UserTerminalState

Setup Messages
""""""""""""""

.. autosummary::
   :nosignatures:

   ~mu_mimo.types.TransmitPilotMessage
   ~mu_mimo.types.ReceivePilotMessage
   ~mu_mimo.types.TransmitFeedbackMessage
   ~mu_mimo.types.ReceiveFeedbackMessage
   ~mu_mimo.types.TransmitFeedforwardMessage
   ~mu_mimo.types.ReceiveFeedforwardMessage


See Also
--------

- :doc:`detailed_documentation/data_types`

.. toctree::
   :hidden:
   
   detailed_documentation/data_types
