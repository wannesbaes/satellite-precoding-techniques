Datatypes (Detailed Documentation)
==================================

This page provides a complete documentation of the custom datatypes used in the simulation framework.

System Configuration
""""""""""""""""""""

.. autoclass:: mu_mimo.configs.BaseStationConfig

.. autoclass:: mu_mimo.configs.ChannelConfig

.. autoclass:: mu_mimo.configs.UserTerminalConfig

.. autoclass:: mu_mimo.configs.SystemConfig
   :members:

.. autoclass:: mu_mimo.configs.SimConfig
   :members:

.. autofunction:: mu_mimo.configs.setup_sys_configs

.. autofunction:: mu_mimo.configs.setup_sim_configs


System State
""""""""""""

.. autoclass:: mu_mimo.types.ChannelStateInformation

.. autoclass:: mu_mimo.types.ChannelState

.. autoclass:: mu_mimo.types.BaseStationState

.. autoclass:: mu_mimo.types.UserTerminalState

Setup Messages
""""""""""""""

.. autoclass:: mu_mimo.types.TransmitPilotMessage

.. autoclass:: mu_mimo.types.ReceivePilotMessage

.. autoclass:: mu_mimo.types.TransmitFeedbackMessage

.. autoclass:: mu_mimo.types.ReceiveFeedbackMessage

.. autoclass:: mu_mimo.types.TransmitFeedforwardMessage

.. autoclass:: mu_mimo.types.ReceiveFeedforwardMessage