Channel
=======

The :class:`~mu_mimo.core.system.Channel` is responsible for propagating the transmitted signal from the base station to the user terminals. It models the physical channel properties.

.. figure:: /_static/mu-mimo/figures/channel.png
   :width: 60%
   :align: center
   :alt: channel block diagram


Documentation
-------------

.. autoclass:: mu_mimo.core.system.Channel()
   :no-index:

   .. autosummary::
      :nosignatures:

      ~Channel.reset
      ~Channel.proceed
      ~Channel.propagate_pilots
      ~Channel.propagate_feedback
      ~Channel.propagate_feedforward
      ~Channel.propagate


See Also
--------

- :doc:`detailed_documentation/channel`
- :doc:`mu_mimo_system`

.. toctree::
   :hidden:
   
   detailed_documentation/channel
