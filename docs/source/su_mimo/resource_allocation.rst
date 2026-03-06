Resource Allocation
===================

This module contains the implementation of the considered resource allocation techniques for a SU-MIMO system, including power allocation and bit allocation strategies. 

It also includes functions to visualize the optimal power allocation across the data streams (each corresponding to an eigenchannel) using waterfilling, as well as the capacity and information bit rate of the data streams for different SNR values and power allocation strategies.


Documentation
-------------

.. autosummary::
   :toctree: ../_autosummary/resource_allocation/
   :nosignatures:

   ~su_mimo.resource_allocation.resource_allocation.waterfilling_v1
   ~su_mimo.resource_allocation.resource_allocation.equal_power_allocation
   ~su_mimo.resource_allocation.resource_allocation.eigenbeamforming
   ~su_mimo.resource_allocation.resource_allocation.plot_waterfilling
   ~su_mimo.resource_allocation.resource_allocation.demo_waterfilling
   ~su_mimo.resource_allocation.resource_allocation.adaptive_bit_allocation
   ~su_mimo.resource_allocation.resource_allocation.plot_adaptive_bit_allocation
   ~su_mimo.resource_allocation.resource_allocation.demo_adaptive_bit_allocation
