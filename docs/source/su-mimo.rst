SU-MIMO
=======

This module contains the implementation of a Single-User MIMO digital communication system, where the channel state information (CSI) is available at both the transmitter and receiver.

The channel is diagonalized by using the Singular Value Decomposition (SVD) of the channel for precoding and combining. 
The implementation is designed to simulate the performance for different system configurations (modulation schemes, resource allocation strategies, antenna counts, etc.).

.. toctree::
   :maxdepth: 1
   :caption: Components:

   su-mimo/transmitter
   su-mimo/channel
   su-mimo/receiver
   su-mimo/resource_allocation
   su-mimo/su_mimo_system