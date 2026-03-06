SU-MIMO
=======

This module contains the documentation of the single-user MIMO digital communication system, where the channel state information (CSI) is available at both the transmitter and receiver.

The channel is diagonalized by using the Singular Value Decomposition (SVD) of the channel for precoding and combining. 
The implementation is designed to simulate the performance for different system configurations (modulation schemes, resource allocation strategies, antenna counts, etc.).

.. toctree::
   :maxdepth: 1
   :caption: Components:
   
   su_mimo/su_mimo_system
   su_mimo/transmitter
   su_mimo/channel
   su_mimo/receiver
   su_mimo/resource_allocation