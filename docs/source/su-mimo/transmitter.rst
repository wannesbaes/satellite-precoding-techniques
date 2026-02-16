Transmitter
===========

This page documents the transmitter component of the SU-MIMO digital communication system, in which the CSI is available at the transmitter (and the receiver).


When the transmitter is called and given an input bit sequence, it performs the following operations [:meth:`~transmitter.Transmitter.simulate`]:

* Determines the power allocation and bit allocation for each bit stream, based on the given resource allocation strategy (RAS) [:meth:`~transmitter.Transmitter.resource_allocation`]
* Distributes the input bits accross the data streams [:meth:`~transmitter.Transmitter.bit_allocator`]
* Maps the input bit sequences to the corresponding data symbol sequences according to a specified modulation constellation for each data stream [:meth:`~transmitter.Transmitter.mapper`]
* Allocates power across data streams using [:meth:`~transmitter.Transmitter.power_allocator`]
* Precodes the data symbols using the right singular vectors of the channel matrix [:meth:`~transmitter.Transmitter.precoder`]


.. figure:: /_static/su-mimo/figures/transmitter.png
   :width: 80%
   :align: center
   :alt: transmitter block diagram

   Figure: block diagram of the transmitter component in the SU-MIMO DigCom system.


Documentation
-------------

.. autoclass:: transmitter.Transmitter
   :members:
   :undoc-members:
   :show-inheritance:
