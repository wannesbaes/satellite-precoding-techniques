Receiver
========

This page documents the receiver component of the SU-MIMO digital communication system, in which the CSI is available at the receiver (and the transmitter).


When the receiver is called and given a received input signal y, it performs the following operations:

* determines (or retrieves) the power allocation and constellation size on each receive antenna [:meth:`~receiver.Receiver.resource_allocation`]
* combines the received signal using the left singular vectors of the channel matrix H [:meth:`~receiver.Receiver.combiner`]
* deallocates the power from the scaled decision variables [:meth:`~receiver.Receiver.power_deallocator`]
* searches the most probable transmitted data symbol vectors based on the decision variables [:meth:`~receiver.Receiver.detector`]
* demaps them into bit vectors [:meth:`~receiver.Receiver.demapper`]
* and combines the reconstructed bits of each data stream to create the output bitstream [:meth:`~receiver.Receiver.bit_deallocator`].


.. figure:: /_static/su-mimo/figures/receiver.png
   :width: 80%
   :align: center
   :alt: receiver block diagram
   
   Figure: block diagram of the receiver component in the SU-MIMO DigCom system.


Documentation
-------------

.. autoclass:: receiver.Receiver
   :members:
   :undoc-members:
   :show-inheritance:
