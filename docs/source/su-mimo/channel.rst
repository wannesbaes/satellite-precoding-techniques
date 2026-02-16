Channel
=======

This page documents the channel component of the SU-MIMO digital communication system.

The channel is modeled as a Rayleigh slow-fading channel, i.e. the elements of the channel matrix are independent and identically distributed (i.i.d.) complex Gaussian random variables with zero mean and unit variance and remain constant over the duration of at least one transmission block. The channel can also be initialized with a user-provided channel matrix.

In addition, the channel adds complex proper, circularly-symmetric additive white Gaussian noise (AWGN) to the transmitted symbols, based on a specified signal-to-noise ratio (SNR) (in dB).


When the channel is called and given a transmitted signal, it performs the following operations [:meth:`~channel.Channel.simulate`]:

* Transmit the precoded symbols through the MIMO channel.
* Add noise to the received symbols.

.. figure:: /_static/su-mimo/figures/channel.png
   :width: 80%
   :align: center
   :alt: channel block diagram

   Figure: block diagram of the channel component in the SU-MIMO DigCom system.


Documentation
-------------

.. autoclass:: channel.Channel
   :members:
   :undoc-members:
   :show-inheritance: