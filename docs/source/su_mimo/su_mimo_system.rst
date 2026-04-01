SU-MIMO System
==============

This page documents the SU-MIMO DigCom system itself.
 
It consists of a :doc:`transmitter <transmitter>` component, a :doc:`channel <channel>` component and a :doc:`receiver <receiver>` component.

.. figure:: /_static/su-mimo/figures/su_mimo_system.png
   :width: 70%
   :align: center
   :alt: SU-MIMO system block diagram

   Figure: block diagram of the SU-MIMO DigCom system.

Documentation
-------------

.. autosummary::
   :nosignatures:

   ~su_mimo.su_mimo_svd.SuMimoSVD.set_CSI
   ~su_mimo.su_mimo_svd.SuMimoSVD.reset_CSI
   ~su_mimo.su_mimo_svd.SuMimoSVD.set_RAS
   ~su_mimo.su_mimo_svd.SuMimoSVD.simulate
   ~su_mimo.su_mimo_svd.SuMimoSVD.BERs_simulation
   ~su_mimo.su_mimo_svd.SuMimoSVD.BERs_eigenchs_simulation
   ~su_mimo.su_mimo_svd.SuMimoSVD.BERs_analytical
   ~su_mimo.su_mimo_svd.SuMimoSVD.plot_scatter_diagram
   ~su_mimo.su_mimo_svd.SuMimoSVD.print_simulation_example

See Also
--------

- :doc:`detailed_documentation/su_mimo_system`
- :doc:`../su_mimo`

.. toctree::
   :hidden:

   detailed_documentation/su_mimo_system