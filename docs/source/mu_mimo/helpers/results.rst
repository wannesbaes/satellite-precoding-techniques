Results
=======

The :mod:`~mu_mimo.core.results` module provides the main interface for managing the simulation results.
It contains a data class :class:`~mu_mimo.core.results.SimResult` that encapsulates the performance metrics of a complete simulation run and a class :class:`~mu_mimo.core.results.SimResultManager` that provides methods to save, load, and analyze the simulation results.

Documentation
-------------

.. autoclass:: mu_mimo.core.results.SimResult()
   :members:
   :no-index:

.. autoclass:: mu_mimo.core.results.SimResultManager
   :no-index:

   .. autosummary::
      :nosignatures:

      ~SimResultManager.search_results
      ~SimResultManager.load_results
      ~SimResultManager.save_results
      
      ~SimResultManager.display
      ~SimResultManager.plot_system_performance
      ~SimResultManager.plot_ut_performance
      ~SimResultManager.plot_stream_performance
      ~SimResultManager.plot_system_performance_comparison
      ~SimResultManager.plot_ut_performance_comparison


See Also
--------

- :doc:`detailed_documentation/single_snr_result`
- :doc:`detailed_documentation/result_manager`

.. toctree::
   :hidden:
   
   detailed_documentation/single_snr_result
   detailed_documentation/result_manager
