MU-MIMO
=======

This module provides a simulation framework for multi-user MIMO digital communication systems,
designed to evaluate and compare different precoding strategies and channel models in a structured way.


Model Architecture
------------------

The simulation is built around a modular class hierarchy that separates the core system components from the interchangeable processing algorithms. The :doc:`simulation runner <mu-mimo/core/simulation_runner>` orchestrates the simulation loop, driving a :doc:`MU-MIMO system <mu-mimo/core/mu_mimo_system>` that encapsulates a :doc:`base station (BS) <mu-mimo/core/base_station>`, a set of :doc:`user terminal (UT) <mu-mimo/core/user_terminal>` instances, and a :doc:`channel <mu-mimo/core/channel>`.
Each core component delegates its signal processing to abstract processing classes, which can be swapped out independently to test different algorithms, as shown in the class diagram below.

.. figure:: /_static/mu-mimo/figures/class_diagram.png
   :width: 100%
   :align: center
   :alt: Simulation Model Class Diagram

   Figure: class diagram of the simulation model.


Each simulation run iterates over a range of SNR values, with multiple channel realizations per SNR point.
Every iteration progresses through three phases:

1. **Reset:** The system state is cleared and a new channel realization is generated.
2. **Configuration:** Precoding and combining matrices are computed from the current channel state, alongside the resource allocation according to the specified strategy.
3. **Data Transmission:** The base station transmits data, the channel propagates it, and each user terminal detects and decodes its received signal.

After every single iteration, performance metrics are computed and stored for analysis later on.

These phases are illustrated in the :download:`sequence diagram </_static/mu-mimo/figures/sequence_diagram.png>`, in which the interactions between the different components become clear as well.


Example Usage
-------------

.. code-block:: python

   from mu_mimo.datatypes import SystemConfig, SimConfig, SimResult
   from mu_mimo.core.simulation_runner import SimulationRunner

   # Define the system configuration settings
   system_configs = SystemConfig(
       pass
   )

   # Define the simulation configuration settings
   sim_configs = SimConfig(
      pass
   )

   # Create and run the simulation
   runner = SimulationRunner(sim_configs, system_configs)
   results = runner.run()


Documentation
-------------

The complete documentation of the MU-MIMO module is found in the following sections.

Core Components
^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   mu-mimo/core/simulation_runner.rst
   mu-mimo/core/datatypes.rst

   mu-mimo/core/mu_mimo_system.rst
   mu-mimo/core/base_station.rst
   mu-mimo/core/user_terminal.rst
   mu-mimo/core/channel.rst


Processing Components
^^^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1
   
   mu-mimo/processing/precoding_and_combining.rst
   mu-mimo/processing/resource_allocation.rst
   mu-mimo/processing/channel.rst
   mu-mimo/processing/modulation_and_detection.rst