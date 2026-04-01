Modulation and Detection
========================

Modulation
----------

Our simulation framework assumes an equivalent discrete-time baseband model. Therefore, the modulation process only consists of mapping the bits to symbols.

Documentation
^^^^^^^^^^^^^

.. autoclass:: mu_mimo.processing.modulation.Mapper
    :no-index:

.. autoclass:: mu_mimo.processing.modulation.NeutralMapper
    :show-inheritance:
    :no-index:

.. autoclass:: mu_mimo.processing.modulation.GrayCodeMapper
    :show-inheritance:
    :no-index:

Detection
---------

The detection process consists of three steps. First, the scaled decision variables passed through the equalizer the obtain the decision variables. Then, the detector maps the decision variables to the corresponding constellation points. Finally, the demapper maps the constellation points to bits.

Documentation
^^^^^^^^^^^^^

.. autoclass:: mu_mimo.processing.modulation.Equalizer
    :no-index:

.. autoclass:: mu_mimo.processing.modulation.Detector
    :no-index:

.. autoclass:: mu_mimo.processing.modulation.NeutralDetector
    :show-inheritance:
    :no-index:

.. autoclass:: mu_mimo.processing.modulation.MDDetector
    :show-inheritance:
    :no-index:

.. autoclass:: mu_mimo.processing.modulation.Demapper
    :no-index:

.. autoclass:: mu_mimo.processing.modulation.NeutralDemapper
    :show-inheritance:
    :no-index:

.. autoclass:: mu_mimo.processing.modulation.GrayCodeDemapper
    :show-inheritance:
    :no-index:


See Also
--------

- :doc:`detailed_documentation/modulation_and_detection`
- :doc:`../../mu_mimo`

.. toctree::
   :hidden:
   
   detailed_documentation/modulation_and_detection