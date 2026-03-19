# mu-mimo/main.py

import numpy as np
import matplotlib.pyplot as plt
from mu_mimo import *

simulation_configuration_settings = [

    # Standard Simulation Configution Settings.
    {'number': 0,    'SNR values (in dB)': np.arange(-10, 31, 2.5),    'channel realizations per SNR value': 1000,    'bit errors per SNR value': 250,    'Scope of bit errors': 'system-wide',    'Transmission per channel realization': 512,    'name': 'Sim Config 0',    'description': "Standard Simulation Configution Settings"},

    # Standard Simulation Configution Settings, bit errors per UT and per stream.
    {'number': 1,    'SNR values (in dB)': np.arange(-10, 31, 2.5),    'channel realizations per SNR value': 1000,    'bit errors per SNR value': 250,    'Scope of bit errors': 'uts',            'Transmission per channel realization': 1024,    'name': 'Sim Config 1',    'description': "Standard Simulation Configution Settings (UT-level bit errors counting)"},
    {'number': 2,    'SNR values (in dB)': np.arange(-10, 31, 2.5),    'channel realizations per SNR value': 1000,    'bit errors per SNR value': 250,    'Scope of bit errors': 'streams',        'Transmission per channel realization': 1024,    'name': 'Sim Config 2',    'description': "Standard Simulation Configution Settings (Stream-level bit errors counting)"},

    # Test Simulation Configution Settings for code validation and debugging purposes only.
    {'number': 3,    'SNR values (in dB)': np.arange(-10, 31, 10),    'channel realizations per SNR value': 10,       'bit errors per SNR value': 10,     'Scope of bit errors': 'system-wide',    'Transmission per channel realization': 1,     'name': 'Sim Config 3',    'description': "First Test Simulation Configution Settings"},
    {'number': 4,    'SNR values (in dB)': np.arange(-10, 31, 5),     'channel realizations per SNR value': 100,       'bit errors per SNR value': 50,     'Scope of bit errors': 'system-wide',    'Transmission per channel realization': 512,     'name': 'Sim Config 4',    'description': "Second Test Simulation Configution Settings"},
]

system_configuration_settings = [

    # Reference System 0: SISO reference system for code validation and debugging purposes only.
    {'number': 0,     'Pt': 1.0,    'B': 0.5,    'K': 1,    'Nr': 1,    'Nt': 1,    'constellation types': "PAM",    'constellation sizes': 1,       'capacity fractions': None,    'bit loader': NeutralBitLoader,     'mapper': NeutralMapper,    'precoder': NeutralPrecoder,    'channel model': NeutralChannelModel,        'noise model': NeutralNoiseModel,    'combiner': NeutralCombiner,    'equalizer': Equalizer,    'detector': NeutralDetector,    'demapper': NeutralDemapper,    'name': "Reference System 0",     'description': "SISO system with neutral components only. For code validation and debugging purposes only."},
    
    # Reference Systems 1-4: Non-coordinated Beamforming with ZF Precoder, RSV Combiner and Optimal Power Allocation. Fixed bit loading with 4-QAM constellation for each UT.
    {'number': 1,     'Pt': 1.0,    'B': 0.5,    'K': 1,    'Nr': 8,    'Nt': 8,    'constellation types': "QAM",    'constellation sizes': 2,       'capacity fractions': None,    'bit loader': FixedBitLoader,       'mapper': GrayCodeMapper,    'precoder': ZFPrecoder,        'channel model': IIDRayleighChannelModel,    'noise model': CSAWGNNoiseModel,      'combiner': LSVCombiner,       'equalizer': Equalizer,    'detector': MDDetector,         'demapper': GrayCodeDemapper,    'name': "Reference System 1",     'description': "Non-coordinated Beamforming with ZF Precoder, RSV Combiner and Optimal Power Allocation.\n 8 transmit antennas, 1 UT with 8 receive antennas.\n Fixed bit loading with 4-QAM constellation."},
    {'number': 2,     'Pt': 1.0,    'B': 0.5,    'K': 2,    'Nr': 4,    'Nt': 8,    'constellation types': "QAM",    'constellation sizes': 2,       'capacity fractions': None,    'bit loader': FixedBitLoader,       'mapper': GrayCodeMapper,    'precoder': ZFPrecoder,        'channel model': IIDRayleighChannelModel,    'noise model': CSAWGNNoiseModel,      'combiner': LSVCombiner,       'equalizer': Equalizer,    'detector': MDDetector,         'demapper': GrayCodeDemapper,    'name': "Reference System 2",     'description': "Non-coordinated Beamforming with ZF Precoder, RSV Combiner and Optimal Power Allocation.\n 8 transmit antennas, 2 UTs with 4 receive antennas each.\n Fixed bit loading with 4-QAM constellation."},
    {'number': 3,     'Pt': 1.0,    'B': 0.5,    'K': 4,    'Nr': 2,    'Nt': 8,    'constellation types': "QAM",    'constellation sizes': 2,       'capacity fractions': None,    'bit loader': FixedBitLoader,       'mapper': GrayCodeMapper,    'precoder': ZFPrecoder,        'channel model': IIDRayleighChannelModel,    'noise model': CSAWGNNoiseModel,      'combiner': LSVCombiner,       'equalizer': Equalizer,    'detector': MDDetector,         'demapper': GrayCodeDemapper,    'name': "Reference System 3",     'description': "Non-coordinated Beamforming with ZF Precoder, RSV Combiner and Optimal Power Allocation.\n 8 transmit antennas, 4 UTs with 2 receive antennas each.\n Fixed bit loading with 4-QAM constellation."},
    {'number': 4,     'Pt': 1.0,    'B': 0.5,    'K': 8,    'Nr': 1,    'Nt': 8,    'constellation types': "QAM",    'constellation sizes': 2,       'capacity fractions': None,    'bit loader': FixedBitLoader,       'mapper': GrayCodeMapper,    'precoder': ZFPrecoder,        'channel model': IIDRayleighChannelModel,    'noise model': CSAWGNNoiseModel,      'combiner': LSVCombiner,       'equalizer': Equalizer,    'detector': MDDetector,         'demapper': GrayCodeDemapper,    'name': "Reference System 4",     'description': "Non-coordinated Beamforming with ZF Precoder, RSV Combiner and Optimal Power Allocation.\n 8 transmit antennas, 8 UTs with 1 receive antenna each.\n Fixed bit loading with 4-QAM constellation."},

    # Reference Systems 5-8: Non-coordinated Beamforming with ZF Precoder, RSV Combiner and Optimal Power Allocation. Fixed bit loading with 16-QAM constellation for each UT.
    {'number': 5,     'Pt': 1.0,    'B': 0.5,    'K': 1,    'Nr': 8,    'Nt': 8,    'constellation types': "QAM",    'constellation sizes': 4,       'capacity fractions': None,    'bit loader': FixedBitLoader,       'mapper': GrayCodeMapper,    'precoder': ZFPrecoder,        'channel model': IIDRayleighChannelModel,    'noise model': CSAWGNNoiseModel,      'combiner': LSVCombiner,       'equalizer': Equalizer,    'detector': MDDetector,         'demapper': GrayCodeDemapper,    'name': "Reference System 5",     'description': "Non-coordinated Beamforming with ZF Precoder, RSV Combiner and Optimal Power Allocation.\n 8 transmit antennas, 1 UT with 8 receive antennas.\n Fixed bit loading with 16-QAM constellation."},
    {'number': 6,     'Pt': 1.0,    'B': 0.5,    'K': 2,    'Nr': 4,    'Nt': 8,    'constellation types': "QAM",    'constellation sizes': 4,       'capacity fractions': None,    'bit loader': FixedBitLoader,       'mapper': GrayCodeMapper,    'precoder': ZFPrecoder,        'channel model': IIDRayleighChannelModel,    'noise model': CSAWGNNoiseModel,      'combiner': LSVCombiner,       'equalizer': Equalizer,    'detector': MDDetector,         'demapper': GrayCodeDemapper,    'name': "Reference System 6",     'description': "Non-coordinated Beamforming with ZF Precoder, RSV Combiner and Optimal Power Allocation.\n 8 transmit antennas, 2 UTs with 4 receive antennas each.\n Fixed bit loading with 16-QAM constellation."},
    {'number': 7,     'Pt': 1.0,    'B': 0.5,    'K': 4,    'Nr': 2,    'Nt': 8,    'constellation types': "QAM",    'constellation sizes': 4,       'capacity fractions': None,    'bit loader': FixedBitLoader,       'mapper': GrayCodeMapper,    'precoder': ZFPrecoder,        'channel model': IIDRayleighChannelModel,    'noise model': CSAWGNNoiseModel,      'combiner': LSVCombiner,       'equalizer': Equalizer,    'detector': MDDetector,         'demapper': GrayCodeDemapper,    'name': "Reference System 7",     'description': "Non-coordinated Beamforming with ZF Precoder, RSV Combiner and Optimal Power Allocation.\n 8 transmit antennas, 4 UTs with 2 receive antennas each.\n Fixed bit loading with 16-QAM constellation."},
    {'number': 8,     'Pt': 1.0,    'B': 0.5,    'K': 8,    'Nr': 1,    'Nt': 8,    'constellation types': "QAM",    'constellation sizes': 4,       'capacity fractions': None,    'bit loader': FixedBitLoader,       'mapper': GrayCodeMapper,    'precoder': ZFPrecoder,        'channel model': IIDRayleighChannelModel,    'noise model': CSAWGNNoiseModel,      'combiner': LSVCombiner,       'equalizer': Equalizer,    'detector': MDDetector,         'demapper': GrayCodeDemapper,    'name': "Reference System 8",     'description': "Non-coordinated Beamforming with ZF Precoder, RSV Combiner and Optimal Power Allocation.\n 8 transmit antennas, 8 UTs with 1 receive antenna each.\n Fixed bit loading with 16-QAM constellation."},

    # Reference Systems 9-12: Non-coordinated Beamforming with ZF Precoder, RSV Combiner and Optimal Power Allocation. Adaptive bit loading (100% of the channel capacities) with QAM constellation for each UT.
    {'number': 9,     'Pt': 1.0,    'B': 0.5,    'K': 1,    'Nr': 8,    'Nt': 8,    'constellation types': "QAM",    'constellation sizes': None,    'capacity fractions': 1.0,     'bit loader': AdaptiveBitLoader,     'mapper': GrayCodeMapper,    'precoder': ZFPrecoder,        'channel model': IIDRayleighChannelModel,    'noise model': CSAWGNNoiseModel,      'combiner': LSVCombiner,       'equalizer': Equalizer,    'detector': MDDetector,         'demapper': GrayCodeDemapper,    'name': "Reference System 9",     'description': "Non-coordinated Beamforming with ZF Precoder, RSV Combiner and Optimal Power Allocation.\n 8 transmit antennas, 1 UT with 8 receive antennas.\n Adaptive bit loading (100%% of the channel capacities) with QAM constellation."},
    {'number': 10,    'Pt': 1.0,    'B': 0.5,    'K': 2,    'Nr': 4,    'Nt': 8,    'constellation types': "QAM",    'constellation sizes': None,    'capacity fractions': 1.0,     'bit loader': AdaptiveBitLoader,     'mapper': GrayCodeMapper,    'precoder': ZFPrecoder,        'channel model': IIDRayleighChannelModel,    'noise model': CSAWGNNoiseModel,      'combiner': LSVCombiner,       'equalizer': Equalizer,    'detector': MDDetector,         'demapper': GrayCodeDemapper,    'name': "Reference System 10",    'description': "Non-coordinated Beamforming with ZF Precoder, RSV Combiner and Optimal Power Allocation.\n 8 transmit antennas, 2 UTs with 4 receive antennas each.\n Adaptive bit loading (100%% of the channel capacities) with QAM constellation."},
    {'number': 11,    'Pt': 1.0,    'B': 0.5,    'K': 4,    'Nr': 2,    'Nt': 8,    'constellation types': "QAM",    'constellation sizes': None,    'capacity fractions': 1.0,     'bit loader': AdaptiveBitLoader,     'mapper': GrayCodeMapper,    'precoder': ZFPrecoder,        'channel model': IIDRayleighChannelModel,    'noise model': CSAWGNNoiseModel,      'combiner': LSVCombiner,       'equalizer': Equalizer,    'detector': MDDetector,         'demapper': GrayCodeDemapper,    'name': "Reference System 11",    'description': "Non-coordinated Beamforming with ZF Precoder, RSV Combiner and Optimal Power Allocation.\n 8 transmit antennas, 4 UTs with 2 receive antennas each.\n Adaptive bit loading (100%% of the channel capacities) with QAM constellation."},
    {'number': 12,    'Pt': 1.0,    'B': 0.5,    'K': 8,    'Nr': 1,    'Nt': 8,    'constellation types': "QAM",    'constellation sizes': None,    'capacity fractions': 1.0,     'bit loader': AdaptiveBitLoader,     'mapper': GrayCodeMapper,    'precoder': ZFPrecoder,        'channel model': IIDRayleighChannelModel,    'noise model': CSAWGNNoiseModel,      'combiner': LSVCombiner,       'equalizer': Equalizer,    'detector': MDDetector,         'demapper': GrayCodeDemapper,    'name': "Reference System 12",    'description': "Non-coordinated Beamforming with ZF Precoder, RSV Combiner and Optimal Power Allocation.\n 8 transmit antennas, 8 UTs with 1 receive antenna each.\n Adaptive bit loading (100%% of the channel capacities) with QAM constellation."},

    # Reference Systems 13-16: Non-coordinated Beamforming with ZF Precoder, RSV Combiner and Optimal Power Allocation. Adaptive bit loading (75% of the channel capacities) with QAM constellation for each UT.
    {'number': 13,    'Pt': 1.0,    'B': 0.5,    'K': 1,    'Nr': 8,    'Nt': 8,    'constellation types': "QAM",    'constellation sizes': None,    'capacity fractions': 0.75,    'bit loader': AdaptiveBitLoader,     'mapper': GrayCodeMapper,    'precoder': ZFPrecoder,        'channel model': IIDRayleighChannelModel,    'noise model': CSAWGNNoiseModel,      'combiner': LSVCombiner,       'equalizer': Equalizer,    'detector': MDDetector,         'demapper': GrayCodeDemapper,    'name': "Reference System 13",    'description': "Non-coordinated Beamforming with ZF Precoder, RSV Combiner and Optimal Power Allocation.\n 8 transmit antennas, 1 UT with 8 receive antennas.\n Adaptive bit loading (75%% of the channel capacities) with QAM constellation."},
    {'number': 14,    'Pt': 1.0,    'B': 0.5,    'K': 2,    'Nr': 4,    'Nt': 8,    'constellation types': "QAM",    'constellation sizes': None,    'capacity fractions': 0.75,    'bit loader': AdaptiveBitLoader,     'mapper': GrayCodeMapper,    'precoder': ZFPrecoder,        'channel model': IIDRayleighChannelModel,    'noise model': CSAWGNNoiseModel,      'combiner': LSVCombiner,       'equalizer': Equalizer,    'detector': MDDetector,         'demapper': GrayCodeDemapper,    'name': "Reference System 14",    'description': "Non-coordinated Beamforming with ZF Precoder, RSV Combiner and Optimal Power Allocation.\n 8 transmit antennas, 2 UTs with 4 receive antennas each.\n Adaptive bit loading (75%% of the channel capacities) with QAM constellation."},
    {'number': 15,    'Pt': 1.0,    'B': 0.5,    'K': 4,    'Nr': 2,    'Nt': 8,    'constellation types': "QAM",    'constellation sizes': None,    'capacity fractions': 0.75,    'bit loader': AdaptiveBitLoader,     'mapper': GrayCodeMapper,    'precoder': ZFPrecoder,        'channel model': IIDRayleighChannelModel,    'noise model': CSAWGNNoiseModel,      'combiner': LSVCombiner,       'equalizer': Equalizer,    'detector': MDDetector,         'demapper': GrayCodeDemapper,    'name': "Reference System 15",    'description': "Non-coordinated Beamforming with ZF Precoder, RSV Combiner and Optimal Power Allocation.\n 8 transmit antennas, 4 UTs with 2 receive antennas each.\n Adaptive bit loading (75%% of the channel capacities) with QAM constellation."},
    {'number': 16,    'Pt': 1.0,    'B': 0.5,    'K': 8,    'Nr': 1,    'Nt': 8,    'constellation types': "QAM",    'constellation sizes': None,    'capacity fractions': 0.75,    'bit loader': AdaptiveBitLoader,     'mapper': GrayCodeMapper,    'precoder': ZFPrecoder,        'channel model': IIDRayleighChannelModel,    'noise model': CSAWGNNoiseModel,      'combiner': LSVCombiner,       'equalizer': Equalizer,    'detector': MDDetector,         'demapper': GrayCodeDemapper,    'name': "Reference System 16",    'description': "Non-coordinated Beamforming with ZF Precoder, RSV Combiner and Optimal Power Allocation.\n 8 transmit antennas, 8 UTs with 1 receive antenna each.\n Adaptive bit loading (75%% of the channel capacities) with QAM constellation."},

    # Reference System 17: Non-coordinated Beamforming with ZF Precoder, RSV Combiner and Optimal Power Allocation. Fixed bit loading with 4-QAM and 16-PSK constellation.
    {'number': 17,    'Pt': 1.0,    'B': 0.5,    'K': 2,    'Nr': 4,    'Nt': 8,    'constellation types': ["QAM", "PSK"],    'constellation sizes': [2, 4],       'capacity fractions': None,    'bit loader': FixedBitLoader,       'mapper': GrayCodeMapper,    'precoder': ZFPrecoder,        'channel model': IIDRayleighChannelModel,    'noise model': CSAWGNNoiseModel,      'combiner': LSVCombiner,       'equalizer': Equalizer,    'detector': MDDetector,         'demapper': GrayCodeDemapper,    'name': "Reference System 17",    'description': "Non-coordinated Beamforming with ZF Precoder, RSV Combiner and Optimal Power Allocation.\n 8 transmit antennas, 2 UTs with 4 receive antennas each.\n Fixed bit loading with 4-QAM constellation for UT1 and 16-PSK constellation for UT2."},

    # Reference System 18: Non-coordinated Beamforming with ZF Precoder, RSV Combiner and Optimal Power Allocation. Adaptive bit loading (100% and 75% of the channel capacities) with QAM constellation.
    {'number': 18,    'Pt': 1.0,    'B': 0.5,    'K': 2,    'Nr': 4,    'Nt': 8,    'constellation types': "QAM",         'constellation sizes': None,     'capacity fractions': [1.0, 0.75],      'bit loader': AdaptiveBitLoader,     'mapper': GrayCodeMapper,    'precoder': ZFPrecoder,        'channel model': IIDRayleighChannelModel,    'noise model': CSAWGNNoiseModel,      'combiner': LSVCombiner,       'equalizer': Equalizer,    'detector': MDDetector,         'demapper': GrayCodeDemapper,    'name': "Reference System 18",    'description': "Non-coordinated Beamforming with ZF Precoder, RSV Combiner and Optimal Power Allocation.\n 8 transmit antennas, 2 UTs with 4 receive antennas each.\n Adaptive bit loading with QAM constellation. UT1 is loaded with 100%% of the channel capacities and UT2 is loaded with 75%% of the channel capacities."},

    # Reference System 18: Non-coordinated Beamforming with ZF Precoder, RSV Combiner and Optimal Power Allocation. Adaptive bit loading (100% and 75% of the channel capacities) with QAM constellation.
    {'number': 19,    'Pt': 1.0,    'B': 0.5,    'K': 3,    'Nr': 2,    'Nt': 6,    'constellation types': "QAM",         'constellation sizes': 4,     'capacity fractions': None,      'bit loader': FixedBitLoader,     'mapper': GrayCodeMapper,    'precoder': ZFPrecoder,        'channel model': IIDRayleighChannelModel,    'noise model': CSAWGNNoiseModel,      'combiner': NeutralCombiner,       'equalizer': Equalizer,    'detector': MDDetector,         'demapper': GrayCodeDemapper,    'name': "Reference System 19",    'description': "Non-coordinated Beamforming with ZF Precoder, RSV Combiner and Optimal Power Allocation.\n 6 transmit antennas, 3 UTs with 2 receive antennas each.\n Adaptive bit loading with QAM constellation. UT1 is loaded with 100%% of the channel capacities and UT2 is loaded with 75%% of the channel capacities."},

    # Reference System 18: Non-coordinated Beamforming with ZF Precoder, RSV Combiner and Optimal Power Allocation. Adaptive bit loading (100% and 75% of the channel capacities) with QAM constellation.
    {'number': 20,    'Pt': 1.0,    'B': 0.5,    'K': 3,    'Nr': 2,    'Nt': 6,    'constellation types': "QAM",         'constellation sizes': 4,     'capacity fractions': None,      'bit loader': FixedBitLoader,     'mapper': GrayCodeMapper,    'precoder': ZFPrecoder,        'channel model': IIDRayleighChannelModel,    'noise model': CSAWGNNoiseModel,      'combiner': LSVCombiner,       'equalizer': Equalizer,    'detector': MDDetector,         'demapper': GrayCodeDemapper,    'name': "Reference System 20",    'description': "Non-coordinated Beamforming with ZF Precoder, RSV Combiner and Optimal Power Allocation.\n 6 transmit antennas, 3 UTs with 2 receive antennas each.\n Adaptive bit loading with QAM constellation. UT1 is loaded with 100%% of the channel capacities and UT2 is loaded with 75%% of the channel capacities."},


    # Presentation Reference Systems.

    {'number': 101,     'Pt': 1.0,    'B': 0.5,    'K': 4,    'Nr': 2,    'Nt': 8,    'constellation types': "QAM",    'constellation sizes': 2,       'capacity fractions': None,    'bit loader': FixedBitLoader,       'mapper': GrayCodeMapper,    'precoder': ZFPrecoder,        'channel model': IIDRayleighChannelModel,    'noise model': CSAWGNNoiseModel,      'combiner': NeutralCombiner,       'equalizer': Equalizer,    'detector': MDDetector,         'demapper': GrayCodeDemapper,    'name': "4-QAM",     'description': ""},
    {'number': 102,     'Pt': 1.0,    'B': 0.5,    'K': 4,    'Nr': 2,    'Nt': 8,    'constellation types': "PSK",    'constellation sizes': 3,       'capacity fractions': None,    'bit loader': FixedBitLoader,       'mapper': GrayCodeMapper,    'precoder': ZFPrecoder,        'channel model': IIDRayleighChannelModel,    'noise model': CSAWGNNoiseModel,      'combiner': NeutralCombiner,       'equalizer': Equalizer,    'detector': MDDetector,         'demapper': GrayCodeDemapper,    'name': "8-PSK",     'description': ""},
    {'number': 103,     'Pt': 1.0,    'B': 0.5,    'K': 4,    'Nr': 2,    'Nt': 8,    'constellation types': "QAM",    'constellation sizes': 4,       'capacity fractions': None,    'bit loader': FixedBitLoader,       'mapper': GrayCodeMapper,    'precoder': ZFPrecoder,        'channel model': IIDRayleighChannelModel,    'noise model': CSAWGNNoiseModel,      'combiner': NeutralCombiner,       'equalizer': Equalizer,    'detector': MDDetector,         'demapper': GrayCodeDemapper,    'name': "16-QAM",     'description': ""},
    {'number': 104,     'Pt': 1.0,    'B': 0.5,    'K': 4,    'Nr': 2,    'Nt': 8,    'constellation types': "PSK",    'constellation sizes': 5,       'capacity fractions': None,    'bit loader': FixedBitLoader,       'mapper': GrayCodeMapper,    'precoder': ZFPrecoder,        'channel model': IIDRayleighChannelModel,    'noise model': CSAWGNNoiseModel,      'combiner': NeutralCombiner,       'equalizer': Equalizer,    'detector': MDDetector,         'demapper': GrayCodeDemapper,    'name': "32-PSK",     'description': ""},
    {'number': 105,     'Pt': 1.0,    'B': 0.5,    'K': 4,    'Nr': 2,    'Nt': 8,    'constellation types': "QAM",    'constellation sizes': 6,       'capacity fractions': None,    'bit loader': FixedBitLoader,       'mapper': GrayCodeMapper,    'precoder': ZFPrecoder,        'channel model': IIDRayleighChannelModel,    'noise model': CSAWGNNoiseModel,      'combiner': NeutralCombiner,       'equalizer': Equalizer,    'detector': MDDetector,         'demapper': GrayCodeDemapper,    'name': "64-QAM",     'description': ""},
    

    {'number': 206,     'Pt': 1.0,    'B': 0.5,    'K': 4,    'Nr': 2,    'Nt': 8,    'constellation types': "QAM",    'constellation sizes': None,       'capacity fractions': 1.0,    'bit loader': AdaptiveBitLoader,       'mapper': GrayCodeMapper,    'precoder': ZFPrecoder,        'channel model': IIDRayleighChannelModel,    'noise model': CSAWGNNoiseModel,      'combiner': NeutralCombiner,       'equalizer': Equalizer,    'detector': MDDetector,         'demapper': GrayCodeDemapper,    'name': "100%%",     'description': ""},
    {'number': 207,     'Pt': 1.0,    'B': 0.5,    'K': 4,    'Nr': 2,    'Nt': 8,    'constellation types': "QAM",    'constellation sizes': None,       'capacity fractions': 0.75,    'bit loader': AdaptiveBitLoader,       'mapper': GrayCodeMapper,    'precoder': ZFPrecoder,        'channel model': IIDRayleighChannelModel,    'noise model': CSAWGNNoiseModel,      'combiner': NeutralCombiner,       'equalizer': Equalizer,    'detector': MDDetector,         'demapper': GrayCodeDemapper,    'name': "75%%",     'description': ""},
    
    {'number': 101,     'Pt': 1.0,    'B': 0.5,    'K': 4,    'Nr': 2,    'Nt': 12,    'constellation types': "QAM",    'constellation sizes': 2,       'capacity fractions': None,    'bit loader': FixedBitLoader,       'mapper': GrayCodeMapper,    'precoder': ZFPrecoder,        'channel model': IIDRayleighChannelModel,    'noise model': CSAWGNNoiseModel,      'combiner': NeutralCombiner,       'equalizer': Equalizer,    'detector': MDDetector,         'demapper': GrayCodeDemapper,    'name': "4-QAM ",     'description': ""},
    {'number': 103,     'Pt': 1.0,    'B': 0.5,    'K': 4,    'Nr': 2,    'Nt': 12,    'constellation types': "QAM",    'constellation sizes': 4,       'capacity fractions': None,    'bit loader': FixedBitLoader,       'mapper': GrayCodeMapper,    'precoder': ZFPrecoder,        'channel model': IIDRayleighChannelModel,    'noise model': CSAWGNNoiseModel,      'combiner': NeutralCombiner,       'equalizer': Equalizer,    'detector': MDDetector,         'demapper': GrayCodeDemapper,    'name': "16-QAM ",     'description': ""},
    {'number': 105,     'Pt': 1.0,    'B': 0.5,    'K': 4,    'Nr': 2,    'Nt': 12,    'constellation types': "QAM",    'constellation sizes': 6,       'capacity fractions': None,    'bit loader': FixedBitLoader,       'mapper': GrayCodeMapper,    'precoder': ZFPrecoder,        'channel model': IIDRayleighChannelModel,    'noise model': CSAWGNNoiseModel,      'combiner': NeutralCombiner,       'equalizer': Equalizer,    'detector': MDDetector,         'demapper': GrayCodeDemapper,    'name': "64-QAM ",     'description': ""},
    
]


def _setup_settings(sim_config_idx, sys_config_idx):

    # 1. Simulation Configuration Settings.
    sim_config_settings = simulation_configuration_settings[sim_config_idx]
    sim_config = SimConfig(
        snr_dB_values            = sim_config_settings['SNR values (in dB)'],
        num_channel_realizations = sim_config_settings['channel realizations per SNR value'],
        num_bit_errors           = sim_config_settings['bit errors per SNR value'],
        num_bit_errors_scope     = sim_config_settings['Scope of bit errors'],
        M                        = sim_config_settings['Transmission per channel realization'],
        name                     = sim_config_settings['name'],
    )


    # 2. System Configuration Settings.
    sys_config_settings = system_configuration_settings[sys_config_idx]

    # constellation configurations.
    c_configs = ConstConfig(
        types                   = sys_config_settings['constellation types'],
        sizes                   = sys_config_settings['constellation sizes'],
        capacity_fractions      = sys_config_settings['capacity fractions'],
    )

    # base station configurations.
    base_station_configs = BaseStationConfig(
        precoder               = sys_config_settings['precoder'],
        bit_loader             = sys_config_settings['bit loader'],
        mapper                 = sys_config_settings['mapper'],
    )

    # channel configurations.
    channel_configs = ChannelConfig(
        channel_model          = sys_config_settings['channel model'],
        noise_model            = sys_config_settings['noise model'],
    )

    # user terminal configerations.
    user_terminal_configs = UserTerminalConfig(
        combiner              = sys_config_settings['combiner'],
        equalizer             = sys_config_settings['equalizer'],
        detector              = sys_config_settings['detector'],
        demapper              = sys_config_settings['demapper'],
    )

    # system configurations.
    system_config = SystemConfig(
        Pt                    = sys_config_settings['Pt'],
        B                     = sys_config_settings['B'],
        K                     = sys_config_settings['K'],
        Nr                    = sys_config_settings['Nr'],
        Nt                    = sys_config_settings['Nt'],
        c_configs             = c_configs,
        base_station_configs  = base_station_configs,
        channel_configs       = channel_configs,
        user_terminal_configs = user_terminal_configs,
        name                  = sys_config_settings['name'],
    )

    return sim_config, system_config

def main(sim_config_indices: IntArray, sys_config_indices: IntArray):

    results = []

    for sim_config_idx in sim_config_indices:
        for sys_config_idx in sys_config_indices:

            # 1. SETTINGS
            sim_config, system_config = _setup_settings(sim_config_idx, sys_config_idx)
            
            # 2. SIMULATION.
            runner = SimulationRunner(sim_config=sim_config, system_config=system_config)
            result = runner.run()

            # 3. RESULT.
            results.append(result)
            # ResultManager.display(result)
            # ResultManager.plot_system_performance(result)
            # ResultManager.plot_ut_performance(result)
            # ResultManager.plot_stream_performance(result)

    return results


if __name__ == "__main__":
    results = main(sim_config_indices = [4], sys_config_indices = [28, 29, 30])
    ResultManager.plot_system_performance_comparison(results)