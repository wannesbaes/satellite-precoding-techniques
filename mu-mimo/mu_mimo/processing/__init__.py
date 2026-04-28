"""
mu_mimo.processing package

@author Wannes Baes
@date 2026
"""

from .bit_loading import BitLoader, NeutralBitLoader, FixedBitLoader, AdaptiveBitLoader

from .modulation import (
    Constellation, NumberRepresentation,
    Mapper, NeutralMapper, GrayCodeMapper,
    Demapper, NeutralDemapper, GrayCodeDemapper,
    Equalizer,
    Detector, NeutralDetector, MDDetector,
)

from .precoding import Precoder, NeutralPrecoder, SVDPrecoder, ZFPrecoder, BDPrecoder, WMMSEPrecoder, waterfilling_v1

from .combining import Combiner, NeutralCombiner, LSVCombiner

from .channel_estimation import (
    ChannelEstimator, NeutralChannelEstimator,
    ChannelPredictor, NeutralChannelPredictor, ARPredictor
)

from .channel import (
    ChannelModel, NeutralChannel, IIDRayleighChannel, RiceanIIDTCChannel,
    NoiseModel, NeutralNoise, CSAWGNNoise
)


__all__ = [
    "BitLoader", "NeutralBitLoader", "FixedBitLoader", "AdaptiveBitLoader",
    "Constellation", "NumberRepresentation",
    "Mapper", "NeutralMapper", "GrayCodeMapper",
    "Demapper", "NeutralDemapper", "GrayCodeDemapper",
    "Equalizer",
    "Detector", "NeutralDetector", "MDDetector",
    "Precoder", "NeutralPrecoder", "SVDPrecoder", "ZFPrecoder", "BDPrecoder", "WMMSEPrecoder",
    "Combiner", "NeutralCombiner", "LSVCombiner",
    "ChannelEstimator", "NeutralChannelEstimator",
    "ChannelPredictor", "NeutralChannelPredictor", "ARPredictor",
    "ChannelModel", "NeutralChannel", "IIDRayleighChannel", "RiceanIIDTCChannel",
    "NoiseModel", "NeutralNoise", "CSAWGNNoise",
]