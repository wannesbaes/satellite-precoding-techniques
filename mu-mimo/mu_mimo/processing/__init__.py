"""
mu_mimo.processing package

@author Wannes Baes
@date 2026
"""

from .precoding import Precoder, NeutralPrecoder, SVDPrecoder, ZFPrecoder, BDPrecoder, WMMSEPrecoder, waterfilling_v1

from .combining import Combiner, NeutralCombiner, LSVCombiner

from .bit_loading import BitLoader, NeutralBitLoader, FixedBitLoader, AdaptiveBitLoader

from .modulation import (
    Constellation, NumberRepresentation,
    Mapper, NeutralMapper, GrayCodeMapper,
    Demapper, NeutralDemapper, GrayCodeDemapper,
    Equalizer,
    Detector, NeutralDetector, MDDetector,
)

from .channel import (
    ChannelModel, NeutralChannelModel, IIDRayleighFadingChannelModel, RiceanFadingChannelModel,
    NoiseModel, NeutralNoiseModel, CSAWGNNoiseModel
)


__all__ = [
    "Precoder", "NeutralPrecoder", "SVDPrecoder", "ZFPrecoder", "BDPrecoder", "WMMSEPrecoder",
    "Combiner", "NeutralCombiner", "LSVCombiner",
    "BitLoader", "NeutralBitLoader", "FixedBitLoader", "AdaptiveBitLoader",
    "Constellation", "NumberRepresentation",
    "Mapper", "NeutralMapper", "GrayCodeMapper",
    "Demapper", "NeutralDemapper", "GrayCodeDemapper",
    "Equalizer",
    "Detector", "NeutralDetector", "MDDetector",
    "ChannelModel", "NeutralChannelModel", "IIDRayleighFadingChannelModel", "RiceanFadingChannelModel",
    "NoiseModel", "NeutralNoiseModel", "CSAWGNNoiseModel",
]