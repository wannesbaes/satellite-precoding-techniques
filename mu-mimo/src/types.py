# mu-mimo/src/types.py

from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from typing import Literal
from .processing import (
    Precoder, Combiner,
    PowerAllocator, PowerDeallocator,
    BitAllocator, BitDeallocator,
    Mapper, Demapper,
    Detector,
    ChannelModel, NoiseModel )


RealArray = NDArray[np.floating]
ComplexArray = NDArray[np.complexfloating]
IntArray = NDArray[np.integer]
BitArray = NDArray[np.integer]


@dataclass(slots=True)
class ChannelStateInformation:
    snr: float | None = None
    H: ComplexArray | None = None


@dataclass(slots=True)
class TransmitPilotMessage:

    pass

@dataclass(slots=True)
class ReceivePilotMessage:
    H_k: ComplexArray

@dataclass(slots=True)
class TransmitFeedbackMessage:
    ut_id: int
    H_eff_k: ComplexArray

@dataclass(slots=True)
class ReceiveFeedbackMessage:
    snr: float
    H_eff: ComplexArray

@dataclass(slots=True)
class TransmitFeedforwardMessage:
    P: RealArray
    ibr: IntArray
    G: ComplexArray | None

@dataclass(slots=True)
class ReceiveFeedforwardMessage:
    ut_id: int
    P_k: RealArray
    ibr_k: IntArray
    G_k: ComplexArray | None


@dataclass(frozen=True)
class BaseStationConfig:
    bit_allocator: type[BitAllocator]
    mapper: type[Mapper]
    power_allocator: type[PowerAllocator]
    precoder: type[Precoder]

@dataclass(frozen=True)
class UserTerminalConfig:
    combiner: type[Combiner]
    power_deallocator: type[PowerDeallocator]
    detector: type[Detector]
    demapper: type[Demapper]
    bit_deallocator: type[BitDeallocator]

@dataclass(frozen=True)
class ChannelConfig:
    channel_model: type[ChannelModel]
    noise_model: type[NoiseModel]

@dataclass
class SystemConfig:

    K: int
    Nt: int
    Nr: int
    Ns: int

    base_station_configs: BaseStationConfig
    user_terminal_configs: UserTerminalConfig
    channel_configs: ChannelConfig


    def __post_init__(self):

        # Validate the number of user terminals, transmit antennas, receive antennas, and data streams.
        if self.K <= 0: raise ValueError("The number of UTs must be a positive integer.")
        if self.Nt <= 0: raise ValueError("The number of transmit antennas must be a positive integer.")
        if self.Nr <= 0: raise ValueError("The number of receive antennas per UT must be a positive integer.")
        if self.Ns <= 0: raise ValueError("The number of data streams per UT must be a positive integer.")
        if self.K * self.Nr > self.Nt: raise ValueError("We assume that the base station has at least as many transmit antennas as the total number of receive antennas across all UTs.")
        if self.Ns > self.Nr: raise ValueError("We assume that the number of data streams per UT does not exceed the number of receive antennas per UT.")
        
        # Validate the types of the processing components in the BS and UT configurations.
        if self.user_terminal_configs.demapper is not self.base_station_configs.mapper.demapper_class: raise ValueError("Mapper and demapper do not match.")
        if self.user_terminal_configs.combiner is not self.base_station_configs.precoder.combiner_class: raise ValueError("Precoder and combiner do not match.")
        if self.base_station_configs.power_allocator is not self.base_station_configs.precoder.power_allocator_class: raise ValueError("This power allocator is not compatible with the precoder.")

    def __eq__(self, other: object) -> bool:
        
        if not isinstance(other, SystemConfig):
            return NotImplemented
        
        return (
            self.K == other.K and
            self.Nt == other.Nt and
            self.Nr == other.Nr and
            self.Ns == other.Ns and
            self.base_station_configs == other.base_station_configs and
            self.user_terminal_configs == other.user_terminal_configs and
            self.channel_configs == other.channel_configs
        )


@dataclass
class SimConfig:

    snr_dB_values: RealArray = field(default_factory=lambda: np.arange(-5, 31, 2.5))
    num_channel_realizations: int = 200
    num_bit_errors: int = 250
    num_bit_errors_scope: Literal["system-wide", "uts", "streams"] = "uts"
    num_symbols: int = 4800

    @property
    def snr_values(self) -> RealArray:
        return 10 ** (self.snr_dB_values / 10)

    def __post_init__(self):
        if self.num_channel_realizations <= 0: raise ValueError("The minimum number of channel realizations must be a positive integer.")
        if self.num_bit_errors <= 0: raise ValueError("The minimum number of bit errors per SNR value must be a positive integer.")
        if self.num_symbols <= 0: raise ValueError("The minimum number of symbols that are sent per channel realization must be a positive integer.")

    def __eq__(self, other: object) -> bool:
        
        if not isinstance(other, SimConfig):
            return NotImplemented
        
        return (
            np.array_equal(self.snr_dB_values, other.snr_dB_values) and
            self.num_channel_realizations == other.num_channel_realizations and
            self.num_bit_errors == other.num_bit_errors and
            self.num_bit_errors_scope == other.num_bit_errors_scope and
            self.num_symbols == other.num_symbols
        )

@dataclass
class SingleSnrSimResult:
    """
    Simulation result for a single SNR point, averaged over different channel realizations.
    
    Attributes
    ----------
    stream_ibrs : list[IntArray] (list of K arrays, each shape (Ns,))
        Per-UT per-stream information bit rates.
    stream_bers : list[RealArray] (list of K arrays, each shape (Ns,))
        Per-UT per-stream bit error rates.
    ut_ibrs : IntArray (array of shape (K,))
        Per-UT information bit rates.
    ut_bers : RealArray (array of shape (K,))
        Per-UT bit error rates.
    ibr : float
        System-wide information bit rate.
    ber : float
        System-wide bit error rate.
    """
    
    stream_ibrs : list[IntArray]
    stream_bers : list[RealArray]
    
    ut_ibrs : IntArray
    ut_bers : RealArray
    
    ibr : float
    ber : float  

@dataclass
class SimResult:

    filename: Path
    
    sim_configs: SimConfig
    system_configs: SystemConfig

    snr_dB_values: RealArray
    simulation_results: list[SingleSnrSimResult]
