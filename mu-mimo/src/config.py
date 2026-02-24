# mu-mimo/src/config.py

from dataclasses import dataclass
from .processing import (
    Precoder, Combiner,
    PowerAllocator, PowerDeallocator,
    BitAllocator, BitDeallocator,
    Mapper, Demapper,
    Detector,
    ChannelModel, NoiseModel )
import numpy as np
from numpy.typing import NDArray


RealArray = NDArray[np.floating]
ComplexArray = NDArray[np.complexfloating]
IntArray = NDArray[np.integer]
BitArray = NDArray[np.integer]


@dataclass
class ChannelStateInformation:
    snr: float | None = None
    H: ComplexArray | None = None


@dataclass
class TransmitPilotMessage:

    pass

@dataclass
class ReceivePilotMessage:
    H_k: ComplexArray

@dataclass
class TransmitFeedbackMessage:
    ut_id: int
    H_eff_k: ComplexArray

@dataclass
class ReceiveFeedbackMessage:
    snr: float
    H_eff: ComplexArray

@dataclass
class TransmitFeedforwardMessage:
    P: RealArray
    ibr: IntArray
    G: ComplexArray | None

@dataclass
class ReceiveFeedforwardMessage:
    ut_id: int
    P_k: RealArray
    ibr_k: IntArray
    G_k: ComplexArray | None



@dataclass
class BaseStationConfig:
    bit_allocator: type[BitAllocator]
    mapper: type[Mapper]
    power_allocator: type[PowerAllocator]
    precoder: type[Precoder]

@dataclass
class UserTerminalConfig:
    combiner: type[Combiner]
    power_deallocator: type[PowerDeallocator]
    detector: type[Detector]
    demapper: type[Demapper]
    bit_deallocator: type[BitDeallocator]

@dataclass
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



@dataclass
class SimulationConfig:

    snr_dB_values: RealArray
    num_channel_realizations: int
    num_errors: int
    num_symbols: int = 4800

    def __post_init__(self):
        
        # Validate the number of channel realizations and the number of bit errors.
        if self.num_channel_realizations <= 0: raise ValueError("The minimum number of channel realizations must be a positive integer.")
        if self.num_errors <= 0: raise ValueError("The minimum number of bit errors per SNR value must be a positive integer.")

@dataclass
class SimulationResult:
    
    sim_configs: SimulationConfig
    system_configs: SystemConfig

    snr_dB_values: RealArray
    bers_list: list[RealArray]
    ibrs_list: list[IntArray]
    ars_list: list[RealArray]

    filename: str | None = None

    def __post_init__(self):

        # Validate the lengths of the SNR values, BERs, IBRs, and ARs.
        
        # initialize the filename.

        pass



