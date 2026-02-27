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
ConstType = Literal["PAM", "PSK", "QAM"]


@dataclass()
class ChannelStateInformation:
    """
    The channel state information (CSI).

    Parameters
    ----------
    snr : float
        The signal-to-noise ratio. (optional, default None)
    H_eff : ComplexArray, shape (K*Nr, Nt) or (Ns_total, Nt)
        The effective channel matrix. (optional, default None)
        In case of coordinated beamforming, the effective channel matrix equals the actual channel matrix.
        In case of non-coordinated beamforming, the effective channel matrix equals the the actual channel matrix followed by the compound combining matrix (G * H).
    """
    snr: float | None = None
    H_eff: ComplexArray | None = None


@dataclass()
class BaseStationState:
    """
    The state of a base station (for a specific channel and SNR value).

    Parameters
    ----------
    F : ComplexArray, shape (Nt, Ns_total) (Ns_total = sum of the number of data streams across all UTs)
        The compound precoding matrix.
    P : RealArray, shape (Ns_total,)
        The power allocation vector. It contains the power allocated to each data stream of each UT.
    ibr : IntArray, shape (K, Nr)
        The information bit rates for each data stream of each UT.
    Ns : IntArray, shape (K,)
        The number of data streams for each UT.
    G : ComplexArray, shape (Ns_total, K * Nr)
        The compound combining matrix.
    """
    F: ComplexArray
    P: RealArray
    ibr: IntArray
    Ns: IntArray
    G: ComplexArray

@dataclass()
class UserTerminalState:
    """
    The state of a single user terminal (for a specific channel and SNR value).

    Parameters
    ----------
    H_k : ComplexArray, shape (Nr, Nt)
        The channel matrix of this UT.
    G_k : ComplexArray, shape (Ns_k, Nr)
        The combining matrix of this UT.
    c_type_k : ConstType
        The constellation type for the data streams to this UT.
    P_k : RealArray, shape (Ns_k,)
        The power allocation vector of this UT.
    ibr_k : IntArray, shape (Ns_k,)
        The information bit rates for each data stream of this UT.
    Ns_k : int
        The number of data streams for this UT.
    """
    H_k: ComplexArray
    G_k: ComplexArray
    c_type_k: ConstType | None
    P_k: RealArray | None
    ibr_k: IntArray | None
    Ns_k: int | None


@dataclass()
class TransmitPilotMessage:
    pass

@dataclass()
class ReceivePilotMessage:
    """
    The pilot message received by the UT.

    Normally, the received pilot message would contain the received pilot symbols. However, since we consider channel estimation out of scope for this framework, we directly include the channel matrix in the received pilot message.

    Parameters
    ----------
    H_k : ComplexArray, shape (Nr, Nt)
        The channel matrix of this UT.
    """
    H_k: ComplexArray

@dataclass()
class TransmitFeedbackMessage:
    """
    The feedback message transmitted by the UT.

    Parameters
    ----------
    ut_id : int
        The ID of the UT that transmits this feedback message.
    H_eff_k : ComplexArray, shape (Nr, Nt) or (Ns_k, Nt)
        The effective channel matrix of this UT.
    """
    ut_id: int
    H_eff_k: ComplexArray

@dataclass()
class ReceiveFeedbackMessage:
    """
    The feedback message received by the BS.

    Parameters
    ----------
    csi : ChannelStateInformation
        The channel state information (CSI).
    """
    csi: ChannelStateInformation

@dataclass()
class TransmitFeedforwardMessage:
    """
    The feedforward message transmitted by the BS.

    Parameters
    ----------
    c_type : list[ConstType]
        The constellation types for the data streams to each UT.
    P : RealArray, shape (Ns_total,)
        The power allocation vector. It contains the power allocated to each data stream of each UT.
    ibr : IntArray, shape (Ns_total,)
        The information bit rates for each data stream of each UT.
    Ns : IntArray, shape (K,)
        The number of data streams for each UT.
    G : ComplexArray, shape (Ns_total, K * Nr)
        The compound combining matrix. It is None in case of non-coordinated beamforming.
    """
    c_type: list[ConstType]
    P: RealArray
    ibr: IntArray
    Ns: IntArray
    G: ComplexArray | None

@dataclass()
class ReceiveFeedforwardMessage:
    """
    The feedforward message received by the UT.

    Parameters
    ----------
    ut_id : int
        The ID of the UT that receives this feedforward message.
    c_type_k : ConstType
        The constellation type for the data streams to this UT.
    P_k : RealArray, shape (Ns_k,)
        The power allocation vector of this UT.
    ibr_k : IntArray, shape (Ns_k,)
        The information bit rates for each data stream of this UT.
    Ns_k : int
        The number of data streams for this UT.
    G_k : ComplexArray, shape (Ns_k, Nr)
        The combining matrix of this UT. It is None in case of non-coordinated beamforming.
    """
    ut_id: int
    c_type_k: ConstType
    P_k: RealArray
    ibr_k: IntArray
    Ns_k: int
    G_k: ComplexArray | None


@dataclass()
class ConstConfig:
    """
    The constellation configuration settings for the data transmission between the BS and one UT.

    Parameters
    ----------
    types : list[ConstType]
        The constellation types for the data streams to each UT.
    sizes : int
        The constellation sizes in bits, i.e. the number of bits per data symbol (point in the constellation), for the data streams to each UT.
    """
    types: ConstType | list[ConstType]
    sizes: int | IntArray | None = None


@dataclass()
class BaseStationConfig:
    """
    The configuration settings of a base station.

    Parameters
    ----------
    bit_allocator : type[BitAllocator]
        The type of the bit allocator (the concrete bit allocator class).
    mapper : type[Mapper]
        The type of the mapper (the concrete mapper class).
    power_allocator : type[PowerAllocator]
        The type of the power allocator (the concrete power allocator class).
    precoder : type[Precoder]
        The type of the precoder (the concrete precoder class).
    """
    bit_allocator: type[BitAllocator]
    mapper: type[Mapper]
    power_allocator: type[PowerAllocator]
    precoder: type[Precoder]

@dataclass()
class UserTerminalConfig:
    """
    The configuration settings of a user terminal.

    Parameters
    ----------
    combiner : type[Combiner]
        The type of the combiner (the concrete combiner class).
    power_deallocator : type[PowerDeallocator]
        The type of the power deallocator (the concrete power deallocator class).
    detector : type[Detector]
        The type of the detector (the concrete detector class).
    demapper : type[Demapper]
        The type of the demapper (the concrete demapper class).
    bit_deallocator : type[BitDeallocator]
        The type of the bit deallocator (the concrete bit deallocator class).
    """
    combiner: type[Combiner]
    power_deallocator: type[PowerDeallocator]
    detector: type[Detector]
    demapper: type[Demapper]
    bit_deallocator: type[BitDeallocator]

@dataclass()
class ChannelConfig:
    """
    The configuration settings of a channel.

    Parameters
    ----------
    channel_model : type[ChannelModel]
        The type of the channel model (the concrete channel model class).
    noise_model : type[NoiseModel]
        The type of the noise model (the concrete noise model class).
    """
    channel_model: type[ChannelModel]
    noise_model: type[NoiseModel]

@dataclass()
class SystemConfig:
    """
    The configuration settings of a MU-MIMO system.

    Parameters
    ----------
    K : int
        The number of user terminals.
    Nt : int
        The number of transmit antennas at the base station.
    Nr : int
        The number of receive antennas per user terminal.
    
    c_configs : ConstConfig
        The constellation configuration settings for each UT.
    
    base_station_configs : BaseStationConfig
        The configuration settings of the base station.
    user_terminal_configs : UserTerminalConfig
        The configuration settings of the user terminals.
    channel_configs : ChannelConfig
        The configuration settings of the channel.
    """

    K: int
    Nt: int
    Nr: int

    c_configs: ConstConfig

    base_station_configs: BaseStationConfig
    user_terminal_configs: UserTerminalConfig
    channel_configs: ChannelConfig


    def __post_init__(self):

        # Validate the number of user terminals, transmit antennas, receive antennas, and data streams.
        if self.K <= 0: raise ValueError("The number of UTs must be a positive integer.")
        if self.Nt <= 0: raise ValueError("The number of transmit antennas must be a positive integer.")
        if self.Nr <= 0: raise ValueError("The number of receive antennas per UT must be a positive integer.")
        if self.K * self.Nr > self.Nt: raise ValueError("We assume that the base station has at least as many transmit antennas as the total number of receive antennas across all UTs.")
        
        # Validate the types of the processing components in the BS and UT configurations.
        for k in range(self.K):
            if self.user_terminal_configs[k].demapper is not self.base_station_configs.mapper.demapper_class: raise ValueError("Mapper and demapper do not match.")
            if self.user_terminal_configs[k].combiner is not self.base_station_configs.precoder.combiner_class: raise ValueError("Precoder and combiner do not match.")
            if self.base_station_configs.power_allocator is not self.base_station_configs.precoder.power_allocator_class: raise ValueError("This power allocator is not compatible with the precoder.")
        
        # Validate the constellation configuration settings.
        if isinstance(self.c_configs.types, ConstType):
            self.c_configs.types = [self.c_configs.types] * self.K
        else:
            if len(self.c_configs.types) != self.K:
                raise ValueError("The number of different constellation types must match the number of user terminals.")

        if isinstance(self.c_configs.sizes, int):
            self.c_configs.sizes = [self.c_configs.sizes] * self.K
        else:
            if len(self.c_configs.sizes) != self.K:
                raise ValueError("The number of different constellation sizes must match the number of user terminals.")

    def __eq__(self, other: object) -> bool:
        
        if not isinstance(other, SystemConfig):
            return NotImplemented
        
        return (
            self.K == other.K and
            self.Nt == other.Nt and
            self.Nr == other.Nr and
            self.base_station_configs == other.base_station_configs and
            self.user_terminal_configs == other.user_terminal_configs and
            self.channel_configs == other.channel_configs
        )


@dataclass
class SimConfig:
    """
    The configuration settings of a simulation.

    Parameters
    ----------
    snr_dB_values : RealArray
        The SNR values in dB.
    snr_values : RealArray
        The SNR values in linear scale. It is derived from snr_dB_values.
    num_channel_realizations : int
        The minimum number of channel realizations per SNR value.
    num_bit_errors : int
        The minimum number of bit errors per SNR value.
    num_bit_errors_scope : Literal["system-wide", "uts", "streams"]
        The scope over which the minimum number of bit errors are considered.
    num_symbols : int
        The number of symbol vectors that are sent at once (per channel realization).
    """

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
        if self.num_symbols <= 0: raise ValueError("The minimum number of symbol vectors that are sent per channel realization must be a positive integer.")

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
    The result of a simulation for a single SNR point, averaged over different channel realizations.
    
    Attributes
    ----------
    stream_ibrs : list[IntArray] (list of K arrays, each shape (Nr,))
        Per-UT per-stream information bit rates.
    stream_becs : list[RealArray] (list of K arrays, each shape (Nr,))
        Per-UT per-stream bit error counts.
    stream_ars : list[BitArray] (list of K arrays, each shape (Nr,))
        Per-UT per-stream stream activation rates (1 if the stream is active, 0 otherwise).
    
    ut_ibrs : IntArray (array of shape (K,))
        Per-UT information bit rates.
    ut_becs : RealArray (array of shape (K,))
        Per-UT bit error counts.
    ut_ars : BitArray (array of shape (K,))
        Per-UT UT activation rates (1 if the UT is active, 0 otherwise).
    
    ibr : float
        System-wide information bit rate.
    bec : float
        System-wide bit error count.
    
    stream_ars_avg : float
        Average stream activation rate.
    ut_ars_avg : float
        Average UT activation rate.
    
    num_symbols : int
        The number of symbol vectors that were sent per channel realization.
    num_channel_realizations : int
        The number of channel realizations that were simulated.
    
    stream_bers : list[RealArray] | None
        Per-UT per-stream bit error rates. None if num_channel_realizations == 1.
    ut_bers : RealArray | None
        Per-UT bit error rates. None if num_channel_realizations == 1.
    ber : float | None
        System-wide bit error rate. None if num_channel_realizations == 1.
    """
    
    stream_ibrs : list[IntArray]
    stream_becs : list[RealArray]
    stream_ars : list[BitArray]
    
    ut_ibrs : IntArray
    ut_becs : RealArray
    ut_ars : BitArray
    
    ibr : float
    bec : float

    stream_ars_avg : float
    ut_ars_avg : float

    num_symbols : int
    num_channel_realizations : int


    stream_bers : list[RealArray] | None = None
    ut_bers : RealArray | None = None
    ber: float | None = None

    def __post_init__(self):

        if self.num_channel_realizations > 1:

            K = len(self.stream_ibrs)
            self.stream_bers = [ self.stream_becs[k] / (self.stream_ibrs[k] * self.num_symbols * self.num_channel_realizations) for k in range(K) ]
            self.ut_bers = self.ut_becs / ( self.ut_ibrs * self.num_symbols * self.num_channel_realizations)
            self.ber = self.bec / (self.ibr * self.num_symbols * self.num_channel_realizations)

@dataclass
class SimResult:
    """
    The results of a simulation.
    
    Attributes
    ----------
    filename : Path
        The path to the file where the simulation results are stored.
    sim_configs : SimConfig
        The configuration settings of the simulation.
    system_configs : SystemConfig
        The configuration settings of the system.
    """

    filename: Path
    
    sim_configs: SimConfig
    system_configs: SystemConfig

    snr_dB_values: RealArray
    simulation_results: list[SingleSnrSimResult]
