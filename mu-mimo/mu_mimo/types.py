# mu-mimo/mu_mimo/types.py

from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from typing import Literal, Type, TYPE_CHECKING

# from .base_types import (
#     RealArray, ComplexArray, IntArray, BitArray, ConstType,
#     ChannelStateInformation, ConstConfig,
# )

if TYPE_CHECKING:
    from .processing import (
        BitLoader, Mapper, Precoder,
        ChannelModel, NoiseModel,
        Combiner, Equalizer, Detector, Demapper
    )

RealArray    = NDArray[np.floating]
ComplexArray = NDArray[np.complexfloating]
IntArray     = NDArray[np.integer]
BitArray     = NDArray[np.integer]
ConstType    = Literal["PAM", "PSK", "QAM"]


@dataclass()
class ChannelStateInformation:
    """
    The channel state information (CSI).

    Parameters
    ----------
    snr : float
        The signal-to-noise ratio. (optional, default None)
    H_eff : ComplexArray, shape (K*Nr, Nt) or (Ns_total, Nt)
        The effective channel matrix. (optional, default None)\\
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
    F : ComplexArray, shape (Nt, K*Nr)
        The compound precoding matrix.
    C_eq : ComplexArray, shape (K*Nr,)
        The equalization coefficients for each data stream.
    ibr : IntArray, shape (K*Nr,)
        The information bit rates for each data stream of each UT.
    Ns : IntArray, shape (K,)
        The number of data streams for each UT.
    G : ComplexArray, shape (K*Nr, K*Nr)
        The compound combining matrix.
    """
    F: ComplexArray
    C_eq: ComplexArray
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
    G_k : ComplexArray, shape (Nr, Nr)
        The combining matrix of this UT.
    c_type_k : ConstType
        The constellation type for the data streams to this UT.
    C_eq_k : ComplexArray, shape (Nr,)
        The equalization coefficients for each data stream of this UT.
    ibr_k : IntArray, shape (Nr,)
        The information bit rates for each data stream of this UT.
    Ns_k : int
        The number of data streams for this UT.
    """
    H_k: ComplexArray
    G_k: ComplexArray
    c_type_k: ConstType | None
    C_eq_k: ComplexArray | None
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
    H_eff_k : ComplexArray, shape (Nr, Nt)
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
    C_eq : ComplexArray, shape (K*Nr,)
        The equalization coefficients for each data stream of each UT.
    ibr : IntArray, shape (K*Nr,)
        The information bit rates for each data stream of each UT.
    Ns : IntArray, shape (K,)
        The number of data streams for each UT.
    G : ComplexArray, shape (K*Nr, K*Nr)
        The compound combining matrix. It is None in case of non-coordinated beamforming.
    """
    c_type: list[ConstType]
    C_eq: ComplexArray
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
    C_eq_k : ComplexArray, shape (Nr,)
        The equalization coefficients for each data stream of this UT.
    ibr_k : IntArray, shape (Nr,)
        The information bit rates for each data stream of this UT.
    Ns_k : int
        The number of data streams for this UT.
    G_k : ComplexArray, shape (Nr, Nr)
        The combining matrix of this UT. It is None in case of non-coordinated beamforming.
    """
    ut_id: int
    c_type_k: ConstType
    C_eq_k: ComplexArray
    ibr_k: IntArray
    Ns_k: int
    G_k: ComplexArray | None


@dataclass()
class ConstConfig:
    """
    The constellation configuration settings for the data transmission between the BS and one UT.

    Parameters
    ----------
    types : list[ConstType], shape (K,)
        The constellation types for the data streams to each UT. \\
        (If the same constellation type is used for all UTs, this can also be provided as a single ConstType at the moment of initialization.)
    sizes : IntArray, shape (K,) | None
        The constellation sizes in bits, i.e. the number of bits per data symbol (point in the constellation), for the data streams to each UT. \\
        If the same constellation size is used for all UTs, this can also be provided as a single int at the moment of initialization. \n
        In case of adaptive bit allocation, the constellation sizes are not predetermined but will be calculated by the bit allocator. In that case, this can be set to None.
    capacity_fractions : RealArray, shape (K,) | None
        The fractions of channel capacities that are allocated to each UT. \\
        If the same capacity fraction is allocated to all UTs, this can also be provided as a single float at the moment of initialization. \n
        For adaptive bit allocation, the bit allocator computes the achievable rates (shannon capacity) for each stream of all UTs. Then it calculates the information bit rates for the data streams to each UT as the fraction of their achievable rates.\\
        In case of fixed bit allocation, the information bit rates are predetermined. In that case, this can be set to None.
    """
    types: ConstType | list[ConstType]
    sizes: int | IntArray | None = None
    capacity_fractions: float | RealArray | None = None

    def __eq__(self, other: object) -> bool:
        
        if not isinstance(other, ConstConfig):
            return NotImplemented
        
        return (
            self.types == other.types and
            np.array_equal(self.sizes, other.sizes) and
            np.array_equal(self.capacity_fractions, other.capacity_fractions)
        )


@dataclass()
class BaseStationConfig:
    """
    The configuration settings of a base station.

    Parameters
    ----------
    bit_loader : type[BitLoader]
        The type of the bit loader (the concrete bit loader class).
    mapper : type[Mapper]
        The type of the mapper (the concrete mapper class).
    precoder : type[Precoder]
        The type of the precoder (the concrete precoder class).
    """
    bit_loader: type["BitLoader"]
    mapper: type["Mapper"]
    precoder: type["Precoder"]

@dataclass()
class UserTerminalConfig:
    """
    The configuration settings of a user terminal.

    Parameters
    ----------
    combiner : type[Combiner]
        The type of the combiner (the concrete combiner class).
    equalizer : type[Equalizer]
        The type of the equalizer (the concrete equalizer class).
    detector : type[Detector]
        The type of the detector (the concrete detector class).
    demapper : type[Demapper]
        The type of the demapper (the concrete demapper class).
    """
    combiner: type["Combiner"]
    equalizer: type["Equalizer"]
    detector: type["Detector"]
    demapper: type["Demapper"]

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
    channel_model: type["ChannelModel"]
    noise_model: type["NoiseModel"]

@dataclass()
class SystemConfig:
    """
    The configuration settings of a MU-MIMO system.

    Parameters
    ----------
    Pt : float
        The total available transmit power (in Watt).
    B : float
        The system frequency bandwidth (in Hertz).

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

    Pt: float
    B: float

    K: int
    Nt: int
    Nr: int

    c_configs: ConstConfig

    base_station_configs: BaseStationConfig
    user_terminal_configs: UserTerminalConfig
    channel_configs: ChannelConfig


    def __post_init__(self):

        # Validate dimensions.
        if self.K <= 0: raise ValueError("The number of UTs must be a positive integer.")
        if self.Nt <= 0: raise ValueError("The number of transmit antennas must be a positive integer.")
        if self.Nr <= 0: raise ValueError("The number of receive antennas per UT must be a positive integer.")
        if self.K * self.Nr > self.Nt: raise ValueError("The BS must have at least as many transmit antennas as the total number of receive antennas across all UTs.")

        # Validate constellation types.
        if isinstance(self.c_configs.types, str):
            self.c_configs.types = [self.c_configs.types] * self.K
        elif len(self.c_configs.types) != self.K:
            raise ValueError("The number of constellation types must match K.")

        # Validate constellation sizes.
        if isinstance(self.c_configs.sizes, int):
            self.c_configs.sizes = np.array([self.c_configs.sizes] * self.K, dtype=int)
        elif isinstance(self.c_configs.sizes, np.ndarray) and len(self.c_configs.sizes) != self.K:
            raise ValueError("The number of constellation sizes must match K.")

        # Validate capacity fractions.
        if isinstance(self.c_configs.capacity_fractions, (int, float)):
            self.c_configs.capacity_fractions = np.array([self.c_configs.capacity_fractions] * self.K, dtype=float)
        elif isinstance(self.c_configs.capacity_fractions, np.ndarray):
            if len(self.c_configs.capacity_fractions) != self.K:
                raise ValueError("The number of capacity fractions must match K.")
            if not (np.all(self.c_configs.capacity_fractions >= 0) and np.all(self.c_configs.capacity_fractions <= 1)):
                raise ValueError("Capacity fractions must be between 0 and 1.")
        
    def __eq__(self, other: object) -> bool:
    
        if not isinstance(other, SystemConfig):
            return NotImplemented
        
        return (
            self.Pt == other.Pt and
            self.B == other.B and
            self.K == other.K and
            self.Nt == other.Nt and
            self.Nr == other.Nr and
            self.c_configs == other.c_configs and
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
    M : int
        The number of symbol vector transmissions for each channel realization.
    """

    snr_dB_values: RealArray = field(default_factory=lambda: np.arange(-5, 31, 2.5))
    num_channel_realizations: int = 200
    num_bit_errors: int = 250
    num_bit_errors_scope: Literal["system-wide", "uts", "streams"] = "uts"
    M: int = 4800

    @property
    def snr_values(self) -> RealArray:
        return 10 ** (self.snr_dB_values / 10)

    def __post_init__(self):
        if self.num_channel_realizations <= 0: raise ValueError("The minimum number of channel realizations must be a positive integer.")
        if self.num_bit_errors <= 0: raise ValueError("The minimum number of bit errors per SNR value must be a positive integer.")
        if self.M <= 0: raise ValueError("The minimum number of symbol vector transmissions for each channel realization must be a positive integer.")

    def __eq__(self, other: object) -> bool:
        
        if not isinstance(other, SimConfig):
            return NotImplemented
        
        return (
            np.array_equal(self.snr_dB_values, other.snr_dB_values) and
            self.num_channel_realizations == other.num_channel_realizations and
            self.num_bit_errors == other.num_bit_errors and
            self.num_bit_errors_scope == other.num_bit_errors_scope and
            self.M == other.M
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
    
    ut_ibrs : IntArray, shape (K,)
        Per-UT information bit rates.
    ut_becs : RealArray, shape (K,)
        Per-UT bit error counts.
    ut_ars : BitArray, shape (K,)
        Per-UT UT activation rates (1 if the UT is active, 0 otherwise).
    
    ibr : float
        System-wide information bit rate.
    bec : float
        System-wide bit error count.
    
    stream_ars_avg : float
        Average stream activation rate.
    ut_ars_avg : float
        Average UT activation rate.
    
    M : int
        The number of symbol vector transmissions for each channel realization.
    num_channel_realizations : int
        The number of channel realizations that were simulated.
    
    stream_bers : list[RealArray] (list of K arrays, each shape (Nr,)) | None
        Per-UT per-stream bit error rates. None if num_channel_realizations == 1.
    ut_bers : RealArray, shape (K,) | None
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

    M : int
    num_channel_realizations : int


    stream_bers : list[RealArray] | None = None
    ut_bers : RealArray | None = None
    ber: float | None = None

    def __post_init__(self):

        if self.num_channel_realizations > 1:

            K = len(self.stream_ibrs)

            self.stream_bers = []
            for k in range(K):
                denom = self.stream_ibrs[k] * self.M * self.num_channel_realizations
                ber_k = np.where(denom > 0, self.stream_becs[k] / denom, np.nan)
                self.stream_bers.append(ber_k)
            
            ut_denom = self.ut_ibrs * self.M * self.num_channel_realizations
            self.ut_bers = np.where(ut_denom > 0, self.ut_becs / ut_denom, np.nan)
            
            total_denom = self.ibr * self.M * self.num_channel_realizations
            self.ber = self.bec / total_denom if total_denom > 0 else np.nan

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


__all__ = [
    "RealArray", "ComplexArray", "IntArray", "BitArray", "ConstType",
    "ChannelStateInformation", "ConstConfig",
    "BaseStationState", "UserTerminalState",
    "TransmitPilotMessage", "ReceivePilotMessage",
    "TransmitFeedbackMessage", "ReceiveFeedbackMessage",
    "TransmitFeedforwardMessage", "ReceiveFeedforwardMessage",
    "BaseStationConfig", "UserTerminalConfig", "ChannelConfig",
    "SystemConfig", "SimConfig", "SingleSnrSimResult", "SimResult",
]
