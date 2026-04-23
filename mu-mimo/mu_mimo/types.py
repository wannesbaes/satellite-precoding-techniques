# mu-mimo/mu_mimo/types.py

from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray
from typing import Literal

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
    H_hat : ComplexArray, shape (p, K * Nr, Nt)
        The p most up-to-date channel estimates.
    """
    snr: float | None = None
    H_hat: ComplexArray | None = None

@dataclass()
class ChannelState:
    """
    The state of the channel.

    Parameters
    ----------
    H : ComplexArray, shape (K * Nr, Nt) or (Msv, K * Nr, Nt)
        The channel matrix.\\
        In case of a static channel, the channel matrix has shape (K * Nr, Nt).
        In case of a time-varying channel, the channel matrix has shape (Msv, K * Nr, Nt), where Msv is the number of channel uses for a single data transmission block.
    csi : ChannelStateInformation
        The channel state information (CSI) corresponding to this channel state.\\
        Depending on the delay on the CSI feedback message, the CSI may be outdated! 
    """
    H: ComplexArray | None = None
    csi: ChannelStateInformation | None = None

@dataclass()
class BaseStationState:
    """
    The state of a base station (for a specific channel and SNR value).

    Parameters
    ----------
    F : ComplexArray, shape (Nt, K * Nr)
        The compound precoding matrix.
    C_eq : ComplexArray, shape (K * Nr,)
        The equalization coefficients for each data stream.
    ibr : IntArray, shape (K * Nr,)
        The information bit rates for each data stream of each UT.
    Ns : IntArray, shape (K,)
        The number of data streams for each UT.
    G : ComplexArray, shape (K * Nr, K * Nr) or None
        The compound combining matrix. It is None in case of non-coordinated beamforming.
    """
    F: ComplexArray
    C_eq: ComplexArray
    ibr: IntArray
    Ns: IntArray
    G: ComplexArray | None

@dataclass()
class UserTerminalState:
    """
    The state of a single user terminal (for a specific channel and SNR value).

    Parameters
    ----------
    snr_k : float
        The signal-to-noise ratio for this UT.
    H_k_hathat : ComplexArray, shape (Nr, Nt)
        The estimated and/or predicted most up-to-date channel matrix for this UT.
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
    snr_k: float
    H_k_hathat: ComplexArray
    G_k: ComplexArray
    c_type_k: ConstType | None
    C_eq_k: ComplexArray | None
    ibr_k: IntArray | None


@dataclass()
class TransmitPilotMessage:
    """
    The pilot message transmitted by the BS.
    """
    pass

@dataclass()
class ReceivePilotMessage:
    """
    The pilot message received by the UT.

    Parameters
    ----------
    csi_k : ChannelStateInformation
        The channel state information (CSI) corresponding to the CSI in UT k after receiving the pilot messages and estimating the channel.
    
    Note
    ----
    Normally, the received pilot message would contain the received pilot symbols.
    However, since channel estimation is not yet considered in this framework, the CSI is directly included in the received pilot message.
    """
    csi_k: ChannelStateInformation

@dataclass()
class TransmitFeedbackMessage:
    """
    The feedback message transmitted by the UT.

    Parameters
    ----------
    ut_id : int
        The ID of the UT that transmits this feedback message.
    snr_k : float
        The signal-to-noise ratio for this UT.
    H_eff_k : ComplexArray, shape (Nr, Nt)
        The effective channel matrix for this UT.
    """
    ut_id: int
    snr_k: float
    H_eff_k: ComplexArray

@dataclass()
class ReceiveFeedbackMessage:
    """
    The feedback message received by the BS.

    Parameters
    ----------
    snr : float
        The signal-to-noise ratio.
    H_eff : ComplexArray, shape (K * Nr, Nt)
        The compound effective channel matrix.
    """
    snr : float
    H_eff : ComplexArray

@dataclass()
class TransmitFeedforwardMessage:
    """
    The feedforward message transmitted by the BS.

    Parameters
    ----------
    c_type : list[ConstType]
        The constellation types for the data streams to each UT.
    C_eq : ComplexArray, shape (K * Nr,)
        The equalization coefficients for each data stream of each UT.
    ibr : IntArray, shape (K * Nr,)
        The information bit rates for each data stream of each UT.
    G : ComplexArray, shape (K * Nr, K * Nr)
        The compound combining matrix. It is None in case of non-coordinated beamforming.
    """
    c_type: list[ConstType]
    C_eq: ComplexArray
    ibr: IntArray
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
    G_k : ComplexArray, shape (Nr, Nr)
        The combining matrix of this UT. It is None in case of non-coordinated beamforming.
    """
    ut_id: int
    c_type_k: ConstType
    C_eq_k: ComplexArray
    ibr_k: IntArray
    G_k: ComplexArray | None


__all__ = [
    "RealArray", "ComplexArray", "IntArray", "BitArray", "ConstType",
    "ChannelStateInformation",
    "ChannelState", "BaseStationState", "UserTerminalState",
    "TransmitPilotMessage", "ReceivePilotMessage",
    "TransmitFeedbackMessage", "ReceiveFeedbackMessage",
    "TransmitFeedforwardMessage", "ReceiveFeedforwardMessage",
]
