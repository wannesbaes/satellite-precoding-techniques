# mu-mimo/mu_mimo/processing/precoding.py

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from ..types import ComplexArray, RealArray, IntArray, BitArray, ChannelStateInformation


class Precoder(ABC):
    """
    The Precoder Abstract Base Class (ABC).

    A precoder class is responsible for implementing a precoding strategy and effectively precoding the transmitted signals in the base station.\\
    In case of coordinated beamforming, the precoder is responsible for computing the combining matrices for each UT as well! These are then later sent to the UTs.

    In addition, the precoder is responsible for computing the equalization coefficients for each data stream. These will be used by the UTs to correctly rescale the received symbols before the decoding process.
    """

    @staticmethod
    @abstractmethod
    def compute(csi: ChannelStateInformation, Pt: float, K: int,) -> tuple[ComplexArray, ComplexArray | None, ComplexArray]:
        """
        Compute the compound precoding matrix.\\
        In case of coordinated beamforming, the combining matrices for each UT are computed as well.

        In addition, the equalization coefficients for each data stream are computed as well. These will be used by the UTs to correctly rescale the received symbols before the decoding process.
        
        Parameters
        ----------
        csi : ChannelStateInformation
            The channel state information (CSI) of the system.
        Pt : float
            The total transmit power available at the BS.
        K : int
            The number of user terminals (UTs) in the system.
        
        Returns
        -------
        F : ComplexArray, shape (Nt, K*Nr)
            The compound precoding matrix for all UTs.
        G : ComplexArray, shape (K*Nr, K*Nr) or None
            The compound combining matrix (block diagonal) for all UTs in case of coordinated beamforming. None otherwise.
        C_eq : ComplexArray, shape (K*Nr,)
            The equalization coefficients for all UTs.
        """
        raise NotImplementedError

    @staticmethod
    def apply(a: ComplexArray, F: ComplexArray, ibr: IntArray) -> ComplexArray:
        """
        Apply the precoding matrix to the data symbols.

        Parameters
        ----------
        a : ComplexArray, shape (Ns_total, M)
            The data symbol streams for all UTs.
        F : ComplexArray, shape (Nt, K*Nr)
            The compound precoding matrix for all UTs.
        ibr : IntArray, shape (K*Nr,)
            The number of bits per symbol for each data stream (active and inactve).

        Returns
        -------
        x : ComplexArray, shape (Nt, M)
            The precoded signal to be transmitted by the BS.
        """

        x = F[:, ibr > 0] @ a
        return x

class NeutralPrecoder(Precoder):
    """
    Neutral Precoder.

    This precoder acts as a 'neutral element' for precoding.\\
    It does not perform any precoding and simply passes the data symbols through without any modification.

    In addition, the power allocation is uniform across the data streams (so no real power allocation is performed either).

    Finally, in case of coordinated beamforming, the combining matrices for each UT are set to the identity matrix (so no real combining is performed either).
    """

    @staticmethod
    def compute(csi: ChannelStateInformation, Pt: float, K: int) -> tuple[ComplexArray, ComplexArray | None, ComplexArray]:

        Nt = csi.H_eff.shape[1]
        Nr = csi.H_eff.shape[0] // K

        F = np.eye(Nt, K*Nr)
        G = np.eye(K*Nr)
        C_eq = np.ones(K*Nr)
        return F, G, C_eq

class ZFPrecoder(Precoder):
    """
    Zero-Forcing (ZF) Precoder.

    The precoder aims to completely eliminate all interference at the user terminals.\\
    The precoding matrix is therefore computed as the pseudo-inverse of the effective channel matrix H_eff.

    In addition, the power allocation across the data streams is optimal in the sense that it maximizes the sum rate of the system under the total power constraint Pt.\\
    """

    @staticmethod
    def compute(csi: ChannelStateInformation, Pt: float, K: int) -> tuple[ComplexArray, ComplexArray | None, ComplexArray]:

        # The combining matrix is not computed in the ZF precoding strategy.
        G = None

        # The power allocation across the data streams is computed using the waterfilling algorithm to maximize the sum rate under the total power constraint Pt.
        gamma = np.real(np.diag(np.linalg.inv(csi.H_eff @ csi.H_eff.conj().T))) * (csi.snr / Pt)
        P = ZFPrecoder._waterfilling_v1(gamma=gamma, pt=Pt)

        # The precoding matrix is computed as the pseudo-inverse of the effective channel matrix.
        F = np.linalg.pinv(csi.H_eff)
        FP = F @ np.diag(np.sqrt(P))

        # Compute the equalization coefficients.
        C_eq = ZFPrecoder._equalization_coefficients(csi.H_eff, F, P)

        return FP, G, C_eq
    
    @staticmethod
    def _equalization_coefficients(H_eff: ComplexArray, F: ComplexArray, P: ComplexArray) -> ComplexArray:
        r"""
        Compute the equalization coefficients for each data stream.

        .. math::
            C_{eq \; (k,nr)}  = \left( G H F \right)_{(k,nr),(k,nr)} \cdot \sqrt{P_{(k,nr)}}

        Parameters
        ----------
        H_eff : ComplexArray, shape (K*Nr, Nt)
            The effective channel matrix.
        F : ComplexArray, shape (Nt, K*Nr)
            The compound precoding matrix for all UTs.
        P : ComplexArray, shape (K*Nr,)
            The power allocation for each data stream.
        
        Returns
        -------
        C_eq : ComplexArray, shape (K*Nr,)
            The equalization coefficients for each data stream.
        """
        C_eq = np.diag( H_eff @ F @ np.diag(np.sqrt(P)) )
        return C_eq
    
    @staticmethod
    def _waterfilling_v1(gamma, pt):
        r"""
        Waterfilling algorithm.

        This function implements the waterfilling algorithm to find the optimal power allocation across N transmission streams, given the channel-to-noise ratio (CNR) coefficients `gamma` and the total available transmit power `pt`.

        In particular, it solves the following constraint optimization problem:

        .. math::

            \begin{aligned}
                & \underset{\{p_n\}}{\text{max}}
                & & \sum_{n=1}^{N} \log_2 \left( 1 + \gamma_n \, p_n \right) \\
                & \text{s. t.}
                & & \sum_{n=1}^{N} p_n = p_t \\
                & & & \forall n \in \{1, \ldots, N\} : \, p_n \geq 0
            \end{aligned}

        Parameters
        ----------
        gamma : RealArray, shape (N,)
            Channel-to-Noise Ratio (CNR) coefficients for each eigenchannel.
        pt : float
            Total available transmit power.

        Returns
        -------
        p : RealArray, shape (N,)
            Optimal power allocation across the eigenchannels.
        """

        # STEP 0: Sort the CNR coefficients in descending order.
        sorted_indices = np.argsort(gamma)[::-1]
        gamma = gamma[sorted_indices]

        # STEP 1: Determine the number of active streams.
        pt_iter = lambda as_iter: np.sum( (1 / gamma[as_iter]) - (1 / gamma[:as_iter]) )
        as_UB = len(gamma)
        as_LB = 0

        while as_UB - as_LB > 1:
            as_iter = (as_UB + as_LB) // 2
            if pt > pt_iter(as_iter): as_LB = as_iter
            elif pt < pt_iter(as_iter): as_UB = as_iter
        
        # STEP 2: Compute the optimal power allocation for each active stream.
        p_step1 = ( (1 / gamma[as_LB]) - (1 / gamma[:as_LB]) )
        p_step1 = np.concatenate( (p_step1, np.zeros(as_UB - as_LB)) )

        power_remaining = pt - np.sum(p_step1)
        p_step2 = (1 / as_UB) * power_remaining

        p_sorted = np.concatenate( (p_step1 + p_step2, np.zeros(len(gamma) - as_UB)) )

        # STEP 3: Reorder the power allocation to match the original order of the streams.
        p = np.empty_like(p_sorted)
        p[sorted_indices] = p_sorted

        return p
