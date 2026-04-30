# mu-mimo/mu_mimo/processing/channel.py

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter
from scipy.special import j0
from scipy.special import i0, i1
from scipy.optimize import brentq
from scipy.linalg import toeplitz

from ..types import ComplexArray, RealArray, IntArray



# CHANNEL MODELS

class ChannelModel(ABC):
    """
    The Channel Model Abstract Base Class (ABC).

    A channel model is responsible for generating the channel matrix according to a specific channel model type, and the corresponding CSI feedback message.
    Also, it is responsible for applying the channel effects to the transmitted symbol vectors.
    """

    def __init__(self, Nt: int, Nr: int, K: int):
        """
        Instantiate the channel model.

        Parameters
        ----------
        Nt : int
            The number of transmit antennas at the base station.
        Nr : int
            The number of receive antennas per user terminal.
        K : int
            The number of user terminals.
        """

        self.Nt = Nt
        self.Nr = Nr
        self.K = K

        self.Msv = None
        self._H = None
        self._H_CSI = None
    
    def __str__(self) -> str:
        """ Return a string representation of the channel model. """
        return "Channel Model (abstract base class)"
    
    def __eq__(self, other: object) -> bool:
        
        if not isinstance(other, ChannelModel):
            return NotImplemented
        
        return (
            type(self) == type(other) and
            self.Nt == other.Nt and
            self.Nr == other.Nr and
            self.K == other.K
        )
    
    def get_CSI(self) -> ComplexArray:
        """
        Get the channel state information (CSI) feedback message.\\
        This message includes the p most up-to-date channel estimates, where p is the order of the AR-model at the BS (1 by default).
        
        Returns
        -------
        H : ComplexArray, shape (p, K*Nr, Nt)
            The channel matrices corresponding to the p previous channel estimates.\\
            If the channel model is known to both BS and UTs, it contains the parameters of the channel model instead of the channel matrices.
        
        Notes
        -----
        The channel estimation is in practice done by the UTs. 
        However, in our simulation model, we assume that this estimation is perfect, without any errors.
        """
        H_CSI = self._H_CSI
        return H_CSI
    
    @abstractmethod
    def generate(self) -> ComplexArray:
        """
        Generate a new channel and store the CSI corresponding to this channel.\\
        In case of a fading channel, it generates the channel matrices for all symbol vector transmissions of the consecutive block.

        Returns
        -------
        H : ComplexArray, shape (K*Nr, Nt) or (Msv, K*Nr, Nt)
            The new channel matrix.
        """
        raise NotImplementedError
    
    def apply(self, x: ComplexArray) -> ComplexArray:
        """
        Apply the channel effects to the transmitted signals.

        Parameters
        ----------
        x : ComplexArray, shape (Nt, Msv)
            The transmitted signals.

        Returns
        -------
        y : ComplexArray, shape (K*Nr, Msv)
            The received signals.
        """
        
        if len(self._H.shape) == 2:
            y = self._H @ x
        
        elif len(self._H.shape) == 3:
            y = np.array([ self._H[m] @ x[:, m] for m in range(self.Msv) ]).T
        
        else:
            raise ValueError(f"Invalid channel matrix shape: {self._H.shape}.\nExpected a shape of (K*Nr, Nt) or (Msv, K*Nr, Nt) in case of a fading channel.")
        
        return y

class NeutralChannel(ChannelModel):
    """
    Neutral Channel.
    
    This channel acts as a 'neutral element' for the channel model.\\
    In particular, it generates an identity channel matrix, which means that the symbols are transmitted to the receive antennas for which they are intended, and without any interference.
    """

    def __str__(self) -> str:
        return "Neutral Channel"

    def generate(self) -> ComplexArray:
        H = np.eye(self.K*self.Nr, self.Nt, dtype=complex)
        self._H = H
        self._H_CSI = H[np.newaxis, :, :]
        return H
    
class IIDRayleighChannel(ChannelModel):
    """
    Independent and Identically Distributed (IID) Rayleigh Channel.

    The channel gains of each propagation link are independent and identically distributed (IID) circularly-symmetric zero-mean unit-variance complex Gaussian distributed.\\
    """

    def __str__(self) -> str:
        return "IID Rayleigh Channel"
    
    def generate(self) -> ComplexArray:
        H = (1 / np.sqrt(2)) * (np.random.randn(self.K*self.Nr, self.Nt) + 1j * np.random.randn(self.K*self.Nr, self.Nt))
        self._H = H
        self._H_CSI = H[np.newaxis, :, :]
        return H

class RiceanIIDTCChannel(ChannelModel):
    r"""
    The Ricean Channel with an IID time-correlated fading NLoS component.

    The LoS component is modeled as a deterministic component independent of time and across users.
    The NLoS component follows Jake's model.

    .. math::

        H_k(t) = e^{j \theta_k} \left(\sqrt{\frac{K}{K+1}} + \sqrt{\frac{1}{K+1}} \mathbf{H}_{\text{NLoS},k}(t)\right)

    The parameters are:

    - :math:`K \in [0, +\infty)`: The Rice factor. It quantifies the strength of the deterministic LoS component relative to the scattered multipath.
    - :math:`\theta_k`: The arbitrary channel phase, uniformly distributed over $[-\pi, \pi]$ and independent across users $k$.
    - :math:`\mathbf{H}_{\text{NLoS},k}(t)`: The NLoS components. Each entry is an i.i.d. zero-mean unit-variance complex Gaussian process correlated in time. The PSD is:
    
    .. math::

        S(f) = \frac{1}{\pi f_D \sqrt{1 - \left( \frac{f}{f_D} \right)^2}}, \quad |f| < f_D

    """

    def __init__(self, Nt: int, Nr: int, K: int, K_rician: float, Trtt_2_Tc: float, Tpilot_2_Tc: float, Twindow_2_Tc: float = 2):
        r"""
        Instantiate the channel.

        Parameters
        ----------
        Nt : int
            The number of transmit antennas at the base station.
        Nr : int
            The number of receive antennas per user terminal.
        K : int
            The number of user terminals.
        
        K_rician : float
            The Rice factor. It quantifies the strength of the deterministic LoS component relative to the scattered multipath.
        
        Trtt_2_Tc : float
            The round trip time :math:`T_{\text{RTT}}` relative to the coherence time :math:`T_c`.
        Tpilot_2_Tc : float
            The time period between two CSI feedback messages relative to the coherence time :math:`T_c`.
        Twindow_2_Tc : float, optional
            The time window over which the channel will be generated relative to the coherence time :math:`T_c`.
            (This also sets an upper bound on the order of the AR predictor.)
        """
        
        # Initialize the base class.
        super().__init__(Nt, Nr, K)

        # Validate the parameters.
        assert K_rician >= 0, f"The Rice factor K_rician must be non-negative.\nCurrent value: {K_rician}"
        assert 0 <= Trtt_2_Tc, f"The round trip time relative to the coherence time must be non-negative.\nCurrent value: {Trtt_2_Tc}"
        assert 0 < Tpilot_2_Tc <= Trtt_2_Tc or Trtt_2_Tc == 0, f"The time period between two CSI feedback messages relative to the coherence time must be positive and less than or equal to the round trip time.\nCurrent value: {Tpilot_2_Tc}"
        assert Twindow_2_Tc > 0, f"The time window relative to the coherence time must be positive.\nCurrent value: {Twindow_2_Tc}"
        
        # Store the channel parameters.
        self.K_rician = K_rician
        self.Trtt_2_Tc = Trtt_2_Tc
        self.Tpilot_2_Tc = Tpilot_2_Tc
        self.Twindow_2_Tc = Twindow_2_Tc

        
        # Compute and store the helper parameters

        # the number of channel samples to generate the NLoS process for (we generate 2 coherence periods, always)
        self.num_samples = int((1 / Tpilot_2_Tc) * self.Twindow_2_Tc)

        # the delay on the CSI feedback message (in number of channel samples)
        self.CSI_delay = int(np.ceil(Trtt_2_Tc / Tpilot_2_Tc))

        # the maximum Doppler frequency times the coherence time
        self.fD_times_Tc = (1 / (2*np.pi)) * brentq(lambda z: j0(z) - 0.5, 0, 2.5)

        return

    def __str__(self) -> str:
        return f"Ricean Channel with an IID time-correlated fading NLoS component.\n   K_rician = {round(10 * np.log10(self.K_rician))} dB, RTT = {np.round(self.Trtt_2_Tc, 2)} Tc, CSI feedback rate = {np.round(1 / self.Tpilot_2_Tc, 1)} messages per Tc"
    
    def __eq__(self, other):
        
        if not isinstance(other, RiceanIIDTCChannel):
            return NotImplemented
        
        return (
            self.Nt == other.Nt and
            self.Nr == other.Nr and
            self.K == other.K and
            self.K_rician == other.K_rician and
            self.Trtt_2_Tc == other.Trtt_2_Tc and
            self.Tpilot_2_Tc == other.Tpilot_2_Tc
        )

    def generate(self) -> ComplexArray:

        # Generate the arbitrary channel phase.
        theta_k = np.random.uniform(low=-np.pi, high=np.pi, size=self.K)
        theta = np.repeat(theta_k, self.Nr*self.Nt).reshape(self.K*self.Nr, self.Nt)

        # Generate the NLoS components for all propagation links and time instants.
        H_NLoS_process = self._generate_NLoS(self.num_samples)

        # Compute the current channel matrix.
        H_NLoS = H_NLoS_process[-1]
        H = np.exp(1j * theta) * (np.sqrt(self.K_rician / (self.K_rician + 1)) + np.sqrt(1 / (self.K_rician + 1)) * H_NLoS)

        # Compute the CSI corresponding to the current channel.
        H_NLoS_CSI = H_NLoS_process[:self.num_samples-self.CSI_delay]
        H_CSI = np.exp(1j * theta) * (np.sqrt(self.K_rician / (self.K_rician + 1)) + np.sqrt(1 / (self.K_rician + 1)) * H_NLoS_CSI)
        
        # Update the values of the current channel and the corresponding CSI.
        self._H = H
        self._H_CSI = H_CSI

        return H

    def _generate_NLoS(self, num_channel_samples: int) -> ComplexArray:
        r"""
        Generate the NLoS component for user k using the Cholesky decomposition method for generating a Gaussian process with a specified auto-correlation function.

        The Cholesky decomposition method for generating a zero-mean unit-variance complex Gaussian process :math:`\mathbf{h}` of length :math:`N` with a specified autocorrelation function :math:`R_h(\tau)`:
        
        - **Step 1:** Build the :math:`N \times N` covariance matrix :math:`\mathbf{C}` with entries :math:`C_{i,j} = R_h((i-j) \, T_{\text{sample}})`.
        - **Step 2:** Compute the Cholesky decomposition of the covariance matrix :math:`\mathbf{C} = \mathbf{L}\mathbf{L}^H`, where :math:`\mathbf{L}` is a lower triangular matrix.
        - **Step 3:** Generate a column vector :math:`\mathbf{w}` of :math:`N` i.i.d. white complex Gaussian random variables with zero mean and unit variance.
        - **Step 4:** Find the desired Gaussian process as :math:`\mathbf{h} = \mathbf{L} \, \mathbf{w}`.

        Parameters
        ----------
        num_channel_samples : int
            The total number of channel samples to generate.\\
            This is equal to the total number of symbol vectors that will be transmitted during the simulation plus the delay on the CSI feedback message in symbol vector transmissions.
        
        Returns
        -------
        H_NLoS : ComplexArray, shape (num_channel_samples, K * Nr, Nt)
            The generated NLoS component for all propagation links and all time instants.
        
        Note
        ----
        For Jake's model, where the PSD of the Gaussian processes equals the Doppler spectrum, the autocorrelation function :math:`R_h(\tau)` corresponds to a zeroth-order Bessel function of the first kind :math:`J_0(2\pi f_D \tau)`.

        The sample period :math:`T_{\text{sample}}` of the channel realizations is equal to the symbol period :math:`T_{\text{symbol}}`.
        We can compute the covariance matrix in function of the symbol period to coherence time ratio:
        
        .. math::
            R_h((i-j) \, T_{\text{sample}}) 
            &= R_h((i-j) \, T_{\text{symbol}}) \\
            &= J_0(2\pi \, f_D \cdot (i-j) \, T_{\text{symbol}}) \\
            &= J_0(2\pi \, (f_D \, T_c) \cdot (i-j) \, \frac{T_{\text{symbol}}}{T_c})
            
        where we used
        
        .. math::
            R_h(\tau) \geq \frac{1}{2} \iff \left| 2\pi f_D \tau \right| \leq 1.521 \implies \tau_c \approx (0.242) \cdot \frac{1}{f_D}
        
        """

        # STEP 1.
        Tsample_2_Tc = self.Tpilot_2_Tc
        C = toeplitz( j0(2*np.pi * (self.fD_times_Tc * Tsample_2_Tc) * np.arange(num_channel_samples))  )
        C += (1e-10) * np.eye(num_channel_samples)

        # STEP 2.
        L = np.linalg.cholesky(C)

        # STEP 3.
        w = (1 / np.sqrt(2)) * (np.random.randn(num_channel_samples, self.K*self.Nr*self.Nt) + 1j * np.random.randn(num_channel_samples, self.K*self.Nr*self.Nt))

        # STEP 4.
        H_NLoS = (L @ w).reshape(num_channel_samples, self.K*self.Nr, self.Nt)

        return H_NLoS

class SatelliteChannel(ChannelModel):
    """
    Satellite Channel.
    """

    def __init__(self, Nt: int, Nr: int, K: int, K_rician: float, Trtt_2_Tc: float, Tpilot_2_Tc: float, Twindow_2_Tc: float = 2, L1: int = 2, w1: float = 0.75, L2: int = 2, w2: float = 0.25, dx_BS: float = 1, dy_BS: float = 1, dx_UT: float = 0.5, dy_UT: float = 0.5, theta_max: float = 30 * (np.pi / 180), sigma_theta: float = 4 * (np.pi / 180), sigma_phi: float = 4 * (np.pi / 180)):
        r"""
        Instantiate the Satellite Channel.

        Parameters
        ----------
        Nt : int
            Total number of transmit antennas at the BS.
        Nr : int
            Number of receive antennas per UT.
        K : int
            Number of user terminals.
        K_rician : float
            Rice factor.  Ratio of LoS power to scattered power.
        Trtt_2_Tc : float
            Round-trip time relative to the coherence time :math:`T_{\text{RTT}} / T_c`.
        Tpilot_2_Tc : float
            Time period between two CSI message acquisitions relative to the coherence time :math:`T_{\text{pilot}} / T_c`.
        Twindow_2_Tc : float, optional
            Length of the generated time window relative to :math:`T_c`.  Default: 2.
        
        L1 : int, optional
            Number of NLoS rays per UT in the first cluster. Default: 2.
        w1 : float, optional
            Weight of the first cluster.  Default: 0.75.
        L2 : int, optional
            Number of NLoS rays per UT in the second cluster. Default: 2.
        w2 : float, optional
            Weight of the second cluster.  Default: 0.25.
        dx_BS, dy_BS : float, optional
            Normalised BS antenna spacings at the BS (satellite) :math:`d/\lambda`.  Default: 1.
        dx_UT, dy_UT : float, optional
            Normalised UT antenna spacings at the UT :math:`d/\lambda`.  Default: 0.5.
        theta_max : float, optional
            Maximum elevation angle [rad].  Default: 30°.
        sigma_theta : float, optional
            Laplace scale for NLoS elevation perturbations [rad].  Default: 4°.
        sigma_phi : float, optional
            Laplace scale for NLoS azimuth perturbations [rad].  Default: 4°.
        """
        super().__init__(Nt, Nr, K)

        # Validate parameters.
        assert K_rician >= 0,             f"K_rician must be non-negative. Got {K_rician}"
        assert L1 >= 1,                   f"L1 must be >= 1. Got {L1}"
        assert L2 >= 1,                   f"L2 must be >= 1. Got {L2}"
        assert w1 + w2 == 1,              f"w1 and w2 must sum to 1. Got w1={w1}, w2={w2}"
        assert Trtt_2_Tc >= 0,            f"Trtt_2_Tc must be non-negative. Got {Trtt_2_Tc}"
        assert 0 < Tpilot_2_Tc <= Trtt_2_Tc or Trtt_2_Tc == 0, f"Tpilot_2_Tc must be in (0, Trtt_2_Tc]. Got {Tpilot_2_Tc}"
        assert Twindow_2_Tc > 0,          f"Twindow_2_Tc must be positive. Got {Twindow_2_Tc}"

        # Store channel parameters.
        self.K_rician     = K_rician

        self.Trtt_2_Tc    = Trtt_2_Tc
        self.Tpilot_2_Tc  = Tpilot_2_Tc
        self.Twindow_2_Tc = Twindow_2_Tc

        self.L1            = L1
        self.w1            = w1
        self.L2            = L2
        self.w2            = w2

        self.dx_BS        = dx_BS
        self.dy_BS        = dy_BS
        self.dx_UT        = dx_UT
        self.dy_UT        = dy_UT

        self.theta_max    = theta_max
        self.sigma_theta  = sigma_theta
        self.sigma_phi    = sigma_phi

        # Compute and store the helper parameters for the array response vector computations.
        self.Nx_BS        = next(Nx_BS for Nx_BS in range(int(math.isqrt(Nt)), 0, -1) if Nt % Nx_BS == 0)
        self.Ny_BS        = Nt // self.Nx_BS
        self.Nx_UT        = next(Nx_UT for Nx_UT in range(int(math.isqrt(Nr)), 0, -1) if Nr % Nx_UT == 0)
        self.Ny_UT        = Nr // self.Nx_UT

        # Compute and store the helper parameters for the time-correlated fading of the NLoS component.
        self.num_samples  = int((1 / Tpilot_2_Tc) * Twindow_2_Tc)
        self.CSI_delay    = int(np.ceil(Trtt_2_Tc / Tpilot_2_Tc))
        self.fD_times_Tc  = (1 / (2 * np.pi)) * brentq(lambda z: j0(z) - 0.5, 0, 2.5)

    def __str__(self) -> str:
        return f"Ricean Satellite Channel with space- and time-correlated fading NLoS component.\n   K_rician = {round(10 * np.log10(self.K_rician))} dB, RTT = {np.round(self.Trtt_2_Tc, 2)} Tc, CSI feedback rate = {np.round(1 / self.Tpilot_2_Tc, 1)} messages per Tc, {self.L1} and {self.L2} rays in the first and second clusters per UT"

    def __eq__(self, other):
        
        if not isinstance(other, SatelliteChannel):
            return NotImplemented
        
        return (
            self.Nt == other.Nt and
            self.Nr == other.Nr and
            self.K == other.K and
            self.K_rician == other.K_rician and
            self.Trtt_2_Tc == other.Trtt_2_Tc and
            self.Tpilot_2_Tc == other.Tpilot_2_Tc and
            self.Twindow_2_Tc == other.Twindow_2_Tc and
            self.L1 == other.L1 and
            self.w1 == other.w1 and
            self.L2 == other.L2 and
            self.w2 == other.w2 and
            self.dx_BS == other.dx_BS and
            self.dy_BS == other.dy_BS and
            self.dx_UT == other.dx_UT and
            self.dy_UT == other.dy_UT and
            self.theta_max == other.theta_max and
            self.sigma_theta == other.sigma_theta and
            self.sigma_phi == other.sigma_phi
        )

    def generate(self) -> ComplexArray:

        # STEP 1: initialization.

        # elevation angles
        theta_BS_LoS     = self._sample_BS_LoS_elevation_angle(self.K, self.theta_max)                                   # shape: (K,)
        theta_UT_LoS     = self._sample_UT_LoS_elevation_angle(self.K, theta_BS_LoS)                                     # shape: (K,)
        theta_UT_NLoS_L1 = self._sample_UT_NLoS_elevation_angles_L1(self.K, self.L1, theta_UT_LoS, self.sigma_theta)     # shape: (K, L)
        theta_UT_NLoS_L2 = self._sample_UT_NLoS_elevation_angles_L2(self.K, self.L2, theta_UT_LoS, self.sigma_theta)     # shape: (K, L)

        # Azimuth angles
        phi_BS_LoS     = self._sample_BS_azimuth_angle(self.K)                                                            # shape: (K,)
        phi_UT_LoS     = self._sample_UT_LoS_azimuth_angle(self.K, phi_BS_LoS)                                            # shape: (K,)
        phi_UT_NLoS_L1 = self._sample_UT_NLoS_azimuth_angles_L1(self.K, self.L1, phi_UT_LoS, self.sigma_phi)              # shape: (K, L)
        phi_UT_NLoS_L2 = self._sample_UT_NLoS_azimuth_angles_L2(self.K, self.L2, phi_UT_LoS, self.sigma_phi)              # shape: (K, L)
        
        # channel gains
        alpha_LoS          = self._sample_LoS_channel_gain(self.num_samples, self.K, theta_UT_LoS, self.fD_times_Tc, self.Tpilot_2_Tc)        # shape: (K,)
        alpha_NLoS_process = self._sample_NLoS_channel_gains(self.num_samples, self.K, self.L1+self.L2, self.fD_times_Tc, self.Tpilot_2_Tc)   # shape: (T, K, L)

        # array response vectors
        a_BS          = np.array([self._array_response_vector(self.Nx_BS, self.Ny_BS, self.dx_BS, self.dy_BS, theta_BS_LoS[k], phi_BS_LoS[k]) for k in range(self.K)])                                              # (K, Nt)
        a_UT_LoS      = np.array([self._array_response_vector(self.Nx_UT, self.Ny_UT, self.dx_UT, self.dy_UT, theta_UT_LoS[k], phi_UT_LoS[k]) for k in range(self.K)])                                              # (K, Nr)
        a_UT_NLoS_L1  = np.array([[self._array_response_vector(self.Nx_UT, self.Ny_UT, self.dx_UT, self.dy_UT, theta_UT_NLoS_L1[k, l], phi_UT_NLoS_L1[k, l]) for l in range(self.L1)] for k in range(self.K)])       # (K, L, Nr)
        a_UT_NLoS_L2  = np.array([[self._array_response_vector(self.Nx_UT, self.Ny_UT, self.dx_UT, self.dy_UT, theta_UT_NLoS_L2[k, l], phi_UT_NLoS_L2[k, l]) for l in range(self.L2)] for k in range(self.K)])       # (K, L, Nr)


        # STEP 2: assembly.
        
        H_process = np.zeros((self.num_samples, self.K*self.Nr, self.Nt), dtype=complex)
        
        for k in range(self.K):

            # LoS component
            H_LoS_k = np.sqrt(self.K_rician / (self.K_rician + 1))  * alpha_LoS[:, k, np.newaxis, np.newaxis] * np.outer(a_UT_LoS[k], a_BS[k].conj())[np.newaxis]

            # NLoS component
            v_NLoS_L1_k = np.einsum('tl,li->ti', alpha_NLoS_process[:, k, : self.L1], a_UT_NLoS_L1[k])
            H_NLoS_L1_k = np.sqrt(1 / (self.K_rician + 1)) * np.sqrt(self.w1 / self.L1)  *  np.einsum('ti,j->tij', v_NLoS_L1_k, a_BS[k].conj())

            v_NLoS_L2_k = np.einsum('tl,li->ti', alpha_NLoS_process[:, k, self.L1 : self.L1+self.L2], a_UT_NLoS_L2[k])
            H_NLoS_L2_k = np.sqrt(1 / (self.K_rician + 1)) * np.sqrt(self.w2 / self.L2)  *  np.einsum('ti,j->tij', v_NLoS_L2_k, a_BS[k].conj())

            H_NLoS_k = H_NLoS_L1_k + H_NLoS_L2_k

            # channel matrix for UT k
            H_k = H_LoS_k[np.newaxis, :, :] + H_NLoS_k
            H_process[:, k*self.Nr : (k+1)*self.Nr, :] = H_k

        
        # STEP 3: termination.
        
        # retrieve the current channel and the corresponding CSI from the generated process.
        H     = H_process[-1]
        H_CSI = H_process[:self.num_samples - self.CSI_delay]

        # Update the stored values.
        self._H     = H
        self._H_CSI = H_CSI

        return H

    
    def autocorrelation_norm(self, tau: RealArray, nu_LoS: float = 0.0) -> ComplexArray:
        r"""
        Compute the theoretical normalized autocorrelation function of a single propagation link.

        For a fixed LoS Doppler shift :math:`\nu^{\text{LoS}}`, the autocorrelation of the scalar channel :math:`h(t) = [\mathbf{H}_k(t)]_{n_r, n_t}` is:

        .. math::

            R_h(\tau)
            = \left[ \frac{K}{K+1} \, e^{j 2\pi \nu^{\text{LoS}} \tau} + \frac{J_0\!\left(2\pi\, f_D T_{\text{pilot}}\, \tau\right)}{K+1} \right]

        The two terms correspond to the LoS and NLoS contributions respectively.
        The LoS term is a complex tone that rotates at the LoS Doppler rate; it does not
        decay with lag.  
        The NLoS term follows Jakes' autocorrelation.

        When averaged over the random LoS Doppler direction :math:`\xi \sim \mathrm{U}[0, 2\pi)`, the expected value of the LoS tone becomes :math:`J_0(2\pi f_D T_{\text{pilot}} \tau)` as well, so the ensemble-averaged autocorrelation reduces to a pure Jakes function:

        .. math::

            \bar{R}_h(\tau) = \frac{J_0(2\pi\, f_D T_{\text{pilot}}\, \tau)}{N_t N_r}

        Parameters
        ----------
        tau : RealArray
            Lag values in number of pilot periods (i.e. samples).
        nu_LoS : float, optional
            Normalised LoS Doppler shift :math:`\nu^{\text{LoS}} = f_D T_{\text{pilot}} \cos(\xi)`.
            Default: 0 (static LoS, reduces to a time-invariant model).

        Returns
        -------
        R : ComplexArray, same shape as ``tau``
            The autocorrelation values at the requested lags.
        """
        
        R_LoS  = (self.K_rician / (self.K_rician + 1)) * np.exp(1j * 2*np.pi * nu_LoS * tau)
        R_NLoS = (1  / (self.K_rician + 1)) * j0(2*np.pi * (self.fD_times_Tc * self.Tpilot_2_Tc) * tau)

        return (R_LoS + R_NLoS)
    
    
    @staticmethod
    def _array_response_vector(Nx: int, Ny: int, dx: float, dy: float, theta: float, phi: float) -> ComplexArray:
        """
        Compute the UPA array response vector.

        Parameters
        ----------
        Nx, Ny : int
            Number of antennas along the x- and y-direction.
        dx, dy : float
            Normalized spacings between the antennas along the x- and y-direction, respectively, relative to the wavelength.
        theta : float
            Elevation angle [rad].
        phi : float
            Azimuth angle [rad].

        Returns
        -------
        a : ComplexArray, shape (Nx * Ny,)
            The UPA array response vector.
        """

        nx = np.arange(Nx)
        ny = np.arange(Ny)

        phase_x =  dx * np.sin(theta) * np.cos(phi)
        phase_y =  dy * np.sin(theta) * np.sin(phi)

        phases = (np.outer(nx, np.ones(Ny))*phase_x + np.outer(np.ones(Nx), ny)*phase_y).flatten()

        a = (1 / np.sqrt(Nx*Ny)) * np.exp(1j * 2*np.pi * phases)
        return a

    
    @staticmethod
    def _sample_BS_LoS_elevation_angle(K: int, theta_max: float) -> RealArray:
        r"""
        Sample the LoS elevation angles :math:`\theta_k` for each UT k according to the following PDF derived from a uniform distribution of UTs on the circular ground disk of radius :math:`r_{\max} = h \tan(\theta_{\max})`:

        .. math::

            P_{\Theta^{\text{BS}}}(\theta)
            = \frac{2\,\tan\theta}{\tan^2(\theta_{\max})\,\cos^2\theta}, \quad \theta \in [0, \theta_{\max}]

        Samples are drawn via the inverse CDF method. The CDF is

        .. math::

            F_\Theta(\theta) = \frac{\tan^2\theta}{\tan^2(\theta_{\max})}

        so that :math:`\theta = \arctan\!\bigl(\tan(\theta_{\max})\sqrt{U}\bigr)` with :math:`U \sim \mathrm{Uniform}[0, 1]`.

        Parameters
        ----------
        K : int
            The number of user terminals.
        theta_max : float
            The maximum elevation angle [rad].

        Returns
        -------
        theta_BS_LoS : RealArray, shape (K,)
            The sampled LoS elevation angles at the BS for each UT.
        """
        
        U = np.random.uniform(0.0, 1.0, size=K)
        theta_BS_LoS = np.arctan(np.tan(theta_max) * np.sqrt(U))
        return theta_BS_LoS

    @staticmethod
    def _sample_UT_LoS_elevation_angle(K: int, theta_BS_LoS: RealArray) -> RealArray:
        """
        Sample the LoS elevation angles at the UT :math:`\theta_k^{\text{UT}}` for each UT k, given the LoS elevation angles at the BS :math:`\theta_k^{\text{BS}}`.

        The elevation angle of the LoS path of UT k at the UT is identical to the elevation angle of the LoS path at the BS, i.e., :math:`\theta_k^{\text{UT}} = \theta_k^{\text{BS}}` for all k.

        Parameters
        ----------
        K : int
            The number of user terminals.
        theta_BS_LoS : RealArray, shape (K,)
            The LoS elevation angles at the BS for each user terminal.
        
        Returns
        -------
        theta_UT_LoS : RealArray, shape (K,)
            The sampled LoS elevation angles at the UT for each user terminal.
        """
        theta_UT_LoS = theta_BS_LoS
        return theta_UT_LoS
    
    @staticmethod
    def _sample_UT_NLoS_elevation_angles_L1(K: int, L1: int, theta_UT_LoS: RealArray, sigma_theta: float) -> RealArray:
        r"""
        Sample the elevation angles of the rays at the UT :math:`\theta_{k,\ell}^{\text{UT}}` for each UT `k` and rays `l` of the first cluster.

        The elevation angle of each ray l of UT k at the UT is modeled as a small perturbation around the LoS elevation angle :math:`\theta_k^{\text{UT}}`:

        .. math::

            \theta^{\text{UT}}_{k,\ell} = \theta^{\text{UT}}_k + \sigma_{\theta} \, u_{k,\ell} \quad \text{with} \quad u_{k,\ell} \sim \text{Laplace}(0, 1) \text{ and } \sigma_{\theta} \approx 4^\circ

        Parameters
        ----------
        K : int
            The number of user terminals.
        L1 : int
            The number of rays (bundle of NLoS propagation paths) per UT in the first cluster.
        theta_UT_LoS : RealArray, shape (K,)
            The elevation angles of the LoS path at the UT, for each UT.
        sigma_theta : float
            The scale parameter of the Laplace distribution modeling the deviation of the NLoS elevation angles from the LoS elevation angles.

        Returns
        -------
        theta_UT_NLoS : RealArray, shape (K, L1)
            The sampled NLoS elevation angles for all rays `l` of the first cluster at each UT `k`.
        """
        
        u = np.random.laplace(loc=0.0, scale=1.0, size=(K, L1))
        theta_UT_NLoS_L1 = theta_UT_LoS[:, np.newaxis] + (sigma_theta * u)
        return theta_UT_NLoS_L1

    @staticmethod
    def _sample_UT_NLoS_elevation_angles_L2(K: int, L2: int, theta_UT_LoS: RealArray, sigma_theta: float) -> RealArray:
        r"""
        Sample the elevation angles of the rays at the UT :math:`\theta_{k,\ell}^{\text{UT}}` for each UT `k` and rays `l` of the second cluster.

        The elevation angle of each ray l of UT k at the UT is modeled as a small perturbation around the LoS elevation angle :math:`\theta_k^{\text{UT}}`:

        .. math::

            \theta^{\text{UT}}_{k,\ell} = \theta^{\text{UT}}_k + \sigma_{\theta} \, u_{k,\ell} + \delta\theta_k \quad \text{with} \quad u_{k,\ell} \sim \text{Laplace}(0, 1) \text{ and } \sigma_{\theta} \approx 4^\circ \text{ and } \delta\theta_k \sim \mathrm{U}[2^\circ, 10^\circ]

        Parameters
        ----------
        K : int
            The number of user terminals.
        L2 : int
            The number of rays (bundle of NLoS propagation paths) per UT in the second cluster.
        theta_UT_LoS : RealArray, shape (K,)
            The elevation angles of the LoS path at the UT, for each UT.
        sigma_theta : float
            The scale parameter of the Laplace distribution modeling the deviation of the NLoS elevation angles from the LoS elevation angles.

        Returns
        -------
        theta_UT_NLoS : RealArray, shape (K, L2)
            The sampled NLoS elevation angles for all rays `l` of the second cluster at each UT `k`.
        """
        
        u = np.random.laplace(loc=0.0, scale=1.0, size=(K, L2))
        delta_theta = np.random.uniform(2*(np.pi/180), 10*(np.pi/180), size=K)
        theta_UT_NLoS_L2 = theta_UT_LoS[:, np.newaxis] + (sigma_theta * u) + delta_theta[:, np.newaxis]
        return theta_UT_NLoS_L2
    

    @staticmethod
    def _sample_BS_azimuth_angle(K: int) -> RealArray:
        r"""
        Sample the azimuth angles at the BS :math:`\phi_k^{\text{BS}}` for each UT k.

        The azimuth angle of the LoS path of UT :math:`k` at the BS is uniformly distributed within the UT's dedicated angular sector of width :math:`2\pi/K`:

        .. math::

            P_{\Phi^{\text{BS}}_k}(\phi) = \frac{K}{2\pi},
            \quad \phi \in \left[k\,\frac{2\pi}{K},\; (k+1)\,\frac{2\pi}{K}\right)

        Sampling directly:

        .. math::

            \phi_k^{\text{BS}} = \frac{2\pi}{K}\,(k + U_k), \quad U_k \sim \mathrm{Uniform}[0, 1)

        Parameters
        ----------
        K : int
            The number of user terminals.

        Returns
        -------
        phi_BS_LoS : RealArray, shape (K,)
            The sampled LoS azimuth angles at the BS for each UT [rad].
        """
        k = np.arange(K)
        U = np.random.uniform(0.0, 1.0, size=K)
        phi_BS_LoS = (2 * np.pi / K) * (k + U)
        return phi_BS_LoS

    @staticmethod
    def _sample_UT_LoS_azimuth_angle(K: int, phi_BS_LoS: RealArray) -> RealArray:
        r"""
        Sample the LoS azimuth angles at the UT :math:`\phi_k^{\text{UT}}` for each UT k.

        The orientation of each UT's array relative to the satellite is determined by a rotation :math:`\rho_k` around the vertical axis, which is uniformly distributed in :math:`[0, 2\pi)`.
        This makes the UT azimuth angle of the LoS path uniformly distributed over the full circle, independently of the BS azimuth angle:

        .. math::

            \phi_k^{\text{UT}} \sim \mathrm{Uniform}[0, 2\pi), \quad \text{i.i.d. across } k

        Parameters
        ----------
        K : int
            The number of user terminals.
        phi_BS_LoS : RealArray, shape (K,)
            The LoS azimuth angles at the BS for each UT.

        Returns
        -------
        phi_UT_LoS : RealArray, shape (K,)
            The sampled LoS azimuth angles [rad] at the UT, for each UT.
        """
        phi_UT_LoS = np.random.uniform(0.0, 2*np.pi, size=K)
        return phi_UT_LoS
    
    @staticmethod
    def _sample_UT_NLoS_azimuth_angles_L1(K: int, L1: int, phi_UT_LoS: RealArray, sigma_phi: float) -> RealArray:
        r"""
        Sample the NLoS azimuth angles at the UT :math:`\phi_{k,\ell}^{\text{UT}}` for each UT `k` and ray `l` of the first cluster.

        The NLoS azimuth angles are modeled as Laplace-distributed deviations from the LoS azimuth angles for the corresponding UT `k` and ray `l`:

        .. math::

            \phi_{k,\ell}^{\text{UT}} = \phi_k^{\text{UT}} + \sigma_{\phi} \, u_{k,\ell}, \quad u_{k,\ell} \sim \text{Laplace}(0, 1), \quad \sigma_{\phi} \approx 4^\circ

        Parameters
        ----------
        K : int
            The number of user terminals.
        L1 : int
            The number of rays (bundle of NLoS propagation paths) per UT in the first cluster.
        phi_UT_LoS : RealArray, shape (K,)
            The LoS azimuth angles at the UT for each UT.
        sigma_phi : float
            The scale parameter of the Laplace distribution modeling the deviation of the NLoS azimuth angles from the LoS azimuth angles.

        Returns
        -------
        phi_UT_NLoS : RealArray, shape (K, L1)
            The sampled NLoS azimuth angles for all rays `l` at each UT `k` of the first cluster [rad].
        """
        
        u = np.random.laplace(loc=0.0, scale=1.0, size=(K, L1))
        phi_UT_NLoS = phi_UT_LoS[:, np.newaxis] + (sigma_phi * u)
        return phi_UT_NLoS
    
    @staticmethod
    def _sample_UT_NLoS_azimuth_angles_L2(K: int, L2: int, phi_UT_LoS: RealArray, sigma_phi: float) -> RealArray:
        r"""
        Sample the NLoS azimuth angles at the UT :math:`\phi_{k,\ell}^{\text{UT}}` for each UT `k` and ray `l` of the second cluster.

        The NLoS azimuth angles are modeled as Laplace-distributed deviations from the LoS azimuth angles for the corresponding UT `k` and ray `l`:

        .. math::

            \phi_{k,\ell}^{\text{UT}} = \phi_k^{\text{UT}} + \sigma_{\phi} \, u_{k,\ell} + \delta\phi_k, \quad u_{k,\ell} \sim \text{Laplace}(0, 1), \quad \sigma_{\phi} \approx 4^\circ \text{ and } \delta\phi_k \sim \mathrm{U}[10^\circ, 30^\circ]

        Parameters
        ----------
        K : int
            The number of user terminals.
        L2 : int
            The number of rays (bundle of NLoS propagation paths) per UT in the second cluster.
        phi_UT_LoS : RealArray, shape (K,)
            The LoS azimuth angles at the UT for each UT.
        sigma_phi : float
            The scale parameter of the Laplace distribution modeling the deviation of the NLoS azimuth angles from the LoS azimuth angles.

        Returns
        -------
        phi_UT_NLoS : RealArray, shape (K, L2)
            The sampled NLoS azimuth angles for all rays `l` at each UT `k` of the second cluster [rad].
        """
        
        u = np.random.laplace(loc=0.0, scale=1.0, size=(K, L2))
        delta_phi = np.random.uniform(low=10*(np.pi/180), high=30*(np.pi/180), size=K)
        phi_UT_NLoS = phi_UT_LoS[:, np.newaxis] + (sigma_phi * u) + delta_phi[:, np.newaxis]
        return phi_UT_NLoS

    
    @staticmethod
    def _sample_LoS_channel_gain(T: int, K: int, theta_k: RealArray, fD_times_Tc: float, Tpilot_2_Tc: float) -> ComplexArray:
        r"""
        Sample the time-varying LoS channel gains :math:`\alpha^{\text{LoS}}_k(t)` for each UT k.

        The LoS path of a moving UT experiences a deterministic Doppler shift determined by the
        projection of the UT's velocity onto the satellite direction.  The gain is modeled as a
        complex tone with a random initial phase and a random (but fixed per realisation) normalised
        Doppler frequency:

        .. math::

            \alpha^{\text{LoS}}_k(t) = e^{j\left(\psi_k + 2\pi \nu_k^{\text{LoS}} t\right)}

        where

        .. math::

            \psi_k \sim \mathrm{Uniform}[0, 2\pi), \qquad
            \nu_k^{\text{LoS}} = f_D T_{\text{pilot}} \cos(\pi/2 - \theta_k)

        Here :math:`\theta_k` is the elevation angle of the LoS path at the UT, so that :math:`\nu_k^{\text{LoS}} \in [-f_D T_{\text{pilot}},\, f_D T_{\text{pilot}}]`.
        Both :math:`\psi_k` and :math:`\theta_k` are drawn independently across users and held constant throughout the generated time window.

        Parameters
        ----------
        T : int
            Number of time samples to generate.
        K : int
            Number of user terminals.
        theta_k : RealArray, shape (K,)
            The elevation angles of the LoS path at the UT, for each UT.
        fD_times_Tc : float
            Maximum Doppler frequency times the coherence time :math:`f_D T_c`.
        Tpilot_2_Tc : float
            Time period between two CSI acquisitions relative to the coherence time
            :math:`T_{\text{pilot}} / T_c`.

        Returns
        -------
        alpha_LoS : ComplexArray, shape (T, K)
            Time-varying LoS channel gains for every time instant and user.
        """

        psi    = np.random.uniform(0.0, 2*np.pi, size=K)
        xi     = np.pi/2 - theta_k
        nu_LoS = fD_times_Tc * Tpilot_2_Tc * np.cos(xi)
        t      = np.arange(T)

        alpha_LoS = np.exp(1j * (psi[np.newaxis, :] + 2*np.pi * np.outer(t, nu_LoS)))
        return alpha_LoS
        
    @staticmethod
    def _sample_NLoS_channel_gains(T: int, K: int, L: int, fD_times_Tc: float, Tpilot_2_Tc: float) -> ComplexArray:
        r"""
        Generate time-correlated NLoS channel gains :math:`\alpha^{\text{NLoS}}_{k,\ell}(t)` for all users, rays (bundle of NLoS propagation paths), and time instants using the Cholesky decomposition method.

        Each gain is an i.i.d. (across users and rays) zero-mean unit-variance complex Gaussian process whose power spectral density follows Jakes' Doppler spectrum (Clarke's fading model):

        .. math::

            \alpha^{\text{NLoS}}_{k,\ell}(t) \sim \mathcal{CN}(0, 1), \quad S(f) = \frac{1}{\pi f_D \sqrt{1 - (f/f_D)^2}}, \quad |f| < f_D

        For Clarke's model the autocorrelation function is :math:`R(\tau) = J_0(2\pi f_D \tau)`, so the :math:`T \times T` covariance matrix has entries :math:`C_{ij} = J_0\!\bigl(2\pi\,(f_D T_c)\,(i-j)\,T_\text{pilot}/T_c\bigr)`.

        The Cholesky steps are:

        1. Build the Toeplitz covariance matrix :math:`\mathbf{C}`.
        2. Compute its Cholesky factor :math:`\mathbf{C} = \mathbf{L}\mathbf{L}^H`.
        3. Draw :math:`K \times L` independent white :math:`\mathcal{CN}(0,1)` sequences :math:`\mathbf{w}` of length :math:`T`.
        4. Obtain the correlated gains as :math:`\boldsymbol{\alpha} = \mathbf{L}\,\mathbf{w}`.

        Parameters
        ----------
        T : int
            Number of time samples to generate.
        K : int
            Number of user terminals.
        L : int
            Number of NLoS rays per user terminal.
        fD_times_Tc : float
            Maximum Doppler frequency times the coherence time :math:`f_D T_c`.
        Tpilot_2_Tc : float
            Time period between two CSI acquisitions relative to the coherence time :math:`T_\text{pilot}/T_c`.

        Returns
        -------
        alpha_NLoS : ComplexArray, shape (T, K, L)
            Time-correlated NLoS channel gains for every time instant, user, and ray.
        """
        
        # STEP 1: build the covariance matrix.
        lags = np.arange(T)
        C = toeplitz(j0(2 * np.pi * fD_times_Tc * Tpilot_2_Tc * lags))
        C += 1e-10 * np.eye(T)  # regularise for numerical stability

        # STEP 2: cholesky decomposition
        L_cholesky = np.linalg.cholesky(C)

        # STEP 3: white gaussian samples
        w = (1 / np.sqrt(2)) * (np.random.randn(T, K*L) + 1j * np.random.randn(T, K*L))

        # STEP 4: impose the temporal correlation and reshape
        alpha_NLoS = (L_cholesky @ w).reshape(T, K, L)

        return alpha_NLoS
    


# NOISE MODELS

class NoiseModel(ABC):
    """
    Noise Abstract Base Class (ABC).
    
    A noise class is responsible for generating the noise vectors according to a specific noise type and applying the noise effects to the received signals.
    """

    def __init__(self, Nr: int, K: int):
        """
        Instantiate the noise.

        Parameters
        ----------
        Nr : int
            The number of receive antennas per user terminal.
        K : int
            The number of user terminals.
        """

        self.Nr = Nr
        self.K = K

    def __str__(self) -> str:
        """ Return a string representation of the noise model. """
        return "Noise Model (abstract base class)"
    
    def __eq__(self, other: object) -> bool:
        
        if not isinstance(other, NoiseModel):
            return NotImplemented
        
        return (
            type(self) == type(other) and
            self.Nr == other.Nr and
            self.K == other.K
        )

    @abstractmethod
    def get_noise(self, snr: float, x: ComplexArray) -> ComplexArray:
        """
        Generate the noise vectors.

        Parameters
        ----------
        snr : float
            The signal-to-noise (SNR) ratio.
        x : ComplexArray, shape (Nt, Msv)
            The transmitted signals.

        Returns
        -------
        n : ComplexArray, shape (K*Nr, Msv)
            The generated noise vectors.
        """
        raise NotImplementedError
    
    @staticmethod
    def apply(y_noiseless: ComplexArray, n: ComplexArray) -> ComplexArray:
        """
        Apply the noise effects to the received signals.

        Parameters
        ----------
        y_noiseless : ComplexArray, shape (K*Nr, Msv)
            The received signals without noise.
        n : ComplexArray, shape (K*Nr, Msv)
            The noise vectors.
        
        Returns
        -------
        y : ComplexArray, shape (K*Nr, Msv)
            The received signals with noise.
        """
        y = y_noiseless + n
        return y

class NeutralNoise(NoiseModel):
    """
    Neutral Noise.

    This noise acts as a 'neutral element' for the noise model.\\
    It does not add any noise to the received signals but simply lets the noiseless received signals pass through.
    """

    def __str__(self) -> str:
        return "Zero Noise"

    def get_noise(self, snr: float, x: ComplexArray) -> ComplexArray:
        n = np.zeros((self.K * self.Nr, x.shape[1]), dtype=complex)
        return n

class CSAWGNNoise(NoiseModel):
    """
    Circularly-Symmetric Additive White Gaussian Noise (CSAWGN).

    This noise is complex proper, circularly-symmetric additive white Gaussian (AWGN) distributed, with a power based on the specified signal-to-noise ratio (SNR).
    """

    def __str__(self) -> str:
        return "CS Additive White Gaussian Noise"
    
    def get_noise(self, snr: float, x: ComplexArray) -> ComplexArray:
        
        # Compute the noise power based on the current SNR and the signal power of x.
        p_signal = np.mean( np.sum( np.abs(x)**2, axis=0 ) )
        p_noise = p_signal / snr
        sigma = np.sqrt(p_noise / 2)

        # Generate complex proper, circularly-symmetric AWGN noise vectors with the computed noise power.
        n = sigma * (np.random.randn(self.K*self.Nr, x.shape[1]) + 1j * np.random.randn(self.K*self.Nr, x.shape[1]))
        return n

