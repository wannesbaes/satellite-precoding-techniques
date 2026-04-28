# mu-mimo/mu_mimo/processing/channel.py

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

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

    def __init__(self, Nt: int, Nr: int, K: int, K_rice: float, Trtt_2_Tc: float, Tpilot_2_Tc: float, Twindow_2_Tc: float = 2):
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
        
        K_rice : float
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
        assert K_rice >= 0, f"The Rice factor K_rice must be non-negative.\nCurrent value: {K_rice}"
        assert 0 <= Trtt_2_Tc, f"The round trip time relative to the coherence time must be non-negative.\nCurrent value: {Trtt_2_Tc}"
        assert 0 < Tpilot_2_Tc <= Trtt_2_Tc or Trtt_2_Tc == 0, f"The time period between two CSI feedback messages relative to the coherence time must be positive and less than or equal to the round trip time.\nCurrent value: {Tpilot_2_Tc}"
        assert Twindow_2_Tc > 0, f"The time window relative to the coherence time must be positive.\nCurrent value: {Twindow_2_Tc}"
        
        # Store the channel parameters.
        self.K_rice = K_rice
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
        return f"Ricean Channel with an IID time-correlated fading NLoS component.\n   K_rice = {self.K_rice}, RTT = {np.round(self.Trtt_2_Tc, 2)} Tc, CSI feedback rate = {np.round(1 / self.Tpilot_2_Tc, 2)} per Tc"
    
    def __eq__(self, other):
        
        if not isinstance(other, RiceanIIDTCChannel):
            return NotImplemented
        
        return (
            self.Nt == other.Nt and
            self.Nr == other.Nr and
            self.K == other.K and
            self.K_rice == other.K_rice and
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
        H = np.exp(1j * theta) * (np.sqrt(self.K_rice / (self.K_rice + 1)) + np.sqrt(1 / (self.K_rice + 1)) * H_NLoS)

        # Compute the CSI corresponding to the current channel.
        H_NLoS_CSI = H_NLoS_process[:self.num_samples-self.CSI_delay]
        H_CSI = np.exp(1j * theta) * (np.sqrt(self.K_rice / (self.K_rice + 1)) + np.sqrt(1 / (self.K_rice + 1)) * H_NLoS_CSI)
        
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

