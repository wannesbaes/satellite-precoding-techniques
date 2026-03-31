# mu-mimo/mu_mimo/processing/channel.py

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from ..types import ComplexArray

# CHANNEL MODELS

class ChannelModel(ABC):
    """
    The Channel Model Abstract Base Class (ABC).

    A channel model is responsible for generating the channel matrix according to a specific channel model and applying the channel effects to the transmitted signals.
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
    
    @abstractmethod
    def generate(self) -> ComplexArray:
        """
        Generate the channel matrix.

        Returns
        -------
        H : ComplexArray, shape (K*Nr, Nt)
            The generated channel matrix.
        """
        raise NotImplementedError

    @staticmethod
    def apply(x: ComplexArray, H: ComplexArray) -> ComplexArray:
        """
        Apply the channel effects to the transmitted signals.

        Parameters
        ----------
        x : ComplexArray, shape (Nt, M)
            The transmitted signals.
        H : ComplexArray, shape (K*Nr, Nt)
            The channel matrix.

        Returns
        -------
        y : ComplexArray, shape (K*Nr, M)
            The received signals.
        """
        y = H @ x
        return y

class NeutralChannelModel(ChannelModel):
    """
    Neutral Channel Model.
    
    This channel model acts as a 'neutral element' for the channel.\\
    In particular, it generates an identity channel matrix, which means that the symbols are transmitted to the receive antennas for which they are intended, and without any interference.
    """

    def generate(self) -> ComplexArray:
        H = np.eye(self.K*self.Nr, self.Nt, dtype=complex)
        return H

class IIDRayleighFadingChannelModel(ChannelModel):
    """
    Independent and Identically Distributed (IID) Rayleigh Fading Channel Model.

    This channel model generates a channel matrix with independent and identically distributed (IID) circularly-symmetric zero-mean unit-variance complex Gaussian entries.\\
    The Rayleigh fading aspect is captured by the fact that the channel coefficients change independently after M transmissions.
    """

    def generate(self) -> ComplexArray:
        H = (1 / np.sqrt(2)) * (np.random.randn(self.K*self.Nr, self.Nt) + 1j * np.random.randn(self.K*self.Nr, self.Nt))
        return H

class RiceanChannelModel(ChannelModel):
    r"""
    The ricean channel model.
    The LoS component is modeled as a deterministic component independent of time and across users. The NLoS component is modeled according to Jake's model.

    ..math:
    \begin{equation}
        H_k(t) = e^{j\theta_k} \left( \sqrt{\frac{K}{K+1}} + \sqrt{\frac{1}{K+1}} H_{\text{NLoS},k}(t) \right),
    \end{equation}
    
    \begin{itemize}
        \item $K \in [0, +\infty)$: The Rice factor. It quantifies the strength of the deterministic LoS component relative to the scattered multipath.
        \item $\theta_k$: : The arbitrary channel phase., uniformly distributed over $[-\pi, \pi]$ and statistically independent for different users $k$.
        \item $H_{\text{NLoS},k}(t)$: The No Line-of-Sight (NLoS) components.\\
        The entries of $H_{\text{NLoS},k}(t)$ are i.i.d. zero-mean unit-variance complex Gaussian random processes. Each process is correlated in time. The power spectral density (PSD) is given by Jake's model: $S(f) = \frac{1}{\pi f_D \sqrt{1 - \left(\tfrac{f}{f_D}\right)^2}}, \quad |f| < f_D.$
    \end{itemize}
    """


# NOISE MODELS

class NoiseModel(ABC):
    """
    Noise Model Abstract Base Class (ABC).
    
    A noise model is responsible for generating the noise vectors according to a specific noise model and applying the noise effects to the received signals.
    """

    def __init__(self, Nr: int, K: int):
        """
        Instantiate the noise model.

        Parameters
        ----------
        Nr : int
            The number of receive antennas per user terminal.
        K : int
            The number of user terminals.
        """

        self.Nr = Nr
        self.K = K
    
    @abstractmethod
    def generate(self, snr: float, x: ComplexArray) -> ComplexArray:
        """
        Generate the noise vectors.

        Parameters
        ----------
        snr : float
            The signal-to-noise ratio.
        x : ComplexArray, shape (Nt, M)
            The transmitted signals.

        Returns
        -------
        n : ComplexArray, shape (K*Nr, M)
            The generated noise vectors.
        """
        raise NotImplementedError

    @staticmethod
    def apply(y_noiseless: ComplexArray, n: ComplexArray) -> ComplexArray:
        """
        Apply the noise effects to the received signals.

        Parameters
        ----------
        y_noiseless : ComplexArray, shape (K*Nr, M)
            The received signals without noise.
        n : ComplexArray, shape (K*Nr, M)
            The noise vectors.
        
        Returns
        -------
        y : ComplexArray, shape (K*Nr, M)
            The received signals with noise.
        """
        y = y_noiseless + n
        return y

class NeutralNoiseModel(NoiseModel):
    """
    Neutral Noise Model.

    This noise model acts as a 'neutral element' for noise.\\
    It does not add any noise to the received signals but simply lets the noiseless received signals pass through.
    """

    def generate(self, snr: float, x: ComplexArray) -> ComplexArray:
        n = np.zeros((self.K * self.Nr, x.shape[1]), dtype=complex)
        return n

class CSAWGNNoiseModel(NoiseModel):
    """
    Circularly-Symmetric Additive White Gaussian Noise (CSAWGN) Model.

    This noise model generates complex proper, circularly-symmetric additive white Gaussian noise (AWGN) vectors based on the specified signal-to-noise ratio (SNR).
    """

    def generate(self, snr: float, x: ComplexArray) -> ComplexArray:
        
        # Compute the noise power based on the current SNR and the signal power of x.
        p_signal = np.mean( np.sum( np.abs(x)**2, axis=0 ) )
        p_noise = p_signal / snr
        sigma = np.sqrt(p_noise / 2)

        # Generate complex proper, circularly-symmetric AWGN noise vectors with the computed noise power.
        n = sigma * (np.random.randn(self.K*self.Nr, x.shape[1]) + 1j * np.random.randn(self.K*self.Nr, x.shape[1]))
        return n
