# mu-mimo/mu_mimo/processing/channel.py

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np
from scipy.special import j0
from ..types import ComplexArray, RealArray, IntArray


# HELPERS

@dataclass
class CholeskyMethodState:
    """
    State of the Cholesky method for generating the NLoS component of the channel.
    The state contains the already generated NLoS component for later use in the next time instants.

    Attributes
    ----------
    H_NLoS : ComplexArray | None, shape (K * Nr, Nt, M)
        The generated NLoS component for all propagation links and all time instants.
    m : IntArray, shape (K,)
        The current time instant for each user k.
    """

    H_NLoS : ComplexArray | None
    m : IntArray


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
    The Ricean channel model.

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

    def __init__(self, Nt: int, Nr: int, K: int, K_rice: float, fD: float, NLoS_method : str = "Cholesky-decomposition method"):
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
        
        K_rice : float
            The Rice factor. It quantifies the strength of the deterministic LoS component relative to the scattered multipath.
        fD : float
            The maximum Doppler frequency (in Hertz).
        
        NLoS_method : str
            The method to use for generating the NLoS component. Choose between: 'Cholesky-decomposition method', 'spectral method', 'FIR filter method'.
        """
        
        super().__init__(Nt, Nr, K)
        
        self.K_rice = K_rice
        self.fD = fD
        self.NLoS_method = NLoS_method

        self.num_channel_realizations = 2000         # The number of channel realizations. Needed for methods that require pre-generating the NLoS component for all time instants. Hardcoded for now.
        self.Ts = 20e-3                              # The time interval between two consecutive channel realizations. Needed for methods that require pre-generating the NLoS component for all time instants. Hardcoded for now.       

        # Generate the channel phase for each user k, which is uniformly distributed over [-pi, pi) and independent of time and across users k.
        self._theta = np.random.uniform(-np.pi, np.pi, size=self.K)

        # Cholesky State (only used when the Cholesky-decomposition method is chosen).
        self._cholesky_state = CholeskyMethodState(H_NLoS=None, m=np.zeros(self.K, dtype=int))

    def generate(self) -> ComplexArray:
        
        H = np.empty((self.K*self.Nr, self.Nt), dtype=complex)

        for k in range(self.K):

            # Retrieve the channel phase for user k. 
            theta_k = self._theta[k]

            # Retrieve the current NLoS component for user k, using the specified method.
            H_NLoS_k = self._retrieve_NLoS(k, method = self.NLoS_method)
            
            # Generate the current channel matrix for user k.
            H_k = np.exp(1j * theta_k) * (np.sqrt(self.K_rice / (self.K_rice + 1)) + np.sqrt(1 / (self.K_rice + 1)) * H_NLoS_k)

            # Store the current channel matrix for user k in the compound channel matrix H.
            H[k*self.Nr : (k+1)*self.Nr] = H_k
        
        return H

    def _retrieve_NLoS(self, k: int, method: str) -> ComplexArray:
        """
        Retrieve the NLoS component for user k using the specified method.

        Parameters
        ----------
        k : int
            The user terminal ID for which to generate the NLoS component.
        method : str
            The method to use for generating the NLoS component. Choose between: 'Cholesky-decomposition method', 'spectral method', 'FIR filter method'.

        Returns
        -------
        H_NLoS_k : ComplexArray, shape (Nr, Nt)
            The generated NLoS component for user k.
        """
        
        if method == "Cholesky-decomposition method":
            
            # If this is the first time instant, we need to generate the NLoS component for all time instants and all users, and store it in the state for later use.
            if np.all(self._cholesky_state.m == 0):
                self._cholesky_state.H_NLoS = self._generate_NLoS_cholesky()

            # Retrieve the current NLoS component for user k.
            H_NLoS_k = self._cholesky_state.H_NLoS[k*self.Nr : (k+1)*self.Nr, :, self._cholesky_state.m[k]]
            self._cholesky_state.m[k] += 1
        
        elif method == "spectral method":
            pass
        
        elif method == "FIR filter method":
            pass
        
        else:
            raise ValueError(f"Unknown method '{method}' for generating the Ricean channel matrix. Valid options are: 'Cholesky-decomposition method', 'spectral method', 'FIR filter method'.")
        
        return H_NLoS_k

    def _generate_NLoS_cholesky(self) -> ComplexArray:
        """
        Generate the NLoS component for user k using the Cholesky decomposition method for generating a Gaussian process with a specified auto-correlation function.

        The Cholesky decomposition method for generating a zero-mean unit-variance complex Gaussian process :math:`\mathbf{h}` of length :math:`N` with a specified autocorrelation function :math:`R_h(\tau)`:
        - **Step 1:** Build the :math:`N \times N` covariance matrix :math:`\mathbf{C}` with entries :math:`C_{i,j} = R_h((i-j)T_s)`.
        - **Step 2:** Compute the Cholesky decomposition of the covariance matrix :math:`\mathbf{C} = \mathbf{L}\mathbf{L}^H`, where :math:`\mathbf{L}` is a lower triangular matrix.
        - **Step 3:** Generate a column vector :math:`\mathbf{w}` of :math:`N` i.i.d. white complex Gaussian random variables with zero mean and unit variance.
        - **Step 4:** Find the desired Gaussian process as :math:`\mathbf{h} = \mathbf{L}\mathbf{w}`.

        Returns
        -------
        H_NLoS : ComplexArray, shape (K * Nr, Nt, M)
            The generated NLoS component for all propagation links and all time instants.
        """

        # STEP 1.
        R_h = lambda tau: j0(2*np.pi * self.fD * tau)
        
        i = np.arange(self.num_channel_realizations).reshape(-1,1)
        j = np.arange(self.num_channel_realizations)
        C = R_h((i - j)*self.Ts)

        # STEP 2.
        L = np.linalg.cholesky(C)

        # STEP 3.
        w = (1 / np.sqrt(2)) * (np.random.randn(self.K*self.Nr, self.Nt, self.num_channel_realizations) + 1j * np.random.randn(self.K*self.Nr, self.Nt, self.num_channel_realizations))

        # STEP 4.
        H_NLoS = w @ L.T

        return H_NLoS

    def _generate_NLoS_spectral(self, k: int) -> ComplexArray:
        """
        Generate the NLoS component for user k using the spectral method for generating a Gaussian process with a specified power spectral density (PSD) function.
        """
        pass

    def _generate_NLoS_FIR_filter(self, k: int) -> ComplexArray:
        """
        Generate the NLoS component for user k using the FIR filter method for generating a Gaussian process with a specified power spectral density (PSD) function.
        """
        pass

    


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
