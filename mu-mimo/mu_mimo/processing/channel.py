# mu-mimo/mu_mimo/processing/channel.py

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j0
from ..types import ComplexArray, RealArray, IntArray



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

    def __str__(self) -> str:
        """ Return a string representation of the channel model. """
        return "Neutral Channel"

    def generate(self) -> ComplexArray:
        H = np.eye(self.K*self.Nr, self.Nt, dtype=complex)
        return H

class IIDRayleighFadingChannelModel(ChannelModel):
    """
    Independent and Identically Distributed (IID) Rayleigh Fading Channel Model.

    This channel model generates a channel matrix with independent and identically distributed (IID) circularly-symmetric zero-mean unit-variance complex Gaussian entries.\\
    The Rayleigh fading aspect is captured by the fact that the channel coefficients change independently after M transmissions.
    """

    def __str__(self) -> str:
        """ Return a string representation of the channel model. """
        return "IID Rayleigh Fading Channel"
    
    def generate(self) -> ComplexArray:
        H = (1 / np.sqrt(2)) * (np.random.randn(self.K*self.Nr, self.Nt) + 1j * np.random.randn(self.K*self.Nr, self.Nt))
        return H

class RiceanFadingChannelModel(ChannelModel):
    r"""
    The Ricean fading channel model.

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

        # Store the NLoS component for later use if the generating method requires pre-generating the NLoS component for all time instants.
        self.H_NLoS = None
        self.M = np.zeros(self.K, dtype=int)

    def __str__(self) -> str:
        """ Return a string representation of the channel model. """
        return f"Ricean Fading Channel (K_rice={self.K_rice}, fD={self.fD}, NLoS_method='{self.NLoS_method}')"
    
    def reset(self) -> None:
        """
        Reset the time instant to zero and delete all pre-generated NLoS components.
        """
        self.H_NLoS = None
        self.M = np.zeros(self.K, dtype=int)
        return

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
        Retrieve the current NLoS component for user k.

        If this is the first time instant, generate the NLoS component for all time instants and all users using the specified method and store it in the state for later use. Then, retrieve the current NLoS component for user k.
        
        Parameters
        ----------
        k : int
            The user terminal ID for which to retrieve the current NLoS component.
        method : str
            The method to use for generating the NLoS component. Choose between: 'Cholesky-decomposition method', 'spectral method', 'FIR filter method'.
        
        Returns
        -------
        H_NLoS_k : ComplexArray, shape (Nr, Nt)
            The current NLoS component for user k.
        """

        # If this is the first time instant, generate the NLoS component for all time instants and all users and store it in the state for later use.
        if np.all(self.M == 0):
            self.H_NLoS = self._generate_NLoS(method=method)

        # Retrieve the current NLoS component for user k.
        H_NLoS_k = self.H_NLoS[k*self.Nr : (k+1)*self.Nr, :, self.M[k]]
        self.M[k] += 1

        return H_NLoS_k
    
    def _generate_NLoS(self, method: str) -> ComplexArray:
        """
        Generate the NLoS component for all users and all time instants, using the specified method.

        Parameters
        ----------
        method : str
            The method to use for generating the NLoS component. Choose between: 'Cholesky-decomposition method', 'spectral method', 'FIR filter method'.

        Returns
        -------
        H_NLoS : ComplexArray, shape (K * Nr, Nt, num_channel_realizations)
            The generated NLoS component of the channel, for all time instants.
        """
        
        if method == "Cholesky-decomposition method":
            H_NLoS = self._generate_NLoS_cholesky()
        
        elif method == "spectral method":
            H_NLoS = self._generate_NLoS_spectral()
        
        elif method == "FIR filter method":
            H_NLoS = self._generate_NLoS_FIR_filter()
        
        else:
            raise ValueError(f"Unknown method '{method}' for generating the Ricean channel matrix. Valid options are: 'Cholesky-decomposition method', 'spectral method', 'FIR filter method'.")
        
        return H_NLoS

    def _generate_NLoS_cholesky(self) -> ComplexArray:
        r"""
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
        R_h = lambda tau: j0(2*np.pi * self.fD * (tau))
        
        i = np.arange(self.num_channel_realizations).reshape(-1,1)
        j = np.arange(self.num_channel_realizations)
        C = R_h((i - j)*self.Ts)
        C += (10e-10)*np.eye(self.num_channel_realizations)

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

    def plot_autocorrelation(self, max_lag: int = 200, num_samples: int = 1) -> None:
        r"""
        Plot the autocorrelation function of the generated NLoS process and compare it to the analytical expression.

        The simulated autocorrelation is computed as the empirical autocorrelation, averaged over num_samples independent realizations of the NLoS process:
        .. math::
            \hat{R}_h(k) = \frac{1}{N \cdot M} \sum_{m=1}^{M} \sum_{n=0}^{N-k-1} h_m[n+k] h_m^*[n], \quad k = 0, 1, \ldots, N-1
        
        The analytical autocorrelation equals the zero-order Bessel function of the first kind:
        .. math::
            R_h(\tau) = J_0(2\pi f_D \tau)

        Parameters
        ----------
        max_lag : int
            The maximum lag (in samples) to compute the autocorrelation for.
        num_samples : int
            The number of independent realizations of the NLoS process to average over. Default is 1.
            Should be smaller than or equal to K * Nr * Nt, because the average cannot be taken accross different simulations.
        
        Results
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the plot.
        ax : matplotlib.axes.Axes
            The axes object containing the plot.

        Notes
        -----
        This function is intended to be used for validating the generated NLoS process! It will reset the channel model to ensure we start at time instant zero and that all pre-generated NLoS components are deleted. Then, it will generate num_samples sequences of the NLoS process using the specified method, and compute the averaged empirical autocorrelation for both positive and negative lags. Finally, it will compute the analytical autocorrelation and plot both the empirical and analytical autocorrelation functions.
        """
 
        # 0. Reset the channel model to ensure we start at time instant zero and that all pre-generated NLoS components are deleted.
        assert num_samples <= self.K * self.Nr * self.Nt, f"num_samples should be smaller than or equal to K * Nr * Nt = {self.K * self.Nr * self.Nt}, because the average cannot be taken accross different simulations. However, num_samples={num_samples} was given."
        self.reset()
        
        # 1. Compute empirical autocorrelation.

        # Generate enough realizations of the NLoS process using the specified method.
        H_NLoS = self._generate_NLoS(method=self.NLoS_method)
        H_NLoS_flat = H_NLoS.reshape(self.K*self.Nr*self.Nt, self.num_channel_realizations)

        # Compute the empirical autocorrelation for each realization.
        tau_sim = np.arange(-max_lag, max_lag + 1) * self.Ts
        R_empirical = np.zeros(2 * max_lag + 1, dtype=complex)
        
        for sample_num in range(num_samples):
            
            h = H_NLoS_flat[sample_num, :]
            R_realization = np.zeros(2 * max_lag + 1, dtype=complex)
            R_realization[max_lag] = np.mean(np.abs(h)**2)
            for lag in range(1, max_lag + 1):
                R_realization[max_lag + lag] = np.mean(h[lag:] * np.conj(h[:-lag]))
            R_realization[:max_lag] = np.conj(R_realization[max_lag + 1:])[::-1]

            R_empirical += R_realization
        
        R_empirical /= num_samples
 
        # 2. Compute the analytical autocorrelation.
        tau_analytical = np.linspace(-max_lag * self.Ts, max_lag * self.Ts, 10_000)
        R_analytical = j0(2 * np.pi * self.fD * tau_analytical)
 
        # 3. Plot.
        fig, ax = plt.subplots(figsize=(10, 4))
        
        ax.plot(tau_sim, np.real(R_empirical),  color="tab:blue", label=r"$R_h(\tau)$ (Simulated)")
        ax.plot(tau_analytical, R_analytical, color="black", linestyle="--", label=r"$R_h(\tau) = J_0(2\pi f_D \tau)$ (Analytical)", lw=0.75)
        
        ax.set_xlabel(r"$\tau$ [s]")
        ax.set_ylabel(r"$R_h(\tau)$")
        ax.set_title("Autocorrelation of a NLoS process" if num_samples == 1 else f"Autocorrelation of the NLoS process\n(averaged over {num_samples} process realizations)")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
 
        plot_Rh_filename = f"Rh NLoS process ({self.NLoS_method}) ({num_samples} samples).png"
        plot_Rh_dir = Path(__file__).parents[2] / "report" / "analytical_results" / "channel_statistics" / "plots"
        plot_Rh_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_Rh_dir / plot_Rh_filename, dpi=300, bbox_inches="tight")
        return fig, ax

    def plot_psd(self, num_samples: int = 1) -> None:
        r"""
        Plot the power spectral density (PSD) of the generated NLoS process and compare it to the analytical expression.

        The simulated PSD is calculated by averaging the periodograms of num_samples independent realizations (Bartlett's method):
        .. math::
            \hat{S}_h(f) = \frac{1}{M} \sum_{m=1}^{M} \frac{T_s}{N} \left| H_m(f) \right|^2

        where M is the number of realizations (num_samples).
        
        The analytical PSD is given by Jake's Doppler Spectrum:
        .. math::
            S_h(f) = \frac{1}{\pi f_D \sqrt{1 - \left( \frac{f}{f_D} \right)^2}}, \quad |f| < f_D

        Parameters
        ----------
        num_samples : int
            The number of periodogram realizations to average over. Default is 1. 
            Should be smaller than or equal to K * Nr * Nt, because the average cannot be taken across different simulations.

        Results
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the plot.
        ax : matplotlib.axes.Axes
            The axes object containing the plot.

        Notes
        -----
        This function is intended to be used for validating the generated NLoS process! It will reset the channel model to ensure we start at time instant zero and that all pre-generated NLoS components are deleted. Then, it will generate num_samples sequences of the NLoS process using the specified method, compute the periodogram for each, and average them (Bartlett's method). Finally, it will compute the analytical PSD and plot both the empirical and analytical PSDs.
        """
 
        # 0. Reset the channel model to ensure we start at time instant zero and that all pre-generated NLoS components are deleted.
        assert num_samples <= self.K * self.Nr * self.Nt, f"num_samples should be smaller than or equal to K * Nr * Nt = {self.K * self.Nr * self.Nt}, because the average cannot be taken across different simulations. However, num_samples={num_samples} was given."
        self.reset()
        
        # 1. Compute the power spectral density (Bartlett's method).

        # Generate enough realizations of the NLoS process using the specified method.
        H_NLoS = self._generate_NLoS(method=self.NLoS_method)
        H_NLoS_flat = H_NLoS.reshape(self.K*self.Nr*self.Nt, self.num_channel_realizations)

        # Compute the periodogram for each realization.
        S_sim = np.zeros(self.num_channel_realizations, dtype=float)
        for sample_num in range(num_samples):
            h = H_NLoS_flat[sample_num, :]
            S_sim_realization = (self.Ts / self.num_channel_realizations) * np.abs(np.fft.fft(h))**2
            S_sim += S_sim_realization
        S_sim /= num_samples
        
        # Shift to center zero frequency.
        f_sim = np.fft.fftshift(np.fft.fftfreq(self.num_channel_realizations, d=self.Ts))
        S_sim = np.fft.fftshift(S_sim)
 
        # 2. Compute the analytical PSD (avoid singularity at ±fD).
        f_analytical = np.linspace(-0.999 * self.fD, 0.999 * self.fD, 10_000)
        S_analytical = 1.0 / (np.pi * self.fD * np.sqrt(1 - (f_analytical / self.fD)**2))
 
        # 3. Plot.
        fig, ax = plt.subplots(figsize=(10, 4))
 
        ax.plot(f_sim, S_sim, color="tab:blue", label=r"$S_h(f)$ (Simulated)")
        ax.plot(f_analytical, S_analytical, color="black", linestyle="--", label=r"$S_h(f) = \frac{1}{\pi f_D \sqrt{1-(f/f_D)^2}}$ (Analytical)")
        ax.axvline(-self.fD, color="gray", linestyle=":", lw=1)
        ax.axvline(self.fD, color="gray", linestyle=":", lw=1)
 
        ax.set_xlabel(r"$f$ [Hz]")
        ax.set_ylabel(r"$S_h(f)$")
        ax.set_title("PSD of a NLoS process" if num_samples == 1 else f"PSD of the NLoS process\n(averaged over {num_samples} process realizations)")
        ax.set_xlim(-1.5 * self.fD, 1.5 * self.fD)
        ax.set_xticks(ticks=np.arange(-self.fD, self.fD + 1, self.fD / 4), labels=[r"$-f_D$", r"$-\frac{3}{4}f_D$", r"$-\frac{1}{2}f_D$", r"$-\frac{1}{4}f_D$", r"$0$", r"$\frac{1}{4}f_D$", r"$\frac{1}{2}f_D$", r"$\frac{3}{4}f_D$", r"$f_D$"])
        ax.set_ylim(0, 1.0)
        ax.set_yticks(ticks=np.linspace(0, 1.0, 5))
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
 
        plot_psd_filename = f"PSD NLoS process ({self.NLoS_method}) ({num_samples} samples).png"
        plot_psd_dir = Path(__file__).parents[2] / "report" / "analytical_results" / "channel_statistics" / "plots"
        plot_psd_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_psd_dir / plot_psd_filename, dpi=300, bbox_inches="tight")
        return fig, ax


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

    def __str__(self) -> str:
        """ Return a string representation of the noise model. """
        return "Zero Noise"

    def generate(self, snr: float, x: ComplexArray) -> ComplexArray:
        n = np.zeros((self.K * self.Nr, x.shape[1]), dtype=complex)
        return n

class CSAWGNNoiseModel(NoiseModel):
    """
    Circularly-Symmetric Additive White Gaussian Noise (CSAWGN) Model.

    This noise model generates complex proper, circularly-symmetric additive white Gaussian noise (AWGN) vectors based on the specified signal-to-noise ratio (SNR).
    """

    def __str__(self) -> str:
        """ Return a string representation of the noise model. """
        return "CS Additive White Gaussian Noise"
    
    def generate(self, snr: float, x: ComplexArray) -> ComplexArray:
        
        # Compute the noise power based on the current SNR and the signal power of x.
        p_signal = np.mean( np.sum( np.abs(x)**2, axis=0 ) )
        p_noise = p_signal / snr
        sigma = np.sqrt(p_noise / 2)

        # Generate complex proper, circularly-symmetric AWGN noise vectors with the computed noise power.
        n = sigma * (np.random.randn(self.K*self.Nr, x.shape[1]) + 1j * np.random.randn(self.K*self.Nr, x.shape[1]))
        return n
