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

        # The current channel matrix H.
        self._H = None

        # The current time index m_ch
        self._m_ch = -1
    
    def __str__(self) -> str:
        """ Return a string representation of the channel model. """
        return "Channel Model (abstract base class)"
    
    def reset(self) -> None:
        """
        Reset the channel model to its initial state.
        """
        self._H = None
        self._m_ch = -1
        return
    
    @abstractmethod
    def proceed(self) -> ComplexArray:
        """
        Update the channel state (time index and current channel matrix).

        Returns
        -------
        H : ComplexArray, shape (K * Nr, Nt)
            The new, current channel matrix.
        """
        
        # Update the time index.
        self._m_ch += 1

        return
    
    @abstractmethod
    def get_channel(self) -> ComplexArray:
        """
        Retrieve the channel matrix.
        
        If there is a delay on the CSI, it will return a previous version of the channel matrix!

        Returns
        -------
        H : ComplexArray, shape (K * Nr, Nt)
            The channel matrix (corresping to the current CSI).
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
        y = self._H @ x
        return y

    def __eq__(self, other: object) -> bool:
        
        if not isinstance(other, ChannelModel):
            return NotImplemented
        
        return (
            type(self) == type(other) and
            self.Nt == other.Nt and
            self.Nr == other.Nr and
            self.K == other.K
        )

class NeutralChannelModel(ChannelModel):
    """
    Neutral Channel Model.
    
    This channel model acts as a 'neutral element' for the channel.\\
    In particular, it generates an identity channel matrix, which means that the symbols are transmitted to the receive antennas for which they are intended, and without any interference.
    """

    def __str__(self) -> str:
        """ Return a string representation of the channel model. """
        return "Neutral Channel"

    def proceed(self) -> ComplexArray:
        
        # Update the time index.
        super().proceed()

        # Update the current channel matrix.
        H = np.eye(self.K*self.Nr, self.Nt, dtype=complex)
        self._H = H
        
        return H
    
    def get_channel(self) -> ComplexArray:
        
        # Return the channel matrix corresponding to the current CSI.
        H = self._H
        return H

class IIDRayleighFadingChannelModel(ChannelModel):
    """
    Independent and Identically Distributed (IID) Rayleigh Fading Channel Model.

    This channel model generates a channel matrix with independent and identically distributed (IID) circularly-symmetric zero-mean unit-variance complex Gaussian entries.\\
    The Rayleigh fading aspect is captured by the fact that the channel coefficients change independently after Msv transmissions.
    """

    def __str__(self) -> str:
        """ Return a string representation of the channel model. """
        return "IID Rayleigh Fading Channel"
    
    def proceed(self) -> ComplexArray:
        
        # Update the time index.
        super().proceed()

        # Update the current channel matrix.
        H = (1 / np.sqrt(2)) * (np.random.randn(self.K*self.Nr, self.Nt) + 1j * np.random.randn(self.K*self.Nr, self.Nt))
        self._H = H

        return H

    def get_channel(self) -> ComplexArray:
        
        # Return the channel matrix corresponding to the current CSI.
        H = self._H
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

    def __init__(self, Nt: int, Nr: int, K: int, K_rice: float, fD: float, mode: str, Mch_max: int = 210, NLoS_method : str = "Cholesky-decomposition method"):
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
        mode : str
            The mode of the channel model, a.k.a. the scenario in which the channel model is used.
            Choose between 'terrestrial' and 'satellite'. In satellite mode, the CSI given to the receiver is a delayed version of the CSI.
        
        Mch_max : int
            The maximum number of channel realizations. Needed for methods that require pre-generating the NLoS component for all time instants.
        Tc : float
            The coherence time of the channel. This is the time duration between two consecutive channel realizations.
        
        NLoS_method : str
            The method to use for generating the NLoS component. Choose between: 'Cholesky-decomposition method', 'spectral method', 'FIR filter method'.
        """
        
        super().__init__(Nt, Nr, K)
        
        self.K_rice = K_rice
        self.fD = fD
        self.mode = mode

        self.Mch_max = Mch_max
        self.Tc = 25e-3    

        # Set the delay on the CSI according to the mode.
        self.delay = 0 if mode == "terrestrial" else 10

        # Set the method to use for generating the NLoS component.
        self.NLoS_method = NLoS_method

        # Initialize the channel state.
        self._theta = None
        self._H_NLoS = None
        self._m_ch += self.delay

    def __str__(self) -> str:
        """ Return a string representation of the channel model. """
        return f"Ricean Fading Channel (K = {self.K_rice}, fD = {self.fD}, " + ("satellite" if self.delay > 0 else "terrestrial") + " mode)"
    
    def reset(self) -> None:
        """
        Reset the channel model to its initial state.

        Then, generate the channel matrices for the upcoming simulation.
        """
        
        # Reset the channel model to its initial state.
        super().reset()
        self._theta = None
        self._H_NLoS = None
        self._m_ch += self.delay

        # Generate the channel matrices for the upcoming simulation.
        self._generate()

        return

    def proceed(self) -> ComplexArray:
        
        # Update and validate the time index. It cannot exceed the maximum number of channel realizations Mch_max.
        super().proceed()

        if self._m_ch >= self.Mch_max:
            raise IndexError(f"The time index m_ch cannot exceed the maximum number of channel realizations 2*Mch_max = {self.Mch_max}. However, m_ch = {self._m_ch} was given. Please take a closer look at these parameters.")

        # Update the current channel matrix.
        theta = np.repeat(self._theta, self.Nr*self.Nt).reshape(self.K*self.Nr, self.Nt)
        H_NLoS = self._H_NLoS[:, :, self._m_ch]

        H = np.exp(1j * theta) * (np.sqrt(self.K_rice / (self.K_rice + 1)) + np.sqrt(1 / (self.K_rice + 1)) * H_NLoS)
        self._H = H

        return H

    def get_channel(self):

        # Get the LoS and NLoS component.
        theta = np.repeat(self._theta, self.Nr*self.Nt).reshape(self.K*self.Nr, self.Nt)
        H_NLoS = self._H_NLoS[:, :, self._m_ch - self.delay]

        # Return the channel corresonding to the CSI.
        H = np.exp(1j * theta) * (np.sqrt(self.K_rice / (self.K_rice + 1)) + np.sqrt(1 / (self.K_rice + 1)) * H_NLoS)
        return H
        
    def _generate(self) -> tuple[RealArray, ComplexArray]:
        """
        Generate the channel matrices for all users and all future time instances.
        Store them in the channel state.

        Returns
        -------
        theta : RealArray, shape (K,)
            The arbitrary channel phase, uniformly distributed over [-pi, pi) and statistically independent across different users k.
        H_NLoS : ComplexArray, shape (K * Nr, Nt, Mch_max)
            The No Line-of-Sight (NLoS) component of the channel. 
            It is a complex Gaussian random process with zero mean and unit variance. The channel gains of the different propagation links are correlated in time and uncorrelate din space.
        """

        # Generate the arbitrary channel phase.
        theta = np.random.uniform(low=-np.pi, high=np.pi, size=self.K)
        self._theta = theta

        # Generate the NLoS component.
        H_NLoS = self._generate_NLoS(method=self.NLoS_method)
        self._H_NLoS = H_NLoS

        return theta, H_NLoS
    
    def _generate_NLoS(self, method: str) -> ComplexArray:
        """
        Generate the NLoS component for all users and all time instants, using the specified method.

        Parameters
        ----------
        method : str
            The method to use for generating the NLoS component. Choose between: 'Cholesky-decomposition method', 'spectral method', 'FIR filter method'.

        Returns
        -------
        H_NLoS : ComplexArray, shape (K * Nr, Nt, Mch_max)
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
        H_NLoS : ComplexArray, shape (K * Nr, Nt, Mch_max)
            The generated NLoS component for all propagation links and all time instants.
        """

        # STEP 1.
        R_h = lambda tau: j0(2*np.pi * self.fD * (tau))
        
        i = np.arange(self.Mch_max).reshape(-1,1)
        j = np.arange(self.Mch_max)
        C = R_h((i - j)*self.Tc)
        C += (10e-10)*np.eye(self.Mch_max)

        # STEP 2.
        L = np.linalg.cholesky(C)


        # STEP 3.
        w = (1 / np.sqrt(2)) * (np.random.randn(self.K*self.Nr, self.Nt, self.Mch_max) + 1j * np.random.randn(self.K*self.Nr, self.Nt, self.Mch_max))

        # STEP 4.
        H_NLoS = w @ L.T

        return H_NLoS

    def _generate_NLoS_spectral(self) -> ComplexArray:
        """
        Generate the NLoS component for all users using the spectral method for generating a Gaussian process with a specified power spectral density (PSD) function.

        Returns
        -------
        H_NLoS : ComplexArray, shape (K * Nr, Nt, Mch_max)
            The generated NLoS component for all propagation links and all time instants.
        """
        raise NotImplementedError("The spectral method for generating the NLoS component is not implemented yet.")

    def _generate_NLoS_FIR_filter(self) -> ComplexArray:
        """
        Generate the NLoS component for all users using the FIR filter method for generating a Gaussian process with a specified power spectral density (PSD) function.
        
        Returns
        -------
        H_NLoS : ComplexArray, shape (K * Nr, Nt, Mch_max)
            The generated NLoS component for all propagation links and all time instants.
        """
        raise NotImplementedError("The FIR filter method for generating the NLoS component is not implemented yet.")

    def plot_autocorrelation(self, max_lag: int = 200, num_samples: int = 1, component: str = "NLoS") -> None:
        r"""
        Plot the autocorrelation function of the generated channel gain process and compare it to the analytical expression.

        The simulated autocorrelation is computed as the empirical autocorrelation, averaged over num_samples independent realizations of the NLoS process:
        
        .. math::

            \hat{R}_h(k) = \frac{1}{Msv} \sum_{m_{sv}=1}^{Msv} \cdot \frac{1}{N-k} \sum_{n=0}^{N-k-1} h_{m_{sv}}[n+k] h_{m_{sv}}^*[n], \quad k = 0, 1, \ldots, N-1
        
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
        component : str
            The component of the channel to plot the autocorrelation for. 
            Choose between: 'LoS', 'NLoS' and 'LoS + NLoS'.

        
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the plot.
        ax : matplotlib.axes.Axes
            The axes object containing the plot.

        Notes
        -----
        This function is intended to be used for validating the generated NLoS process! It will reset the channel model to ensure we start at time instant zero and that all pre-generated NLoS components are deleted. Then, it will generate num_samples sequences of the NLoS process using the specified method, and compute the averaged empirical autocorrelation for both positive and negative lags. Finally, it will compute the analytical autocorrelation and plot both the empirical and analytical autocorrelation functions.
        """
 
        # 0. Reset the channel model.
        assert num_samples <= self.K * self.Nr * self.Nt, f"num_samples should be smaller than or equal to K * Nr * Nt = {self.K * self.Nr * self.Nt}, because the average cannot be taken accross different simulations. However, num_samples={num_samples} was given."
        self.reset()
        
        # 1. Compute empirical autocorrelation.

        # Generate the channel gain process (for the specified component) for each propagation link.
        
        if component == "LoS":
            theta = np.broadcast_to(self._theta[:, None, None, None], (self.K, self.Nr, self.Nt, self.Mch_max)).reshape(self.K*self.Nr, self.Nt, self.Mch_max)
            H = np.exp(1j * theta)
        
        elif component == "NLoS":
            H_NLoS = self._H_NLoS
            H = H_NLoS
        
        elif component == "LoS + NLoS":
            theta = np.broadcast_to(self._theta[:, None, None, None], (self.K, self.Nr, self.Nt, self.Mch_max)).reshape(self.K*self.Nr, self.Nt, self.Mch_max)
            H_NLoS = self._H_NLoS
            H = np.exp(1j * theta) * (np.sqrt(self.K_rice / (self.K_rice + 1)) + np.sqrt(1 / (self.K_rice + 1)) * H_NLoS)
        
        else:
            raise ValueError(f"Unknown component '{component}' for plotting the autocorrelation function. Valid options are: 'LoS', 'NLoS' and 'LoS + NLoS'.")

        H_flat = H.reshape(self.K*self.Nr*self.Nt, self.Mch_max)


        # Compute the empirical autocorrelation for each realization.
        
        tau_sim = np.arange(-max_lag, max_lag + 1) * self.Tc
        R_empirical = np.zeros(2 * max_lag + 1, dtype=complex)
        
        for sample_num in range(num_samples):
            
            h = H_flat[sample_num, :]
            R_realization = np.zeros(2 * max_lag + 1, dtype=complex)
            R_realization[max_lag] = np.mean(np.abs(h)**2)
            for lag in range(1, max_lag + 1):
                R_realization[max_lag + lag] = np.mean(h[lag:] * np.conj(h[:-lag]))
            R_realization[:max_lag] = np.conj(R_realization[max_lag + 1:])[::-1]

            R_empirical += R_realization
        
        R_empirical /= num_samples
 
        # 2. Compute the analytical autocorrelation.
        tau_analytical = np.linspace(-max_lag * self.Tc, max_lag * self.Tc, 10_000)
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
 
        plot_Rh_filename = f"Rh {component} process ({self.NLoS_method}) ({num_samples} samples).png"
        plot_Rh_dir = Path(__file__).parents[2] / "report" / "analytical_results" / "channel_statistics" / "plots" / "satellite channel autocorrelation"
        plot_Rh_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_Rh_dir / plot_Rh_filename, dpi=300, bbox_inches="tight")
        return fig, ax

    def plot_NLoS_PSD(self, num_samples: int = 1) -> None:
        r"""
        Plot the power spectral density (PSD) of the generated NLoS process and compare it to the analytical expression.

        The simulated PSD is calculated by averaging the periodograms of num_samples independent realizations (Bartlett's method):
        
        .. math::

            \hat{S}_h(f) = \frac{1}{M} \sum_{m=1}^{M} \frac{T_s}{N} \left| H_{m}(f) \right|^2

        where M is the number of realizations (num_samples).
        
        The analytical PSD is given by Jake's Doppler Spectrum:
        
        .. math::

            S_h(f) = \frac{1}{\pi f_D \sqrt{1 - \left( \frac{f}{f_D} \right)^2}}, \quad |f| < f_D

        Parameters
        ----------
        num_samples : int
            The number of periodogram realizations to average over. Default is 1. 
            Should be smaller than or equal to K * Nr * Nt, because the average cannot be taken across different simulations.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the plot.
        ax : matplotlib.axes.Axes
            The axes object containing the plot.

        Notes
        -----
        This function is intended to be used for validating the generated NLoS process! It will reset the channel model to ensure we start at time instant zero and that all pre-generated NLoS components are deleted. Then, it will generate num_samples sequences of the NLoS process using the specified method, compute the periodogram for each, and average them (Bartlett's method). Finally, it will compute the analytical PSD and plot both the empirical and analytical PSDs.
        """
 
        # 0. Reset the channel model.
        assert num_samples <= self.K * self.Nr * self.Nt, f"num_samples should be smaller than or equal to K * Nr * Nt = {self.K * self.Nr * self.Nt}, because the average cannot be taken across different simulations. However, num_samples={num_samples} was given."
        self.reset()
        
        # 1. Compute the power spectral density (Bartlett's method).

        # Get enough realizations of the NLoS process using the specified method.
        H_NLoS_flat = self._H_NLoS.reshape(self.K*self.Nr*self.Nt, self.Mch_max)

        # Compute the periodogram for each realization.
        S_sim = np.zeros(self.Mch_max, dtype=float)
        for sample_num in range(num_samples):
            h = H_NLoS_flat[sample_num, :]
            S_sim_realization = (self.Tc / self.Mch_max) * np.abs(np.fft.fft(h))**2
            S_sim += S_sim_realization
        S_sim /= num_samples
        
        # Shift to center zero frequency.
        f_sim = np.fft.fftshift(np.fft.fftfreq(self.Mch_max, d=self.Tc))
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
        plot_psd_dir = Path(__file__).parents[2] / "report" / "analytical_results" / "channel_statistics" / "plots" / "satellite channel PSD"
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

    def reset(self) -> None:
        """
        Reset the noise model to its initial state.
        """
        return
    
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

    def __eq__(self, other: object) -> bool:
        
        if not isinstance(other, NoiseModel):
            return NotImplemented
        
        return (
            type(self) == type(other) and
            self.Nr == other.Nr and
            self.K == other.K
        )

class NeutralNoiseModel(NoiseModel):
    """
    Neutral Noise Model.

    This noise model acts as a 'neutral element' for noise.\\
    It does not add any noise to the received signals but simply lets the noiseless received signals pass through.
    """

    def __str__(self) -> str:
        """ Return a string representation of the noise model. """
        return "Zero Noise"

    def get_noise(self, snr: float, x: ComplexArray) -> ComplexArray:
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
    
    def get_noise(self, snr: float, x: ComplexArray) -> ComplexArray:
        
        # Compute the noise power based on the current SNR and the signal power of x.
        p_signal = np.mean( np.sum( np.abs(x)**2, axis=0 ) )
        p_noise = p_signal / snr
        sigma = np.sqrt(p_noise / 2)

        # Generate complex proper, circularly-symmetric AWGN noise vectors with the computed noise power.
        n = sigma * (np.random.randn(self.K*self.Nr, x.shape[1]) + 1j * np.random.randn(self.K*self.Nr, x.shape[1]))
        return n
