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
            The channel matrices corresponding to the p previous channel estimates.
        
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
    
class RiceanIIDFadingChannel(ChannelModel):
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

    def __init__(self, Nt: int, Nr: int, K: int, K_rice: float, Trtt_2_Tc: float, Msv: int, Tc_scale: float):
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
            The ratio between the round trip time :math:`T_{rtt}` and the coherence time :math:`T_c`.\\
            It quantifies the block period and the delay on the CSI.
        Msv : int
            The number of symbol vector transmissions per channel realization.
            It quantifies the ratio between the symbol period and the coherence time :math:`T_{\text{symbol}} / T_c`.
        Tc_scale : float
            The dimensionless scaling factor applied to the conventional coherence time.\\
            It defines the coherence time, such that :math:`T_c = \text{Tc\_scale} \cdot T_{c,\text{ conventional}}`, where :math:`T_{c,\text{conventional}}` is the conventional coherence time defined as the time lag at which the autocorrelation function of the channel gains reaches :math:`0.5` (:math:`R_h(T_{c,\text{conventional}}) = \frac{1}{2}`).
        
        Notes
        -----
        Computation of the symbol period to coherence time ratio:

        .. math::

            \frac{T_{\text{block}}}{T_{\text{symbol}}} = M_{\text{sv}} \iff \frac{T_{\text{symbol}}}{T_c} = \frac{\frac{T_{\text{block}}}{T_c}}{M_{\text{sv}}}

        
        Computation of the the delay on the CSI feedback message (in symbol vector periods):

        .. math::

            \text{CSI delay}
            &= \frac{T_{\text{RTT}}}{T_{\text{symbol}}} \\
            &= \frac{\frac{T_{\text{RTT}}}{T_c}}{\frac{T_{\text{symbol}}}{T_c}} \\
            &= \frac{\frac{T_{\text{RTT}}}{T_c}}{\frac{T_{\text{block}}}{T_c}} \cdot M_{\text{sv}} \\
            &  \qquad \bullet \frac{T_{\text{block}}}{T_c} = 1 - \frac{T_{\text{RTT}}}{T_c} \\
            &= \frac{\frac{T_{\text{RTT}}}{T_c}}{1 - \frac{T_{\text{RTT}}}{T_c}} \cdot M_{\text{sv}}

        """
        
        # Initialize the base class.
        super().__init__(Nt, Nr, K)

        # Validate the parameters.
        assert K_rice >= 0, f"The Rice factor K_rice must be non-negative.\nCurrent value: {K_rice}"
        assert 0 <= Trtt_2_Tc, f"The round trip time to coherence time ratio Trtt_2_Tc must be non-negative.\nCurrent value: {Trtt_2_Tc}"
        assert 0 < Tc_scale, f"The coherence time scaling factor Tc_scale must be strictly positive.\nCurrent value: {Tc_scale}"
        
        # Store the channel parameters.
        self.K_rice = K_rice
        self.Trtt_2_Tc = Trtt_2_Tc
        self.Msv = Msv
        self.Tc_scale = Tc_scale

        Tblock_2_Tc = (1 - self.Trtt_2_Tc) if self.Trtt_2_Tc < 1 else 1.0

        # Compute and store the helper parameters

        # the symbol period to coherence time ratio
        self.Tsymbol_2_Tc = Tblock_2_Tc * (1 / Msv)

        # the delay on the CSI feedback message (in symbol vector periods)
        self.CSI_delay = int( np.ceil( (self.Trtt_2_Tc / Tblock_2_Tc) * Msv ) )

        # the maximum Doppler frequency times the coherence time
        self.fD_times_Tc = (1 / (Tc_scale * 2*np.pi)) * brentq(lambda z: j0(z) - 0.5, 0, 2.5) if Tc_scale != np.inf else 0.0

        return

    def __str__(self) -> str:
        return f"Ricean Channel with an IID time-correlated fading NLoS component.\n   K_rice = {self.K_rice}, Trtt_2_Tc = {np.round(self.Trtt_2_Tc, 2)}, Rh({np.round(self.Tc_scale, 1)} Tc) = 0.5)"
    
    def __eq__(self, other):
        
        if not isinstance(other, RiceanIIDFadingChannel):
            return NotImplemented
        
        return (
            self.Nt == other.Nt and
            self.Nr == other.Nr and
            self.K == other.K and
            self.K_rice == other.K_rice and
            self.Trtt_2_Tc == other.Trtt_2_Tc and
            self.Tc_scale == other.Tc_scale
        )

    def generate(self) -> ComplexArray:

        # Generate the arbitrary channel phase.
        theta_k = np.random.uniform(low=-np.pi, high=np.pi, size=self.K)
        theta = np.repeat(theta_k, self.Nr*self.Nt).reshape(self.K*self.Nr, self.Nt)

        # Generate the NLoS components for all propagation links and time instants.
        H_NLoS_process = self._generate_NLoS(self.CSI_delay + self.Msv)

        # Compute the channel matrices for all symbol vector transmissions of the consecutive block.
        H_NLoS = H_NLoS_process[-self.Msv : ]
        H = np.exp(1j * theta) * (np.sqrt(self.K_rice / (self.K_rice + 1)) + np.sqrt(1 / (self.K_rice + 1)) * H_NLoS)
        self._H = H

        # Compute the CSI corresponding to this channel.
        H_NLoS_CSI = H_NLoS_process[0]
        H_CSI = np.exp(1j * theta) * (np.sqrt(self.K_rice / (self.K_rice + 1)) + np.sqrt(1 / (self.K_rice + 1)) * H_NLoS_CSI)
        self._H_CSI = H_CSI[np.newaxis, :, :]

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

        # If Tc -> 0, all channel matrices of the consecutive block are fully correlated!
        if self.fD_times_Tc == 0:
            w = (1 / np.sqrt(2)) * (np.random.randn(self.K*self.Nr*self.Nt) + 1j * np.random.randn(self.K*self.Nr*self.Nt))
            H_NLoS = np.tile(w, (num_channel_samples, 1)).reshape(num_channel_samples, self.K*self.Nr, self.Nt)
            return H_NLoS
        
        # STEP 1.
        i = np.arange(num_channel_samples).reshape(-1,1)
        j = np.arange(num_channel_samples)
        C = j0(2*np.pi * self.fD_times_Tc * (i-j) * self.Tsymbol_2_Tc)
        C += (1e-10)*np.eye(num_channel_samples)

        # STEP 2.
        L = np.linalg.cholesky(C)

        # STEP 3.
        w = (1 / np.sqrt(2)) * (np.random.randn(num_channel_samples, self.K*self.Nr*self.Nt) + 1j * np.random.randn(num_channel_samples, self.K*self.Nr*self.Nt))

        # STEP 4.
        H_NLoS = (L @ w).reshape(num_channel_samples, self.K*self.Nr, self.Nt)

        return H_NLoS

    def plot_channel_gain_process(self, N_Tc: int = 5, component: str = "NLoS", expected_value_ref: bool = True, uncorrelated_process_ref: bool = False, block_fading_assumption_ref: bool = False) -> tuple[plt.Figure, plt.Axes]:
        r"""
        Plot the magnitude of the channel gain process of a single propagation link over time.

        Parameters
        ----------
        N_Tc : int
            The number of coherence periods to plot. Default is 5.
        component : str
            The component of the channel to plot. Choose between: 'LoS', 'NLoS' and 'LoS + NLoS'.
        
        expected_value_ref : bool
            Whether to plot the expected value of the magnitude as a reference line. Default is True.
        uncorrelated_process_ref : bool
            Whether to plot an uncorrelated channel gain process with the same average power and the same number of samples for comparison. Default is False.
        block_fading_assumption_ref : bool
            Whether to plot the channel gain process used in the simulations as a reference. This is a channel gain process that is constant during one block period!
        
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the plot.
        ax : matplotlib.axes.Axes
            The axes object containing the plot.
        """

        # 0. To speed up the plot generation, limit the number of samples!
        Tsymbol_2_Tc_original = self.Tsymbol_2_Tc
        M_SVPTC = 50

        self.Tsymbol_2_Tc = 1 / M_SVPTC
        num_samples = N_Tc * M_SVPTC + 1

        
        # 1. Generate channel gain process and compute the magnitude.
        if component == "LoS":
            theta_k = np.random.uniform(low=-np.pi, high=np.pi)
            h = np.exp(1j * theta_k) * np.ones(num_samples)
        elif component == "NLoS":
            h_NLoS = self._generate_NLoS(num_samples)[:, 0, 0]
            h = h_NLoS
        elif component == "LoS + NLoS":
            theta_k = np.random.uniform(low=-np.pi, high=np.pi)
            h_NLoS = self._generate_NLoS(num_samples)[:, 0, 0]
            h = np.exp(1j * theta_k) * (np.sqrt(self.K_rice / (self.K_rice + 1)) + np.sqrt(1 / (self.K_rice + 1)) * h_NLoS)
        else:
            raise ValueError(f"Unknown component '{component}' for plotting the channel gain process. Choose between: 'LoS', 'NLoS' and 'LoS + NLoS'.")

        h_magnitude = np.abs(h)
        t = np.arange(num_samples)


        # 2. Plot.

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(t, h_magnitude, color="tab:blue", linewidth=2, label="Clark's Fading Model")

        # plot an uncorrelated channel gain process with the same average power and the same number of samples for comparison if asked.
        if uncorrelated_process_ref:

            if component == "LoS":
                h_uncorrelated = np.exp(1j * theta_k) * np.ones(num_samples)
            elif component == "NLoS":
                h_uncorrelated = (1 / np.sqrt(2)) * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
            elif component == "LoS + NLoS":
                h_uncorrelated = np.exp(1j * theta_k) * (np.sqrt(self.K_rice / (self.K_rice + 1)) + np.sqrt(1 / (self.K_rice + 1)) * (1 / np.sqrt(2)) * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)))

            h_uncorrelated_magnitude = np.abs(h_uncorrelated)
            ax.plot(t, h_uncorrelated_magnitude, color="tab:red", linestyle="--", linewidth=1, label="Uncorrelated Fading")

        # plot the channel gain process used in the simulations as a reference if asked. This is a channel gain process that is constant during one block period.
        if block_fading_assumption_ref:

            h_assumption_magnitude = np.abs(h[::M_SVPTC])
            ax.step(t[::M_SVPTC], h_assumption_magnitude, where="post", color="tab:red", linestyle="--", linewidth=1.5, label="Block Fading Assumption")

        # plot the expected value of the magnitude as a reference line.
        if expected_value_ref:

            if component == "LoS":
                h_magnitude_exp = 1
            elif component == "NLoS":
                h_magnitude_exp = np.sqrt(np.pi / 4)
            elif component == "LoS + NLoS":
                h_magnitude_exp = np.sqrt(np.pi / (4 * (self.K_rice + 1))) * np.exp(-self.K_rice/2) * ( (1 + self.K_rice) * i0(self.K_rice/2) + self.K_rice * i1(self.K_rice/2) )

            ax.axhline(h_magnitude_exp, color="black", linestyle="-", linewidth=1, label=r"$\mathbb{E}\left[|h(t)|\right]$")

        # x-axis ticks and labels.
        def Tc_formatter(x, pos):
            n = int(round(x / M_SVPTC))
            if   n == 0:  return r"$0$"
            elif n == 1:  return r"$T_c$"
            elif n == -1: return r"$-T_c$"
            else:         return rf"${n} \, T_c$"
        ax.xaxis.set_major_locator(MultipleLocator(num_samples // 5))
        ax.xaxis.set_major_formatter(FuncFormatter(Tc_formatter))

        # plot settings.
        ax.set_xlabel(r"$t \; [s]$")
        ax.set_ylabel(r"$\left|h(t)\right|$")
        ax.set_title(f"Channel gain process of a propagation link\n{component} component")
        ax.legend()
        ax.grid(True, which="both", linestyle="--")
        plt.tight_layout()

        # save the plot.
        plot_filename = f"channel gain process {component} ({N_Tc} coherence periods)" + (" (uncorrelated ref)" if uncorrelated_process_ref else "") + (" (block fading assumption ref)" if block_fading_assumption_ref else "") + ".png"
        plot_dir = Path(__file__).parents[2] / "report" / "analytical_results" / "channel_statistics" / "plots" / "ricean time-correlated fading channel" / "channel gain process"
        plot_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_dir / plot_filename, dpi=300, bbox_inches="tight")


        # restore the original symbol period to coherence time ratio.
        self.Tsymbol_2_Tc = Tsymbol_2_Tc_original

        return fig, ax

    def plot_autocorrelation(self, N_Tc: int = 10, component: str = "NLoS", average: bool = False) -> tuple[plt.Figure, plt.Axes]:
        r"""
        Plot the empirical autocorrelation function of the generated channel gain process and compare it to the analytical expression.

        The simulated, empirical autocorrelation is calculated as follows:
        
        .. math::

            \hat{R}_h(k) = \frac{1}{\hat{R}_h(0)} \cdot \frac{1}{N} \sum_{n=0}^{N-k-1} h[n + k] \cdot h[n]^*
        
        where :math:`N` is the total number of channel realizations and :math:`k` is the lag, expressed in number of samples.
        
        The analytical autocorrelation equals the zero-order Bessel function of the first kind:
        
        .. math::

            R_h(\tau) = J_0(2\pi f_D \tau)

        Parameters
        ----------
        N_Tc : int
            The number of coherence periods to compute and plot the autocorrelation for.
        component : str
            The component of the channel to plot the autocorrelation for. 
            Choose between: 'LoS', 'NLoS' and 'LoS + NLoS'.
        average : bool
            Whether to average the empirical autocorrelation over multiple realizations of the channel gain process. Default is False.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the plot.
        ax : matplotlib.axes.Axes
            The axes object containing the plot.
        """

        # 0. To speed up the plot generation, limit the number of samples!
        Tsymbol_2_Tc_original = self.Tsymbol_2_Tc
        M_SVPTC = 25

        self.Tsymbol_2_Tc = 1 / M_SVPTC
        num_samples = 10 * N_Tc * M_SVPTC


        # 1.1 Generate channel gain process
        if component == "LoS":
            theta_k = np.random.uniform(low=-np.pi, high=np.pi)
            h = np.exp(1j * theta_k) * np.ones((self.K*self.Nr*self.Nt, num_samples))
        elif component == "NLoS":
            h = self._generate_NLoS(num_samples).reshape(num_samples, self.K*self.Nr*self.Nt).T
        elif component == "LoS + NLoS":
            theta_k = np.random.uniform(low=-np.pi, high=np.pi)
            h_NLoS = self._generate_NLoS(num_samples).reshape(num_samples, self.K*self.Nr*self.Nt).T
            h = np.exp(1j * theta_k) * (np.sqrt(self.K_rice / (self.K_rice + 1)) + np.sqrt(1 / (self.K_rice + 1)) * h_NLoS)
        else:
            raise ValueError(f"Unknown component '{component}' for plotting the autocorrelation. Choose between: 'LoS', 'NLoS' and 'LoS + NLoS'.")

        # 1.2 Compute empirical autocorrelation.
        h = h[[0]] if not average else h
        Rh_simulation = []
        for h_i in h:
            Rh_i = (1 / len(h_i)) * np.correlate(h_i, h_i, mode='full')[(len(h_i)-1) - (N_Tc//2 * M_SVPTC) : (len(h_i)-1) + (N_Tc//2 * M_SVPTC)]
            Rh_simulation.append(np.real(Rh_i / Rh_i[N_Tc//2 * M_SVPTC]))
        Rh_simulation = np.mean(Rh_simulation, axis=0)

        tau = np.arange(-N_Tc//2 * M_SVPTC, N_Tc//2 * M_SVPTC)


        # 2. Compute the analytical autocorrelation.
        Rh_analytical = j0(2 * np.pi * self.fD_times_Tc * tau * self.Tsymbol_2_Tc)


        # 3. Plot.
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(tau, np.real(Rh_simulation), color="tab:blue", linewidth=2.5, label=r"$R_h(\tau)$ (Simulated)")
        ax.plot(tau, Rh_analytical, color="black", linestyle="--", linewidth=1.5, label=r"$R_h(\tau) = J_0(2\pi f_D \tau)$ (Analytical)")
        if component == "LoS + NLoS": ax.axhline(self.K_rice / (self.K_rice + 1), color="slategray", linestyle="-", linewidth=1.5, label=r"$\frac{K}{K+1}$")

        # x-axis ticks and labels.
        def Tc_formatter(x, pos):
            n = int(round(x / M_SVPTC))
            if   n == 0:  return r"$0$"
            elif n == 1:  return r"$T_c$"
            elif n == -1: return r"$-T_c$"
            else:         return rf"${n} \, T_c$"
        ax.xaxis.set_major_locator(MultipleLocator((N_Tc * M_SVPTC) / 10))
        ax.xaxis.set_major_formatter(FuncFormatter(Tc_formatter))

        # y-axis ticks and labels.
        ax.yaxis.set_major_locator(MultipleLocator(0.25))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y:.2f}"))

        ax.set_xlabel(r"$\tau \; [s]$")
        ax.set_ylabel(r"$R_h(\tau)$")
        ax.set_title(f"Autocorrelation of a {component} process" + ("\naveraged across all propagation links" if average else ""))
        ax.legend()
        ax.grid(True, which="both", linestyle="--")
        plt.tight_layout()

        plot_Rh_filename = f"Rh {component} process ({N_Tc} coherence periods)" + (" (averaged)" if average else "") + ".png"
        plot_Rh_dir = Path(__file__).parents[2] / "report" / "analytical_results" / "channel_statistics" / "plots" / "ricean time-correlated fading channel" / "autocorrelation"
        plot_Rh_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_Rh_dir / plot_Rh_filename, dpi=300, bbox_inches="tight")

        # restore the original symbol period to coherence time ratio.
        self.Tsymbol_2_Tc = Tsymbol_2_Tc_original

        return fig, ax

    def plot_NLoS_PSD(self, length_segments: int, num_segments: int, average: bool = True) -> tuple[plt.Figure, plt.Axes]:
        r"""
        Plot the power spectral density (PSD) of the generated NLoS process and compare it to the analytical expression.

        The simulated PSD is calculated using Bartlett's method:
        
        .. math::

            \hat{S}_h(f_k) = \frac{1}{M} \sum_{m=0}^{M-1} \hat{S}_h^{(m)}(f_k), \quad \text{with } \hat{S}_h^{(m)}(f_k) = \frac{T_{\text{sym}}}{L} \left| \sum_{n=0}^{L-1} h[mL + n] \, e^{-j 2\pi k n / L} \right|^2 \quad \text{and} \quad f_k = \frac{k}{L \cdot T_{\text{sym}}}

        where :math:`L` is the segment length, :math:`M` is the number of segments (`num_segments`), and :math:`N = M \cdot L` is the total number of channel realizations.
        
        The analytical PSD is given by Jake's Doppler Spectrum:
        
        .. math::

            S_h(f) = \frac{1}{\pi f_D \sqrt{1 - \left( \frac{f}{f_D} \right)^2}}, \quad |f| < f_D

        Parameters
        ----------
        length_segments : int
            The number of samples per segment.\\
            This is the FFT length per segment. It controls frequency resolution. A larger value gives finer resolution. Default is 200.
        num_segments : int
            The number of independent segments to average over when computing the simulated PSD using Bartlett's method.\\
            It controls the variance of the PSD estimate. A larger value gives a smoother PSD. Default is 10.
        average : bool
            If this is True, the PSDs will be average over all propagation links.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the plot.
        ax : matplotlib.axes.Axes
            The axes object containing the plot.
        """

        assert num_segments > 0, "`num_segments` must be a positive integer."
        assert length_segments > 0, "`length_segments` must be a positive integer."

        # 0. To speed up the plot generation, limit the number of samples!
        Tsymbol_2_Tc_original = self.Tsymbol_2_Tc
        self.Tsymbol_2_Tc = 1.0

        num_samples = length_segments * num_segments


        # 1. Generate NLoS process and compute the PSD (Bartlett's method).
        H_NLoS = self._generate_NLoS(num_samples)

        if average:
            h = H_NLoS
            h_segments = h.reshape(num_segments, length_segments, self.K*self.Nr, self.Nt)
        else:
            h = H_NLoS[:, 0, 0]
            h_segments = h.reshape(num_segments, length_segments)

        H_fft = np.fft.fft(h_segments, axis=1)
        periodogram = (self.fD_times_Tc * self.Tsymbol_2_Tc / length_segments) * np.abs(H_fft)**2

        S_simulation = np.fft.fftshift(np.mean(periodogram, axis=(0 if not average else (0, 2, 3))))
        f_simulation = np.fft.fftshift(np.fft.fftfreq(length_segments, d=self.fD_times_Tc * self.Tsymbol_2_Tc))


        # 2. Compute the analytical PSD in normalized units.
        f_analytical_norm = np.linspace(-0.999, 0.999, 10_000)
        S_analytical_norm = 1.0 / (np.pi * np.sqrt(1 - f_analytical_norm**2))


        # 3. Plot.
        fig, ax = plt.subplots(figsize=(10, 4))

        ax.plot(f_simulation, S_simulation, color="tab:blue", linewidth=2.5, label=r"$S_h(f)$ (Simulated)")
        ax.plot(f_analytical_norm, S_analytical_norm, color="black", linestyle="--", linewidth=1.5, label=r"$S_h(f) = \frac{1}{\pi f_D \sqrt{1-(f/f_D)^2}}$ (Analytical)")

        ax.axvline(-1.0, color="gray", linestyle=":", lw=1)
        ax.axvline( 1.0, color="gray", linestyle=":", lw=1)

        # x-axis ticks and labels.
        def fD_formatter(x, pos):
            x_mul_4 = int(round(x * 4))
            if   x_mul_4 ==  0: return r"$0$"
            elif x_mul_4 ==  1: return r"$\frac{1}{4} \, f_D$"
            elif x_mul_4 == -1: return r"$\frac{-1}{4} \, f_D$"
            elif x_mul_4 ==  2: return r"$\frac{1}{2} \, f_D$"
            elif x_mul_4 == -2: return r"$\frac{-1}{2} \, f_D$"
            elif x_mul_4 ==  3: return r"$\frac{3}{4} \, f_D$"
            elif x_mul_4 == -3: return r"$\frac{-3}{4} \, f_D$"
            elif x_mul_4 ==  4: return r"$f_D$"
            elif x_mul_4 == -4: return r"$-f_D$"
            elif x_mul_4 ==  5: return r"$\frac{5}{4} \, f_D$"
            elif x_mul_4 == -5: return r"$\frac{-5}{4} \, f_D$"
            else:               return rf"${x_mul_4/4:.2f} \, f_D$"
        ax.xaxis.set_major_locator(MultipleLocator(0.25))
        ax.xaxis.set_major_formatter(FuncFormatter(fD_formatter))

        # y-axis ticks and labels.
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y:.1f}"))

        ax.set_xlabel(r"$f \; [Hz]$")
        ax.set_ylabel(r"$S_h(f) \cdot f_D$")
        ax.set_xlim(-1.25, 1.25)
        ax.set_ylim(0, 4.0)
        ax.set_title("PSD of a NLoS process" + ("\naveraged across all propagation links" if average else ""))
        ax.legend()
        ax.grid(True, which="both", linestyle="--")
        plt.tight_layout()

        plot_psd_filename = f"PSD NLoS process ({num_segments} segments, {length_segments} samples per segment)" + (" (average)" if average else "") + ".png"
        plot_psd_dir = Path(__file__).parents[2] / "report" / "analytical_results" / "channel_statistics" / "plots" / "ricean time-correlated fading channel" / "PSD"
        plot_psd_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_psd_dir / plot_psd_filename, dpi=300, bbox_inches="tight")

        # restore the original symbol period to coherence time ratio.
        self.Tsymbol_2_Tc = Tsymbol_2_Tc_original

        return fig, ax


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


# TESTING AND VALIDATION

if __name__ == "__main__":

    channel_model = RiceanIIDFadingChannel(Nt=8, Nr=2, K=4, K_rice=10**(5/10), Trtt_2_Tc=0, Msv=500, Tc_scale=1)

    # channel_model.plot_channel_gain_process()
    # channel_model.plot_channel_gain_process(N_Tc=20)
    # channel_model.plot_channel_gain_process(N_Tc=5, uncorrelated_process_ref=True)
    # channel_model.plot_channel_gain_process(N_Tc=5, block_fading_assumption_ref=True)

    # channel_model.plot_autocorrelation()
    # channel_model.plot_autocorrelation(average=True)
    # channel_model.plot_autocorrelation(N_Tc=50)
    # channel_model.plot_autocorrelation(N_Tc=50, average=True)
    # channel_model.plot_autocorrelation(component="LoS + NLoS")
    # channel_model.plot_autocorrelation(component="LoS + NLoS", average=True)
    # channel_model.plot_autocorrelation(N_Tc=50, component="LoS + NLoS")
    # channel_model.plot_autocorrelation(N_Tc=50, component="LoS + NLoS", average=True)

    # channel_model.plot_NLoS_PSD(length_segments=400, num_segments=10, average=True)
    # channel_model.plot_NLoS_PSD(length_segments=400, num_segments=10, average=False)
    # channel_model.plot_NLoS_PSD(length_segments=200, num_segments=20, average=True)
    # channel_model.plot_NLoS_PSD(length_segments=200, num_segments=20, average=False)
