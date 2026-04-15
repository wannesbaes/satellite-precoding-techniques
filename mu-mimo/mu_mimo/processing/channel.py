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

    def __init__(self, Nt: int, Nr: int, K: int, K_rice: float, Trtt_2_Tc: float, Mch_max: int = 2000):
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
        Trtt_2_Tc : float
            The ratio between the round trip time :math:`T_{rtt}` and the coherence time :math:`T_c`. It quantifies the delay on the CSI.
        
        Mch_max : int
            The maximum number of channel realizations (or blocks). Needed for methods that require pre-generating the NLoS component for all time instants.
        """
        
        # Initialize the base class.
        super().__init__(Nt, Nr, K)
        
        # Set the Rice factor.
        self.K_rice = K_rice

        # Compute and store the block period and the delay on the CSI.
        self.T_block = self._compute_T_block(Trtt_2_Tc)
        self.CSI_delay = self._compute_CSI_delay(Trtt_2_Tc)

        # Set the maximum number of channel realizations (or thus blocks).
        self.Mch_max = Mch_max + self.CSI_delay

        # Initialize the channel state.
        self._theta = None
        self._H_NLoS = None
        self._m_ch += self.CSI_delay

    def __str__(self) -> str:
        """ Return a string representation of the channel model. """
        return f"Ricean Fading Channel (K_rice = {self.K_rice}), " + ("terrestrial mode" if self.CSI_delay == 0 else (f"satellite mode: T_block = {round(self.T_block, 2)} * Tc, CSI delay = {self.CSI_delay} * T_block"))
    
    def reset(self) -> None:
        """
        Reset the channel model to its initial state.

        Then, generate the channel matrices for the upcoming simulation.
        """
        
        # Reset the channel model to its initial state.
        super().reset()
        self._theta = None
        self._H_NLoS = None
        self._m_ch += self.CSI_delay

        # Generate the channel matrices for the upcoming simulation.
        self._generate()

        return

    def proceed(self) -> ComplexArray:
        
        # Update and validate the time index. It cannot exceed the maximum number of channel realizations Mch_max.
        super().proceed()

        if self._m_ch >= self.Mch_max:
            raise IndexError(f"The time index `m_ch` cannot exceed the maximum number of channel realizations 2*Mch_max = {self.Mch_max}. However, m_ch = {self._m_ch} was given. Please take a closer look at these parameters.")

        # Update the current channel matrix.
        theta = np.repeat(self._theta, self.Nr*self.Nt).reshape(self.K*self.Nr, self.Nt)
        H_NLoS = self._H_NLoS[:, :, self._m_ch]

        H = np.exp(1j * theta) * (np.sqrt(self.K_rice / (self.K_rice + 1)) + np.sqrt(1 / (self.K_rice + 1)) * H_NLoS)
        self._H = H

        return H

    def get_channel(self):

        # Get the LoS and NLoS component.
        theta = np.repeat(self._theta, self.Nr*self.Nt).reshape(self.K*self.Nr, self.Nt)
        H_NLoS = self._H_NLoS[:, :, self._m_ch - self.CSI_delay]

        # Return the channel corresonding to the CSI.
        H = np.exp(1j * theta) * (np.sqrt(self.K_rice / (self.K_rice + 1)) + np.sqrt(1 / (self.K_rice + 1)) * H_NLoS)
        return H
        
    def _compute_T_block(self, Trtt_2_Tc: float) -> float:
        r"""
        Compute the block period :math:`T_{\text{block}}` as a fraction of the coherence time :math:`T_c`, based on the round trip time to coherence time ratio :math:`\frac{T_{RTT}}{T_c}`.

        If :math:`\frac{T_{RTT}}{T_c} \leq \frac{1}{2}`, then there is no delay on the CSI. This is the terrestrial communication scenario.
        In this case, we set the block period :math:`T_{\text{block}}` equal to the largest fraction of half the coherence time :math:`T_c` that is an integer multiple of the round trip time :math:`T_{RTT}`.
        
        .. math::
            T_{\text{block}} = \left\lfloor \frac{1/2}{\frac{T_{RTT}}{T_c}} \right\rfloor \, \frac{T_{RTT}}{T_c} \cdot T_c \text{ (in seconds)}

        If :math:`\frac{T_{RTT}}{T_c} > \frac{1}{2}`, then there is a delay on the CSI. This is a satellite communication scenario.
        In this case, we set the block period :math:`T_{\text{block}}` equal to the largest fraction of the coherence time :math:`T_c` that is a divisor of the round trip time to coherence time ratio :math:`\frac{T_{RTT}}{T_c}`.
        
        .. math::
            T_{\text{block}} = \frac{\frac{T_{RTT}}{T_c}}{\left\lceil \frac{T_{RTT}}{T_c} \right\rceil} \cdot T_c \text{ (in seconds)}
        
        Parameters
        ----------
        Trtt_2_Tc : float
            The round trip time to coherence time ratio :math:`\frac{T_{RTT}}{T_c}`.
        
        Returns
        -------
        Tb : float (0 < Tb <= 1)
            The block period :math:`T_{\text{block}}`, expressed as a fraction of the coherence time :math:`T_c`!
        
        Note
        ----
         - To get the block period in seconds, multiply the return value by the coherence time :math:`T_c`.
         - The block period is equal to the sample period of the channel gain process. During one sample period (:math:`< T_c`), the channel is assumed to be constant (block fading).
        """
        
        # terrestrial communication scenario.
        if Trtt_2_Tc <= 0.5: 
            T_block = int(np.floor(0.5 / Trtt_2_Tc)) * Trtt_2_Tc
        
        # satellite communication scenario.
        else: 
            T_block = Trtt_2_Tc / int(np.ceil(Trtt_2_Tc))

        # return the block period.
        return T_block
    
    def _compute_CSI_delay(self, Trtt_2_Tc: float) -> int:
        r"""
        Compute the delay on the CSI, expressed as an integer multiple of the block period :math:`T_{\text{block}}`.

        In the satellite communication scenario (:math:`\frac{T_{RTT}}{T_c} > \frac{1}{2}`), there is a delay on the CSI:
        
        .. math::
            \text{delay} = \lceil \frac{T_{RTT}}{T_c} \rceil \cdot T_{\text{block}} \text{ (in seconds)}
        
        Parameters
        ----------
        Trtt_2_Tc : float
            The round trip time to coherence time ratio :math:`\frac{T_{RTT}}{T_c}`.
        
        Returns
        -------
        delay : int
            The delay on the CSI, expressed as an integer multiple of the block period :math:`T_{\text{block}}`.

        Note
        ----
         - To get the delay in seconds, multiply the return value by the block period :math:`T_{\text{block}}`.
         - In the terrestrial communication scenario, there is no delay on the CSI, so the delay equals zero.
        """
        delay = int(np.ceil(Trtt_2_Tc)) if Trtt_2_Tc > 0.5 else 0
        return delay
    
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
        H_NLoS = self._generate_NLoS(method="Cholesky-decomposition method")
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
        
        - **Step 1:** Build the :math:`N \times N` covariance matrix :math:`\mathbf{C}` with entries :math:`C_{i,j} = R_h((i-j) \, T_{\text{sample}})`.
        - **Step 2:** Compute the Cholesky decomposition of the covariance matrix :math:`\mathbf{C} = \mathbf{L}\mathbf{L}^H`, where :math:`\mathbf{L}` is a lower triangular matrix.
        - **Step 3:** Generate a column vector :math:`\mathbf{w}` of :math:`N` i.i.d. white complex Gaussian random variables with zero mean and unit variance.
        - **Step 4:** Find the desired Gaussian process as :math:`\mathbf{h} = \mathbf{L}\mathbf{w}`.

        Returns
        -------
        H_NLoS : ComplexArray, shape (K * Nr, Nt, Mch_max)
            The generated NLoS component for all propagation links and all time instants.
        
        Note
        ----
        For Jake's model, where the PSD of the Gaussian processes equals the Doppler spectrum, the autocorrelation function :math:`R_h(\tau)` corresponds to a zeroth-order Bessel function of the first kind :math:`J_0(2\pi f_D \tau)`.

        The sample period :math:`T_{\text{sample}}` of the channel realizations is equal to the block period :math:`T_{\text{block}} (< T_c)`.
        Because the block period attribute `self.T_block` is expressed as a fraction of the coherence time, we can compute the covariance matrix in function of this fraction only:
        
        .. math::
            R_h((i-j) \, T_{\text{sample}}) 
            &= R_h((i-j) \, T_{\text{block}}) \\
            &= J_0(2\pi \, f_D \cdot (i-j) \, T_{\text{block}}) \\
            &= J_0(2\pi \, (0.242) \frac{1}{T_c} \cdot (i-j) \cdot \text{self.T\_block} \cdot T_c) \\
            &= J_0(2\pi \, (0.242) \, (i-j) \ \text{self.T\_block}) \\
            
        where we used
        
        .. math::
            R_h(\tau) \geq \frac{1}{2} \iff \left| 2\pi f_D \tau \right| \leq 1.521 \implies \tau_c \approx (0.242) \cdot \frac{1}{f_D}
        
        """

        # STEP 1.
        i = np.arange(self.Mch_max).reshape(-1,1)
        j = np.arange(self.Mch_max)
        C = j0(2*np.pi * (0.242097596) * (i-j) * self.T_block)
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
        
        Note
        ----
        The spectral method is not implemented yet. It will raise a `NotImplementedError` when called.
        Use the Cholesky decomposition method instead.
        """
        raise NotImplementedError("The spectral method for generating the NLoS component is not implemented yet.")

    def _generate_NLoS_FIR_filter(self) -> ComplexArray:
        """
        Generate the NLoS component for all users using the FIR filter method for generating a Gaussian process with a specified power spectral density (PSD) function.
        
        Returns
        -------
        H_NLoS : ComplexArray, shape (K * Nr, Nt, Mch_max)
            The generated NLoS component for all propagation links and all time instants.
        
        Note
        ----
        The FIR filter method is not implemented yet. It will raise a `NotImplementedError` when called.
        Use the Cholesky decomposition method instead.
        """
        raise NotImplementedError("The FIR filter method for generating the NLoS component is not implemented yet.")

    def plot_channel_gain_process(self, Mch: int = 10, samples_per_block: int = 100, component: str = "NLoS", expected_value_ref: bool = True, uncorrelated_ref: bool = False, assumption_ref: bool = False, prop_link_idx: tuple[int, int] = (0, 0)) -> tuple[plt.Figure, plt.Axes]:
        r"""
        Plot the channel gain process of a single propagation link over time.

        The magnitude of the complex channel gain process is plotted in dB:

        .. math::
            |h(t)| \; [\text{dB}] = 20 \log_{10}(|h[m]|), \quad m = 0, 1, \ldots, M-1

        Parameters
        ----------
        Mch : int
            The number of channel realizations to plot. One channel realization corresponds to one block. The duration of a block is at most equal to the coherence time :math:`T_c`. Default is 15.
        samples_per_block : int
            The number of samples to plot per block. This is for visualization purposes, to get a smoother curve. Default is 100.
        component : str
            The component of the channel to plot. Choose between: 'LoS', 'NLoS' and 'LoS + NLoS'.
        
        expected_value_ref : bool
            Whether to plot the expected value of the magnitude as a reference line. Default is True.
        uncorrelated_ref : bool
            Whether to plot an uncorrelated channel gain process with the same average power and the same number of samples for comparison. Default is False.
        assumption_ref : bool
            Whether to plot the channel gain process used in the simulations as a reference. This is a channel gain process that is constant during one block period!
        
        prop_link_idx : tuple[int, int]
            The index of the propagation link to plot, in the form (k * Nr + nr, nt). Default is (0, 0).

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the plot.
        ax : matplotlib.axes.Axes
            The axes object containing the plot.

        Notes
        -----
        The time axis is expressed in units of the coherence time :math:`T_c`.
        The reference line at 0 dB corresponds to unit amplitude (average power = 1).

        This function resets the channel model before plotting, so any previously generated channel state is discarded.
        """

        assert prop_link_idx[0] < self.Nt and prop_link_idx[1] < self.K * self.Nr, ( f"`prop_link_idx` must be smaller than (Nt, K * Nr) = ({self.Nt}, {self.K * self.Nr}). However, `prop_link_idx` = {prop_link_idx} was given." )
        assert Mch*samples_per_block <= self.Mch_max, ( f"`Mch * samples_per_block` must be smaller than or equal to the maximum number of channel realizations per simulation. (`Mch_max` = {self.Mch_max}). However, `Mch` = {Mch} and `samples_per_block` = {samples_per_block} were given." )

        # 0. Reset the channel model.

        self.T_block /= samples_per_block
        self.CSI_delay *= samples_per_block
        self.reset()

        # 1. Extract the channel gain process and compute the magnitude.
        
        if component == "LoS":
            h = np.exp(1j * self._theta[prop_link_idx[0] // self.Nr]) * np.ones(Mch*samples_per_block)

        elif component == "NLoS":
            h = self._H_NLoS[prop_link_idx[0], prop_link_idx[1], :]

        elif component == "LoS + NLoS":
            h = np.exp(1j * self._theta[prop_link_idx[0] // self.Nr]) * (np.sqrt(self.K_rice / (self.K_rice + 1)) + np.sqrt(1 / (self.K_rice + 1)) * self._H_NLoS[prop_link_idx[0], prop_link_idx[1], :])

        else:
            raise ValueError(f"Unknown component '{component}' for plotting the channel gain process. Choose between: 'LoS', 'NLoS' and 'LoS + NLoS'.")

        h_magnitude = np.abs(h[:Mch*samples_per_block])
        t_norm = np.arange(Mch*samples_per_block) * self.T_block

        # 2. Plot.

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(t_norm, h_magnitude, color="tab:blue", linewidth=2, label="Clark's Fading Model")

        # plot an uncorrelated channel gain process with the same average power and the same number of samples for comparison if asked.
        if uncorrelated_ref:
            
            if component == "LoS": 
                h_uncorrelated = np.exp(1j * self._theta[prop_link_idx[0] // self.Nr]) * np.ones(Mch*samples_per_block)
            elif component == "NLoS": 
                h_uncorrelated = (1 / np.sqrt(2)) * (np.random.randn(Mch*samples_per_block) + 1j * np.random.randn(Mch*samples_per_block))
            elif component == "LoS + NLoS":
                h_uncorrelated = np.exp(1j * self._theta[prop_link_idx[0] // self.Nr]) * (np.sqrt(self.K_rice / (self.K_rice + 1)) + np.sqrt(1 / (self.K_rice + 1)) * (1 / np.sqrt(2)) * (np.random.randn(Mch*samples_per_block) + 1j * np.random.randn(Mch*samples_per_block)))

            h_uncorrelated_magnitude = np.abs(h_uncorrelated)
            ax.plot(t_norm, h_uncorrelated_magnitude, color="tab:red", linestyle="--", linewidth=1, label="Uncorrelated Fading")
        
        # plot the channel gain process used in the simulations as a reference if asked. This is a channel gain process that is constant during one block period.
        if assumption_ref:
            
            if component == "LoS": 
                h_assumption = np.exp(1j * self._theta[prop_link_idx[0] // self.Nr]) * np.ones(Mch)
            elif component == "NLoS": 
                h_assumption = self._H_NLoS[prop_link_idx[0], prop_link_idx[1], np.arange(0, Mch*samples_per_block, samples_per_block)]
            elif component == "LoS + NLoS":
                h_assumption = np.exp(1j * self._theta[prop_link_idx[0] // self.Nr]) * (np.sqrt(self.K_rice / (self.K_rice + 1)) + np.sqrt(1 / (self.K_rice + 1)) * self._H_NLoS[prop_link_idx[0], prop_link_idx[1], np.arange(0, Mch*samples_per_block, samples_per_block)])
            
            h_assumption_magnitude = np.abs(h_assumption)
            ax.step(t_norm[::samples_per_block], h_assumption_magnitude, where="post", color="slategray", linestyle="--", linewidth=1.5, label="Block Fading Assumption")

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
            n = int(round(x))
            if   n == 0:  return r"$0$"
            elif n == 1:  return r"$T_c$"
            elif n == -1: return r"$-T_c$"
            else:         return rf"${n} \, T_c$"
        ax.xaxis.set_major_locator(MultipleLocator( max(1, int(np.round(t_norm[-1] / 10))) ))
        ax.xaxis.set_major_formatter(FuncFormatter(Tc_formatter))

        # plot settings.
        ax.set_xlabel(r"$t \; [s]$")
        ax.set_ylabel(r"$\left|h(t)\right|$")
        ax.set_title(f"Channel gain process of a propagation link\n{component} component")
        ax.legend()
        ax.grid(True, which="both", linestyle="--")
        plt.tight_layout()

        # save the plot.
        plot_filename = f"channel gain process {component} ({Mch} blocks, {samples_per_block} samples per block)" + (" (uncorrelated ref)" if uncorrelated_ref else "") + (" (block fading assumption ref)" if assumption_ref else "") + ".png"
        plot_dir = Path(__file__).parents[2] / "report" / "analytical_results" / "channel_statistics" / "plots" / "satellite channel gain process"
        plot_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_dir / plot_filename, dpi=300, bbox_inches="tight")
        
        # restore the original block period and CSI delay.
        self.T_block *= samples_per_block
        self.CSI_delay //= samples_per_block

        return fig, ax
    
    def plot_autocorrelation(self, max_lag: int = 20, num_samples: int = 1, component: str = "NLoS"):
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
        The plot is normalized in the sense that the lag axis is expressed in units of the coherence time :math:`T_c`.

        This function is intended to be used for validating the generated NLoS process! It will reset the channel model to ensure we start at time instant zero and that all pre-generated NLoS components are deleted. Then, it will generate num_samples sequences of the NLoS process using the specified method, and compute the averaged empirical autocorrelation for both positive and negative lags. Finally, it will compute the analytical autocorrelation and plot both the empirical and analytical autocorrelation functions.
        """
 
        assert num_samples <= self.K * self.Nr * self.Nt, f"num_samples should be smaller than or equal to K * Nr * Nt = {self.K * self.Nr * self.Nt}, because the average will be taken accross at most one simulation. However, {num_samples} num_samples were given."
        
        # 0. Reset the channel model.
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


        # Compute the empirical autocorrelation for each realization. ( Normalized lag axis: tau_norm[k] = tau[k] / Tc = (k * T_sample) / Tc = (k * self.T_block * Tc) / Tc = k * self.T_block )
        
        tau_sim_norm = np.arange(-max_lag, max_lag + 1) * self.T_block
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

        # 2. Compute the analytical autocorrelation in normalized units ( Normalized lag axis: 2pi * fD * tau[k] = 2pi * (0.242 / Tc) * tau[k] = 2pi * (0.242) * tau_norm[k] )
        tau_analytical_norm = np.linspace(tau_sim_norm[0], tau_sim_norm[-1], 10_000)
        R_analytical = j0(2 * np.pi * 0.242097596 * tau_analytical_norm)

        # 3. Plot.
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(tau_sim_norm, np.real(R_empirical), color="tab:blue", linewidth=2.5, label=r"$R_h(\tau)$ (Simulated)")
        ax.plot(tau_analytical_norm, R_analytical, color="black", linestyle="--", linewidth=1.5, label=r"$R_h(\tau) = J_0(2\pi f_D \tau)$ (Analytical)")

        #ax.axvline(1.0, color="tab:green", linestyle="-", lw=1)
        #ax.axvline(-1.0, color="tab:green", linestyle="-", lw=1)

        # x-axis ticks and labels.
        def Tc_formatter(x, pos):
            n = int(round(x))
            if   n == 0:  return r"$0$"
            elif n == 1:  return r"$T_c$"
            elif n == -1: return r"$-T_c$"
            else:         return rf"${n} \, T_c$"
        ax.xaxis.set_major_locator(MultipleLocator(max(1, int(np.round(tau_sim_norm[-1] / 5)))))
        ax.xaxis.set_major_formatter(FuncFormatter(Tc_formatter))

        # y-axis ticks and labels.
        ax.yaxis.set_major_locator(MultipleLocator(0.25))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y:.2f}"))
    

        ax.set_xlabel(r"$\tau \; [s]$")
        ax.set_ylabel(r"$R_h(\tau)$")
        ax.set_title(f"Autocorrelation of a {component} process" if num_samples == 1 else f"Autocorrelation of a {component} process\n(averaged over {num_samples} realizations)" )
        ax.legend()
        ax.grid(True, which="both", linestyle="--")
        plt.tight_layout()

        plot_Rh_filename = f"Rh {component} process ({max_lag} lags, {num_samples} samples).png"
        plot_Rh_dir = Path(__file__).parents[2] / "report" / "analytical_results" / "channel_statistics" / "plots" / "satellite channel autocorrelation"
        plot_Rh_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_Rh_dir / plot_Rh_filename, dpi=300, bbox_inches="tight")
        return fig, ax

    def plot_NLoS_PSD(self, num_samples: int = 1) -> None:
        r"""
        Plot the power spectral density (PSD) of the generated NLoS process and compare it to the analytical expression.

        The simulated PSD is calculated by averaging the periodograms of `num_samples` independent realizations (Bartlett's method):
        
        .. math::

            \hat{S}_h(f) = \frac{1}{M} \sum_{m=1}^{M} T_s \cdot \left| H_{m}(f) \right|^2

        where M is the number of process realizations (`num_samples`).
        
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
        The plot is normalized in the sense that the frequency axis is expressed in units of the maximum Doppler frequency :math:`f_D`.

        This function is intended to be used for validating the generated NLoS process! It will reset the channel model to ensure we start at time instant zero and that all pre-generated NLoS components are deleted. Then, it will generate num_samples sequences of the NLoS process using the specified method, compute the periodogram for each, and average them (Bartlett's method). Finally, it will compute the analytical PSD and plot both the empirical and analytical PSDs.
        """
 
        assert num_samples <= self.K * self.Nr * self.Nt, (f"num_samples should be smaller than or equal to K * Nr * Nt = {self.K * self.Nr * self.Nt}, because the average cannot be taken across different simulations. However, num_samples={num_samples} was given.")
        
        # 0. Reset the channel model.
        self.reset()

        # 1. Compute the PSD (Bartlett's method).
        H_NLoS_flat = self._H_NLoS.reshape(self.K*self.Nr*self.Nt, self.Mch_max)

        # Compute the periodogram for each realization.
        S_sim = np.zeros(self.Mch_max, dtype=float)
        for sample_num in range(num_samples):
            h = H_NLoS_flat[sample_num, :]
            S_sim_realization = (self.T_block*(0.242097596) / self.Mch_max) * np.abs(np.fft.fft(h))**2
            S_sim += S_sim_realization
        S_sim /= num_samples

        # Normalized frequency axis.
        f_sim_norm = np.fft.fftshift(np.fft.fftfreq(self.Mch_max, d=self.T_block*(0.242097596)))
        S_sim = np.fft.fftshift(S_sim)

        # 2. Compute the analytical PSD in normalized units.
        f_analytical_norm = np.linspace(-0.999, 0.999, 10_000)
        S_analytical_norm = 1.0 / (np.pi * np.sqrt(1 - f_analytical_norm**2))

        # 3. Plot.
        fig, ax = plt.subplots(figsize=(10, 4))
        
        ax.plot(f_sim_norm, S_sim, color="tab:blue", linewidth=2.5, label=r"$S_h(f)$ (Simulated)")
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
        ax.set_title("PSD of the NLoS process" if num_samples == 1 else f"PSD of the NLoS process\n(averaged over {num_samples} realizations)")
        ax.legend()
        ax.grid(True, which="both", linestyle="--")
        plt.tight_layout()

        plot_psd_filename = f"PSD NLoS process ({num_samples} samples).png"
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


# TESTING AND VALIDATION

if __name__ == "__main__":

    channel_model = RiceanFadingChannelModel(Nt=16, Nr=2, K=4, K_rice=10**(5/10), Trtt_2_Tc=0.5)
    
    # channel_model.plot_autocorrelation(max_lag=15, num_samples=1)
    # channel_model.plot_autocorrelation(max_lag=15, num_samples=128)
    # channel_model.plot_autocorrelation(max_lag=100, num_samples=1)
    # channel_model.plot_autocorrelation(max_lag=100, num_samples=64)
    # channel_model.plot_autocorrelation(max_lag=100, num_samples=128)

    # channel_model.plot_autocorrelation(max_lag=15, num_samples=1, component="LoS + NLoS")
    # channel_model.plot_autocorrelation(max_lag=15, num_samples=128, component="LoS + NLoS")
    # channel_model.plot_autocorrelation(max_lag=100, num_samples=1, component="LoS + NLoS")
    # channel_model.plot_autocorrelation(max_lag=100, num_samples=128, component="LoS + NLoS")

    # channel_model.plot_NLoS_PSD(num_samples=1)
    # channel_model.plot_NLoS_PSD(num_samples=64)
    # channel_model.plot_NLoS_PSD(num_samples=128)

    # channel_model.plot_channel_gain_process(Mch=10, samples_per_block=20, component="LoS + NLoS", uncorrelated_ref=True)
    # channel_model.plot_channel_gain_process(Mch=10, samples_per_block=20, component="LoS + NLoS", assumption_ref=True)
    # channel_model.plot_channel_gain_process(Mch=100, samples_per_block=20, component="LoS + NLoS")