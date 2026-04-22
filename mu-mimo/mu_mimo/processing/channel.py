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

        # The current time index m.
        self._m = 0
    
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
    
    def reset(self, Mch_max: int, Msv: int) -> None:
        """
        Reset the channel model state to its initial state.

        Parameters
        ----------
        Mch_max : int
            In case of an uncorrelated fading channel (:math:`T_c = 0`), the maximum number of channel realizations per SNR value.\\
            In case of a correlated fading channel (:math:`T_c > 0`), the (maximum) number of coherence periods of the channel.
        Msv : int
            In case of an uncorrelated fading channel (:math:`T_c = 0`), the number of symbol vector transmissions for each channel realization.\\
            In case of a correlated fading channel (:math:`T_c > 0`), the Coherence Time To Symbol Period Ratio (:math:`\frac{T_c}{T{\text{symbol}}}`).
            In other words, the number of symbol vectors per block when using full block-level precoding (:math:`T_c = T{\text{block}}`).
        
        Returns
        -------
        Mblocks_max : int
            The maximum number of blocks to transmit during the simulation for this SNR value.
        Msvpb : int
            The number of symbol vectors per block.
        """
        self._H = None
        self._m = 0
        return Mch_max, Msv
    
    @abstractmethod
    def proceed(self) -> ComplexArray:
        """
        Update the channel state (time index and current channel matrix).

        Returns
        -------
        H : ComplexArray, shape (K * Nr, Nt)
            The new, current channel matrix for the consecutive symbol block.
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_channel(self) -> ComplexArray:
        """
        Retrieve the channel matrix.
        
        If there is a delay on the available CSI at the BS, it will return the most up-to-date channel estimate (matrix) available at the BS.

        Returns
        -------
        H : ComplexArray, shape (K * Nr, Nt)
            The channel matrix corresponding to the most up-to-date CSI available at the BS.
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
        
        # case 1: uncorrelated fading channel / symbol-level precoding with instantaneous CSI.
        if len(self._H.shape) == 2:
            y = self._H @ x
        
        # case 2.1: correlated fading channel, symbol-level precoding with instantaneous CSI.
        elif (len(self._H.shape) == 3 and self._H.shape[0] == 1):
            y = self._H[0] @ x
        
        # case 2: correlated fading channel, block-level precoding.
        elif len(self._H.shape) == 3:
            
            assert self._H.shape[0] == x.shape[1], f"The number of channel realizations within one block must be equal to the number of symbol vector transmissions. Please take a closer look at these parameters.\nNumber of channel realizations within one block: {self._H.shape[0]}\nNumber of symbol vector transmissions: {x.shape[1]}"
            
            y = np.empty((self.K*self.Nr, x.shape[1]), dtype=complex)
            for m in range(x.shape[1]):
                y[:, m] = self._H[m] @ x[:, m]

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

    def proceed(self) -> ComplexArray:
        
        # Update the time index.
        self._m += 1

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
        self._m += 1

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
    The ricean time-correlated fading channel.

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

    def __init__(self, Nt: int, Nr: int, K: int, K_rice: float, Trtt_2_Tc: float, Rh_Tc: float = 0.5):
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
            The ratio between the round trip time :math:`T_{rtt}` and the coherence time :math:`T_c`.\\
            It quantifies the block period and the delay on the CSI.
        Rh_Tc : float
            The channel correlation at a lag equal to the coherence time :math:`T_c`.\\
            It quantifies the product between the maximum Doppler frequency and the channel coherence time :math:`f_D \cdot T_c`.
        """
        
        # Initialize the base class.
        super().__init__(Nt, Nr, K)

        # Validate the parameters.
        assert K_rice >= 0, f"The Rice factor K_rice must be non-negative.\nCurrent value: {K_rice}"
        assert Trtt_2_Tc >= 0, f"The round trip time to coherence time ratio Trtt_2_Tc must be non-negative.\nCurrent value: {Trtt_2_Tc}"
        assert 0.5 <= Rh_Tc <= 1, f"The minimum channel correlation at a lag equal to the coherence time is 0.5, the maximum is 1 (0.5 <= R_h(Tc) <= 1).\nCurrent value: {Rh_Tc}"
        
        # Store the Rice factor, the the round trip time to the coherence time ratio and the channel correlation at a lag equal to the coherence time.
        self.K_rice = K_rice
        self.Trtt_2_Tc = Trtt_2_Tc
        self.Rh_Tc = Rh_Tc

        # Initialize the channel time period parameters.
        self.Tsymbol_2_Tc : float | None = None
        self.Tblock_2_Tsymbol : int | None = None
        self.CSI_fb_delay : int | None = None
        self.fD_times_Tc : float | None = None

        # Initialize the channel state.
        self.SYMBOL_LEVEL : bool | None = None
        self._theta : RealArray | None = None
        self._H_NLoS : ComplexArray | None = None
        self._H_NLoS_CSI : ComplexArray | None = None

    def __str__(self) -> str:
        """ Return a string representation of the channel model. """
        return f"Ricean Time-Correlated Fading Channel (K_rice = {self.K_rice}, Trtt_2_Tc = {np.round(self.Trtt_2_Tc, 2)}, Rh(Tc) = {np.round(self.Rh_Tc, 2)})"
    
    def __eq__(self, other):
        
        if not isinstance(other, RiceanFadingChannelModel):
            return NotImplemented
        
        return (
            self.Nt == other.Nt and
            self.Nr == other.Nr and
            self.K == other.K and
            self.K_rice == other.K_rice and
            self.Trtt_2_Tc == other.Trtt_2_Tc and
            self.Rh_Tc == other.Rh_Tc
        )
    
    def reset(self, N_Tc_max: int, Tc_2_Tsymbol: int) -> None:
        r"""
        Reset the channel model to its initial state.\\
        Then, compute the channel time period parameters.\\
        Finally, generate the channel matrices for the upcoming simulation.

        Parameters
        ----------
        N_Tc_max : int
            The maximum number of coherence periods of the channel to simulate.
        Tc_2_Tsymbol : int
            The Coherence Time To Symbol Period Ratio.
            In other words, the number of symbol vectors that can be transmitted during one coherence period.
        
        Returns
        -------
        M_bl_max : int
            The maximum number of blocks to transmit during the simulation for this SNR value.
        M_svpb : int
            The number of symbol vectors per block.
        
        Notes
        -----
        - Computation of block period to coherence time ratio :math:`\frac{T_{\text{block}}}{T_c}`.
        .. math::
            T_{\text{block}} = T_c - T_{\text{RTT}} \quad \text{if } T_{\text{RTT}} \leq T_c \text{  (semi block-level precoding)}\\
            T_{\text{block}} = T_c \quad \text{if } T_{\text{RTT}} > T_c \text{  (full block-level precoding)}
        
        """
        
        # 1. Reset the channel model to its initial state.
        super().reset(N_Tc_max, Tc_2_Tsymbol)
        self._theta = None
        self._H_NLoS = None

        # 2. Compute the channel time period parameters.
        num_channel_realizations, M_bl_max, M_svpb = self._compute_parameters(N_Tc_max, Tc_2_Tsymbol)
 
        # 3. Generate the channel matrices for the upcoming simulation.
        self._generate(num_channel_realizations)

        return M_bl_max, M_svpb

    def proceed(self) -> ComplexArray:
        
        # Update the time index.
        self._m += self.Tblock_2_Tsymbol

        # Validate the time index. It cannot exceed the maximum number of channel realizations.
        if self._m >= self._H_NLoS.shape[0]:
            raise IndexError(f"The time index `m` cannot exceed the maximum number of channel realizations. Please take a closer look at these parameters.\nCurrent time index `m`: {self._m}\nMaximum number of channel realizations: {self._H_NLoS.shape[0]}")

        # Update the current channel matrix.
        theta = np.repeat(self._theta, self.Nr*self.Nt).reshape(self.K*self.Nr, self.Nt)
        H_NLoS = self._H_NLoS[self._m : self._m + self.Tblock_2_Tsymbol]

        H = np.exp(1j * theta) * (np.sqrt(self.K_rice / (self.K_rice + 1)) + np.sqrt(1 / (self.K_rice + 1)) * H_NLoS)
        self._H = H
        
        return H

    def get_channel(self):

        # Get the LoS and NLoS component.
        theta = np.repeat(self._theta, self.Nr*self.Nt).reshape(self.K*self.Nr, self.Nt)
        if not self.SYMBOL_LEVEL:
            H_NLoS = self._H_NLoS[self._m - self.CSI_fb_delay]
        else:
            H_NLoS = self._H_NLoS_CSI[self._m]

        # Return the channel corresponding to the most up-to-date CSI available at the BS.
        H_CSI = np.exp(1j * theta) * (np.sqrt(self.K_rice / (self.K_rice + 1)) + np.sqrt(1 / (self.K_rice + 1)) * H_NLoS)
        
        return H_CSI
    

    def _compute_parameters(self, N_Tc_max: int, Tc_2_Tsymbol: int):

        # block-level precoding.
        if self.Rh_Tc < 1.0:

            # [Tblock_2_Tsymbol] Compute and store the number of symbol vectors that can be transmitted during one block period.
            Tblock_2_Tc = (1.0 - self.Trtt_2_Tc) if self.Trtt_2_Tc <= 1 else 1.0
            M_sv = int(max(np.round(Tblock_2_Tc * Tc_2_Tsymbol), 1))
            self.Tblock_2_Tsymbol = M_sv

            # [CSI_fb_delay] Compute and store the delay on the CSI feedback, in number of symbol vector transmissions.
            CSI_fb_delay = int(max(np.ceil(self.Trtt_2_Tc * Tc_2_Tsymbol), 0))
            self.CSI_fb_delay = CSI_fb_delay
            self._m += self.CSI_fb_delay

            # [Tsymbol_2_Tc] Compute and store the symbol period to coherence time ratio.
            self.Tsymbol_2_Tc = (1 / Tc_2_Tsymbol)

            # [fD_times_Tc] Compute and store the product between the maximum Doppler frequency and the coherence time.
            self.fD_times_Tc = (1 / (2*np.pi)) * brentq(lambda z: j0(z) - self.Rh_Tc, 0, 2.5)
            
            # Compute the number of channel samples to generate and the maximum number of blocks to transmit during the simulation.
            num_channel_realizations = N_Tc_max * Tc_2_Tsymbol
            M_blch_max = int(np.floor((N_Tc_max * Tc_2_Tsymbol) / M_sv))

            self.SYMBOL_LEVEL = False
        

        # symbol-level precoding.
        else:

            # [Tblock_2_Tsymbol] Compute and store the number of symbol vectors that can be transmitted during one block period.
            self.Tblock_2_Tsymbol = 1
            
            # [CSI_fb_delay] Compute and store the delay on the CSI feedback, in number of symbol vector transmissions.
            CSI_fb_delay = int(max(np.ceil(self.Trtt_2_Tc * Tc_2_Tsymbol), 0))
            self.CSI_fb_delay = CSI_fb_delay

            # [Tsymbol_2_Tc] Compute and store the symbol period to coherence time ratio.
            self.Tsymbol_2_Tc = 1 / Tc_2_Tsymbol

            # [fD_times_Tc] Compute and store the product between the maximum Doppler frequency and the coherence time.
            RH_TC = 0.5
            self.fD_times_Tc = (1 / (2*np.pi)) * brentq(lambda z: j0(z) - RH_TC, 0, 2.5)
            
            # Compute the number of channel samples to generate, number of channel realizations for the simulation and the number of symbol vectors to transmit per channel realization.
            num_channel_realizations = N_Tc_max
            M_blch_max = N_Tc_max
            M_sv = Tc_2_Tsymbol

            self.SYMBOL_LEVEL = True


        return num_channel_realizations, M_blch_max, M_sv
    

    def _generate(self, num_channel_realizations: int) -> tuple[RealArray, ComplexArray]:
        """
        Generate the channel matrices for all users and all future time instances.
        Store them in the channel state.

        Parameters
        ----------
        num_channel_realizations : int
            The total number of channel realizations to generate.
            This is equal to the total number of symbol vectors that will be transmitted during the simulation.

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
        H_NLoS = self._generate_NLoS(num_channel_realizations, method="Cholesky-decomposition method")
        if not self.SYMBOL_LEVEL:
            self._H_NLoS = H_NLoS
        else:
            self._H_NLoS = H_NLoS["actual"]
            self._H_NLoS_CSI = H_NLoS["CSI"]

        return theta, H_NLoS
    
    def _generate_NLoS(self, num_channel_realizations: int, method: str) -> ComplexArray:
        """
        Generate the NLoS component for all users and all time instants, using the specified method.

        Parameters
        ----------
        num_channel_realizations : int
            The total number of channel realizations to generate.
            This is equal to the total number of symbol vectors that will be transmitted during the simulation.
        method : str
            The method to use for generating the NLoS component. Choose between: 'Cholesky-decomposition method', 'spectral method', 'FIR filter method'.

        Returns
        -------
        H_NLoS : ComplexArray, shape (K * Nr, Nt, Mch_max)
            The generated NLoS component of the channel, for all time instants.
        """
        
        if not self.SYMBOL_LEVEL:

            if method == "Cholesky-decomposition method":
                H_NLoS = self._generate_NLoS_cholesky(num_channel_realizations)
            
            elif method == "spectral method":
                H_NLoS = self._generate_NLoS_spectral(num_channel_realizations)
            
            elif method == "FIR filter method":
                H_NLoS = self._generate_NLoS_FIR_filter(num_channel_realizations)
            
            else:
                raise ValueError(f"Unknown method '{method}' for generating the Ricean channel matrix. Valid options are: 'Cholesky-decomposition method', 'spectral method', 'FIR filter method'.")
        
        else:
            
            H_NLoS = self._generate_SL_NLoS(num_channel_realizations)

        return H_NLoS

    def _generate_NLoS_cholesky(self, num_channel_realizations: int) -> ComplexArray:
        r"""
        Generate the NLoS component for user k using the Cholesky decomposition method for generating a Gaussian process with a specified auto-correlation function.

        The Cholesky decomposition method for generating a zero-mean unit-variance complex Gaussian process :math:`\mathbf{h}` of length :math:`N` with a specified autocorrelation function :math:`R_h(\tau)`:
        
        - **Step 1:** Build the :math:`N \times N` covariance matrix :math:`\mathbf{C}` with entries :math:`C_{i,j} = R_h((i-j) \, T_{\text{sample}})`.
        - **Step 2:** Compute the Cholesky decomposition of the covariance matrix :math:`\mathbf{C} = \mathbf{L}\mathbf{L}^H`, where :math:`\mathbf{L}` is a lower triangular matrix.
        - **Step 3:** Generate a column vector :math:`\mathbf{w}` of :math:`N` i.i.d. white complex Gaussian random variables with zero mean and unit variance.
        - **Step 4:** Find the desired Gaussian process as :math:`\mathbf{h} = \mathbf{L} \, \mathbf{w}`.

        Parameters
        ----------
        num_channel_realizations : int
            The total number of channel realizations to generate.
            This is equal to the total number of symbol vectors that will be transmitted during the simulation.
        
        Returns
        -------
        H_NLoS : ComplexArray, shape (num_channel_realizations, K * Nr, Nt)
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
            &= J_0(2\pi \, (0.242) \frac{1}{T_c} \cdot (i-j) \, T_{\text{symbol}})) \\
            &= J_0(2\pi \, (0.242) \, (i-j) \cdot \frac{T_{\text{symbol}}}{T_c}) \\
            
        where we used
        
        .. math::
            R_h(\tau) \geq \frac{1}{2} \iff \left| 2\pi f_D \tau \right| \leq 1.521 \implies \tau_c \approx (0.242) \cdot \frac{1}{f_D}
        
        """

        # STEP 1.
        i = np.arange(num_channel_realizations).reshape(-1,1)
        j = np.arange(num_channel_realizations)
        C = j0(2*np.pi * self.fD_times_Tc * (i-j) * self.Tsymbol_2_Tc)
        C += (10e-10)*np.eye(num_channel_realizations)

        # STEP 2.
        L = np.linalg.cholesky(C)

        # STEP 3.
        w = (1 / np.sqrt(2)) * (np.random.randn(num_channel_realizations, self.K*self.Nr*self.Nt) + 1j * np.random.randn(num_channel_realizations, self.K*self.Nr*self.Nt))

        # STEP 4.
        H_NLoS = (L @ w).reshape(num_channel_realizations, self.K*self.Nr, self.Nt)

        return H_NLoS

    def _generate_NLoS_spectral(self, num_channel_realizations: int) -> ComplexArray:
        """
        Generate the NLoS component for all users using the spectral method for generating a Gaussian process with a specified power spectral density (PSD) function.

        Parameters
        ----------
        num_channel_realizations : int
            The total number of channel realizations to generate.

        Returns
        -------
        H_NLoS : ComplexArray, shape (K * Nr, Nt, num_channel_realizations)
            The generated NLoS component for all propagation links and all time instants.
        
        Note
        ----
        The spectral method is not implemented yet. It will raise a `NotImplementedError` when called.
        Use the Cholesky decomposition method instead.
        """
        raise NotImplementedError("The spectral method for generating the NLoS component is not implemented yet.")

    def _generate_NLoS_FIR_filter(self, num_channel_realizations: int) -> ComplexArray:
        """
        Generate the NLoS component for all users using the FIR filter method for generating a Gaussian process with a specified power spectral density (PSD) function.
        
        Parameters
        ----------
        num_channel_realizations : int
            The total number of channel realizations to generate.

        Returns
        -------
        H_NLoS : ComplexArray, shape (K * Nr, Nt, num_channel_realizations)
            The generated NLoS component for all propagation links and all time instants.
        
        Note
        ----
        The FIR filter method is not implemented yet. It will raise a `NotImplementedError` when called.
        Use the Cholesky decomposition method instead.
        """
        raise NotImplementedError("The FIR filter method for generating the NLoS component is not implemented yet.")

    def _generate_SL_NLoS(self, num_channel_realizations: int) -> dict[str, ComplexArray]:
        """
        Generate the NLoS component for all channel realizations.

        Both the actual channel and the corresponding CSI available at the BS will be returned.
        """

        H_NLoS = np.empty((num_channel_realizations, self.K*self.Nr, self.Nt), dtype=complex)
        H_NLoS_CSI = np.empty((num_channel_realizations, self.K*self.Nr, self.Nt), dtype=complex)

        for m_ch in range(num_channel_realizations):

            H_NLoS_process = self._generate_NLoS_cholesky(self.CSI_fb_delay + 1)
            H_NLoS[m_ch] = H_NLoS_process[-1]
            H_NLoS_CSI[m_ch] = H_NLoS_process[0]
        
        return {"actual": H_NLoS, "CSI": H_NLoS_CSI}


    def plot_channel_gain_process(self, N_Tc: int = 5, component: str = "NLoS", expected_value_ref: bool = True, uncorrelated_ref: bool = False, assumption_ref: bool = False, prop_link_idx: tuple[int, int] = (0, 0)) -> tuple[plt.Figure, plt.Axes]:
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
        """

        # Validation.
        assert self._H_NLoS is not None, "The channel model must be reset before plotting the channel gain process. Please call the `reset` method of the channel model before plotting."
        
        Msvptc = int(round(1 / self.Tsymbol_2_Tc))
        assert N_Tc * Msvptc <= self._H_NLoS.shape[0], ( f"`N_Tc` must be smaller than or equal to the maximum number of coherence periods per simulation. However, `N_Tc` = {N_Tc} and `N_Tc_max` = {self._H_NLoS.shape[0] * self.Tsymbol_2_Tc} were given." )
        assert prop_link_idx[0] < self.K * self.Nr and prop_link_idx[1] < self.Nt, ( f"`prop_link_idx` must be an index element of the channel matrix (shape: (Nt, K * Nr) = ({self.Nt}, {self.K * self.Nr})). However, `prop_link_idx` = {prop_link_idx} was given." )
        
        # 1. Extract the channel gain process and compute the magnitude.
        
        if component == "LoS":
            h = np.exp(1j * self._theta[prop_link_idx[0] // self.Nr]) * np.ones(N_Tc * Msvptc)
        elif component == "NLoS":
            h = self._H_NLoS[0 : N_Tc * Msvptc, prop_link_idx[0], prop_link_idx[1]]
        elif component == "LoS + NLoS":
            h = np.exp(1j * self._theta[prop_link_idx[0] // self.Nr]) * (np.sqrt(self.K_rice / (self.K_rice + 1)) + np.sqrt(1 / (self.K_rice + 1)) * self._H_NLoS[0 : N_Tc * Msvptc, prop_link_idx[0], prop_link_idx[1]])
        else:
            raise ValueError(f"Unknown component '{component}' for plotting the channel gain process. Choose between: 'LoS', 'NLoS' and 'LoS + NLoS'.")

        h_magnitude = np.abs(h)
        t = np.arange(N_Tc * Msvptc)

        # 2. Plot.

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(t, h_magnitude, color="tab:blue", linewidth=2, label="Clark's Fading Model")

        # plot an uncorrelated channel gain process with the same average power and the same number of samples for comparison if asked.
        if uncorrelated_ref:
            
            if component == "LoS": 
                h_uncorrelated = np.exp(1j * self._theta[prop_link_idx[0] // self.Nr]) * np.ones(N_Tc * Msvptc)
            elif component == "NLoS": 
                h_uncorrelated = (1 / np.sqrt(2)) * (np.random.randn(N_Tc * Msvptc) + 1j * np.random.randn(N_Tc * Msvptc))
            elif component == "LoS + NLoS":
                h_uncorrelated = np.exp(1j * self._theta[prop_link_idx[0] // self.Nr]) * (np.sqrt(self.K_rice / (self.K_rice + 1)) + np.sqrt(1 / (self.K_rice + 1)) * (1 / np.sqrt(2)) * (np.random.randn(N_Tc * Msvptc) + 1j * np.random.randn(N_Tc * Msvptc)))

            h_uncorrelated_magnitude = np.abs(h_uncorrelated)
            ax.plot(t, h_uncorrelated_magnitude, color="tab:red", linestyle="--", linewidth=1, label="Uncorrelated Fading")
        
        # plot the channel gain process used in the simulations as a reference if asked. This is a channel gain process that is constant during one block period.
        if assumption_ref:
            
            if component == "LoS": 
                h_assumption = np.exp(1j * self._theta[prop_link_idx[0] // self.Nr]) * np.ones(N_Tc)
            elif component == "NLoS": 
                h_assumption = self._H_NLoS[np.arange(0, N_Tc * Msvptc, Msvptc), prop_link_idx[0], prop_link_idx[1]]
            elif component == "LoS + NLoS":
                h_assumption = np.exp(1j * self._theta[prop_link_idx[0] // self.Nr]) * (np.sqrt(self.K_rice / (self.K_rice + 1)) + np.sqrt(1 / (self.K_rice + 1)) * self._H_NLoS[np.arange(0, N_Tc * Msvptc, Msvptc), prop_link_idx[0], prop_link_idx[1]])
            
            h_assumption_magnitude = np.abs(h_assumption)
            ax.step(t[::Msvptc], h_assumption_magnitude, where="post", color="slategray", linestyle="--", linewidth=1.5, label="Block Fading Assumption")

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
            n = int(round(x / Msvptc))
            if   n == 0:  return r"$0$"
            elif n == 1:  return r"$T_c$"
            elif n == -1: return r"$-T_c$"
            else:         return rf"${n} \, T_c$"
        ax.xaxis.set_major_locator(MultipleLocator(Msvptc))
        ax.xaxis.set_major_formatter(FuncFormatter(Tc_formatter))

        # plot settings.
        ax.set_xlabel(r"$t \; [s]$")
        ax.set_ylabel(r"$\left|h(t)\right|$")
        ax.set_title(f"Channel gain process of a propagation link\n{component} component")
        ax.legend()
        ax.grid(True, which="both", linestyle="--")
        plt.tight_layout()

        # save the plot.
        plot_filename = f"channel gain process {component} ({N_Tc} coherence periods)" + (" (uncorrelated ref)" if uncorrelated_ref else "") + (" (block fading assumption ref)" if assumption_ref else "") + ".png"
        plot_dir = Path(__file__).parents[2] / "report" / "analytical_results" / "channel_statistics" / "plots" / "ricean time-correlated fading channel" / "channel gain process"
        plot_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_dir / plot_filename, dpi=300, bbox_inches="tight")
        
        return fig, ax
    
    def plot_autocorrelation(self, N_Tc: int = 10, component: str = "NLoS", prop_link_idx: tuple[int, int] = (0, 0)) -> tuple[plt.Figure, plt.Axes]:
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
        prop_link_idx : tuple[int, int]
            The index of the propagation link to plot, in the form (k * Nr + nr, nt). Default is (0, 0).

        
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the plot.
        ax : matplotlib.axes.Axes
            The axes object containing the plot.
        """
 
        # Validation.
        assert self._H_NLoS is not None, "The channel model must be reset before plotting the autocorrelation. Please call the `reset` method of the channel model before plotting."
        
        Msvptc = int(round(1 / self.Tsymbol_2_Tc))
        tau = np.arange(-N_Tc//2 * Msvptc, N_Tc//2 * Msvptc)
        assert N_Tc * Msvptc <= self._H_NLoS.shape[0], ( f"`N_Tc` must be smaller than or equal to the maximum number of coherence periods per simulation. However, `N_Tc` = {N_Tc} and `N_Tc_max` = {self._H_NLoS.shape[0] * self.Tsymbol_2_Tc} were given." )
        assert prop_link_idx[0] < self.K * self.Nr and prop_link_idx[1] < self.Nt, ( f"`prop_link_idx` must be an index element of the channel matrix (shape: (Nt, K * Nr) = ({self.Nt}, {self.K * self.Nr})). However, `prop_link_idx` = {prop_link_idx} was given." )
        
        # 1. Extract the channel gain process and compute empirical autocorrelation.
        
        if component == "LoS":
            h = np.exp(1j * self._theta[prop_link_idx[0] // self.Nr]) * np.ones(self._H_NLoS.shape[0])
        elif component == "NLoS":
            h = self._H_NLoS[:, prop_link_idx[0], prop_link_idx[1]]
        elif component == "LoS + NLoS":
            h = np.exp(1j * self._theta[prop_link_idx[0] // self.Nr]) * (np.sqrt(self.K_rice / (self.K_rice + 1)) + np.sqrt(1 / (self.K_rice + 1)) * self._H_NLoS[:, prop_link_idx[0], prop_link_idx[1]])
        else:
            raise ValueError(f"Unknown component '{component}' for plotting the channel gain process. Choose between: 'LoS', 'NLoS' and 'LoS + NLoS'.")

        Rh_simulation = (1 / len(h)) * np.correlate(h, h, mode='full')[(len(h)-1) - (N_Tc//2 * Msvptc) : (len(h)-1) + (N_Tc//2 * Msvptc)]
        Rh_simulation /= Rh_simulation[N_Tc//2 * Msvptc]


        # 2. Compute the analytical autocorrelation.
        Rh_analytical = j0(2 * np.pi * self.fD_times_Tc * tau*self.Tsymbol_2_Tc)

        # 3. Plot.
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(tau, np.real(Rh_simulation), color="tab:blue", linewidth=2.5, label=r"$R_h(\tau)$ (Simulated)")
        ax.plot(tau, Rh_analytical, color="black", linestyle="--", linewidth=1.5, label=r"$R_h(\tau) = J_0(2\pi f_D \tau)$ (Analytical)")
        if component == "LoS + NLoS": ax.axhline(self.K_rice / (self.K_rice + 1), color="slategray", linestyle="-", linewidth=1.5, label=r"$\frac{K}{K+1}$")

        # x-axis ticks and labels.
        def Tc_formatter(x, pos):
            n = int(round(x / Msvptc))
            if   n == 0:  return r"$0$"
            elif n == 1:  return r"$T_c$"
            elif n == -1: return r"$-T_c$"
            else:         return rf"${n} \, T_c$"
        ax.xaxis.set_major_locator(MultipleLocator(((N_Tc * Msvptc) / 10)))
        ax.xaxis.set_major_formatter(FuncFormatter(Tc_formatter))

        # y-axis ticks and labels.
        ax.yaxis.set_major_locator(MultipleLocator(0.25))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y:.2f}"))
    

        ax.set_xlabel(r"$\tau \; [s]$")
        ax.set_ylabel(r"$R_h(\tau)$")
        ax.set_title(f"Autocorrelation of a {component} process")
        ax.legend()
        ax.grid(True, which="both", linestyle="--")
        plt.tight_layout()

        plot_Rh_filename = f"Rh {component} process ({N_Tc} coherence periods).png"
        plot_Rh_dir = Path(__file__).parents[2] / "report" / "analytical_results" / "channel_statistics" / "plots" / "ricean time-correlated fading channel" / "autocorrelation"
        plot_Rh_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_Rh_dir / plot_Rh_filename, dpi=300, bbox_inches="tight")
        return fig, ax

    def plot_NLoS_PSD(self, num_segments: int = 1, prop_link_idx: tuple[int, int] | None = (0, 0)) -> tuple[plt.Figure, plt.Axes]:
        r"""
        Plot the power spectral density (PSD) of the generated NLoS process and compare it to the analytical expression.

        The simulated PSD is calculated using Bartlett's method:
        
        .. math::

            \hat{S}_h(f_k) = \frac{1}{M} \sum_{m=0}^{M-1} \hat{S}_h^{(m)}(f_k), \quad \text{with } \hat{S}_h^{(m)}(f_k) = \frac{T_{\text{sym}}}{L} \left| \sum_{n=0}^{L-1} h[mL + n] \, e^{-j 2\pi k n / L} \right|^2 \quad \text{and} \quad f_k = \frac{k}{L \cdot T_{\text{sym}}}

        where :math:`L = \lfloor N / M \rfloor` is the segment length, :math:`M` is the number of segments (`num_segments`), and :math:`N` is the total number of channel realizations.
        
        The analytical PSD is given by Jake's Doppler Spectrum:
        
        .. math::

            S_h(f) = \frac{1}{\pi f_D \sqrt{1 - \left( \frac{f}{f_D} \right)^2}}, \quad |f| < f_D

        Parameters
        ----------
        num_segments : int
            The number of independent segments to average over when computing the simulated PSD using Bartlett's method. Default is 1 (no averaging).
        prop_link_idx : tuple[int, int] | None
            The index of the propagation link to plot, in the form (k * Nr + nr, nt).
            If this is None, the PSDs will be averaged over all propagation links.
        
        
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the plot.
        ax : matplotlib.axes.Axes
            The axes object containing the plot.
        """
 
        # Validation.
        assert self._H_NLoS is not None, "The channel model must be reset before plotting the PSD. Please call the `reset` method of the channel model before plotting."
        assert prop_link_idx is None or (prop_link_idx[0] < self.K * self.Nr and prop_link_idx[1] < self.Nt), ( f"`prop_link_idx` must be an index element of the channel matrix (shape: (Nt, K * Nr) = ({self.Nt}, {self.K * self.Nr})). However, `prop_link_idx` = {prop_link_idx} was given." )
        assert num_segments > 0, "`num_segments` must be a positive integer."

        # 1. Compute the PSD (Bartlett's method).
        length_segments = self._H_NLoS.shape[0] // num_segments
        
        if prop_link_idx is None: 
            h = self._H_NLoS[:num_segments * length_segments]
            h_segments = h.reshape(num_segments, length_segments, self.K*self.Nr, self.Nt)
        else: 
            h = self._H_NLoS[:num_segments * length_segments, prop_link_idx[0], prop_link_idx[1]]
            h_segments = h.reshape(num_segments, length_segments)
        
        H_fft = np.fft.fft(h_segments, axis=1)
        periodogram = (self.fD_times_Tc * self.Tsymbol_2_Tc / length_segments) * np.abs(H_fft)**2
        
        S_simulation = np.fft.fftshift(np.mean(periodogram, axis=(0 if prop_link_idx is not None else (0, 2, 3))))
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
        ax.set_title("PSD of a NLoS process" + ("\naveraged across all propagation links" if prop_link_idx is None else ""))
        ax.legend()
        ax.grid(True, which="both", linestyle="--")
        plt.tight_layout()

        plot_psd_filename = f"PSD NLoS process ({num_segments} segments)" + (" averaged" if prop_link_idx is None else "") + ".png"
        plot_psd_dir = Path(__file__).parents[2] / "report" / "analytical_results" / "channel_statistics" / "plots" / "ricean time-correlated fading channel" / "PSD"
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

    channel_model = RiceanFadingChannelModel(Nt=8, Nr=2, K=4, K_rice=10**(5/10), Trtt_2_Tc=0.5)
    # channel_model.reset(1000, 20)

    # channel_model.plot_channel_gain_process(N_Tc=5)
    # channel_model.plot_channel_gain_process(N_Tc=20)
    # channel_model.plot_channel_gain_process(N_Tc=5, uncorrelated_ref=True)
    # channel_model.plot_channel_gain_process(N_Tc=5, assumption_ref=True)
    
    # channel_model.plot_autocorrelation(component="NLoS")
    # channel_model.plot_autocorrelation(N_Tc=40, component="NLoS")
    # channel_model.plot_autocorrelation(component="LoS + NLoS")
    # channel_model.plot_autocorrelation(N_Tc=40, component="LoS + NLoS")

    # channel_model.plot_NLoS_PSD(num_segments=1, prop_link_idx=None)
    # channel_model.plot_NLoS_PSD(num_segments=10, prop_link_idx=None)
