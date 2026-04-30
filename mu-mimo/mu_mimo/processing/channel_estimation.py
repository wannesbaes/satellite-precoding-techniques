# mu-mimo/mu_mimo/processing/channel_estimation.py

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from scipy.linalg import toeplitz
from scipy.special import j0
from scipy.optimize import brentq

from ..types import ComplexArray, ChannelStateInformation, ReceivePilotMessage


# CHANNEL ESTIMATION

class ChannelEstimator(ABC):
    """
    The Channel Estimator Abstract Base Class (ABC).

    This class is responsible for the channel estimation in each user terminal (UT).
    """

    def __eq__(self, other: object) -> bool:
        
        if not isinstance(other, ChannelEstimator):
            return NotImplemented
        
        return self.__class__ == other.__class__
    
    @abstractmethod
    def estimate(self, rx_pilots_msg: ReceivePilotMessage) -> ChannelStateInformation:
        """
        Estimate the channel matrix based on the received pilot message.

        Parameters
        ----------
        rx_pilots_msg : ReceivePilotMessage
            The pilot message received by this UT.

        Returns
        -------
        csi_k : ChannelStateInformation
            The estimated channel state information for this UT.
        """
        raise NotImplementedError

class NeutralChannelEstimator(ChannelEstimator):
    """
    Neutral Channel Estimator.

    Acts as a neutral element for channel estimation.
    """

    def __init__(self):
        return

    def estimate(self, rx_pilots_msg: ReceivePilotMessage) -> ChannelStateInformation:
        csi_k = rx_pilots_msg.csi_k
        return csi_k


# CHANNEL PREDICTION

class ChannelPredictor(ABC):
    """
    The Channel Predictor Abstract Base Class (ABC).

    This class is responsible for the channel prediction in each user terminal (UT).
    """

    def __eq__(self, other: object) -> bool:
        
        if not isinstance(other, ChannelPredictor):
            return NotImplemented
        
        return self.__class__ == other.__class__

    @abstractmethod
    def predict(self, csi_k: ChannelStateInformation) -> ComplexArray:
        """
        Predict the most recent channel matrix for this UT based on the estimated CSI.

        Parameters
        ----------
        csi_k : ChannelStateInformation
            The estimated channel state information for this UT.

        Returns
        -------
        H_k_hathat : ComplexArray, shape (Nr, Nt)
            The predicted most up-to-date channel matrix for this UT.
        """
        raise NotImplementedError

class NeutralChannelPredictor(ChannelPredictor):
    """
    Neutral Channel Predictor.

    Acts as a neutral element for channel prediction.\\
    It simply returns the most recent available channel estimate.
    """

    def __init__(self):
        return
    
    def predict(self, csi_k: ChannelStateInformation) -> ComplexArray:
        H_k_hathat = csi_k.H_hat[-1]
        return H_k_hathat

class ARPredictor(ChannelPredictor):
    """
    Autoregressive (AR) Channel Predictor.

    This class implements an AR(p) predictor for channel state information (CSI) prediction.
    """

    def __init__(self, channel_model: str, channel_params: dict):
        
        # Compute the prediction horizon in samples based on the CSI feedback delay and time period between two CSI feedback messages.
        k = int(np.ceil(channel_params['Trtt_2_Tc'] / channel_params['Tpilot_2_Tc']))

        # Compute the order of the AR predictor based on the physical time window of the received CSI and time period between two CSI feedback messages.
        p = int((1 / channel_params['Tpilot_2_Tc']) * channel_params['Twindow_2_Tc']) - k

        # Compute the optimal AR predictor coefficients!
        self.a = self._compute_optimal_coefficients(p, k, channel_model, channel_params)
    
    def __eq__(self, other: object) -> bool:
        
        if not isinstance(other, ARPredictor):
            return NotImplemented
        
        return np.array_equal(self.a, other.a)

    def predict(self, csi_k: ChannelStateInformation) -> ComplexArray:
        H_k_hathat = np.sum(self.a[:, None, None] * csi_k.H_hat[::-1], axis=0)
        return H_k_hathat
    
    def _compute_optimal_coefficients(self, p: int, k: int, channel_model: str, channel_params: dict) -> np.ndarray:
        r"""
        Compute the optimal AR(p) predictor coefficients for k-step-ahead prediction.

        Solves the Wiener-Hopf equations using the analytical autocorrelation function:

        .. math::
            \mathbf{R}_p \, \mathbf{a}^{(k)} = \mathbf{r}^{(k)}

        where :math:`\mathbf{R}_p` is the :math:`p \times p` Toeplitz autocorrelation matrix with entries :math:`[\mathbf{R}_p]_{i,j} = R_h((i-j) \, T_{\text{sample}})`, and :math:`\mathbf{r}^{(k)}` is the cross-correlation vector with entries :math:`r_i^{(k)} = R_h((k + i - 1) \, T_{\text{sample}})`.

        The predicted channel k steps ahead is then

        .. math::
            \hat{h}[n+k] = \sum_{l=1}^{p} a_l^{(k)} \, h[n - l + 1] = (\mathbf{a}^{(k)})^T \, \mathbf{h}_{\text{past}}

        where :math:`\mathbf{h}_{\text{past}} = [h[n], h[n-1], \ldots, h[n-p+1]]^T`.

        Parameters
        ----------
        p : int
            Number of taps.
        k : int
            Prediction horizon in samples (CSI feedback delay).
        channel_model : str
            The channel model for which to compute the AR predictor coefficients.
        channel_params : dict
            The parameters of the channel model.

        Returns
        -------
        a : np.ndarray, shape (p,)
            The real-valued AR predictor coefficients.

        Notes
        -----
        The coefficients are real-valued because :math:`R_h(\tau) = J_0(2\pi f_D \tau)` is a symmetric real function.
        The prediction window covers a physical duration of :math:`p \cdot T_{\text{pilot}}` seconds.
        """
        
        if channel_params["Trtt_2_Tc"] == 0:
            a = np.concatenate( (np.ones(1), np.zeros(p-1)) )
        
        elif channel_model == "Ricean IID TC NLoS":
            a = self._compute_optimal_coefficients_ricean_iid_tc_nlos(p, k, channel_params['K_ricean'], channel_params['Tpilot_2_Tc'])
        elif channel_model == "Satellite Channel":
            a = self._compute_optimal_coefficients_satellite_channel(p, k, channel_params)
        else:
            raise ValueError(f"Unsupported channel model for AR prediction!\n Channel Model: {channel_model}")
        
        return a

    def _compute_optimal_coefficients_ricean_iid_tc_nlos(self, p: int, k: int, K_ricean: float, Tpilot_2_Tc: float) -> np.ndarray:
        
        # STEP 1:
        fD_times_Tc = (1 / (2*np.pi)) * brentq(lambda z: j0(z) - 0.5, 0, 2.5)
        R_p = toeplitz( (K_ricean / (K_ricean+1)) + (1 / (K_ricean+1)) * j0(2*np.pi * (fD_times_Tc * Tpilot_2_Tc) * np.arange(p))  )

        # STEP 2:
        r_k = (K_ricean / (K_ricean+1)) + (1 / (K_ricean+1)) * j0(2*np.pi * (fD_times_Tc * Tpilot_2_Tc) * (k + np.arange(p)))

        # STEP 3:
        a = np.linalg.solve(R_p, r_k)

        return a

    def _compute_optimal_coefficients_satellite_channel(self, p: int, k: int, channel_params: dict) -> np.ndarray:

        def _compute_fD_times_Tc(K_rician: float, theta: float | None) -> float:

            # Coherence time of the NLoS component only.
            fD_times_Tc__NLoS = (1 / (2*np.pi)) * brentq(lambda z: j0(z) - 0.5, 0, 2.5)

            if theta is None:
                return fD_times_Tc__NLoS

            # Define the function whose root we want to find.
            f = lambda x: (K_rician**2 + j0(x)**2 + 2 * K_rician * j0(x) * np.cos(x * np.sin(theta)) - ((K_rician + 1) / 2)**2)

            # Coarse scan to locate the first sign change (first crossing of 0.5).
            x_grid = np.linspace(1e-6, 20.0, 20_000)
            f_grid = f(x_grid)
            sign_changes = np.where(np.diff(np.sign(f_grid)))[0]

            if len(sign_changes) == 0:
                return fD_times_Tc__NLoS

            # Refine the root with brentq
            idx = sign_changes[0]
            x_c = brentq(f, x_grid[idx], x_grid[idx + 1])
            fD_times_Tc = x_c / (2 * np.pi)
            return fD_times_Tc

        Tpilot_2_Tc = channel_params['Tpilot_2_Tc']
        K_rician    = channel_params['K_rician']
        theta_k     = channel_params['theta']
        fD_times_Tc = _compute_fD_times_Tc(K_rician, theta_k)

        fD_Tpilot = fD_times_Tc * Tpilot_2_Tc
        nu_LoS    = fD_Tpilot * np.sin(theta_k)

        def R(tau):
            return (K_rician / (K_rician + 1)) * np.exp(1j * 2*np.pi * nu_LoS * tau) + (1 / (K_rician + 1)) * j0(2*np.pi * fD_Tpilot * tau)

        # STEP 1.
        first_row = R(np.arange(p))
        R_p = toeplitz(first_row, first_row.conj())
        R_p += 1e-10 * np.eye(p)

        # STEP 2.
        r_k = R(k + np.arange(p))

        # STEP 3.
        a = np.linalg.solve(R_p, r_k)

        return a
