# mu-mimo/mu_mimo/processing/channel_estimation.py

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from ..types import ComplexArray, ChannelStateInformation, ReceivePilotMessage


# CHANNEL ESTIMATION

class ChannelEstimator(ABC):
    """
    The Channel Estimator Abstract Base Class (ABC).

    This class is responsible for the channel estimation in each user terminal (UT).
    """

    @staticmethod
    @abstractmethod
    def estimate(rx_pilots_msg: ReceivePilotMessage) -> ChannelStateInformation:
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

    @staticmethod
    def estimate(rx_pilots_msg: ReceivePilotMessage) -> ChannelStateInformation:
        csi_k = rx_pilots_msg.csi_k
        return csi_k


# CHANNEL PREDICTION

class ChannelPredictor(ABC):
    """
    The Channel Predictor Abstract Base Class (ABC).

    This class is responsible for the channel prediction in each user terminal (UT).
    """

    @staticmethod
    @abstractmethod
    def predict(csi_k: ChannelStateInformation) -> ComplexArray:
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

    @staticmethod
    def predict(csi_k: ChannelStateInformation) -> ComplexArray:
        H_k_hathat = csi_k.H_hat[-1]
        return H_k_hathat
