# mu-mimo/mu_mimo/processing/channel.py

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
#from ..types import ComplexArray, RealArray
from tqdm import tqdm
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os

ComplexArray = np.ndarray
RealArray = np.ndarray

# CHANNEL MODELS

class ChannelModel(ABC):
    """
    The Channel Model Abstract Base Class (ABC).

    A channel model is responsible for generating the channel matrix according to a specific channel model and applying the channel effects to the transmitted signals.
    """

    @staticmethod
    @abstractmethod
    def generate(Nr_total: int, Nt: int) -> ComplexArray:
        """
        Generate the channel matrix.

        Parameters
        ----------
        Nr_total : int
            The total number of receive antennas across all UTs.
        Nt : int
            The number of transmit antennas at the BS.
        
        Returns
        -------
        H : ComplexArray, shape (Nr_total, Nt)
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

    @staticmethod
    def generate( Nr_total: int, Nt: int) -> ComplexArray:
        H = np.eye(Nr_total, Nt, dtype=complex)
        return H

class IIDRayleighChannelModel(ChannelModel):
    """
    Independent and Identically Distributed (IID) Rayleigh Fading Channel Model.

    This channel model generates a channel matrix with independent and identically distributed (IID) circularly-symmetric zero-mean unit-variance complex Gaussian entries.\\
    The Rayleigh fading aspect is captured by the fact that the channel coefficients change independently after M transmissions.
    """

    @staticmethod
    def generate(Nr_total: int, Nt: int) -> ComplexArray:
        H = (1 / np.sqrt(2)) * (np.random.randn(Nr_total, Nt) + 1j * np.random.randn(Nr_total, Nt))
        return H


# NOISE MODELS

class NoiseModel(ABC):
    """
    Noise Model Abstract Base Class (ABC).
    
    A noise model is responsible for generating the noise vectors according to a specific noise model and applying the noise effects to the received signals.
    """

    @staticmethod
    @abstractmethod
    def generate(snr: float, x: ComplexArray, Nr_total: int) -> ComplexArray:
        """
        Generate the noise vectors.

        Parameters
        ----------
        snr : float
            The signal-to-noise ratio.
        x : ComplexArray, shape (Nt, M)
            The transmitted signals.
        Nr_total : int
            The total number of receive antennas across all UTs.

        Returns
        -------
        n : ComplexArray, shape (Nr_total, M)
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

    @staticmethod
    def generate(snr: float, x: ComplexArray, Nr_total: int) -> ComplexArray:
        n = np.zeros((Nr_total, x.shape[1]), dtype=complex)
        return n

class CSAWGNNoiseModel(NoiseModel):
    """
    Circularly-Symmetric Additive White Gaussian Noise (CSAWGN) Model.

    This noise model generates complex proper, circularly-symmetric additive white Gaussian noise (AWGN) vectors based on the specified signal-to-noise ratio (SNR).
    """

    @staticmethod
    def generate(snr: float, x: ComplexArray, Nr_total: int) -> ComplexArray:
        
        # Compute the noise power based on the current SNR and the signal power of x.
        p_signal = np.mean( np.sum( np.abs(x)**2, axis=0 ) )
        p_noise = p_signal / snr
        sigma = np.sqrt(p_noise / 2)

        # Generate complex proper, circularly-symmetric AWGN noise vectors with the computed noise power.
        n = sigma * (np.random.randn(Nr_total, x.shape[1]) + 1j * np.random.randn(Nr_total, x.shape[1]))
        return n
