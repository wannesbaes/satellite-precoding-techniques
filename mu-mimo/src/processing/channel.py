# mu-mimo/src/processing/channel.py

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from ..types import (ComplexArray, RealArray, IntArray, BitArray)


# CHANNEL MODELS

class ChannelModel(ABC):

    @abstractmethod
    def generate(self, Nr_total: int, Nt: int) -> ComplexArray:
        raise NotImplementedError

    def apply(self, H: ComplexArray, x: ComplexArray) -> ComplexArray:
        y = H @ x
        return y


class NeutralChannelModel(ChannelModel):

    def generate(self, Nr_total: int, Nt: int) -> ComplexArray:
        H = np.eye(Nr_total, Nt, dtype=complex)
        return H


class IIDRayleighChannelModel(ChannelModel):

    def generate(self, Nr_total: int, Nt: int) -> ComplexArray:
        H = (1 / np.sqrt(2)) * (np.random.randn(Nr_total, Nt) + 1j * np.random.randn(Nr_total, Nt))
        return H


# NOISE MODELS

class NoiseModel(ABC):

    @abstractmethod
    def generate(self, snr: float, x: ComplexArray, Nr_total: int) -> ComplexArray:
        raise NotImplementedError

    def apply(self, noise: ComplexArray, y_noiseless: ComplexArray) -> ComplexArray:
        y = y_noiseless + noise
        return y


class NeutralNoiseModel(NoiseModel):

    def generate(self, snr: float, x: ComplexArray, Nr_total: int) -> ComplexArray:
        noise = np.zeros((Nr_total, x.shape[1]), dtype=complex)
        return noise


class CSAWGNNoiseModel(NoiseModel):

    def generate(self, snr: float, x: ComplexArray, Nr_total: int) -> ComplexArray:
        
        # Compute the noise power based on the current SNR and the signal power of x.
        p_signal = np.mean( np.sum( np.abs(x)**2, axis=0 ) )
        p_noise = p_signal / snr
        sigma = np.sqrt(p_noise / 2)

        # Generate complex proper, circularly-symmetric AWGN noise vectors with the computed noise power.
        noise = sigma * (np.random.randn(Nr_total, x.shape[1]) + 1j * np.random.randn(Nr_total, x.shape[1]))
        return noise
