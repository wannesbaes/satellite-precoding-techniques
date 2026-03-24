# mu-mimo/mu_mimo/processing/bit_loading.py

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
import numpy as np
from ..types import ComplexArray, RealArray, IntArray, BitArray, ChannelStateInformation
if TYPE_CHECKING: from ..configs import ConstConfig


class BitLoader(ABC):
    """
    The Bit Loader Abstract Base Class (ABC).

    A bit loader class is responsible for implementing a bit loading strategy and effectively allocating bits for each single data stream.
    """

    @staticmethod
    @abstractmethod
    def compute(csi: ChannelStateInformation, F: ComplexArray, G: ComplexArray | None, c_configs: ConstConfig, Pt: float, B: float) -> tuple[IntArray, IntArray]:
        """
        Implementation of the bit loading strategy.

        The information bit rate (number of bits per symbol) for each data stream of each user terminal is computed based on the channel state information (the effective channel matrix and the SNR), the compound precoding matrix, and in case of coordinated beamforming the compound combiner matrix and according to a specific bit loading strategy.

        In case of predefined bit allocations (e.g., for fixed modulation schemes), the constellation configurations c_configs provide the necessary information.

        Parameters
        ----------
        csi : ChannelStateInformation
            The channel state information.
        F : ComplexArray, shape (Nt, K*Nr)
            The compound precoding matrix.
        G : ComplexArray | None, shape (K*Nr, K*Nr)
            The compound combiner matrix (only available in case of coordinated beamforming).
        c_configs : ConstConfig
            The constellation configurations for the data streams of each user terminal.
        Pt : float
            The total available transmit power.
        B : float
            The bandwidth of the system.

        
        Returns
        -------
        ibr : IntArray, shape (K*Nr,)
            The information bit rate (number of bits per symbol) for each data stream of each user terminal.
        Ns : IntArray, shape (K,)
            The number of data streams for each user terminal.
        """
        raise NotImplementedError

    @staticmethod
    def apply(ibr: IntArray, M: int, Ns: IntArray) -> tuple[list[list[BitArray]], list[BitArray]]:
        """
        Apply the bit loader.

        The bit loader allocates the right number of bits for each active data stream for each UT.
        Non-active data streams (i.e., data streams with ibr = 0) are not allocated any bits.

        Parameters
        ----------
        ibr : IntArray, shape (K*Nr,)
            The information bit rate for each data stream of each UT.
        M : int
            The number of symbol vectors to be transmitted at once.
        Ns : IntArray, shape (K,)
            The number of active data streams for each UT.
        
        Returns
        -------
        tx_bits_list : list[list[BitArray]], shape (K, Ns[k], ibr[k][s]*M)
            The list of allocated bits for each data stream of each UT.
        b : list[BitArray], shape (Ns_total, ibr[s]*M)
            The concatenated list of allocated bits for all data streams.
        """
        
        b = []
        for s in range(len(ibr)):
            if ibr[s] > 0:
                b_s = np.random.randint(0, 2, size=(ibr[s] * M,), dtype=np.uint8)
                b.append(b_s)
        
        Ns_cumulative = np.concatenate(([0], np.cumsum(Ns)))
        tx_bits_list = [ [ b[Ns_cumulative[k] + s] for s in range(Ns[k]) ] for k in range(len(Ns)) ]

        return tx_bits_list, b

class NeutralBitLoader(BitLoader):
    """
    Neutral Bit Loader.

    Acts as a 'neutral element' for bit loading. 
    It always allocates one bit per symbol to each data stream, and creates as many data streams per UT as it has antennas.
    """

    @staticmethod
    def compute(csi: ChannelStateInformation, F: ComplexArray, G: ComplexArray | None, c_configs: ConstConfig, Pt: float, B: float) -> tuple[IntArray, IntArray]:
        
        # Determine the number of data streams for each UT and the number of receive antennas at each UT.
        K = len(c_configs.types)
        Nr = csi.H_eff.shape[0] // K
        
        # Define the information bit rate for each data stream and the number of data streams for each UT.
        ibr = np.ones(K * Nr, dtype=int)
        Ns = np.full(K, Nr, dtype=int)
        
        return ibr, Ns

class FixedBitLoader(BitLoader):
    """
    Fixed Bit Loader.

    Allocates a predefined fixed number of bits to the data streams of each UT. Each UT gets assigned as many data streams as it has receive antennas, so every UTs gets assigned the same number of data streams as we assume each UT has the same number of receive antennas.
    
    For example, c_configs tells us that the data streams of UT 1 use 2-PSK modulation and the data streams of all other UTs use 16-QAM modulation, then the fixed bit loader will allocate 2 bits per symbol to each data stream of UT 1 and 4 bits per symbol to all other data streams.
    """

    @staticmethod
    def compute(csi: ChannelStateInformation, F: ComplexArray, G: ComplexArray | None, c_configs: ConstConfig, Pt: float, B: float) -> tuple[IntArray, IntArray]:

        # Validate the constellation sizes for each UT.
        if c_configs.sizes is None:
            raise ValueError("The constellation size for each UT must be provided beforehand when using the FixedBitLoader.")
        
        # Determine the number of data streams for each UT and the number of receive antennas at each UT.
        K = len(c_configs.sizes)
        Nr = csi.H_eff.shape[0] // K

        # Determine the number of data streams for each UT.
        Ns = np.full(K, Nr, dtype=int)
        Ns[c_configs.sizes == 0] = 0

        # Determine the information bit rates for the data streams to each UT.
        ibr = np.array([c_configs.sizes[k] for k in range(K) for _ in range(Nr)], dtype=int)

        return ibr, Ns

class AdaptiveBitLoader(BitLoader):
    """
    Adaptive Bit Loader.

    Allocates a variable number of bits to the data streams of each UT based on the channel capacity that UT. More specifically, the bit loader computes the achievable rates (shannon capacity) for each stream of all UTs. Then it calculates the information bit rates for the data streams to each UT as a fraction of the achievable rates.
    """

    @staticmethod
    def compute(csi: ChannelStateInformation, F: ComplexArray, G: ComplexArray | None, c_configs: ConstConfig, Pt: float, B: float) -> tuple[IntArray, IntArray]:

        # Validate the capacity fractions for each UT.
        if c_configs.capacity_fractions is None:
            raise ValueError("The capacity fraction for each UT must be provided beforehand when using the AdaptiveBitLoader.")

        # Determine the number of receive antennas at each UT.
        K = len(c_configs.capacity_fractions)
        Nr = csi.H_eff.shape[0] // K

        # Computes the achievable rates for each UT.
        ch_capacities = AdaptiveBitLoader._compute_achievable_rates(csi, F, G, Pt, B)

        # Determine the information bit rates as a fraction of the achievable rates, and the number of data streams for each UT.
        ibr = np.empty(K*Nr, dtype=int)
        Ns = np.empty(K, dtype=int)
        
        for k in range(K):
            
            if c_configs.types[k] == "QAM":
                ibr_k = 2 * np.floor( (ch_capacities[k*Nr : (k+1)*Nr] * c_configs.capacity_fractions[k]) / 2 ).astype(int)
            elif c_configs.types[k] in ["PAM", "PSK"]:
                ibr_k = np.floor( ch_capacities[k*Nr : (k+1)*Nr] * c_configs.capacity_fractions[k] ).astype(int)
            
            ibr[k*Nr : (k+1)*Nr] = ibr_k
            Ns[k] = np.sum(ibr_k > 0)

        return ibr, Ns
    
    @staticmethod
    def _compute_achievable_rates(csi: ChannelStateInformation, F: ComplexArray, G: ComplexArray | None, Pt: float, B: float) -> RealArray:
        """
        Compute the achievable rates (shannon capacity) for each stream of all UTs.

        Parameters
        ----------
        csi : ChannelStateInformation
            The channel state information.
        F : ComplexArray, shape (Nt, K*Nr)
            The compound precoding matrix.
        G : ComplexArray | None, shape (K*Nr, K*Nr)
            The compound combiner matrix (only available in case of coordinated beamforming).
        Pt : float
            The available total transmit power.
        B : float
            The bandwidth of the system.
        
        Returns
        -------
        abr : RealArray, shape (K*Nr,)
            The achievable rates for each stream of all UTs.
        """
        
        # Compute the transfer matrix T = G @ H @ F = H_eff @ F.
        H_eff = csi.H_eff if G is None else (G @ csi.H_eff)
        T = H_eff @ F
        
        # Compute the power of the noise, the interference, and the useful signal for each data stream.
        p_noise = Pt / csi.snr      # p_noise = N0 = sigma^2 = Pt / SNR
        p_interference = np.sum( np.abs( T - np.diag(np.diagonal(T)) )**2, axis=1 )
        p_useful = np.abs( np.diagonal(T) )**2

        # Compute the SINR for each data stream.
        sinr = p_useful / (p_interference + p_noise)

        # Compute the achievable bit rates.
        abr = 2*B * np.log2(1 + sinr)

        return abr
