# mu-mimo/src/processing/resource_allocation.py

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from ..types import (ComplexArray, RealArray, IntArray, BitArray, ChannelStateInformation, ConstConfig)
from typing import Literal



# BIT ALLOCATION

# Bit Deallocator.

class BitDeallocator():
    """
    The Bit Deallocator.

    This class is responsible for deallocating the bits for each data stream of this user terminal.
    """

    def execute(self, ibr_k: IntArray, b_hat_k: list[BitArray]) -> list[BitArray]:
        """
        Deallocate the bits for each data stream of this UT.

        Parameters
        ----------
        ibr_k : IntArray
            The information bit rate (number of bits per symbol) for each data stream of this UT.
        b_hat_k : list[BitArray], shape (Ns, ibr_k[s])
            The reconstructed bits for each data stream of this UT.
        """
        return b_hat_k

# Bit Allocators.

class BitAllocator(ABC):
    """
    The Bit Allocator Abstract Base Class (ABC).

    This class is responsible for implementing a bit allocation strategy and effectively allocating bits for each single data stream.
    """

    @abstractmethod
    def compute(self, csi: ChannelStateInformation, F: ComplexArray, G: ComplexArray | None, P: RealArray, c_configs: ConstConfig, Pt: float, B: float) -> tuple[IntArray, IntArray]:
        """
        Implementation of the bit allocation strategy.

        The information bit rate (number of bits per symbol) for each data stream of each user terminal is computed based on the channel state information (the effective channel matrix and the SNR), the compound precoding matrix, the power allocated to each data stream and in case of coordinated beamforming the compound combiner matrix and according to a specific bit allocation strategy.

        In case of predefined bit allocations (e.g., for fixed modulation schemes), the constellation configurations c_configs provide the necessary information.

        Parameters
        ----------
        csi : ChannelStateInformation
            The channel state information.
        F : ComplexArray, shape (Nt, K*Nr)
            The compound precoding matrix.
        G : ComplexArray | None, shape (K*Nr, K*Nr)
            The compound combiner matrix (only available in case of coordinated beamforming).
        P : RealArray, shape (K*Nr,)
            The power allocated to each data stream.
        c_configs : ConstConfig
            The constellation configurations for the data streams of each user terminal.
        Pt : float
            The total available transmit power.
        B : float
            The bandwidth of the system.

        
        Returns
        -------
        ibr : IntArray, shape (Ns_total,)
            The information bit rate (number of bits per symbol) for each data stream of each user terminal.
        Ns : IntArray, shape (K,)
            The number of data streams for each user terminal.
        """
        raise NotImplementedError

    def apply(self, ibr: IntArray, num_symbols: int, Ns: IntArray) -> tuple[list[list[BitArray]], list[BitArray]]:
        """
        Apply the bit allocator.

        The bit allocator allocates the right number of bits for each data stream for each UT.

        Parameters
        ----------
        ibr : IntArray, shape (Ns_total,)
            The information bit rate for each data stream of each UT.
        num_symbols : int
            The number of symbol vectors to be transmitted at once.
        Ns : IntArray, shape (K,)
            The number of data streams for each UT.
        
        Returns
        -------
        tx_bits_list : list[list[BitArray]], shape (K, Ns[k], ibr[k][s]*num_symbols)
            The list of allocated bits for each data stream of each UT.
        b : list[BitArray], shape (Ns_total, ibr[s]*num_symbols)
            The concatenated list of allocated bits for all data streams.
        """
        
        b = []
        for s in range(len(ibr)):
            if ibr[s] > 0:
                b_s = np.random.randint(0, 2, size=(ibr[s] * num_symbols,), dtype=np.uint8)
                b.append(b_s)
        
        Ns_cumulative = np.concatenate(([0], np.cumsum(Ns)))
        tx_bits_list = [ [ b[Ns_cumulative[k] + s] for s in range(Ns[k]) ] for k in range(len(Ns)) ]

        return tx_bits_list, b

class NeutralBitAllocator(BitAllocator):
    """
    Neutral Bit Allocator.

    Acts as a 'neutral element' for bit allocation. 
    It always allocates one bit per symbol to each data stream, and creates as many data streams per UT as it has antennas.
    """

    def compute(self, csi: ChannelStateInformation, F: ComplexArray, G: ComplexArray | None, P: RealArray, c_configs: ConstConfig, Pt: float, B: float) -> tuple[IntArray, IntArray]:
        
        # Determine the number of data streams for each UT and the number of receive antennas at each UT.
        K = len(c_configs.types)
        Nr = csi.H_eff.shape[0] // K
        
        # Define the information bit rate for each data stream and the number of data streams for each UT.
        ibr = np.ones(K * Nr, dtype=int)
        Ns = np.full(K, Nr, dtype=int)
        
        return ibr, Ns

class FixedBitAllocator(BitAllocator):
    """
    Fixed Bit Allocator.

    Allocates a predefined fixed number of bits to the data streams of each UT. Each UT gets assigned as many data streams as it has receive antennas, so every UTs gets assigned the same number of data streams as we assume each UT has the same number of receive antennas.
    
    For example, c_configs tells us that the data streams of UT 1 use 2-PSK modulation and the data streams of all other UTs use 16-QAM modulation, then the fixed bit allocator will allocate 2 bits per symbol to each data stream of UT 1 and 4 bits per symbol to all other data streams.
    """

    def compute(self, csi: ChannelStateInformation, F: ComplexArray, G: ComplexArray | None, P: RealArray, c_configs: ConstConfig, Pt: float, B: float) -> tuple[IntArray, IntArray]:

        # Validate the constellation sizes for each UT.
        if c_configs.sizes is None:
            raise ValueError("The constellation size for each UT must be provided beforehand when using the FixedBitAllocator.")
        
        # Determine the number of data streams for each UT and the number of receive antennas at each UT.
        K = len(c_configs.sizes)
        Nr = csi.H_eff.shape[0] // K

        # Determine the number of data streams for each UT.
        Ns = np.full(K, Nr, dtype=int)
        Ns[c_configs.sizes == 0] = 0

        # Determine the information bit rates for the data streams to each UT.
        ibr = np.array([c_configs.sizes[k] for k in range(K) for _ in range(Ns[k])], dtype=int)
        ibr = ibr[ibr > 0]

        return ibr, Ns

class AdaptiveBitAllocator(BitAllocator):
    """
    Adaptive Bit Allocator.

    Allocates a variable number of bits to the data streams of each UT based on the channel capacity that UT. More specifically, the bit allocator computes the achievable rates (shannon capacity) for each stream of all UTs. Then it calculates the information bit rates for the data streams to each UT as a fraction of the achievable rates.
    """

    def compute(self, csi: ChannelStateInformation, F: ComplexArray, G: ComplexArray | None, P: RealArray, c_configs: ConstConfig, Pt: float, B: float) -> tuple[IntArray, IntArray]:

        # Validate the capacity fractions for each UT.
        if c_configs.capacity_fractions is None:
            raise ValueError("The capacity fraction for each UT must be provided beforehand when using the AdaptiveBitAllocator.")

        # Determine the number of data streams for each UT and the number of receive antennas at each UT.
        K = len(c_configs.capacity_fractions)
        Nr = csi.H_eff.shape[0] // K

        # Computes the achievable rates for each UT.
        ch_capacities = self._compute_achievable_rates(csi, F, G, P, K, Nr)

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
        
        ibr = ibr[ibr > 0]

        return ibr, Ns
    
    def _compute_achievable_rates(self, csi: ChannelStateInformation, F: ComplexArray, G: ComplexArray | None, P: RealArray, Pt: float, B: float) -> RealArray:
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
        P : RealArray, shape (K*Nr,)
            The power allocated to each data stream.
        Pt : float
            The available total transmit power.
        B : float
            The bandwidth of the system.
        
        Returns
        -------
        abr : RealArray, shape (K*Nr,)
            The achievable rates for each stream of all UTs.
        """
        
        # ...
        H_eff = csi.H_eff if G is None else (G @ csi.H_eff)
        T = H_eff @ F
        
        # ...
        p_noise = Pt / csi.snr
        p_interference = ( P * (np.sum(T, axis=1) - np.diagonal(T)) )**2
        p_useful = ( P * np.diagonal(T) )**2

        # ...
        sinr = p_useful / (p_interference + p_noise)

        # Compute the achievable bit rates.
        abr = 2*B * np.log2(1 + sinr)

        return abr


# POWER ALLOCATION

# Power Deallocator.

class PowerDeallocator():
    pass

# Power Allocators.

class PowerAllocator(ABC):
    pass

class NeutralPowerAllocator(PowerAllocator):
    pass

class EqualPowerAllocator(PowerAllocator):
    pass

class BeamformingPowerAllocator(PowerAllocator):
    pass

class WaterfillingPowerAllocator(PowerAllocator):
    pass

