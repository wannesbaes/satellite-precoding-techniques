# mu-mimo/src/processing/resource_allocation.py

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from ..types import (ComplexArray, RealArray, IntArray, BitArray, ChannelStateInformation)
from typing import Literal



# BIT ALLOCATION

# Bit Deallocators.

class BitDeallocator(ABC):
    """
    The Bit Deallocator Abstract Base Class (ABC).

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

    def __init__(self):
        self.c_type: Literal["PAM", "PSK", "QAM"] = "QAM"

    @abstractmethod
    def compute(self, csi: ChannelStateInformation, F: ComplexArray, G: ComplexArray | None, P: RealArray) -> tuple[IntArray, IntArray]:
        """
        Implementation of the bit allocation strategy.

        The information bit rate (number of bits per symbol) for each data stream of each user terminal is computed based on the channel state information (the effective channel matrix and the SNR), the compound precoding matrix, the power allocated to each data stream and in case of coordinated beamforming the compound combiner matrix and according to a specific bit allocation strategy.

        Parameters
        ----------
        csi : ChannelStateInformation
            The channel state information.
        F : ComplexArray, shape (Nt, Ns_total)
            The compound precoding matrix.
        G : ComplexArray | None, shape (Ns_total, K*Nr)
            The compound combiner matrix (only available in case of coordinated beamforming).
        P : RealArray, shape (Ns_total,)
            The power allocated to each data stream.
        
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
    It always allocates one bit per symbol to each data stream (2 in case of QAM modulation, since 1 is not allowed), and creates as many data streams per UT as it has antennas.
    """

    def compute(self, csi: ChannelStateInformation, F: ComplexArray, G: ComplexArray | None, P: RealArray) -> tuple[IntArray, IntArray]:
        
        # Determine the number of data streams for each UT and the number of receive antennas at each UT.
        K = len(P)
        Nr = csi.H_eff.shape[0] // K
        
        # Define the information bit rate for each data stream and the number of data streams for each UT.
        ibr = np.ones(K * Nr, dtype=int) * (2 if self.c_type == "QAM" else 1)
        Ns = np.full(K, Nr, dtype=int)
        
        return ibr, Ns



# POWER ALLOCATION

class PowerDeallocator(ABC):

    def execute(self, P_k: RealArray, z_k: ComplexArray) -> RealArray:
        pass


class PowerAllocator(ABC):

    @abstractmethod
    def compute(self, Pt: float, H_eff: ComplexArray, snr: int, K: int, Ns: int, Nt: int) -> RealArray:
        raise NotImplementedError

    def execute(self, P: RealArray, a: ComplexArray) -> RealArray:
        a_p = np.diag(P) * a
        return a_p


class NeutralPowerAllocator(PowerAllocator):

    def compute(self, Pt: float, H_eff: ComplexArray, snr: int, K: int, Ns: int, Nt: int) -> RealArray:
        P = np.ones(K * Nt)
        return P
