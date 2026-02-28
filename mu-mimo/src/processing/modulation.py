# mu-mimo/src/processing/modulation.py

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from ..types import (ComplexArray, RealArray, IntArray, BitArray, ConstType)
from typing import Literal



# HELPERS

@dataclass()
class Constellation:
    """
    A constellation.

    Parameters
    ----------
    type : ConstType
        The constellation type.
    size : int
        The constellation size (number of points in the constellation).
    points : ComplexArray, shape (size,)
        The constellation points.
    """

    type: ConstType
    size: int

    points : ComplexArray | None = None

    def __post_init__(self):

        # Validate the constellation size.
        if (self.size & (self.size - 1) == 0) and ((self.size & 0xAAAAAAAA) == 0 or self.type != 'QAM'):
            raise ValueError("The constellation size is invalid. For PAM and PSK modulation, the constellation size must be a power of 2. For QAM modulation, the constellation size must be a power of 4 (or thus even power of 2).")
        
        # Generate the constellation points for a PAM constellation.
        if self.type == "PAM":
            self.points = np.arange(-(self.size-1), (self.size-1) + 1, 2) * np.sqrt(3/(self.size**2-1))
        
        # Generate the constellation points for a PSK constellation.
        elif self.type == "PSK":
            self.points = np.exp(1j * 2*np.pi * np.arange(self.size) / self.size)
        
        # Generate the constellation points for a QAM constellation.
        elif self.type == "QAM":
            sqrtM_PAM = np.arange(-(np.sqrt(self.size)-1), (np.sqrt(self.size)-1) + 1, 2) * np.sqrt(3 / (2*(self.size-1)))
            real_grid, imaginary_grid = np.meshgrid(sqrtM_PAM, sqrtM_PAM)
            self.points = (real_grid + 1j*imaginary_grid).ravel()
        
        else:
            raise ValueError(f'The constellation type is invalid.\nChoose between "PAM", "PSK", or "QAM". Right now, type is {self.type}.')

class NumberRepresentation:
    """
    Number Representation Utility Class.

    This class provides helper functions for the conversion between different number representations.
    """

    @staticmethod
    def binary_to_decimal(b: BitArray) -> int:
        """
        Convert a binary number to its decimal representation.

        Parameters
        ----------
        b : BitArray
            The binary number.

        Returns
        -------
        d : int
            The corresponding decimal representation.
        """
        powers = 1 << np.arange(b.size - 1, -1, -1, dtype=int)
        d = int(b @ powers)
        return d

    @staticmethod
    def decimal_to_binary(d: int, length: int = None) -> BitArray:
        """
        Convert a decimal number to its binary representation.

        Parameters
        ----------
        d : int
            The decimal number.
        length : int, optional
            The length of the binary representation.\\
            If None, the length is the minimum required to represent the decimal number.

        Returns
        -------
        b : BitArray
            The corresponding binary representation.
        """
        if d < 0: 
            raise ValueError("The decimal input must be non-negative.")
        
        b = np.array(list(np.binary_repr(int(d), width=length)), dtype=int)
        return b

    @staticmethod
    def gray_to_binary(g: BitArray) -> BitArray:
        """
        Convert a Gray code number to its binary representation.

        Parameters
        ----------
        g : BitArray
            The Gray code number.

        Returns
        -------
        b : BitArray
            The corresponding binary representation.
        """
        b =  np.bitwise_xor.accumulate(g).astype(int)
        return b

    @staticmethod
    def binary_to_gray(b: BitArray) -> BitArray:
        """
        Convert a binary number to its Gray code representation.

        Parameters
        ----------
        b : BitArray
            The binary number.

        Returns
        -------
        g : BitArray
            The corresponding Gray code representation.
        """
        if b.size == 0: 
            return np.array([], dtype=int)
        
        g = np.empty_like(b, dtype=int)
        g[0] = b[0]
        if b.size > 1:
            g[1:] = np.bitwise_xor(b[:-1], b[1:])

        return g

    @staticmethod
    def gray_to_decimal(g: BitArray) -> int:
        """
        Convert a Gray code number to its decimal representation.

        Parameters
        ----------
        g : BitArray
            The Gray code number.

        Returns
        -------
        d : int
            The corresponding decimal representation.
        """
        b = NumberRepresentation.gray_to_binary(g)
        d = NumberRepresentation.binary_to_decimal(b)
        return d

    @staticmethod
    def decimal_to_gray(d: int, length: int = None) -> BitArray:
        """
        Convert a decimal number to its Gray code representation.

        Parameters
        ----------
        d : int
            The decimal number.
        length : int, optional
            The length of the Gray code representation.\\
            If None, the length is the minimum required to represent the decimal number.

        Returns
        -------
        g : BitArray
            The corresponding Gray code representation.
        """
        b = NumberRepresentation.decimal_to_binary(d, length=length)
        g = NumberRepresentation.binary_to_gray(b)
        return g


# MAPPING & DEMAPPING

# Mapping.

class Mapper(ABC):
    """
    Mapper Abstract Base Class.
    """

    @abstractmethod
    def apply(self, b: list[BitArray], ibr: IntArray, c_types: list[ConstType], Ns: IntArray) -> ComplexArray:
        """
        Apply the mapper operation to the bitstreams.

        For each data stream of each user terminal, the mapper converts the bitstream into a data symbol stream according to the information bit rate (number of bits per symbol or thus the constellation size in bits).

        Parameters
        ----------
        b : list[BitArray], shape (Ns_total, ibr[s] * num_symbols)
            The compound bitstream vector.
        ibr : IntArray, shape (Ns_total,)
            The information bit rate for each data stream.
        c_types : list[ConstType], shape (K,)
            The constellation types for the data streams to each UT.
        Ns : IntArray, shape (K,)
            The number of data streams for each UT.

        Returns
        -------
        a : ComplexArray, shape (Ns_total, num_symbols)
            The compound data symbol vector.
        """
        raise NotImplementedError
    
class NeutralMapper(Mapper):
    """
    Neutral Mapper.

    Acts as a 'neutral element' for mapping.\\
    It simply converts the bitstream into a data symbol stream by interpreting the bits as integers, without applying any modulation scheme. So in practice, the bitstreams pass through the mapper without any change. A requirement for this mapper is that the information bit rate equals one bit per symbol for each data stream (neutral bit allocation)!
    """

    def apply(self, b: list[BitArray], ibr: IntArray, c_types: list[ConstType], Ns: IntArray) -> ComplexArray:

        if not np.all(ibr == 1):
            raise ValueError("The information bit rate must be equal to one bit per symbol for each data stream when using the NeutralMapper.")
        
        a = np.array([b], dtype=complex)
        return a

class GrayCodeMapper(Mapper):
    """
    Gray Code Mapper.
    """

    def apply(self, b: list[BitArray], ibr: IntArray, c_types: list[ConstType], Ns: IntArray) -> ComplexArray:

        # Determine the total number of data streams and the number of symbols vectors.
        Ns_total = len(b)
        num_symbols = len(b[0]) // ibr[0]
        c_types = [c_types[k] for k in range(len(c_types)) for _ in range(Ns[k])]
        
        # Convert the binary bitstreams, interpret as Gray code numbers, to their corresponding decimal representations.
        d = np.empty((Ns_total, num_symbols), dtype=int)
        for s in range(Ns_total):
            for i in range(num_symbols):
                d[s, i] = NumberRepresentation.gray_to_decimal(b[s][i*ibr[s] : (i+1)*ibr[s]])

        # Map the decimal numbers to their corresponding constellation points.
        a = np.empty((Ns_total, num_symbols), dtype=complex)
        for s in range(Ns_total):
            constellation_points = Constellation(type=c_types[s], size=2**ibr[s]).points
            a[s] = constellation_points[d[s]]

        return a

# Demapping.

class Demapper(ABC):
    """
    Demapper Abstract Base Class.
    """

    def apply(cpi_k_hat: IntArray, ibr_k: IntArray, c_type: ConstType) -> list[BitArray]:
        """
        Apply the demapper operation to the reconstructed symbol streams.

        For each data stream of this user terminal, the demapper converts the reconstructed data symbol stream into a bitstream, based to the used modulation scheme (constellation typpe and sizes) and the constellation point indices of the reconstructed data symbols.

        Parameters
        ----------
        cpi_k_hat : IntArray, shape (Ns_k, num_symbols)
            The indices of the constellation points (decimal integers) corresponding to the reconstructed data symbols, for each data stream of this user terminal.
        ibr_k : IntArray, shape (Ns_k,)
            The information bit rate for each data stream of this user terminal.
        c_type : ConstType
            The constellation type for the data streams to this user terminal.
        
        Returns
        -------
        b_k_hat : BitArray, shape (Ns_k, ibr_k[s] * num_symbols)
            The list of reconstructed bitstreams of this user terminal.
        """
        raise NotImplementedError

class NeutralDemapper(Demapper):
    """
    Neutral Demapper.

    Acts as a 'neutral element' for demapping.\\
    It simply converts the constellation point indices into a bitstream by interpreting the indices of the constellation points as bits, without applying any demodulation scheme. So in practice, the constellation point indices pass through the demapper without any change.
    """

    def apply(cpi_k_hat: IntArray, ibr_k: IntArray, c_type: ConstType) -> list[BitArray]:

        if not np.all(ibr_k == 1):
            raise ValueError("The information bit rate must be equal to one bit per symbol for each data stream when using the NeutralDemapper.")
        
        if not np.all(np.isin(cpi_k_hat, [0, 1])):
            raise ValueError("The reconstructed data symbol stream must consist of symbols that are either 0 or 1 when using the NeutralDemapper.")
        
        b_k_hat = [cpi_k_hat[s] for s in range(cpi_k_hat.shape[0])]
        return b_k_hat

class GrayCodeDemapper(Demapper):
    """
    Gray Code Demapper.
    """

    def apply(cpi_k_hat: IntArray, ibr_k: IntArray, c_type: ConstType) -> list[BitArray]:

        # Determine the number of data streams and the number of symbol vectors.
        Ns_k = cpi_k_hat.shape[0]
        num_symbols = cpi_k_hat.shape[1]
        
        # Convert the decimal index numbers to their Gray code representations.
        b_k_hat = [np.empty(ibr_k[s]*num_symbols) for s in range(Ns_k)]
        for s in range(Ns_k):
            for i in range(num_symbols):
                b_k_hat[s][i*ibr_k[s] : (i+1)*ibr_k[s]] = NumberRepresentation.decimal_to_gray(cpi_k_hat[s, i], length=ibr_k[s])
        
        return b_k_hat


# DETECTION

class Detector(ABC):
    """
    Detector Abstract Base Class.
    """

    def apply(u_k: ComplexArray, ibr_k: IntArray, c_type: ConstType) -> IntArray:
        """
        Apply the detector operation to the decision variable streams.

        For each data stream of this user terminal, the detector converts the decision variable stream into a stream of constellation point indices (decimal integers), based on the used modulation scheme. The indices correspond to the constellation points that are most likely transmitted by the base station.

        The desicion of which constellation points that are most likely depends on the type of the detector and might be suboptimal for certain detector types.

        Parameters
        ----------
        u_k : ComplexArray, shape (Ns_k, num_symbols)
            The decision variable streams of this user terminal.
        ibr_k : IntArray, shape (Ns_k,)
            The information bit rate for each data stream of this user terminal.
        c_type : ConstType
            The constellation type for the data streams to this user terminal.
        
        Returns
        -------
        cpi_k_hat : IntArray, shape (Ns_k, num_symbols)
            The indices of the constellation points (decimal integers) corresponding to the reconstructed data symbols, for each data stream of this user terminal.
        """
        raise NotImplementedError

class NeutralDetector(Detector):
    """
    Neutral Detector.

    Acts as a 'neutral element' for detection.\\
    It simply converts the decision variable stream into a stream of constellation point indices by interpreting the decision variables as integers, without applying any detection scheme. So in practice, the decision variable streams pass through the detector without any change.
    """

    def apply(u_k: ComplexArray, ibr_k: IntArray, c_type: ConstType) -> IntArray:

        if not np.all(ibr_k == 1):
            raise ValueError("The information bit rate must be equal to one bit per symbol for each data stream when using the NeutralDetector.")
        
        cpi_k_hat = np.array(u_k, dtype=int)
        return cpi_k_hat

class MDDetector(Detector):
    """
    Minimum Distance Detector.
    """
    
    def apply(u_k: ComplexArray, ibr_k: IntArray, c_type: ConstType) -> IntArray:

        # Determine the number of data streams and the number of symbol vectors.
        Ns_k = u_k.shape[0]
        num_symbols = u_k.shape[1]

        # Decide the constellation points that are most likely transmitted by finding the constellation points that are closest to the decision variables. Then, retrieve the corresponding constellation point indices (decimal integers) of the decided constellation points.
        cpi_k_hat = np.empty((Ns_k, num_symbols), dtype=int)
        for s in range(Ns_k):
            constellation_points = Constellation(type=c_type, size=2**ibr_k[s]).points
            cpi_k_hat[s] = np.argmin( np.abs(np.tile(constellation_points, (num_symbols, 1)) - u_k[s][:, np.newaxis]), axis=1)

        return cpi_k_hat
