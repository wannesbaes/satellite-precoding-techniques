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
        The constellation size.
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
    def binary_to_decimal(b: BitArray) -> IntArray:
        """
        Convert a binary number to its decimal representation.

        Parameters
        ----------
        b : BitArray
            The binary number.

        Returns
        -------
        d : IntArray, shape (N,)
            The corresponding decimal representation.
        """
        raise NotImplementedError

    @staticmethod
    def decimal_to_binary(d: int) -> BitArray:
        """
        Convert a decimal number to its binary representation.

        Parameters
        ----------
        d : int
            The decimal number.

        Returns
        -------
        b : BitArray
            The corresponding binary representation.
        """
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

    @staticmethod
    def decimal_to_gray(d: int) -> BitArray:
        """
        Convert a decimal number to its Gray code representation.

        Parameters
        ----------
        d : int
            The decimal number.

        Returns
        -------
        g : BitArray
            The corresponding Gray code representation.
        """
        raise NotImplementedError


# MAPPING & DEMAPPING

# Mapping.

class Mapper(ABC):

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

    Acts as a 'neutral element' for mapping.
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
    pass

class NeutralDemapper(Demapper):
    pass

class GrayCodeDemapper(Demapper):
    pass

