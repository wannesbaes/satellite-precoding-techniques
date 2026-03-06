# mu-mimo/mu_mimo/processing/modulation.py

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from ..types import (ComplexArray, RealArray, IntArray, BitArray, ConstType)



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
        is_power_of_2 = (self.size > 0) and ((self.size & (self.size - 1)) == 0)
        is_power_of_4 = is_power_of_2 and (int(np.log2(self.size)) % 2 == 0)

        if self.type in ("PAM", "PSK") and not is_power_of_2:
            raise ValueError("For PAM and PSK modulation, the constellation size must be a power of 2.")
        elif self.type == "QAM" and not is_power_of_4:
            raise ValueError("For QAM modulation, the constellation size must be a power of 4.")
        
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

    def __eq__(self, other: object) -> bool:
        
        if not isinstance(other, Constellation):
            return NotImplemented
        
        return (
            self.type == other.type and
            self.size == other.size and
            np.array_equal(self.points, other.points)
        )

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

class Mapper(ABC):
    """
    Mapper Abstract Base Class.
    """

    @staticmethod
    @abstractmethod
    def apply(b: list[BitArray], ibr: IntArray, c_types: list[ConstType], Ns: IntArray) -> ComplexArray:
        """
        Apply the mapper operation to the bitstreams.

        For each data stream of each user terminal, the mapper converts the bitstream into a data symbol stream according to the information bit rate (number of bits per symbol or thus the constellation size in bits).

        Parameters
        ----------
        b : list[BitArray], shape (Ns_total, ibr[s] * M)
            The compound bitstream vector.
        ibr : IntArray, shape (K*Nr,)
            The information bit rate for each data stream.
        c_types : list[ConstType], shape (K,)
            The constellation types for the data streams to each UT.
        Ns : IntArray, shape (K,)
            The number of active data streams for each UT.

        Returns
        -------
        a : ComplexArray, shape (Ns_total, M)
            The compound data symbol vector.
        """
        raise NotImplementedError

class NeutralMapper(Mapper):
    """
    Neutral Mapper.

    Acts as a 'neutral element' for mapping.\\
    It simply converts the bitstream into a data symbol stream by interpreting the bits as integers, without applying any modulation scheme. So in practice, the bitstreams pass through the mapper without any change. A requirement for this mapper is that the information bit rate equals one bit per symbol for each data stream (neutral bit allocation)!
    """

    @staticmethod
    def apply(b: list[BitArray], ibr: IntArray, c_types: list[ConstType], Ns: IntArray) -> ComplexArray:

        if not np.all(ibr == 1):
            raise ValueError("The information bit rate must be equal to one bit per symbol for each data stream when using the NeutralMapper.")
        
        a = np.array(b, dtype=complex)
        return a

class GrayCodeMapper(Mapper):
    """
    Gray Code Mapper.
    """

    @staticmethod
    def apply(b: list[BitArray], ibr: IntArray, c_types: list[ConstType], Ns: IntArray) -> ComplexArray:

        # Initialization.
        K    = len(c_types)
        Nr   = ibr.size // K
        mask = np.arange(Nr) < Ns[:, None]
        ibr  = ibr.reshape(K, Nr)[mask]         # Only keep the ibr values of the active data streams, shape (Ns_total,)

        c_types = [c_types[k] for k in range(K) for _ in range(Ns[k])]
        
        Ns_total = np.sum(Ns)
        M        = len(b[np.argmax(ibr)]) // np.max(ibr)  
        
        # Convert the binary bitstreams, interpret as Gray code numbers, to their corresponding decimal representations.
        d = np.empty((Ns_total, M), dtype=int)
        for s in range(Ns_total):
            for m in range(M):
                d[s, m] = NumberRepresentation.gray_to_decimal(b[s][m*ibr[s] : (m+1)*ibr[s]])

        # Map the decimal numbers to their corresponding constellation points.
        a = np.empty((Ns_total, M), dtype=complex)
        for s in range(Ns_total):
            constellation_points = Constellation(type=c_types[s], size=2**ibr[s]).points
            a[s] = constellation_points[d[s]]

        return a


class Demapper(ABC):
    """
    Demapper Abstract Base Class.
    """

    @staticmethod
    @abstractmethod
    def apply(cpi_k_hat: IntArray, ibr_k: IntArray, Ns_k: int) -> list[BitArray]:
        """
        Apply the demapper operation to the reconstructed symbol streams.

        For each data stream of this user terminal, the demapper converts the reconstructed data symbol stream into a bitstream, based to the used modulation scheme (constellation typpe and sizes) and the constellation point indices of the reconstructed data symbols.

        Parameters
        ----------
        cpi_k_hat : IntArray, shape (Ns_k, M)
            The indices of the constellation points (decimal integers) corresponding to the reconstructed data symbols, for each data stream of this user terminal.
        ibr_k : IntArray, shape (Nr,)
            The information bit rate for each data stream of this user terminal.
        Ns_k : int
            The number of active data streams for this user terminal.
        
        Returns
        -------
        b_k_hat : BitArray, shape (Ns_k, ibr_k[s] * M)
            The list of reconstructed bitstreams of this user terminal.
        """
        raise NotImplementedError

class NeutralDemapper(Demapper):
    """
    Neutral Demapper.

    Acts as a 'neutral element' for demapping.\\
    It simply converts the constellation point indices into a bitstream by interpreting the indices of the constellation points as bits, without applying any demodulation scheme. So in practice, the constellation point indices pass through the demapper without any change.
    """

    @staticmethod
    def apply(cpi_k_hat: IntArray, ibr_k: IntArray, Ns_k: int) -> list[BitArray]:

        if not np.all(ibr_k[:Ns_k] == 1):
            raise ValueError("The information bit rate must be equal to one bit per symbol for each data stream when using the NeutralDemapper.")
        
        if not np.all(np.isin(cpi_k_hat, [0, 1])):
            raise ValueError("The reconstructed data symbol stream must consist of symbols that are either 0 or 1 when using the NeutralDemapper.")
        
        b_k_hat = [cpi_k_hat[s] for s in range(Ns_k)]
        return b_k_hat

class GrayCodeDemapper(Demapper):
    """
    Gray Code Demapper.
    """

    @staticmethod
    def apply(cpi_k_hat: IntArray, ibr_k: IntArray, Ns_k: int) -> list[BitArray]:

        # Determine the number of data streams and the number of symbol vectors.
        M = cpi_k_hat.shape[1]
        ibr_k = ibr_k[:Ns_k]
        
        # Convert the decimal index numbers to their Gray code representations.
        b_k_hat = [np.empty(ibr_k[s]*M) for s in range(Ns_k)]
        for s in range(Ns_k):
            for m in range(M):
                b_k_hat[s][m*ibr_k[s] : (m+1)*ibr_k[s]] = NumberRepresentation.decimal_to_gray(cpi_k_hat[s, m], length=ibr_k[s])
        
        return b_k_hat


# EQUALIZATION

class Equalizer():
    """
    Equalization Class.
    """

    @staticmethod
    def apply(z_k: ComplexArray, C_eq_k: ComplexArray, Ns_k: int) -> ComplexArray:
        """
        Apply the equalization operation to the combined signal.

        Each data stream of this user terminal is multiplied by its corresponding equalization coefficient to rescale the received symbols before the decoding process.

        Parameters
        ----------
        z_k : ComplexArray, shape (Ns_k, M)
            The combined signal for this UT.
        C_eq_k : ComplexArray, shape (Nr,)
            The equalization coefficients for each data stream of this UT.
        Ns_k : int
            The number of active data streams for this UT.
        
        Returns
        -------
        u_k : ComplexArray, shape (Ns_k, M)
            The decision variable streams for this UT.
        """
        u_k = z_k / C_eq_k[:Ns_k, np.newaxis]
        return u_k


# DETECTION

class Detector(ABC):
    """
    Detector Abstract Base Class.
    """

    @staticmethod
    @abstractmethod
    def apply(u_k: ComplexArray, ibr_k: IntArray, c_type_k: ConstType, Ns_k: int) -> IntArray:
        """
        Apply the detector operation to the decision variable streams.

        For each data stream of this user terminal, the detector converts the decision variable stream into a stream of constellation point indices (decimal integers), based on the used modulation scheme. The indices correspond to the constellation points that are most likely transmitted by the base station.

        The desicion of which constellation points that are most likely depends on the type of the detector and might be suboptimal for certain detector types.

        Parameters
        ----------
        u_k : ComplexArray, shape (Ns_k, M)
            The decision variable streams of this user terminal.
        ibr_k : IntArray, shape (Nr,)
            The information bit rate for each data stream of this user terminal.
        c_type_k : ConstType
            The constellation type for the data streams to this user terminal.
        Ns_k : int
            The number of active data streams for this user terminal.
        
        Returns
        -------
        cpi_k_hat : IntArray, shape (Ns_k, M)
            The indices of the constellation points (decimal integers) corresponding to the reconstructed data symbols, for each data stream of this user terminal.
        """
        raise NotImplementedError

class NeutralDetector(Detector):
    """
    Neutral Detector.

    Acts as a 'neutral element' for detection.\\
    It simply converts the decision variable stream into a stream of constellation point indices by interpreting the decision variables as integers, without applying any detection scheme. So in practice, the decision variable streams pass through the detector without any change.
    """

    @staticmethod
    def apply(u_k: ComplexArray, ibr_k: IntArray, c_type_k: ConstType, Ns_k: int) -> IntArray:

        if not np.all(ibr_k == 1):
            raise ValueError("The information bit rate must be equal to one bit per symbol for each data stream when using the NeutralDetector.")
        
        cpi_k_hat = np.array(u_k, dtype=int)
        return cpi_k_hat

class MDDetector(Detector):
    """
    Minimum Distance (MD) Detector.
    """
    
    @staticmethod
    def apply(u_k: ComplexArray, ibr_k: IntArray, c_type_k: ConstType, Ns_k: int) -> IntArray:

        # Determine the number of symbol vectors.
        M = u_k.shape[1]
        ibr_k = ibr_k[:Ns_k]

        # Decide the constellation points that are most likely transmitted by finding the constellation points that are closest to the decision variables. Then, retrieve the corresponding constellation point indices (decimal integers) of the decided constellation points.
        cpi_k_hat = np.empty((Ns_k, M), dtype=int)
        for s in range(Ns_k):
            constellation_points = Constellation(type=c_type_k, size=2**ibr_k[s]).points
            cpi_k_hat[s] = np.argmin( np.abs(np.tile(constellation_points, (M, 1)) - u_k[s][:, np.newaxis]), axis=1)

        return cpi_k_hat
