# src/mu_mimo_sim/processing/modulation.py

from __future__ import annotations
import abc

import numpy as np
Array = np.ndarray


class Mapper(abc.ABC):
    """
    Abstract base class for mappers.
    
    A mapper converts bit vectors into data symbol vectors according to a specified modulation constellation.
    """

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abc.abstractmethod
    def configure(self, *args, **kwargs) -> None:
        """
        Configure the mapper.
        """
        raise NotImplementedError("The configure method must be implemented by the subclass of Mapper.")

    @abc.abstractmethod
    def execute(self, b: list[Array] | Array) -> Array:
        """
        Convert the input bit sequences into an output data symbol sequences.

        Parameters
        ----------
        b : list[Array] | Array
            Input bit sequence(s).

        Returns
        -------
        a : Array
            Output data symbol sequence.
        """
        raise NotImplementedError("The execute method must be implemented by the subclass of Mapper.")


class SingleStreamGrayMapper(Mapper):
    pass

class MultipleStreamGrayMapper(Mapper):
    pass




#########
# DRAFT #
#########

class SingleStreamGrayMapper(Mapper):
    """
    Gray-coded mapper supporting PAM, PSK, and QAM modulation.
    
    Converts a bit sequence into data symbol sequence using Gray coding for given constellation size and type.
    """

    @staticmethod
    def validate_parameters(constellation_size: int, constellation_type: str) -> None:

        if constellation_type not in ['PAM', 'PSK', 'QAM']:
            raise ValueError(f"Invalid constellation type: {constellation_type}. \nChoose 'PAM', 'PSK', or 'QAM'.")
        
        if constellation_type != 'QAM' and not ((constellation_size & (constellation_size - 1)) == 0 and constellation_size > 0):
            raise ValueError(f"Constellation size {constellation_size} must be a power of 2.")
        if constellation_type == 'QAM' and not ((constellation_size & 0xAAAAAAAA) == 0):
            raise ValueError(f"For QAM, constellation size {constellation_size} must be a power of 4.")

    def __init__(self, constellation_size: int, constellation_type: str):
        """
        Initialize the Gray-coded mapper.

        Parameters
        ----------
        constellation_size : int
            Modulation constellation size.
        constellation_type : str
            Modulation constellation type.
        
        Raises
        ------
        ValueError
            If the constellation type is invalid. The constellation type must be 'PAM', 'PSK', or 'QAM'.
        ValueError
            If the constellation size is invalid. The constellation size must be a power of 2 for PAM and PSK modulation, and a power of 4 for QAM modulation.
        """
        
        self.validate_parameters(constellation_size, constellation_type)
        super().__init__(constellation_size, constellation_type)

    def set_constellation(self, constellation_size: int | None = None, constellation_type: str | None = None) -> None:
        """
        Set the constellation size and type of the Gray code Mapper.

        Parameters
        ----------
        constellation_size : int, optional
            New modulation constellation size if provided.
        constellation_type : str, optional
            New modulation constellation type if provided.
        
        Raises
        ------
        ValueError
            If the constellation type is invalid. The constellation type must be 'PAM', 'PSK', or 'QAM'.
        ValueError
            If the constellation size is invalid. The constellation size must be a power of 2 for PAM and PSK modulation, and a power of 4 for QAM modulation.
        """

        new_c_size = constellation_size if constellation_size is not None else self.c_size
        new_c_type = constellation_type if constellation_type is not None else self.c_type
        self.validate_parameters(new_c_size, new_c_type)

        super().set_constellation(constellation_size, constellation_type)


    def __call__(self, b: Array) -> Array:
        """
        Convert the input bit sequences into an output data symbol sequences.

        Parameters
        ----------
        b : Array, shape (N_symbols * log2(c_size),), dtype int
            Input bit sequence.

        Returns
        -------
        a : Array, shape (N_symbols,), dtype complex
            Output data symbol sequence.
        
        Raises
        ------
        ValueError
            If the input bit sequence length is not a multiple of log2(c_size).
        """

        # Validate input parameters.
        if b.size % int(np.log2(self.c_size)) != 0: raise ValueError(f"Input bit sequence length {b.size} must be a multiple of log2(constellation_size) = {int(np.log2(self.c_size))}.")


        # Reshape the bit sequence into blocks of m bits. Each block corresponds to one symbol.
        bit_blocks = b.reshape((-1, int(np.log2(self.c_size))))
        
        # Convert the Gray code blocks to decimal values. Each decimal value corresponds to a symbol index in the constellation.
        decimals = self._gray_to_decimal(bit_blocks)
        
        # Map the decimal values to constellation symbols to obtain the output data symbols.
        symbols = self._get_constellation_symbols(decimals)

        # Reshape the data symbols into a sequence.
        a = symbols.reshape(1, -1)
        
        return a

    def _gray_to_decimal(self, graycodes: Array) -> Array:
        """
        Convert gray code blocks to decimal values.

        Parameters
        ----------
        graycodes : Array, shape (N_blocks, log2(c_size)), dtype int
            Gray code representation of the input bit blocks.

        Returns
        -------
        decimals : Array, shape (N_blocks,), dtype int
            Decimal representation of the input gray code blocks.
        """
        m = int(np.log2(self.c_size))

        binarycodes = np.zeros_like(graycodes)
        binarycodes[:, 0] = graycodes[:, 0]
        
        for i in range(1, m):
            binarycodes[:, i] = binarycodes[:, i-1] ^ graycodes[:, i]
        
        decimals = np.dot(binarycodes, (2**np.arange(m))[::-1])

        return decimals

    def _get_constellation_symbols(self, decimals: Array) -> Array:
        """
        Map decimal values to constellation symbols.

        Parameters
        ----------
        decimals : Array, shape (N_symbols,), dtype int
            Decimal indices of the constellation symbols.

        Returns
        -------
        symbols : Array, shape (N_symbols,), dtype complex
            Constellation symbols.
        """

        if self.c_type == 'PAM':
            return self._get_pam_constellation_symbols(decimals)
        
        elif self.c_type == 'PSK':
            return self._get_psk_constellation_symbols(decimals)
        
        elif self.c_type == 'QAM':
            return self._get_qam_constellation_symbols(decimals)

    def _get_pam_constellation_symbols(self, decimals: Array) -> Array:
        """
        Generate PAM symbols with unit average power for the given constellation indices.

        Parameters
        ----------
        decimals : Array, shape (N_symbols,), dtype int
            Decimal indices of the PAM symbols.

        Returns
        -------
        symbols : Array, shape (N_symbols,), dtype complex
            PAM symbols corresponding to the input decimal indices.
        """
        delta = np.sqrt(3 / (self.c_size**2 - 1))
        symbols = (2 * decimals - (self.c_size - 1)) * delta
        return symbols

    def _get_psk_constellation_symbols(self, decimals: Array) -> Array:
        """
        Generate PSK symbols with unit average power for the given constellation indices.

        Parameters
        ----------
        decimals : Array, shape (N_symbols,), dtype int
            Decimal indices of the PSK symbols.
        
        Returns
        -------
        symbols : Array, shape (N_symbols,), dtype complex
            PSK symbols corresponding to the input decimal indices.
        """
        symbols = np.exp(1j * 2 * np.pi * decimals / self.c_size)
        return symbols

    def _get_qam_constellation_symbols(self, decimals: Array) -> Array:
        """
        Generate QAM symbols with unit average power for the given constellation indices.

        Parameters
        ----------
        decimals : Array, shape (N_symbols,), dtype int
            Decimal indices of the QAM symbols.

        Returns
        -------
        symbols : Array, shape (N_symbols,), dtype complex
            QAM symbols corresponding to the input decimal indices.
        """
        
        sqrt_M = int(np.sqrt(self.c_size))
        c_sqrtM_PAM = np.arange(-(sqrt_M - 1), sqrt_M, 2) * np.sqrt(3 / (2 * (self.c_size - 1)))
        
        real_grid, imag_grid = np.meshgrid(c_sqrtM_PAM, c_sqrtM_PAM[::-1])
        constellation = real_grid + 1j * imag_grid
        
        constellation[1::2] = constellation[1::2, ::-1]
        constellation = constellation.flatten()
        
        symbols = constellation[decimals]
        return symbols

