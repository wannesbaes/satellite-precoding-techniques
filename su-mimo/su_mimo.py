# This module contains the implemtation of a SU-MIMO SVD communication system.

import numpy as np
import math

class SuMimoSVD:
    """
    ## Description

    A class representing a single-user multiple-input multiple-output (SU-MIMO) communication system using singlular value decomposition (SVD).

    The communication system consists of a transmitter, a channel, and a receiver. The transmitter maps the input bit sequence to a data symbol sequence according to a specified modulation constellation. In addition, the transmitter uses the SVD of the channel matrix to precode the transmitted symbols and allocates power across the transmit antennas. The channel applies a MIMO channel matrix to the transmitted symbols and adds white Gaussian noise. The receiver uses the SVD of the channel matrix to decode the received symbols back into bit sequences.

    ## Attributes
        
        ### The communication system parameters.

        Nt (int): Number of transmit antennas.
        Nr (int): Number of receive antennas.

        M (int): Size of the modulation constellation (e.g., 2, 4, 16, 64).
        type (str): Type of modulation constellation ('PAM', 'PSK', or 'QAM').

        SNR (float): Signal-to-noise ratio (SNR) in decibel (dB).


        ### The channel parameters.

        H (numpy.ndarray): The MIMO channel matrix of shape (Nr, Nt). The entries of H represent the complex channel gains between each transmit and receive antenna. They are i.i.d. complex Gaussian random variables with zero mean and unit variance by default.
        U (numpy.ndarray): Left singular vectors of the channel matrix H. The columns of U represent the orthogonal basis for the received signal space. The shape of U is (Nr, Nr).
        SIGMA (numpy.ndarray): A rectangular diagonal matrix of shape (Nr, Nt) containing the singular values of the channel matrix H on its diagonal. The singular values represent the strength of the communication modes between the transmit and receive antennas. They are non-negative real numbers sorted in descending order. The number of non-zero singular values is equal to the rank of H.
        Vh (numpy.ndarray): Right singular vectors of the channel matrix H. The rows of Vh represent the orthogonal basis for the transmitted signal space. The shape of Vh is (Nt, Nt).

        
        ### The communication system signals.

        bits (numpy.ndarray): Input bit sequences of shape (Nt, Nbits).
        symbols (numpy.ndarray): Transmitted data symbol sequences of shape (Nt, Nsymbols).
        w (numpy.ndarray): Additive white Gaussian noise of shape (Nr, Nsymbols).
        r (numpy.ndarray): Received data symbol sequences of shape (Nr, Nsymbols).
        symbols_hat (numpy.ndarray): Estimated data symbol sequences of shape (Nt, Nsymbols).
        bits_hat (numpy.ndarray): Estimated bit sequences of shape (Nt, Nbits).

    
        
    ## Methods

        ...

    """


    def __init__(self, Nt, Nr, constellation_size, constellation_type, SNR, H=None):
        """ Initialize the SU-MIMO SVD communication system. """
        
        # The communication system parameters (number of antennas and constellation).
        self.Nt = Nt
        self.Nr = Nr

        self.M = constellation_size
        self.type = constellation_type

        self.SNR = SNR

        # The channel parameters.
        self._H = H
        self._U = None
        self._SIGMA = None
        self._Vh = None

        # The communication system signals.
        self._bits = None
        self._symbols = None
        self._w = None
        self._r = None
        self._symbols_hat = None
        self._bits_hat = None


    def __str__(self):
        pass

    def __call__(self, *args, **kwds):
        pass

    def simulate(self, Nbits):
        pass


    # BUILDING BLOCKS OF THE COMMUNICATION SYSTEM.

    def mapper(self):
        """
        Convert bit sequences into data symbol sequences according to the specified modulation constellation. The mapper is the inverse operation of the demapper.

        Args:
            bits (numpy.ndarray): Input bit sequences (1D or 2D array of 0s and 1s).
            M (int): Size of the constellation (e.g., 2, 4, 16, 64).
            type (str): Type of constellation ('PAM', 'PSK', or 'QAM').

        Returns:
            symbols (numpy.ndarray): Output data symbol sequences (1D or 2D array of complex numbers).
        """

        
        #1 Divide the input bit sequences into blocks of mc bits, where M = 2^mc.
        
        mc = int(np.log2(self.M))
        if (self._bits.shape[1] % mc != 0) : raise ValueError('The length of the bit sequences is invalid. They must be a multiple of log2(M).')    
        
        bits = self._bits.flatten()
        bits = bits.reshape((bits.size // mc, mc))


        #2 Convert the blocks of mc bits from gray code to the corresponding decimal value.
        
        graycodes = bits
        binarycodes = np.zeros_like(graycodes)
        binarycodes[:, 0] = graycodes[:, 0]

        for i in range(1, graycodes.shape[1]):
            binarycodes[:, i] = binarycodes[:, i-1] ^ graycodes[:, i]
        
        decimals = np.dot(binarycodes, (2**np.arange(mc))[::-1])  
    

        #3 Convert the decimal values to the corresponding data symbols, according to the specified constellation type.

        if self.type == 'PAM' :
            dmin = math.sqrt(12/(self.M**2-1))
            symbols = (decimals - (self.M-1)/2) * dmin
        
        elif self.type == 'PSK' :
            symbols = np.exp(2 * np.pi * decimals * 1j / self.M)

        elif self.type == 'QAM' :
            if (mc % 2 != 0) : raise ValueError('The constellation size M is invalid. For QAM Modulation, M must be a power of 4 (e.g., 4, 16, 64).')
            dmin = math.sqrt(6/(self.M-1))
            symbols_real_part = ((decimals//math.sqrt(self.M)) - (math.sqrt(self.M)-1)/2) * dmin
            symbols_imaginary_part = ( np.where( ((decimals//int(math.sqrt(self.M))) % 2 == 0), (int(math.sqrt(self.M))-1) - (decimals % int(math.sqrt(self.M))), (decimals % int(math.sqrt(self.M))) ) - (int(math.sqrt(self.M))-1)/2) * dmin
            symbols = symbols_real_part + (symbols_imaginary_part * 1j)

        else :
            raise ValueError('The constellation type is invalid. Choose between "PAM", "PSK", or "QAM".')
        

        #4 Store the output data symbol sequences.

        symbols = symbols.reshape((self._bits.shape[0], self._bits.shape[1] // mc)) if self._bits.ndim != 1 else symbols.flatten()
        self._symbols = symbols

    def demapper(self):
        """
        Convert data symbol sequences into bit sequences according to the specified modulation constellation. The demapper is the inverse operation of the mapper.

        Args:
            symbols (numpy.ndarray): Input data symbol sequence (1D or 2D array of complex numbers).
            M (int): Size of the constellation (e.g., 2, 4, 16, 64).
            type (str): Type of constellation ('PAM', 'PSK', or 'QAM').
        
        Returns:
            bits (numpy.ndarray): Output bit sequence (1D or 2D array of bits).
        """

        #1 Setup.

        mc = int(np.log2(self.M))
        symbols = self._symbols_hat.flatten()


        #2 Convert the data symbols to the corresponding decimal values, according to the specified constellation type.
        
        if self.type == 'PAM':
            dmin = math.sqrt(12/(self.M**2-1))
            decimals = np.round(symbols/dmin + (self.M-1)/2).astype(int)
        
        elif self.type == 'PSK':
            phases = np.angle(symbols)
            phases[phases < 0] += 2*np.pi
            decimals = np.round((phases * self.M) / (2*np.pi)).astype(int)
        
        elif self.type == 'QAM':
            if (mc % 2 != 0): raise ValueError('The constellation size M is invalid. For QAM Modulation, M must be a power of 4 (e.g., 4, 16, 64).')
            dmin = math.sqrt(6/(self.M-1))
            real_parts = np.round( np.real(symbols)/dmin + (int(math.sqrt(self.M))-1)/2 ).astype(int)
            imaginary_parts = np.round( np.imag(symbols)/dmin + (int(math.sqrt(self.M))-1)/2 ).astype(int)
            decimals = (real_parts * int(math.sqrt(self.M))) + np.where((real_parts % 2 == 0), (int(math.sqrt(self.M))-1) - imaginary_parts, imaginary_parts)

        else:
            raise ValueError('The constellation type is invalid. Choose between "PAM", "PSK", or "QAM".')


        #3 Convert the decimal values to the corresponding blocks of mc bits in gray code.

        binarycodes = ((decimals[:, None].astype(int) & (1 << np.arange(mc))[::-1].astype(int)) > 0).astype(int)

        graycodes = np.zeros_like(binarycodes)
        graycodes[:, 0] = binarycodes[:, 0]
        for i in range(1, mc):
            graycodes[:, i] = binarycodes[:, i] ^ binarycodes[:, i - 1]


        #4 Store the output bit sequences.

        bits = graycodes.flatten()
        bits = bits.reshape((self._symbols_hat.shape[0], self._symbols_hat.shape[1] * mc)) if self._symbols_hat.ndim != 1 else bits.flatten()
        self._bits_hat = bits


    def set_channel(self):
        """
        Initialize the MIMO channel matrix and compute its SVD!
        If no channel is provided, the channel matrix is initialized with i.i.d. complex Gaussian (zero mean and unit variance) random variables.
        """          
        
        # Initialize the MIMO channel matrix H with i.i.d. complex Gaussian (zero mean and unit variance) random variables, if no channel is provided.
        if self._H is None: H = (np.random.randn(self.Nr, self.Nt) + 1j * np.random.randn(self.Nr, self.Nt)) / np.sqrt(2)
        self._H = H
        
        # Compute the SVD of the channel matrix H and store the results in U, SIGMA, and Vh.
        self._U, s, self._Vh = np.linalg.svd(H)
        self._SIGMA = np.zeros((self.Nr, self.Nt), dtype=complex)
        np.fill_diagonal(self._SIGMA, s)

    def get_channel(self):
        """ Get the current MIMO channel matrix H."""
        return self._H


    def generate_noise(self):
        """
        Generate complex circularly-symmetric additive white Gaussian noise (AWGN) for the communication system, based on the specified SNR.
        """

        # Calculate the noise variance based on the specified SNR (in dB) and the average symbol power Es.
        Es = 1.0
        SNR_linear = 10.0 ** (self.SNR/10.0)
        var_n = Es / SNR_linear

        # Sample complex circularly-symmetric AWGN with the calculated noise variance.
        w = np.sqrt(var_n/2.0) * (np.random.randn(self.Nr, self._symbols.shape[1]) + 1j * np.random.randn(self.Nr, self._symbols.shape[1]))

        # Store the noise.
        self._w = w

    def get_noise(self):
        """ Get the current noise matrix w."""
        return self._w


    # PERFORMANCE METRICS OF THE COMMUNICATION SYSTEM.

    def ber(self):
        return np.mean(self._bits != self._bits_hat)

    def plot_ber():
        pass

