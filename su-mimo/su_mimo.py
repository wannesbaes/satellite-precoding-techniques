# This module contains the implemtation of a SU-MIMO SVD communication system.

import numpy as np
import math


class Transmitter:
    """
    ## Description
    A class representing the transmitter of a single-user multiple-input multiple-output (SU-MIMO) communication system, in which the channel state information is available at the transmitter.
    The transmitter maps the input bit sequence to a data symbol sequence according to a specified modulation constellation. In addition, the transmitter uses the SVD of the channel matrix to precode the transmitted symbols and (!!TODO!!) allocates power across the transmit antennas.
    
    ## Attributes ##
        
        ### The transmitter parameters.
        Nt (int): Number of transmit antennas.

        ### The necessary communication system parameters.
        M (int): Size of the modulation constellation (e.g., 2, 4, 16, 64).
        type (str): Type of modulation constellation ('PAM', 'PSK', or 'QAM').

        ### The channel state information.
        _Vh (numpy.ndarray): The right singular vectors of the channel matrix H, obtained from the SVD of H.

        ### The transmitter signals.
        _bits (numpy.ndarray): The input bit sequences of shape (Nt, Nbits).
        _symbols (numpy.ndarray): The complex data symbol sequences of shape (Nt, Nsymbols).
        _s (numpy.ndarray): The output signal of shape (Nt, Nsymbols).
    
        
    ## Methods
       
        ### Special methods.
        __init__(): Initialize the transmitter parameters and signals.
        __str__(): Return a string representation of the transmitter object.
        __call__(): Allow the transmitter object to be called as a function. When called, it executes the simulate() method.

        ### Operations of the transmitter.
        mapper(): Map the input bit sequences to data symbol sequences according to the specified modulation constellation and store them.
        precoder(): Precode the data symbols using the precoding matrix and store them.
        simulate(): Execute the operations of the transmitter.
    """

    def __init__(self, Nt, constellation_size, constellation_type, Vh):
        """ Initialize the transmitter. """

        # The transmitter parameters.
        self.Nt = Nt

        # The necessary communication system parameters.
        self.M = constellation_size
        self.type = constellation_type

        # The channel state information.
        self._Vh = Vh
        
        # The transmitter signals.
        self._bits = None
        self._symbols = None
        self._s = None

    def __str__(self):
        """ Return a string representation of the transmitter object. """
        return f"Transmitter: \n  - Number of antennas = {self.Nt}\n  - Constellation = {self.M}-{self.type}"

    def __call__(self):
        """ Allow the transmitter object to be called as a function. When called, it executes the simulate() method. """
        return self.simulate()
    

    def set_bits(self, bits):
        """ Set the input bit sequences to be transmitted. """
        self._bits = bits

    def mapper(self):
        """ Convert the bit sequences into data symbol sequences according to the specified modulation constellation. The mapper is the inverse operation of the demapper. """

        
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

    def precoder(self):
        """ Precode the data symbols using the right singular vectors of the channel matrix H and store them. So, we assume the channel state information is available at the transmitter. """
        self._s = np.dot(self._Vh.conj().T, self._symbols)

    def simulate(self):
        """
        Simulate the transmitter operations: get the channel state information, generate the bit sequences, map them to complex symbols, and precode the symbols.
        The precoded symbols are ready to be transmitted through the MIMO channel.

        Returns:
            _s (numpy.ndarray): The output signal. This is the precoded data symbol sequences ready to be transmitted through the channel.
        """

        # Simulate the transmitter operations.
        self.mapper()
        self.precoder()

        # Return the output signal.
        return self._s


class Channel:
    """
    ## Description
    ...
    
    ## Attributes
    ...
    
    ## Methods
    ...
    """

    def __init__(self, Nt, Nr, SNR, H=None, s=None):
        """ Initialize the channel. """

        # The necessary communication system parameters.
        self.Nt = Nt
        self.Nr = Nr
        
        # The channel state information (CSI).        
        self._H = H
        self._U = None
        self._S = None
        self._Vh = None
        self.set_CSI()

        # The noise parameters.
        self.SNR = SNR
        self._w = None

        # The channel signals.
        self._s = s
        self._r = None
    
    def __str__(self):
        """ Return a string representation of the channel object. """
        return f"Channel: \n  - Number of transmitting and receiving antennas is {self.Nt} and {self.Nr}\n  - SNR = {self.SNR} dB\n  - H = {'Provided' if self._H is not None else 'i.i.d. complex Gaussian (0 mean, 1 variance) variables.'}"
    
    def __call__(self):
        """ Allow the channel object to be called as a function. When called, it executes the simulate() method. """
        return self.simulate()


    def set_s(self, s):
        """ Set the input signal to be transmitted through the channel. """
        self._s = s
    
    def set_CSI(self):
        """
        Initialize the MIMO channel matrix and compute its SVD!
        If no channel is provided, the channel matrix is initialized with i.i.d. complex Gaussian (zero mean and unit variance) random variables.
        """          
        
        # Initialize the MIMO channel matrix H with i.i.d. complex Gaussian (zero mean and unit variance) random variables, if no channel is provided.
        if self._H is None: self._H = (np.random.randn(self.Nr, self.Nt) + 1j * np.random.randn(self.Nr, self.Nt)) / np.sqrt(2)
        
        # Compute the SVD of the channel matrix H and store the results in U, SIGMA, and Vh.
        self._U, self._S, self._Vh = np.linalg.svd(self._H)

    def get_CSI(self):
        """ Get the current channel state information (CSI), in terms of the channel matrix H and its SVD (U, S, Vh)."""
        return self._H, self._U, self._S, self._Vh

    def generate_noise(self):
        """ Generate complex circularly-symmetric additive white Gaussian noise (AWGN) on the channel, based on the specified SNR. """

        # Calculate the noise variance based on the specified SNR (in dB) and the average symbol power Es.
        Es = 1.0
        SNR_linear = 10.0 ** (self.SNR/10.0)
        var_n = Es / SNR_linear

        # Sample complex circularly-symmetric AWGN with the calculated noise variance.
        w = np.sqrt(var_n/2.0) * (np.random.randn(self.Nr, self._s.shape[1]) + 1j * np.random.randn(self.Nr, self._s.shape[1]))

        # Store the noise.
        self._w = w

    def simulate(self):
        """ 
        Simulete the channel operation: transmit the precoded symbols through the MIMO channel and add noise.
        The output signal is ready to be processed by the receiver.
        
        Returns:
            _r (numpy.ndarray): The output signal. This is the received data symbol sequences after passing through the MIMO channel and adding noise.
        """

        # Generate the noise based on the specified SNR.
        self.generate_noise()

        # Transmit the precoded symbols through the MIMO channel and add noise.
        self._r = np.dot(self._H, self._s) + self._w

        # Return the output signal.
        return self._r


class Receiver:
    """
    ## Description
    ...
    
    ## Attributes
    ...
    
    ## Methods
    ...
    """

    def __init__(self, Nr, constellation_size, constellation_type, U, S):
        """ Initialize the receiver. """

        # The receiver parameters.
        self.Nr = Nr

         # The necessary communication system parameters.
        self.M = constellation_size
        self.type = constellation_type

        # The channel state information.
        self._U = U
        self._S = S

        # The receiver signals.
        self._r = None
        self._y = None
        self._estimation_var = None
        self._symbols_hat = None
        self._bits_hat = None

    def __str__(self):
        """ Return a string representation of the receiver object. """
        return f"Receiver: \n  - Number of antennas = {self.Nr}\n  - Constellation = {self.M}-{self.type}"

    def __call__(self):
        """ Allow the receiver object to be called as a function. When called, it executes the simulate() method. """
        return self.simulate()


    def set_r(self, r):
        """ Set the input signal received from the channel. """
        self._r = r

    def postcoder(self):
        """ Postcode the received symbols using the left singular vectors of the channel matrix H and store them. """
        self._y = np.dot(self._U.conj().T, self._r)
    
    def equalizer(self):
        """ Equalize the postcoded symbols using the singular values of the channel matrix H and store them. """
        self._estimation_var = np.zeros((self.Nr, self._r.shape[1]), dtype=complex)
        self._estimation_var[:self._S.shape[0], :] = self._y[:self._S.shape[0], :] / self._S[:, np.newaxis]

    def estimator(self):
        """ Convert the received (equalized postcoded) data symbols into the most probable data symbols and store them. """
        
        # Initialize the constellation points.

        if self.type == 'PAM':
            constellation = np.linspace(-(self.M-1), self.M-1, self.M) * (np.sqrt(12/(self.M**2-1)) / 2)
            
        elif self.type == 'PSK':
            constellation = np.exp(1j * 2*np.pi * np.arange(self.M) / self.M)

        elif self.type == 'QAM':
            if (math.log2(self.M) % 2 != 0): raise ValueError('The constellation size M is invalid. For QAM Modulation, M must be a power of 4 (e.g., 4, 16, 64).')
            real_parts = np.linspace(-(math.sqrt(self.M)-1), math.sqrt(self.M)-1) * (np.sqrt(6/(self.M-1)) / 2)
            imaginary_parts = np.linspace(-(math.sqrt(self.M)-1), math.sqrt(self.M)-1) * (np.sqrt(6/(self.M-1)) / 2)
            real_grid, imaginary_grid = np.meshgrid(real_parts, imaginary_parts)
            constellation = (real_grid + 1j*imaginary_grid).flatten()
            
        else :
            raise ValueError('The constellation type is invalid. Choose between "PAM", "PSK", or "QAM".')
        
        # Find the most probable data symbols by minimizing the Euclidean distance between the received symbols and the constellation points.
        distances = np.abs(constellation[:, np.newaxis] - self._estimation_var.flatten())
        symbols_hat = constellation[np.argmin(distances, axis=0)]

        # Store the estimated data symbol sequences.
        self._symbols_hat = symbols_hat.reshape((self.Nr, self._r.shape[1])) if self._r.ndim != 1 else symbols_hat.flatten()

    def demapper(self):
        """ Convert the data symbol sequences into bit sequences according to the specified modulation constellation. The demapper is the inverse operation of the mapper. """

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

    def simulate(self):
        """
        Simulate the receiver operations: postcode the received symbols, equalize them, estimate the transmitted symbols, and demap them to bits.
        The estimated bits are the output of the receiver.

        Returns:
            _bits_hat (numpy.ndarray): The output bit sequences. This is the estimated bit sequences after processing the received symbols.
        """

        # Simulate the receiver operations.
        self.postcoder()
        self.equalizer()
        self.estimator()
        self.demapper()

        # Return the output bit sequences.
        return self._bits_hat


class SuMimoSVD:
    """
    ## Description
    ...

    ## Attributes
    ...

    ## Methods
    ...
    """

    def __init__(self, Nt, Nr, constellation_size, constellation_type, SNR, Nbits=64, bits=None, H=None):
        """ Initialize the communication system parameters. """

        # The necessary communication system parameters.
        self.Nt = Nt
        self.Nr = Nr
        self.M = constellation_size
        self.type = constellation_type
        self.SNR = SNR
        
        # The channel, transmitter, and receiver of the communication system.
        self.channel = Channel(self.Nt, self.Nr, self.SNR, H=H)
        self.transmitter = Transmitter(self.Nt, self.M, self.type, self.channel.get_CSI()[3])
        self.receiver = Receiver(Nr, self.M, self.type, self.channel.get_CSI()[1], self.channel.get_CSI()[2])

        # The communication system signals.
        self._bits = np.random.randint(0, 2, (self.Nt, Nbits)) if bits is None else bits
        self._s = None
        self._r = None
        self._bits_hat = None

    def __str__(self):
        """ Return a string representation of the communication system object. """
        return f"SU-MIMO SVD Communication System: \n  - Number of transmitting and receiving antennas = {self.Nt} and {self.Nr}\n  - Constellation = {self.M}-{self.type}\n  - SNR = {self.SNR} dB\n  - Channel Matrix H = {'Provided' if self.channel._H is not None else 'i.i.d. complex Gaussian (0 mean, 1 variance) variables.'}"

    def __call__(self):
        """ Allow the communication system object to be called as a function. When called, it executes the simulate() method. """
        return self.simulate()

    def simulate(self):
        """
        Simulate the communication system: generate the bit sequences, transmit them through the MIMO channel, and estimate the transmitted bits at the receiver.
        Store the internal signals.
        """

        self.transmitter.set_bits(self._bits)
        self._s = self.transmitter()

        self.channel.set_s(self._s)
        self._r = self.channel()

        self.receiver.set_r(self._r)
        self._bits_hat = self.receiver()

