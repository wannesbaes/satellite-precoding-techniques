# This module contains the implementation of the SU-MIMO communication system.

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

import transmitter as tx
import channel as ch
import receiver as rx


class SuMimoSVD:
    """
    Attributes
    ----------
    Nt : int
        Number of transmit antennas.
    Nr : int
        Number of receive antennas.
    c_type : str
        Constellation type. (Choose between 'PAM', 'PSK', or 'QAM'.)
    Pt : float, optional
        Total transmit power (in W). Default is 1.0.
    B : float, optional
        Bandwidth of the communication system (in Hz). Default is 0.5.
    RAS : dict, optional
        Resource allocation settings. See the documentation of resource_allocation() method in the Transmitter class for more details about these strategy settings.
    SNR : float, optional
        Signal-to-noise ratio (in dB). Default is infinity (no noise).
    H : 2D numpy array (dtype: complex, shape: (Nr, Nt)), optional
        Channel matrix. If None, a random complex Gaussian (CS, zero-mean, unit-variance) channel matrix is generated. Default is None.
    
    Methods
    -------
    __init__()
        Initialize the SU-MIMO SVD digital communication system with the given parameters.
    __str__()
        Return a string representation of the SU-MIMO SVD digital communication system.
    """
    
    # INITIALIZATION AND REPRESENTATION

    def __init__(self, Nt, Nr, c_type, Pt=1.0, B=0.5, RAS={}, SNR=np.inf, H=None):
        """
        Initialize the SU-MIMO SVD digital communication system.

        Parameters
        ----------
        Nt : int
            Number of transmit antennas.
        Nr : int
            Number of receive antennas.
        c_type : str
            Constellation type. (Choose between 'PAM', 'PSK', or 'QAM'.)
        Pt : float, optional
            Total available transmit power (in W). Default is 1.0.
        B : float, optional
            Bandwidth of the communication system (in Hz). Default is 0.5.
        RAS : dict, optional
            Resource allocation settings. See the documentation of resource_allocation() method in the Transmitter class for more details.
        SNR : float, optional
            Signal-to-noise ratio (in dB). Default is infinity (no noise).
        H : 2D numpy array (dtype: complex, shape: (Nr, Nt)), optional
            Channel matrix. If None, a random complex Gaussian (CS, zero-mean, unit-variance) channel matrix is generated. Default is None.
        """

        # System settings.
        self.Nt = Nt
        self.Nr = Nr
        self.c_type = c_type

        self.Pt = Pt
        self.B = B

        self.RAS = RAS if RAS != {} else { 'power allocation': 'optimal', 'bit allocation': 'adaptive', 'data rate': 1.0, 'constellation sizes': None, 'control channel': True}

        # System components.
        self.transmitter = tx.Transmitter(Nt, c_type, Pt, B, self.RAS)
        self.channel = ch.Channel(Nt, Nr, SNR, H)
        self.receiver = rx.Receiver(Nr, c_type, Pt, B, self.RAS)

    def __str__(self):
        """ String representation of the SU-MIMO DigCom system. """
        return f'{self.Nt}x{self.Nr} {self.c_type} SU-MIMO SVD DigCom System'


    # FUNCTIONALITY

    def set_CSI(self, SNR=None, H=None):
        """
        Set the channel state information (CSI) of the SU-MIMO SVD digital communication system. Also, the resource allocation at the transmitter and receiver is updated accordingly, so no invalid configurations remain.\n
        If no new values are provided, the current values are kept.

        Parameters
        ----------
        SNR : float, optional
            Signal-to-noise ratio (in dB).
        H : 2D numpy array (dtype: complex, shape: (Nr, Nt)), optional
            Channel matrix.
        """
        
        # Set the channel properties.
        self.channel.set_CSI(SNR, H)

        # Update the resource allocation at the transmitter and receiver.
        self.transmitter.resource_allocation(self.channel.get_CSI())
        self.receiver.resource_allocation(self.channel.get_CSI(), self.transmitter.get_CCI())

    def reset_CSI(self, SNR=None, H=None):
        """
        Reset the channel state information (CSI) of the SU-MIMO SVD digital communication system. Also, the resource allocation at the transmitter and receiver is updated accordingly, so no invalid configurations remain.\n
        If no new value is provided, the default initialization values are used. For the SNR value, the default is infinity (no noise). For the channel matrix, the default is a random i.i.d. complex circularly-symmetric Gaussian (zero mean, unit variance) MIMO channel.

        Parameters
        ----------
        SNR : float, optional
            Signal-to-noise ratio (in dB).
        H : 2D numpy array (dtype: complex, shape: (Nr, Nt)), optional
            Channel matrix.
        """
        
        # Reset the channel properties.
        self.channel.reset_CSI(SNR, H)

        # Update the resource allocation at the transmitter and receiver.
        self.transmitter.resource_allocation(self.channel.get_CSI())
        self.receiver.resource_allocation(self.channel.get_CSI(), self.transmitter.get_CCI())
    
    def set_RAS(self, RAS):
        """
        Set the resource allocation strategy of the SU-MIMO SVD digital communication system. Also, the resource allocation at the transmitter and receiver is updated accordingly, so no invalid configurations remain.\n
        If no new value for a certain setting is provided, the current value for that setting is kept.

        Parameters
        ----------
        RAS : dict
            Resource Allocation Strategy. See the documentation of resource_allocation() method in the Transmitter class for more details on these strategy settings.
        """

        # Update the resource allocation strategy.
        self.RAS |= RAS

        self.transmitter.set_RAS(self.RAS)
        self.receiver.set_RAS(self.RAS)

        # Update the resource allocation at the transmitter and receiver.
        self.transmitter.resource_allocation(self.channel.get_CSI())
        self.receiver.resource_allocation(self.channel.get_CSI(), self.transmitter.get_CCI())


    def simulate(self, bitstream):
        """
        Simulate the SU-MIMO SVD digital communication system for a given input bitstream. Return the reconstructed bitstream at the receiver.\n
        If no capacity is available on the channel, the transmission fails and None is returned.

        Parameters
        ----------
        bitstream : 1D numpy array (dtype: int, length: N_bits)
            Input - The bit stream.
        
        Returns
        -------
        bitstream_hat : 1D numpy array (dtype: int, length: N_bits)
            Output - The reconstructed bit stream.
        """

        # 1. Transmitter
        x = self.transmitter.simulate(bitstream, self.channel.get_CSI())
        if x is None: return None

        # 2. Channel
        y = self.channel.simulate(x)

        # 3. Receiver
        bitstream_hat = self.receiver.simulate(y, self.channel.get_CSI(), self.transmitter.get_CCI())

        return bitstream_hat[:len(bitstream)]
    
    def BERs_simulation(self, SNRs, num_errors=500, num_channels=100):
        """
        Simulate the SU-MIMO SVD digital communication system over a range of SNR values until a specified number of bit errors are reached. Also, average over a specified number of channel realizations.\n
        Return the simulated BERs for each SNR value, along with the average information bit rate and activation rate.
        
        Parameters
        ----------
        SNRs : 1D numpy array (dtype: float, length: N_SNRs)
            Input - The range of signal-to-noise ratio (SNR) values in dB to simulate over.
        num_errors : int
            The minimum number of bit errors that must be reached for each SNR value before stopping the simulation.
        num_channels : int
            The minimum number of different channel realizations to average over for each SNR value.
        
        Returns
        -------
        BERs : 1D numpy array (dtype: float, length: N_SNRs)
            Output - The simulated Bit Error Rates (BERs) corresponding to each SNR value.
        IBRs : 1D numpy array (dtype: int, length: N_SNRs)
            Output - The simulated information bit rates (IBRs) (bits per symbol vector) corresponding to each SNR value.
        ARs : 1D numpy array (dtype: float, length: N_SNRs)
            Output - The simulated activation rates (ARs) of the channel corresponding to each SNR value. (Indicates the fraction of channel realizations for which enough capacity was available to transmit data.)
        """
        
        def BER_simulation(bitstream):
            """
            Simulate the SU-MIMO SVD communication system for a given input bitstream. Return the Bit Error Rate (BER).\n
            If no capacity is available on the channel, NaN is returned.

            Parameters
            ----------
            bitstream : 1D numpy array (dtype: int, length: N_bits)
                Input - The bit stream.
            
            Returns
            -------
            BER : float
                Output - The Bit Error Rate (BER).
            """
            
            bitstream_hat = self.simulate(bitstream)
            BER = np.sum(bitstream_hat != bitstream) / bitstream.size if bitstream_hat is not None else np.nan

            return BER
        
        print(f"\n\nStarting BER simulation for \n{str(self)} ...\n")


        # Initialization.
        error_progress, channel_progress = 0, 0
        counted_bits = np.zeros(len(SNRs), dtype=int)
        counted_errors = np.zeros(len(SNRs), dtype=float)
        used_channels = np.zeros(len(SNRs), dtype=int)
        realized_channels = np.zeros(len(SNRs), dtype=int)
        IBRs = [[] for _ in range(len(SNRs))]


        # Iteration.
        while np.any(counted_errors < num_errors) or np.any(used_channels < num_channels):
            
            if (((np.min(counted_errors)/num_errors) >= error_progress + 0.10) and error_progress < 1.0) or (((np.min(used_channels)/num_channels) >= channel_progress + 0.10) and channel_progress < 1.0):
                error_progress, channel_progress = (np.min(counted_errors)/num_errors), (np.min(used_channels)/num_channels)
                print(f"\r    progress update...    errors: {error_progress:.0%}    channels: {channel_progress:.0%}")
            
            for i, SNR_idx in enumerate(np.where( (counted_errors < num_errors) | (used_channels < num_channels) )[0]):

                if i == 0: self.reset_CSI(SNR=SNRs[SNR_idx])
                else: self.set_CSI(SNR=SNRs[SNR_idx])

                num_bits = 4800
                bitstream = np.random.randint(0, 2, size=num_bits)
                
                BER = BER_simulation(bitstream)
                IBR = np.sum(np.log2(self.transmitter._Mi))

                
                realized_channels[SNR_idx] += 1
                
                IBRs[SNR_idx].append(IBR)
                
                if IBR == 0: continue
                counted_bits[SNR_idx] += num_bits
                counted_errors[SNR_idx] += num_bits*BER
                used_channels[SNR_idx] += 1
        

        # Termination.
        BERs = counted_errors / counted_bits
        IBRs = np.array([np.mean(np.array(IBRs[i], dtype=float)) for i in range(len(SNRs))])
        ARs = used_channels / realized_channels
        
        return BERs, IBRs, ARs

    def BERs_eigenchs_simulation(self, SNRs, num_errors=500, num_channels=100):
        """
        Simulate the SU-MIMO SVD digital communication system over a range of SNR values until a specified number of bit errors are reached. Also, average over a specified number of channel realizations.\n
        Return the simulated BERs of each eigenchannel (!) for every SNR value.

        Parameters
        ----------
        SNRs : 1D numpy array (dtype: float, length: N_SNRs)
            Input - The range of signal-to-noise ratio (SNR) values in dB to simulate over.
        num_errors : int
            The minimum number of bit errors that must be reached in at least one eigenchannel for each SNR value before stopping the simulation.
        num_channels : int
            The minimum number of different channel realizations in at least one eigenchannel for each SNR value to average over.
        
        Returns
        -------
        BERs : 2D numpy array (dtype: float, shape: (min(Nt, Nr), N_SNRs))
            Output - The simulated Bit Error Rates (BERs) of each eigenchannel, corresponding to each SNR value.
        IBRs : 2D numpy array (dtype: float, shape: (min(Nt, Nr), N_SNRs))
            Output - The simulated information bit rates (IBRs) (bits per symbol) of each eigenchannel, corresponding to each SNR value.
        ARs : 1D numpy array (dtype: float, length: N_SNRs)
            Output - The simulated activation rates (ARs) of every eigenchannel corresponding to each SNR value. (Indicates the fraction of channel realizations for which enough capacity was available to transmit data throught that eigenchannel.)
        """
        
        def BER_eigenchs_simulation(bitstream):
            """
            Simulate the SU-MIMO SVD digital communication system for a given input bitstream. Return the Bit Error Rate (BER) of each eigenchannel.
            If no capacity is available on the eigenchannel, np.NaN is returned for that eigenchannel.

            Parameters
            ----------
            bitstream : 1D numpy array (dtype: int, length: N_bits)
                Input - The bit stream.
            
            Returns
            -------
            BER_eigenchs : 1D numpy array (dtype: float, length: min(Nt, Nr))
                Output - The Bit Error Rates (BERs) of each eigenchannel.
            """
            
            # Simulate the system.
            bitstream_hat = self.simulate(bitstream)
            if bitstream_hat is None: return np.array([np.nan]*min(self.Nt, self.Nr))

            # Allocate bits to the eigenchannels.
            bitstream_eigenchs = self.transmitter.bit_allocator(bitstream)
            bitstream_hat_eigenchs = self.transmitter.bit_allocator(bitstream_hat)

            # Calculate the BER of each eigenchannel.
            BER_eigenchs = np.array([ ( (np.sum(bitstream_hat_eigenchs[i] != bitstream_eigenchs[i]) / bitstream_eigenchs[i].size) if bitstream_eigenchs[i].size > 0 else np.nan ) for i in range(min(self.Nt, self.Nr)) ])

            return BER_eigenchs
        
        print(f"\nStarting eigenchannel BER simulation for \n{str(self)} ...")


        # Initialization.
        error_progress, channel_progress = 0, 0
        counted_bits = np.zeros((min(self.Nt, self.Nr), len(SNRs)), dtype=float)
        counted_errors = np.zeros((min(self.Nt, self.Nr), len(SNRs)), dtype=float)
        used_channels = np.zeros((min(self.Nt, self.Nr), len(SNRs)), dtype=int)
        realized_channels = np.zeros((min(self.Nt, self.Nr), len(SNRs)), dtype=int)
        IBRs = [ [[] for _ in range(len(SNRs))] for _ in range(min(self.Nt, self.Nr)) ]


        # Iteration.
        while np.any(counted_errors[0, :] < num_errors) or np.any(used_channels[0, :] < num_channels):

            if (((np.min(counted_errors[0, :])/num_errors) >= error_progress + 0.10) and error_progress < 1.0) or (((np.min(used_channels[0, :])/num_channels) >= channel_progress + 0.10) and channel_progress < 1.0):
                error_progress, channel_progress = (np.min(counted_errors[0, :])/num_errors), (np.min(used_channels[0, :])/num_channels)
                print(f"\r    progress update...    errors: {error_progress:.0%}    channels: {channel_progress:.0%}")

            for i, SNR_idx in enumerate(np.where( (counted_errors[0, :] < num_errors) | (used_channels[0, :] < num_channels) )[0]):

                if i == 0: self.reset_CSI(SNR=SNRs[SNR_idx])
                else: self.set_CSI(SNR=SNRs[SNR_idx])

                num_bits = 2400
                bitstream = np.random.randint(0, 2, size=num_bits)
                
                BERi = BER_eigenchs_simulation(bitstream)
                IBRi = np.pad( np.log2(self.transmitter._Mi), pad_width=(0, min(self.Nt, self.Nr) - len(self.transmitter._Mi)), mode='constant', constant_values=0 )
                

                realized_channels[:, SNR_idx] += 1
                
                for eigench_idx in range(min(self.Nt, self.Nr)): 
                    IBRs[eigench_idx][SNR_idx].append(IBRi[eigench_idx])

                if np.sum(IBRi) == 0: continue
                num_bits_i = np.ceil(num_bits * (IBRi[IBRi > 0] / np.sum(IBRi)))
                counted_bits[(IBRi > 0), SNR_idx] += num_bits_i
                counted_errors[(IBRi > 0), SNR_idx] += num_bits_i*BERi[IBRi > 0]
                used_channels[(IBRi > 0), SNR_idx] += (IBRi > 0)[IBRi > 0]
        

        # Termination.
        BERs = np.divide(counted_errors, counted_bits, out=np.full((min(self.Nt, self.Nr), len(SNRs)), np.nan, dtype=float), where=(counted_bits != 0))
        IBRs = np.array([ [np.mean(np.array(IBRs[eigench_idx][SNR_idx], dtype=float)) for SNR_idx in range(len(SNRs))] for eigench_idx in range(min(self.Nt, self.Nr)) ])
        ARs = used_channels / realized_channels
        
        return BERs, IBRs, ARs

    def BERs_analytical(self, SNRs, num_channels=100, settings={'mode': 'approximation', 'eigenchannels': False}):
        """
        Calculate an analytical approximation or upper bound of the Bit Error Rate (BER) over a range of SNR values, for every eigenchannel. Average the result over a specified number of channel realizations.\n

        Return the analytical BERs for each SNR value, along with the information bit rates and activation rates.
        Depending on the settings, either the BERs of every eigenchannel are returned separately, or the weighted average BER over all eigenchannels is returned. 

        Parameters
        ----------
        SNRs : 1D numpy array (dtype: float, length: N_SNRs)
            The range of signal-to-noise ratio (SNR) values in dB to calculate the analytical BER over.
        num_channels : int
            The number of different channel realizations to average over.
        settings : dict
            The calculation settings.
                - mode: (str)
                    'upper bound' or 'approximation'. For more information on the calculation method, see the documentation under the section 'Bit Error Rate Calculation'.
                - eigenchannels: (bool)
                    If True, return the BERs for every eigenchannel separately (outputs are 2D arrays). If False, return the weighted average BER over all eigenchannels (outputs are 1D arrays).
        
        Returns
        -------
        BERs : 1D/2D numpy array (dtype: float, shape: (min(Nt, Nr), N_SNRs))
            Output - The analytical Bit Error Rates (BERs) corresponding to each SNR value, whether or not for every eigenchannel.
        IBRs : 1D/2D numpy array (dtype: float, shape: (min(Nt, Nr), N_SNRs))
            Output - The information bit rates (IBRs) (bits per symbol vector) corresponding to each SNR value, whether or not for every eigenchannel.
        ARs : 1D/2D numpy array (dtype: float, shape: (min(Nt, Nr), N_SNRs))
            Output - The activation rates (ARs) of the channel corresponding to each SNR value, whether or not for every eigenchannel. (Indicates the fraction of channel realizations for which enough capacity was available to transmit data.)
        """
    
        def BER_analytical(mode):
            """
            Calculate an analytical approximation or upper bound of the Bit Error Rate (BER) of each eigenchannel for the current channel realization and the current SNR. 
            Return the analytical BERs, along with the information bit rates of each eigenchannel.

            Parameters
            ----------
            mode: str
                'upper bound' or 'approximation'. For more information on the calculation method, see the documentation under the section 'Bit Error Rate Calculation'.
            
            Returns
            -------
            BERi : 1D numpy array (dtype: float, length: min(Nt, Nr))
                Output - The analytical Bit Error Rates (BERs) of each eigenchannel.
            IBRi : 1D numpy array (dtype: float, length: min(Nt, Nr))
                Output - The information bit rates (IBRs) (bits per symbol) of each eigenchannel.
            """

            def demap(symbols_hat, M, c_type):
                """
                Convert detected data symbols into the corresponding bits according to the specified modulation constellation.

                Parameters
                ----------
                symbols_hat : 1D numpy array (dtype: complex, length: num_symbols)
                    Input - detected data symbols.
                M : int
                    Constellation size.
                c_type : str
                    Constellation type. (Choose between 'PAM', 'PSK', 'QAM'.)
                
                Returns
                -------
                bits_hat : 1D numpy array (dtype: int, length: num_symbols * log2(M))
                    Output - reconstructed bits.
                """

                # 1. Convert the data symbols to the corresponding decimal values, according to the specified constellation type.

                if c_type == 'PAM':
                    delta = np.sqrt(3/(M**2-1))
                    decimals = np.round((1/2) * (symbols_hat / delta + (M-1))).astype(int)
                
                elif c_type == 'PSK':
                    phases = np.angle(symbols_hat) % (2*np.pi)
                    decimals = np.round((phases * M) / (2 * np.pi)).astype(int)
                
                elif c_type == 'QAM':
                    c_sqrtM_PAM = np.arange(-(np.sqrt(M)-1), (np.sqrt(M)-1) + 1, 2) * np.sqrt(3 / (2*(M-1)))
                    real_grid, imaginary_grid = np.meshgrid(c_sqrtM_PAM, c_sqrtM_PAM[::-1])
                    constellation = (real_grid + 1j*imaginary_grid)
                    constellation[1::2] = constellation[1::2, ::-1]
                    constellation = constellation.flatten()

                    sort_idx = np.argsort(constellation)
                    pos = np.searchsorted(constellation[sort_idx], symbols_hat)
                    decimals = sort_idx[pos]

                else:
                    raise ValueError(f'The constellation type is invalid.\nChoose between "PAM", "PSK", or "QAM". Right now, type is {c_type}.')
                

                # 2. Convert the decimal values to the corresponding blocks of m bits in gray code.

                m = int(np.log2(M))
                binarycodes = ((decimals[:, None].astype(int) & (1 << np.arange(m))[::-1].astype(int)) > 0).astype(int)

                graycodes = np.zeros_like(binarycodes)
                graycodes[:, 0] = binarycodes[:, 0]
                for i in range(1, m):
                    graycodes[:, i] = binarycodes[:, i] ^ binarycodes[:, i - 1]


                # 3. Convert the gray code blocks to a single bit sequence and return it.

                bits_hat = graycodes.flatten()
                return bits_hat

            def construct_constellation(M, c_type):

                if c_type == 'PAM':
                    constellation = np.arange(-(M-1), (M-1) + 1, 2) * np.sqrt(3/(M**2-1))

                elif c_type == 'PSK':
                    constellation = np.exp(1j * 2*np.pi * np.arange(M) / M)

                elif c_type == 'QAM':
                    c_sqrtM_PAM = np.arange(-(np.sqrt(M)-1), (np.sqrt(M)-1) + 1, 2) * np.sqrt(3 / (2*(M-1)))
                    real_grid, imaginary_grid = np.meshgrid(c_sqrtM_PAM, c_sqrtM_PAM)
                    constellation = (real_grid + 1j*imaginary_grid).ravel()

                else :
                    raise ValueError(f'The constellation type is invalid.\nChoose between "PAM", "PSK", or "QAM". Right now, type is {c_type}.')
            
                return constellation

            def Q(x):
                """ Implementation of the Q-function. The tail probability of standard normal distribution: Q(x) = P[X > x] with X ~ N(0,1). """
                return 0.5 * sp.special.erfc(x / np.sqrt(2))

            def N(c, M, c_type):
                """ Return the Hamming distance between any two constellation points. Cartesian product of the constellation c is created in the same way as in d(). """
                b = (demap(c, M, c_type)).reshape(len(c), -1)
                i, j = np.meshgrid(np.arange(len(c)), np.arange(len(c)))
                N = (np.sum(np.abs(b[i] - b[j]), axis=2)).ravel()
                return N

            def d(c):
                """ Return the Euclidean distance between any two constellation points. Cartesian product of the constellation c is created in the same way as in N(). """
                c1, c2 = np.meshgrid(c, c)
                d = np.abs((c1 - c2).ravel())
                return d

            def K(Mi, c_type):
                """ Calculate the average number of nearest neighbors K of a given constellation. """
                if c_type == 'PAM': return 2*(Mi-1) / Mi
                elif c_type == 'PSK': return np.where(Mi == 2, 1, 2)
                elif c_type == 'QAM': return 4*(np.sqrt(Mi)-1) / np.sqrt(Mi)
                else: raise ValueError(f'The constellation type is invalid.\nChoose between "PAM", "PSK", or "QAM". Right now, type is {c_type}.')
            
            def d_min(Mi, c_type):
                """ Calculate the distance between the nearest neighbors of a given constellation. """
                if c_type == 'PAM': return np.sqrt(12 / (Mi**2 - 1))
                elif c_type == 'PSK': return 2 * np.abs( np.sin(np.pi / Mi) )
                elif c_type == 'QAM': return np.sqrt(6 / (Mi - 1))
                else: raise ValueError(f'The constellation type is invalid.\nChoose between "PAM", "PSK", or "QAM". Right now, type is {c_type}.')
            
            # Initialization.
            CSI = self.channel.get_CSI()
            S = CSI['S']
            N0 = self.Pt / ((10**(CSI['SNR']/10.0)) * 2*self.B)

            self.transmitter.resource_allocation(CSI)
            Mi = self.transmitter.get_CCI()['Mi']
            Pi = self.transmitter.get_CCI()['Pi'][:len(Mi)]

            # Edge case: No data is could be transmitted due to lack of capacity.
            if len(Mi) == 0: return np.full(min(self.Nt, self.Nr), np.nan), np.zeros(min(self.Nt, self.Nr), dtype=int)
            

            # Mode 1: Approximation.
            if mode == 'approximation':
                Ki = K(Mi, self.c_type)
                dmin = d_min(Mi, self.c_type)
                BERi = (Ki / np.log2(Mi)) * Q( (S[:len(Mi)] * np.sqrt(Pi / (2*N0))) * dmin )

            # Mode 2: Upper Bound.
            elif mode == 'upper bound':
                BERi = np.empty(len(Mi))
                for i in range(len(Mi)):
                    c = construct_constellation(Mi[i], self.c_type)
                    N_value = N(c, Mi[i], self.c_type)
                    Q_value = Q( (S[i] * np.sqrt(Pi[i] / (2*N0))) * d(c) )
                    BERi[i] = (1 / (Mi[i]*np.log2(Mi[i]))) * np.sum( N_value * Q_value )

            # Invalid Mode.
            else: 
                raise ValueError(f'The mode is invalid.\nChoose between "upper bound" or "approximation". Right now, mode is {mode}.')
            

            # Termination.
            BERi = np.pad(BERi, pad_width=(0, min(self.Nt, self.Nr) - len(BERi)), mode='constant', constant_values=np.nan)
            IBRi = np.pad(np.log2(Mi).astype(int), pad_width=(0, min(self.Nt, self.Nr) - len(Mi)), mode='constant', constant_values=0)
            return BERi, IBRi

        print("\nStarting " + ("eigenchannel" if settings['eigenchannels'] else "") + f" BER {settings['mode']} calculation for \n{str(self)} ...")


        # Initialization.
        BERs = np.empty((min(self.Nt, self.Nr), len(SNRs), num_channels), dtype=float)
        IBRs = np.empty((min(self.Nt, self.Nr), len(SNRs), num_channels), dtype=float)


        # Iteration.
        for channel_idx in tqdm(range(num_channels)):

            for SNR_idx, SNR in enumerate(SNRs):

                if SNR_idx == 0: self.reset_CSI(SNR=SNR)
                else: self.set_CSI(SNR=SNR)

                BERi, IBRi = BER_analytical(settings['mode'])
                BERs[:, SNR_idx, channel_idx] = BERi
                IBRs[:, SNR_idx, channel_idx] = IBRi
        

        # Termination.
        ARs = np.sum( IBRs > 0, axis=2 ) / num_channels
        BERs = np.nanmean(BERs, axis=2)
        IBRs = np.mean(IBRs, axis=2)
        
        if not settings['eigenchannels']:
            BERs = np.nansum( (IBRs / np.where(IBRs[0]==0, np.nan, np.sum(IBRs, axis=0))) * BERs, axis=0 )
            IBRs = np.sum(IBRs, axis=0)
            ARs = ARs[0]

        return BERs, IBRs, ARs


    # TESTS AND PLOTS

    def plot_scatter_diagram(self, K=100):
        """
        Plot a scatter diagram of the received symbol vectors.\n
        Every distinct transmitted symbol is represented by a different color. Every used eigenchannel (antenna) has its own subplot. The amount of symbol vectors to consider is specified by K.

        Parameters
        ----------
        K : int
            The amount of symbol vectors to consider for the scatter diagram.
        
        Returns
        -------
        plot : matplotlib figure
            The created scatter diagram plot.
        """
        
        def generate_constellation(M, c_type):
            """
            Description
            -----------
            Generate the constellation points for a given constellation size (M) and type (c_type). 

            Parameters
            ----------
            M : int
                Input - Constellation size.
            c_type : str
                Input - Constellation type. (Choose between 'PAM', 'PSK', or 'QAM'.)
            
            Returns
            -------
            constellation : 1D numpy array (dtype: complex, length: M)
                Output - Constellation points.
            """

            if c_type == 'PAM':
                constellation = np.arange(-(M-1), (M-1) + 1, 2) * np.sqrt(3/(M**2-1))
            
            elif c_type == 'PSK':
                constellation = np.exp(1j * 2*np.pi * np.arange(M) / M)

            elif c_type == 'QAM':
                c_sqrtM_PAM = np.arange(-(np.sqrt(M)-1), (np.sqrt(M)-1) + 1, 2) * np.sqrt(3 / (2*(M-1)))
                real_grid, imaginary_grid = np.meshgrid(c_sqrtM_PAM, c_sqrtM_PAM)
                constellation = (real_grid + 1j*imaginary_grid)
                constellation[1::2] = constellation[1::2, ::-1]
                constellation = constellation.flatten()

            else :
                raise ValueError(f'The constellation type is invalid.\nChoose between "PAM", "PSK", or "QAM". Right now, type is {c_type}.')

            return constellation

        def generate_colormaps(Mi, c_type):
            """
            Description
            -----------
            Generate a color map for each used eigenchannel that allocates a unique color to every constellation point.

            Parameters
            ----------
            Mi : 1D numpy array (dtype: int, length: min(Nt, Nr))
                Input - Constellation sizes used on each eigenchannel.
            c_type : str
                Input - Constellation type. (Choose between 'PAM', 'PSK', or 'QAM'.)
            
            Returns
            -------
            colormaps : 1D numpy array (dtype: dict, length: min(Nt, Nr))
                Output - Colormaps, one for each eigenchannel. Each colormap is a dictionary that maps every constellation point index (key) to a unique color (value).
            """
            
            colormaps = []
            
            for M in Mi:
                colors = plt.get_cmap('turbo', M).colors
                constellation = generate_constellation(M, c_type)
                eigenchannel_colormaps = {constellation[i]: colors[i] for i in range(M)}
                colormaps.append(eigenchannel_colormaps)

            return np.array(colormaps)
        
        def scatter_simulate(K):
            """
            Description
            -----------
            Simulate the SU-MIMO SVD digital communication system for K symbol vectors with a random input bitstream, with the purpose of obtaining the transmitted and received symbols for the scatter diagram. 

            Parameters
            ----------
            K : int, optional
                Input - The number of symbol vectors to consider for illustration (equal to the number of points in each scatter diagram).
            
            Returns
            -------
            a : 2D numpy array (dtype: complex, shape: (min(Nt, Nr), K))
                Output - The data symbols (before power allocation and precoding, so this are constellation points).
            r_prime : 2D numpy array (dtype: complex, shape: (min(Nt, Nr), K))
                Output - The received data symbols (after combining).
            """
            
            # Transmitter Setup.
            self.transmitter.resource_allocation(self.channel.get_CSI())
            total_bits = K * np.sum( np.log2(self.transmitter.get_CCI()['Mi']) )
            bitstream = np.random.randint(0, 2, size=int(total_bits))
            if total_bits == 0: return None, None

            # 1. Simulate the transmitter operations.
            b = self.transmitter.bit_allocator(bitstream)
            a = self.transmitter.mapper(b)
            s_prime = self.transmitter.power_allocator(a)
            x = self.transmitter.precoder(s_prime, self.channel.get_CSI()['Vh'])

            # 2. Simulate the channel operations.
            y = self.channel.simulate(x)

            # 3. Simulate the receiver operations untill the combiner.
            r_prime = self.receiver.combiner(y, self.channel.get_CSI()['U'])

            # Return
            return a, r_prime

        # Similate part of the system to obtain the transmitted data symbols (before power allocation and precoding) and the received data symbols (after combining).
        a, r_prime = scatter_simulate(K)

        # Retrieve the CSI and CCI.
        SNR = self.channel.get_CSI()['SNR']
        S = self.channel.get_CSI()['S']
        Mi = self.transmitter.get_CCI()['Mi']
        Pi = self.transmitter.get_CCI()['Pi']

        # Initialize the plot. Build the colormaps.
        fig, axes = plt.subplots(1, len(Mi), figsize=(6*len(Mi), 6))
        colormaps = generate_colormaps(Mi, self.c_type)

        # Create a scatter diagram for every eigenchannel.
        for eigenchannel in range(len(Mi)):
            
            ax = axes[eigenchannel] if len(Mi) > 1 else axes
            
            # Plot the received symbols for the current eigenchannel with colors based on the transmitted symbols.
            colors = [colormaps[eigenchannel][symbol] for symbol in a[eigenchannel]]
            ax.scatter(r_prime[eigenchannel].real, r_prime[eigenchannel].imag, color=colors, marker='.', alpha= 0.75, s=50)

            # Plot the constellation points for reference.
            constellation_points = generate_constellation(Mi[eigenchannel], self.c_type)
            colors = [colormaps[eigenchannel][point] for point in constellation_points]
            constellation_points = constellation_points * (S[eigenchannel] * np.sqrt(Pi[eigenchannel]))
            ax.scatter(constellation_points.real, constellation_points.imag, color=colors, edgecolor='black', marker='X', alpha= 1.0, s=50)

            # Plot settings.
            ax.axhline(0, color='black', linewidth=0.5)
            ax.axvline(0, color='black', linewidth=0.5)
            ax.set_title(f'Eigenchannel {eigenchannel+1}: {Mi[eigenchannel]}-{self.c_type}')
            ax.set_xlabel('Real Part')
            ax.set_ylabel('Imaginary Part')

            def set_ticks_and_grid(constellation_points, ax):
                """ Set a grid on the scatter diagram based on the constellation points. """
                
                x_cpoints = np.sort(np.unique(constellation_points.real))
                x_midpoints = (x_cpoints[:-1] + x_cpoints[1:]) / 2
                x_ticks = np.empty(len(x_cpoints) + len(x_midpoints))
                x_ticks[0::2] = x_cpoints
                x_ticks[1::2] = x_midpoints
                
                ax.set_xticks(x_ticks, minor=True)
                ax.set_xticks(np.arange(np.floor(x_ticks.min()), np.ceil(x_ticks.max()) + 1))

                y_cpoints = np.sort(np.unique(constellation_points.imag))
                y_midpoints = (y_cpoints[:-1] + y_cpoints[1:]) / 2
                y_ticks = np.empty(len(y_cpoints) + len(y_midpoints))
                y_ticks[0::2] = y_cpoints
                y_ticks[1::2] = y_midpoints

                ax.set_yticks(y_ticks, minor=True)
                ax.set_yticks(np.arange(np.floor(y_ticks.min()), np.ceil(y_ticks.max()) + 1))

                ax.tick_params(which='minor', length=0)
                ax.grid(True, which='minor', linestyle='--', alpha=0.7)
            
            set_ticks_and_grid(constellation_points, ax)
            ax.axis('equal')

        # Overall plot settings.
        #fig.suptitle(f'{str(self)}' + f'\n\nScatter Diagram after SVD Processing' + f'\nSNR: {SNR} dB & ' + f'data rate: {round(self.RAS['data rate']*100)}%')
        fig.tight_layout()
        fig.savefig(f"su-mimo/report/plots/1_simulation/scatter_plots/2__{self.Nt}x{self.Nr}_{self.c_type}__SNR_{SNR}__pa__{self.RAS['power allocation']}" + (f"__ba_adaptive__R_{round(self.RAS['data rate']*100)}" if self.RAS['bit allocation'] == 'adaptive' else f"__ba_fixed__M_{(np.log2(self.RAS['constellation sizes'])).astype(int)}") + datetime.now().strftime("__%Y%m%d_%H%M%S") + ".png", dpi=300, bbox_inches="tight")
    
        # Return the plot.
        return fig, axes

    def print_simulation_example(self, bitstream, K=1):
        """ 
        Print a step-by-step example of the transmitter, channel and receiver operations (see simulate() methods) for a given input bitstream. Only the first K data symbols vectors are considered for illustration.

        Parameters
        ----------
        bitstream : 1D numpy array (dtype: int, length: N_bits)
            Input - The bit stream.
        K : int, optional
            The maximum number of symbol vectors to consider for illustration.
        
        Notes
        -----
        For demonstration purposes only.
        """
        
        print(f"\nStarting Simulation Example for \n{str(self)} ...\n\n")

        x = self.transmitter.print_simulation_example(bitstream, self.channel.get_CSI(), K)
        y = self.channel.print_simulation_example(x, K)
        bits_hat = self.receiver.print_simulation_example(y, self.channel.get_CSI(), self.transmitter.get_CCI(), K)

        min_len = min(len(bitstream), len(bits_hat))
        BER = np.sum(bitstream[:min_len] != bits_hat[:min_len]) / bitstream.size
        print(f"\n\n========== Final Result ==========\n Bit Error Rate (BER): {BER}\n\n\n")

        return
