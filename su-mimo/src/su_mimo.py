# This module contains the implementation of the SU-MIMO communication system.

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba, to_hex
from matplotlib.collections import LineCollection

from . import transmitter as tx
from . import channel as ch
from . import receiver as rx


class SuMimoSVD:
    """
    Description
    -----------
    A single-user multiple-input multiple-output (SU-MIMO) digital communication system, in which the channel state information is available at both the transmitter and receiver.
    
    The communication system consists of a transmitter, a flat-fading MIMO channel, and a receiver. 
    The singular value decomposition (SVD) of the channel matrix is used for precoding at the transmitter and combining at the receiver.

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
    data_rate : float, optional
        Target data rate, as a fraction of the channel capacity. Default is 100%.
    SNR : float, optional
        Signal-to-noise ratio (in dB). Default is infinity (no noise).
    H : 2D numpy array (dtype: complex, shape: (Nr, Nt)), optional
        Channel matrix. If None, a random complex Gaussian (CS, zero-mean, unit-variance) channel matrix is generated. Default is None.
    c_size : int, optional
        Constellation size. If None, adaptive modulation is used based on the channel conditions. Default is None.
    
    Methods
    -------
    __init__()
        Initialize the SU-MIMO SVD digital communication system with the given parameters.
    __str__()
        Return a string representation of the SU-MIMO SVD digital communication system.

    simulate()
        Simulate the SU-MIMO SVD digital communication system for a given input bitstream. Return the reconstructed bitstream.
    BERs_simulation()
        Simulate the SU-MIMO SVD digital communication system over a range of SNR values. Return the simulated BERs, information bit rates, and activation rates.
    BERs_eigenchs_simulation()
        Simulate the SU-MIMO SVD digital communication system over a range of SNR values. Return the simulated BERs, information bit rates, and activation rates for each eigenchannel separately.
    BERs_eigenchs_analytical()
        Calculate an analytical approximation or upper bound of the BER over a range of SNR values. Return the analytical BERs, information bit rates, and activation rates whether or not for each eigenchannel separately.
    
    plot_performance()
        Create the performance evaluation plots for the requested evaluation metrics.
    plot_scatter_diagram()
        Plot a scatter diagram of the received symbol vectors.
    print_simulation_example()
        Print an example simulation of the SU-MIMO SVD digital communication system. For demonstration purposes only.
    """
    
    # INITIALIZATION AND REPRESENTATION

    def __init__(self, Nt, Nr, c_type, Pt=1.0, B=0.5, data_rate=1.0, SNR=np.inf, H=None, c_size=None):

        self.Nt = Nt
        self.Nr = Nr

        self.c_type = c_type
        self.M = c_size

        self.Pt = Pt
        self.B = B
        self.data_rate = data_rate

        self.transmitter = tx.Transmitter(Nt, c_type, data_rate, Pt, B, c_size)
        self.channel = ch.Channel(Nt, Nr, SNR, H)
        self.receiver = rx.Receiver(Nr, c_type, data_rate, Pt, B, c_size)

    def __str__(self):
        """ String representation of the SU-MIMO DigCom system. """
        return f'{self.Nt}x{self.Nr} ' + (f'{self.M}-' if self.M is not None else '') + f'{self.c_type} SU-MIMO SVD DigCom System'


    # FUNCTIONALITY

    def simulate(self, bitstream):
        """
        Description
        -----------
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
        s = self.transmitter.simulate(bitstream, self.channel.get_CSI())
        if s is None: return None

        # 2. Channel
        r = self.channel.simulate(s)

        # 3. Receiver
        bitstream_hat = self.receiver.simulate(r, self.channel.get_CSI(), self.transmitter.get_CCI())

        return bitstream_hat[:len(bitstream)]
    
    def BERs_simulation(self, SNRs, num_errors=200, num_channels=50):
        """
        Description
        -----------
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
        data_Rs : 1D numpy array (dtype: int, length: N_SNRs)
            Output - The information bit rates (bits per symbol vector) corresponding to each SNR value.
        activation_Rs : 1D numpy array (dtype: float, length: N_SNRs)
            Output - The activation rate of the channel corresponding to each SNR value. (Indicates the fraction of channel realizations for which enough capacity was available to transmit data.)
        """
        
        def BER_simulation(bitstream):
            """
            Description
            -----------
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
        
        print(f"\nStarting BER simulation for \n{str(self)} ...")


        # Initialization.
        counted_bits = np.zeros(len(SNRs), dtype=int)
        counted_errors = np.zeros(len(SNRs), dtype=float)
        used_channels = np.zeros(len(SNRs), dtype=int)
        realized_channels = np.zeros(len(SNRs), dtype=int)
        data_Rs = [[] for _ in range(len(SNRs))]


        # Iteration.
        while np.any(counted_errors < num_errors) or np.any(used_channels < num_channels):

            for i, SNR_idx in enumerate(np.where( (counted_errors < num_errors) | (used_channels < num_channels) )[0]):

                H = self.channel.get_CSI()['H'] if i != 0 else None
                self.channel.reset(SNR=SNRs[SNR_idx], H=H)

                num_bits = 2400
                bitstream = np.random.randint(0, 2, size=num_bits)
                
                BER = BER_simulation(bitstream)
                R = np.sum(np.log2(self.transmitter.get_CCI()['Mi']))

                
                realized_channels[SNR_idx] += 1
                
                data_Rs[SNR_idx].append(R)
                
                if R == 0: continue
                counted_bits[SNR_idx] += num_bits
                counted_errors[SNR_idx] += num_bits*BER
                used_channels[SNR_idx] += 1
        

        # Termination.
        BERs = counted_errors / counted_bits
        data_Rs = np.array([np.mean(np.array(data_Rs[i], dtype=float)) for i in range(len(SNRs))])
        activation_Rs = used_channels / realized_channels
        
        return BERs, data_Rs, activation_Rs

    def BERs_eigenchs_simulation(self, SNRs, num_errors=200, num_channels=50):
        """
        Description
        -----------
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
        data_Rs : 2D numpy array (dtype: float, shape: (min(Nt, Nr), N_SNRs))
            Output - The total data transmit rate (bits per symbol) of each eigenchannel, corresponding to each SNR value.
        activation_Rs : 1D numpy array (dtype: float, length: N_SNRs)
            Output - The activation rate of every eigenchannel corresponding to each SNR value. (Indicates the fraction of channel realizations for which enough capacity was available to transmit data throught that eigenchannel.)
        """
        
        def BER_eigenchs_simulation(bitstream):
            """
            Description
            -----------
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
        counted_bits = np.zeros((min(self.Nt, self.Nr), len(SNRs)), dtype=float)
        counted_errors = np.zeros((min(self.Nt, self.Nr), len(SNRs)), dtype=float)
        used_channels = np.zeros((min(self.Nt, self.Nr), len(SNRs)), dtype=int)
        realized_channels = np.zeros((min(self.Nt, self.Nr), len(SNRs)), dtype=int)
        data_Rs = [ [[] for _ in range(len(SNRs))] for _ in range(min(self.Nt, self.Nr)) ]


        # Iteration.
        while np.any(counted_errors[0, :] < num_errors) or np.any(used_channels[0, :] < num_channels):

            for i, SNR_idx in enumerate(np.where( (counted_errors[0, :] < num_errors) | (used_channels[0, :] < num_channels) )[0]):

                H = self.channel.get_CSI()['H'] if i != 0 else None
                self.channel.reset(SNR=SNRs[SNR_idx], H=H)

                num_bits = 2400
                bitstream = np.random.randint(0, 2, size=num_bits)
                
                BERi = BER_eigenchs_simulation(bitstream)
                data_Ri = np.pad( np.log2(self.transmitter.get_CCI()['Mi']), pad_width=(0, min(self.Nt, self.Nr) - len(self.transmitter.get_CCI()['Mi'])), mode='constant', constant_values=0 )
                

                realized_channels[:, SNR_idx] += 1
                
                for eigench_idx in range(min(self.Nt, self.Nr)): 
                    data_Rs[eigench_idx][SNR_idx].append(data_Ri[eigench_idx])

                if np.sum(data_Ri) == 0: continue
                num_bits_i = np.ceil(num_bits * (data_Ri[data_Ri > 0] / np.sum(data_Ri)))
                counted_bits[(data_Ri > 0), SNR_idx] += num_bits_i
                counted_errors[(data_Ri > 0), SNR_idx] += num_bits_i*BERi[data_Ri > 0]
                used_channels[(data_Ri > 0), SNR_idx] += (data_Ri > 0)[data_Ri > 0]
        

        # Termination.
        BERs = np.divide(counted_errors, counted_bits, out=np.full((min(self.Nt, self.Nr), len(SNRs)), np.nan, dtype=float), where=(counted_bits != 0))
        data_Rs = np.array([ [np.mean(np.array(data_Rs[eigench_idx][SNR_idx], dtype=float)) for SNR_idx in range(len(SNRs))] for eigench_idx in range(min(self.Nt, self.Nr)) ])
        activation_Rs = used_channels / realized_channels
        
        return BERs, data_Rs, activation_Rs

    def BERs_analytical(self, SNRs, num_channels, settings):
        """
        Description
        -----------
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
        data_Rs : 1D/2D numpy array (dtype: float, shape: (min(Nt, Nr), N_SNRs))
            Output - The information bit rates (bits per symbol vector) corresponding to each SNR value, whether or not for every eigenchannel.
        activation_Rs : 1D/2D numpy array (dtype: float, shape: (min(Nt, Nr), N_SNRs))
            Output - The activation rates of the channel corresponding to each SNR value, whether or not for every eigenchannel. (Indicates the fraction of channel realizations for which enough capacity was available to transmit data.)
        """
    
        def BER_analytical(mode):
            """
            Description
            -----------
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
            data_Ri : 1D numpy array (dtype: float, length: min(Nt, Nr))
                Output - The information bit rates (bits per symbol) of each eigenchannel.
            """

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
                b = (self.receiver.demap(c, M, c_type)).reshape(len(c), -1)
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
            data_Ri = np.pad(np.log2(Mi).astype(int), pad_width=(0, min(self.Nt, self.Nr) - len(Mi)), mode='constant', constant_values=0)
            return BERi, data_Ri

        print(f"\nStarting eigenchannel BER {settings['mode']} calculation for \n{str(self)} ...")


        # Initialization.
        BERs = np.empty((min(self.Nt, self.Nr), len(SNRs), num_channels), dtype=float)
        data_Rs = np.empty((min(self.Nt, self.Nr), len(SNRs), num_channels), dtype=float)


        # Iteration.
        for channel_idx in range(num_channels):

            for SNR_idx, SNR in enumerate(SNRs):

                H = self.channel.get_CSI()['H'] if SNR_idx != 0 else None
                self.channel.reset(SNR=SNR, H=H)

                BERi, data_Ri = BER_analytical(settings['mode'])
                BERs[:, SNR_idx, channel_idx] = BERi
                data_Rs[:, SNR_idx, channel_idx] = data_Ri
        

        # Termination.
        activation_Rs = np.sum( data_Rs > 0, axis=2 ) / num_channels
        BERs = np.nanmean(BERs, axis=2)
        data_Rs = np.mean(data_Rs, axis=2)
        
        if not settings['eigenchannels']:
            BERs = np.nansum( (data_Rs / np.where(data_Rs[0]==0, np.nan, np.sum(data_Rs, axis=0))) * BERs, axis=0 )
            data_Rs = np.sum(data_Rs, axis=0)
            activation_Rs = activation_Rs[0]

        return BERs, data_Rs, activation_Rs


    # TESTS AND PLOTS

    def plot_performance(self, SNRs, BERs, data_Rs, settings):
        """
        Description
        -----------
        Create the performance evaluation plots for the requested evaluation metrics.

        Parameters
        ----------
        SNRs : 1D numpy array (dtype: float, length: N_SNRs)
            The range of signal-to-noise ratio (SNR) values in dB, for which the performance data is provided.
        BERs : 2D numpy array (dtype: float, shape: (N_curves, N_SNRs))
            The simulated Bit Error Rates (BERs) corresponding to each SNR value.
        data_Rs : 2D numpy array (dtype: float, shape: (N_curves, N_SNRs))
            The total data transmit rate (bits per symbol vector) corresponding to each SNR value.
        settings : dict
            A dictionary containing the plot settings for each evaluation metric.
            - labels : the label, for each curve to be displayed in the legend. (list of str, length: N_curves)
            - colors : the color, for each curve. Optional. (list of str, length: N_curves)
            - markers : the type of the markers, for each curve. Optional. (list of str, length: N_curves)
            - marker_colors : the color of the markers, for each curve. Optional. (list of str, length: N_curves)
            - opacity : the opacity of the markers and curve parts, for each curve. Optional. (2D numpy array, dtype: float, shape: (N_curves, N_SNRs))
            - titles : the title of each performance evaluation plot. \n
            This also defines which different metrics are plotted! 
            Possible options are:
                * 'BER' : Bit Error Rate plot.
                * 'data_R' : Data Transmit Rate plot.
                * 'Eb' : The energy per bit to the noise power spectral density (BNR) plot.
        
        Returns
        -------
        plots : dict
            A dictionary containing the created plots (values) for each requested evaluation metric (keys).
        """

        # Initialize a dictionary to hold the result plots.
        plots = {metric: None for metric in settings['titles'].keys()}

        # Initialize the plot settings.
        labels = settings['labels']
        colors = settings.get('colors', [to_hex(c) for c in plt.get_cmap('tab10').colors] + ['black']*(len(labels) - 10))
        markers = settings.get('markers', ['v', '^', '<', '>', 'o', 's', '*', 'd', '8', 'p'] + ['o']*(len(labels) - 10))
        marker_colors = settings.get('marker_colors', colors)
        opacity = settings.get('opacity', None)
        titles = settings['titles']

        # Create the requested plots.
        for metric in settings['titles'].keys():

            # Get the data and data specific plot settings.
            if metric == 'BER': 
                data = BERs
                title = titles[metric]
                y_label = 'Bit Error Rate (BER)'
                y_scale = 'log'
                y_lim_bottom, y_lim_top = 1e-6, 1
            elif metric == 'data_R': 
                data = data_Rs
                title = titles[metric]
                y_label = r'Information Bit Rate $R_b$ [bits per symbol vector]'
                y_scale = 'linear'
                y_lim_bottom, y_lim_top = None, None
            elif metric == 'Eb':
                data = SNRs - np.where(data_Rs == 0, np.nan, 10*np.log10(data_Rs))
                title = titles[metric]
                y_label = r'$\frac{\mathrm{E}_b}{\mathrm{N}_0}$ [dB]'
                y_scale = 'linear'
                y_lim_bottom, y_lim_top = None, None
            else: 
                raise ValueError(f'The performance metric "{metric}" is not recognized.')

            # Initialize the plot.
            fig, ax = plt.subplots()

            # Create each curve.
            for i in range(len(labels)):
                
                if opacity is not None:
                    
                    points = np.array([SNRs, data[i]]).T.reshape(-1, 1, 2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
                    segment_colors = np.tile(to_rgba(colors[i]), (len(segments), 1))
                    segment_colors[:, 3] = (opacity[i][:-1] + opacity[i][1:]) / 2
                    
                    lc = LineCollection(segments, colors=segment_colors, linewidth=1.5)
                    ax.add_collection(lc)

                    for j in range(len(SNRs)):
                        ax.scatter(SNRs[j], data[i][j], marker=markers[i], color=marker_colors[i], edgecolors=marker_colors[i], alpha=opacity[i][j], s=36, zorder=3)

                    ax.plot([], [], label=(labels[i] if i < len(labels) else None), color=colors[i], marker=markers[i], markeredgecolor=marker_colors[i], markerfacecolor=marker_colors[i], linestyle='-')
                
                else:
                    ax.plot(SNRs, data[i], label=(labels[i] if i < len(labels) else None), color=colors[i], marker=markers[i], markeredgecolor=marker_colors[i], markerfacecolor=marker_colors[i], linestyle='-')
            
            # Set the plot settings.
            ax.set_title(title)
            ax.set_yscale(y_scale)
            ax.set_xlabel('SNR [dB]')
            ax.set_ylabel(y_label)
            ax.set_xlim(SNRs[0]-(SNRs[1]-SNRs[0]), SNRs[-1]+(SNRs[1]-SNRs[0]))
            ax.set_ylim(y_lim_bottom, y_lim_top)
            ax.grid(True, which='both', linestyle='--', alpha=0.6)
            ax.legend()
            fig.tight_layout()
            
            # Store the plot.
            fig.savefig(f"su-mimo/plots/performance/SNR_curves/{title.replace(' - ', '__').replace(' ', '_')}.png", dpi=300, bbox_inches="tight")
            plots[metric] = (fig, ax)
        
        return plots

    def plot_scatter_diagram(self, K=100):
        """
        Description
        -----------
        Plot a scatter diagram of the received symbol vectors.
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
            s = self.transmitter.precoder(s_prime, self.channel.get_CSI()['Vh'])

            # 2. Simulate the channel operations.
            r = self.channel.simulate(s)

            # 3. Simulate the receiver operations untill the combiner.
            r_prime = self.receiver.combiner(r, self.channel.get_CSI()['U'])

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
            ax.grid(True, which='both', linestyle='--', alpha=0.7)
            ax.axis('equal')
        
        # Overall plot settings.
        fig.suptitle(f'{str(self)}' + f'\n\nScatter Diagram after SVD Processing \nSNR = {SNR} dB & ' + r'$R_b$' + f' = {round(self.data_rate*100)}%')
        fig.tight_layout()
        fig.savefig(f"su-mimo/plots/performance/scatter_plots/{(str(self) + '__SNR_' + str(SNR) + '__R_' + str(round(self.data_rate*100))).replace(' ', '_').replace('-', '__')}.png", dpi=300, bbox_inches="tight")
    
        # Return the plot.
        return fig, axes

    def print_simulation_example(self, bitstream, K=1):
        """ 
        Description
        -----------
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

        s = self.transmitter.print_simulation_example(bitstream, self.channel.get_CSI(), K)
        r = self.channel.print_simulation_example(s, K)
        bits_hat = self.receiver.print_simulation_example(r, self.channel.get_CSI(), self.transmitter.get_CCI(), K)

        min_len = min(len(bitstream), len(bits_hat))
        BER = np.sum(bitstream[:min_len] != bits_hat[:min_len]) / bitstream.size
        print(f"\n\n========== Final Result ==========\n Bit Error Rate (BER): {BER}\n\n\n")

        return
