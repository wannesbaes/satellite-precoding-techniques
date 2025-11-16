# This module contains the implementation of the SU-MIMO communication system.

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba, to_hex
from matplotlib.collections import LineCollection

import transmitter as tx
import channel as ch
import receiver as rx


class SuMimoSVD:
    """
    Description
    -----------
    A single-user multiple-input multiple-output (SU-MIMO) digital communication system, in which the channel state information is available at both the transmitter and receiver.
    
    The communication system consists of a transmitter, a distortion-free MIMO channel, and a receiver. 
    The singular value decomposition (SVD) of the channel matrix is used for precoding at the transmitter and postcoding at the receiver.
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
        Return the simulated BERs for each SNR value.
        
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
            Output - The total data transmit rate (bits per symbol vector) corresponding to each SNR value.
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
        
        print(f"\nStarting BER Simulation for \n{str(self)} ...")

        # Initialization.
        counted_bits = np.zeros(len(SNRs), dtype=int)
        counted_errors = np.zeros(len(SNRs), dtype=float)
        used_channels = np.zeros(len(SNRs), dtype=int)
        realized_channels = np.zeros(len(SNRs), dtype=int)
        data_Rs = [[] for _ in range(len(SNRs))]

        # Run the simulation until the required number of errors and channels is reached for every SNR value.
        while np.any(counted_errors < num_errors) or np.any(used_channels < num_channels):

            # Reset the channel properties (new channel matrix).
            self.channel.reset()
            
            # Iterate over every SNR value for which the required number of errors or usable channel realization has not yet been reached.
            for i in np.where( (counted_errors < num_errors) | (used_channels < num_channels) )[0]:

                # Initialize the bitstream.
                num_bits = 2400
                bitstream = np.random.randint(0, 2, size=num_bits)

                # Run the simulation for the current channel and SNR value.
                self.channel.reset(SNR=SNRs[i], H=self.channel.get_CSI()['H'])
                BER = BER_simulation(bitstream)
                R = np.sum(np.log2(self.transmitter.get_CCI()['Mi']))

                # Store the channel activation rate.
                realized_channels[i] += 1

                # Store the data transmit rate.
                data_Rs[i].append(R)

                # Store the bit error rate. (Only if actual data was transmitted)
                if R == 0: continue
                counted_bits[i] += num_bits
                counted_errors[i] += num_bits*BER
                used_channels[i] += 1
        
        # Calculate the average BER and data transmit rate for each SNR value.
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
        
        print(f"\nStarting BER Eigenchannel Simulation for \n{str(self)} ...")

        # Initialization.
        counted_bits = np.zeros((min(self.Nt, self.Nr), len(SNRs)), dtype=float)
        counted_errors = np.zeros((min(self.Nt, self.Nr), len(SNRs)), dtype=float)
        used_channels = np.zeros((min(self.Nt, self.Nr), len(SNRs)), dtype=int)
        realized_channels = np.zeros((min(self.Nt, self.Nr), len(SNRs)), dtype=int)
        data_Rs = [ [[] for _ in range(len(SNRs))] for _ in range(min(self.Nt, self.Nr)) ]

        # Run the simulation until the required number of errors (on every eigenchannel) and usable channel realization is reached for every SNR value.
        while np.any(counted_errors[0, :] < num_errors) or np.any(used_channels[0, :] < num_channels):

            # Reset the channel properties (new channel matrix).
            self.channel.reset()
            
            # Iterate over every SNR value for which the required number of errors or usable channel realizations has not yet been reached.
            for SNR_idx in np.where( (counted_errors[0, :] < num_errors) | (used_channels[0, :] < num_channels) )[0]:

                # Initialize the bitstream.
                num_bits = 2400
                bitstream = np.random.randint(0, 2, size=num_bits)

                # Run the simulation for the current channel and SNR value.
                self.channel.reset(SNR=SNRs[SNR_idx], H=self.channel.get_CSI()['H'])
                BERi = BER_eigenchs_simulation(bitstream)
                data_Ri = np.pad( np.log2(self.transmitter.get_CCI()['Mi']), pad_width=(0, min(self.Nt, self.Nr) - len(self.transmitter.get_CCI()['Mi'])), mode='constant', constant_values=0 )
                
                # Store the eigenchannel activation rate.
                realized_channels[:, SNR_idx] += 1
                
                # Store the data transmit rate.
                for eigench_idx in range(min(self.Nt, self.Nr)): data_Rs[eigench_idx][SNR_idx].append(data_Ri[eigench_idx])

                # Store the bit error rate. (Only if capacity is available on the eigenchannel.)
                if np.sum(data_Ri) == 0: continue
                num_bits_i = np.ceil(num_bits * (data_Ri[data_Ri > 0] / np.sum(data_Ri)))
                counted_bits[(data_Ri > 0), SNR_idx] += num_bits_i
                counted_errors[(data_Ri > 0), SNR_idx] += num_bits_i*BERi[data_Ri > 0]
                used_channels[(data_Ri > 0), SNR_idx] += (data_Ri > 0)[data_Ri > 0]
        
        # Calculate the average BER, data transmit rate and activation rate for each SNR value.
        BERs = np.divide(counted_errors, counted_bits, out=np.full((min(self.Nt, self.Nr), len(SNRs)), np.nan, dtype=float), where=(counted_bits != 0))
        data_Rs = np.array([ [np.mean(np.array(data_Rs[eigench_idx][SNR_idx], dtype=float)) for SNR_idx in range(len(SNRs))] for eigench_idx in range(min(self.Nt, self.Nr)) ])
        activation_Rs = used_channels / realized_channels
        
        return BERs, data_Rs, activation_Rs

    def BERs_analytical(self, SNRs: np.ndarray, num_channels: int, mode: str, eigenchannels: bool = False) -> float:
        """ 
        Description
        -----------
        Calculate a theoretical upper boound of the BERs for eigenchannel i, over a range of SNR values.

        If a number of channels is provided, the BERs are averaged over that number of different channel realizations. By default, the BERs are calculated for the current channel only. The state of the system (and thus channel) is not changed after executing this function.

        In addition to the BERs, the total used capacity is returned. This indicates the total number of bits that are transmitted per data symbol vector (or thus per time unit) over the SU-MIMO SVD system at each SNR value.
        
        Parameters
        ----------
        SNRs : 1D numpy array (dtype: float)
            The range of signal-to-noise ratio (SNR) values in dB to calculate the theoretical BER approximation over.
        num_channels : int
            The number of different channel realizations to average over.
        mode: str
            Choose between 'upper bound' and 'approximation'.
        eigenchannels : bool, optional
            If True, the BERs and used capacity of each eigenchannel separately are returned as well. 

        Returns
        -------
        BERs : 1D numpy array (dtype: float)
            The theoretical Bit Error Rates (BERs) corresponding to each SNR value.
        Cs_total : 1D numpy array (dtype: float)
            The total used capacity (bits per symbol vector) of the SU-MIMO SVD system corresponding to each SNR value.
        BERs_eigench: 2D numpy array (dtype: float), shape: (min(Nt, Nr), len(SNRs))
            The theoretical Bit Error Rates (BERs) of each eigenchannel, for every SNR value. (Only if eigenchannels is True.)
        Cs_eigench: 2D numpy array (dtype: float), shape: (min(Nt, Nr), len(SNRs))
            The total used capacity (bits per symbol) of each eigenchannel, for every SNR value. (Only if eigenchannels is True.)
        """
        print(f"\nStarting BER {mode} calculation for \n{str(self)} ...")

        def construct_constellation(M: int, type: str) -> np.ndarray:

            if type == 'PAM':
                constellation = np.arange(-(M-1), (M-1) + 1, 2) * np.sqrt(3/(M**2-1))

            elif type == 'PSK':
                constellation = np.exp(1j * 2*np.pi * np.arange(M) / M)

            elif type == 'QAM':
                c_sqrtM_PAM = np.arange(-(np.sqrt(M)-1), (np.sqrt(M)-1) + 1, 2) * np.sqrt(3 / (2*(M-1)))
                real_grid, imaginary_grid = np.meshgrid(c_sqrtM_PAM, c_sqrtM_PAM)
                constellation = (real_grid + 1j*imaginary_grid).ravel()

            else :
                raise ValueError(f'The constellation type is invalid.\nChoose between "PAM", "PSK", or "QAM". Right now, type is {type}.')
        
            return constellation

        def Q(x: np.ndarray) -> np.ndarray:
            """ Implementation of the Q-function. The tail probability of standard normal distribution: Q(x) = P[X > x] with X ~ N(0,1). """
            return sp.stats.norm.sf(x)

        def N(c: np.ndarray, Mi: int, type: str) -> np.ndarray:
            """ Return the Hamming distance between any two constellation points. Cartesian product of c is created in the same way as in d(). """
            b = (self.receiver.demap(c, Mi, type)).reshape(len(c), -1)
            i, j = np.meshgrid(np.arange(len(c)), np.arange(len(c)))
            N = (np.sum(np.abs(b[i] - b[j]), axis=2)).ravel()
            return N

        def d(c: np.ndarray) -> np.ndarray:
            """ Return the Euclidean distance between any two constellation points. Cartesian product of c is created in the same way as in N(). """
            c1, c2 = np.meshgrid(c, c)
            d = np.abs((c1 - c2).ravel())
            return d

        def K(Mi: np.ndarray, type: str) -> float:
            """ Calculate the average number of nearest neighbors K of a given constellation. """
            if type == 'PAM': return 2*(Mi-1) / Mi
            elif type == 'PSK': return np.where(Mi == 2, 1, 2)
            elif type == 'QAM': return 4*(np.sqrt(Mi)-1) / np.sqrt(Mi)
            else: raise ValueError(f'The constellation type is invalid.\nChoose between "PAM", "PSK", or "QAM". Right now, type is {type}.')
        
        def d_min(Mi: np.ndarray, type: str) -> float:
            """ Calculate the distance between the nearest neighbors of a given constellation. """
            if type == 'PAM': return np.sqrt(12 / (Mi**2 - 1))
            elif type == 'PSK': return 2 * np.abs( np.sin(np.pi / Mi) )
            elif type == 'QAM': return np.sqrt(6 / (Mi - 1))
            else: raise ValueError(f'The constellation type is invalid.\nChoose between "PAM", "PSK", or "QAM". Right now, type is {type}.')
        
        # Store the current channel.
        channel = self.channel

        # Initialization.
        BERs_channels = np.zeros((num_channels, len(SNRs)), dtype=float)
        Cs_channels = np.zeros((num_channels, len(SNRs)), dtype=float)

        if eigenchannels:
            BERs_eigenchs_channels = np.zeros((num_channels, min(self.Nt, self.Nr), len(SNRs)), dtype=float)
            Cs_eigenchs_channels = np.zeros((num_channels, min(self.Nt, self.Nr), len(SNRs)), dtype=float)


        for channel_i in range(num_channels):

            # Reset the channel and retrieve the CSI.
            if num_channels > 1: self.channel.reset()
            H, U, S, Vh = self.channel.get_CSI()

            # Initialization of outer-loop variables.
            BERs_channel_SNRs = []
            Cs_channel_SNRs = []

            if eigenchannels:
                BERs_eigench_channel_SNRs = np.zeros((min(self.Nt, self.Nr), len(SNRs)))
                Cs_eigench_channel_SNRs = np.zeros((min(self.Nt, self.Nr), len(SNRs)))

            for i, SNR in enumerate(SNRs):

                # Calculate the used capacity and allocated power on each eigenchannel.
                N0 = self.Pt / ((10**(SNR/10.0)) * 2*self.B)
                Pi, Ci, Mi = self.transmitter.waterfilling(H, S, N0)
                Pi = Pi[Mi > 1]
                Mi = Mi[Mi > 1]

                # Calculate the total used capacity of the system, and store it.
                C_total = np.sum( np.log2(Mi) )
                Cs_channel_SNRs.append( C_total )


                # Calculate the bit error rate of the eigenchannels of the system!

                if mode == 'upper bound':
                    BERi = np.empty(len(Pi))
                    for i in range(len(Pi)):
                        c = construct_constellation(Mi[i], self.type)
                        BERi[i] = (1 / (Mi[i]*np.log2(Mi[i]))) * np.sum( N(c, Mi[i], self.type) * Q( ((Pi[i] * S[i]**2) / N0) * d(c) ) )
                
                elif mode == 'approximation':
                    Ki = K(Mi, self.type)
                    dmin = d_min(Mi, self.type)
                    BERi = (Ki / np.log2(Mi)) * Q( ((Pi * S[:len(Pi)]**2) / N0) * dmin )

                else: 
                    raise ValueError(f'The mode is invalid.\nChoose between "upper bound" or "approximation". Right now, mode is {mode}.')


                # Optionally, store the bit error rates and used capacity of each eigenchannel, for the current SNR value and channel realization.
                if eigenchannels:
                    BERs_eigench_channel_SNRs[:, i] = np.concatenate([BERi, np.array( [np.nan] * (min(self.Nt, self.Nr) - len(BERi)) )])
                    Cs_eigench_channel_SNRs[:, i] = np.concatenate([Mi, np.array( [0] * (min(self.Nt, self.Nr) - len(Mi)) )])


                # Calculate the bit error rate of the complete system as a weighted average, and store it.
                BERs_channel_SNR = (1 / C_total) * np.sum( np.log2(Mi) * BERi ) if C_total > 0 else np.nan
                BERs_channel_SNRs.append( BERs_channel_SNR )


            # Store the results for the current channel realization.
            BERs_channels[channel_i] = BERs_channel_SNRs
            Cs_channels[channel_i] = Cs_channel_SNRs

            if eigenchannels:
                BERs_eigenchs_channels[channel_i] = BERs_eigench_channel_SNRs
                Cs_eigenchs_channels[channel_i] = Cs_eigench_channel_SNRs


        # Average the results over the different channel realizations.
        BERs = np.nanmean(BERs_channels, axis=0)
        Cs = np.mean(Cs_channels, axis=0)

        if eigenchannels:
            BERs_eigench = np.nanmean(BERs_eigenchs_channels, axis=0)
            Cs_eigench = np.mean(Cs_eigenchs_channels, axis=0)

        # Restore the current channel.
        self.channel = channel

        # Return the result.
        result = [BERs, Cs]
        if eigenchannels: result += [BERs_eigench, Cs_eigench]
        
        return result



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
                * 'data_Rs' : Data Transmit Rate plot.
                * 'Ebs' : The energy per bit to the noise power spectral density (BNR) plot.
        
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
                y_lim_bottom, y_lim_top = 1e-7, 1
            elif metric == 'data_Rs': 
                data = data_Rs
                title = titles[metric]
                y_label = 'Data Transmit Rate (bits per symbol vector)'
                y_scale = 'linear'
                y_lim_bottom, y_lim_top = None, None
            elif metric == 'Ebs':
                data = np.divide(np.tile(SNRs, (len(labels), 1)), data_Rs, out=np.full((len(labels), len(SNRs)), np.nan), where=(data_Rs != 0))
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

                    ax.plot([], [], label=labels[i], color=colors[i], marker=markers[i], markeredgecolor=marker_colors[i], markerfacecolor=marker_colors[i], linestyle='-')
                
                else:
                    ax.plot(SNRs, data[i], label=labels[i], color=colors[i], marker=markers[i], markeredgecolor=marker_colors[i], markerfacecolor=marker_colors[i], linestyle='-')
            
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
                Output - The received data symbols (after postcoding).
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

            # 3. Simulate the receiver operations untill the postcoder.
            r_prime = self.receiver.postcoder(r, self.channel.get_CSI()['U'])

            # Return
            return a, r_prime

        # Similate part of the system to obtain the transmitted data symbols (before power allocation and precoding) and the received data symbols (after postcoding).
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
            ax.scatter(constellation_points.real, constellation_points.imag, color=colors, edgecolor='black', marker='o', alpha= 1.0, s=50)

            # Plot settings.
            ax.axhline(0, color='black', linewidth=0.5)
            ax.axvline(0, color='black', linewidth=0.5)
            ax.set_title(f'Eigenchannel {eigenchannel+1}: {Mi[eigenchannel]}-{self.c_type}')
            ax.set_xlabel('Real Part')
            ax.set_ylabel('Imaginary Part')
            ax.grid(True, which='both', linestyle='--', alpha=0.7)
            ax.axis('equal')
        
        # Overall plot settings.
        fig.suptitle(f'{str(self)}' + f'\n\nScatter Diagram after SVD Processing \nSNR = {SNR} dB & R = {round(self.data_rate*100)}%')
        fig.tight_layout()
    
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


if __name__ == "__main__":

    test = 'test 0'
    
    # TEST 1: Example simulation of SU-MIMO SVD system.
    if test == 'test 1':

        Nt = 3
        Nr = 2
        type = 'PSK'
        H = np.array([[3, 2, 2], [2, 3, -2]])

        su_mimo_svd = SuMimoSVD(Nt=Nt, Nr=Nr, constellation_type=type, H=H)
        ber = su_mimo_svd.print_simulation_example(bits=np.random.randint(0, 2, size=16000), SNR=np.inf, K=3)


    # TEST 2: Performance evaluation of multiple SU-MIMO SVD systems.
    if test == 'test 2':

        def test_performance_systems(system_configs, curves, SNRs, num_errors_sim, num_channels):

            BERs_list = []
            Cs_list = []
            
            labels = []
            colors = []

            markers = [marker for marker in ['o', 's', 'D', 'v', '*', 'p', '8', '^', '<', '>'][:len(system_configs)] for _ in range(len(curves))]


            for Nt, Nr, type in system_configs:

                # Initialize SU-MIMO SVD system.
                su_mimo_svd = SuMimoSVD(Nt, Nr, type)
                label = f'{Nt}x{Nr} {type}'

                # Calculate analytical BERs upper bound and the used capacities, and store the results.
                if 'upper bound' in curves:

                    BERs_upper_bound, Cs_upper_bound, BERis_upper_bound = su_mimo_svd.BERs_analytical(SNRs, num_channels['upper bound'], mode='upper bound')

                    BERs_list.append(BERs_upper_bound)
                    Cs_list.append(Cs_upper_bound)
                    
                    labels.append(label + ' upper bound')
                    colors.append('tab:red')

                    np.savez(f'su-mimo/inter_sim_results/{Nt}x{Nr}_{type}_performance_upper_bound.npz', SNRs=SNRs, BERs=BERs_upper_bound, Cs=Cs_upper_bound, BERis=BERis_upper_bound)
                
                # Calculate analytical BERs approximation and the used capacities, and store the results.
                if 'approximation' in curves:

                    BERs_approx, Cs_approx, BERis_approx = su_mimo_svd.BERs_analytical(SNRs, num_channels['approximation'], mode='approximation')

                    BERs_list.append(BERs_approx)
                    Cs_list.append(Cs_approx)

                    labels.append(label + ' approximation')
                    colors.append('tab:orange')

                    np.savez(f'su-mimo/inter_sim_results/{Nt}x{Nr}_{type}_performance_approximation.npz', SNRs=SNRs, BERs=BERs_approx, Cs=Cs_approx, BERis=BERis_approx)

                # Calculate simulated BERs and the used capacities, and store the results.
                if 'simulation' in curves:

                    BERs_sim, Cs_sim = su_mimo_svd.BERs_simulation(SNRs, num_errors_sim, num_channels['simulation'])

                    BERs_list.append(BERs_sim)
                    Cs_list.append(Cs_sim)
                    
                    labels.append(label + ' simulation')
                    colors.append('tab:blue')

                    np.savez(f'su-mimo/inter_sim_results/{Nt}x{Nr}_{type}_performance_simulation.npz', SNRs=SNRs, BERs=BERs_sim, Cs=Cs_sim)


            # Plot results.
            fig1, ax1 = su_mimo_svd.plot_BERs(SNRs, BERs_list, labels, colors=colors, marker_colors=colors, markers=markers,title=f'SU-MIMO SVD DigCom System')
            fig2, ax2 = su_mimo_svd.plot_Cs(SNRs, Cs_list, labels, colors=colors, marker_colors=colors, markers=markers, title=f'SU-MIMO SVD DigCom System')
            fig3, ax3 = su_mimo_svd.plot_BERs_Cs(SNRs, BERs_list, Cs_list, labels, colors=colors, marker_colors=colors, markers=markers, title=f'SU-MIMO SVD DigCom System')
            
            # Return
            return fig1, ax1, fig2, ax2, fig3, ax3
    
        system_configs = [ (4, 4, 'PSK') ]
        curves = ['simulation']
        SNRs = np.arange(-5, 31, 5)
        num_errors_sim = 200
        num_channels = {'simulation': 50}
        fig1, ax1, fig2, ax2, fig3, ax3 = test_performance_systems(system_configs, curves, SNRs, num_errors_sim, num_channels)
        plt.show()


    # TEST 3: Performance evaluation of the different eigenchannels.
    if test == 'test 3':

        def test_performance_eigenchannels_simulation(system, SNRs, num_errors, num_channels):
            
            Nt, Nr, type = system
            su_mimo_svd = SuMimoSVD(Nt, Nr, type)

            BERs_eigenchs, Cs_eigenchs = su_mimo_svd.BERs_eigenchs_simulation(SNRs, num_errors, num_channels)
            
            BERs_eigenchs = [x for x in BERs_eigenchs]
            Cs_eigenchs = [x for x in Cs_eigenchs]
            labels = [f'eigenchannel {i+1}' for i in range(min(Nt, Nr))]
            title = f'Eigenchannel Performance (Simulation)\n{Nt}x{Nr} {type} System.'

            fig1, ax1 = su_mimo_svd.plot_BERs(SNRs, BERs_eigenchs, labels, markers=(['o']*min(Nt, Nr)), title=title)
            fig2, ax2 = su_mimo_svd.plot_Cs(SNRs, Cs_eigenchs, labels, markers=(['o']*min(Nt, Nr)), title=title)
            fig3, ax3 = su_mimo_svd.plot_BERs_Cs(SNRs, BERs_eigenchs, Cs_eigenchs, labels, markers=(['o']*min(Nt, Nr)), title=title)

            np.savez(f'su-mimo/inter_sim_results/{Nt}x{Nr}_{type}_eigenchannel_performance_simulation.npz', SNRs=SNRs, BERs=BERs_eigenchs, Cs=Cs_eigenchs, labels=labels)

            return fig1, ax1, fig2, ax2, fig3, ax3
        
        def test_performance_eigenchannels_analytical(system, SNRs, num_channels, mode):
            
            Nt, Nr, type = system
            su_mimo_svd = SuMimoSVD(Nt, Nr, type)

            BERs_eigenchs, Cs_eigenchs = su_mimo_svd.BERs_analytical(SNRs, num_channels, mode, eigenchannels=True)[2:]

            BERs_eigenchs = [x for x in BERs_eigenchs]
            Cs_eigenchs = [x for x in Cs_eigenchs]
            labels = [f'eigenchannel {i+1}' for i in range(min(Nt, Nr))]
            title = f'Eigenchannel Performance (Analytical {mode})\n{Nt}x{Nr} {type} System.'

            fig1, ax1 = su_mimo_svd.plot_BERs(SNRs, BERs_eigenchs, labels, markers=(['o']*min(Nt, Nr)), title=title)
            fig2, ax2 = su_mimo_svd.plot_Cs(SNRs, Cs_eigenchs, labels, markers=(['o']*min(Nt, Nr)), title=title)
            fig3, ax3 = su_mimo_svd.plot_BERs_Cs(SNRs, BERs_eigenchs, Cs_eigenchs, labels, markers=(['o']*min(Nt, Nr)), title=title)

            np.savez(f'su-mimo/inter_sim_results/{Nt}x{Nr}_{type}_eigenchannel_performance_{mode.replace(" ", "_")}.npz', SNRs=SNRs, BERs=BERs_eigenchs, Cs=Cs_eigenchs, labels=labels)

            return fig1, ax1, fig2, ax2, fig3, ax3
        

    # TEST 4: Performance evaluation at different data rates.
    if test == 'test 4':

        def test_performance_data_rates_simulation(system, SNRs, c_sizes, SNRs_list, data_rates, num_errors, num_channels):
        
            Nt, Nr, c_type = system

            BERs_list = []
            Rs_list = []
            labels = []

            for c_size in c_sizes:

                su_mimo_svd = SuMimoSVD(Nt, Nr, c_type, c_size=c_size)
                BERs, Rs = su_mimo_svd.BERs_simulation(SNRs, num_errors, num_channels)

                BERs_list.append(BERs)
                Rs_list.append(Rs)
                labels.append(r'$R =$' + f' {np.log2(c_size).astype(int)} bits/antenna')

            for data_rate, SNRs_i in zip(data_rates, SNRs_list):

                su_mimo_svd = SuMimoSVD(Nt, Nr, c_type, data_rate=data_rate)
                BERs, Rs = su_mimo_svd.BERs_simulation(SNRs_i, num_errors, num_channels)

                BERs = np.concatenate([np.full(len(SNRs) - len(SNRs_i), np.nan), BERs])
                Rs = np.concatenate([np.full(len(SNRs) - len(SNRs_i), np.nan), Rs])

                BERs_list.append(BERs)
                Rs_list.append(Rs)
                labels.append(r'$R \approx$' + f' {round(data_rate*100)}' + r'$\%$')


            title = f'Performance at Different Data Rates (Simulation)\n{Nt}x{Nr} {c_type} System.'
            fig1, ax1 = su_mimo_svd.plot_BERs(SNRs, BERs_list, labels, markers=['o']*len(BERs_list), title=title)
            fig2, ax2 = su_mimo_svd.plot_Cs(SNRs, Rs_list, labels, markers=['o']*len(Rs_list), title=title)
            fig3, ax3 = su_mimo_svd.plot_BERs_Cs(SNRs, BERs_list, Rs_list, labels, markers=['o']*len(BERs_list), title=title)

            return fig1, ax1, fig2, ax2, fig3, ax3



