# This module contains the implementation of the SU-MIMO communication system.

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os

import transmitter as tx
import channel as ch
import receiver as rx

class SuMimoSVD:
    """
    Description
    -----------
    A single-user multiple-input multiple-output (SU-MIMO) digital communication system, in which the channel state information is available at both the transmitter and receiver.
    
    The communication system consists of a transmitter, a flat-fading MIMO channel, and a receiver. The singular value decomposition (SVD) of the channel matrix is used for precoding at the transmitter and postcoding at the receiver.
    """
    
    # INITIALIZATION AND REPRESENTATION

    def __init__(self, Nt: int, Nr: int, c_type: str, c_size: int = None, data_rate: float = 1.0, H: np.ndarray = None, Pt: float = 1.0, B: int = 0.5):

        self.Nt = Nt
        self.Nr = Nr

        self.type = c_type
        self.M = c_size
        self.data_rate = data_rate

        self.Pt = Pt
        self.B = B

        self.transmitter = tx.Transmitter(Nt, c_type, c_size, data_rate, Pt, B)
        self.channel = ch.Channel(Nt, Nr, H)
        self.receiver = rx.Receiver(Nr, c_type, c_size, data_rate, Pt, B)

    def __str__(self):
        """ String representation of the SU-MIMO DigCom system. """
        return f'{self.Nt}x{self.Nr} ' + (f'{self.M}-' if self.M is not None else '') + f'{self.type} SU-MIMO SVD DigCom System'
    
    def __call__(self):
        pass


    # FUNCTIONALITY

    def simulate(self, bits: np.ndarray, SNR: float) -> np.ndarray:
        """
        Description
        -----------
        Simulate the SU-MIMO SVD communication system for a given input bit stream and SNR value. Return the reconstructed bit stream at the receiver.

        Parameters
        ----------
        bits : 1D numpy array (dtype: int)
            The input bit stream to be transmitted.
        SNR : float
            The signal-to-noise ratio (SNR) in dB.
        
        Returns
        -------
        bits_hat : 1D numpy array (dtype: int)
            The reconstructed bit stream at the receiver after transmission through the SU-MIMO SVD system.
        C_total : int
            The total used capacity of the SU-MIMO SVD system during the transmission.
        """

        # 1. Transmitter
        s, total_transmit_R = self.transmitter.simulate(bits, SNR, self.channel.get_CSI())
        if total_transmit_R == 0: 
            print('Transmission Failed! Not enough capacity available.')
            return None, 0

        # 2. Channel
        r = self.channel.simulate(s, SNR)

        # 3. Receiver
        bits_hat = self.receiver.simulate(r, SNR, self.channel.get_CSI())

        return bits_hat[:len(bits)], total_transmit_R
    

    def BER_simulation(self, bits: np.ndarray, SNR: float) -> float:
        """
        Description
        -----------
        Simulate the SU-MIMO SVD communication system for a given input bit stream and SNR value. Return the Bit Error Rate (BER).

        Parameters
        ----------
        bits : 1D numpy array (dtype: int)
            The input bit stream to be transmitted.
        SNR : float
            The signal-to-noise ratio (SNR) in dB.
        
        Returns
        -------
        ber : float
            The Bit Error Rate (BER) at the receiver after transmission through the SU-MIMO SVD system.
            If the transmission fails (due to a zero capacity channel), the BER is set to NaN.
        C_total : int
            The total used capacity of the SU-MIMO SVD system during the transmission.
        """
        
        bits_hat, C_total = self.simulate(bits, SNR)
        BER = np.sum(bits_hat != bits) / bits.size if C_total > 0 else np.nan

        return BER, C_total

    def BERs_simulation(self, SNRs: np.ndarray, num_errors: int = 200, num_channels: int = 20):
        """
        Description
        -----------
        Simulate the SU-MIMO SVD communication system over a range of SNR values until a specified number of bit errors are reached. Also, average over a specified number of channel realizations. Return the simulated BERs for each SNR value.

        Parameters
        ----------
        SNRs : 1D numpy array (dtype: float)
            The range of signal-to-noise ratio (SNR) values in dB to simulate over.
        num_errors : int
            The minimum number of bit errors to reach for each SNR value before stopping the simulation.
        num_channels : int
            The minimum number of different channel realizations to average over for each SNR value.
        
        Returns
        -------
        BERs : 1D numpy array (dtype: float)
            The simulated Bit Error Rates (BERs) corresponding to each SNR value.
        Cs : 1D numpy array (dtype: int)
            The total used capacity corresponding to each SNR value. This is the number of bits per data symbol vector.
        """
        print(f"\nStarting BER Simulation for \n{str(self)} ...")


        BERs = []
        Cs = []

        for SNR in SNRs:

            counted_errors = 0
            counted_channels = 0
            Cs_SNR = []

            # num_bits = (1 / self.BER_approximation([SNR])[0][0]) * (num_errors / num_channels)
            num_bits = 2400
            bits = np.random.randint(0, 2, size=round(num_bits))
            
            while counted_errors < num_errors or counted_channels < num_channels:

                self.channel.reset()
                BER, C_SNR = self.BER_simulation(bits, SNR)

                Cs_SNR.append(C_SNR)
                if C_SNR == 0: continue

                counted_errors += round(num_bits*BER)
                counted_channels += 1
            
            BERs.append(counted_errors / (num_bits*counted_channels))
            Cs.append(np.mean(np.array(Cs_SNR)))

        return BERs, Cs


    def BER_eigenchs_simulation(self, bits: np.ndarray, SNR: float, Mi: np.ndarray) -> float:
        """
        Description
        -----------
        Simulate the SU-MIMO SVD communication system for a given input bit stream and SNR value. Return the Bit Error Rate (BER) of each eigenchannel.

        Parameters
        ----------
        bits : 1D numpy array (dtype: int)
            The input bit stream to be transmitted.
        SNR : float
            The signal-to-noise ratio (SNR) in dB.
        
        Returns
        -------
        BERi : 1D numpy array (length: number of used eigenchannels)
            The Bit Error Rate (BER) of each eigenchannel at the receiver after transmission through the SU-MIMO SVD system.
            If the transmission fails (due to a zero capacity channel), the BER is set to NaN.
        Ci : 1D numpy array (length: number of used eigenchannels)
            The total used capacity of each eigenchannel of the SU-MIMO SVD system during the transmission.
        """
        
        # Simulate the system.
        bits_hat, C_total = self.simulate(bits, SNR)
        if C_total == 0: return np.nan

        # Allocate bits to the eigenchannels.
        bits_eigenchs = self.transmitter.bit_allocator(bits, np.concatenate([Mi, np.ones(self.Nt - len(Mi), dtype=int)]))
        bits_eigenchs_hat = self.transmitter.bit_allocator(bits_hat, np.concatenate([Mi, np.ones(self.Nt - len(Mi), dtype=int)]))

        # Calculate the BER of each eigenchannel.
        BER_eigenchs = np.array([ (np.sum(bits_eigenchs_hat[i] != bits_eigenchs[i]) / bits_eigenchs[i].size) for i in range(len(Mi)) ])

        return BER_eigenchs

    def BERs_eigenchs_simulation(self, SNRs: np.ndarray, num_errors: int = 200, num_channels: int = 20):
        """
        Description
        -----------
        Simulate the SU-MIMO SVD communication system over a range of SNR values until a specified number of bit errors are reached. Also, average over a specified number of channel realizations. Return the simulated BERs of each eigenchannel (!) for every SNR value.

        Parameters
        ----------
        SNRs : 1D numpy array (dtype: float)
            The range of signal-to-noise ratio (SNR) values in dB to simulate over.
        num_errors : int
            The minimum number of bit errors to reach for each SNR value before stopping the simulation.
        num_channels : int
            The minimum number of different channel realizations to average over for each SNR value.
        
        Returns
        -------
        BERs : 2D numpy array (dtype: float), shape: (min(Nt, Nr), len(SNRs))
            The simulated Bit Error Rates (BERs) of each eigenchannel for every SNR value.
        Cs : 2D numpy array (dtype: int), shape: (min(Nt, Nr), len(SNRs))
            The total used capacity of each eigenchannel for every SNR value. This is the number of bits per data symbol vector.
        """
        print(f"\nStarting BER Simulation for \n{str(self)} ...")

        # Initialization.
        BER_eigenchs = np.empty((min(self.Nt, self.Nr), len(SNRs)), dtype=float)
        C_eigenchs = np.empty((min(self.Nt, self.Nr), len(SNRs)), dtype=float)

        for i, SNR in enumerate(SNRs):
            
            # Initialization.
            N0 = self.Pt / ((10**(SNR/10.0)) * 2*self.B)
            
            counted_errors = 0
            counted_channels = 0

            BER_eigenchs_SNR = []
            C_eigenchs_SNR = []

            while counted_errors < num_errors or counted_channels < num_channels:

                # Generate random bits to transmit.
                num_bits = 2400
                bits = np.random.randint(0, 2, size=round(num_bits))

                # Calculate the used constellation sizes of the eigenchannels.
                self.channel.reset()
                H, U, S, Vh = self.channel.get_CSI()
                Mi = self.transmitter.waterfilling(H, S, N0)[2]
                Mi = (Mi[Mi > 1]).astype(int)

                # Store the total bits per symbol of each eigenchannel for the current SNR value and the current channel realization.
                Ci = np.concatenate( [ np.log2(Mi).astype(int), np.array([0]*(min(self.Nt, self.Nr) - len(Mi))) ] )
                C_eigenchs_SNR.append(Ci)
                if np.sum(Ci) == 0: continue
                
                # Simulate the communication and calculate the BER of each eigenchannel for the current SNR value and the current channel realization. Store the result.
                BER_eigenchs_SNR_channel = self.BER_eigenchs_simulation(bits, SNR, Mi)
                BER_eigenchs_SNR_channel = np.concatenate( [ BER_eigenchs_SNR_channel, np.array([np.nan]*(min(self.Nt, self.Nr) - len(Mi))) ] )
                BER_eigenchs_SNR.append(BER_eigenchs_SNR_channel)

                # Update the counted errors and channels.
                counted_errors += round( num_bits * ((1/np.sum(Mi)) * np.sum(Mi * BER_eigenchs_SNR_channel[:len(Mi)])) )
                counted_channels += 1
            
            # Store the average BER of each eigenchannel over the different channel realizations for the current SNR value.
            BER_eigenchs_SNR = np.nanmean(np.array(BER_eigenchs_SNR), axis=0)
            BER_eigenchs[:, i] = BER_eigenchs_SNR

            # Store the average bits per symbol (C) of each eigenchannel over the different channel realizations for the current SNR value.
            C_eigenchs_SNR = np.mean(np.array(C_eigenchs_SNR), axis=0)
            C_eigenchs[:, i] = C_eigenchs_SNR

        return BER_eigenchs, C_eigenchs


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

    def plot_BERs(self, SNRs: np.ndarray, BERs_list: list[np.ndarray], labels: list[str], colors: list[str] = None, marker_colors: list[str] = None, markers: list[str] = None, title: str = None) -> None:
        """
        Description
        -----------
        Plot the simulated Bit Error Rates (BERs) over a range of SNR values on a semi-logarithmic scale.
        It is possible to plot multiple BER curves on the same figure by providing a list of BER arrays and corresponding labels.

        Parameters
        ----------
        SNRs : 1D numpy array (dtype: float)
            The range of signal-to-noise ratio (SNR) values in dB.
        BERs : list of 1D numpy arrays (dtype: float), each with length len(SNRs)
            The simulated Bit Error Rates (BERs) corresponding to each SNR value.
        labels : list of strings with length len(BERs)
            The labels for each BER curve to be displayed in the legend.
        colors : list of strings with length len(BERs), optional
            The colors for each BER curve. Default uses default matplotlib tableau palette colors.
        marker_colors : list of strings with length len(BERs), optional
            The colors for each marker in the BER curves. Default uses default matplotlib tableau palette colors.
        markers : list of strings with length len(BERs), optional
            The markers for each BER curve. Default is a predefined set of 10 different markers.
        title : str, optional
            The title of the plot. Default is no title.
        """

        # Set default colors and markers if not provided.
        if colors is None: colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'] + ['black']*(len(BERs_list) - 10)
        if marker_colors is None: marker_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'] + ['black']*(len(BERs_list) - 10)
        if markers is None: markers = ['v', '^', '<', '>', 'o', 's', '*', 'd', '8', 'p'] + ['o']*(len(BERs_list) - 10)

        # Create the plot.
        fig, ax = plt.subplots()
        for i in range(len(BERs_list)):
            ax.semilogy(SNRs, BERs_list[i], label=labels[i], color=colors[i], markeredgecolor=marker_colors[i], markerfacecolor=marker_colors[i], marker=markers[i], linestyle='-')
        if title is not None: ax.set_title(title)
        ax.set_xlabel('SNR [dB]')
        ax.set_ylabel('Bit Error Rate (BER)')
        ax.grid(True, which='both', linestyle='--', alpha=0.6)
        ax.set_ylim(1e-8, 1)
        ax.set_xlim(SNRs[0]-(SNRs[1]-SNRs[0]), SNRs[-1]+(SNRs[1]-SNRs[0]))
        ax.legend()
        fig.tight_layout()
        
        return fig, ax

    def plot_Cs(self, SNRs: np.ndarray, Cs_list: list[np.ndarray],  labels: list[str], colors: list[str] = None, marker_colors: list[str] = None, markers: list[str] = None, title: str = None) -> None:
        """
        Description
        -----------
        Plot the total used capacity, i.e. the total amount of bits that are transmitted per data symbol vector (or thus per time unit), over a range of SNR values on a semi-logarithmic scale.
        It is possible to plot multiple capacity curves on the same figure by providing a list of capacity arrays and corresponding labels.

        Parameters
        ----------
        SNRs : 1D numpy array (dtype: float)
            The range of signal-to-noise ratio (SNR) values in dB.
        Cs : list of 1D numpy arrays (dtype: float), each with length len(SNRs)
            The total used capacities corresponding to each SNR value.
        labels : list of strings with length len(Cs)
            The labels for each capacity curve to be displayed in the legend.
        colors : list of strings with length len(Cs), optional
            The colors for each capacity curve. Default uses default matplotlib tableau palette colors.
        marker_colors : list of strings with length len(Cs), optional
            The colors for each marker in the capacity curves. Default uses default matplotlib tableau palette colors.
        markers : list of strings with length len(Cs), optional
            The markers for each capacity curve. Default is a predefined set of 10 different markers.
        title : str, optional
            The title of the plot. Default is None.
        """

        # Set default colors and markers if not provided.
        if colors is None: colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'] + ['black']*(len(Cs_list) - 10)
        if marker_colors is None: marker_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'] + ['black']*(len(Cs_list) - 10)
        if markers is None: markers = ['v', '^', '<', '>', 'o', 's', '*', 'd', '8', 'p'] + ['o']*(len(Cs_list) - 10)

        # Create the plot.
        fig, ax = plt.subplots()
        for i in range(len(Cs_list)):
            ax.plot(SNRs, Cs_list[i], label=labels[i], color=colors[i], markeredgecolor=marker_colors[i], markerfacecolor=marker_colors[i], marker=markers[i], linestyle='-')
        if title is not None: ax.set_title(title)
        ax.set_xlabel('SNR [dB]')
        ax.set_ylabel('Total Used Capacity [bits/symbol vector]')
        ax.grid(True, which='both', linestyle='--', alpha=0.6)
        ax.set_xlim(SNRs[0]-(SNRs[1]-SNRs[0]), SNRs[-1]+(SNRs[1]-SNRs[0]))
        ax.set_ylim(0, None)
        ax.legend()
        fig.tight_layout()
        
        return fig, ax
    
    def plot_BERs_Cs(self, SNRs: np.ndarray, BERs_list: list[np.ndarray], Cs_list: list[np.ndarray], labels: list[str], colors: list[str] = None, marker_colors: list[str] = None, markers: list[str] = None, title: str = None) -> None:
        """
        Description
        -----------
        Plot the simulated Bit Error Rates (BERs) in function of the ratio between the energy per bit and the noise PSD (E_b/N_0) on a semi-logarithmic scale.
        It is possible to plot multiple BER curves on the same figure by providing a list of BER arrays and corresponding labels.

        Parameters
        ----------
        SNRs : 1D numpy array (dtype: float)
            The range of signal-to-noise ratio (SNR) values in dB.
        Cs : list of 1D numpy arrays (dtype: float), each with length len(SNRs)
            The total used capacities corresponding to each SNR value.
        labels : list of strings with length len(Cs)
            The labels for each capacity curve to be displayed in the legend.
        colors : list of strings with length len(Cs), optional
            The colors for each capacity curve. Default uses default matplotlib tableau palette colors.
        marker_colors : list of strings with length len(Cs), optional
            The colors for each marker in the capacity curves. Default uses default matplotlib tableau palette colors.
        markers : list of strings with length len(Cs), optional
            The markers for each capacity curve. Default is a predefined set of 10 different markers.
        title : str, optional
            The title of the plot. Default is None.
        """

        # Set default colors and markers if not provided.
        if colors is None: colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'] + ['black']*(len(BERs_list) - 10)
        if marker_colors is None: marker_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'] + ['black']*(len(BERs_list) - 10)
        if markers is None: markers = ['v', '^', '<', '>', 'o', 's', '*', 'd', '8', 'p'] + ['o']*(len(BERs_list) - 10)

        # Create the plot.
        fig, ax = plt.subplots()
        for i in range(len(BERs_list)):
            ax.semilogy(SNRs / Cs_list[i], BERs_list[i], label=labels[i], color=colors[i], markeredgecolor=marker_colors[i], markerfacecolor=marker_colors[i], marker=markers[i], linestyle='-')
        if title is not None: ax.set_title(title)
        ax.set_xlabel(r'$\frac{\mathrm{E}_b}{\mathrm{N}_0}$ [dB]')
        ax.set_ylabel('Bit Error Rate (BER)')
        ax.grid(True, which='both', linestyle='--', alpha=0.6)
        ax.set_ylim(1e-8, 1)
        ax.legend()
        fig.tight_layout()
        
        return fig, ax

    def print_simulation_example(self, bits: np.ndarray, SNR: float, K: int = 1) -> None:
        """ 
        Description
        -----------
        Print a step-by-step example of the transmitter, channel and receiver operations (see simulate() methods) for given input bits, SNR, and optionally a channel state information (CSI). Only the first K data symbols vectors are considered.
        """
        print(self)

        s = self.transmitter.print_simulation_example(bits, SNR, self.channel.get_CSI(), K)
        r = self.channel.print_simulation_example(s, SNR, K)
        bits_hat = self.receiver.print_simulation_example(r, SNR, self.channel.get_CSI(), K)


        min_len = min(len(bits), len(bits_hat))
        BER = np.sum(bits[:min_len] != bits_hat[:min_len]) / bits.size
        print(f"\n\n========== Final Result ==========\n Bit Error Rate (BER): {BER}\n\n\n")

        return




if __name__ == "__main__":

    test = 'test 3'

    # TEST 1: Example simulation of SU-MIMO SVD system.
    if test == 'test 1':

        Nt = 3
        Nr = 2
        type = 'PSK'
        H = np.array([[3, 2, 2], [2, 3, -2]])

        su_mimo_svd = SuMimoSVD(Nt=Nt, Nr=Nr, constellation_type=type, H=H)
        ber = su_mimo_svd.print_simulation_example(bits=np.random.randint(0, 2, size=16000), SNR=np.inf, K=3)


    # TEST 2: Performance analysis of multiple SU-MIMO SVD systems.
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
                label = f'{type} ({Nt}x{Nr})'

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
    
    # TEST 3: Performance analysis of different eigenchannels.
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
        

    # TEST 4: Performance analysis at different data rates.

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



