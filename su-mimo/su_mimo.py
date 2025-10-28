# This module contains the implementation of the SU-MIMO communication system.

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

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

    def __init__(self, Nt: int, Nr: int, constellation_type: str, M: int = None, Pt: float = 1.0, B: int = 0.5):

        self.Nt = Nt
        self.Nr = Nr
        self.type = constellation_type
        self.M = M
        self.Pt = Pt
        self.B = B

        self.transmitter = tx.Transmitter(Nt, constellation_type, Pt, B)
        self.channel = ch.Channel(Nt, Nr)
        self.receiver = rx.Receiver(Nr, constellation_type, Pt, B)

    def __str__(self):
        """ String representation of the SU-MIMO DigCom system. """
        return f"SU-MIMO System: \n\n  - Transmitter: {str(self.transmitter)}\n\n  - Channel: {str(self.channel)}\n\n  - Receiver: {str(self.receiver)}\n\n"
    
    def __call__(self):
        pass


    # FUNCTIONALITY

    def BER_approximation(self, SNRs: np.ndarray, num_channels: int = 1) -> float:
        """ 
        Description
        -----------
        Calculate the theoretical approximation of the BERs for eigenchannel i, over a range of SNR values.

        If a number of channels is provided, the BERs are averaged over that number of different channel realizations. By default, the BERs are calculated for the current channel only. The state of the system (and thus channel) is not changed after executing this function.

        In addition to the BERs, the total used capacity is returned. This indicates the total number of bits that are transmitted per data symbol vector (or thus per time unit) over the SU-MIMO SVD system at each SNR value.
        
        Parameters
        ----------
        SNRs : 1D numpy array (dtype: float)
            The range of signal-to-noise ratio (SNR) values in dB to calculate the theoretical BER approximation over.
        num_channels : int, optional
            The number of different channel realizations to average over. Default is only the current channel (num_channels=1).

        Returns
        -------
        BERs : 1D numpy array (dtype: float)
            The theoretical Bit Error Rates (BERs) corresponding to each SNR value.
        Cs_total : 1D numpy array (dtype: float)
            The total used capacity of the SU-MIMO SVD system corresponding to each SNR value.
        """

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
        
        channel = self.channel
        BERs_channels = np.zeros((num_channels, len(SNRs)), dtype=float)
        Cs_channels = np.zeros((num_channels, len(SNRs)), dtype=float)

        for channel_i in range(num_channels):

            # Reset the channel and retrieve the CSI.
            if num_channels > 1: self.channel.reset()
            H, U, S, Vh = self.channel.get_CSI()

            BERs_channel_SNRs = []
            Cs_channel_SNRs = []

            for SNR in SNRs:

                # Calculate the used capacity and allocated power on each eigenchannel.
                N0 = self.Pt / ((10**(SNR/10.0)) * 2*self.B)
                Pi, Ci, Mi = self.transmitter.waterfilling(H, S, N0)
                Pi = Pi[Mi > 1]
                Mi = Mi[Mi > 1]

                # Calculate the total used capacity of the system.
                C_total = np.sum( np.log2(Mi) )
                Cs_channel_SNRs.append( C_total )

                # Calculate the bit error rate of the eigenchannels of the system.
                Ki = K(Mi, self.type)
                dmin = d_min(Mi, self.type)
                BERi = (Ki / np.log2(Mi)) * sp.stats.norm.sf( ((Pi * S[:len(Pi)]**2) / N0) * dmin )

                # Calculate the bit error weight of the complete system as a weighted average.
                BERs_channel_SNR = (1 / C_total) * np.sum( np.log2(Mi) * BERi ) if C_total > 0 else np.nan
                BERs_channel_SNRs.append(BERs_channel_SNR)

            BERs_channels[channel_i] = BERs_channel_SNRs
            Cs_channels[channel_i] = Cs_channel_SNRs

        BERs = np.nanmean(BERs_channels, axis=0)
        Cs_total = np.mean(Cs_channels, axis=0)
        self.channel = channel

        return BERs, Cs_total

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
        """

        # 1. Transmitter
        s = self.transmitter(bits, SNR, self.channel.get_CSI(), self.M)

        # 2. Channel
        r = self.channel(s, SNR)

        # 3. Receiver
        bits_hat = self.receiver(r, SNR, self.channel.get_CSI(), self.M)

        return bits_hat[:len(bits)]
    
    def simulate_BER(self, bits: np.ndarray, SNR: float) -> float:
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
        """
        
        bits_hat = self.simulate(bits, SNR)
        ber = np.sum(bits_hat != bits) / bits.size

        return ber

    def simulate_BERs(self, SNRs: np.ndarray, num_errors: int = 200, num_channels: int = 20):
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
        SNRs : 1D numpy array (dtype: float)
            The input range of SNR values in dB.
        BERs : 1D numpy array (dtype: float)
            The simulated Bit Error Rates (BERs) corresponding to each SNR value.
        """
        
        BERs = []

        for SNR in SNRs:

            counted_errors = 0
            counted_channels = 0
            num_bits = (1 / self.BER_approximation(SNR)) * (num_errors / num_channels)
            bits = np.random.randint(0, 2, size=round(num_bits))
            
            while counted_errors < num_errors or counted_channels < num_channels:

                self.channel.reset()
                ber = self.simulate_BER(bits, SNR)

                counted_errors += round(num_bits*ber)
                counted_channels += 1
            
            BERs.append(counted_errors / (num_bits*counted_channels))

        return SNRs, BERs


    # TESTS AND PLOTS

    def plot_BERs(self, SNRs: np.ndarray, BERs_list: list[np.ndarray], labels: list[str], title: str = None) -> None:
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
        title : str, optional
            The title of the plot. Default is None.
        """

        fig, ax = plt.subplots()
        for BERs, label in zip(BERs_list, labels): 
            ax.semilogy(SNRs, BERs, marker='o', linestyle='-', label=label)
        if title is not None: ax.set_title(title)
        ax.set_xlabel('SNR [dB]')
        ax.set_ylabel('Bit Error Rate (BER)')
        ax.grid(True, which='both', linestyle='--', alpha=0.6)
        ax.set_ylim(1e-8, 1)
        ax.set_xlim(SNRs[0]-(SNRs[1]-SNRs[0]), SNRs[-1]+(SNRs[1]-SNRs[0]))
        ax.legend()
        fig.tight_layout()
        
        return fig, ax

    def plot_Cs(self, SNRs: np.ndarray, Cs_list: list[np.ndarray], labels: list[str], title: str = None) -> None:
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
        labels : list of strings with length len(BERs)
            The labels for each BER curve to be displayed in the legend.
        title : str, optional
            The title of the plot. Default is None.
        """

        fig, ax = plt.subplots()
        for Cs, label in zip(Cs_list, labels): 
            ax.plot(SNRs, Cs, marker='o', linestyle='-', label=label)
        if title is not None: ax.set_title(title)
        ax.set_xlabel('SNR [dB]')
        ax.set_ylabel('Total Used Capacity [bits/s/Hz]')
        ax.grid(True, which='both', linestyle='--', alpha=0.6)
        ax.set_xlim(SNRs[0]-(SNRs[1]-SNRs[0]), SNRs[-1]+(SNRs[1]-SNRs[0]))
        ax.set_ylim(0, None)
        ax.legend()
        fig.tight_layout()
        
        return fig, ax
    
    def print_simulation_example(self, bits: np.ndarray, SNR: float) -> None:
        pass

    

if __name__ == "__main__":


    test = 2


    if test == 1:
        system_configs = [ (1, 1, 'PAM'), (2, 2, 'PAM'), (4, 2, 'PAM'), (2, 4, 'PAM'), (4, 4, 'PAM'), (8, 8, 'PAM') ]
        num_channels = 10000

        BERs_list = []
        Cs_total_list = []
        labels = []

        for Nt, Nr, type in system_configs:

            # Initialize SU-MIMO SVD system.
            su_mimo_svd = SuMimoSVD(Nt, Nr, type)
            label = f'{type} ({Nt}x{Nr})'

            # Calculate theoretical BER approximations and the used capacities.
            SNRs = np.arange(-5, 26, 2.5)
            BERs, Cs_total = su_mimo_svd.BER_approximation(SNRs, num_channels=num_channels)

            # Store results.
            BERs_list.append(BERs)
            Cs_total_list.append(Cs_total)
            labels.append(label)


        # Plot results.
        fig, ax = su_mimo_svd.plot_BERs(SNRs, BERs_list, labels, title=f'SU-MIMO SVD System Theoretical BER Approximation\n (averaged over {num_channels//1000}K channel realizations)')
        fig, ax = su_mimo_svd.plot_Cs(SNRs, Cs_total_list, labels, title=f'SU-MIMO SVD System Bit Rate\n (averaged over {num_channels//1000}K channel realizations)')
        plt.show()


    if test == 2:
        
        # Initialize SU-MIMO SVD system.
        su_mimo_svd = SuMimoSVD(Nt=2, Nr=2, constellation_type='PSK', M=4)

        ber = su_mimo_svd.simulate_BER(bits=np.random.randint(0, 2, size=1000), SNR=25)
        print(ber)