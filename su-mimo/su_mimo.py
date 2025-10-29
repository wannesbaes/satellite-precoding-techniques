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

    def __init__(self, Nt: int, Nr: int, constellation_type: str, H: np.ndarray = None, M: int = None, Pt: float = 1.0, B: int = 0.5):

        self.Nt = Nt
        self.Nr = Nr
        self.type = constellation_type
        self.M = M
        self.Pt = Pt
        self.B = B

        self.transmitter = tx.Transmitter(Nt, constellation_type, Pt, B)
        self.channel = ch.Channel(Nt, Nr, H)
        self.receiver = rx.Receiver(Nr, constellation_type, Pt, B)

    def __str__(self):
        """ String representation of the SU-MIMO DigCom system. """
        return f"({self.Nt}x{self.Nr} {self.type}) SU-MIMO SVD DigCom System"
    
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
        print(f"\nStarting SU-MIMO SVD BER Calculation for \n{str(self)} ...")

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
        Cs = np.mean(Cs_channels, axis=0)
        self.channel = channel

        return BERs, Cs

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
        s, C_total = self.transmitter.simulate(bits, SNR, self.channel.get_CSI(), self.M)
        if C_total == 0: return None, C_total

        # 2. Channel
        r = self.channel.simulate(s, SNR)

        # 3. Receiver
        bits_hat = self.receiver.simulate(r, SNR, self.channel.get_CSI(), self.M)

        return bits_hat[:len(bits)], C_total
    
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
            If the transmission fails (due to a zero capacity channel), the BER is set to NaN.
        C_total : int
            The total used capacity of the SU-MIMO SVD system during the transmission.
        """
        
        bits_hat, C_total = self.simulate(bits, SNR)
        BER = np.sum(bits_hat != bits) / bits.size if C_total > 0 else np.nan

        return BER, C_total

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
        BERs : 1D numpy array (dtype: float)
            The simulated Bit Error Rates (BERs) corresponding to each SNR value.
        Cs : 1D numpy array (dtype: int)
            The total used capacity corresponding to each SNR value. This is the number of bits per data symbol vector.
        """
        print(f"\nStarting SU-MIMO SVD BER Simulation for \n{str(self)} ...")


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
                ber, C_SNR = self.simulate_BER(bits, SNR)

                Cs_SNR.append(C_SNR)
                if C_SNR == 0: continue

                counted_errors += round(num_bits*ber)
                counted_channels += 1
            
            BERs.append(counted_errors / (num_bits*counted_channels))
            Cs.append(np.mean(np.array(Cs_SNR)))

        return BERs, Cs


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
            The markers for each BER curve. Default is an 'o' marker for every curve.
        title : str, optional
            The title of the plot. Default is no title.
        """

        # Set default colors and markers if not provided.
        if colors is None: colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'] + ['black']*(len(BERs_list) - 10)
        if marker_colors is None: marker_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'] + ['black']*(len(BERs_list) - 10)
        if markers is None: markers = ['o'] * len(BERs_list)

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
            The markers for each capacity curve. Default is an 'o' marker for every curve.
        title : str, optional
            The title of the plot. Default is None.
        """

        # Set default colors and markers if not provided.
        if colors is None: colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'] + ['black']*(len(Cs_list) - 10)
        if marker_colors is None: marker_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'] + ['black']*(len(Cs_list) - 10)
        if markers is None: markers = ['o'] * len(Cs_list)

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

    test = 'Theoretical BER & Simulated BER'

    if test == 'Theoretical BER Approximation':

        # Setup.
        SNRs = np.arange(-5, 26, 2.5)
        system_configs = [ (1, 1, 'PSK'), (2, 2, 'PSK'), (4, 2, 'PSK'), (2, 4, 'PSK'), (4, 4, 'PSK'), (8, 8, 'PSK') ]
        num_channels = 50000

        BERs_list = []
        Cs_list = []
        labels = []

        for Nt, Nr, type in system_configs:

            # Initialize SU-MIMO SVD system.
            su_mimo_svd = SuMimoSVD(Nt, Nr, type)
            label = f'{type} ({Nt}x{Nr})'

            # Calculate theoretical BER approximations and the used capacities.
            BERs, Cs = su_mimo_svd.BER_approximation(SNRs, num_channels=num_channels)

            # Store results.
            BERs_list.append(BERs)
            Cs_list.append(Cs)
            labels.append(label)


        # Plot results.
        fig, ax = su_mimo_svd.plot_BERs(SNRs, BERs_list, labels, title=f'SU-MIMO SVD System Theoretical BER Approximation\n (averaged over {num_channels//1000}K channel realizations)')
        fig, ax = su_mimo_svd.plot_Cs(SNRs, Cs_list, labels, title=f'SU-MIMO SVD System Bit Rate\n (averaged over {num_channels//1000}K channel realizations)')
        plt.show()

    if test == 'Simulation Example':

        # Initialize the system's properties.
        Nt = 3
        Nr = 2
        type = 'PSK'

        # Initialize a predefined channel matrix.
        H1 = np.eye(Nt, Nr) 
        H2 = np.array([[3, 2, 2], [2, 3, -2]])

        # Initialize SU-MIMO SVD system.
        su_mimo_svd = SuMimoSVD(Nt=Nt, Nr=Nr, constellation_type=type, H=H2)
        ber = su_mimo_svd.print_simulation_example(bits=np.random.randint(0, 2, size=16000), SNR=np.inf, K=100)

    if test == 'Simulated BER':

        # Setup.
        SNRs = np.arange(-5, 31, 5)
        system_configs = [ (1, 1, 'QAM'), (2, 2, 'QAM'), (4, 2, 'QAM'), (2, 4, 'QAM'), (4, 4, 'QAM'), (8, 8, 'QAM') ]
        num_errors = 500
        num_channels = 20

        BERs_list = []
        Cs_list = []
        labels = []

        for Nt, Nr, type in system_configs:

            # Initialize SU-MIMO SVD system.
            su_mimo_svd = SuMimoSVD(Nt, Nr, type)
            label = f'{type} ({Nt}x{Nr})'

            # Calculate theoretical BER approximations and the used capacities.
            BERs, Cs = su_mimo_svd.simulate_BERs(SNRs, num_errors, num_channels)

            # Store results.
            BERs_list.append(BERs)
            Cs_list.append(Cs)
            labels.append(label)


        # Plot results.
        fig, ax = su_mimo_svd.plot_BERs(SNRs, BERs_list, labels, title=f'SU-MIMO SVD System\n Simulation BER vs SNR')
        fig, ax = su_mimo_svd.plot_Cs(SNRs, Cs_list, labels, title=f'SU-MIMO SVD System\n Simulation Bit Rate vs SNR')
        plt.show()

    if test == 'Theoretical BER & Simulated BER':
        
        # Setup.
        SNRs = np.arange(-5, 26, 5)
        system_configs = [(2, 2, 'PAM'), (2, 4, 'PAM'), (4, 4, 'PAM')]

        BERs_list = []
        Cs_list = []
        labels = []
        colors = ['tab:red', 'tab:blue'] * len(system_configs)
        marker_colors = [color for color in ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'] for _ in range(2)] + ['black']*(len(system_configs) - 8)
        num_channels_theory = 10000
        num_errors_sim = 500
        num_channels_sim = 20


        for Nt, Nr, type in system_configs:

            # Initialize SU-MIMO SVD system.
            su_mimo_svd = SuMimoSVD(Nt, Nr, type)
            label = f'{type} ({Nt}x{Nr})'

            # Calculate theoretical BER and the used capacities.
            BERs_theory, Cs_theory = su_mimo_svd.BER_approximation(SNRs, num_channels_theory)

            # Store results.
            BERs_list.append(BERs_theory)
            Cs_list.append(Cs_theory)
            labels.append(label + ' theoretical')

            # Calculate simulated BER and the used capacities.
            BERs_sim, Cs_sim = su_mimo_svd.simulate_BERs(SNRs, num_errors_sim, num_channels_sim)

            # Store results.
            BERs_list.append(BERs_sim)
            Cs_list.append(Cs_sim)
            labels.append(label + ' simulation')


        # Plot results.
        fig, ax = su_mimo_svd.plot_BERs(SNRs, BERs_list, labels, colors=colors, marker_colors=marker_colors, title=f'SU-MIMO SVD DigCom System\nTheoretical Approximations & Simulations')
        fig, ax = su_mimo_svd.plot_Cs(SNRs, Cs_list, labels, colors=colors, marker_colors=marker_colors, title=f'SU-MIMO SVD DigCom System\nTheoretical Approximations & Simulations')
        plt.show()