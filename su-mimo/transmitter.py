# This module contains the implementation of the transmitter component of a SU-MIMO SVD communication system.

import numpy as np
import matplotlib.pyplot as plt


class Transmitter:
    """
    Description
    -----------
    The transmitter of a single-user multiple-input multiple-output (SU-MIMO) communication system, in which the channel state information is available at the transmitter (and receiver).

    When the transmitter is called and given an input bit sequence, it executes the waterfilling algorithm to determine the constellation size and power allocation for each transmit antenna, allocates the input bits across the transmit antennas, maps the input bit sequences to the corresponding data symbol sequences according to a specified modulation constellation for each transmit antenna, allocates power across the transmit antennas, and precodes the data symbols using the right singular vectors of the channel matrix H.

    Attributes
    ----------
    Nt : int
        Number of transmit antennas.
    constellation_type : str
        Type of modulation constellation ('PAM', 'PSK', or 'QAM').
    Pt : float, optional
        Total available transmit power. Default is 1.0.
    B : float, optional
        Bandwidth of the communication system. Default is 0.5.
    
    Methods
    -------
    __init__()
        Initialize the transmitter parameters.
    __str__()
        Return a string representation of the transmitter object.
    __call__()
        Allow the transmitter object to be called as a function. When called, it executes the simulate() method.
    
    waterfilling()
        Execute the waterfilling algorithm to determine the power allocation and constellation size for each transmit antenna.
    bit_allocator()
        Allocate the input bits across the transmit antennas based on the constellation size for each antenna.
    mapper()
        Convert bit sequence vectors into the corresponding data symbol vectors according to the specified modulation constellation for each transmit antenna.
    power_allocator()
        Allocate power across the transmit antennas based on the power allocation for each antenna.
    precoder()
        Precode the data symbols using the right singular vectors of the channel matrix H.
    simulate()
        Simulate the transmitter operations and return the output signal ready to be transmitted through the MIMO channel.
    
    plot_bit_allocation()
        Plot the bit allocation across the transmit antennas as determined by the waterfilling algorithm.
    plot_power_allocation()
        Plot the power allocation across the transmit antennas as determined by the waterfilling algorithm.
    plot_channel_capacities()
        Plot the channel capacities and used channel capacities in function of the SNR.
    plot_data_symbols()
        Plot the data symbols in the complex plane.
    """


    # INITIALIZATION AND REPRESENTATION

    def __init__(self, Nt: int, constellation_type: str, Pt: float = 1.0, B: float = 0.5) -> None:
        """ Initialize the transmitter. """

        self.Nt = Nt
        self.Pt = Pt
        self.B = B
        self.type = constellation_type

    def __str__(self) -> str:
        """ Return a string representation of the transmitter object. """
        return f"Transmitter: \n  - Number of antennas = {self.Nt}\n  - Constellation = {self.type}"

    def __call__(self, bits: np.ndarray, SNR: float, CSI: tuple, M: int = None) -> np.ndarray:
        """ Allow the transmitter object to be called as a function. When called, it executes the simulate() method. """
        return self.simulate(bits, SNR, CSI, M)

    # FUNCTIONALITY

    def waterfilling(self, H: np.ndarray, S: np.ndarray, N0: float) -> tuple:
        """
        Description
        -----------
        Allocate power and constellation size across the transmit antennas so the total capacity of the MIMO system is maximized, using the waterfilling algorithm.\n
        (1) Execute the waterfilling algorithm. The total transmit power Pt, the bandwidth B, the noise power spectral density N0, and the CSI are given.\n
        (2) Calculate the capacity of each eigenchannel.\n
        (3) Determine the constellation size for each transmit antenna.\n

        Parameters
        ----------
        H : numpy.ndarray
            Channel matrix.
        S : numpy.ndarray
            Singular values of the channel matrix H.
        N0 : float
            Noise power spectral density.

        Returns
        -------
        Pi, Ci, Mi : tuple (3 times 1D numpy array of length used_eigenchannels)
            - Pi: The optimal power allocation for each transmit antenna.
            - Ci: The capacity of each eigenchannel.
            - Mi: The constellation size for each transmit antenna.

        Notes
        -----
        In real-world scenarios, every antenna itself has a power constraint as well. However, in this implementation, we only consider a total power constraint across all antennas.
        """

        # 1. Waterfilling Algorithm.

        # Initialization.
        gamma = self.Pt / (2*self.B*N0)
        used_eigenchannels = np.linalg.matrix_rank(H)
        waterlevel = (gamma / used_eigenchannels) + (1 / used_eigenchannels) * np.sum(1 / (S[:used_eigenchannels]**2))

        # Iteration.
        while ( waterlevel < (1 / (S[used_eigenchannels-1]**2)) ):
            used_eigenchannels -= 1
            waterlevel = (gamma / used_eigenchannels) + (1 / used_eigenchannels) * np.sum(1 / (S[:used_eigenchannels]**2))

        # Termination.
        Pi = np.maximum((waterlevel - (1 / (S[:used_eigenchannels]**2))) * (2*self.B*N0), 0)


        # 2. Eigenchannel Capacities.
        Ci = 2*self.B * np.log2( 1 + (Pi * (S[:used_eigenchannels]**2)) / (2*self.B*N0) )


        # 3. Constellation Sizes.
        Mi = 2 ** np.floor( Ci ).astype(int) if self.type != 'QAM' else 4 ** np.floor( Ci / 2 ).astype(int)


        # 4. Return.
        return Pi, Ci, Mi

    def bit_allocator(self, bitstream: np.ndarray, Mi: np.ndarray) -> np.ndarray:
        """
        Description
        -----------
        Allocate the input bitstream across the transmit antennas based on the constellation size for each antenna. 
        
        Every antenna will have to send an equal amount of data symbols. If the number of bits in the bitstream does not perfectly align with an an equal amount of symbols per antenna, it is padded with zeros.
        If zero power is allocated to an eigenchannel (or thus the used capacity of that eigenchannel equals zero), no bits will be allocated to that eigenchannel.

        Parameters
        ----------
        bitstream : 1D numpy array
            Input bit sequence to be allocated across the transmit antennas.
        Mi : 1D numpy array
            Size of the constellation for each transmit antenna.

        Returns
        -------
        bits : list of 1D numpy arrays
            Output bit sequence vector allocated across the transmit antennas.

        Raises
        ------
        ValueError
            If the number of transmit antennas does not match the length of Mi.
        ValueError
            If the constellation size M of any antenna is invalid.
        """

        assert self.Nt == Mi.size, f'The number of transmit antennas does not match the number of given constellation sizes.\nNumber of transmit antennas is {self.Nt}, while length of Mi is {Mi.size}.'
        for i, M in enumerate(Mi): assert (M & (M - 1) == 0) and ((M & 0xAAAAAAAA) == 0 or self.type != 'QAM'), f'The constellation size M of antenna {i} is invalid.\nFor PAM and PSK modulation, it must be a power of 2. For QAM Modulation, M must be a power of 4. Right now, M equals {M} and the type is {self.type}.'
        
        C_eigenchannels = np.log2(Mi).astype(int)
        C_total = np.sum( C_eigenchannels ).astype(int) # What if C_total = 0 ??
        N_symbols = np.ceil( len(bitstream) / C_total ).astype(int)
        bitstream = np.pad( bitstream, pad_width=(0, N_symbols*C_total - len(bitstream)), mode='constant', constant_values=0 )

        bits = [np.array([], dtype=int)] * self.Nt
        while bitstream.size > 0:
            for i, mc in enumerate(C_eigenchannels):
                bits[i] = np.concatenate((bits[i], bitstream[:mc]))
                bitstream = bitstream[mc:]

        return bits

    def map(self, bits: np.ndarray, M: int, type: str) -> np.ndarray:
        """
        Description
        -----------
        Convert a bit sequence into the corresponding data symbol sequence according to the specified modulation constellation.

        Parameters
        ----------
        bits : 1D numpy array
            Input bit sequence to be mapped.
        M : int
            Size of the constellation.
        type : str
            Type of the constellation ('PAM', 'PSK', or 'QAM').
        
        Returns
        -------
        symbols : 1D numpy array
            Output data symbol sequence corresponding to the input bit sequence.
        
        Raises
        ------
        ValueError
            If the constellation size M is invalid. It must be greater than 1 and a power of 2. For QAM modulation, it must be a power of 4 instead of 2.
        ValueError
            If the length of the bit sequence is invalid. It must be a multiple of log2(M).
        ValueError
            If the constellation type is invalid. Choose only between 'PAM', 'PSK', or 'QAM'.
        """

        
        # 1. Divide the input bit sequences into blocks of mc bits, where M = 2^mc.

        assert (M & (M - 1) == 0) and ((M & 0xAAAAAAAA) == 0 or self.type != 'QAM'), f'The constellation size M of antenna {i} is invalid.\nFor PAM and PSK modulation, it must be a power of 2. For QAM Modulation, M must be a power of 4. Right now, M equals {M} and the type is {self.type}.'
        assert (bits.size % int(np.log2(M)) == 0), f'The length of the bit sequences is invalid.\nThey must be a multiple of log2(M). Right now, length is {bits.size} and log2(M) is {mc}.' 

        mc = int(np.log2(M))
        bits = bits.reshape((bits.size // mc, mc))


        # 2. Convert the blocks of mc bits from gray code to the corresponding decimal value.
        
        graycodes = bits
        binarycodes = np.zeros_like(graycodes)
        binarycodes[:, 0] = graycodes[:, 0]

        for i in range(1, graycodes.shape[1]):
            binarycodes[:, i] = binarycodes[:, i-1] ^ graycodes[:, i]
        
        decimals = np.dot(binarycodes, (2**np.arange(mc))[::-1])  


        # 3. Convert the decimal values to the corresponding data symbols, according to the specified constellation type.

        if type == 'PAM' :
            delta = np.sqrt(3/(M**2-1))
            symbols = (2*decimals - (M-1)) * delta

        elif type == 'PSK' :
            symbols = np.exp(1j * 2*np.pi * decimals / M)

        elif type == 'QAM' :
            c_sqrtM_PAM = np.arange(-(np.sqrt(M)-1), (np.sqrt(M)-1) + 1, 2) * np.sqrt(3 / (2*(M-1)))
            real_grid, imaginary_grid = np.meshgrid(c_sqrtM_PAM, c_sqrtM_PAM[::-1])
            constellation = (real_grid + 1j*imaginary_grid)
            constellation[1::2] = constellation[1::2, ::-1]
            constellation = constellation.flatten()

            symbols = constellation[decimals]



        else :
            raise ValueError(f'The constellation type is invalid.\nChoose between "PAM", "PSK", or "QAM". Right now, type is {type}.')
        

        # 4. Return
        return symbols

    def mapper(self, bits: list, Mi: np.ndarray) -> np.ndarray:
        """
        Description
        -----------
        Convert a bit sequence vector into the corresponding data symbol vectors according to the specified modulation constellation for each transmit antenna.

        Parameters
        ----------
        bits : list of 1D numpy arrays
            Input bit sequence vector to be mapped.
        Mi : 1D numpy array
            Size of the constellation for each transmit antenna.
        type : str
            Type of the constellation ('PAM', 'PSK', or 'QAM').

        Returns
        -------
        symbols : 2D numpy array
            Output data symbol vectors corresponding to the input bit sequence vectors.
        
        Raises
        ------
        ValueError
            If the number of transmit antennas does not match the number of given bit sequences.
        ValueError
            If the number of transmit antennas does not match the length of Mi.
        """

        assert self.Nt == Mi.size, f'The number of transmit antennas does not match the number of given constellation sizes.\nNumber of transmit antennas is {self.Nt}, while length of Mi is {Mi.size}.'
        assert self.Nt == len(bits), f'The number of transmit antennas does not match the number of given bit sequences.\nNumber of transmit antennas is {self.Nt}, while number of bit sequences is {len(bits)}.'

        symbols = [0] * self.Nt
        for tx_antenna in range(self.Nt):
            symbols[tx_antenna] = self.map(bits[tx_antenna], Mi[tx_antenna], self.type) if Mi[tx_antenna] >= 2 else np.zeros_like(symbols[tx_antenna-1], dtype=complex)
        symbols = np.array(symbols)
        return symbols

    def power_allocator(self, symbols: np.ndarray, Pi: np.ndarray) -> np.ndarray:
        """
        Description
        -----------
        Allocate power across the transmit antennas so the total capacity of the MIMO system is maximized.

        Parameters
        ----------
        symbols : 2D numpy array
            Input data symbol sequence for each transmit antenna. Shape is (Nt, N_symbols).
        Pi : 1D numpy array
            Optimal power allocation for each transmit antenna. Shape is (Nt,).
        
        Returns
        -------
        powered_symbols : 2D numpy array
            Data symbol sequence with power allocated for each transmit antenna. Shape is (Nt, N_symbols).
        """
        
        powered_symbols = np.dot(np.diag(np.sqrt(Pi)), symbols)
        return powered_symbols

    def precoder(self, powered_symbols: np.ndarray, Vh: np.ndarray) -> np.ndarray:
        """ 
        Description
        -----------
        Precode the powered data symbols using the right singular vectors of the channel matrix H.

        Parameters
        ----------
        powered_symbols : 2D numpy array
            Input data symbol sequence with power allocated for each transmit antenna. Shape is (Nt, N_symbols).
        Vh : 2D numpy array
            Right singular vectors of the channel matrix H. Shape is (Nt, Nt).
        Returns
        -------
        precoded_symbols : 2D numpy array
            Precoded data symbol sequence ready for transmission through the MIMO channel. Shape is (Nt, N_symbols).
        """

        precoded_symbols = np.dot(Vh.conj().T, powered_symbols)
        return precoded_symbols

    def simulate(self, bits: np.ndarray, SNR: float, CSI: tuple, M: int = None) -> np.ndarray:
        """
        Description
        -----------
        Simulate the transmitter operations:\n
        (1) Get the channel state information.\n
        (2) Execute the waterfilling algorithm to determine the constellation size and power allocation for each transmit antenna.\n Note: If a fixed constellation size M is provided as input, the waterfilling algorithm is skipped and this constellation size is used for all antennas.\n
        (3) [bit_allocator] Divide the input bits across the transmit antennas.\n
        (4) [mapper] Map the input bit sequence to the corresponding data symbol sequence for each transmit antenna.\n
        (5) [power_allocator] Allocate power across the transmit antennas.\n
        (6) [precoder] Precode the data symbols using the right singular vectors of the channel matrix H.\n
        (7) Transmit the precoded symbols through the MIMO channel.\n

        The output signal is ready to be transmitted through the MIMO channel.

        Parameters
        ----------
        bits : numpy.ndarray
            The input bit sequences for each transmit antenna to be transmitted.
        SNR : float
            The signal-to-noise ratio in dB.
        CSI : tuple
            The channel state information, consisting of the channel matrix H, left singular vectors U, singular values S, and right singular vectors Vh.
        M : int, optional
            The fixed constellation size for all transmit antennas. If provided, the waterfilling algorithm is skipped and this constellation size is used for all antennas.

        Returns
        -------
            s (numpy.ndarray): The output signal.
        """

        # Setup.
        N0 = self.Pt / ((10**(SNR/10.0)) * 2*self.B)
        H, U, S, Vh = CSI

        if M is None:
            Pi, Ci, Mi = self.waterfilling(H, S, N0)
            Pi = np.pad(Pi, pad_width=(0, self.Nt - len(Pi)), mode='constant', constant_values=0)
            Mi = np.pad(Mi, pad_width=(0, self.Nt - len(Mi)), mode='constant', constant_values=1)
        else:
            rank_H = np.linalg.matrix_rank(CSI[0])
            Mi = np.array([M] * rank_H)
            Pi = np.array([self.Pt / rank_H] * rank_H)

        # Transmitter Operations.
        bits = self.bit_allocator(bits, Mi)
        symbols = self.mapper(bits, Mi)
        powered_symbols = self.power_allocator(symbols, Pi)
        precoded_symbols = self.precoder(powered_symbols, Vh)

        return precoded_symbols


    # TESTS AND PLOTS

    def plot_bit_allocation(self, SNR: float, CSI: tuple) -> None:
        """
        Plot the bit allocation across the transmit antennas as determined by the waterfilling algorithm, so that the total capacity of the MIMO system is maximized.
        On the x-axis, the transmit antennas are shown. On the y-axis, the number of bits allocated to each transmit antenna is shown, as well as the capacity of the eigenchannel corresponding to each transmit antenna.

        Parameters
        ----------
        SNR : float
            The signal-to-noise ratio in dB.
        CSI : tuple
            The channel state information (H, U, S, Vh).
        """

        N0 = self.Pt / ((10**(SNR/10.0)) * 2*self.B)
        Pi, Ci, Mi = self.waterfilling(CSI[0], CSI[2], N0)
        mc = np.log2(Mi).astype(int)

        x = np.arange(len(Ci))
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(x - 0.35/2, Ci, width=0.35, color='tab:green', label='Channel capacity')
        ax.bar(x + 0.35/2, mc, width=0.35, color='tab:blue', label='Constellation size')
        ax.set_xlabel('Transmit antenna')
        ax.set_ylabel('Capacity [bits]')
        ax.set_title(f'Antenna Bit Allocation\n\n{self.Nt} transmit antennas and {CSI[0].shape[0]} receive antennas \nSNR: {SNR} dB')
        ax.set_xticks(x)
        ax.set_xticklabels(x + 1)
        ax.set_xlim(-0.5, len(Ci) - 0.5)
        ax.legend(loc='upper right')
        fig.tight_layout()

        return fig, ax

    def plot_power_allocation(self, SNR: float, CSI: tuple) -> None:
        """
        Plot the power allocation across the transmit antennas as determined by the waterfilling algorithm.
        On the x-axis, the transmit antennas are shown. On the y-axis, the inverse channel gain is shown in grey and the amount of power allocated to each transmit antenna is shown in blue.

        Parameters
        ----------
        SNR : float
            The signal-to-noise ratio in dB.
        CSI : tuple
            The channel state information (H, U, S, Vh).
        
        Returns
        -------
        fig, ax : tuple
            The figure and axis objects of the plot.
        """

        N0 = self.Pt / ((10**(SNR/10.0)) * 2*self.B)
        S = CSI[2]
        Pi = self.waterfilling(CSI[0], S, N0)[0]
        Pi = np.pad(Pi, pad_width=(0, self.Nt - len(Pi)), mode='constant', constant_values=0)

        inverse_channel_gains = (2*self.B*N0) / (S[S > 0] ** 2)
        waterlevels = (Pi[Pi > 0] + inverse_channel_gains[Pi[:len(S[S > 0])] > 0])

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(np.arange(1, len(waterlevels) + 1), waterlevels, color='tab:blue', label='Allocated power')
        ax.bar(np.arange(1, len(inverse_channel_gains) + 1), inverse_channel_gains, color='tab:grey', label='Inverse channel gain')
        ax.axhline(y=np.mean(waterlevels), color='tab:red', linestyle='--', linewidth=3, label='Water level')
        ax.set_xlabel('Transmit antenna')
        ax.set_ylabel('Power')
        ax.set_title(f'Antenna Power Allocation\n\n{self.Nt} transmit antennas and {CSI[0].shape[0]} receive antennas \nSNR: {SNR} dB, Total transmit power: {round(self.Pt, 2)} W, Noise Power: {round(2*self.B*N0, 2)} W')
        ax.set_xticks(np.arange(1, len(inverse_channel_gains) + 1))
        ax.set_xlim(0.5, len(inverse_channel_gains) + 0.5)
        ax.legend(loc='upper left')
        fig.tight_layout()
        
        return fig, ax

    def plot_channel_capacities(self, CSI: tuple = None, N: int = 100, Nr: int = None) -> None:
        """
        Description
        -----------
        Plot the channel capacities and used channel capacities versus SNR. Because the constellation sizes are constrained to powers of 2 (or 4 for QAM), the used channel capacity for each eigenchannel is generally lower than the theoretical channel capacity for that eigenchannel.

        Two options are possible:
            - Default option. An average is taken over N random channel realizations.
            - Provide the channel state information (CSI) as input. In this case, the channel capacities are calculated for that specific channel.

        Parameters
        ----------
        CSI : tuple, optional
            The channel state information (H, U, S, Vh). If provided, the channel capacities are calculated for that specific channel. If None, an average is taken over N random channel realizations. Default is None.
        N : int, optional
            The number of random channel realizations to average over if CSI is not provided. Default is 100.
        Nr : int, optional
            The number of receive antennas. Required if CSI is not provided. Default is None.
        """

        SNRs = range(-10, 41, 5)
        channel_capacities = []
        used_channel_capacities = []

        for SNR in SNRs:

            N0 = self.Pt / ((10**(SNR/10.0)) * 2*self.B)

            if CSI is not None:

                Pi, Ci, Mi = self.waterfilling(CSI[0], CSI[2], N0)
                channel_capacities.append( np.sum(Ci) )
                used_channel_capacities.append( np.sum( np.log2(Mi) ) )

            else:

                channel_capacities_i = []
                used_channel_capacities_i = []

                for _ in range(N):
                    H = (1/np.sqrt(2)) * (np.random.randn(Nr, self.Nt) + 1j * np.random.randn(Nr, self.Nt))
                    S = np.linalg.svd(H)[1]

                    Pi, Ci, Mi = self.waterfilling(H, S, N0)
                    channel_capacities_i.append( np.sum(Ci) )
                    used_channel_capacities_i.append( np.sum( np.log2(Mi) ) )

                channel_capacities.append( np.mean(channel_capacities_i) )
                used_channel_capacities.append( np.mean(used_channel_capacities_i) )


        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(SNRs, channel_capacities, color='tab:green', marker='o', label='Channel Capacity')
        ax.plot(SNRs, used_channel_capacities, color='tab:blue', marker='o', label='Used Channel Capacity')
        ax.set_xlabel('SNR [dB]')
        ax.set_ylabel('Capacity [bits]')
        ax.set_title((f'Channel Capacity vs SNR\n\n{self.Nt} transmit antennas, {Nr if Nr is not None else H.shape[0]} receive antennas, {self.type} modulation\n') + (f'Averaged over {N} random channel realizations.' if CSI is None else 'Calculated for specific channel realization.'))
        ax.set_xticks(SNRs)
        ax.legend(loc='upper left')
        ax.grid(True)
        fig.tight_layout()
        
        return fig, ax

    def plot_data_symbols(self, symbols: np.ndarray, N: int = 1) -> None:
        """
        Plot the first N data symbols vectors in the complex plane. 
        Each data symbol vector corresponds to the data symbols transmitted from all transmit antennas at a specific time instant. The color of the markers corresponds to the transmit antenna.

        Parameters
        ----------
        symbols : 2D numpy array
            Data symbol sequence for each transmit antenna. Shape is (Nt, N_symbols).
        N : int, optional
            Number of data symbol vectors to plot. Must be less than or equal to N_symbols. Default is 1.

        Returns
        -------
        fig, ax : tuple
            The figure and axis objects of the plot.
        """

        fig, ax = plt.subplots(figsize=(6, 6))
        for tx_antenna in range(self.Nt):
            ax.scatter(symbols[tx_antenna, :N].real, symbols[tx_antenna, :N].imag, alpha=0.7, s=50, label=f'Antenna {tx_antenna+1}')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.set_xlabel('Real Part')
        ax.set_ylabel('Imaginary Part')
        ax.set_title(f'Data Symbols in Complex Plane\n\n{self.Nt} transmit antennas, {self.type} modulation')
        ax.grid()
        ax.axis('equal')
        ax.legend()
        fig.tight_layout()
                
        return fig, ax

    def print_simulation_example(self, bits: np.ndarray, SNR: float, CSI: tuple, K: int = 1) -> None:
        """
        Description
        -----------
        Print a step-by-step example of the transmitter operations (see simulate() method) for given input bits, SNR, and channel state information (CSI). Only the first K data symbols vectors are considered.

        Parameters
        ----------
        bits : numpy.ndarray
            The input bit sequences for each transmit antenna to be transmitted.
        SNR : float
            The signal-to-noise ratio in dB.
        CSI : tuple
            The channel state information (H, U, S, Vh).
        K : int, optional
            The maximum number of data symbol vectors to consider in the example.
        
        Notes
        -----
        For demonstration purposes only.
        """

        # PRINTING EXAMPLE
        print("\n\n========== Transmitter Simulation Example ==========\n")
        
        # 0. Print the input bit sequence.
        print(f"----- the input bit sequence -----\n{bits}\n\n")
        
        # 1. Get the channel state information.
        N0 = self.Pt / ((10**(SNR/10.0)) * 2*self.B)
        H, U, S, Vh = CSI
        print(f"----- the channel state information -----\n\n{np.round(H, 2)}\nS = {np.round(S, 2)}\n\n")

        # 2. Execute the waterfilling algorithm to determine the constellation size and power allocation for each transmit antenna.
        Pi, Ci, Mi = self.waterfilling(H, S, N0)
        Pi = np.pad(Pi, pad_width=(0, self.Nt - len(Mi)), mode='constant', constant_values=0)
        Mi = np.pad(Mi, pad_width=(0, self.Nt - len(Mi)), mode='constant', constant_values=1)
        print(f"----- waterfilling algorithm results -----\n Power allocation Pi = {np.round(Pi, 2)},\n Channel capacities Ci = {np.round(Ci, 2)},\n Constellation sizes Mi = {Mi}\n\n")
        
        # 3. Divide the input bits accross the transmit antennas.
        bits = self.bit_allocator(bits, Mi)
        bits = [b[:K * int(np.log2(Mi[i]))] for i, b in enumerate(bits)]
        print(f"----- the input bits, allocated across transmit antennas -----\n")
        for i, row in enumerate(bits): print(f'Antenna {i+1}: ' + '[' + ' '.join(map(str, row)) + ']')

        # 4. Map the input bit sequence to the corresponding data symbol sequence for each transmit antenna.
        symbols = self.mapper(bits, Mi)
        print(f"\n\n----- the data symbols for each transmit antenna -----\n{np.round(symbols, 2)}\n\n")

        # 5. Allocate power across the transmit antennas.
        powered_symbols = self.power_allocator(symbols, Pi)
        print(f"----- the data symbols with power allocated for each transmit antenna -----\n{np.round(powered_symbols, 2)}\n\n")

        # 6. Precode the data symbols.
        precoded_symbols = self.precoder(powered_symbols, Vh)
        print(f"----- the precoded data symbols ready for transmission -----\n{np.round(precoded_symbols, 2)}\n\n")

        print("======== End of Simulation Example ========")


        # PLOTS
        fig1, ax1 = self.plot_bit_allocation(SNR, CSI)
        fig2, ax2 = self.plot_power_allocation(SNR, CSI)
        fig3, ax3 = self.plot_data_symbols(symbols, N=K)
        plt.show()


        # RETURN
        return




if __name__ == "__main__":

    # Initialize the transmitter.
    transmitter = Transmitter(Nt=5, constellation_type='PSK')
    
    # Initialize the channel.
    H = (1/np.sqrt(2)) * (np.random.randn(transmitter.Nt-1, transmitter.Nt) + 1j * np.random.randn(transmitter.Nt-1, transmitter.Nt))
    U, S, Vh = np.linalg.svd(H)
    CSI = (H, U, S, Vh)

    # Transmitter simulation example.
    transmitter.print_simulation_example(np.random.randint(0, 2, size=100), SNR=25, CSI=CSI, K=3)
