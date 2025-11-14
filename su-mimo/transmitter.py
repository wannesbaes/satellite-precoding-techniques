# This module contains the implementation of the transmitter component of a SU-MIMO SVD communication system.

import numpy as np
import matplotlib.pyplot as plt


class Transmitter:
    """
    Description
    -----------
    The transmitter of a single-user multiple-input multiple-output (SU-MIMO) digital communication system, in which the channel state information is available at the transmitter (and the receiver).

    When the transmitter is called and given an input bit sequence, it ...
        executes the waterfilling algorithm to determine the constellation size and power allocation for each transmit antenna (unless a constant constellation size is specified)
        allocates the input bits across the transmit antennas
        maps the input bit sequences to the corresponding data symbol sequences according to a specified modulation constellation for each transmit antenna
        allocates power across the transmit antennas
        and precodes the data symbols using the right singular vectors of the channel matrix H.

    Attributes
    ----------
    Nt : int
        Number of transmit antennas.
    c_type : str
        Constellation type. (Choose between 'PAM', 'PSK', or 'QAM')
    c_size : int, optional
        Constellation size. Only used if a fixed constellation size is desired. Default is None.
    data_rate : float, optional
        The data rate at which data is transmitted. It is specified is the fraction of the channel capacity. Default is 1.0.
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
    
    get_resource_allocation()
        Return the power allocation, eigenchannel capacities, and constellation sizes of each transmit antenna, for the current CSI and SNR.
    waterfilling()
        Execute the waterfilling algorithm to determine the power allocation and constellation size for each transmit antenna.
    map()
        Convert a bit sequence into the corresponding data symbol sequence according to the specified modulation constellation.
    
    resource_allocation()
        Determine and store the constellation size and power allocation for each transmit antenna, for the current channel state information (CSI) and signal-to-noise ratio (SNR).
    bit_allocator()
        Allocate the input bitstream across the transmit antennas based on the calculated constellation size for each antenna.
    mapper()
        Convert bit vectors into the corresponding data symbol vectors according to the specified modulation constellation for each transmit antenna.
    power_allocator()
        Allocate power across the transmit antennas based on the calculated power for each antenna.
    precoder()
        Precode the powered data symbol vectors using the right singular vectors of the channel matrix H.
    simulate()
        Simulate the transmitter operations and return the output signal ready to be transmitted through the MIMO channel.

    plot_bit_allocation()
        Plot the bit allocation across the transmit antennas as determined by the waterfilling algorithm.
    plot_power_allocation()
        Plot the power allocation across the transmit antennas as determined by the waterfilling algorithm.
    print_simulation_example()
        Print a step-by-step example of the transmitter operations for given input bits, SNR, and channel state information (CSI).
    """


    # INITIALIZATION AND REPRESENTATION

    def __init__(self, Nt, c_type, data_rate=1.0, c_size=None, Pt=1.0, B=0.5):
        """ Initialize the transmitter. """

        # Transmitter Settings.
        self.Nt = Nt
        self.data_rate = data_rate

        self.M = c_size
        self.c_type = c_type

        self.Pt = Pt
        self.B = B

        # Resource Allocation.
        self._Pi = None
        self._Ci = None
        self._Mi = None

    def __str__(self):
        """ Return a string representation of the transmitter object. """
        return f'Transmitter: \n  - Number of antennas = {self.Nt}\n  - Constellation = ' + (f'{self.M}' if self.M != None else '') + f'{self.c_type}'

    def __call__(self, bits, SNR, CSI):
        """ Allow the transmitter object to be called as a function. When called, it executes the simulate() method. """
        return self.simulate(bits, SNR, CSI)

    # FUNCTIONALITY

    def get_resource_allocation(self):
        """ Return the power allocation, eigenchannel capacities, and constellation sizes of each transmit antenna, for the current CSI and SNR. """
        return self._Pi, self._Ci, self._Mi

    def waterfilling(self, N0, H, S):
        """
        Description
        -----------
        Allocate power and constellation size across the transmit antennas so the total capacity of the MIMO system is maximized, using the waterfilling algorithm.\n
        (1) Execute the waterfilling algorithm. The total transmit power Pt, the bandwidth B, the noise power spectral density N0, and the CSI are given.\n
        (2) Calculate the capacity of each eigenchannel.\n
        (3) Determine the constellation size for each transmit antenna.\n

        Parameters
        ----------
        N0 : float
            Noise power spectral density.
        H : numpy.ndarray
            Channel matrix.
        S : numpy.ndarray
            Singular values of the channel matrix H.

        Returns
        -------
        Pi, Ci, Mi : tuple (3x 1D numpy array of length used_eigenchannels)
            - Pi: The optimal power allocation for each transmit antenna.
            - Ci: The capacity of each eigenchannel.
            - Mi: The constellation size for each transmit antenna.

        Notes
        -----
        In real-world scenarios, every antenna itself has a power constraint as well. However, in this implementation, we only consider a total power constraint across all antennas.
        """

        # 1. Waterfilling Algorithm.

        # Edge Case: The PSD of the noise is zero. The optimal strategy is to equally divide the power across all eigenchannels. The capacity of each eigenchannel becomes infinite. That's why we set the constelations sizes to a certain constant value (we can not choose an infinite constellation size).
        if N0 == 0:
            used_eigenchannels = np.linalg.matrix_rank(H)
            Pi = np.array( [self.Pt / used_eigenchannels] * used_eigenchannels )
            Ci = np.array([np.inf] * used_eigenchannels)
            Mi = np.array([4] * used_eigenchannels)
            return Pi, Ci, Mi

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
        Mi = 2 ** np.floor( Ci * self.data_rate ).astype(int) if self.c_type != 'QAM' else 4 ** np.floor( (Ci * self.data_rate) / 2 ).astype(int)


        # 4. Return.
        return Pi, Ci, Mi

    def map(self, bits, M, c_type):
        """
        Description
        -----------
        Convert a bit sequence into the corresponding data symbol sequence according to the specified modulation constellation.

        Parameters
        ----------
        bits : 1D numpy array (dtype=int, length=N_bits)
            Input bit sequence.
        M : int
            Constellation size.
        c_type : str
            Constellation type. (Choose between 'PAM', 'PSK', or 'QAM')
        
        Returns
        -------
        symbols : 1D numpy array (dtype=complex, length=N_bits/log2(M))
            Output data symbol sequence.
        
        Raises
        ------
        ValueError
            If the constellation size M is invalid. It must be greater than 1 and a power of 2. For QAM modulation, it must be a power of 4 instead of 2.
        ValueError
            If the length of the bit sequence is invalid. It must be a multiple of log2(M).
        ValueError
            If the constellation type is invalid. Choose only between 'PAM', 'PSK', or 'QAM'.
        """

        
        # 1. Divide the input bit sequences into blocks of m bits, where M = 2^m.

        assert (M & (M - 1) == 0) and ((M & 0xAAAAAAAA) == 0 or c_type != 'QAM'), f'The constellation size M of antenna {i} is invalid.\nFor PAM and PSK modulation, it must be a power of 2. For QAM Modulation, M must be a power of 4. Right now, M equals {M} and the type is {c_type}.'
        assert (bits.size % int(np.log2(M)) == 0), f'The length of the bit sequences is invalid.\nThey must be a multiple of log2(M). Right now, length is {bits.size} and log2(M) is {m}.' 

        m = int(np.log2(M))
        bits = bits.reshape((bits.size // m, m))


        # 2. Convert the blocks of m bits from gray code to the corresponding decimal value.
        
        graycodes = bits
        binarycodes = np.zeros_like(graycodes)
        binarycodes[:, 0] = graycodes[:, 0]

        for i in range(1, graycodes.shape[1]):
            binarycodes[:, i] = binarycodes[:, i-1] ^ graycodes[:, i]
        
        decimals = np.dot(binarycodes, (2**np.arange(m))[::-1])  


        # 3. Convert the decimal values to the corresponding data symbols, according to the specified constellation type.

        if c_type == 'PAM' :
            delta = np.sqrt(3/(M**2-1))
            symbols = (2*decimals - (M-1)) * delta

        elif c_type == 'PSK' :
            symbols = np.exp(1j * 2*np.pi * decimals / M)

        elif c_type == 'QAM' :
            c_sqrtM_PAM = np.arange(-(np.sqrt(M)-1), (np.sqrt(M)-1) + 1, 2) * np.sqrt(3 / (2*(M-1)))
            real_grid, imaginary_grid = np.meshgrid(c_sqrtM_PAM, c_sqrtM_PAM[::-1])
            constellation = (real_grid + 1j*imaginary_grid)
            constellation[1::2] = constellation[1::2, ::-1]
            constellation = constellation.flatten()

            symbols = constellation[decimals]

        else :
            raise ValueError(f'The constellation type is invalid.\nChoose between "PAM", "PSK", or "QAM". Right now, type is {c_type}.')
        

        # 4. Return
        return symbols


    def resource_allocation(self, SNR, CSI):
        """
        Description
        -----------
        Determine and store the constellation size and power allocation for each transmit antenna, for the current channel state information (CSI) and signal-to-noise ratio (SNR).\n

        There are two possible cases:
        - Case 1. No fixed constellation size M is provided as input. In this case, the waterfilling algorithm is executed to determine the optimal (!!) constellation size and power allocation for each transmit antenna.
        - Case 2. A fixed constellation size M is provided as input. In this case, the waterfilling algorithm is skipped and this constant constellation size is used for all antennas. The power is equally divided across the used eigenchannels.

        Parameters
        ----------
        SNR : float
            The signal-to-noise ratio in dB.
        CSI : tuple
            The channel state information (H, U, S, Vh).
        """

        # Case 1.
        if self.M is None:
            N0 = self.Pt / ((10**(SNR/10.0)) * 2*self.B)
            H, U, S, Vh = CSI
            Pi, Ci, Mi = self.waterfilling(N0, H, S)
        
        # Case 2.
        else:
            rank_H = np.linalg.matrix_rank(CSI[0])
            Pi = np.array([self.Pt / rank_H] * rank_H)
            Mi = np.array([self.M] * rank_H)
        
        # Pad Pi and Mi to match the number of transmit antennas Nt.
        Pi = np.pad(Pi, pad_width=(0, self.Nt - len(Pi)), mode='constant', constant_values=0)
        Mi = np.pad(Mi, pad_width=(0, self.Nt - len(Mi)), mode='constant', constant_values=1)

        # Store the results.
        self._Pi = Pi
        self._Ci = Ci
        self._Mi = Mi
        return

    def bit_allocator(self, bitstream):
        """
        Description
        -----------
        Allocate the input bitstream across the transmit antennas based on the calculated constellation size for each antenna. 
        
        Every antenna will have to send an equal amount of data symbols. If the number of bits in the bitstream does not perfectly align with an an equal amount of symbols per antenna, it is padded with zeros.
        If zero power is allocated to an eigenchannel (or thus the used capacity of that eigenchannel equals zero), no bits will be allocated to that eigenchannel.

        Parameters
        ----------
        bitstream : 1D numpy array (dtype=int, length=N_bits)
            Input bitstream.

        Returns
        -------
        b : list of 1D numpy arrays (dtype=int, length=(N_symbols * log2(Mi[tx_antenna])))
            Output bit vectors.
        """

        transmit_Rs = np.log2(self._Mi).astype(int)
        total_transmit_R = np.sum(transmit_Rs)
        
        N_symbols = np.ceil( len(bitstream) / total_transmit_R ).astype(int)
        bitstream = np.pad( bitstream, pad_width=(0, N_symbols*total_transmit_R - len(bitstream)), mode='constant', constant_values=0 )

        bits = [np.array([], dtype=int)] * self.Nt
        while bitstream.size > 0:
            for i, mc in enumerate(transmit_Rs):
                bits[i] = np.concatenate((bits[i], bitstream[:mc]))
                bitstream = bitstream[mc:]

        return bits

    def mapper(self, b):
        """
        Description
        -----------
        Convert bit vectors into the corresponding data symbol vectors according to the specified modulation constellation for each transmit antenna.

        Parameters
        ----------
        b : list of 1D numpy arrays (dtype=int, length=(N_symbols * log2(Mi[tx_antenna])))
            Input bit vectors to be mapped.

        Returns
        -------
        a : 2D numpy array (dtype=complex, shape=(Nt, N_symbols))
            Output data symbol vectors.
        """

        a = np.array([ self.map(b[tx_antenna], self._Mi[tx_antenna], self.c_type) for tx_antenna in range(len(self._Mi[self._Mi > 1])) ])
        a = np.concatenate( (a, np.zeros((self.Nt - a.shape[0], a.shape[1]), dtype=complex)) )
        return a

    def power_allocator(self, a):
        """
        Description
        -----------
        Allocate power across the transmit antennas based on the calculated power for each antenna.

        Parameters
        ----------
        a : 2D numpy array (dtype=complex), shape=(Nt, N_symbols)
            Input data symbol vectors. 
        
        Returns
        -------
        s_prime : 2D numpy array (dtype=complex), shape=(Nt, N_symbols)
            Output powered data symbol vectors.
        """
        
        s_prime = np.diag(np.sqrt(self._Pi)) @ a
        return s_prime

    def precoder(self, s_prime, Vh):
        """ 
        Description
        -----------
        Precode the powered data symbol vectors using the right singular vectors of the channel matrix H.

        Parameters
        ----------
        s_prime : 2D numpy array (dtype=complex, shape=(Nt, N_symbols))
            Input powered data symbol vectors.
        Vh : 2D numpy array (dtype=complex, shape=(Nt, Nt))
            Right singular vectors of the channel matrix H.
        
        Returns
        -------
        s : 2D numpy array (dtype=complex, shape=(Nt, N_symbols))
            Output precoded data symbol vectors.
        """

        s = Vh.conj().T @ s_prime
        return s

    def simulate(self, bitstream, SNR, CSI):
        """
        Description
        -----------
        Simulate the transmitter operations:\n
        (1) Get the channel state information.\n
        (2) [resource_allocation] Execute the waterfilling algorithm to determine the constellation size and power allocation for each transmit antenna.\n Note: If a fixed constellation size M is provided as input, the waterfilling algorithm is skipped and this constant constellation size is used for all antennas.\n
        (3) [bit_allocator] Divide the input bits across the transmit antennas.\n
        (4) [mapper] Map the input bit sequence to the corresponding data symbol sequence for each transmit antenna.\n
        (5) [power_allocator] Allocate power across the transmit antennas.\n
        (6) [precoder] Precode the data symbols using the right singular vectors of the channel matrix H.\n
        (7) Transmit the precoded symbols through the MIMO channel.\n

        The output signal is ready to be transmitted through the MIMO channel.

        Parameters
        ----------
        bitstream : 1D numpy array (dtype=int, length=N_bits)
            The input bitstream.
        SNR : float
            The signal-to-noise ratio in dB.
        CSI : tuple
            The channel state information. It consists of the channel matrix H, left singular vectors U, singular values S, and right singular vectors Vh.

        Returns
        -------
        s : 2D numpy array (dtype=complex, shape=(Nt, N_symbols))
            The output signal.
        """

        # Transmitter Setup.
        self.resource_allocation(SNR, CSI)

        # Edge Case: Transmission fails due to zero useful channel capacity.
        if np.sum( np.log2(self._Mi) ) == 0: return None

        # Transmitter Operations.
        b = self.bit_allocator(bitstream)
        a = self.mapper(b)
        s_prime = self.power_allocator(a)
        s = self.precoder(s_prime, CSI[3])

        return s


    # TESTS AND PLOTS

    def plot_bit_allocation(self, SNR, CSI) -> None:
        """
        Description
        -----------
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
        Pi, Ci, Mi = self.waterfilling(N0, CSI[0], CSI[2])
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

    def plot_power_allocation(self, SNR, CSI) -> None:
        """
        Description
        -----------
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
        Pi = self.waterfilling(N0, CSI[0], S)[0]
        Pi = np.pad(Pi, pad_width=(0, self.Nt - len(Pi)), mode='constant', constant_values=0)

        inverse_channel_gains = (2*self.B*N0) / (S[S > 0] ** 2)
        waterlevels = (Pi[Pi > 0] + inverse_channel_gains[Pi[:len(S[S > 0])] > 0])

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(np.arange(1, len(waterlevels) + 1), waterlevels, color='tab:blue', label='Allocated power')
        ax.bar(np.arange(1, len(inverse_channel_gains) + 1), inverse_channel_gains, color='tab:grey', label='Inverse channel gain')
        ax.axhline(y=np.mean(waterlevels), color='tab:red', linestyle='--', linewidth=3, label='Water level')
        ax.set_xlabel('Transmit antenna')
        ax.set_ylabel('Power')
        ax.set_title(f'Antenna Power Allocation\n\n{self.Nt} transmit antennas and {CSI[0].shape[0]} receive antennas \nSNR: {SNR} dB, Total Transmit Power: {round(self.Pt, 2)} W, Noise Power: {round(2*self.B*N0, 2)} W')
        ax.set_xticks(np.arange(1, len(inverse_channel_gains) + 1))
        ax.set_xlim(0.5, len(inverse_channel_gains) + 0.5)
        ax.legend(loc='upper left')
        fig.tight_layout()
        
        return fig, ax

    def print_simulation_example(self, bitstream, SNR, CSI, K=1) -> None:
        """
        Description
        -----------
        Print a step-by-step example of the transmitter operations (see simulate() method) for given input bits, SNR, and CSI. Only the first K data symbols vectors are considered.

        Parameters
        ----------
        bitstream : 1D numpy array (dtype=int, length=N_bits)
            The input bitstream.
        SNR : float
            The signal-to-noise ratio in dB.
        CSI : tuple
            The channel state information (H, U, S, Vh).
        K : int, optional
            The maximum number of data symbol vectors to consider in the example.
        
        Return
        ------
        s : 2D numpy array (dtype=complex, shape=(Nt, K))
            The output precoded data symbol vectors (first K vectors only).
        
        Notes
        -----
        For demonstration purposes only.
        """

        # PRINTING EXAMPLE
        print("\n\n========== Transmitter Simulation Example ==========\n")
        
        # 0. Print the input bit sequence.
        print(f"----- the input bit sequence -----\n{bitstream}\n\n")
        
        # 1. Get the channel state information.
        N0 = self.Pt / ((10**(SNR/10.0)) * 2*self.B)
        H, U, S, Vh = CSI
        print(f"----- the channel state information -----\n\nH = \n{np.round(H, 2)}\n\nS = {np.round(S, 2)}\n\nU =\n {np.round(U, 2)}\n\nVh =\n {np.round(Vh, 2)}\n\nNoise power spectral density N0 = {round(N0, 4)} W/Hz\n\n\n")

        # 2. Execute the waterfilling algorithm to determine the constellation size and power allocation for each transmit antenna.
        Pi, Ci, Mi = self.waterfilling(N0, H, S)
        self._Pi = np.pad(Pi, pad_width=(0, self.Nt - len(Mi)), mode='constant', constant_values=0)
        self._Mi = np.pad(Mi, pad_width=(0, self.Nt - len(Mi)), mode='constant', constant_values=1)
        print(f"----- waterfilling algorithm results -----\n Power allocation Pi = {np.round(Pi, 2)},\n Channel capacities Ci = {np.round(Ci, 2)},\n Constellation sizes Mi = {Mi}\n\n")
        
        # 3. Divide the input bits accross the transmit antennas.
        b = self.bit_allocator(bitstream)
        b = [b[:K * int(np.log2(self._Mi[i]))] for i, b in enumerate(b)]
        print(f"----- the input bits, allocated across transmit antennas -----\n")
        for i, row in enumerate(b): print(f'Antenna {i+1}: ' + '[' + ' '.join(map(str, row)) + ']')

        # 4. Map the input bit sequence to the corresponding data symbol sequence for each transmit antenna.
        a = self.mapper(b)
        print(f"\n\n----- the data symbols for each transmit antenna -----\n{np.round(a, 2)}\n\n")

        # 5. Allocate power across the transmit antennas.
        s_prime = self.power_allocator(a)
        print(f"----- the data symbols with power allocated for each transmit antenna -----\n{np.round(s_prime, 2)}\n\n")

        # 6. Precode the data symbols.
        s = self.precoder(s_prime, Vh)
        print(f"----- the precoded data symbols ready for transmission -----\n{np.round(s, 2)}\n\n")

        print("======== End Transmitter Simulation Example ========")


        # PLOTS
        fig1, ax1 = self.plot_bit_allocation(SNR, CSI)
        fig2, ax2 = self.plot_power_allocation(SNR, CSI)
        plt.show()


        # RETURN
        return s



if __name__ == "__main__":

    # Initialize the transmitter.
    transmitter = Transmitter(Nt=5, c_type='QAM')
    
    # Initialize the channel.
    H = (1/np.sqrt(2)) * (np.random.randn(transmitter.Nt-1, transmitter.Nt) + 1j * np.random.randn(transmitter.Nt-1, transmitter.Nt))
    U, S, Vh = np.linalg.svd(H)
    CSI = (H, U, S, Vh)

    # Transmitter simulation example.
    transmitter.print_simulation_example(np.random.randint(0, 2, size=100), SNR=25, CSI=CSI, K=3)
