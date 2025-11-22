# This module contains the implementation of the transmitter component of a SU-MIMO SVD communication system.

import numpy as np
import matplotlib.pyplot as plt


class Transmitter:
    """
    Description
    -----------
    The transmitter of a single-user multiple-input multiple-output (SU-MIMO) digital communication system, in which the channel state information is available at the transmitter (and the receiver).

    When the transmitter is called and given an input bit sequence, it ...
        determines the power allocation and constellation size for each transmit antenna, based on the given resource allocation settings (ras)
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
    data_rate : float, optional
        The data rate at which data is transmitted. It is specified is the fraction of the channel capacity. Default is 1.0.
    Pt : float, optional
        Total available transmit power. Default is 1.0.
    B : float, optional
        Bandwidth of the communication system. Default is 0.5.
    c_size : int, optional
        Constellation size. Only used if a fixed constellation size is desired. Default is None.
    
    _Pi : 1D numpy array (dtype: float, length: Nt)
        The power allocation for each transmit antenna for the current CSIT.
    _Ci : 1D numpy array (dtype: float, length: Nt)
        The capacity of each eigenchannel for the current CSIT.
    _Mi : 1D numpy array (dtype: int, length: Nt)
        The constellation size for each transmit antenna for the current CSIT.
    
    Methods
    -------
    __init__()
        Initialize the transmitter parameters.
    __str__()
        Return a string representation of the transmitter object.
    __call__()
        Allow the transmitter object to be called as a function. When called, it executes the simulate() method.
    
    get_CCI()
        Return the power allocation, eigenchannel capacities, and constellation sizes of each transmit antenna, for the current CSIT.
    
    resource_allocation()
        Determine and store power allocation and constellation size for each transmit antenna, based on the given resource allocation settings.
    bit_allocator()
        Allocate the input bitstream across the transmit antennas based on the calculated constellation size for each antenna.
    mapper()
        Convert bit vectors into the corresponding data symbol vectors according to the specified modulation constellation for each transmit antenna.
    power_allocator()
        Allocate power across the transmit antennas based on the calculated power for each antenna.
    precoder()
        Precode the powered data symbol vectors using the right singular vectors of the channel matrix H.
    simulate()
        Simulate the transmitter operations. Return the transmitter output signal.

    plot_bit_allocation()
        Plot the bit allocation across the transmit antennas.
    plot_power_allocation()
        Plot the power allocation across the transmit antennas.
    print_simulation_example()
        Print a step-by-step example of the transmitter operations for given input bits and channel state information (CSIT).
    """


    # INITIALIZATION AND REPRESENTATION

    def __init__(self, Nt, c_type, ras, Pt=1.0, B=0.5):
        """
        Description
        -----------
        Initialize the transmitter.

        Parameters
        ----------
        Nt : int
            Number of transmit antennas.
        c_type : str
            Constellation type. (Choose between 'PAM', 'PSK', or 'QAM')
        ras : dict
            The resource allocation settings. We refer to the function description of resource_allocation() for more details on the meaning of these settings.
            - 'power allocation': 'optimal', 'eigenbeamforming' or 'equal'.
            - 'constellation size': 'adaptive' or 'fixed'. 
        Pt : float, optional
            Total available transmit power. Default is 1.0.
        B : float, optional
            Bandwidth of the communication system. Default is 0.5.
        """

        # Transmitter Settings.
        self.Nt = Nt
        self.c_type = c_type

        self.Pt = Pt
        self.B = B

        self.ras = ras

        # Resource Allocation.
        self._Pi = None
        self._Mi = None

    def __str__(self):
        """ Return a string representation of the transmitter object. """
        return f'Transmitter: \n  - Number of antennas = {self.Nt}\n  - Constellation = ' + (f'{self.M}' if self.M != None else '') + f'{self.c_type}' + f'\n  - Data rate = {self.data_rate*100}% \n  - Total transmit power Pt = {self.Pt} W\n  - Bandwidth B = {self.B} Hz\n'

    def __call__(self, bits, CSIT):
        """ Allow the transmitter object to be called as a function. When called, it executes the simulate() method. """
        return self.simulate(bits, CSIT)


    # FUNCTIONALITY

    def get_CCI(self):
        """
        Description
        -----------
        Return the power allocation and bit allocation (constellation sizes) of each transmit antenna, for the current CSIT. The function represents a control channel between the transmitter and receiver.

        Returns
        -------
        CCI : dict
            The current channel capacity information (CCI).
            - Pi: The power allocation for each used eigenchannel.
            - Mi: The constellation size for each used eigenchannel.
        """
        
        CCI = {'Pi': self._Pi[self._Pi > 0], 'Mi': self._Mi[self._Mi > 1]}
        return CCI


    def resource_allocation(self, ras, CSIT):
        """
        Description
        -----------
        Determine and store the power allocation and bit allocation (constellation size) for each transmit antenna, based on the given resource allocation settings (ras).

        There are three possible options for the power allocation:
            (1) 'optimal': Execute the waterfilling algorithm to determine the optimal power allocation across the transmit antennas. CSIT is required for this mode.
            (2) 'eigenbeamforming': Allocate all power to the best eigenchannel. The waterfilling algorithm is omitted. CSIT is required for this mode.
            (3) 'equal': Equally divide the available transmit power across all transmit antennas. The waterfilling algorithm is omitted and CSIT is not required for this mode.
        
        There are two possible options for the constellation size:
            (1) 'adaptive': Determine the constellation size based on the eigenchannel capacities. CSIT is required for this mode. An extra key 'data rate' must be provided in the dictionary to specify the fraction of the channel capacity that is utilized.
            (2) 'fixed': Use a constant constellation size for all transmit antennas. CSIT is not required for this mode. An extra key 'fixed sizes' must be provided in the dictionary to specify the constellation size on each transmit antenna (in case of equal constellation sizes across all transmit antennas, the value might be an integer instead of an array).\n

        Parameters
        ----------
        ras : dict
            The resource allocation settings.
            - 'power allocation': 'optimal', 'eigenbeamforming' or 'equal'.
            - 'constellation size': 'adaptive' or 'fixed'.
        CSIT : dict
            The channel state information (SNR, H, U, S, Vh).
        
        Returns
        -------
        Pi : 1D numpy array (dtype: float, length: Nt)
            The power allocation for each transmit antenna.
        Mi : 1D numpy array (dtype: int, length: Nt)
            The constellation size for each transmit antenna.
        """

        def waterfilling(CSIT):
            """
            Description
            -----------
            Execute the waterfilling algorithm to determine the optimal power allocation for each transmit antenna, given the channel state information (CSI).\n

            Parameters
            ----------
            CSIT : dict
                The channel state information (SNR, H, U, S, Vh).

            Returns
            -------
            Pi : 1D numpy array (dtype: float, length: Nt)
                The optimal power allocation for each transmit antenna.

            Notes
            -----
            In real-world scenarios, every antenna itself has a power constraint as well. However, in this implementation, we only consider a total power constraint across all antennas.
            """

            # Parameters.
            N0 = self.Pt / ((10**(CSIT['SNR']/10.0)) * 2*self.B)
            S = CSIT['S']
            rank_H = np.linalg.matrix_rank(CSIT['H'])

            # Edge Case: The PSD of the noise is zero. The optimal strategy is to equally divide the power across all eigenchannels.
            if N0 == 0: return np.array([self.Pt / rank_H] * rank_H + [0] * (self.Nt - rank_H))

            # Initialization.
            gamma = self.Pt / (2*self.B*N0)
            used_eigenchannels = rank_H
            waterlevel = (gamma / used_eigenchannels) + (1 / used_eigenchannels) * np.sum(1 / (S[:used_eigenchannels]**2))

            # Iteration.
            while ( waterlevel < (1 / (S[used_eigenchannels-1]**2)) ):
                used_eigenchannels -= 1
                waterlevel = (gamma / used_eigenchannels) + (1 / used_eigenchannels) * np.sum(1 / (S[:used_eigenchannels]**2))

            # Termination.
            Pi = np.maximum((waterlevel - (1 / (S[:used_eigenchannels]**2))) * (2*self.B*N0), 0)
            Pi = np.pad(Pi, (0, self.Nt - used_eigenchannels), 'constant')
            return Pi

        def eigenchannel_capacities(Pi, CSIT):
            """
            Description
            -----------
            Calculate the capacity of each eigenchannel, given the power allocation and channel state information (CSI).\n

            Parameters
            ----------
            Pi : 1D numpy array (dtype: float, length: Nt)
                The power allocation for each transmit antenna.
            CSIT : dict
                The channel state information (SNR, H, U, S, Vh).

            Returns
            -------
            Ci : 1D numpy array (dtype: float, length: Nt)
                The capacity of each eigenchannel (length: min(Nt, Nr)), padded with zeros to length Nt.
            """

            # Parameters.
            num_eigenchannels = min(self.Nt, CSIT['H'].shape[0])
            rank_H = np.linalg.matrix_rank(CSIT['H'])
            
            N0 = self.Pt / ((10**(CSIT['SNR']/10.0)) * 2*self.B)
            S = np.pad(CSIT['S'], (0, self.Nt - len(CSIT['S'])), 'constant')

            # Edge Case: The PSD of the noise is zero. The capacity of each eigenchannel becomes infinite.
            if N0 == 0: return np.array([np.inf] * rank_H + [0] * (self.Nt - rank_H))

            # Calculate the capacity for each eigenchannel.
            Ci = 2*self.B * np.log2( 1 + (Pi * (S**2)) / (2*self.B*N0) )
            return Ci


        # Power Allocation.

        if ras.get('power allocation') == 'optimal':
            Pi = waterfilling(CSIT)
        
        elif ras.get('power allocation') == 'eigenbeamforming':
            Pi = np.array([self.Pt] + [0]*(self.Nt - 1))
        
        elif ras.get('power allocation') == 'equal':
            Pi = np.array([self.Pt / self.Nt] * self.Nt)
        
        else: raise ValueError(f'The power allocation method is invalid.\nChoose between "optimal", "eigenbeamforming", or "equal".')
        

        # Bit Allocation.

        if ras.get('constellation size') == 'adaptive':
            Ci = eigenchannel_capacities(Pi, CSIT)
            data_rate = ras.get('data rate', 1.0)
            Mi = 2 ** np.floor( Ci * data_rate ).astype(int) if self.c_type != 'QAM' else 4 ** np.floor( (Ci * data_rate) / 2 ).astype(int)
        
        elif ras.get('constellation size') == 'fixed':
            Mi = np.full(self.Nt, ras.get('fixed sizes')) if isinstance(ras.get('fixed sizes'), int) else np.array(ras.get('fixed sizes'))

        else: raise ValueError(f'The constellation size method is invalid.\nChoose between "adaptive" or "fixed".')


        # Store the results. 
           
        self._Pi = Pi
        self._Mi = Mi
        return Pi, Mi

    def bit_allocator(self, bitstream):
        """
        Description
        -----------
        Allocate the input bitstream across the transmit antennas based on the calculated constellation size for each antenna. 
        
        Every antenna will have to send an equal amount of data symbols. If the number of bits in the bitstream does not perfectly align with an an equal amount of symbols per antenna, it is padded with zeros.
        If zero power is allocated to an eigenchannel (or thus the used capacity of that eigenchannel equals zero), no bits will be allocated to that eigenchannel.

        Parameters
        ----------
        bitstream : 1D numpy array (dtype: int, length: N_bits)
            Input - bitstream.

        Returns
        -------
        b : list of 1D numpy arrays (dtype: int, length: N_symbols * log2(Mi[tx_antenna]))
            Output - bit vectors.
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
        b : list of 1D numpy arrays (dtype: int, length: N_symbols * log2(Mi[tx_antenna]))
            Input - bit vectors.

        Returns
        -------
        a : 2D numpy array (dtype: complex, shape: (Nt, N_symbols))
            Output - data symbol vectors.
        """

        def map(bits, M, c_type):
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
                If the constellation size or type is invalid.
            ValueError
                If the length of the bit sequence is invalid. It must be a multiple of log2(M).
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

        a = np.array([ map(b[tx_antenna], self._Mi[tx_antenna], self.c_type) for tx_antenna in range(len(self._Mi[self._Mi > 1])) ])
        a = np.concatenate( (a, np.zeros((self.Nt - a.shape[0], a.shape[1]), dtype=complex)), axis=0 )
        return a

    def power_allocator(self, a):
        """
        Description
        -----------
        Allocate power across the transmit antennas based on the calculated power for each antenna.

        Parameters
        ----------
        a : 2D numpy array (dtype: complex, shape: (Nt, N_symbols))
            Input - data symbol vectors. 
        
        Returns
        -------
        s_tilda : 2D numpy array (dtype: complex, shape: (Nt, N_symbols))
            Output - powered data symbol vectors.
        """
        
        s_tilda = np.diag(np.sqrt(self._Pi)) @ a
        return s_tilda

    def precoder(self, s_tilda, Vh):
        """ 
        Description
        -----------
        Precode the powered data symbol vectors using the right singular vectors of the channel matrix H.

        Parameters
        ----------
        s_tilda : 2D numpy array (dtype: complex, shape: (Nt, N_symbols))
            Input - powered data symbol vectors.
        Vh : 2D numpy array (dtype: complex, shape: (Nt, Nt))
            Right singular vectors of the channel matrix H.
        
        Returns
        -------
        s : 2D numpy array (dtype: complex, shape: (Nt, N_symbols))
            Output - precoded data symbol vectors.
        """

        s = Vh.conj().T @ s_tilda
        return s

    def simulate(self, bitstream, CSIT):
        """
        Description
        -----------
        Simulate the transmitter operations:\n
        (1) Get the channel state information.\n
        (2) [resource_allocation] Determine and store the power allocation and constellation size for each transmit antenna, based on the given resource allocation settings (ras).\n
        (3) [bit_allocator] Divide the input bits across the transmit antennas.\n
        (4) [mapper] Map the input bit sequence to the corresponding data symbol sequence for each transmit antenna.\n
        (5) [power_allocator] Allocate power across the transmit antennas.\n
        (6) [precoder] Precode the data symbols using the right singular vectors of the channel matrix H.\n
        (7) Transmit the precoded symbols through the MIMO channel.\n

        The output signal is ready to be transmitted through the MIMO channel.

        Parameters
        ----------
        bitstream : 1D numpy array (dtype=int, length=N_bits)
            Input - bitstream.
        CSIT : dict
            The channel state information at the transmitter.
            - SNR: The signal-to-noise ratio in dB. (float)
            - H : The channel matrix. (2D numpy array, dtype: complex, shape: (Nr, Nt)).
            - U : The left singular vectors of H. (2D numpy array, dtype: complex, shape: (Nr, Nr)).
            - S : The singular values of H. (1D numpy array, dtype: float, length: Rank(H)).
            - Vh : The right singular vectors of H. (2D numpy array, dtype: complex, shape: (Nt, Nt)). 

        Returns
        -------
        s : 2D numpy array (dtype: complex, shape: (Nt, N_symbols))
            Output - transmitted signal.
        """

        # Transmitter Setup.
        self.resource_allocation(self.ras, CSIT)

        # Edge Case: Transmission fails due to zero useful channel capacity.
        if np.sum( np.log2(self._Mi) ) == 0: return None

        # Transmitter Operations.
        b = self.bit_allocator(bitstream)
        a = self.mapper(b)
        s_tilda = self.power_allocator(a)
        s = self.precoder(s_tilda, CSIT['Vh'])

        return s


    # TESTS AND PLOTS

    def plot_bit_allocation(self, ras, CSIT) -> None:
        """
        Description
        -----------
        Plot the bit allocation across the transmit antennas as determined by the resource allocation settings.
        On the x-axis, the transmit antennas are shown. On the y-axis, the number of bits allocated to each transmit antenna is shown, as well as the capacity of the eigenchannel corresponding to each transmit antenna.

        Parameters
        ----------
        ras : dict
            The resource allocation settings (power allocation, constellation size). We refer to the function description of resource_allocation() for more details on the meaning of these settings.
        CSIT : dict
            The channel state information (SNR, H, U, S, Vh).
        
        Returns
        -------
        fig, ax : tuple
            The figure and axis objects of the plot.
        """
        
        def generate_title(ras, SNR):
            title = f'{str(self)}\nAntenna Bit Allocation (SNR = {round(SNR)} dB)\n\n'
            settings = f'power allocation: {ras.get("power allocation")}, bit allocation: {ras.get("constellation size")}'
            detail_settings = (f'R = {ras.get("data rate")*100}%' if ras.get("constellation size") == 'adaptive' else f'M = {str(ras.get("fixed sizes"))}')
            title = title + settings + detail_settings
            return title

        def generate_file_name(ras, SNR):
            location = 'su-mimo/plots/resource_allocation/bit_allocation/'
            system_name = str(self).replace(' - ', '__').replace('-', '__').replace(' ', '_')
            settings = '__SNR_' + str(SNR) + '__pa_' + str(ras.get('power allocation')) + '__ba_' + str(ras.get('constellation size'))
            detail_settings = (f'__R_{round(ras.get("data rate")*100)}%' if ras.get("constellation size") == 'adaptive' else f'__M_{str(ras.get("fixed sizes"))}')
            extension = '.png'
            return location + system_name + settings + detail_settings + extension

        # Determine the bit allocation.
        Pi, Mi = self.resource_allocation(self.ras, CSIT)
        
        mc = np.log2(Mi).astype(int)
        Ci = 2*self.B * np.log2( 1 + (10**(CSIT['SNR']/10.0)) * ((Pi[:min(self.Nt, CSIT['H'].shape[0])] * (CSIT['S']**2)) / self.Pt) )

        # Plot.
        fig, ax = plt.subplots(figsize=(8, 4))
        
        x = np.arange(min(self.Nt, CSIT['H'].shape[0]))
        ax.bar(x - 0.35/2, Ci, width=0.35, color='tab:green', label='Eigenchannel capacity')
        ax.bar(x + 0.35/2, mc, width=0.35, color='tab:blue', label='Constellation size')
        
        ax.set_xlabel('Transmit Antenna')
        ax.set_ylabel('Bits')
        ax.set_title(generate_title(self.ras, CSIT['SNR']))
        ax.set_xticks(x)
        ax.set_xticklabels(x + 1)
        ax.set_xlim(-0.5, len(Ci) - 0.5)
        ax.legend(loc='upper right')
        fig.tight_layout()
        fig.savefig(generate_file_name(self.ras, CSIT['SNR']), dpi=300, bbox_inches='tight')

        return fig, ax

    def plot_power_allocation(self,CSIT):
        """
        Description
        -----------
        Plot the power allocation across the transmit antennas, as determined by the resource allocation settings.
        On the x-axis, the transmit antennas are shown. On the y-axis, the inverse channel gain is shown in grey and the amount of power allocated to each transmit antenna is shown in blue.

        Parameters
        ----------
        CSIT : dict
            The channel state information (SNR, H, U, S, Vh).
        
        Returns
        -------
        fig, ax : tuple
            The figure and axis objects of the plot.
        """

        def generate_title(CSIT):
            title = f'{str(self)}\nAntenna Power Allocation\n\n'
            settings = f'power allocation: {self.ras.get("power allocation")}, SNR: {round(CSIT["SNR"])} dB'
            return title + settings

        def generate_file_name(CSIT):
            location = 'su-mimo/plots/resource_allocation/power_allocation/'
            system_name = str(self).replace(' - ', '__').replace('-', '__').replace(' ', '_')
            settings = '__SNR_' + str(round(CSIT['SNR'])) + '__pa_' + str(self.ras.get('power allocation'))
            extension = '.png'
            return location + system_name + settings + extension

        # Determine the power allocation.
        Pi, _ = self.resource_allocation(self.ras, CSIT)

        N0 = self.Pt / ((10**(CSIT['SNR']/10.0)) * 2*self.B)
        S = CSIT['S']
        inverse_channel_gains = (2*self.B*N0) / (S[S > 0] ** 2)
        waterlevels = Pi[Pi > 0] + inverse_channel_gains[Pi > 0]

        # Plot.
        fig, ax = plt.subplots(figsize=(8, 4))
        
        ax.bar(np.arange(1, len(waterlevels) + 1), waterlevels, color='tab:blue', label='Allocated power')
        ax.bar(np.arange(1, len(inverse_channel_gains) + 1), inverse_channel_gains, color='tab:grey', label='Inverse channel gain')
        ax.axhline(y=np.mean(waterlevels), color='tab:red', linestyle='--', linewidth=3, label='Water level')
        
        ax.set_xlabel('Transmit antenna')
        ax.set_ylabel('Power [W]')
        ax.set_title(generate_title(CSIT))
        ax.set_xticks(np.arange(1, len(inverse_channel_gains) + 1))
        ax.set_xlim(0.5, len(inverse_channel_gains) + 0.5)
        ax.legend(loc='upper left')
        fig.tight_layout()
        fig.savefig(generate_file_name(CSIT['SNR']), dpi=300, bbox_inches='tight')
        
        return fig, ax

    def print_simulation_example(self, bitstream, CSIT, K=1) -> None:
        """
        Description
        -----------
        Print a step-by-step example of the transmitter operations (see simulate() method) for given input bits and CSIT. Only the first K data symbols vectors are considered.

        Parameters
        ----------
        bitstream : 1D numpy array (dtype: int, length: N_bits)
            The input bitstream.
        CSIT : dict
            The channel state information (SNR, H, U, S, Vh).
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
        N0 = self.Pt / ((10**(CSIT['SNR']/10.0)) * 2*self.B)
        print(f"----- the channel state information -----\n\nH = \n{np.round(CSIT['H'], 2)}\n\nS = {np.round(CSIT['S'], 2)}\n\nU =\n {np.round(CSIT['U'], 2)}\n\nVh =\n {np.round(CSIT['Vh'], 2)}\n\nNoise power spectral density N0 = {round(N0, 4)} W/Hz\n\n\n")

        # 2. Execute the waterfilling algorithm to determine the constellation size and power allocation for each transmit antenna.
        Pi, Ci, Mi = self.waterfilling(N0, CSIT['H'], CSIT['S'])
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
        s_tilda = self.power_allocator(a)
        print(f"----- the data symbols with power allocated for each transmit antenna -----\n{np.round(s_tilda, 2)}\n\n")

        # 6. Precode the data symbols.
        s = self.precoder(s_tilda, CSIT['Vh'])
        print(f"----- the precoded data symbols ready for transmission -----\n{np.round(s, 2)}\n\n")

        print("======== End Transmitter Simulation Example ========")


        # PLOTS
        fig1, ax1 = self.plot_bit_allocation(CSIT)
        fig2, ax2 = self.plot_power_allocation(CSIT)
        plt.show()


        # RETURN
        return s
