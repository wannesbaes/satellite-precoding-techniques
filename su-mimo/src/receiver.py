# This module contains the implementation of the receiver component of a SU-MIMO SVD communication system.

import numpy as np


class Receiver:
    """
    Description
    -----------
    The receiver of a single-user multiple-input multiple-output (SU-MIMO) digital communication system, in which the channel state information is available at the receiver (and transmitter).

    When the receiver is called and given an received input signal y, it ...
        determines the power allocation and constellation size on each receive antenna
        combines and equalizes the received symbol vectors
        searches the most probable transmitted data symbol vectors
        demaps them into bit vectors
        and combines the reconstructed bits on each antenna to create the output bitstream.

    Attributes
    ----------
    Nr : int
        Number of receiving antennas.
    c_type : str
        Constellation type. (Choose between 'PAM', 'PSK', or 'QAM'.)
    Pt : float
        Total available transmit power. Default is 1.0.
    B : float
        Bandwidth of the communication system. Default is 0.5.
    RAS : dict
        The resource allocation strategy. We refer to the function description of resource_allocation() for more details on the meaning of these strategy settings.
    
    _Pi : 1D numpy array (dtype: float, length: Nr)
        The power allocation for each receive antenna for the current CSI.
    _Ci : 1D numpy array (dtype: float, length: Nr)
        The capacity of each eigenchannel for the current CSI.
    _Mi : 1D numpy array (dtype: int, length: Nr)
        The constellation size for each receive antenna for the current CSI.
    
    Methods
    -------
    __init__():
        Initialize the receiver.
    __str__():
        Return a string representation of the receiver object.
    __call__():
        Allow the receiver object to be called as a function. When called, it executes the simulate() method.
    
    resource_allocation():
        Determine and store the power allocation and constellation size for each receive antenna, based on the given control channel information (CCI) or resource allocation strategy (RAS).
    combiner():
        Combine the input signal using the left singular vectors of the channel matrix H.
    equalizer():
        Equalize the combined symbol vectors, based on S and Pi.
    detector():
        Convert the decision variable vectors into the most probable transmitted data symbol vectors.
    demapper():
        Convert the detected data symbol vectors into the corresponding bit vectors according to the specified modulation constellation.
    bit_deallocator():
        Combine the reconstructed bits on each antenna to create the output bitstream.
    simulate():
        Simulate the receiver operations. Return the reconstructed bitstream.
    """

    
    # INITIALIZATION AND REPRESENTATION

    def __init__(self, Nr, c_type, Pt=1.0, B=0.5, RAS={}):
        """
        Description
        -----------
        Initialize the receiver.

        Parameters
        ----------
        Nr : int
            Number of receive antennas.
        c_type : str
            Constellation type. (Choose between 'PAM', 'PSK', or 'QAM')
        Pt : float, optional
            Total available transmit power. Default is 1.0.
        B : float, optional
            Bandwidth of the communication system. Default is 0.5.
        RAS : dict
            The resource allocation strategy. We refer to the function description of resource_allocation() for more details on the meaning of these settings.
            - 'control channel': True or False.
            - 'power allocation': 'optimal', 'eigenbeamforming' or 'equal'.
            - 'bit allocation': 'adaptive' or 'fixed'. 
        """

        # Transmitter Settings.
        self.Nr = Nr
        self.c_type = c_type

        self.Pt = Pt
        self.B = B

        self._RAS = RAS

        # Resource Allocation.
        self._Pi = None
        self._Mi = None

    def __str__(self):
        """ Return a string representation of the receiver object. """
        return f'Receiver:\n  - Number of antennas: {self.Nr}\n  - Total transmit power (Pt): {self.Pt} W\n  - Bandwidth (B): {self.B} Hz\n  - Power allocation: {self._RAS["power allocation"]}\n  - Bit Allocation: {self._RAS["bit allocation"]}\n' + (f'  - Data Rate: {round(self._RAS["data rate"]*100)}%\n' if "data rate" in self._RAS else '') + f'  - Constellation type: {self.c_type}\n\n'

    def __call__(self, y, CSI):
        """ Allow the receiver object to be called as a function. When called, it executes the simulate() method. """
        return self.simulate(y, CSI)


    # FUNCTIONALITY

    def set_RAS(self, RAS):
        """
        Description
        -----------
        Update the resource allocation strategy (RAS) for the receiver.

        Parameters
        ----------
        RAS : dict
            The resource allocation strategy. We refer to the function description of resource_allocation() for more details on the meaning of these settings.
        """

        self._RAS |= RAS
        return

    
    def resource_allocation(self, CSIR, CCI=None):
        """
        Description
        -----------
        Determine and store the power allocation and bit allocation (constellation size) for each receive antenna, based on the given resource allocation strategy (RAS).

        If a control channel is available, the resource allocation is acquired from the the control channel information (CCI).
        Otherwise, the resource allocation is calculated completely analogously as in the transmitter.

        There are three possible options for the power allocation:
            (1) 'optimal': Execute the waterfilling algorithm to determine the optimal power allocation across the receive antennas. CSIR is required for this mode. (Default)
            (2) 'eigenbeamforming': Allocate all power to the best eigenchannel. The waterfilling algorithm is omitted and CSIR is not required for this mode.
            (3) 'equal': Equally divide the available transmit power across all receive antennas. The waterfilling algorithm is omitted and CSIR is not required for this mode.
        
        There are two possible options for the bit allocation:
            (1) 'adaptive': Determine the constellation size based on the eigenchannel capacities. CSIR is required for this mode. An extra key 'data rate' must be provided in the dictionary to specify the fraction of the channel capacity that is utilized. (Default)
            (2) 'fixed': Use a constant constellation size for all receive antennas. CSIR is not required for this mode. An extra key 'constellation sizes' must be provided in the dictionary to specify the constellation size on each receive antenna (in case of equal constellation sizes across all receive antennas, the value might be an integer instead of an array).\n

        Parameters
        ----------
        CSIR : dict
            The channel state information at the receiver (SNR, H, U, S, Vh).
        CCI : dict, optional
            The control channel information (Pi, Mi). Only required if a control channel is available.
        
        Returns
        -------
        Pi : 1D numpy array (dtype: float, length: Nr)
            The power allocation for each receive antenna.
        Mi : 1D numpy array (dtype: int, length: Nr)
            The constellation size for each receive antenna.
        """

        def waterfilling(CSIR):
            """
            Description
            -----------
            Execute the waterfilling algorithm to determine the optimal power allocation for each receive antenna, given the channel state information (CSI).\n
            
            Parameters
            ----------
            CSIR : dict
                The channel state information (SNR, H, U, S, Vh).
            
            Returns
            -------
            Pi : 1D numpy array (dtype: float, length: Nr)
                The optimal power allocation for each receive antenna.
            
            Notes
            -----
            In real-world scenarios, every antenna itself has a power constraint as well. However, in this implementation, we only consider a total power constraint across all antennas.
            """

            # Parameters.
            N0 = self.Pt / ((10**(CSIR['SNR']/10.0)) * 2*self.B)
            S = CSIR['S']
            rank_H = np.linalg.matrix_rank(CSIR['H'])

            # Edge Case: The PSD of the noise is zero. The optimal strategy is to equally divide the power across all eigenchannels.
            if N0 == 0: return np.array([self.Pt / rank_H] * rank_H + [0] * (self.Nr - rank_H))

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
            Pi = np.pad(Pi, (0, self.Nr - used_eigenchannels), 'constant')
            return Pi

        def eigenchannel_capacities(Pi, CSIR):
            """
            Description
            -----------
            Calculate the capacity of each eigenchannel, given the power allocation and channel state information (CSI).\n

            Parameters
            ----------
            Pi : 1D numpy array (dtype: float, length: Nr)
                The power allocation for each receive antenna.
            CSIR : dict
                The channel state information (SNR, H, U, S, Vh).

            Returns
            -------
            Ci : 1D numpy array (dtype: float, length: Nr)
                The capacity of each eigenchannel (length: min(Nt, Nr)), padded with zeros to length Nr.
            """

            # Parameters.
            N0 = self.Pt / ((10**(CSIR['SNR']/10.0)) * 2*self.B)
            S = np.pad(CSIR['S'], (0, self.Nr - len(CSIR['S'])), 'constant')

            # Edge Case: The PSD of the noise is zero. The capacity of each eigenchannel becomes infinite.
            if N0 == 0:
                rank_H = np.linalg.matrix_rank(CSIR['H'])
                Ci = np.array([np.inf] * rank_H + [0] * (self.Nr - rank_H))
                return Ci

            # Calculate the capacity for each eigenchannel.
            Ci = 2*self.B * np.log2( 1 + (Pi * (S**2)) / (2*self.B*N0) )
            return Ci


        # CASE 1: Control Channel Available.
        
        if self._RAS.get('control channel'):
            
            Pi = np.pad(CCI['Pi'], pad_width=(0, self.Nr - len(CCI['Pi'])), mode='constant', constant_values=0)
            Mi = np.pad(CCI['Mi'], pad_width=(0, self.Nr - len(CCI['Mi'])), mode='constant', constant_values=1)
            
            self._Pi = Pi
            self._Mi = Mi
            return Pi, Mi


        # CASE 2: No Control Channel Available.

        # Power Allocation.

        if self._RAS.get('power allocation') == 'optimal':
            Pi = waterfilling(CSIR)
        
        elif self._RAS.get('power allocation') == 'eigenbeamforming':
            Pi = np.array([self.Pt] + [0]*(self.Nr - 1))
        
        elif self._RAS.get('power allocation') == 'equal':
            Pi = np.array([self.Pt / self.Nr] * self.Nr)
        
        else: raise ValueError(f'The power allocation method is invalid.\nChoose between "optimal", "eigenbeamforming", or "equal".')
        

        # Bit Allocation.

        if self._RAS.get('bit allocation') == 'adaptive':
            Ci = eigenchannel_capacities(Pi, CSIR)
            data_rate = self._RAS.get('data rate', 1.0)
            Mi = 2 ** np.floor( Ci * data_rate ).astype(int) if self.c_type != 'QAM' else 4 ** np.floor( (Ci * data_rate) / 2 ).astype(int)
        
        elif self._RAS.get('bit allocation') == 'fixed':
            Mi = np.full(self.Nr, self._RAS.get('constellation sizes')) if isinstance(self._RAS.get('constellation sizes'), int) else np.array(self._RAS.get('constellation sizes'))
            Mi[Pi == 0] = 1

        else: raise ValueError(f'The bit allocation method is invalid.\nChoose between "adaptive" or "fixed".')

        # Store the results. 
           
        self._Pi = Pi
        self._Mi = Mi
        return Pi, Mi

    def combiner(self, y, U):
        """
        Description
        -----------
        Combine the input signal (distorted data symbol vectors) using the left singular vectors of the channel matrix H.

        Parameters
        ----------
        y : 2D numpy array (dtype: complex, shape: (Nr, N_symbols))
            Input - received signal.
        U : 2D numpy array (dtype: complex, shape: (Nr, Nr))
            The left singular vectors of the channel matrix H.
        
        Returns
        -------
        y_tilda : 2D numpy array (dtype: complex, shape: (Nr, N_symbols))
            Output - combined symbol vectors.
        """

        y_tilda = U.conj().T @ y
        return y_tilda

    def equalizer(self, y_tilda, S):
        """
        Description
        -----------
        Equalize the combined symbol vectors using the singular values of the channel matrix H and the allocated power on each antenna.

        Parameters
        ----------
        y_tilda : 2D numpy array (dtype: complex, shape: (Nr, N_symbols))
            Input - combined symbol vectors.
        S : 1D numpy array (dtype: float, length=rank_H)
            The singular values of the channel matrix H.

        Returns
        -------
        u : 2D numpy array (dtype: complex, shape: (Nr, N_symbols))
            Output - decision variable vectors.
        """

        useful_eigenchannels = min(len(self._Pi[self._Pi>0]), len(S))
        u = y_tilda[:useful_eigenchannels] / (S * np.sqrt(self._Pi[:len(S)]))[:useful_eigenchannels][:, np.newaxis]

        return u

    def detector(self, u):
        """
        Description
        -----------
        Convert the decision variable vectors (distorted (equalized & combined) data symbol vectors) into the most probable (minimum distance (MD) detection) transmitted data symbol vectors according to the specified modulation constellation for each transmit antenna.

        Parameters
        ----------
        u : 2D numpy array (dtype: complex, shape: (Nr, N_symbols))
            Input - decision variable vectors.
        
        Returns
        -------
        a_hat : 2D numpy array (dtype: complex, shape: (Nr, N_symbols))
            Output - detected data symbol vectors.
        """
        
        def detect(decision_variables, M, c_type):
            """
            Description
            -----------
            Convert the decision variables into the most probable (minimum distance detection) transmitted data symbols according to the specified modulation constellation.

            Parameters
            ----------
            decision_variables : 1D numpy array (dtype: complex, length: num_symbols)
                Input - decision variables.
            M : int
                Constellation size.
            c_type : str
                Constellation type. (Choose between 'PAM', 'PSK', 'QAM'.)
            
            Returns
            -------
            datasymbols_hat : 1D numpy array (dtype: complex, length: num_symbols)
                Output - detected data symbols.
            
            Raises
            ------
            ValueError
                If the constellation size or type is invalid.
            """

            # 1. Construct the modulation constellation.

            assert (M & (M - 1) == 0) and ((M & 0xAAAAAAAA) == 0 or c_type != 'QAM'), f'The constellation size M is invalid.\nFor PAM and PSK modulation, it must be a power of 2. For QAM Modulation, M must be a power of 4. Right now, M equals {M} and the type is {c_type}.'

            if c_type == 'PAM':
                constellation = np.arange(-(M-1), (M-1) + 1, 2) * np.sqrt(3/(M**2-1))

            elif c_type == 'PSK':
                constellation = np.exp(1j * 2*np.pi * np.arange(M) / M)

            elif c_type == 'QAM':
                c_sqrtM_PAM = np.arange(-(np.sqrt(M)-1), (np.sqrt(M)-1) + 1, 2) * np.sqrt(3 / (2*(M-1)))
                real_grid, imaginary_grid = np.meshgrid(c_sqrtM_PAM, c_sqrtM_PAM)
                constellation = (real_grid + 1j*imaginary_grid).flatten()

            else :
                raise ValueError(f'The constellation type is invalid.\nChoose between "PAM", "PSK", or "QAM". Right now, type is {c_type}.')
            

            # 2. Map each decision variable to the nearest data symbol in the constellation.

            distances = np.abs(constellation[:, np.newaxis] - decision_variables)
            datasymbols_hat = constellation[np.argmin(distances, axis=0)]

            # 3. Return
            return datasymbols_hat

        a_hat = np.array([ detect(u[rx_antenna, :], self._Mi[rx_antenna], self.c_type) for rx_antenna in range(len(self._Mi[self._Mi > 1])) ])
        a_hat = np.concatenate( (a_hat, np.zeros((self.Nr - a_hat.shape[0], a_hat.shape[1]), dtype=complex)), axis=0 )
        return a_hat

    def demapper(self, a_hat):
        """
        Description
        -----------
        Convert the detected data symbol vectors into the corresponding bit vectors according to the specified modulation constellation. It performs the inverse operation of the mapper in the transmitter.

        Parameters
        ----------
        a_hat : 2D numpy array (dtype: complex, shape: (Nr, N_symbols))
            Input - detected data symbol vectors.
              
        Returns
        -------
        b_hat : list of 1D numpy arrays (dtype: int, length: N_symbols * log2(Mi[rx_antenna]))
            Output - reconstructed bit vectors.
        """

        def demap(symbols_hat, M, c_type):
            """
            Description
            -----------
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
        
        b_hat = [ demap(a_hat[rx_antenna, :], self._Mi[rx_antenna], self.c_type) for rx_antenna in range(len(self._Mi[self._Mi > 1])) ] + [np.array([], dtype=int)]*(len(self._Mi[self._Mi <= 1]))
        return b_hat
    
    def bit_deallocator(self, b_hat):
        """
        Description
        -----------
        Combine the reconstructed bit vectors to create the output bitstream. It performs the inverse operation of the bit_allocator in the transmitter.

        Parameters
        ----------
        bits_hat : list of 1D numpy arrays (dtype: int, length: N_symbols * log2(Mi[rx_antenna]))
            Input - reconstructed bit vectors.
        
        Return
        ------
        bitstream_hat: 1D numpy array (dtype: int, length: Nr * num_symbols * log2(Mi[rx_antenna]))
            Output - reconstructed bitstream.
        """

        bitstream_hat = np.array([], dtype=int)
        m = np.log2(self._Mi).astype(int)
        num_symbols = len(b_hat[0]) // m[0]

        for symbol_i in range(num_symbols):
            for rx_antenna in range(len(self._Mi[self._Mi > 1])):
                bitstream_hat = np.concatenate((bitstream_hat, b_hat[rx_antenna][symbol_i*m[rx_antenna]: (symbol_i+1)*m[rx_antenna]]))

        return bitstream_hat

    def simulate(self, y, CSI, CCI=None):
        """
        Description
        -----------
        Simulate the receiver operations:\n
        (1) Get the channel state information.\n
        (2) [resource_allocation] Determine and store power allocation and constellation size for each receive antenna, based on the given control channel information or resource allocation strategy.\n
        (3) [combiner] Combine the received symbol vectors using the left singular vectors of the channel matrix H.\n
        (4) [equalizer] Equalize the combined symbol vectors using the singular values of the channel matrix H and the allocated power on each antenna.\n
        (5) [detector] Convert the decision variable vectors into the most probable data symbol vectors.\n
        (6) [demapper] Convert the reconstructed data symbol vectors into the corresponding bit vectors according to the specified modulation constellation.\n
        (7) [bit deallocator] Combine the reconstructed bit vectors to create the output bitstream.\n
        (8) Return the output bitstream.\n

        Parameters
        ----------
        y : 2D numpy array (dtype: complex, shape: (Nr, N_symbols))
            Input - received signal.
        CSI : dict
            The channel state information.
            - SNR: The signal-to-noise ratio in dB. (float)
            - H : The channel matrix. (2D numpy array, dtype: complex, shape: (Nr, Nt)).
            - U : The left singular vectors of H. (2D numpy array, dtype: complex, shape: (Nr, Nr)).
            - S : The singular values of H. (1D numpy array, dtype: float, length: Rank(H)).
            - Vh : The right singular vectors of H. (2D numpy array, dtype: complex, shape: (Nt, Nt)).
        CCI : dict, optional
            The control channel information.
            - Pi : The power allocation for each receive antenna. (1D numpy array, dtype: float)
            - Mi : The constellation size for each receive antenna. (1D numpy array, dtype: int)
        
        Returns
        -------
        bitstream_hat : 1D numpy array (dtype: int, length: Nr * N_symbols * log2(Mi[rx_antenna]))
            Output - reconstructed bitstream.
        """

        # Receiver Setup.
        self.resource_allocation(CSI, CCI)
        
        # Receiver Operations.
        y_tilda = self.combiner(y, CSI['U'])
        u = self.equalizer(y_tilda, CSI['S'])
        a_hat = self.detector(u)
        b_hat = self.demapper(a_hat)
        bitstream_hat = self.bit_deallocator(b_hat)

        return bitstream_hat


    # TESTS AND PLOTS

    def print_simulation_example(self, y, CSI, CCI, K=1):
        """
        Description
        -----------
        Print a step-by-step example of the receiver operations (see simulate() method) for given input signal y, and channel state information (CSI). Only the first K data symbols vectors are considered.

        Parameters
        ----------
        y : 2D numpy array (dtype: complex, shape: (Nr, N_symbols))
            Input - received signal.
        CSI : dict
            The channel state information (SNR, H, U, S, Vh).
        CCI : dict
            The control channel information (Pi, Ci, Mi).
        K : int, optional
            Number of data symbol vectors to consider in the illustration. Default is 1.
        
        Notes
        -----
        For demonstration purposes only.
        """

        # PRINTING EXAMPLE

        print("\n\n========== Receiver Simulation Example ==========\n")
        print(str(self))

        # 0. Print the input signal.
        print(f"----- the input signal (distorted data symbols) -----\n{np.round(y, 2)}\n\n")

        # 1. Get the channel state information.
        N0 = self.Pt / ((10**(CSI['SNR']/10.0)) * 2*self.B)
        print(f"----- the channel state information -----\n\nH = \n{np.round(CSI['H'], 2)}\n\nS = {np.round(CSI['S'], 2)}\n\nU =\n {np.round(CSI['U'], 2)}\n\nVh =\n {np.round(CSI['Vh'], 2)}\n\nNoise power spectral density N0 = {round(N0, 4)} W/Hz\n\n\n")

        # 2. Set the resource allocation parameters.
        self.resource_allocation(CSI, CCI)
        print(f"----- resource allocation results -----\n Power allocation Pi = {np.round(self._Pi, 2)},\n Constellation sizes Mi = {self._Mi}\n\n")

        # 3. Combine the received symbols.
        y_tilda = self.combiner(y[:, :K], CSI['U'])
        print(f"----- the combined symbols -----\n{np.round(y_tilda, 2)}\n\n")

        # 4. Equalize the combined symbols.
        u = self.equalizer(y_tilda, CSI['S'])
        print(f"----- the equalized symbols -----\n{np.round(u, 2)}\n\n")

        # 5. Detect the transmitted data symbols.
        a_hat = self.detector(u)
        print(f"----- the estimated symbols -----\n{np.round(a_hat, 2)}\n\n")

        # 6. Demap the estimated data symbols into bit sequences.
        b_hat = self.demapper(a_hat)
        print(f"----- the reconstructed bit vector -----\n")
        for rx_antenna in range(self.Nr):
            print(f" Receive Antenna {rx_antenna+1}: bits_hat = {b_hat[rx_antenna]}\n")
        
        # 7. Combine the bit sequences to create the output bitstream.
        bitstream_hat = self.bit_deallocator(b_hat)
        print(f"\n----- the reconstructed bitstream -----\n bits_hat = {bitstream_hat}\n\n")
        
        print("\n\n========== End Receiver Simulation Example ==========\n")


        # RETURN
        return bitstream_hat
