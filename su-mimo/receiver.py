# This module contains the implementation of the receiver component of a SU-MIMO SVD communication system.

import numpy as np
import matplotlib.pyplot as plt


class Receiver:
    """
    Description
    -----------
    The receiver of a single-user multiple-input multiple-output (SU-MIMO) digital communication system, in which the channel state information is available at the receiver (and transmitter).

    When the receiver is called and given an received input signal r, it ...
        determines the power allocation and constellation size on each receive antenna
        postcodes and equalizes the received symbol vectors
        searches the most probable transmitted data symbol vectors
        demaps them into bit vectors
        and combines the reconstructed bits on each antenna to create the output bitstream.

    Attributes
    ----------
    Nr : int
        Number of receiving antennas.
    c_type : str
        Constellation type. (Choose between 'PAM', 'PSK', or 'QAM'.)
    data_rate : float
        The data rate at which data is transmitted. It is specified is the fraction of the channel capacity. Default is 1.0.
    Pt : float
        Total available transmit power. Default is 1.0.
    B : float
        Bandwidth of the communication system. Default is 0.5.
    c_size : int, optional
        Constellation size. Only used if a fixed constellation size is desired. Default is None.
    
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
    
    waterfilling():
        Determine the optimal power allocation and constellation size on each receive antenna using the waterfilling algorithm.
    detect():
        Convert the decision variables into the most probable transmitted data symbols according to the specified modulation constellation.
    demap():
        Convert detected data symbols into the corresponding bits according to the specified modulation constellation.
    
    resource_allocation():
        Determine and store the constellation size and power allocation for each receive antenna.
    postcode():
        Postcode the input signal using the left singular vectors of the channel matrix H.
    equalizer():
        Equalize the postcoded symbol vectors, based on S and Pi.
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

    def __init__(self, Nr, c_type, data_rate=1.0, c_size=None, Pt=1.0, B=0.5):
        """ Initialize the receiver. """

        # Receiver Settings.
        self.Nr = Nr
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
        """ Return a string representation of the receiver object. """
        return f'Receiver: \n  - Number of antennas = {self.Nr}\n  - Constellation = ' + (f'{self.M}' if self.M != None else '') + f'{self.c_type}' + f'\n  - Data rate = {self.data_rate*100}% \n  - Total transmit power Pt = {self.Pt} W\n  - Bandwidth B = {self.B} Hz\n'

    def __call__(self, r, CSI):
        """ Allow the receiver object to be called as a function. When called, it executes the simulate() method. """
        return self.simulate(r, CSI)


    # FUNCTIONALITY

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

    def detect(self, decision_variables, M, c_type):
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

    def demap(self, symbols_hat, M, c_type):
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


    def resource_allocation(self, CSI, CCI):
        """
        Description
        -----------
        Determine and store the constellation size and power allocation for each transmit antenna.

        If a control channel is available, set the resource allocation parameters from the control channel information.
        
        Otherwise, there are two cases:
        - Case 1. No fixed constellation size M is provided as input. In this case, the waterfilling algorithm is executed to determine the optimal (!!) constellation size and power allocation for each transmit antenna.
        - Case 2. A fixed constellation size M is provided as input. In this case, the waterfilling algorithm is skipped and this constant constellation size is used for all antennas. The power is equally divided across the used eigenchannels.

        Parameters
        ----------
        CSI : dict
            The channel state information (SNR, H, U, S, Vh).
        CCI : dict, optional
            The control channel information (Pi, Ci, Mi).
        """

        # If a control channel is available, set the resource allocation parameters from the control channel information.
        if CCI is not None:
            self._Pi = CCI['Pi']
            self._Ci = CCI['Ci']
            self._Mi = CCI['Mi']
            return
        
        # Case 1: Determine the optimal power allocation and constellation size using the waterfilling algorithm.
        if self.M is None:
            N0 = self.Pt / ((10**(CSI['SNR']/10.0)) * 2*self.B)
            Pi, Ci, Mi = self.waterfilling(N0, CSI['H'], CSI['S'])
        
        # Case 2: Use the fixed constellation size M and equally divide the power across the useful eigenchannels.
        else:
            rank_H = np.linalg.matrix_rank(CSI['H'])
            Pi = np.array([self.Pt / rank_H] * rank_H)
            Mi = np.array([self.M] * rank_H)
        
        # Pad Pi and Mi to match the number of transmit antennas Nt.
        Pi = np.pad(Pi, pad_width=(0, self.Nr - len(Pi)), mode='constant', constant_values=0)
        Mi = np.pad(Mi, pad_width=(0, self.Nr - len(Mi)), mode='constant', constant_values=1)

        # Store the results.
        self._Pi = Pi
        self._Ci = Ci
        self._Mi = Mi
        return

    def postcoder(self, r, U):
        """
        Description
        -----------
        Postcode the input signal (distorted data symbol vectors) using the left singular vectors of the channel matrix H.

        Parameters
        ----------
        r : 2D numpy array (dtype: complex, shape: (Nr, N_symbols))
            Input - received signal.
        U : 2D numpy array (dtype: complex, shape: (Nr, Nr))
            The left singular vectors of the channel matrix H.
        
        Returns
        -------
        r_prime : 2D numpy array (dtype: complex, shape: (Nr, N_symbols))
            Output - postcoded symbol vectors.
        """

        r_prime = U.conj().T @ r
        return r_prime

    def equalizer(self, r_prime, S):
        """
        Description
        -----------
        Equalize the postcoded symbol vectors using the singular values of the channel matrix H and the allocated power on each antenna.

        Parameters
        ----------
        r_prime : 2D numpy array (dtype: complex, shape: (Nr, N_symbols))
            Input - postcoded symbol vectors.
        S : 1D numpy array (dtype: float, length=rank_H)
            The singular values of the channel matrix H.

        Returns
        -------
        u : 2D numpy array (dtype: complex, shape: (Nr, N_symbols))
            Output - decision variable vectors.
        """

        useful_eigenchannels = min(len(self._Pi[self._Pi>0]), len(S))
        u = r_prime[:useful_eigenchannels] / (S * np.sqrt(self._Pi[:len(S)]))[:useful_eigenchannels][:, np.newaxis]

        return u

    def detector(self, u):
        """
        Description
        -----------
        Convert the decision variable vectors (distorted (equalized & postcoded) data symbol vectors) into the most probable (minimum distance (MD) detection) transmitted data symbol vectors according to the specified modulation constellation for each transmit antenna.

        Parameters
        ----------
        u : 2D numpy array (dtype: complex, shape: (Nr, N_symbols))
            Input - decision variable vectors.
        
        Returns
        -------
        a_hat : 2D numpy array (dtype: complex, shape: (Nr, N_symbols))
            Output - detected data symbol vectors.
        """
        
        a_hat = np.array([ self.detect(u[rx_antenna, :], self._Mi[rx_antenna], self.c_type) for rx_antenna in range(len(self._Mi[self._Mi > 1])) ])
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

        b_hat = [ self.demap(a_hat[rx_antenna, :], self._Mi[rx_antenna], self.c_type) for rx_antenna in range(len(self._Mi[self._Mi > 1])) ] + [np.array([], dtype=int)]*(len(self._Mi[self._Mi <= 1]))
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

    def simulate(self, r, CSI, CCI=None):
        """
        Description
        -----------
        Simulate the receiver operations:\n
        (1) Get the channel state information.\n
        (2) [resource_allocation] Set the resource allocation parameters, obtained from the control channel information.\n In case there is no control channel, execute the waterfilling algorithm to determine the constellation size and allocated power that will be received on each antenna.\n Note: If a fixed constellation size M is provided as input, the waterfilling algorithm is omitted. The available power is equally allocated across all antennas and this constant constellation size is used for all antennas.\n
        (3) [postcoder] Postcode the received symbol vectors using the left singular vectors of the channel matrix H.\n
        (4) [equalizer] Equalize the postcoded symbol vectors using the singular values of the channel matrix H and the allocated power on each antenna.\n
        (5) [detector] Convert the decision variable vectors into the most probable data symbol vectors.\n
        (6) [demapper] Convert the reconstructed data symbol vectors into the corresponding bit vectors according to the specified modulation constellation.\n
        (7) [bit deallocator] Combine the reconstructed bit vectors to create the output bitstream.\n
        (8) Return the output bitstream.\n

        Parameters
        ----------
        r : 2D numpy array (dtype: complex, shape: (Nr, N_symbols))
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
            - Pi : The power allocation for each transmit antenna. (1D numpy array, dtype: float)
            - Ci : The capacity of each eigenchannel. (1D numpy array, dtype: float)
            - Mi : The constellation size for each transmit antenna. (1D numpy array, dtype: int)
        
        Returns
        -------
        bitstream_hat : 1D numpy array (dtype: int, length: Nr * N_symbols * log2(Mi[rx_antenna]))
            Output - reconstructed bitstream.
        """

        # Receiver Setup.
        self.resource_allocation(CCI)
        
        # Receiver Operations.
        r_prime = self.postcoder(r, CSI['U'])
        u = self.equalizer(r_prime, CSI['S'])
        a_hat = self.detector(u)
        b_hat = self.demapper(a_hat)
        bitstream_hat = self.bit_deallocator(b_hat)

        return bitstream_hat


    # TESTS AND PLOTS

    def plot_detector(self, decision_variable, M, c_type, SNR):
        """
        Description
        -----------
        Plot the decision variable, the complete constellation and the chosen constellation point to visualize the detection process.

        Parameters
        ----------
        decision_variable : complex
            The decision variable to be detected.
        M : int
            Constellation size.
        c_type : str
            Constellation type. (Choose between 'PAM', 'PSK', 'QAM'.)
        SNR : float
            The signal-to-noise ratio (SNR) in dB.
        
        Returns
        -------
        fig, ax : tuple
            The figure and axis objects of the plot.
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
        
        # 2. Detect the decision variable.
        symbol_hat = self.detect(np.array([decision_variable]), M, c_type)[0]

        # 3. Plot.
        fig, ax = plt.subplots()
        ax.scatter(np.real(decision_variable), np.imag(decision_variable), color='tab:orange', s=60, label='Decision Variable')
        ax.scatter(np.real(symbol_hat), np.imag(symbol_hat), color='tab:blue', alpha=1, s=60, label='Detected Symbol')
        ax.scatter(np.real(constellation[constellation != decision_variable]), np.imag(constellation[constellation != decision_variable]), color='tab:blue', alpha=0.5, s=50, label='Constellation Points')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.set_title(f'Detector Visualization for {M}-{c_type} \nSNR: {SNR} dB')
        ax.set_xlabel('Real Part')
        ax.set_ylabel('Imaginary Part')
        ax.legend()
        ax.grid()
        ax.axis('equal')
        fig.tight_layout()

        return fig, ax

    def print_simulation_example(self, r, CSI, CCI, K=1):
        """
        Description
        -----------
        Print a step-by-step example of the receiver operations (see simulate() method) for given input signal r, and channel state information (CSI). Only the first K data symbols vectors are considered.

        Parameters
        ----------
        r : 2D numpy array (dtype: complex, shape: (Nr, N_symbols))
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

        # 0. Print the input signal.
        print(f"----- the input signal (distorted data symbols) -----\n{np.round(r, 2)}\n\n")

        # 1. Get the channel state information.
        N0 = self.Pt / ((10**(CSI['SNR']/10.0)) * 2*self.B)
        print(f"----- the channel state information -----\n\nH = \n{np.round(CSI['H'], 2)}\n\nS = {np.round(CSI['S'], 2)}\n\nU =\n {np.round(CSI['U'], 2)}\n\nVh =\n {np.round(CSI['Vh'], 2)}\n\nNoise power spectral density N0 = {round(N0, 4)} W/Hz\n\n\n")

        # 2. Set the resource allocation parameters.
        Pi, self._Ci, Mi = self.waterfilling(N0, CSI['H'], CSI['S']) if CCI is None else (CCI['Pi'], CCI['Ci'], CCI['Mi'])
        self._Pi = np.pad(Pi, pad_width=(0, self.Nr - len(Pi)), mode='constant', constant_values=0)
        self._Mi = np.pad(Mi, pad_width=(0, self.Nr - len(Mi)), mode='constant', constant_values=1)
        print(f"----- waterfilling algorithm results -----\n Power allocation Pi = {np.round(self._Pi, 2)},\n Channel capacities Ci = {np.round(self._Ci, 2)},\n Constellation sizes Mi = {self._Mi}\n\n")

        # 3. Postcode the received symbols.
        r_prime = self.postcoder(r[:, :K], CSI['U'])
        print(f"----- the postcoded symbols -----\n{np.round(r_prime, 2)}\n\n")

        # 4. Equalize the postcoded symbols.
        u = self.equalizer(r_prime, CSI['S'])
        print(f"----- the equalized symbols -----\n{np.round(u, 2)}\n\n")

        # 5. Detect the transmitted data symbols.
        a_hat = self.detector(u)
        print(f"----- the estimated symbols -----\n{np.round(a_hat, 2)}\n\n")

        # 6. Demap the estimated data symbols into bit sequences.
        b_hat = self.demapper(a_hat)
        print(f"----- the reconstructed bitstreams -----\n")
        for rx_antenna in range(self.Nr):
            print(f" Receive Antenna {rx_antenna+1}: bits_hat = {b_hat[rx_antenna]}\n")
        
        # 7. Combine the bit sequences to create the output bitstream.
        bitstream_hat = self.bit_deallocator(b_hat)
        print(f"\n----- the reconstructed bitstream -----\n bits_hat = {bitstream_hat}\n\n")
        
        print("\n\n========== End Receiver Simulation Example ==========\n")


        # RETURN
        return bitstream_hat



if __name__ == "__main__":

    # Initialize the receiver.
    receiver = Receiver(Nr=4, c_type='QAM')

    # Initialize the transmitter and channel.
    import transmitter as tx
    transmitter = tx.Transmitter(Nt=5, c_type='QAM')
    import channel as ch
    channel = ch.Channel(Nt=5, Nr=4, SNR=15)

    s = transmitter(bits=np.random.randint(0, 2, size=100), CSI=channel.get_CSI())
    r = channel(s=s)

    # Receiver simulation example.
    receiver.print_simulation_example(r=r, CSI=channel.get_CSI(), CCI=transmitter.get_CCI(), K=1)