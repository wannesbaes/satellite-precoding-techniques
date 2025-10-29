# This module contains the implementation of the receiver component of a SU-MIMO SVD communication system.

import numpy as np
import matplotlib.pyplot as plt


class Receiver:
    """
    Description
    -----------
    The receiver of a single-user multiple-input multiple-output (SU-MIMO) communication system, in which the channel state information is available at the receiver (and transmitter).

    When a receiver is called and given an received input signal r containing distorted data symbols, it executes the waterfilling algorithm to determine the constellation size and allocated power on each receive antenna, postcodes and equalizes the received symbols, searches the most probable transmitted data symbols, demaps them into bit sequences, and returns the reconstructed bitstream.

    Attributes
    ----------
    Nr : int
        Number of receiving antennas.
    Pt : float
        Total available transmit power.
    B : float
        Bandwidth of the communication system.
    type : str
        Type of the modulation constellation ('PAM', 'PSK', or 'QAM').
    
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
    postcode():
        Postcode the received symbols using the left singular vectors of the channel matrix H.
    equalizer():
        Equalize the postcoded symbols using the singular values of the channel matrix H and the allocated power on each antenna.
    estimator():
        Search for the most probable transmitted data symbol vectors from the distorted (equalized & postcoded) data symbol vectors.
    demapper():
        Convert the estimated data symbol vectors into the corresponding bit sequence vectors according to the specified modulation constellation on each receive antenna.
    bit_deallocator()
        Combine the reconstructed bits on each antenna to create the output bitstream.
    simulate():
        Simulate the receiver operations and return the reconstructed bitstream.
    """

    
    # INITIALIZATION AND REPRESENTATION

    def __init__(self, Nr, constellation_type, Pt=1, B=1):
        """ Initialize the receiver. """

        self.Nr = Nr
        self.Pt = Pt
        self.B = B
        self.type = constellation_type

    def __str__(self):
        """ Return a string representation of the receiver object. """
        return f"Receiver: \n  - Number of antennas = {self.Nr}\n  - Constellation = {self.type}"

    def __call__(self, r: np.ndarray, SNR: float, CSI: tuple, M: int = None) -> None:
        """ Allow the receiver object to be called as a function. When called, it executes the simulate() method. """
        return self.simulate(r, SNR, CSI, M)


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
        Pi, Ci, Mi : tuple (3 times a 1D numpy array of length used_eigenchannels)
            - Pi: The optimal power allocation for each receive antenna.
            - Ci: The capacity of each eigenchannel.
            - Mi: The constellation size on each receive antenna.

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
        Mi = 2 ** np.floor( Ci ).astype(int) if self.type != 'QAM' else 4 ** np.floor( Ci / 2 ).astype(int)


        # 4. Return.
        return Pi, Ci, Mi

    def postcoder(self, r: np.ndarray, U: np.ndarray) -> np.ndarray:
        """
        Description
        -----------
        Postcode the received symbols using the left singular vectors of the channel matrix H.

        Parameters
        ----------
        r : (Nr x num_symbols) numpy array
            The input signal of distorted data symbols received from the channel and to be processed by the receiver.
        U : (Nr x Nr) numpy array
            The left singular vectors of the channel matrix H.
        
        Returns
        -------
        postcoded_symbols: (Nr x num_symbols) numpy array
            The postcoded symbols.
        """

        postcoded_symbols = U.conj().T @ r
        return postcoded_symbols

    def equalizer(self, postcoded_symbols: np.ndarray, S: np.ndarray, Pi: np.ndarray) -> np.ndarray:
        """
        Description
        -----------
        Equalize the postcoded symbols using the singular values of the channel matrix H and the allocated power on each antenna.

        Parameters
        ----------
        postcoded_symbols : (Nr x num_symbols) numpy array
            The postcoded symbols to be equalized.
        S : (Nr,) numpy array
            The singular values of the channel matrix H.
        Pi : (Nr,) numpy array
            The allocated power on each antenna.

        Returns
        -------
        equalized_symbols : (Nr x num_symbols) numpy array
            The equalized symbols.
        """

        useful_eigenchannels = min(len(Pi[Pi>0]), len(S))
        equalized_symbols = postcoded_symbols[:useful_eigenchannels] / (S*np.sqrt(Pi[:len(S)]))[:useful_eigenchannels][:, np.newaxis]

        return equalized_symbols

    def estimate(self, equalized_symbols: np.ndarray, M: int, type: str) -> np.ndarray:
        """
        Description
        -----------
        Convert a  distorted (equalized & postcoded) data symbol sequence (the decision variables) into the most probable transmitted data symbol sequence according to the specified modulation constellation.

        Parameters
        ----------
        equalized_symbols : 1D numpy array (dtype: complex)
            The distorted (equalized & postcoded) data symbol sequence (the decision variables).
        M : int
            The size of the modulation constellation.
        type : str
            The type of modulation constellation (choose between 'PAM', 'PSK', 'QAM').
        
        Returns
        -------
        datasymbols_hat : 1D numpy array (dtype: complex)
            The estimated transmitted data symbol sequence.
        """

        # 1. Construct the modulation constellation.

        assert (M & (M - 1) == 0) and ((M & 0xAAAAAAAA) == 0 or self.type != 'QAM'), f'The constellation size M is invalid.\nFor PAM and PSK modulation, it must be a power of 2. For QAM Modulation, M must be a power of 4. Right now, M equals {M} and the type is {self.type}.'

        if type == 'PAM':
            constellation = np.arange(-(M-1), (M-1) + 1, 2) * np.sqrt(3/(M**2-1))

        elif type == 'PSK':
            constellation = np.exp(1j * 2*np.pi * np.arange(M) / M)

        elif type == 'QAM':
            c_sqrtM_PAM = np.arange(-(np.sqrt(M)-1), (np.sqrt(M)-1) + 1, 2) * np.sqrt(3 / (2*(M-1)))
            real_grid, imaginary_grid = np.meshgrid(c_sqrtM_PAM, c_sqrtM_PAM)
            constellation = (real_grid + 1j*imaginary_grid).flatten()

        else :
            raise ValueError(f'The constellation type is invalid.\nChoose between "PAM", "PSK", or "QAM". Right now, type is {type}.')
        

        # 2. Map each decision variable to the nearest data symbol in the constellation.

        distances = np.abs(constellation[:, np.newaxis] - equalized_symbols)
        datasymbols_hat = constellation[np.argmin(distances, axis=0)]

        # 3. Return
        return datasymbols_hat

    def estimator(self, equalized_symbols: np.ndarray, Mi: np.ndarray) -> np.ndarray:
        """
        Description
        -----------
        Convert the distorted (equalized & postcoded) data symbol vectors (the decision variable vectors) into the most probable transmitted data symbol vectors according to the specified modulation constellation for each transmit antenna.

        Parameters
        ----------
        equalized_symbols : 2D numpy array (dtype: complex)
            The distorted (equalized & postcoded) data symbol vectors (the decision variable vectors).
        Mi : 1D numpy array (dtype: int)
            The size of the constellation for each transmit antenna.
        
        Returns
        -------
        symbols_hat : 2D numpy array (dtype: complex)
            The estimated transmitted data symbol vectors.
        """
        
        num_symbols = len(equalized_symbols[0])
        symbols_hat = np.zeros((self.Nr, num_symbols), dtype=complex)
        
        for rx_antenna in range(self.Nr):
            symbols_hat[rx_antenna, :] = self.estimate(equalized_symbols[rx_antenna, :], Mi[rx_antenna], self.type) if Mi[rx_antenna] >= 2 else np.zeros(num_symbols, dtype=complex)
        
        return symbols_hat

    def demap(self, symbols_hat: np.ndarray, M: int, type: str) -> np.ndarray:
        """
        Description
        -----------
        Convert an estimated data symbol sequence into the corresponding bit sequence according to the specified modulation constellation.

        Parameters
        ----------
        symbols_hat : 1D numpy array (dtype: complex) (length: num_symbols)
            The estimated data symbol sequence.
        M : int
            The size of the modulation constellation.
        type : str
            The type of modulation constellation (choose between 'PAM', 'PSK', 'QAM').
        
        Returns
        -------
        bit_array : 1D numpy array (dtype: int) (length: num_symbols * log2(M))
            The corresponding bit sequence.
        """

        # 1. Convert the data symbols to the corresponding decimal values, according to the specified constellation type.

        assert (M & (M - 1) == 0) and ((M & 0xAAAAAAAA) == 0 or self.type != 'QAM'), f'The constellation size M of antenna {i} is invalid.\nFor PAM and PSK modulation, it must be a power of 2. For QAM Modulation, M must be a power of 4. Right now, M equals {M} and the type is {self.type}.'
        
        if type == 'PAM':
            delta = np.sqrt(3/(M**2-1))
            decimals = np.round((1/2) * (symbols_hat / delta + (M-1))).astype(int)
        
        elif type == 'PSK':
            phases = np.angle(symbols_hat) % (2*np.pi)
            decimals = np.round((phases * M) / (2 * np.pi)).astype(int)
        
        elif type == 'QAM':
            c_sqrtM_PAM = np.arange(-(np.sqrt(M)-1), (np.sqrt(M)-1) + 1, 2) * np.sqrt(3 / (2*(M-1)))
            real_grid, imaginary_grid = np.meshgrid(c_sqrtM_PAM, c_sqrtM_PAM[::-1])
            constellation = (real_grid + 1j*imaginary_grid)
            constellation[1::2] = constellation[1::2, ::-1]
            constellation = constellation.flatten()

            sort_idx = np.argsort(constellation)
            pos = np.searchsorted(constellation[sort_idx], symbols_hat)
            decimals = sort_idx[pos]

        else:
            raise ValueError(f'The constellation type is invalid.\nChoose between "PAM", "PSK", or "QAM". Right now, type is {type}.')
        

        # 2. Convert the decimal values to the corresponding blocks of m bits in gray code.

        m = int(np.log2(M))
        binarycodes = ((decimals[:, None].astype(int) & (1 << np.arange(m))[::-1].astype(int)) > 0).astype(int)

        graycodes = np.zeros_like(binarycodes)
        graycodes[:, 0] = binarycodes[:, 0]
        for i in range(1, m):
            graycodes[:, i] = binarycodes[:, i] ^ binarycodes[:, i - 1]


        # 3. Convert the gray code blocks to a single bit sequence and return it.

        bit_array = graycodes.flatten()
        return bit_array

    def demapper(self, symbols_hat: np.ndarray, Mi: np.ndarray) -> list:
        """
        Description
        -----------
        Convert the estimated data symbol vectors into the corresponding bit sequence vectors according to the specified modulation constellation.

        Parameters
        ----------
        symbols_hat : 2D numpy array (dtype: complex) (shape: Nr x num_symbols)
            The estimated data symbol vectors.
        Mi : 1D numpy array (dtype: int) (length: used_eigenchannels)
            The size of the constellation for each transmit antenna.
        
        Returns
        -------
        bits_hat : list of 1D numpy arrays (dtype: int) (shape: Nr x num_symbols * log2(Mi[rx_antenna]))
            The corresponding bit sequence vectors.
        """

        Mi = np.pad(Mi, (0, max(0, self.Nr - len(Mi))), mode='constant', constant_values=1)
        bits_hat = [0] * self.Nr
        for rx_antenna in range(self.Nr):
            bits_hat[rx_antenna] = self.demap(symbols_hat[rx_antenna, :], Mi[rx_antenna], self.type) if Mi[rx_antenna] >= 2 else np.array([], dtype=int)
        return bits_hat

    def bit_deallocator(self, bits_hat: np.ndarray, Mi: np.ndarray) -> np.ndarray:
        """
        Description
        -----------
        Combine the reconstructed bits on each antenna to create the output bitstream. It performs the inverse operation of the bit_allocator in the receiver.

        Parameters
        ----------
        bits_hat : list of 1D numpy arrays (dtype: int) (shape: Nr x num_symbols * log2(Mi[rx_antenna]))
            The reconstructed bits on each antenna.
        Mi : 1D numpy array (dtype: int) (length: Nr)
            The constellation size on each receive antenna.
        
        Return
        ------
        bitstream: 1D numpy array (dtype: int) (length: Nr * num_symbols * log2(Mi[rx_antenna]))
            The reconstructed bitstream. This is the output of the receiver.
        """

        bitstream = np.array([], dtype=int)
        m = np.log2(Mi).astype(int)
        num_symbols = len(bits_hat[0]) // m[0]

        for symbol_i in range(num_symbols):
            for rx_antenna in range(len(Mi[Mi>=2])):
                bitstream = np.concatenate((bitstream, bits_hat[rx_antenna][symbol_i*m[rx_antenna]: (symbol_i+1)*m[rx_antenna]]))

        return bitstream

    def simulate(self, r: np.ndarray, SNR: float, CSI: tuple, M: int = None) -> np.ndarray:
        """
        Description
        -----------
        Simulate the receiver operations:\n
        (1) Get the channel state information.\n
        (2) Execute the waterfilling algorithm to determine the constellation size and allocated power that will be received on each antenna.\n Note: If a fixed constellation size M is provided as input, the waterfilling algorithm is skipped and this constellation size is used for all antennas.\n
        (3) [postcoder] Postcode the received symbols using the left singular vectors of the channel matrix H.\n
        (4) [equalizer] Equalize the postcoded symbols using the singular values of the channel matrix H and the allocated power on each antenna.\n
        (5) [estimator] Convert the received (equalized postcoded) data symbols into the most probable data symbols.\n
        (6) [demapper] Convert the data symbol sequences into the corresponding bit sequences according to the specified modulation constellation.\n
        (7) [bit deallocator] Combine the reconstructed bit sequences to create the output bitstream.\n
        (8) Return the output bitstream.\n

        Parameters
        ----------
        r : np.ndarray
            The input signal of distorted data symbols received from the channel and to be processed by the receiver.
        SNR : float
            The signal-to-noise ratio (SNR) in dB.
        CSI : tuple
            A tuple containing the channel state information of the MIMO channel: (H, U, S, Vh).
        M : int, optional
            The fixed constellation size for all receive antennas. If provided, the waterfilling algorithm is skipped and this constellation size is used for all antennas.
        
        Returns
        -------
        bits_hat : np.ndarray
            The reconstructed bitstream.
        """

        # Setup.
        N0 = self.Pt / ((10**(SNR/10.0)) * 2*self.B)
        H, U, S, Vh = CSI

        if M is None:
            Pi, Ci, Mi = self.waterfilling(H, S, N0)
            Pi = np.pad(Pi, pad_width=(0, self.Nr - len(Pi)), mode='constant', constant_values=0)
            Mi = np.pad(Mi, pad_width=(0, self.Nr - len(Mi)), mode='constant', constant_values=1)
        else:
            rank_H = np.linalg.matrix_rank(CSI[0])
            Mi = np.array([M] * rank_H)
            Pi = np.array([self.Pt / rank_H] * rank_H)
        
        # Receiver Operations.
        postcoded_symbols = self.postcoder(r, U)
        equalized_symbols = self.equalizer(postcoded_symbols, S, Pi)
        symbols_hat = self.estimator(equalized_symbols, Mi)
        bits_hat = self.demapper(symbols_hat, Mi)
        bits_hat = self.bit_deallocator(bits_hat, Mi)

        return bits_hat


    # TESTS AND PLOTS

    def plot_estimator(self, decision_variable: complex, M: int, SNR: float) -> None:
        """
        Description
        -----------
        Plot the decision variable, the complete constellation and the chosen constellation point to visualize the estimation process.

        Parameters
        ----------
        decision_variable : complex
            The decision variable to be estimated.
        M : int
            The size of the modulation constellation.
        type : str
            The type of modulation constellation (choose between 'PAM', 'PSK', 'QAM').
        """

        # 1. Construct the modulation constellation.

        assert (M & (M - 1) == 0) and ((M & 0xAAAAAAAA) == 0 or self.type != 'QAM'), f'The constellation size M is invalid.\nFor PAM and PSK modulation, it must be a power of 2. For QAM Modulation, M must be a power of 4. Right now, M equals {M} and the type is {self.type}.'

        if self.type == 'PAM':
            constellation = np.arange(-(M-1), (M-1) + 1, 2) * np.sqrt(3/(M**2-1))

        elif self.type == 'PSK':
            constellation = np.exp(1j * 2*np.pi * np.arange(M) / M)

        elif self.type == 'QAM':
            c_sqrtM_PAM = np.arange(-(np.sqrt(M)-1), (np.sqrt(M)-1) + 1, 2) * np.sqrt(3 / (2*(M-1)))
            real_grid, imaginary_grid = np.meshgrid(c_sqrtM_PAM, c_sqrtM_PAM)
            constellation = (real_grid + 1j*imaginary_grid).flatten()

        else :
            raise ValueError(f'The constellation type is invalid.\nChoose between "PAM", "PSK", or "QAM". Right now, type is {type}.')
        

        # 2. Estimate the decision variable.
        symbol_hat = self.estimate(np.array([decision_variable]), M, self.type)[0]

        # 3. Plot.
        fig, ax = plt.subplots()
        ax.scatter(np.real(decision_variable), np.imag(decision_variable), color='tab:orange', s=60, label='Decision Variable')
        ax.scatter(np.real(symbol_hat), np.imag(symbol_hat), color='tab:blue', alpha=1, s=60, label='Estimated Symbol')
        ax.scatter(np.real(constellation[constellation != decision_variable]), np.imag(constellation[constellation != decision_variable]), color='tab:blue', alpha=0.5, s=50, label='Constellation Points')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.set_title(f'Estimator Visualization for {M}-{self.type} \nSNR: {SNR} dB')
        ax.set_xlabel('Real Part')
        ax.set_ylabel('Imaginary Part')
        ax.legend()
        ax.grid()
        ax.axis('equal')
        fig.tight_layout()

        return fig, ax

    def print_simulation_example(self, r: np.ndarray, SNR: float, CSI: tuple, K: int = 1) -> None:
        """
        Description
        -----------
        Print a step-by-step example of the receiver operations (see simulate() method) for given input signal r, SNR, and channel state information (CSI). Only the first K data symbols vectors are considered.

        Parameters
        ----------
        r : np.ndarray
            The input signal of distorted data symbols received from the channel and to be processed by the receiver.
        SNR : float
            The signal-to-noise ratio (SNR) in dB.
        CSI : tuple
            A tuple containing the channel state information of the MIMO channel: (H, U, S, Vh).
        K : int, optional
            The maximum number of symbol vectors to consider for illustration.
        """

        # PRINTING EXAMPLE

        print("\n\n========== Receiver Simulation Example ==========\n")

        # 0. Print the input signal.
        print(f"----- the input signal (distorted data symbols) -----\n{np.round(r, 2)}\n\n")

        # 1. Get the channel state information.
        N0 = self.Pt / ((10**(SNR/10.0)) * 2*self.B)
        H, U, S, Vh = CSI
        print(f"----- the channel state information -----\n\nH = \n{np.round(H, 2)}\n\nS = {np.round(S, 2)}\n\nU =\n {np.round(U, 2)}\n\nVh =\n {np.round(Vh, 2)}\n\nNoise power spectral density N0 = {round(N0, 4)} W/Hz\n\n\n")

        # 2. Execute the waterfilling algorithm to determine the constellation size and power allocation on each receive antenna.
        Pi, Ci, Mi = self.waterfilling(H, S, N0)
        Pi = np.pad(Pi, pad_width=(0, self.Nr - len(Pi)), mode='constant', constant_values=0)
        Mi = np.pad(Mi, pad_width=(0, self.Nr - len(Mi)), mode='constant', constant_values=1)
        print(f"----- waterfilling algorithm results -----\n Power allocation Pi = {np.round(Pi, 2)},\n Channel capacities Ci = {np.round(Ci, 2)},\n Constellation sizes Mi = {Mi}\n\n")

        # 3. Postcode the received symbols.
        postcoded_symbols = self.postcoder(r[:, :K], U)
        print(f"----- the postcoded symbols -----\n{np.round(postcoded_symbols, 2)}\n\n")

        # 4. Equalize the postcoded symbols.
        equalized_symbols = self.equalizer(postcoded_symbols, S, Pi)
        print(f"----- the equalized symbols -----\n{np.round(equalized_symbols, 2)}\n\n")

        # 5. Estimate the transmitted data symbols.
        symbols_hat = self.estimator(equalized_symbols, Mi)
        print(f"----- the estimated symbols -----\n{np.round(symbols_hat, 2)}\n\n")

        # 6. Demap the estimated data symbols into bit sequences.
        bits_hat = self.demapper(symbols_hat, Mi)
        print(f"----- the reconstructed bitstreams -----\n")
        for rx_antenna in range(self.Nr):
            print(f" Receive Antenna {rx_antenna+1}: bits_hat = {bits_hat[rx_antenna]}\n")
        
        # 7. Combine the bit sequences to create the output bitstream.
        bits_hat = self.bit_deallocator(bits_hat, Mi)
        print(f"\n----- the reconstructed bitstream -----\n bits_hat = {bits_hat}\n\n")
        
        print("\n\n========== End Receiver Simulation Example ==========\n")


        # PLOTS
        # fig1, ax1 = self.plot_estimator(decision_variable=equalized_symbols[1, 0], M=Mi[1], SNR=SNR)
        # plt.show()

        # RETURN
        return bits_hat



if __name__ == "__main__":

    # Initialize the receiver.
    receiver = Receiver(Nr=4, constellation_type='QAM')

    # Initialize the transmitter and channel.
    import transmitter as tx
    transmitter = tx.Transmitter(Nt=5, constellation_type='QAM')
    import channel as ch
    channel = ch.Channel(Nt=5, Nr=4)

    s = transmitter(bits=np.random.randint(0, 2, size=100), SNR=10, CSI=channel.get_CSI())
    r = channel(s=s, SNR=10)

    # Receiver simulation example.
    receiver.print_simulation_example(r=r, SNR=10, CSI=channel.get_CSI())