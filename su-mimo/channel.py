# This module contains the implementation of the channel component of a SU-MIMO SVD communication system.

import numpy as np
import matplotlib.pyplot as plt


class Channel:
    """
    Description
    -----------
    The channel of a single-user multiple-input multiple-output (SU-MIMO) communication system. 

    The channel is modeled as a distortion-free MIMO channel. The channel matrix can be either provided or initialized with independent and identically distributed (i.i.d.) complex Gaussian random variables. 
    In addition, the channel adds complex proper, circularly-symmetric additive white Gaussian noise (AWGN) to the transmitted symbols, based on a specified signal-to-noise ratio (SNR in dB).

    Attributes
    ----------
    Nt : int
        The number of transmitting antennas.
    Nr : int
        The number of receiving antennas.
    H : 2D numpy array (dtype: complex)
        The MIMO channel matrix (shape: Nr x Nt).
    U : 2D numpy array (dtype: complex)
        The left singular vectors of the channel matrix H (shape: Nr x Nr).
    S : 1D numpy array (dtype: float)
        The singular values of the channel matrix H (shape: (Rank(H),)).
    Vh : 2D numpy array (dtype: complex)
        The right singular vectors of the channel matrix H (shape: Nt x Nt).
    
    Methods
    -------
    __init__()
        Initialize the channel object.
    __str__()
        Return a string representation of the channel object.
    __call__()
        Allow the channel object to be called as a function. When called, it executes the simulate() method.
    
    set_CSI()
        Initialize the MIMO channel matrix and compute its SVD.
    get_CSI()
        Get the current channel state information (CSI), in terms of the channel matrix H and its SVD (U, S, Vh).
    generate_noise()
        Generate complex proper, circularly-symmetric additive white Gaussian noise (AWGN) vectors for every transmitted symbol vector, based on the specified SNR.
    simulate()
        Simulate the channel operations. Return the channel output signal r.
    
    plot_scatter_diagram()
        Plot a scatter diagram of the received symbol vectors for a given SNR and input signal.
    print_simulation_example()
        Print a step-by-step example of the channel operations for a given SNR and input signal.
    """


    # INITIALIZATION AND REPRESENTATION

    def __init__(self, Nt: int, Nr: int, H: np.ndarray = None) -> None:
        """ Initialize the channel. """

        self.Nt = Nt
        self.Nr = Nr

        self._H = H
        self._U = None
        self._S = None
        self._Vh = None
        self.set_CSI()
    
    def __str__(self) -> str:
        """ Return a string representation of the channel object. """
        return f"Channel: \n  - Number of transmitting and receiving antennas is {self.Nt} and {self.Nr}\n\n   - H = \n{self._H}\n\n  - S = \n{self._S}\n\n"

    def __call__(self, s: np.ndarray, SNR: float) -> np.ndarray:
        """ Allow the channel object to be called as a function. When called, it executes the simulate() method. """
        return self.simulate(s, SNR)

    
    # FUNCTIONALITY

    def set_CSI(self):
        """
        Description
        -----------
        Initialize the MIMO channel matrix and compute its SVD!
        If no channel is provided, the channel matrix is initialized with i.i.d. complex Gaussian (zero mean and unit variance) random variables.
        """          
        
        # Initialize the MIMO channel matrix H with i.i.d. complex Gaussian (zero mean and unit variance) random variables, if no channel is provided.
        if self._H is None: self._H = (np.random.randn(self.Nr, self.Nt) + 1j * np.random.randn(self.Nr, self.Nt)) / np.sqrt(2)
        
        # Compute the SVD of the channel matrix H and store the results in U, SIGMA, and Vh.
        self._U, self._S, self._Vh = np.linalg.svd(self._H)

    def get_CSI(self):
        """ Get the current channel state information (CSI), in terms of the channel matrix H and its SVD (U, S, Vh)."""
        return self._H, self._U, self._S, self._Vh

    def reset(self, H: np.ndarray = None):
        """
        Description
        -----------
        Reset the channel properties by re-initializing the MIMO channel matrix and its SVD.

        Parameters
        ----------
        H : 2D numpy array (dtype: complex), optional
            The MIMO channel matrix (shape: Nr x Nt). If not provided, the channel matrix is initialized with i.i.d. complex Gaussian (zero mean and unit variance) random variables.
        """
        
        self._H = H
        self._U = None
        self._S = None
        self._Vh = None
        self.set_CSI()

    def generate_noise(self, s: np.ndarray, SNR: float) -> np.ndarray:
        """
        Description
        -----------
        Generate complex proper, circularly-symmetric additive white Gaussian noise (AWGN) vectors (w[k]) for every transmitted symbol vector, based on the specified SNR.

        Parameters
        ----------
        s : np.ndarray
            The input signal. This signal consists of the precoded symbols that the transmitter sends through the channel. Shape: (Nt, N_symbols)
        SNR : float
            The signal-to-noise ratio (SNR) in dB.
        
        Returns
        -------
        noise : np.ndarray
            The generated noise vectors. Shape: (Nr, N_symbols)
        """
        
        # 1. Compute the noise power based on the specified SNR and the signal power of s.
        P_signal = np.mean( np.sum( np.abs(s)**2, axis=0 ) )
        P_noise = P_signal / (10**(SNR/10))
        sigma = np.sqrt(P_noise / 2)

        # 2. Generate complex proper, circularly-symmetric AWGN noise vectors with the computed noise power.
        noise = sigma * (np.random.randn(self.Nr, s.shape[1]) + 1j * np.random.randn(self.Nr, s.shape[1]))

        # 3. Return.
        return noise

    def simulate(self, s: np.ndarray, SNR: float) -> np.ndarray:
        """
        Description
        -----------
        Simulate the channel operations:\n
        (1) Transmit the precoded symbols through the MIMO channel. This is modeled as a matrix multiplication between the channel matrix H and the precoded symbols.\n
        (2) Add complex proper, circularly-symmetric additive white Gaussian noise (AWGN) to the received symbols, based on the specified SNR.\n

        Parameters
        ----------
        s : np.ndarray
            The input signal. This signal consists of the precoded symbols that the transmitter sends through the channel. Shape: (Nt, N_symbols)
        
        Returns
        -------
        r : np.ndarray
            The output signal. This signal consists of the received symbols after passing the input signal through the channel and adding noise. Shape: (Nr, N_symbols)
        """

        # 1. Transmit the precoded symbols through the MIMO channel.
        r = self._H @ s

        # 2. Add complex proper, circularly-symmetric additive white Gaussian noise (AWGN) to the received symbols, based on the specified SNR.
        w = self.generate_noise(s, SNR)
        r = r + w

        # 3. Return.
        return r


    # TESTS AND PLOTS

    def plot_scatter_diagram(self, s: np.ndarray, Pi: np.ndarray, c_sizes: np.ndarray, c_type: str, SNR: float, K: int = 50):
        """
        Description
        -----------
        Plot a scatter diagram of the received symbol vectors.
        Every transmitted symbol is represented by a different color. Every used eigenchannel (antenna) has its own subplot. The number of symbol vectors to be considered is specified by K.

        Parameters
        ----------
        s : np.ndarray
            The input signal. This signal consists of the precoded symbols that the transmitter sends through the channel. Shape: (Nt, N_symbols)
        Pi : np.ndarray
            The power allocation vector. Shape: (Nr,)
        c_sizes : np.ndarray
            The constellation sizes for every eigenchannel. Shape: (Nr,)
        c_type : str
            The constellation type.
        SNR : float
            The signal-to-noise ratio (SNR) in dB.
        K : int, optional
            The maximum number of symbol vectors to consider for illustration.
        
        Returns
        -------
        fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
            The created figure and axes objects.
        """
        
        def generate_constellation(c_size: int, c_type: str) -> np.ndarray:
            """ Generate the constellation points for a given constellation size M and type. """

            if c_type == 'PAM':
                constellation = np.arange(-(c_size-1), (c_size-1) + 1, 2) * np.sqrt(3/(c_size**2-1))
            
            elif c_type == 'PSK':
                constellation = np.exp(1j * 2*np.pi * np.arange(c_size) / c_size)

            elif c_type == 'QAM':
                c_sqrtM_PAM = np.arange(-(np.sqrt(c_size)-1), (np.sqrt(c_size)-1) + 1, 2) * np.sqrt(3 / (2*(c_size-1)))
                real_grid, imaginary_grid = np.meshgrid(c_sqrtM_PAM, c_sqrtM_PAM)
                constellation = (real_grid + 1j*imaginary_grid)
                constellation[1::2] = constellation[1::2, ::-1]
                constellation = constellation.flatten()

            else :
                raise ValueError(f'The constellation type is invalid.\nChoose between "PAM", "PSK", or "QAM". Right now, type is {c_type}.')

            return constellation

        def generate_color_map(c_sizes: np.ndarray) -> list[dict]:
            """ Generate a color map that allocates a unique color to each different transmitted symbol based on its value, for every eigenchannel. """
            
            color_map = []
            
            for c_size in c_sizes:
                colors = plt.get_cmap('hsv', c_size+1)
                constellation = generate_constellation(c_size, c_type)
                eigenchannel_color_map = {constellation[i]: colors(i) for i in range(c_size)}
                color_map.append(eigenchannel_color_map)

            return color_map
        

        def reconstrct_tx_symbols(s: np.ndarray, Pi: np.ndarray, c_sizes: np.ndarray) -> np.ndarray:
            """ Reconstruct the transmitted symbols from the precoded symbols s. """

            s_prime = self._Vh @ s
            s_prime = s_prime[:np.sum(c_sizes > 1), :K]
            symbols = s_prime / np.sqrt(Pi[c_sizes > 1].reshape(-1, 1))

            for eigenchannel, c_size in enumerate(c_sizes[c_sizes > 1]):
                constellation = generate_constellation(c_size, c_type)
                distances = np.abs(constellation.reshape(-1, 1) - symbols[eigenchannel])
                symbols[eigenchannel] = constellation[np.argmin(distances, axis=0)]
            
            return symbols

        def construct_rx_symbols(s: np.ndarray, SNR: float) -> np.ndarray:
            """ Construct the received symbols from the precoded symbols s, according to the current channel and SNR. """
            
            r = self.simulate(s, SNR)
            r_prime = self._U.conj().T @ r
            r_prime = r_prime[:np.sum(c_sizes > 1), :K]
            
            return r_prime

        symbols = reconstrct_tx_symbols(s, Pi, c_sizes)
        r_prime = construct_rx_symbols(s, SNR)
        
        color_map = generate_color_map(c_sizes)
        fig, axes = plt.subplots(1, symbols.shape[0], figsize=(6*symbols.shape[0], 6))

        for eigenchannel in range(symbols.shape[0]):
            
            ax = axes[eigenchannel] if symbols.shape[0] > 1 else axes
            
            # Plot the received symbols for the current eigenchannel with colors based on the transmitted symbols.
            colors = [color_map[eigenchannel][symbol] for symbol in symbols[eigenchannel]]
            ax.scatter(r_prime[eigenchannel].real, r_prime[eigenchannel].imag, color=colors, marker='.', alpha= 0.75, s=50)


            # Plot the constellation points for reference.
            constellation_points = generate_constellation(c_sizes[eigenchannel], c_type)
            colors = [color_map[eigenchannel][point] for point in constellation_points]
            constellation_points = constellation_points * (self._S[eigenchannel] * np.sqrt(Pi[eigenchannel]))
            ax.scatter(constellation_points.real, constellation_points.imag, color=colors, edgecolor='black', marker='o', alpha= 1.0, s=50)

            ax.axhline(0, color='black', linewidth=0.5)
            ax.axvline(0, color='black', linewidth=0.5)
            ax.set_title(f'Eigenchannel {eigenchannel+1}: {c_sizes[eigenchannel]}-{c_type}')
            ax.set_xlabel('Real Part')
            ax.set_ylabel('Imaginary Part')
            ax.grid(True, which='both', linestyle='--', alpha=0.7)
            ax.axis('equal')
            
        fig.suptitle(f'Scatter Diagram after SVD Processing \nSNR = {SNR} dB & R = {SNR}%')
        fig.tight_layout()
    
        return fig, axes

    def print_simulation_example(self, s: np.ndarray, SNR: float, K: int = 1) -> tuple:
        """
        Description
        -----------
        Print a step-by-step example of the channel operations (see simulate() method) for a given SNR and input signal s. Only the first K symbol vectors are considered for illustration.

        Parameters
        ----------
        s : np.ndarray
            The input signal. This signal consists of the precoded symbols that the transmitter sends through the channel. Shape: (Nt, N_symbols)
        SNR : float
            The signal-to-noise ratio (SNR) in dB.
        K : int, optional
            The maximum number of symbol vectors to consider for illustration.
        
        Return
        ------
        r : 2D numpy array of shape (Nr, K)
            The input signal for the receiver.

        Notes
        -----
        For demonstration purposes only.
        """

        # PRINTING EXAMPLE
        print("\n\n========== Channel Simulation Example ==========\n")

        # 0. Print the input symbol vector sequence.
        print(f"----- the input symbol vector sequence -----\n{np.round(s, 2)}\n\n")

        # 1. Transmit the precoded symbols through the MIMO channel.
        r = self._H @ s
        print(f"----- the received symbol vector sequence before adding noise -----\n{np.round(r, 2)}\n\n")

        # 2. Add complex proper, circularly-symmetric additive white Gaussian noise (AWGN) to the received symbols, based on the specified SNR.
        w = self.generate_noise(s, SNR)
        r = r + w
        SNR_calculated = 10 * np.log10( np.mean( np.sum( np.abs(self._H @ s)**2, axis=0 ) ) / np.mean( np.sum( np.abs(w)**2, axis=0 ) ) )
        print(f"----- the received symbol vector sequence after adding noise -----\n{np.round(r, 2)}\nSNR [calculated]: {np.round(SNR_calculated, 2)} dB\n\n")

        print("======== End Channel Simulation Example ========\n\n")

        # RETURN
        return r


if __name__ == "__main__":

    # Initialize the channel.
    channel = Channel(Nt=5, Nr=4)
    
    # Initialize the transmitted signal s.
    import transmitter as tx
    transmitter = tx.Transmitter(Nt=5, c_type='QAM')
    s, Pi, Mi = transmitter.simulate(bits=np.random.randint(0, 2, size=2400), SNR=7, CSI=channel.get_CSI())

    # Channel simulation example.
    # channel.print_simulation_example(s=s, SNR=25, K=2)

    # Scatter diagram.
    fig, ax = channel.plot_scatter_diagram(s=s, Pi=Pi, c_sizes=Mi, c_type=transmitter.type, SNR=7, K=100)
    plt.show()