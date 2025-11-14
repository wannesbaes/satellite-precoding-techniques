# This module contains the implementation of the channel component of a SU-MIMO SVD communication system.

import numpy as np
import matplotlib.pyplot as plt


class Channel:
    """
    Description
    -----------
    The channel of a single-user multiple-input multiple-output (SU-MIMO) digital communication system.

    The channel is modeled as a distortion-free MIMO channel. The channel matrix can be either provided or initialized with independent and identically distributed (i.i.d.) complex Gaussian random variables. 
    In addition, the channel adds complex proper, circularly-symmetric additive white Gaussian noise (AWGN) to the transmitted symbols, based on a specified signal-to-noise ratio (SNR) (in dB).

    Attributes
    ----------
    Nt : int
        The number of transmitting antennas.
    Nr : int
        The number of receiving antennas.
    SNR : float
        The signal-to-noise ratio (SNR) in dB.
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
        Get the current channel state information (CSI), in terms of the signal-to-noise ratio (SNR), the channel matrix (H) and its SVD (U, S, Vh).
    reset()
        Reset the channel properties.
    generate_noise()
        Generate complex white Gaussian noise (AWGN) vectors for every transmitted symbol vector, based on the current SNR.
    simulate()
        Simulate the channel operations. Return the channel output signal r.
    
    print_simulation_example()
        Print a step-by-step example of the channel operations for a given input signal s.
    """


    # INITIALIZATION AND REPRESENTATION

    def __init__(self, Nt, Nr, SNR=np.inf, H=None):
        """ Initialize the channel. """

        self.Nt = Nt
        self.Nr = Nr

        self._SNR = SNR

        self._H = H
        self._U = None
        self._S = None
        self._Vh = None
        self.set_CSI()
    
    def __str__(self):
        """ Return a string representation of the channel object. """
        return f"Channel of an {self.Nt} and {self.Nr} SU-MIMO system.\n\n  -The signal-to-noise ratio (SNR) in dB: {self._SNR}\n\n  - The channel matrix H: \n{self._H}\n\n  - The singular values S: \n{self._S}\n\n"

    def __call__(self, s):
        """ Allow the channel object to be called as a function. When called, it executes the simulate() method. """
        return self.simulate(s)

    
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
        """
        Description
        -----------
        Get the current channel state information (CSI), in terms of the SNR, the channel matrix H and its SVD (U, S, Vh).

        Returns
        -------
        CSI : dict
            The current channel state information (CSI).
            - SNR : the signal-to-noise ratio (SNR) in dB. (float)
            - H : the MIMO channel matrix. (2D numpy array, dtype: complex, shape: (Nr, Nt))
            - U : the left singular vectors of the channel matrix H. (2D numpy array, dtype: complex, shape: (Nr, Nr))
            - S : the singular values of the channel matrix H. (1D numpy array, dtype: float, shape: (Rank(H),))
            - Vh : the right singular vectors of the channel matrix H. (2D numpy array, dtype: complex, shape: (Nt, Nt))
        """
        
        CSI = {'SNR': self._SNR, 'H': self._H, 'U': self._U, 'S': self._S, 'Vh': self._Vh}
        return CSI

    def reset(self, SNR=np.inf, H=None):
        """
        Description
        -----------
        Reset the channel properties by re-initializing the signal-to-noise ratio (SNR), the MIMO channel matrix and its SVD.

        Parameters
        ----------
        SNR : float, optional
            The signal-to-noise ratio (SNR) in dB.
        H : 2D numpy array (dtype: complex, shape=(Nr, Nt)), optional
            The channel matrix. If not provided, the channel matrix is initialized with i.i.d. complex Gaussian (zero mean and unit variance) random variables.
        """

        self._SNR = SNR
        
        self._H = H
        self._U = None
        self._S = None
        self._Vh = None
        self.set_CSI()

    def generate_noise(self, s):
        """
        Description
        -----------
        Generate complex proper, circularly-symmetric additive white Gaussian noise (AWGN) vectors (w[k]) for every transmitted symbol vector, based on the current SNR of the channel.

        Parameters
        ----------
        s : 2D numpy array (dtype: complex, shape=(Nt, N_symbols))
            The input signal.
        
        Returns
        -------
        noise : 2D numpy array (dtype: complex, shape=(Nr, N_symbols))
            The generated noise vectors. Shape: (Nr, N_symbols)
        """
        
        # 1. Compute the noise power based on the current SNR and the signal power of s.
        P_signal = np.mean( np.sum( np.abs(s)**2, axis=0 ) )
        P_noise = P_signal / (10**(self._SNR/10))
        sigma = np.sqrt(P_noise / 2)

        # 2. Generate complex proper, circularly-symmetric AWGN noise vectors with the computed noise power.
        noise = sigma * (np.random.randn(self.Nr, s.shape[1]) + 1j * np.random.randn(self.Nr, s.shape[1]))

        # 3. Return.
        return noise

    def simulate(self, s):
        """
        Description
        -----------
        Simulate the channel operations:\n
        (1) Transmit the precoded symbols through the MIMO channel. This is modeled as a matrix multiplication between the channel matrix H and the precoded symbols.\n
        (2) Add complex proper, circularly-symmetric additive white Gaussian noise (AWGN) to the received symbols, based on the current SNR of the channel.\n

        Parameters
        ----------
        s : 2D numpy array (dtype: complex, shape=(Nt, N_symbols))
            The input signal.
        
        Returns
        -------
        r : 2D numpy array (dtype: complex, shape=(Nr, N_symbols))
            The output signal.
        """

        # 1. Transmit the precoded symbols through the MIMO channel.
        r = self._H @ s

        # 2. Add complex proper, circularly-symmetric additive white Gaussian noise (AWGN) to the received symbols, based on the current SNR of the channel.
        w = self.generate_noise(s)
        r = r + w

        # 3. Return.
        return r


    # TESTS AND PLOTS

    def print_simulation_example(self, s, K=1):
        """
        Description
        -----------
        Print a step-by-step example of the channel operations (see simulate() method) for a given input signal s. Only the first K symbol vectors are considered for illustration.

        Parameters
        ----------
        s : 2D numpy array (dtype: complex, shape=(Nt, N_symbols))
            The input signal.
        K : int, optional
            The maximum number of symbol vectors to consider for illustration.
        
        Return
        ------
        r : 2D numpy array (dtype: complex, shape=(Nr, N_symbols))
            The output signal.

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
        w = self.generate_noise(s, self._SNR)
        r = r + w
        SNR_calculated = 10 * np.log10( np.mean( np.sum( np.abs(self._H @ s)**2, axis=0 ) ) / np.mean( np.sum( np.abs(w)**2, axis=0 ) ) )
        print(f"----- the received symbol vector sequence after adding noise -----\n{np.round(r, 2)}\nSNR [calculated]: {np.round(SNR_calculated, 2)} dB\n\n")

        print("======== End Channel Simulation Example ========\n\n")

        # RETURN
        return r


if __name__ == "__main__":

    # Initialize the channel.
    channel = Channel(Nt=5, Nr=4, SNR=15)
    
    # Initialize the transmitted signal s.
    import transmitter as tx
    transmitter = tx.Transmitter(Nt=5, c_type='QAM')
    s = transmitter.simulate(bits=np.random.randint(0, 2, size=2400), CSI=channel.get_CSI())

    # Channel simulation example.
    channel.print_simulation_example(s=s, K=2)