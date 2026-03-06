# This module contains the implementation of the channel component of a SU-MIMO SVD communication system.

import numpy as np


class Channel:
    """
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
    """


    # INITIALIZATION AND REPRESENTATION

    def __init__(self, Nt, Nr, SNR=None, H=None):
        """ 
        Initialize the channel.

        Parameters
        ----------
        Nt : int
            The number of transmitting antennas.
        Nr : int
            The number of receiving antennas.
        SNR : float, optional
            The signal-to-noise ratio (SNR) in dB. Default is infinity (no noise).
        H : 2D numpy array (dtype: complex, shape: (Nr, Nt)), optional
            The channel matrix. Default is a random i.i.d. complex circularly-symmetric Gaussian (zero mean, unit variance) matrix.
        """

        # System parameters
        self.Nt = Nt
        self.Nr = Nr

        # Channel parameters.
        self._SNR = None
        self._H = None
        self._U = None
        self._S = None
        self._Vh = None
        self.reset_CSI(SNR, H)
    
    def __str__(self):
        """ Return a string representation of the channel object. """
        return f"Channel:\n  - The signal-to-noise ratio (SNR) in dB: {self._SNR}\n  - The singular values S: {self._S}\n  - The channel matrix H: \n{self._H}\n\n"

    def __call__(self, x):
        """ Allow the channel object to be called as a function. When called, it executes the simulate() method. """
        return self.simulate(x)

    
    # FUNCTIONALITY

    def get_CSI(self):
        """
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

    def set_CSI(self, SNR=None, H=None):
        """
        Set the SNR value and the channel matrix. Compute the singular-value-decomposition (SVD) of the channel matrix and store it.\n
        If no new value is provided, the old value is left unchanged.

        Parameters
        ----------
        SNR : float, optional
            The signal-to-noise ratio (SNR) in dB.
        H : 2D numpy array (dtype: complex, shape: (Nr, Nt)), optional
            The channel matrix.
        """

        if SNR is not None: 
            self._SNR = SNR
        
        if H is not None:
            self._H = H
            self._U, self._S, self._Vh = np.linalg.svd(self._H)
    
    def reset_CSI(self, SNR=None, H=None):
        """
        Reset the SNR value and the channel matrix. Compute the singular-value-decomposition (SVD) of the channel matrix and store it.\n
        If no new value is provided, the default initialization values are used. For the SNR value, the default is infinity (no noise). For the channel matrix, the default is a random i.i.d. complex circularly-symmetric Gaussian (zero mean, unit variance) MIMO channel.

        Parameters
        ----------
        SNR : float, optional
            The signal-to-noise ratio (SNR) in dB.
        H : 2D numpy array (dtype: complex, shape: (Nr, Nt)), optional
            The channel matrix.
        """
        
        if SNR is None: SNR = np.inf
        if H is None: H = (np.random.randn(self.Nr, self.Nt) + 1j * np.random.randn(self.Nr, self.Nt)) / np.sqrt(2)
        
        self.set_CSI(SNR, H)    

    def generate_noise(self, x):
        """
        Generate complex proper, circularly-symmetric additive white Gaussian noise (AWGN) vectors (w[k]) for every transmitted symbol vector, based on the current SNR of the channel.

        Parameters
        ----------
        x : 2D numpy array (dtype: complex, shape=(Nt, N_symbols))
            The input signal.
        
        Returns
        -------
        noise : 2D numpy array (dtype: complex, shape=(Nr, N_symbols))
            The generated noise vectors. Shape: (Nr, N_symbols)
        """
        
        # 1. Compute the noise power based on the current SNR and the signal power of x.
        P_signal = np.mean( np.sum( np.abs(x)**2, axis=0 ) )
        P_noise = P_signal / (10**(self._SNR/10))
        sigma = np.sqrt(P_noise / 2)

        # 2. Generate complex proper, circularly-symmetric AWGN noise vectors with the computed noise power.
        noise = sigma * (np.random.randn(self.Nr, x.shape[1]) + 1j * np.random.randn(self.Nr, x.shape[1]))

        # 3. Return.
        return noise

    def simulate(self, x):
        """
        Simulate the channel operations:\n
        (1) Transmit the precoded symbols through the MIMO channel. This is modeled as a matrix multiplication between the channel matrix H and the precoded symbols.\n
        (2) Add complex proper, circularly-symmetric additive white Gaussian noise (AWGN) to the received symbols, based on the current SNR of the channel.\n

        Parameters
        ----------
        x : 2D numpy array (dtype: complex, shape=(Nt, N_symbols))
            The input signal.
        
        Returns
        -------
        y : 2D numpy array (dtype: complex, shape=(Nr, N_symbols))
            The output signal.
        """

        # 1. Transmit the precoded symbols through the MIMO channel.
        y = self._H @ x

        # 2. Add complex proper, circularly-symmetric additive white Gaussian noise (AWGN) to the received symbols, based on the current SNR of the channel.
        w = self.generate_noise(x)
        y = y + w

        # 3. Return.
        return y


    # TESTS AND PLOTS

    def print_simulation_example(self, x, K=1):
        """
        Print a step-by-step example of the channel operations (see simulate() method) for a given input signal x. Only the first K symbol vectors are considered for illustration.

        Parameters
        ----------
        x : 2D numpy array (dtype: complex, shape=(Nt, N_symbols))
            The input signal.
        K : int, optional
            The maximum number of symbol vectors to consider for illustration.
        
        Returns
        -------
        y : 2D numpy array (dtype: complex, shape=(Nr, N_symbols))
            The output signal.

        Notes
        -----
        For demonstration purposes only.
        """

        # PRINTING EXAMPLE
        print("\n\n========== Channel Simulation Example ==========\n")

        # 0. Print the input signal (precoded data symbol vector).
        print(f"----- the input signal -----\n{np.round(x, 2)}\n\n")

        # 1. Transmit the precoded symbols through the MIMO channel.
        y = self._H @ x
        print(f"----- the output signal before adding noise -----\n{np.round(y, 2)}\n\n")

        # 2. Add complex proper, circularly-symmetric additive white Gaussian noise (AWGN) to the received symbols, based on the specified SNR.
        w = self.generate_noise(x)
        y = y + w
        SNR_calculated = 10 * np.log10( np.mean( np.sum( np.abs(self._H @ x)**2, axis=0 ) ) / np.mean( np.sum( np.abs(w)**2, axis=0 ) ) )
        print(f"----- the output signal -----\n{np.round(y, 2)}\nSNR [calculated]: {np.round(SNR_calculated, 2)} dB\n\n")

        print("======== End Channel Simulation Example ========\n\n")

        # RETURN
        return y
