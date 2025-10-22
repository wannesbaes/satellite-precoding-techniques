# This module contains the implementation of the channel component of a SU-MIMO SVD communication system.

import numpy as np
import matplotlib.pyplot as plt
import math


class Channel:
    """
    Description
    -----------
    The channel of a single-user multiple-input multiple-output (SU-MIMO) communication system. 

    The channel is modeled as a flat-fading MIMO channel. The channel matrix can be either provided or initialized with independent and identically distributed (i.i.d.) complex Gaussian random variables. 
    In addition, the channel adds complex proper, circularly-symmetric additive white Gaussian noise (AWGN) to the transmitted symbols, based on a specified signal-to-noise ratio (SNR in dB).
    """


    # INITIALIZATION AND REPRESENTATION

    def __init__(self, Nt: int, Nr: int, H=None) -> None:
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
        return f"Channel: \n  - Number of transmitting and receiving antennas is {self.Nt} and {self.Nr}\n  - SNR = {self.SNR} dB\n  - H = {'Provided' if self._H is not None else 'i.i.d. complex Gaussian (0 mean, 1 variance) variables.'}"
    
    def __call__(self):
        """ Allow the channel object to be called as a function. When called, it executes the simulate() method. """
        return self.simulate()

    
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

    def generate_noise(self, s: np.ndarray, SNR: float) -> np.ndarray:
        """
        Description
        -----------
        Generate complex proper, circularly-symmetric additive white Gaussian noise (AWGN) vectors (n[k]) for every transmitted symbol vector, based on the specified SNR.

        Parameters
        ----------
        s : np.ndarray
            The input signal. This signal consists of the precoded symbols that the transmitter sends through the channel. Shape: (Nt, N_symbols)
        SNR : float
            The signal-to-noise ratio (SNR) in dB.
        
        Returns
        -------
        w : np.ndarray
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

    def plot_before_after_noise(self, s: np.ndarray, SNR: float, K: int = 1) -> tuple:
        """
        Description
        -----------
        Plot the first K received symbol vectors before and after adding noise for a given SNR and input signal s. 
        
        The symbols before adding noise are half-transparent, while the symbols after adding noise are fully opaque.
        Every color represents a different receive antenna.

        Parameters
        ----------
        s : np.ndarray
            The input signal. This signal consists of the precoded symbols that the transmitter sends through the channel. Shape: (Nt, N_symbols)
        SNR : float
            The signal-to-noise ratio (SNR) in dB.
        K : int, optional
            The maximum number of symbol vectors to consider for illustration.

        Returns
        -------
        r_before_noise : np.ndarray
            The received symbol vectors before adding noise. Shape: (Nr, N_symbols)
        r_after_noise : np.ndarray
            The received symbol vectors after adding noise. Shape: (Nr, N_symbols)
        """
        
        r_before_noise = self._H @ s
        r_after_noise = r_before_noise + self.generate_noise(s, SNR)
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        fig, ax = plt.subplots(figsize=(6, 6))
        for rx_antenna in range(self.Nr):
            ax.scatter(r_before_noise[rx_antenna, :K].real, r_before_noise[rx_antenna, :K].imag, color=colors[rx_antenna%len(colors)], alpha=0.5, s=100)
            ax.scatter(r_after_noise[rx_antenna, :K].real, r_after_noise[rx_antenna, :K].imag, color=colors[rx_antenna%len(colors)], alpha=1.0, s=100, label=f'Receive Antenna {rx_antenna+1}')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.set_title(f'Received Symbols Before (Transparent) and After (Opaque) Noise (SNR = {SNR} dB)')
        ax.set_xlabel('Real Part')
        ax.set_ylabel('Imaginary Part')
        ax.grid()
        ax.axis('equal')
        ax.legend()
        fig.tight_layout()
                
        return fig, ax

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

        Notes
        -----
        For demonstration purposes only.
        """

        # PRINTING EXAMPLE
        print("\n\n========== Channel Simulation Example ==========\n")

        # 0. Print the input symbol vector sequence.
        print(f"----- the input symbol vector sequence -----\n{s}\n\n")

        # 1. Transmit the precoded symbols through the MIMO channel.
        r = self._H @ s
        print(f"----- the received symbol vector sequence before adding noise -----\n{np.round(r, 2)}\n\n")

        # 2. Add complex proper, circularly-symmetric additive white Gaussian noise (AWGN) to the received symbols, based on the specified SNR.
        w = self.generate_noise(s, SNR)
        r = r + w
        SNR_calculated = 10 * np.log10( np.mean( np.sum( np.abs(self._H @ s)**2, axis=0 ) ) / np.mean( np.sum( np.abs(w)**2, axis=0 ) ) )
        print(f"----- the received symbol vector sequence after adding noise -----\n{np.round(r, 2)}\nSNR [calculated]: {np.round(SNR_calculated, 2)} dB\n\n")

        print("======== End of Simulation Example ========\n\n")


        # PLOTS
        fig, ax = self.plot_before_after_noise(s=s, SNR=SNR, K=K)
        plt.show()


        # RETURN
        return



if __name__ == "__main__":

    # Initialize the channel.
    channel = Channel(Nt=5, Nr=4)
    
    # Initialize the transmitted signal s.
    import transmitter as tx
    transmitter = tx.Transmitter(Nt=5, constellation_type='PSK', Pt=1.0, B=0.5)
    s = transmitter.simulate(bits=np.random.randint(0, 2, size=100), SNR=15, CSI=channel.get_CSI())

    # Channel simulation example.
    channel.print_simulation_example(s=s, SNR=15, K=2)