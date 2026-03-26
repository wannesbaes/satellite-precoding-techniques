# mu-mimo/mu_mimo/processing/combining.py

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from ..types import ComplexArray, IntArray


class Combiner(ABC):
    """
    The Combiner Abstract Base Class (ABC).

    This class is responsible for implementing a combining strategy and effectively combining the received signals in each user terminal.
    """

    @staticmethod 
    @abstractmethod
    def compute(H_k: ComplexArray) -> ComplexArray:
        """
        Compute the combining matrix for a given user terminal.

        In case of coordinated beamforming, the combiner matrix is computed in the BS and sent to the user terminal. Then, the identity matrix is returned by this method. 

        Parameters
        ----------
        H_k : ComplexArray, shape (Nr, Nt)
            The channel matrix for the this UT.

        Returns
        -------
        G_k : ComplexArray, shape (Nr, Nr)
            The computed combining matrix for the this UT.
        """
        raise NotImplementedError

    @staticmethod
    def apply(y_k: ComplexArray, G_k: ComplexArray,  ibr_k: IntArray) -> ComplexArray:
        """
        Apply the combining matrix to the received signal.

        Parameters
        ----------
        y_k : ComplexArray, shape (Nr, M)
            The received signal for the this UT.
        G_k : ComplexArray, shape (Nr, Nr)
            The combining matrix for the this UT.
        ibr_k : IntArray, shape (Nr,)
            The information bit rate for each data stream of the this UT.

        Returns
        -------
        z_k : ComplexArray, shape (Ns_k, M)
            The combined signal for the this UT.
        """
        z_k = G_k[ibr_k > 0, :] @ y_k
        return z_k

class NeutralCombiner(Combiner):
    """
    Neutral Combiner.

    This combiner acts as a 'neutral element' for combining.\\
    It does not perform any combining and simply passes the received signal through without any modification. 
    """

    @staticmethod
    def compute(H_k: ComplexArray) -> ComplexArray:
        Nr = H_k.shape[0]
        G_k = np.eye(Nr, Nr)
        return G_k

class LSVCombiner(Combiner):
    """
    Left Singular Vector (LSV) Combiner.

    The rows of the combining matrix are computed as the conjugate-transposed left singular vectors of the channel matrix H_k. The order of the singular vectors is determined by the descending order of the corresponding singular values. In that way, the strongest singular modes of the channel are used when effectively applying the combining matrix to the Ns_k data streams.
    """
    
    @staticmethod
    def compute(H_k: ComplexArray) -> ComplexArray:
        U_k, _, _ = np.linalg.svd(H_k)
        G_k = U_k.conj().T
        return G_k

