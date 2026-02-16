# src/mu_mimo_sim/processing/combiners.py

from __future__ import annotations
import abc

import numpy as np
Array = np.ndarray


class Combiner(abc.ABC):
    """
    Abstract base class for the combiners.

    A combiner combines the received signal at the UT into the Ns receive streams.
    """

    @property
    def name(self) -> str:
        return self.__class__.__name__
    
    def __init__(self):
        """"
        Create a combiner.
        """
        self.W_k: Array | None = None

    def set_combining_matrix(self, W_k: Array) -> None:
        """
        Update the combining matrix W_k of the combiner.

        Parameters
        ----------
        W_k : Array, shape (Ns, Nr), dtype complex
            Combining matrix for user terminal k.
        """
        self.W_k = W_k
    
    def get_combining_matrix(self) -> Array:
        """
        Get the current combining matrix W_k of the combiner.

        Returns
        -------
        W_k : Array, shape (Ns, Nr), dtype complex
            Current combining matrix for user terminal k.
        
        Raises
        ------
        ValueError
            If the combining matrix W_k has not been set yet.
        """
        if self.W_k is None: raise ValueError("Combining matrix W_k has not been set yet.")
        return self.W_k

    @abc.abstractmethod
    def configure(self, *args, **kwargs) -> None:
        """
        Configure the combiner for a new channel realization.

        The configuration consists of computing or retreiving the combining matrix W_k for this UT and updating the state of the combiner with the new combining matrix W_k.

        Parameters
        ----------
        *args, **kwargs
            Arguments needed to compute or retreive the combining matrix W_k for this UT.
        
        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.
        """
        raise NotImplementedError("The configure method must be implemented in a subclass of Combiner.")

    def execute(self, y_k: Array) -> Array:
        """
        Combine the received signal y_k at the UT into the Ns receive streams.

        Parameters
        ----------
        y_k : Array, shape (Nr,), dtype complex
            Received signal at user terminal k.

        Returns
        -------
        z_k : Array, shape (Ns,), dtype complex
            Scaled decision variables for user terminal k.
        
        Raises
        ------
        ValueError
            If the combining matrix W_k has not been set yet.
        """
        W_k = self.get_combining_matrix()
        z_k = W_k @ y_k
        return z_k


class RSVCombiner(Combiner):
    """
    Non-coordinated Right-Singular-Vector (RSV) combiner.
    """

    def configure(self, H_k: Array, Ns: int) -> Array:
        """
        Configure the RSV combiner for a given channel realization.

        The rows of the combining matrix W_k are the Ns most dominant singular vectors of the channel matrix H_k.

        Parameters
        ----------
        H_k : Array, shape (Nr, Nt), dtype complex
            Channel matrix for user terminal k.
        Ns : int
            Number of receive streams.

        Returns
        -------
        W_k : Array, shape (Ns, Nr), dtype complex
            RSV combining matrix for user terminal k.
        """

        U_k, _, _ = np.linalg.svd(H_k)
        W_k = U_k.conj().T[ : Ns]

        self.set_combining_matrix(W_k)


class CoordinatedCombiner(Combiner):
    """
    Coordinated combiner.
    """

    def configure(self, W_k: Array) -> None:
        """
        Configure the coordinated combiner with the provided combining matrix W_k.

        The combining matrix W_k is computed by the BS (coordinator) and provided to the UT.

        Parameters
        ----------
        W_k : Array, shape (Ns, Nr), dtype complex
            Combining matrix for user terminal k.
        """
        self.set_combining_matrix(W_k)