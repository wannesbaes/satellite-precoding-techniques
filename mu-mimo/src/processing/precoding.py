# src/mu_mimo_sim/processing/precoding.py

from __future__ import annotations
import abc

import numpy as np
Array = np.ndarray


class Precoder(abc.ABC):
    """
    Abstract base class for the precoders.

    A precoder converts the data symbols into the transmit signal at the BS, resulting in K*Ns transmit streams.
    """

    @property
    def name(self) -> str:
        return self.__class__.__name__
    
    def __init__(self):
        """
        Create a precoder.
        """
        self.F: Array | None = None

    def set_precoding_matrix(self, F: Array) -> None:
        """
        Update the compound precoding matrix F.

        Parameters
        ----------
        F : Array, shape (Nt, K*Ns), dtype complex
            Compound precoding matrix F.
        """
        self.F = F
    
    def get_precoding_matrix(self) -> Array:
        """
        Get the current compound precoding matrix F.

        Returns
        -------
        F : Array, shape (Nt, K*Ns), dtype complex
            Current compound precoding matrix.
        
        Raises
        ------
        ValueError
            If the compound precoding matrix F has not been set yet.
        """
        if self.F is None: raise ValueError("Compound precoding matrix F has not been set yet.")
        return self.F

    @abc.abstractmethod
    def configure(self, *args, **kwargs) -> None:
        """
        Configure the precoder for a new channel realization.

        The configuration consists of computing the compound precoding matrix F and updating the state of the precoder with the new compound precoding matrix F.
        In case of coordinated beamforming, the configuration also consists of computing the compound combining matrix W and updating the state of the precoder with the new compound combining matrix W.
        
        Parameters
        ----------
        *args, **kwargs
            The parameters needed to compute the compound precoding matrix F depend on the specific precoding algorithm implemented in the subclass.
        """
        raise NotImplementedError("The configure method must be implemented in a subclass.")
    
    def execute(self, a: Array) -> Array:
        """
        Execute the precoding operation on the data symbols a to obtain the precoded transmit signal x: x = F @ a.

        Parameters
        ----------
        a : Array, shape (K*Ns,), dtype complex
            Input data symbols a.
        
        Returns
        -------
        x : Array, shape (Nt,), dtype complex
            Output transmit signal x.
        """
        F = self.get_precoding_matrix()
        x = F @ a
        return x


class ZFPrecoder(Precoder):
    """
    Zero-Forcing (ZF) precoder.
    """
    
    def configure(self, H_BS: Array) -> None:
        """
        Configure the ZF precoder for a given channel realization H_BS.

        The precoder is obtained by computing the pseudoinverse of the effective channel H_BS seen by the BS (i.e., the compound channel followed by the compound combiner): F = (W H)† = H_BS†

        Parameters
        ----------
        H_BS : Array, shape (K*Nr, Nt), dtype complex
            Effective channel matrix H_BS seen by the BS.
        """
        F = np.linalg.pinv(H_BS)
        self.set_precoding_matrix(F)


class WMMSEPrecoder(Precoder):
    """
    Coordinated Weighted Minimum Mean Square Error (WMMSE) precoder.
    """

    def __init__(self):
        """
        Create a WMMSE precoder.
        """
        super().__init__()
        self.W: Array | None = None
    
    def set_combining_matrix(self, W: Array) -> None:
        """
        Update the compound combining matrix W.

        Parameters
        ----------
        W : Array, shape (K*Ns, K*Nr), dtype complex
            Compound combining matrix W.
        """
        self.W = W
    
    def get_combining_matrix(self) -> Array:
        """
        Get the current compound combining matrix W.

        Returns
        -------
        W : Array, shape (K*Ns, K*Nr), dtype complex
            Current compound combining matrix.
        
        Raises
        ------
        ValueError
            If the compound combining matrix W has not been set yet.
        """
        if self.W is None: raise ValueError("Compound combining matrix W has not been set yet.")
        return self.W

    def configure(self, H: Array) -> None:
        raise NotImplementedError