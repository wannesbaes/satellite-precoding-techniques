# src/mu_mimo_sim/processing/modulation.py

from __future__ import annotations
import abc

import numpy as np
Array = np.ndarray


class Mapper(abc.ABC):
    """
    Abstract base class for mappers.
    
    A mapper converts bit vectors into data symbol vectors according to a specified modulation constellation.
    """

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abc.abstractmethod
    def configure(self, *args, **kwargs) -> None:
        """
        Configure the mapper.
        """
        raise NotImplementedError("The configure method must be implemented by the subclass of Mapper.")

    @abc.abstractmethod
    def execute(self, b: list[Array] | Array) -> Array:
        """
        Convert the input bit sequences into an output data symbol sequences.

        Parameters
        ----------
        b : list[Array] | Array
            Input bit sequence(s).

        Returns
        -------
        a : Array
            Output data symbol sequence.
        """
        raise NotImplementedError("The execute method must be implemented by the subclass of Mapper.")

