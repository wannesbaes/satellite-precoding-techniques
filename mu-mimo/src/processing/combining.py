# mu-mimo/src/processing/combining.py

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from ..types import (ComplexArray, RealArray, IntArray, BitArray)



class Combiner(ABC):

    @abstractmethod
    def compute(self, H_k: ComplexArray, Ns: int, Nr: int) -> ComplexArray:
        raise NotImplementedError

    def execute(self, G_k: ComplexArray, y_k: ComplexArray) -> ComplexArray:
        z_k = G_k @ y_k
        return z_k



class NeutralCombiner(Combiner):

    def compute(self, H_k: ComplexArray, Ns: int, Nr: int) -> ComplexArray:
        G_k = np.eye(Ns, Nr)
        return G_k


class RSVCombiner(Combiner):
    pass


class WMMSECombiner(Combiner):
    pass
