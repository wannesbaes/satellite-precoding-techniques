# mu-mimo/src/processing/precoding.py

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from ..types import (ComplexArray, RealArray, IntArray, BitArray)



class Precoder(ABC):

    @abstractmethod
    def compute(self, H_eff: ComplexArray, K: int, Ns: int, Nt: int) -> tuple[ComplexArray, ComplexArray | None]:
        raise NotImplementedError

    def execute(self, F: ComplexArray, a_p: ComplexArray) -> ComplexArray:
        x = F @ a_p
        return x



class NeutralPrecoder(Precoder):

    def compute(self, H_eff: ComplexArray, K: int, Ns: int, Nt: int) -> tuple[ComplexArray, ComplexArray | None]:
        F = np.eye(Nt, K * Ns)
        return F, None


class ZFPrecoder(Precoder):
    pass


class WMMSEPrecoder(Precoder):
    pass
