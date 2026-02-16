# mu_mimo/src/core/system.py

from __future__ import annotations
import abc

import numpy as np
Array = np.ndarray

from ..processing.combining import Combiner
from ..processing.precoding import Precoder
from src.processing.resource_allocation import PowerAllocator, PowerDeallocator, BitAllocator, BitDeallocator
from src.processing.modulation import Mapper, Demapper
from src.processing.detection import Detector
from src.processing.channel import ChannelModel, NoiseModel


class UserTerminal:
    """ 
    Represents a user terminal (UT) in the MU-MIMO downlink system. 
    """

    def __init__(self, id: int, Nr: int, Ns: int, combiner: Combiner, power_deallocator: PowerDeallocator, detector: Detector, demapper: Demapper, bit_deallocator: BitDeallocator):

        self.id = id
        self.Nr = Nr
        self.Ns = Ns

        self.power_deallocator = power_deallocator
        self.combiner = combiner
        self.detector = detector
        self.demapper = demapper
        self.bit_deallocator = bit_deallocator
    
    def configure(self, P_k: Array, ibr_k: Array, W_k: Array | None = None, H_k: Array | None = None) -> None:
        """
        Configure the UT for a new channel realization.

        The configuration consists of the following steps:
        1. The combiner computes the combining matrix W_k for this UT in case of non-coordinated beamforming, or retreives the combining matrix W_k for this UT in case of coordinated beamforming, and updates its state with the new combining matrix W_k.
        2. The power deallocator retreives the power allocation for each stream and updates its state.
        3. The bit deallocator retreives the information bit rate for each stream and updates its state.

        Parameters
        ----------
        P_k : Array, shape (Ns,), dtype float
            Power allocation for each stream of this UT.
        ibr_k : Array, shape (Ns,), dtype int
            Information bit rate for each stream of this UT.
        W_k : Array, shape (Ns, Nr), dtype complex, optional
            Combining matrix for this UT. 
            In case of non-coordinated beamforming, the combining matrix is computed by the UT itself and thus not provided by the BS (coordinator).
        H_k : Array, shape (Nr, Nt), dtype complex, optional
            Channel matrix for this UT. 
            In case of non-coordinated beamforming, the computation of the combining matrix by the UT may depend on the channel matrix.
        
        Raises
        ------
        ValueError
            If both W_k and H_k are provided, or if neither W_k nor H_k is provided. In this case, confusion arises about how the combining matrix W_k is computed and provided to the UT.
        """
        
        if W_k is not None and H_k is None:
            self.combiner.configure(W_k)
        elif H_k is not None and W_k is None:
            self.combiner.configure(H_k, self.Ns)
        else:
            raise ValueError("Either W_k or H_k must be provided to define the combining matrix.")
        
        self.power_deallocator.configure(P_k)

        self.bit_deallocator.configure(ibr_k)

    def receive(self, y_k: Array) -> Array:
        """
        Execute the receive processing chain of the UT on the received signal y_k to obtain the estimated bitstream b_hat_k. 
        
        The processing chain consists of the following steps:
        1. Combining - apply the combining matrix W_k to the received signal y_k to obtain the scaled decision variables z_k
        2. Power deallocation - invert the power allocation applied by the base station to obtain the decision variables u_k
        3. Detection - estimate the transmitted symbol vectors a_k based on the decision variables u_k
        4. Demapping - convert the estimated symbol vectors a_hat_k to the corresponding estimated bit vectors b_hat_k
        5. Bit deallocation - reconstruct the estimated bitstream from the estimated bit vectors b_hat_k

        Parameters
        ----------
        y_k : Array, shape (Nr, M), dtype complex
            Received signal at the UT.
        
        Returns
        -------
        bitstream_hat_k : Array, shape (L,), dtype int
            Estimated output bitstream.
        """
        
        z_k = self.combiner.execute(y_k)
        u_k = self.power_deallocator.execute(z_k)
        a_hat_k = self.detector.execute(u_k)
        b_hat_k = self.demapper.execute(a_hat_k)
        bitstream_hat_k = self.bit_deallocator.execute(b_hat_k)
        
        return bitstream_hat_k


class BaseStation:
    """
    Represents the base station (BS) in the MU-MIMO downlink system.
    """

    def __init__(self,  Nt: int, Ns: int, bit_allocator: BitAllocator, mapper: Mapper, power_allocator: PowerAllocator, precoder: Precoder):

        self.Nt = Nt
        self.Ns = Ns

        self.bit_allocator = bit_allocator
        self.mapper = mapper
        self.power_allocator = power_allocator
        self.precoder = precoder

    def configure(self) -> None:
        """
        Configure the BS for a new channel realization.

        The configuration consists of the following steps:
        1. The power allocator computes the power allocation (P) for each stream and updates its state.
        2. The bit allocator computes the bit allocation (ibr) for each stream and updates its state.
        3. The precoder computes the compound precoding matrix (F) for each stream, as well as the compound combining matrix (W) in case of coordinated beamforming, and updates its state.

        Parameters
        ----------
        Not implemented yet.
        """

        self.power_allocator.configure()
        self.bit_allocator.configure()
        self.mapper.configure()
        self.precoder.configure()

    def transmit(self, bitstream: Array) -> Array:
        """
        Execute the transmit processing chain of the BS on the input bitstream to obtain the transmitted signal x.

        The processing chain consists of the following steps:
        1. Bit allocation - distribute the input bitstream over the K users and Ns streams per user
        2. Mapping - convert the bit vectors b to the corresponding data symbol vectors a
        3. Power allocation - allocate power to the symbol vectors a to obtain the power-scaled symbol vectors a_p
        4. Precoding - apply the precoding matrix F to the power-scaled symbol vectors a_p to obtain the transmitted signal x

        Parameters
        ----------
        bitstream : Array, shape (L,), dtype int
            Input bitstream.

        Returns
        -------
        x : Array, shape (Nt, M), dtype complex
            Transmitted signal.
        """

        b = self.bit_allocator.execute(bitstream)
        a = self.mapper.execute(b)
        a_p = self.power_allocator.execute(a)
        x = self.precoder.execute(a_p)

        return x


class Channel:
    """
    Represents the wireless channel in the MU-MIMO downlink system.
    """

    def __init__(self, K: int, Nr: int, Nt: int, channel_model: ChannelModel, noise_model: NoiseModel):

        self.K = K
        self.Nr = Nr
        self.Nt = Nt

        self.channel_model = channel_model
        self.noise_model = noise_model

        self.H: Array | None = None
    
    def configure(self) -> None:
        """
        Generate a new channel realization, according to the specified channel model.
        """
        self.H = self.channel_model.generate(self.K, self.Nr, self.Nt)

    def propagate(self, x: Array) -> Array:
        """
        Execute the channel propagation of the transmitted signal x to obtain the received signal y.

        The channel propagation consists of the following steps:
        1. Apply the channel matrix H to the transmitted signal x to obtain the noiseless received signal y_noiseless.
        2. Generate the noise samples according to the specified noise.
        3. Add the noise to the noiseless received signal y_noiseless to obtain the actual received signal y.

        Parameters
        ----------
        x : Array, shape (Nt, M), dtype complex
            Transmitted signal.
        
        Returns
        -------
        y : Array, shape (K*Nr, M), dtype complex
            Received signal at the UTs.
        """

        y_noiseless = self.H @ x
        noise = self.noise_model.generate(self.K, self.Nr, x.shape[1])
        y = y_noiseless + noise

        return y

