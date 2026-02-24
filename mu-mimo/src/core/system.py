# mu-mimo/src/core/system.py

from __future__ import annotations
import numpy as np
from ..config import (
    RealArray, ComplexArray, IntArray, BitArray,
    ChannelStateInformation,
    TransmitPilotMessage, ReceivePilotMessage, TransmitFeedbackMessage, ReceiveFeedbackMessage, TransmitFeedforwardMessage, ReceiveFeedforwardMessage,
    SystemConfig, BaseStationConfig, UserTerminalConfig, ChannelConfig,
    SimulationConfig )
from ..processing import (
    Precoder, Combiner,
    PowerAllocator, PowerDeallocator,
    BitAllocator, BitDeallocator,
    Mapper, Demapper, Detector,
    ChannelModel, NoiseModel )



class SimulationRunner:
    pass


class MuMimoSystem:
    """
    Represents the MU-MIMO downlink system, orchestrating the base station, channel, and user terminals.
    """

    def __init__(self, system_config: SystemConfig):

        UserTerminal.reset_id_counter()

        K = system_config.K
        Nt = system_config.Nt
        Nr = system_config.Nr
        Ns = system_config.Ns

        self.bs = BaseStation(Nt, Ns, system_config.base_station_configs)
        self.channel = Channel(K, Nr, Nt, Ns, system_config.channel_configs)
        self.uts = [UserTerminal(Nr, Ns, system_config.user_terminal_configs) for _ in range(K)]

    def reset(self, csi: ChannelStateInformation) -> ChannelStateInformation:

        # Reset the base station state.
        self.bs.reset_state()

        # Reset each user terminal state.
        for ut in self.uts: ut.reset_state()

        # Reset the channel state.
        csi = self.channel.reset(csi)

        return csi

    def configure(self) -> None:

        # Pilot Phase.
        tx_pilot_msg = self.bs.transmit_pilots()
        rx_pilot_msgs = self.channel.propagate_pilots(tx_pilot_msg)
        for ut, pilot_msg in zip(self.uts, rx_pilot_msgs): ut.receive_pilots(pilot_msg)

        # Feedback Phase.
        tx_fb_msgs = [ut.transmit_feedback() for ut in self.uts]
        rx_fb_msg = self.channel.propagate_feedback(tx_fb_msgs)
        self.bs.receive_feedback(rx_fb_msg)

        # Feedforward Phase.
        tx_ff_msg = self.bs.transmit_feedforward()
        rx_ff_msgs = self.channel.propagate_feedforward(tx_ff_msg)
        for ut, ff_msg in zip(self.uts, rx_ff_msgs): ut.receive_feedforward(ff_msg)

    def communicate(self, num_symbols: int) -> tuple[list[BitArray], list[BitArray]]:

        # Transmit from the BS.
        antenna_bitstreams_list, x = self.bs.transmit(num_symbols)

        # Propagate through the channel.
        y_k_list = self.channel.propagate(x)

        # Receive at the UTs.
        antenna_bitstreams_list_hat = [ut.receive(y_k) for ut, y_k in zip(self.uts, y_k_list)]

        return antenna_bitstreams_list, antenna_bitstreams_list_hat


class BaseStation:
    """
    Represents the base station (BS) in the MU-MIMO downlink system.
    """

    def __init__(self,  Nt: int, Ns: int, configs: BaseStationConfig):

        self.Nt = Nt
        self.Ns = Ns

        # Processing Components.
        self.bit_allocator = configs.bit_allocator()
        self.mapper = configs.mapper()
        self.power_allocator = configs.power_allocator()
        self.precoder = configs.precoder()

        # State.
        self.F: ComplexArray | None = None
        self.P: RealArray | None = None
        self.ibr: IntArray | None = None
        self.G: ComplexArray | None = None

    def reset_state(self) -> None:
        
        self.F = None
        self.P = None
        self.ibr = None
        self.G = None

        return


    def transmit_pilots(self) -> TransmitPilotMessage:

        pilot_msg = TransmitPilotMessage()
        return pilot_msg

    def receive_feedback(self, rx_fb_msg: ReceiveFeedbackMessage) -> None:

        # Retrieve the effective channel matrix and the snr from the received feedback message.
        H_eff = rx_fb_msg.H_eff
        snr = rx_fb_msg.snr

        # Compute the precoder matrix, the power allocation, and the information bit rate for the current channel realization and SNR. In case of coordinated beamforming, also compute the combining matrices.
        F, G = self.precoder.compute(H_eff)
        P = self.power_allocator.compute(H_eff, snr)
        ibr = self.bit_allocator.compute(H_eff, P)

        # Update the state of the BS to the current channel realization and SNR.
        self.F = F
        self.P = P
        self.ibr = ibr
        self.G = G

        return

    def transmit_feedforward(self) -> TransmitFeedforwardMessage:

        tx_ff_msg = TransmitFeedforwardMessage(P=self.P, ibr=self.ibr, G=self.G)
        return tx_ff_msg


    def transmit(self, num_symbols: int) -> ComplexArray:

        b = self.bit_allocator.execute(self.ibr, num_symbols)
        a = self.mapper.execute(self.ibr, b)
        a_p = self.power_allocator.execute(self.P, a)
        x = self.precoder.execute(self.F, a_p)

        return x


class Channel:
    """
    Represents the wireless channel in the MU-MIMO downlink system.
    """

    def __init__(self, K: int, Nr: int, Nt: int, Ns:int, configs: ChannelConfig):

        self.K = K
        self.Nr = Nr
        self.Nt = Nt
        self.Ns = Ns

        # Channel model and noise model.
        self.channel_model = configs.channel_model()
        self.noise_model = configs.noise_model()

        # State.
        self.snr: float | None = None
        self.H: ComplexArray | None = None

    def set(self, csi: ChannelStateInformation) -> None:
       
        if csi.snr is not None: self.snr = csi.snr
        if csi.H is not None: self.H = csi.H

        return

    def reset(self, csi: ChannelStateInformation) -> ChannelStateInformation:
        
        if csi.snr is None: csi.snr = np.inf
        if csi.H is None: csi.H = self.channel_model.generate(self.K, self.Nr, self.Nt)
        
        self.set(csi)

        return csi


    def propagate_pilots(self, tx_pilot_msg: TransmitPilotMessage) -> list[ReceivePilotMessage]:

        rx_pilot_msgs = [ReceivePilotMessage(H_k = self.H[k*self.Nr:(k+1)*self.Nr, :]) for k in range(self.K)]
        return rx_pilot_msgs

    def propagate_feedback(self, tx_fb_msgs: list[TransmitFeedbackMessage]) -> ReceiveFeedbackMessage:
        
        # Aggregate the effective channel matrices from the received feedback messages.
        H_eff = np.empty((self.K*self.Ns, self.Nt), dtype=complex)
        for tx_fb_msg_k in tx_fb_msgs:
            H_eff[tx_fb_msg_k.ut_id*self.Ns : (tx_fb_msg_k.ut_id+1)*self.Ns, : ] = tx_fb_msg_k.H_eff_k
        
        # Generate the feedback message that will be received by the BS.
        rx_fb_msg = ReceiveFeedbackMessage(snr=self.snr, H_eff=H_eff)

        return rx_fb_msg

    def propagate_feedforward(self, tx_ff_msg: TransmitFeedforwardMessage) -> list[ReceiveFeedforwardMessage]:
        
        # Retrieve the transmitted feedforward message elements.
        P = tx_ff_msg.P
        ibr = tx_ff_msg.ibr
        G = tx_ff_msg.G

        # Split the transmitted feedforward message elemtens into K different elements for each UT.
        P_k_list = [ P[k*self.Ns : (k+1)*self.Ns] for k in range(self.K)]
        ibr_k_list = [ ibr[k*self.Ns : (k+1)*self.Ns] for k in range(self.K)]
        G_k_list = [ G[k*self.Ns : (k+1)*self.Ns, k*self.Nr : (k+1)*self.Nr] for k in range(self.K)] if G is not None else [None]*self.K

        # Generate the feedforward messages that will be received by the UTs.
        rx_ff_msgs = [ReceiveFeedforwardMessage(ut_id=k, P_k=P_k, ibr_k=ibr_k, G_k=G_k) for k, (P_k, ibr_k, G_k) in enumerate(zip(P_k_list, ibr_k_list, G_k_list))]

        return rx_ff_msgs


    def propagate(self, x: ComplexArray) -> list[ComplexArray]:

        # Generate the noise samples according to the specified noise model.
        noise = self.noise_model.generate(self.snr, self.K, self.Nr, x.shape[1])

        # Apply the channel matrix H to the transmitted signal x and add the noise to obtain the received signal y.
        y_noiseless = self.channel_model.execute(self.H, x)
        y = self.noise_model.execute(noise, y_noiseless)

        # Split the received signal y into K different signals y_k, one for each UT.
        y_k_list = [ y[k*self.Nr:(k+1)*self.Nr, :] for k in range(self.K) ]

        return y_k_list


class UserTerminal:
    """ 
    Represents a user terminal (UT) in the MU-MIMO downlink system. 
    """

    _id_counter = 0

    @classmethod
    def reset_id_counter(cls) -> None:
        cls._id_counter = 0


    def __init__(self, Nr: int, Ns: int, configs: UserTerminalConfig):

        self.Nr = Nr
        self.Ns = Ns

        # Processing Components.
        self.power_deallocator = configs.power_deallocator()
        self.combiner = configs.combiner()
        self.detector = configs.detector()
        self.demapper = configs.demapper()
        self.bit_deallocator = configs.bit_deallocator()

        # State.
        self.H_k: ComplexArray | None = None
        self.G_k: ComplexArray | None = None
        self.P_k: RealArray | None = None
        self.ibr_k: IntArray | None = None

        # UT ID.
        self.ut_id = UserTerminal._id_counter
        UserTerminal._id_counter += 1

    def reset_state(self) -> None:
        
        self.H_k = None
        self.G_k = None
        self.P_k = None
        self.ibr_k = None

        return


    def receive_pilots(self, rx_pilots_msg: ReceivePilotMessage) -> None:
        
        # Retrieve the channel estimate from the received pilot message.
        H_k = rx_pilots_msg.H_k
        
        # Compute the combining matrix for this UT based on the current channel realization. 
        G_k = self.combiner.compute(H_k)

        # Update the state of the UT to the current channel realization.
        self.H_k = H_k
        self.G_k = G_k
        
        return
    
    def transmit_feedback(self) -> TransmitFeedbackMessage:

        tx_fb_msg = TransmitFeedbackMessage(ut_id = self.ut_id, H_eff_k = self.G_k @ self.H_k)
        return tx_fb_msg

    def receive_feedforward(self, rx_ff_msg: ReceiveFeedforwardMessage) -> None:
        
        # Update the state of the UT to the current channel realization.
        self.G_k = rx_ff_msg.G_k if rx_ff_msg.G_k is not None else self.G_k
        self.P_k = rx_ff_msg.P_k
        self.ibr_k = rx_ff_msg.ibr_k

        return


    def receive(self, y_k: ComplexArray) -> BitArray:
        
        z_k = self.combiner.execute(self.G_k, y_k)
        u_k = self.power_deallocator.execute(self.P_k, z_k)
        a_hat_k = self.detector.execute(self.ibr_k, u_k)
        b_hat_k = self.demapper.execute(self.ibr_k, a_hat_k)
        bitstream_hat_k = self.bit_deallocator.execute(self.ibr_k, b_hat_k)
        
        return bitstream_hat_k

