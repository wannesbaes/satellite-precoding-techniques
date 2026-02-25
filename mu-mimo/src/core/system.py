# mu-mimo/src/core/system.py

from __future__ import annotations
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ..types import (
    RealArray, ComplexArray, IntArray, BitArray,
    ChannelStateInformation,
    TransmitPilotMessage, ReceivePilotMessage, TransmitFeedbackMessage, ReceiveFeedbackMessage, TransmitFeedforwardMessage, ReceiveFeedforwardMessage,
    SystemConfig, BaseStationConfig, UserTerminalConfig, ChannelConfig,
    SimConfig, SimResult, SingleSnrSimResult )
from ..processing import (
    Precoder, Combiner,
    PowerAllocator, PowerDeallocator,
    BitAllocator, BitDeallocator,
    Mapper, Demapper, Detector,
    ChannelModel, NoiseModel )



class SimulationRunner:
    """
    Orchestrates the execution of the MU-MIMO downlink system simulation.
    """
     
    def __init__(self, system_config: SystemConfig, sim_config: SimConfig):

        self.system_config = system_config
        self.sim_config = sim_config

        self.mu_mimo_system = MuMimoSystem(system_config)
    
    def run(self) -> SimResult:
        
        # Generate the filepath for this simulation.
        filepath: Path = self._generate_filepath()

        # Check if this simulation has already been executed. If so, load the results and return them.
        if filepath.exists():
            sim_result = self._load_results(filepath)
            return sim_result
        
        # Run the simulation.
        simulation_results: list[SingleSnrSimResult] = []

        # Outer loop: Iterate over the SNR values.
        for snr in tqdm(self.sim_config.snr_values, desc="SNR values"):

            inner_loop_results: list[SingleSnrSimResult] = []
            bit_error_count = 0
            channel_realization_count = 0

            # Inner loop: Iterate over the channel realizations.
            while (channel_realization_count < self.sim_config.num_channel_realizations) or (bit_error_count < self.sim_config.num_bit_errors):

                # Reset.
                csi = ChannelStateInformation(snr=snr, H=None)
                self.mu_mimo_system.reset(csi)

                # Configuration.
                self.mu_mimo_system.configure()

                # Communication.
                tx_bits_list, rx_bits_list = self.mu_mimo_system.communicate(self.sim_config.num_symbols)

                # Store inner loop results.
                inner_loop_result = self._calculate_inner_loop_result(tx_bits_list, rx_bits_list)
                inner_loop_results.append(inner_loop_result)

                # Stopping criterion.
                channel_realization_count += 1
                bit_error_count += self._calculate_bit_error_count_update(inner_loop_result)

            # Store outer loop results.
            simulation_results.append(self._calculate_outer_loop_result(inner_loop_results))

        # Save simulation results.
        simulation_result = SimResult(
            filename = filepath,
            sim_configs = self.sim_config,
            system_configs = self.system_config,
            snr_dB_values = self.sim_config.snr_dB_values,
            simulation_results = simulation_results)
        self._save_results(simulation_result)

        return simulation_result
        
    def _generate_filepath(self) -> Path:

        # Create the results directory if it does not exist.
        results_dir = Path(__file__).resolve().parents[2] / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Generate the filename based on the system and simulation configurations.
        snr_dB_values = self.sim_config.snr_dB_values
        bs_cfg = self.system_config.base_station_configs
        ut_cfg = self.system_config.user_terminal_configs
        ch_cfg = self.system_config.channel_configs

        name_parts = [
            "mu_mimo_downlink_sim",
            f"K_{self.system_config.K}",
            f"Nt_{self.system_config.Nt}",
            f"Nr_{self.system_config.Nr}",
            f"Ns_{self.system_config.Ns}",
            f"snr_min_{snr_dB_values.min()}",
            f"snr_max_{snr_dB_values.max()}",
            f"snr_step_{(snr_dB_values[-1] - snr_dB_values[-2]):.1f}" if len(snr_dB_values) > 1 else "0",
            f"ncr_{self.sim_config.num_channel_realizations}",
            f"nbe_{self.sim_config.num_bit_errors}",
            f"nbe_scope_{self.sim_config.num_bit_errors_scope}",
            f"ns_{self.sim_config.num_symbols}",
            f"prec_{bs_cfg.precoder.__name__}",
            f"comb_{ut_cfg.combiner.__name__}",
            f"pa_{bs_cfg.power_allocator.__name__}",
            f"ba_{bs_cfg.bit_allocator.__name__}",
            f"cm_{ch_cfg.channel_model.__name__}",
            f"nm_{ch_cfg.noise_model.__name__}",
        ]
        filename = "__".join(name_parts) + ".npz"

        # Generate the full file path.
        filepath = results_dir / filename
        return filepath

    def _calculate_bit_error_count_update(self, inner_loop_result: SingleSnrSimResult) -> int:
        
        if self.sim_config.num_bit_errors_scope == "system-wide":
            becu = inner_loop_result.ibr * inner_loop_result.ber * self.sim_config.num_symbols
            becu = int(becu) if not np.isnan(becu) else 0
            return becu
        
        elif self.sim_config.num_bit_errors_scope == "uts":
            becus = inner_loop_result.ut_ibrs * inner_loop_result.ut_bers * self.sim_config.num_symbols
            becu = int(np.nanmin(becus)) if not np.all(np.isnan(becus)) else 0
            return becu
        
        elif self.sim_config.num_bit_errors_scope == "streams":
            becus = np.array([ stream_ibrs * stream_bers * self.sim_config.num_symbols for stream_ibrs, stream_bers in zip(inner_loop_result.stream_ibrs, inner_loop_result.stream_bers) ])
            becu = int(np.nanmin(becus)) if not np.all(np.isnan(becus)) else 0
            return becu
        
        else:
            raise ValueError(f"The specified scope for the minimum number of bit errors per SNR value is not valid: {self.sim_config.num_bit_errors_scope}. Valid options are: 'system-wide', 'uts', 'streams'.")

    def _calculate_inner_loop_result(self, tx_bits_list: list[list[BitArray]], rx_bits_list: list[list[BitArray]]) -> SingleSnrSimResult:
        
        # Initialization.
        K = len(tx_bits_list)
        Ns = len(tx_bits_list[0])
        
        stream_ibrs = [np.empty((Ns,), dtype=int) for k in range(K)]
        stream_bers = [np.empty((Ns,), dtype=float) for k in range(K)]

        ut_ibrs = np.empty((K,), dtype=int)
        ut_bers = np.empty((K,), dtype=float)

        # Iteration.
        for k in range(K):

            for s in range(Ns):
                stream_ibrs[k][s] = len(tx_bits_list[k][s]) // self.sim_config.num_symbols
                stream_bers[k][s] = np.mean(tx_bits_list[k][s] != rx_bits_list[k][s]) if len(tx_bits_list[k][s]) > 0 else np.nan
            
            ut_ibrs[k] = np.sum(stream_ibrs[k])
            ut_bers[k] = np.nansum(stream_ibrs[k] * stream_bers[k]) / np.sum(stream_ibrs[k]) if np.sum(stream_ibrs[k]) > 0 else np.nan
        
        ibr = np.sum(ut_ibrs)
        ber = np.nansum(ut_ibrs * ut_bers) / np.sum(ut_ibrs) if np.sum(ut_ibrs) > 0 else np.nan

        # Termination.
        inner_loop_result = SingleSnrSimResult(
            stream_ibrs = stream_ibrs,
            stream_bers = stream_bers,
            ut_ibrs = ut_ibrs,
            ut_bers = ut_bers,
            ibr = ibr,
            ber = ber)
        return inner_loop_result

    def _calculate_outer_loop_result(self, inner_loop_results: list[SingleSnrSimResult]) -> SingleSnrSimResult:


        # Initialization.
        K = len(inner_loop_results[0].stream_ibrs)
        

        # Iteration.

        # Result per stream (for single SNR value, averaged over channel realizations).
        stream_ibrs: list[IntArray] = []
        stream_bers: list[RealArray] = []

        for k in range(K):

            stream_ibrs_inner_loops = np.stack([res.stream_ibrs[k] for res in inner_loop_results], axis=0)
            stream_ibrs_outer_loop = np.mean(stream_ibrs_inner_loops, axis=0)
            stream_ibrs.append(stream_ibrs_outer_loop)

            stream_bers_inner_loops = np.stack([res.stream_bers[k] for res in inner_loop_results], axis=0)
            stream_bers_outer_loop = np.nanmean(stream_bers_inner_loops, axis=0)
            stream_bers.append(stream_bers_outer_loop)

        # Result per UT (for single SNR value, averaged over channel realizations).
        ut_ibrs_inner_loops = np.stack([res.ut_ibrs for res in inner_loop_results], axis=0)
        ut_ibrs = np.mean(ut_ibrs_inner_loops, axis=0)

        ut_bers_inner_loops = np.stack([res.ut_bers for res in inner_loop_results], axis=0)
        ut_bers = np.nanmean(ut_bers_inner_loops, axis=0)

        # Result system-wide (for single SNR value, averaged over channel realizations).
        ibrs_inner_loops = np.array([res.ibr for res in inner_loop_results])
        ibr = float(np.mean(ibrs_inner_loops))

        bers_inner_loops = np.array([res.ber for res in inner_loop_results])
        ber = float(np.nanmean(bers_inner_loops))


        # Termination.
        outer_loop_result = SingleSnrSimResult(
            stream_ibrs = stream_ibrs,
            stream_bers = stream_bers,
            ut_ibrs = ut_ibrs,
            ut_bers = ut_bers,
            ibr = ibr,
            ber = ber)
        return outer_loop_result

    def _load_results(self, filepath: Path) -> SimResult:
        
        # Load the simulation results from the .npz file.
        loaded_data = np.load(filepath, allow_pickle=True)
        sim_result = SimResult(
            filename = Path(str(loaded_data["filename"])),
            sim_configs = loaded_data["sim_configs"].item(),
            system_configs = loaded_data["system_configs"].item(),
            snr_dB_values = loaded_data["snr_dB_values"],
            simulation_results = loaded_data["simulation_results"].tolist())
        
        # Validate that the loaded simulation results match the current simulation and system configuration.
        if self.sim_config != sim_result.sim_configs or self.system_config != sim_result.system_configs:
            raise ValueError("The loaded simulation results do not match the current simulation and system configuration. However their filename suggests that they should. Please check the filename and the contents of the loaded simulation results to resolve this issue.")
        
        return sim_result

    def _save_results(self, simulation_result: SimResult) -> None:

        # Save the simulation results to a .npz file.
        np.savez(simulation_result.filename,
            filename = simulation_result.filename,
            sim_configs = simulation_result.sim_configs,
            system_configs = simulation_result.system_configs,
            snr_dB_values = simulation_result.snr_dB_values,
            simulation_results = np.array(simulation_result.simulation_results, dtype=object))
        
        return


class MuMimoSystem:
    """
    Represents the MU-MIMO downlink system, orchestrating the base station, channel, and user terminals.
    """

    def __init__(self, system_config: SystemConfig):

        K = system_config.K
        Nt = system_config.Nt
        Nr = system_config.Nr
        Ns = system_config.Ns

        self.bs = BaseStation(Nt, Ns, system_config.base_station_configs)
        self.channel = Channel(K, Nr, Nt, Ns, system_config.channel_configs)
        self.uts = [UserTerminal(k, Nr, Ns, system_config.user_terminal_configs) for k in range(K)]

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
        for ut, rx_pilot_msg in zip(self.uts, rx_pilot_msgs): ut.receive_pilots(rx_pilot_msg)

        # Feedback Phase.
        tx_fb_msgs = [ut.transmit_feedback() for ut in self.uts]
        rx_fb_msg = self.channel.propagate_feedback(tx_fb_msgs)
        self.bs.receive_feedback(rx_fb_msg)

        # Feedforward Phase.
        tx_ff_msg = self.bs.transmit_feedforward()
        rx_ff_msgs = self.channel.propagate_feedforward(tx_ff_msg)
        for ut, rx_ff_msg in zip(self.uts, rx_ff_msgs): ut.receive_feedforward(rx_ff_msg)

    def communicate(self, num_symbols: int) -> tuple[list[list[BitArray]], list[list[BitArray]]]:

        # Transmit from the BS.
        tx_bits_list, x = self.bs.transmit(num_symbols)

        # Propagate through the channel.
        y_k_list = self.channel.propagate(x)

        # Receive at the UTs.
        rx_bits_list = [ut.receive(y_k) for ut, y_k in zip(self.uts, y_k_list)]

        return tx_bits_list, rx_bits_list


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

    def transmit(self, num_symbols: int) -> tuple[list[list[BitArray]], ComplexArray]:

        b = self.bit_allocator.execute(self.ibr, num_symbols)
        a = self.mapper.execute(self.ibr, b)
        a_p = self.power_allocator.execute(self.P, a)
        x = self.precoder.execute(self.F, a_p)

        tx_bits_list = [ [ b[k*self.Ns + s] for s in range(self.Ns) ] for k in range(len(b) // self.Ns) ]

        return tx_bits_list, x


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

    def __init__(self, ut_id: int, Nr: int, Ns: int, configs: UserTerminalConfig):

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
        self.ut_id = ut_id

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

    def receive(self, y_k: ComplexArray) -> list[BitArray]:
        
        z_k = self.combiner.execute(self.G_k, y_k)
        u_k = self.power_deallocator.execute(self.P_k, z_k)
        a_hat_k = self.detector.execute(self.ibr_k, u_k)
        b_hat_k = self.demapper.execute(self.ibr_k, a_hat_k)
        rx_bits_list = self.bit_deallocator.execute(self.ibr_k, b_hat_k)
        
        return rx_bits_list

