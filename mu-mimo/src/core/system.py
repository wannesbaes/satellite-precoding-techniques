# mu-mimo/src/core/system.py

from __future__ import annotations
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ..types import (
    RealArray, ComplexArray, IntArray, BitArray,
    ChannelStateInformation, BaseStationState, UserTerminalState,
    TransmitPilotMessage, ReceivePilotMessage, TransmitFeedbackMessage, ReceiveFeedbackMessage, TransmitFeedforwardMessage, ReceiveFeedforwardMessage,
    ConstConfig, SystemConfig, BaseStationConfig, UserTerminalConfig, ChannelConfig,
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
     
    def __init__(self, sim_config: SimConfig, system_config: SystemConfig):
        """
        Initialize a simulation runner.

        Parameters
        ----------
        sim_config: SimConfig
            The configuration of the simulation. This includes the SNR values, the minimum number of channel realizations, the minimum number of bit errors, and the number of symbols to be transmitted at once.
        system_config: SystemConfig
            The configuration of the MU-MIMO system. 
            This includes the number of user terminals (UTs), the number of transmit antennas at the base station (BS), the number of receive antennas per UT. Also, it includes the configurations of the BS, UTs and channel.
        """

        self.system_config = system_config
        self.sim_config = sim_config

        self.mu_mimo_system = MuMimoSystem(system_config)
    
    def run(self) -> SimResult:
        """
        Run the MU-MIMO downlink system simulation.

        The simulation consists of an outer loop iterating over the SNR values and an inner loop iterating over the channel realizations for each SNR value until the stopping criterion is met. The stopping criterion is based on the minimum number of channel realizations and the minimum number of bit errors for each SNR value. The performance metrics are calculated for each channel realization and later averaged over the channel realizations for each SNR value. Finally, the simulation results are saved to a .npz file.

        Results
        -------
        simulation_result : SimResult
            The simulation results.
        """
        
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
        """
        Generate a unique filepath for each distinct simulation and system configuration.

        Returns
        -------
        filepath : Path
            The generated filepath for the simulation results. 
        """

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
        """
        Calculate the update of the bit error count stopping criterion based on the specified scope.

        Parameters
        ----------
        inner_loop_result : SingleSnrSimResult
            The performance metrics for the current channel realization and SNR value. (we refer to the SingleSnrSimResult dataclass for more details)
        
        Returns
        -------
        becu : int
            The update of the bit error count stopping criterion.
        """
        
        if self.sim_config.num_bit_errors_scope == "system-wide":
            becu = inner_loop_result.bec
        
        elif self.sim_config.num_bit_errors_scope == "uts":
            becu = np.nanmin(inner_loop_result.ut_becs)

        elif self.sim_config.num_bit_errors_scope == "streams":
            becu = np.nanmin(np.array(inner_loop_result.stream_becs))
        
        else:
            raise ValueError(f"The specified scope for the minimum number of bit errors per SNR value is not valid: {self.sim_config.num_bit_errors_scope}. Valid options are: 'system-wide', 'uts', 'streams'.")
        
        becu = int(becu) if not np.isnan(becu) else 0
        return becu

    def _calculate_inner_loop_result(self, tx_bits_list: list[list[BitArray]], rx_bits_list: list[list[BitArray]]) -> SingleSnrSimResult:
        """
        Calculate the performance metrics for a simulation corresponding to a single channel realization and SNR value.

        Parameters
        ----------
        tx_bits_list : list[list[BitArray]], shape (K, Ns_k, ibr_k_s * num_symbols)
            The list of bitstreams for each UT k and each data stream s that were transmitted by the BS.
        rx_bits_list : list[list[BitArray]], shape (K, Ns_k, ibr_k_s * num_symbols)
            The list of bitstreams for each UT k and each data stream s that were reconstructed by the UTs.
        
        Returns
        -------
        inner_loop_result : SingleSnrSimResult
            The performance metrics. (we refer to the SingleSnrSimResult dataclass for more details)
        """
        
        # Initialization.
        K = self.system_config.K
        Nr = self.system_config.Nr
        Ns = [len(tx_bits_list[k]) for k in range(K)]
        
        stream_ibrs = [np.zeros((Nr,), dtype=int) for k in range(K)]
        stream_becs = [np.full((Nr,), np.nan, dtype=float) for k in range(K)]
        stream_ars = [np.zeros((Nr,), dtype=int) for k in range(K)]

        ut_ibrs = np.empty((K,), dtype=int)
        ut_becs = np.empty((K,), dtype=float)
        ut_ars = np.empty((K,), dtype=float)

        # Iteration.
        for k in range(K):

            for s in range(Ns[k]):
                stream_ibrs[k][s] = len(tx_bits_list[k][s]) // self.sim_config.num_symbols
                stream_becs[k][s] = np.sum(tx_bits_list[k][s] != rx_bits_list[k][s])
                stream_ars[k][s] = 1
            
            ut_ibrs[k] = np.sum(stream_ibrs[k])
            ut_becs[k] = np.nansum(stream_becs[k]) if not np.all(np.isnan(stream_becs[k])) else np.nan
            ut_ars[k] = 1 if ut_ibrs[k] > 0 else 0
        
        ibr = np.sum(ut_ibrs)
        bec = np.nansum(ut_becs) if not np.all(np.isnan(ut_becs)) else np.nan

        stream_ars_avg = np.sum([np.sum(stream_ars[k]) for k in range(K)]) / (K*Nr)
        ut_ars_avg = np.mean(ut_ars)

        # Termination.
        inner_loop_result = SingleSnrSimResult(
            stream_ibrs = stream_ibrs,
            stream_bers = stream_becs,
            stream_ars = stream_ars,
            ut_ibrs = ut_ibrs,
            ut_becs = ut_becs,
            ut_ars = ut_ars,
            ibr = ibr,
            bec = bec,
            stream_ars_avg = stream_ars_avg,
            ut_ars_avg = ut_ars_avg,
            num_symbols = self.sim_config.num_symbols,
            num_channel_realizations = 1,)
        return inner_loop_result

    def _calculate_outer_loop_result(self, inner_loop_results: list[SingleSnrSimResult]) -> SingleSnrSimResult:
        """
        Calculate the performance metrics for a simulation corresponding to a single SNR value, averaged over different channel realizations.

        Parameters
        ----------
        inner_loop_results : list[SingleSnrSimResult]
            The list of performance metrics for each channel realization.
        
        Returns
        -------
        outer_loop_result : SingleSnrSimResult
            The performance metrics. (we refer to the SingleSnrSimResult dataclass for more details)
        """


        # Initialization.
        K = len(inner_loop_results[0].stream_ibrs)
        

        # Iteration.

        # Result per stream (for single SNR value, averaged over channel realizations).
        stream_ibrs: list[IntArray] = []
        stream_becs: list[RealArray] = []
        stream_ars: list[BitArray] = []

        for k in range(K):

            stream_ibrs_inner_loops = np.stack([ilr.stream_ibrs[k] for ilr in inner_loop_results], axis=0)
            stream_ibrs_outer_loop = np.mean(stream_ibrs_inner_loops, axis=0)
            stream_ibrs.append(stream_ibrs_outer_loop)

            stream_becs_inner_loops = np.stack([ilr.stream_becs[k] for ilr in inner_loop_results], axis=0)
            stream_becs_outer_loop = np.nansum(stream_becs_inner_loops, axis=0)
            stream_becs.append(stream_becs_outer_loop)

            stream_ars_inner_loops = np.stack([ilr.stream_ars[k] for ilr in inner_loop_results], axis=0)
            stream_ars_outer_loop = np.mean(stream_ars_inner_loops, axis=0)
            stream_ars.append(stream_ars_outer_loop)

        # Result per UT (for single SNR value, averaged over channel realizations).
        ut_ibrs_inner_loops = np.stack([ilr.ut_ibrs for ilr in inner_loop_results], axis=0)
        ut_ibrs = np.mean(ut_ibrs_inner_loops, axis=0)

        ut_becs_inner_loops = np.stack([ilr.ut_becs for ilr in inner_loop_results], axis=0)
        ut_becs = np.nansum(ut_becs_inner_loops, axis=0)

        ut_ars_inner_loops = np.stack([ilr.ut_ars for ilr in inner_loop_results], axis=0)
        ut_ars = np.mean(ut_ars_inner_loops, axis=0)

        # Result system-wide (for single SNR value, averaged over channel realizations).
        ibrs_inner_loops = np.array([ilr.ibr for ilr in inner_loop_results])
        ibr = float(np.mean(ibrs_inner_loops))

        becs_inner_loops = np.array([ilr.bec for ilr in inner_loop_results])
        bec = float(np.nansum(becs_inner_loops))

        stream_ars_avg = float(np.mean([np.mean(stream_ars[k]) for k in range(K)]))
        ut_ars_avg = float(np.mean(ut_ars))


        # Termination.
        outer_loop_result = SingleSnrSimResult(
            stream_ibrs = stream_ibrs,
            stream_becs = stream_becs,
            stream_ars = stream_ars,
            ut_ibrs = ut_ibrs,
            ut_becs = ut_becs,
            ut_ars = ut_ars,
            ibr = ibr,
            bec = bec,
            stream_ars_avg = stream_ars_avg,
            ut_ars_avg = ut_ars_avg,
            num_symbols = self.sim_config.num_symbols,
            num_channel_realizations = len(inner_loop_results))
        return outer_loop_result

    def _load_results(self, filepath: Path) -> SimResult:
        """
        Load simulation results from a previously executed simulation with the same simulation and system configuration.

        Parameters
        ----------
        filepath : Path
            The path to the .npz file containing the simulation results.
        
        Returns
        -------
        sim_result : SimResult
            The loaded simulation results.
        """
        
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
        """
        Save the simulation results to a .npz file.

        Parameters
        ----------
        simulation_result : SimResult
            The simulation results to save.
        """

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
    Represents a MU-MIMO downlink digital communication system.
    """

    def __init__(self, system_config: SystemConfig):
        """
        Initialize a MU-MIMO system.

        Parameters
        ----------
        system_config: SystemConfig
            The configuration of the MU-MIMO system.\\
            This includes the total available transmit power, the system bandwidth, the number of user terminals (UTs), the number of transmit antennas at the base station (BS), the number of receive antennas per UT and the modulation constellation settings. Also, it includes the configurations of the BS, UTs and channel.
        """

        Pt = system_config.Pt
        B = system_config.B
        
        K = system_config.K
        Nt = system_config.Nt
        Nr = system_config.Nr

        self.bs = BaseStation(Pt, B, K, Nt, system_config.base_station_configs, system_config.c_configs)
        self.channel = Channel(K, Nr, Nt, system_config.channel_configs)
        self.uts = [UserTerminal(k, Nr, system_config.user_terminal_configs) for k in range(K)]

    def reset(self, csi: ChannelStateInformation) -> ChannelStateInformation:
        """
        Resets the MU-MIMO system. 
        
        After resetting, the system is ready for configuration.
        For more datail on the reset phase, we refer to the reset_state() methods of the BS, UTs and channel.

        Parameters
        ----------
        csi : ChannelStateInformation
            The channel state information to reset to.
        """

        # Reset the base station state.
        self.bs.reset_state()

        # Reset each user terminal state.
        for ut in self.uts: ut.reset_state()

        # Reset the channel state.
        csi = self.channel.reset(csi)

        return csi

    def configure(self) -> None:
        """
        Configuration the MU-MIMO system. 
        
        Make sure the system has been reset first. After configuration, the system is ready for communication.
        For more datails on the configuration phase, we refer to the receive, propagate and transmit methods of the pilot signals, feedback messages and feedforward messages of the BS, UTs and channel.
        """

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

        return

    def communicate(self, num_symbols: int) -> tuple[list[list[BitArray]], list[list[BitArray]]]:
        """
        Communication of the MU-MIMO system.

        Make sure the system has been configured first. After communication, performance evaluation can be performed.
        For more datails on the communication phase, we refer to the transmit method of the BS, the propagate method of the channel and the receive method of the UTs.

        Parameters
        ----------
        num_symbols : int
            The number of symbols to be transmitted for this channel realization and SNR at once.
        
        Returns
        -------
        tx_bits_list : list[list[BitArray]], shape (K, Ns_k, ibr_k_s * num_symbols)
            The list of bitstreams for each UT k and each data stream s that were transmitted by the BS.
        rx_bits_list : list[list[BitArray]], shape (K, Ns_k, ibr_k_s * num_symbols)
            The list of bitstreams for each UT k and each data stream s that were received by the UTs.
        """

        # Transmit from the BS.
        tx_bits_list, x = self.bs.transmit(num_symbols)

        # Propagate through the channel.
        y_k_list = self.channel.propagate(x)

        # Receive at the UTs.
        rx_bits_list = [ut.receive(y_k) for ut, y_k in zip(self.uts, y_k_list)]

        return tx_bits_list, rx_bits_list

class BaseStation:
    """
    Represents the base station (BS) in a MU-MIMO downlink system.
    """

    def __init__(self, Pt: float, B: float, K: int, Nt: int, configs: BaseStationConfig, c_configs: ConstConfig):
        """
        Initialize a base station.

        Parameters
        ----------
        Pt : float
            The total transmit power at the BS.
        B : float
            The bandwidth of the system.

        K: int
            The number of user terminals (UTs) in the system.
        Nt: int
            The number of transmit antennas at the BS.
        
        configs: BaseStationConfig
            The configuration of the processing components (bit allocator, mapper, power allocator and precoder) in the BS.
        c_configs: ConstConfig
            The constellation configuration settings for each UT.
        """

        self.Pt = Pt
        self.B = B
        
        self.K = K
        self.Nt = Nt

        # Processing Components.
        self.bit_allocator: BitAllocator = configs.bit_allocator()
        self.mapper: Mapper = configs.mapper()
        self.power_allocator: PowerAllocator = configs.power_allocator()
        self.precoder: Precoder = configs.precoder()

        # State.
        self.c_configs: ConstConfig = c_configs
        self.state: BaseStationState | None = None

    def reset_state(self) -> None:
        """
        Resets the state of the BS. 
        
        It clears the precoder F, the power allocation P, the information bit rates ibr, the number of data streams for each UT Ns and the combining matrices G for each UT in case of coordinated beamforming from the previous channel realization and SNR.
        """
        
        self.state = None
        return

    def transmit_pilots(self) -> TransmitPilotMessage:
        """
        Transmits pilot signals from the BS through the channel.

        Returns
        -------
        pilot_msg : TransmitPilotMessage
            The pilot messages transmitted by the BS.
        """

        pilot_msg = TransmitPilotMessage()
        return pilot_msg

    def receive_feedback(self, rx_fb_msg: ReceiveFeedbackMessage) -> None:
        """
        Receive and process the feedback messages from the UTs.

        The feedback message contains the current CSI, which is used to compute the precoder matrix, the power allocation, and the information bit rate for the current channel realization and SNR, according to the specific algorithms implemented in the processing components of the BS. 
        In case of non-coordinated beamforming, the effective channel matrix (part of the CSI) equals the channel matrix H followed by the compound combining matrix G. 

        Parameters
        ----------
        rx_fb_msg : ReceiveFeedbackMessage
            The compound feedback message.
        """

        # Compute the precoder matrix, the power allocation, and the information bit rate for the current channel realization and SNR. In case of coordinated beamforming, also compute the combining matrices.
        F, G = self.precoder.compute(rx_fb_msg.csi)
        P = self.power_allocator.compute(rx_fb_msg.csi, F, G, self.Pt, B=self.B)
        ibr, Ns = self.bit_allocator.compute(rx_fb_msg.csi, F, G, P, self.c_configs, Pt=self.Pt, B=self.B)

        # Update the state of the BS to the current channel realization and SNR.
        self.state = BaseStationState(F=F, P=P, ibr=ibr, Ns=Ns, G=G)

        return

    def transmit_feedforward(self) -> TransmitFeedforwardMessage:
        """
        Transmits the feedforward messages from the BS through the channel.

        The feedforward message contains the constellation type, the power allocation, the information bit rates and the number of data streams for each UT and the combining matrices for each UT in case of coordinated beamforming for the current channel realization and SNR.

        Returns
        -------
        tx_ff_msg : TransmitFeedforwardMessage
            The feedforward messages.
        """

        tx_ff_msg = TransmitFeedforwardMessage(c_type=self.c_configs.types, P=self.state.P, ibr=self.state.ibr, Ns=self.state.Ns, G=self.state.G)
        return tx_ff_msg

    def transmit(self, num_symbols: int) -> tuple[list[list[BitArray]], ComplexArray]:
        """
        Simulate the transmit processing chain of the BS to obtain the transmitted signal x.

        The processing chain consists of the following steps:
        1. Bit Allocation - generate the bitstreams b_s for each data stream s based on the information bit rates ibr and the number of symbols to be transmitted num_symbols.
        2. Mapping - convert the bitstreams b_s to the corresponding data symbol streams a_s based on the modulation scheme (determined by the information bit rates ibr).
        3. Power Allocation - allocate power to the symbol streams a_s based on the power allocation P to obtain the power-scaled symbol streams a_p_s.
        4. Precoding - apply the precoding matrix F to the power-scaled symbol streams a_p_s to obtain the transmitted signal x.

        Parameters
        ----------
        num_symbols : int
            The number of symbols to be transmitted for this channel realization and SNR at once.

        Returns
        -------
        tx_bits_list : list[list[BitArray]], shape (K, Ns_k, ibr_k_s * num_symbols)
            The list of bitstreams for each UT k and each data stream s.
        x : ComplexArray, shape (Nt, num_symbols)
            The transmitted signal.
        """

        tx_bits_list, b = self.bit_allocator.apply(self.state.ibr, num_symbols, self.state.Ns)
        a = self.mapper.apply(b, self.state.ibr, self.c_configs.types, self.state.Ns)
        a_p = self.power_allocator.apply(a, self.state.P)
        x = self.precoder.apply(a_p, self.state.F)

        return tx_bits_list, x

class Channel:
    """
    Represents the wireless channel in a MU-MIMO downlink system.
    """

    def __init__(self, K: int, Nr: int, Nt: int, configs: ChannelConfig):
        """
        Initialize a channel.

        Parameters
        ----------
        K: int
            The number of user terminals (UTs) in the system.
        Nr: int
            The number of receive antennas per UT.
        Nt: int
            The number of transmit antennas at the BS.
        configs: ChannelConfig
            The channel configurations (channel model & noise model).
        """

        self.K = K
        self.Nr = Nr
        self.Nt = Nt

        # Channel model and noise model.
        self.channel_model = configs.channel_model()
        self.noise_model = configs.noise_model()

        # State.
        self.state: ChannelStateInformation | None = None

    def set(self, csi: ChannelStateInformation) -> None:
        """
        Sets the channel state information (CSI).

        The CSI consists of the SNR and the channel matrix H. If either of them is not provided, the old value is kept.

        Parameters
        ----------
        csi : ChannelStateInformation
            The channel state information to set.
        """
       
        if csi.snr is not None: self.state.snr = csi.snr
        if csi.H_eff is not None: self.state.H_eff = csi.H_eff

        return

    def reset(self, csi: ChannelStateInformation) -> ChannelStateInformation:
        """
        Resets the channel state information (CSI).

        If the SNR or the channel matrix H is not provided in the input CSI, they are set to default values.
        For the SNR, the default value is infinity, which corresponds to the absence of noise. For the channel matrix H, the default value is a randomly generated channel matrix according to the specified channel model.

        Parameters
        ----------
        csi : ChannelStateInformation
            The channel state information to reset to.

        Returns
        -------
        ChannelStateInformation
            The new channel state information.
        """
        
        if csi.snr is None: csi.snr = np.inf
        if csi.H_eff is None: csi.H_eff = self.channel_model.generate(self.K * self.Nr, self.Nt)
        
        self.set(csi)

        return csi

    def propagate_pilots(self, tx_pilot_msg: TransmitPilotMessage) -> list[ReceivePilotMessage]:
        """
        Propagates the pilot signals from the BS to the UTs.

        The compound pilot message transmitted by the BS is split into K different pilot messages, one for each UT.

        Parameters
        ----------
        tx_pilot_msg : TransmitPilotMessage
            The compound pilot message transmitted by the BS.
        
        Returns
        -------
        rx_pilot_msgs : list[ReceivePilotMessage]
            The list of pilot messages that will be received by each UT.
        """

        rx_pilot_msgs = [ReceivePilotMessage(H_k = self.state.H_eff[k*self.Nr:(k+1)*self.Nr, :]) for k in range(self.K)]
        return rx_pilot_msgs

    def propagate_feedback(self, tx_fb_msgs: list[TransmitFeedbackMessage]) -> ReceiveFeedbackMessage:
        """
        Propagates the feedback messages from the UTs to the BS.

        The feedback messages transmitted by the UTs are aggregated to form the compound feedback message received by the BS, which contains the CSI (effective channel matrix and SNR) for the current channel realization and SNR.

        Parameters
        ----------
        tx_fb_msgs : list[TransmitFeedbackMessage]
            The list of feedback messages transmitted by the UTs.

        Returns
        -------
        rx_fb_msg : ReceiveFeedbackMessage
            The compound feedback message that will be received by the BS.
        """
        
        # Aggregate the effective channel matrices from the received feedback messages.
        H_eff = np.empty((self.K*self.Nr, self.Nt), dtype=complex)
        for tx_fb_msg_k in tx_fb_msgs:
            H_eff[tx_fb_msg_k.ut_id*self.Nr : (tx_fb_msg_k.ut_id+1)*self.Nr, : ] = tx_fb_msg_k.H_eff_k
        
        # Generate the feedback message that will be received by the BS.
        rx_fb_msg = ReceiveFeedbackMessage(snr=self.state.snr, H_eff=H_eff)

        return rx_fb_msg

    def propagate_feedforward(self, tx_ff_msg: TransmitFeedforwardMessage) -> list[ReceiveFeedforwardMessage]:
        """
        Propagates the feedforward messages from the BS to the UTs.

        The feedforward message transmitted by the BS is split into K different feedforward messages, one for each UT. Each feedforward message contains the constellation type, the power allocation, the information bit rates, the number of data streams for each UT and, in case of coordinated beamforming, the combining matrices for each UT for the current channel realization and SNR.

        Parameters
        ----------
        tx_ff_msg : TransmitFeedforwardMessage
            The feedforward message transmitted by the BS.
        
        Returns
        -------
        rx_ff_msgs : list[ReceiveFeedforwardMessage]
            The list of feedforward messages that will be received by each UT.
        """
        
        # Retrieve the transmitted feedforward message elements.
        c_type = tx_ff_msg.c_type
        P = tx_ff_msg.P
        ibr = tx_ff_msg.ibr
        Ns = tx_ff_msg.Ns
        G = tx_ff_msg.G

        # Split the transmitted feedforward message elemtens into K different elements for each UT.
        Ns_cumulative = np.concatenate(([0], np.cumsum(Ns)))
        P_k_list = [ P[ Ns_cumulative[k] : Ns_cumulative[k+1] ] for k in range(self.K)]
        ibr_k_list = [ ibr[ Ns_cumulative[k] : Ns_cumulative[k+1] ] for k in range(self.K)]
        G_k_list = [ G[ Ns_cumulative[k] : Ns_cumulative[k+1], k*self.Nr : (k+1)*self.Nr] for k in range(self.K)] if G is not None else [None]*self.K

        # Generate the feedforward messages that will be received by the UTs.
        rx_ff_msgs = [ReceiveFeedforwardMessage(ut_id=k, c_type_k=c_type[k], P_k=P_k, ibr_k=ibr_k, Ns_k=Ns[k], G_k=G_k) for k, (P_k, ibr_k, G_k) in enumerate(zip(P_k_list, ibr_k_list, G_k_list))]

        return rx_ff_msgs

    def propagate(self, x: ComplexArray) -> list[ComplexArray]:
        """
        Simulate the signal propagation through the channel of the transmitted signal x to obtain the received signal y.

        The channel propagation consists of the following steps:
        1. Apply the channel matrix H to the transmitted signal x to obtain the noiseless received signal y_noiseless.
        2. Generate the noise samples according to the specified noise model.
        3. Add the noise to the noiseless received signal y_noiseless to obtain the actual received signal y.

        Parameters
        ----------
        x : ComplexArray, shape (Nt, num_symbols)
            The transmitted signal.
        
        Returns
        -------
        y_k_list : list[ComplexArray], shape (K, Nr, num_symbols)
            The list of received signals for each UT k.
        """

        # Generate the noise samples according to the specified noise model.
        noise = self.noise_model.generate(self.state.snr, x, self.K * self.Nr)

        # Apply the channel matrix H to the transmitted signal x and add the noise to obtain the received signal y.
        y_noiseless = self.channel_model.apply(x, self.state.H_eff)
        y = self.noise_model.apply(y_noiseless, noise)

        # Split the received signal y into K different signals y_k, one for each UT.
        y_k_list = [ y[k*self.Nr:(k+1)*self.Nr, :] for k in range(self.K) ]

        return y_k_list

class UserTerminal:
    """ 
    Represents a user terminal (UT) in a MU-MIMO downlink system. 
    """

    def __init__(self, ut_id: int, Nr: int, configs: UserTerminalConfig):
        """
        Initialize a user terminal.

        Parameters
        ----------
        ut_id: int
            The ID of the UT, which is used to identify the UT in the system.
        Nr: int
            The number of receive antennas at the UT. This is equal for all UTs in the system.
        configs: UserTerminalConfig
            The configuration of the processing components (combiner, detector, demapper and bit deallocator) in the UT.
        """

        self.Nr = Nr

        # Processing Components.
        self.power_deallocator = configs.power_deallocator()
        self.combiner = configs.combiner()
        self.detector = configs.detector()
        self.demapper = configs.demapper()
        self.bit_deallocator = configs.bit_deallocator()

        # State.
        self.state: UserTerminalState | None = None

        # UT ID.
        self.ut_id = ut_id

    def reset_state(self) -> None:
        """
        Reset the state of the user terminal.

        It clears the channel matrix H_k, the combining matrix G_k, the power allocation P_k, the information bit rates ibr_k and the number of data streams Ns_k for the previous channel realization and SNR.
        """
        
        self.state = None
        return

    def receive_pilots(self, rx_pilots_msg: ReceivePilotMessage) -> None:
        """
        Receive and process the pilot message from the BS.

        Normally, the pilot signals are used to estimate the current channel matrix H_k. We do not consider channel estimation in this framework, so we directly retrieve the channel matrix H_k from the received pilot message. 

        In case of non-coordinated beamforming, a suboptimal combining matrix G_k is computed based on the current channel. The state of the UT is updated to the current channel realization.

        Parameters
        ----------
        rx_pilots_msg : ReceivePilotMessage
            The pilot message received by the UT.
        """
        
        # Retrieve the channel estimate from the received pilot message.
        H_k = rx_pilots_msg.H_k
        
        # Compute the combining matrix for this UT based on the current channel realization. 
        G_k = self.combiner.compute(H_k)

        # Update the state of the UT to the current channel realization.
        self.state = UserTerminalState(H_k=H_k, G_k=G_k, c_type_k=None, P_k=None, ibr_k=None, Ns_k=None)
        
        return
    
    def transmit_feedback(self) -> TransmitFeedbackMessage:
        """
        Transmit the feedback message from the UT through the channel.

        The feedback message contains the effective channel matrix for this UT and the UT ID.
        In case of non-coordinated beamforming, the effective channel matrix is equal to the product of the combining matrix G_k and the channel matrix H_k. In case of coordinated beamforming, the effective channel matrix is equal to the channel matrix H_k.

        Returns
        -------
        tx_fb_msg : TransmitFeedbackMessage
            The feedback message transmitted by the UT.
        """

        tx_fb_msg = TransmitFeedbackMessage(ut_id=self.ut_id, H_eff_k= self.state.G_k @ self.state.H_k)
        return tx_fb_msg

    def receive_feedforward(self, rx_ff_msg: ReceiveFeedforwardMessage) -> None:
        """
        Receive and process the feedforward message from the BS.

        The feedforward message contains the constellation type c_type_k, the power allocation P_k, the information bit rates ibr_k, and the number of data streams Ns_k of this UT, for the current channel realization and SNR. In case of coordinated beamforming, it also contains the combining matrix G_k for this UT.

        Parameters
        ----------
        rx_ff_msg : ReceiveFeedforwardMessage
            The feedforward message received by the UT.
        """
        
        # Update the state of the UT to the current channel realization.
        if rx_ff_msg.G_k is not None: self.state.G_k = rx_ff_msg.G_k
        self.state.c_type_k = rx_ff_msg.c_type_k
        self.state.P_k = rx_ff_msg.P_k
        self.state.ibr_k = rx_ff_msg.ibr_k
        self.state.Ns_k = rx_ff_msg.Ns_k

        return

    def receive(self, y_k: ComplexArray) -> list[BitArray]:
        """
        Simulate the receive processing chain of the UT on the received signal y_k to obtain the estimated bitstreams b_k_s_hat. 
        
        The processing chain consists of the following steps:
        1. Combining - apply the combining matrix G_k to the received signal y_k to obtain the scaled decision variables z_k_s
        2. Power Deallocation - invert the power allocation applied by the base station to obtain the decision variables u_k_s
        3. Detection - estimate the transmitted symbol streams a_k_s based on the decision variables u_k_s
        4. Demapping - convert the estimated symbol streams a_k_s_hat to the corresponding estimated bit streams b_k_s_hat
        5. Bit Deallocation - reconstruct the estimated bitstream from the estimated bit streams b_k_s_hat

        Parameters
        ----------
        y_k : ComplexArray, shape (Nr, num_symbols)
            The received signal at this UT.
        
        Returns
        -------
        rx_bits_list : list[BitArray], shape (Ns_k, ibr_k_s * num_symbols)
            The list of estimated bitstreams for each data stream s of this UT.
        """
        
        z_k = self.combiner.apply(y_k, self.state.G_k)
        u_k = self.power_deallocator.apply(z_k, self.state.P_k)
        a_hat_k = self.detector.apply(u_k, self.state.ibr_k)
        b_hat_k = self.demapper.apply(a_hat_k, self.state.ibr_k)
        rx_bits_list = self.bit_deallocator.apply(b_hat_k, self.state.ibr_k)
        
        return rx_bits_list
