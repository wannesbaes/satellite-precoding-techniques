# mu-mimo/mu_mimo/core/system.py

from __future__ import annotations
import numpy as np
from tqdm import tqdm

from ..types import *
from ..configs import *
from .results import *
from ..processing import *


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
        
        
        # Check if this simulation has already been executed. If so, load the results and return them.
        if SimResultManager.search_results(self.sim_config, self.system_config):
            sim_result = SimResultManager.load_results(self.sim_config, self.system_config)
            print("="*60 + "\n  MU-MIMO Downlink Simulation \n" + f"  Results from {sim_result.system_configs.name} for {sim_result.sim_configs.name} successfully loaded.\n" + "="*60)
            return sim_result
        else:
            print("\n" + "="*60 + "\n  MU-MIMO Downlink Simulation \n" + "="*60 + "\n" + self.system_config.display() + "\n" + self.sim_config.display() + "\n")
        
        # Run the simulation.
        simulation_results: list[SingleSnrSimResult] = []

        # Outer loop: Iterate over the SNR values.
        for snr in tqdm(self.sim_config.snr_values, desc="SNR values"):

            inner_loop_results: list[SingleSnrSimResult] = []
            bit_error_count = 0
            channel_realization_count = 0
            hard_stop = 0

            # Reset.
            self.mu_mimo_system.reset(snr)

            # Inner loop: Iterate over the channel realizations.
            while (channel_realization_count < self.sim_config.Mch_min) or (bit_error_count < self.sim_config.num_bit_errors) and (hard_stop < 2 * self.sim_config.Mch_min):

                # Configuration.
                stream_Rs = self.mu_mimo_system.configure()

                # Communication.
                tx_bits_list, rx_bits_list = self.mu_mimo_system.communicate(self.sim_config.Msv)

                # Store inner loop results.
                inner_loop_result = self._calculate_inner_loop_result(snr, stream_Rs, tx_bits_list, rx_bits_list)
                inner_loop_results.append(inner_loop_result)

                # Stopping criterion.
                channel_realization_count += 1
                bit_error_count += self._calculate_bit_error_count_update(inner_loop_result)
                hard_stop += 1

            # Store outer loop results.
            simulation_results.append(self._calculate_outer_loop_result(inner_loop_results))

        # Save simulation results.
        simulation_result = SimResult(
            sim_configs = self.sim_config,
            system_configs = self.system_config,
            simulation_results = simulation_results)
        SimResultManager.save_results(simulation_result)

        return simulation_result

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

    def _calculate_inner_loop_result(self, snr: float, stream_Rs: RealArray, tx_bits_list: list[list[BitArray]], rx_bits_list: list[list[BitArray]]) -> SingleSnrSimResult:
        """
        Calculate the performance metrics for a simulation corresponding to a single channel realization and SNR value.

        Parameters
        ----------
        snr : float
            The SNR value for this simulation.
        stream_Rs : RealArray, shape (K, Nr)
            The achievable rates of each UT and each stream for this channel realization (and SNR value and system configuration).
        tx_bits_list : list[list[BitArray]], shape (K, Ns_k, ibr_k_s * Msv)
            The list of bitstreams for each UT k and each data stream s that were transmitted by the BS.
        rx_bits_list : list[list[BitArray]], shape (K, Ns_k, ibr_k_s * Msv)
            The list of bitstreams for each UT k and each data stream s that were reconstructed by the UTs.
        
        Returns
        -------
        inner_loop_result : SingleSnrSimResult
            The performance metrics. (we refer to the SingleSnrSimResult dataclass for more details)
        """
        
        # Initialization.
        K = self.system_config.K
        Nr = self.system_config.Nr
        ibr = self.mu_mimo_system.bs.state.ibr
        Ns = self.mu_mimo_system.bs.state.Ns

        snr_dB = 10 * np.log10(snr)
        
        stream_ibrs = [np.zeros((Nr,), dtype=int) for k in range(K)]
        stream_becs = [np.full((Nr,), np.nan, dtype=float) for k in range(K)]
        stream_ars = [np.zeros((Nr,), dtype=int) for k in range(K)]

        ut_ibrs = np.empty((K,), dtype=int)
        ut_becs = np.empty((K,), dtype=float)
        ut_ars = np.empty((K,), dtype=float)
        ut_Rs = np.empty((K,), dtype=float)

        # Iteration.
        for k in range(K):

            for a_s in range(Ns[k]):
                active_streams = np.where(ibr[k*Nr : (k+1)*Nr] > 0)[0]
                stream_ibrs[k][active_streams[a_s]] = len(tx_bits_list[k][a_s]) // self.sim_config.Msv
                stream_becs[k][active_streams[a_s]] = np.sum(tx_bits_list[k][a_s] != rx_bits_list[k][a_s])
                stream_ars[k][active_streams[a_s]] = 1
            
            ut_ibrs[k] = np.sum(stream_ibrs[k])
            ut_becs[k] = np.nansum(stream_becs[k]) if not np.all(np.isnan(stream_becs[k])) else np.nan
            ut_ars[k] = 1 if ut_ibrs[k] > 0 else 0
            ut_Rs[k] = np.sum(stream_Rs[k])
        
        ibr = np.sum(ut_ibrs)
        bec = np.nansum(ut_becs) if not np.all(np.isnan(ut_becs)) else np.nan
        ar = 1 if ibr > 0 else 0
        R = np.sum(ut_Rs)

        stream_ars_avg = np.sum([np.sum(stream_ars[k]) for k in range(K)]) / (K*Nr)
        ut_ars_avg = np.mean(ut_ars)

        # Termination.
        inner_loop_result = SingleSnrSimResult(
            
            snr_dB = snr_dB,

            stream_ibrs = stream_ibrs,
            stream_becs = stream_becs,
            stream_ars = stream_ars,
            stream_Rs = stream_Rs,

            ut_ibrs = ut_ibrs,
            ut_becs = ut_becs,
            ut_ars = ut_ars,
            ut_Rs = ut_Rs,
            
            ibr = ibr,
            bec = bec,
            ar = ar,
            R = R,
            
            stream_ars_avg = stream_ars_avg,
            ut_ars_avg = ut_ars_avg,

            Msv = self.sim_config.Msv,
            Mch = 1
        )
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
        snr_dB = inner_loop_results[0].snr_dB
        if not all(snr_dB == ilr.snr_dB for ilr in inner_loop_results):
            raise ValueError("The SNR values of the inner loop results are not all the same. Please check the inner loop results to resolve this issue.")
        
        K = self.system_config.K
        

        # Iteration.

        # Result per stream (for single SNR value, averaged over channel realizations).
        stream_ibrs: list[IntArray] = []
        stream_becs: list[RealArray] = []
        stream_ars: list[BitArray] = []
        stream_Rs: list[RealArray] = []

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

            stream_Rs_inner_loops = np.stack([ilr.stream_Rs[k] for ilr in inner_loop_results], axis=0)
            stream_Rs_outer_loop = np.mean(stream_Rs_inner_loops, axis=0)
            stream_Rs.append(stream_Rs_outer_loop)


        # Result per UT (for single SNR value, averaged over channel realizations).
        ut_ibrs_inner_loops = np.stack([ilr.ut_ibrs for ilr in inner_loop_results], axis=0)
        ut_ibrs = np.mean(ut_ibrs_inner_loops, axis=0)

        ut_becs_inner_loops = np.stack([ilr.ut_becs for ilr in inner_loop_results], axis=0)
        ut_becs = np.nansum(ut_becs_inner_loops, axis=0)

        ut_ars_inner_loops = np.stack([ilr.ut_ars for ilr in inner_loop_results], axis=0)
        ut_ars = np.mean(ut_ars_inner_loops, axis=0)

        ut_Rs_inner_loops = np.stack([ilr.ut_Rs for ilr in inner_loop_results], axis=0)
        ut_Rs = np.mean(ut_Rs_inner_loops, axis=0)

        # Result system-wide (for single SNR value, averaged over channel realizations).
        ibrs_inner_loops = np.array([ilr.ibr for ilr in inner_loop_results])
        ibr = float(np.mean(ibrs_inner_loops))

        becs_inner_loops = np.array([ilr.bec for ilr in inner_loop_results])
        bec = float(np.nansum(becs_inner_loops))

        ars_inner_loops = np.array([ilr.ar for ilr in inner_loop_results])
        ar = float(np.mean(ars_inner_loops))

        Rs_inner_loops = np.array([ilr.R for ilr in inner_loop_results])
        R = float(np.mean(Rs_inner_loops))


        stream_ars_avg = float(np.mean([np.mean(stream_ars[k]) for k in range(K)]))
        ut_ars_avg = float(np.mean(ut_ars))


        # Termination.
        outer_loop_result = SingleSnrSimResult(
            
            snr_dB = snr_dB,

            stream_ibrs = stream_ibrs,
            stream_becs = stream_becs,
            stream_ars = stream_ars,
            stream_Rs = stream_Rs,

            ut_ibrs = ut_ibrs,
            ut_becs = ut_becs,
            ut_ars = ut_ars,
            ut_Rs = ut_Rs,

            ibr = ibr,
            bec = bec,
            ar = ar,
            R = R,

            stream_ars_avg = stream_ars_avg,
            ut_ars_avg = ut_ars_avg,

            Msv = self.sim_config.Msv,
            Mch = len(inner_loop_results)
        )
        return outer_loop_result

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

        self.system_config = system_config
        self.bs = BaseStation(Pt, B, K, Nt, system_config.base_station_configs, system_config.c_configs)
        self.channel = Channel(K, Nr, Nt, system_config.channel_configs)
        self.uts = [UserTerminal(k, Nr, system_config.user_terminal_configs) for k in range(K)]

    def reset(self, snr: float) -> None:
        """
        Resets the MU-MIMO system. 
        
        Before starting the simulation for a new SNR value, the system must be resetted.
        For more datail on the reset phase, we refer to the reset_state() methods of the channel.

        Parameters
        ----------
        snr : float
            The SNR value to reset the system to.
        """

        # Reset the channel state.
        self.channel.reset(snr)

        return

    def configure(self) -> None:
        """
        Configuration the MU-MIMO system. 
        
        First, the state of the system components (precoding matrices, combiners, equalizers, etc.) are cleared. Then, a new state is computed based on the current channel realization and SNR value.
        After configuration, the system is ready for communication.

        For more datails on the configuration phase, we refer to the receive, propagate and transmit methods of the pilot signals, feedback messages and feedforward messages of the BS, UTs and channel.

        Returns
        -------
        capacity : RealArray, shape (K, Nr)
            The capacity of each stream of each UT for the current channel realization, SNR value and system configurations.
        """

        # Clearing Phase.
        self.bs.clear_state()
        for ut in self.uts: ut.clear_state()
        self.channel.proceed()
        
        # Pilot Phase.
        tx_pilot_msg = self.bs.transmit_pilots()
        rx_pilot_msgs = self.channel.propagate_pilots(tx_pilot_msg)
        for ut in self.uts: ut.receive_pilots(rx_pilot_msgs[ut.ut_id])

        # Feedback Phase.
        tx_fb_msgs = [ut.transmit_feedback() for ut in self.uts]
        rx_fb_msg = self.channel.propagate_feedback(tx_fb_msgs)
        self.bs.receive_feedback(rx_fb_msg)

        # Feedforward Phase.
        tx_ff_msg = self.bs.transmit_feedforward()
        rx_ff_msgs = self.channel.propagate_feedforward(tx_ff_msg)
        for ut in self.uts: ut.receive_feedforward(rx_ff_msgs[ut.ut_id])

        return self._compute_capacity()

    def communicate(self, Msv: int) -> tuple[list[list[BitArray]], list[list[BitArray]]]:
        """
        Communication of the MU-MIMO system.

        Make sure the system has been configured first. After communication, performance evaluation can be performed.
        For more datails on the communication phase, we refer to the transmit method of the BS, the propagate method of the channel and the receive method of the UTs.

        Parameters
        ----------
        Msv : int
            The number of symbol vector transmissions for this channel realization and SNR.
        
        Returns
        -------
        tx_bits_list : list[list[BitArray]], shape (K, Ns_k, ibr_k_s * Msv)
            The list of bitstreams for each UT k and each data stream s that were transmitted by the BS.
        rx_bits_list : list[list[BitArray]], shape (K, Ns_k, ibr_k_s * Msv)
            The list of bitstreams for each UT k and each data stream s that were received by the UTs.
        """

        # Transmit from the BS.
        tx_bits_list, x = self.bs.transmit(Msv)

        # Propagate through the channel.
        y_k_list = self.channel.propagate(x)

        # Receive at the UTs.
        rx_bits_list = [ut.receive(y_k_list[ut.ut_id]) for ut in self.uts]

        return tx_bits_list, rx_bits_list

    def _compute_capacity(self) -> RealArray:
        r"""
        Compute the capacity of the MU-MIMO system for the current system cofiguration, channel realization and SNR value.

        .. math::
            
            R_{k, s} = 2 B \cdot \log_2 \left( 1 + \text{SINR}_{k, s} \right)

            \mathrm{SINR}_{k, s} = \frac
            {
                p_{k, s} \, \left| \left( \mathbf{W}_k \mathbf{H}_k \mathbf{F}_k \right)_{(s, s)} \right|^2
            }
            { 
                \displaystyle \sum_{\substack{s' = 1 \\ s' \neq s}}^{N_s} \; p_{k, s'} \, \left| \left( \mathbf{W}_k \mathbf{H}_k \mathbf{F}_k \right)_{(s, s')} \right|^2 \; + \; \sum_{\substack{k' = 1 \\ k' \neq k}}^{K}  \sum_{s'=1}^{N_s} \; p_{k', s'} \, \left| \left( \mathbf{W}_k \mathbf{H}_k \mathbf{F}_{k'} \right)_{(s, s')} \right|^2 \; + \; N_0 \, \left\| \left( \mathbf{W}_k \right)_{(s, s)} \right\|^2
            }

        Returns
        -------
        capacity : RealArray, shape (K, Nr)
            The capacity of each stream of each UT.
        """

        # Initialization.
        Pt = self.system_config.Pt
        B = self.system_config.B
        Nr = self.system_config.Nr
        K = self.system_config.K

        snr = self.channel.state.snr
        H = self.channel.state.H

        F = self.bs.state.F
        if self.bs.state.G is not None:
            G = self.bs.state.G
        else:
            G = np.zeros((K*Nr, K*Nr), dtype=complex)
            for ut in self.uts:
                k = ut.ut_id
                G[k*Nr: (k+1)*Nr, k*Nr: (k+1)*Nr] = ut.state.G_k
        
        # Compute the transfer matrix T = G @ H @ F.
        T = G @ H @ F
        
        # Compute the power of the noise, the interference, and the useful signal for each data stream.
        p_noise = (Pt / snr) * np.sum( np.abs(G)**2, axis=1 )
        p_interference = np.sum( np.abs( T - np.diag(np.diagonal(T)) )**2, axis=1 )
        p_useful = np.abs( np.diagonal(T) )**2

        # Compute the SINR for each data stream.
        sinr = p_useful / (p_interference + p_noise)
        # sinr = np.where((p_useful == 0) & (p_interference + p_noise == 0), 0, p_useful / (p_interference + p_noise))

        # Compute the achievable bit rates.
        capacity = 2*B * np.log2(1 + sinr)
        capacity = capacity.reshape((K, Nr))

        return capacity

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
        self.bit_loader: type[BitLoader] = configs.bit_loader
        self.mapper: type[Mapper] = configs.mapper
        self.precoder: type[Precoder] = configs.precoder

        # State.
        self.c_configs: ConstConfig = c_configs
        self.state: BaseStationState | None = None

    def clear_state(self) -> None:
        """
        Clears the state of the BS. 
        
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
        F, G, C_eq = self.precoder.compute(rx_fb_msg.csi, self.Pt, self.K)
        ibr, Ns = self.bit_loader.compute(rx_fb_msg.csi, F, G, self.c_configs, Pt=self.Pt, B=self.B)

        # Update the state of the BS to the current channel realization and SNR.
        self.state = BaseStationState(F=F, C_eq=C_eq, ibr=ibr, Ns=Ns, G=G)

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

        tx_ff_msg = TransmitFeedforwardMessage(c_type=self.c_configs.types, C_eq=self.state.C_eq, ibr=self.state.ibr, G=self.state.G)
        return tx_ff_msg

    def transmit(self, Msv: int) -> tuple[list[list[BitArray]], ComplexArray]:
        """
        Simulate the transmit processing chain of the BS to obtain the transmitted signal x.

        The processing chain consists of the following steps:

        1. Bit Allocation - generate the bitstreams b_s for each data stream s based on the information bit rates ibr and the number of symbols to be transmitted Msv.
        2. Mapping - convert the bitstreams b_s to the corresponding data symbol streams a_s based on the modulation scheme (determined by the information bit rates ibr).
        3. Precoding - apply the precoding matrix F to the symbol streams a_s to obtain the transmitted signal x.

        Parameters
        ----------
        Msv : int
            The number of symbol vector transmissions for this channel realization and SNR.

        Returns
        -------
        tx_bits_list : list[list[BitArray]], shape (K, Ns_k, ibr_k_s * Msv)
            The list of bitstreams for each UT k and each data stream s.
        x : ComplexArray, shape (Nt, Msv)
            The transmitted signal.
        """

        tx_bits_list, b = self.bit_loader.apply(self.state.ibr, Msv, self.state.Ns)
        a = self.mapper.apply(b, self.state.ibr, self.c_configs.types, self.state.Ns)
        x = self.precoder.apply(a, self.state.F, self.state.ibr)

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

        # System dimensions.
        self.K = K
        self.Nr = Nr
        self.Nt = Nt

        # Channel model and noise model.
        self.channel_model: ChannelModel = configs.channel_model
        self.noise_model: NoiseModel = configs.noise_model

        # State.
        self.state: ChannelState | None = None

    def reset(self, snr: float) -> None:
        """
        Resets the channel.

        First, the state of the channel is reset.
        Then, the channel model and noise model are reset.
        After this method is called, the channel is ready for a new simulation for the specified SNR value.

        Parameters
        ----------
        snr : float
            The signal-to-noise ratio (SNR) to reset the channel to.
        """
        
        # Reset the state of the channel.
        self.state = ChannelState(snr=snr, H=None)

        # Reset the channel model and noise model.
        self.channel_model.reset()
        self.noise_model.reset()

        return

    def proceed(self) -> None:
        """
        Proceed to the next channel realization.

        This method should be called when the coherence time of the channel is passed.
        When this method is called, the system is ready to continue the configuration by sending the configuration messages.
        """
        self.state.H = self.channel_model.proceed()
        return

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

        Notes
        -----
        Because we do not consider channel estimation at this point, the pilot messages contain the exact current channel state information, retrieved from the channel model! The receiver does not have to do any estimation. 
        Also, in case a satellite channel, the CSI given to the receiver is a delayed compared to the channel used to propagate the information data.
        """

        H = self.channel_model.get_channel()
        rx_pilot_msgs = [ReceivePilotMessage(H_k = H[k*self.Nr:(k+1)*self.Nr, :]) for k in range(self.K)]
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
        rx_fb_msg = ReceiveFeedbackMessage(csi=ChannelStateInformation(snr=self.state.snr, H_eff=H_eff))

        return rx_fb_msg

    def propagate_feedforward(self, tx_ff_msg: TransmitFeedforwardMessage) -> list[ReceiveFeedforwardMessage]:
        """
        Propagates the feedforward messages from the BS to the UTs.

        The feedforward message transmitted by the BS is split into K different feedforward messages, one for each UT. Each feedforward message contains the constellation type, the equalization coefficients, the information bit rates, the number of data streams for each UT and, in case of coordinated beamforming, the combining matrices for each UT for the current channel realization and SNR.

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
        C_eq = tx_ff_msg.C_eq
        ibr = tx_ff_msg.ibr
        G = tx_ff_msg.G

        # Split the transmitted feedforward message elemtens into K different elements for each UT.
        C_eq_k_list = [ C_eq[k*self.Nr : (k+1)*self.Nr] for k in range(self.K)]
        ibr_k_list  = [ ibr[k*self.Nr : (k+1)*self.Nr] for k in range(self.K)]
        G_k_list    = [ G[k*self.Nr:(k+1)*self.Nr, k*self.Nr:(k+1)*self.Nr] for k in range(self.K)] if G is not None else [None]*self.K

        # Generate the feedforward messages that will be received by the UTs.
        rx_ff_msgs = [ReceiveFeedforwardMessage(ut_id=k, c_type_k=c_type[k], C_eq_k=C_eq_k_list[k], ibr_k=ibr_k_list[k], G_k=G_k_list[k]) for k in range(self.K) ]

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
        x : ComplexArray, shape (Nt, Msv)
            The transmitted signal.
        
        Returns
        -------
        y_k_list : list[ComplexArray], shape (K, Nr, Msv)
            The list of received signals for each UT k.
        """

        # Generate the noise samples according to the specified noise model.
        n = self.noise_model.get_noise(self.state.snr, x)

        # Apply the channel matrix H to the transmitted signal x and add the noise to obtain the received signal y.
        y_noiseless = self.channel_model.apply(x)
        y = self.noise_model.apply(y_noiseless, n)

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
        self.combiner: type[Combiner] = configs.combiner
        self.equalizer: type[Equalizer] = configs.equalizer
        self.detector: type[Detector] = configs.detector
        self.demapper: type[Demapper] = configs.demapper

        # State.
        self.state: UserTerminalState | None = None

        # UT ID.
        self.ut_id = ut_id

    def clear_state(self) -> None:
        """
        Clears the state of the user terminal.

        It clears the channel matrix H_k, the combining matrix G_k, the equalization coefficients C_eq_k, the information bit rates ibr_k and the number of data streams Ns_k for the previous channel realization and SNR.
        """
        
        self.state = None
        return

    def receive_pilots(self, rx_pilots_msg: ReceivePilotMessage) -> None:
        """
        Receive and process the pilot message from the BS.

        In case of non-coordinated beamforming, a suboptimal combining matrix G_k is computed based on the current channel. The state of the UT is updated to the current channel realization.

        Parameters
        ----------
        rx_pilots_msg : ReceivePilotMessage
            The pilot message received by the UT.
        
        Notes
        -----
        Normally, the pilot signals are used to estimate the current channel matrix H_k. We do not consider channel estimation in this framework, so we directly retrieve the channel matrix H_k from the received pilot message. 
        """
        
        # Retrieve the channel estimate from the received pilot message.
        H_k = rx_pilots_msg.H_k
        
        # Compute the combining matrix for this UT based on the current channel realization. 
        G_k = self.combiner.compute(H_k)

        # Update the state of the UT to the current channel realization.
        self.state = UserTerminalState(H_k=H_k, G_k=G_k, c_type_k=None, C_eq_k=None, ibr_k=None)
        
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
        self.state.C_eq_k = rx_ff_msg.C_eq_k
        self.state.ibr_k = rx_ff_msg.ibr_k

        return

    def receive(self, y_k: ComplexArray) -> list[BitArray]:
        """
        Simulate the receive processing chain of the UT on the received signal y_k to obtain the estimated bitstreams b_k_s_hat. 
        
        The processing chain consists of the following steps:
        
        1. Combining - apply the combining matrix G_k to the received signal y_k to obtain the scaled decision variables z_k_s
        2. Equalization - apply the equalization coefficients C_eq_k to the scaled decision variables z_k_s to obtain the decision variables u_k_s
        3. Detection - estimate the transmitted symbol streams a_k_s based on the decision variables u_k_s
        4. Demapping - convert the estimated symbol streams a_k_s_hat to the corresponding estimated bit streams b_k_s_hat
        5. Bit Deallocation - reconstruct the estimated bitstream from the estimated bit streams b_k_s_hat

        Parameters
        ----------
        y_k : ComplexArray, shape (Nr, Msv)
            The received signal at this UT.
        
        Returns
        -------
        rx_bits_list : list[BitArray], shape (Ns_k, ibr_k_s * Msv)
            The list of estimated bitstreams for each data stream s of this UT.
        """

        z_k = self.combiner.apply(y_k, self.state.G_k, self.state.ibr_k)
        u_k = self.equalizer.apply(z_k, self.state.C_eq_k, self.state.ibr_k)
        cpi_k_hat = self.detector.apply(u_k, self.state.ibr_k, self.state.c_type_k)
        b_k_hat = self.demapper.apply(cpi_k_hat, self.state.ibr_k)
        rx_bits = b_k_hat
        
        return rx_bits
