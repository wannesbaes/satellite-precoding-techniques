# mu-mimo/main.py

import numpy as np
import matplotlib.pyplot as plt
from mu_mimo import *


def main():
    """Run a basic MU-MIMO downlink simulation."""

    # ──────────────────────────────────────────────
    # System parameters
    # ──────────────────────────────────────────────
    K = 2       # Number of user terminals
    Nt = 4      # Number of transmit antennas at the BS
    Nr = 2      # Number of receive antennas per UT
    Pt = 1.0    # Total transmit power [W]
    B = 1.0     # System bandwidth [Hz]

    # ──────────────────────────────────────────────
    # Simulation parameters
    # ──────────────────────────────────────────────
    snr_dB_values = np.array([10, 15, 20, 25])             # SNR range in dB
    num_channel_realizations = 10                    # Min channel realizations per SNR
    num_bit_errors = 5                              # Min bit errors per SNR (stopping criterion)
    num_bit_errors_scope = "system-wide"              # Scope of the bit error stopping criterion
    M = 100                                           # Number of symbol vectors per transmission

    # ──────────────────────────────────────────────
    # Constellation configuration
    # ──────────────────────────────────────────────
    c_configs = ConstConfig(
        types=["QAM"] * K,
        sizes=[2, 2],                                   # Determined adaptively
        capacity_fractions=np.array([0.5] * K),       # Use 50% of Shannon capacity
    )

    # ──────────────────────────────────────────────
    # Component configurations
    # ──────────────────────────────────────────────
    base_station_configs = BaseStationConfig(
        precoder=ZFPrecoder,
        bit_loader=FixedBitLoader,
        mapper=GrayCodeMapper,
    )

    user_terminal_configs = UserTerminalConfig(
        combiner=LSVCombiner,
        equalizer=Equalizer,
        detector=MDDetector,
        demapper=GrayCodeDemapper,
    )

    channel_configs = ChannelConfig(
        channel_model=IIDRayleighChannelModel,
        noise_model=CSAWGNNoiseModel,
    )

    # ──────────────────────────────────────────────
    # Build simulation and system configurations
    # ──────────────────────────────────────────────
    sim_config = SimConfig(
        snr_dB_values=snr_dB_values,
        num_channel_realizations=num_channel_realizations,
        num_bit_errors=num_bit_errors,
        num_bit_errors_scope=num_bit_errors_scope,
        M=M,
    )

    system_config = SystemConfig(
        Pt=Pt,
        B=B,
        K=K,
        Nt=Nt,
        Nr=Nr,
        c_configs=c_configs,
        base_station_configs=base_station_configs,
        user_terminal_configs=user_terminal_configs,
        channel_configs=channel_configs,
    )

    # ──────────────────────────────────────────────
    # Run simulation
    # ──────────────────────────────────────────────
    print("=" * 60)
    print("  MU-MIMO Downlink Simulation")
    print("=" * 60)
    print(f"  K  = {K} UTs, Nt = {Nt}, Nr = {Nr}")
    print(f"  Precoder  : {base_station_configs.precoder.__name__}")
    print(f"  Combiner  : {user_terminal_configs.combiner.__name__}")
    print(f"  BitLoader : {base_station_configs.bit_loader.__name__}")
    print(f"  Channel   : {channel_configs.channel_model.__name__}")
    print(f"  Noise     : {channel_configs.noise_model.__name__}")
    print(f"  SNR range : {snr_dB_values[0]} – {snr_dB_values[-1]} dB")
    print(f"  M = {M} symbols/transmission")
    print("=" * 60)

    runner = SimulationRunner(sim_config=sim_config, system_config=system_config)
    result = runner.run()

    # ──────────────────────────────────────────────
    # Display results
    # ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Simulation Complete")
    print("=" * 60)
    print(f"  Results saved to: {result.filename}")
    print("=" * 60)
    result.display(detailed=True)


if __name__ == "__main__":
    main()