# mu-mimo/main.py

import numpy as np
import json
from pathlib import Path
from mu_mimo import *

SIM_CONFIG_PATH = Path(__file__).parent / 'sim_configs.json'
SYSTEM_CONFIG_PATH = Path(__file__).parent / 'system_configs.json'


def main(sim_ref_numbers: list[str], sys_ref_numbers: list[str]) -> list[SimResult]:

    results = []
    
    sim_configs = setup_sim_configs(sim_ref_numbers, SIM_CONFIG_PATH)
    system_configs = setup_sys_configs(sys_ref_numbers, SYSTEM_CONFIG_PATH)

    for sim_ref_number in sim_ref_numbers:
        for sys_ref_number in sys_ref_numbers:
            
            runner = SimulationRunner(sim_config=sim_configs[sim_ref_number], system_config=system_configs[sys_ref_number])
            result = runner.run()

            # SimResultManager.plot_system_performance(result)
            # SimResultManager.plot_ut_performance(result)
            # SimResultManager.plot_stream_performance(result)

            results.append(result)

    return results

def main_ch_stats(sys_ref_numbers: list[str]) -> None:

    system_configs = setup_sys_configs(sys_ref_numbers, SYSTEM_CONFIG_PATH)

    for sys_ref_number in sys_ref_numbers:

        channel_statistics = ChannelStatistics(system_configs[sys_ref_number], num_channel_samples=1_000_000)
        channel_statistics_data = channel_statistics.evaluate()

        channel_statistics.plot_streamchannel_gains_pdf()
        channel_statistics.plot_streamchannel_gains_ecdf()
    
    return


if __name__ == "__main__":

    # CHOOSE THE SIMULATION AND SYSTEM CONFIGURATIONS HERE.
    sim_ref_numbers = []
    system_ref_numbers = []

    system_ref_numbers_ZF_fixed    = [f"2{ch}.1.1.{sd}" for ch in ['t', 's1', 's2', 's4', 's8'] for sd in ['1', '2', '3']]
    system_ref_numbers_ZF_adaptive = [f"2{ch}.1.3.{sd}" for ch in ['t', 's1', 's2', 's4', 's8'] for sd in ['1', '2', '3']]

    system_ref_numbers_WMMSE_fixed    = [f"2{ch}.4.1.{sd}" for ch in ['t', 's1', 's2', 's4', 's8'] for sd in ['1', '2', '3']]
    system_ref_numbers_WMMSE_adaptive = [f"2{ch}.4.3.{sd}" for ch in ['t', 's1', 's2', 's4', 's8'] for sd in ['1', '2', '3']]

    
    # RUN OR LOAD YOUR SIMULATIONS HERE.
    results = main(["1.0"], system_ref_numbers_WMMSE_adaptive)
    
    # PLOT THE RESULTS HERE.
    # SimResultManager.plot_system_performance_comparison(results, label_type="default")
