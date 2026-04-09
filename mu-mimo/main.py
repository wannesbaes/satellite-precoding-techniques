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
    sim_ref_numbers = ["1.0"]
    system_ref_numbers = [f"1.4.{i}.{j}" for i in range(1, 5) for j in range(1, 4)]
    
    system_ref_numbers_1 = [f"1.4.1.{j}" for j in range(1, 4)]
    system_ref_numbers_2 = [f"1.4.2.{j}" for j in range(1, 4)]
    system_ref_numbers_3 = [f"1.4.3.{j}" for j in range(1, 4)]
    system_ref_numbers_4 = [f"1.4.4.{j}" for j in range(1, 4)]
    
    # RUN OR LOAD YOUR SIMULATIONS HERE.
    results = main(sim_ref_numbers, system_ref_numbers_4)

    # PLOT THE RESULTS HERE.
    SimResultManager.plot_system_performance_comparison(results, label_type="PT")
    # SimResultManager.plot_ut_performance_comparison(results, label_type="default")
