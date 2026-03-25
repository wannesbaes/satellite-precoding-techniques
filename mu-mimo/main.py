# mu-mimo/main.py

import json
from pathlib import Path
from mu_mimo import *

SIM_CONFIG_PATH = Path(__file__).parent / 'sim_configs.json'
SYSTEM_CONFIG_PATH = Path(__file__).parent / 'system_configs.json'


def main(sim_ref_numbers: list[str], sys_ref_numbers: list[str]) -> list[SimResult]:

    sim_configs = setup_sim_configs(sim_ref_numbers, SIM_CONFIG_PATH)
    system_configs = setup_sys_configs(sys_ref_numbers, SYSTEM_CONFIG_PATH)

    for sim_ref_number in sim_ref_numbers:
        for sys_ref_number in sys_ref_numbers:
            
            runner = SimulationRunner(sim_config=sim_configs[sim_ref_number], system_config=system_configs[sys_ref_number])
            result = runner.run()

            #SimResultManager.plot_system_performance(result)
            #SimResultManager.plot_ut_performance(result)
            #SimResultManager.plot_stream_performance(result)

    return

def main_ch_stats(sys_ref_numbers: list[str]) -> None:

    system_configs = setup_sys_configs(sys_ref_numbers, SYSTEM_CONFIG_PATH)

    for sys_ref_number in sys_ref_numbers:

        channel_statistics = ChannelStatistics(system_configs[sys_ref_number], num_channel_realizations=1_000_000)
        channel_statistics_data = channel_statistics.evaluate()

        channel_statistics._plot_streamchannel_pdf(num_uts=1, seperate_plots=True)
        channel_statistics._plot_streamchannel_ecdf(num_uts=1, seperate_plots=True)
    
    return


if __name__ == "__main__":

    # CHOOSE THE SIMULATION AND SYSTEM CONFIGURATIONS HERE.
    sim_ref_numbers = ["1.0"]
    sys_ref_numbers = [f"1.{i}.{j}.{k}" for i in range(1,4) for j in range(1, 5) for k in range(1, 4)]

    sys_ref_numbers_ZF = [f"1.1.{j}.{k}" for j in range(1, 5) for k in range(1, 4)]
    sys_ref_numbers_ZF_LSV = [f"1.2.{j}.{k}" for j in range(1, 5) for k in range(1, 4)]
    sys_ref_numbers_BD = [f"1.3.{j}.{k}" for j in range(1, 5) for k in range(1, 4)]
    
    # RUN OR LOAD YOUR SIMULATIONS HERE.
    main(sim_ref_numbers, sys_ref_numbers_BD)
    #main_ch_stats(sys_ref_numbers)
    #main_ergodic_capacity(sys_ref_numbers)
