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


if __name__ == "__main__":

    # CHOOSE THE SIMULATION AND SYSTEM CONFIGURATIONS HERE.
    sim_ref_numbers = []
    system_ref_numbers = []

    # RUN OR LOAD YOUR SIMULATIONS HERE.
    results = main(sim_ref_numbers, system_ref_numbers)
    
    # PLOT THE RESULTS HERE.
    SimResultManager.plot_system_performance_comparison(results, label_type="default", ibr=False)



# 1. OUTDATED CSI

# 1.1 TC SCALE = 1

# Fixed Bitloader (16-QAM),  Underloaded System,  ZF
system_ref_numbers = ["2_1_0.1.5.1", "2_1_1.1.5.1", "2_1_2.1.5.1", "2_1_3.1.5.1", "2_1_4.1.5.1", "2_1_5.1.5.1"]

# Fixed Bitloader (16-QAM),  Underloaded System,  WMMSE
system_ref_numbers = ["2_1_0.4.5.1", "2_1_1.4.5.1", "2_1_2.4.5.1", "2_1_3.4.5.1", "2_1_4.4.5.1", "2_1_5.4.5.1"]

# Fixed Bitloader (16-QAM),  Underloaded System,  ZF vs WMMSE
system_ref_numbers = [["2_1_0.1.5.1", "2_1_0.4.5.1"], ["2_1_1.1.5.1", "2_1_1.4.5.1"], ["2_1_2.1.5.1", "2_1_2.4.5.1"], ["2_1_3.1.5.1", "2_1_3.4.5.1"], ["2_1_4.1.5.1", "2_1_4.4.5.1"], ["2_1_5.1.5.1", "2_1_5.4.5.1"]]


# Fixed Bitloader (16-QAM),  Fully Loaded System (2 UTs),  ZF
system_ref_numbers = ["2_1_0.1.5.2", "2_1_1.1.5.2", "2_1_2.1.5.2", "2_1_3.1.5.2", "2_1_4.1.5.2", "2_1_5.1.5.2"]

# Fixed Bitloader (16-QAM),  Fully Loaded System (2 UTs),  WMMSE
system_ref_numbers = ["2_1_0.4.5.2", "2_1_1.4.5.2", "2_1_2.4.5.2", "2_1_3.4.5.2", "2_1_4.4.5.2", "2_1_5.4.5.2"]

# Fixed Bitloader (16-QAM),  Fully Loaded System (2 UTs),  ZF vs WMMSE
system_ref_numbers = [["2_1_0.1.5.2", "2_1_0.4.5.2"], ["2_1_1.1.5.2", "2_1_1.4.5.2"], ["2_1_2.1.5.2", "2_1_2.4.5.2"], ["2_1_3.1.5.2", "2_1_3.4.5.2"], ["2_1_4.1.5.2", "2_1_4.4.5.2"], ["2_1_5.1.5.2", "2_1_5.4.5.2"]]


# Fixed Bitloader (16-QAM),  Fully Loaded System (4 UTs),  ZF
system_ref_numbers = ["2_1_0.1.5.3", "2_1_1.1.5.3", "2_1_2.1.5.3", "2_1_3.1.5.3", "2_1_4.1.5.3", "2_1_5.1.5.3"]

# Fixed Bitloader (16-QAM),  Fully Loaded System (4 UTs),  WMMSE
system_ref_numbers = ["2_1_0.4.5.3", "2_1_1.4.5.3", "2_1_2.4.5.3", "2_1_3.4.5.3", "2_1_4.4.5.3", "2_1_5.4.5.3"]

# Fixed Bitloader (16-QAM),  Fully Loaded System (4 UTs),  ZF vs WMMSE
system_ref_numbers = [["2_1_0.1.5.3", "2_1_0.4.5.3"], ["2_1_1.1.5.3", "2_1_1.4.5.3"], ["2_1_2.1.5.3", "2_1_2.4.5.3"], ["2_1_3.1.5.3", "2_1_3.4.5.3"], ["2_1_4.1.5.3", "2_1_4.4.5.3"], ["2_1_5.1.5.3", "2_1_5.4.5.3"]]


# 1.2 TC SCALE = 2

