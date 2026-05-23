# mu-mimo/main.py

import numpy as np
import json
from pathlib import Path
from mu_mimo import *

SIM_CONFIG_PATH = Path(__file__).parent / 'sim_configs.json'
SYSTEM_CONFIG_PATH = Path(__file__).parent / 'system_configs.json'

# Matplotlib LaTeX style
import matplotlib
matplotlib.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 13,
    "axes.labelsize": 14,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
})


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
    sim_ref_numbers = ["1.test"]
    sys_ref_numbers =[
        "2_sl_0.4.5.3", "2_sl_1on4.4.5.3", "2_sl_1on2.4.5.3", "2_sl_3on4.4.5.3", "2_sl_1.4.5.3", "2_sl_5on4.4.5.3", "2_sl_3on2.4.5.3",
        "3_sl_0.4.5.3", "3_sl_1on4.4.5.3", "3_sl_1on2.4.5.3", "3_sl_3on4.4.5.3", "3_sl_1.4.5.3", "3_sl_5on4.4.5.3", "3_sl_3on2.4.5.3",
        "2_sl_0.1.5.3", "2_sl_1on4.1.5.3", "2_sl_1on2.1.5.3", "2_sl_3on4.1.5.3", "2_sl_1.1.5.3", "2_sl_5on4.1.5.3", "2_sl_3on2.1.5.3",
        "3_sl_0.1.5.3", "3_sl_1on4.1.5.3", "3_sl_1on2.1.5.3", "3_sl_3on4.1.5.3", "3_sl_1.1.5.3", "3_sl_5on4.1.5.3", "3_sl_3on2.1.5.3",        
    ]

    # RUN OR LOAD YOUR SIMULATIONS HERE.
    results = main(sim_ref_numbers, sys_ref_numbers)
    
    # PLOT THE RESULTS HERE.
    # SimResultManager.plot_system_performance_comparison(results, label_type="RTT")
