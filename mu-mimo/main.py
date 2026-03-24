# mu-mimo/main.py

import json
from pathlib import Path
import numpy as np
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

            results.append(result)

    return results


if __name__ == "__main__":

    # CHOOSE THE SIMULATION AND SYSTEM CONFIGURATIONS HERE.
    sim_ref_numbers = ["2.1"]
    sys_ref_numbers = ["1.1.1.1", "1.1.2.1", "1.1.3.1", "1.1.4.1", "1.2.1.1", "1.2.2.1", "1.2.3.1", "1.2.4.1"]
    
    # RUN OR LOAD YOUR SIMULATIONS HERE.
    results = main(sim_ref_numbers, sys_ref_numbers)


    # PLOT YOUR RESULTS HERE.
    for result in results:
        # ResultManager.plot_system_performance(result)
        pass