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
    system_ref_numbers = [
        "3_sl_2on4.4.5.3", "3_sl_3on4.4.5.3", "3_sl_4on4.4.5.3", "3_sl_5on4.4.5.3", "3_sl_6on4.4.5.3", "3_sl_7on4.4.5.3", "3_sl_8on4.4.5.3",
        "3_sl_2on4.1.5.3", "3_sl_3on4.1.5.3", "3_sl_4on4.1.5.3", "3_sl_5on4.1.5.3", "3_sl_6on4.1.5.3", "3_sl_7on4.1.5.3", "3_sl_8on4.1.5.3",
    ]
    
    # RUN OR LOAD YOUR SIMULATIONS HERE.
    results = main(sim_ref_numbers, system_ref_numbers)
    
    # PLOT THE RESULTS HERE.
    # SimResultManager.plot_system_performance_comparison(results, label_type="RTT", ibr=False, ber=False)

    
    
    
    # import matplotlib.pyplot as plt

    # # order
    # sys_ref_numbers_p = [
    #     ["3_sl_2on4_p2.4.5.3", "3_sl_2on4_p3.4.5.3", "3_sl_2on4_p4.4.5.3", "3_sl_2on4_p5.4.5.3", "3_sl_2on4_p6.4.5.3", "3_sl_2on4_p7.4.5.3"],
    #     ["3_sl_3on4_p2.4.5.3", "3_sl_3on4_p3.4.5.3", "3_sl_3on4_p4.4.5.3", "3_sl_3on4_p5.4.5.3", "3_sl_3on4_p6.4.5.3", "3_sl_3on4_p7.4.5.3"],
    #     ["3_sl_4on4_p2.4.5.3", "3_sl_4on4_p3.4.5.3", "3_sl_4on4_p4.4.5.3", "3_sl_4on4_p5.4.5.3", "3_sl_4on4_p6.4.5.3", "3_sl_4on4_p7.4.5.3"],
    #     ["3_sl_5on4_p2.4.5.3", "3_sl_5on4_p3.4.5.3", "3_sl_5on4_p4.4.5.3", "3_sl_5on4_p5.4.5.3", "3_sl_5on4_p6.4.5.3", "3_sl_5on4_p7.4.5.3"],
    # ]

    # # window length
    # sys_ref_numbers_w = [
    #     ["3_sl_2on4_w025.4.5.3", "3_sl_2on4_w050.4.5.3", "3_sl_2on4_w075.4.5.3", "3_sl_2on4_w100.4.5.3", "3_sl_2on4_w125.4.5.3", "3_sl_2on4_w150.4.5.3"],
    #     ["3_sl_3on4_w025.4.5.3", "3_sl_3on4_w050.4.5.3", "3_sl_3on4_w075.4.5.3", "3_sl_3on4_w100.4.5.3", "3_sl_3on4_w125.4.5.3", "3_sl_3on4_w150.4.5.3"],
    #     ["3_sl_4on4_w025.4.5.3", "3_sl_4on4_w050.4.5.3", "3_sl_4on4_w075.4.5.3", "3_sl_4on4_w100.4.5.3", "3_sl_4on4_w125.4.5.3", "3_sl_4on4_w150.4.5.3"],
    #     ["3_sl_5on4_w025.4.5.3", "3_sl_5on4_w050.4.5.3", "3_sl_5on4_w075.4.5.3", "3_sl_5on4_w100.4.5.3", "3_sl_5on4_w125.4.5.3", "3_sl_5on4_w150.4.5.3"],
    # ]

    # results = [main(sim_ref_numbers, sys_ref_number) for sys_ref_number in sys_ref_numbers_w]
    
    # fig_R, ax_R = plt.subplots(figsize=(6, 3.75))

    # colors = ["tab:green", "tab:red", "tab:purple", "tab:brown"]

    # x = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5])
    # for i, sim_results in enumerate(results):
    #     y = []
    #     for sim_result in sim_results:
    #         R = sim_result.simulation_results[0].R
    #         y.append(R)
    #     ax_R.plot(x, y, marker="o", linestyle="-", color=colors[i % len(colors)], label=fr"$\tau_{{\mathrm{{CSI}}}} = \frac{{{i+2}}}{{4}} \, T_c^{{\mathrm{{NLoS}}}}$")
    
    
    # ax_R.set_xlabel(r"context window length $[\mathrm{s}/T_c]$")
    # ax_R.set_ylabel(r"SR [bits/channel use]")
    # ax_R.set_xlim(0, 1.75)
    # ax_R.set_xticks(x)
    # ax_R.set_ylim(0, None)
    # ax_R.grid(True, which="both", linestyle="--", alpha=0.6)
    # ax_R.legend()
    # fig_R.tight_layout()

    # plot_filename = Path(__file__).resolve().parents[0] / "report" / "plots" / "window_comparison.png"
    # fig_R.savefig(plot_filename, dpi=300)
    # print(f"\n Saved system R comparison plot to:\n {plot_filename}")
