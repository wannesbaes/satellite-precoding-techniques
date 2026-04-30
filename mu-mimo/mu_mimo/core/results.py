# mu-mimo/mu_mimo/core/results.py

from dataclasses import dataclass, field
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from pathlib import Path

from ..types import BitArray, IntArray, RealArray
from ..configs import SimConfig, SystemConfig


@dataclass
class SingleSnrSimResult:
    """
    The result of a simulation for a single SNR point, averaged over different channel realizations.
    
    Attributes
    ----------
    snr_dB : float
        The SNR value in dB for which the simulation results are reported.

    stream_ibrs : list[IntArray] (list of K arrays, each shape (Nr,))
        Per-UT per-stream information bit rates.
    stream_becs : list[RealArray] (list of K arrays, each shape (Nr,))
        Per-UT per-stream bit error counts.
    stream_ars : list[BitArray] (list of K arrays, each shape (Nr,))
        Per-UT per-stream stream activation rates.
    stream_Rs : list[RealArray] (list of K arrays, each shape (Nr,))
        Per-UT per-stream achievable rates.
    
    ut_ibrs : IntArray, shape (K,)
        Per-UT information bit rates.
    ut_becs : RealArray, shape (K,)
        Per-UT bit error counts.
    ut_ars : BitArray, shape (K,)
        Per-UT activation rates.
    ut_Rs : RealArray, shape (K,)
        Per-UT achievable rates.
    
    ibr : float
        System-wide information bit rate.
    bec : float
        System-wide bit error count.
    ar : float
        System-wide activation rate.
    R : float
        System-wide achievable rate.
    
    stream_ars_avg : float
        Average stream activation rate.
    ut_ars_avg : float
        Average UT activation rate.
    
    Msv : int
        The number of symbol vector transmissions per channel realization.
    Mch_tot : int
        The total number of channel realizations generated during the simulation for this SNR value.
    
    stream_bers : list[RealArray] (list of K arrays, each shape (Nr,)) | None
        Per-UT per-stream bit error rates. None if Mch_tot == 1.
    ut_bers : RealArray, shape (K,) | None
        Per-UT bit error rates. None if Mch_tot == 1.
    ber : float | None
        System-wide bit error rate. None if Mch_tot == 1.
    """
    
    snr_dB: float
    
    stream_ibrs : list[IntArray]
    stream_becs : list[RealArray]
    stream_ars : list[BitArray]
    stream_Rs : list[RealArray]
    
    ut_ibrs : IntArray
    ut_becs : RealArray
    ut_ars : BitArray
    ut_Rs : RealArray
    
    ibr : float
    bec : float
    ar : float
    R : float

    stream_ars_avg : float
    ut_ars_avg : float

    Msv : int
    Mch_tot : int


    stream_bers : list[RealArray] | None = None
    ut_bers : RealArray | None = None
    ber: float | None = None

    def __post_init__(self):

        if self.Mch_tot > 1:

            K = len(self.stream_ibrs)

            self.stream_bers = []
            for k in range(K):
                denom = self.stream_ibrs[k] * self.Msv * self.Mch_tot
                ber_k = np.where(denom > 0, self.stream_becs[k] / denom, np.nan)
                self.stream_bers.append(ber_k)
            
            ut_denom = self.ut_ibrs * self.Msv * self.Mch_tot
            self.ut_bers = np.where(ut_denom > 0, self.ut_becs / ut_denom, np.nan)
            
            total_denom = self.ibr * self.Msv * self.Mch_tot
            self.ber = self.bec / total_denom if total_denom > 0 else np.nan

@dataclass
class SimResult:
    """
    The results of a simulation.
    
    Attributes
    ----------
    sim_configs : SimConfig
        The configuration settings of the simulation.
    system_configs : SystemConfig
        The configuration settings of the system.
    simulation_results : list[SingleSnrSimResult]
        The list of simulation results for each SNR point.
    """
    
    sim_configs: SimConfig
    system_configs: SystemConfig
    simulation_results: list[SingleSnrSimResult]

@dataclass
class AnaResult:
    """
    The results of an analytical computation.

    Attributes
    ----------
    system_configs : SystemConfig
        The configuration settings of the system for which the analytical results are computed.
    
    snr_dB_BER : RealArray, shape (N_snr,)
        The SNR values in dB for which the BER results are computed.
    BER_system : RealArray, shape (N_snr,)
        The BER of the complete system for each SNR value.
    BER_uts : RealArray, shape (K, N_snr)
        The BER of each UT for each SNR value.
    BER_streams : RealArray, shape (K*Nr, N_snr)
        The BER of each datastream for each SNR value.

    snr_dB_R : RealArray, shape (N_snr,)
        The SNR values in dB for which the achievable rate results are computed.
    R_system : RealArray, shape (N_snr,)
        The expected achievable rates of the complete system for each SNR value.
    R_uts : RealArray, shape (K, N_snr)
        The expected achievable rates of each UT for each SNR value.
    R_streams : RealArray, shape (K*Nr, N_snr)
        The expected achievable rates of each datastream for each SNR value.
    """

    system_configs: SystemConfig

    snr_dB_BER: RealArray
    BER_system: RealArray | None
    BER_uts: RealArray | None
    BER_streams: RealArray | None

    snr_dB_R: RealArray
    R_system: RealArray | None
    R_uts: RealArray | None
    R_streams: RealArray | None


class SimResultManager:
    """
    The Simulation Result Manager.

    This class is responsible for managing the simulation results.
    This includes saving, loading, displaying, plotting, etc.
    """

    # LOAD & SAVE.

    @staticmethod
    def _filepath(sim_configs: SimConfig, system_configs: SystemConfig) -> Path:
        """
        Generate the file path where the simulation results should be saved.
        The filename is generated based on the name of the simulation configuration and the name of the system configuration.

        Parameters
        ----------
        sim_configs : SimConfig
            The configuration settings of the simulation.
        system_configs : SystemConfig
            The configuration settings of the system.
        
        Returns
        -------
        filepath : Path
            The filepath for the simulation results. 
        """

        # Create the results directory if it does not exist.
        results_dir = Path(__file__).resolve().parents[2] / "report" / "simulation_results"
        if "test" in sim_configs.name: results_dir = results_dir / "test_simulation_results"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Generate the filename based on the system and simulation configurations.
        filename = f"{system_configs.name}.npz"

        # Return the full file path.
        filepath = results_dir / filename
        return filepath
    
    @staticmethod
    def search_results(sim_configs: SimConfig, system_configs: SystemConfig) -> bool:
        """
        Search for previously executed simulation results with the same simulation and system configuration.
        If they exist, return True. Otherwise, return False.

        Parameters
        ----------
        sim_configs : SimConfig
            The configuration settings of the simulation.
        system_configs : SystemConfig
            The configuration settings of the system.
        
        Returns
        -------
        exists : bool
            True if the simulation results exist already, False otherwise.
        """
        filepath = SimResultManager._filepath(sim_configs, system_configs)
        return filepath.exists()
    
    @staticmethod
    def load_results(sim_configs: SimConfig, system_configs: SystemConfig) -> SimResult:
        """
        Load simulation results from a previously executed simulation with the same simulation and system configuration.

        Parameters
        ----------
        sim_configs : SimConfig
            The configuration settings of the simulation.
        system_configs : SystemConfig
            The configuration settings of the system.
        
        Returns
        -------
        sim_result : SimResult
            The loaded simulation results.
        """
        
        # Generate the appropiate file path.
        filepath = SimResultManager._filepath(sim_configs, system_configs)

        # Load the simulation results from the .npz file.
        loaded_data = np.load(filepath, allow_pickle=True)
        sim_result = SimResult(
            sim_configs = loaded_data["sim_configs"].item(),
            system_configs = loaded_data["system_configs"].item(),
            simulation_results = loaded_data["simulation_results"].tolist()
        )
        
        # Validate that the loaded simulation results match the current simulation and system configuration.
        if sim_configs != sim_result.sim_configs or system_configs != sim_result.system_configs:
            print("\nCurrent simulation configuration: \n", sim_configs.display())
            print("Loaded simulation configuration: \n", sim_result.sim_configs.display(), "\n\n")
            print("Current system configuration: \n", system_configs.display())
            print("Loaded system configuration: \n", sim_result.system_configs.display(), "\n\n")
            raise ValueError("The loaded simulation results do not match the current simulation and system configuration. However their filename suggests that they should. Please check the filename and the contents of the loaded simulation results to resolve this issue.")
        
        return sim_result

    @staticmethod
    def save_results(sim_result: SimResult) -> None:
        """
        Save the simulation results to a .npz file. 

        Parameters
        ----------
        sim_result : SimResult
            The simulation results to save.
        """

        filepath = SimResultManager._filepath(sim_result.sim_configs, sim_result.system_configs)
        np.savez(filepath,
            sim_configs = sim_result.sim_configs,
            system_configs = sim_result.system_configs,
            simulation_results = np.array(sim_result.simulation_results, dtype=object))
        
        print(f"\n Simulation results saved to:\n {filepath}")
        return
    
    # DISPLAY.

    @staticmethod
    def display(sim_result: SimResult, configs: bool = False, detailed: bool = True, precision: int = 3) -> str:
        """
        Display simulation results in a readable table format.

        Parameters
        ----------
        sim_result : SimResult
            The simulation results to display.
        configs : bool
            If True, also prints the system and simulation configuration settings in the header.
        detailed : bool
            If True, also prints per-UT metrics for each SNR point.
        precision : int
            Number of decimal digits for floating-point formatting.
        """


        lines: list[str] = []

        # Title.
        lines.append("\n")
        lines.append(f"=" * 60)
        lines.append(f"  MU-MIMO Downlink Simulation Results")
        lines.append(f"=" * 60)

        # System configuration summary.
        if configs: lines.append(f"\n{sim_result.system_configs.display()}")

        # Simulation configuration summary.
        if configs: lines.append(f"\n{sim_result.sim_configs.display()}")

        # Results table.
        lines.append(f"\n\n  Simulation results:\n")
        
        header = " " + f"{'SNR [dB]':>10} | {'BER':>10} | {'IBR':>10} | {'R':>10}" + (f" | {'UT AR avg':>12} | {'Stream AR avg':>12}" if detailed else "")
        lines.append( " " + "-" * len(header))
        lines.append(header)
        lines.append( " " + "-" * len(header))

        for sim_res in sim_result.simulation_results:
            ber_str = f"{sim_res.ber:.{precision}e}" if not np.isnan(sim_res.ber) else "N/A"
            R_str = f"{sim_res.R:.{precision}f}" if not np.isnan(sim_res.R) else "N/A"
            lines.append(" " + f"{int(sim_res.snr_dB):>10} | " + f"{ber_str:>10} | " + f"{int(sim_res.ibr):>10} | " + f"{R_str:>10}" + (f" | {sim_res.ut_ars_avg:>12.1%} | {sim_res.stream_ars_avg:>12.1%}" if detailed else "") )

            if detailed:
                lines.append("")
                for k in range(sim_result.system_configs.K):
                    ut_ber_str = f"{sim_res.ut_bers[k]:.{precision}e}" if not np.isnan(sim_res.ut_bers[k]) else "N/A"
                    ut_R_str = f"{sim_res.ut_Rs[k]:.{precision}f}" if not np.isnan(sim_res.ut_Rs[k]) else "N/A"
                    lines.append( f"        UT {k}: " + f"{ut_ber_str:>10} | " + f"{int(sim_res.ut_ibrs[k]):>10} | " + f"{ut_R_str:>10}" + (f" | {sim_res.ut_ars[k]:>12.1%}" if detailed else "") )
                lines.append(" " + "-" * len(header))

        # Return the formatted string.
        str_display = "\n".join(lines)
        return str_display

    # PLOT.

    @staticmethod
    def _plot_filename(sim_results: list[SimResult], plot_type: str) -> Path:
        """
        Generate the file path where the plot should be saved.
        The filename is generated based on the name of the simulation configuration and the name of the system configuration, and the type of plot.

        Parameters
        ----------
        sim_results : list[SimResult]
            A list of simulation results for which the plot is generated.
        plot_type : str
            A string indicating the type of plot.
        
        Returns
        -------
        filepath : Path
            The filepath for the plot.
        """
        
        # Create the plots directory if it does not exist.
        if len(sim_results) == 1: plots_dir = Path(__file__).resolve().parents[2] / "report" / "plots" / "reference systems" / f"{sim_results[0].system_configs.name}"
        else: plots_dir = Path(__file__).resolve().parents[2] / "report" / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Generate the filename based on the system and simulation configurations, and the type of plot.
        system_names = [f"{sim_result.system_configs.name}" for sim_result in sim_results]
        filename = f"{' - '.join(system_names)}" + f" -- {plot_type}" + f" -- {sim_results[0].sim_configs.name}" + ".png"

        # Return the full file path.
        filepath = plots_dir / filename
        return filepath

    @staticmethod
    def _plot_get_label(system_configs: SystemConfig, label_type: str = "default") -> str | None:
        """
        Generate the label for the plot legend based on the system configuration.\\
        Multiple labels for the same system configuration are available, and the label_type parameter is used to specify which one to use.

        Parameters
        ----------
        system_configs : SystemConfig
            The system configuration for which the label is generated.
        label_type : str
            The type of label.

            Possible options:
                - 'default': Default label type, which is the name of the system.
                - 'CH': The channel model used in the system. (e.g. 'Rayleigh')
                - 'PT': The precoding technique used in the system. (e.g. 'WMMSE')
                - 'BL': The bit loader configurations. (e.g. '4-QAM')
                - 'SD': The system dimensions. (e.g. 'Nt=8, Nr=2, K=2')

        Returns
        -------
        label : str | None
            The generated label for the plot legend.
        """
        
        label = None
        system_name = system_configs.name
        reference_number = system_name[12:]

        if label_type == "default":
            label = reference_number
        
        elif label_type == "CH":
            CH_mapping = {"1": "Rayleigh", "2": "Ricean TC Fading", "3": "Ricean TC Fading", "2s": "Satellite Channel", "3s": "Satellite Channel"}
            CH_number = (reference_number.split(".")[0]).split("_")[0]
            label = CH_mapping.get(CH_number, None)
        
        elif label_type == "RTT":
            RTT_mapping = {"0": "Instant CSI", "1": r"$T_{\text{RTT}} = \frac{1}{6} \, T^{\text{NLoS}}_c$", "2": r"$T_{\text{RTT}} = \frac{1}{3} \, T^{\text{NLoS}}_c$", "3": r"$T_{\text{RTT}} = \frac{1}{2} \, T^{\text{NLoS}}_c$", "4": r"$T_{\text{RTT}} = \frac{2}{3} \, T^{\text{NLoS}}_c$", "5": r"$T_{\text{RTT}} = \frac{5}{6} \, T^{\text{NLoS}}_c$"}
            RTT_number = (reference_number.split(".")[0]).split("_")[2]
            label = RTT_mapping.get(RTT_number, None)
        
        elif label_type == "CE":
            CE_mapping = {"1": "Perfect CSI", "2": "Outdated CSI", "3": "Predicted CSI"}
            CE_number = (reference_number.split(".")[0]).split("_")[0][0]
            label = CE_mapping.get(CE_number, None)
        
        elif label_type == "PT":
            PT_mapping = {"1": "ZF", "2": "ZF+LSV", "3": "BD", "4": "WMMSE"}
            PT_number = reference_number.split(".")[1]
            label = PT_mapping.get(PT_number, None)

        elif label_type == "BL":
            BL_mapping = {"1": "4-QAM", "2": "64-QAM", "3": r"$\approx R$-QAM", "4": r"$\approx \frac{3}{4}R$-QAM", "5": "16-QAM"}
            BL_number = reference_number.split(".")[2]
            label = BL_mapping.get(BL_number, None)

        elif label_type == "SD":
            SD_mapping = {"1": "Underloaded (8x2x2)", "2": "Fully loaded (2 UTs) (8x4x2)", "3": "Fully loaded (4 UTs) (8x2x4)"}
            SD_number = reference_number.split(".")[3]
            label = SD_mapping.get(SD_number, None)

        else:
            print(f"Warning: Unknown label type '{label_type}'. No label will be generated for the plot legend.")

        return label
    
    @staticmethod
    def _plot_curve(ax, x, y, ar, color, marker, label):

        x_v, y_v, ar_v = x[~np.isnan(y)], y[~np.isnan(y)], ar[~np.isnan(y)]
        if len(x_v) == 0: return

        if np.allclose(ar_v, 1.0):
            
            ax.plot(x_v, y_v, color=color, marker=marker, markeredgecolor=color, markerfacecolor='none', label=label)
        
        else:

            points = np.column_stack([x_v, y_v]).reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            seg_colors = np.tile(to_rgba(color), (len(segments), 1))
            seg_colors[:, 3] = (ar_v[:-1] + ar_v[1:]) / 2

            lc = LineCollection(segments, colors=seg_colors, linewidth=1.5)
            ax.add_collection(lc)

            for j in range(len(x_v)):
                ax.scatter(x_v[j], y_v[j], marker=marker, color=color, alpha=ar_v[j], s=36, zorder=3)

            ax.plot([], [], color=color, marker=marker, label=label)
        
        return ax
    
    @staticmethod
    def plot_system_performance(sim_result: SimResult, ber: bool = True, ibr: bool = True, R: bool = True, ana_result: AnaResult | None = None):
        """
        Plot the system performance.

        Saves three separate plots: system-wide BER, IBR, and achievable rate as a function of the SNR.
        The opacity of the points in the plots is proportional to the average data stream activation rate (only needed if not all stream AR are 100%).

        Parameters
        ----------
        sim_result : SimResult
            The simulation results to plot.
        ber : bool, optional
            Whether to plot and save the system-wide BER. Default is True.
        ibr : bool, optional
            Whether to plot and save the system-wide IBR. Default is True.
        R : bool, optional
            Whether to plot and save the system-wide achievable rate. Default is True.
        ana_result : AnaResult or None, optional
            If provided, the analytical results are overlaid on the simulation plots. Default is None.

        Returns
        -------
        fig_ber : matplotlib.figure.Figure
            The figure object of the BER plot.
        fig_ibr : matplotlib.figure.Figure
            The figure object of the IBR plot.
        fig_R : matplotlib.figure.Figure
            The figure object of the R plot.
        """

        # Extract data arrays.
        snr_dB = np.array([sim_res.snr_dB for sim_res in sim_result.simulation_results], dtype=float)
        bers = np.array([sim_res.ber for sim_res in sim_result.simulation_results], dtype=float)
        ibrs = np.array([sim_res.ibr for sim_res in sim_result.simulation_results], dtype=float)
        Rs = np.array([sim_res.R for sim_res in sim_result.simulation_results], dtype=float)
        ars = np.array([sim_res.ar for sim_res in sim_result.simulation_results], dtype=float)

        # BER vs SNR.
        if ber:
            fig_ber, ax_ber = plt.subplots(figsize=(6, 5))
            SimResultManager._plot_curve(ax_ber, snr_dB, bers, ars, color="tab:blue", marker="o", label="Simulation")
            if ana_result is not None and ana_result.BER_system is not None:
                ax_ber.plot(ana_result.snr_dB_BER, ana_result.BER_system, color="black", linestyle="--", label="Analytical")
            ax_ber.set_xlabel("SNR [dB]")
            ax_ber.set_ylabel("BER")
            ax_ber.set_yscale("log")
            ax_ber.set_ylim(None, 1)
            ax_ber.grid(True, which="both", linestyle="--", alpha=0.6)
            if ana_result is not None: ax_ber.legend()
            fig_ber.tight_layout()

            plot_filename = SimResultManager._plot_filename([sim_result], plot_type="system BER")
            fig_ber.savefig(plot_filename, dpi=300)
            print(f"\n Saved system BER plot to:\n {plot_filename}")

        # IBR vs SNR.
        if ibr:
            fig_ibr, ax_ibr = plt.subplots(figsize=(6, 5))
            SimResultManager._plot_curve(ax_ibr, snr_dB, ibrs, ars, color="tab:blue", marker="o", label="Simulation")
            ax_ibr.set_xlabel("SNR [dB]")
            ax_ibr.set_ylabel("IBR")
            ax_ibr.set_ylim(0, None)
            ax_ibr.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax_ibr.grid(True, which="both", linestyle="--", alpha=0.6)
            fig_ibr.tight_layout()

            plot_filename = SimResultManager._plot_filename([sim_result], plot_type="system IBR")
            fig_ibr.savefig(plot_filename, dpi=300)
            print(f"\n Saved system IBR plot to:\n {plot_filename}")

        # R vs SNR.
        if R:
            fig_R, ax_R = plt.subplots(figsize=(6, 5))
            SimResultManager._plot_curve(ax_R, snr_dB, Rs, ars, color="tab:blue", marker="o", label="Simulation")
            if ana_result is not None and ana_result.R_system is not None:
                ax_R.plot(ana_result.snr_dB_R, ana_result.R_system, color="black", linestyle="--", label="Analytical")
            ax_R.set_xlabel("SNR [dB]")
            ax_R.set_ylabel("R [bits/s/Hz]")
            ax_R.set_ylim(0, None)
            ax_R.grid(True, which="both", linestyle="--", alpha=0.6)
            if ana_result is not None: ax_R.legend()
            fig_R.tight_layout()

            plot_filename = SimResultManager._plot_filename([sim_result], plot_type="system R")
            fig_R.savefig(plot_filename, dpi=300)
            print(f"\n Saved system R plot to:\n {plot_filename}")

        figs = (fig_ber if ber else None, fig_ibr if ibr else None, fig_R if R else None)
        return figs

    @staticmethod
    def plot_ut_performance(sim_result: SimResult, ber: bool = True, ibr: bool = True, R: bool = True, ana_result: AnaResult | None = None):
        """
        Plot the performance of each UT in the system.

        Saves three separate plots: per-UT BER, IBR, and R as a function of the SNR.
        The opacity of the points in the plots is proportional to the average UT activation rate (only needed if not all UT ARs are 100%).
        Different UTs are plotted in different colors.

        Parameters
        ----------
        sim_result : SimResult
            The simulation results to plot.
        ber : bool, optional
            Whether to plot the BER (default is True).
        ibr : bool, optional
            Whether to plot the IBR (default is True).
        R : bool, optional
            Whether to plot the achievable rate (default is True).
        ana_result : AnaResult or None, optional
            If provided, the analytical results are overlaid on the simulation plots. Default is None.

        Returns
        -------
        fig_ber : matplotlib.figure.Figure
            The figure object of the BER plot.
        fig_ibr : matplotlib.figure.Figure
            The figure object of the IBR plot.
        fig_R : matplotlib.figure.Figure
            The figure object of the R plot.
        """

        # Extract data arrays.
        K = sim_result.system_configs.K
        colors = [f"C{k}" for k in range(K)]
        snr_dB = np.array([sim_res.snr_dB for sim_res in sim_result.simulation_results], dtype=float)
        bers = np.transpose(np.array([sim_res.ut_bers for sim_res in sim_result.simulation_results], dtype=float))
        ibrs = np.transpose(np.array([sim_res.ut_ibrs for sim_res in sim_result.simulation_results], dtype=float))
        ut_Rs = np.transpose(np.array([sim_res.ut_Rs for sim_res in sim_result.simulation_results], dtype=float))
        ut_ars = np.transpose(np.array([sim_res.ut_ars for sim_res in sim_result.simulation_results], dtype=float))

        # BER vs SNR.
        if ber:
            fig_ber, ax_ber = plt.subplots(figsize=(6, 5))
            for k in range(K):
                SimResultManager._plot_curve(ax_ber, snr_dB, bers[k], ut_ars[k], color=colors[k], marker="o", label=f"UT {k+1}")
            if ana_result is not None and ana_result.BER_uts is not None:
                for k in range(K):
                    ax_ber.plot(ana_result.snr_dB_BER, ana_result.BER_uts[k], color=colors[k], linestyle="--", label=f"UT {k+1} (analytical)")
            ax_ber.set_xlabel("SNR [dB]")
            ax_ber.set_ylabel("BER")
            ax_ber.set_yscale("log")
            ax_ber.set_ylim(None, 1)
            ax_ber.grid(True, which="both", linestyle="--", alpha=0.6)
            ax_ber.legend()
            fig_ber.tight_layout()

            plot_filename = SimResultManager._plot_filename([sim_result], plot_type="UT BER")
            fig_ber.savefig(plot_filename, dpi=300)
            print(f"\n Saved per-UT BER plot to:\n {plot_filename}")

        # IBR vs SNR.
        if ibr:
            fig_ibr, ax_ibr = plt.subplots(figsize=(6, 5))
            for k in range(K):
                SimResultManager._plot_curve(ax_ibr, snr_dB, ibrs[k], ut_ars[k], color=colors[k], marker="o", label=f"UT {k+1}")
            ax_ibr.set_xlabel("SNR [dB]")
            ax_ibr.set_ylabel("IBR")
            ax_ibr.set_ylim(0, None)
            ax_ibr.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax_ibr.grid(True, which="both", linestyle="--", alpha=0.6)
            ax_ibr.legend()
            fig_ibr.tight_layout()

            plot_filename = SimResultManager._plot_filename([sim_result], plot_type="UT IBR")
            fig_ibr.savefig(plot_filename, dpi=300)
            print(f"\n Saved per-UT IBR plot to:\n {plot_filename}")

        # Achievable Rate vs SNR.
        if R:
            fig_R, ax_R = plt.subplots(figsize=(6, 5))
            for k in range(K):
                SimResultManager._plot_curve(ax_R, snr_dB, ut_Rs[k], ut_ars[k], color=colors[k], marker="o", label=f"UT {k+1}")
            if ana_result is not None and ana_result.R_uts is not None:
                for k in range(K):
                    ax_R.plot(ana_result.snr_dB_R, ana_result.R_uts[k], color=colors[k], linestyle="--", label=f"UT {k+1} (analytical)")
            ax_R.set_xlabel("SNR [dB]")
            ax_R.set_ylabel("R [bits/s/Hz]")
            ax_R.set_ylim(0, None)
            ax_R.grid(True, which="both", linestyle="--", alpha=0.6)
            ax_R.legend()
            fig_R.tight_layout()

            plot_filename = SimResultManager._plot_filename([sim_result], plot_type="UT R")
            fig_R.savefig(plot_filename, dpi=300)
            print(f"\n Saved per-UT R plot to:\n {plot_filename}")

        figs = (fig_ber if ber else None, fig_ibr if ibr else None, fig_R if R else None)
        return figs

    @staticmethod
    def plot_stream_performance(sim_result: SimResult, ber: bool = True, ibr: bool = True, R: bool = True, ana_result: AnaResult | None = None):
        """
        Plot the performance of each stream in the system.

        Saves three separate plots: per-stream BER, IBR, and R as a function of the SNR.
        The opacity of the points in the plots is proportional to the average stream activation rate (only needed if not all stream ARs are 100%).
        Different UTs are plotted in different colors. Different streams are plotted with different markers.

        Parameters
        ----------
        sim_result : SimResult
            The simulation results to plot.
        ber : bool, optional
            Whether to plot the BER (default is True).
        ibr : bool, optional
            Whether to plot the IBR (default is True).
        R : bool, optional
            Whether to plot the R (default is True).
        ana_result : AnaResult or None, optional
            If provided, the analytical results are overlaid on the simulation plots. Default is None.

        Returns
        -------
        fig_ber : matplotlib.figure.Figure
            The figure object of the BER plot.
        fig_ibr : matplotlib.figure.Figure
            The figure object of the IBR plot.
        fig_R : matplotlib.figure.Figure
            The figure object of the R plot.
        """

        # Extract data arrays.
        K = sim_result.system_configs.K
        Nr = sim_result.system_configs.Nr
        colors = [f"C{k}" for k in range(K)]
        markers = ['o', 's', 'd', '*', '+', 'p', 'v', '^', '<', '>']
        
        snr_dB = np.array([sim_res.snr_dB for sim_res in sim_result.simulation_results], dtype=float)
        stream_bers = np.array([np.transpose(np.array([sim_res.stream_bers[k] for sim_res in sim_result.simulation_results], dtype=float)) for k in range(K)])
        stream_ibrs = np.array([np.transpose(np.array([sim_res.stream_ibrs[k] for sim_res in sim_result.simulation_results], dtype=float)) for k in range(K)])
        stream_Rs = np.array([np.transpose(np.array([sim_res.stream_Rs[k] for sim_res in sim_result.simulation_results], dtype=float)) for k in range(K)])
        stream_ars = np.array([np.transpose(np.array([sim_res.stream_ars[k] for sim_res in sim_result.simulation_results], dtype=float)) for k in range(K)])

        # BER vs SNR.
        if ber:
            fig_ber, ax_ber = plt.subplots(figsize=(6, 5))
            for k in range(K):
                for nr in range(Nr):
                    SimResultManager._plot_curve(ax_ber, snr_dB, stream_bers[k][nr], stream_ars[k][nr], color=colors[k], marker=markers[nr % len(markers)], label=f"UT {k+1}, Stream {nr+1}")
            if ana_result is not None and ana_result.BER_streams is not None:
                for k in range(K):
                    for nr in range(Nr):
                        ax_ber.plot(ana_result.snr_dB_BER, ana_result.BER_streams[k*Nr + nr], color=colors[k], marker=markers[nr % len(markers)], linestyle="--", label=f"UT {k+1}, Stream {nr+1} (analytical)")
            ax_ber.set_xlabel("SNR [dB]")
            ax_ber.set_ylabel("BER")
            ax_ber.set_yscale("log")
            ax_ber.set_ylim(None, 1)
            ax_ber.grid(True, which="both", linestyle="--", alpha=0.6)
            ax_ber.legend()
            fig_ber.tight_layout()

            plot_filename = SimResultManager._plot_filename([sim_result], plot_type="stream BER")
            fig_ber.savefig(plot_filename, dpi=300)
            print(f"\n Saved per-stream BER plot to:\n {plot_filename}")

        # IBR vs SNR.
        if ibr:
            fig_ibr, ax_ibr = plt.subplots(figsize=(6, 5))
            for k in range(K):
                for nr in range(Nr):
                    SimResultManager._plot_curve(ax_ibr, snr_dB, stream_ibrs[k][nr], stream_ars[k][nr], color=colors[k], marker=markers[nr % len(markers)], label=f"UT {k+1}, Stream {nr+1}")
            ax_ibr.set_xlabel("SNR [dB]")
            ax_ibr.set_ylabel("IBR")
            ax_ibr.set_ylim(0, None)
            ax_ibr.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax_ibr.grid(True, which="both", linestyle="--", alpha=0.6)
            ax_ibr.legend()
            fig_ibr.tight_layout()

            plot_filename = SimResultManager._plot_filename([sim_result], plot_type="stream IBR")
            fig_ibr.savefig(plot_filename, dpi=300)
            print(f"\n Saved per-stream IBR plot to:\n {plot_filename}")

        # Achievable Rate vs SNR.
        if R:
            fig_R, ax_R = plt.subplots(figsize=(6, 5))
            for k in range(K):
                for nr in range(Nr):
                    SimResultManager._plot_curve(ax_R, snr_dB, stream_Rs[k][nr], stream_ars[k][nr], color=colors[k], marker=markers[nr % len(markers)], label=f"UT {k+1}, Stream {nr+1}")
            if ana_result is not None and ana_result.R_streams is not None:
                for k in range(K):
                    for nr in range(Nr):
                        ax_R.plot(ana_result.snr_dB_R, ana_result.R_streams[k*Nr + nr], color=colors[k], marker=markers[nr % len(markers)], linestyle="--", label=f"UT {k+1}, Stream {nr+1} (analytical)")
            ax_R.set_xlabel("SNR [dB]")
            ax_R.set_ylabel("R [bit/s/Hz]")
            ax_R.set_ylim(0, None)
            ax_R.grid(True, which="both", linestyle="--", alpha=0.6)
            ax_R.legend()
            fig_R.tight_layout()

            plot_filename = SimResultManager._plot_filename([sim_result], plot_type="stream R")
            fig_R.savefig(plot_filename, dpi=300)
            print(f"\n Saved per-stream R plot to:\n {plot_filename}")

        figs = (fig_ber if ber else None, fig_ibr if ibr else None, fig_R if R else None)
        return figs

    @staticmethod
    def plot_system_performance_comparison(sim_results: list[SimResult], ber: bool = True, ibr: bool = True, R: bool = True, label_type: str = "default", ana_results: list[AnaResult] | None = None):
        """
        Plot the system performance of multiple systems for comparison.

        Saves three separate plots: system-wide BER, IBR, and R as a function of the SNR.
        Different systems are plotted in different colors.

        Parameters
        ----------
        sim_results : list[SimResult]
            A list of simulation results to plot.
        ber : bool, optional
            Whether to plot and save the system-wide BER comparison. Default is True.
        ibr : bool, optional
            Whether to plot and save the system-wide IBR comparison. Default is True.
        R : bool, optional
            Whether to plot and save the system-wide R comparison. Default is True.
        label_type: str, optional
            The type of label to use for the legend.

            Possible options:
                - 'default': Default label type, which is the name of the system.
                - 'CH': The channel model used in the system. (e.g. 'Rayleigh')
                - 'PT': The precoding technique used in the system. (e.g. 'WMMSE')
                - 'BL': The bit loader configurations. (e.g. '4-QAM')
                - 'SD': The system dimensions. (e.g. 'Nt=8, Nr=2, K=2')
        ana_results : list[AnaResult] or None, optional
            If provided, analytical results are overlaid on the simulation plots. Must have the same length as sim_results. Default is None.
        
        Returns
        -------
        fig_ber : matplotlib.figure.Figure
            The figure object of the system-wide BER comparison plot. None if ber=False.
        fig_ibr : matplotlib.figure.Figure
            The figure object of the system-wide IBR comparison plot. None if ibr=False.
        fig_R : matplotlib.figure.Figure
            The figure object of the system-wide R comparison plot. None if R=False.
        """
        
        # Validate the simulation configuration settings.
        if not all(sim_result.sim_configs == sim_results[0].sim_configs for sim_result in sim_results):
            raise ValueError("All results must have the same simulation configuration settings to be compared in the same plot.")
        
        # BER vs SNR.
        if ber:
            
            fig_ber, ax_ber = plt.subplots(figsize=(6, 5))
            
            for i, sim_result in enumerate(sim_results):
                snr_dB = np.array([sim_res.snr_dB for sim_res in sim_result.simulation_results], dtype=float)
                bers = np.array([sim_res.ber for sim_res in sim_result.simulation_results], dtype=float)
                stream_ars = np.array([sim_res.stream_ars_avg for sim_res in sim_result.simulation_results], dtype=float)
                label = SimResultManager._plot_get_label(sim_result.system_configs, label_type=label_type)
                SimResultManager._plot_curve(ax_ber, snr_dB, bers, stream_ars, color=f"C{i}", marker="o", label=label)
            
            if ana_results is not None:
                for i, ana_result in enumerate(ana_results):
                    if ana_result is not None and ana_result.BER_system is not None:
                        ax_ber.plot(ana_result.snr_dB_BER, ana_result.BER_system, color=f"C{i}", linestyle="--", label=f"{ana_result.system_configs.name} (analytical)")

            ax_ber.set_xlabel("SNR [dB]")
            ax_ber.set_ylabel("BER")
            ax_ber.set_yscale("log")
            ax_ber.set_ylim(None, 1)
            ax_ber.grid(True, which="both", linestyle="--", alpha=0.6)
            ax_ber.legend()
            fig_ber.tight_layout()

            plot_filename = SimResultManager._plot_filename(sim_results, plot_type="system BER comparison")
            fig_ber.savefig(plot_filename, dpi=300)
            print(f"\n Saved system BER comparison plot to:\n {plot_filename}")

        # IBR vs SNR.
        if ibr:
            
            fig_ibr, ax_ibr = plt.subplots(figsize=(6, 5))
            
            for i, sim_result in enumerate(sim_results):
                snr_dB = np.array([sim_res.snr_dB for sim_res in sim_result.simulation_results], dtype=float)
                ibrs = np.array([sim_res.ibr for sim_res in sim_result.simulation_results], dtype=float)
                stream_ars = np.array([sim_res.stream_ars_avg for sim_res in sim_result.simulation_results], dtype=float)
                label = SimResultManager._plot_get_label(sim_result.system_configs, label_type=label_type)
                SimResultManager._plot_curve(ax_ibr, snr_dB, ibrs, stream_ars, color=f"C{i}", marker="o", label=label)

            ax_ibr.set_xlabel("SNR [dB]")
            ax_ibr.set_ylabel("IBR")
            ax_ibr.set_ylim(0, None)
            ax_ibr.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax_ibr.grid(True, which="both", linestyle="--", alpha=0.6)
            ax_ibr.legend()
            fig_ibr.tight_layout()

            plot_filename = SimResultManager._plot_filename(sim_results, plot_type="system IBR comparison")
            fig_ibr.savefig(plot_filename, dpi=300)
            print(f"\n Saved system IBR comparison plot to:\n {plot_filename}")

        # Achievable Rate vs SNR.
        if R:

            fig_R, ax_R = plt.subplots(figsize=(6, 5))

            for i, sim_result in enumerate(sim_results):
                snr_dB = np.array([sim_res.snr_dB for sim_res in sim_result.simulation_results], dtype=float)
                Rs = np.array([sim_res.R for sim_res in sim_result.simulation_results], dtype=float)
                stream_ars = np.array([sim_res.stream_ars_avg for sim_res in sim_result.simulation_results], dtype=float)
                label = SimResultManager._plot_get_label(sim_result.system_configs, label_type=label_type)
                SimResultManager._plot_curve(ax_R, snr_dB, Rs, stream_ars, color=f"C{i}", marker="o", label=label)

            if ana_results is not None:
                for i, ana_result in enumerate(ana_results):
                    if ana_result is not None and ana_result.R_system is not None:
                        ax_R.plot(ana_result.snr_dB_R, ana_result.R_system, color=f"C{i}", linestyle="--", label=f"{ana_result.system_configs.name} (analytical)")

            ax_R.set_xlabel("SNR [dB]")
            ax_R.set_ylabel("R [bits/s/Hz]")
            ax_R.set_ylim(0, None)
            ax_R.grid(True, which="both", linestyle="--", alpha=0.6)
            ax_R.legend()
            fig_R.tight_layout()

            plot_filename = SimResultManager._plot_filename(sim_results, plot_type="system R comparison")
            fig_R.savefig(plot_filename, dpi=300)
            print(f"\n Saved system R comparison plot to:\n {plot_filename}")

        figs = (fig_ber if ber else None, fig_ibr if ibr else None, fig_R if R else None)
        return figs

    @staticmethod
    def plot_ut_performance_comparison(sim_results: list[SimResult], ber: bool = True, ibr: bool = True, R: bool = True, label_type: str = "default", ana_results: list[AnaResult] | None = None):
        """
        Plot the user terminal performance of multiple systems for comparison.

        Saves three separate plots: per-UT BER, IBR, and achievable rate as a function of the SNR.
        Different systems are plotted in different colors. Different UTs are plotted with different markers.

        Parameters
        ----------
        sim_results : list[SimResult]
            A list of simulation results to plot.
        ber : bool, optional
            Whether to plot and save the per-UT BER comparison. Default is True.
        ibr : bool, optional
            Whether to plot and save the per-UT IBR comparison. Default is True.
        R : bool, optional
            Whether to plot and save the per-UT achievable rate comparison. Default is True.
        label_type : str, optional
            The type of label to use for the plots.

            Possible options:
                - 'default': Default label type, which is the name of the system.
                - 'PT': The precoding technique used in the system. (e.g. 'WMMSE')
                - 'BL': The bit loader configurations. (e.g. '4-QAM')
                - 'SD': The system dimensions. (e.g. 'Nt=8, Nr=2, K=2')
        ana_results : list[AnaResult] or None, optional
            If provided, analytical results are overlaid on the simulation plots. Must have the same length as sim_results. Default is None.
        
        Returns
        -------
        fig_ber : matplotlib.figure.Figure
            The figure object of the per-UT BER comparison plot. None if ber=False.
        fig_ibr : matplotlib.figure.Figure
            The figure object of the per-UT IBR comparison plot. None if ibr=False.
        fig_R : matplotlib.figure.Figure
            The figure object of the per-UT achievable rate comparison plot. None if R=False.
        """

        # Validate the simulation configuration settings.
        if not all(sim_result.sim_configs == sim_results[0].sim_configs for sim_result in sim_results):
            raise ValueError("All results must have the same simulation configuration settings to be compared in the same plot.")

        # Marker per UT (constant across systems), color per system.
        markers = ["o", "s", "d", "*", "+", "p", "v", "^", "<", ">"]

        # BER vs SNR.
        if ber:
            
            fig_ber, ax_ber = plt.subplots(figsize=(6, 5))
            
            for i, sim_result in enumerate(sim_results):
                snr_dB = np.array([sim_res.snr_dB for sim_res in sim_result.simulation_results], dtype=float)
                ut_bers = np.transpose(np.array([sim_res.ut_bers for sim_res in sim_result.simulation_results], dtype=float))
                ut_ars = np.transpose(np.array([sim_res.ut_ars for sim_res in sim_result.simulation_results], dtype=float))
                label = SimResultManager._plot_get_label(sim_result.system_configs, label_type=label_type)
                for k in range(sim_result.system_configs.K):
                    SimResultManager._plot_curve(ax_ber, snr_dB, ut_bers[k], ut_ars[k], color=f"C{i}", marker=markers[k % len(markers)], label=f"{label} - UT {k+1}")
            
            if ana_results is not None:
                for i, ana_result in enumerate(ana_results):
                    if ana_result is not None and ana_result.BER_uts is not None:
                        K = ana_result.BER_uts.shape[0]
                        label = SimResultManager._plot_get_label(ana_result.system_configs, label_type=label_type)
                        for k in range(K):
                            ax_ber.plot(ana_result.snr_dB_BER, ana_result.BER_uts[k], color=f"C{i}", linestyle="--", label=f"{label} - UT {k+1} (analytical)")

            ax_ber.set_xlabel("SNR [dB]")
            ax_ber.set_ylabel("BER")
            ax_ber.set_yscale("log")
            ax_ber.set_ylim(0.5e-5, 1)
            ax_ber.grid(True, which="both", linestyle="--", alpha=0.6)
            ax_ber.legend()
            fig_ber.tight_layout()

            plot_filename = SimResultManager._plot_filename(sim_results, plot_type="UT BER comparison")
            fig_ber.savefig(plot_filename, dpi=300)
            print(f"\n Saved UT BER comparison plot to:\n {plot_filename}")

        # IBR vs SNR.
        if ibr:
            
            fig_ibr, ax_ibr = plt.subplots(figsize=(6, 5))
            
            for i, sim_result in enumerate(sim_results):
                snr_dB = np.array([sim_res.snr_dB for sim_res in sim_result.simulation_results], dtype=float)
                ut_ibrs = np.transpose(np.array([sim_res.ut_ibrs for sim_res in sim_result.simulation_results], dtype=float))
                ut_ars = np.transpose(np.array([sim_res.ut_ars for sim_res in sim_result.simulation_results], dtype=float))
                label = SimResultManager._plot_get_label(sim_result.system_configs, label_type=label_type)
                for k in range(sim_result.system_configs.K):
                    SimResultManager._plot_curve(ax_ibr, snr_dB, ut_ibrs[k], ut_ars[k], color=f"C{i}", marker=markers[k % len(markers)], label=f"{label} - UT {k+1}")

            ax_ibr.set_xlabel("SNR [dB]")
            ax_ibr.set_ylabel("IBR")
            ax_ibr.set_ylim(0, None)
            ax_ibr.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax_ibr.grid(True, which="both", linestyle="--", alpha=0.6)
            ax_ibr.legend()
            fig_ibr.tight_layout()

            plot_filename = SimResultManager._plot_filename(sim_results, plot_type="UT IBR comparison")
            fig_ibr.savefig(plot_filename, dpi=300)
            print(f"\n Saved UT IBR comparison plot to:\n {plot_filename}")

        # Achievable Rate vs SNR.
        if R:
            
            fig_R, ax_R = plt.subplots(figsize=(6, 5))
            
            for i, sim_result in enumerate(sim_results):
                snr_dB = np.array([sim_res.snr_dB for sim_res in sim_result.simulation_results], dtype=float)
                ut_Rs = np.transpose(np.array([sim_res.ut_Rs for sim_res in sim_result.simulation_results], dtype=float))
                ut_ars = np.transpose(np.array([sim_res.ut_ars for sim_res in sim_result.simulation_results], dtype=float))
                label = SimResultManager._plot_get_label(sim_result.system_configs, label_type=label_type)
                for k in range(sim_result.system_configs.K):
                    SimResultManager._plot_curve(ax_R, snr_dB, ut_Rs[k], ut_ars[k], color=f"C{i}", marker=markers[k % len(markers)], label=f"{label} - UT {k+1}")
            
            if ana_results is not None:
                for i, ana_result in enumerate(ana_results):
                    if ana_result is not None and ana_result.R_uts is not None:
                        label = SimResultManager._plot_get_label(ana_result.system_configs, label_type=label_type)
                        K = ana_result.R_uts.shape[0]
                        for k in range(K):
                            ax_R.plot(ana_result.snr_dB_R, ana_result.R_uts[k], color=f"C{i}", linestyle="--", label=f"{label} - UT {k+1} (analytical)")

            ax_R.set_xlabel("SNR [dB]")
            ax_R.set_ylabel("R [bits/s/Hz]")
            ax_R.set_ylim(0, None)
            ax_R.grid(True, which="both", linestyle="--", alpha=0.6)
            ax_R.legend()
            fig_R.tight_layout()

            plot_filename = SimResultManager._plot_filename(sim_results, plot_type="UT R comparison")
            fig_R.savefig(plot_filename, dpi=300)
            print(f"\n Saved UT R comparison plot to:\n {plot_filename}")

        figs = (fig_ber if ber else None, fig_ibr if ibr else None, fig_R if R else None)
        return figs

class AnaResultManager:
    """
    The Analytical Result Manager.

    This class is responsible for managing the analytical results.
    This includes saving, loading, displaying, plotting, etc.
    """

    # LOAD & SAVE.

    @staticmethod
    def _filepath(system_configs: SystemConfig) -> Path:
        """
        Generate the file path where the analytical results should be saved.
        The filename is generated based on the name of the system configuration.

        Parameters
        ----------
        system_configs : SystemConfig
            The configuration settings of the system.
        
        Returns
        -------
        filepath : Path
            The filepath for the analytical results.
        """

        # Create the results directory if it does not exist.
        results_dir = Path(__file__).resolve().parents[2] / "report" / "analytical_results"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Generate the filename based on the system configuration.
        filename = f"{system_configs.name}.npz"

        # Return the full file path.
        filepath = results_dir / filename
        return filepath

    @staticmethod
    def search_results(system_configs: SystemConfig) -> bool:
        """
        Search for previously computed analytical results with the same system configuration.
        If they exist, return True. Otherwise, return False.

        Parameters
        ----------
        system_configs : SystemConfig
            The configuration settings of the system.
        
        Returns
        -------
        exists : bool
            True if the analytical results exist already, False otherwise.
        """
        filepath = AnaResultManager._filepath(system_configs)
        return filepath.exists()

    @staticmethod
    def load_results(system_configs: SystemConfig) -> AnaResult:
        """
        Load analytical results from a previously computed analysis with the same system configuration.

        Parameters
        ----------
        system_configs : SystemConfig
            The configuration settings of the system.
        
        Returns
        -------
        ana_result : AnaResult
            The loaded analytical results.
        """

        # Generate the appropriate file path.
        filepath = AnaResultManager._filepath(system_configs)

        # Load the analytical results from the .npz file.
        loaded_data = np.load(filepath, allow_pickle=True)
        ana_result = AnaResult(
            system_configs = loaded_data["system_configs"].item(),
            
            snr_dB_R = loaded_data["snr_dB_R"],
            R_system = loaded_data["R_system"],
            R_uts = loaded_data["R_uts"],
            R_streams = loaded_data["R_streams"],
            
            snr_dB_BER = loaded_data["snr_dB_BER"],
            BER_system = loaded_data["BER_system"],
            BER_uts = loaded_data["BER_uts"],
            BER_streams = loaded_data["BER_streams"],
        )

        # Validate that the loaded analytical results match the current system configuration.
        if system_configs != ana_result.system_configs:
            raise ValueError("The loaded analytical results do not match the current system configuration. However their filename suggests that they should. Please check the filename and the contents of the loaded analytical results to resolve this issue.")

        return ana_result

    @staticmethod
    def save_results(ana_result: AnaResult) -> None:
        """
        Save the analytical results to a .npz file.

        Parameters
        ----------
        ana_result : AnaResult
            The analytical results to save.
        """

        filepath = AnaResultManager._filepath(ana_result.system_configs)
        np.savez(filepath,
            system_configs = ana_result.system_configs,
            
            snr_dB_R = ana_result.snr_dB_R,
            R_system = ana_result.R_system,
            R_uts = ana_result.R_uts,
            R_streams = ana_result.R_streams,
            
            snr_dB_BER = ana_result.snr_dB_BER,
            BER_system = ana_result.BER_system,
            BER_uts = ana_result.BER_uts,
            BER_streams = ana_result.BER_streams,
        )

        print(f"\n Analytical results saved to:\n {filepath}")
        return

    # DISPLAY.

    @staticmethod
    def display(ana_result: AnaResult, configs: bool = False, detailed: bool = True, precision: int = 3) -> str:
        """
        Display analytical results in a readable table format.

        Parameters
        ----------
        ana_result : AnaResult
            The analytical results to display.
        configs : bool
            If True, also prints the system configuration settings in the header.
        detailed : bool
            If True, also prints per-UT metrics for each SNR point.
        precision : int
            Number of decimal digits for floating-point formatting.
        """

        lines: list[str] = []

        # Title.
        lines.append("\n")
        lines.append(f"=" * 60)
        lines.append(f"  MU-MIMO Downlink Analytical Results")
        lines.append(f"=" * 60)

        # System configuration summary.
        if configs: lines.append(f"\n{ana_result.system_configs.display()}")

        # Achievable Rate results table.
        lines.append(f"\n\n  Achievable Rate results:\n")

        header_R = " " + f"{'SNR [dB]':>10} | {'R_system':>10}"
        if detailed:
            K = ana_result.R_uts.shape[0]
            for k in range(K):
                header_R += f" | {'R_UT' + str(k):>10}"
        lines.append(" " + "-" * len(header_R))
        lines.append(header_R)
        lines.append(" " + "-" * len(header_R))

        for i, snr in enumerate(ana_result.snr_dB_R):
            R_sys_str = f"{ana_result.R_system[i]:.{precision}f}" if not np.isnan(ana_result.R_system[i]) else "N/A"
            row = " " + f"{snr:>10.1f} | " + f"{R_sys_str:>10}"
            if detailed:
                for k in range(K):
                    R_ut_str = f"{ana_result.R_uts[k, i]:.{precision}f}" if not np.isnan(ana_result.R_uts[k, i]) else "N/A"
                    row += f" | {R_ut_str:>10}"
            lines.append(row)

        # BER results table.
        lines.append(f"\n\n  BER results:\n")

        header_BER = " " + f"{'SNR [dB]':>10} | {'BER_system':>12}"
        if detailed:
            K = ana_result.BER_uts.shape[0]
            for k in range(K):
                header_BER += f" | {'BER_UT' + str(k):>12}"
        lines.append(" " + "-" * len(header_BER))
        lines.append(header_BER)
        lines.append(" " + "-" * len(header_BER))

        for i, snr in enumerate(ana_result.snr_dB_BER):
            BER_sys_str = f"{ana_result.BER_system[i]:.{precision}e}" if not np.isnan(ana_result.BER_system[i]) else "N/A"
            row = " " + f"{snr:>10.1f} | " + f"{BER_sys_str:>12}"
            if detailed:
                for k in range(K):
                    BER_ut_str = f"{ana_result.BER_uts[k, i]:.{precision}e}" if not np.isnan(ana_result.BER_uts[k, i]) else "N/A"
                    row += f" | {BER_ut_str:>12}"
            lines.append(row)

        # Return the formatted string.
        str_display = "\n".join(lines)
        return str_display

    # PLOT.

    @staticmethod
    def _plot_filename(ana_results: list[AnaResult], plot_type: str) -> Path:
        """
        Generate the file path where the plot should be saved.
        The filename is generated based on the name of the system configuration and the type of plot.

        Parameters
        ----------
        ana_results : list[AnaResult]
            A list of analytical results for which the plot is generated.
        plot_type : str
            A string indicating the type of plot.
        
        Returns
        -------
        filepath : Path
            The filepath for the plot.
        """

        # Create the plots directory if it does not exist.
        if len(ana_results) == 1: plots_dir = Path(__file__).resolve().parents[2] / "report" / "plots" / "reference systems" / f"{ana_results[0].system_configs.name}"
        else: plots_dir = Path(__file__).resolve().parents[2] / "report" / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Generate the filename based on the system configurations and the type of plot.
        system_names = [f"{ana_result.system_configs.name}" for ana_result in ana_results]
        filename = f"{' - '.join(system_names)}" + f" -- {plot_type}" + ".png"

        # Return the full file path.
        filepath = plots_dir / filename
        return filepath

    @staticmethod
    def plot_system_performance(ana_result: AnaResult, ber: bool = True, R: bool = True):
        """
        Plot the system performance.

        Saves separate plots: system-wide BER and achievable rate as a function of the SNR.

        Parameters
        ----------
        ana_result : AnaResult
            The analytical results to plot.
        ber : bool, optional
            Whether to plot and save the system-wide BER. Default is True.
        R : bool, optional
            Whether to plot and save the system-wide achievable rate. Default is True.

        Returns
        -------
        fig_ber : matplotlib.figure.Figure | None
            The figure object of the BER plot. None if ber=False.
        fig_R : matplotlib.figure.Figure | None
            The figure object of the R plot. None if R=False.
        """

        # BER vs SNR.
        if ber and ana_result.BER_system is not None:
            fig_ber, ax_ber = plt.subplots(figsize=(6, 5))
            ax_ber.plot(ana_result.snr_dB_BER, ana_result.BER_system, color="tab:blue", marker="o", markeredgecolor="tab:blue", markerfacecolor='none')
            ax_ber.set_xlabel("SNR [dB]")
            ax_ber.set_ylabel("BER")
            ax_ber.set_yscale("log")
            ax_ber.set_ylim(None, 1)
            ax_ber.grid(True, which="both", linestyle="--", alpha=0.6)
            fig_ber.tight_layout()

            plot_filename = AnaResultManager._plot_filename([ana_result], plot_type="analytical system BER")
            fig_ber.savefig(plot_filename, dpi=300)
            print(f"\n Saved analytical system BER plot to:\n {plot_filename}")

        # R vs SNR.
        if R and ana_result.R_system is not None:
            fig_R, ax_R = plt.subplots(figsize=(6, 5))
            ax_R.plot(ana_result.snr_dB_R, ana_result.R_system, color="tab:blue", marker="o", markeredgecolor="tab:blue", markerfacecolor='none')
            ax_R.set_xlabel("SNR [dB]")
            ax_R.set_ylabel("R [bits/s/Hz]")
            ax_R.set_ylim(0, None)
            ax_R.grid(True, which="both", linestyle="--", alpha=0.6)
            fig_R.tight_layout()

            plot_filename = AnaResultManager._plot_filename([ana_result], plot_type="analytical system R")
            fig_R.savefig(plot_filename, dpi=300)
            print(f"\n Saved analytical system R plot to:\n {plot_filename}")

        figs = (fig_ber if ber else None, fig_R if R else None)
        return figs

    @staticmethod
    def plot_ut_performance(ana_result: AnaResult, ber: bool = True, R: bool = True):
        """
        Plot the performance of each UT in the system.

        Saves separate plots: per-UT BER and R as a function of the SNR.
        Different UTs are plotted in different colors.

        Parameters
        ----------
        ana_result : AnaResult
            The analytical results to plot.
        ber : bool, optional
            Whether to plot the BER (default is True).
        R : bool, optional
            Whether to plot the achievable rate (default is True).

        Returns
        -------
        fig_ber : matplotlib.figure.Figure | None
            The figure object of the BER plot. None if ber=False.
        fig_R : matplotlib.figure.Figure | None
            The figure object of the R plot. None if R=False.
        """

        K = ana_result.R_uts.shape[0]
        colors = [f"C{k}" for k in range(K)]

        # BER vs SNR.
        if ber and ana_result.BER_uts is not None:
            fig_ber, ax_ber = plt.subplots(figsize=(6, 5))
            for k in range(K):
                ax_ber.plot(ana_result.snr_dB_BER, ana_result.BER_uts[k], color=colors[k], marker="o", markeredgecolor=colors[k], markerfacecolor='none', label=f"UT {k}")
            ax_ber.set_xlabel("SNR [dB]")
            ax_ber.set_ylabel("BER")
            ax_ber.set_yscale("log")
            ax_ber.set_ylim(None, 1)
            ax_ber.grid(True, which="both", linestyle="--", alpha=0.6)
            ax_ber.legend()
            fig_ber.tight_layout()

            plot_filename = AnaResultManager._plot_filename([ana_result], plot_type="analytical UT BER")
            fig_ber.savefig(plot_filename, dpi=300)
            print(f"\n Saved analytical per-UT BER plot to:\n {plot_filename}")

        # R vs SNR.
        if R and ana_result.R_uts is not None:
            fig_R, ax_R = plt.subplots(figsize=(6, 5))
            for k in range(K):
                ax_R.plot(ana_result.snr_dB_R, ana_result.R_uts[k], color=colors[k], marker="o", markeredgecolor=colors[k], markerfacecolor='none', label=f"UT {k}")
            ax_R.set_xlabel("SNR [dB]")
            ax_R.set_ylabel("R [bits/s/Hz]")
            ax_R.set_ylim(0, None)
            ax_R.grid(True, which="both", linestyle="--", alpha=0.6)
            ax_R.legend()
            fig_R.tight_layout()

            plot_filename = AnaResultManager._plot_filename([ana_result], plot_type="analytical UT R")
            fig_R.savefig(plot_filename, dpi=300)
            print(f"\n Saved analytical per-UT R plot to:\n {plot_filename}")

        figs = (fig_ber if ber else None, fig_R if R else None)
        return figs

    @staticmethod
    def plot_stream_performance(ana_result: AnaResult, ber: bool = True, R: bool = True):
        """
        Plot the performance of each stream in the system.

        Saves separate plots: per-stream BER and R as a function of the SNR.
        Different UTs are plotted in different colors. Different streams are plotted with different markers.

        Parameters
        ----------
        ana_result : AnaResult
            The analytical results to plot.
        ber : bool, optional
            Whether to plot the BER (default is True).
        R : bool, optional
            Whether to plot the R (default is True).

        Returns
        -------
        fig_ber : matplotlib.figure.Figure | None
            The figure object of the BER plot. None if ber=False.
        fig_R : matplotlib.figure.Figure | None
            The figure object of the R plot. None if R=False.
        """

        K = ana_result.system_configs.K
        Nr = ana_result.system_configs.Nr
        colors = [f"C{k}" for k in range(K)]
        markers = ['o', 's', 'd', '*', '+', 'p', 'v', '^', '<', '>']

        # BER vs SNR.
        if ber and ana_result.BER_streams is not None:
            fig_ber, ax_ber = plt.subplots(figsize=(6, 5))
            for k in range(K):
                for nr in range(Nr):
                    idx = k * Nr + nr
                    ax_ber.plot(ana_result.snr_dB_BER, ana_result.BER_streams[idx], color=colors[k], marker=markers[nr % len(markers)], markeredgecolor=colors[k], markerfacecolor='none', label=f"UT {k}, Stream {nr}")
            ax_ber.set_xlabel("SNR [dB]")
            ax_ber.set_ylabel("BER")
            ax_ber.set_yscale("log")
            ax_ber.set_ylim(None, 1)
            ax_ber.grid(True, which="both", linestyle="--", alpha=0.6)
            ax_ber.legend()
            fig_ber.tight_layout()

            plot_filename = AnaResultManager._plot_filename([ana_result], plot_type="analytical stream BER")
            fig_ber.savefig(plot_filename, dpi=300)
            print(f"\n Saved analytical per-stream BER plot to:\n {plot_filename}")

        # R vs SNR.
        if R and ana_result.R_streams is not None:
            fig_R, ax_R = plt.subplots(figsize=(6, 5))
            for k in range(K):
                for nr in range(Nr):
                    idx = k * Nr + nr
                    ax_R.plot(ana_result.snr_dB_R, ana_result.R_streams[idx], color=colors[k], marker=markers[nr % len(markers)], markeredgecolor=colors[k], markerfacecolor='none', label=f"UT {k}, Stream {nr}")
            ax_R.set_xlabel("SNR [dB]")
            ax_R.set_ylabel("R [bits/s/Hz]")
            ax_R.set_ylim(0, None)
            ax_R.grid(True, which="both", linestyle="--", alpha=0.6)
            ax_R.legend()
            fig_R.tight_layout()

            plot_filename = AnaResultManager._plot_filename([ana_result], plot_type="analytical stream R")
            fig_R.savefig(plot_filename, dpi=300)
            print(f"\n Saved analytical per-stream R plot to:\n {plot_filename}")

        figs = (fig_ber if ber else None, fig_R if R else None)
        return figs

    @staticmethod
    def plot_system_performance_comparison(ana_results: list[AnaResult], ber: bool = True, R: bool = True):
        """
        Plot the system performance of multiple systems for comparison.

        Saves separate plots: system-wide BER and R as a function of the SNR.
        Different systems are plotted in different colors.

        Parameters
        ----------
        ana_results : list[AnaResult]
            A list of analytical results to plot.
        ber : bool, optional
            Whether to plot and save the system-wide BER comparison. Default is True.
        R : bool, optional
            Whether to plot and save the system-wide R comparison. Default is True.
        
        Returns
        -------
        fig_ber : matplotlib.figure.Figure | None
            The figure object of the system-wide BER comparison plot. None if ber=False.
        fig_R : matplotlib.figure.Figure | None
            The figure object of the system-wide R comparison plot. None if R=False.
        """

        # BER vs SNR.
        if ber:
            fig_ber, ax_ber = plt.subplots(figsize=(6, 5))
            for i, ana_result in enumerate(ana_results):
                if ana_result.BER_system is not None:
                    ax_ber.plot(ana_result.snr_dB_BER, ana_result.BER_system, color=f"C{i}", marker="o", markeredgecolor=f"C{i}", markerfacecolor='none', label=ana_result.system_configs.name)

            ax_ber.set_xlabel("SNR [dB]")
            ax_ber.set_ylabel("BER")
            ax_ber.set_yscale("log")
            ax_ber.set_ylim(None, 1)
            ax_ber.grid(True, which="both", linestyle="--", alpha=0.6)
            ax_ber.legend()
            fig_ber.tight_layout()

            plot_filename = AnaResultManager._plot_filename(ana_results, plot_type="analytical system BER comparison")
            fig_ber.savefig(plot_filename, dpi=300)
            print(f"\n Saved analytical system BER comparison plot to:\n {plot_filename}")

        # R vs SNR.
        if R:
            fig_R, ax_R = plt.subplots(figsize=(6, 5))
            for i, ana_result in enumerate(ana_results):
                if ana_result.R_system is not None:
                    ax_R.plot(ana_result.snr_dB_R, ana_result.R_system, color=f"C{i}", marker="o", markeredgecolor=f"C{i}", markerfacecolor='none', label=ana_result.system_configs.name)

            ax_R.set_xlabel("SNR [dB]")
            ax_R.set_ylabel("R [bits/s/Hz]")
            ax_R.set_ylim(0, None)
            ax_R.grid(True, which="both", linestyle="--", alpha=0.6)
            ax_R.legend()
            fig_R.tight_layout()

            plot_filename = AnaResultManager._plot_filename(ana_results, plot_type="analytical system R comparison")
            fig_R.savefig(plot_filename, dpi=300)
            print(f"\n Saved analytical system R comparison plot to:\n {plot_filename}")

        figs = (fig_ber if ber else None, fig_R if R else None)
        return figs

    @staticmethod
    def plot_ut_performance_comparison(ana_results: list[AnaResult], ber: bool = True, R: bool = True):
        """
        Plot the user terminal performance of multiple systems for comparison.

        Saves separate plots: per-UT BER and achievable rate as a function of the SNR.
        Different systems are plotted in different colors. Different UTs are plotted with different markers.

        Parameters
        ----------
        ana_results : list[AnaResult]
            A list of analytical results to plot.
        ber : bool, optional
            Whether to plot and save the per-UT BER comparison. Default is True.
        R : bool, optional
            Whether to plot and save the per-UT achievable rate comparison. Default is True.
        
        Returns
        -------
        fig_ber : matplotlib.figure.Figure | None
            The figure object of the per-UT BER comparison plot. None if ber=False.
        fig_R : matplotlib.figure.Figure | None
            The figure object of the per-UT R comparison plot. None if R=False.
        """

        # Marker per UT (constant across systems), color per system.
        markers = ["o", "s", "d", "*", "+", "p", "v", "^", "<", ">"]

        # BER vs SNR.
        if ber:
            fig_ber, ax_ber = plt.subplots(figsize=(6, 5))
            for i, ana_result in enumerate(ana_results):
                if ana_result.BER_uts is not None:
                    K = ana_result.BER_uts.shape[0]
                    for k in range(K):
                        ax_ber.plot(ana_result.snr_dB_BER, ana_result.BER_uts[k], color=f"C{i}", marker=markers[k % len(markers)], markeredgecolor=f"C{i}", markerfacecolor='none', label=f"{ana_result.system_configs.name} - UT {k}")

            ax_ber.set_xlabel("SNR [dB]")
            ax_ber.set_ylabel("BER")
            ax_ber.set_yscale("log")
            ax_ber.set_ylim(None, 1)
            ax_ber.grid(True, which="both", linestyle="--", alpha=0.6)
            ax_ber.legend()
            fig_ber.tight_layout()

            plot_filename = AnaResultManager._plot_filename(ana_results, plot_type="analytical UT BER comparison")
            fig_ber.savefig(plot_filename, dpi=300)
            print(f"\n Saved analytical UT BER comparison plot to:\n {plot_filename}")

        # R vs SNR.
        if R:
            fig_R, ax_R = plt.subplots(figsize=(6, 5))
            for i, ana_result in enumerate(ana_results):
                if ana_result.R_uts is not None:
                    K = ana_result.R_uts.shape[0]
                    for k in range(K):
                        ax_R.plot(ana_result.snr_dB_R, ana_result.R_uts[k], color=f"C{i}", marker=markers[k % len(markers)], markeredgecolor=f"C{i}", markerfacecolor='none', label=f"{ana_result.system_configs.name} - UT {k}")

            ax_R.set_xlabel("SNR [dB]")
            ax_R.set_ylabel("R [bits/s/Hz]")
            ax_R.set_ylim(0, None)
            ax_R.grid(True, which="both", linestyle="--", alpha=0.6)
            ax_R.legend()
            fig_R.tight_layout()

            plot_filename = AnaResultManager._plot_filename(ana_results, plot_type="analytical UT R comparison")
            fig_R.savefig(plot_filename, dpi=300)
            print(f"\n Saved analytical UT R comparison plot to:\n {plot_filename}")

        figs = (fig_ber if ber else None, fig_R if R else None)
        return figs


        
__all__ = [
    "SingleSnrSimResult", "SimResult", "SimResultManager",
    "AnaResult", "AnaResultManager",
]
