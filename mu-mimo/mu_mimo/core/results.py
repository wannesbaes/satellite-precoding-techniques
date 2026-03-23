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
    
    M : int
        The number of symbol vector transmissions for each channel realization.
    num_channel_realizations : int
        The number of channel realizations that were simulated.
    
    stream_bers : list[RealArray] (list of K arrays, each shape (Nr,)) | None
        Per-UT per-stream bit error rates. None if num_channel_realizations == 1.
    ut_bers : RealArray, shape (K,) | None
        Per-UT bit error rates. None if num_channel_realizations == 1.
    ber : float | None
        System-wide bit error rate. None if num_channel_realizations == 1.
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

    M : int
    num_channel_realizations : int


    stream_bers : list[RealArray] | None = None
    ut_bers : RealArray | None = None
    ber: float | None = None

    def __post_init__(self):

        if self.num_channel_realizations > 1:

            K = len(self.stream_ibrs)

            self.stream_bers = []
            for k in range(K):
                denom = self.stream_ibrs[k] * self.M * self.num_channel_realizations
                ber_k = np.where(denom > 0, self.stream_becs[k] / denom, np.nan)
                self.stream_bers.append(ber_k)
            
            ut_denom = self.ut_ibrs * self.M * self.num_channel_realizations
            self.ut_bers = np.where(ut_denom > 0, self.ut_becs / ut_denom, np.nan)
            
            total_denom = self.ibr * self.M * self.num_channel_realizations
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


class ResultManager:
    """
    The Result Manager.

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
        results_dir.mkdir(parents=True, exist_ok=True)

        # Generate the filename based on the system and simulation configurations.
        filename = f"{system_configs.name} - {sim_configs.name}.npz"

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
        filepath = ResultManager._filepath(sim_configs, system_configs)
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
        filepath = ResultManager._filepath(sim_configs, system_configs)

        # Load the simulation results from the .npz file.
        loaded_data = np.load(filepath, allow_pickle=True)
        sim_result = SimResult(
            sim_configs = loaded_data["sim_configs"].item(),
            system_configs = loaded_data["system_configs"].item(),
            simulation_results = loaded_data["simulation_results"].tolist()
        )
        
        # Validate that the loaded simulation results match the current simulation and system configuration.
        if sim_configs != sim_result.sim_configs or system_configs != sim_result.system_configs:
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

        filepath = ResultManager._filepath(sim_result.sim_configs, sim_result.system_configs)
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
        plots_dir = Path(__file__).resolve().parents[2] / "report" / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Generate the filename based on the system and simulation configurations, and the type of plot.
        system_names = [f"{sim_result.system_configs.name}" for sim_result in sim_results]
        filename = f"{' - '.join(system_names)}" + f" -- {plot_type}" + f" -- {sim_results[0].sim_configs.name}" + ".png"

        # Return the full file path.
        filepath = plots_dir / filename
        return filepath

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
    def plot_system_performance(sim_result: SimResult, ber: bool = True, ibr: bool = True, R: bool = True):
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
            ResultManager._plot_curve(ax_ber, snr_dB, bers, ars, color="tab:blue", marker="o", label="")
            ax_ber.set_xlabel("SNR [dB]")
            ax_ber.set_ylabel("BER")
            ax_ber.set_yscale("log")
            ax_ber.set_ylim(0.5e-4, 1)
            ax_ber.grid(True, which="both", linestyle="--", alpha=0.6)
            fig_ber.tight_layout()

            plot_filename = ResultManager._plot_filename([sim_result], plot_type="system BER")
            fig_ber.savefig(plot_filename, dpi=300)
            print(f"\n Saved system BER plot to:\n {plot_filename}")

        # IBR vs SNR.
        if ibr:
            fig_ibr, ax_ibr = plt.subplots(figsize=(6, 5))
            ResultManager._plot_curve(ax_ibr, snr_dB, ibrs, ars, color="tab:blue", marker="o", label="")
            ax_ibr.set_xlabel("SNR [dB]")
            ax_ibr.set_ylabel("IBR")
            ax_ibr.set_ylim(0, None)
            ax_ibr.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax_ibr.grid(True, which="both", linestyle="--", alpha=0.6)
            fig_ibr.tight_layout()

            plot_filename = ResultManager._plot_filename([sim_result], plot_type="system IBR")
            fig_ibr.savefig(plot_filename, dpi=300)
            print(f"\n Saved system IBR plot to:\n {plot_filename}")

        # R vs SNR.
        if R:
            fig_R, ax_R = plt.subplots(figsize=(6, 5))
            ResultManager._plot_curve(ax_R, snr_dB, Rs, ars, color="tab:blue", marker="o", label="")
            ax_R.set_xlabel("SNR [dB]")
            ax_R.set_ylabel("R [bits/s/Hz]")
            ax_R.set_ylim(0, None)
            ax_R.grid(True, which="both", linestyle="--", alpha=0.6)
            fig_R.tight_layout()

            plot_filename = ResultManager._plot_filename([sim_result], plot_type="system R")
            fig_R.savefig(plot_filename, dpi=300)
            print(f"\n Saved system R plot to:\n {plot_filename}")

        figs = (fig_ber if ber else None, fig_ibr if ibr else None, fig_R if R else None)
        return figs

    @staticmethod
    def plot_ut_performance(sim_result: SimResult, ber: bool = True, ibr: bool = True, R: bool = True):
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
                ResultManager._plot_curve(ax_ber, snr_dB, bers[k], ut_ars[k], color=colors[k], marker="o", label=f"UT {k}")
            ax_ber.set_xlabel("SNR [dB]")
            ax_ber.set_ylabel("BER")
            ax_ber.set_yscale("log")
            ax_ber.set_ylim(0.5e-4, 1)
            ax_ber.grid(True, which="both", linestyle="--", alpha=0.6)
            ax_ber.legend()
            fig_ber.tight_layout()

            plot_filename = ResultManager._plot_filename([sim_result], plot_type="UT BER")
            fig_ber.savefig(plot_filename, dpi=300)
            print(f"\n Saved per-UT BER plot to:\n {plot_filename}")

        # IBR vs SNR.
        if ibr:
            fig_ibr, ax_ibr = plt.subplots(figsize=(6, 5))
            for k in range(K):
                ResultManager._plot_curve(ax_ibr, snr_dB, ibrs[k], ut_ars[k], color=colors[k], marker="o", label=f"UT {k}")
            ax_ibr.set_xlabel("SNR [dB]")
            ax_ibr.set_ylabel("IBR")
            ax_ibr.set_ylim(0, None)
            ax_ibr.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax_ibr.grid(True, which="both", linestyle="--", alpha=0.6)
            ax_ibr.legend()
            fig_ibr.tight_layout()

            plot_filename = ResultManager._plot_filename([sim_result], plot_type="UT IBR")
            fig_ibr.savefig(plot_filename, dpi=300)
            print(f"\n Saved per-UT IBR plot to:\n {plot_filename}")

        # Achievable Rate vs SNR.
        if R:
            fig_R, ax_R = plt.subplots(figsize=(6, 5))
            for k in range(K):
                ResultManager._plot_curve(ax_R, snr_dB, ut_Rs[k], ut_ars[k], color=colors[k], marker="o", label=f"UT {k}")
            ax_R.set_xlabel("SNR [dB]")
            ax_R.set_ylabel("R [bits/s/Hz]")
            ax_R.set_ylim(0, None)
            ax_R.grid(True, which="both", linestyle="--", alpha=0.6)
            ax_R.legend()
            fig_R.tight_layout()

            plot_filename = ResultManager._plot_filename([sim_result], plot_type="UT R")
            fig_R.savefig(plot_filename, dpi=300)
            print(f"\n Saved per-UT R plot to:\n {plot_filename}")

        figs = (fig_ber if ber else None, fig_ibr if ibr else None, fig_R if R else None)
        return figs

    @staticmethod
    def plot_stream_performance(sim_result: SimResult, ber: bool = True, ibr: bool = True, R: bool = True):
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
                    ResultManager._plot_curve(ax_ber, snr_dB, stream_bers[k][nr], stream_ars[k][nr], color=colors[k], marker=markers[nr % len(markers)], label=f"UT {k}, Stream {nr}")
            ax_ber.set_xlabel("SNR [dB]")
            ax_ber.set_ylabel("BER")
            ax_ber.set_yscale("log")
            ax_ber.set_ylim(0.5e-4, 1)
            ax_ber.grid(True, which="both", linestyle="--", alpha=0.6)
            ax_ber.legend()
            fig_ber.tight_layout()

            plot_filename = ResultManager._plot_filename([sim_result], plot_type="stream BER")
            fig_ber.savefig(plot_filename, dpi=300)
            print(f"\n Saved per-stream BER plot to:\n {plot_filename}")

        # IBR vs SNR.
        if ibr:
            fig_ibr, ax_ibr = plt.subplots(figsize=(6, 5))
            for k in range(K):
                for nr in range(Nr):
                    ResultManager._plot_curve(ax_ibr, snr_dB, stream_ibrs[k][nr], stream_ars[k][nr], color=colors[k], marker=markers[nr % len(markers)], label=f"UT {k}, Stream {nr}")
            ax_ibr.set_xlabel("SNR [dB]")
            ax_ibr.set_ylabel("IBR")
            ax_ibr.set_ylim(0, None)
            ax_ibr.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax_ibr.grid(True, which="both", linestyle="--", alpha=0.6)
            ax_ibr.legend()
            fig_ibr.tight_layout()

            plot_filename = ResultManager._plot_filename([sim_result], plot_type="stream IBR")
            fig_ibr.savefig(plot_filename, dpi=300)
            print(f"\n Saved per-stream IBR plot to:\n {plot_filename}")

        # Achievable Rate vs SNR.
        if R:
            fig_R, ax_R = plt.subplots(figsize=(6, 5))
            for k in range(K):
                for nr in range(Nr):
                    ResultManager._plot_curve(ax_R, snr_dB, stream_Rs[k][nr], stream_ars[k][nr], color=colors[k], marker=markers[nr % len(markers)], label=f"UT {k}, Stream {nr}")
            ax_R.set_xlabel("SNR [dB]")
            ax_R.set_ylabel("R [bit/s/Hz]")
            ax_R.set_ylim(0, None)
            ax_R.grid(True, which="both", linestyle="--", alpha=0.6)
            ax_R.legend()
            fig_R.tight_layout()

            plot_filename = ResultManager._plot_filename([sim_result], plot_type="stream R")
            fig_R.savefig(plot_filename, dpi=300)
            print(f"\n Saved per-stream R plot to:\n {plot_filename}")

        figs = (fig_ber if ber else None, fig_ibr if ibr else None, fig_R if R else None)
        return figs

    @staticmethod
    def plot_system_performance_comparison(sim_results: list[SimResult], ber: bool = True, ibr: bool = True, R: bool = True):
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
                ResultManager._plot_curve(ax_ber, snr_dB, bers, stream_ars, color=f"C{i}", marker="o", label=sim_result.system_configs.name)

            ax_ber.set_xlabel("SNR [dB]")
            ax_ber.set_ylabel("BER")
            ax_ber.set_yscale("log")
            ax_ber.set_ylim(0.5e-4, 1)
            ax_ber.grid(True, which="both", linestyle="--", alpha=0.6)
            ax_ber.legend()
            fig_ber.tight_layout()

            plot_filename = ResultManager._plot_filename(sim_results, plot_type="system BER comparison")
            fig_ber.savefig(plot_filename, dpi=300)
            print(f"\n Saved system BER comparison plot to:\n {plot_filename}")

        # IBR vs SNR.
        if ibr:
            fig_ibr, ax_ibr = plt.subplots(figsize=(6, 5))
            for i, sim_result in enumerate(sim_results):
                snr_dB = np.array([sim_res.snr_dB for sim_res in sim_result.simulation_results], dtype=float)
                ibrs = np.array([sim_res.ibr for sim_res in sim_result.simulation_results], dtype=float)
                stream_ars = np.array([sim_res.stream_ars_avg for sim_res in sim_result.simulation_results], dtype=float)
                ResultManager._plot_curve(ax_ibr, snr_dB, ibrs, stream_ars, color=f"C{i}", marker="o", label=sim_result.system_configs.name)

            ax_ibr.set_xlabel("SNR [dB]")
            ax_ibr.set_ylabel("IBR")
            ax_ibr.set_ylim(0, None)
            ax_ibr.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax_ibr.grid(True, which="both", linestyle="--", alpha=0.6)
            ax_ibr.legend()
            fig_ibr.tight_layout()

            plot_filename = ResultManager._plot_filename(sim_results, plot_type="system IBR comparison")
            fig_ibr.savefig(plot_filename, dpi=300)
            print(f"\n Saved system IBR comparison plot to:\n {plot_filename}")

        # Achievable Rate vs SNR.
        if R:
            fig_R, ax_R = plt.subplots(figsize=(6, 5))
            for i, sim_result in enumerate(sim_results):
                snr_dB = np.array([sim_res.snr_dB for sim_res in sim_result.simulation_results], dtype=float)
                Rs = np.array([sim_res.R for sim_res in sim_result.simulation_results], dtype=float)
                stream_ars = np.array([sim_res.stream_ars_avg for sim_res in sim_result.simulation_results], dtype=float)
                ResultManager._plot_curve(ax_R, snr_dB, Rs, stream_ars, color=f"C{i}", marker="o", label=sim_result.system_configs.name)

            ax_R.set_xlabel("SNR [dB]")
            ax_R.set_ylabel("R [bits/s/Hz]")
            ax_R.set_ylim(0, None)
            ax_R.grid(True, which="both", linestyle="--", alpha=0.6)
            ax_R.legend()
            fig_R.tight_layout()

            plot_filename = ResultManager._plot_filename(sim_results, plot_type="system R comparison")
            fig_R.savefig(plot_filename, dpi=300)
            print(f"\n Saved system R comparison plot to:\n {plot_filename}")

        figs = (fig_ber if ber else None, fig_ibr if ibr else None, fig_R if R else None)
        return figs

    @staticmethod
    def plot_ut_performance_comparison(sim_results: list[SimResult], ber: bool = True, ibr: bool = True, R: bool = True):
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
                for k in range(sim_result.system_configs.K):
                    ResultManager._plot_curve(ax_ber, snr_dB, ut_bers[k], ut_ars[k], color=f"C{i}", marker=markers[k % len(markers)], label=f"{sim_result.system_configs.name} - UT {k}")

            ax_ber.set_xlabel("SNR [dB]")
            ax_ber.set_ylabel("BER")
            ax_ber.set_yscale("log")
            ax_ber.set_ylim(0.5e-4, 1)
            ax_ber.grid(True, which="both", linestyle="--", alpha=0.6)
            ax_ber.legend()
            fig_ber.tight_layout()

            plot_filename = ResultManager._plot_filename(sim_results, plot_type="UT BER comparison")
            fig_ber.savefig(plot_filename, dpi=300)
            print(f"\n Saved UT BER comparison plot to:\n {plot_filename}")

        # IBR vs SNR.
        if ibr:
            fig_ibr, ax_ibr = plt.subplots(figsize=(6, 5))
            for i, sim_result in enumerate(sim_results):
                snr_dB = np.array([sim_res.snr_dB for sim_res in sim_result.simulation_results], dtype=float)
                ut_ibrs = np.transpose(np.array([sim_res.ut_ibrs for sim_res in sim_result.simulation_results], dtype=float))
                ut_ars = np.transpose(np.array([sim_res.ut_ars for sim_res in sim_result.simulation_results], dtype=float))
                for k in range(sim_result.system_configs.K):
                    ResultManager._plot_curve(ax_ibr, snr_dB, ut_ibrs[k], ut_ars[k], color=f"C{i}", marker=markers[k % len(markers)], label=f"{sim_result.system_configs.name} - UT {k}")

            ax_ibr.set_xlabel("SNR [dB]")
            ax_ibr.set_ylabel("IBR")
            ax_ibr.set_ylim(0, None)
            ax_ibr.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax_ibr.grid(True, which="both", linestyle="--", alpha=0.6)
            ax_ibr.legend()
            fig_ibr.tight_layout()

            plot_filename = ResultManager._plot_filename(sim_results, plot_type="UT IBR comparison")
            fig_ibr.savefig(plot_filename, dpi=300)
            print(f"\n Saved UT IBR comparison plot to:\n {plot_filename}")

        # Achievable Rate vs SNR.
        if R:
            fig_R, ax_R = plt.subplots(figsize=(6, 5))
            for i, sim_result in enumerate(sim_results):
                snr_dB = np.array([sim_res.snr_dB for sim_res in sim_result.simulation_results], dtype=float)
                ut_Rs = np.transpose(np.array([sim_res.ut_Rs for sim_res in sim_result.simulation_results], dtype=float))
                ut_ars = np.transpose(np.array([sim_res.ut_ars for sim_res in sim_result.simulation_results], dtype=float))
                for k in range(sim_result.system_configs.K):
                    ResultManager._plot_curve(ax_R, snr_dB, ut_Rs[k], ut_ars[k], color=f"C{i}", marker=markers[k % len(markers)], label=f"{sim_result.system_configs.name} - UT {k}")

            ax_R.set_xlabel("SNR [dB]")
            ax_R.set_ylabel("R [bits/s/Hz]")
            ax_R.set_ylim(0, None)
            ax_R.grid(True, which="both", linestyle="--", alpha=0.6)
            ax_R.legend()
            fig_R.tight_layout()

            plot_filename = ResultManager._plot_filename(sim_results, plot_type="UT R comparison")
            fig_R.savefig(plot_filename, dpi=300)
            print(f"\n Saved UT R comparison plot to:\n {plot_filename}")

        figs = (fig_ber if ber else None, fig_ibr if ibr else None, fig_R if R else None)
        return figs


        
__all__ = [
    "SingleSnrSimResult", "SimResult", "ResultManager"
]
