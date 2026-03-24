# mu-mimo/mu_mimo/analytical/channel_stats.py

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from ..types import ComplexArray, RealArray
from ..configs import SystemConfig, setup_sys_configs
from ..processing import ChannelModel

SYSTEM_CONFIG_PATH = Path(__file__).parent.parent.parent / 'system_configs.json'

@dataclass
class ChannelStatisticsData:
    """
    Data class for storing the channel statistics.

    This class contains the channel statistics of the virtual independent parallel streamchannels of a single UT in a multi-user MIMO system.\\
    The computation is based on the channel model, precoding technique, and combining technique used in the system, and is averaged over multiple channel realizations.

    Parameters
    ----------
    system_configs : SystemConfig
        The system configuration settings for which the channel statistics are computed.
    num_channel_samples : int
        The number of channel realizations used to compute the channel statistics.
    
    mean : RealArray, shape (K*Nr,)
        The mean of the channel gains of the virtual independent parallel streamchannels across all channel realizations.
    var : RealArray, shape (K*Nr,)
        The variance of the channel gains of the virtual independent parallel streamchannels across all channel realizations.
    
    histograms : RealArray, shape (K*Nr, num_bins-1)
        The histograms of the channel gains of the virtual independent parallel streamchannels across all channel realizations, where num_bins is the number of bins used for the histogram.
    bin_edges : RealArray, shape (K*Nr, num_bins)
        The bin edges of the histograms of the channel gains of the virtual independent parallel streamchannels
    
    ecdf : RealArray, shape (num_bins, K*Nr)
        The empirical cumulative distribution function (ECDF) values of the channel gains of the virtual independent parallel streamchannels across all channel realizations.
    quantiles : RealArray, shape (num_bins,)
        The quantiles corresponding to the ECDF values of the channel gains of the virtual independent parallel streamchannels across all channel realizations.
    """
    
    # System parameters.
    system_configs: SystemConfig
    num_channel_samples: int

    # Channel statistics.
    mean: RealArray
    var: RealArray
     
    histograms: RealArray
    bin_edges: RealArray
    
    ecdf: RealArray
    quantiles: RealArray

class ChannelStatistics:

    def __init__(self, system_config: SystemConfig, num_channel_samples: int = 1_000_000):
        
        self.system_config: SystemConfig = system_config
        self.num_channel_samples: int = num_channel_samples

        self.channel_statistics_data: ChannelStatisticsData = None

        # Argument validation.
        if system_config.base_station_configs.precoder.__name__ == "SVDPrecoder" and self.system_config.K > 1:
            raise ValueError("SVD precoding is only applicable for single-user MIMO systems (K=1). Please choose K=1 or a different precoding technique.")

    def evaluate(self, plot: bool = False) -> ChannelStatisticsData:
        """
        Evaluate the channel statistics of the virtual independent parallel streamchannels.

        The channel gain statistics of the virtual independent parallel streamchannels are computed based on the used precoding technique for multiple channel realizations.

        Parameters
        ----------
        plot : bool, optional
            Whether to plot the channel statistics, by default False.\\

        Returns
        -------
        channel_statistics_data : ChannelStatisticsData
            The computed channel statistics data of the virtual independent parallel streamchannels.
        """

        # 0. Try to load the channel statistics from an existing .npz file. If the file does not yet exist, compute the channel statistics.
        self.channel_statistics_data = self._load_channel_statistics()
        if self.channel_statistics_data is not None:
            print(f"\nChannel statistics loaded successfully from:\n    {self._generate_filepath().with_suffix('.npz')}\n\n")
            return self

        # 1. Compute the corresponding channel gains statistics of the virtual independent parallel streamchannels.
        K = self.system_config.K
        Nr = self.system_config.Nr
        g = np.empty((self.num_channel_samples, K * Nr), dtype=float)
        for i in tqdm(range(self.num_channel_samples), desc="Generating channel realizations"):
            H = self._generate_channel()
            g[i] = self._compute_streamchannel_gains(H)

        print("Computing channel statistics...")
        channel_statistics_data = self._compute_channel_statistics(g)

        # 3. Store the computed channel statistics.
        self.channel_statistics_data = channel_statistics_data
        self._store_channel_statistics()
        print(f"Channel statistics computed successfully and stored to:\n    {self._generate_filepath().with_suffix('.npz')}")

        # 4. Plot the channel statistics.
        if plot:
            print("Plotting channel statistics...")
            self._plot_streamchannel_pdf(num_uts=min(K, 1), seperate_plots=True)
            self._plot_streamchannel_ecdf(num_uts=min(K, 1), seperate_plots=True)
            print(f"Channel statistics plots generated successfully and stored to:\n    {self._generate_filepath().with_suffix('.png')}\n\n")

        return channel_statistics_data

    def _generate_filepath(self) -> Path:
        """
        Generate the file name for storing the channel statistics and plots.

        Returns
        -------
        file_name : Path
            The generated file name for storing the channel statistics.
        """
        
        # Generate a unique file name based on the system parameters and the used precoding technique.
        filename = Path(f"stats virtual streamchannel gains ({self.num_channel_samples//1_000_000}M samples) -- {self.system_config.name}")

        # Ensure both output trees exist, because plotting code saves into subfolders.
        stats_dir = Path(__file__).resolve().parents[2] / "report" / "analytical_results" /"channel_statistics" / "stats"
        plots_dir = Path(__file__).resolve().parents[2] / "report" / "analytical_results" / "channel_statistics" / "plots"
        stats_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)

        return filename

    def _store_channel_statistics(self) -> None:
        """
        Store the computed channel statistics in a .npz file.
        """

        # Validate the channel statistics data before storing.
        if self.channel_statistics_data is None:
            raise ValueError("No channel statistics available. Compute them before storing.")
        
        # Generate the file name.
        filename = self._generate_filepath().with_suffix(".npz")
        dirname = Path(__file__).resolve().parents[2] / "report" / "analytical_results" / "channel_statistics" / "stats"

        # Store the channel statistics data in a .npz file.
        data = self.channel_statistics_data
        np.savez_compressed(
            
            dirname / filename,
            
            system_configs = data.system_configs,
            num_channel_samples = data.num_channel_samples,

            mean        = data.mean,
            var         = data.var,
            histograms  = data.histograms,
            bin_edges   = data.bin_edges,
            ecdf        = data.ecdf,
            quantiles   = data.quantiles,
        )
        return

    def _load_channel_statistics(self) -> ChannelStatisticsData | None:
        """
        Load the channel statistics from a .npz file.

        Returns
        -------
        channel_statistics_data : ChannelStatisticsData
            The loaded channel statistics data. If the file does not yet exist, return None.
        """
        
        filename = self._generate_filepath().with_suffix(".npz")
        dirname = Path(__file__).resolve().parents[2] / "report" / "analytical_results" / "channel_statistics" / "stats"
        pathname = dirname / filename

        if not pathname.exists():
            return None

        loaded = np.load(pathname, allow_pickle=True)
        data = ChannelStatisticsData(
            
            system_configs           = loaded["system_configs"].item(),
            num_channel_samples = int(loaded["num_channel_samples"].item()),
            
            mean       = np.asarray(loaded["mean"], dtype=float),
            var        = np.asarray(loaded["var"], dtype=float),
            histograms = np.asarray(loaded["histograms"], dtype=float),
            bin_edges  = np.asarray(loaded["bin_edges"], dtype=float),
            ecdf       = np.asarray(loaded["ecdf"], dtype=float),
            quantiles  = np.asarray(loaded["quantiles"], dtype=float),
        )

        if (data.system_configs != self.system_config or data.num_channel_samples != self.num_channel_samples):
            raise ValueError("The configuration settings of the loaded channel statistics data do not match the configuration settings of the current ChannelStatistics instance. Please check the file name and the configuration settings of the current ChannelStatistics instance to solve this issue.")

        return data

    def _generate_channel(self) -> ComplexArray:
        """
        Generate the channel matrix according to the specified channel model.

        Returns
        -------
        H : ComplexArray, shape (K*Nr, Nt)
            The generated channel matrix.
        """
        
        K = self.system_config.K
        Nr = self.system_config.Nr
        Nt = self.system_config.Nt
        channel_model: ChannelModel = self.system_config.channel_configs.channel_model()
        
        H = channel_model.generate(K * Nr, Nt)
        return H

    def _compute_streamchannel_gains(self, H: ComplexArray) -> RealArray:
        """
        Compute the channel gains of the virtual independent parallel streamchannels for a given channel matrix, based on the used precoding technique.

        Parameters
        ----------
        H : ComplexArray, shape (K*Nr, Nt)
            The channel matrix.

        Returns
        -------
        g : RealArray, shape (K*Nr,)
            The channel gains of the virtual independent parallel streamchannels.
        """
        
        def svd_streamchannel_gains(H: ComplexArray) -> RealArray:
            Sigma = sp.linalg.svdvals(H)
            g = Sigma**2
            return g
        
        def zf_streamchannel_gains(H: ComplexArray) -> RealArray:
            g = 1 / ((np.linalg.inv(H @ H.conj().T)).conj()).diagonal()
            return g.real

        def zf_lsv_streamchannel_gains(H: ComplexArray) -> RealArray:

            K = self.system_config.K
            Nr = self.system_config.Nr
            
            H_eff = np.empty_like(H, dtype=complex)
            for k in range(K):
                H_k = H[k*Nr : (k+1)*Nr]
                U_k, Sigma_k, Vh_k = sp.linalg.svd(H_k)
                H_eff_k = np.transpose(U_k.conj()) @ H_k
                H_eff[k*Nr : (k+1)*Nr] = H_eff_k
            
            g = zf_streamchannel_gains(H_eff)
            return g

        def bd_streamchannel_gains(H: ComplexArray) -> RealArray:

            K = self.system_config.K
            Nr = self.system_config.Nr
            Nt = self.system_config.Nt
            
            r_ring = Nt - (K-1)*Nr
            F1 = np.empty((Nt, r_ring*K), dtype=complex)
            for k in range(K):
                H_ring_k = np.delete(H, slice(k*Nr, (k+1)*Nr), axis=0)
                _, _, Vh_ring_k = sp.linalg.svd(H_ring_k)
                V_ring_k = Vh_ring_k.conj().T
                F1_k = V_ring_k[:, Nt-r_ring : Nt]
                F1[:, k*r_ring : (k+1)*r_ring] = F1_k
            H_eff = H @ F1

            g = np.empty(K * Nr, dtype=float)
            for k in range(K):
                H_eff_k = H_eff[k*Nr : (k+1)*Nr, k*r_ring : (k+1)*r_ring]
                g_k = svd_streamchannel_gains(H_eff_k)
                g[k*Nr : (k+1)*Nr] = g_k
            
            return g

        def wmmse_streamchannel_gains(H: ComplexArray) -> RealArray:
            raise NotImplementedError("WMMSE streamchannel gains computation is not implemented yet.")
        
        precoder_name = self.system_config.base_station_configs.precoder.__name__
        combiner_name = self.system_config.user_terminal_configs.combiner.__name__
        
        if precoder_name == "SVDPrecoder" and combiner_name == "SVDCombiner":
            g = svd_streamchannel_gains(H)
        elif precoder_name == "ZFPrecoder" and combiner_name == "NeutralCombiner":
            g = zf_streamchannel_gains(H)
        elif precoder_name == "ZFPrecoder" and combiner_name == "LSVCombiner":
            g = zf_lsv_streamchannel_gains(H)
        elif precoder_name == "BDPrecoder":
            g = bd_streamchannel_gains(H)
        elif precoder_name == "WMMSEPrecoder":
            g = wmmse_streamchannel_gains(H)
        else:
            raise ValueError("Unsupported precoding and combining technique combination.")
        
        return g

    def _compute_channel_statistics(self, g: RealArray, num_bins: int = 100) -> ChannelStatisticsData:
        """
        Compute the channel statistics based on the provided channel gains of the virtual independent parallel streamchannels for multiple channel realizations.
        
        Parameters
        ----------
        g : RealArray, shape (num_channel_samples, K*Nr)
            The channel gains of the virtual independent parallel streamchannels for multiple channel realizations.
        num_bins : int, optional
            The number of bins to use for the histogram, by default 100.
        
        Returns
        -------
        channel_statistics : ChannelStatisticsData
            The computed channel statistics.
        """
        
        mean = np.mean(g, axis=0)
        var = np.var(g, axis=0)

        histograms_bins = [np.histogram(g[:, s], bins=num_bins, density=True) for s in range(self.system_config.K * self.system_config.Nr)]
        histograms = np.asarray([hb[0] for hb in histograms_bins], dtype=float)
        bin_edges = np.asarray([hb[1] for hb in histograms_bins], dtype=float)

        quantiles = np.linspace(0.0, 1.0, num_bins+1)
        ecdf = np.quantile(g, quantiles, axis=0)

        channel_statistics_data = ChannelStatisticsData(
            
            system_configs = self.system_config,
            num_channel_samples = self.num_channel_samples,
            
            mean       = mean,
            var        = var,
            histograms = histograms,
            bin_edges  = bin_edges,
            ecdf       = ecdf,
            quantiles  = quantiles,
            
        )
        return channel_statistics_data

    def _plot_streamchannel_pdf(self, num_uts: int = 1, seperate_plots: bool = False) -> None:
        """
        Plot the probability density function (PDF) of the channel gains of the virtual independent parallel streamchannels for multiple channel realizations, and save the plot as a .png file.

        Parameters
        ----------
        num_uts : int, optional
            The number of UTs to consider for the plot, by default 1.\\
            e.g., if num_uts=2, the plot will show the PDF of the channel gains of the virtual independent parallel streamchannels for the first 2 UTs (i.e., for the first 2*Nr streamchannels).
        seperate_plots : bool, optional
            Whether to plot the PDFs of the channel gains of the virtual independent parallel streamchannels for each UT in a separate subplot, by default False.\\
        """
        
        # 1. Plot initialization.
        if not seperate_plots:
            plot_pdf, ax = plt.subplots(figsize=(8, 5))
            axs = np.array([ax])
        else:
            num_colums = int(np.ceil(np.sqrt(num_uts)))
            num_rows = int(np.ceil(num_uts / num_colums))
            plot_pdf, axs = plt.subplots(num_rows, num_colums, figsize=(8*num_colums, 5*num_rows), sharex=True)
            axs = np.array(axs).reshape(-1)

        # 2. Plot the PDFs.
        for k in range(num_uts):

            ax = axs[k]
            Nr = self.system_config.Nr

            mean = self.channel_statistics_data.mean[k*Nr : (k+1)*Nr]
            var = self.channel_statistics_data.var[k*Nr : (k+1)*Nr]
            
            histograms = self.channel_statistics_data.histograms[k*Nr : (k+1)*Nr]
            bin_edges = self.channel_statistics_data.bin_edges[k*Nr : (k+1)*Nr]

            for nr in range(Nr):

                # The Probability Density Function (PDF) using histograms.
                edges = bin_edges[nr]
                centers = 0.5 * (edges[:-1] + edges[1:])
                ax.plot(centers, histograms[nr], label=f"UT {k+1}, Stream {nr+1}")

                # The mean line
                ax.axvline(mean[nr], linestyle="--", linewidth=1, color=ax.lines[-1].get_color())

                # The standard deviation area
                ax.axvspan(mean[nr] - np.sqrt(var[nr]), mean[nr] + np.sqrt(var[nr]), alpha=0.10, color=ax.lines[-1].get_color())

            # Plot settings.
            ax.set_xlabel("Virtual Channel Gain")
            ax.set_ylabel("Probability Density")
            ax.set_title("")
            ax.grid(True, alpha=0.3)
            ax.set_xlim(xmax = (bin_edges[1][-1] + 0.25 * (bin_edges[0][-1] - bin_edges[1][-1])) if Nr > 1 else (bin_edges[0][-1]) )
            ax.legend()

        # 3. Save the plot.
        plot_pdf_filename = Path(str(self._generate_filepath()) + f" ({num_uts} UTs plotted)" + ".png")
        plot_pdf_dir = Path(__file__).parents[2] / "report" / "channel_statistics" / "plots" / "pdf"
        plot_pdf.savefig(plot_pdf_dir / plot_pdf_filename, dpi=300, bbox_inches="tight")
        return

    def _plot_streamchannel_ecdf(self, num_uts: int = 1, seperate_plots: bool = False) -> None:
        """
        Plot the empirical cumulative distribution function (ECDF) of the channel gains of the virtual independent parallel streamchannels for multiple channel realizations, and save the plot as a .png file.

        Parameters
        ----------
        num_uts : int, optional
            The number of UTs to consider for the plot, by default 1.\\
            e.g., if num_uts=2, the plot will show the ECDF of the channel gains of the virtual independent parallel streamchannels for the first 2 UTs (i.e., for the first 2*Nr streamchannels).
        """
        
        # 1. Plot initialization.
        if not seperate_plots:
            plot_ecdf, ax = plt.subplots(figsize=(8, 5))
            axs = np.array([ax])
        else:
            num_colums = int(np.ceil(np.sqrt(num_uts)))
            num_rows = int(np.ceil(num_uts / num_colums))
            plot_ecdf, axs = plt.subplots(num_rows, num_colums, figsize=(8*num_colums, 5*num_rows), sharex=True)
            axs = np.array(axs).reshape(-1)

        # 2. Plot the ECDFs.
        for k in range(num_uts):

            ax = axs[k]
            Nr = self.system_config.Nr

            mean = self.channel_statistics_data.mean[k*Nr : (k+1)*Nr]
            var = self.channel_statistics_data.var[k*Nr : (k+1)*Nr]

            quantiles = self.channel_statistics_data.quantiles
            ecdf_vals = self.channel_statistics_data.ecdf[:, k*Nr : (k+1)*Nr]

            for nr in range(Nr):
            
                # The Empirical Cumulative Distribution Function (ECDF).
                ax.plot(ecdf_vals[:, nr], quantiles, label=f"UT {k+1}, Stream {nr+1}")

                # The mean marker.
                mean_q = np.interp(mean[nr], ecdf_vals[:, nr], quantiles)
                ax.plot(mean[nr], mean_q, marker="o", color=ax.lines[-1].get_color())

                # The standard deviation area.
                ax.axvspan((mean[nr] - np.sqrt(var[nr])), (mean[nr] + np.sqrt(var[nr])), alpha=0.10, color=ax.lines[-1].get_color())

            ax.set_xlabel("Virtual Channel Gain")
            ax.set_ylabel("Quantiles")
            ax.set_title(f"")
            ax.grid(True, alpha=0.3)
            ax.set_yticks(np.linspace(0, 1, 11))
            ax.set_xlim( xmax = (ecdf_vals[-1,1] + 0.25 * (ecdf_vals[-1,0] - ecdf_vals[-1,1])) if Nr > 1 else (ecdf_vals[-1,0]) )
            ax.legend()

        # 3. Save the plot.
        plot_ecdf_filename = Path(str(self._generate_filepath()) + f" ({num_uts} UTs plotted)" + ".png")
        plot_ecdf_dir = Path(__file__).parents[2] / "report" / "channel_statistics" / "plots" / "ecdf"
        plot_ecdf.savefig(plot_ecdf_dir / plot_ecdf_filename, dpi=300, bbox_inches="tight")
        return


