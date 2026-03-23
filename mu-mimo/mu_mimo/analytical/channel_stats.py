# mu-mimo/mu_mimo/analytical/channel_stats.py

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from tqdm import tqdm
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from ..types import ComplexArray, RealArray
from ..configs import SimConfig, SystemConfig

# CHANNEL MODELS

class ChannelModel(ABC):
    """
    The Channel Model Abstract Base Class (ABC).

    A channel model is responsible for generating the channel matrix according to a specific channel model and applying the channel effects to the transmitted signals.
    """

    @staticmethod
    @abstractmethod
    def generate(Nr_total: int, Nt: int) -> ComplexArray:
        """
        Generate the channel matrix.

        Parameters
        ----------
        Nr_total : int
            The total number of receive antennas across all UTs.
        Nt : int
            The number of transmit antennas at the BS.
        
        Returns
        -------
        H : ComplexArray, shape (Nr_total, Nt)
            The generated channel matrix.
        """
        raise NotImplementedError

    @staticmethod
    def apply(x: ComplexArray, H: ComplexArray) -> ComplexArray:
        """
        Apply the channel effects to the transmitted signals.

        Parameters
        ----------
        x : ComplexArray, shape (Nt, M)
            The transmitted signals.
        H : ComplexArray, shape (K*Nr, Nt)
            The channel matrix.

        Returns
        -------
        y : ComplexArray, shape (K*Nr, M)
            The received signals.
        """
        y = H @ x
        return y

class NeutralChannelModel(ChannelModel):
    """
    Neutral Channel Model.
    
    This channel model acts as a 'neutral element' for the channel.\\
    In particular, it generates an identity channel matrix, which means that the symbols are transmitted to the receive antennas for which they are intended, and without any interference.
    """

    @staticmethod
    def generate( Nr_total: int, Nt: int) -> ComplexArray:
        H = np.eye(Nr_total, Nt, dtype=complex)
        return H

class IIDRayleighChannelModel(ChannelModel):
    """
    Independent and Identically Distributed (IID) Rayleigh Fading Channel Model.

    This channel model generates a channel matrix with independent and identically distributed (IID) circularly-symmetric zero-mean unit-variance complex Gaussian entries.\\
    The Rayleigh fading aspect is captured by the fact that the channel coefficients change independently after M transmissions.
    """

    @staticmethod
    def generate(Nr_total: int, Nt: int) -> ComplexArray:
        H = (1 / np.sqrt(2)) * (np.random.randn(Nr_total, Nt) + 1j * np.random.randn(Nr_total, Nt))
        return H


# CHANNEL STATISTICS

@dataclass
class ChannelStatisticsData:
    """
    Data class for storing the channel statistics.

    Parameters
    ----------
    Nt : int
        The number of transmit antennas at the BS.
    K : int
        The number of UTs.
    Nr : int
        The number of receive antennas per UT.
    num_channel_realizations : int
        The number of channel realizations used to compute the channel statistics.
    channel_model : type[ChannelModel]
        The channel model used to generate the channel matrices.
    precoding_technique : Literal["SVDPrecoder", "ZFPrecoder", "BDPrecoder", "WMMSEPrecoder"]
        The precoding technique to consider when computing the channel gains of the virtual independent parallel streamchannels.
    combining_technique : Literal["NeutralCombiner", "SVDCombiner", "LSVCombiner"]
        The combining technique to consider when computing the channel gains of the virtual independent parallel streamchannels.
    
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
    Nt: int
    K: int
    Nr: int
    num_channel_realizations: int
    channel_model: type[ChannelModel]
    precoding_technique: Literal["SVDPrecoder", "ZFPrecoder", "BDPrecoder", "WMMSEPrecoder"]
    combining_technique: Literal["NeutralCombiner", "SVDCombiner", "LSVCombiner"]

    # Channel statistics.
    mean: RealArray
    var: RealArray
     
    histograms: RealArray
    bin_edges: RealArray
    
    ecdf: RealArray
    quantiles: RealArray

class ChannelStatistics:

    def __init__(self, Nt: int, K: int, Nr: int, num_channel_realizations: int, channel_model: type[ChannelModel], precoding_technique: Literal["SVDPrecoder", "ZFPrecoder", "BDPrecoder", "WMMSEPrecoder"], combining_technique: Literal["NeutralCombiner", "SVDCombiner", "LSVCombiner"]) -> None:
        
        self.Nt: int = Nt
        self.K: int = K
        self.Nr: int = Nr
        self.num_channel_realizations: int = num_channel_realizations
        self.channel_model: ChannelModel = channel_model()
        self.precoding_technique: Literal["SVDPrecoder", "ZFPrecoder", "BDPrecoder", "WMMSEPrecoder"] = precoding_technique
        self.combining_technique: Literal["NeutralCombiner", "SVDCombiner", "LSVCombiner"] = combining_technique

        self.channel_statistics_data: ChannelStatisticsData = None

        # Argument validation.
        if precoding_technique == "SVDPrecoder" and K > 1:
            raise ValueError("SVD precoding is only applicable for single-user MIMO systems (K=1). Please choose K=1 or a different precoding technique.")

    def evaluate(self) -> ChannelStatistics:
        """
        Evaluate the channel statistics of the virtual independent parallel streamchannels.

        The channel gain statistics of the virtual independent parallel streamchannels are computed based on the used precoding technique for multiple channel realizations.

        Returns
        -------
        channel_statistics : ChannelStatistics
            The channel statistics instance with the computed channel statistics data.
        """

        # 0. Try to load the channel statistics from an existing .npz file. If the file does not yet exist, compute the channel statistics.
        self.channel_statistics_data = self._load_channel_statistics()
        if self.channel_statistics_data is not None:
            print(f"Channel statistics loaded successfully from: {self._generate_filepath().with_suffix('.npz')}")
            return self

        # 1. Compute the corresponding channel gains statistics of the virtual independent parallel streamchannels.
        g = np.empty((self.num_channel_realizations, self.K * self.Nr), dtype=float)
        for i in tqdm(range(self.num_channel_realizations), desc="Generating channel realizations"):
            H = self._generate_channel()
            g[i] = self._compute_streamchannel_gains(H)

        print("Computing channel statistics...")
        self.channel_statistics_data = self._compute_channel_statistics(g)

        # 3. Store the computed channel statistics.
        self._store_channel_statistics()
        print(f"Channel statistics computed successfully and stored to: {self._generate_filepath().with_suffix('.npz')}")

        # 4. Plot the channel statistics.
        print("Plotting channel statistics...")
        self._plot_streamchannel_pdf(num_uts=min(self.K, 2), seperate_plots=True)
        self._plot_streamchannel_ecdf(num_uts=min(self.K, 2), seperate_plots=True)
        print(f"Channel statistics plots generated successfully and stored to: {self._generate_filepath().with_suffix('.png')}\n")

        return self

    def _generate_filepath(self) -> Path:
        """
        Generate the file name for storing the channel statistics and plots.

        Returns
        -------
        file_name : Path
            The generated file name for storing the channel statistics.
        """
        
        # Generate a unique file name based on the system parameters and the used precoding technique.
        filename = Path(
            f"virtual_streamchannel_gains"
            f"__Nt_{self.Nt}_K_{self.K}_Nr_{self.Nr}"
            f"__{self.channel_model.__class__.__name__}__{self.precoding_technique}__{self.combining_technique}"
            f"__N_{self.num_channel_realizations//1_000_000:.0f}M"
        )

        # Ensure both output trees exist, because plotting code saves into subfolders.
        stats_dir = Path(__file__).resolve().parents[2] / "report" / "channel_statistics" / "stats"
        plots_dir = Path(__file__).resolve().parents[2] / "report" / "channel_statistics" / "plots"
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
        dirname = Path(__file__).resolve().parents[2] / "report" / "channel_statistics" / "stats"

        # Store the channel statistics data in a .npz file.
        data = self.channel_statistics_data
        np.savez_compressed(
            
            dirname / filename,
            
            Nt = data.Nt,
            K  = data.K,
            Nr = data.Nr,
            num_channel_realizations = data.num_channel_realizations,
            channel_model            = data.channel_model,
            precoding_technique      = data.precoding_technique,
            combining_technique      = data.combining_technique,

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
        dirname = Path(__file__).resolve().parents[2] / "report" / "channel_statistics" / "stats"
        pathname = dirname / filename

        if not pathname.exists():
            return None

        loaded = np.load(pathname, allow_pickle=True)
        data = ChannelStatisticsData(
            
            Nt = int(loaded["Nt"]),
            K  = int(loaded["K"]),
            Nr = int(loaded["Nr"]),
            num_channel_realizations = int(loaded["num_channel_realizations"]),
            channel_model            = loaded["channel_model"].item(),
            precoding_technique      = loaded["precoding_technique"].item(),
            combining_technique      = loaded["combining_technique"].item(),
            
            mean       = np.asarray(loaded["mean"], dtype=float),
            var        = np.asarray(loaded["var"], dtype=float),
            histograms = np.asarray(loaded["histograms"], dtype=float),
            bin_edges  = np.asarray(loaded["bin_edges"], dtype=float),
            ecdf       = np.asarray(loaded["ecdf"], dtype=float),
            quantiles  = np.asarray(loaded["quantiles"], dtype=float),
        )

        if (data.Nt != self.Nt or data.K != self.K or data.Nr != self.Nr or data.num_channel_realizations != self.num_channel_realizations or data.channel_model != self.channel_model.__class__.__name__  or data.precoding_technique != self.precoding_technique  or data.combining_technique != self.combining_technique):
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
        H = self.channel_model.generate(self.K * self.Nr, self.Nt)
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
            
            H_eff = np.empty_like(H, dtype=complex)
            for k in range(self.K):
                H_k = H[k*self.Nr : (k+1)*self.Nr]
                U_k, Sigma_k, Vh_k = sp.linalg.svd(H_k)
                H_eff_k = np.transpose(U_k.conj()) @ H_k
                H_eff[k*self.Nr : (k+1)*self.Nr] = H_eff_k
            
            g = zf_streamchannel_gains(H_eff)
            return g

        def bd_streamchannel_gains(H: ComplexArray) -> RealArray:
            
            r_ring = self.Nt - (self.K-1)*self.Nr
            F1 = np.empty((self.Nt, r_ring*self.K), dtype=complex)
            for k in range(self.K):
                H_ring_k = np.delete(H, slice(k*self.Nr, (k+1)*self.Nr), axis=0)
                _, _, Vh_ring_k = sp.linalg.svd(H_ring_k)
                V_ring_k = Vh_ring_k.conj().T
                F1_k = V_ring_k[:, self.Nt-r_ring : self.Nt]
                F1[:, k*r_ring : (k+1)*r_ring] = F1_k
            H_eff = H @ F1

            g = np.empty(self.K * self.Nr, dtype=float)
            for k in range(self.K):
                H_eff_k = H_eff[k*self.Nr : (k+1)*self.Nr, k*r_ring : (k+1)*r_ring]
                g_k = svd_streamchannel_gains(H_eff_k)
                g[k*self.Nr : (k+1)*self.Nr] = g_k
            
            return g

        def wmmse_streamchannel_gains(H: ComplexArray) -> RealArray:
            raise NotImplementedError("WMMSE streamchannel gains computation is not implemented yet.")
        
        if self.precoding_technique == "SVDPrecoder" and self.combining_technique == "SVDCombiner":
            g = svd_streamchannel_gains(H)
        elif self.precoding_technique == "ZFPrecoder" and self.combining_technique == "NeutralCombiner":
            g = zf_streamchannel_gains(H)
        elif self.precoding_technique == "ZFPrecoder" and self.combining_technique == "LSVCombiner":
            g = zf_lsv_streamchannel_gains(H)
        elif self.precoding_technique == "BDPrecoder":
            g = bd_streamchannel_gains(H)
        elif self.precoding_technique == "WMMSEPrecoder":
            g = wmmse_streamchannel_gains(H)
        else:
            raise ValueError("Unsupported precoding and combining technique combination.")
        
        return g

    def _compute_channel_statistics(self, g: RealArray, num_bins: int = 100) -> ChannelStatisticsData:
        """
        Compute the channel statistics based on the provided channel gains of the virtual independent parallel streamchannels for multiple channel realizations.
        
        Parameters
        ----------
        g : RealArray, shape (num_channel_realizations, K*Nr)
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

        histograms_bins = [np.histogram(g[:, s], bins=num_bins, density=True) for s in range(self.K * self.Nr)]
        histograms = np.asarray([hb[0] for hb in histograms_bins], dtype=float)
        bin_edges = np.asarray([hb[1] for hb in histograms_bins], dtype=float)

        quantiles = np.linspace(0.0, 1.0, num_bins+1)
        ecdf = np.quantile(g, quantiles, axis=0)

        channel_statistics_data = ChannelStatisticsData(
            
            Nt = self.Nt,
            K  = self.K,
            Nr = self.Nr,
            num_channel_realizations = self.num_channel_realizations,
            channel_model            = self.channel_model.__class__.__name__,
            precoding_technique      = self.precoding_technique,
            combining_technique      = self.combining_technique,
            
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

            mean = self.channel_statistics_data.mean[k*self.Nr : (k+1)*self.Nr]
            var = self.channel_statistics_data.var[k*self.Nr : (k+1)*self.Nr]
            
            histograms = self.channel_statistics_data.histograms[k*self.Nr : (k+1)*self.Nr]
            bin_edges = self.channel_statistics_data.bin_edges[k*self.Nr : (k+1)*self.Nr]

            for nr in range(self.Nr):

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
            ax.set_xlim( xmax = (bin_edges[1][-1] + 0.25 * (bin_edges[0][-1] - bin_edges[1][-1])) if self.Nr > 1 else (bin_edges[0][-1]) )
            ax.legend()

        # 3. Save the plot.
        plot_pdf_filename = Path(str(self._generate_filepath()) + f"__UTs_{num_uts}__pdf").with_suffix(".png")
        plot_pdf_dir = Path(__file__).parents[2] / "report" / "channel_statistics" / "plots"
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

            mean = self.channel_statistics_data.mean[k*self.Nr : (k+1)*self.Nr]
            var = self.channel_statistics_data.var[k*self.Nr : (k+1)*self.Nr]

            quantiles = self.channel_statistics_data.quantiles
            ecdf_vals = self.channel_statistics_data.ecdf[:, k*self.Nr : (k+1)*self.Nr]

            for nr in range(self.Nr):
            
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
            ax.set_xlim( xmax = (ecdf_vals[-1,1] + 0.25 * (ecdf_vals[-1,0] - ecdf_vals[-1,1])) if self.Nr > 1 else (ecdf_vals[-1,0]) )
            ax.legend()

        # 3. Save the plot.
        plot_ecdf_filename = Path(str(self._generate_filepath()) + f"__UTs_{num_uts}__ecdf").with_suffix(".png")
        plot_ecdf_dir = Path(__file__).parents[2] / "report" / "channel_statistics" / "plots"
        plot_ecdf.savefig(plot_ecdf_dir / plot_ecdf_filename, dpi=300, bbox_inches="tight")
        return


# EXPECTED ACHIEVABLE RATES

@dataclass
class ExpectedAchievableRateData:
    """
    TO DO: add docstring.
    """

    # System parameters.
    Nt: int
    Nr: int
    channel_model: type[ChannelModel]
    precoding_technique: Literal["SVDPrecoder", "ZFPrecoder", "BDPrecoder", "WMMSEPrecoder"]
    combining_technique: Literal["NeutralCombiner", "SVDCombiner", "LSVCombiner"]

    # Computation parameters.
    num_channel_realizations: int
    num_bins: int

    # Data.
    snr_dB_values: RealArray
    expected_achievable_rate: RealArray
    expected_achievable_rate_ub: RealArray


class ExpectedAchievableRate:

    def __init__(self, channel_statistics: ChannelStatisticsData) -> None:
        pass

    def _generate_filepath(self) -> Path:
        pass

    def _store_expected_achievable_rate(self) -> None:
        pass

    def _load_expected_achievable_rate(self) -> ExpectedAchievableRateData | None:
        pass

    def evaluate(self) -> None:
        pass
    
    @staticmethod
    def _waterfilling_v1(gamma, pt):
        r"""
        Waterfilling algorithm.

        This function implements the waterfilling algorithm to find the optimal power allocation across N transmission streams, given the channel-to-noise ratio (CNR) coefficients `gamma` and the total available transmit power `pt`.

        In particular, it solves the following constraint optimization problem:

        .. math::

            \begin{aligned}
                & \underset{\{p_n\}}{\text{max}}
                & & \sum_{n=1}^{N} \log_2 \left( 1 + \gamma_n \, p_n \right) \\
                & \text{s. t.}
                & & \sum_{n=1}^{N} p_n = p_t \\
                & & & \forall n \in \{1, \ldots, N\} : \, p_n \geq 0
            \end{aligned}

        Parameters
        ----------
        gamma : RealArray, shape (N,)
            Channel-to-Noise Ratio (CNR) coefficients for each eigenchannel.
        pt : float
            Total available transmit power.

        Returns
        -------
        p : RealArray, shape (N,)
            Optimal power allocation across the eigenchannels.
        """

        # STEP 0: Sort the CNR coefficients in descending order.
        sorted_indices = np.argsort(gamma)[::-1]
        gamma = gamma[sorted_indices]

        # STEP 1: Determine the number of active streams.
        pt_iter = lambda as_iter: np.sum( (1 / gamma[as_iter]) - (1 / gamma[:as_iter]) )
        as_UB = len(gamma)
        as_LB = 0

        while as_UB - as_LB > 1:
            as_iter = (as_UB + as_LB) // 2
            if pt > pt_iter(as_iter): as_LB = as_iter
            elif pt < pt_iter(as_iter): as_UB = as_iter
        
        # STEP 2: Compute the optimal power allocation for each active stream.
        p_step1 = ( (1 / gamma[as_LB]) - (1 / gamma[:as_LB]) )
        p_step1 = np.concatenate( (p_step1, np.zeros(as_UB - as_LB)) )

        power_remaining = pt - np.sum(p_step1)
        p_step2 = (1 / as_UB) * power_remaining

        p_sorted = np.concatenate( (p_step1 + p_step2, np.zeros(len(gamma) - as_UB)) )

        # STEP 3: Reorder the power allocation to match the original order of the streams.
        p = np.empty_like(p_sorted)
        p[sorted_indices] = p_sorted

        return p





if __name__ == "__main__":


    settings = [

        {'Nt': 8, 'K': 2, 'Nr': 2, 'num_channel_realizations': 1_000_000, 'channel_model': IIDRayleighChannelModel, 'precoding_technique': "ZFPrecoder", 'combining_technique': "NeutralCombiner"},
        {'Nt': 8, 'K': 2, 'Nr': 2, 'num_channel_realizations': 1_000_000, 'channel_model': IIDRayleighChannelModel, 'precoding_technique': "ZFPrecoder", 'combining_technique': "LSVCombiner"},
        {'Nt': 8, 'K': 2, 'Nr': 2, 'num_channel_realizations': 1_000_000, 'channel_model': IIDRayleighChannelModel, 'precoding_technique': "BDPrecoder", 'combining_technique': "NeutralCombiner"},
        # {'Nt': 8, 'K': 2, 'Nr': 2, 'num_channel_realizations': 1_000_000, 'channel_model': IIDRayleighChannelModel, 'precoding_technique': "WMMSEPrecoder", 'combining_technique': "NeutralCombiner"},

        {'Nt': 8, 'K': 4, 'Nr': 2, 'num_channel_realizations': 1_000_000, 'channel_model': IIDRayleighChannelModel, 'precoding_technique': "ZFPrecoder", 'combining_technique': "NeutralCombiner"},
        {'Nt': 8, 'K': 4, 'Nr': 2, 'num_channel_realizations': 1_000_000, 'channel_model': IIDRayleighChannelModel, 'precoding_technique': "ZFPrecoder", 'combining_technique': "LSVCombiner"},
        {'Nt': 8, 'K': 4, 'Nr': 2, 'num_channel_realizations': 1_000_000, 'channel_model': IIDRayleighChannelModel, 'precoding_technique': "BDPrecoder", 'combining_technique': "NeutralCombiner"},
        # {'Nt': 8, 'K': 4, 'Nr': 2, 'num_channel_realizations': 1_000_000, 'channel_model': IIDRayleighChannelModel, 'precoding_technique': "WMMSEPrecoder", 'combining_technique': "NeutralCombiner"},

        {'Nt': 64, 'K': 16, 'Nr': 2, 'num_channel_realizations': 1_000_000, 'channel_model': IIDRayleighChannelModel, 'precoding_technique': "ZFPrecoder", 'combining_technique': "NeutralCombiner"},
        {'Nt': 64, 'K': 16, 'Nr': 2, 'num_channel_realizations': 1_000_000, 'channel_model': IIDRayleighChannelModel, 'precoding_technique': "ZFPrecoder", 'combining_technique': "LSVCombiner"},
        {'Nt': 64, 'K': 16, 'Nr': 2, 'num_channel_realizations': 1_000_000, 'channel_model': IIDRayleighChannelModel, 'precoding_technique': "BDPrecoder", 'combining_technique': "NeutralCombiner"},
        # {'Nt': 64, 'K': 16, 'Nr': 2, 'num_channel_realizations': 1_000_000, 'channel_model': IIDRayleighChannelModel, 'precoding_technique': "WMMSEPrecoder", 'combining_technique': "NeutralCombiner"},

    ]
    
    for setting in settings:

        channel_statistics = ChannelStatistics(
            Nt = setting['Nt'],
            K  = setting['K'],
            Nr = setting['Nr'],
            num_channel_realizations = setting['num_channel_realizations'],
            channel_model = setting['channel_model'],
            precoding_technique = setting['precoding_technique'],
            combining_technique = setting['combining_technique'],
        )

        channel_statistics = channel_statistics.evaluate()
        channel_statistics._plot_streamchannel_pdf(num_uts=min(channel_statistics.K, 2), seperate_plots=True)

