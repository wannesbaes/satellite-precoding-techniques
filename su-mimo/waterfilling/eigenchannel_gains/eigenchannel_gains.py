# This file makes it possible to simulate and store the eigenchannel gain statistics (expected value, variance, probability density function, and empirical cumulative distribution function) for slow Rayleigh fading MIMO channels. Also, it contains a demonstration function to plot these statistics. Different combinations of numbers of transmit and receive antennas can be specified.


import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from tqdm import tqdm

def store_eigenchannel_gains_stats(Nt_list, Nr_list, num_samples, output_dir):
    """
    Description
    -----------
    Simulate and store the eigenchannel gain statistics (expected value, variance, histograms, and empirical cumulative distribution functions) for slow Rayleigh fading MIMO channels. Different combinations of numbers of transmit and receive antennas can be specified.

    Parameters
    ----------
    Nt_list : list of int
        The different numbers of transmit antennas to consider.
    Nr_list : list of int
        The different numbers of receive antennas to consider.
    num_samples : int
        The number of channel realizations to use for the computation of the eigenchannel gain statistics.
    """

    def simulate_eigenchannel_gains(Nt, Nr, num_samples):
        """
        Description
        -----------
        Simulate the eigenchannel gains of each eigenchannel of a MIMO channel with Nt transmitters and Nr receivers over num_samples channel realizations.

        Parameters
        ----------
        Nt : int
            The number of transmit antennas.
        Nr : int
            The number of receive antennas.
        num_samples : int
            The number of channel realizations to use for the computation of the eigenchannel gain statistics.
        
        Returns
        -------
        g : 2D numpy array (dtype: float, shape: (num_samples, min(Nt, Nr)))
            The eigenchannel gains of each eigenchannel (columns), in descending order. The rows correspond to different channel realizations.
        """
        
        H = (1/np.sqrt(2)) * ( np.random.randn(num_samples, Nr, Nt) + 1j * np.random.randn(num_samples, Nr, Nt) )

        g = np.zeros((num_samples, min(Nt, Nr)))
        for i in tqdm(range(num_samples), desc=f'{Nt}x{Nr}'):
            G = H[i] @ H[i].conj().T if Nr <= Nt else H[i].conj().T @ H[i]
            g[i] = np.linalg.eigvalsh(G)[::-1]
        
        return g

    def process_eigenchannel_gains(g):
        """
        Description
        -----------
        Process the eigenchannel gains to compute their expected value, variance, histograms, and empirical cumulative distribution functions (ECDF).

        Parameters
        ----------
        g : 2D numpy array (dtype: float, shape: (num_samples, min(Nt, Nr)))
            The eigenchannel gains of each eigenchannel (columns), in descending order. The rows correspond to different channel realizations.
        
        Returns
        -------
        results : dict
            A dictionary containing the following keys:
            - "mean": 1D numpy array (dtype: float, shape: (min(Nt, Nr),)) - The expected value of the eigenchannel gains.
            - "var": 1D numpy array (dtype: float, shape: (min(Nt, Nr),)) - The variance of the eigenchannel gains.
            - "histograms": 2D numpy array (dtype: float, shape: (min(Nt, Nr), num_bins)) - The histograms of the eigenchannel gains.
            - "bin_edges": 1D numpy array (dtype: float, shape: (num_bins + 1,)) - The edges of the histogram bins.
            - "ecdf": 2D numpy array (dtype: float, shape: (101, min(Nt, Nr))) - The empirical cumulative distribution functions of the eigenchannel gains.
        """

        num_samples, num_eigenchannels = g.shape

        mean = np.mean(g, axis=0)
        var = np.var(g, axis=0)

        histograms_bins = [np.histogram(g[:, k], bins=100, density=True) for k in range(num_eigenchannels)]
        histograms = [histograms_bins[k][0] for k in range(num_eigenchannels)]
        bin_edges = [histograms_bins[k][1] for k in range(num_eigenchannels)]

        quantiles = np.linspace(0.0, 1.0, 101)
        ecdf = np.quantile(g, quantiles, axis=0)

        results = {"mean": mean, "var": var, "histograms": histograms, "bin_edges": bin_edges, "quantiles": quantiles, "ecdf": ecdf}
        return results

    for Nt, Nr in product(Nt_list, Nr_list):

        if os.path.exists(os.path.join(output_dir, f'{Nt}x{Nr}__{num_samples//1e6:.0f}M_samples.npz')): continue
        g = simulate_eigenchannel_gains(Nt, Nr, num_samples)
        results = process_eigenchannel_gains(g)
        np.savez_compressed(
            os.path.join(output_dir, f'{Nt}x{Nr}__{num_samples//1e6:.0f}M_samples.npz'),
            Nt=Nt,
            Nr=Nr,
            num_samples=num_samples,
            mean=results["mean"],
            var=results["var"],
            histograms=results["histograms"],
            bin_edges=results["bin_edges"],
            quantiles=results["quantiles"],
            ecdf=results["ecdf"]
        )

def demo_eigenchannel_gains_stats(Nt_list, Nr_list, num_samples, input_dir, output_dir, pdf=True, ecdf=True):
    """
    Description
    -----------
    Demonstrate the eigenchannel gain statistics.
    First load the eigenchannel gain statistics from a .npz file. If the file does not exist yet, the statistics will first be simulated and stored.
    Then plot the probability density function (PDF) using histograms and/or empirical cumulative distribution function (ECDF) plots for each eigenchannel gain. These plots are saved to the specified output directory.

    Parameters
    ----------
    Nt_list : list of int
        The different numbers of transmit antennas to consider.
    Nr_list : list of int
        The different numbers of receive antennas to consider.
    num_samples : int
        The number of channel realizations used for the computation of the eigenchannel gain statistics.
    pdf : bool, optional
        Whether to plot the probability density function (PDF) using histograms. Default is True.
    ecdf : bool, optional
        Whether to plot the empirical cumulative distribution function (ECDF). Default is True.
    input_dir : str, optional
        The directory where the eigenchannel gain statistics files are stored. Default is 'mu-mimo/src/helpers/eigenchannel_gains_stats/'.
    output_dir : str, optional
        The directory where the plots will be saved. Default is 'mu-mimo/src/helpers/eigenchannel_gains_stats/plots/'.
    """

    def load_eigenchannel_gains_stats(Nt, Nr, num_samples, input_dir):
        """
        Description
        -----------
        Load the eigenchannel gain statistics from a .npz file. If the file does not exist yet, the statistics will first be simulated and stored.

        Parameters
        ----------
        Nt : int
            The number of transmit antennas.
        Nr : int
            The number of receive antennas.
        num_samples : int
            The number of channel realizations used for the computation of the eigenchannel gain statistics.
        input_dir : str, optional
            The directory where the eigenchannel gain statistics files are stored. Default is 'mu-mimo/src/helpers/eigenchannel_gains_stats/'.
        
        Returns
        -------
        results : dict
            A dictionary containing the following keys:
            - "Nt": int - The number of transmit antennas.
            - "Nr": int - The number of receive antennas.
            - "num_samples": int - The number of channel realizations used.
            - "mean": 1D numpy array (dtype: float, shape: (min(Nt, Nr),)) - The expected value of the eigenchannel gains.
            - "var": 1D numpy array (dtype: float, shape: (min(Nt, Nr),)) - The variance of the eigenchannel gains.
            - "histograms": 2D numpy array (dtype: float, shape: (min(Nt, Nr), num_bins)) - The histograms of the eigenchannel gains.
            - "bin_edges": 1D numpy array (dtype: float, shape: (num_bins + 1,)) - The edges of the histogram bins.
            - "ecdf": 2D numpy array (dtype: float, shape: (101, min(Nt, Nr))) - The empirical cumulative distribution functions of the eigenchannel gains.
        """
        
        filepath = os.path.join(input_dir, f'{Nt}x{Nr}__{num_samples//1e6:.0f}M_samples.npz')
        
        if not os.path.exists(filepath): store_eigenchannel_gains_stats((Nt,), (Nr,), num_samples, output_dir=input_dir)
        
        data = np.load(filepath, allow_pickle=True)
        results = { "Nt": data["Nt"].item(), "Nr": data["Nr"].item(), "num_samples": data["num_samples"].item(), "mean": data["mean"], "var": data["var"], "histograms": data["histograms"], "bin_edges": data["bin_edges"], "quantiles": data["quantiles"], "ecdf": data["ecdf"] }
        return results
    
    def plot_eigenchannel_gains_stats(results, output_dir, pdf=True, ecdf=True):
        """
        Description
        -----------
        Plot the eigenchannel gain statistics. This function creates a plot of the probability density function (PDF) and/or empirical cumulative distribution function (ECDF) plot for each eigenchannel gain, based on the provided results.

        Parameters
        ----------
        results : dict
            A dictionary containing the eigenchannel gain statistics. See the output of the `load_eigenchannel_gains_stats` function for the expected structure.
        pdf : bool, optional
            Whether to plot the probability density function (PDF) using histograms. Default is True.
        ecdf : bool, optional
            Whether to plot the empirical cumulative distribution function (ECDF). Default is True.
        output_dir : str, optional
            The directory where the plots will be saved. Default is 'mu-mimo/src/helpers/eigenchannel_gains_stats/plots/'.
        
        Returns
        -------
        plot_pdf : matplotlib.figure.Figure or None
            The figure object containing the PDF plots, or None if pdf=False.
        plot_ecdf : matplotlib.figure.Figure or None
            The figure object containing the ECDF plots, or None if ecdf=False.
        """

        # Load the parameters.

        Nt = results.get("Nt", None)
        Nr = results.get("Nr", None)
        num_samples = results.get("num_samples", None)
        mean = results["mean"]
        std = np.sqrt(results["var"])

        plot_pdf = None
        plot_ecdf = None


        # Plot the PDF.  

        if pdf:

            plot_pdf, ax = plt.subplots(figsize=(8, 5))

            histograms = results["histograms"]
            bin_edges = results["bin_edges"]

            for k in range(min(Nt, Nr)):

                # The Probability Density Function (PDF) using histograms.
                edges = bin_edges[k]
                centers = 0.5 * (edges[:-1] + edges[1:])
                ax.plot(centers, histograms[k], label=f"Eigenchannel {k+1}")

                # The mean line
                ax.axvline(mean[k], linestyle="--", linewidth=1, color=ax.lines[-1].get_color())

                # The standard deviation area
                ax.axvspan(mean[k] - std[k], mean[k] + std[k], alpha=0.10, color=ax.lines[-1].get_color())

            ax.set_xlabel("Eigenchannel Gain")
            ax.set_ylabel("Probability Density")
            ax.set_title(f"Eigenchannel Gain PDF ({Nt}x{Nr})")
            ax.grid(True, alpha=0.3)
            ax.set_xlim( xmax = (bin_edges[1][-1] + 0.25 * (bin_edges[0][-1] - bin_edges[1][-1])) if min(Nt, Nr) > 1 else (bin_edges[0][-1]) )
            ax.legend()

            pdf_filename = os.path.join(output_dir, f"pdf_{Nt}x{Nr}.png")
            plot_pdf.savefig(pdf_filename, dpi=300, bbox_inches="tight")


        # Plot the ECDF.

        if ecdf:

            plot_ecdf, ax = plt.subplots(figsize=(8, 5))

            quantiles = results["quantiles"]
            ecdf_vals = results["ecdf"]

            for k in range(min(Nt, Nr)):
                
                # The Empirical Cumulative Distribution Function (ECDF).
                ax.plot(ecdf_vals[:, k], quantiles, label=f"Eigenchannel {k+1}")

                # The mean marker.
                mean_q = np.interp(mean[k], ecdf_vals[:, k], quantiles)
                ax.plot(mean[k], mean_q, marker="o", color=ax.lines[-1].get_color())

                # The standard deviation area.
                ax.axvspan((mean[k] - std[k]), (mean[k] + std[k]), alpha=0.10, color=ax.lines[-1].get_color())

            ax.set_xlabel("Eigenchannel gain")
            ax.set_ylabel("Quantiles")
            ax.set_title(f"Eigenchannel Gain ECDF ({Nt}x{Nr})")
            ax.grid(True, alpha=0.3)
            ax.set_yticks(np.linspace(0, 1, 11))
            ax.set_xlim( xmax = (ecdf_vals[-1,1] + 0.25 * (ecdf_vals[-1,0] - ecdf_vals[-1,1])) if min(Nt, Nr) > 1 else (ecdf_vals[-1,0]) )
            ax.legend()

            ecdf_filename = os.path.join(output_dir, f"ecdf_{Nt}x{Nr}.png")
            plot_ecdf.savefig(ecdf_filename, dpi=300, bbox_inches="tight")

        return plot_pdf, plot_ecdf

    for Nt, Nr in product(Nt_list, Nr_list):
        results = load_eigenchannel_gains_stats(Nt, Nr, num_samples, input_dir)
        plot_eigenchannel_gains_stats(results, output_dir, pdf, ecdf)
        plt.close('all')


if __name__ == "__main__":

    N = 10e6                                                        # Number of channel realizations.
    Nt = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)                            # Number of transmit antennas.
    Nr = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)                            # Number of receive antennas.
    stats_dir = 'mu-mimo/src/helpers/eigenchannel_gains/stats/'     # Directory to store the eigenchannel gain statistics.
    plots_dir = 'mu-mimo/src/helpers/eigenchannel_gains/plots/'     # Directory to store the eigenchannel gain plots.
    os.makedirs(stats_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)


    store_eigenchannel_gains_stats(Nt, Nr, int(N), stats_dir)
    demo_eigenchannel_gains_stats(Nt, Nr, int(N), stats_dir, plots_dir, pdf=True, ecdf=True)