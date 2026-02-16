# This file contains the implementation of the resource allocation techniques for a single-user MIMO system, including power allocation and bit allocation strategies. It also includes functions to visualize the optimal power allocation across the eigenchannels using waterfilling, as well as the capacity and information bit rate of the eigenchannels for different SNR values and power allocation strategies.

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# 1. POWER ALLOCATION

def waterfilling_v1(gamma, pt):
    r"""
    Waterfilling algorithm for optimal power allocation across eigenchannels.

    This function implements the waterfilling algorithm to find the optimal power allocation across N transmission channels (eigenchannels in a single-user MIMO system, given the channel-to-noise ratio (CNR) coefficients `gamma` and the total available transmit power `pt`.

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
    gamma : ndarray, shape (N,), dtype=float
        Channel-to-Noise Ratio (CNR) coefficients for each eigenchannel.
    pt : float
        Total available transmit power.

    Returns
    -------
    p : ndarray, shape (N,), dtype=float
        Optimal power allocation across the eigenchannels.
    """

    # STEP 0: Check the validity of the input parameters.
    assert pt > 0, "The total transmit power 'pt' must be positive."

    # STEP 1: Determine the number of active eigenchannels.
    pt_iter = lambda aes_iter: np.sum( (1 / gamma[aes_iter]) - (1 / gamma[:aes_iter]) )
    aes_UB = len(gamma)
    aes_LB = 0

    while aes_UB - aes_LB > 1:
        aes_iter = (aes_UB + aes_LB) // 2
        if pt > pt_iter(aes_iter): aes_LB = aes_iter
        elif pt < pt_iter(aes_iter): aes_UB = aes_iter
    
    # STEP 2: Compute the optimal power allocation for each active eigenchannel.
    p_step1 = ( (1 / gamma[aes_LB]) - (1 / gamma[:aes_LB]) )
    p_step1 = np.concatenate( (p_step1, np.zeros(aes_UB - aes_LB)) )

    power_remaining = pt - np.sum(p_step1)
    p_step2 = (1 / aes_UB) * power_remaining

    p = np.concatenate( (p_step1 + p_step2, np.zeros(len(gamma) - aes_UB)) )

    return p

def equal_power_allocation(Ns, pt):
    r"""
    Equal power allocation across the eigenchannels of a SU-MIMO system.

    Each eigenchannel is allocated an equal share of the total available transmit power, regardless of the channel conditions. This strategy is simple to implement and does not require any channel state information (CSI) at the transmitter.

    Parameters
    ----------
    Ns : int
        Number of eigenchannels (or spatial streams) to allocate power to.
    pt : float
        Total available transmit power.
    
    Returns
    -------
    p : ndarray, shape (Ns,), dtype=float
        Equal power allocation across the eigenchannels.
    """

    # STEP 0: Check the validity of the input parameters.
    assert pt > 0, "The total transmit power 'pt' must be positive."

    # STEP 1: Allocate equal power across the eigenchannels.
    p = np.full(Ns, pt / Ns, dtype=float)

    return p

def eigenbeamforming(Ns, pt):
    r"""
    Eigenbeamforming power allocation across the eigenchannels of a SU-MIMO system.

    All of the available transmit power is allocated to the strongest eigenchannel, while the remaining eigenchannels are allocated zero power.

    Parameters
    ----------
    Ns : int
        Number of eigenchannels (or spatial streams) to allocate power to.
    pt : float
        Total available transmit power.
    
    Returns
    -------
    p : ndarray, shape (Ns,), dtype=float
        Power allocation across the eigenchannels based on the eigenbeamforming method.
    """

    # STEP 0: Check the validity of the input parameters.
    assert pt > 0, "The total transmit power 'pt' must be positive."

    # STEP 1: Allocate all power to the eigenchannel with the highest CNR coefficient.
    p = np.zeros(Ns, dtype=float)
    p[0] = pt

    return p


def plot_waterfilling(Nt, Nr, snr_dB, p_signal=None, p_noise=None, num_samples=1e7):
    """
    Generate a static waterfilling plot for a SU-MIMO system.

    This function visualizes the optimal power allocation across the eigenchannels of a single-user MIMO system at a given SNR. The eigenchannel gains are taken as the expected values of the squared singular values of a random Gaussian zero-mean unit-variance channel matrix.

    The eigenchannels are shown horizontally (on the x-axis), and ordered by their expected gain. The inverse channel-to-noise ratio (ICNR), the allocated power per eigenchannel, and the water level are shown vertically (on the y-axis).

    Parameters
    ----------
    Nt : int
        Number of transmit antennas.
    Nr : int
        Number of receive antennas.
    snr_dB : float
        Signal-to-Noise Ratio in decibels.
    p_signal : float, optional
        Total transmit power. If provided, the noise power is scaled according to `snr_dB`.
    p_noise : float, optional
        Total noise power. If provided, the transmit power is scaled according to `snr_dB`.
    num_samples : int, optional
        Number of Monte Carlo samples used to compute the mean eigenchannel gains (default: 1e7).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the water-filling plot.
    ax : matplotlib.axes.Axes
        The axes object of the figure.
    """

    # STEP 0: Parameters check.
    filepath_g = "su-mimo/src/resource_allocation/eigenchannel_gains/stats/" + f"{Nt}x{Nr}__{num_samples//1e6:.0f}M_samples.npz"
    assert os.path.exists(filepath_g), FileNotFoundError(f"Data file not found: {filepath_g}. Please ensure the eigenchannel gains statistics file is available.")
    constant_power = "signal" if p_signal is not None else "noise"
    assert (p_signal is None) != (p_noise is None), "Exactly one of 'p_signal' or 'p_noise' input parameters must be provided to compute the other using the SNR values.\nWhich one to provide determines the scaling of the power levels in the plot."
    
    # STEP 1: Determine the optimal power allocation across the eigenchannels.
    ev_g = np.load(filepath_g, allow_pickle=True)["mean"]
    if constant_power == "signal": p_noise = p_signal / (10**(snr_dB / 10))
    elif constant_power == "noise": p_signal = (10**(snr_dB / 10)) * p_noise
    gamma = ev_g / p_noise
    pt = p_signal

    inv_cnr = (1 / gamma)
    p = waterfilling_v1(gamma, pt)
    
    # STEP 2: Create a plot visualizing the inverse channel-to-noise ratio and allocated power.
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(np.arange(1, min(Nt, Nr) + 1), inv_cnr, color='tab:grey', label='Inverse CNR')
    ax.bar(np.arange(1, min(Nt, Nr) + 1), p, bottom=inv_cnr, color='tab:blue', label='Allocated Power')
    ax.axhline(y = (inv_cnr[0] + p[0]), color='tab:red', linestyle='--', linewidth=3, label='Water Level')

    wl = (inv_cnr[0] + p[0])
    y_max = 1.1 * wl if wl > inv_cnr[-1] else 1.6 * wl

    text_inv_cnr = [ax.text(i + 1, 0.5*(inv_cnr[i] if inv_cnr[i] < y_max else y_max), rf"$\mathrm{{\gamma_{{{i+1}}}^{{-1}}}}$", ha='center', va='center', fontsize=10, visible=False) for i in range(min(Nt, Nr))]
    text_p = [ax.text(i + 1, inv_cnr[i] + 0.5*p[i], rf"$\mathrm{{P_{{{i+1}}}}}$", ha='center', va='center', fontsize=10, visible=False) for i in range(min(Nt, Nr))]

    for i in range(min(Nt, Nr)):
        if inv_cnr[i] / y_max > 0.1: text_inv_cnr[i].set_visible(True)
        if p[i] / y_max > 0.1: text_p[i].set_visible(True)

    ax.set_title(f'Power Allocation')
    ax.set_xlabel('Eigenchannel Index')
    ax.set_ylabel('Power [W]')
    ax.set_xticks(np.arange(1, min(Nt, Nr) + 1))
    ax.set_xlim(0.5, min(Nt, Nr) + 0.5)
    ax.set_ylim(0, y_max)
    ax.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(f'su-mimo/src/resource_allocation/power_allocation/plots/{Nt}x{Nr}' + f'__SNR_{snr_dB:.0f}dB.png')
    plt.close(fig)
    
    return fig, ax

def demo_waterfilling(Nt, Nr, snr_dB_list, p_signal=None, p_noise=None, num_samples=1e7):
    """
    Generate a waterfilling video plot for a SU-MIMO system across multiple SNR values.

    This function visualizes how the optimal power allocation across the eigenchannels changes with increasing SNR. The eigenchannel gains are taken as the expected values of the squared singular values of a random Gaussian zero-mean unit-variance channel matrix.

    The eigenchannels are shown horizontally (on the x-axis), and ordered by their expected gain. The inverse channel-to-noise ratio (ICNR), the allocated power per eigenchannel, and the water level are shown vertically (on the y-axis).

    Parameters
    ----------
    Nt : int
        Number of transmit antennas.
    Nr : int
        Number of receive antennas.
    snr_dB_list : ndarray, shape (N_snr_frames,), dtype=float
        Sequence of SNR values (in dB) to visualize.
    p_signal : float, optional
        Total transmit power. If provided, noise power is adjusted according to `snr_dB`.
    p_noise : float, optional
        Total noise power. If provided, transmit power is adjusted according to `snr_dB`.
    num_samples : int, optional
        Number of Monte Carlo samples used to compute the mean eigenchannel gains (default: 1e7).

    Returns
    -------
    animation : matplotlib.animation.FuncAnimation
        The animation object containing the water-filling video plot.

    Raises
    ------
    FileNotFoundError
        If the eigenchannel gains statistics file cannot be found.
    ValueError
        If neither or both of `p_signal` and `p_noise` are provided. Exactly one must be specified to compute the other using the SNR values.
    """


    # STEP 0: Parameters check.
    filepath_g = "su-mimo/src/resource_allocation/eigenchannel_gains/stats/" + f"{Nt}x{Nr}__{num_samples//1e6:.0f}M_samples.npz"
    assert os.path.exists(filepath_g), FileNotFoundError(f"Data file not found: {filepath_g}. Please ensure the eigenchannel gains statistics file is available.")
    constant_power = 'signal' if p_signal is not None else 'noise'
    assert (p_signal is None) != (p_noise is None), "Exactly one of 'p_signal' or 'p_noise' input parameters must be provided to compute the other using the SNR values.\nWhich one to provide determines the scaling of the power levels in the video."
    

    # STEP 1: Determine the optimal power allocation across the eigenchannels for each SNR value.
    ev_g = np.load(filepath_g, allow_pickle=True)["mean"]
    p = np.empty( (len(snr_dB_list), min(Nt, Nr)) )
    inv_cnr = np.empty( (len(snr_dB_list), min(Nt, Nr)) )
    
    for idx, snr_dB in enumerate(snr_dB_list):
        
        if constant_power == 'signal': p_noise = p_signal / (10**(snr_dB / 10))
        elif constant_power == 'noise': p_signal = (10**(snr_dB / 10)) * p_noise
        gamma = ev_g / p_noise
        pt = p_signal

        p[idx, :] = waterfilling_v1(gamma, pt)
        inv_cnr[idx, :] = (1 / gamma)
    
    # STEP 2: Create a video plot visualizing the inverse channel gains and allocated power for increasing SNR values.
    fig, ax = plt.subplots(figsize=(8, 5))

    bars_inv_cnr = ax.bar(np.arange(1, min(Nt, Nr) + 1), inv_cnr[0, :], color='tab:grey', label='Inverse CNR' + r' $(\gamma_{i}^{-1})$')
    bars_p = ax.bar(np.arange(1, min(Nt, Nr) + 1), p[0, :], bottom=inv_cnr[0, :], color='tab:blue', label='Allocated Power' + r' $(P_i)$')
    line_wl = ax.axhline(y = (inv_cnr[0, 0] + p[0, 0]), color='tab:red', linestyle='--', linewidth=3, label='Water Level')
    title = ax.set_title(f'SNR = {snr_dB_list[0]:.1f} dB')

    text_inv_cnr = [ax.text(i + 1, 0, rf"$\mathrm{{\gamma_{{{i+1}}}^{{-1}}}}$", ha='center', va='center', fontsize=10, visible=False) for i in range(min(Nt, Nr))]
    text_p = [ax.text(i + 1, 0, rf"$\mathrm{{P_{{{i+1}}}}}$", ha='center', va='center', fontsize=10, visible=False) for i in range(min(Nt, Nr))]

    ax.set_xlabel('Eigenchannel Index')
    ax.set_ylabel('Power [W]')
    ax.set_xticks(np.arange(1, min(Nt, Nr) + 1))
    ax.set_xlim(0.5, min(Nt, Nr) + 0.5)
    ax.legend(loc='upper left')
    fig.tight_layout()

    def update(snr_frame):
        
        # Update the bars heights that represent the inverse CNR and allocated power.
        for bar_inv_cnr, h_inv_cnr, bar_p, h_p in zip(bars_inv_cnr, inv_cnr[snr_frame], bars_p, p[snr_frame]): 
            bar_inv_cnr.set_height(h_inv_cnr)
            bar_p.set_height(h_p)
            bar_p.set_y(h_inv_cnr)
        
        # Update the water level height (red dashed line).
        wl = (inv_cnr[snr_frame][p[snr_frame] > 0] + p[snr_frame][p[snr_frame] > 0])[0]
        line_wl.set_ydata([wl, wl])
        
        # Update the scaling of the y-axis based on the current water level and maximum inverse CNR.
        if constant_power == 'signal':
            if wl < inv_cnr[snr_frame][-1]: y_max = 1.9 * wl
            elif inv_cnr[snr_frame][-1] < wl and 1.1*wl < 1.9*inv_cnr[snr_frame][-1]: y_max = 1.9 * inv_cnr[snr_frame][-1]
            elif 1.1*wl > 1.9*inv_cnr[snr_frame][-1]: y_max = 1.1 * wl
        elif constant_power == 'noise':
            if wl < 1.45*inv_cnr[snr_frame][-1]: y_max = 1.55 * inv_cnr[snr_frame][-1]
            elif wl > 1.45*inv_cnr[snr_frame][-1]: y_max = 1.55/1.45 * wl
        ax.set_ylim(0, y_max)

        # Update the position of the text labels on each bar.
        for i in range(min(Nt, Nr)):
            
            if inv_cnr[snr_frame][i] / y_max > 0.1:
                text_inv_cnr[i].set_position((i + 1, 0.5*(inv_cnr[snr_frame][i] if inv_cnr[snr_frame][i] < y_max else y_max)))
                text_inv_cnr[i].set_visible(True)
            else:
                text_inv_cnr[i].set_visible(False)

            if p[snr_frame][i] / y_max > 0.1:
                text_p[i].set_position((i + 1, inv_cnr[snr_frame][i] + 0.5*p[snr_frame][i]))
                text_p[i].set_visible(True)
            else:
                text_p[i].set_visible(False)

            
        # Update the title with the current SNR value.
        title.set_text(f'SNR = {snr_dB_list[snr_frame]:.1f} dB')
        
        return bars_p, bars_inv_cnr, line_wl, title, text_inv_cnr, text_p

    animation = FuncAnimation( fig, update, frames=len(snr_dB_list), blit=False)
    animation.save(f'su-mimo/src/resource_allocation/power_allocation/demos/{Nt}x{Nr}__' + ('Pt' if constant_power == 'signal' else 'N0') + '_constant.mp4', fps=30)
    plt.close(fig)

    return animation


# 2. BIT ALLOCATION

def adaptive_bit_allocation(gamma, p, B=0.5, R=1.0, c_type="QAM"):
    r"""
    Adaptive bit allocation for a SU-MIMO system based on the capacity of each eigenchannel.

    Each eigenchannel is allocated a constellation size (in bits), i.e. the number of bits per symbol, equal to a fraction of the capacity of the eigenchannel.

    Parameters
    ----------
    gamma : ndarray, shape (Ns,), dtype=float
        Channel-to-Noise Ratio (CNR) coefficients for each eigenchannel.
    p : ndarray, shape (Ns,), dtype=float
        Power allocation vector for each eigenchannel.
    B : float, optional
        The available bandwidth in Hz. Default is 0.5 Hz.
    R : float, optional
        The data rate. Represents the fraction of the eigenchannel capacity used for information transmission. Must be in the range (0, 1). Default is 1.0 (i.e., using the full capacity).
    c_type : str, optional
        The constellation type. Must be either 'PAM', 'PSK', or 'QAM'. Default is 'QAM'.

    Returns
    -------
    c : ndarray, shape (Ns,), dtype=float
        The capacity of each eigenchannel.
    ibr : ndarray, shape (Ns,), dtype=int
        The information bit rate (IBR) for each eigenchannel.
    """

    # STEP 0: Check the validity of the input parameters.
    assert len(p) == len(gamma), "The power allocation vector 'p' must have the same length as the CNR vector 'gamma'."
    assert B > 0, "The bandwidth 'B' must be positive."
    assert R >= 0 and R <= 1, "The data rate 'R' represents the fraction of the eigenchannel capacity used for information transmission and thus must be in the range [0, 1]."
    assert c_type in ['PAM', 'PSK', 'QAM'], "The modulation type 'c_type' must be either 'PAM', 'PSK', or 'QAM'."

    # STEP 1: Compute the capacity of each eigenchannel.
    c = 2*B * np.log2(1 + (gamma * p))

    # STEP 2: Compute the information bit rate (IBR) for each eigenchannel.
    ibr = np.floor( c * R ).astype(int) if c_type != 'QAM' else 2 * np.floor( (c * R) / 2 ).astype(int)

    return c, ibr


def plot_adaptive_bit_allocation(Nt, Nr, snr_dB, p_signal=1.0, pa_strategy="optimal", B=0.5, R=1.0, c_type="QAM", num_samples=1e7):
    r"""
    Generate a static plot for the capacity and information bit rate of the eigenchannels of a SU-MIMO system.

    This function visualizes the capacity and information bit rate of the eigenchannels of a single-user MIMO system for a given SNR value. The power allocation across the eigenchannels can be determined using different strategies, such as optimal waterfilling, equal power allocation, or eigenbeamforming. The eigenchannel gains are taken as the expected values of the squared singular values of a random Gaussian zero-mean unit-variance channel matrix.

    The eigenchannels are shown horizontally (on the x-axis), and ordered by their expected gain. The capacity and information bit rate for each eigenchannel are shown as vertical bars, with the capacity bars colored in red and the information bit rate bars colored in orange.

    Parameters
    ----------
    Nt : int
        Number of transmit antennas.
    Nr : int
        Number of receive antennas.
    snr_dB : float
        Signal-to-noise ratio (SNR) in decibels (dB).
    p_signal : float, optional
        The total available transmit power. Default is 1.0.
    pa_strategy : str, optional
        The power allocation strategy to use. Must be one of the following: 'optimal', 'equal', 'eigenbeamforming'. Default is 'optimal'.
    B : float, optional
        The available bandwidth in Hz. Default is 0.5 Hz.
    R : float, optional
        The data rate. Represents the fraction of the eigenchannel capacity used for information transmission. Must be in the range [0, 1]. Default is 1.0 (i.e., using the full capacity).
    c_type : str, optional
        The constellation type. Must be either 'PAM', 'PSK', or 'QAM'. Default is 'QAM'.
    num_samples : int, optional
        The number of channel realizations used to compute the expected eigenchannel gains. Default is 10 million (1e7).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    ax : matplotlib.axes.Axes
        The axes object containing the plot.
    """
    
    # STEP 0: Check the validity of the input parameters.
    assert pa_strategy in ['optimal', 'equal', 'eigenbeamforming'], "The power allocation strategy 'pa_strategy' must be one of the following: 'optimal', 'equal', 'eigenbeamforming'."
    
    filepath_g = "su-mimo/src/resource_allocation/eigenchannel_gains/stats/" + f"{Nt}x{Nr}__{num_samples//1e6:.0f}M_samples.npz"
    assert os.path.exists(filepath_g), FileNotFoundError(f"Data file not found: {filepath_g}. Please ensure the eigenchannel gains statistics file is available.")
    
    

    # STEP 1: Determine the capacity and information bit rate for the eigenchannels.
    ev_g = np.load(filepath_g, allow_pickle=True)["mean"]
    p_noise = p_signal / (10**(snr_dB / 10))
    gamma = ev_g / p_noise
    pt = p_signal

    if pa_strategy == "optimal": p = waterfilling_v1(gamma, pt)
    elif pa_strategy == "equal": p = equal_power_allocation(len(gamma[gamma > 0]), pt)
    elif pa_strategy == "eigenbeamforming": p = eigenbeamforming(len(gamma[gamma > 0]), pt)

    c, ibr = adaptive_bit_allocation(gamma, p, B=B, R=R, c_type=c_type)


    # STEP 2: Create a plot visualizing the capacity and information bit rate for the eigenchannels.
    fig, ax = plt.subplots(figsize=(8, 5))
        
    x = np.arange(min(Nt, Nr))
    ax.bar(x - 0.35/2, c, width=0.35, color='tab:red', label='Capacities ' + r'$\mathrm{C_i}$')
    ax.bar(x + 0.35/2, ibr, width=0.35, color='tab:orange', label='Information Bit Rates ' + r'$\mathrm{R_{b,i}}$')
    
    y_max = np.ceil(c[0]) + 0.5
    text_c = [ax.text(x[i] - 0.35/2, 0.5*c[i], rf"$\mathrm{{C_{{{i+1}}}}}$", ha='center', va='center', fontsize=10, visible=False) for i in range(min(Nt, Nr))]
    text_ibr = [ax.text(x[i] + 0.35/2, 0.5*ibr[i], rf"$\mathrm{{R_{{b,{i+1}}}}}$", ha='center', va='center', fontsize=10, visible=False) for i in range(min(Nt, Nr))]

    for i in range(min(Nt, Nr)):
        if c[i] / y_max > 0.1: text_c[i].set_visible(True)
        if ibr[i] / y_max > 0.1: text_ibr[i].set_visible(True)

    ax.set_title('Bit Allocation')
    ax.set_xlabel("Eigenchannel Index")
    ax.set_xticks(x)
    ax.set_xticklabels(x + 1)
    ax.set_xlim(-0.5, len(c) - 0.5)
    ax.set_ylabel('Bits [bps]')
    ax.set_ylim(0, y_max)
    ax.set_yticks(np.arange(0, y_max, 1, dtype=int))
    ax.grid(True, linestyle='dashed', alpha=0.4, axis='y')
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(f'su-mimo/src/resource_allocation/bit_allocation/plots/{Nt}x{Nr}_{c_type}' + f'__PA_{pa_strategy}' + f'__R_{R*100:.0f}' + f'__SNR_{snr_dB:.0f}dB.png')
    plt.close(fig)
    
    return fig, ax

def demo_adaptive_bit_allocation(Nt, Nr, snr_dB_list, p_signal=1.0, pa_strategy="optimal", B=0.5, R=1.0, c_type="QAM", num_samples=1e7):
    r"""
    Generate an animation for the capacity and information bit rate of the eigenchannels of a SU-MIMO system.

    This function visualizes the capacity and information bit rate of the eigenchannels of a single-user MIMO system for a range of SNR values. The power allocation across the eigenchannels can be determined using different strategies, such as optimal waterfilling, equal power allocation, or eigenbeamforming. The eigenchannel gains are taken as the expected values of the squared singular values of a random Gaussian zero-mean unit-variance channel matrix.

    The eigenchannels are shown horizontally (on the x-axis), and ordered by their expected gain. The capacity and information bit rate for each eigenchannel are shown as vertical bars, with the capacity bars colored in red and the information bit rate bars colored in orange. 
    
    The animation shows how the capacity and information bit rate of each eigenchannel evolve as a function of the SNR.

    Parameters
    ----------
    Nt : int
        Number of transmit antennas.
    Nr : int
        Number of receive antennas.
    snr_dB_list : array-like, shape (K,)
        List of SNR values in decibels (dB) for which to compute the capacity and information bit rate of the eigenchannels.
    p_signal : float, optional
        The total available transmit power. Default is 1.0.
    pa_strategy : str, optional
        The power allocation strategy to use. Must be one of the following: 'optimal', 'equal', 'eigenbeamforming'. Default is 'optimal'.
    B : float, optional
        The available bandwidth in Hz. Default is 0.5 Hz.
    R : float, optional
        The data rate. Represents the fraction of the eigenchannel capacity used for information transmission. Must be in the range [0, 1]. Default is 1.0 (i.e., using the full capacity).
    c_type : str, optional
        The constellation type. Must be either 'PAM', 'PSK', or 'QAM'. Default is 'QAM'.
    num_samples : int, optional
        The number of channel realizations used to compute the expected eigenchannel gains. Default is 10 million (1e7).

    Returns
    -------
    animation : matplotlib.animation.FuncAnimation
        The animation object visualizing the capacity and information bit rate of the eigenchannels as a function of the SNR.
    """
    
    # STEP 0: Check the validity of the input parameters.
    assert pa_strategy in ['optimal', 'equal', 'eigenbeamforming'], "The power allocation strategy 'pa_strategy' must be one of the following: 'optimal', 'equal', 'eigenbeamforming'."
    filepath_g = "su-mimo/src/resource_allocation/eigenchannel_gains/stats/" + f"{Nt}x{Nr}__{num_samples//1e6:.0f}M_samples.npz"
    assert os.path.exists(filepath_g), FileNotFoundError(f"Data file not found: {filepath_g}. Please ensure the eigenchannel gains statistics file is available.")
    

    # STEP 1: Determine the capacity and information bit rate of the eigenchannels for each SNR value.
    ev_g = np.load(filepath_g, allow_pickle=True)["mean"]
    p = np.empty( (len(snr_dB_list), min(Nt, Nr)) )
    c = np.empty( (len(snr_dB_list), min(Nt, Nr)) )
    ibr = np.empty( (len(snr_dB_list), min(Nt, Nr)), dtype=int )

    for idx, snr_dB in enumerate(snr_dB_list):
        
        p_noise = p_signal / (10**(snr_dB / 10))
        gamma = ev_g / p_noise
        pt = p_signal

        if pa_strategy == "optimal": p[idx, :] = waterfilling_v1(gamma, pt)
        elif pa_strategy == "equal": p[idx, :] = equal_power_allocation(len(gamma[gamma > 0]), pt)
        elif pa_strategy == "eigenbeamforming": p[idx, :] = eigenbeamforming(len(gamma[gamma > 0]), pt)

        c[idx, :], ibr[idx, :] = adaptive_bit_allocation(gamma, p[idx, :], B=B, R=R, c_type=c_type)
    

    # STEP 2: Create an animation visualizing the capacity and information bit rate of the eigenchannels as a function of the SNR.
    fig, ax = plt.subplots(figsize=(8, 5))
        
    x = np.arange(min(Nt, Nr))
    y_max = np.ceil(c).max() + 0.5

    bars_c = ax.bar(x - 0.35/2, c[0, :], width=0.35, color='tab:red', label='Capacities ' + r'$\mathrm{C_i}$')
    bars_ibr = ax.bar(x + 0.35/2, ibr[0, :], width=0.35, color='tab:orange', label='Information Bit Rates ' + r'$\mathrm{R_{b,i}}$')
    
    text_c = [ax.text(x[i] - 0.35/2, 0.5*c[0, i], rf"$\mathrm{{C_{{{i+1}}}}}$", ha='center', va='center', fontsize=10, visible=False) for i in range(min(Nt, Nr))]
    text_ibr = [ax.text(x[i] + 0.35/2, 0.5*ibr[0, i], rf"$\mathrm{{R_{{b,{i+1}}}}}$", ha='center', va='center', fontsize=10, visible=False) for i in range(min(Nt, Nr))]

    title = ax.set_title(f'SNR = {snr_dB_list[0]:.1f} dB')

    ax.set_xlabel("Eigenchannel Index")
    ax.set_xticks(x)
    ax.set_xticklabels(x + 1)
    ax.set_xlim(-0.5, min(Nt, Nr) - 0.5)
    ax.set_ylabel('Bits [bps]')
    ax.set_ylim(0, y_max)
    ax.set_yticks(np.arange(0, y_max, 1, dtype=int))
    ax.grid(True, linestyle='dashed', alpha=0.4, axis='y')
    ax.legend(loc='upper right')
    fig.tight_layout()

    def update(snr_frame):

        # Update the bars heights that represent the eigenchannel capacities and information bit rates.
        for i in range(min(Nt, Nr)):
            bars_c[i].set_height(c[snr_frame, i])
            bars_ibr[i].set_height(ibr[snr_frame, i])
        
        # Update the title with the current SNR value.
        title.set_text(f'SNR = {snr_dB_list[snr_frame]:.1f} dB')

        # Update the position of the text labels on each bar.
        for i in range(min(Nt, Nr)):
            
            if c[snr_frame][i] / y_max > 0.1:
                text_c[i].set_position((x[i] - 0.35/2, 0.5*(c[snr_frame][i] if c[snr_frame][i] < y_max else y_max)))
                text_c[i].set_visible(True)
            else:
                text_c[i].set_visible(False)

            if ibr[snr_frame][i] / y_max > 0.1:
                text_ibr[i].set_position((x[i] + 0.35/2, 0.5*(ibr[snr_frame][i] if ibr[snr_frame][i] < y_max else y_max)))
                text_ibr[i].set_visible(True)
            else:
                text_ibr[i].set_visible(False)
        
        return bars_c, bars_ibr, title, text_c, text_ibr
    
    animation = FuncAnimation( fig, update, frames=len(snr_dB_list), blit=False)
    animation.save(f'su-mimo/src/resource_allocation/bit_allocation/demos/{Nt}x{Nr}_{c_type}' + f'__PA_{pa_strategy}' + f'__R_{R*100:.0f}' + '.mp4', fps=50)
    plt.close(fig)

    return animation


# TESTS

if __name__ == "__main__":

    for snr_dB in range(-10, 31, 5):
        plot_waterfilling(Nt=4, Nr=4, snr_dB=snr_dB, p_signal=1)

    demo_waterfilling(Nt=4, Nr=4, snr_dB_list=np.arange(-10.00, 30.05, 0.05), p_signal=1)
    demo_waterfilling(Nt=4, Nr=4, snr_dB_list=np.arange(-10.00, 30.05, 0.05), p_noise=0.05)

    for c_type in ['QAM', 'PSK']:
    
        for snr_dB in range(-10, 31, 5):
            plot_adaptive_bit_allocation(Nt=4, Nr=4, snr_dB=snr_dB, c_type=c_type)

        demo_adaptive_bit_allocation(Nt=4, Nr=4, snr_dB_list=np.arange(-10.00, 30.05, 0.05), c_type=c_type)