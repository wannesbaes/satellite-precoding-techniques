

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def waterfilling_v1(gamma, pt):
    r"""
    Waterfilling algorithm for optimal power allocation across eigenchannels.

    This function implements the waterfilling algorithm to find the optimal power allocation across N transmission channels (eigenchannels in a single-user MIMO system, given the channel-to-noise ratio (CNR) coefficients `gamma` and the total available transmit power `pt`.

    In particular, it solves the following constraint optimization problem:

    .. math::

        \begin{aligned}
            & \underset{\{p_n\}}{\text{max}}
            & & \sum_{n=1}^{N} \log_2 \left( 1 + a_n \, p_n \right) \\
            & \text{s. t.}
            & & \sum_{n=1}^{N} p_n = p_t \\
            & & & \forall n \in \{1, \ldots, N\} : \, p_n \geq 0
        \end{aligned}

    Parameters
    ----------
    gamma : ndarray, shape (N,), dtype=float
        Channel-to-noise ratio (CNR) coefficients for each eigenchannel.
    pt : float
        Total available transmit power.

    Returns
    -------
    p : ndarray, shape (N,), dtype=float
        Optimal power allocation across the N eigenchannels.
    """


    # STEP 0: Check the validy of the input parameters.
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
        Signal-to-noise ratio in decibels.
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

    # 0. Parameters check.
    filepath_g = "su-mimo/waterfilling/eigenchannel_gains/stats/" + f"{Nt}x{Nr}__{num_samples//1e6:.0f}M_samples.npz"
    assert os.path.exists(filepath_g), FileNotFoundError(f"Data file not found: {filepath_g}. Please ensure the eigenchannel gains statistics file is available.")
    constant_power = 'signal' if p_signal is not None else 'noise'
    assert (p_signal is None) != (p_noise is None), "Exactly one of 'p_signal' or 'p_noise' input parameters must be provided to compute the other using the SNR values.\nWhich one to provide determines the scaling of the power levels in the video."
    
    # 1. Determine the optimal power allocation across the eigenchannels.
    ev_g = np.load(filepath_g, allow_pickle=True)["mean"]
    if constant_power == 'signal': p_noise = p_signal / (10**(snr_dB / 10))
    elif constant_power == 'noise': p_signal = (10**(snr_dB / 10)) * p_noise
    gamma = ev_g / p_noise
    pt = p_signal

    inv_cnr = (1 / gamma)
    p = waterfilling_v1(gamma, pt)
    
    # 2. Create a plot visualizing the inverse channel-to-noise ratio and allocated power.
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(np.arange(1, min(Nt, Nr) + 1), inv_cnr, color='tab:grey', label='Inverse CNR')
    ax.bar(np.arange(1, min(Nt, Nr) + 1), p, bottom=inv_cnr, color='tab:blue', label='Allocated Power')
    ax.axhline(y = (inv_cnr[0] + p[0]), color='tab:red', linestyle='--', linewidth=3, label='Water Level')

    wl = (inv_cnr[0] + p[0])
    y_max = 1.1 * wl if wl > inv_cnr[-1] else 1.6 * wl
    for eigenchannel_idx, (inv_cnr_i, p_i) in enumerate(zip(inv_cnr, p), start=1):
        if inv_cnr_i / y_max > 0.1: 
            ax.text(eigenchannel_idx, 0.5*inv_cnr_i if (inv_cnr_i < y_max) else 0.5*y_max, rf"$\mathrm{{\gamma_{{{eigenchannel_idx}}}^{{-1}}}}$", ha='center', va='center', fontsize=10)
        if p_i / y_max > 0.1: 
            ax.text(eigenchannel_idx, inv_cnr_i + 0.5*p_i, rf"$\mathrm{{P_{{{eigenchannel_idx}}}}}$", ha='center', va='center', fontsize=10)

    ax.set_xlabel('Eigenchannel Index')
    ax.set_ylabel('Power [W]')
    ax.set_xticks(np.arange(1, min(Nt, Nr) + 1))
    ax.set_xlim(0.5, min(Nt, Nr) + 0.5)
    ax.set_ylim(0, y_max)
    ax.set_title(f'Optimal Power Allocation ({Nt}x{Nr})\nSNR = {snr_dB:.0f} dB')
    ax.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(f'su-mimo/waterfilling/plots/{Nt}x{Nr}' + f'__SNR_{snr_dB:.0f}dB.png')
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


    # 0. Parameters check.
    filepath_g = "su-mimo/waterfilling/eigenchannel_gains/stats/" + f"{Nt}x{Nr}__{num_samples//1e6:.0f}M_samples.npz"
    assert os.path.exists(filepath_g), FileNotFoundError(f"Data file not found: {filepath_g}. Please ensure the eigenchannel gains statistics file is available.")
    constant_power = 'signal' if p_signal is not None else 'noise'
    assert (p_signal is None) != (p_noise is None), "Exactly one of 'p_signal' or 'p_noise' input parameters must be provided to compute the other using the SNR values.\nWhich one to provide determines the scaling of the power levels in the video."
    

    # 1. Determine the optimal power allocation across the eigenchannels for each SNR value.
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
    
    # 2. Create a video plot visualizing the inverse channel gains and allocated power for increasing SNR values.
    fig, ax = plt.subplots(figsize=(8, 5))

    bars_inv_cnr = ax.bar(np.arange(1, min(Nt, Nr) + 1), inv_cnr[0, :], color='tab:grey', label='Inverse CNR' + rf' $(\gamma_i)$')
    bars_p = ax.bar(np.arange(1, min(Nt, Nr) + 1), p[0, :], bottom=inv_cnr[0, :], color='tab:blue', label='Allocated Power' + rf' $(P_i)$')
    line_wl = ax.axhline(y = (inv_cnr[0, 0] + p[0, 0]), color='tab:red', linestyle='--', linewidth=3, label='Water Level')
    title = ax.set_title(f'Optimal Power Allocation ({Nt}x{Nr})\nSNR = {snr_dB_list[0]:.1f} dB')

    ax.set_xlabel('Eigenchannel Index')
    ax.set_ylabel('Power [W]')
    ax.set_xticks(np.arange(1, min(Nt, Nr) + 1))
    ax.set_xlim(0.5, min(Nt, Nr) + 0.5)
    ax.legend(loc='upper left')
    fig.tight_layout()

    def update(snr_frame):
        
        for bar_inv_cnr, h_inv_cnr, bar_p, h_p in zip(bars_inv_cnr, inv_cnr[snr_frame], bars_p, p[snr_frame]): 
            bar_inv_cnr.set_height(h_inv_cnr)
            bar_p.set_height(h_p)
            bar_p.set_y(h_inv_cnr)
        
        wl = (inv_cnr[snr_frame][p[snr_frame] > 0] + p[snr_frame][p[snr_frame] > 0])[0]
        line_wl.set_ydata([wl, wl])
        
        if constant_power == 'signal':
            if wl < inv_cnr[snr_frame][-1]: y_max = 1.9 * wl
            elif inv_cnr[snr_frame][-1] < wl and 1.1*wl < 1.9*inv_cnr[snr_frame][-1]: y_max = 1.9 * inv_cnr[snr_frame][-1]
            elif 1.1*wl > 1.9*inv_cnr[snr_frame][-1]: y_max = 1.1 * wl
        elif constant_power == 'noise':
            if wl < 1.45*inv_cnr[snr_frame][-1]: y_max = 1.55 * inv_cnr[snr_frame][-1]
            elif wl > 1.45*inv_cnr[snr_frame][-1]: y_max = 1.55/1.45 * wl
        ax.set_ylim(0, y_max)
        
        title.set_text(f'Optimal Power Allocation ({Nt}x{Nr})\nSNR = {snr_dB_list[snr_frame]:.1f} dB')
        
        return bars_p, bars_inv_cnr, line_wl, title

    animation = FuncAnimation( fig, update, frames=len(snr_dB_list), blit=False)
    animation.save(f'su-mimo/waterfilling/demos/{Nt}x{Nr}__' + ('Pt' if constant_power == 'signal' else 'N0') + '_constant.mp4', fps=25)
    plt.close(fig)

    return animation


if __name__ == "__main__":

    plot_waterfilling(Nt=4, Nr=4, snr_dB=10, p_signal=1, num_samples=1e7)
    #demo_waterfilling(Nt=4, Nr=4, snr_dB_list=np.arange(-10.00, 30.05, 0.05), p_signal=1, num_samples=1e7)
    #demo_waterfilling(Nt=4, Nr=4, snr_dB_list=np.arange(-10.00, 30.05, 0.05), p_noise=0.05, num_samples=1e7)