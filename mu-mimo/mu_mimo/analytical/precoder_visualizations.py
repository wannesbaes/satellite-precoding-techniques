# mu-mimo/mu_mimo/analytical/precoder_visualizations.py

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D

from ..processing.channel import IIDRayleighFadingChannelModel, RiceanFadingChannelModel
from ..processing.precoding import ZFPrecoder
from ..types import RealArray, ChannelStateInformation

def visualize_ZF_2x1(channel_model: str = "Rayleigh", H: RealArray | None = None) -> None:
    r"""
    Visualise the zero-forcing (ZF) precoding operation for a system with 2 transmit antennas at the base station (BS) and 1 single-antenna user (UT).

    The plot will show the following vectors in the 2D transmit signal space (spanned by the two transmit antennas):
        - 1 channel vector (:math:`\mathbf{h}_1`) that represents the direction of the channel to the UT.
        - 1 precoding vector (:math:`\mathbf{f}_1`) that represents the direction of the precoder of the data stream ment the UT.

    Because there is only one UT in the system, there is no interference to cancell. The precoder matrix can be chosen freely in the 2D transmit signal space, as long as it is not perpendicular to the channel vector. 
    Choosing the precoder direction to be parallel to the channel direction results in the highest possible signal-to-noise ratio (SNR) at the UT, and thus the best performance. This is the same precoder direction as the maximum ratio transmission (MRT) precoder.

    Parameters
    ----------
    channel_model : str, optional
        The channel model to use for generating the channel matrix. This parameter is only used if the channel matrix :math:`\mathbf{H}` is not provided.
        Options include 'Rayleigh' (default) and 'Rician'.
    H : RealArray shape (1, 2), optional
        The channel matrix. If not provided, a random channel vector will be generated according to the specified channel model.

    Note
    ----
    For visualization purposes, we assume each UT to have only one receive antenna (:math:`Nr = 1`).
    This allows us to visualize direction of the channel of each user as a vector, whereas in general the channel of each user correspond to the subspace (matrix) spanned Nr vectors.
    """

    # Input validation.
    assert H is None or H.shape == (1, 2), "The provided channel matrix H must have shape (1, 2) for this visualization."
    
    # Initialize the channel matrix H.
    if H is None:
        if channel_model == "Rayleigh": H = np.sqrt(2) * np.real(IIDRayleighFadingChannelModel(2, 1, 1).proceed())
        elif channel_model == "Rician": H = np.sqrt(2) * np.real(RiceanFadingChannelModel(2, 1, 1, K_rice=3.16, fD=9, mode='terrestrial').proceed())
        else: raise ValueError(f"Unsupported channel model: {channel_model}. Choose between 'Rayleigh' and 'Rician'.")
    
    # Compute the ZF precoder matrix F.
    F = ZFPrecoder.compute(csi=ChannelStateInformation(snr=10, H_eff=H), Pt=1.0, K=1)[0]


    # Construct the plot.
    fig, ax = plt.subplots(figsize=(7, 6))

    # colors.
    C_h1 = "#A52306"
    C_f1 = "#3F5A00"

    # vectors.
    h1 = H[0, :]
    f1 = F[:, 0]
    zo_h1, zo_f1 = (2, 3) if np.linalg.norm(h1) >= np.linalg.norm(f1) else (3, 2)

    ax.annotate("", xy=h1, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=C_h1, lw=2.5, mutation_scale=20), label=r"$\mathbf{h}_1$", zorder=zo_h1)
    ax.text(*(h1 + np.array([0.033, 0.075])), r"$\mathbf{h}_1$", color=C_h1, fontsize=13, ha="center", zorder=zo_h1)

    ax.annotate("", xy=f1, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=C_f1, lw=2.5, mutation_scale=20), label=r"$\mathbf{f}_1$", zorder=zo_f1)
    ax.text(*(f1 + np.array([0.033, 0.075])), r"$\mathbf{f}_1$", color=C_f1, fontsize=13, ha="center", zorder=zo_f1)
    
    # precoder has 1 degree of freedom.
    ax.set_facecolor((*plt.matplotlib.colors.to_rgb(C_f1), 0.20))

    # null space of the channel vector for UT1.
    h1_perp = np.array([-h1[1], h1[0]])
    h1_perp_norm = h1_perp / np.linalg.norm(h1_perp)
    lim = np.max(np.abs(np.concatenate([h1, f1]))) * 1.25
    ax.plot(np.linspace(-lim, lim, 100)*h1_perp_norm[0], np.linspace(-lim, lim, 100)*h1_perp_norm[1], linestyle='-.', color=C_h1, linewidth=2, label=r"$\perp \mathbf{h}_1$")
    
    # legend.
    handles = [
        mpatches.Patch(color=C_h1, label=r"$\mathbf{h}_1$ — channel direction for $UT_{1}$"),
        mpatches.Patch(color=C_f1, label=r"$\mathbf{f}_1$ — precoder direction for $UT_{1}$"),
        plt.Line2D([0], [0], color=C_h1, linestyle='--', lw=2, label=r"Null space of $\mathbf{h}_1$"),
    ]

    # plot settings.
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.axhline(0, color="black", lw=1.5)
    ax.axvline(0, color="black", lw=1.5)
    ax.set_xlabel(r"$\hat{e}_1$", fontsize=11)
    ax.set_ylabel(r"$\hat{e}_2$", fontsize=11)
    ax.set_aspect("equal")
    ax.grid(True, which="both", ls="--", lw=1.0, color="lightgray")
    ax.set_title("")
    ax.legend(handles=handles, fontsize=10, framealpha=0.9)

    plt.tight_layout()
    plt.show()

def visualize_ZF_2x2(channel_model: str = "Rayleigh", H: RealArray | None = None) -> None:
    pass

if __name__ == "__main__":
    
    visualize_ZF_2x1(H = np.array([[-0.25, 0.075]]))

