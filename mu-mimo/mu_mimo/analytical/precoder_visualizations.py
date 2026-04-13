# mu-mimo/mu_mimo/analytical/precoder_visualizations.py

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

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
    F = np.linalg.pinv(H)


    # Construct the plot.
    fig, ax = plt.subplots(figsize=(7, 6))

    # colors.
    C_h1 = "#A52306"
    C_f1 = "#3F5A00"

    # vectors.
    h1 = H[0, :]
    f1 = F[:, 0]

    # draw the vectors.
    zo_h1, zo_f1 = (2, 3) if np.linalg.norm(h1) >= np.linalg.norm(f1) else (3, 2)
    
    ax.annotate("", xy=h1, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=C_h1, lw=2.5, mutation_scale=20), label=r"$\mathbf{h}_1$", zorder=zo_h1)
    ax.text(*(h1 + np.array([0.025, 0.075])), r"$\mathbf{h}_1$", color=C_h1, fontsize=13, ha="center", zorder=zo_h1)

    ax.annotate("", xy=f1, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=C_f1, lw=2.5, mutation_scale=20), label=r"$\mathbf{f}_1$", zorder=zo_f1)
    ax.text(*(f1 + np.array([0.025, 0.050])), r"$\mathbf{f}_1$", color=C_f1, fontsize=13, ha="center", zorder=zo_f1)
    
    # draw the null-space of the channel.
    h1_perp = np.array([-h1[1], h1[0]])
    h1_perp_norm = h1_perp / np.linalg.norm(h1_perp)
    lim = np.max(np.abs(np.concatenate([h1, f1]))) * 1.25
    
    ax.set_facecolor((*plt.matplotlib.colors.to_rgb(C_f1), 0.20))
    ax.plot(np.linspace(-2*lim, 2*lim, 200)*h1_perp_norm[0], np.linspace(-2*lim, 2*lim, 200)*h1_perp_norm[1], linestyle='-', color="white", linewidth=2, label=r"$\perp \mathbf{h}_1$")
    
    # legend.
    handles = [
        mpatches.Patch(color=C_h1, label=r"$\mathbf{h}_1$ — channel direction for $UT_{1}$"),
        mpatches.Patch(color=C_f1, label=r"$\mathbf{f}_1$ — precoder direction for $UT_{1}$"),
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
    r"""
    Visualise the zero-forcing (ZF) precoding operation for a system with 2 transmit antennas at the base station (BS) and 2 single-antenna users (UTs).

    The plot will show the following vectors in the 2D transmit signal space (spanned by the two transmit antennas):
        - 2 channel vectors (:math:`\mathbf{h}_1` and :math:`\mathbf{h}_2`) that represent the directions of the channel to the UTs.
        - 2 precoding vectors (:math:`\mathbf{f}_1` and :math:`\mathbf{f}_2`) that represent the directions of the precoders of the data streams meant for each UT.
    
    The ZF precoder is designed to cancel the interference between the two UTs. This means that the precoding vector for each UT is chosen to be perpendicular to the channel vector of the other UT.
    
    Parameters
    ----------
    channel_model : str, optional
        The channel model to use for generating the channel matrix. This parameter is only used if the channel matrix :math:`\mathbf{H}` is not provided.
        Options include 'Rayleigh' (default) and 'Rician'.
    H : RealArray shape (2, 2), optional
        The channel matrix. If not provided, a random channel matrix will be generated according to the specified channel model.

    Note
    ----
    For visualization purposes, we assume each UT to have only one receive antenna (:math:`Nr = 1`).
    This allows us to visualize direction of the channel of each user as a vector, whereas in general the channel of each user correspond to the subspace (matrix) spanned Nr vectors.
    """
    
    # Input validation.
    assert H is None or H.shape == (2, 2), "The provided channel matrix H must have shape (2, 2) for this visualization."

    # Initialize the channel matrix H.
    if H is None:
        if channel_model == "Rayleigh": H = np.sqrt(2) * np.real(IIDRayleighFadingChannelModel(2, 1, 2).proceed())
        elif channel_model == "Rician": H = np.sqrt(2) * np.real(RiceanFadingChannelModel(2, 1, 2, K_rice=3.16, fD=9, mode='terrestrial').proceed())
        else: raise ValueError(f"Unsupported channel model: {channel_model}. Choose between 'Rayleigh' and 'Rician'.")

    # Compute the ZF precoder matrix F.
    F = np.linalg.pinv(H)

    # Validate the ZF condition.
    print(f"H · F = \n{np.round(H @ F, 2)}\n")


    # Construct the plot.
    fig, ax = plt.subplots(figsize=(7, 6))

    # colors.
    C_h1 = "#CA2500"
    C_h2 = "#730505"
    C_f1 = "#597E00"
    C_f2 = "#01520B"

    # vectors.
    h1, h2 = H[0, :], H[1, :]
    f1, f2 = F[:, 0], F[:, 1]
    lim = np.max(np.abs(np.concatenate([h1, h2, f1, f2]))) * 1.3

    # draw null-space lines.
    h1_perp = np.array([-h1[1], h1[0]]) / np.linalg.norm(h1)
    h2_perp = np.array([-h2[1], h2[0]]) / np.linalg.norm(h2)
    t = np.linspace(-2*lim, 2*lim, 200)
    
    ax.plot(t*h2_perp[0], t*h2_perp[1], linestyle='-.', color=C_f1, linewidth=1.5, zorder=1)
    ax.plot(t*h1_perp[0], t*h1_perp[1], linestyle='-.', color=C_f2, linewidth=1.5, zorder=1)

    # draw right-angle markers.
    def _right_angle_2d(u: RealArray, v: RealArray, size: float) -> None:
        u_n, v_n = u / np.linalg.norm(u), v / np.linalg.norm(v)
        pts = np.array([size*u_n, size*(u_n + v_n), size*v_n])
        ax.plot(pts[:, 0], pts[:, 1], color="dimgray", lw=1.2, zorder=5)

    _right_angle_2d(-f1, h2, np.min([np.linalg.norm(f1), np.linalg.norm(h2)]) * 0.175)
    _right_angle_2d(-f2, h1, np.min([np.linalg.norm(f2), np.linalg.norm(h1)]) * 0.175)

    # draw channel and precoder directions.
    def _draw_vector_2d(v: RealArray, color: str, lbl: str, lbl_pos: tuple[float, float], zo: int) -> None:
        ax.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=color, lw=2.5, mutation_scale=20), zorder=zo)
        ax.text(*(v + lbl_pos), lbl, color=color, fontsize=13, ha="center", zorder=zo)

    for zo, (v, col, lbl, lbl_pos) in enumerate(sorted([(h1, C_h1, r"$\mathbf{h}_1$", (0.2, -0.2)), (h2, C_h2, r"$\mathbf{h}_2$", (0, 0.1)), (f1, C_f1, r"$\mathbf{f}_1$", (0.25, 0.025)), (f2, C_f2, r"$\mathbf{f}_2$", (0.25, 0))], key=lambda x: -np.linalg.norm(x[0])), start=2):
        _draw_vector_2d(v, col, lbl, lbl_pos, zo)

    # legend.
    handles = [
        mpatches.Patch(color=C_h1, label=r"$\mathbf{h}_1$ — channel direction for $UT_1$"),
        mpatches.Patch(color=C_h2, label=r"$\mathbf{h}_2$ — channel direction for $UT_2$"),
        mpatches.Patch(color=C_f1, label=r"$\mathbf{f}_1$ — precoder for $UT_1$"),
        mpatches.Patch(color=C_f2, label=r"$\mathbf{f}_2$ — precoder for $UT_2$"),
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

def visualize_ZF_3x2(channel_model: str = "Rayleigh", H: RealArray | None = None) -> None:
    r"""
    Visualise the zero-forcing (ZF) precoding operation for a system with 3 transmit antennas at the base station (BS) and 2 single-antenna users (UTs).

    The plot will show the following vectors in the 3D signal space (spanned by the three transmit antennas):
        - 2 channel vectors (:math:`\mathbf{h}_1` and :math:`\mathbf{h}_2`) that represent the directions of the channel to the UTs.
        - 2 precoding vectors (:math:`\mathbf{f}_1` and :math:`\mathbf{f}_2`) that represent the directions of the precoders of the data streams meant for each UT.

    The ZF precoder is designed to cancel the interference between the two UTs. This means that the precoding vector for each UT is chosen to be perpendicular to the channel vector of the other UT, while still being as aligned as possible with the channel vector of its intended UT.

    Parameters
    ----------
    channel_model : str, optional
        The channel model to use for generating the channel matrix. This parameter is only used if the channel matrix :math:`\mathbf{H}` is not provided.
        Options include 'Rayleigh' (default) and 'Rician'.
    H : RealArray shape (2, 3), optional
        The channel matrix. If not provided, a random channel matrix will be generated according to the specified channel model.

    Note
    ----
    For visualization purposes, we assume each UT to have only one receive antenna (:math:`Nr = 1`).
    This allows us to visualize direction of the channel of each user as a vector, whereas in general the channel of each user correspond to the subspace (matrix) spanned Nr vectors.
    Analogously, the precoder for each user is visualized as a vector, whereas in general the precoder for each user correspond to the subspace (matrix) spanned by the precoding vectors of all data streams meant for that user. This space is then chosen to be a subspace of the null space of the channel of the other user(s) to achieve interference cancellation.
    """

    # Input validation.
    assert H is None or H.shape == (2, 3), "The provided channel matrix H must have shape (2, 3) for this visualization."

    # Initialize the channel matrix H.
    if H is None:
        if channel_model == "Rayleigh": H = np.sqrt(2) * np.real(IIDRayleighFadingChannelModel(3, 1, 2).proceed())
        elif channel_model == "Rician": H = np.sqrt(2) * np.real(RiceanFadingChannelModel(3, 1, 2, K_rice=3.16, fD=9, mode='terrestrial').proceed())
        else: raise ValueError(f"Unsupported channel model: {channel_model}. Choose between 'Rayleigh' and 'Rician'.")

    # Compute the ZF precoder matrix.
    F = np.linalg.pinv(H)


    # Construct the plot.
    fig = plt.figure(figsize=(10, 8), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    lim = np.max([np.max(np.abs(F)), np.max(np.abs(H))]) * 1.3

    # colors.
    C_h1 = "#CA2500"
    C_h2 = "#510000"
    C_f1 = "#597E00"
    C_f2 = "#013A08"

    # vectors.
    h1, h2 = H[0, :], H[1, :]
    f1, f2 = F[:, 0], F[:, 1]

    # draw null-space planes.
    def _draw_shaded_null_plane(normal: np.ndarray, color: str, alpha: float = 0.25) -> None:
        n = normal / np.linalg.norm(normal)
        _, _, Vh = np.linalg.svd(n.reshape(1, -1))
        b1, b2 = Vh[1], Vh[2]
        t = np.array([-lim, lim])
        S, T = np.meshgrid(t, t)
        ax.plot_surface(S*b1[0] + T*b2[0], S*b1[1] + T*b2[1], S*b1[2] + T*b2[2], color=color, alpha=alpha)

    _draw_shaded_null_plane(h2, color=C_f1)
    _draw_shaded_null_plane(h1, color=C_f2)

    # draw right-angle markers.
    def _right_angle_3d(u: np.ndarray, v: np.ndarray, size: float) -> None:
        u_n, v_n = u / np.linalg.norm(u), v / np.linalg.norm(v)
        pts = np.array([size*u_n, size*(u_n + v_n), size*v_n])
        ax.plot3D(*pts.T, color="dimgray", lw=1.2)

    _right_angle_3d(-f1, h2, lim * 0.1)
    _right_angle_3d(-f2, h1, lim * 0.1)

    # draw precoder directions in the channel null-space.
    f1_n, f2_n = f1 / np.linalg.norm(f1), f2 / np.linalg.norm(f2)
    t = np.linspace(-lim, lim, 200)
    ax.plot3D(t*f1_n[0], t*f1_n[1], t*f1_n[2], linestyle='-.', color=C_f1, linewidth=1.5)
    ax.plot3D(t*f2_n[0], t*f2_n[1], t*f2_n[2], linestyle='-.', color=C_f2, linewidth=1.5)

    # draw channel and precoder directions.
    def _draw_vector(v: RealArray, color: str, lbl: str, lbl_pos: tuple[float, float, float]) -> None:
        ax.quiver(0, 0, 0, v[0], v[1], v[2], color=color, arrow_length_ratio=0.12, linewidth=2.5)
        v_n = v / np.linalg.norm(v)
        ax.text(*(v + v_n * lim * 0.08 + np.array(lbl_pos)), lbl, color=color, fontsize=12, ha="center")

    for v, col, lbl, lbl_pos in sorted([(h1, C_h1, r"$\mathbf{h}_1$", (0.2, 0, -0.2)), (h2, C_h2, r"$\mathbf{h}_2$", (-0.3, 0, 0.1)), (f1, C_f1, r"$\mathbf{f}_1$", (0.2, 0, 0.1)), (f2, C_f2, r"$\mathbf{f}_2$", (0.2, 0, 0))], key=lambda x: -np.linalg.norm(x[0]) ):
        _draw_vector(v, col, lbl, lbl_pos)
    
    # draw the axes.
    # ax.plot3D([-lim, lim], [0, 0], [0, 0], color="black", lw=0.75)
    # ax.plot3D([0, 0], [-lim, lim], [0, 0], color="black", lw=0.75)
    # ax.plot3D([0, 0], [0, 0], [-lim, lim], color="black", lw=0.75)

    # legend.
    handles = [
        mpatches.Patch(color=C_h1, label=r"$\mathbf{h}_1$ — channel direction for $UT_1$"),
        mpatches.Patch(color=C_h2, label=r"$\mathbf{h}_2$ — channel direction for $UT_2$"),
        mpatches.Patch(color=C_f1, label=r"$\mathbf{f}_1$ — precoder for $UT_1$"),
        mpatches.Patch(color=C_f2, label=r"$\mathbf{f}_2$ — precoder for $UT_2$"),
        mpatches.Patch(color=C_f1, alpha=0.5, label=r"null-space of $\mathbf{h}_2$"),
        mpatches.Patch(color=C_f2, alpha=0.5, label=r"null-space of $\mathbf{h}_1$"),
    ]
    
    fig_legend = plt.figure(figsize=(3, 2), dpi=200)
    ax_legend = fig_legend.add_subplot(111)
    ax_legend.axis("off")
    ax_legend.legend(handles=handles, fontsize=10, framealpha=0.9, loc="center")

    fig_legend.savefig(str(Path.home() / "Desktop" / "legend_3x2.png"), bbox_inches="tight", dpi=300)
    plt.close(fig_legend)

    # plot settings.
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])
    ax.set_xlabel(r"$\hat{e}_1$", fontsize=11, labelpad=6)
    ax.set_ylabel(r"$\hat{e}_2$", fontsize=11, labelpad=6)
    ax.set_zlabel(r"$\hat{e}_3$", fontsize=11, labelpad=6)
    ax.set_title("")
    

    plt.tight_layout()
    plt.show()

def visualize_ZF_3x3(channel_model: str = "Rayleigh", H: RealArray | None = None) -> None:
    r"""
    Visualise the zero-forcing (ZF) precoding operation for a system with 3 transmit antennas at the base station (BS) and 3 single-antenna users (UTs).

    The plot will show the following vectors in the 3D signal space (spanned by the three transmit antennas):
        - 3 channel vectors (:math:`\mathbf{h}_1`, :math:`\mathbf{h}_2`, and :math:`\mathbf{h}_3`) that represent the directions of the channel to the UTs.
        - 3 precoding vectors (:math:`\mathbf{f}_1`, :math:`\mathbf{f}_2`, and :math:`\mathbf{f}_3`) that represent the directions of the precoders of the data streams meant for each UT.

    The ZF precoder is designed to cancel the interference between the three UTs. This means that the precoding vector for each UT is chosen to be perpendicular to the channel vectors of the other two UTs.

    Parameters
    ----------
    channel_model : str, optional
        The channel model to use for generating the channel matrix. This parameter is only used if the channel matrix :math:`\mathbf{H}` is not provided.
        Options include 'Rayleigh' (default) and 'Rician'.
    H : RealArray shape (3, 3), optional
        The channel matrix. If not provided, a random channel matrix will be generated according to the specified channel model.

    Note
    ----
    For visualization purposes, we assume each UT to have only one receive antenna (:math:`Nr = 1`).
    This allows us to visualize direction of the channel of each user as a vector, whereas in general the channel of each user correspond to the subspace (matrix) spanned Nr vectors.
    Analogously, the precoder for each user is visualized as a vector, whereas in general the precoder for each user correspond to the subspace (matrix) spanned by the precoding vectors of all data streams meant for that user. This space is then chosen to be a subspace of the null space of the channel of the other user(s)
    """
        # Input validation.
    assert H is None or H.shape == (3, 3), "The provided channel matrix H must have shape (3, 3) for this visualization."

    # Initialize the channel matrix H.
    if H is None:
        if channel_model == "Rayleigh": H = np.sqrt(2) * np.real(IIDRayleighFadingChannelModel(3, 1, 3).proceed())
        elif channel_model == "Rician": H = np.sqrt(2) * np.real(RiceanFadingChannelModel(3, 1, 3, K_rice=3.16, fD=9, mode='terrestrial').proceed())
        else: raise ValueError(f"Unsupported channel model: {channel_model}. Choose between 'Rayleigh' and 'Rician'.")

    # Compute the ZF precoder matrix F.
    F = np.linalg.pinv(H)


    # Construct the plot.
    fig = plt.figure(figsize=(10, 8), dpi=115)
    ax = fig.add_subplot(111, projection="3d")

    lim = np.max([np.max(np.abs(F)), np.max(np.abs(H))]) * 1.3

    # colors.
    C_h1 = "#CF0000"
    C_h2 = "#A10000"
    C_h3 = "#610000"
    C_f1 = "#809745"
    C_f2 = "#546F02"
    C_f3 = "#2E4500"

    # vectors.
    h1, h2, h3 = H[0, :], H[1, :], H[2, :]
    f1, f2, f3 = F[:, 0], F[:, 1], F[:, 2]

    # right-angle markers.
    def _right_angle_3d(u: RealArray, v: RealArray, size: float) -> None:
        return
        u_n, v_n = u / np.linalg.norm(u), v / np.linalg.norm(v)
        pts = np.array([size*u_n, size*(u_n + v_n), size*v_n])
        ax.plot3D(*pts.T, color="dimgray", lw=1.2)

    size = lim * 0.1
    _right_angle_3d(f1, h2, size)
    _right_angle_3d(f1, h3, size)
    _right_angle_3d(f2, h1, size)
    _right_angle_3d(f2, h3, size)
    _right_angle_3d(f3, h1, size)
    _right_angle_3d(f3, h2, size)

    # draw channel and precoder directions.
    def _draw_vector(v: RealArray, color: str, lbl: str) -> None:
        ax.quiver(0, 0, 0, v[0], v[1], v[2], color=color, arrow_length_ratio=0.1, linewidth=2.5)
        v_n = v / np.linalg.norm(v)
        ax.text(*(v + v_n * lim * 0.05), lbl, color=color, fontsize=12, ha="center")

    for v, col, lbl in sorted( [(h1, C_h1, r"$\mathbf{h}_1$"), (h2, C_h2, r"$\mathbf{h}_2$"), (h3, C_h3, r"$\mathbf{h}_3$"), (f1, C_f1, r"$\mathbf{f}_1$"), (f2, C_f2, r"$\mathbf{f}_2$"), (f3, C_f3, r"$\mathbf{f}_3$")], key=lambda x: -np.linalg.norm(x[0]) ):
        _draw_vector(v, col, lbl)

    # legend.
    handles = [
        mpatches.Patch(color=C_h1, label=r"$\mathbf{h}_1$ — channel direction for $UT_1$"),
        mpatches.Patch(color=C_h2, label=r"$\mathbf{h}_2$ — channel direction for $UT_2$"),
        mpatches.Patch(color=C_h3, label=r"$\mathbf{h}_3$ — channel direction for $UT_3$"),
        mpatches.Patch(color=C_f1, label=r"$\mathbf{f}_1$ — precoder for $UT_1$  ($\perp\,\mathbf{h}_2,\,\mathbf{h}_3$)"),
        mpatches.Patch(color=C_f2, label=r"$\mathbf{f}_2$ — precoder for $UT_2$  ($\perp\,\mathbf{h}_1,\,\mathbf{h}_3$)"),
        mpatches.Patch(color=C_f3, label=r"$\mathbf{f}_3$ — precoder for $UT_3$  ($\perp\,\mathbf{h}_1,\,\mathbf{h}_2$)"),
    ]

    fig_legend = plt.figure(figsize=(3, 2), dpi=200)
    ax_legend = fig_legend.add_subplot(111)
    ax_legend.axis("off")
    ax_legend.legend(handles=handles, fontsize=10, framealpha=0.9, loc="center")

    fig_legend.savefig(str(Path.home() / "Desktop" / "legend_3x3.png"), bbox_inches="tight", dpi=300)
    plt.close(fig_legend)

    # plot settings.
    xlim_min, xlim_max = np.min([h1[0], h2[0], h3[0], f1[0], f2[0], f3[0]]) * 1.25, np.max([h1[0], h2[0], h3[0], f1[0], f2[0], f3[0]]) * 1.25
    ax.set_xlim([xlim_min, xlim_max])
    ylim_min, ylim_max = np.min([h1[1], h2[1], h3[1], f1[1], f2[1], f3[1]]) * 1.25, np.max([h1[1], h2[1], h3[1], f1[1], f2[1], f3[1]]) * 1.25
    ax.set_ylim([ylim_min, ylim_max])
    zlim_min, zlim_max = np.min([h1[2], h2[2], h3[2], f1[2], f2[2], f3[2]]) * 1.25, np.max([h1[2], h2[2], h3[2], f1[2], f2[2], f3[2]]) * 1.25
    ax.set_zlim([zlim_min, zlim_max])
    ax.set_xlabel(r"$\hat{e}_1$", fontsize=11, labelpad=6)
    ax.set_ylabel(r"$\hat{e}_2$", fontsize=11, labelpad=6)
    ax.set_zlabel(r"$\hat{e}_3$", fontsize=11, labelpad=6)
    ax.set_title("")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    # CASE 1: Nt=2, K=1.
    alpha = np.pi / 9
    H = np.array([[0.75*np.cos(alpha), 0.75*np.sin(alpha)]])
    visualize_ZF_2x1(H = H)
    visualize_ZF_2x1()
    
    # CASE 2: Nt=2, K=2.
    alpha = np.pi / 9
    epsilon = np.pi / 18
    
    # favorable scenario.
    H_favorable = np.array([[np.cos(alpha - epsilon/2),            np.sin(alpha - epsilon/2)], 
                            [np.cos(alpha + np.pi/2 + epsilon/2),  np.sin(alpha + np.pi/2 + epsilon/2)]])
    visualize_ZF_2x2(H = H_favorable)

    # unfavorable scenario.
    H_unfavorable = 1.25 * np.array([[np.cos(alpha - epsilon/2),  np.sin(alpha - epsilon/2)], 
                                     [np.cos(alpha + epsilon/2),  np.sin(alpha + epsilon/2)]])
    visualize_ZF_2x2(H = H_unfavorable)

    # random scenario.
    visualize_ZF_2x2()
    

    # CASE 3:  Nt=3, K=2.

    alpha = np.pi / 9
    epsilon = np.pi / 18
    
    # favorable scenario.
    H_favorable = np.array([[np.cos(alpha - epsilon/2),           0.0, np.sin(alpha - epsilon/2)], 
                            [np.cos(alpha + np.pi/2 + epsilon/2), 0.0, np.sin(alpha + np.pi/2 + epsilon/2)]])
    visualize_ZF_3x2(H = H_favorable)
    
    # unfavorable scenario.
    H_unfavorable = 1.25 * np.array([[np.cos(alpha - epsilon/2), 0.0, np.sin(alpha - epsilon/2)], 
                                     [np.cos(alpha + epsilon/2), 0.0, np.sin(alpha + epsilon/2)]])
    visualize_ZF_3x2(H = H_unfavorable)

    # random scenario.
    visualize_ZF_3x2()
    

    # CASE 4: Nt=3, K=3.
    
    alpha = np.pi / 9
    epsilon = np.pi / 15

    # favorable scenario.
    H_favorable = np.array([[np.cos(alpha - epsilon/2),             0.0,    np.sin(alpha - epsilon/2)], 
                            [np.cos(alpha + np.pi/2 + epsilon/2),   0.0,    np.sin(alpha + np.pi/2 + epsilon/2)], 
                            [np.cos(np.pi/2 - epsilon),             1.0,    np.sin(epsilon)]])
    visualize_ZF_3x3(H = H_favorable)
    
    # unfavorable scenario.
    H_unfavorable = 1.25 * np.array([[np.cos(alpha - epsilon/2),   -0.1,    np.sin(alpha - epsilon/2)], 
                                     [np.cos(alpha + epsilon/2),   -0.1,    np.sin(alpha + epsilon/2)], 
                                     [np.cos(alpha),                0.2,    np.sin(alpha)]])
    visualize_ZF_3x3(H = H_unfavorable)

    # random scenario.
    visualize_ZF_3x3()
