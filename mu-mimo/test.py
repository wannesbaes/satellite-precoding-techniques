import numpy as np
import matplotlib.pyplot as plt

def _sample_elevation_angle(K: int, theta_max: float = 30 * (np.pi / 180)) -> np.ndarray:
    r"""
    Sample the elevation angles :math:`\theta_k` for each UT k according to the PDF
    derived from a uniform distribution of UTs on the circular ground disk of radius
    :math:`r_{\max} = h \tan(\theta_{\max})`:

    .. math::

        P_{\Theta^{\text{BS}}}(\theta)
        = \frac{2\,\tan\theta}{\tan^2(\theta_{\max})\,\cos^2\theta},
        \quad \theta \in [0, \theta_{\max}]

    Samples are drawn via the inverse CDF method. The CDF is

    .. math::

        F_\Theta(\theta) = \frac{\tan^2\theta}{\tan^2(\theta_{\max})}

    so that :math:`\theta = \arctan\!\bigl(\tan(\theta_{\max})\sqrt{U}\bigr)`
    with :math:`U \sim \mathrm{Uniform}[0, 1]`.

    Parameters
    ----------
    K : int
        The number of user terminals.
    theta_max : float, optional
        The maximum elevation angle [rad]. Default: 30°.
    """
    U = np.random.uniform(0.0, 1.0, size=K)
    theta = np.arctan(np.tan(theta_max) * np.sqrt(U))
    return theta

def _plot_elevation_angle_pdf(K_samples: int = 200_000, theta_max: float = 30 * (np.pi / 180)) -> None:
    """
    Plot the empirical histogram of sampled elevation angles against the theoretical PDF.

    Parameters
    ----------
    K_samples : int, optional
        Number of samples to draw for the histogram.
    theta_max : float, optional
        Maximum elevation angle [rad].
    """
    # --- samples ---
    theta_samples_deg = np.degrees(
        _sample_elevation_angle(K_samples, theta_max)
    )

    # --- theoretical PDF (in degrees) ---
    theta_plot_rad = np.linspace(0, theta_max, 500)
    # PDF per radian; convert to per degree by multiplying by pi/180
    pdf_per_rad = (2 * np.tan(theta_plot_rad)
                   / (np.tan(theta_max)**2 * np.cos(theta_plot_rad)**2))
    pdf_per_deg = pdf_per_rad * (np.pi / 180)

    # --- plot ---
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(theta_samples_deg, bins=120, density=True,
            alpha=0.6, label=f'Samples ($N={K_samples:,}$)')
    ax.plot(np.degrees(theta_plot_rad), pdf_per_deg,
            color='crimson', linewidth=2, label='Theoretical PDF')
    ax.set_xlabel(r'Elevation angle $\theta$ [°]')
    ax.set_ylabel(r'$P_\Theta(\theta)$ [1/°]')
    ax.set_title('Elevation Angle Distribution')
    ax.legend()
    fig.tight_layout()
    plt.show()


def _sample_BS_azimuth_angle(K: int) -> np.ndarray:
    r"""
    Sample the azimuth angles at the BS :math:`\phi_k^{\text{BS}}` for each UT k.

    The azimuth angle of the LoS path of UT :math:`k` at the BS is uniformly
    distributed within the UT's dedicated angular sector of width :math:`2\pi/K`:

    .. math::

        P_{\Phi^{\text{BS}}_k}(\phi) = \frac{K}{2\pi},
        \quad \phi \in \left[k\,\frac{2\pi}{K},\; (k+1)\,\frac{2\pi}{K}\right)

    Sampling directly:

    .. math::

        \phi_k^{\text{BS}} = \frac{2\pi}{K}\,(k + U_k), \quad U_k \sim \mathrm{Uniform}[0, 1)

    Parameters
    ----------
    K : int
        The number of user terminals.

    Returns
    -------
    phi_BS : np.ndarray, shape (K,)
        The sampled BS azimuth angles for each user terminal [rad], in :math:`[0, 2\pi)`.
    """
    k = np.arange(K)
    U = np.random.uniform(0.0, 1.0, size=K)
    phi_BS = (2 * np.pi / K) * (k + U)
    return phi_BS

def _plot_BS_azimuth_angle_pdf(K: int = 8, K_samples: int = 100_000) -> None:
    """
    Plot the empirical distribution of BS azimuth angles against the theoretical PDF.

    Each of the K users is shown in a distinct colour on a polar histogram,
    illustrating that each user occupies its own non-overlapping sector.

    Parameters
    ----------
    K : int, optional
        Number of user terminals (= number of sectors).
    K_samples : int, optional
        Number of independent channel realisations to draw per user.
    """
    # Draw K_samples realisations — each call gives one sample per user,
    # so we call the method K_samples times and stack.
    samples = np.stack(
        [_sample_BS_azimuth_angle(K) for _ in range(K_samples)],
        axis=0
    )  # shape (K_samples, K)

    sector_width = 2 * np.pi / K
    colors = plt.cm.tab10(np.linspace(0, 1, K))

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 6))

    bins = np.linspace(0, 2 * np.pi, 72 + 1)  # 5° bins

    for k in range(K):
        counts, _ = np.histogram(samples[:, k], bins=bins)
        # Normalise to probability density
        density = counts / (K_samples * np.diff(bins))
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        ax.bar(bin_centers, density, width=np.diff(bins),
               alpha=0.6, color=colors[k], label=f'UT {k}')

    # Theoretical PDF: K/(2π) within each sector, 0 outside
    phi_plot = np.linspace(0, 2 * np.pi, 1000)
    pdf_theory = np.full_like(phi_plot, K / (2 * np.pi))
    ax.plot(phi_plot, pdf_theory, color='black', linewidth=1.5,
            linestyle='--', label='Theoretical PDF')

    ax.set_title(f'BS Azimuth Angle Distribution  ($K={K}$)', pad=15)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=8)
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    #_plot_elevation_angle_pdf()
    _plot_BS_azimuth_angle_pdf()