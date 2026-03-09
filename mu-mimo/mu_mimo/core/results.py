# mu-mimo/mu_mimo/core/results.py

from dataclasses import dataclass, field
import numpy as np
from pathlib import Path

from ..types import BitArray, IntArray, RealArray
from ..configs import SimConfig, SystemConfig


@dataclass
class SingleSnrSimResult:
    """
    The result of a simulation for a single SNR point, averaged over different channel realizations.
    
    Attributes
    ----------
    stream_ibrs : list[IntArray] (list of K arrays, each shape (Nr,))
        Per-UT per-stream information bit rates.
    stream_becs : list[RealArray] (list of K arrays, each shape (Nr,))
        Per-UT per-stream bit error counts.
    stream_ars : list[BitArray] (list of K arrays, each shape (Nr,))
        Per-UT per-stream stream activation rates (1 if the stream is active, 0 otherwise).
    
    ut_ibrs : IntArray, shape (K,)
        Per-UT information bit rates.
    ut_becs : RealArray, shape (K,)
        Per-UT bit error counts.
    ut_ars : BitArray, shape (K,)
        Per-UT UT activation rates (1 if the UT is active, 0 otherwise).
    
    ibr : float
        System-wide information bit rate.
    bec : float
        System-wide bit error count.
    
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
    
    stream_ibrs : list[IntArray]
    stream_becs : list[RealArray]
    stream_ars : list[BitArray]
    
    ut_ibrs : IntArray
    ut_becs : RealArray
    ut_ars : BitArray
    
    ibr : float
    bec : float

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
    filename : Path
        The path to the file where the simulation results are stored.
    sim_configs : SimConfig
        The configuration settings of the simulation.
    system_configs : SystemConfig
        The configuration settings of the system.
    """

    filename: Path
    
    sim_configs: SimConfig
    system_configs: SystemConfig

    snr_dB_values: RealArray
    simulation_results: list[SingleSnrSimResult]

    def display(self, configs: bool = False, detailed: bool = True, precision: int = 3) -> str:
        """
        Display simulation results in a readable table format.

        Parameters
        ----------
        configs : bool
            If True, also prints the system and simulation configuration settings in the header.
        detailed : bool
            If True, also prints per-UT metrics for each SNR point.
        precision : int
            Number of decimal digits for floating-point formatting.
        """


        if len(self.snr_dB_values) != len(self.simulation_results):
            raise ValueError( "Length mismatch: snr_dB_values and simulation_results must have the same length." )

        lines: list[str] = []

        # Title.
        lines.append("\n")
        lines.append(f"=" * 60)
        lines.append(f"  MU-MIMO Downlink Simulation Results")
        lines.append(f"=" * 60)

        # System configuration summary.
        if configs: lines.append(f"\n{self.system_configs.display()}")

        # Simulation configuration summary.
        if configs: lines.append(f"\n{self.sim_configs.display()}")

        # Results table.
        lines.append(f"\n\n  Simulation results:\n")
        
        header = " " + f"{'SNR [dB]':>10} | {'BER':>10} | {'IBR':>10} | " + (f"{'UT AR avg':>12} | {'Stream AR avg':>12}" if detailed else "")
        lines.append( " " + "-" * len(header))
        lines.append(header)
        lines.append( " " + "-" * len(header))

        for snr_db, res in zip(self.snr_dB_values, self.simulation_results):
            ber_str = f"{res.ber:.{precision}e}" if not np.isnan(res.ber) else "N/A"
            lines.append(" " + f"{int(snr_db):>10} | " + f"{ber_str:>10} | " + f"{int(res.ibr):>10} | " + (f"{res.ut_ars_avg:>12.1%} | " + f"{res.stream_ars_avg:>12.1%}" if detailed else "") )

            if detailed:
                lines.append("")
                for k in range(self.system_configs.K):
                    ut_ber_str = f"{res.ut_bers[k]:.{precision}e}" if not np.isnan(res.ut_bers[k]) else "N/A"
                    lines.append( f"        UT {k}: " + f"{ut_ber_str:>10} | " + f"{int(res.ut_ibrs[k]):>10} | " + f"{res.ut_ars[k]:>12.1%} | " + (f"{res.stream_ars[k].mean():>12.1%}" if res.stream_ars[k].size > 0 else "N/A"))
                lines.append(" " + "-" * len(header))

        # Return the formatted string.
        str_display = "\n".join(lines)
        return str_display

    def plot_performance_system(self):
        pass

    def plot_performance_uts(self):
        pass

    def plot_performance_streams(self):
        pass


__all__ = [
    "SingleSnrSimResult", "SimResult",
]
