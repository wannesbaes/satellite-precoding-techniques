# mu-mimo/mu_mimo/analytical/expected_achievable_rate.py

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from .channel_stats import ChannelStatistics, ChannelStatisticsData
from ..types import ComplexArray, RealArray
from ..configs import SystemConfig, setup_sys_configs
from ..processing.precoding import waterfilling_v1

SYSTEM_CONFIG_PATH = Path(__file__).parent.parent.parent / 'system_configs.json'

@dataclass
class ExpectedAchievableRateData:
    """
    """

    # System parameters.
    system_configs: SystemConfig
    num_channel_samples: int

    # Analytical results.
    snr_dB_R: RealArray
    R_system: RealArray
    R_uts: RealArray
    R_streams: RealArray


class ExpectedAchievableRate:

    @staticmethod
    def compute_ub(system_config: SystemConfig, snr_dB_values: RealArray = np.arange(-10, 31, 2.5), num_channel_samples: int = 1_000_000) -> ExpectedAchievableRateData:

        Pt = system_config.Pt
        B = system_config.B

        K = system_config.K
        Nr = system_config.Nr
        
        # Try to load the analytical results from an existing .npz file. If the file does not yet exist, compute the expected achievable rate.

        # Channel Statistics.
        channel_stats = ChannelStatistics(system_config, num_channel_samples)
        channel_stats_data = channel_stats.evaluate()


        # ...
        N = K*Nr
        num_bins = channel_stats_data.bin_edges.shape[1]

        g = np.zeros((N, num_bins), dtype=float)
        p_g = np.zeros((N, num_bins), dtype=float)

        for n in range(N):
            g[n] = (channel_stats_data.bin_edges[n][1:] + channel_stats_data.bin_edges[n][:-1])/2
            p_g[n] = channel_stats_data.histograms[n] / num_channel_samples

        # ...
        

        # ...

        R_streams = np.zeros((N, len(snr_dB_values)), dtype=float)

        for snr_dB in snr_dB_values:
            
            snr = 10**(snr_dB/10)
            Pn = Pt / N
            N0 = Pt / snr
            
            for n in range(N):
                R_streams[n] = 2*B * np.sum(np.log2(1 + g[n] * Pn / N0) * p_g[n])



        R_uts = None
        R_system = np.sum(R_uts)


        # Save the analytical results to a .npz file.
        
        
        return data
    
    @staticmethod
    def compute_ub_jensen(system_config: SystemConfig, snr_dB_values: RealArray = np.arange(-10, 31, 2.5), num_channel_samples: int = 1_000_000) -> ExpectedAchievableRateData:
        pass