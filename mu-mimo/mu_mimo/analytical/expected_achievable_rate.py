

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



