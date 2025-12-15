import numpy as np

def waterfilling(alpha, a, pt, epsilon=None):
    """
    Description
    -----------
    Implementation of the waterfilling algorithm (through a bisection search) to find the optimal power allocation. 

    More generally, this function solves the following optimization problem:
    \begin{equation}
        \begin{aligned}
            & \underset{\mathbf{p}}{\text{max}}
            & & \sum_{n=1}^{N} \alpha_n \log_2 \left( 1 + a_n \, p_n \right) \\
            & \text{s. t.}
            & & \sum_{n=1}^{N} p_n = p_t \\
            & & & \forall n \in \{1, \ldots, N\} : \, p_n \geq 0
        \end{aligned}
    \end{equation}
    
    Paramaters
    ----------
    alpha : 1D numpy array (dtype: float, shape: (N,))
        Weighting factors for each transmission channel. They determine the relative importance of each channel in the optimization problem. Their sum should be equal to 1 (normalized).
    a : 1D numpy array (dtype: float, shape: (N,))
        Effective signal-to-noise ratio (SNR) coefficients for each transmission channel when unit transmit power is allocated.
    pt : float
        Total available transmit power.

    Returns
    -------
    p : 1D numpy array (dtype: float, shape: (N,))
        Optimal power allocation across the N transmission channels.

    """

    # Initialization.
    p = lambda mu, alpha, a: np.clip((alpha / (mu * np.log(2))) - (1 / a), min=0, max=None)
    epsilon = pt * 10e-6 if epsilon is None else epsilon
    
    l = 0
    u = ( (alpha[0]*a[0]) / (np.log(2)*(1+a[0]*pt)) )
    mu = (l+u)/2
    pt_iter = np.sum(p(mu, alpha, a))

    # Iteration.
    while np.abs(pt - pt_iter) > epsilon:

        if pt > pt_iter: l = mu
        elif pt < pt_iter: u = mu
        mu = (l+u)/2
        pt_iter = np.sum(p(mu, alpha, a))

    # Termination.
    p = np.clip((alpha / (mu * np.log(2))) - (1 / a), min=0, max=None)
    return p
