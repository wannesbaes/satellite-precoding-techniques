# mu-mimo/mu_mimo/processing/precoding.py

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import scipy as sp
from ..types import ComplexArray, RealArray, IntArray, BitArray, ChannelStateInformation


class Precoder(ABC):
    """
    The Precoder Abstract Base Class (ABC).

    A precoder class is responsible for implementing a precoding strategy and effectively precoding the transmitted signals in the base station.\\
    In case of coordinated beamforming, the precoder is responsible for computing the combining matrices for each UT as well! These are then later sent to the UTs.

    In addition, the precoder is responsible for computing the equalization coefficients for each data stream. These will be used by the UTs to correctly rescale the received symbols before the decoding process.
    """

    @staticmethod
    @abstractmethod
    def compute(csi: ChannelStateInformation, Pt: float, K: int,) -> tuple[ComplexArray, ComplexArray | None, ComplexArray]:
        """
        Compute the compound precoding matrix.\\
        In case of coordinated beamforming, the combining matrices for each UT are computed as well.

        In addition, the equalization coefficients for each data stream are computed as well. These will be used by the UTs to correctly rescale the received symbols before the decoding process.
        
        Parameters
        ----------
        csi : ChannelStateInformation
            The channel state information (CSI) of the system.
        Pt : float
            The total transmit power available at the BS.
        K : int
            The number of user terminals (UTs) in the system.
        
        Returns
        -------
        F : ComplexArray, shape (Nt, K*Nr)
            The compound precoding matrix for all UTs. It contains the power allocation across the data streams as well.
        G : ComplexArray, shape (K*Nr, K*Nr) or None
            The compound combining matrix (block diagonal) for all UTs in case of coordinated beamforming. None otherwise.
        C_eq : ComplexArray, shape (K*Nr,)
            The equalization coefficients for all UTs.
        """
        raise NotImplementedError

    @staticmethod
    def apply(a: ComplexArray, F: ComplexArray, ibr: IntArray) -> ComplexArray:
        """
        Apply the precoding matrix to the data symbols.

        Parameters
        ----------
        a : ComplexArray, shape (Ns_total, Msv)
            The data symbol streams for all UTs.
        F : ComplexArray, shape (Nt, K*Nr)
            The compound precoding matrix for all UTs.
        ibr : IntArray, shape (K*Nr,)
            The number of bits per symbol for each data stream (active and inactve).

        Returns
        -------
        x : ComplexArray, shape (Nt, Msv)
            The precoded signal to be transmitted by the BS.
        """

        x = F[:, ibr > 0] @ a
        return x

class NeutralPrecoder(Precoder):
    """
    Neutral Precoder.

    This precoder acts as a 'neutral element' for precoding.\\
    It does not perform any precoding and simply passes the data symbols through without any modification.

    In addition, the power allocation is uniform across the data streams (so no real power allocation is performed either).

    Finally, in case of coordinated beamforming, the combining matrices for each UT are set to the identity matrix (so no real combining is performed either).
    """

    @staticmethod
    def compute(csi: ChannelStateInformation, Pt: float, K: int) -> tuple[ComplexArray, ComplexArray | None, ComplexArray]:

        Nt = csi.H_eff.shape[1]
        Nr = csi.H_eff.shape[0] // K

        F = np.eye(Nt, K*Nr)
        G = np.eye(K*Nr)
        C_eq = np.ones(K*Nr)
        return F, G, C_eq

class SVDPrecoder(Precoder):
    """
    Singular Value Decomposition (SVD) Precoder.

    The optimal precoding strategy for a single-user MIMO system.\\
    This precoding strategy is only available for single-user systems, since it requires combining all data streams.
    """

    @staticmethod
    def compute(csi: ChannelStateInformation, Pt: float, K: int) -> tuple[ComplexArray, ComplexArray | None, ComplexArray]:

        # Validate that the SVD precoding strategy is only applied in single-user systems.
        if K > 1:
            raise ValueError("The SVD precoding strategy is only available for single-user systems. Please choose a different precoding strategy for multi-user systems.")
        
        # Execute the SVD of the effective channel matrix.
        U, Sigma, Vh = np.linalg.svd(csi.H_eff, full_matrices=False)

        # Determine the precoder and combiner matrices.
        F = Vh.conj().T
        G = U.conj().T

        # Determine the optimal power allocation across all the data streams.
        gamma = (Sigma**2) * (csi.snr / Pt)
        P = waterfilling_v1(gamma=gamma, pt=Pt)
        FP = F @ np.diag(np.sqrt(P))

        # Compute the equalization coefficients.
        C_eq = np.diag( G @ csi.H_eff @ FP )

        return FP, G, C_eq

class ZFPrecoder(Precoder):
    """
    Zero-Forcing (ZF) Precoder.

    The precoder aims to completely eliminate all interference at the user terminals.\\
    The precoding matrix is therefore computed as the pseudo-inverse of the effective channel matrix H_eff.

    In addition, the power allocation across the data streams is optimal in the sense that it maximizes the sum rate of the system under the total power constraint Pt.\\
    """

    @staticmethod
    def compute(csi: ChannelStateInformation, Pt: float, K: int) -> tuple[ComplexArray, ComplexArray | None, ComplexArray]:

        # The combining matrix is not computed in the ZF precoding strategy.
        G = None

        # The precoding matrix is computed as the pseudo-inverse of the effective channel matrix.
        F = np.linalg.pinv(csi.H_eff)
        F_norm = F / np.linalg.norm(F, axis=0)

        # The power allocation across the data streams is computed using the waterfilling algorithm to maximize the sum rate under the total power constraint Pt.
        gamma = (1 / np.linalg.norm(F, axis=0)) * (csi.snr / Pt)
        P = waterfilling_v1(gamma=gamma, pt=Pt)
        FP = F_norm @ np.diag(np.sqrt(P))

        # Compute the equalization coefficients.
        C_eq = np.diag( csi.H_eff @ FP )

        return FP, G, C_eq
    
class BDPrecoder(Precoder):
    """
    Block Diagonalization (BD) Precoder.

    Firstly, the multi-user precoder aims to completely eliminate all inter-user interference at the user terminals. This is achieved by choosing the multi-user precoding matrix of each UT such that it spans the null space of the interfering channel matrix of that UT.\\
    
    Secondly, the single-user precoder for each UT is computed as the SVD precoder of the effective channel matrix of that UT after applying the multi-user precoding matrix. Analogue to SVD precoding in a SU-MIMO system. Same for the single-user combiner.
    """

    @staticmethod
    def compute(csi: ChannelStateInformation, Pt: float, K: int) -> tuple[ComplexArray, ComplexArray | None, ComplexArray]:
        
        # Parameter initialization.
        H = csi.H_eff
        Nr = H.shape[0] // K
        Nt = H.shape[1]

        # STEP 1: the multi-user precoding matrix.
        r_ring = Nt - (K-1)*Nr
        F1 = np.empty((Nt, K*r_ring), dtype=complex)
        for k in range(K):
            H_ring_k = np.delete(H, slice(k*Nr, (k+1)*Nr), axis=0)
            _, _, Vh_ring_k = sp.linalg.svd(H_ring_k)
            V_ring_k = Vh_ring_k.conj().T
            F1_k = V_ring_k[:, Nt-r_ring : Nt]
            F1[:, k*r_ring : (k+1)*r_ring] = F1_k
        
        # STEP 2: the single-user precoding matrix.
        F2 = np.zeros((K*r_ring, K*Nr), dtype=complex)
        G = np.zeros((K*Nr, K*Nr), dtype=complex)
        for k in range(K):
            H_k = H[k*Nr : (k+1)*Nr, :]
            F1_k = F1[:, k*r_ring : (k+1)*r_ring]
            U_k, Sigma_k, Vh_k = np.linalg.svd(H_k @ F1_k, full_matrices=False)
            F2_k = Vh_k.conj().T[:, :Nr]
            G_k = U_k.conj().T
            F2[k*r_ring : (k+1)*r_ring, k*Nr : (k+1)*Nr] = F2_k
            G[k*Nr : (k+1)*Nr, k*Nr : (k+1)*Nr] = G_k
        
        # STEP 3: the power allocation.
        F = F1 @ F2
        T = G @ H @ F
        gamma = np.abs(np.diagonal(T))**2 * (csi.snr / Pt)
        P = waterfilling_v1(gamma=gamma, pt=Pt)
        FP = F @ np.diag(np.sqrt(P))

        # Compute the equalization coefficients.
        C_eq = np.diag( G @ H @ FP )

        return FP, G, C_eq

class WMMSEPrecoder(Precoder):
    """
    Weighted Minimum Mean Squared Error (WMMSE) Precoder.

    The goal of this precoding technique is to maximize the weighted sum-rate.
    """
    
    @staticmethod
    def compute(csi: ChannelStateInformation, Pt: float, K: int) -> tuple[ComplexArray, ComplexArray | None, ComplexArray]:

        H = csi.H_eff
        snr = csi.snr
        R = 0

        for mode in ['MF', 'random', 'random', 'random']:

            # Initialization.
            MAX_ITER = 10
            nu = (1/K) * np.ones(K)
            F_iter = WMMSEPrecoder._init_precoders(H, snr, Pt, K, mode=mode)

            # Iteration.
            for _ in range(MAX_ITER):
                G_iter = WMMSEPrecoder._update_combiners_fast(H, snr, Pt, K, F_iter)
                W_iter = WMMSEPrecoder._update_MSE_weights_fast(H, snr, Pt, K, F_iter, nu)
                F_iter = WMMSEPrecoder._update_precoders(H, snr, Pt, K, G_iter, W_iter)

            # Termination.
            R_new = WMMSEPrecoder.__compute_achievable_rate_fast(K, W_iter, nu)
            if R_new > R:
                R = R_new
                F = F_iter
                G = G_iter
                C_eq = np.diag( G @ H @ F )

        return F, G, C_eq
    
    
    @staticmethod
    def _init_precoders(H: ComplexArray, snr: float, Pt: float, K: int, mode: str = 'MF') -> ComplexArray:
        r"""
        Initialize the precoding matrices for each UT.

        Depending on selected initialization strategy, this can be done in different ways.
        The possible options are:

            - Random initialization (mode = 'random').
            - Matched filter initialization (mode = 'MF').
            - Existing precoding strategy initialization (mode = 'ZF', 'BD').
        
        Parameters
        ----------
        H : ComplexArray, shape (K*Nr, Nt)
            The compound channel matrix.
        snr : float
            The signal-to-noise ratio (SNR).
        Pt : float
            The total transmit power available at the BS.
        K : int
            The total number of user terminals (UTs).
        mode : str, optional
            The initialization strategy for the precoding matrices. The default is 'MF'.
        
        Returns
        -------
        F : ComplexArray, shape (Nt, K*Nr)
            The initialized compound precoding matrix for all UTs.
        """
        
        if mode == 'random':
            Nr = H.shape[0] // K
            Nt = H.shape[1]
            F = np.random.randn(Nt, K*Nr) + 1j * np.random.randn(Nt, K*Nr)
            F = np.sqrt(Pt / (K * Nr)) * (F / np.linalg.norm(F, axis=0))

        elif mode == 'MF':
            F = H.conj().T
            F = np.sqrt(Pt) * (F / np.linalg.norm(F, axis=0))

        elif mode == 'ZF':
            F = ZFPrecoder.compute(csi=ChannelStateInformation(H_eff=H, snr=snr), Pt=Pt, K=K)[0]

        elif mode == 'BD':
            F = BDPrecoder.compute(csi=ChannelStateInformation(H_eff=H, snr=snr), Pt=Pt, K=K)[0]

        else:
            raise ValueError(f"Unknown initialization mode: '{mode}'. Choose between 'random', 'MF', 'ZF' and 'BD'.")

        return F

    @staticmethod
    def _update_combiners(H: ComplexArray, snr: float, Pt: float, K: int, F: ComplexArray) -> ComplexArray:
        r"""
        Compute the MMSE combiner matrices for each UT based on the channel matrix and the precoder matrix of the previous iteration.

        .. math::
            \begin{aligned}
                \mathbf{G}^{\text{MMSE}}_k &= \arg \, \underset{\mathbf{G}_k}{\text{min}} \, \mathbb{E} \left[ \| \mathbf{G}_k^H \mathbf{y}_k - \mathbf{a}_k \|^2 \right] \\
                &= \mathbf{F}_k^H \, \mathbf{H}_k^H \; \left( \mathbf{H}_k \, \mathbf{F}_k \; \mathbf{F}_k^H \, \mathbf{H}_k^H + \mathbf{R}_{\tilde{n}_k, \tilde{n}_k} \right)^{-1}
            \end{aligned}

        .. math::
            \text{with} \quad \mathbf{R}_{\tilde{n}_k, \tilde{n}_k} = N_0 \, \mathbf{I} + \sum_{\substack{k' = 1 \\ k' \neq k}}^{K} \mathbf{H}_k \, \mathbf{F}_{k'} \, \mathbf{F}_{k'}^H \, \mathbf{H}_k^H

        Parameters
        ----------
        H : ComplexArray, shape (K*Nr, Nt)
            The compound channel matrix.
        snr : float
            The signal-to-noise ratio (SNR).
        Pt : float
            The total transmit power.
        K : int
            The total number of user terminals (UTs).
        F : ComplexArray, shape (Nt, K*Nr)
            The compound precoding matrix from the previous iteration.
        
        Returns
        -------
        G : ComplexArray, shape (K*Nr, K*Nr)
            The updated block diagonal compound combining matrix for all UTs.
        """

        # Initialization.
        Nr = H.shape[0] // K
        R_n_tilde = WMMSEPrecoder.__compute_interference_plus_noise_covariance_matrix(H=H, F=F, Pt=Pt, snr=snr, K=K)
        
        G = np.zeros((K*Nr, K*Nr), dtype=complex)
        
        # Iteration.
        for k in range(K):

            H_k = H[k*Nr : (k+1)*Nr, :]
            F_k = F[:, k*Nr : (k+1)*Nr]
            
            G_k = F_k.conj().T @ H_k.conj().T @ ( np.linalg.inv( H_k @ F_k @ F_k.conj().T @ H_k.conj().T + R_n_tilde[k] ) )
            G[k*Nr : (k+1)*Nr, k*Nr : (k+1)*Nr] = G_k

        # Termination.
        return G

    @staticmethod
    def _update_MSE_weights(H: ComplexArray, snr: float, Pt: float, K: int, F: ComplexArray, nu: RealArray) -> ComplexArray:
        r"""
        Compute the MSE-weights for each data stream based on the channel matrix and the precoder matrix of the previous iteration.

        .. math::
            \mathbf{W}_k = \nu_k \, \text{diag}\left(e_{k, 1}^{-1}, \ldots, e_{k, N_r}^{-1}\right)

        .. math::
            \text{with} \quad \mathbf{E}_k = \left( \mathbf{I} + \mathbf{F}_k^H \, \mathbf{H}_k^H \, \mathbf{R}_{\tilde{n}_k, \tilde{n}_k}^{-1} \, \mathbf{H}_k \, \mathbf{F}_k \right)^{-1} \quad \text{and} \quad \mathbf{R}_{\tilde{n}_k, \tilde{n}_k} = N_0 \, \mathbf{I} + \sum_{\substack{k' = 1 \\ k' \neq k}}^{K} \mathbf{H}_k \, \mathbf{F}_{k'} \, \mathbf{F}_{k'}^H \, \mathbf{H}_k^H

        Parameters
        ----------
        H : ComplexArray, shape (K*Nr, Nt)
            The compound channel matrix.
        snr : float
            The signal-to-noise ratio (SNR).
        Pt : float
            The total transmit power.
        K : int
            The total number of user terminals (UTs).
        F : ComplexArray, shape (Nt, K*Nr)
            The compound precoding matrix from the previous iteration.
        nu : RealArray, shape (K,)
            The weights of each UT.
        
        Returns
        -------
        W : ComplexArray, shape (K*Nr, K*Nr)
            The updated block diagonal compound MSE-weight matrix for all UTs.
        """
        
        # Initialization.
        Nr = H.shape[0] // K
        E = WMMSEPrecoder.__compute_MMSE_error_covariance_matrix(H=H, F=F, Pt=Pt, snr=snr, K=K)
        
        W_diag = np.zeros(K*Nr)

        # Iteration.
        for k in range(K):
            nu_k = nu[k]
            E_k = E[k]
            W_diag_k = nu_k * (1 / np.diag(E_k))
            W_diag[k*Nr : (k+1)*Nr] = W_diag_k
        
        # Termination.
        W = np.diag(W_diag)
        return W

    @staticmethod
    def _update_precoders(H: ComplexArray, snr: float, Pt: float, K: int, G: ComplexArray, W: ComplexArray) -> ComplexArray:
        r"""
        Compute the updated precoding matrices for each UT based on the channel matrix and the combiner matrix of the current iteration.

        .. math::
            \mathbf{F} = p \; \mathbf{\tilde{F}}

        .. math::
            \begin{aligned}
                \text{with} \quad \mathbf{\tilde{F}} &= \left( \mathbf{H}^H \, \mathbf{G}^H \, \mathbf{W} \, \mathbf{G} \, \mathbf{H} + \frac{1}{\text{SNR}} \, \text{tr}\left[\mathbf{W} \, \mathbf{G} \, \mathbf{G}^H\right] \, \mathbf{I} \right)^{-1} \cdot \mathbf{H}^H \, \mathbf{G}^H \, \mathbf{W} \\
                \text{and} \quad p &= \sqrt{\frac{P_t}{\left\| \mathbf{\tilde{F}} \right\|_F^2}}
            \end{aligned}
        
        Parameters
        ----------
        H : ComplexArray, shape (K*Nr, Nt)
            The compound channel matrix.
        snr : float
            The signal-to-noise ratio (SNR).
        Pt : float
            The total transmit power.
        G : ComplexArray, shape (K*Nr, K*Nr)
            The block diagonal compound combining matrix for all UTs from the current iteration.
        W : ComplexArray, shape (K*Nr, K*Nr)
            The block diagonal compound MSE-weight matrix for all UTs from the current iteration.

        
        Returns
        -------
        F : ComplexArray, shape (Nt, K*Nr)
            The updated compound precoding matrix for all UTs.
        """

        # Computation.
        F_tilde = np.linalg.inv( H.conj().T @ G.conj().T @ W @ G @ H + (1/snr) * np.trace(W @ G @ G.conj().T) * np.eye(H.shape[1]) ) @ H.conj().T @ G.conj().T @ W
        p = np.sqrt(Pt / np.linalg.norm(F_tilde, 'fro')**2)
        F = p * F_tilde
        
        # Termination.
        return F

    @staticmethod
    def __compute_interference_plus_noise_covariance_matrix(H: ComplexArray, F: ComplexArray, Pt: float, snr: float, K: int) -> ComplexArray:
        r"""
        Compute the interference plus noise covariance matrix :math:`\mathbf{R}_{\tilde{n}_k, \tilde{n}_k}` for all UTs.

        .. math::
            \mathbf{R}_{\tilde{n}_k, \tilde{n}_k} = N_0 \, \mathbf{I} + \sum_{\substack{k' = 1 \\ k' \neq k}}^{K} \mathbf{H}_k \, \mathbf{F}_{k'} \, \mathbf{F}_{k'}^H \, \mathbf{H}_k^H

        Parameters
        ----------
        H : ComplexArray, shape (K*Nr, Nt)
            The compound channel matrix.
        F : ComplexArray, shape (Nt, K*Nr)
            The compound precoding matrix from the previous iteration.
        Pt : float
            The total transmit power.
        snr : float
            The signal-to-noise ratio (SNR).
        K : int
            The number of UTs.
        
        Returns
        -------
        R : ComplexArray, shape (K, Nr, Nr)
            The interference plus noise covariance matrix for all UTs.
        """

        # Initialization.
        Nr = H.shape[0] // K
        N0 = Pt / snr

        R = np.zeros((K, Nr, Nr), dtype=complex)

        # Iteration.
        for k in range(K):
            
            H_k = H[k*Nr:(k+1)*Nr, :]
            F_k = F[:, k*Nr:(k+1)*Nr]

            R_k = N0 * np.eye(Nr, dtype=complex)
            for k_accent in range(K):
                if k_accent != k:
                    F_kp = F[:, k_accent*Nr:(k_accent+1)*Nr]
                    R_k += H_k @ F_kp @ F_kp.conj().T @ H_k.conj().T
            
            R[k] = R_k
        
        # Termination.
        return R

    @staticmethod
    def __compute_MMSE_error_covariance_matrix(H: ComplexArray, F: ComplexArray, Pt: float, snr: float, K: int) -> ComplexArray:
        r"""
        Compute the MMSE error covariance matrix :math:`\mathbf{E}_k` for all UTs.

        .. math::
            \mathbf{E}_k = \left( \mathbf{I} + \mathbf{F}_k^H \, \mathbf{H}_k^H \, \mathbf{R}_{\tilde{n}_k, \tilde{n}_k}^{-1} \, \mathbf{H}_k \, \mathbf{F}_k \right)^{-1}

        Parameters
        ----------
        H : ComplexArray, shape (K*Nr, Nt)
            The compound channel matrix.
        F : ComplexArray, shape (Nt, K*Nr)
            The compound precoding matrix from the previous iteration.
        Pt : float
            The total transmit power.
        snr : float
            The signal-to-noise ratio (SNR).
        K : int
            The number of UTs.
        
        Returns
        -------
        E : ComplexArray, shape (K, Nr, Nr)
            The MMSE error covariance matrix for all UTs.
        """
        
        # Initialization.
        Nr = H.shape[0] // K
        R_n_tilde = WMMSEPrecoder.__compute_interference_plus_noise_covariance_matrix(H=H, F=F, Pt=Pt, snr=snr, K=K)
        
        E = np.zeros((K, Nr, Nr), dtype=complex)

        # Iteration.
        for k in range(K):
            
            H_k = H[k*Nr:(k+1)*Nr, :]
            F_k = F[:, k*Nr:(k+1)*Nr]

            E_k = np.linalg.inv( np.eye(Nr, dtype=complex)  +  F_k.conj().T @ H_k.conj().T @ np.linalg.inv(R_n_tilde[k]) @ H_k @ F_k )
            E[k] = E_k

        # Termination.
        return E

    @staticmethod
    def __compute_achievable_rate(H: ComplexArray, snr: float, Pt: float, K: int, F: ComplexArray, nu: RealArray) -> float:
        """
        Compute the achievable weighted sum-rate of the system for the current precoder matrix.

        .. math::
            R = \sum_{k=1}^{K} \nu_k \, \log_2 \left( \det\left( \mathbf{I} + \mathbf{F}_k^H \, \mathbf{H}_k^H \, \mathbf{R}_{\tilde{n}_k, \tilde{n}_k}^{-1} \, \mathbf{H}_k \, \mathbf{F}_k \right) \right)
        
        Parameters
        ----------
        H : ComplexArray, shape (K*Nr, Nt)
            The compound channel matrix.
        snr : float
            The signal-to-noise ratio (SNR).
        Pt : float
            The total transmit power.
        K : int
            The number of UTs.
        F : ComplexArray, shape (Nt, K*Nr)
            The compound precoding matrix from the previous iteration.
        nu : RealArray, shape (K,)
            The weights of each UT.
        
        Returns
        -------
        R : float
            The achievable weighted sum-rate of the system.
        """

        R = np.zeros(K)
        Nr = H.shape[0] // K
        R_n_tilde = WMMSEPrecoder.__compute_interference_plus_noise_covariance_matrix(H=H, F=F, Pt=Pt, snr=snr, K=K)
        
        for k in range(K):
            
            H_k = H[k*Nr:(k+1)*Nr, :]
            F_k = F[:, k*Nr:(k+1)*Nr]

            E_k_inv = np.eye(Nr) +  F_k.conj().T @ H_k.conj().T @ np.linalg.inv(R_n_tilde[k]) @ H_k @ F_k
            R[k] = np.log2(np.linalg.det(E_k_inv))
        
        R = np.sum(nu * R)
        return R

    @staticmethod
    def __compute_achievable_rate_fast(K: int, W: ComplexArray, nu: RealArray) -> float:
        """
        Fast implementation of the `__compute_achievable_rate` method.

        .. math::
            R = \sum_{k=1}^{K} \nu_k \, \sum_{nr=1}^{N_r} \log_2 \left( e_{k,nr}^{-1} \right)

        Parameters
        ----------
        K : int
            The number of UTs.
        W : ComplexArray, shape (K*Nr, K*Nr)
            The block diagonal compound MSE-weight matrix for all UTs.
        nu : RealArray, shape (K,)
            The weights of each UT.
        
        Returns
        -------
        R : float
            The achievable weighted sum-rate of the system.
        """

        Nr = W.shape[0] // K
        nu_repeated = np.repeat(nu, Nr)
        R = np.sum(nu_repeated * np.log2(np.diag(np.real(W)) / nu_repeated))
        return R


    @staticmethod
    def _update_combiners_fast(H: ComplexArray, snr: float, Pt: float, K: int, F: ComplexArray) -> ComplexArray:
        """
        Fast implementation of the `_update_combiners` method.
        """
        
        # Initialization.
        Nr = H.shape[0] // K
        N0 = Pt / snr

        H_b = H.reshape(K, Nr, -1)
        F_b = F.T.reshape(K, Nr, -1).transpose(0, 2, 1)

        # Computation.
        G_b = (F_b.conj().transpose(0, 2, 1) @ H_b.conj().transpose(0, 2, 1)) @ np.linalg.inv( N0 * np.eye(Nr) + H_b @ (F @ F.conj().T) @ H_b.conj().transpose(0, 2, 1) )
        G = sp.linalg.block_diag(*G_b)

        # Termination.
        return G

    @staticmethod
    def _update_MSE_weights_fast(H: ComplexArray, snr: float, Pt: float, K: int, F: ComplexArray, nu: RealArray) -> ComplexArray:
        """
        Fast implementation of the `_update_MSE_weights` method.
        """
        
        # Initialization.
        Nr = H.shape[0] // K
        N0 = Pt / snr

        H_b = H.reshape(K, Nr, -1)
        F_b = F.T.reshape(K, Nr, -1).transpose(0, 2, 1)

        # Computation.
        HF_b = H_b @ F_b
        R_b = N0*np.eye(Nr) + H_b @ (F @ F.conj().T) @ H_b.conj().transpose(0, 2, 1) - HF_b @ HF_b.conj().transpose(0, 2, 1)
        E_b = np.linalg.inv(np.eye(Nr) +  HF_b.conj().transpose(0, 2, 1) @ np.linalg.inv(R_b) @ HF_b)
        e_diag = E_b[:, np.arange(Nr), np.arange(Nr)]
        W = np.diag((nu[:, np.newaxis] / e_diag).flatten())

        # Termination.
        return W


def waterfilling_v1(gamma: RealArray, pt: float) -> RealArray:
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
        elif pt <= pt_iter(as_iter): as_UB = as_iter

    
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
