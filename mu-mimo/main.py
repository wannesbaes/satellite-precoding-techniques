
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def simulate_mu_mimo(M, K, Nr, Nt, Ns, nu, SNR_dB):


    def generate_compound_channel_matrix(K, Nr, Nt):
        H = (1/np.sqrt(2)) * (np.random.randn(K*Nr, Nt) + 1j * np.random.randn(K*Nr, Nt))
        return H

    def compute_compound_combining_matrix(H, K, Nr, Ns):
        W = np.zeros((K*Ns, K*Nr), dtype=complex)
        for k in range(K):
            H_k = H[k*Nr : (k+1)*Nr]
            U_k, Sigma_k, Vh_k = np.linalg.svd(H_k)
            W_k = U_k.conj().T[ : Ns]
            W[k*Ns : (k+1)*Ns, k*Nr : (k+1)*Nr] = W_k
        return W

    def compute_compound_precoding_matrix(H, W):
        H_BS = W @ H
        F = H_BS.conj().T @ np.linalg.inv(H_BS @ H_BS.conj().T)
        return F

    def compute_compound_power_allocation_matrix(K, Ns, nu, H, W, N0):
        P = np.eye(K*Ns) / (K*Ns)
        return P


    def generate_noise_vector(K, Nr, SNR_dB, M):
        N0 = 10 ** (-SNR_dB / 10)
        n = np.sqrt(N0/2) * (np.random.randn(K*Nr, M) + 1j * np.random.randn(K*Nr, M))
        return n

    def genereate_data_symbols(M, K, Ns):
        b = np.random.randint(0, 2, size=(K*Ns, M))
        a = 1 - 2 * b
        return a

    def detect_data_symbols(u):
        a_hat = (np.where(u.real < 0, -1, 1))
        return a_hat


    H = generate_compound_channel_matrix(K, Nr, Nt)
    W = compute_compound_combining_matrix(H, K, Nr, Ns)
    F = compute_compound_precoding_matrix(H, W)
    P = compute_compound_power_allocation_matrix(K, Ns, nu, H, W, SNR_dB)

    n = generate_noise_vector(K, Nr, SNR_dB, M)
    a = genereate_data_symbols(M, K, Ns)
    x = F @ P @ a
    y = H @ x + n
    z = W @ y
    # u = (1/P) @ z
    a_hat = detect_data_symbols(z)

    ber = np.mean(a != a_hat)
    return ber

def simulate_su_mimo(M, Nr, Nt, nu, SNR_dB):

    def generate_channel_matrix(Nr, Nt):
        H = (1/np.sqrt(2)) * (np.random.randn(Nr, Nt) + 1j * np.random.randn(Nr, Nt))
        return H
    
    def compute_combining_matrix(H):
        U, Sigma, Vh = np.linalg.svd(H)
        W = U.conj().T
        return W
    
    def compute_precoding_matrix(H):
        U, Sigma, Vh = np.linalg.svd(H)
        F = Vh.conj().T
        return F
    
    def compute_power_allocation_matrix(Nr, nu, H, W, N0):
        P = np.eye(Nr) / Nr
        return P
    

    def generate_noise_vector(Nr, SNR_dB, M):
        N0 = 10 ** (-SNR_dB / 10)
        n = np.sqrt(N0/2) * (np.random.randn(Nr, M) + 1j * np.random.randn(Nr, M))
        return n
    
    def generate_data_symbols(M, Nr):
        b = np.random.randint(0, 2, size=(Nr, M))
        a = 1 - 2 * b
        return a
    
    def detect_data_symbols(u):
        a_hat = (np.where(u.real < 0, -1, 1))
        return a_hat


    H = generate_channel_matrix(Nr, Nt)
    W = compute_combining_matrix(H)
    F = compute_precoding_matrix(H)
    P = compute_power_allocation_matrix(Nr, nu, H, W, SNR_dB)
    
    n = generate_noise_vector(Nr, SNR_dB, M)
    a = generate_data_symbols(M, Nr)
    x = F[:, :Nr] @ P @ a
    y = H @ x + n
    z = W @ y
    #u = (1/P) @ z
    a_hat = detect_data_symbols(z)

    ber = np.mean(a != a_hat)
    return ber

def su_theory_ber(snr_dB, Nr, Nt):

    def Q(x):
        return 0.5 * sp.special.erfc(x / np.sqrt(2))

    H = (1/np.sqrt(2)) * (np.random.randn(Nr, Nt) + 1j * np.random.randn(Nr, Nt))
    N0 = 10 ** (-snr_dB / 10)
    P = np.full(Nr, 1/Nr)
    _, sigma, _ = np.linalg.svd(H)

    ber = np.mean(Q( np.sqrt(P*(sigma**2) / (2*N0)) ))
    return ber

def plot_ber_vs_snr(snrs_dB, bers_mu, bers_su_8x8, bers_su_2x2, bers_su_8x2):
    
    plt.semilogy(snrs_dB, bers_mu, c='tab:blue', marker='o', label='MU (8x2 BPSK, K=4)')
    plt.semilogy(snrs_dB, bers_su_8x8, c='darkgreen', marker='o', label='SU (8x8 BPSK)')
    plt.semilogy(snrs_dB, bers_su_2x2, c='tab:green', marker='o', label='SU (2x2 BPSK)')
    plt.semilogy(snrs_dB, bers_su_8x2, c='tab:olive', marker='o', label='SU (8x2 BPSK)')

    plt.xlabel('SNR (dB)')
    plt.ylabel('BER')
    plt.title('Performance\nMU-MIMO vs SU-MIMO')
    plt.grid(True, which='both', linestyle='--')
    plt.legend()
    plt.ylim(1e-5, 1)
    plt.savefig('ber_mu_vs_su_mimo.png')
    plt.show()


if __name__ == "__main__":

    K = 4     # Number of UTs
    Nt = 8    # Number of transmit antennas at BS
    Nr = 2    # Number of receive antennas per UT
    Ns = 2    # Number of data streams per UT
    
    snrs_dB = np.arange(-5, 30, 2.5)
    
    bers_mu = []
    for SNR_dB in snrs_dB:
        ber = np.mean( [simulate_mu_mimo(1000, K, Nr, Nt, Ns, nu=1, SNR_dB=SNR_dB) for _ in range(1000)] )
        bers_mu.append(ber)
    print("MU-MIMO simulation done.")
    
    bers_su_8x8 = []
    for SNR_dB in snrs_dB:
        ber = np.mean( [simulate_su_mimo(1000, Nt, Nt, nu=1, SNR_dB=SNR_dB) for _ in range(1000)] )
        bers_su_8x8.append(ber)
    print("SU-MIMO 8x8 simulation done.")
    
    bers_su_2x2 = []
    for SNR_dB in snrs_dB:
        ber = np.mean( [simulate_su_mimo(1000, Nr, Nr, nu=1, SNR_dB=SNR_dB) for _ in range(1000)] )
        bers_su_2x2.append(ber)
    print("SU-MIMO 2x2 simulation done.")

    bers_su_8x2 = []
    for SNR_dB in snrs_dB:
        ber = np.mean( [simulate_su_mimo(1000, Nr, Nt, nu=1, SNR_dB=SNR_dB) for _ in range(1000)] )
        bers_su_8x2.append(ber)
    print("SU-MIMO 8x2 simulation done.")
    
    # bers_su_8x8_theory = []
    # for SNR_dB in snrs_dB:
    #     ber = np.mean( [su_theory_ber(SNR_dB, Nt, Nt) for _ in range(1000)] )
    #     bers_su_8x8_theory.append(ber)
    # print("SU-MIMO 8x8 theory done.")
    
    # bers_su_2x2_theory = []
    # for SNR_dB in snrs_dB:
    #     ber = np.mean( [su_theory_ber(SNR_dB, Nr, Nr) for _ in range(1000)] )
    #     bers_su_2x2_theory.append(ber)
    # print("SU-MIMO 2x2 theory done.")

    plot_ber_vs_snr(snrs_dB, bers_mu, bers_su_8x8, bers_su_2x2, bers_su_8x2)
    

    
    
