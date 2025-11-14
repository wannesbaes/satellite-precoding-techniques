import numpy as np
import matplotlib.pyplot as plt

def plot_singular_value_pdfs(Nt, Nr, n_samples=1_000_000, bins=50):

    def calc_singular_values(Nt, Nr):
        H = (np.random.randn(Nr, Nt) + 1j * np.random.randn(Nr, Nt)) / np.sqrt(2)
        _, S, _ = np.linalg.svd(H)
        return S
    
    Si = np.array([calc_singular_values(Nt, Nr) for _ in range(n_samples)])
    
    fig, ax = plt.subplots(figsize=(7, 5))
    for i in range(Si.shape[1]):
        ax.hist(Si[:, i], bins=bins, density=True, alpha=0.6, label=r"$\sigma_{" + f"{i+1}" + r"}$")
    ax.set_xlabel("Singular value")
    ax.set_ylabel("Probability density")
    ax.set_title(f"PDFs of singular values for {Nr}x{Nt} complex Gaussian matrix")
    ax.legend()
    ax.grid(True)

    return fig, ax

result = []
configs = [(4, 8), (8, 4)]
for Nt, Nr in configs:
    result.extend(plot_singular_value_pdfs(Nt=Nt, Nr=Nr))
    print(f"Calculations for {Nr}x{Nt} matrix done.")
plt.show()