import numpy as np
import matplotlib.pyplot as plt
import os

condition_numbers = {
    100: 13507.582092897444,
    1000: 1351031.1299646497,
    10000: 135103385.93200016,
}


def theoretical_bound(k, kappa, initial_error=1.0):
    factor = (np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)
    return 2 * (factor**k) * initial_error


fig, axes = plt.subplots(1, 3, figsize=(15, 5))
N_values = [100, 1000, 10000]
for i, N in enumerate(N_values):
    kappa = condition_numbers[N]
    residuals_file = f"residuals-{N}.dat"
    data = np.loadtxt(residuals_file)
    iterations = data[:, 0].astype(int)
    residuals = data[:, 1]
    axes[i].semilogy(iterations, residuals, "b-", label="Actual Residuals")

    # Calculate and plot theoretical bound
    initial_error = residuals[0]
    bound_iterations = np.arange(max(iterations) + 1)
    bounds = [theoretical_bound(k, kappa, initial_error) for k in bound_iterations]
    axes[i].semilogy(bound_iterations, bounds, "r--", label="Theoretical Bound")

    # Set labels and title
    axes[i].set_title(rf"$N={N},\kappa={kappa:.2e}$")
    axes[i].set_xlabel("Iteration")
    axes[i].set_ylabel("Residual")
    axes[i].legend()
    axes[i].grid(True)

plt.tight_layout()
plt.savefig("convergence.png", dpi=300)
