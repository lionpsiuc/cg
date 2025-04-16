import matplotlib.pyplot as plt
import numpy as np

grid_sizes = [8, 16, 32, 64, 128, 256]
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))
axes = axes.flatten()

# Loop through each grid size and plot the corresponding solution
for i, N in enumerate(grid_sizes):
    filename = f"analytical-{N}.dat"
    data = np.genfromtxt(filename)
    u = data[:, 2].reshape((N, N))
    x = data[:, 0].reshape((N, N))
    y = data[:, 1].reshape((N, N))

    # Determine plot extent based on the physical coordinates
    extent = [x.min(), x.max(), y.min(), y.max()]

    # Plot the heatmap
    im = axes[i].imshow(u, extent=extent, origin="lower", cmap="viridis", aspect="auto")
    axes[i].set_title(rf"Heatmap, $N={N}$")
    axes[i].set_xlabel(r"$x$")
    axes[i].set_ylabel(r"$y$")
    cbar = fig.colorbar(
        im, ax=axes[i], orientation="vertical", fraction=0.046, pad=0.04
    )
    cbar.set_label(r"$u(x,y)$")

plt.tight_layout()
plt.savefig("analytical.png")
