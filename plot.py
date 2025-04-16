import matplotlib.pyplot as plt
import numpy as np

data = np.genfromtxt("solution.dat")
N = int(np.sqrt(data.shape[0]))
u = data[:, 2].reshape(N, N)
x = data[:, 0].reshape(N, N)
y = data[:, 1].reshape(N, N)
extent = [x.min(), x.max(), y.min(), y.max()]
plt.figure(figsize=(8, 6))
heatmap = plt.imshow(u, extent=extent, origin="lower", cmap="viridis", aspect="auto")
plt.title(rf"Heatmap of $u(x,y)$ for N = {N}")
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.colorbar(heatmap, label=r"$u(x,y)$")
plt.tight_layout()
plt.savefig("solution.png")
