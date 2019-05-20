import matplotlib.pylab as plt
import numpy as np
s, d, l = np.loadtxt("sigma.csv", delimiter=',', skiprows=1, unpack=True)

plt.semilogy(s, d, label=r"$\delta$")
plt.semilogy(s, l, label=r"$\lambda$")

# 1e-7 is "about right", per AMJ
plt.semilogy((0,1), (1e-7,1e-7), ":k")

plt.title("Nucleation Probability (Enriched Material)")

plt.xlim([0., 0.2])
plt.xlabel(r"$\sigma$")

plt.ylim([5e-9, 0.05])
plt.ylabel(r"$\mathcal{P}$")

plt.legend(loc="best")
plt.savefig("sigma.png", dpi=400, bbox_inches="tight")
