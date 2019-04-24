import matplotlib.pylab as plt
import numpy as np
s, d, l = np.loadtxt("sigma.csv", delimiter=',', skiprows=1, unpack=True)

plt.semilogy(s, d, label=r"$\delta$");
plt.semilogy(s, l, label=r"$\lambda$");
plt.xlabel(r"$\sigma$")
plt.ylabel(r"$\mathcal{P}$")
plt.legend(loc="best")
plt.savefig("sigma.png", dpi=400, bbox_inches="tight")
