import matplotlib.pylab as plt
import numpy as np

# === Interfacial Energy ===

s, d, l = np.loadtxt("sigma.csv", delimiter=',', skiprows=1, unpack=True)

plt.semilogy(s, d, label=r"$\delta$")
plt.semilogy(s, l, label=r"$\lambda$")

# 1e-7 is "about right", per AMJ
plt.semilogy((0,1), (1e-7,1e-7), ":k")

plt.title("Nucleation Probability (Enriched Material)")

plt.xlim([0., 0.25])
plt.xlabel(r"$\sigma$")

plt.ylim([5e-9, 0.05])
plt.ylabel(r"$\mathcal{P}$")

plt.legend(loc="best")
plt.savefig("sigma.png", dpi=400, bbox_inches="tight")
plt.close()

"""
# === Composition ===

xCr, xNb, dGdel, Pdel, dGlav, Plav = np.loadtxt("composition.csv", delimiter=',', skiprows=1, unpack=True)

plt.plot(xNb, dGdel, label=r"$\delta$")
plt.plot(xNb, dGlav, label=r"$\lambda$")
plt.plot((0,1), (0, 0), ":k")

plt.title("Driving Force (Enriched Material)")
plt.xlabel(r"$\chi_{\mathrm{Nb}}$")
plt.ylabel(r"$\Delta G_{\mathrm{nuc}}$")

plt.legend(loc="best")
plt.savefig("composition.png", dpi=400, bbox_inches="tight")
plt.close()
"""
