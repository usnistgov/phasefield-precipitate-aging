# -*- coding: utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print("Usage: {0} filename [timestep]".format(sys.argv[0]))
else:
    filename = sys.argv[1]
    timestep = 3.125e-8

    if len(sys.argv) > 2:
        timestep = float(sys.argv[2])

    delta, lam, f, w = np.genfromtxt(
        filename, skip_header=1, skip_footer=0, usecols=(4, 5, 6, 7), unpack=True, dtype=float
    )

    t = np.arange(0, len(delta))

    ax1 = plt.subplot(311)
    plt.xlabel("$t$", fontsize=20)
    plt.ylabel("$\phi$", rotation = 0, fontsize=20, labelpad=20)
    plt.plot(t, delta, label="$\delta$")
    plt.plot(t, lam, label="$\lambda$")
    plt.legend(loc="best")

    plt.subplot(312, sharex = ax1)
    plt.xlabel("$t$", fontsize=20)
    plt.ylabel("$\mathcal{F}$", rotation = 0, fontsize=20, labelpad=20)
    plt.plot(t, f, "-k")

    plt.subplot(313, sharex = ax1)
    plt.xlabel("$t$", fontsize=20)
    plt.ylabel("$2\lambda$", rotation = 0, fontsize=20, labelpad=20)
    plt.plot(t, w, "-k")
    plt.plot((t[0], t[-1]), (7.5e-10, 7.5e-10), ":k")

    plt.show()
    plt.close()
