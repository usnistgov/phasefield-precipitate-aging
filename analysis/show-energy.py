# -*- coding: utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print("Usage: {0} filename [timestep]".format(sys.argv[0]))
else:
    filename = sys.argv[1]
    timestep = 5e-7

    if len(sys.argv) > 2:
        timestep = float(sys.argv[2])

    delta, lam, f = np.genfromtxt(
        filename, skip_header=1, usecols=(4, 5, 6), unpack=True, dtype=float
    )

    t = np.arange(0, len(delta))

    plt.subplot(121)
    plt.title("Phase Fractions")
    plt.xlabel("$t$")
    plt.plot(t, delta, label="$\delta$")
    plt.plot(t, lam, label="$\lambda$")
    plt.legend(loc="best")

    plt.subplot(122)
    plt.title("Free Energy")
    plt.xlabel("$t$")
    plt.plot(t, f)

    plt.show()
    plt.close()
