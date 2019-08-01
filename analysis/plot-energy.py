# -*- coding: utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print("Usage: {0} filename [timestep]".format(sys.argv[0]))
else:
    filename = sys.argv[1]
    imgname = filename.replace("log", "png")
    if imgname == filename:
        imgname = filename + ".png"
    timestep = 3.125e-8

    if len(sys.argv) > 2:
        timestep = float(sys.argv[2])

    delta, lam, f = np.genfromtxt(
        filename, skip_header=1, skip_footer=0, usecols=(4, 5, 6), unpack=True, dtype=float
    )

    t = np.arange(0, len(delta))

    fig, ax = plt.subplots(2, 1, sharex=True)

    ax[0].set_xlabel("$t$", fontsize=20)
    ax[0].set_ylabel("$\phi$", rotation = 0, fontsize=20, labelpad=20)
    ax[0].plot(t, delta, label="$\delta$")
    ax[0].plot(t, lam, label="$\lambda$")
    ax[0].legend(loc="best")

    ax[1].set_xlabel("$t$", fontsize=20)
    ax[1].set_ylabel("$\mathcal{F}$", rotation = 0, fontsize=20, labelpad=20)
    ax[1].plot(t, f, "-k")

    plt.savefig(imgname, dpi=400, bbox_inches="tight")
    plt.close()
