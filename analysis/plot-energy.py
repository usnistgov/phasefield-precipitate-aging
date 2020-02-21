# -*- coding: utf-8 -*-

from matplotlib import use as mplBackEnd
mplBackEnd("Agg")

import sys
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) < 3:
    print("Usage: {0} input output".format(sys.argv[0]))
else:
    datname = sys.argv[1]
    imgname = sys.argv[2]
    imgtitl = imgname.replace(".png", "")

    delta, lam, f = np.genfromtxt(
        datname, skip_header=1, skip_footer=0, usecols=(4, 5, 6), unpack=True, dtype=float
    )

    dt = 0.1
    t = dt * np.arange(0, len(delta))

    fig, ax = plt.subplots(2, 1, sharex=True)

    ax[0].set_title(imgtitl, fontsize=20)
    ax[0].set_xlim([0, 600])
    ax[0].set_ylim([0, 0.75])
    ax[0].set_xlabel("$t$", fontsize=20)
    ax[0].set_ylabel("$\phi$", rotation = 0, fontsize=20, labelpad=20)
    ax[0].plot(t, delta, label="$\delta$")
    ax[0].plot(t, lam, label="$\lambda$")
    ax[0].legend(loc="best")

    ax[1].set_xlabel("$t$", fontsize=20)
    ax[1].set_ylabel("$\mathcal{F}$", rotation = 0, fontsize=20, labelpad=20)
    ax[1].set_ylim([1e-7, 2.5e-6])
    ax[1].semilogy(t, f, "-k")

    plt.savefig(imgname, dpi=400, bbox_inches="tight")
    plt.close()
