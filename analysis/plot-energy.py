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

    t, delta, lam, f = np.genfromtxt(
        datname, skip_header=1, skip_footer=0, usecols=(0, 4, 5, 6), unpack=True, dtype=float
    )

    fig, ax = plt.subplots(2, 1, sharex=True, sharey=False)

    ax[0].set_title(imgtitl, fontsize=20)
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
