# -*- coding: utf-8 -*-

# Overlay phase-field simulation compositions on ternary phase diagram
# Before executing this script, run the mmsp2comp utility
# for each checkpoint file in the directories of interest.
# Usage: python analysis/pathways173.py

import re
import numpy as np
from math import floor, sqrt
from os import path
from sys import argv
import glob
import matplotlib.pylab as plt

import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from CALPHAD_energies import *

density = 500
skipsz = 9

labels = [r"$\gamma$", r"$\delta$", "Laves"]
colors = ["red", "green", "blue"]

for datdir in glob.glob(
    "/usr/local/tnk10/data/phase-field/alloy625/TKR4p173/run*"
):  # {0}".format(j) for j in (2,3,6,12,21,63)]:
    base = path.basename(datdir)
    if (
        path.isdir(datdir)
        and len(glob.glob("{0}/../../coord/TKR4p173/{1}/*.xy".format(datdir, base))) > 0
        and len(
            glob.glob("{0}/../../coord/TKR4p173/diffusion_{1}.xc".format(datdir, base))
        )
        > 0
    ):
        # Plot phase diagram
        plt.figure(0, figsize=(10, 7.5))  # inches
        plt.plot(XS, YS, "-k")
        plt.plot(X0, Y0, "-k", zorder=1)
        plt.title("Cr-Nb-Ni at %.0f K" % temp, fontsize=18)
        plt.xlabel(r"$x_\mathrm{Nb}$", fontsize=18)
        plt.ylabel(r"$x_\mathrm{Cr}$", fontsize=18)
        plt.xticks(np.linspace(0, 1, 21))
        plt.scatter(Xtick, Ytick, color="black", s=3)
        gann = plt.text(simX(0.010, 0.495), simY(0.495), r"$\gamma$", fontsize=14)
        dann = plt.text(simX(0.230, 0.010), simY(0.010), r"$\delta$", fontsize=14)
        lann = plt.text(simX(0.340, 0.275), simY(0.275), r"L", fontsize=14)

        # Add composition pathways
        fnames = sorted(
            glob.glob("{0}/../../coord/TKR4p173/{1}/*.xy".format(datdir, base))
        )
        for file in fnames[::10]:
            try:
                x, xcr, xnb, P = np.loadtxt(file, delimiter=",", unpack=True)
                num = int(re.search("[0-9]{5,16}", file).group(0)) * 7.5e-5
                plt.plot(
                    simX(xnb, xcr),
                    simY(xcr),
                    "-",
                    linewidth=1,
                    zorder=1,
                    color="gray",
                    label=r"%.0f s" % num,
                )
                # plt.plot(simX(xnb, xcr), simY(xcr), linewidth=1, zorder=1)
            except:
                print("Empty file: ", file)

        try:
            dgcr, dgnb, dcr, dnb, lgcr, lgnb, lcr, lnb = np.loadtxt(
                "{0}/../../coord/TKR4p173/diffusion_{1}.xc".format(datdir, base),
                delimiter=",",
                skiprows=1,
                usecols=(3, 4, 5, 6, 11, 12, 13, 14),
                unpack=True,
            )
            plt.plot(
                simX(dgnb, dgcr), simY(dgcr), c="blue"
            )  # , label=r'$\gamma(\delta)$'
            plt.plot(
                simX(lgnb, lgcr), simY(lgcr), c="green"
            )  # , label=r'$\gamma($L$)$'
            plt.plot(simX(dnb, dcr), simY(dcr), c="coral")  # , label=r'$\delta$'
            plt.plot(simX(lnb, lcr), simY(lcr), c="magenta")  # , label=r'L'
        except:
            print(
                "Empty file: {0}/../../coord/TKR4p173/diffusion_{1}.xc".format(
                    datdir, base
                )
            )

        plt.xlim([0, 0.6])
        plt.ylim([0, rt3by2 * 0.6])
        plt.legend(loc="best")
        plt.savefig(
            "diagrams/TKR4p173/pathways/pathways_{0}.png".format(base),
            dpi=400,
            bbox_inches="tight",
        )

        dann.remove()
        lann.remove()
        plt.xticks([])
        plt.yticks([])

        plt.xlim([0.175, 0.425])
        plt.ylim([0.275, 0.275 + rt3by2 * 0.25])
        plt.savefig(
            "diagrams/TKR4p173/pathways/pathways_zm_gam_{0}.png".format(base),
            dpi=400,
            bbox_inches="tight",
        )

        gann.remove()
        dann = plt.text(simX(0.2375, 0.010), simY(0.010), r"$\delta$", fontsize=14)

        plt.xlim([0.2375, 0.3])
        plt.ylim([0.0, rt3by2 * 0.05125])
        plt.savefig(
            "diagrams/TKR4p173/pathways/pathways_zm_del_{0}.png".format(base),
            dpi=400,
            bbox_inches="tight",
        )

        dann.remove()
        lann = plt.text(simX(0.345, 0.3), simY(0.3), r"L", fontsize=14)

        plt.xlim([0.45, 0.55])
        plt.ylim([0.25, 0.25 + rt3by2 * 0.1])
        plt.savefig(
            "diagrams/TKR4p173/pathways/pathways_zm_lav_{0}.png".format(base),
            dpi=400,
            bbox_inches="tight",
        )

        plt.close()

        # Plot phase diagram
        plt.figure(1, figsize=(10, 7.5))  # inches
        plt.plot(XS, YS, "-k")
        plt.plot(X0, Y0, "-k", zorder=1)
        plt.axis("off")
        gann = plt.text(simX(0.010, 0.495), simY(0.495), r"$\gamma$", fontsize=14)
        dann = plt.text(simX(0.230, 0.010), simY(0.010), r"$\delta$", fontsize=14)
        lann = plt.text(simX(0.340, 0.275), simY(0.275), r"L", fontsize=14)
        xCr0, xNb0 = np.genfromtxt(
            "{0}/c.log".format(datdir),
            usecols=(2, 3),
            delimiter="\t",
            skip_header=1,
            unpack=True,
        )
        plt.scatter(simX(xNb0[-1], xCr0[-1]), simY(xCr0[-1]), zorder=1, color="black")
        # geometric bisector is meaningless, since the triangle is not equilateral
        # xB, yB = draw_bisector(0.5, 0.5)
        # plt.plot(xB, yB, ':k', zorder=1)
        plt.xlim([0, 0.6])
        plt.ylim([0, rt3by2 * 0.6])
        plt.savefig(
            "diagrams/TKR4p173/triangles/composition_{0}.png".format(base),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
    else:
        print("Skipping {0}".format(datdir))
