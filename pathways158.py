# coding: utf-8

# Overlay phase-field simulation compositions on ternary phase diagram
# Before executing this script, run the mmsp2comp utility
# for each checkpoint file in the directories of interest.

# Usage: python pathways158.py

import re
import numpy as np
from math import floor, sqrt
from os import path
from sys import argv
import glob
import matplotlib.pylab as plt

density = 500
skipsz = 9

labels = [r'$\gamma$', r'$\delta$', 'Laves']
colors = ['red', 'green', 'blue']

temp = 870 + 273.15 # 1143 Kelvin
fr1by2 = 1.0 / 2
rt3by2 = np.sqrt(3.0)/2
RT = 8.3144598*temp # J/mol/K

# Coexistence vertices
gamCr = 0.490
gamNb = 0.025
delCr = 0.015
delNb = 0.245
lavCr = 0.300
lavNb = 0.328

def simX(xnb, xcr):
    return xnb + fr1by2 * xcr

def simY(xcr):
    return rt3by2 * xcr

def draw_bisector(A, B):
    bNb = (A * delNb + B * lavNb) / (A + B)
    bCr = (A * delCr + B * lavCr) / (A + B)
    x = [simX(gamNb, gamCr), simX(bNb, bCr)]
    y = [simY(gamCr), simY(bCr)]
    return x, y

# triangle bounding the Gibbs simplex
XS = [0, simX(1, 0), simX(0, 1), 0]
YS = [0, simY(0), simY(1), 0]

# triangle bounding three-phase coexistence
X0 = [simX(gamNb, gamCr), simX(delNb, delCr), simX(lavNb, lavCr), simX(gamNb, gamCr)]
Y0 = [simY(gamCr), simY(delCr), simY(lavCr), simY(gamCr)]

# Tick marks along simplex edges
Xtick = []
Ytick = []
for i in range(20):
    # Cr-Ni edge
    xcr = 0.05*i
    xni = 1.0 - xcr
    Xtick.append(xcr/2 - 0.002)
    Ytick.append(rt3by2*xcr)
    # Cr-Nb edge
    xcr = 0.05*i
    xnb = 1.0 - xcr
    Xtick.append(xnb + xcr/2 + 0.002)
    Ytick.append(rt3by2*xcr)

for datdir in ["data/alloy625/TKR4p158/run{0}".format(j) for j in (11,13,58,64)]:
    if path.isdir(datdir) and len(glob.glob("{0}/*.xy".format(datdir))) > 0:
        base = path.basename(datdir)
        # Plot phase diagram
        plt.figure(0, figsize=(10, 7.5)) # inches
        plt.plot(XS, YS, '-k')
        plt.plot(X0, Y0, '-k', zorder=1)
        plt.title("Cr-Nb-Ni at %.0fK"%temp, fontsize=18)
        plt.xlabel(r'$x_\mathrm{Nb}$', fontsize=18)
        plt.ylabel(r'$x_\mathrm{Cr}$', fontsize=18)
        plt.xticks(np.linspace(0, 1, 21))
        plt.scatter(Xtick, Ytick, color='black', s=3)
        plt.text(simX(0.010, 0.495), simY(0.495), r'$\gamma$', fontsize=14)
        dann = plt.text(simX(0.230, 0.010), simY(0.010), r'$\delta$', fontsize=14)
        lann = plt.text(simX(0.340, 0.275), simY(0.275), r'L',        fontsize=14)

        # Plot system composition and bisector
        # xCr0, xNb0 = np.genfromtxt("{0}/c.log".format(datdir), usecols=(2, 3), delimiter='\t', skip_header=1, unpack=True)
        xb, yb = draw_bisector(5., 7.)
        plt.plot(xb, yb, ls=':', c="green", zorder=1)
        # plt.scatter(simX(xNb0[-1], xCr0[-1]), simY(xCr0[-1]), zorder=1)

        # Add composition pathways
        fnames = sorted(glob.glob("{0}/*.xy".format(datdir)))
        for file in fnames[::10]:
            try:
                x, xcr, xnb, P = np.loadtxt(file, delimiter=',', unpack=True)
                # num = int(re.search('[0-9]{5,16}', file).group(0)) / 1000000
                plt.plot(simX(xnb, xcr), simY(xcr), '-', markersize=2, linewidth=1, zorder=1, c='gray')
                # plt.plot(simX(xnb, xcr), simY(xcr), linewidth=1, zorder=1, label=num)
            except:
                print("Empty file: ", file)

        try:
            dgcr, dgnb, dcr, dnb, lgcr, lgnb, lcr, lnb = np.loadtxt("{0}/diffusion_{1}.xc".format(datdir, base), delimiter=',', skiprows=1, usecols=(3,4,5,6,10,11,12,13), unpack=True)
            plt.plot(simX(dgnb, dgcr), simY(dgcr), label=r'$\gamma-\delta$', c='blue')
            plt.plot(simX(lgnb, lgcr), simY(lgcr), label=r'$\gamma-$L', c='green')
            plt.plot(simX(dnb, dcr), simY(dcr), label=r'$\delta$', c='coral')
            plt.plot(simX(lnb, lcr), simY(lcr), label=r'L', c='magenta')
        except:
            print("Empty file: {0}/diffusion_{1}.xc".format(datdir, base))

        plt.xlim([0, 0.6])
        plt.ylim([0, rt3by2*0.6])
        plt.legend(loc='best')
        plt.savefig("diagrams/TKR4p158/pathways_{0}.png".format(base), dpi=400, bbox_inches='tight')

        dann.remove()
        lann.remove()

        plt.xlim([0.175, 0.425])
        plt.ylim([0.275, 0.275+rt3by2*0.25])
        plt.savefig("diagrams/TKR4p158/pathways_zoom_{0}.png".format(base), dpi=400, bbox_inches='tight')
        plt.close()
    else:
        print("Skipping {0}".format(datdir))

