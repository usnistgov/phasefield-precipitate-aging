#!/usr/bin/python
# -*- coding: utf-8 -*-

#####################################################################################
# This software was developed at the National Institute of Standards and Technology #
# by employees of the Federal Government in the course of their official duties.    #
# Pursuant to title 17 section 105 of the United States Code this software is not   #
# subject to copyright protection and is in the public domain. NIST assumes no      #
# responsibility whatsoever for the use of this code by other parties, and makes no #
# guarantees, expressed or implied, about its quality, reliability, or any other    #
# characteristic. We would appreciate acknowledgement if the software is used.      #
#                                                                                   #
# This software can be redistributed and/or modified freely provided that any       #
# derivative works bear some notice that they are derived from it, and any modified #
# versions bear some notice that they have been modified.                           #
#####################################################################################

# Overlay phase-field simulation compositions on ternary phase diagram
# Before executing this script, run the mmsp2comp utility
# for each checkpoint file in the directories of interest.
# Usage: python analysis/pathways158.py

import re
import numpy as np
from math import floor, sqrt
from os import path
from sys import argv
import glob
import matplotlib.pylab as plt

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from CALPHAD_energies import *

density = 500
skipsz = 9

labels = [r'$\gamma$', r'$\delta$', 'Laves']
colors = ['red', 'green', 'blue']

def draw_bisector(A, B):
    bNb = (A * xe_del_Nb + B * xe_lav_Nb) / (A + B)
    bCr = (A * xe_del_Cr + B * xe_lav_Cr) / (A + B)
    x = [simX(xe_gam_Nb, xe_gam_Cr), simX(bNb, bCr)]
    y = [simY(xe_gam_Cr), simY(bCr)]
    return x, y

for datdir in glob.glob('/data/tnk10/phase-field/alloy625/TKR4p158/run*'): #{0}".format(j) for j in (2,3,6,12,21,63)]:
    if path.isdir(datdir) and len(glob.glob("{0}/*.xy".format(datdir))) > 0:
        base = path.basename(datdir)
        # Plot phase diagram
        plt.figure(0, figsize=(10, 7.5)) # inches
        plt.plot(XS, YS, '-k')
        plt.plot(X0, Y0, '-k', zorder=1)
        plt.title("Cr-Nb-Ni at %.0f K"%temp, fontsize=18)
        plt.xlabel(r'$x_\mathrm{Nb}$', fontsize=18)
        plt.ylabel(r'$x_\mathrm{Cr}$', fontsize=18)
        plt.xticks(np.linspace(0, 1, 21))
        plt.scatter(Xtick, Ytick, color='black', s=3)
        gann = plt.text(simX(0.010, 0.495), simY(0.495), r'$\gamma$', fontsize=14)
        dann = plt.text(simX(0.230, 0.010), simY(0.010), r'$\delta$', fontsize=14)
        lann = plt.text(simX(0.340, 0.275), simY(0.275), r'L',        fontsize=14)

        # Add composition pathways
        fnames = sorted(glob.glob("{0}/*.xy".format(datdir)))
        for file in fnames[::10]:
            try:
                x, xcr, xnb, P = np.loadtxt(file, delimiter=',', unpack=True)
                num = int(re.search('[0-9]{5,16}', file).group(0)) * 7.5e-5
                plt.plot(simX(xnb, xcr), simY(xcr), '-', linewidth=1, zorder=1, color='gray', label=r'%.0f s' % num)
                # plt.plot(simX(xnb, xcr), simY(xcr), linewidth=1, zorder=1)
            except:
                print("Empty file: ", file)

        try:
            dgcr, dgnb, dcr, dnb, lgcr, lgnb, lcr, lnb = np.loadtxt("{0}/diffusion_{1}.xc".format(datdir, base), delimiter=',', skiprows=1, usecols=(3,4,5,6,10,11,12,13), unpack=True)
            plt.plot(simX(dgnb, dgcr), simY(dgcr), c='blue') # , label=r'$\gamma(\delta)$'
            plt.plot(simX(lgnb, lgcr), simY(lgcr), c='green') # , label=r'$\gamma($L$)$'
            plt.plot(simX(dnb, dcr), simY(dcr), c='coral') # , label=r'$\delta$'
            plt.plot(simX(lnb, lcr), simY(lcr), c='magenta') # , label=r'L'
        except:
            print("Empty file: {0}/diffusion_{1}.xc".format(datdir, base))

        plt.xlim([0, 0.6])
        plt.ylim([0, rt3by2*0.6])
        plt.legend(loc='best')
        plt.savefig("diagrams/TKR4p158/pathways/pathways_{0}.png".format(base), dpi=400, bbox_inches='tight')

        dann.remove()
        lann.remove()
        plt.xticks([])
        plt.yticks([])

        plt.xlim([0.175, 0.425])
        plt.ylim([0.275, 0.275+rt3by2*0.25])
        plt.savefig("diagrams/TKR4p158/pathways/pathways_zm_gam_{0}.png".format(base), dpi=400, bbox_inches='tight')

        gann.remove()
        dann = plt.text(simX(0.2375, 0.010), simY(0.010), r'$\delta$', fontsize=14)

        plt.xlim([0.2375, 0.3])
        plt.ylim([0.0, rt3by2 * 0.05125])
        plt.savefig("diagrams/TKR4p158/pathways/pathways_zm_del_{0}.png".format(base), dpi=400, bbox_inches='tight')

        dann.remove()
        lann = plt.text(simX(0.345, 0.3), simY(0.3), r'L',        fontsize=14)

        plt.xlim([0.45, 0.55])
        plt.ylim([0.25, 0.25 + rt3by2*0.1])
        plt.savefig("diagrams/TKR4p158/pathways/pathways_zm_lav_{0}.png".format(base), dpi=400, bbox_inches='tight')

        plt.close()

        # Plot phase diagram
        plt.figure(1, figsize=(10, 7.5)) # inches
        plt.plot(XS, YS, '-k')
        plt.plot(X0, Y0, '-k', zorder=1)
        plt.axis('off')
        #plt.xlabel(r'$x_\mathrm{Nb}$', fontsize=18)
        #plt.ylabel(r'$x_\mathrm{Cr}$', fontsize=18)
        #plt.xticks(np.linspace(0, 1, 21))
        #plt.scatter(Xtick, Ytick, color='black', s=3)
        gann = plt.text(simX(0.010, 0.495), simY(0.495), r'$\gamma$', fontsize=14)
        dann = plt.text(simX(0.230, 0.010), simY(0.010), r'$\delta$', fontsize=14)
        lann = plt.text(simX(0.340, 0.275), simY(0.275), r'L',        fontsize=14)
        xCr0, xNb0 = np.genfromtxt("{0}/c.log".format(datdir), usecols=(2, 3), delimiter='\t', skip_header=1, unpack=True)
        plt.scatter(simX(xNb0[-1], xCr0[-1]), simY(xCr0[-1]), zorder=1, color='black')
        plt.xlim([0, 0.6])
        plt.ylim([0, rt3by2*0.6])
        plt.savefig("diagrams/TKR4p158/triangles/composition_{0}.png".format(base), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        print("Skipping {0}".format(datdir))

