#!/usr/bin/python
# coding: utf-8

# Plot compositions in ternary simplex

# Numerical libraries
import numpy as np
from os import path, stat
from sys import argv

# Check data directory
datdir = argv[1]

if path.isdir(datdir):
    base = path.basename(datdir)
    
    # Visualization libraries
    import matplotlib.pylab as plt
    
    def simX(x2, x3):
        return x2 + fr1by2 * x3
        
    def simY(x3):
        return rt3by2 * x3
    
    fr1by2 = 1.0/2
    rt3by2 = np.sqrt(3)/2
    
    # triangle bounding the Gibbs simplex
    XS = [0.0, simX(1,0), simX(0,1), 0.0]
    YS = [0.0, simY(0),   simY(1),   0.0]
    
    # triangle bounding three-phase coexistence
    X0 = [simX(0.025, 0.490), simX(0.245, 0.015), simX(0.283, 0.300)]
    Y0 = [simY(0.490),        simY(0.015),        simY(0.300)]
    
    # Tick marks along simplex edges
    Xtick = []
    Ytick = []
    tickdens = 10
    for i in range(tickdens):
        # Cr-Ni edge
        xcr = (1.0 * i) / tickdens
        xni = 1.0 - xcr
        Xtick.append(simX(-0.002, xcr))
        Ytick.append(simY(xcr))
        # Cr-Nb edge
        xcr = (1.0 * i) / tickdens
        xnb = 1.0 - xcr
        Xtick.append(simX(xnb+0.002, xcr))
        Ytick.append(simY(xcr))
        # Nb-Ni edge
        xnb = (1.0 * i) / tickdens
        Xtick.append(xnb)
        Ytick.append(-0.002)
    
    # Triangular grid
    XG = [[]]
    YG = [[]]
    for a in np.arange(0, 1, 0.1):
        # x1--x2: lines of constant x2=a
        XG.append([simX(a, 0), simX(a, 1-a)])
        YG.append([simY(0),    simY(1-a)])
        # x2--x3: lines of constant x3=a
        XG.append([simX(0, a), simX(1-a, a)])
        YG.append([simY(a),    simY(a)])
        # x1--x3: lines of constant x1=1-a
        XG.append([simX(0, a), simX(a, 0)])
        YG.append([simY(a),    simY(0)])
    
    # Plot ternary axes and labels
    plt.figure(figsize=(10, 7.5)) # inches
    plt.plot(XS, YS, '-k')
    plt.title("Cr-Nb-Ni Fictitious Compositions", fontsize=18)
    plt.xlabel(r'$x_\mathrm{Nb}$', fontsize=24)
    plt.ylabel(r'$x_\mathrm{Cr}$', fontsize=24)
    plt.xlim([0, 1])
    plt.ylim([0, rt3by2])
    plt.xticks(np.linspace(0, 1, 11))
    plt.scatter(Xtick, Ytick, color='black', s=3, zorder=10)
    plt.scatter(X0, Y0, color='black', s=9, zorder=10)
    for a in range(len(XG)):
        plt.plot(XG[a], YG[a], ':k', linewidth=0.5, alpha=0.5)
    
    # Plot compositions given to rootsolver
    thebad = datdir + "/badroots.log"
    if path.isfile(thebad) and stat(thebad).st_size > 0:
        gam_xcr, gam_xnb, del_xcr, del_xnb, lav_xcr, lav_xnb = np.loadtxt(thebad, delimiter=',', unpack=True)
        plt.scatter(simX(gam_xnb, gam_xcr), simY(gam_xcr), color='red', s=2, zorder=1, label="bad $\gamma$")
        plt.scatter(simX(del_xnb, del_xcr), simY(del_xcr), color='yellow', s=2, zorder=1, label="bad $\delta$")
        plt.scatter(simX(lav_xnb, lav_xcr), simY(lav_xcr), color='orange', s=2, zorder=1, label="bad L")
    
    thegud = datdir + "/gudroots.log" 
    if path.isfile(thegud) and stat(thegud).st_size > 0:
        gam_xcr, gam_xnb, del_xcr, del_xnb, lav_xcr, lav_xnb = np.loadtxt(thegud, delimiter=',', unpack=True)
        plt.scatter(simX(gam_xnb, gam_xcr), simY(gam_xcr), color='green', s=1.25, zorder=1, label="$\gamma$")
        plt.scatter(simX(del_xnb, del_xcr), simY(del_xcr), color='blue', s=1.25, zorder=1, label="$\delta$")
        plt.scatter(simX(lav_xnb, lav_xcr), simY(lav_xcr), color='red', s=1.25, zorder=1, label="L")
    
    plt.legend(loc='best')
    plt.savefig("diagrams/pathways_{0}.png".format(base), dpi=400, bbox_inches='tight')
    plt.close()
else:
    print("Invalid argument: {0} is not a directory.".format(datdir))
    print("Usage: {0} path/to/data".format(argv[0]))
