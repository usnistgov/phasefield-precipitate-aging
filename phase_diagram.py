#!/usr/bin/python

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

from CALPHAD_energies import *

# Generate ternary axes
labels = [r'$\gamma$', r'$\delta$', r'$\mu$', 'Laves']
colors = ['red', 'green', 'blue', 'cyan']

# Generate ternary phase diagram

density = 1001

allY = []
allX = []
allG = []
allID = []
points = []
dc = 0.01

for phi in np.linspace(-1-dc, 1+dc, density):
    for psi in np.linspace(-dc, 1, density, endpoint=False):
        xni = (1+phi)*(1-psi)/2
        xnb = (1-phi)*(1-psi)/2
        xcr = 1 - xni - xnb
        if  (xcr < 1  and xcr > -dc) \
        and (xnb < 1+dc  and xnb > -dc) \
        and (xni < 1+dc  and xni > -dc):
            f = (TG(xcr, xnb),
                 TD(xcr, xnb),
                 TU(xcr, xnb),
                 TL(xcr, xnb))
            for n in range(len(f)):
                allX.append(simX(xnb,xcr))
                allY.append(simY(xcr))
                allG.append(f[n])
                allID.append(n)

points = np.array([allX, allY, allG]).T
hull = ConvexHull(points)

# Prepare arrays for plotting
X = [[],[],[],[]]
Y = [[],[],[],[]]
tielines = []

for simplex in hull.simplices:
    if len(simplex) != 3:
        print simplex
    for i in simplex:
        X[allID[i]].append(allX[i])
        Y[allID[i]].append(allY[i])
        for j in simplex:
            if j>i and allID[i] != allID[j]:
                tielines.append([[allX[i], allX[j]], [allY[i], allY[j]]])

# Plot phase diagram
pltsize = 20
plt.figure(figsize=(pltsize, rt3by2*pltsize))
plt.title("Cr-Nb-Ni (Taylor approx) at %.0fK"%temp, fontsize=18)
plt.xlim([-dc, 1+dc])
plt.ylim([simY(-dc), simY(1+dc)])
plt.xlabel(r'$x_\mathrm{Nb}$', fontsize=24)
plt.ylabel(r'$x_\mathrm{Cr}$', fontsize=24)
plt.plot(XS, YS, '-k')
for tie in tielines:
    plt.plot(tie[0], tie[1], '-k', alpha=0.1)
for i in range(len(labels)):
    plt.scatter(X[i], Y[i], color=colors[i], s=4, label=labels[i])
plt.scatter(X0, Y0, color='black', marker='s', s=8)
plt.xticks(np.linspace(0, 1, 21))
plt.scatter(Xtick, Ytick, color='black', marker='+', s=8)
plt.legend(loc='best')
plt.savefig("taylor_phase_diagram.png", bbox_inches='tight', dpi=400)
plt.close()

# Compare CALPHAD and parabolic expressions (const. Cr)
stepsz = 0.005

# Plot phase diagram
plt.figure()
plt.title("Cr-Nb-Ni at %.0fK"%temp)
plt.xlabel(r'$x_\mathrm{Nb}$')
plt.ylabel(r'$\mathcal{F}$')
plt.ylim([-1e10, 0])

for xcr in (0.01, 0.3):
    x = []

    cgam = []
    cdel = []
    cmu  = []
    clav = []

    for xnb in np.arange(0.01, 0.98, stepsz):
        xni = 1-xcr-xnb
        x.append(xnb)
        cgam.append(GG(xcr, xnb))
        cdel.append(GD(xcr, xnb))
        cmu.append( GU(xcr, xnb))
        clav.append(GL(xcr, xnb))

    plt.plot(x, cgam, color=colors[0], label=r'$\gamma$, $x_{\mathrm{Cr}}=%.2f$'%xcr)
    plt.plot(x, cdel, color=colors[1], label=r'$\delta$, $x_{\mathrm{Cr}}=%.2f$'%xcr)
    plt.plot(x, cmu,  color=colors[2], label=r'$\mu$, $x_{\mathrm{Cr}}=%.2f$'%xcr)
    plt.plot(x, clav, color=colors[3], label=r'L, $x_{\mathrm{Cr}}=%.2f$'%xcr)

# un-indent using for loops
plt.legend(loc='best', fontsize=6)
plt.savefig("linescan_Nb.png", bbox_inches='tight', dpi=400)
plt.close()


# Compare CALPHAD and parabolic expressions (const. Nb)

# Plot phase diagram
plt.figure()
plt.title("Cr-Nb-Ni at %.0fK"%temp)
plt.xlabel(r'$x_\mathrm{Cr}$')
plt.ylabel(r'$\mathcal{F}$')
plt.ylim([-1e10, 0])

for xnb in (0.02, 0.15):
    x = []

    cgam = []
    cdel = []
    cmu  = []
    clav = []

    for xcr in np.arange(0.01, 0.6, stepsz):
        x.append(xcr)
        cgam.append(GG(xcr, xnb))
        cdel.append(GD(xcr, xnb))
        cmu.append( GU(xcr, xnb))
        clav.append(GL(xcr, xnb))

    plt.plot(x, cgam, color=colors[0], label=r'$\gamma$, $x_{\mathrm{Nb}}=%.2f$'%xnb)
    plt.plot(x, cdel, color=colors[1], label=r'$\delta$, $x_{\mathrm{Nb}}=%.2f$'%xnb)
    plt.plot(x, cmu,  color=colors[2], label=r'$\mu$, $x_{\mathrm{Nb}}=%.2f$'%xnb)
    plt.plot(x, clav, color=colors[3], label=r'L, $x_{\mathrm{Nb}}=%.2f$'%xnb)

# un-indent using for loops
plt.legend(loc='best', fontsize=6)
plt.savefig("linescan_Cr.png", bbox_inches='tight', dpi=400)
plt.close()
