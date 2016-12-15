#!/usr/bin/python

from CALPHAD_energies import *

print "Parabolic Gamma: ", p_gamma
print "Parabolic Delta: ", p_delta
print "Parabolic Mu:    ", p_mu
print "Parabolic Laves: ", p_laves
print ""

print "Taylor Gamma: ", t_gamma
print "Taylor Delta: ", t_delta
print "Taylor Mu:    ", t_mu
print "Taylor Laves: ", t_laves
print ""

# Generate ternary axes
labels = [r'$\gamma$', r'$\delta$', r'$\mu$', 'Laves']
colors = ['red', 'green', 'blue', 'cyan']

# triangle bounding the Gibbs simplex
XS = [0.0, 1.0, 0.5, 0.0]
YS = [0.0, 0.0,rt3by2, 0.0]
# triangle bounding three-phase coexistence
XT = [0.25, 0.4875+0.025/2,0.5375+0.4625/2, 0.25]
YT = [0.0,  0.025*rt3by2, 0.4625*rt3by2, 0.0]
# Tick marks along simplex edges
Xtick = []
Ytick = []
for i in range(20):
    # Cr-Ni edge
    xcr = 0.05*i
    Xtick.append(xcr/2 - 0.002)
    Ytick.append(rt3by2*xcr)
    # Cr-Nb edge
    xcr = 0.05*i
    xnb = 1.0 - xcr
    Xtick.append(xnb + xcr/2 + 0.002)
    Ytick.append(rt3by2*xcr)


# Generate ternary phase diagram

density = 101

allY = []
allX = []
allG = []
allID = []
points = []

for phi in np.linspace(-1.1, 1.1, density): #, endpoint=False):
    for psi in np.linspace(-0.05, 1, density, endpoint=False):
        xni = (1+phi)*(1-psi)/2 #+ 0.5*(np.random.random_sample() - 0.5)/(density-1)
        xnb = (1-phi)*(1-psi)/2 #+ 0.5*(np.random.random_sample() - 0.5)/(density-1)
        xcr = 1 - xni - xnb     # psi + (np.random.random_sample() - 0.5)/(density-1)
        if  (xcr < 1  and xcr > -0.05) \
        and (xnb < 1.1  and xnb > -0.1) \
        and (xni < 1.1  and xni > -0.1):
            f = (c_gamma.subs({GAMMA_XCR: xcr, GAMMA_XNB: xnb, GAMMA_XNI: xni}),
                 c_delta.subs({DELTA_XCR: xcr, DELTA_XNB: xnb, DELTA_XNI: xni}),
                 c_mu.subs(   {MU_XCR: xcr, MU_XNB: xnb, MU_XNI: xni}),
                 c_laves.subs({LAVES_XCR: xcr, LAVES_XNB: xnb, LAVES_XNB: xnb, LAVES_XNI: xni}))
            for n in range(len(f)):
                allX.append(xnb + xcr/2)
                allY.append(rt3by2*xcr)
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
plt.title("Cr-Nb-Ni at %.0fK"%temp, fontsize=18)
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1*rt3by2, 1.1*rt3by2])
plt.xlabel(r'$x_\mathrm{Nb}$', fontsize=24)
plt.ylabel(r'$x_\mathrm{Cr}$', fontsize=24)
plt.plot(XS, YS, '-k')
for tie in tielines:
    plt.plot(tie[0], tie[1], '-k', alpha=0.025)
for i in range(len(labels)):
    plt.scatter(X[i], Y[i], color=colors[i], s=2.5, label=labels[i])
plt.scatter(X0, Y0, color='black', s=6)
plt.xticks(np.linspace(0, 1, 21))
plt.scatter(Xtick, Ytick, color='black', s=3)
#plt.scatter(0.02+0.3/2, rt3by2*0.3, color='red', s=8)
#plt.scatter(simX(0.1625, 0.013), simY(0.013), color='black', s=5)
plt.legend(loc='best')
plt.savefig("parabolic_energy.png", bbox_inches='tight', dpi=100)
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
        cgam.append(c_gamma.subs({GAMMA_XCR: xcr, GAMMA_XNB: xnb, GAMMA_XNI: xni}))
        cdel.append(c_delta.subs({DELTA_XCR: xcr, DELTA_XNB: xnb, DELTA_XNI: xni}))
        cmu.append( c_mu.subs(   {MU_XCR:    xcr, MU_XNB:    xnb, MU_XNI:    xni}))
        clav.append(c_laves.subs({LAVES_XCR: xcr, LAVES_XNB: xnb, LAVES_XNI: xni}))

    plt.plot(x, cgam, color=colors[0], label=r'$\gamma$, $x_{\mathrm{Cr}}=%.2f$'%xcr)
    plt.plot(x, cdel, color=colors[1], label=r'$\delta$, $x_{\mathrm{Cr}}=%.2f$'%xcr)
    plt.plot(x, cmu,  color=colors[2], label=r'$\mu$, $x_{\mathrm{Cr}}=%.2f$'%xcr)
    plt.plot(x, clav, color=colors[3], label=r'L, $x_{\mathrm{Cr}}=%.2f$'%xcr)

# un-indent using for loops
plt.legend(loc='best', fontsize=6)
plt.savefig("linescan_Nb.png", bbox_inches='tight', dpi=300)
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
        cgam.append(c_gamma.subs({GAMMA_XCR: xcr, GAMMA_XNB: xnb, GAMMA_XNI: xni}))
        cdel.append(c_delta.subs({DELTA_XCR: xcr, DELTA_XNB: xnb, DELTA_XNI: xni}))
        cmu.append( c_mu.subs(   {MU_XCR:    xcr, MU_XNB:    xnb, MU_XNI:    xni}))
        clav.append(c_laves.subs({LAVES_XCR: xcr, LAVES_XNB: xnb, LAVES_XNI: xni}))

    plt.plot(x, cgam, color=colors[0], label=r'$\gamma$, $x_{\mathrm{Nb}}=%.2f$'%xnb)
    plt.plot(x, cdel, color=colors[1], label=r'$\delta$, $x_{\mathrm{Nb}}=%.2f$'%xnb)
    plt.plot(x, cmu,  color=colors[2], label=r'$\mu$, $x_{\mathrm{Nb}}=%.2f$'%xnb)
    plt.plot(x, clav, color=colors[3], label=r'L, $x_{\mathrm{Nb}}=%.2f$'%xnb)

# un-indent using for loops
plt.legend(loc='best', fontsize=6)
plt.savefig("linescan_Cr.png", bbox_inches='tight', dpi=300)
plt.close()
