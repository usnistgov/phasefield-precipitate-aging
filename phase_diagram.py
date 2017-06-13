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
from tqdm import tqdm

phases = ["gamma", "delta", "laves"]
labels = [r"$\gamma$", r"$\delta$", "Laves"]
colors = ['red', 'green', 'blue']

### LINESCANS
stepsz = 0.01
dc = 0.05

# Plot Nb linescans

pltgam = plt.figure(0)
pltdel = plt.figure(1)
pltlav = plt.figure(2)

for j in range(len(phases)):
	plt.figure(j)
	plt.title("{0} linescan at {1}K".format(labels[j], temp))
	plt.xlabel(r'$x_\mathrm{Nb}$')
	plt.ylabel(r'$\mathcal{F}$')
	plt.ylim([-1e10, 3e10])

n = 0
for xcr in (0.1, 0.2, 0.3):
    x = []
	
    g = [[], [], []] # CALPHAD array
    t = [[], [], []] # Taylor array
    p = [[], [], []] # parabolic array
	
    for xnb in np.arange(-dc, 1+dc, stepsz):
        xni = 1-xcr-xnb
        x.append(xnb)
        #g[0].append(GG(xcr, xnb))
        #g[1].append(GD(xcr, xnb))
        #g[2].append(GL(xcr, xnb))
        t[0].append(TG(xcr, xnb))
        t[1].append(TD(xcr, xnb))
        t[2].append(TL(xcr, xnb))
        p[0].append(PG(xcr, xnb))
        p[1].append(PD(xcr, xnb))
        p[2].append(PL(xcr, xnb))
	
    for j in range(len(phases)):
    	plt.figure(j)
    	#plt.plot(x, g[j], color=colors[n], label=r'CALPHAD $x_{\mathrm{Cr}}=%.2f$'%xcr)
    	plt.plot(x, t[j], color=colors[n], ls='-', label=r'Taylor $x_{\mathrm{Cr}}=%.2f$'%xcr)
    	plt.plot(x, p[j], color=colors[n], ls='-.', label=r'parabola $x_{\mathrm{Cr}}=%.2f$'%xcr)
    
    n += 1

for j in range(len(phases)):
	plt.figure(j)
	plt.legend(loc='best', fontsize=6)
	plt.savefig("linescan_{0}_Nb.png".format(phases[j]), bbox_inches='tight', dpi=400)
	plt.close()

print("Finished plotting Nb linescan.")

# Plot Cr linescans

pltgam = plt.figure(0)
pltdel = plt.figure(1)
pltlav = plt.figure(2)

for j in range(len(phases)):
	plt.figure(j)
	plt.title("{0} linescan at {1}K".format(labels[j], temp))
	plt.xlabel(r'$x_\mathrm{Cr}$')
	plt.ylabel(r'$\mathcal{F}$')
	plt.ylim([-1e10, 3e10])

n = 0
for xnb in (0.01, 0.05, 0.10):
    x = []
	
    g = [[], [], []] # CALPHAD array
    t = [[], [], []] # Taylor array
    p = [[], [], []] # parabolic array
	
    for xcr in np.arange(-dc, 1+dc, stepsz):
        x.append(xcr)
        #g[0].append(GG(xcr, xnb))
        #g[1].append(GD(xcr, xnb))
        #g[2].append(GL(xcr, xnb))
        t[0].append(TG(xcr, xnb))
        t[1].append(TD(xcr, xnb))
        t[2].append(TL(xcr, xnb))
        p[0].append(PG(xcr, xnb))
        p[1].append(PD(xcr, xnb))
        p[2].append(PL(xcr, xnb))
	
    for j in range(len(phases)):
    	plt.figure(j)
    	#plt.plot(x, g[j], color=colors[n], label=r'CALPHAD $x_{\mathrm{Nb}}=%.2f$'%xnb)
    	plt.plot(x, t[j], color=colors[n], ls='-', label=r'Taylor $x_{\mathrm{Nb}}=%.2f$'%xnb)
    	plt.plot(x, p[j], color=colors[n], ls='-.', label=r'parabola $x_{\mathrm{Nb}}=%.2f$'%xnb)
    
    n += 1

for j in range(len(phases)):
	plt.figure(j)
	plt.legend(loc='best', fontsize=6)
	plt.savefig("linescan_{0}_Cr.png".format(phases[j]), bbox_inches='tight', dpi=400)
	plt.close()

print("Finished plotting Cr linescan.")



# Generate ternary phase diagram

density = 1001

allY = []
allX = []
allG = []
allID = []
points = []
dc = 0.005

for xcr in tqdm(np.linspace(dc, 1-dc, density)):
    for xnb in np.linspace(dc, 1-dc, density, endpoint=False):
        xni = 1 - xcr - xnb
        if xni < 1-dc  and xni > dc:
            # CALPHAD expressions
            #f = (GG(xcr, xnb),
            #     GD(xcr, xnb),
            #     GL(xcr, xnb))
            # Taylor approximation
            #f = (TG(xcr, xnb),
            #     TD(xcr, xnb),
            #     TL(xcr, xnb))
            # Parabolic approximation
            f = (PG(xcr, xnb),
                 PD(xcr, xnb),
                 PL(xcr, xnb))
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
    for i in simplex:
        X[allID[i]].append(allX[i])
        Y[allID[i]].append(allY[i])
    for i in simplex:
        for j in simplex:
            # Record facets of the simplex as tie-lines
            #if j>i and allID[i] != allID[j]:
            #    tielines.append([[allX[i], allX[j]], [allY[i], allY[j]]])
            for k in simplex:
                # Record only the boundaries of three-phase coexistence fields
                if j>i and k>j and allID[i] != allID[j] and allID[j] != allID[k] and allID[i] != allID[k]:
                    tielines.append([[allX[i], allX[j]], [allY[i], allY[j]]])
                    tielines.append([[allX[j], allX[k]], [allY[j], allY[k]]])
                    tielines.append([[allX[i], allX[k]], [allY[i], allY[k]]])

# Plot phase diagram
pltsize = 20
plt.figure(figsize=(pltsize, rt3by2*pltsize))
#plt.title("Cr-Nb-Ni (CALPHAD expr) at %.0fK"%temp, fontsize=18)
#plt.title("Cr-Nb-Ni (Taylor approx) at %.0fK"%temp, fontsize=18)
plt.title("Cr-Nb-Ni (Parabolic approx) at %.0fK"%temp, fontsize=18)
#plt.xlim([-dc, 1+dc])
#plt.ylim([simY(-dc), simY(1+dc)])
plt.xlabel(r'$x_\mathrm{Nb}$', fontsize=24)
plt.ylabel(r'$x_\mathrm{Cr}$', fontsize=24)
plt.plot(XS, YS, '-k')
for tie in tielines:
    plt.plot(tie[0], tie[1], '-k', alpha=0.5)
for i in range(len(labels)):
    plt.scatter(X[i], Y[i], color=colors[i], s=4, label=labels[i])
#plt.scatter(XT, YT, color='black', marker='s', s=8)
plt.scatter(X0, Y0, color='black', marker='s', s=8)
plt.xticks(np.linspace(0, 1, 21))
plt.scatter(Xtick, Ytick, color='black', marker='+', s=8)
plt.legend(loc='best')
#plt.savefig("calphad_phase_diagram.png", bbox_inches='tight', dpi=400)
#plt.savefig("taylor_phase_diagram.png", bbox_inches='tight', dpi=400)
plt.savefig("parabolic_phase_diagram.png", bbox_inches='tight', dpi=400)
plt.close()

print("Finished plotting ternary phase diagram.")
