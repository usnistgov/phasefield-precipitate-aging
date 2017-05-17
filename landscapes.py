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
from pycalphad import equilibrium
from pycalphad import variables as v

# setup global variables

Titles = (r'$\gamma$', r'$\delta$', r'Laves')
npts = 100
nfun = 3
span = (-0.1, 1.1)
yspan = (-0.1, 0.9)
x = np.linspace(span[0], span[1], npts)
y = np.linspace(yspan[0], yspan[1], npts)
z = np.ndarray(shape=(nfun,len(x)*len(y)), dtype=float)

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


# Plot CALPHAD free energies using extracted equations
datmin = 1.0e10 * np.ones(nfun)
datmax = -1.0e10 * np.ones(nfun)
p = np.zeros(len(x)*len(y))
q = np.zeros(len(x)*len(y))
n = 0
z.fill(0.0)
for j in tqdm(np.nditer(y)):
    for i in np.nditer(x):
        xcr = 1.0*j / rt3by2
        xnb = 1.0*i - 0.5 * j / rt3by2
        p[n] = i
        q[n] = j
        z[0][n] = GG(xcr, xnb)
        z[1][n] = GD(xcr, xnb)
        z[2][n] = GL(xcr, xnb)
        for k in range(nfun):
        	datmin[k] = min(datmin[k], z[k][n])
        	datmax[k] = max(datmax[k], z[k][n])
        n += 1

print "Data spans [%.4g, %.4g]" % (np.amin(datmin), np.amax(datmax))


f, axarr = plt.subplots(nrows=1, ncols=3, sharex='col', sharey='row')
f.suptitle("IN625 Ternary Potentials (Restricted)",fontsize=14)
n=0
for ax in axarr.reshape(-1):
    #levels = np.logspace(np.log2(datmin[n]-1.01*datmin[n]), np.log2(datmax[n]-1.01*datmin[n]), num=50, base=2.0)
    ax.set_title(Titles[n],fontsize=10)
    ax.axis('equal')
    ax.set_xlim(span)
    ax.set_ylim(yspan)
    ax.axis('off')
    for a in range(len(XG)):
        ax.plot(XG[a], YG[a], ':w', linewidth=0.5)
    #ax.tricontourf(p, q, z[n]-1.011*datmin[n], levels, cmap=plt.cm.get_cmap('coolwarm'), norm=LogNorm())
    ax.tricontourf(p, q, z[n], cmap=plt.cm.get_cmap('coolwarm'))
    ax.plot(XS, YS, 'k', linewidth=0.5)
    ax.scatter(X0[n], Y0[n], color='black', s=2.5)
    n+=1
plt.figtext(x=0.5, y=0.0625, ha='center', fontsize=8, \
            s=r'White triangles enclose Gibbs simplex, $x_{\mathrm{Cr}}+x_{\mathrm{Nb}}+x_{\mathrm{Ni}}=1$.')
f.savefig('ternary.png', dpi=400, bbox_inches='tight')
plt.close()

images = ['diagrams/gamma_calphad.png', 'diagrams/delta_calphad.png', 'diagrams/Laves_calphad.png']
# files  = ['diagrams/gamma_calphad.txt', 'diagrams/delta_calphad.txt', 'diagrams/Laves_calphad.txt']
for n in range(nfun):
    levels = np.logspace(np.log2(datmin[n]-1.01*datmin[n]), np.log2(datmax[n]-1.01*datmin[n]), num=50, base=2.0)
    plt.axis('equal')
    plt.xlim(span)
    plt.ylim(yspan)
    plt.axis('off')
    for a in range(len(XG)):
        plt.plot(XG[a], YG[a], ':w', linewidth=0.5)
    plt.tricontourf(p, q, z[n]-1.011*datmin[n], levels, cmap=plt.cm.get_cmap('coolwarm'), norm=LogNorm())
    #plt.tricontourf(p, q, z[n], cmap=plt.cm.get_cmap('coolwarm'))
    plt.plot(XS, YS, 'k', linewidth=0.5)
    plt.scatter(X0[n], Y0[n], color='black', s=2.5)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(images[n], transparent=True, dpi=400, bbox_inches='tight', pad_inches=0)
    plt.close()
    # points = np.array([p, q, z[n]])
    # np.savetxt(files[n], points.T)


# Plot Taylor series approximate free energy landscapes
datmin = 1.0e10 * np.ones(nfun)
datmax = -1.0e10 * np.ones(nfun)
p = np.zeros(len(x)*len(y))
q = np.zeros(len(x)*len(y))
n = 0
z.fill(0.0)
for j in tqdm(np.nditer(y)):
    for i in np.nditer(x):
        xcr = 1.0*j / rt3by2
        xnb = 1.0*i - 0.5 * j / rt3by2
        p[n] = i
        q[n] = j
        z[0][n] = TG(xcr, xnb)
        z[1][n] = TD(xcr, xnb)
        z[2][n] = TL(xcr, xnb)
        for k in range(nfun):
        	datmin[k] = min(datmin[k], z[k][n])
        	datmax[k] = max(datmax[k], z[k][n])
        n += 1

print "Data spans [%.4g, %.4g]" % (np.amin(datmin), np.amax(datmax))


f, axarr = plt.subplots(nrows=1, ncols=3, sharex='col', sharey='row')
f.suptitle("IN625 Ternary Potentials (Taylor)",fontsize=14)
n=0
for ax in axarr.reshape(-1):
    #levels = np.logspace(np.log2(datmin[n]-1.01*datmin[n]), np.log2(datmax[n]-1.01*datmin[n]), num=50, base=2.0)
    ax.set_title(Titles[n],fontsize=10)
    ax.axis('equal')
    ax.set_xlim(span)
    ax.set_ylim(yspan)
    ax.axis('off')
    for a in range(len(XG)):
        ax.plot(XG[a], YG[a], ':w', linewidth=0.5)
    #ax.tricontourf(p, q, z[n]-1.011*datmin[n], levels, cmap=plt.cm.get_cmap('coolwarm'), norm=LogNorm())
    ax.tricontourf(p, q, z[n], cmap=plt.cm.get_cmap('coolwarm'))
    ax.plot(XS, YS, 'k', linewidth=0.5)
    ax.scatter(X0[n], Y0[n], color='black', s=2.5)
    n+=1
plt.figtext(x=0.5, y=0.0625, ha='center', fontsize=8, \
            s=r'White triangles enclose Gibbs simplex, $x_{\mathrm{Cr}}+x_{\mathrm{Nb}}+x_{\mathrm{Ni}}=1$.')
f.savefig('ternary_taylor.png', dpi=400, bbox_inches='tight')
plt.close()

images = ['diagrams/gamma_taylor.png', 'diagrams/delta_taylor.png', 'diagrams/Laves_taylor.png']
# files  = ['diagrams/gamma_taylor.txt', 'diagrams/delta_taylor.txt', 'diagrams/Laves_taylor.txt']
for n in range(nfun):
    #levels = np.logspace(np.log2(datmin[n]-1.01*datmin[n]), np.log2(datmax[n]-1.01*datmin[n]), num=50, base=2.0)
    plt.axis('equal')
    plt.xlim(span)
    plt.ylim(yspan)
    plt.axis('off')
    for a in range(len(XG)):
        plt.plot(XG[a], YG[a], ':w', linewidth=0.5)
    #plt.tricontourf(p, q, z[n]-1.011*datmin[n], levels, cmap=plt.cm.get_cmap('coolwarm'), norm=LogNorm())
    plt.tricontourf(p, q, z[n], cmap=plt.cm.get_cmap('coolwarm'))
    plt.plot(XS, YS, 'k', linewidth=0.5)
    plt.scatter(X0[n], Y0[n], color='black', s=2.5)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(images[n], transparent=True, dpi=400, bbox_inches='tight', pad_inches=0)
    plt.close()
    # points = np.array([p, q, z[n]])
    # np.savetxt(files[n], points.T)
