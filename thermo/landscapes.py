# -*- coding: utf-8 -*-

# Usage: python thermo/landscapes.py

from tqdm import tqdm
from pycalphad import equilibrium
from pycalphad import variables as v
from CALPHAD_energies import *

Titles = (r'$\gamma$', r'$\delta$', r'Laves')
xspan = (-0.05, 1.05)
yspan = (-0.05, 0.95)
nfun = 3
npts = 500
ncon = 100
xmin = 1.0e-4
xmax = 1.0e11
x = np.linspace(xspan[0], xspan[1], npts)
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

# Plot paraboloid free energy landscapes

datmin = xmin * np.ones(nfun)
datmax = xmin * np.ones(nfun)
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
        z[0][n] = PG(xcr, xnb)
        z[1][n] = PD(xcr, xnb)
        z[2][n] = PL(xcr, xnb)
        for k in range(nfun):
        	datmin[k] = min(datmin[k], z[k][n])
        	datmax[k] = max(datmax[k], z[k][n])
        n += 1

print "Paraboloid data spans [%.4g, %.4g]" % (np.amin(datmin), np.amax(datmax))

f, axarr = plt.subplots(nrows=1, ncols=3, sharex='col', sharey='row')
f.suptitle("IN625 Ternary Potentials (Taylor)",fontsize=14)
n=0
for ax in axarr.reshape(-1):
    levels = np.logspace(np.log2(xmin), np.log2(xmax), num=ncon, base=2.0)
    ax.set_title(Titles[n],fontsize=10)
    ax.axis('equal')
    ax.set_xlim(xspan)
    ax.set_ylim(yspan)
    ax.axis('off')
    for a in range(len(XG)):
        ax.plot(XG[a], YG[a], ':w', linewidth=0.5)
    ax.tricontourf(p, q, z[n]-datmin[n]+xmin, levels, cmap=plt.cm.get_cmap('coolwarm'), norm=LogNorm())
    #ax.tricontourf(p, q, z[n], cmap=plt.cm.get_cmap('coolwarm'))
    ax.plot(XS, YS, 'k', linewidth=0.5)
    ax.scatter(X0[n], Y0[n], color='black', s=2.5)
    n+=1
plt.figtext(x=0.5, y=0.0625, ha='center', fontsize=8, \
            s=r'White triangles enclose Gibbs simplex, $x_{\mathrm{Cr}}+x_{\mathrm{Nb}}+x_{\mathrm{Ni}}=1$.')
f.savefig('diagrams/ternary_parabola.png', dpi=400, bbox_inches='tight')
plt.close()

images = ['diagrams/gamma_parabola.png', 'diagrams/delta_parabola.png', 'diagrams/Laves_parabola.png']
for n in range(nfun):
    levels = np.logspace(np.log2(xmin), np.log2(xmax), num=ncon, base=2.0)
    plt.axis('equal')
    plt.xlim(xspan)
    plt.ylim(yspan)
    plt.axis('off')
    for a in range(len(XG)):
        plt.plot(XG[a], YG[a], ':w', linewidth=0.5)
    plt.tricontourf(p, q, z[n]-datmin[n]+xmin, levels, cmap=plt.cm.get_cmap('coolwarm'), norm=LogNorm())
    #plt.tricontourf(p, q, z[n], cmap=plt.cm.get_cmap('coolwarm'))
    plt.plot(XS, YS, 'k', linewidth=0.5)
    plt.scatter(X0[n], Y0[n], color='black', s=2.5)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(images[n], transparent=True, dpi=400, bbox_inches='tight', pad_inches=0)
    plt.close()
