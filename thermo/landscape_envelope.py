# -*- coding: utf-8 -*-

# Usage: python landscape_envelope.py

from tqdm import tqdm
from pycalphad import equilibrium
from pycalphad import variables as v
from constants import *

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
z = np.zeros(len(x)*len(y))

# Plot 2nd-order Taylor series approximate free energy landscapes

p = np.zeros(len(x)*len(y))
q = np.zeros(len(x)*len(y))
n = 0
z.fill(0.0)
for j in tqdm(np.nditer(y)):
    for i in np.nditer(x):
        xcr = 1.0 * j / rt3by2
        xnb = 1.0 * i - 0.5 * j / rt3by2
        p[n] = i
        q[n] = j
        z[n] = np.min((PG(xcr, xnb), PD(xcr, xnb), PL(xcr, xnb)))
        n += 1

datmin = np.min(z)
datmax = np.max(z)

levels = np.logspace(np.log2(xmin), np.log2(xmax), num=ncon, base=2.0)
plt.axis('equal')
plt.xlim(xspan)
plt.ylim(yspan)
plt.axis('off')
for a in range(len(XG)):
    plt.plot(XG[a], YG[a], ':w', linewidth=0.5)
plt.tricontourf(p, q, z - datmin + xmin, levels, cmap=plt.cm.get_cmap('coolwarm'), norm=LogNorm())
plt.plot(XS, YS, 'k', linewidth=0.5)
plt.scatter(X0, Y0, color='black', s=2.5)
plt.text(simX(0.010, 0.495), simY(0.495), r'$\gamma$', fontsize=14)
plt.text(simX(0.230, 0.010), simY(0.010), r'$\delta$', fontsize=14)
plt.text(simX(0.340, 0.275), simY(0.275), r'L',        fontsize=14)
plt.margins(0,0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.savefig('../diagrams/landscape_parabola.png', transparent=True, dpi=400, bbox_inches='tight', pad_inches=0)
plt.close()
