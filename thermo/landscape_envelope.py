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

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from CALPHAD_energies import *

from tqdm import tqdm
from pycalphad import equilibrium
from pycalphad import variables as v

# setup global variables

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
plt.savefig( '../diagrams/landscape_parabola.png', transparent=True, dpi=400, bbox_inches='tight', pad_inches=0)
plt.close()
