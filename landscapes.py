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

Titles = (r'$\gamma$', r'$\delta$', r'$\mu$', r'Laves')
npts = 76
nfun = 4
#levels = 100
#levels = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10, 100, 1000, 10000, 1e5, 1.0e10, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12]
span = (-0.001, 1.001)
yspan = (-0.001, rt3by2 + 0.001)
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

p = np.zeros(len(x)*len(y))
q = np.zeros(len(x)*len(y))


# Plot CALPHAD free energies using extracted equations

n = 0
z.fill(0.0)
for j in tqdm(np.nditer(y)):
    for i in np.nditer(x):
        xcr = 1.0*j / rt3by2
        xnb = 1.0*i - 0.5 * j / rt3by2
        xni = 1.0 - xcr - xnb
        p[n] = i
        q[n] = j
        z[0][n] = GG(xcr, xnb) #c_gamma.subs({GAMMA_XCR: xcr, GAMMA_XNB: xnb, GAMMA_XNI: xni}) #GG(xcr, xnb)
        z[1][n] = GD(xcr, xnb) #c_delta.subs({DELTA_XCR: xcr, DELTA_XNB: xnb, DELTA_XNI: xni}) #GD(xcr, xnb)
        z[2][n] = GU(xcr, xnb) #c_mu.subs({MU_XCR: xcr, MU_XNB: xnb, MU_XNI: xni}) #GU(xcr, xnb)
        z[3][n] = GL(xcr, xnb) #c_laves.subs({LAVES_XCR: xcr, LAVES_XNB: xnb, LAVES_XNI: xni}) #GL(xcr, xnb)
        n += 1

datmin = 1.0e10
datmax = -1.0e10
for n in range(len(z)):
    mymin = np.min(z[n])
    mymax = np.max(z[n])
    datmin = min(datmin, mymin)
    datmax = max(datmax, mymax)

datmin = float(datmin)
datmax = float(datmax)

print "Data spans [%.4g, %.4g]" % (datmin, datmax)

levels = np.logspace(np.log2(datmin-1.01*datmin), np.log2(datmax-1.01*datmin), num=50, base=2.0)

f, axarr = plt.subplots(nrows=2, ncols=2, sharex='col', sharey='row')
f.suptitle("IN625 Ternary Potentials (Restricted)",fontsize=14)
n=0
for ax in axarr.reshape(-1):
    ax.set_title(Titles[n],fontsize=10)
    ax.axis('equal')
    ax.set_xlim(span)
    ax.set_ylim(yspan)
    ax.axis('off')
    for a in range(len(XG)):
        ax.plot(XG[a], YG[a], ':w', linewidth=0.5)
    ax.tricontourf(p, q, z[n]-1.01*datmin, levels, cmap=plt.cm.get_cmap('coolwarm'), norm=LogNorm())
    #ax.tricontourf(p, q, z[n], cmap=plt.cm.get_cmap('coolwarm'))
    ax.plot(XS, YS, 'k', linewidth=0.5)
    ax.scatter(X0[n], Y0[n], color='black', s=2.5)
    n+=1
plt.figtext(x=0.5, y=0.0625, ha='center', fontsize=8, \
            s=r'White triangles enclose Gibbs simplex, $x_{\mathrm{Cr}}+x_{\mathrm{Nb}}+x_{\mathrm{Ni}}=1$.')
f.savefig('ternary.png', dpi=400, bbox_inches='tight')
plt.close()

files = ['diagrams/gamma_parabola.png', 'diagrams/delta_parabola.png', 'diagrams/mu_parabola.png', 'diagrams/Laves_parabola.png']
for n in range(len(z)):
    plt.axis('equal')
    plt.xlim(span)
    plt.ylim(yspan)
    plt.axis('off')
    for a in range(len(XG)):
        plt.plot(XG[a], YG[a], ':w', linewidth=0.5)
    plt.tricontourf(p, q, z[n]-1.01*datmin, levels, cmap=plt.cm.get_cmap('coolwarm'), norm=LogNorm())
    #plt.tricontourf(p, q, z[n], cmap=plt.cm.get_cmap('coolwarm'))
    plt.plot(XS, YS, 'k', linewidth=0.5)
    plt.scatter(X0[n], Y0[n], color='black', s=2.5)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(files[n], transparent=True, dpi=400, bbox_inches='tight', pad_inches=0)
    plt.close()



# Plot Taylor series approximate free energy landscapes

n = 0
z.fill(0.0)
for j in tqdm(np.nditer(y)):
    for i in np.nditer(x):
        xcr = 1.0*j / rt3by2
        xnb = 1.0*i - 0.5 * j / rt3by2
        xni = 1.0 - xcr - xnb
        p[n] = i
        q[n] = j
        z[0][n] = TG(xcr, xnb)
        z[1][n] = TD(xcr, xnb)
        z[2][n] = TU(xcr, xnb)
        z[3][n] = TL(xcr, xnb)
        n += 1

datmin = 1.0e10
datmax = -1.0e10
for n in range(len(z)):
    mymin = np.min(z[n])
    mymax = np.max(z[n])
    datmin = min(datmin, mymin)
    datmax = max(datmax, mymax)

print "Data spans [%.4g, %.4g]" % (datmin, datmax)

levels = np.logspace(np.log2(datmin-1.01*datmin), np.log2(datmax-1.01*datmin), num=50, base=2.0)

f, axarr = plt.subplots(nrows=2, ncols=2, sharex='col', sharey='row')
f.suptitle("IN625 Ternary Potentials (Taylor)",fontsize=14)
n=0
for ax in axarr.reshape(-1):
    ax.set_title(Titles[n],fontsize=10)
    ax.axis('equal')
    ax.set_xlim(span)
    ax.set_ylim(yspan)
    ax.axis('off')
    for a in range(len(XG)):
        ax.plot(XG[a], YG[a], ':w', linewidth=0.5)
    ax.tricontourf(p, q, z[n]-1.01*datmin, levels, cmap=plt.cm.get_cmap('coolwarm'), norm=LogNorm())
    #ax.tricontourf(p, q, z[n], cmap=plt.cm.get_cmap('coolwarm'))
    ax.plot(XS, YS, 'k', linewidth=0.5)
    ax.scatter(X0[n], Y0[n], color='black', s=2.5)
    n+=1
plt.figtext(x=0.5, y=0.0625, ha='center', fontsize=8, \
            s=r'White triangles enclose Gibbs simplex, $x_{\mathrm{Cr}}+x_{\mathrm{Nb}}+x_{\mathrm{Ni}}=1$.')
f.savefig('ternary_taylor.png', dpi=400, bbox_inches='tight')
plt.close()

files = ['diagrams/gamma_taylor.png', 'diagrams/delta_taylor.png', 'diagrams/mu_taylor.png', 'diagrams/Laves_taylor.png']
for n in range(len(z)):
    plt.axis('equal')
    plt.xlim(span)
    plt.ylim(yspan)
    plt.axis('off')
    for a in range(len(XG)):
        plt.plot(XG[a], YG[a], ':w', linewidth=0.5)
    plt.tricontourf(p, q, z[n]-1.01*datmin, levels, cmap=plt.cm.get_cmap('coolwarm'), norm=LogNorm())
    #plt.tricontourf(p, q, z[n], cmap=plt.cm.get_cmap('coolwarm'))
    plt.plot(XS, YS, 'k', linewidth=0.5)
    plt.scatter(X0[n], Y0[n], color='black', s=2.5)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(files[n], transparent=True, dpi=400, bbox_inches='tight', pad_inches=0)
    plt.close()






"""
# Plot ternary free energy landscapes from CALPHAD
XX  = [[], [], [], []]
YY  = [[], [], [], []]
AllG  = [[], [], [], []]
XCr = [[], [], [], []]
XNb = [[], [], [], []]
G   = [[], [], [], []]


fulldb = Database('Du_Cr-Nb-Ni.tdb')
elements = ['CR', 'NB', 'NI', 'VA']

gamma_data = calculate(fulldb, elements, 'FCC_A1', T=temp)
XCr[0] = np.ravel(gamma_data.X.sel(component='CR'))
XNb[0] = np.ravel(gamma_data.X.sel(component='NB'))
G[0]   = np.ravel(gamma_data.GM)

delta_data = calculate(fulldb, elements, 'D0A_NBNI3', T=temp)
XCr[1] = np.ravel(delta_data.X.sel(component='CR'))
XNb[1] = np.ravel(delta_data.X.sel(component='NB'))
G[1]   = np.ravel(delta_data.GM)

mu_data = calculate(fulldb, elements, 'D85_NI7NB6', T=temp)
XCr[2] = np.ravel(mu_data.X.sel(component='CR'))
XNb[2] = np.ravel(mu_data.X.sel(component='NB'))
G[2]   = np.ravel(mu_data.GM)

laves_data = calculate(fulldb, elements, 'C14_LAVES', T=temp)
XCr[3] = np.ravel(laves_data.X.sel(component='CR'))
XNb[3] = np.ravel(laves_data.X.sel(component='NB'))
G[3]   = np.ravel(laves_data.GM)

for n in range(4):
    for i in range(len(XCr[n])):
        XX[n].append(simX(XNb[n][i], XCr[n][i]))
        YY[n].append(simY(XCr[n][i]))

for i in range(len(G[0])):
    AllG[0].append(G[0][i])

for i in range(len(G[1])):
    AllG[1].append(G[1][i])

for i in range(len(G[2])):
    AllG[2].append(G[2][i])

for i in range(len(G[3])):
    AllG[3].append(G[3][i])


datmin = 1.0e10
datmax = -1.0e10
for n in range(len(AllG)):
    mymin = np.min(AllG[n])
    mymax = np.max(AllG[n])
    datmin = min(datmin, mymin)
    datmax = max(datmax, mymax)

print "Data spans [%.4g, %.4g]" % (datmin, datmax)

#levels = np.logspace(np.log2(datmin-1.0001*datmin), np.log2(datmax-1.01*datmin), num=50, base=2.0)
#levels = np.linspace(datmin-1.01*datmin, datmax-1.01*datmin, 50)
#levels = np.linspace(1.001*float(datmin), 1.001*float(datmax), num=50, dtype=float)

files = ['diagrams/gamma_calphad.png', 'diagrams/delta_calphad.png', 'diagrams/mu_calphad.png', 'diagrams/Laves_calphad.png']
for n in range(len(G)):
    plt.axis('equal')
    plt.xlim(span)
    plt.ylim(yspan)
    plt.axis('off')
    for a in range(len(XG)):
        plt.plot(XG[a], YG[a], ':w', linewidth=0.5)
    #offset = [1.01*datmin] * len(AllG[n])
    #plt.tricontourf(XX[n], YY[n], G[n]-1.0001*datmin, levels, cmap=plt.cm.get_cmap('coolwarm'), norm=LogNorm())
    #plt.tricontourf(XX[n], YY[n], AllG[n]-offset, levels, cmap=plt.cm.get_cmap('coolwarm'))
    #plt.tricontourf(XX[n], YY[n], AllG[n], levels=levels, cmap=plt.cm.get_cmap('coolwarm'))
    plt.tricontourf(XX[n], YY[n], AllG[n], cmap=plt.cm.get_cmap('coolwarm'))
    plt.plot(XS, YS, 'k', linewidth=0.5)
    plt.scatter(X0[n], Y0[n], color='black', s=2.5)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(files[n], transparent=True, dpi=400, bbox_inches='tight', pad_inches=0)
    plt.close()



f, axarr = plt.subplots(nrows=2, ncols=2, sharex='col', sharey='row')
f.suptitle("IN625 Ternary Potentials (CALPHAD)",fontsize=14)
n=0
for ax in axarr.reshape(-1):
    ax.set_title(Titles[n],fontsize=10)
    ax.axis('equal')
    ax.set_xlim(span)
    ax.set_ylim(yspan)
    ax.axis('off')
    for a in range(len(XG)):
        ax.plot(XG[a], YG[a], ':w', linewidth=0.5)
    #offset = [1.01*datmin] * len(AllG[n])
    #ax.tricontourf(XX[n], YY[n], AllG[n]-1.0001*datmin, levels, cmap=plt.cm.get_cmap('coolwarm'), norm=LogNorm())
    #ax.tricontourf(XX[n], YY[n], AllG[n]-offset, levels, cmap=plt.cm.get_cmap('coolwarm'))
    #ax.tricontourf(XX[n], YY[n], AllG[n], levels=levels, cmap=plt.cm.get_cmap('coolwarm'))
    ax.tricontourf(XX[n], YY[n], AllG[n], cmap=plt.cm.get_cmap('coolwarm'))
    ax.plot(XS, YS, 'k', linewidth=0.5)
    #ax.scatter(X0[n], Y0[n], color='black', s=2.5)
    n+=1

plt.figtext(x=0.5, y=0.0625, ha='center', fontsize=8, \
            s=r'White triangles enclose Gibbs simplex, $x_{\mathrm{Cr}}+x_{\mathrm{Nb}}+x_{\mathrm{Ni}}=1$.')
f.savefig('ternary_calphad.png', dpi=400, bbox_inches='tight')
plt.close()

"""



#fulldb = Database('Du_Cr-Nb-Ni.tdb')
#elements = ['CR', 'NB', 'NI']

p = np.zeros(len(x)*len(y))
q = np.zeros(len(x)*len(y))

# Plot CALPHAD free energy landscape using pycalphad
n = 0
z.fill(0.0)
for j in tqdm(np.nditer(y)):
    for i in np.nditer(x):
        xcr = 1.0*j / rt3by2
        xnb = 1.0*i - 0.5 * j / rt3by2
        xsum = xcr + xnb
        xni = 1.0 - xcr - xnb
        p[n] = i
        q[n] = j
        if xsum < 1 and xcr > 0 and xnb > 0 and xni > 0:
            g_nrg = inVm * float(equilibrium(tdb, elements, 'FCC_A1',     {T:temp, v.P:101325, v.X('CR'):xcr, v.X('NB'):xnb}, output='GM').GM)
            z[0][n] = g_nrg
        if xsum < 1 and xcr > 0 and xnb > 0 and xni > 0 and xnb < 0.25 and xcr < 0.75:
            d_nrg = inVm * float(equilibrium(tdb, elements, 'D0A_NBNI3',  {T:temp, v.P:101325, v.X('CR'):xcr, v.X('NB'):xnb}, output='GM').GM)
            z[1][n] = d_nrg
        if xsum < 1 and xcr > 0 and xnb > 0 and xni > 0 and xnb > fr6by13:
            m_nrg = inVm * float(equilibrium(tdb, elements, 'D85_NI7NB6', {T:temp, v.P:101325, v.X('CR'):xcr, v.X('NB'):xnb}, output='GM').GM)
            z[2][n] = m_nrg
        if xsum < 1 and xcr > 0 and xnb > 0 and xni > 0 and xnb < fr1by3 and xni < fr2by3:
            l_nrg = inVm * float(equilibrium(tdb, elements, 'C14_LAVES',  {T:temp, v.P:101325, v.X('CR'):xcr, v.X('NB'):xnb}, output='GM').GM)
            z[3][n] = l_nrg
        n += 1

datmin = 1.0e10
datmax = -1.0e10
for n in range(len(z)):
    mymin = np.min(z[n])
    mymax = np.max(z[n])
    datmin = min(datmin, mymin)
    datmax = max(datmax, mymax)

print "Data spans [%.4g, %.4g]" % (datmin, datmax)

levels = np.logspace(np.log2(datmin-1.01*datmin), np.log2(datmax-1.01*datmin), num=50, base=2.0)

f, axarr = plt.subplots(nrows=2, ncols=2, sharex='col', sharey='row')
f.suptitle("IN625 Ternary Potentials (CALPHAD)",fontsize=14)
n=0
for ax in axarr.reshape(-1):
    ax.set_title(Titles[n],fontsize=10)
    ax.axis('equal')
    ax.set_xlim(span)
    ax.set_ylim(yspan)
    ax.axis('off')
    for a in range(len(XG)):
        ax.plot(XG[a], YG[a], ':w', linewidth=0.5)
    ax.tricontourf(p, q, z[n]-1.01*datmin, levels, cmap=plt.cm.get_cmap('coolwarm'), norm=LogNorm())
    #ax.tricontourf(p, q, z[n], cmap=plt.cm.get_cmap('coolwarm'))
    ax.plot(XS, YS, 'k', linewidth=0.5)
    ax.scatter(X0[n], Y0[n], color='black', s=2.5)
    n+=1
plt.figtext(x=0.5, y=0.0625, ha='center', fontsize=8, \
            s=r'White triangles enclose Gibbs simplex, $x_{\mathrm{Cr}}+x_{\mathrm{Nb}}+x_{\mathrm{Ni}}=1$.')
f.savefig('ternary_calphad.png', dpi=400, bbox_inches='tight')
plt.close()

files = ['diagrams/gamma_CALPHAD.png', 'diagrams/delta_CALPHAD.png', 'diagrams/mu_CALPHAD.png', 'diagrams/Laves_CALPHAD.png']
for n in range(len(z)):
    plt.axis('equal')
    plt.xlim(span)
    plt.ylim(yspan)
    plt.axis('off')
    for a in range(len(XG)):
        plt.plot(XG[a], YG[a], ':w', linewidth=0.5)
    plt.tricontourf(p, q, z[n]-1.01*datmin, levels, cmap=plt.cm.get_cmap('coolwarm'), norm=LogNorm())
    #plt.tricontourf(p, q, z[n], cmap=plt.cm.get_cmap('coolwarm'))
    plt.plot(XS, YS, 'k', linewidth=0.5)
    plt.scatter(X0[n], Y0[n], color='black', s=2.5)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(files[n], transparent=True, dpi=400, bbox_inches='tight', pad_inches=0)
    plt.close()
