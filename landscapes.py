#!/usr/bin/python

from CALPHAD_energies import *

from tqdm import tqdm

# Plot ternary free energy landscapes
Titles = (r'$\gamma$', r'$\delta$', r'$\mu$', r'Laves')
npts = 75
nfun = 4
#levels = 100
#levels = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10, 100, 1000, 10000, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12]
#span = (-0.05, 1.05)
#yspan = (-0.15, 0.95)
span = (-0.01, 1.01)
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

p = np.zeros(len(x)*len(y))
q = np.zeros(len(x)*len(y))

n = 0
for j in tqdm(np.nditer(y)):
    for i in np.nditer(x):
        xcr = 1.0*j / rt3by2
        xnb = 1.0*i - 0.5 * j / rt3by2
        xni = 1.0 - xcr - xnb
        p[n] = i
        q[n] = j
        z[0][n] = GG(xcr, xnb)
        z[1][n] = GD(xcr, xnb)
        z[2][n] = GU(xcr, xnb)
        z[3][n] = GL(xcr, xnb)
        n += 1

datmin = 1e13
datmax = 1e-13
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
        ax.plot(XG[a], YG[a], ':w')
    ax.tricontourf(p, q, z[n]-1.01*datmin, levels, cmap=plt.cm.get_cmap('coolwarm'), norm=LogNorm())
    ax.plot(XS, YS, ':w')
    ax.scatter(X0[n], Y0[n], color='black', s=2.5)
    n+=1
plt.figtext(x=0.5, y=0.0625, ha='center', fontsize=8, \
            s=r'White triangles enclose Gibbs simplex, $x_{\mathrm{Cr}}+x_{\mathrm{Nb}}+x_{\mathrm{Ni}}=1$.')
f.savefig('ternary.png', dpi=600, bbox_inches='tight')
plt.close()

files = ['diagrams/gamma_parabola.png', 'diagrams/delta_parabola.png', 'diagrams/mu_parabola.png', 'diagrams/Laves_parabola.png']
for n in range(len(z)):
    plt.axis('equal')
    plt.xlim(span)
    plt.ylim(yspan)
    plt.axis('off')
    for a in range(len(XG)):
        plt.plot(XG[a], YG[a], ':w')
    plt.tricontourf(p, q, z[n]-1.01*datmin, levels, cmap=plt.cm.get_cmap('coolwarm'), norm=LogNorm())
    plt.plot(XS, YS, ':w')
    plt.scatter(X0[n], Y0[n], color='black', s=2.5)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(files[n], transparent=True, dpi=600, bbox_inches='tight', pad_inches=0)
    plt.close()



# Plot ternary free energy landscapes
Titles = (r'$\gamma$', r'$\delta$', r'$\mu$', r'Laves')
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

n = 0
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

datmin = 1e13
datmax = 1e-13
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
        ax.plot(XG[a], YG[a], ':w')
    ax.tricontourf(p, q, z[n]-1.01*datmin, levels, cmap=plt.cm.get_cmap('coolwarm'), norm=LogNorm())
    ax.plot(XS, YS, ':w')
    ax.scatter(X0[n], Y0[n], color='black', s=2.5)
    n+=1
plt.figtext(x=0.5, y=0.0625, ha='center', fontsize=8, \
            s=r'White triangles enclose Gibbs simplex, $x_{\mathrm{Cr}}+x_{\mathrm{Nb}}+x_{\mathrm{Ni}}=1$.')
f.savefig('ternary_taylor.png', dpi=600, bbox_inches='tight')
plt.close()

files = ['diagrams/gamma_taylor.png', 'diagrams/delta_taylor.png', 'diagrams/mu_taylor.png', 'diagrams/Laves_taylor.png']
for n in range(len(z)):
    plt.axis('equal')
    plt.xlim(span)
    plt.ylim(yspan)
    plt.axis('off')
    for a in range(len(XG)):
        plt.plot(XG[a], YG[a], ':w')
    plt.tricontourf(p, q, z[n]-1.01*datmin, levels, cmap=plt.cm.get_cmap('coolwarm'), norm=LogNorm())
    plt.plot(XS, YS, ':w')
    plt.scatter(X0[n], Y0[n], color='black', s=2.5)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(files[n], transparent=True, dpi=600, bbox_inches='tight', pad_inches=0)
    plt.close()

'''

span = (-0.01, 1.01)
yspan = (-0.1, 0.9)

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


# Plot ternary free energy landscapes from CALPHAD
Titles = (r'$\gamma$', r'$\delta$', r'$\mu$', r'Laves')

XX  = [[], [], [], []]
YY  = [[], [], [], []]
AllG  = [[], [], [], []]
XCr = [[], [], [], []]
XNb = [[], [], [], []]
G   = [[], [], [], []]


gamma_data = calculate(tdb, elements, 'FCC_A1', T=temp, output='GM')
XCr[0] = np.ravel(gamma_data.X.sel(component='CR'))
XNb[0] = np.ravel(gamma_data.X.sel(component='NB'))
G[0]   = np.ravel(gamma_data.GM)

delta_data = calculate(tdb, elements, 'D0A_NBNI3', T=temp, output='GM')
XCr[1] = np.ravel(delta_data.X.sel(component='CR'))
XNb[1] = np.ravel(delta_data.X.sel(component='NB'))
G[1]   = np.ravel(delta_data.GM)

mu_data = calculate(tdb, elements, 'D85_NI7NB6', T=temp, output='GM')
XCr[2] = np.ravel(mu_data.X.sel(component='CR'))
XNb[2] = np.ravel(mu_data.X.sel(component='NB'))
G[2]   = np.ravel(mu_data.GM)

laves_data = calculate(tdb, elements, 'C15_LAVES', T=temp, output='GM')
XCr[3] = np.ravel(laves_data.X.sel(component='CR'))
XNb[3] = np.ravel(laves_data.X.sel(component='NB'))
G[3]   = np.ravel(laves_data.GM)

for n in range(4):
    for i in range(len(XCr[n])):
        XX[n].append(simX(XNb[n][i], XCr[n][i]))
        YY[n].append(simY(XCr[n][i]))

for i in range(len(G[0])):
    AllG[0].append(G[0][i] - TG(xcr, xnb))

for i in range(len(G[1])):
    AllG[1].append(G[1][i] - TG(xcr, xnb))

for i in range(len(G[2])):
    AllG[2].append(G[2][i] - TU(xcr, xnb))

for i in range(len(G[3])):
    AllG[3].append(G[3][i] - TL(xcr, xnb))


#datmin = 1.0e8
#datmax = -1.0e8
#for n in range(len(AllG)):
#    mymin = np.min(AllG[n])
#    mymax = np.max(AllG[n])
#    datmin = min(datmin, mymin)
#    datmax = max(datmax, mymax)
#
#print "Data spans [%.4g, %.4g]" % (datmin, datmax)

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
        plt.plot(XG[a], YG[a], ':w')
    #offset = [1.01*datmin] * len(AllG[n])
    #plt.tricontourf(XX[n], YY[n], G[n]-1.01*np.min(G[n]), levels, cmap=plt.cm.get_cmap('coolwarm'), norm=LogNorm())
    #plt.tricontourf(XX[n], YY[n], AllG[n]-offset, levels, cmap=plt.cm.get_cmap('coolwarm'))
    #plt.tricontourf(XX[n], YY[n], AllG[n], levels=levels, cmap=plt.cm.get_cmap('coolwarm'))
    plt.tricontourf(XX[n], YY[n], AllG[n], cmap=plt.cm.get_cmap('coolwarm'))
    plt.plot(XS, YS, ':w')
    plt.scatter(X0[n], Y0[n], color='black', s=2.5)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(files[n], transparent=True, dpi=600, bbox_inches='tight', pad_inches=0)
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
        ax.plot(XG[a], YG[a], ':w')
    #offset = [1.01*datmin] * len(AllG[n])
    #ax.tricontourf(XX[n], YY[n], AllG[n]-1.01*np.min(AllG[n]), levels, cmap=plt.cm.get_cmap('coolwarm'), norm=LogNorm())
    #ax.tricontourf(XX[n], YY[n], AllG[n]-offset, levels, cmap=plt.cm.get_cmap('coolwarm'))
    #ax.tricontourf(XX[n], YY[n], AllG[n], levels=levels, cmap=plt.cm.get_cmap('coolwarm'))
    ax.tricontourf(XX[n], YY[n], AllG[n], cmap=plt.cm.get_cmap('coolwarm'))
    ax.plot(XS, YS, ':w')
    #ax.scatter(X0[n], Y0[n], color='black', s=2.5)
    n+=1

plt.figtext(x=0.5, y=0.0625, ha='center', fontsize=8, \
            s=r'White triangles enclose Gibbs simplex, $x_{\mathrm{Cr}}+x_{\mathrm{Nb}}+x_{\mathrm{Ni}}=1$.')
f.savefig('ternary_calphad.png', dpi=600, bbox_inches='tight')
plt.close()

'''
