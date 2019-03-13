#!/usr/bin/python
# -*- coding: utf-8 -*-

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

# Generate ternary phase diagram
# Usage: python phase_diagram.py

from CALPHAD_energies import *
from math import ceil, sqrt
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from tqdm import tqdm

spill = 0.001
density = 150
deltamu = 1.e-12

phases = ["gamma", "delta", "laves"]
labels = [r"$\gamma$", r"$\delta$", "Laves"]
colors = ['red', 'green', 'blue']

# Plot phase diagram
pltsize = 20
plt.figure(figsize=(pltsize, rt3by2*pltsize))
plt.title("Cr-Nb-Ni (Parabolic approx) at %.0f K"%temp, fontsize=18)
plt.xlabel(r'$x_\mathrm{Nb}$', fontsize=24)
plt.ylabel(r'$x_\mathrm{Cr}$', fontsize=24)
plt.plot(XS, YS, '-k')

def GammaDeltaSolver(xCr, xNb):
  def system(X):
    xCr1, xNb1, xCr2, xNb2 = X
    fA = PG(xCr1, xNb1)
    fB = PD(xCr2, xNb2)
    dfAdxCr = PGdxCr(xCr1, xNb1)
    dfAdxNb = PGdxNb(xCr1, xNb1)
    dfBdxCr = PDdxCr(xCr2, xNb2)
    dfBdxNb = PDdxNb(xCr2, xNb2)
    return [dfAdxCr - dfBdxCr,
            dfAdxNb - dfBdxNb,
            fA  + dfAdxCr * (xCr1 - xCr2) + dfAdxNb * (xNb1 - xNb2) - fB,
            (xCr - xCr2) * (xNb1 - xNb2) - (xCr1 - xCr2) * (xNb - xNb2)
           ]

  def jacobian(X):
    xCr1, xNb1, xCr2, xNb2 = X
    dfAdxCr = PGdxCr(xCr1, xNb1)
    dfAdxNb = PGdxNb(xCr1, xNb1)
    dfBdxCr = PDdxCr(xCr2, xNb2)
    dfBdxNb = PDdxNb(xCr2, xNb2)
    dxCr = xCr1 - xCr2
    dxNb = xNb1 - xNb2
    return [[ p_d2Ggam_dxCrCr, p_d2Ggam_dxCrNb,-p_d2Gdel_dxCrCr,-p_d2Gdel_dxCrNb],
            [ p_d2Ggam_dxNbCr, p_d2Ggam_dxNbNb,-p_d2Gdel_dxNbCr,-p_d2Gdel_dxNbNb],
            [ p_d2Ggam_dxCrCr * dxCr + 2*dfAdxCr + p_d2Ggam_dxNbCr * dxNb,
              p_d2Ggam_dxCrNb * dxCr + 2*dfAdxNb + p_d2Ggam_dxNbNb * dxNb,
              -dfBdxCr - dfAdxCr,
              -dfBdxNb - dfAdxNb],
            [-xNb + xNb2, xCr - xCr2, -xNb1 + xNb, -xCr + xCr1]
           ]
  # returns the tuple [xCrGamDel, xNbGamDel, xCrDelGam, xNbDelGam]
  return fsolve(func=system, x0=[xCr, xNb, xCr, xNb], fprime=jacobian)

def GammaLavesSolver(xCr, xNb):
  def system(X):
    xCr1, xNb1, xCr2, xNb2 = X
    fA = PG(xCr1, xNb1)
    fB = PL(xCr2, xNb2)
    dfAdxCr = PGdxCr(xCr1, xNb1)
    dfAdxNb = PGdxNb(xCr1, xNb1)
    dfBdxCr = PLdxCr(xCr2, xNb2)
    dfBdxNb = PLdxNb(xCr2, xNb2)
    return [dfAdxCr - dfBdxCr,
            dfAdxNb - dfBdxNb,
            fA  + dfAdxCr * (xCr1 - xCr2) + dfAdxNb * (xNb1 - xNb2) - fB,
            (xCr - xCr2) * (xNb1 - xNb2) - (xCr1 - xCr2) * (xNb - xNb2)
           ]

  def jacobian(X):
    xCr1, xNb1, xCr2, xNb2 = X
    dfAdxCr = PGdxCr(xCr1, xNb1)
    dfAdxNb = PGdxNb(xCr1, xNb1)
    dfBdxCr = PLdxCr(xCr2, xNb2)
    dfBdxNb = PLdxNb(xCr2, xNb2)
    dxCr = xCr1 - xCr2
    dxNb = xNb1 - xNb2
    return [[ p_d2Ggam_dxCrCr, p_d2Ggam_dxCrNb,-p_d2Glav_dxCrCr,-p_d2Glav_dxCrNb],
            [ p_d2Ggam_dxNbCr, p_d2Ggam_dxNbNb,-p_d2Glav_dxNbCr,-p_d2Glav_dxNbNb],
            [ p_d2Ggam_dxCrCr * dxCr + 2*dfAdxCr + p_d2Ggam_dxNbCr * dxNb,
              p_d2Ggam_dxCrNb * dxCr + 2*dfAdxNb + p_d2Ggam_dxNbNb * dxNb,
              -dfBdxCr - dfAdxCr,
              -dfBdxNb - dfAdxNb],
            [-xNb + xNb2, xCr - xCr2, -xNb1 + xNb, -xCr + xCr1]
           ]
  # returns the tuple [xCrGamLav, xNbGamLav, xCrLavGam, xNbLavGam]
  return fsolve(func=system, x0=[xCr, xNb, xCr, xNb], fprime=jacobian)

def DeltaLavesSolver(xCr, xNb):
  def system(X):
    xCr1, xNb1, xCr2, xNb2 = X
    fA = PD(xCr1, xNb1)
    fB = PL(xCr2, xNb2)
    dfAdxCr = PDdxCr(xCr1, xNb1)
    dfAdxNb = PDdxNb(xCr1, xNb1)
    dfBdxCr = PLdxCr(xCr2, xNb2)
    dfBdxNb = PLdxNb(xCr2, xNb2)
    return [dfAdxCr - dfBdxCr,
            dfAdxNb - dfBdxNb,
            fA  + dfAdxCr * (xCr1 - xCr2) + dfAdxNb * (xNb1 - xNb2) - fB,
            (xCr - xCr2) * (xNb1 - xNb2) - (xCr1 - xCr2) * (xNb - xNb2)
           ]

  def jacobian(X):
    xCr1, xNb1, xCr2, xNb2 = X
    dfAdxCr = PDdxCr(xCr1, xNb1)
    dfAdxNb = PDdxNb(xCr1, xNb1)
    dfBdxCr = PLdxCr(xCr2, xNb2)
    dfBdxNb = PLdxNb(xCr2, xNb2)
    dxCr = xCr1 - xCr2
    dxNb = xNb1 - xNb2
    return [[ p_d2Gdel_dxCrCr, p_d2Gdel_dxCrNb,-p_d2Glav_dxCrCr,-p_d2Glav_dxCrNb],
            [ p_d2Gdel_dxNbCr, p_d2Gdel_dxNbNb,-p_d2Glav_dxNbCr,-p_d2Glav_dxNbNb],
            [ p_d2Gdel_dxCrCr * dxCr + 2*dfAdxCr + p_d2Gdel_dxNbCr * dxNb,
              p_d2Gdel_dxCrNb * dxCr + 2*dfAdxNb + p_d2Gdel_dxNbNb * dxNb,
              -dfBdxCr - dfAdxCr,
              -dfBdxNb - dfAdxNb],
            [-xNb + xNb2, xCr - xCr2, -xNb1 + xNb, -xCr + xCr1]
           ]
  # returns the tuple [xCrDelLav, xNbDelLav, xCrLavDel, xNbLavDel]
  return fsolve(func=system, x0=[xCr, xNb, xCr, xNb], fprime=jacobian)


pureGamma = []
pureDelta = []
pureLaves = []
tieGamDel = []
tieGamLav = []
tieDelLav = []

for xNbTest in tqdm(np.linspace(spill, 1 - spill, density)):
  for xCrTest in np.linspace(spill, 1 - spill - xNbTest, max(1, ceil((1 - spill - xNbTest) * density))):

      xCrGamDel, xNbGamDel, xCrDelGam, xNbDelGam = GammaDeltaSolver(xCrTest, xNbTest)
      xCrGamLav, xNbGamLav, xCrLavGam, xNbLavGam = GammaLavesSolver(xCrTest, xNbTest)
      xCrDelLav, xNbDelLav, xCrLavDel, xNbLavDel = DeltaLavesSolver(xCrTest, xNbTest)

      fGam = PG(xCrTest, xNbTest)
      fDel = PD(xCrTest, xNbTest)
      fLav = PL(xCrTest, xNbTest)
      fGamDel = 1.0e12
      fGamLav = 1.0e12
      fDelLav = 1.0e12

      # Filter unphysical results
      GamDelIsPhysical = (xCrGamDel > 0 and xCrGamDel < 0.25 and
                          xNbGamDel > 0 and xNbGamDel < 0.20 and
                          xCrGamDel + xNbGamDel       < 1 and
                          xCrDelGam > 0 and xCrDelGam < 1 and
                          xNbDelGam > 0 and xNbDelGam < 1 and
                          xCrDelGam + xNbDelGam       < 1)
      GamLavIsPhysical = (xCrGamLav > 0 and xCrGamLav < 1 and
                          xNbGamLav > 0 and xNbGamLav < 0.25 and
                          xCrGamLav + xNbGamLav       < 1 and
                          xCrLavGam > 0 and xCrLavGam < 1 and
                          xNbLavGam > 0 and xNbLavGam < 1 and
                          xCrLavGam + xNbLavGam       < 1)
      DelLavIsPhysical = (xCrDelLav > 0 and xCrDelLav < 1 and
                          xNbDelLav > 0 and xNbDelLav < 1 and
                          xCrDelLav + xNbDelLav       < 1 and
                          xCrLavDel > 0 and xCrLavDel < 1 and
                          xNbLavDel > 0 and xNbLavDel < 1 and
                          xCrLavDel + xNbLavDel       < 1)

      # Filter mismatched slopes
      GamDelCrSlopeSimilar = (PGdxCr(xCrGamDel, xNbGamDel) - PDdxCr(xCrDelGam, xNbDelGam) < deltamu)
      GamDelNbSlopeSimilar = (PGdxNb(xNbGamDel, xNbGamDel) - PDdxNb(xNbDelGam, xNbDelGam) < deltamu)
      GamLavCrSlopeSimilar = (PGdxCr(xCrGamLav, xNbGamLav) - PLdxCr(xCrLavGam, xNbLavGam) < deltamu)
      GamLavNbSlopeSimilar = (PGdxNb(xNbGamLav, xNbGamLav) - PLdxNb(xNbLavGam, xNbLavGam) < deltamu)
      DelLavCrSlopeSimilar = (PDdxCr(xCrDelLav, xNbDelLav) - PLdxCr(xCrLavDel, xNbLavDel) < 8*deltamu)
      DelLavNbSlopeSimilar = (PDdxNb(xNbDelLav, xNbDelLav) - PLdxNb(xNbLavDel, xNbLavDel) < 8*deltamu)

      if (GamDelIsPhysical and GamDelCrSlopeSimilar and GamDelNbSlopeSimilar):
        fGamDel = PG(xCrGamDel, xNbGamDel) + PGdxCr(xCrGamDel, xNbGamDel) * (xCrGamDel - xCrTest) \
                                           + PGdxNb(xCrGamDel, xNbGamDel) * (xNbGamDel - xNbTest)
      if (GamLavIsPhysical and GamLavCrSlopeSimilar and GamLavNbSlopeSimilar):
        fGamLav = PG(xCrGamLav, xNbGamLav) + PGdxCr(xCrGamLav, xNbGamLav) * (xCrGamLav - xCrTest) \
                                           + PGdxNb(xCrGamLav, xNbGamLav) * (xNbGamLav - xNbTest)
      if (DelLavIsPhysical and DelLavCrSlopeSimilar and DelLavNbSlopeSimilar):
        fDelLav = PD(xCrLavDel, xNbLavDel) + PDdxCr(xCrDelLav, xNbDelLav) * (xCrDelLav - xCrTest) \
                                           + PDdxNb(xCrDelLav, xNbDelLav) * (xNbDelLav - xNbTest)

      minima = np.asarray([fGam, fDel, fLav, fGamDel, fGamLav, fDelLav])
      minidx = np.argmin(minima)

      if (minidx == 0):
        pureGamma.append([simX(xNbTest, xCrTest), simY(xCrTest)])
      elif (minidx == 1):
        pureDelta.append([simX(xNbTest, xCrTest), simY(xCrTest)])
      elif (minidx == 2):
        pureLaves.append([simX(xNbTest, xCrTest), simY(xCrTest)])
      elif (minidx == 3):
        tieGamDel.append([simX(xNbGamDel, xCrGamDel), simY(xCrGamDel),
                          simX(xNbDelGam, xCrDelGam), simY(xCrDelGam)])
      elif (minidx == 4):
        tieGamLav.append([simX(xNbGamLav, xCrGamLav), simY(xCrGamLav),
                          simX(xNbLavGam, xCrLavGam), simY(xCrLavGam)])
      elif (minidx == 5):
        tieDelLav.append([simX(xNbDelLav, xCrDelLav), simY(xCrDelLav),
                          simX(xNbLavDel, xCrLavDel), simY(xCrLavDel)])
"""
for x, y in pureGamma:
  plt.scatter(x, y, c=colors[0], edgecolor=colors[0], s=1)
for x, y in pureDelta:
  plt.scatter(x, y, c=colors[1], edgecolor=colors[1], s=1)
for x, y in pureLaves:
  plt.scatter(x, y, c=colors[2], edgecolor=colors[2], s=1)
"""

for xa, ya, xb, yb in tieGamDel:
  plt.scatter(xa, ya, c=colors[0], edgecolor=colors[0], s=2)
  plt.scatter(xb, yb, c=colors[1], edgecolor=colors[1], s=2)
  plt.plot([xa, xb], [ya, yb], color="gray", linewidth=0.5)
for xa, ya, xb, yb in tieGamLav:
  plt.scatter(xa, ya, c=colors[0], edgecolor=colors[0], s=2)
  plt.scatter(xb, yb, c=colors[2], edgecolor=colors[2], s=2)
  plt.plot([xa, xb], [ya, yb], color="gray", linewidth=0.5)
for xa, ya, xb, yb in tieDelLav:
  plt.scatter(xa, ya, c=colors[1], edgecolor=colors[1], s=2)
  plt.scatter(xb, yb, c=colors[2], edgecolor=colors[2], s=2)
  plt.plot([xa, xb], [ya, yb], color="gray", linewidth=0.5)

plt.savefig("ternary-diagram.png", dpi=400, bbox_inches="tight")
