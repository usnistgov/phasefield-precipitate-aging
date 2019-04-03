# -*- coding:utf-8 -*-
import numpy as np

print("AM IN625, per ASTM F3056")

atoms = ('C'  , 'Mn' , 'Si' , 'P'   , 'S'   , 'Cr', 'Co' , 'Mo' , 'Nb' , 'Ti' , 'Al' , 'Fe' , 'Ni'  )
minWt = ( 0   ,  0   ,  0   ,  0    ,  0    ,  20.,  0   ,  8.  ,  3.15,  0   ,  0   ,  0   ,  68.85)
maxWt = ( 0.1 ,  0.5 ,  0.5 ,  0.015,  0.015,  23.,  1   , 10.  ,  4.15,  0.4 ,  0.4 ,  5.  ,  54.92)
frmWt = (12.01, 54.94, 28.08, 30.97 , 32.06 ,  52., 58.93, 95.94, 92.91, 47.87, 26.98, 55.84,  58.69)

minAt = np.divide(minWt, frmWt)
maxAt = np.divide(maxWt, frmWt)

minN = np.sum(minAt)
maxN = np.sum(maxAt)

for i in range(len(atoms)):
    print("{0:2s}: {1:.5f}, {2:.5f}".format(atoms[i], minAt[i]/minN, maxAt[i]/maxN))

print("\nTernary (Cr+Mo)-Nb-Ni Analogue")

atoms = ('Cr', 'Mo' , 'Nb' , 'Ni' )
minWt = ( 20.,  8.  ,  3.15, 68.85)
maxWt = ( 23., 10.  ,  4.15, 62.85)
frmWt = ( 52., 95.94, 92.91, 58.69)

minAt = np.divide(minWt, frmWt)
maxAt = np.divide(maxWt, frmWt)

minN = np.sum(minAt)
maxN = np.sum(maxAt)

for i in range(len(atoms)):
    print("{0:2s}: {1:.5f}, {2:.5f}".format(atoms[i], minAt[i]/minN, maxAt[i]/maxN))

print("\nDICTRA Solidification")

atoms = ('C'  , 'Cr', 'Fe' , 'Mo' , 'Nb' , 'Ni' )
wtPct = ( 0.13, 13.6,  0.35, 13.9 , 23.5 , 48.52)
frmWt = (12.01, 52.0, 55.84, 95.94, 92.91, 58.69)

molFr = np.divide(wtPct, frmWt)

N = np.sum(molFr)

for i in range(len(atoms)):
    print("{0:2s}: {1:.5f}".format(atoms[i], molFr[i]/N))

print("\nEnriched (Cr+Mo)-Nb-Ni Analogue")

atoms = ('Cr', 'Mo' , 'Nb' , 'Ni' )
wtPct = (13.6, 13.9 , 23.5 , 49.  )
frmWt = (52. , 95.94, 92.91, 58.69)

molFr = np.divide(wtPct, frmWt)

N = np.sum(molFr)

for i in range(len(atoms)):
    print("{0:2s}: {1:.5f}".format(atoms[i], molFr[i]/N))
