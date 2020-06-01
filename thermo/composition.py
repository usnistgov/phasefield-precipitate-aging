# -*- coding:utf-8 -*-

import numpy as np

wmin = {"C": 0,
        "Mn": 0,
        "Si": 0,
        "P": 0,
        "S": 0,
        "Cr": 20,
        "Co": 0,
        "Mo": 8,
        "Nb": 3.15,
        "Ti": 0,
        "Al": 0,
        "Fe": 0,
        "Ni": 68.85
}

wmax = {"C": 0.1,
        "Mn": 0.5,
        "Si": 0.5,
        "P": 0.015,
        "S": 0.015,
        "Cr": 23,
        "Co": 1,
        "Mo": 10,
        "Nb": 4.15,
        "Ti": 0.4,
        "Al": 0.4,
        "Fe": 5.0,
        "Ni": 54.92
}

gmol = {"C": 12.01,
        "Mn": 54.94,
        "Si": 28.08,
        "P": 30.97,
        "S": 32.06,
        "Cr": 52.0,
        "Co": 58.93,
        "Mo": 95.94,
        "Nb": 92.91,
        "Ti": 47.87,
        "Al": 26.98,
        "Fe": 55.84,
        "Ni": 58.69
}

def molfrac(weights):
    keys = weights.keys()
    moles = {}
    N = 0.0
    for key in keys:
        # Assume 1 g of material
        moles[key] = weights[key] / gmol[key]
        N += moles[key]
    for key in keys:
        moles[key] /= N
    return moles

def weightfrac(moles):
    keys = moles.keys()
    weights = {}
    W = 0.0
    for key in keys:
        # Assume 1 g of material
        weights[key] = moles[key] * gmol[key]
        W += weights[key]
    for key in keys:
        weights[key] /= W
    return weights

def printdict(d):
    for key in d.keys():
        print("  {0:2s}: {1:6.4f}".format(key, d[key]))
    print("")

"""
print("Mean composition of ASTM AM IN625 (mol frac)")
astm = {}
for key in wmin.keys():
    astm[key] = 0.5 * (wmin[key] + wmax[key])
printdict(molfrac(astm))
"""

"""
print("\nTernary (Cr+Mo)-Nb-Ni Analogue (mol frac)")
tern = {"Cr": 0.5 * (20 + 23),
        "Mo": 0.5 * (8 + 10),
        "Nb": 0.5 * (3.15 + 4.15),
        "Ni": 0.5 * (68.85 + 62.85)
}
printdict(molfrac(tern))
"""

"""
print("\nDICTRA Solid (mol frac)")
dctr = {"Cr": 13.6,
        "Mo": 13.9,
        "Nb": 23.5,
        "Ni": 49.0
}
printdict(weightfrac(dctr))
"""

"""
print("\nEnriched (Cr+Mo)-Nb-Ni Analogue (mol frac)")
enrc = {"Cr": dctr["Cr"] + dctr["Mo"],
        "Nb": 23.5,
        "Ni": 49.0
}
printdict(weightfrac(enrc))
"""

print("DICTRA Solid from paper (wt)")
dctr = {"Cr": .15,
        "Fe": .008,
        "Mo": .14,
        "Nb": .18,
        "Ni": 1-.15-.008-.14-.18
}
printdict(dctr)

print("DICTRA Solid from paper (mol)")
printdict(molfrac(dctr))

print("DICTRA Enriched from paper (wt)")
dctr = {"Cr": .11,
        "Fe": .003,
        "Mo": .25,
        "Nb": .21,
        "Ni": 1-.11-.003-.25-.21
}
printdict(dctr)

print("DICTRA Solid from paper (mol)")
printdict(molfrac(dctr))

