# Analysis Scripts

This directory contains several small programs to help with data conversion
and analysis.

- [adsorption](adsorption.cpp) computes the degree of solute adsorption to
  the interface. For the KKS interfacial model, this ought to be zero; so
  far, this has been borne out.
- [interface-composition](interface-composition.cpp) prints the composition
  of points in γ-δ, δ-λ, and γ-λ interfaces extracted from MMSP grid data.
- [mmsp2comp](mmsp2comp.cpp) prints the compositions at every point in an MMSP grid.
- [mmsp2frac](mmsp2frac.cpp) computes the phase fractions from an MMSP grid.
