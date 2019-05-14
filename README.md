# Phase field precipitate aging model

Phase field model for precipitate aging in ternary analogues to Ni-based
superalloys. Please cite using the following DOI:

[![DOI](https://zenodo.org/badge/80764108.svg)](https://zenodo.org/badge/latestdoi/80764108)

## Table of Contents

- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [Details](#details)
- [Contribute](#contribute)
- [References](#references)
- [License](#license)

## Background

This repository contains a phase-field model for solid-state
transformations in Inconel 625 based on [Zhou *et al.*](#zhou-2014), which
involves a ternary generalization of the binary [KKS model](#kim-1999).

To capture δ and λ intermetallic precipitates in a γ matrix, I have chosen the
ternary Cr-Nb-Ni system. The three-phase three-component model is represented
using two composition fields (Cr and Nb) and two phase fields (δ and λ). There
is one dependent composition (Ni) and one dependent phase (γ). Based on
[ASTM F3056](#astm-f3056), combining Cr with Mo under the assumption that their
influences on the alloy are similar, this codebase considers system compositions
between (Ni--0.0202 Nb--0.2794 Cr) and (Ni--0.0269 Nb--0.3288 Cr), expressed as
molar fractions. Based on [DICTRA simulations](#keller-2018), this work
considers enrichment of interdendritic regions to compositions between
(Ni--0.1659 Nb--0.2473 Cr) and (Ni--0.1726 Nb--0.2967 Cr).

Free energies for each constituent phase are computed using a CALPHAD database,
simplified from [Du *et al.*](#du-2005) to enable one-to-one mapping of
sublattice compositions to system compositions. This modified database is
[provided](Du_Cr-Nb-Ni_simple.tdb). The CALPHAD expressions are further
simplified using 2<sup>nd</sup>-order Taylor series (paraboloid) approximations.

Further details are provided in [src/README](src/README.md) and
[thermo/README](thermo/README.md).

## Install

This repository contains Python code to handle the CALPHAD database and C++ code
to perform the phase-field simulation. A Python 3 interpreter and C++11 compiler
are recommended. You will need to satisfy the following dependencies:

- Python
  - [Cloudpickle](https://github.com/cloudpipe/cloudpickle)
  - [Dill](https://github.com/uqfoundation/dill)
  - [NumPy](https://www.numpy.org/)
  - [PyCALPHAD](http://pycalphad.readthedocs.io/)
  - [SymPy](http://www.sympy.org/)
  - [TinyDB](https://tinydb.readthedocs.io/)
  - [Xarray](http://xarray.pydata.org/)

- C++
  - [CUDA](https://developer.nvidia.com/cuda-toolkit)
  - [MMSP](https://github.com/mesoscale/mmsp)
  - [matplotlib-cpp](https://github.com/tkphd/matplotlib-cpp/tree/develop) (submodule)

After downloading MMSP, please set the environmental variable `MMSP_PATH`
to its location. If you are using `bash`, do something similar to

```bash
$ echo "MMSP_PATH=~/Downloads/mmsp" >> ~/.bashrc
$ . ~/.bashrc
```

You will also want to build the MMSP utilities, as described in the MMSP
documentation.

## Usage

2. `make`. This will compile the source code into a binary, `src/alloy625`.
3. Run the code. Since your executable is built against `MMSP.main.hpp`,
   the options of that program apply to your binary. For usage suggestions,
   run `./alloy625 --help`. A typical MMSP run comprises two steps:
   initialization and update loops. So you would normally do:
   - `./alloy625 --example 2 data.dat`
   - `./alloy625 data.dat 10000000 1000000`
   - `mmsp2pvd data.dat data.*.dat` to generate VTK visualization
     files, then use a VTK viewer such as ParaView or Mayavi to see the
     results.
4. Remix, run, and analyze your own variants.

## Contribute

Pull requests are welcome! Comments are also appreciated via
[issues](https://github.com/usnistgov/phasefield-precipitate-aging/issues)
and [e-mail](mailto:trevor.keller@nist.gov).

## References

### ASTM F3056

  "Standard Specification for Additive Manufacturing Nickel Alloy (UNS N06625)
  with Powder Bed Fusion."
  URL: https://www.astm.org/Standards/F3056.htm

### Du 2005

  Du, Y.; Liu, S.; Chang, Y. and Yang, Y.
  "A thermodynamic modeling of the Cr–Nb–Ni system."
  *Calphad* **29** (2005) 140–148.
  DOI: [10.1016/j.calphad.2005.06.001](https://doi.org/10.1016/j.calphad.2005.06.001)

### Jokisaari 2016

  Jokisaari, A.M.; Permann, C.; Thornton, K.
  "A nucleation algorithm for the coupled conserved-nonconserved phase field model."
  *Computational Materials Science* **112** (2016) 128–138.
  DOI: [10.1016/j.commatsci.2015.10.009](https://doi.org/10.1016/j.commatsci.2015.10.009)

### Karunaratne 2005

  Karunaratne, M. S. A. and Reed, R. C.
  "Interdiffusion of Niobium and Molybdenum in Nickel between 900 - 1300&deg;C."
  *Defect and Diffusion Forum* **237-240** (2005) 420–425.
  DOI: [10.4028/www.scientific.net/DDF.237-240.420](https://doi.org/10.4028/www.scientific.net/DDF.237-240.420)

### Kim 1999

  Kim, S. G.; Kim, W. T. and Suzuki, T.
  "Phase-field model for binary alloys."
  *Physical Review E* **60** (1999) 7186–7197.
  DOI: [10.1103/PhysRevE.60.7186](https://doi.org/10.1103/PhysRevE.60.7186)

### Keller 2018

  Keller, T.; Lindwall, G.; Ghosh, S.; Ma, L.; Lane, B.; Zhang, F.; Kattner, U.; Lass, E.; Heigel, J.; Idell, Y.; Williams, M.; Allen, A.; Guyer, J.; and Levine, L.
  "Application of finite element, phase-field, and CALPHAD-based methods to additive manufacturing of Ni-based superalloys."
  *Acta Materialia* **139** (2018) 244-253.
  DOI: [10.1016/j.actamat.2017.05.003](https://doi.org/10.1016/j.actamat.2017.05.003)

### Provatas 2010

  Provatas, N. and Elder, K.
  [*Phase-Field Methods in Materials Science and Engineering.*](http://www.wiley.com/WileyCDA/WileyTitle/productCd-3527407472.html)
  Wiley-VCH: Weinheim, 2010. ISBN: 978-3-527-40747-7.

### Xu 2016

  Xu, G.; Liu, Y. and Kang, Z.
  "Atomic Mobilities and Interdiffusivities for fcc Ni-Cr-Nb Alloys."
  *Metallurgical Transactions B* **47B** (2016) 3126–3131.
  DOI: [10.1007/s11663-016-0726-6](https://doi.org/10.1007/s11663-016-0726-6)

### Zhou 2014

  Zhou, N.; Lv, D.; Zhang, H.; McAllister, D.; Zhang, F.; Mills, M. and
  Wang, Y. "Computer simulation of phase transformation and plastic
  deformation in IN718 superalloy: Microstructural evolution during
  precipitation." *Acta Materialia* **65** (2014) 270–286. DOI:
  [10.1016/j.actamat.2013.10.069](https://doi.org/10.1016/j.actamat.2013.10.069)

## License

See [LICENSE](LICENSE.md).

The source files (`.py`, `.hpp`, and `.cpp`) in this repository
were written by an employee of the United States federal government in the
course of their employment, and are therefore not subject to copyright.
They are public domain. However, the Mesoscale Microstructure Simulation
Project (MMSP) is subject to the General Public License v3.0, and this
software `#include`s major aspects of that work. Therefore, if you are
not an employee of the US government, your derivative works will likely be
subject to the terms and conditions of GPLv3.
