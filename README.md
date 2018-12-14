# Phase field precipitate aging model

> Phase field model for precipitate aging in ternary analogues to Ni-based superalloys

## Table of Contents

- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [Details](#details)
- [Contribute](#contribute)
- [References](#references)
- [License](#license)

## Background

This repository contains a phase-field model for solid-state transformations in
Inconel 625 based on [Zhou *et al.*](#zhou-2014), which is a ternary generalization
of the binary [KKS model](#kim-1999).

To capture δ, μ, and Laves precipitates in a γ matrix, I have
chosen Ni–30%Cr–2%Nb as the model system. The interdendritic regions in
additive manufacturing get enriched to Ni–31%Cr–13%Nb.
The four-phase three-component model is represented using two composition fields
(Cr and Nb) and three phase fields (δ, μ, and Laves). There is one
dependent composition (Ni) and one dependent phase (γ).

Free energies for each constituent phase are computed using a CALPHAD database,
simplified from [Du *et al.*](#du-2005) to enable one-to-one mapping of sublattice
compositions to system compositions. This database is provided in
[Du_Cr-Nb-Ni_simple.tdb](Du_Cr-Nb-Ni_simple.tdb). The CALPHAD expressions are
further simplified using 2<sup>nd</sup>-order Taylor series (paraboloid)
approximations.

## Install

This repository contains Python code to handle the CALPHAD database and C++
code to perform the phase-field simulation. A Python 3 interpreter and
C++11 compiler are recommended. You will need to satisfy the following
dependencies:

- Python
  - [PyCALPHAD](http://pycalphad.readthedocs.io/)
  - [SymPy](http://www.sympy.org/)

- C++
  - [MMSP](https://github.com/mesoscale/mmsp)
  
After downloading MMSP, the core dependency of this model implementation,
please set the environmental variable ```MMSP_PATH``` to its location. If
you are using ```bash```, do something similar to

```
  echo "MMSP_PATH=~/Downloads/mmsp" >> ~/.bashrc
  . ~/.bashrc
```

You will also want to build the MMSP utilities, as described in the MMSP
documentation.

## Usage

1. ```cd thermo; python CALPHAD_energies.py; python nucleation.py``` This
   will use pycalphad to read the database and extract expressions, which
   are then manipulated and written into C-code by SymPy.
2. ```make``` (OpenMP, Nvidia CUDA compiler). This will compile the source
   code into a binary, ```alloy625```.
3. Run the code. Since your executable is built against
   ```MMSP.main.hpp```, the options of that program apply to your binary.
   For usage suggestions, run ```./alloy625 --help``` or ```./serial
   --help``` or ```mpirun -np 1 parallel --help```, depending on which
   executable you built. A typical MMSP run comprises two steps:
   initialization and update loops. So you would normally do:
   - ```./alloy625 --example 2 data.dat```
   - ```./alloy625 data.dat 10000000 1000000```
   - ```mmsp2pvd data.dat data.*.dat``` to generate VTK visualization
     files, then use a VTK viewer such as ParaView or Mayavi to see the
     results.
4. Remix, run, and analyze your own variants.

## Details

Due to the paraboloid free energy representation, we can leverage the
[HiPerC](https://github.com/usnistgov/hiperc) project to implement the
equations of motion on the GPU using CUDA.

### Workflow

#### `MMSP::generate()`

Initial conditions are generated using the existing MMSP code, and written to a
compressed MMSP checkpoint, with no work assigned to the GPU.

#### Initialization

1. `main.cpp` was adapted from MMSP, rather than HiPerC.
2. The initial condition checkpoint gets read back into an MMSP::grid
   object.
3. The Laplacian kernel gets written into const cache on the GPU.
4. 12 device arrays get allocated on the GPU: one old and one new for each
   field variable.
    1. `x_Cr`
    2. `x_Nb`
    3. `phi_del`
    4. `phi_lav`
    5. `x_gam_Cr`
    6. `x_gam_Nb`
5. 12 identical host arrays get allocated on the CPU.

#### `MMSP::update()`

##### Before timestepping:

1. Data gets read from the MMSP::grid into the host arrays.
2. Data gets copied from the host to device arrays.

##### For each iteration:

1. Boundary conditions get applied on each of the "old" device arrays.
2. Laplacian values gets computed and recorded in each of the "new" arrays.
3. Boundary conditions get applied on each of the "new" device arrays.
4. Updated field values get computed from the "old" and "new" device
   arrays.
5. Fictitious matrix phase compositions get computed and written into the
   "new" device array.

##### After timestepping:

1. Data gets copied from the 6 "new" device arrays into the corresponding
   host arrays.
2. Data gets read from the host arrays into the MMSP::grid object.
3. The MMSP::grid object gets written to a compressed MMSP checkpoint.

### Cleanup

Arrays gets freed from the host and device once the last checkpoint is
written.

## Contribute

Pull requests are welcome! Comments are also appreciated via
[issues](https://github.com/usnistgov/phasefield-precipitate-aging/issues)
and [e-mail](mailto:trevor.keller@nist.gov).

## References

### Du 2005

  Du, Y.; Liu, S.; Chang, Y. and Yang, Y.
  "A thermodynamic modeling of the Cr–Nb–Ni system."
  *Calphad* **29** (2005) 140–148.
  DOI: [10.1016/j.calphad.2005.06.001](http://dx.doi.org/10.1016/j.calphad.2005.06.001)

### Kim 1999

  Kim, S. G.; Kim, W. T. and Suzuki, T.
  "Phase-field model for binary alloys."
  *Phys. Rev. E* **60** (1999) 7186–7197.
  DOI: [10.1103/PhysRevE.60.7186](http://dx.doi.org/10.1103/PhysRevE.60.7186)

### Karunaratne 2005

  Karunaratne, M. S. A. and Reed, R. C.
  "Interdiffusion of Niobium and Molybdenum in Nickel between 900 - 1300&deg;C."
  *Defect and Diffusion Forum* **237-240** (2005) 420–425.
  DOI: [10.4028/www.scientific.net/DDF.237-240.420](http://dx.doi.org/10.4028/www.scientific.net/DDF.237-240.420)

### Provatas 2010

  Provatas, N. and Elder, K.
  [*Phase-Field Methods in Materials Science and Engineering.*](http://www.wiley.com/WileyCDA/WileyTitle/productCd-3527407472.html)
  Wiley-VCH: Weinheim, 2010. ISBN: 978-3-527-40747-7.

### Xu 2016

  Xu, G.; Liu, Y. and Kang, Z.
  "Atomic Mobilities and Interdiffusivities for fcc Ni-Cr-Nb Alloys."
  *Met. Trans. B* **47B** (2016) 3126–3131.
  DOI: [10.1007/s11663-016-0726-6](http://dx.doi.org/10.1007/s11663-016-0726-6)

### Zhou 2014

  Zhou, N.; Lv, D.; Zhang, H.; McAllister, D.; Zhang, F.; Mills, M. and
  Wang, Y. "Computer simulation of phase transformation and plastic
  deformation in IN718 superalloy: Microstructural evolution during
  precipitation." *Acta Mater.* **65** (2014) 270–286. DOI:
  [10.1016/j.actamat.2013.10.069](http://dx.doi.org/10.1016/j.actamat.2013.10.069)

## License

As a work of the United States Government, this software is in the public
domain within the United States.

### Derivative Works

The source files (```.py```, ```.hpp```, and ```.cpp```) in this repository
were written by an employee of the United States federal government in the
course of their employment, and are therefore not subject to copyright.
They are public domain. However, the Mesoscale Microstructure Simulation
Project (MMSP) is subject to the General Public License v3.0, and this
software ```#include```s major aspects of that work. Therefore, if you are
not an employee of the US government, your derivative works will likely be
subject to the terms and conditions of the GPL.
