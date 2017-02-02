# Phase field precipitate aging model

> Phase field model for precipitate aging in ternary analogues to Ni-based superalloys


## Table of Contents

- [Background](Background)
- [Install](#Install)
- [Usage](#Usage)
- [Contribute](#Contribute)
- [References](#References)
- [License](#license)


## Background

This repository contains a phase-field model for solid-state transformations in
Inconel 625 based on [Zhou *et al.*][Zhou2014], which is a ternary generalization
of the binary [KKS model][Kim1999].

To capture δ, μ, and Laves precipitates in a γ matrix, I have
chosen Ni–30%Cr–2%Nb as the model system. The interdendritic regions in
additive manufacturing get enriched to Ni–31%Cr–13%Nb.
The four-phase three-component model is represented using two composition fields
(Cr and Nb) and three phase fields (δ, μ, and Laves). There is one
dependent composition (Ni) and one dependent phase (γ).
For a more thorough discussion of the model, please build the PDF description:

```
    pdflatex description.tex
```

Free energies for each constituent phase are computed using a CALPHAD database,
simplified from [Du *et al.*][Du2005] to enable one-to-one mapping of sublattice
compositions to system compositions. This database is provided in
[Du_Cr-Nb-Ni_simple.tdb](Du_Cr-Nb-Ni_simple.tdb).


## Install

This repository contains Python code to handle the CALPHAD database and C++
code to perform the phase-field simulation. A Python 3 interpreter and C++11
compiler are recommended. You will need to satisfy the following dependencies:

- Python
  - [PyCALPHAD](http://pycalphad.readthedocs.io/en/latest/)
  - [SymPy](http://www.sympy.org/en/index.html)

- C++
  - [GNU Scientific Library](https://www.gnu.org/software/gsl/)
  - [MMSP](https://github.com/mesoscale/mmsp)
  
After downloading MMSP, the core dependency of this model implementation,
please set the environmental variable ```MMSP_PATH``` to its location. If
you are using ```bash```, do something similar to

```
  echo "MMSP_PATH=~/Downloads/mmsp" >> ~/.bashrc
  . ~/.bashrc
```

You will also want to build the MMSP utilities,
as described in the MMSP documentation.


## Usage

1. ```python CALPHAD_energies.py```
   This will use pycalphad to read the database and extract expressions,
   which are then manipulated and written into C-code by sympy.

2. ```make``` (OpenMP, Intel compiler).
  This will compile the source code into a binary, ```alloy625```.
  - or ```make serial``` (serial, GNU compiler).
  This will compile the source code into a binary, ```serial```.
  - or ```make parallel``` (MPI, GNU compiler)
  This will compile the source code into a binary, ```parallel```.

3. Run the code. Since your executable is built against ```MMSP.main.hpp```,
  the options of that program apply to your binary. For usage suggestions,
  run ```alloy625 --help``` or ```serial --help``` or
  ```mpirun -np 1 parallel --help```. A typical MMSP run comprises two steps:
  initialization and update loops. So you would normally do:
  - ```./alloy625 --example 1 data.dat```
  - ```./alloy625 data.dat 1000000 100000```
  - ```mmsp2pvd data.dat data.*.dat``` to generate VTK visualization files,
    then use a VTK viewer such as ParaView or Mayavi to see the results
  - or ```for f in *.dat; do mmsp2tsv $f; done``` to generate tab-delimited
  text files.

4. Remix, run, and distribute your own variants.


## Contribute

Pull requests are invited. Comments are welcome through issues and by e-mail.


## References

[Du2005]:  Du, Y.; Liu, S.; Chang, Y.; and Yang, Y.
          "A thermodynamic modeling of the Cr–Nb–Ni system."
          *Calphad* **29** (2005) 140–148.
          DOI: [10.1016/j.calphad.2005.06.001](http://dx.doi.org/10.1016/j.calphad.2005.06.001).

[Kim1999]: Kim, S. G.; Kim, W. T. and Suzuki, T.
		  "Phase-field model for binary alloys."
		  *Phys. Rev. E* **60** (1999) 7186–7197.
		  DOI: [10.1103/PhysRevE.60.7186](http://dx.doi.org/10.1103/PhysRevE.60.7186).

[Karunaratne2005]: Karunaratne, M. S. A. and Reed, R. C.
		          "Interdiffusion of Niobium and Molybdenum in Nickel between 900 - 1300&deg;C."
		          *Defect and Diffusion Forum* **237-240** (2005) 420–425.
		          DOI: [10.4028/www.scientific.net/DDF.237-240.420](http://dx.doi.org/10.4028/www.scientific.net/DDF.237-240.420).

[Provatas2010]: Provatas, N. and Elder, K.
		       *Phase-Field Methods in Materials Science and Engineering.*
		       Wiley-VCH: Weinheim, 2010.
		       ISBN: 978-3-527-40747-7.

[Xu2016]: Xu, G.; Liu, Y.; and Kang, Z.
		 "Atomic Mobilities and Interdiffusivities for fcc Ni-Cr-Nb Alloys."
		 *Met. Trans. B* **47B** (2016) 3126–3131.
		 DOI: [10.1007/s11663-016-0726-6](http://dx.doi.org/10.1007/s11663-016-0726-6).

[Zhou2014]: Zhou, N.; Lv, D.; Zhang, H.; McAllister, D.; Zhang, F.; Mills, M. and Wang, Y.
		   "Computer simulation of phase transformation and plastic deformation in IN718 superalloy: Microstructural evolution during precipitation."
		   *Acta Mater.* **65** (2014) 270–286.
		   DOI: [10.1016/j.actamat.2013.10.069](http://dx.doi.org/10.1016/j.actamat.2013.10.069).


## License

As a work of the United States Government, this software is in the public domain within the United States.


### Derivative Works

The source files (```.py```, ```.hpp```, and ```.cpp```) in this repository were
written by an employee of the United States federal government in the course of
their employment, and are therefore not subject to copyright. They are public
domain. However, the Mesoscale Microstructure Simulation Project (MMSP) is subject
to the General Public License v3.0, and this software ```#include```s major
aspects of that work. Therefore, if you are not an employee of the US government,
your derivative works will likely be subject to the terms and conditions of the GPL.
