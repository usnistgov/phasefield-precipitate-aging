# Makefile
# GNU makefile for Ni-base superalloy decomposition
# Questions/comments to trevor.keller@nist.gov (Trevor Keller)
# Compiler optimizations after http://www.nersc.gov/users/computational-systems/retired-systems/hopper/performance-and-optimization/compiler-comparisons/

# compilers: Intel, GCC, MPI
icompiler = icc
gcompiler = /usr/bin/g++
pcompiler = /usr/bin/mpic++


# libraries: z, gsl, mpiP
stdlinks = -lz -lgsl -lgslcblas
mpilinks = -lmpiP -lbfd -liberty


# precompiler directives
# Options: -DCALPHAD	-DPARABOLA	-DADAPTIVE_TIMESTEPS	-DNDEBUG
directives = -DPARABOLA


# flags: common, debug, Intel, GNU, and MPI
stdflags  = -Wall -std=c++11 -I $(MMSP_PATH)/include $(directives)
dbgflags  = $(stdflags) -O1 -pg
idbgflags = $(stdflags) -O1 -profile-functions -profile-loops=all -profile-loops-report=2

iflags = $(stdflags) -w3 -diag-disable:remark -xCORE-AVX2 -O3 -funroll-loops -opt-prefetch -fast
gflags = $(stdflags) -pedantic -O3 -funroll-loops -ffast-math 
pflags = -include mpi.h $(dbgflags) $(mpilinks)


# WORKSTATION

# default program (shared memory, OpenMP)
alloy625: alloy625.cpp
	$(icompiler) $< -o $@ $(iflags) $(stdlinks) -openmp

# profiling program (no parallelism or optimization)
serial: alloy625.cpp
	$(gcompiler) $< -o $@ $(gflags) $(stdlinks)

iserial: alloy625.cpp
	$(icompiler) $< -o $@ $(iflags) $(stdlinks)


# CLUSTER

# threaded program (shared memory, OpenMP)
smp: alloy625.cpp
	$(gcompiler) $< -o $@ $(gflags) $(stdlinks) -fopenmp

# parallel program (distributed memory, MPI)
parallel: alloy625.cpp $(core)
	$(pcompiler) $< -o $@ $(pflags) $(stdlinks)

smpi: alloy625.cpp
	$(pcompiler) $< -o $@ $(pflags) $(stdlinks) -fopenmp

ibtest: alloy625.cpp
	/usr/local/bin/mpicxx $< -o $@ $(pflags) $(stdlinks) -fopenmp

# PGI compiler
pgparallel: alloy625.cpp
	$(pcompiler) -fast -Mipa=fast -Mfprelaxed -std=c++11 -I $(MMSP_PATH)/include -include mpi.h $< -o $@ $(stdlinks) -mp


# DESCRIPTION
description: phasefield-precipitate-aging_description.tex
	pdflatex -interaction=nonstopmode $<


# UTILITIES

# extract composition from line profile
mmsp2comp: mmsp2comp.cpp
	$(gcompiler) $(stdflags) $< -o $@ -lz

# check interfacial adsorption (should be zero)
adsorption: adsorption.cpp
	$(gcompiler) $(stdflags) $< -o $@ -lz

# generate equilibrium phase diagram information
equilibrium: equilibrium.cpp
	$(gcompiler) -Wall -std=c++11 -DPARABOLA $< -o $@ -lgsl -lgslcblas

clean:
	rm -f adsorption alloy625 equilibrium ibtest iserial mmsp2comp parallel pgparallel serial smp smpi
