# Makefile
# GNU makefile for Ni-base superalloy decomposition
# Questions/comments to trevor.keller@nist.gov (Trevor Keller)
# Compiler optimizations after http://www.nersc.gov/users/computational-systems/retired-systems/hopper/performance-and-optimization/compiler-comparisons/

# compilers: GCC, Intel, MPI
gcompiler = g++
icompiler = icc
pcompiler = mpicxx

# flags: common, debug, GNU, Intel, and MPI
stdflags  = -Wall -pedantic -std=c++11 -I$(MMSP_PATH)/include -I..
dbgflags  = $(stdflags) -O0 -pg
gccflags = $(stdflags) -pedantic -O3 -funroll-loops -ffast-math
iccflags = $(stdflags) -w3 -diag-disable:remark -O3 -funroll-loops -qopt-prefetch
mpiflags = $(gccflags) -I$(MPI_PATH) -include mpi.h

deps = alloy625.h


# === WORKSTATION EXECUTABLES ===

all: alloy625 analysis
.PHONY: all alloy625.h

# free energy landscape
parabola625.c: CALPHAD_energies.py
	python2 $<

# default program (shared memory, OpenMP)
alloy625: alloy625.cpp $(deps)
	$(icompiler) $< -o $@ $(iccflags) -lz -qopenmp

# profiling program (no parallelism or optimization)
serial: alloy625.cpp $(deps)
	$(gcompiler) $< -o $@ $(dbgflags) -lz


# === CLUSTER EXECUTABLES ===

# shared thread memory (OpenMP) parallelism
smp: alloy625.cpp $(deps)
	$(gcompiler) $< -o $@ $(gccflags) -lz -fopenmp

# distributed node memory (MPI), distributed thread memory (MPI) parallelism
parallel: alloy625.cpp $(deps)
	$(pcompiler) $< -o $@ $(mpiflags) -lz

# distributed node memory (MPI), shared thread memory (OpenMP) parallelism
smpi: alloy625.cpp $(deps)
	$(pcompiler) $< -o $@ $(mpiflags) -lz -fopenmp

# distributed node memory (MPI), shared thread memory (OpenMP) parallelism
ismpi: alloy625.cpp $(deps)
	$(pcompiler) $< -o $@ $(iccflags) -include mpi.h -L/usr/lib64 -lz -fopenmp


# === ANALYSIS EXECUTABLES ===

.PHONY: analysis
analysis:
	$(MAKE) -C analysis

# === CLEANUP ===

.PHONY: clean
clean:
	rm -f alloy625 ismpi parallel serial smp smpi *.o *.pyc
