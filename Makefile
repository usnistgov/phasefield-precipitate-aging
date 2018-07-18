# Makefile
# GNU makefile for Ni-base superalloy decomposition
# Questions/comments to trevor.keller@nist.gov (Trevor Keller)
# Compiler optimizations after http://www.nersc.gov/users/computational-systems/retired-systems/hopper/performance-and-optimization/compiler-comparisons/

# compilers: Intel, GCC, MPI
gcompiler = g++
icompiler = icc
pcompiler = mpicxx

# flags: common, debug, Intel, GNU, and MPI
stdflags  = -Wall -std=c++11 -I $(MMSP_PATH)/include -I ..
dbgflags  = -pedantic $(stdflags) $(dbgdirect) -O0 -pg
idbgflags = $(stdflags) $(dbgdirect) -O0 -profile-functions -profile-loops=all -profile-loops-report=2

gccflags = $(stdflags) -pedantic -O3 -funroll-loops -ffast-math
iccflags = $(stdflags) -w3 -diag-disable:remark -O3 -funroll-loops -qopt-prefetch
mpiflags = $(gccflags) -include mpi.h

deps = alloy625.h parabola625.c

# WORKSTATION EXECUTABLES
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

iserial: alloy625.cpp $(deps)
	$(icompiler) $< -o $@ $(idbgflags) -lz


# CLUSTER EXECUTABLES

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


# ANALYSIS EXECUTABLES
.PHONY: analysis
analysis:
	$(MAKE) -C analysis

.PHONY: clean
clean:
	rm -f alloy625 ibtest iserial ismpi parallel serial smp smpi *.pyc
