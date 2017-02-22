# Makefile
# GNU makefile for Ni-base superalloy decomposition
# Questions/comments to trevor.keller@nist.gov (Trevor Keller)
# Compiler optimizations after http://www.nersc.gov/users/computational-systems/retired-systems/hopper/performance-and-optimization/compiler-comparisons/

# includes and locations
mmspdir = $(MMSP_PATH)/include
links = -lz -lgsl -lgslcblas
icompiler = icc
gcompiler = /usr/bin/g++
pcompiler = mpic++

# Performance and energy switches
directives = -DADAPTIVE_TIMESTEPS -DNDEBUG -DGSL_RANGE_CHECK_OFF

# Compiler flags: common, debug, Intel, GNU, and MPI
stdflags = -Wall -std=c++11 -I $(mmspdir) $(directives)
dbgflags = $(stdflags) -O1 -pg
idbgflags = $(stdflags) -O1 -profile-functions -profile-loops=all -profile-loops-report=2

iflags = $(stdflags) -O3 -xCORE-AVX2 -unroll-aggressive -opt-prefetch
gflags = $(stdflags) -O3 -ffast-math -funroll-loops

pflags = $(gflags) -include mpi.h


# WORKSTATION

# default program (shared memory, OpenMP)
alloy625: alloy625.cpp
	$(icompiler) $(iflags) $< -o $@ $(links) -fopenmp

# profiling program (no parallelism or optimization)
serial: alloy625.cpp
	$(gcompiler) $(dbgflags) $< -o $@ $(links)

iserial: alloy625.cpp
	$(icompiler) $(idbgflags) $< -o $@ $(links)


# CLUSTER

# threaded program (shared memory, OpenMP)
smp: alloy625.cpp
	$(gcompiler) $(gflags) $< -o $@ $(links) -fopenmp

# parallel program (distributed memory, MPI)
parallel: alloy625.cpp
	$(pcompiler) $(pflags) $< -o $@ $(links)

smpi: alloy625.cpp
	$(pcompiler) $(pflags) $< -o $@ $(links) -fopenmp

ibtest: alloy625.cpp
	/usr/local/bin/mpicxx $(pflags) $< -o $@ $(links) -fopenmp


# PGI compiler
pgparallel: alloy625.cpp
	$(pcompiler) -fast -Mipa=fast -Mfprelaxed -std=c++11 -I $(mmspdir) -include mpi.h $< -o $@ $(links) -mp



# Utilities
mmsp2comp: mmsp2comp.cpp
	$(gcompiler) $(stdflags) $< -o $@ -lz

adsorption: adsorption.cpp
	$(gcompiler) $(stdflags) $< -o $@ -lz

clean:
	rm -f adsorption alloy625 ibtest iserial mmsp2comp parallel pgparallel serial smp smpi
