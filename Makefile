# Makefile
# GNU makefile for Ni-base superalloy decomposition
# Questions/comments to trevor.keller@nist.gov (Trevor Keller)

# includes and common flags
incdir = $(MMSP_PATH)/include
stdflags = -Wall -std=c++11 -I $(incdir)
links = -lz -lgsl -lgslcblas
icompiler = icc -xCORE-AVX2
gcompiler = /usr/bin/g++
pcompiler = mpic++

# Performance and energy switches
directives = -DADAPTIVE_TIMESTEPS -DNDEBUG -DGSL_RANGE_CHECK_OFF

fflags = -O3 $(directives) $(stdflags)
gflags = -O1 $(directives) $(stdflags) -pg
pflags = -O3 $(directives) $(stdflags) -include mpi.h


# WORKSTATION

# default program (shared memory, OpenMP)
alloy625: alloy625.cpp
	$(icompiler) $(fflags) $< -o $@ $(links) -fopenmp

# profiling program (no parallelism or optimization)
serial: alloy625.cpp
	$(gcompiler) $(gflags) $< -o $@ $(links)


# CLUSTER

# threaded program (shared memory, OpenMP)
smp: alloy625.cpp
	$(gcompiler) $(fflags) $< -o $@ $(links) -fopenmp

# parallel program (distributed memory, MPI)
parallel: alloy625.cpp
	$(pcompiler) $(pflags) $< -o $@ $(links)

ibtest: alloy625.cpp
	mpicxx.mpich2 $(pflags) $< -o $@ $(links)

pgparallel: alloy625.cpp
	$(pcompiler) -fastsse -Minfo -std=c++11 -I $(incdir) -include mpi.h $< -o $@ $(links) -mp

mmsp2comp: mmsp2comp.cpp
	$(gcompiler) $(stdflags) $< -o $@ -lz

clean:
	rm -f alloy625 ibtest mmsp2comp parallel pgparallel serial smp

