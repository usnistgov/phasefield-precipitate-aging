# Makefile
# GNU makefile for Ni-base superalloy decomposition
# Questions/comments to trevor.keller@nist.gov (Trevor Keller)

# includes and boilerplate
incdir = $(MMSP_PATH)/include
boiler = -Wall -std=c++11 -I $(incdir)
links = -lz -lgsl -lgslcblas
icompiler = icc
gcompiler = /usr/bin/g++
pcompiler = mpic++

fflags = -O3  $(boiler) -DPARABOLIC
gflags = -pg $(boiler)
pflags = -O3 $(boiler) -include mpi.h


# WORKSTATION

# default program (shared memory, OpenMP)
alloy625: alloy625.cpp
	$(icompiler) $(fflags) $< -o $@ $(links) -fopenmp

# profiling program (no parallelism or optimization)
gflags = -pg $(boiler)
serial: alloy625.cpp
	$(gcompiler) $(gflags) $< -o $@ $(links)


# CLUSTER

# threaded program (shared memory, OpenMP)
smp: alloy625.cpp
	$(gcompiler) $(fflags) $< -o $@ $(links) -fopenmp

# parallel program (distributed memory, MPI)
parallel: alloy625.cpp
	$(pcompiler) $(pflags) $< -o $@ $(links) -fopenmp


clean:
	rm -f alloy625 parallel smp

