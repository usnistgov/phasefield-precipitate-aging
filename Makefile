# Makefile
# GNU makefile for Ni-base superalloy decomposition
# Questions/comments to trevor.keller@nist.gov (Trevor Keller)

# includes
incdir = $(MMSP_PATH)/include

# compilers/flags
compiler = icc
ccompiler = /usr/bin/g++
pcompiler = /usr/bin/mpic++.openmpi

#flags = -O3 -Wall -std=c++11 -I $(incdir)
#flags = -O3 -Wall -std=c++11 -I $(incdir) -DJACOBIAN
flags = -O3 -Wall -std=c++11 -I $(incdir) -DJACOBIAN -DPARABOLIC

pflags = $(flags) -include mpi.h

# serial program
alloy625: alloy625.cpp
	$(compiler) $(flags) $< -o $@ -lz -lgsl -lgslcblas -fopenmp

# parallel program (shared memory, OpenMP)
smp: alloy625.cpp
	$(ccompiler) $(flags) $< -o $@ -lz -lgsl -lgslcblas -fopenmp

# parallel program (distributed memory, MPI)
parallel: alloy625.cpp
	$(pcompiler) $(pflags) $< -o $@ -lz -lgsl -lgslcblas

minima: minima.c
	g++ -O3 -Wall $< -o $@ -lgsl -lgslcblas

clean:
	rm -f alloy625 minima parallel smp

#alloy718: alloy718.cpp
#	$(compiler) $(flags) $< -o $@ -lz
#
#parallel: alloy718.cpp
#	$(pcompiler) $(pflags) $< -o $@ -lz
#
#clean:
#	rm -f alloy718 parallel
