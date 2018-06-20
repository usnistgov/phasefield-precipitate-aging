# Makefile
# GNU makefile for Ni-base superalloy decomposition
# Questions/comments to trevor.keller@nist.gov (Trevor Keller)
# Compiler optimizations after http://www.nersc.gov/users/computational-systems/retired-systems/hopper/performance-and-optimization/compiler-comparisons/

# compilers: Intel, GCC, MPI
gcompiler = /usr/bin/g++-4.9
stdlinks = -lz

stddirect = -DPARABOLA -DNDEBUG

# flags: common, debug, Intel, GNU, and MPI
stdflags  = -Wall -pedantic -O2 -std=c++11 -I $(MMSP_PATH)/include -I ..

# generate equilibrium phase diagram information
equilibrium: equilibrium.cpp
	$(gcompiler) $(stdflags) $(stddirect) $< -o $@ -lgsl -lgslcblas

clean:
	rm -f equilibrium
