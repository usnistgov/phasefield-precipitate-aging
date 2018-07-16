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

all: adsorption ifcomp mmsp2comp mmsp2frac
.PHONY: all

# check interfacial adsorption (should be zero)
adsorption: adsorption.cpp
	$(gcompiler) $(stdflags) $(stddirect) -O2 $< -o $@ -lz

# extract composition from line profile
ifcomp: interface-composition.cpp
	$(gcompiler) $(stdflags) $(stddirect) -O2 $< -o $@ -lz

# extract composition from line profile
mmsp2comp: mmsp2comp.cpp
	$(gcompiler) $(stdflags) $(stddirect) -O2 $< -o $@ -lz

# extract phase fractions
mmsp2frac: mmsp2frac.cpp
	$(gcompiler) $(stdflags) $(stddirect) -O2 $< -o $@ -lz

# compare Newton-Raphson (numerical) to linear root solver
fictitious-comparison: fictitious-comparison.cpp
	$(gcompiler) $(stdflags) $(stddirect) -g $< -o $@ -lz

clean:
	rm -f adsorption ifcomp mmsp2comp mmsp2frac
