# Makefile
# GNU makefile for Ni-base superalloy decomposition
# Questions/comments to trevor.keller@nist.gov (Trevor Keller)

gcompiler = /usr/bin/g++
stdflags  = -Wall -pedantic -O2 -std=c++11 -I $(MMSP_PATH)/include -I ..

all: adsorption ifcomp mmsp2comp mmsp2frac
.PHONY: all

# check interfacial adsorption (should be zero)
adsorption: adsorption.cpp
	$(gcompiler) $(stdflags) -O2 $< -o $@ -lz

# extract composition from line profile
ifcomp: interface-composition.cpp
	$(gcompiler) $(stdflags) -O2 $< -o $@ -lz

# extract composition from line profile
mmsp2comp: mmsp2comp.cpp
	$(gcompiler) $(stdflags) -O2 $< -o $@ -lz

# extract phase fractions
mmsp2frac: mmsp2frac.cpp
	$(gcompiler) $(stdflags) -O2 $< -o $@ -lz

clean:
	rm -f adsorption ifcomp mmsp2comp mmsp2frac
