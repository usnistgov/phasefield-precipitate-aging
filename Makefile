# GNU Makefile for Ni-base superalloy evolution using solid-state
# multi-phase-field methods in C++ with OpenMP and CUDA.
# Questions/comments to trevor.keller@nist.gov (Trevor Keller)

all: thermo src
	$(MAKE) -C src/
	$(MAKE) -C src/
.PHONY: all thermo src clean clean-src clean-thermo

src:
	$(MAKE) -C $@

thermo:
	$(MAKE) -C $@

clean: clean-src clean-thermo

clean-src:
	$(MAKE) -C src clean

clean-thermo:
	$(MAKE) -C thermo clean
