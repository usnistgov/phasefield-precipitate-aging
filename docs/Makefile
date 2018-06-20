# Makefile
# GNU makefile for Ni-base superalloy decomposition
# Questions/comments to trevor.keller@nist.gov (Trevor Keller)
# Compiler optimizations after http://www.nersc.gov/users/computational-systems/retired-systems/hopper/performance-and-optimization/compiler-comparisons/

# DOCUMENTATION

.PHONY: docs
docs: phasefield-precipitate-aging_description.tex
	pdflatex -interaction=nonstopmode $<

clean:
	rm -f *.aux *.log
