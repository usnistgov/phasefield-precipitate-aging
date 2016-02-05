# Makefile
# GNU makefile for alloy718 decomposition example code
# Questions/comments to gruberja@gmail.com (Jason Gruber)

# includes
incdir = $(MMSP_PATH)/include

# compilers/flags
compiler = g++
flags = -O3 -std=c++11 -I $(incdir)
pcompiler = mpic++
pflags = $(flags) -include mpi.h

# dependencies
core = $(incdir)/MMSP.main.hpp \
       $(incdir)/MMSP.utility.hpp \
       $(incdir)/MMSP.grid.hpp

# the program
alloy718: alloy718.cpp $(core)
	$(compiler) $(flags) $< -o $@ -lz

parallel: alloy718.cpp $(core)
	$(pcompiler) $(pflags) $< -o $@ -lz

clean:
	rm -f alloy718 parallel
