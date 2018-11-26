# GNU Makefile for Ni-base superalloy evolution using solid-state
# multi-phase-field methods in C++ with OpenMP and CUDA.
# Questions/comments to trevor.keller@nist.gov (Trevor Keller)

NVCXX = nvcc
CXX = g++ -fPIC
CXXFLAGS = -Wall -std=c++11 -funroll-loops -ffast-math -fopenmp -I. -I$(MMSP_PATH)/include
NVCXXFLAGS = -std=c++11 -D_FORCE_INLINES -Wno-deprecated-gpu-targets --compiler-options="$(CXXFLAGS)"
LINKS = -lcuda -lm -lpng -lz

OBJS = boundaries.o data.o discretization.o mesh.o numerics.o output.o d_parabola625.o parabola625.o

# Executable
alloy625: src/alloy625.cpp $(OBJS)
	$(NVCXX) $(NVCXXFLAGS) $(OBJS) $< -o $@ $(LINKS)

profile: CXXFLAGS += -O0 -g
profile: NVCXXFLAGS += -O0 -g -lineinfo -arch=$(CUDARCH)
profile: cuda625

# CUDA objects
boundaries.o: src/cuda_boundaries.cu
	$(NVCXX) $(NVCXXFLAGS) -dc $< -o $@

data.o: src/cuda_data.cu
	$(NVCXX) $(NVCXXFLAGS) -dc $< -o $@

discretization.o: src/cuda_discretization.cu
	$(NVCXX) $(NVCXXFLAGS) -dc $< -o $@

d_parabola625.o: src/parabola625.cu
	$(NVCXX) $(NVCXXFLAGS) -dc $< -o $@

# Common objects
mesh.o: src/mesh.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

numerics.o: src/numerics.cpp
	$(CXX) $(CXXFLAGS) -c $<

output.o: src/output.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

parabola625.o: src/parabola625.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm -f alloy625 *.o
