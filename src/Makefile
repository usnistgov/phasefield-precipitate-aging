# GNU Makefile for Ni-base superalloy evolution using solid-state
# multi-phase-field methods in C++ with OpenMP and CUDA.
# Questions/comments to trevor.keller@nist.gov (Trevor Keller)

# Note: The project will not build unless you have exported
#       CUDA_HDR_PATH - directory containing curand_kernel.h
#       MMSP_PATH     - root directory of cloned MMSP repository

NVCXX = nvcc
CXX = g++ -fPIC
CXXFLAGS = -Wall -std=c++11 -funroll-loops -ffast-math -fopenmp -I$(CUDA_HDR_PATH) -I$(MMSP_PATH)/include
NVCXXFLAGS = -std=c++11 -D_FORCE_INLINES -Wno-deprecated-gpu-targets --compiler-options="$(CXXFLAGS)"
LINKS = -lcuda -lcurand -lm -lpng -lz

OBJS = data.o discretization.o d_parabola625.o \
       mesh.o enrichment.o numerics.o output.o parabola625.o

# Executable
alloy625: alloy625.cpp main.cpp $(OBJS)
	$(NVCXX) $(NVCXXFLAGS) $(OBJS) $< -o $@ $(LINKS)

profile: CXXOPT = -O1 -g
profile: CXXFLAGS += $(CXXOPT)
profile: NVCXXFLAGS += $(CXXOPT) -lineinfo # -arch=$(CUDARCH)
profile: alloy625

# CUDA objects
data.o: cuda_data.cu cuda_data.h
	$(NVCXX) $(NVCXXFLAGS) -dc $< -o $@

discretization.o: cuda_discretization.cu cuda_kernels.cuh
	$(NVCXX) $(NVCXXFLAGS) -dc $< -o $@

d_parabola625.o: parabola625.cu parabola625.cuh parabola625.h
	$(NVCXX) $(NVCXXFLAGS) -dc $< -o $@

parabola625.cu: energy_functions.sh ../thermo/parabola625.c
	bash $<

# Common objects
mesh.o: mesh.cpp mesh.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

enrichment.o: enrichment.c enrichment.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

numerics.o: numerics.cpp numerics.h
	$(CXX) $(CXXFLAGS) -c $<

output.o: output.cpp output.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

parabola625.o: parabola625.cpp parabola625.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm -f alloy625 $(OBJS)
