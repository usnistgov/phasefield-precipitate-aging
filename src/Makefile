# GNU Makefile for Ni-base superalloy evolution using solid-state
# multi-phase-field methods in C++ with OpenMP and CUDA.
# Questions/comments to trevor.keller@nist.gov (Trevor Keller)

# Note: The project will not build unless you have exported
#       CUDA_HDR_PATH - directory containing curand_kernel.h
#       MMSP_PATH     - root directory of cloned MMSP repository

PYVER = 2.7
PLT_SRC = -I./matplotlib-cpp
PLT_LIB = -I/usr/include/python$(PYVER) \
          -I/usr/include/x86_64-linux-gnu/python$(PYVER) \
          -I/usr/lib/python$(PYVER)/dist-packages/numpy/core/include
PLT_LNK = -lpython$(PYVER)
PLTFLAGS = $(PLT_SRC) $(PLT_LIB)

CXX = g++
INCLUDES = -I. -I../thermo -I$(MMSP_PATH)/include -I$(CUDA_HDR_PATH) $(PLTFLAGS)

CXXFLAGS = -O3 -Wall -std=c++11 \
           -funroll-loops -ffast-math -fopenmp \
           $(INCLUDES)

NVCXX = nvcc
NVCXXFLAGS = -std=c++11 -D_FORCE_INLINES -Wno-deprecated-gpu-targets \
             $(INCLUDES) --compiler-options="$(CXXFLAGS)"

LINKS = -lcuda -lcurand -lm $(PLT_LNK) -lz

OBJS = data.o discretization.o d_nucleation.o d_parabola625.o mesh.o \
       enrichment.o nucleation.o numerics.o output.o parabola625.o

# Executable

alloy625: alloy625.cpp globals.h main.cpp $(OBJS)
	$(NVCXX) $(NVCXXFLAGS) $(OBJS) $< -o $@ $(LINKS)

# Debugging

debug: CXXOPT = -O0 -g
debug: CXXFLAGS += $(CXXOPT)
debug: NVCXXFLAGS += $(CXXOPT) -lineinfo # -arch=$(CUDARCH)
debug: alloy625

profile: CXXOPT = -O1 -pg
profile: CXXFLAGS += $(CXXOPT)
profile: NVCXXFLAGS += $(CXXOPT) -lineinfo # -arch=$(CUDARCH)
profile: alloy625

# CUDA objects

data.o: cuda_data.cu cuda_data.h
	$(NVCXX) $(NVCXXFLAGS) -dc $< -o $@

discretization.o: cuda_discretization.cu globals.h cuda_data.h cuda_kernels.cuh parabola625.cu parabola625.cuh
	$(NVCXX) $(NVCXXFLAGS) -dc $< -o $@

d_nucleation.o: nucleation.cu nucleation.cuh
	$(NVCXX) $(NVCXXFLAGS) -dc $< -o $@

d_parabola625.o: parabola625.cu parabola625.cuh
	$(NVCXX) $(NVCXXFLAGS) -dc $< -o $@

nucleation.cu: energy_functions.sh ../thermo/nucleation.c
	bash $<

parabola625.cu: energy_functions.sh ../thermo/parabola625.c
	bash $<

# Common objects

mesh.o: mesh.cpp mesh.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

enrichment.o: ../thermo/enrichment.c ../thermo/enrichment.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

nucleation.o: ../thermo/nucleation.c ../thermo/nucleation.h
	$(CXX) $(CXXFLAGS) -c $<

numerics.o: numerics.cpp numerics.h
	$(CXX) $(CXXFLAGS) -c $<

output.o: output.cpp output.h matplotlib-cpp/matplotlibcpp.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

parabola625.o: ../thermo/parabola625.c ../thermo/parabola625.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm -vf alloy625 $(OBJS)
