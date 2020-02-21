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
CXXFOPTS = -funroll-loops -ffast-math -fopenmp
CXXFLAGS = -O3 -Wall -std=c++11 $(CXXFOPTS) $(INCLUDES)

NVCXX = nvcc
NVCXXFLAGS = -std=c++11 -D_FORCE_INLINES -Wno-deprecated-gpu-targets \
             $(INCLUDES) --compiler-options="$(CXXFLAGS)"

LINKS = -lcuda -lcurand -lgsl -lgslcblas -lm $(PLT_LNK) -lz

OBJS = data_cu.o discretization_cu.o nucleation_cu.o parabola625_cu.o \
       enrichment_c.o nucleation_c.o parabola625_c.o \
       mesh_cpp.o output_cpp.o phasefrac_cpp.o

# Executables

alloy625: alloy625.cpp parameters.h main.cpp $(OBJS)
	$(NVCXX) $(NVCXXFLAGS) $(OBJS) $< -o $@ $(LINKS)

debug: CXXOPT = -O0 -g -DPLANAR
debug: CXXFLAGS += $(CXXOPT)
debug: NVCXXFLAGS += $(CXXOPT) -lineinfo
debug: alloy625

profile: CXXOPT = -O1 -pg -DPLANAR
profile: CXXFLAGS += $(CXXOPT)
profile: NVCXXFLAGS += $(CXXOPT) -lineinfo
profile: alloy625

# Initial Condition Flags

planar: CXXFLAGS += -DPLANAR
planar: NVCXXFLAGS += -DPLANAR
planar: alloy625

pair: CXXFLAGS += -DPAIR
pair: NVCXXFLAGS += -DPAIR
pair: alloy625

nucleation: CXXFLAGS += -DNUCLEATION
nucleation: NVCXXFLAGS += -DNUCLEATION
nucleation: alloy625

pair: CXXFLAGS += -DTANH
tanh: NVCXXFLAGS += -DTANH
tanh: alloy625

# Derived source code
nucleation.cu: energy_functions.sh ../thermo/nucleation.c
	bash $<

parabola625.cu: energy_functions.sh ../thermo/parabola625.c
	bash $<

# CUDA objects
%_cu.o: %.cu %.cuh
	$(NVCXX) $(NVCXXFLAGS) -dc $< -o $@

# Host objects
%_cpp.o: %.cpp %.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

%_c.o: ../thermo/%.c ../thermo/%.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm -vf alloy625 $(OBJS)
