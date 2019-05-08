# GNU Makefile for Ni-base superalloy evolution using solid-state
# multi-phase-field methods in C++ with OpenMP and CUDA.
# Questions/comments to trevor.keller@nist.gov (Trevor Keller)

# Note: The project will not build unless you have exported
#       CUDA_HDR_PATH - directory containing curand_kernel.h
#       MMSP_PATH     - root directory of cloned MMSP repository

# If matplotlib is desired, then un-comment this block.
PLT_SRC = -Imatplotlib-cpp
PLT_LIB = -I/usr/include/python2.7
PLT_LNK = -L/usr/lib/python2.7 -lpython2.7
PLTFLAGS = $(PLT_SRC) $(PLT_LIB)

# If VTK is desired, then un-comment this block.
# VTK_SRC = -I $(shell dirname $(shell locate --limit 1 vtkImageData.h))
# VTK_LIB = -L $(shell dirname $(shell locate --limit 1 vtkCommonCore.so))
# VTK_LNK = -lvtkCommonCore -lvtkCommonDataModel -lvtkIOCore -lvtkIOXML
# VTKFLAGS = $(VTK_SRC) $(VTK_LIB)

CXX = g++ -O3
CXXFLAGS = -Wall -std=c++11 -funroll-loops -ffast-math -fopenmp \
           -I. -I../thermo -I$(MMSP_PATH)/include \
           -I$(CUDA_HDR_PATH) $(PLTFLAGS)

NVCXX = nvcc
NVCXXFLAGS = -std=c++11 -D_FORCE_INLINES -Wno-deprecated-gpu-targets \
             --compiler-options="$(CXXFLAGS)"

LINKS = -lcuda -lcurand -lm -lpng $(PLT_LNK) -lz

OBJS = data.o discretization.o d_nucleation.o d_parabola625.o \
       mesh.o enrichment.o numerics.o output.o parabola625.o

# Executable
alloy625: alloy625.cpp globals.h main.cpp $(OBJS)
	$(NVCXX) $(NVCXXFLAGS) $(OBJS) $< -o $@ $(LINKS)

profile: CXXOPT = -O1 -g
profile: CXXFLAGS += $(CXXOPT)
profile: NVCXXFLAGS += $(CXXOPT) -lineinfo # -arch=$(CUDARCH)
profile: alloy625

# CUDA objects

data.o: cuda_data.cu cuda_data.h
	$(NVCXX) $(NVCXXFLAGS) -dc $< -o $@

discretization.o: cuda_discretization.cu globals.h cuda_kernels.cuh parabola625.cu parabola625.cuh
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

numerics.o: numerics.cpp numerics.h
	$(CXX) $(CXXFLAGS) -c $<

output.o: output.cpp output.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

parabola625.o: ../thermo/parabola625.c ../thermo/parabola625.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm -f alloy625 $(OBJS)
