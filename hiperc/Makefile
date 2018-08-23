# Makefile for HiPerC spinodal code
# CUDA implementation

NVCXX = nvcc
CXX = g++ -fPIC
CXXFLAGS = -Wall -std=c++11 -funroll-loops -ffast-math -fopenmp -I. -I$(MMSP_PATH)/include
NVCXXFLAGS = -std=c++11 -D_FORCE_INLINES -Wno-deprecated-gpu-targets --compiler-options="$(CXXFLAGS)"
LINKS = -lcuda -lm -lpng -lz
OBJS = boundaries.o data.o discretization.o mesh.o numerics.o output.o d_parabola625.o parabola625.o

# Executable
cuda625: cuda625.cpp $(OBJS)
	$(NVCXX) $(NVCXXFLAGS) $(OBJS) $< -o $@ $(LINKS)

profile: CXXFLAGS += -O0 -g
profile: NVCXXFLAGS += -O0 -g -lineinfo -arch=$(CUDARCH)
profile: cuda625

# CUDA objects
boundaries.o: cuda_boundaries.cu
	$(NVCXX) $(NVCXXFLAGS) -dc $< -o $@

data.o: cuda_data.cu
	$(NVCXX) $(NVCXXFLAGS) -dc $< -o $@

discretization.o: cuda_discretization.cu
	$(NVCXX) $(NVCXXFLAGS) -dc $< -o $@

d_parabola625.o: parabola625.cu
	$(NVCXX) $(NVCXXFLAGS) -dc $< -o $@

# Common objects
mesh.o: mesh.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

numerics.o: numerics.cpp
	$(CXX) $(CXXFLAGS) -c $<

output.o: output.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

parabola625.o: parabola625.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm -f cuda625 *.o
