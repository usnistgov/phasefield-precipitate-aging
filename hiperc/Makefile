# Makefile for HiPerC spinodal code
# CUDA implementation

NVCXX = nvcc
        NVCXXFLAGS = -D_FORCE_INLINES -Wno-deprecated-gpu-targets -std=c++11 -I$(MMSP_PATH)/include \
                     --compiler-options="-O3 -Wall -ffast-math -fopenmp -funroll-loops"
                                        LINKS = -lcuda -lm -lpng -lz

                                                OBJS = boundaries.o data.o discretization.o mesh.o numerics.o output.o d_parabola625.o parabola625.o

# Executable
                                                        cuda625:
                                                        cuda625.cpp $(OBJS)
                                                        $(NVCXX) $(NVCXXFLAGS) $(OBJS) $< -o $@ $(LINKS)

# CUDA objects

                                                        boundaries.o:
                                                        cuda_boundaries.cu
                                                        $(NVCXX) $(NVCXXFLAGS) -dc $< -o $@

                                                        data.o:
                                                        cuda_data.cu
                                                        $(NVCXX) $(NVCXXFLAGS) -dc $< -o $@

                                                        discretization.o:
                                                        cuda_discretization.cu
                                                        $(NVCXX) $(NVCXXFLAGS) -dc $< -o $@

                                                        d_parabola625.o:
                                                        parabola625.cu
                                                        $(NVCXX) $(NVCXXFLAGS) -dc $< -o $@

# Common objects

                                                        mesh.o:
                                                        mesh.c
                                                        $(NVCXX) $(NVCXXFLAGS) -c $< -o $@

                                                        numerics.o:
                                                        numerics.c
                                                        $(NVCXX) $(NVCXXFLAGS) -c $<

                                                        output.o:
                                                        output.c
                                                        $(NVCXX) $(NVCXXFLAGS) -c $< -o $@

                                                        parabola625.o:
                                                        parabola625.c
                                                        $(NVCXX) $(NVCXXFLAGS) -c $< -o $@

                                                        .PHONY:
                                                        cleanobjects
                                                        cleanobjects:
                                                        rm -f cuda625 *.o

                                                        .PHONY:
                                                        clean
                                                        clean:
                                                        cleanobjects
