#!/bin/bash
# Prepend CUDA directives on SymPy functions

sed "s/^double/__device__ double/g" ../parabola625.h > parabola625.cuh
sed "s/^double/__device__ double/g" ../parabola625.c > parabola625.cu
