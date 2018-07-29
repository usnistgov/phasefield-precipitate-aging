#!/bin/bash
# Prepend CUDA directives on SymPy functions

cp ../parabola625.h parabola625.h
cp ../parabola625.c parabola625.cpp

sed "s/^double /__device__ double d_/g" ../parabola625.h > parabola625.cuh
sed "s/^double /__device__ double d_/g" ../parabola625.c > parabola625.cu
