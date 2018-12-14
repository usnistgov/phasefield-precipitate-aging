#!/bin/bash
# Prepend CUDA directives on SymPy functions

cp ../src/parabola625.h parabola625.h
cp ../src/parabola625.c parabola625.cpp

sed "s/^double /__device__ double d_/g" parabola625.h   > parabola625.cuh
sed "s/^double /__device__ double d_/g" parabola625.cpp > parabola625.cu

sed -e 's/pow(\(f_[a-z]\{3\}\), 2)/\1*\1/g' \
    -e 's/pow(\([a-z]\), 2)/\1*\1/g' \
    -e 's/pow(\([a-z]\), 3)/\1*\1*\1/g' \
    -e 's/pow(\([[:print:]]\{1,8\}\) \([-+]\) \([[:print:]]\{1,8\}\), 2)/(\1 \2 \3)*(\1 \2 \3)/g' \
    -i parabola625.cu
