#!/bin/bash
# Prepend CUDA directives on SymPy functions

cp ../parabola625.h parabola625.h
cp ../parabola625.c parabola625.cpp

sed "s/^double /__device__ double d_/g" ../parabola625.h > parabola625.cuh
sed "s/^double /__device__ double d_/g" ../parabola625.c > parabola625.cu

sed -i -e "s/pow(XCR, 2)/XCR*XCR/g" \
    -e "s/pow(XNB, 2)/XNB*XNB/g" \
    -e "s/pow(x, 2)/x*x/g" \
    -e "s/pow(x, 3)/x*x*x/g" \
       -e "s/pow(f_gam, 2)/f_gam*f_gam/g" \
       -e "s/pow(f_del, 2)/f_del*f_del/g" \
       -e "s/pow(f_lav, 2)/f_lav*f_lav/g" \
       parabola625.cu
