#!/bin/bash
# Prepend CUDA directives on SymPy functions

sed -e "s/^void /__device__ void d_/g" \
    -e "s/NUCLEATION/D_NUCLEATION/g" \
    -e "s/^fp_t /__device__ fp_t d_/g" \
    ../thermo/nucleation.h > nucleation.cuh
astyle --style=linux --indent-col1-comments --indent=tab \
       --indent-preprocessor --indent-preproc-cond --pad-header \
       --align-pointer=type --keep-one-line-blocks --suffix=none \
       nucleation.cuh

sed -e "s/^double /__device__ double d_/g" \
    -e "s/^void /__device__ void d_/g" \
    -e "s/^fp_t /__device__ fp_t d_/g" \
    -e "s/ g_/ d_g_/g" \
    -e "s/ dg_/ d_dg_/g" \
    -e "s/xe_/ d_xe_/g" \
    -e "s/kT/ d_kT/g" \
    -e "s/.h\"/.cuh\"/g" \
    -e "s/.0L/.0/g" \
    ../thermo/nucleation.c > nucleation.cu
sed -e "s/globals.cuh/globals.h/g" \
    -i nucleation.cu
astyle --style=linux --indent-col1-comments --indent=tab \
       --indent-preprocessor --indent-preproc-cond --pad-header \
       --align-pointer=type --keep-one-line-blocks --suffix=none \
       nucleation.cu


sed -e "s/^double /__device__ double d_/g" \
    -e "s/PRECIPITATEAGING/D_PRECIPITATEAGING/g" \
    ../thermo/parabola625.h > parabola625.cuh
sed -e "s/^double /__device__ double d_/g" \
    -e "s/.h\"/.cuh\"/g" \
    -e "s/.0L/.0/g" \
    ../thermo/parabola625.c > parabola625.cu
sed -e 's/pow(\(f_[a-z]\{3\}\), 2)/\1*\1/g' \
    -e 's/pow(\([a-z]\), 2)/\1*\1/g' \
    -e 's/pow(\([a-z]\), 3)/\1*\1*\1/g' \
    -e 's/pow(\([[:print:]]\{1,8\}\) \([-+]\) \([[:print:]]\{1,8\}\), 2)/(\1 \2 \3)*(\1 \2 \3)/g' \
    -i parabola625.cu
astyle --style=linux --indent-col1-comments --indent=tab \
       --indent-preprocessor --indent-preproc-cond --pad-header \
       --align-pointer=type --keep-one-line-blocks --suffix=none \
       parabola625.cu
