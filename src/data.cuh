/**
 \file  data.cuh
 \brief Declaration of CUDA data container
*/

/** \cond SuppressGuard */
#ifndef _CUDA_DATA_H_
#define _CUDA_DATA_H_
/** \endcond */

#include "type.h"

/**
 \brief Handle device errors
 */
cudaError_t checkCuda(cudaError_t result);

/**
 \brief Initialize CUDA device memory before marching
*/
void init_cuda(struct HostData* host,
               const int nx, const int ny, const int nm,
               const fp_t* kappa, const fp_t* omega, const fp_t* Lmob,
               const fp_t* D_Cr, const fp_t* D_Nb,
               struct CudaData* dev);

/**
 \brief Free CUDA device memory after marching
*/
void free_cuda(struct CudaData* dev);

/**
   \brief Copy fields from device to host
*/
void read_out_result(struct CudaData* dev, struct HostData* host, const int nx, const int ny);

/** \cond SuppressGuard */
#endif /* _CUDA_DATA_H_ */
/** \endcond */
