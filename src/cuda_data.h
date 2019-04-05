/**********************************************************************************
 HiPerC: High Performance Computing Strategies for Boundary Value Problems
 written by Trevor Keller and available from https://github.com/usnistgov/hiperc

 This software was developed at the National Institute of Standards and Technology
 by employees of the Federal Government in the course of their official duties.
 Pursuant to title 17 section 105 of the United States Code this software is not
 subject to copyright protection and is in the public domain. NIST assumes no
 responsibility whatsoever for the use of this software by other parties, and makes
 no guarantees, expressed or implied, about its quality, reliability, or any other
 characteristic. We would appreciate acknowledgement if the software is used.

 This software can be redistributed and/or modified freely provided that any
 derivative works bear some notice that they are derived from it, and any modified
 versions bear some notice that they have been modified.

 Questions/comments to Trevor Keller (trevor.keller@nist.gov)
 **********************************************************************************/

/**
 \file  cuda_data.h
 \brief Declaration of CUDA data container
*/

/** \cond SuppressGuard */
#ifndef _CUDA_DATA_H_
#define _CUDA_DATA_H_
/** \endcond */

#include "type.h"

/**
 \brief Initialize CUDA device memory before marching
*/
void init_cuda(struct HostData* host,
               const int nx, const int ny, const int nm,
               struct CudaData* dev);
/**
 \brief Free CUDA device memory after marching
*/
void free_cuda(struct CudaData* dev);

/**
   \brief Compute number of tiles along an axis
*/
float nTiles(int domain_size, int tile_loc, int mask_size);

/**
 \brief Apply boundary conditions to fields on device
*/
void device_boundaries(struct CudaData* dev,
                       const int nx, const int ny, const int nm,
                       const int bx, const int by);

/**
   \brief Apply boundary conditions to Laplacian fields on device
*/
void device_laplacian_boundaries(struct CudaData* dev,
                                 const int nx, const int ny, const int nm,
                                 const int bx, const int by);

/**
   \brief Update Laplacian fields on device
*/
void device_laplacian(struct CudaData* dev,
                      const int nx, const int ny, const int nm,
                      const int bx, const int by);

/**
 \brief Step equations of motion to update fields on device
*/
void device_evolution(struct CudaData* dev,
                      const int nx, const int ny, const int nm,
                      const int bx, const int by,
                      const fp_t* D_Cr, const fp_t* D_Nb,
                      const fp_t alpha, const fp_t kappa, const fp_t omega,
                      const fp_t M_del, const fp_t M_lav,
                      const fp_t dt);

/**
   \brief Initialize PRNG on device
*/
void device_init_prng(struct CudaData* dev,
                      const int nx, const int ny, const int nm,
                      const int bx, const int by);

/**
 \brief Stochastically seed nuclei on device
*/
void device_nucleation(struct CudaData* dev,
                       const int nx, const int ny, const int nm,
                       const int bx, const int by,
                       const fp_t* D_Cr, const fp_t* D_Nb,
                       const fp_t sigma_del, const fp_t sigma_lav,
                       const fp_t unit_a, const fp_t ifce_width,
                       const fp_t dx, const fp_t dy, const fp_t dz,
                       const fp_t dt);

/**
 \brief Update fictitious composition fields on device
*/
void device_fictitious(struct CudaData* dev,
                       const int nx, const int ny, const int nm,
                       const int bx, const int by);

/**
  \brief Copy fields from device to host
*/
void read_out_result(struct CudaData* dev, struct HostData* host, const int nx, const int ny);

/** \cond SuppressGuard */
#endif /* _CUDA_DATA_H_ */
/** \endcond */
