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
 \file  cuda_kernels.cuh
 \brief Declaration of functions to execute on the GPU (CUDA kernels)
*/

/** \cond SuppressGuard */
#ifndef _CUDA_KERNELS_H_
#define _CUDA_KERNELS_H_
/** \endcond */

#include "numerics.h"

/**
 \brief Convolution mask array on the GPU, allocated in protected memory
*/
__constant__ extern fp_t d_mask[MAX_MASK_W * MAX_MASK_H];

/**
 \brief Boundary condition kernel for execution on the GPU
*/
__global__ void boundary_kernel(fp_t* d_conc_Cr, fp_t* d_conc_Nb,
                                fp_t* d_phi_del,
                                fp_t* d_phi_lav,
                                fp_t* d_gam_Cr, fp_t* d_gam_Nb,
                                const int nx,
                                const int ny,
                                const int nm);

/**
 \brief Tiled convolution algorithm for execution on the GPU
*/
__global__ void convolution_kernel(fp_t* d_conc_old, fp_t* d_conc_new,
                                   const int nx,
                                   const int ny,
                                   const int nm);

/**
 \brief Device kernel to update field variables for composition
*/
__global__ void composition_kernel(fp_t* d_conc_Cr_old, fp_t* d_conc_Nb_old,
                                   fp_t* d_phi_del_old,
                                   fp_t* d_phi_lav_old,
                                   fp_t* d_gam_Cr_old,  fp_t* d_gam_Nb_old,
                                   fp_t* d_conc_Cr_new, fp_t* d_conc_Nb_new,
                                   fp_t* d_gam_Cr_new,  fp_t* d_gam_Nb_new,
                                   const int nx, const int ny, const int nm,
                                   const fp_t D_CrCr, const fp_t D_CrNb,
                                   const fp_t D_NbCr, const fp_t D_NbNb,
                                   const fp_t dt);

/**
 \brief Device kernel to update field variables for delta phase
*/
__global__ void delta_kernel(fp_t* d_conc_Cr_old, fp_t* d_conc_Nb_old,
                             fp_t* d_phi_del_old,
                             fp_t* d_phi_lav_old,
                             fp_t* d_gam_Cr_old,  fp_t* d_gam_Nb_old,
                             fp_t* d_phi_del_new,
                             const int nx, const int ny, const int nm,
                             const fp_t alpha, const fp_t kappa, const fp_t omega,
                             const fp_t M_del, const fp_t M_lav,
                             const fp_t dt);

/**
 \brief Device kernel to update field variables for Laves phase
*/
__global__ void laves_kernel(fp_t* d_conc_Cr_old, fp_t* d_conc_Nb_old,
                             fp_t* d_phi_del_old,
                             fp_t* d_phi_lav_old,
                             fp_t* d_gam_Cr_old,  fp_t* d_gam_Nb_old,
                             fp_t* d_phi_lav_new,
                             const int nx, const int ny, const int nm,
                             const fp_t alpha, const fp_t kappa, const fp_t omega,
                             const fp_t M_del, const fp_t M_lav,
                             const fp_t dt);

/**
 \brief Device kernel to update field variables for fictitious composition
*/
__global__ void fictitious_kernel(fp_t* d_conc_Cr_new, fp_t* d_conc_Nb_new,
                                  fp_t* d_phi_del_new,
                                  fp_t* d_phi_lav_new,
                                  fp_t* d_gam_Cr_new,  fp_t* d_gam_Nb_new,
                                  const int nx, const int ny, const int nm);

/** \cond SuppressGuard */
#endif /* _CUDA_KERNELS_H_ */
/** \endcond */
