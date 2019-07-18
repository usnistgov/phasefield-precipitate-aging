/**
 \file  discretization.cuh
 \brief Declaration of functions to execute on the GPU (CUDA kernels)
*/

/** \cond SuppressGuard */
#ifndef _CUDA_KERNELS_H_
#define _CUDA_KERNELS_H_
/** \endcond */

#include "mesh.h"

/**
 \brief Convolution mask array on the GPU, allocated in protected memory
*/
__constant__ extern fp_t d_mask[MAX_MASK_W * MAX_MASK_H];

/**
 \brief Diffusivity arrays on the GPU, allocated in protected memory
*/
__constant__ extern fp_t d_DCr[NC];
__constant__ extern fp_t d_DNb[NC];

/**
 \brief Kinetic parameter arrays on the GPU, allocated in protected memory
*/
__constant__ extern fp_t d_Kapp[NP];
__constant__ extern fp_t d_Omeg[NP];
__constant__ extern fp_t d_Lmob[NP];

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
                      const fp_t alpha,
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
   \brief Compute Ni composition and order parameter on device
*/
void device_dataviz(struct CudaData* dev,struct HostData* host,
                    const int nx, const int ny, const int nm,
                    const int bx, const int by);

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
__device__ void composition_kernel(const fp_t& d_conc_Cr_old, const fp_t& d_conc_Nb_old,
                                   const fp_t& d_lap_gam_Cr,  const fp_t& d_lap_gam_Nb,
                                   fp_t& d_conc_Cr_new,       fp_t& d_conc_Nb_new,
                                   const fp_t D_CrCr,        const fp_t D_CrNb,
                                   const fp_t D_NbCr,        const fp_t D_NbNb,
                                   const fp_t dt);

/**
 \brief Device kernel to update field variables for delta phase
*/
__device__ void delta_kernel(const fp_t& conc_Cr_old, const fp_t& conc_Nb_old,
                             const fp_t& phi_del_old, const fp_t& phi_lav_old,
                             fp_t& phi_del_new,
                             const fp_t inv_fict_det,
                             const fp_t f_del,       const fp_t f_lav,
                             const fp_t dgGdxCr,     const fp_t dgGdxNb,
                             const fp_t gam_Cr,      const fp_t gam_Nb,
                             const fp_t gam_nrg,     const fp_t alpha,
                             const fp_t dt);

/**
 \brief Device kernel to update field variables for Laves phase
*/
__device__ void laves_kernel(const fp_t& conc_Cr_old, const fp_t& conc_Nb_old,
                             const fp_t& phi_del_old, const fp_t& phi_lav_old,
                             fp_t& phi_lav_new,
                             const fp_t inv_fict_det,
                             const fp_t f_del,       const fp_t f_lav,
                             const fp_t dgGdxCr,     const fp_t dgGdxNb,
                             const fp_t gam_Cr,      const fp_t gam_Nb,
                             const fp_t gam_nrg,     const fp_t alpha,
                             const fp_t dt);

/**
 \brief Device kernel to update composition field variables
*/
__global__ void cahn_hilliard_kernel(fp_t* d_conc_Cr_old, fp_t* d_conc_Nb_old,
                                     fp_t* d_lap_gam_Cr,  fp_t* d_lap_gam_Nb,
                                     fp_t* d_conc_Cr_new, fp_t* d_conc_Nb_new,
                                     fp_t* d_gam_Cr,      fp_t* d_gam_Nb,
                                     const int nx, const int ny, const int nm,
                                     const fp_t dt);

/**
 \brief Device kernel to update phase field variables
*/
__global__ void allen_cahn_kernel(fp_t* d_conc_Cr_old, fp_t* d_conc_Nb_old,
                                  fp_t* d_phi_del_old, fp_t* d_phi_lav_old,
                                  fp_t* d_lap_gam_Cr,  fp_t* d_lap_gam_Nb,
                                  fp_t* d_phi_del_new, fp_t* d_phi_lav_new,
                                  fp_t* d_gam_Cr,      fp_t* d_gam_Nb,
                                  const int nx, const int ny, const int nm,
                                  const fp_t alpha,
                                  const fp_t dt);

/**
 \brief Device kernel to initialize random number generators
*/
__global__ void init_prng_kernel(curandState* prng, const int nx, const int ny);

/**
 \brief Device kernel to initialize supercritical nucleus
*/
__device__ void embed_OPC_kernel(fp_t* d_conc_Cr, fp_t* d_conc_Nb,
                                 fp_t* d_phi_del, fp_t* d_phi_lav,
                                 const int nx, const int ny,
                                 const int x, const int y, const int idx,
                                 const fp_t xCr,
                                 const fp_t xNb,
                                 const fp_t par_xe_Cr,
                                 const fp_t par_xe_Nb,
                                 const fp_t R_precip);

/**
 \brief Device kernel to compute driving force for nucleation and stochastically seed nuclei
*/
__global__ void nucleation_kernel(fp_t* d_conc_Cr, fp_t* d_conc_Nb,
                                  fp_t* d_phi_del, fp_t* d_phi_lav,
                                  const int nx, const int ny, const int nm,
                                  const int bx, const int by,
                                  const fp_t D_CrCr, const fp_t D_NbNb,
                                  const fp_t sigma_del, const fp_t sigma_lav,
                                  const fp_t unit_a, const fp_t ifce_width,
                                  const fp_t dx, const fp_t dy, const fp_t dt);
/**
 \brief Device kernel to update fictitious compositions in matrix phase
*/
__global__ void fictitious_kernel(fp_t* d_conc_Cr, fp_t* d_conc_Nb,
                                  fp_t* d_phi_del, fp_t* d_phi_lav,
                                  fp_t* d_gam_Cr,  fp_t* d_gam_Nb,
                                  const int nx,    const int ny);

__global__ void dataviz_kernel(fp_t* d_conc_Cr, fp_t* d_conc_Nb, fp_t* d_conc_Ni,
                               fp_t* d_ph_del, fp_t* d_phi_lav, fp_t* d_phi,
                               const int nx, const int ny);

/** \cond SuppressGuard */
#endif /* _CUDA_KERNELS_H_ */
/** \endcond */
