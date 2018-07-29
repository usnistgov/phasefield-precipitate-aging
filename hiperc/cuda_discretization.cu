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
 \file  cuda_discretization.cu
 \brief Implementation of boundary condition functions with CUDA acceleration
*/

#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <cuda.h>

extern "C" {
#include "cuda_data.h"
#include "numerics.h"
#include "mesh.h"
#include "timer.h"
}

#include "cuda_kernels.cuh"
#include "parabola625.cuh"

__constant__ fp_t d_mask[MAX_MASK_W * MAX_MASK_H];

__global__ void convolution_kernel(fp_t* d_conc_old, fp_t* d_conc_lap,
                                   const int nx, const int ny, const int nm)
{
	int dst_x, dst_y, dst_nx, dst_ny;
	int src_x, src_y, src_nx, src_ny;
	int til_x, til_y, til_nx;
	fp_t value=0.;

	/* source and tile width include the halo cells */
	src_nx = blockDim.x;
	src_ny = blockDim.y;
	til_nx = src_nx;

	/* destination width excludes the halo cells */
	dst_nx = src_nx - nm + 1;
	dst_ny = src_ny - nm + 1;

	/* determine tile indices on which to operate */
	til_x = threadIdx.x;
	til_y = threadIdx.y;

	dst_x = blockIdx.x * dst_nx + til_x;
	dst_y = blockIdx.y * dst_ny + til_y;

	src_x = dst_x - nm/2;
	src_y = dst_y - nm/2;

	/* copy tile: __shared__ gives access to all threads working on this tile */
	extern __shared__ fp_t d_conc_tile[];

	if (src_x >= 0 && src_x < nx &&
	    src_y >= 0 && src_y < ny ) {
		/* if src_y==0, then dst_y==nm/2: this is a halo row */
		d_conc_tile[til_nx * til_y + til_x] = d_conc_old[nx * src_y + src_x];
	}

	/* tile data is shared: wait for all threads to finish copying */
	__syncthreads();

	/* compute the convolution */
	if (til_x < dst_nx && til_y < dst_ny) {
		for (int j = 0; j < nm; j++) {
			for (int i = 0; i < nm; i++) {
				value += d_mask[j * nm + i] * d_conc_tile[til_nx * (til_y+j) + til_x+i];
			}
		}
		/* record value */
        /* Note: tile is centered on [til_nx*(til_y+nm/2) + (til_x+nm/2)], NOT [til_nx*til_y + til_x] */
		if (dst_y < ny && dst_x < nx) {
          d_conc_lap[nx * dst_y + dst_x] = value;
		}
	}

	/* wait for all threads to finish writing */
	__syncthreads();
}

__global__ void evolution_kernel(fp_t* d_conc_Cr_old, fp_t* d_conc_Nb_old,
                                 fp_t* d_phi_del_old,
                                 fp_t* d_phi_lav_old,
                                 fp_t* d_gam_Cr_old,  fp_t* d_gam_Nb_old,
                                 fp_t* d_conc_Cr_new, fp_t* d_conc_Nb_new,
                                 fp_t* d_phi_del_new,
                                 fp_t* d_phi_lav_new,
                                 fp_t* d_gam_Cr_new,  fp_t* d_gam_Nb_new,
                                 const int nx, const int ny, const int nm,
                                 const fp_t D_CrCr, const fp_t D_CrNb,
                                 const fp_t D_NbCr, const fp_t D_NbNb,
                                 const fp_t alpha, const fp_t kappa, const fp_t omega,
                                 const fp_t M_del, const fp_t M_lav,
                                 const fp_t dt)
{
	int thr_x, thr_y, x, y;

	/* determine indices on which to operate */
	thr_x = threadIdx.x;
	thr_y = threadIdx.y;

	x = blockDim.x * blockIdx.x + thr_x;
	y = blockDim.y * blockIdx.y + thr_y;

	/* explicit Euler solution to the equation of motion */
	if (x < nx && y < ny) {
      const fp_t xCr = d_conc_Cr_old[nx * thr_y + thr_x];
      const fp_t xNb = d_conc_Nb_old[nx * thr_y + thr_x];
      const fp_t phi_del = d_phi_del_old[nx * thr_y + thr_x];
      const fp_t phi_lav = d_phi_lav_old[nx * thr_y + thr_x];
      const fp_t f_del = h(phi_del);
      const fp_t f_lav = h(phi_lav);
      const fp_t f_gam = 1. - f_del - f_lav;

      /* compute fictitious compositions */
      const fp_t gam_Cr = d_gam_Cr_old[nx * thr_y + thr_x];
      const fp_t gam_Nb = d_gam_Nb_old[nx * thr_y + thr_x];
      const fp_t del_Cr = fict_del_Cr(xCr, xNb, f_del, f_gam, f_lav);
      const fp_t del_Nb = fict_del_Nb(xCr, xNb, f_del, f_gam, f_lav);
      const fp_t lav_Cr = fict_lav_Cr(xCr, xNb, f_del, f_gam, f_lav);
      const fp_t lav_Nb = fict_lav_Nb(xCr, xNb, f_del, f_gam, f_lav);

      /* pure phase energies */
      const fp_t gam_nrg = g_gam(gam_Cr, gam_Nb);
      const fp_t del_nrg = g_del(del_Cr, del_Nb);
      const fp_t lav_nrg = g_lav(lav_Cr, lav_Nb);

      /* effective chemical potential */
      const fp_t mu_Cr = dg_gam_dxCr(gam_Cr, gam_Nb);
      const fp_t mu_Nb = dg_gam_dxNb(gam_Cr, gam_Nb);

      /* pressure */
      const fp_t P_del = gam_nrg - del_nrg
                       - mu_Cr * (gam_Cr - del_Cr)
                       - mu_Nb * (gam_Nb - del_Nb);
      const fp_t P_lav = gam_nrg - lav_nrg
                       - mu_Cr * (gam_Cr - lav_Cr)
                       - mu_Nb * (gam_Nb - lav_Nb);

      /* variational derivatives */
      const fp_t dFdPhi_del = -hprime(phi_del) * P_del
                            + 2. * omega * phi_del * (phi_del - 1.) * (2. * phi_del - 1.)
                            + 2. * alpha * phi_del * (phi_lav * phi_lav)
                            - kappa * d_phi_del_new[nx * thr_y + thr_x];
      const fp_t dFdPhi_lav = -hprime(phi_lav) * P_lav
                            + 2. * omega * phi_lav * (phi_lav - 1.) * (2. * phi_lav - 1.)
                            + 2. * alpha * phi_lav * (phi_del * phi_del)
                            - kappa * d_phi_lav_new[nx * thr_y + thr_x];

      /* Cahn-Hilliard equations of motion for composition */
      d_conc_Cr_new[nx * thr_y + thr_x] = d_conc_Cr_old[nx * thr_y + thr_x]
                                        + dt * ( D_CrCr * d_gam_Cr_new[nx * thr_y + thr_x]
                                               + D_CrNb * d_gam_Nb_new[nx * thr_y + thr_x]);
      d_conc_Nb_new[nx * thr_y + thr_x] = d_conc_Nb_old[nx * thr_y + thr_x]
                                        + dt * ( D_NbCr * d_gam_Cr_new[nx * thr_y + thr_x]
                                               + D_NbNb * d_gam_Nb_new[nx * thr_y + thr_x]);

      /* Allen-Cahn equations of motion for phase */
      d_phi_del_new[nx * thr_y + thr_x] = d_phi_del_old[nx * thr_y + thr_x] - dt * M_del * dFdPhi_del;
      d_phi_lav_new[nx * thr_y + thr_x] = d_phi_lav_old[nx * thr_y + thr_x] - dt * M_lav * dFdPhi_lav;

      /* fictitious compositions */
      const fp_t f_del_new = h(d_phi_del_new[nx * thr_y + thr_x]);
      const fp_t f_lav_new = h(d_phi_lav_new[nx * thr_y + thr_x]);
      const fp_t f_gam_new = 1. - f_del - f_lav;
      d_conc_Cr_old[nx * thr_y + thr_x] = fict_gam_Cr(d_conc_Cr_old[nx * thr_y + thr_x],
                                                      d_conc_Cr_old[nx * thr_y + thr_x],
                                                      f_del_new, f_gam_new, f_lav_new);
      d_conc_Nb_old[nx * thr_y + thr_x] = fict_gam_Nb(d_conc_Cr_old[nx * thr_y + thr_x],
                                                      d_conc_Cr_old[nx * thr_y + thr_x],
                                                      f_del_new, f_gam_new, f_lav_new);
    }

	/* wait for all threads to finish writing */
	__syncthreads();
}

void device_boundaries(struct CudaData* dev,
                       const int nx, const int ny, const int nm,
                       const int bx, const int by)
{
	/* divide matrices into blocks of bx * by threads */
	dim3 tile_size(bx, by, 1);
	dim3 num_tiles(ceil(float(nx) / (tile_size.x - nm + 1)),
	               ceil(float(ny) / (tile_size.y - nm + 1)),
	               1);

	boundary_kernel<<<num_tiles,tile_size>>> (
        dev->conc_Cr_old, dev->conc_Nb_old,
        dev->phi_del_old,
        dev->phi_lav_old,
        dev->gam_Cr_old, dev->gam_Nb_old,
        nx, ny, nm
	);
}

void device_fict_boundaries(struct CudaData* dev,
                            const int nx, const int ny, const int nm,
                            const int bx, const int by)
{
	/* divide matrices into blocks of bx * by threads */
	dim3 tile_size(bx, by, 1);
	dim3 num_tiles(ceil(float(nx) / (tile_size.x - nm + 1)),
	               ceil(float(ny) / (tile_size.y - nm + 1)),
	               1);

	fict_boundary_kernel<<<num_tiles,tile_size>>> (
        dev->gam_Cr_new, dev->gam_Nb_new, nx, ny, nm);
}

void device_laplacian(struct CudaData* dev,
                      const int nx, const int ny, const int nm,
                      const int bx, const int by)
{
	/* divide matrices into blocks of bx * by threads */
	dim3 tile_size(bx, by, 1);
	dim3 num_tiles(ceil(float(nx) / (tile_size.x - nm + 1)),
	               ceil(float(ny) / (tile_size.y - nm + 1)),
	               1);
	size_t buf_size = (tile_size.x + nm) * (tile_size.y + nm) * sizeof(fp_t);

	convolution_kernel<<<num_tiles,tile_size,buf_size>>> (
    	dev->conc_Cr_old, dev->conc_Cr_new, nx, ny, nm);
	convolution_kernel<<<num_tiles,tile_size,buf_size>>> (
    	dev->conc_Nb_old, dev->conc_Nb_new, nx, ny, nm);

	convolution_kernel<<<num_tiles,tile_size,buf_size>>> (
    	dev->phi_del_old, dev->phi_del_new, nx, ny, nm);
	convolution_kernel<<<num_tiles,tile_size,buf_size>>> (
    	dev->phi_lav_old, dev->phi_lav_new, nx, ny, nm);

	convolution_kernel<<<num_tiles,tile_size,buf_size>>> (
    	dev->gam_Cr_old, dev->gam_Cr_new, nx, ny, nm);
	convolution_kernel<<<num_tiles,tile_size,buf_size>>> (
    	dev->gam_Nb_old, dev->gam_Nb_new, nx, ny, nm);
}

void device_evolution(struct CudaData* dev,
                      const int nx, const int ny, const int nm,
                      const int bx, const int by,
                      const fp_t D_CrCr, const fp_t D_CrNb,
                      const fp_t D_NbCr, const fp_t D_NbNb,
                      const fp_t alpha, const fp_t kappa, const fp_t omega,
                      const fp_t M_del, const fp_t M_lav,
                      const fp_t dt)
{
	/* divide matrices into blocks of bx * by threads */
	dim3 tile_size(bx, by, 1);
	dim3 num_tiles(ceil(float(nx) / (tile_size.x - nm + 1)),
	               ceil(float(ny) / (tile_size.y - nm + 1)),
	               1);
	evolution_kernel<<<num_tiles,tile_size>>> (
	    dev->conc_Cr_old, dev->conc_Nb_old,
        dev->phi_del_old, dev->phi_lav_old,
        dev->gam_Cr_old, dev->gam_Nb_old,
        dev->conc_Cr_new, dev->conc_Nb_new,
        dev->phi_del_new, dev->phi_lav_new,
        dev->gam_Cr_new, dev->gam_Nb_new,
        nx, ny, nm,
        D_CrCr, D_CrNb,
        D_NbCr, D_NbNb,
        alpha, kappa, omega,
        M_del, M_lav,
        dt);
}

void read_out_result(struct CudaData* dev, struct HostData* host, const int nx, const int ny)
{
  cudaMemcpy(host->conc_Cr_new, dev->conc_Cr_old, nx * ny * sizeof(fp_t),
               cudaMemcpyDeviceToHost);
  cudaMemcpy(host->conc_Nb_new, dev->conc_Nb_old, nx * ny * sizeof(fp_t),
               cudaMemcpyDeviceToHost);
  cudaMemcpy(host->phi_del_new, dev->phi_del_old, nx * ny * sizeof(fp_t),
               cudaMemcpyDeviceToHost);
  cudaMemcpy(host->phi_lav_new, dev->phi_lav_old, nx * ny * sizeof(fp_t),
               cudaMemcpyDeviceToHost);
  cudaMemcpy(host->gam_Cr_new, dev->gam_Cr_old, nx * ny * sizeof(fp_t),
               cudaMemcpyDeviceToHost);
  cudaMemcpy(host->gam_Nb_new, dev->gam_Nb_old, nx * ny * sizeof(fp_t),
               cudaMemcpyDeviceToHost);
}
