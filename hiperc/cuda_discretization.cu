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

#include "cuda_data.h"
#include "numerics.h"
#include "mesh.h"

#include "cuda_kernels.cuh"
#include "parabola625.cuh"

__constant__ fp_t d_mask[MAX_MASK_W * MAX_MASK_H];

__global__ void convolution_kernel(fp_t* d_conc_old, fp_t* d_conc_new,
                                   const int nx, const int ny, const int nm)
{
	/* source and tile width include the halo cells */
	const int src_nx = blockDim.x;
	const int src_ny = blockDim.y;
	const int til_nx = src_nx;

	/* destination width excludes the halo cells */
	const int dst_nx = src_nx - nm + 1;
	const int dst_ny = src_ny - nm + 1;

	/* determine tile indices on which to operate */
	const int til_x = threadIdx.x;
	const int til_y = threadIdx.y;

	const int dst_x = blockIdx.x * dst_nx + til_x;
	const int dst_y = blockIdx.y * dst_ny + til_y;

	const int src_x = dst_x - nm/2;
	const int src_y = dst_y - nm/2;

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
		fp_t value = 0.;
		for (int j = 0; j < nm; j++) {
			for (int i = 0; i < nm; i++) {
				value += d_mask[j * nm + i] * d_conc_tile[til_nx * (til_y+j) + til_x+i];
			}
		}
		/* record value */
		/* Note: tile is centered on [til_nx*(til_y+nm/2) + (til_x+nm/2)], NOT [til_nx*til_y + til_x] */
		if (dst_y < ny && dst_x < nx) {
			d_conc_new[nx * dst_y + dst_x] = value;
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
	/* determine indices on which to operate */
	const int thr_x = threadIdx.x;
	const int thr_y = threadIdx.y;

	const int x = blockDim.x * blockIdx.x + thr_x;
	const int y = blockDim.y * blockIdx.y + thr_y;

	/* explicit Euler solution to the equation of motion */
	if (x < nx && y < ny) {
		const int idx = nx * y + x;

		const fp_t xCr = d_conc_Cr_old[idx];
		const fp_t xNb = d_conc_Nb_old[idx];
		const fp_t phi_del = d_phi_del_old[idx];
		const fp_t phi_lav = d_phi_lav_old[idx];
		const fp_t f_del = d_h(phi_del);
		const fp_t f_lav = d_h(phi_lav);
		const fp_t f_gam = 1. - f_del - f_lav;

		const fp_t lap_Cr = d_gam_Cr_new[idx];
		const fp_t lap_Nb = d_gam_Nb_new[idx];
		const fp_t lap_del = d_phi_del_new[idx];
		const fp_t lap_lav = d_phi_lav_new[idx];

		/* compute fictitious compositions */
		const fp_t gam_Cr = d_gam_Cr_old[idx];
		const fp_t gam_Nb = d_gam_Nb_old[idx];
		const fp_t del_Cr = d_fict_del_Cr(xCr, xNb, f_del, f_gam, f_lav);
		const fp_t del_Nb = d_fict_del_Nb(xCr, xNb, f_del, f_gam, f_lav);
		const fp_t lav_Cr = d_fict_lav_Cr(xCr, xNb, f_del, f_gam, f_lav);
		const fp_t lav_Nb = d_fict_lav_Nb(xCr, xNb, f_del, f_gam, f_lav);

		/* pure phase energies */
		const fp_t gam_nrg = d_g_gam(gam_Cr, gam_Nb);
		const fp_t del_nrg = d_g_del(del_Cr, del_Nb);
		const fp_t lav_nrg = d_g_lav(lav_Cr, lav_Nb);

		/* effective chemical potential */
		const fp_t dgGdxCr = d_dg_gam_dxCr(gam_Cr, gam_Nb);
		const fp_t dgGdxNb = d_dg_gam_dxNb(gam_Cr, gam_Nb);

		/* pressure */
        const fp_t sumPhiSq = phi_del * phi_del + phi_lav * phi_lav;

		const fp_t P_del = gam_nrg - del_nrg
		                 - dgGdxCr * (gam_Cr - del_Cr)
		                 - dgGdxNb * (gam_Nb - del_Nb);
		const fp_t P_lav = gam_nrg - lav_nrg
		                 - dgGdxCr * (gam_Cr - lav_Cr)
		                 - dgGdxNb * (gam_Nb - lav_Nb);

		/* variational derivatives */
		const fp_t dFdPhi_del = -d_hprime(phi_del) * P_del
		                      + 2. * omega * phi_del * (phi_del - 1.) * (2. * phi_del - 1.)
		                      + 2. * alpha * phi_del * (sumPhiSq - phi_del * phi_del)
		                      - kappa * lap_del;
		const fp_t dFdPhi_lav = -d_hprime(phi_lav) * P_lav
		                      + 2. * omega * phi_lav * (phi_lav - 1.) * (2. * phi_lav - 1.)
		                      + 2. * alpha * phi_lav * (sumPhiSq - phi_lav * phi_lav)
		                      - kappa * lap_lav;

		/* Cahn-Hilliard equations of motion for composition */
		const fp_t lap_mu_Cr = D_CrCr * lap_Cr
		                     + D_NbCr * lap_Nb;
		const fp_t lap_mu_Nb = D_CrNb * lap_Cr
		                     + D_NbNb * lap_Nb;

		const fp_t conc_Cr_new = xCr + dt * lap_mu_Cr;
		const fp_t conc_Nb_new = xNb + dt * lap_mu_Nb;
		d_conc_Cr_new[idx] = conc_Cr_new;
		d_conc_Nb_new[idx] = conc_Nb_new;

		/* Allen-Cahn equations of motion for phase */
		const fp_t phi_del_new = phi_del - dt * M_del * dFdPhi_del;
		const fp_t phi_lav_new = phi_lav - dt * M_lav * dFdPhi_lav;
		d_phi_del_new[idx] = phi_del_new;
		d_phi_lav_new[idx] = phi_lav_new;

		/* fictitious compositions */
		const fp_t f_del_new = d_h(phi_del_new);
		const fp_t f_lav_new = d_h(phi_lav_new);
		const fp_t f_gam_new = 1. - f_del_new - f_lav_new;
		d_gam_Cr_new[idx] = d_fict_gam_Cr(conc_Cr_new, conc_Nb_new,
		                                  f_del_new, f_gam_new, f_lav_new);
		d_gam_Nb_new[idx] = d_fict_gam_Nb(conc_Cr_new, conc_Nb_new,
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
	cudaMemcpy(host->conc_Cr_new[0], dev->conc_Cr_old, nx * ny * sizeof(fp_t),
	           cudaMemcpyDeviceToHost);
	cudaMemcpy(host->conc_Nb_new[0], dev->conc_Nb_old, nx * ny * sizeof(fp_t),
	           cudaMemcpyDeviceToHost);
	cudaMemcpy(host->phi_del_new[0], dev->phi_del_old, nx * ny * sizeof(fp_t),
	           cudaMemcpyDeviceToHost);
	cudaMemcpy(host->phi_lav_new[0], dev->phi_lav_old, nx * ny * sizeof(fp_t),
	           cudaMemcpyDeviceToHost);
	cudaMemcpy(host->gam_Cr_new[0], dev->gam_Cr_old, nx * ny * sizeof(fp_t),
	           cudaMemcpyDeviceToHost);
	cudaMemcpy(host->gam_Nb_new[0], dev->gam_Nb_old, nx * ny * sizeof(fp_t),
	           cudaMemcpyDeviceToHost);
}
