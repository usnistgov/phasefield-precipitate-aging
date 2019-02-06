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
#include <curand.h>
#include <curand_kernel.h>

#include "cuda_data.h"
#include "numerics.h"
#include "mesh.h"

#include "cuda_kernels.cuh"
#include "parabola625.cuh"

__constant__ fp_t d_mask[MAX_MASK_W * MAX_MASK_H];

float nTiles(int domain_size, int tile_loc, int mask_size)
{
	return ceil(float(domain_size) / float(tile_loc - mask_size + 1));
}

__global__ void boundary_kernel(fp_t* d_conc_Cr, fp_t* d_conc_Nb,
                                fp_t* d_phi_del, fp_t* d_phi_lav,
                                fp_t* d_gam_Cr,  fp_t* d_gam_Nb,
                                const int nx,
                                const int ny,
                                const int nm)
{
	/* determine indices on which to operate */
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;

	const int row = blockDim.y * blockIdx.y + ty;
	const int col = blockDim.x * blockIdx.x + tx;

	/* apply no-flux boundary conditions: inside to out, sequence matters */

	for (int offset = 0; offset < nm/2; offset++) {
		const int ilo = nm/2 - offset;
		const int ihi = nx - 1 - nm/2 + offset;
		const int jlo = nm/2 - offset;
		const int jhi = ny - 1 - nm/2 + offset;

		if (ilo-1 == col && row < ny) {
			/* left condition */
			d_conc_Cr[row * nx + ilo-1] = d_conc_Cr[row * nx + ilo];
			d_conc_Nb[row * nx + ilo-1] = d_conc_Nb[row * nx + ilo];
			d_phi_del[row * nx + ilo-1] = d_phi_del[row * nx + ilo];
			d_phi_lav[row * nx + ilo-1] = d_phi_lav[row * nx + ilo];
			d_gam_Cr[ row * nx + ilo-1] = d_gam_Cr[ row * nx + ilo];
			d_gam_Nb[ row * nx + ilo-1] = d_gam_Nb[ row * nx + ilo];
		}
		if (ihi+1 == col && row < ny) {
			/* right condition */
			d_conc_Cr[row * nx + ihi+1] = d_conc_Cr[row * nx + ihi];
			d_conc_Nb[row * nx + ihi+1] = d_conc_Nb[row * nx + ihi];
			d_phi_del[row * nx + ihi+1] = d_phi_del[row * nx + ihi];
			d_phi_lav[row * nx + ihi+1] = d_phi_lav[row * nx + ihi];
			d_gam_Cr[ row * nx + ihi+1] = d_gam_Cr[ row * nx + ihi];
			d_gam_Nb[ row * nx + ihi+1] = d_gam_Nb[ row * nx + ihi];
		}
		if (jlo-1 == row && col < nx) {
			/* bottom condition */
			d_conc_Cr[(jlo-1) * nx + col] = d_conc_Cr[jlo * nx + col];
			d_conc_Nb[(jlo-1) * nx + col] = d_conc_Nb[jlo * nx + col];
			d_phi_del[(jlo-1) * nx + col] = d_phi_del[jlo * nx + col];
			d_phi_lav[(jlo-1) * nx + col] = d_phi_lav[jlo * nx + col];
			d_gam_Cr[ (jlo-1) * nx + col] = d_gam_Cr[ jlo * nx + col];
			d_gam_Nb[ (jlo-1) * nx + col] = d_gam_Nb[ jlo * nx + col];
		}
		if (jhi+1 == row && col < nx) {
			/* top condition */
			d_conc_Cr[(jhi+1) * nx + col] = d_conc_Cr[jhi * nx + col];
			d_conc_Nb[(jhi+1) * nx + col] = d_conc_Nb[jhi * nx + col];
			d_gam_Cr[ (jhi+1) * nx + col] = d_gam_Cr[ jhi * nx + col];
			d_gam_Nb[ (jhi+1) * nx + col] = d_gam_Nb[ jhi * nx + col];
			d_phi_del[(jhi+1) * nx + col] = d_phi_del[jhi * nx + col];
			d_phi_lav[(jhi+1) * nx + col] = d_phi_lav[jhi * nx + col];
		}

		__syncthreads();
	}
}

void device_boundaries(struct CudaData* dev,
                       const int nx, const int ny, const int nm,
                       const int bx, const int by)
{
	/* divide matrices into blocks of bx * by threads */
	dim3 tile_size(bx, by, 1);
	dim3 num_tiles(nTiles(nx, tile_size.x, nm),
	               nTiles(ny, tile_size.y, nm),
	               1);

	boundary_kernel<<<num_tiles,tile_size>>> (
	    dev->conc_Cr_old, dev->conc_Nb_old,
	    dev->phi_del_old, dev->phi_lav_old,
	    dev->gam_Cr,      dev->gam_Nb,
	    nx, ny, nm
	);
}

void device_laplacian_boundaries(struct CudaData* dev,
                                 const int nx, const int ny, const int nm,
                                 const int bx, const int by)
{
	/* divide matrices into blocks of bx * by threads */
	dim3 tile_size(bx, by, 1);
	dim3 num_tiles(nTiles(nx, tile_size.x, nm),
	               nTiles(ny, tile_size.y, nm),
	               1);

	boundary_kernel<<<num_tiles,tile_size>>> (
	    dev->conc_Cr_new, dev->conc_Nb_new,
	    dev->phi_del_new, dev->phi_lav_new,
	    dev->lap_gam_Cr,  dev->lap_gam_Nb,
	    nx, ny, nm
	);
}

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
		/* Note: tile is centered on [til_nx*(til_y+nm/2) + (til_x+nm/2)],
		         NOT [til_nx*til_y + til_x] */
		if (dst_y < ny && dst_x < nx) {
			d_conc_new[nx * dst_y + dst_x] = value;
		}
	}

	__syncthreads();
}

void device_laplacian(struct CudaData* dev,
                      const int nx, const int ny, const int nm,
                      const int bx, const int by)
{
	/* divide matrices into blocks of bx * by threads */
	dim3 tile_size(bx, by, 1);
	dim3 num_tiles(nTiles(nx, tile_size.x, nm),
	               nTiles(ny, tile_size.y, nm),
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
	    dev->gam_Cr, dev->lap_gam_Cr, nx, ny, nm);
	convolution_kernel<<<num_tiles,tile_size,buf_size>>> (
	    dev->gam_Nb, dev->lap_gam_Nb, nx, ny, nm);
}

__device__ void composition_kernel(const fp_t& conc_Cr_old, const fp_t& conc_Nb_old,
                                   const fp_t& lap_gam_Cr,  const fp_t& lap_gam_Nb,
                                         fp_t& conc_Cr_new,       fp_t& conc_Nb_new,
                                   const fp_t& D_CrCr,      const fp_t& D_CrNb,
                                   const fp_t& D_NbCr,      const fp_t& D_NbNb,
                                   const fp_t& dt)
{
	/* Cahn-Hilliard equations of motion for composition */
	const fp_t lap_mu_Cr = D_CrCr * lap_gam_Cr
	                     + D_NbCr * lap_gam_Nb;
	const fp_t lap_mu_Nb = D_CrNb * lap_gam_Cr
	                     + D_NbNb * lap_gam_Nb;

	conc_Cr_new = conc_Cr_old + dt * lap_mu_Cr;
	conc_Nb_new = conc_Nb_old + dt * lap_mu_Nb;
}

__device__ void delta_kernel(const fp_t& conc_Cr_old, const fp_t& conc_Nb_old,
                             const fp_t& phi_del_old, const fp_t& phi_lav_old,
                                   fp_t& phi_del_new, const fp_t& inv_fict_det,
                             const fp_t& f_del,       const fp_t& f_lav,
                             const fp_t& dgGdxCr,     const fp_t& dgGdxNb,
                             const fp_t& gam_Cr,      const fp_t& gam_Nb,
                             const fp_t& gam_nrg,     const fp_t& alpha,
                             const fp_t& kappa,       const fp_t& omega,
                             const fp_t& M_del,       const fp_t& dt)
{
	const fp_t f_gam = 1. - f_del - f_lav;
	const fp_t del_Cr = d_fict_del_Cr(inv_fict_det, conc_Cr_old, conc_Nb_old, f_del, f_gam, f_lav);
	const fp_t del_Nb = d_fict_del_Nb(inv_fict_det, conc_Cr_old, conc_Nb_old, f_del, f_gam, f_lav);
	const fp_t del_nrg = d_g_del(del_Cr, del_Nb);

	/* pressure */
	const fp_t P_del = gam_nrg - del_nrg - dgGdxCr * (gam_Cr - del_Cr) - dgGdxNb * (gam_Nb - del_Nb);

	/* variational derivative */
	const fp_t dFdPhi_del = -d_hprime(phi_del_old) * P_del
	                      + 2. * omega * phi_del_old * (phi_del_old - 1.) * (2. * phi_del_old - 1.)
	                      + 2. * alpha * phi_del_old * (phi_lav_old * phi_lav_old)
	                      - kappa * phi_del_new;

	/* Allen-Cahn equation of motion for delta phase */
	phi_del_new = phi_del_old - dt * M_del * dFdPhi_del;
}

__device__ void laves_kernel(const fp_t& conc_Cr_old, const fp_t& conc_Nb_old,
                             const fp_t& phi_del_old, const fp_t& phi_lav_old,
                                   fp_t& phi_lav_new, const fp_t& inv_fict_det,
                             const fp_t& f_del,       const fp_t& f_lav,
                             const fp_t& dgGdxCr,     const fp_t& dgGdxNb,
                             const fp_t& gam_Cr,      const fp_t& gam_Nb,
                             const fp_t& gam_nrg,     const fp_t& alpha,
                             const fp_t& kappa,       const fp_t& omega,
                             const fp_t& M_lav,       const fp_t& dt)
{
	const fp_t f_gam = 1. - f_del - f_lav;
	const fp_t lav_Cr = d_fict_lav_Cr(inv_fict_det, conc_Cr_old, conc_Nb_old, f_del, f_gam, f_lav);
	const fp_t lav_Nb = d_fict_lav_Nb(inv_fict_det, conc_Cr_old, conc_Nb_old, f_del, f_gam, f_lav);
	const fp_t lav_nrg = d_g_lav(lav_Cr, lav_Nb);

	/* pressure */
	const fp_t P_lav = gam_nrg - lav_nrg - dgGdxCr * (gam_Cr - lav_Cr) - dgGdxNb * (gam_Nb - lav_Nb);

	/* variational derivative */
	const fp_t dFdPhi_lav = -d_hprime(phi_lav_old) * P_lav
	                      + 2. * omega * phi_lav_old * (phi_lav_old - 1.) * (2. * phi_lav_old - 1.)
	                      + 2. * alpha * phi_lav_old * (phi_del_old * phi_del_old)
	                      - kappa * phi_lav_new;

	/* Allen-Cahn equation of motion for Laves phase */
	phi_lav_new = phi_lav_old - dt * M_lav * dFdPhi_lav;
}

__global__ void fictitious_kernel(fp_t* d_conc_Cr, fp_t* d_conc_Nb,
                                  fp_t* d_phi_del, fp_t* d_phi_lav,
                                  fp_t* d_gam_Cr,  fp_t* d_gam_Nb,
                                  const int nx, const int ny)
{
	const int thr_x = threadIdx.x;
	const int thr_y = threadIdx.y;
	const int x = blockDim.x * blockIdx.x + thr_x;
	const int y = blockDim.y * blockIdx.y + thr_y;
	const int idx = nx * y + x;

	if (x < nx && y < ny) {
		const fp_t f_del = d_h(d_phi_del[idx]);
		const fp_t f_lav = d_h(d_phi_lav[idx]);
		const fp_t f_gam = 1. - f_del - f_lav;
		const fp_t inv_fict_det = d_inv_fict_det(f_del, f_gam, f_lav);

		d_gam_Cr[idx] = d_fict_gam_Cr(inv_fict_det, d_conc_Cr[idx], d_conc_Nb[idx],
		                              f_del, f_gam, f_lav);
		d_gam_Nb[idx] = d_fict_gam_Nb(inv_fict_det, d_conc_Cr[idx], d_conc_Nb[idx],
		                              f_del, f_gam, f_lav);
	}

	__syncthreads();
}

void device_fictitious(struct CudaData* dev,
                       const int nx, const int ny, const int nm,
                       const int bx, const int by)
{
	/* divide matrices into blocks of bx * by threads */
	dim3 tile_size(bx, by, 1);
	dim3 num_tiles(nTiles(nx, tile_size.x, nm),
	               nTiles(ny, tile_size.y, nm),
	               1);

	fictitious_kernel<<<num_tiles,tile_size>>>(
	    dev->conc_Cr_new, dev->conc_Nb_new,
	    dev->phi_del_new, dev->phi_lav_new,
	    dev->gam_Cr,      dev->gam_Nb,
	    nx, ny);
}

__global__ void evolution_kernel(fp_t* d_conc_Cr_old, fp_t* d_conc_Nb_old,
                                 fp_t* d_phi_del_old, fp_t* d_phi_lav_old,
                                 fp_t* d_lap_gam_Cr,  fp_t* d_lap_gam_Nb,
                                 fp_t* d_conc_Cr_new, fp_t* d_conc_Nb_new,
                                 fp_t* d_phi_del_new, fp_t* d_phi_lav_new,
                                 fp_t* d_gam_Cr,      fp_t* d_gam_Nb,
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
	const int idx = nx * y + x;

	/* explicit Euler solution to the equation of motion */
	if (x < nx && y < ny) {
		const fp_t f_del = d_h(d_phi_del_old[idx]);
		const fp_t f_lav = d_h(d_phi_lav_old[idx]);
		const fp_t inv_fict_det = d_inv_fict_det(f_del, 1.-f_del-f_lav, f_lav);

		/* pure phase energy */
		const fp_t gam_nrg = d_g_gam(d_gam_Cr[idx], d_gam_Nb[idx]);

		/* effective chemical potential */
		const fp_t dgGdxCr = d_dg_gam_dxCr(d_gam_Cr[idx], d_gam_Nb[idx]);
		const fp_t dgGdxNb = d_dg_gam_dxNb(d_gam_Cr[idx], d_gam_Nb[idx]);

		/* Cahn-Hilliard equations of motion for composition */
		composition_kernel(d_conc_Cr_old[idx], d_conc_Nb_old[idx],
		                   d_lap_gam_Cr[idx],  d_lap_gam_Nb[idx],
		                   d_conc_Cr_new[idx], d_conc_Nb_new[idx],
		                   D_CrCr, D_CrNb, D_NbCr, D_NbNb, dt);

		/* Allen-Cahn equations of motion for phase */
		delta_kernel(d_conc_Cr_old[idx], d_conc_Nb_old[idx], d_phi_del_old[idx], d_phi_lav_old[idx],
		             d_phi_del_new[idx], inv_fict_det, f_del, f_lav, dgGdxCr, dgGdxNb,
		             d_gam_Cr[idx], d_gam_Nb[idx], gam_nrg, alpha, kappa, omega,
		             M_del, dt);
		laves_kernel(d_conc_Cr_old[idx], d_conc_Nb_old[idx], d_phi_del_old[idx], d_phi_lav_old[idx],
		             d_phi_lav_new[idx], inv_fict_det, f_del, f_lav, dgGdxCr, dgGdxNb,
		             d_gam_Cr[idx], d_gam_Nb[idx], gam_nrg, alpha, kappa, omega,
		             M_lav, dt);
	}

	__syncthreads();
}

void device_evolution(struct CudaData* dev,
                      const int nx, const int ny, const int nm,
                      const int bx, const int by,
                      const fp_t* D_Cr, const fp_t* D_Nb,
                      const fp_t alpha, const fp_t kappa, const fp_t omega,
                      const fp_t M_del, const fp_t M_lav,
                      const fp_t dt)
{
	/* divide matrices into blocks of bx * by threads */
	dim3 tile_size(bx, by, 1);
	dim3 num_tiles(nTiles(nx, tile_size.x, nm),
	               nTiles(ny, tile_size.y, nm),
	               1);
	evolution_kernel<<<num_tiles,tile_size>>> (
	    dev->conc_Cr_old, dev->conc_Nb_old,
	    dev->phi_del_old, dev->phi_lav_old,
	    dev->lap_gam_Cr,  dev->lap_gam_Nb,
	    dev->conc_Cr_new, dev->conc_Nb_new,
	    dev->phi_del_new, dev->phi_lav_new,
	    dev->gam_Cr,      dev->gam_Nb,
	    nx, ny, nm,
	    D_Cr[0], D_Cr[1],
	    D_Nb[0], D_Nb[1],
	    alpha, kappa, omega,
	    M_del, M_lav,
	    dt);
}

__device__ void nucleation_driving_force(const fp_t xCr, const fp_t xNb, const int index,
                                         fp_t* par_xCr, fp_t* par_xNb, fp_t* dG)
{
	/* compute thermodynamic driving force for nucleation */
	const fp_t a11 = d_dg_gam_dxCr(xCr, xNb);
	const fp_t a12 = d_dg_gam_dxNb(xCr, xNb);
	const fp_t a21 = (index == 0) ? d_dg_del_dxCr(xCr, xNb)
	                              : d_dg_lav_dxCr(xCr, xNb);
	const fp_t a22 = (index == 0) ? d_dg_del_dxNb(xCr, xNb)
	                              : d_dg_lav_dxNb(xCr, xNb);

	const fp_t b11 = (index == 0) ? d_d2g_del_dxCrCr()
	                              : d_d2g_lav_dxCrCr();
	const fp_t b12 = (index == 0) ? d_d2g_del_dxCrNb()
	                              : d_d2g_lav_dxCrNb();
	const fp_t b22 = (index == 0) ? d_d2g_del_dxNbNb()
	                              : d_d2g_lav_dxNbNb();

	const fp_t b1 = d_dg_gam_dxCr(xCr, xNb) + b11 + b12;
	const fp_t b2 = d_dg_gam_dxCr(xCr, xNb) + b12 + b22;

	const fp_t detA = a11 * a22 - a12 * a21;
	const fp_t detB = b1  * a22 - a12 * b2;
	const fp_t detC = a11 * b2  - b1  * a21;

	*par_xCr = detB / detA;
	*par_xNb = detC / detA;

	const fp_t G_matrix = d_g_gam(xCr, xNb) + d_dg_gam_dxCr(xCr, xNb) * (*par_xCr - xCr)
	                    + d_dg_gam_dxNb(xCr, xNb) * (*par_xNb - xNb);
	const fp_t G_precip = (index == 0) ? d_g_del(*par_xCr, *par_xNb)
	                                   : d_g_lav(*par_xCr, *par_xNb);

	*dG = G_matrix - G_precip;
}

__global__ void nucleation_kernel(fp_t* d_conc_Cr, fp_t* d_conc_Nb,
                                  fp_t* d_phi_del, fp_t* d_phi_lav,
                                  const int nx, const int ny, const int nm,
                                  const fp_t D_CrCr, const fp_t D_NbNb,
                                  const fp_t sigma_del, const fp_t sigma_lav,
                                  const fp_t unit_a, const fp_t ifce_width,
                                  const fp_t dx, const fp_t dy, const fp_t dt)
{
	/* determine indices on which to operate */
	const int thr_x = threadIdx.x;
	const int thr_y = threadIdx.y;
	const int x = blockDim.x * blockIdx.x + thr_x;
	const int y = blockDim.y * blockIdx.y + thr_y;
	const int idx = nx * y + x;

	curandState state;
	curand_init((unsigned long long)clock(), x, 0, &state);

	for (int k = 0; k < 2; k++) {
		const fp_t sigma = (k == 0) ? sigma_del
		                            : sigma_lav;
		if (x < nx && y < ny) {
			const fp_t phi_gam = 1.0 - d_h(d_phi_del[idx]) - d_h(d_phi_lav[idx]);
			if (phi_gam > 1.0e-9) {
				const fp_t dV = dx * dy;
				const fp_t xCr = d_conc_Cr[idx];
				const fp_t xNb = d_conc_Nb[idx];
				fp_t dG_chem = 0.;
				fp_t par_xCr = 0., par_xNb = 0.;
				nucleation_driving_force(d_conc_Cr[idx], d_conc_Nb[idx], k,
				                         &par_xCr, &par_xNb, &dG_chem);
				const fp_t denom = unit_a * dG_chem - 2. * sigma;
				const fp_t Vatom = unit_a*unit_a*unit_a / 4.;
				const fp_t Rstar = (sigma * unit_a) / denom;
				const fp_t Zeldov = Vatom * sqrt(6. * denom*denom*denom / d_kT())
				                  / (M_PI*M_PI * unit_a*unit_a * sigma);
				const fp_t N_gam = (nx*dx * ny*dy * unit_a * M_PI) / (3. * sqrt(2.) * Vatom);
				const fp_t BstarCr = (2. * M_PI * sigma * D_CrCr * xCr
				                     * (unit_a * dG_chem - sigma))
				                   / (unit_a*unit_a * denom*denom);
				const fp_t BstarNb = (2. * M_PI * sigma * D_NbNb * xNb
				                      * (unit_a * dG_chem - sigma))
				                   / (unit_a*unit_a * denom*denom);
				const fp_t k1Cr = BstarCr * Zeldov * N_gam;
				const fp_t k1Nb = BstarNb * Zeldov * N_gam;
				const fp_t k2 = dG_chem / d_kT();
				const fp_t dc_Cr = par_xCr - xCr;
				const fp_t dc_Nb = par_xNb - xNb;

				const fp_t JCr = k1Cr * exp(k2 / dc_Cr);
				const fp_t JNb = k1Nb * exp(k2 / dc_Nb);

				const fp_t P_nuc = 1. - exp(-JCr * dt * dV) - exp(-JNb * dt * dV);
				const fp_t rand = (fp_t)curand_uniform_double(&state);

				if (rand < P_nuc) {
					// Embed a particle of type k!
					const int rad = ceil(Rstar / dx);
					for (int j = y - 2*rad; j < y + 2*rad; j++) {
						for (int i = x - 2*rad; i < x + 2*rad; i++) {
							const int r = sqrt(float((i-x)*(i-x) + (j-y)*(j-y)));
							const int idn = nx * j + i;
							if (k == 0) {
								d_phi_del[idn] = d_interface_profile(ifce_width, r);
							} else {
								d_phi_lav[idn] = d_interface_profile(ifce_width, r);
							}
						}
					}
				}
			}
		}

		__syncthreads();
	}
}

void device_nucleation(struct CudaData* dev,
                       const int nx, const int ny, const int nm,
                       const int bx, const int by,
                       const fp_t* D_Cr, const fp_t* D_Nb,
                       const fp_t sigma_del, const fp_t sigma_lav,
                       const fp_t unit_a, const fp_t ifce_width,
                       const fp_t dx, const fp_t dy, const fp_t dt)
{
	/* divide matrices into blocks of bx * by threads */
	dim3 tile_size(bx, by, 1);
	dim3 num_tiles(nTiles(nx, tile_size.x, nm),
	               nTiles(ny, tile_size.y, nm),
	               1);
	nucleation_kernel<<<num_tiles,tile_size>>> (
	    dev->conc_Cr_new, dev->conc_Nb_new,
	    dev->phi_del_new, dev->phi_lav_new,
	    nx, ny, nm,
	    D_Cr[0], D_Nb[1],
	    sigma_del, sigma_lav,
	    unit_a, ifce_width,
	    dx, dy, dt);
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
	cudaMemcpy(host->gam_Cr[0], dev->gam_Cr, nx * ny * sizeof(fp_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(host->gam_Nb[0], dev->gam_Nb, nx * ny * sizeof(fp_t), cudaMemcpyDeviceToHost);
}
