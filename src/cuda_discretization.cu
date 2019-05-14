/**
 \file  cuda_discretization.cu
 \brief Implementation of boundary condition functions with CUDA acceleration
*/

#include <assert.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>

#include "cuda_data.h"
#include "numerics.h"
#include "mesh.h"

#include "cuda_kernels.cuh"
#include "parabola625.cuh"
#include "nucleation.cuh"

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
cudaError_t checkCuda(cudaError_t result)
{
    if (result != cudaSuccess) {
		fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
		assert(result == cudaSuccess);
	}
	return result;
}


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

void device_boundaries(cudaStream_t& stream,
                       struct CudaData* dev,
                       const int nx, const int ny, const int nm,
                       const int bx, const int by)
{
	/* divide matrices into blocks of bx * by threads */
	dim3 tile_size(bx, by, 1);
	dim3 num_tiles(nTiles(nx, tile_size.x, nm),
	               nTiles(ny, tile_size.y, nm),
	               1);

	boundary_kernel<<<num_tiles, tile_size, 0, stream>>> (
	    dev->conc_Cr_old, dev->conc_Nb_old,
	    dev->phi_del_old, dev->phi_lav_old,
	    dev->gam_Cr,      dev->gam_Nb,
	    nx, ny, nm
	);
}

void device_laplacian_boundaries(cudaStream_t& stream,
                                 struct CudaData* dev,
                                 const int nx, const int ny, const int nm,
                                 const int bx, const int by)
{
	/* divide matrices into blocks of bx * by threads */
	dim3 tile_size(bx, by, 1);
	dim3 num_tiles(nTiles(nx, tile_size.x, nm),
	               nTiles(ny, tile_size.y, nm),
	               1);

	boundary_kernel<<<num_tiles, tile_size, 0, stream>>> (
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
}

void device_laplacian(cudaStream_t& stream,
                      struct CudaData* dev,
                      const int nx, const int ny, const int nm,
                      const int bx, const int by)
{
	/* divide matrices into blocks of bx * by threads */
	dim3 tile_size(bx, by, 1);
	dim3 num_tiles(nTiles(nx, tile_size.x, nm),
	               nTiles(ny, tile_size.y, nm),
	               1);
	size_t buf_size = (tile_size.x + nm) * (tile_size.y + nm) * sizeof(fp_t);

	convolution_kernel<<<num_tiles, tile_size, buf_size, stream>>> (
	    dev->conc_Cr_old, dev->conc_Cr_new, nx, ny, nm);
	convolution_kernel<<<num_tiles, tile_size, buf_size, stream>>> (
	    dev->conc_Nb_old, dev->conc_Nb_new, nx, ny, nm);

	convolution_kernel<<<num_tiles, tile_size, buf_size, stream>>> (
	    dev->phi_del_old, dev->phi_del_new, nx, ny, nm);
	convolution_kernel<<<num_tiles, tile_size, buf_size, stream>>> (
	    dev->phi_lav_old, dev->phi_lav_new, nx, ny, nm);

	convolution_kernel<<<num_tiles, tile_size, buf_size, stream>>> (
	    dev->gam_Cr, dev->lap_gam_Cr, nx, ny, nm);
	convolution_kernel<<<num_tiles, tile_size, buf_size, stream>>> (
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
}

void device_fictitious(cudaStream_t& stream,
                       struct CudaData* dev,
                       const int nx, const int ny, const int nm,
                       const int bx, const int by)
{
	/* divide matrices into blocks of bx * by threads */
	dim3 tile_size(bx, by, 1);
	dim3 num_tiles(nTiles(nx, tile_size.x, nm),
	               nTiles(ny, tile_size.y, nm),
	               1);

	fictitious_kernel<<<num_tiles, tile_size, 0, stream>>>(
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
}

void device_evolution(cudaStream_t& stream,
                      struct CudaData* dev,
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
	evolution_kernel<<<num_tiles, tile_size, 0, stream>>> (
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

__global__ void init_prng_kernel(curandState* d_prng, const int nx, const int ny)
{
	/* determine indices on which to operate */
	const int thr_x = threadIdx.x;
	const int thr_y = threadIdx.y;
	const int x = blockDim.x * blockIdx.x + thr_x;
	const int y = blockDim.y * blockIdx.y + thr_y;
	const int idx = nx * y + x;

	if (x < nx && y < ny)
        curand_init((unsigned long long)clock() + idx, x, 0, &(d_prng[idx]));
}

void device_init_prng(cudaStream_t& stream,
                      struct CudaData* dev,
                      const int nx, const int ny, const int nm,
                      const int bx, const int by)
{
	/* divide matrices into blocks of bx * by threads */
	dim3 tile_size(bx, by, 1);
	dim3 num_tiles(nTiles(nx, tile_size.x, nm),
	               nTiles(ny, tile_size.y, nm),
	               1);
	init_prng_kernel<<<num_tiles, tile_size, 0, stream>>> (
	    dev->prng, nx, ny);
}

__global__ void nucleation_kernel(fp_t* d_conc_Cr, fp_t* d_conc_Nb,
                                  fp_t* d_phi_del, fp_t* d_phi_lav,
                                  curandState* d_prng,
                                  const int nx, const int ny, const int nm,
                                  const fp_t D_CrCr, const fp_t D_NbNb,
                                  const fp_t sigma_del, const fp_t sigma_lav,
                                  const fp_t lattice_const, const fp_t ifce_width,
                                  const fp_t dx, const fp_t dy, const fp_t dz,
                                  const fp_t dt)
{
	/* determine indices on which to operate */
	const int thr_x = threadIdx.x;
	const int thr_y = threadIdx.y;
	const int x = blockDim.x * blockIdx.x + thr_x;
	const int y = blockDim.y * blockIdx.y + thr_y;

    const fp_t dV = dx * dy * dz;
    const fp_t Vatom = 0.25 * lattice_const * lattice_const * lattice_const; // assuming FCC
    const fp_t n_gam = M_PI / (3. * sqrt(2.) * Vatom); // assuming FCC
    const fp_t w = ifce_width / dx;
    const fp_t throttle = 5e-7;

    fp_t phi_pre = 0.;
    fp_t dG_chem_del = 0., del_xCr = 0., del_xNb = 0.;
    fp_t dG_chem_lav = 0., lav_xCr = 0., lav_xNb = 0.;
    int R_del, R_lav;
    fp_t r_del, r_del_star;
    fp_t r_lav, r_lav_star;
    fp_t P_nuc_del, P_nuc_lav;
    fp_t rand_del, rand_lav;

    if (x < nx && y < ny) {
        const fp_t rad = 1.75e-9 / dx;
        const int R = 5 * ceil(rad + w) / 4;

        for (int i = -R; i < R; i++) {
            for (int j = -R; j < R; j++) {
                const int idn = nx * (y + j) + (x + i);
                const fp_t r = sqrt(fp_t(i*i + j*j));
                if (idn >= 0 &&
                    idn < nx * ny &&
                    i*i + j*j < R*R)
                    phi_pre = max(phi_pre, d_h(d_phi_del[idn])
                                         + d_h(d_phi_lav[idn]));
            }
        }
    }
    __syncthreads();

    if (x < nx && y < ny && phi_pre < 1e-10) {
        const int idx = nx * y + x;
        const fp_t xCr = d_conc_Cr[idx];
        const fp_t xNb = d_conc_Nb[idx];

        // Test a delta particle
        d_nucleation_driving_force_delta(xCr, xNb, &dG_chem_del);
        d_nucleation_probability_sphere(xCr, xNb,
                                        del_xCr, del_xNb,
                                        dG_chem_del,
                                        D_CrCr, D_NbNb,
                                        sigma_del,
                                        Vatom,
                                        n_gam,
                                        dV, dt,
                                        &r_del_star, &P_nuc_del);
        r_del = r_del_star / dx;
        R_del = 5 * ceil(r_del + w) / 4;
        rand_del = throttle * P_nuc_del - (fp_t)curand_uniform_double(&(d_prng[idx]));

        // Test a Laves particle
        d_nucleation_driving_force_laves(xCr, xNb, &dG_chem_lav);
        d_nucleation_probability_sphere(xCr, xNb,
                                        lav_xCr, lav_xNb,
                                        dG_chem_lav,
                                        D_CrCr, D_NbNb,
                                        sigma_lav,
                                        Vatom,
                                        n_gam,
                                        dV, dt,
                                        &r_lav_star, &P_nuc_lav);
        r_lav = r_lav_star / dx;
        R_lav = 5 * ceil(r_lav + w) / 4;
        rand_lav = throttle * P_nuc_lav - (fp_t)curand_uniform_double(&(d_prng[idx]));

        if (rand_del > 0 && rand_del > rand_lav) {
            for (int i = -R_del; i < R_del; i++) {
                for (int j = -R_del; j < R_del; j++) {
                    const int idn = nx * (y + j) + (x + i);
                    const fp_t r = sqrt(fp_t(i*i + j*j));
                    const fp_t z = r - (r_del + w);
                    if (idn >= 0 && idn < nx * ny)
                        d_phi_del[idn] = d_interface_profile(4 * z / w);
                }
            }
        }

        if (rand_lav > 0 && rand_lav > rand_del) {
            for (int i = -R_lav; i < R_lav; i++) {
                for (int j = -R_lav; j < R_lav; j++) {
                    const int idn = nx * (y + j) + (x + i);
                    const fp_t r = sqrt(fp_t(i*i + j*j));
                    const fp_t z = r - (r_lav + w);
                    if (idn >= 0 && idn < nx * ny)
                        d_phi_lav[idn] = d_interface_profile(4 * z / w);
                }
            }
        }
    }
}

void device_nucleation(cudaStream_t& stream,
                       struct CudaData* dev,
                       const int nx, const int ny, const int nm,
                       const int bx, const int by,
                       const fp_t* D_Cr, const fp_t* D_Nb,
                       const fp_t sigma_del, const fp_t sigma_lav,
                       const fp_t lattice_const, const fp_t ifce_width,
                       const fp_t dx, const fp_t dy, const fp_t dz,
                       const fp_t dt)
{
	/* divide matrices into blocks of bx * by threads */
	dim3 tile_size(bx, by, 1);
	dim3 num_tiles(nTiles(nx, tile_size.x, nm),
	               nTiles(ny, tile_size.y, nm),
	               1);
	nucleation_kernel<<<num_tiles, tile_size, 0, stream>>> (
	    dev->conc_Cr_new, dev->conc_Nb_new,
	    dev->phi_del_new, dev->phi_lav_new,
        dev->prng,
	    nx, ny, nm,
	    D_Cr[0], D_Nb[1],
	    sigma_del, sigma_lav,
	    lattice_const, ifce_width,
	    dx, dy, dz, dt);
}

__global__ void nickel_kernel(fp_t* d_conc_Cr, fp_t* d_conc_Nb, fp_t* d_conc_Ni,
                              const int nx, const int ny)
{
	const int thr_x = threadIdx.x;
	const int thr_y = threadIdx.y;
	const int x = blockDim.x * blockIdx.x + thr_x;
	const int y = blockDim.y * blockIdx.y + thr_y;
	const int idx = nx * y + x;

	if (x < nx && y < ny)
		d_conc_Ni[idx] = 1. - d_conc_Cr[idx] - d_conc_Nb[idx]; 
}

void device_compute_Ni(cudaStream_t& stream,
                       struct CudaData* dev, struct HostData* host,
                       const int nx, const int ny, const int nm,
                       const int bx, const int by)
{
	/* divide matrices into blocks of bx * by threads */
	dim3 tile_size(bx, by, 1);
	dim3 num_tiles(nTiles(nx, tile_size.x, nm),
	               nTiles(ny, tile_size.y, nm),
	               1);

	nickel_kernel<<<num_tiles, tile_size, 0, stream>>>(
	    dev->conc_Cr_old, dev->conc_Nb_old, dev->conc_Ni,
	    nx, ny);

    cudaMemcpyAsync(host->conc_Ni[0], dev->conc_Ni, nx * ny * sizeof(fp_t),
                    cudaMemcpyDeviceToHost, stream);
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
