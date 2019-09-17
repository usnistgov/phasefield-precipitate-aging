/**
 \file  discretization.cu
 \brief Implementation of discretized equations with CUDA acceleration
 Contains functions for boundary conditions, equations of motion, and nucleation.
*/

#include <assert.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>

#include "data.cuh"
#include "discretization.cuh"
#include "parabola625.cuh"
#include "nucleation.cuh"

/**
 \brief Convenience function for checking CUDA runtime API results
 No-op in release builds.
*/
cudaError_t checkCuda(cudaError_t result)
{
	if (result != cudaSuccess) {
		fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
		assert(result == cudaSuccess);
	}
	return result;
}

// Convolution mask array on the GPU, allocated in protected memory
__constant__ fp_t d_mask[MAX_MASK_W * MAX_MASK_H];

// Kinetic parameter arrays on the GPU, allocated in protected memory
__constant__ fp_t d_Kapp[NP];
__constant__ fp_t d_Omeg[NP];
__constant__ fp_t d_Lmob[NP];

float nTiles(int domain_size, int tile_loc, int mask_size)
{
	return ceil(float(domain_size) / float(tile_loc - mask_size + 1));
}

__global__ void fictitious_gam_kernel(fp_t* d_conc_Cr,
                                      fp_t* d_conc_Nb,
                                      fp_t* d_phi_del,
                                      fp_t* d_phi_lav,
                                      fp_t* d_conc_Cr_gam,
                                      fp_t* d_conc_Nb_gam,
                                      const int nx,
                                      const int ny,
                                      const int nm)
{
	/* determine indices on which to operate */
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	const int idx = nx * y + x;

	if (x < nx && y < ny) {
		const fp_t xCr = d_conc_Cr[idx];
		const fp_t xNb = d_conc_Cr[idx];
		const fp_t pDel = d_p(d_phi_del[idx]);
		const fp_t pLav = d_p(d_phi_lav[idx]);
		const fp_t pGam = 1.0 - pDel - pLav;
		const fp_t inv_det = d_inv_fict_det(pDel, pGam, pLav);
		d_conc_Cr_gam[idx] = d_fict_gam_Cr(inv_det, xCr, xNb, pDel, pGam, pLav);
		d_conc_Nb_gam[idx] = d_fict_gam_Nb(inv_det, xCr, xNb, pDel, pGam, pLav);
	}
}

__global__ void fictitious_del_kernel(fp_t* d_conc_Cr,
                                      fp_t* d_conc_Nb,
                                      fp_t* d_phi_del,
                                      fp_t* d_phi_lav,
                                      fp_t* d_conc_Cr_del,
                                      fp_t* d_conc_Nb_del,
                                      const int nx,
                                      const int ny,
                                      const int nm)
{
	/* determine indices on which to operate */
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	const int idx = nx * y + x;

	if (x < nx && y < ny) {
		const fp_t xCr = d_conc_Cr[idx];
		const fp_t xNb = d_conc_Cr[idx];
		const fp_t pDel = d_p(d_phi_del[idx]);
		const fp_t pLav = d_p(d_phi_lav[idx]);
		const fp_t pGam = 1.0 - pDel - pLav;
		const fp_t inv_det = d_inv_fict_det(pDel, pGam, pLav);
		d_conc_Cr_del[idx] = d_fict_del_Cr(inv_det, xCr, xNb, pDel, pGam, pLav);
		d_conc_Nb_del[idx] = d_fict_del_Nb(inv_det, xCr, xNb, pDel, pGam, pLav);
	}
}

__global__ void fictitious_lav_kernel(fp_t* d_conc_Cr,
                                      fp_t* d_conc_Nb,
                                      fp_t* d_phi_del,
                                      fp_t* d_phi_lav,
                                      fp_t* d_conc_Cr_lav,
                                      fp_t* d_conc_Nb_lav,
                                      const int nx,
                                      const int ny,
                                      const int nm)
{
	/* determine indices on which to operate */
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	const int idx = nx * y + x;

	if (x < nx && y < ny) {
		const fp_t xCr = d_conc_Cr[idx];
		const fp_t xNb = d_conc_Cr[idx];
		const fp_t pDel = d_p(d_phi_del[idx]);
		const fp_t pLav = d_p(d_phi_lav[idx]);
		const fp_t pGam = 1.0 - pDel - pLav;
		const fp_t inv_det = d_inv_fict_det(pDel, pGam, pLav);
		d_conc_Cr_lav[idx] = d_fict_lav_Cr(inv_det, xCr, xNb, pDel, pGam, pLav);
		d_conc_Nb_lav[idx] = d_fict_lav_Nb(inv_det, xCr, xNb, pDel, pGam, pLav);
	}
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

    cudaStreamWaitEvent(dev->str_A, dev->ev_A, 0);
    cudaStreamWaitEvent(dev->str_B, dev->ev_B, 0);
    cudaStreamWaitEvent(dev->str_C, dev->ev_C, 0);
    cudaStreamWaitEvent(dev->str_D, dev->ev_D, 0);

	fictitious_gam_kernel
      <<< num_tiles, tile_size, 0, dev->str_A>>>
	(dev->conc_Cr_old, dev->conc_Nb_old,
	 dev->phi_del_old, dev->phi_lav_old,
	 dev->conc_Cr_gam, dev->conc_Nb_gam,
	 nx, ny, nm);
    cudaEventRecord(dev->ev_A, dev->str_A);

	fictitious_del_kernel
	<<< num_tiles, tile_size, 0, dev->str_B>>>
	(dev->conc_Cr_old, dev->conc_Nb_old,
	 dev->phi_del_old, dev->phi_lav_old,
	 dev->conc_Cr_del, dev->conc_Nb_del,
	 nx, ny, nm);
    cudaEventRecord(dev->ev_B, dev->str_B);

    fictitious_lav_kernel
	<<< num_tiles, tile_size, 0, dev->str_C>>>
	(dev->conc_Cr_old, dev->conc_Nb_old,
	 dev->phi_del_old, dev->phi_lav_old,
	 dev->conc_Cr_lav, dev->conc_Nb_lav,
	 nx, ny, nm);
    cudaEventRecord(dev->ev_C, dev->str_C);
}

__global__ void mobility_gam_kernel(fp_t* d_conc_Cr,
                                    fp_t* d_conc_Nb,
                                    fp_t* d_phi_del,
                                    fp_t* d_phi_lav,
                                    fp_t* d_mob_gam_CrCr,
                                    fp_t* d_mob_gam_CrNb,
                                    fp_t* d_mob_gam_NbCr,
                                    fp_t* d_mob_gam_NbNb,
                                    const int nx,
                                    const int ny,
                                    const int nm)
{
	/* determine indices on which to operate */
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	const int idx = nx * y + x;

	if (x < nx && y < ny) {
		const fp_t xCr = d_conc_Cr[idx];
		const fp_t xNb = d_conc_Cr[idx];
		const fp_t pGam = 1.0 - d_p(d_phi_del[idx]) - d_p(d_phi_lav[idx]);
		d_mob_gam_CrCr[idx] = pGam * (d_M_CrCr(xCr, xNb) * d_d2g_gam_dxCrCr() + d_M_CrNb(xCr, xNb) * d_d2g_gam_dxCrNb());
		d_mob_gam_CrNb[idx] = pGam * (d_M_CrCr(xCr, xNb) * d_d2g_gam_dxNbCr() + d_M_CrNb(xCr, xNb) * d_d2g_gam_dxNbNb());
		d_mob_gam_NbCr[idx] = pGam * (d_M_NbCr(xCr, xNb) * d_d2g_gam_dxCrCr() + d_M_NbNb(xCr, xNb) * d_d2g_gam_dxCrNb());
		d_mob_gam_NbNb[idx] = pGam * (d_M_NbCr(xCr, xNb) * d_d2g_gam_dxNbCr() + d_M_NbNb(xCr, xNb) * d_d2g_gam_dxNbNb());
	}
}

__global__ void mobility_del_kernel(fp_t* d_conc_Cr,
                                    fp_t* d_conc_Nb,
                                    fp_t* d_phi_del,
                                    fp_t* d_mob_del_CrCr,
                                    fp_t* d_mob_del_CrNb,
                                    fp_t* d_mob_del_NbCr,
                                    fp_t* d_mob_del_NbNb,
                                    const int nx,
                                    const int ny,
                                    const int nm)
{
	/* determine indices on which to operate */
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	const int idx = nx * y + x;

	if (x < nx && y < ny) {
		const fp_t xCr = d_conc_Cr[idx];
		const fp_t xNb = d_conc_Cr[idx];
		const fp_t pDel = d_p(d_phi_del[idx]);
		d_mob_del_CrCr[idx] = pDel * (d_M_CrCr(xCr, xNb) * d_d2g_del_dxCrCr() + d_M_CrNb(xCr, xNb) * d_d2g_del_dxCrNb());
		d_mob_del_CrNb[idx] = pDel * (d_M_CrCr(xCr, xNb) * d_d2g_del_dxNbCr() + d_M_CrNb(xCr, xNb) * d_d2g_del_dxNbNb());
		d_mob_del_NbCr[idx] = pDel * (d_M_NbCr(xCr, xNb) * d_d2g_del_dxCrCr() + d_M_NbNb(xCr, xNb) * d_d2g_del_dxCrNb());
		d_mob_del_NbNb[idx] = pDel * (d_M_NbCr(xCr, xNb) * d_d2g_del_dxNbCr() + d_M_NbNb(xCr, xNb) * d_d2g_del_dxNbNb());
	}
}

__global__ void mobility_lav_kernel(fp_t* d_conc_Cr,
                                    fp_t* d_conc_Nb,
                                    fp_t* d_phi_lav,
                                    fp_t* d_mob_lav_CrCr,
                                    fp_t* d_mob_lav_CrNb,
                                    fp_t* d_mob_lav_NbCr,
                                    fp_t* d_mob_lav_NbNb,
                                    const int nx,
                                    const int ny,
                                    const int nm)
{
	/* determine indices on which to operate */
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	const int idx = nx * y + x;

	if (x < nx && y < ny) {
		const fp_t xCr = d_conc_Cr[idx];
		const fp_t xNb = d_conc_Cr[idx];
		const fp_t pLav = d_p(d_phi_lav[idx]);
		d_mob_lav_CrCr[idx] = pLav * (d_M_CrCr(xCr, xNb) * d_d2g_lav_dxCrCr() + d_M_CrNb(xCr, xNb) * d_d2g_lav_dxCrNb());
		d_mob_lav_CrNb[idx] = pLav * (d_M_CrCr(xCr, xNb) * d_d2g_lav_dxNbCr() + d_M_CrNb(xCr, xNb) * d_d2g_lav_dxNbNb());
		d_mob_lav_NbCr[idx] = pLav * (d_M_NbCr(xCr, xNb) * d_d2g_lav_dxCrCr() + d_M_NbNb(xCr, xNb) * d_d2g_lav_dxCrNb());
		d_mob_lav_NbNb[idx] = pLav * (d_M_NbCr(xCr, xNb) * d_d2g_lav_dxNbCr() + d_M_NbNb(xCr, xNb) * d_d2g_lav_dxNbNb());
	}
}

void device_mobilities(struct CudaData* dev,
                       const int nx, const int ny, const int nm,
                       const int bx, const int by)
{
	/* divide matrices into blocks of bx * by threads */
	dim3 tile_size(bx, by, 1);
	dim3 num_tiles(nTiles(nx, tile_size.x, nm),
	               nTiles(ny, tile_size.y, nm),
	               1);

    cudaStreamWaitEvent(dev->str_A, dev->ev_A, 0);
    cudaStreamWaitEvent(dev->str_B, dev->ev_B, 0);
    cudaStreamWaitEvent(dev->str_C, dev->ev_C, 0);
    cudaStreamWaitEvent(dev->str_D, dev->ev_D, 0);

	mobility_gam_kernel
	<<< num_tiles, tile_size, 0, dev->str_A>>>
	(dev->conc_Cr_old, dev->conc_Nb_old,
	 dev->phi_del_old, dev->phi_lav_old,
	 dev->mob_gam_CrCr, dev->mob_gam_CrNb,
	 dev->mob_gam_NbCr, dev->mob_gam_NbNb,
	 nx, ny, nm);
    cudaEventRecord(dev->ev_A, dev->str_A);

	mobility_del_kernel
	<<< num_tiles, tile_size, 0, dev->str_B>>>
	(dev->conc_Cr_old, dev->conc_Nb_old,
	 dev->phi_del_old,
	 dev->mob_del_CrCr, dev->mob_del_CrNb,
	 dev->mob_del_NbCr, dev->mob_del_NbNb,
	 nx, ny, nm);
    cudaEventRecord(dev->ev_B, dev->str_B);

    mobility_lav_kernel
	<<< num_tiles, tile_size, 0, dev->str_C>>>
	(dev->conc_Cr_old, dev->conc_Nb_old,
	 dev->phi_lav_old,
	 dev->mob_lav_CrCr, dev->mob_lav_CrNb,
	 dev->mob_lav_NbCr, dev->mob_lav_NbNb,
	 nx, ny, nm);
    cudaEventRecord(dev->ev_C, dev->str_C);
}

__global__ void boundary_kernel(fp_t* d_field,
                                const int nx,
                                const int ny,
                                const int nm)
{
	/* determine indices on which to operate */
	const int row = blockDim.y * blockIdx.y + threadIdx.y;
	const int col = blockDim.x * blockIdx.x + threadIdx.x;

	/* apply no-flux boundary conditions: inside to out, sequence matters */

	for (int offset = 0; offset < nm / 2; offset++) {
		const int ilo = nm / 2 - offset;
		const int ihi = nx - 1 - nm / 2 + offset;
		const int jlo = nm / 2 - offset;
		const int jhi = ny - 1 - nm / 2 + offset;

		if (ilo - 1 == col && row < ny) {
			/* left condition */
			d_field[row * nx + ilo - 1] = d_field[row * nx + ilo];
		}
		if (ihi + 1 == col && row < ny) {
			/* right condition */
			d_field[row * nx + ihi + 1] = d_field[row * nx + ihi];
		}
		if (jlo - 1 == row && col < nx) {
			/* bottom condition */
			d_field[(jlo - 1) * nx + col] = d_field[jlo * nx + col];
		}
		if (jhi + 1 == row && col < nx) {
			/* top condition */
			d_field[(jhi + 1) * nx + col] = d_field[jhi * nx + col];
		}
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
    cudaStreamWaitEvent(dev->str_A, dev->ev_A, 0);
    cudaStreamWaitEvent(dev->str_B, dev->ev_B, 0);
    cudaStreamWaitEvent(dev->str_C, dev->ev_C, 0);
    cudaStreamWaitEvent(dev->str_D, dev->ev_D, 0);

	boundary_kernel <<< num_tiles, tile_size, 0, dev->str_A>>> (dev->conc_Cr_old, nx, ny, nm);
	boundary_kernel <<< num_tiles, tile_size, 0, dev->str_B>>> (dev->conc_Nb_old, nx, ny, nm);
	boundary_kernel <<< num_tiles, tile_size, 0, dev->str_C>>> (dev->phi_del_old, nx, ny, nm);
	boundary_kernel <<< num_tiles, tile_size, 0, dev->str_D>>> (dev->phi_lav_old, nx, ny, nm);

	boundary_kernel <<< num_tiles, tile_size, 0, dev->str_A>>> (dev->conc_Cr_gam, nx, ny, nm);
	boundary_kernel <<< num_tiles, tile_size, 0, dev->str_B>>> (dev->conc_Nb_gam, nx, ny, nm);

	boundary_kernel <<< num_tiles, tile_size, 0, dev->str_C>>> (dev->conc_Cr_del, nx, ny, nm);
	boundary_kernel <<< num_tiles, tile_size, 0, dev->str_D>>> (dev->conc_Nb_del, nx, ny, nm);

	boundary_kernel <<< num_tiles, tile_size, 0, dev->str_A>>> (dev->conc_Cr_lav, nx, ny, nm);
	boundary_kernel <<< num_tiles, tile_size, 0, dev->str_B>>> (dev->conc_Nb_lav, nx, ny, nm);

	boundary_kernel <<< num_tiles, tile_size, 0, dev->str_C>>> (dev->mob_gam_CrCr, nx, ny, nm);
	boundary_kernel <<< num_tiles, tile_size, 0, dev->str_D>>> (dev->mob_gam_CrNb, nx, ny, nm);
	boundary_kernel <<< num_tiles, tile_size, 0, dev->str_A>>> (dev->mob_gam_NbCr, nx, ny, nm);
	boundary_kernel <<< num_tiles, tile_size, 0, dev->str_B>>> (dev->mob_gam_NbNb, nx, ny, nm);

	boundary_kernel <<< num_tiles, tile_size, 0, dev->str_C>>> (dev->mob_del_CrCr, nx, ny, nm);
	boundary_kernel <<< num_tiles, tile_size, 0, dev->str_D>>> (dev->mob_del_CrNb, nx, ny, nm);
	boundary_kernel <<< num_tiles, tile_size, 0, dev->str_A>>> (dev->mob_del_NbCr, nx, ny, nm);
	boundary_kernel <<< num_tiles, tile_size, 0, dev->str_B>>> (dev->mob_del_NbNb, nx, ny, nm);

	boundary_kernel <<< num_tiles, tile_size, 0, dev->str_C>>> (dev->mob_lav_CrCr, nx, ny, nm);
	boundary_kernel <<< num_tiles, tile_size, 0, dev->str_D>>> (dev->mob_lav_CrNb, nx, ny, nm);
	boundary_kernel <<< num_tiles, tile_size, 0, dev->str_A>>> (dev->mob_lav_NbCr, nx, ny, nm);
	boundary_kernel <<< num_tiles, tile_size, 0, dev->str_B>>> (dev->mob_lav_NbNb, nx, ny, nm);

    cudaEventRecord(dev->ev_A, dev->str_A);
    cudaEventRecord(dev->ev_B, dev->str_B);
    cudaEventRecord(dev->ev_C, dev->str_C);
    cudaEventRecord(dev->ev_D, dev->str_D);
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

    cudaStreamWaitEvent(dev->str_A, dev->ev_A, 0);
    cudaStreamWaitEvent(dev->str_B, dev->ev_B, 0);
    cudaStreamWaitEvent(dev->str_C, dev->ev_C, 0);
    cudaStreamWaitEvent(dev->str_D, dev->ev_D, 0);

	boundary_kernel <<< num_tiles, tile_size, 0, dev->str_A>>> (dev->conc_Cr_new, nx, ny, nm);
    cudaEventRecord(dev->ev_A, dev->str_A);
	boundary_kernel <<< num_tiles, tile_size, 0, dev->str_B>>> (dev->conc_Nb_new, nx, ny, nm);
    cudaEventRecord(dev->ev_B, dev->str_B);
	boundary_kernel <<< num_tiles, tile_size, 0, dev->str_C>>> (dev->phi_del_new, nx, ny, nm);
    cudaEventRecord(dev->ev_C, dev->str_C);
	boundary_kernel <<< num_tiles, tile_size, 0, dev->str_D>>> (dev->phi_lav_new, nx, ny, nm);
    cudaEventRecord(dev->ev_D, dev->str_D);
}

__global__ void convolution_kernel(fp_t* d_old,
                                   fp_t* d_new,
                                   const int nx,
                                   const int ny,
                                   const int nm)
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

	const int src_x = dst_x - nm / 2;
	const int src_y = dst_y - nm / 2;

	/* copy tile: __shared__ gives access to all threads working on this tile */
	extern __shared__ double4 d_tile[];

	if (src_x >= 0 && src_x < nx &&
	    src_y >= 0 && src_y < ny ) {
		/* if src_y==0, then dst_y==nm/2: this is a halo row */
		(d_tile[til_nx * til_y + til_x]).x = d_old[nx * src_y + src_x];
	}

	/* tile data is shared: wait for all threads to finish copying */
	__syncthreads();

	/* compute the convolution */
	if (til_x < dst_nx && til_y < dst_ny) {
		fp_t value = 0.;
		for (int j = 0; j < nm; j++) {
			for (int i = 0; i < nm; i++) {
				const double4& mid = (d_tile[til_nx * (til_y + j) + (til_x + i)]);
				value += d_mask[j * nm + i] * mid.x;
			}
		}
		/* record value */
		/* Note: tile is centered on [til_nx*(til_y+nm/2) + (til_x+nm/2)],
		         NOT [til_nx*til_y + til_x] */
		if (dst_y < ny && dst_x < nx) {
			d_new[nx * dst_y + dst_x] = value;
		}
	}
}

__device__ fp_t discrete_laplacian(const fp_t& D_low,
                                   const fp_t& D_mid,
                                   const fp_t& D_hi,
                                   const fp_t& c_low,
                                   const fp_t& c_mid,
                                   const fp_t& c_hi,
                                   const fp_t& dx)
{
	// Five-point stencil, 1 dimension at a time
	return ( (D_hi  + D_mid) * (c_hi  - c_mid)
           - (D_mid + D_low) * (c_mid - c_low) ) / (2.0 * dx * dx);
}

__global__ void chemical_convolution_Cr_kernel(fp_t* d_conc_Cr_gam, fp_t* d_conc_Nb_gam,
                                               fp_t* d_conc_Cr_del, fp_t* d_conc_Nb_del,
                                               fp_t* d_conc_Cr_lav, fp_t* d_conc_Nb_lav,
                                               fp_t* d_mob_gam_CrCr, fp_t* d_mob_gam_CrNb,
                                               fp_t* d_mob_gam_NbCr, fp_t* d_mob_gam_NbNb,
                                               fp_t* d_mob_del_CrCr, fp_t* d_mob_del_CrNb,
                                               fp_t* d_mob_del_NbCr, fp_t* d_mob_del_NbNb,
                                               fp_t* d_mob_lav_CrCr, fp_t* d_mob_lav_CrNb,
                                               fp_t* d_mob_lav_NbCr, fp_t* d_mob_lav_NbNb,
                                               fp_t* d_conc_Cr_new,
                                               const int nx, const int ny, const int nm,
                                               const fp_t dx, const fp_t dy)
{
	/* determine indices on which to operate */
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	const size_t idx = nx * y + x;

	/* compute the 5-point Laplacian with variable coefficients */
	if (x > nm/2 && x < nx - nm/2 &&
	    y > nm/2 && y < ny - nm/2) {
		/* Note: tile is centered on [til_nx*(til_y+nm/2) + (til_x+nm/2)] */
		const size_t mid = idx;
		const size_t lft = nx * y + (x - 1);
		const size_t rgt = nx * y + (x + 1);
		const size_t bot = nx * (y - 1) + x;
		const size_t top = nx * (y + 1) + x;

		// Finite Differences
		// Derivation: TKR5 pp. 301--305

		fp_t divDgradU_Cr = 0.0;

		// TKR5p303, Eqn. 7, term 1
		divDgradU_Cr += discrete_laplacian(d_mob_gam_CrCr[lft], d_mob_gam_CrCr[mid], d_mob_gam_CrCr[rgt],
                                           d_conc_Cr_gam[lft],  d_conc_Cr_gam[mid],  d_conc_Cr_gam[rgt], dx);
		divDgradU_Cr += discrete_laplacian(d_mob_gam_CrCr[bot], d_mob_gam_CrCr[mid], d_mob_gam_CrCr[top],
                                           d_conc_Cr_gam[bot],  d_conc_Cr_gam[mid],  d_conc_Cr_gam[top], dy);

		// TKR5p303, Eqn. 7, term 2
		divDgradU_Cr += discrete_laplacian(d_mob_gam_CrNb[lft], d_mob_gam_CrNb[mid], d_mob_gam_CrNb[rgt],
                                           d_conc_Nb_gam[lft],  d_conc_Nb_gam[mid],  d_conc_Nb_gam[rgt], dx);
		divDgradU_Cr += discrete_laplacian(d_mob_gam_CrNb[bot], d_mob_gam_CrNb[mid], d_mob_gam_CrNb[top],
                                           d_conc_Nb_gam[bot],  d_conc_Nb_gam[mid],  d_conc_Nb_gam[top], dy);

		// TKR5p303, Eqn. 7, term 3
		divDgradU_Cr += discrete_laplacian(d_mob_del_CrCr[lft], d_mob_del_CrCr[mid], d_mob_del_CrCr[rgt],
                                           d_conc_Cr_del[lft],  d_conc_Cr_del[mid],  d_conc_Cr_del[rgt], dx);
		divDgradU_Cr += discrete_laplacian(d_mob_del_CrCr[bot], d_mob_del_CrCr[mid], d_mob_del_CrCr[top],
                                           d_conc_Cr_del[bot],  d_conc_Cr_del[mid],  d_conc_Cr_del[top], dy);

		// TKR5p303, Eqn. 7, term 4
		divDgradU_Cr += discrete_laplacian(d_mob_del_CrNb[lft], d_mob_del_CrNb[mid], d_mob_del_CrNb[rgt],
                                           d_conc_Nb_del[lft],  d_conc_Nb_del[mid],  d_conc_Nb_del[rgt], dx);
		divDgradU_Cr += discrete_laplacian(d_mob_del_CrNb[bot], d_mob_del_CrNb[mid], d_mob_del_CrNb[top],
                                           d_conc_Nb_del[bot],  d_conc_Nb_del[mid],  d_conc_Nb_del[top], dy);

		// TKR5p303, Eqn. 7, term 5
		divDgradU_Cr += discrete_laplacian(d_mob_lav_CrCr[lft], d_mob_lav_CrCr[mid], d_mob_lav_CrCr[rgt],
                                           d_conc_Cr_lav[lft],  d_conc_Cr_lav[mid],  d_conc_Cr_lav[rgt], dx);
		divDgradU_Cr += discrete_laplacian(d_mob_lav_CrCr[bot], d_mob_lav_CrCr[mid], d_mob_lav_CrCr[top],
                                           d_conc_Cr_lav[bot],  d_conc_Cr_lav[mid],  d_conc_Cr_lav[top], dy);

		// TKR5p303, Eqn. 7, term 6
		divDgradU_Cr += discrete_laplacian(d_mob_lav_CrNb[lft], d_mob_lav_CrNb[mid], d_mob_lav_CrNb[rgt],
                                           d_conc_Nb_lav[lft],  d_conc_Nb_lav[mid],  d_conc_Nb_lav[rgt], dx);
		divDgradU_Cr += discrete_laplacian(d_mob_lav_CrNb[bot], d_mob_lav_CrNb[mid], d_mob_lav_CrNb[top],
                                           d_conc_Nb_lav[bot],  d_conc_Nb_lav[mid],  d_conc_Nb_lav[top], dy);

		/* record value */
		d_conc_Cr_new[idx] = divDgradU_Cr;
	}
}

__global__ void chemical_convolution_Nb_kernel(fp_t* d_conc_Cr_gam, fp_t* d_conc_Nb_gam,
                                               fp_t* d_conc_Cr_del, fp_t* d_conc_Nb_del,
                                               fp_t* d_conc_Cr_lav, fp_t* d_conc_Nb_lav,
                                               fp_t* d_mob_gam_CrCr, fp_t* d_mob_gam_CrNb,
                                               fp_t* d_mob_gam_NbCr, fp_t* d_mob_gam_NbNb,
                                               fp_t* d_mob_del_CrCr, fp_t* d_mob_del_CrNb,
                                               fp_t* d_mob_del_NbCr, fp_t* d_mob_del_NbNb,
                                               fp_t* d_mob_lav_CrCr, fp_t* d_mob_lav_CrNb,
                                               fp_t* d_mob_lav_NbCr, fp_t* d_mob_lav_NbNb,
                                               fp_t* d_conc_Nb_new,
                                               const int nx, const int ny, const int nm,
                                               const fp_t dx, const fp_t dy)
{
	/* determine indices on which to operate */
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	const size_t idx = nx * y + x;

	/* compute the 5-point Laplacian with variable coefficients */
	if (x > nm/2 && x < nx - nm/2 &&
	    y > nm/2 && y < ny - nm/2) {
		/* Note: tile is centered on [til_nx*(til_y+nm/2) + (til_x+nm/2)] */
		const size_t mid = idx;
		const size_t lft = nx * y + (x - 1);
		const size_t rgt = nx * y + (x + 1);
		const size_t bot = nx * (y - 1) + x;
		const size_t top = nx * (y + 1) + x;

		// Finite Differences
		// Derivation: TKR5 pp. 301--305

		fp_t divDgradU_Nb = 0.0;

		// TKR5p303, Eqn. 7, term 1
		divDgradU_Nb += discrete_laplacian(d_mob_gam_NbCr[lft], d_mob_gam_NbCr[mid], d_mob_gam_NbCr[rgt],
                                           d_conc_Cr_gam[lft],  d_conc_Cr_gam[mid],  d_conc_Cr_gam[rgt], dx);
		divDgradU_Nb += discrete_laplacian(d_mob_gam_NbCr[bot], d_mob_gam_NbCr[mid], d_mob_gam_NbCr[top],
                                           d_conc_Cr_gam[bot],  d_conc_Cr_gam[mid],  d_conc_Cr_gam[top], dy);

		// TKR5p303, Eqn. 7, term 2
		divDgradU_Nb += discrete_laplacian(d_mob_gam_NbNb[lft], d_mob_gam_NbNb[mid], d_mob_gam_NbNb[rgt],
                                           d_conc_Nb_gam[lft],  d_conc_Nb_gam[mid],  d_conc_Nb_gam[rgt], dx);
		divDgradU_Nb += discrete_laplacian(d_mob_gam_NbNb[bot], d_mob_gam_NbNb[mid], d_mob_gam_NbNb[top],
                                           d_conc_Nb_gam[bot],  d_conc_Nb_gam[mid],  d_conc_Nb_gam[top], dy);

		// TKR5p303, Eqn. 7, term 3
		divDgradU_Nb += discrete_laplacian(d_mob_del_NbCr[lft], d_mob_del_NbCr[mid], d_mob_del_NbCr[rgt],
                                           d_conc_Cr_del[lft],  d_conc_Cr_del[mid],  d_conc_Cr_del[rgt], dx);
		divDgradU_Nb += discrete_laplacian(d_mob_del_NbCr[bot], d_mob_del_NbCr[mid], d_mob_del_NbCr[top],
                                           d_conc_Cr_del[bot],  d_conc_Cr_del[mid],  d_conc_Cr_del[top], dx);

		// TKR5p303, Eqn. 7, term 4
		divDgradU_Nb += discrete_laplacian(d_mob_del_NbNb[lft], d_mob_del_NbNb[mid], d_mob_del_NbNb[rgt],
                                           d_conc_Nb_del[lft],  d_conc_Nb_del[mid],  d_conc_Nb_del[rgt], dx);
		divDgradU_Nb += discrete_laplacian(d_mob_del_NbNb[bot], d_mob_del_NbNb[mid], d_mob_del_NbNb[top],
                                           d_conc_Nb_del[bot],  d_conc_Nb_del[mid],  d_conc_Nb_del[top], dy);

		// TKR5p303, Eqn. 7, term 5
		divDgradU_Nb += discrete_laplacian(d_mob_lav_NbCr[lft], d_mob_lav_NbCr[mid], d_mob_lav_NbCr[rgt],
                                           d_conc_Cr_lav[lft],  d_conc_Cr_lav[mid],  d_conc_Cr_lav[rgt], dx);
		divDgradU_Nb += discrete_laplacian(d_mob_lav_NbCr[bot], d_mob_lav_NbCr[mid], d_mob_lav_NbCr[top],
                                           d_conc_Cr_lav[bot],  d_conc_Cr_lav[mid],  d_conc_Cr_lav[top], dy);

		// TKR5p303, Eqn. 7, term 6
		divDgradU_Nb += discrete_laplacian(d_mob_lav_NbNb[lft], d_mob_lav_NbNb[mid], d_mob_lav_NbNb[rgt],
                                           d_conc_Nb_lav[lft],  d_conc_Nb_lav[mid],  d_conc_Nb_lav[rgt], dx);
		divDgradU_Nb += discrete_laplacian(d_mob_lav_NbNb[bot], d_mob_lav_NbNb[mid], d_mob_lav_NbNb[top],
                                           d_conc_Nb_lav[bot],  d_conc_Nb_lav[mid],  d_conc_Nb_lav[top], dy);

		/* record value */
		d_conc_Nb_new[idx] = divDgradU_Nb;
	}
}

void device_laplacian(struct CudaData* dev,
                      const int nx, const int ny, const int nm,
                      const int bx, const int by,
                      const fp_t dx, const fp_t dy)
{
	/* divide matrices into blocks of bx * by threads */
	dim3 tile_size(bx, by, 1);
	dim3 num_tiles(nTiles(nx, tile_size.x, nm),
	               nTiles(ny, tile_size.y, nm),
	               1);
	const size_t buf_size = (tile_size.x + nm) * (tile_size.y + nm) * sizeof(double4);

    cudaStreamWaitEvent(dev->str_A, dev->ev_A, 0);
    cudaStreamWaitEvent(dev->str_B, dev->ev_B, 0);
    cudaStreamWaitEvent(dev->str_C, dev->ev_C, 0);
    cudaStreamWaitEvent(dev->str_D, dev->ev_D, 0);

	convolution_kernel <<< num_tiles, tile_size, buf_size, dev->str_A>>> (
	    dev->phi_del_old, dev->phi_del_new, nx, ny, nm);
	convolution_kernel <<< num_tiles, tile_size, buf_size, dev->str_B>>> (
	    dev->phi_lav_old, dev->phi_lav_new, nx, ny, nm);

	chemical_convolution_Cr_kernel <<< num_tiles, tile_size, 0, dev->str_C>>> (
	    dev->conc_Cr_gam, dev->conc_Nb_gam,
        dev->conc_Cr_del, dev->conc_Nb_del,
        dev->conc_Cr_lav, dev->conc_Nb_lav,
        dev->mob_gam_CrCr, dev->mob_gam_CrNb,
        dev->mob_gam_NbCr, dev->mob_gam_NbNb,
        dev->mob_del_CrCr, dev->mob_del_CrNb,
        dev->mob_del_NbCr, dev->mob_del_NbNb,
        dev->mob_lav_CrCr, dev->mob_lav_CrNb,
        dev->mob_lav_NbCr, dev->mob_lav_NbNb,
        dev->conc_Cr_new,
	    nx, ny, nm,
	    dx, dy);

	chemical_convolution_Nb_kernel <<< num_tiles, tile_size, 0, dev->str_D>>> (
	    dev->conc_Cr_gam, dev->conc_Nb_gam,
        dev->conc_Cr_del, dev->conc_Nb_del,
        dev->conc_Cr_lav, dev->conc_Nb_lav,
        dev->mob_gam_CrCr, dev->mob_gam_CrNb,
        dev->mob_gam_NbCr, dev->mob_gam_NbNb,
        dev->mob_del_CrCr, dev->mob_del_CrNb,
        dev->mob_del_NbCr, dev->mob_del_NbNb,
        dev->mob_lav_CrCr, dev->mob_lav_CrNb,
        dev->mob_lav_NbCr, dev->mob_lav_NbNb,
        dev->conc_Cr_new,
	    nx, ny, nm,
	    dx, dy);

    cudaEventRecord(dev->ev_A, dev->str_A);
    cudaEventRecord(dev->ev_B, dev->str_B);
    cudaEventRecord(dev->ev_C, dev->str_C);
    cudaEventRecord(dev->ev_D, dev->str_D);
}

__device__ void composition_kernel(const fp_t& d_conc_Cr_old, const fp_t& d_conc_Nb_old,
                                   fp_t& d_conc_Cr_new,       fp_t& d_conc_Nb_new,
                                   const fp_t dt)
{
	/* Cahn-Hilliard equations of motion for composition */
	const fp_t divDgradU_Cr = d_conc_Cr_new;
	const fp_t divDgradU_Nb = d_conc_Nb_new;

	d_conc_Cr_new = d_conc_Cr_old + dt * divDgradU_Cr;
	d_conc_Nb_new = d_conc_Nb_old + dt * divDgradU_Nb;
}

__global__ void cahn_hilliard_kernel(fp_t* d_conc_Cr_old, fp_t* d_conc_Nb_old,
                                     fp_t* d_phi_del_old, fp_t* d_phi_lav_old,
                                     fp_t* d_conc_Cr_new, fp_t* d_conc_Nb_new,
                                     const int nx, const int ny, const int nm,
                                     const fp_t dt)
{
	/* determine indices on which to operate */
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	const int idx = nx * y + x;

	/* explicit Euler solution to the equation of motion */
	if (x < nx && y < ny) {
		/* Cahn-Hilliard equations of motion for composition */
		composition_kernel(d_conc_Cr_old[idx],      d_conc_Nb_old[idx],
		                   d_conc_Cr_new[idx],      d_conc_Nb_new[idx],
		                   dt);
	}
}

__device__ void delta_kernel(const fp_t& conc_Cr_old, const fp_t& conc_Nb_old,
                             const fp_t& phi_del_old, const fp_t& phi_lav_old,
                             fp_t& phi_del_new,
                             const fp_t inv_fict_det,
                             const fp_t pDel,        const fp_t pLav,
                             const fp_t dgGdxCr,     const fp_t dgGdxNb,
                             const fp_t gam_Cr,      const fp_t gam_Nb,
                             const fp_t gam_nrg,     const fp_t alpha,
                             const fp_t dt)
{
	// Derivation: TKR5p281, Eqn. (14)

	const fp_t pGam = 1.0 - pDel - pLav;
	const fp_t del_Cr = d_fict_del_Cr(inv_fict_det, conc_Cr_old, conc_Nb_old, pDel, pGam, pLav);
	const fp_t del_Nb = d_fict_del_Nb(inv_fict_det, conc_Cr_old, conc_Nb_old, pDel, pGam, pLav);
	const fp_t del_nrg = d_g_del(del_Cr, del_Nb);

	/* pressure */
	const fp_t P_del = gam_nrg - del_nrg - dgGdxCr * (gam_Cr - del_Cr) - dgGdxNb * (gam_Nb - del_Nb);

	/* variational derivative */
	const fp_t dFdPhi_del = -d_pPrime(phi_del_old) * P_del
	                        + 2.0 * d_Omeg[0] * phi_del_old * (phi_del_old - 1.0) * (2.0 * phi_del_old - 1.0)
	                        + 2.0 * alpha * phi_del_old * phi_lav_old * phi_lav_old
	                        - d_Kapp[0] * phi_del_new;

	/* Allen-Cahn equation of motion for delta phase */
	phi_del_new = phi_del_old - dt * d_Lmob[0] * dFdPhi_del;
}

__device__ void laves_kernel(const fp_t& conc_Cr_old, const fp_t& conc_Nb_old,
                             const fp_t& phi_del_old, const fp_t& phi_lav_old,
                             fp_t& phi_lav_new,
                             const fp_t inv_fict_det,
                             const fp_t pDel,        const fp_t pLav,
                             const fp_t dgGdxCr,     const fp_t dgGdxNb,
                             const fp_t gam_Cr,      const fp_t gam_Nb,
                             const fp_t gam_nrg,     const fp_t alpha,
                             const fp_t dt)
{
	// Derivation: TKR5p281, Eqn. (14)

	const fp_t pGam = 1.0 - pDel - pLav;
	const fp_t lav_Cr = d_fict_lav_Cr(inv_fict_det, conc_Cr_old, conc_Nb_old, pDel, pGam, pLav);
	const fp_t lav_Nb = d_fict_lav_Nb(inv_fict_det, conc_Cr_old, conc_Nb_old, pDel, pGam, pLav);
	const fp_t lav_nrg = d_g_lav(lav_Cr, lav_Nb);

	/* pressure */
	const fp_t P_lav = gam_nrg - lav_nrg - dgGdxCr * (gam_Cr - lav_Cr) - dgGdxNb * (gam_Nb - lav_Nb);

	/* variational derivative */
	const fp_t dFdPhi_lav = -d_pPrime(phi_lav_old) * P_lav
	                        + 2.0 * d_Omeg[1] * phi_lav_old * (phi_lav_old - 1.0) * (2.0 * phi_lav_old - 1.0)
	                        + 2.0 * alpha * phi_lav_old * phi_del_old * phi_del_old
	                        - d_Kapp[1] * phi_lav_new;

	/* Allen-Cahn equation of motion for Laves phase */
	phi_lav_new = phi_lav_old - dt * d_Lmob[1] * dFdPhi_lav;
}

__global__ void allen_cahn_kernel(fp_t* d_conc_Cr_old, fp_t* d_conc_Nb_old,
                                  fp_t* d_phi_del_old, fp_t* d_phi_lav_old,
                                  fp_t* d_phi_del_new, fp_t* d_phi_lav_new,
                                  const int nx, const int ny, const int nm,
                                  const fp_t alpha,
                                  const fp_t dt)
{
	/* determine indices on which to operate */
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	const int idx = nx * y + x;

	/* explicit Euler solution to the equation of motion */
	if (x < nx && y < ny) {
		const fp_t pDel = d_p(d_phi_del_old[idx]);
		const fp_t pLav = d_p(d_phi_lav_old[idx]);
		const fp_t pGam = 1.0 - pDel - pLav;
		const fp_t inv_fict_det = d_inv_fict_det(pDel, pGam, pLav);
		const fp_t gam_Cr = d_fict_gam_Cr(inv_fict_det, d_conc_Cr_old[idx], d_conc_Nb_old[idx], pDel, pGam, pLav);
		const fp_t gam_Nb = d_fict_gam_Nb(inv_fict_det, d_conc_Cr_old[idx], d_conc_Nb_old[idx], pDel, pGam, pLav);

		/* pure phase energy */
		const fp_t gam_nrg = d_g_gam(gam_Cr, gam_Nb);

		/* effective chemical potential */
		const fp_t dgGdxCr = d_dg_gam_dxCr(gam_Cr, gam_Nb);
		const fp_t dgGdxNb = d_dg_gam_dxNb(gam_Cr, gam_Nb);

		/* Allen-Cahn equations of motion for phase */
		delta_kernel(d_conc_Cr_old[idx], d_conc_Nb_old[idx], d_phi_del_old[idx], d_phi_lav_old[idx],
		             d_phi_del_new[idx], inv_fict_det, pDel, pLav, dgGdxCr, dgGdxNb,
		             gam_Cr, gam_Nb, gam_nrg, alpha, dt);

		laves_kernel(d_conc_Cr_old[idx], d_conc_Nb_old[idx], d_phi_del_old[idx], d_phi_lav_old[idx],
		             d_phi_lav_new[idx], inv_fict_det, pDel, pLav, dgGdxCr, dgGdxNb,
		             gam_Cr, gam_Nb, gam_nrg, alpha, dt);
	}
}

void device_evolution(struct CudaData* dev,
                      const int nx, const int ny, const int nm,
                      const int bx, const int by,
                      const fp_t alpha,
                      const fp_t dt)
{
	/* divide matrices into blocks of bx * by threads */
	dim3 tile_size(bx, by, 1);
	dim3 num_tiles(nTiles(nx, tile_size.x, nm),
	               nTiles(ny, tile_size.y, nm),
	               1);

    cudaStreamWaitEvent(dev->str_A, dev->ev_A, 0);
    cudaStreamWaitEvent(dev->str_B, dev->ev_B, 0);

	cahn_hilliard_kernel <<< num_tiles, tile_size, 0, dev->str_A>>> (
	    dev->conc_Cr_old, dev->conc_Nb_old,
	    dev->phi_del_old, dev->phi_lav_old,
	    dev->conc_Cr_new, dev->conc_Nb_new,
	    nx, ny, nm,
	    dt);

	allen_cahn_kernel <<< num_tiles, tile_size, 0, dev->str_B>>> (
	    dev->conc_Cr_old, dev->conc_Nb_old,
	    dev->phi_del_old, dev->phi_lav_old,
	    dev->phi_del_new, dev->phi_lav_new,
	    nx, ny, nm,
	    alpha,
	    dt);

    cudaEventRecord(dev->ev_A, dev->str_A);
    cudaEventRecord(dev->ev_B, dev->str_B);
}

__global__ void init_prng_kernel(curandState* d_prng, const int nx, const int ny)
{
	/* determine indices on which to operate */
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	const int idx = nx * y + x;

	if (x < nx && y < ny)
		curand_init((unsigned long long)clock() + idx, x, 0, &(d_prng[idx]));
}

void device_init_prng(struct CudaData* dev,
                      const int nx, const int ny, const int nm,
                      const int bx, const int by)
{
	/* divide matrices into blocks of bx * by threads */
	dim3 tile_size(bx, by, 1);
	dim3 num_tiles(nTiles(nx, tile_size.x, nm),
	               nTiles(ny, tile_size.y, nm),
	               1);
	init_prng_kernel <<< num_tiles, tile_size>>> (
	    dev->prng, nx, ny);
}

__device__ void embed_OPC_kernel(fp_t* d_conc_Cr, fp_t* d_conc_Nb,
                                 fp_t* d_phi_del, fp_t* d_phi_lav,
                                 const int nx, const int ny,
                                 const int x, const int y, const int idx,
                                 const fp_t xCr,
                                 const fp_t xNb,
                                 const fp_t par_xe_Cr,
                                 const fp_t par_xe_Nb,
                                 const fp_t R_precip)
{
	const fp_t R_depletion_Cr = fp_t(R_precip) * sqrt((par_xe_Cr - xCr) / (xCr - d_xe_gam_Cr()));
	const fp_t R_depletion_Nb = fp_t(R_precip) * sqrt((par_xe_Nb - xNb) / (xNb - d_xe_gam_Nb()));

	for (int i = -R_precip; i < R_precip; i++) {
		for (int j = -R_precip; j < R_precip; j++) {
			const int idn = nx * (y + j) + (x + i);
			const fp_t r = sqrt(fp_t(i * i + j * j));
			if (idn >= 0 && idn < nx * ny) {
				if (r <= R_precip) {
					d_conc_Cr[idn] = par_xe_Cr;
					d_conc_Nb[idn] = par_xe_Nb;
					d_phi_del[idn] = 1.;
				} else {
					if (r <= R_depletion_Cr)
						d_conc_Cr[idn] = d_xe_gam_Cr();
					if (r <= R_depletion_Nb)
						d_conc_Nb[idn] = d_xe_gam_Nb();
				}
			}
		}
	}
}

__global__ void nucleation_kernel(fp_t* d_conc_Cr, fp_t* d_conc_Nb,
                                  fp_t* d_phi_del, fp_t* d_phi_lav,
                                  curandState* d_prng,
                                  const int nx, const int ny, const int nm,
                                  const fp_t sigma_del, const fp_t sigma_lav,
                                  const fp_t lattice_const,
                                  const fp_t ifce_width,
                                  const fp_t dx, const fp_t dy, const fp_t dz,
                                  const fp_t dt)
{
	/* determine indices on which to operate */
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;

	const fp_t dV = dx * dy * dz;
	const fp_t Vatom = 0.25 * lattice_const * lattice_const * lattice_const; // m³/atom, assuming FCC
	const fp_t n_gam = dV / Vatom; // atoms, assuming FCC

	fp_t phi_pre = 0.;
	fp_t dG_chem = 0.;
	fp_t R_precip, R_star;
	fp_t P_nuc;
	fp_t rand_pre;

	// Scan neighborhood for existing precipitates
	if (x < nx && y < ny) {
		const fp_t R = 1.75e-9 / dx;

		for (int i = -R; i < R; i++) {
			for (int j = -R; j < R; j++) {
				const int idn = nx * (y + j) + (x + i);
				const fp_t r = sqrt(fp_t(i * i + j * j));
				if (idn >= 0 &&
				    idn < nx * ny &&
				    i * i + j * j < R * R)
					phi_pre = max(phi_pre, d_p(d_phi_del[idn]) + d_p(d_phi_lav[idn]));
			}
		}
	}
	__syncthreads();

	if (x < nx && y < ny && phi_pre < 1e-10) {
		const int idx = nx * y + x;
		const fp_t xCr = d_conc_Cr[idx];
		const fp_t xNb = d_conc_Nb[idx];
		const fp_t pDel = d_p(d_phi_del[idx]);
		const fp_t pLav = d_p(d_phi_lav[idx]);
		const fp_t pGam = 1.0 - pDel - pLav;

		// Test a delta particle
		d_nucleation_driving_force_delta(xCr, xNb, &dG_chem);
		d_nucleation_probability_sphere(xCr, xNb,
		                                dG_chem,
		                                pGam * (d_M_CrCr(xCr, xNb) * d_d2g_gam_dxCrCr() + d_M_CrNb(xCr, xNb) * d_d2g_gam_dxCrNb()),
		                                pGam * (d_M_NbCr(xCr, xNb) * d_d2g_gam_dxNbCr() + d_M_NbNb(xCr, xNb) * d_d2g_gam_dxNbNb()),
		                                sigma_del,
		                                Vatom,
		                                n_gam,
		                                dV, dt,
		                                &R_star,
		                                &P_nuc);
		if (R_star > 0.) {
			R_precip = R_star / dx;
			rand_pre = P_nuc - (fp_t)curand_uniform_double(&(d_prng[idx]));

			if (rand_pre > 0)
				embed_OPC_kernel(d_conc_Cr, d_conc_Nb,
				                 d_phi_del, d_phi_lav,
				                 nx, ny,
				                 x, y, idx,
				                 xCr, xNb,
				                 d_xe_del_Cr(), d_xe_del_Nb(),
				                 R_precip);
		}

		// Test a Laves particle
		d_nucleation_driving_force_laves(xCr, xNb, &dG_chem);
		d_nucleation_probability_sphere(xCr, xNb,
		                                dG_chem,
		                                pGam * (d_M_CrCr(xCr, xNb) * d_d2g_gam_dxCrCr() + d_M_CrNb(xCr, xNb) * d_d2g_gam_dxCrNb()),
		                                pGam * (d_M_NbCr(xCr, xNb) * d_d2g_gam_dxNbCr() + d_M_NbNb(xCr, xNb) * d_d2g_gam_dxNbNb()),
		                                sigma_lav,
		                                Vatom,
		                                n_gam,
		                                dV, dt,
		                                &R_star,
		                                &P_nuc);
		if (R_star > 0.) {
			R_precip = R_star / dx;
			rand_pre = P_nuc - (fp_t)curand_uniform_double(&(d_prng[idx]));

			if (rand_pre > 0)
				embed_OPC_kernel(d_conc_Cr, d_conc_Nb,
				                 d_phi_lav, d_phi_lav,
				                 nx, ny,
				                 x, y, idx,
				                 xCr, xNb,
				                 d_xe_lav_Cr(), d_xe_lav_Nb(),
				                 R_precip);
		}
	}
}

void device_nucleation(struct CudaData* dev,
                       const int nx, const int ny, const int nm,
                       const int bx, const int by,
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
	nucleation_kernel <<< num_tiles, tile_size>>> (
	    dev->conc_Cr_new, dev->conc_Nb_new,
	    dev->phi_del_new, dev->phi_lav_new,
	    dev->prng,
	    nx, ny, nm,
	    sigma_del, sigma_lav,
	    lattice_const, ifce_width,
	    dx, dy, dz, dt);
}

__global__ void dataviz_kernel(fp_t* d_conc_Cr, fp_t* d_conc_Cr_viz,
                               fp_t* d_conc_Nb, fp_t* d_conc_Nb_viz,
                               fp_t* d_conc_Ni,
                               fp_t* d_phi_del, fp_t* d_phi_del_viz,
                               fp_t* d_phi_lav, fp_t* d_phi_lav_viz,
                               fp_t* d_phi,
                               const int nx, const int ny)
{
	const int thr_x = threadIdx.x;
	const int thr_y = threadIdx.y;
	const int x = blockDim.x * blockIdx.x + thr_x;
	const int y = blockDim.y * blockIdx.y + thr_y;
	const int idx = nx * y + x;

	if (x < nx && y < ny) {
		d_conc_Ni[idx] = 1.0 - d_conc_Cr[idx] - d_conc_Nb[idx];
		d_phi[idx] = d_p(d_phi_del[idx]) + d_p(d_phi_lav[idx]);
        d_conc_Cr_viz[idx] = d_conc_Cr[idx];
        d_conc_Nb_viz[idx] = d_conc_Nb[idx];
        d_phi_del_viz[idx] = d_phi_del[idx];
        d_phi_lav_viz[idx] = d_phi_lav[idx];
	}
}

void device_dataviz(struct CudaData* dev,
                    struct HostData* host,
                    const int nx, const int ny, const int nm,
                    const int bx, const int by)
{
	/* divide matrices into blocks of bx * by threads */
	dim3 tile_size(bx, by, 1);
	dim3 num_tiles(nTiles(nx, tile_size.x, nm),
	               nTiles(ny, tile_size.y, nm),
	               1);

    cudaStreamWaitEvent(dev->str_A, dev->ev_A, 0);
    cudaStreamWaitEvent(dev->str_B, dev->ev_B, 0);
    cudaStreamWaitEvent(dev->str_C, dev->ev_C, 0);
    cudaStreamWaitEvent(dev->str_D, dev->ev_D, 0);

	dataviz_kernel <<< num_tiles, tile_size>>>(
                                               dev->conc_Cr_old, dev->conc_Cr_viz,
                                               dev->conc_Nb_old, dev->conc_Nb_viz,
                                               dev->conc_Ni,
                                               dev->phi_del_old, dev->phi_del_viz,
                                               dev->phi_lav_old, dev->phi_lav_viz,
                                               dev->phi,
                                               nx, ny);

	cudaMemcpy(host->conc_Cr_new[0], dev->conc_Cr_viz, nx * ny * sizeof(fp_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(host->conc_Nb_new[0], dev->conc_Nb_viz, nx * ny * sizeof(fp_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(host->conc_Ni[0], dev->conc_Ni, nx * ny * sizeof(fp_t), cudaMemcpyDeviceToHost);

	cudaMemcpy(host->phi_del_new[0], dev->phi_del_viz, nx * ny * sizeof(fp_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(host->phi_lav_new[0], dev->phi_lav_viz, nx * ny * sizeof(fp_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(host->phi[0], dev->phi, nx * ny * sizeof(fp_t), cudaMemcpyDeviceToHost);
}
