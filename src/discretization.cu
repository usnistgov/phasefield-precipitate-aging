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

__global__ void boundary_kernel(fp_t* d_conc_Cr,
								fp_t* d_conc_Nb,
                                fp_t* d_phi_del,
								fp_t* d_phi_lav,
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
			d_conc_Cr[row * nx + ilo - 1] = d_conc_Cr[row * nx + ilo];
			d_conc_Nb[row * nx + ilo - 1] = d_conc_Nb[row * nx + ilo];
			d_phi_del[row * nx + ilo - 1] = d_phi_del[row * nx + ilo];
			d_phi_lav[row * nx + ilo - 1] = d_phi_lav[row * nx + ilo];
		}
		if (ihi + 1 == col && row < ny) {
			/* right condition */
			d_conc_Cr[row * nx + ihi + 1] = d_conc_Cr[row * nx + ihi];
			d_conc_Nb[row * nx + ihi + 1] = d_conc_Nb[row * nx + ihi];
			d_phi_del[row * nx + ihi + 1] = d_phi_del[row * nx + ihi];
			d_phi_lav[row * nx + ihi + 1] = d_phi_lav[row * nx + ihi];
		}
		if (jlo - 1 == row && col < nx) {
			/* bottom condition */
			d_conc_Cr[(jlo - 1) * nx + col] = d_conc_Cr[jlo * nx + col];
			d_conc_Nb[(jlo - 1) * nx + col] = d_conc_Nb[jlo * nx + col];
			d_phi_del[(jlo - 1) * nx + col] = d_phi_del[jlo * nx + col];
			d_phi_lav[(jlo - 1) * nx + col] = d_phi_lav[jlo * nx + col];
		}
		if (jhi + 1 == row && col < nx) {
			/* top condition */
			d_conc_Cr[(jhi + 1) * nx + col] = d_conc_Cr[jhi * nx + col];
			d_conc_Nb[(jhi + 1) * nx + col] = d_conc_Nb[jhi * nx + col];
			d_phi_del[(jhi + 1) * nx + col] = d_phi_del[jhi * nx + col];
			d_phi_lav[(jhi + 1) * nx + col] = d_phi_lav[jhi * nx + col];
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

	boundary_kernel <<< num_tiles, tile_size>>> (
	    dev->conc_Cr_old, dev->conc_Nb_old,
	    dev->phi_del_old, dev->phi_lav_old,
	    nx, ny, nm);
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

	boundary_kernel <<< num_tiles, tile_size>>> (
	    dev->conc_Cr_new, dev->conc_Nb_new,
	    dev->phi_del_new, dev->phi_lav_new,
	    nx, ny, nm);
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

__device__ fp_t discrete_laplacian(const fp_t& D_middle,
								   const fp_t& D_left, const fp_t& D_right,
								   const fp_t& D_bottom, const fp_t& D_top,
								   const fp_t& c_middle,
								   const fp_t& c_left, const fp_t& c_right,
								   const fp_t& c_bottom, const fp_t& c_top,
								   const fp_t& dx, const fp_t& dy)
{
	// Five-point stencil
	return ( (D_right + D_middle) * (c_right - c_middle)
	       - (D_middle + D_left) * (c_middle - c_left)) / (2.0 * dx * dx)
	     + ( (D_top + D_middle) * (c_top - c_middle)
		   - (D_middle + D_bottom) * (c_middle - c_bottom)) / (2.0 * dy * dy);
}

__device__ void fictitious(const fp_t xCr, const fp_t xNb, const fp_t pDel, const fp_t pLav,
                           fp_t* gam_Cr, fp_t* gam_Nb, fp_t* del_Cr, fp_t* del_Nb, fp_t* lav_Cr, fp_t* lav_Nb)
{
	const fp_t pGam = 1.0 - pDel - pLav;
    const fp_t inv_det = d_inv_fict_det(pDel, pGam, pLav);
    *gam_Cr = d_fict_gam_Cr(inv_det, xCr, xNb, pDel, pGam, pLav);
    *del_Cr = d_fict_del_Cr(inv_det, xCr, xNb, pDel, pGam, pLav);
    *lav_Cr = d_fict_lav_Cr(inv_det, xCr, xNb, pDel, pGam, pLav);
    *gam_Nb = d_fict_gam_Nb(inv_det, xCr, xNb, pDel, pGam, pLav);
    *del_Nb = d_fict_del_Nb(inv_det, xCr, xNb, pDel, pGam, pLav);
    *lav_Nb = d_fict_lav_Nb(inv_det, xCr, xNb, pDel, pGam, pLav);
}

__device__ fp_t D_gam_CrCr(const fp_t xCr, const fp_t xNb) {
	return d_M_CrCr(xCr, xNb) * d_d2g_gam_dxCrCr() + d_M_CrNb(xCr, xNb) * d_d2g_gam_dxCrNb();
}
__device__ fp_t D_gam_CrNb(const fp_t xCr, const fp_t xNb) {
	return d_M_CrCr(xCr, xNb) * d_d2g_gam_dxNbCr() + d_M_CrNb(xCr, xNb) * d_d2g_gam_dxNbNb();
}
__device__ fp_t D_gam_NbCr(const fp_t xCr, const fp_t xNb) {
	return d_M_NbCr(xCr, xNb) * d_d2g_gam_dxCrCr() + d_M_NbNb(xCr, xNb) * d_d2g_gam_dxCrNb();
}
__device__ fp_t D_gam_NbNb(const fp_t xCr, const fp_t xNb) {
	return d_M_NbCr(xCr, xNb) * d_d2g_gam_dxNbCr() + d_M_NbNb(xCr, xNb) * d_d2g_gam_dxNbNb();
}

__device__ fp_t D_del_CrCr(const fp_t xCr, const fp_t xNb) {
	return d_M_CrCr(xCr, xNb) * d_d2g_del_dxCrCr() + d_M_CrNb(xCr, xNb) * d_d2g_del_dxCrNb();
}
__device__ fp_t D_del_CrNb(const fp_t xCr, const fp_t xNb) {
	return d_M_CrCr(xCr, xNb) * d_d2g_del_dxNbCr() + d_M_CrNb(xCr, xNb) * d_d2g_del_dxNbNb();
}
__device__ fp_t D_del_NbCr(const fp_t xCr, const fp_t xNb) {
	return d_M_NbCr(xCr, xNb) * d_d2g_del_dxCrCr() + d_M_NbNb(xCr, xNb) * d_d2g_del_dxCrNb();
}
__device__ fp_t D_del_NbNb(const fp_t xCr, const fp_t xNb) {
	return d_M_NbCr(xCr, xNb) * d_d2g_del_dxNbCr() + d_M_NbNb(xCr, xNb) * d_d2g_del_dxNbNb();
}

__device__ fp_t D_lav_CrCr(const fp_t xCr, const fp_t xNb) {
	return d_M_CrCr(xCr, xNb) * d_d2g_lav_dxCrCr() + d_M_CrNb(xCr, xNb) * d_d2g_lav_dxCrNb();
}
__device__ fp_t D_lav_CrNb(const fp_t xCr, const fp_t xNb) {
	return d_M_CrCr(xCr, xNb) * d_d2g_lav_dxNbCr() + d_M_CrNb(xCr, xNb) * d_d2g_lav_dxNbNb();
}
__device__ fp_t D_lav_NbCr(const fp_t xCr, const fp_t xNb) {
	return d_M_NbCr(xCr, xNb) * d_d2g_lav_dxCrCr() + d_M_NbNb(xCr, xNb) * d_d2g_lav_dxCrNb();
}
__device__ fp_t D_lav_NbNb(const fp_t xCr, const fp_t xNb) {
	return d_M_NbCr(xCr, xNb) * d_d2g_lav_dxNbCr() + d_M_NbNb(xCr, xNb) * d_d2g_lav_dxNbNb();
}

__global__ void chemical_convolution_kernel(fp_t* d_phi_del_old, fp_t* d_phi_lav_old,
        fp_t* d_conc_Cr_old, fp_t* d_conc_Cr_new,
        fp_t* d_conc_Nb_old, fp_t* d_conc_Nb_new,
        const int nx, const int ny, const int nm,
        const fp_t dx, const fp_t dy)
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
		(d_tile[til_nx * til_y + til_x]).x = d_conc_Cr_old[nx * src_y + src_x];
		(d_tile[til_nx * til_y + til_x]).y = d_conc_Nb_old[nx * src_y + src_x];
		(d_tile[til_nx * til_y + til_x]).z = d_phi_del_old[nx * src_y + src_x];
		(d_tile[til_nx * til_y + til_x]).w = d_phi_lav_old[nx * src_y + src_x];
	}

	/* tile data is shared: wait for all threads to finish copying */
	__syncthreads();

	/* compute the 5-point Laplacian with variable coefficients */
	if (til_x < dst_nx && til_y < dst_ny) {
		/* Note: tile is centered on [til_nx*(til_y+nm/2) + (til_x+nm/2)] */
		const size_t til_mdx = til_x + 1;
		const size_t til_mdy = til_y + 1;
		const size_t til_lft = til_x;
		const size_t til_rgt = til_x + 2;
		const size_t til_bot = til_y;
		const size_t til_top = til_y + 2;

		const double4* mid = &(d_tile[til_nx * til_mdy + til_mdx]);
		const double4* lft = &(d_tile[til_nx * til_mdy + til_lft]);
		const double4* rgt = &(d_tile[til_nx * til_mdy + til_rgt]);
		const double4* bot = &(d_tile[til_nx * til_bot + til_mdx]);
		const double4* top = &(d_tile[til_nx * til_top + til_mdx]);

		const fp_t mid_pDel = d_p(mid->z);
		const fp_t mid_pLav = d_p(mid->w);
		const fp_t mid_pGam = 1.0 - mid_pDel - mid_pLav;

		const fp_t lft_pDel = d_p(lft->z);
		const fp_t lft_pLav = d_p(lft->w);
		const fp_t lft_pGam = 1.0 - lft_pDel - lft_pLav;

		const fp_t rgt_pDel = d_p(rgt->z);
		const fp_t rgt_pLav = d_p(rgt->w);
		const fp_t rgt_pGam = 1.0 - rgt_pDel - rgt_pLav;

		const fp_t bot_pDel = d_p(bot->z);
		const fp_t bot_pLav = d_p(bot->w);
		const fp_t bot_pGam = 1.0 - bot_pDel - bot_pLav;

		const fp_t top_pDel = d_p(top->z);
		const fp_t top_pLav = d_p(top->w);
		const fp_t top_pGam = 1.0 - top_pDel - top_pLav;

		// Fictitious compositions
		fp_t mid_gam_Cr, mid_gam_Nb, mid_del_Cr, mid_del_Nb, mid_lav_Cr, mid_lav_Nb;
		fp_t lft_gam_Cr, lft_gam_Nb, lft_del_Cr, lft_del_Nb, lft_lav_Cr, lft_lav_Nb;
		fp_t rgt_gam_Cr, rgt_gam_Nb, rgt_del_Cr, rgt_del_Nb, rgt_lav_Cr, rgt_lav_Nb;
		fp_t bot_gam_Cr, bot_gam_Nb, bot_del_Cr, bot_del_Nb, bot_lav_Cr, bot_lav_Nb;
		fp_t top_gam_Cr, top_gam_Nb, top_del_Cr, top_del_Nb, top_lav_Cr, top_lav_Nb;
        fictitious(mid->x, mid->y, mid_pDel, mid_pLav, &mid_gam_Cr, &mid_gam_Nb, &mid_del_Cr, &mid_del_Nb, &mid_lav_Cr, &mid_lav_Nb);
        fictitious(lft->x, lft->y, lft_pDel, lft_pLav, &lft_gam_Cr, &lft_gam_Nb, &lft_del_Cr, &lft_del_Nb, &lft_lav_Cr, &lft_lav_Nb);
        fictitious(rgt->x, rgt->y, rgt_pDel, rgt_pLav, &rgt_gam_Cr, &rgt_gam_Nb, &rgt_del_Cr, &rgt_del_Nb, &rgt_lav_Cr, &rgt_lav_Nb);
        fictitious(bot->x, bot->y, bot_pDel, bot_pLav, &bot_gam_Cr, &bot_gam_Nb, &bot_del_Cr, &bot_del_Nb, &bot_lav_Cr, &bot_lav_Nb);
        fictitious(top->x, top->y, top_pDel, top_pLav, &top_gam_Cr, &top_gam_Nb, &top_del_Cr, &top_del_Nb, &top_lav_Cr, &top_lav_Nb);

		// Finite Differences
		// Derivation: TKR5 pp. 301--305

		fp_t divDgradU_Cr = 0.0;
		fp_t divDgradU_Nb = 0.0;
		fp_t mid_D, lft_D, rgt_D, top_D, bot_D;

		{ // k = Cr
			// TKR5p303, Eqn. 7, term 1
			mid_D = mid_pGam * D_gam_CrCr(mid->x, mid->y);
			lft_D = lft_pGam * D_gam_CrCr(lft->x, lft->y);
			rgt_D = rgt_pGam * D_gam_CrCr(rgt->x, rgt->y);
			bot_D = bot_pGam * D_gam_CrCr(bot->x, bot->y);
			top_D = top_pGam * D_gam_CrCr(top->x, top->y);
			divDgradU_Cr += discrete_laplacian(mid_D, lft_D, rgt_D, bot_D, top_D,
											   mid_gam_Cr, lft_gam_Cr, rgt_gam_Cr, bot_gam_Cr, top_gam_Cr, dx, dy);

			// TKR5p303, Eqn. 7, term 2
			mid_D = mid_pGam * D_gam_CrNb(mid->x, mid->y);
			lft_D = lft_pGam * D_gam_CrNb(lft->x, lft->y);
			rgt_D = rgt_pGam * D_gam_CrNb(rgt->x, rgt->y);
			bot_D = bot_pGam * D_gam_CrNb(bot->x, bot->y);
			top_D = top_pGam * D_gam_CrNb(top->x, top->y);
			divDgradU_Cr += discrete_laplacian(mid_D, lft_D, rgt_D, bot_D, top_D,
											   mid_gam_Nb, lft_gam_Nb, rgt_gam_Nb, bot_gam_Nb, top_gam_Nb, dx, dy);

			// TKR5p303, Eqn. 7, term 3
			mid_D = mid_pDel * D_del_CrCr(mid->x, mid->y);
			lft_D = lft_pDel * D_del_CrCr(lft->x, lft->y);
			rgt_D = rgt_pDel * D_del_CrCr(rgt->x, rgt->y);
			bot_D = bot_pDel * D_del_CrCr(bot->x, bot->y);
			top_D = top_pDel * D_del_CrCr(top->x, top->y);
			divDgradU_Cr += discrete_laplacian(mid_D, lft_D, rgt_D, bot_D, top_D,
											   mid_del_Cr, lft_del_Cr, rgt_del_Cr, bot_del_Cr, top_del_Cr, dx, dy);

			// TKR5p303, Eqn. 7, term 4
			mid_D = mid_pDel * D_del_CrNb(mid->x, mid->y);
			lft_D = lft_pDel * D_del_CrNb(lft->x, lft->y);
			rgt_D = rgt_pDel * D_del_CrNb(rgt->x, rgt->y);
			bot_D = bot_pDel * D_del_CrNb(bot->x, bot->y);
			top_D = top_pDel * D_del_CrNb(top->x, top->y);
			divDgradU_Cr += discrete_laplacian(mid_D, lft_D, rgt_D, bot_D, top_D,
											   mid_del_Nb, lft_del_Nb, rgt_del_Nb, bot_del_Nb, top_del_Nb, dx, dy);

			// TKR5p303, Eqn. 7, term 5
			mid_D = mid_pLav * D_lav_CrCr(mid->x, mid->y);
			lft_D = lft_pLav * D_lav_CrCr(lft->x, lft->y);
			rgt_D = rgt_pLav * D_lav_CrCr(rgt->x, rgt->y);
			bot_D = bot_pLav * D_lav_CrCr(bot->x, bot->y);
			top_D = top_pLav * D_lav_CrCr(top->x, top->y);
			divDgradU_Cr += discrete_laplacian(mid_D, lft_D, rgt_D, bot_D, top_D,
											   mid_lav_Cr, lft_lav_Cr, rgt_lav_Cr, bot_lav_Cr, top_lav_Cr, dx, dy);

			// TKR5p303, Eqn. 7, term 6
			mid_D = mid_pLav * D_lav_CrNb(mid->x, mid->y);
			lft_D = lft_pLav * D_lav_CrNb(lft->x, lft->y);
			rgt_D = rgt_pLav * D_lav_CrNb(rgt->x, rgt->y);
			bot_D = bot_pLav * D_lav_CrNb(bot->x, bot->y);
			top_D = top_pLav * D_lav_CrNb(top->x, top->y);
			divDgradU_Cr += discrete_laplacian(mid_D, lft_D, rgt_D, bot_D, top_D,
											   mid_lav_Nb, lft_lav_Nb, rgt_lav_Nb, bot_lav_Nb, top_lav_Nb, dx, dy);
		}
		{ // k = Nb
			// TKR5p303, Eqn. 7, term 1
			mid_D = mid_pGam * D_gam_NbCr(mid->x, mid->y);
			lft_D = lft_pGam * D_gam_NbCr(lft->x, lft->y);
			rgt_D = rgt_pGam * D_gam_NbCr(rgt->x, rgt->y);
			bot_D = bot_pGam * D_gam_NbCr(bot->x, bot->y);
			top_D = top_pGam * D_gam_NbCr(top->x, top->y);
			divDgradU_Nb += discrete_laplacian(mid_D, lft_D, rgt_D, bot_D, top_D,
											   mid_gam_Cr, lft_gam_Cr, rgt_gam_Cr, bot_gam_Cr, top_gam_Cr, dx, dy);

			// TKR5p303, Eqn. 7, term 2
			mid_D = mid_pGam * D_gam_NbNb(mid->x, mid->y);
			lft_D = lft_pGam * D_gam_NbNb(lft->x, lft->y);
			rgt_D = rgt_pGam * D_gam_NbNb(rgt->x, rgt->y);
			bot_D = bot_pGam * D_gam_NbNb(bot->x, bot->y);
			top_D = top_pGam * D_gam_NbNb(top->x, top->y);
			divDgradU_Nb += discrete_laplacian(mid_D, lft_D, rgt_D, bot_D, top_D,
											   mid_gam_Nb, lft_gam_Nb, rgt_gam_Nb, bot_gam_Nb, top_gam_Nb, dx, dy);

			// TKR5p303, Eqn. 7, term 3
			mid_D = mid_pDel * D_del_NbCr(mid->x, mid->y);
			lft_D = lft_pDel * D_del_NbCr(lft->x, lft->y);
			rgt_D = rgt_pDel * D_del_NbCr(rgt->x, rgt->y);
			bot_D = bot_pDel * D_del_NbCr(bot->x, bot->y);
			top_D = top_pDel * D_del_NbCr(top->x, top->y);
			divDgradU_Nb += discrete_laplacian(mid_D, lft_D, rgt_D, bot_D, top_D,
											   mid_del_Cr, lft_del_Cr, rgt_del_Cr, bot_del_Cr, top_del_Cr, dx, dy);

			// TKR5p303, Eqn. 7, term 4
			mid_D = mid_pDel * D_del_NbNb(mid->x, mid->y);
			lft_D = lft_pDel * D_del_NbNb(lft->x, lft->y);
			rgt_D = rgt_pDel * D_del_NbNb(rgt->x, rgt->y);
			bot_D = bot_pDel * D_del_NbNb(bot->x, bot->y);
			top_D = top_pDel * D_del_NbNb(top->x, top->y);
			divDgradU_Nb += discrete_laplacian(mid_D, lft_D, rgt_D, bot_D, top_D,
											   mid_del_Nb, lft_del_Nb, rgt_del_Nb, bot_del_Nb, top_del_Nb, dx, dy);

			// TKR5p303, Eqn. 7, term 5
			mid_D = mid_pLav * D_lav_NbCr(mid->x, mid->y);
			lft_D = lft_pLav * D_lav_NbCr(lft->x, lft->y);
			rgt_D = rgt_pLav * D_lav_NbCr(rgt->x, rgt->y);
			bot_D = bot_pLav * D_lav_NbCr(bot->x, bot->y);
			top_D = top_pLav * D_lav_NbCr(top->x, top->y);
			divDgradU_Nb += discrete_laplacian(mid_D, lft_D, rgt_D, bot_D, top_D,
											   mid_lav_Cr, lft_lav_Cr, rgt_lav_Cr, bot_lav_Cr, top_lav_Cr, dx, dy);

			// TKR5p303, Eqn. 7, term 6
			mid_D = mid_pLav * D_lav_NbNb(mid->x, mid->y);
			lft_D = lft_pLav * D_lav_NbNb(lft->x, lft->y);
			rgt_D = rgt_pLav * D_lav_NbNb(rgt->x, rgt->y);
			bot_D = bot_pLav * D_lav_NbNb(bot->x, bot->y);
			top_D = top_pLav * D_lav_NbNb(top->x, top->y);
			divDgradU_Nb += discrete_laplacian(mid_D, lft_D, rgt_D, bot_D, top_D,
											   mid_lav_Nb, lft_lav_Nb, rgt_lav_Nb, bot_lav_Nb, top_lav_Nb, dx, dy);
		}

		/* record value */
		if (dst_y < ny && dst_x < nx) {
			d_conc_Cr_new[nx * dst_y + dst_x] = divDgradU_Cr;
			d_conc_Nb_new[nx * dst_y + dst_x] = divDgradU_Nb;
		}
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

	convolution_kernel <<< num_tiles, tile_size, buf_size>>> (
	    dev->phi_del_old, dev->phi_del_new, nx, ny, nm);
	convolution_kernel <<< num_tiles, tile_size, buf_size>>> (
	    dev->phi_lav_old, dev->phi_lav_new, nx, ny, nm);

	chemical_convolution_kernel <<< num_tiles, tile_size, buf_size>>> (
	    dev->phi_del_old, dev->phi_lav_old,
	    dev->conc_Cr_old, dev->conc_Cr_new,
	    dev->conc_Nb_old, dev->conc_Nb_new,
	    nx, ny, nm,
	    dx, dy);
}

__device__ void composition_kernel(const fp_t& d_conc_Cr_old, const fp_t& d_conc_Nb_old,
                                   const fp_t& d_frac_del,    const fp_t& d_frac_lav,
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
		                   d_p(d_phi_del_old[idx]), d_p(d_phi_lav_old[idx]),
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
	cahn_hilliard_kernel <<< num_tiles, tile_size>>> (
	    dev->conc_Cr_old, dev->conc_Nb_old,
	    dev->phi_del_old, dev->phi_lav_old,
	    dev->conc_Cr_new, dev->conc_Nb_new,
	    nx, ny, nm,
	    dt);

	allen_cahn_kernel <<< num_tiles, tile_size>>> (
	    dev->conc_Cr_old, dev->conc_Nb_old,
	    dev->phi_del_old, dev->phi_lav_old,
	    dev->phi_del_new, dev->phi_lav_new,
	    nx, ny, nm,
	    alpha,
	    dt);
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

__global__ void dataviz_kernel(fp_t* d_conc_Cr, fp_t* d_conc_Nb, fp_t* d_conc_Ni,
                               fp_t* d_phi_del, fp_t* d_phi_lav, fp_t* d_phi,
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
	}
}

void device_dataviz(struct CudaData* dev, struct HostData* host,
                    const int nx, const int ny, const int nm,
                    const int bx, const int by)
{
	/* divide matrices into blocks of bx * by threads */
	dim3 tile_size(bx, by, 1);
	dim3 num_tiles(nTiles(nx, tile_size.x, nm),
	               nTiles(ny, tile_size.y, nm),
	               1);

	dataviz_kernel <<< num_tiles, tile_size>>>(
	    dev->conc_Cr_old, dev->conc_Nb_old, dev->conc_Ni,
	    dev->phi_del_old, dev->phi_lav_old, dev->phi,
	    nx, ny);

	cudaMemcpy(host->conc_Cr_new[0], dev->conc_Cr_old, nx * ny * sizeof(fp_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(host->conc_Nb_new[0], dev->conc_Nb_old, nx * ny * sizeof(fp_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(host->conc_Ni[0], dev->conc_Ni, nx * ny * sizeof(fp_t), cudaMemcpyDeviceToHost);

	cudaMemcpy(host->phi_del_new[0], dev->phi_del_old, nx * ny * sizeof(fp_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(host->phi_lav_new[0], dev->phi_lav_old, nx * ny * sizeof(fp_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(host->phi[0], dev->phi, nx * ny * sizeof(fp_t), cudaMemcpyDeviceToHost);
}
