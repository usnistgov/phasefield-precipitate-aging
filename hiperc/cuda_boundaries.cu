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
 \file  cuda_boundaries.cu
 \brief Implementation of boundary condition functions with OpenMP threading
*/

#include <math.h>
#include <omp.h>

#include "cuda_kernels.cuh"

__global__ void boundary_kernel(fp_t* d_conc_Cr, fp_t* d_conc_Nb,
                                fp_t* d_phi_del,
                                fp_t* d_phi_lav,
                                fp_t* d_gam_Cr, fp_t* d_gam_Nb,
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
			d_gam_Cr[row * nx + ilo-1]  = d_gam_Cr[row * nx + ilo];
			d_gam_Nb[row * nx + ilo-1]  = d_gam_Nb[row * nx + ilo];
		}
		if (ihi+1 == col && row < ny) {
            /* right condition */
			d_conc_Cr[row * nx + ihi+1] = d_conc_Cr[row * nx + ihi];
			d_conc_Nb[row * nx + ihi+1] = d_conc_Nb[row * nx + ihi];
			d_phi_del[row * nx + ihi+1] = d_phi_del[row * nx + ihi];
			d_phi_lav[row * nx + ihi+1] = d_phi_lav[row * nx + ihi];
			d_gam_Cr[row * nx + ihi+1]  = d_gam_Cr[row * nx + ihi];
			d_gam_Nb[row * nx + ihi+1]  = d_gam_Nb[row * nx + ihi];
		}
		if (jlo-1 == row && col < nx) {
            /* bottom condition */
			d_conc_Cr[(jlo-1) * nx + col] = d_conc_Cr[jlo * nx + col];
			d_conc_Nb[(jlo-1) * nx + col] = d_conc_Nb[jlo * nx + col];
			d_phi_del[(jlo-1) * nx + col] = d_phi_del[jlo * nx + col];
			d_phi_lav[(jlo-1) * nx + col] = d_phi_lav[jlo * nx + col];
			d_gam_Cr[(jlo-1) * nx + col]  = d_gam_Cr[jlo * nx + col];
			d_gam_Nb[(jlo-1) * nx + col]  = d_gam_Nb[jlo * nx + col];
		}
		if (jhi+1 == row && col < nx) {
            /* top condition */
			d_conc_Cr[(jhi+1) * nx + col] = d_conc_Cr[jhi * nx + col];
			d_conc_Nb[(jhi+1) * nx + col] = d_conc_Nb[jhi * nx + col];
			d_gam_Cr[(jhi+1) * nx + col] = d_gam_Cr[jhi * nx + col];
			d_gam_Nb[(jhi+1) * nx + col] = d_gam_Nb[jhi * nx + col];
			d_phi_del[(jhi+1) * nx + col] = d_phi_del[jhi * nx + col];
			d_phi_lav[(jhi+1) * nx + col] = d_phi_lav[jhi * nx + col];
		}

		__syncthreads();
	}
}
