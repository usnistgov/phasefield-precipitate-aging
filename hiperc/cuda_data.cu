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
 \file  cuda_data.cu
 \brief Implementation of functions to create and destroy CudaData struct
*/

extern "C" {
#include "cuda_data.h"
}

#include "cuda_kernels.cuh"

void init_cuda(struct HostData* host, fp_t** mask_lap,
               const int nx, const int ny, const int nm, struct CudaData* dev)
{
	/* allocate memory on device */
	cudaMalloc((void**) &(dev->conc_Cr_old), nx * ny * sizeof(fp_t));
	cudaMalloc((void**) &(dev->conc_Cr_new), nx * ny * sizeof(fp_t));
	cudaMalloc((void**) &(dev->conc_Nb_old), nx * ny * sizeof(fp_t));
	cudaMalloc((void**) &(dev->conc_Nb_new), nx * ny * sizeof(fp_t));

    cudaMalloc((void**) &(dev->phi_del_old), nx * ny * sizeof(fp_t));
	cudaMalloc((void**) &(dev->phi_del_new), nx * ny * sizeof(fp_t));
	cudaMalloc((void**) &(dev->phi_lav_old), nx * ny * sizeof(fp_t));
	cudaMalloc((void**) &(dev->phi_lav_new), nx * ny * sizeof(fp_t));

    cudaMalloc((void**) &(dev->gam_Cr_old), nx * ny * sizeof(fp_t));
	cudaMalloc((void**) &(dev->gam_Cr_new), nx * ny * sizeof(fp_t));
	cudaMalloc((void**) &(dev->gam_Nb_old), nx * ny * sizeof(fp_t));
	cudaMalloc((void**) &(dev->gam_Nb_new), nx * ny * sizeof(fp_t));

	/* transfer mask and boundary conditions to protected memory on GPU */
	cudaMemcpyToSymbol(d_mask, mask_lap[0], nm * nm * sizeof(fp_t));

	/* transfer data from host in to GPU */
	cudaMemcpy(dev->conc_Cr_old, (host->conc_Cr_old)[0], nx * ny * sizeof(fp_t),
	           cudaMemcpyHostToDevice);
	cudaMemcpy(dev->conc_Nb_old, (host->conc_Nb_old)[0], nx * ny * sizeof(fp_t),
	           cudaMemcpyHostToDevice);

	cudaMemcpy(dev->phi_del_old, (host->phi_del_old)[0], nx * ny * sizeof(fp_t),
	           cudaMemcpyHostToDevice);
	cudaMemcpy(dev->phi_lav_old, (host->phi_lav_old)[0], nx * ny * sizeof(fp_t),
	           cudaMemcpyHostToDevice);

    cudaMemcpy(dev->gam_Cr_old, (host->gam_Cr_old)[0], nx * ny * sizeof(fp_t),
	           cudaMemcpyHostToDevice);
	cudaMemcpy(dev->gam_Nb_old, (host->gam_Nb_old)[0], nx * ny * sizeof(fp_t),
	           cudaMemcpyHostToDevice);
}

void free_cuda(struct CudaData* dev)
{
	/* free memory on device */
	cudaFree(dev->conc_Cr_old);
	cudaFree(dev->conc_Cr_new);
	cudaFree(dev->conc_Nb_old);
	cudaFree(dev->conc_Nb_new);

	cudaFree(dev->phi_del_old);
	cudaFree(dev->phi_del_new);
	cudaFree(dev->phi_lav_old);
	cudaFree(dev->phi_lav_new);

    cudaFree(dev->gam_Cr_old);
	cudaFree(dev->gam_Cr_new);
	cudaFree(dev->gam_Nb_old);
	cudaFree(dev->gam_Nb_new);   
}
