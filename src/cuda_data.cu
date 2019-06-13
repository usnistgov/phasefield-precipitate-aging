/**
 \file  cuda_data.cu
 \brief Implementation of functions to create and destroy CudaData struct
*/

#include <curand.h>
#include "cuda_data.h"
#include "cuda_kernels.cuh"

void init_cuda(struct HostData* host,
               const int nx, const int ny, const int nm, struct CudaData* dev)
{
	/* allocate memory on device */
	cudaMalloc((void**) &(dev->conc_Cr_old), nx * ny * sizeof(fp_t));
	cudaMalloc((void**) &(dev->conc_Cr_new), nx * ny * sizeof(fp_t));
	cudaMalloc((void**) &(dev->conc_Nb_old), nx * ny * sizeof(fp_t));
	cudaMalloc((void**) &(dev->conc_Nb_new), nx * ny * sizeof(fp_t));
	cudaMalloc((void**) &(dev->conc_Ni), nx * ny * sizeof(fp_t));

	cudaMalloc((void**) &(dev->phi_del_old), nx * ny * sizeof(fp_t));
	cudaMalloc((void**) &(dev->phi_del_new), nx * ny * sizeof(fp_t));
	cudaMalloc((void**) &(dev->phi_lav_old), nx * ny * sizeof(fp_t));
	cudaMalloc((void**) &(dev->phi_lav_new), nx * ny * sizeof(fp_t));
	cudaMalloc((void**) &(dev->phi),         nx * ny * sizeof(fp_t));

	cudaMalloc((void**) &(dev->gam_Cr), nx * ny * sizeof(fp_t));
	cudaMalloc((void**) &(dev->gam_Nb), nx * ny * sizeof(fp_t));
	cudaMalloc((void**) &(dev->lap_gam_Cr), nx * ny * sizeof(fp_t));
	cudaMalloc((void**) &(dev->lap_gam_Nb), nx * ny * sizeof(fp_t));

	cudaMalloc((void**) &(dev->prng), nx * ny * sizeof(curandState));

	/* transfer mask and boundary conditions to protected memory on GPU */
	cudaMemcpyToSymbol(d_mask, host->mask_lap[0], nm * nm * sizeof(fp_t));

	/* transfer data from host in to GPU */
	cudaMemcpy(dev->conc_Cr_old, host->conc_Cr_old[0], nx * ny * sizeof(fp_t),
	           cudaMemcpyHostToDevice);
	cudaMemcpy(dev->conc_Nb_old, host->conc_Nb_old[0], nx * ny * sizeof(fp_t),
	           cudaMemcpyHostToDevice);

	cudaMemcpy(dev->phi_del_old, host->phi_del_old[0], nx * ny * sizeof(fp_t),
	           cudaMemcpyHostToDevice);
	cudaMemcpy(dev->phi_lav_old, host->phi_lav_old[0], nx * ny * sizeof(fp_t),
	           cudaMemcpyHostToDevice);

	cudaMemcpy(dev->gam_Cr, host->gam_Cr[0], nx * ny * sizeof(fp_t),
	           cudaMemcpyHostToDevice);
	cudaMemcpy(dev->gam_Nb, host->gam_Nb[0], nx * ny * sizeof(fp_t),
	           cudaMemcpyHostToDevice);
}

void free_cuda(struct CudaData* dev)
{
	/* free memory on device */
	cudaFree(dev->conc_Cr_old);
	cudaFree(dev->conc_Cr_new);
	cudaFree(dev->conc_Nb_old);
	cudaFree(dev->conc_Nb_new);
	cudaFree(dev->conc_Ni);

	cudaFree(dev->phi_del_old);
	cudaFree(dev->phi_del_new);
	cudaFree(dev->phi_lav_old);
	cudaFree(dev->phi_lav_new);
	cudaFree(dev->phi);

	cudaFree(dev->gam_Cr);
	cudaFree(dev->gam_Nb);
	cudaFree(dev->lap_gam_Cr);
	cudaFree(dev->lap_gam_Nb);

	cudaFree(dev->prng);
}
