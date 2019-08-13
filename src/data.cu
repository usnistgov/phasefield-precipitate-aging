/**
 \file  data.cu
 \brief Implementation of functions to create and destroy CudaData struct
*/

#include <curand.h>
#include "data.cuh"
#include "discretization.cuh"

void init_cuda(struct HostData* host,
               const int nx, const int ny, const int nm,
			   const fp_t* DCr, const fp_t* DNb,
			   const fp_t* kappa, const fp_t* omega, const fp_t* Lmob, 
			   struct CudaData* dev)
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

	/* transfer mask to protected memory on GPU */
	cudaMemcpyToSymbol(d_mask, host->mask_lap[0], nm * nm * sizeof(fp_t));

	/* transfer mobility data to protected memory on GPU */
	cudaMemcpyToSymbol(d_DCr, DCr, NC * sizeof(fp_t));
	cudaMemcpyToSymbol(d_DNb, DNb, NC * sizeof(fp_t));
	cudaMemcpyToSymbol(d_Kapp, kappa, NP * sizeof(fp_t));
	cudaMemcpyToSymbol(d_Omeg, omega, NP * sizeof(fp_t));
	cudaMemcpyToSymbol(d_Lmob, Lmob,  NP * sizeof(fp_t));

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

void read_out_result(struct CudaData* dev, struct HostData* host,
                     const int nx, const int ny)
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
