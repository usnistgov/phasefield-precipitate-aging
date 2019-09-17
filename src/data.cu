/**
 \file  data.cu
 \brief Implementation of functions to create and destroy CudaData struct
*/

#include <curand.h>
#include "data.cuh"
#include "discretization.cuh"

void init_cuda(struct HostData* host,
               const int nx, const int ny, const int nm,
			   const fp_t* kappa, const fp_t* omega, const fp_t* Lmob, 
			   struct CudaData* dev)
{
	/* allocate memory on device */
	cudaMalloc((void**) &(dev->prng), nx * ny * sizeof(curandState));
	cudaStreamCreate(&(dev->str_A));
	cudaStreamCreate(&(dev->str_B));
	cudaStreamCreate(&(dev->str_C));
	cudaStreamCreate(&(dev->str_D));

	cudaEventCreate(&(dev->ev_A));
	cudaEventCreate(&(dev->ev_B));
	cudaEventCreate(&(dev->ev_C));
	cudaEventCreate(&(dev->ev_D));

	cudaMalloc((void**) &(dev->conc_Cr_old), nx * ny * sizeof(fp_t));
	cudaMalloc((void**) &(dev->conc_Cr_new), nx * ny * sizeof(fp_t));
	cudaMalloc((void**) &(dev->conc_Nb_old), nx * ny * sizeof(fp_t));
	cudaMalloc((void**) &(dev->conc_Nb_new), nx * ny * sizeof(fp_t));

	cudaMalloc((void**) &(dev->phi_del_old), nx * ny * sizeof(fp_t));
	cudaMalloc((void**) &(dev->phi_del_new), nx * ny * sizeof(fp_t));
	cudaMalloc((void**) &(dev->phi_lav_old), nx * ny * sizeof(fp_t));
	cudaMalloc((void**) &(dev->phi_lav_new), nx * ny * sizeof(fp_t));
	cudaMalloc((void**) &(dev->phi),         nx * ny * sizeof(fp_t));

	cudaMalloc((void**) &(dev->conc_Cr_gam), nx * ny * sizeof(fp_t));
	cudaMalloc((void**) &(dev->conc_Cr_del), nx * ny * sizeof(fp_t));
	cudaMalloc((void**) &(dev->conc_Cr_lav), nx * ny * sizeof(fp_t));
	cudaMalloc((void**) &(dev->conc_Nb_gam), nx * ny * sizeof(fp_t));
	cudaMalloc((void**) &(dev->conc_Nb_del), nx * ny * sizeof(fp_t));
	cudaMalloc((void**) &(dev->conc_Nb_lav), nx * ny * sizeof(fp_t));

	cudaMalloc((void**) &(dev->conc_Cr_viz), nx * ny * sizeof(fp_t));
	cudaMalloc((void**) &(dev->conc_Nb_viz), nx * ny * sizeof(fp_t));
	cudaMalloc((void**) &(dev->conc_Ni), nx * ny * sizeof(fp_t));
	cudaMalloc((void**) &(dev->phi_del_viz), nx * ny * sizeof(fp_t));
	cudaMalloc((void**) &(dev->phi_lav_viz), nx * ny * sizeof(fp_t));

	cudaMalloc((void**) &(dev->mob_gam_CrCr), nx * ny * sizeof(fp_t));
	cudaMalloc((void**) &(dev->mob_gam_CrNb), nx * ny * sizeof(fp_t));
	cudaMalloc((void**) &(dev->mob_gam_NbCr), nx * ny * sizeof(fp_t));
	cudaMalloc((void**) &(dev->mob_gam_NbNb), nx * ny * sizeof(fp_t));

	cudaMalloc((void**) &(dev->mob_del_CrCr), nx * ny * sizeof(fp_t));
	cudaMalloc((void**) &(dev->mob_del_CrNb), nx * ny * sizeof(fp_t));
	cudaMalloc((void**) &(dev->mob_del_NbCr), nx * ny * sizeof(fp_t));
	cudaMalloc((void**) &(dev->mob_del_NbNb), nx * ny * sizeof(fp_t));

	cudaMalloc((void**) &(dev->mob_lav_CrCr), nx * ny * sizeof(fp_t));
	cudaMalloc((void**) &(dev->mob_lav_CrNb), nx * ny * sizeof(fp_t));
	cudaMalloc((void**) &(dev->mob_lav_NbCr), nx * ny * sizeof(fp_t));
	cudaMalloc((void**) &(dev->mob_lav_NbNb), nx * ny * sizeof(fp_t));

	/* transfer mask to protected memory on GPU */
	cudaMemcpyToSymbol(d_mask, host->mask_lap[0], nm * nm * sizeof(fp_t));

	/* transfer mobility data to protected memory on GPU */
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
}

void free_cuda(struct CudaData* dev)
{
	/* free memory on device */
	cudaFree(dev->prng);

	cudaStreamDestroy(dev->str_A);
	cudaStreamDestroy(dev->str_B);
	cudaStreamDestroy(dev->str_C);
	cudaStreamDestroy(dev->str_D);

	cudaEventDestroy(dev->ev_A);
	cudaEventDestroy(dev->ev_B);
	cudaEventDestroy(dev->ev_C);
	cudaEventDestroy(dev->ev_D);

	cudaFree(dev->conc_Cr_old);
	cudaFree(dev->conc_Cr_new);
	cudaFree(dev->conc_Nb_old);
	cudaFree(dev->conc_Nb_new);

	cudaFree(dev->phi_del_old);
	cudaFree(dev->phi_del_new);
	cudaFree(dev->phi_lav_old);
	cudaFree(dev->phi_lav_new);
	cudaFree(dev->phi);

	cudaFree(dev->conc_Cr_gam);
	cudaFree(dev->conc_Cr_del);
	cudaFree(dev->conc_Cr_lav);
	cudaFree(dev->conc_Nb_gam);
	cudaFree(dev->conc_Nb_del);
	cudaFree(dev->conc_Nb_lav);

	cudaFree(dev->conc_Cr_viz);
	cudaFree(dev->conc_Nb_viz);
	cudaFree(dev->conc_Ni);
	cudaFree(dev->phi_del_viz);
	cudaFree(dev->phi_lav_viz);

	cudaFree(dev->mob_gam_CrCr);
	cudaFree(dev->mob_gam_CrNb);
	cudaFree(dev->mob_gam_NbCr);
	cudaFree(dev->mob_gam_NbNb);

	cudaFree(dev->mob_del_CrCr);
	cudaFree(dev->mob_del_CrNb);
	cudaFree(dev->mob_del_NbCr);
	cudaFree(dev->mob_del_NbNb);

	cudaFree(dev->mob_lav_CrCr);
	cudaFree(dev->mob_lav_CrNb);
	cudaFree(dev->mob_lav_NbCr);
	cudaFree(dev->mob_lav_NbNb);
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
}
