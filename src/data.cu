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
	checkCuda(cudaMalloc((void**) &(dev->prng), nx * ny * sizeof(curandState)));

	cudaStreamCreate(&(dev->str_A));
	cudaStreamCreate(&(dev->str_B));
	cudaStreamCreate(&(dev->str_C));
	cudaStreamCreate(&(dev->str_D));

	cudaEventCreate(&(dev->ev_A));
	cudaEventCreate(&(dev->ev_B));
	cudaEventCreate(&(dev->ev_C));
	cudaEventCreate(&(dev->ev_D));

	checkCuda(cudaMalloc((void**) &(dev->conc_Cr_old), nx * ny * sizeof(fp_t)));
	checkCuda(cudaMalloc((void**) &(dev->conc_Cr_new), nx * ny * sizeof(fp_t)));
	checkCuda(cudaMalloc((void**) &(dev->conc_Nb_old), nx * ny * sizeof(fp_t)));
	checkCuda(cudaMalloc((void**) &(dev->conc_Nb_new), nx * ny * sizeof(fp_t)));

	checkCuda(cudaMalloc((void**) &(dev->phi_del_old), nx * ny * sizeof(fp_t)));
	checkCuda(cudaMalloc((void**) &(dev->phi_del_new), nx * ny * sizeof(fp_t)));
	checkCuda(cudaMalloc((void**) &(dev->phi_lav_old), nx * ny * sizeof(fp_t)));
	checkCuda(cudaMalloc((void**) &(dev->phi_lav_new), nx * ny * sizeof(fp_t)));
	checkCuda(cudaMalloc((void**) &(dev->phi),         nx * ny * sizeof(fp_t)));

	checkCuda(cudaMalloc((void**) &(dev->conc_Cr_gam), nx * ny * sizeof(fp_t)));
	checkCuda(cudaMalloc((void**) &(dev->conc_Cr_del), nx * ny * sizeof(fp_t)));
	checkCuda(cudaMalloc((void**) &(dev->conc_Cr_lav), nx * ny * sizeof(fp_t)));
	checkCuda(cudaMalloc((void**) &(dev->conc_Nb_gam), nx * ny * sizeof(fp_t)));
	checkCuda(cudaMalloc((void**) &(dev->conc_Nb_del), nx * ny * sizeof(fp_t)));
	checkCuda(cudaMalloc((void**) &(dev->conc_Nb_lav), nx * ny * sizeof(fp_t)));

	checkCuda(cudaMalloc((void**) &(dev->conc_Ni), nx * ny * sizeof(fp_t)));

	checkCuda(cudaMalloc((void**) &(dev->mob_gam_CrCr), nx * ny * sizeof(fp_t)));
	checkCuda(cudaMalloc((void**) &(dev->mob_gam_CrNb), nx * ny * sizeof(fp_t)));
	checkCuda(cudaMalloc((void**) &(dev->mob_gam_NbCr), nx * ny * sizeof(fp_t)));
	checkCuda(cudaMalloc((void**) &(dev->mob_gam_NbNb), nx * ny * sizeof(fp_t)));

	checkCuda(cudaMalloc((void**) &(dev->mob_del_CrCr), nx * ny * sizeof(fp_t)));
	checkCuda(cudaMalloc((void**) &(dev->mob_del_CrNb), nx * ny * sizeof(fp_t)));
	checkCuda(cudaMalloc((void**) &(dev->mob_del_NbCr), nx * ny * sizeof(fp_t)));
	checkCuda(cudaMalloc((void**) &(dev->mob_del_NbNb), nx * ny * sizeof(fp_t)));

	checkCuda(cudaMalloc((void**) &(dev->mob_lav_CrCr), nx * ny * sizeof(fp_t)));
	checkCuda(cudaMalloc((void**) &(dev->mob_lav_CrNb), nx * ny * sizeof(fp_t)));
	checkCuda(cudaMalloc((void**) &(dev->mob_lav_NbCr), nx * ny * sizeof(fp_t)));
	checkCuda(cudaMalloc((void**) &(dev->mob_lav_NbNb), nx * ny * sizeof(fp_t)));

	/* transfer mask to protected memory on GPU */
	checkCuda(cudaMemcpyToSymbol(d_mask, host->mask_lap[0], nm * nm * sizeof(fp_t)));

	/* transfer mobility data to protected memory on GPU */
	checkCuda(cudaMemcpyToSymbol(d_Kapp, kappa, NP * sizeof(fp_t)));
	checkCuda(cudaMemcpyToSymbol(d_Omeg, omega, NP * sizeof(fp_t)));
	checkCuda(cudaMemcpyToSymbol(d_Lmob, Lmob,  NP * sizeof(fp_t)));

	/* transfer data from host in to GPU */
	checkCuda(cudaMemcpy(dev->conc_Cr_old, host->conc_Cr_old[0], nx * ny * sizeof(fp_t),
						 cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(dev->conc_Nb_old, host->conc_Nb_old[0], nx * ny * sizeof(fp_t),
						 cudaMemcpyHostToDevice));

	checkCuda(cudaMemcpy(dev->phi_del_old, host->phi_del_old[0], nx * ny * sizeof(fp_t),
						 cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(dev->phi_lav_old, host->phi_lav_old[0], nx * ny * sizeof(fp_t),
						 cudaMemcpyHostToDevice));
}

void free_cuda(struct CudaData* dev)
{
	/* free memory on device */
	checkCuda(cudaFree(dev->prng));

	cudaStreamDestroy(dev->str_A);
	cudaStreamDestroy(dev->str_B);
	cudaStreamDestroy(dev->str_C);
	cudaStreamDestroy(dev->str_D);

	cudaEventDestroy(dev->ev_A);
	cudaEventDestroy(dev->ev_B);
	cudaEventDestroy(dev->ev_C);
	cudaEventDestroy(dev->ev_D);

	checkCuda(cudaFree(dev->conc_Cr_old));
	checkCuda(cudaFree(dev->conc_Cr_new));
	checkCuda(cudaFree(dev->conc_Nb_old));
	checkCuda(cudaFree(dev->conc_Nb_new));

	checkCuda(cudaFree(dev->phi_del_old));
	checkCuda(cudaFree(dev->phi_del_new));
	checkCuda(cudaFree(dev->phi_lav_old));
	checkCuda(cudaFree(dev->phi_lav_new));
	checkCuda(cudaFree(dev->phi));

	checkCuda(cudaFree(dev->conc_Cr_gam));
	checkCuda(cudaFree(dev->conc_Cr_del));
	checkCuda(cudaFree(dev->conc_Cr_lav));
	checkCuda(cudaFree(dev->conc_Nb_gam));
	checkCuda(cudaFree(dev->conc_Nb_del));
	checkCuda(cudaFree(dev->conc_Nb_lav));

	checkCuda(cudaFree(dev->conc_Ni));

	checkCuda(cudaFree(dev->mob_gam_CrCr));
	checkCuda(cudaFree(dev->mob_gam_CrNb));
	checkCuda(cudaFree(dev->mob_gam_NbCr));
	checkCuda(cudaFree(dev->mob_gam_NbNb));

	checkCuda(cudaFree(dev->mob_del_CrCr));
	checkCuda(cudaFree(dev->mob_del_CrNb));
	checkCuda(cudaFree(dev->mob_del_NbCr));
	checkCuda(cudaFree(dev->mob_del_NbNb));

	checkCuda(cudaFree(dev->mob_lav_CrCr));
	checkCuda(cudaFree(dev->mob_lav_CrNb));
	checkCuda(cudaFree(dev->mob_lav_NbCr));
	checkCuda(cudaFree(dev->mob_lav_NbNb));
}

void read_out_result(struct CudaData* dev, struct HostData* host,
                     const int nx, const int ny)
{
	checkCuda(cudaMemcpy(host->conc_Cr_new[0], dev->conc_Cr_old, nx * ny * sizeof(fp_t),
						 cudaMemcpyDeviceToHost));
	checkCuda(cudaMemcpy(host->conc_Nb_new[0], dev->conc_Nb_old, nx * ny * sizeof(fp_t),
						 cudaMemcpyDeviceToHost));
	checkCuda(cudaMemcpy(host->phi_del_new[0], dev->phi_del_old, nx * ny * sizeof(fp_t),
						 cudaMemcpyDeviceToHost));
	checkCuda(cudaMemcpy(host->phi_lav_new[0], dev->phi_lav_old, nx * ny * sizeof(fp_t),
						 cudaMemcpyDeviceToHost));
}
