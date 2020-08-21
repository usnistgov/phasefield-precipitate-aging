/**
 \file  type.h
 \brief Definition of scalar data type and container structures
*/

/** \cond SuppressGuard */
#ifndef _TYPE_H_
#define _TYPE_H_
/** \endcond */

#include <curand_kernel.h>

#define NP 2 // number of phases
#define NC 2 // number of components

/**
 \brief Precision of floating-point values
 Specify the basic data type to achieve the desired accuracy in floating-point
 arithmetic: float for single-precision, double for double-precision. This
 choice propagates throughout the code, and may significantly affect runtime
 on GPU hardware.
*/
typedef double fp_t;

/**
 \brief Container for pointers to arrays on the CPU
*/
struct HostData {
	fp_t** mask_lap;
	fp_t** conc_Ni;
	fp_t** phi;

	fp_t** chem_nrg;
	fp_t** grad_nrg;

	fp_t** conc_Cr_old;
	fp_t** conc_Cr_new;
	fp_t** conc_Cr_lap;
	fp_t** conc_Cr_gam;

	fp_t** conc_Nb_old;
	fp_t** conc_Nb_new;
	fp_t** conc_Nb_lap;
	fp_t** conc_Nb_gam;

	fp_t** phi_del_old;
	fp_t** phi_del_new;
	fp_t** phi_del_lap;

	fp_t** phi_lav_old;
	fp_t** phi_lav_new;
	fp_t** phi_lav_lap;
};

/**
 \brief Container for pointers to arrays on the GPU
*/
struct CudaData {
	curandState* prng;
	cudaStream_t str_A, str_B, str_C, str_D;
	cudaEvent_t ev_A, ev_B, ev_C, ev_D;

	fp_t* conc_Ni;
	fp_t* phi;

	fp_t* conc_Cr_old;
	fp_t* conc_Cr_new;

	fp_t* conc_Nb_old;
	fp_t* conc_Nb_new;

	fp_t* phi_del_old;
	fp_t* phi_del_new;

	fp_t* phi_lav_old;
	fp_t* phi_lav_new;

	fp_t* conc_Cr_gam;
	fp_t* conc_Cr_del;
	fp_t* conc_Cr_lav;

	fp_t* conc_Nb_gam;
	fp_t* conc_Nb_del;
	fp_t* conc_Nb_lav;

	/*
	fp_t* mob_gam_CrCr;
	fp_t* mob_gam_CrNb;
	fp_t* mob_gam_NbCr;
	fp_t* mob_gam_NbNb;

	fp_t* mob_del_CrCr;
	fp_t* mob_del_CrNb;
	fp_t* mob_del_NbCr;
	fp_t* mob_del_NbNb;

	fp_t* mob_lav_CrCr;
	fp_t* mob_lav_CrNb;
	fp_t* mob_lav_NbCr;
	fp_t* mob_lav_NbNb;

	fp_t* mob_phi_del_Cr;
	fp_t* mob_phi_del_Nb;
	fp_t* mob_phi_lav_Cr;
	fp_t* mob_phi_lav_Nb;
	*/
};

/** \cond SuppressGuard */
#endif /* _TYPE_H_ */
/** \endcond */
