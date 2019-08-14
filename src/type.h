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

	fp_t** conc_Cr_old;
	fp_t** conc_Cr_new;

	fp_t** conc_Nb_old;
	fp_t** conc_Nb_new;

	fp_t** phi_del_old;
	fp_t** phi_del_new;

	fp_t** phi_lav_old;
	fp_t** phi_lav_new;
};

/**
 \brief Container for pointers to arrays on the GPU
*/
struct CudaData {
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

	curandState* prng;
};

/** \cond SuppressGuard */
#endif /* _TYPE_H_ */
/** \endcond */
