// nucleation.c

#include <math.h>
#include <cassert>
#include "parameters.h"
#include "nucleation.cuh"
#include "parabola625.cuh"
#ifdef DEBUG
#include <stdio.h>
#endif

__device__ fp_t d_tanh_interp(const fp_t r, const fp_t w)
{
	// TKR5p243, Boettinger *et al.* 2002
	return 0.5 - 0.5 * tanh(r / w);
}

__device__ void d_nucleation_driving_force_delta(const fp_t& xCr, const fp_t& xNb, fp_t* dG)
{
	// TKR5p271
	*dG = d_g_gam(xCr, xNb);
}

__device__ void d_nucleation_driving_force_laves(const fp_t& xCr, const fp_t& xNb, fp_t* dG)
{
	// TKR5p271
	*dG = d_g_gam(xCr, xNb);
}

__device__ void d_nucleation_probability_sphere(const fp_t& xCr, const fp_t& xNb,
        const fp_t& dG_chem,
        const fp_t& D_CrCr, const fp_t& D_NbNb,
        const fp_t& sigma,
        const fp_t& Vatom,
        const fp_t& n_gam, const fp_t& dV, const fp_t& dt,
        fp_t* Rstar, fp_t* P_nuc)
{
	const fp_t sigma_cubed = sigma * sigma * sigma;
	const fp_t Zeldov = (Vatom * dG_chem * dG_chem)
	                    / (8. * M_PI * sqrt( d_kT() * sigma_cubed));
	const fp_t Gstar = (16. * M_PI * sigma_cubed) / (3. * dG_chem * dG_chem);
	const fp_t BstarCr = (3. * Gstar * D_CrCr * xCr) / (sigma * pow(lattice_const, 4));
	const fp_t BstarNb = (3. * Gstar * D_NbNb * xNb) / (sigma * pow(lattice_const, 4));

	const fp_t k1Cr = BstarCr * Zeldov * n_gam;
	const fp_t k1Nb = BstarNb * Zeldov * n_gam;

	const fp_t k2 = Gstar /  d_kT();

	const fp_t dc_Cr =-xCr +  d_xe_gam_Cr();
	const fp_t dc_Nb = xNb -  d_xe_gam_Nb();

	const fp_t JCr = k1Cr * exp(-k2 / dc_Cr);
	const fp_t JNb = k1Nb * exp(-k2 / dc_Nb);

	const fp_t P0 = exp(-dt * dV * (JCr + JNb)); // zero-event probability

	*P_nuc = 1. - P0;
	*Rstar = (2. * sigma) / dG_chem;

	#ifdef DEBUG
	printf("         Zeldov, G*:    %9.2e  %9.2e\n", Zeldov, Gstar);
	printf("         Bstar(Cr,Nb):  %9.2e  %9.2e\n", BstarCr, BstarNb);
	printf("         k1(Cr,Nb):     %9.2e  %9.2e\n", k1Cr, k1Nb);
	printf("         k2,  d_kT:        %9.2e  %9.2e\n", k2,  d_kT());
	printf("         dc_(Cr,Nb):    %9.2e  %9.2e\n", dc_Cr, dc_Nb);
	printf("         J(Cr,Nb):      %9.2e  %9.2e\n", JCr, JNb);
	printf("         Pnuc, P:       %9.2e  %9.2e\n", *P_nuc, P0);
	#endif
}
