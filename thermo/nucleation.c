// nucleation.c

#include <math.h>
#include "globals.h"
#include "nucleation.h"
#include "parabola625.h"
#ifdef DEBUG
#include <stdio.h>
#endif

void nucleation_driving_force_delta(const fp_t& xCr, const fp_t& xNb, fp_t* dG)
{
    *dG = g_gam(xCr, xNb)
        - dg_gam_dxCr(xCr, xNb) * (xCr - xe_del_Cr())
        - dg_gam_dxNb(xCr, xNb) * (xNb - xe_del_Nb());
}

void nucleation_driving_force_laves(const fp_t& xCr, const fp_t& xNb, fp_t* dG)
{
    *dG = g_gam(xCr, xNb)
        - dg_gam_dxCr(xCr, xNb) * (xCr - xe_lav_Cr())
        - dg_gam_dxNb(xCr, xNb) * (xNb - xe_lav_Nb());
}

void nucleation_probability_sphere(const fp_t& xCr, const fp_t& xNb,
                                   const fp_t par_xCr, const fp_t par_xNb,
                                   const fp_t& dG_chem,
                                   const fp_t& D_CrCr, const fp_t& D_NbNb,
                                   const fp_t& sigma,
                                   const fp_t& Vatom,
                                   const fp_t& n_gam, const fp_t& dV, const fp_t& dt,
                                   fp_t* Rstar, fp_t* P_nuc)
{
    const fp_t Zeldov = (Vatom * dG_chem * dG_chem)
                      / (8 * M_PI * sqrt(kT() * sigma*sigma*sigma));
    const fp_t Gstar = (16 * M_PI * sigma * sigma * sigma) / (3 * dG_chem * dG_chem);
    const fp_t BstarCr = (3 * Gstar * D_CrCr * xCr) / (sigma * pow(lattice_const, 4));
    const fp_t BstarNb = (3 * Gstar * D_NbNb * xNb) / (sigma * pow(lattice_const, 4));

	const fp_t k1Cr = BstarCr * Zeldov * n_gam;
    const fp_t k1Nb = BstarNb * Zeldov * n_gam;

	const fp_t k2 = Gstar / kT();

	const fp_t dc_Cr = xCr - xe_gam_Cr();
	const fp_t dc_Nb = xNb - xe_gam_Nb();

	const fp_t JCr = k1Cr * exp(-k2 / fabs(dc_Cr));
	const fp_t JNb = k1Nb * exp(-k2 / fabs(dc_Nb));

    const fp_t P = exp(-dt * dV * (JCr + JNb));

	*P_nuc = 1. - P;
	*Rstar = (2 * sigma) / dG_chem;

    #ifdef DEBUG
    printf("         Zeldov, G*:    %9.2e  %9.2e\n", Zeldov, Gstar);
    printf("         Bstar(Cr,Nb):  %9.2e  %9.2e\n", BstarCr, BstarNb);
    printf("         k1(Cr,Nb):     %9.2e  %9.2e\n", k1Cr, k1Nb);
    printf("         k2, kT:        %9.2e  %9.2e\n", k2, kT());
    printf("         dc_(Cr,Nb):    %9.2e  %9.2e\n", dc_Cr, dc_Nb);
    printf("         J(Cr,Nb):      %9.2e  %9.2e\n", JCr, JNb);
    printf("         Pnuc, P:       %9.2e  %9.2e\n", *P_nuc, P);
    #endif
}
