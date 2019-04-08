// nucleation.c

#include <math.h>
#include "globals.h"
#include "nucleation.h"
#include "parabola625.h"
#ifdef DEBUG
#include <stdio.h>
#endif

void nucleation_driving_force_delta(const fp_t& xCr, const fp_t& xNb,
                                    fp_t* par_xCr, fp_t* par_xNb, fp_t* dG)
{
	/* compute thermodynamic driving force for nucleation */
	const fp_t a11 = d2g_del_dxCrCr();
	const fp_t a12 = d2g_del_dxCrNb();
	const fp_t a21 = d2g_del_dxNbCr();
	const fp_t a22 = d2g_del_dxNbNb();

	const fp_t b1 = dg_gam_dxCr(xCr, xNb) + a11 * xe_del_Cr() + a12 * xe_del_Nb();
	const fp_t b2 = dg_gam_dxNb(xCr, xNb) + a21 * xe_del_Cr() + a22 * xe_del_Nb();

	const fp_t detA = a11 * a22 - a12 * a21;
	const fp_t detB = b1  * a22 - a12 * b2;
	const fp_t detC = a11 * b2  - b1  * a21;

	*par_xCr = detB / detA;
	*par_xNb = detC / detA;

	const fp_t G_gamma = g_gam(xCr, xNb)
                        - dg_gam_dxCr(xCr, xNb) * (xCr - *par_xCr)
                        - dg_gam_dxNb(xCr, xNb) * (xNb - *par_xNb);
	const fp_t G_delta = g_del(*par_xCr, *par_xNb);

	*dG = (G_gamma - G_delta);
}

void nucleation_driving_force_laves(const fp_t& xCr, const fp_t& xNb,
                                    fp_t* par_xCr, fp_t* par_xNb, fp_t* dG)
{
	/* compute thermodynamic driving force for nucleation */
	const fp_t a11 = d2g_lav_dxCrCr();
	const fp_t a12 = d2g_lav_dxCrNb();
	const fp_t a21 = d2g_lav_dxNbCr();
	const fp_t a22 = d2g_lav_dxNbNb();

	const fp_t b1 = dg_gam_dxCr(xCr, xNb) + a11 * xe_lav_Cr() + a12 * xe_lav_Nb();
	const fp_t b2 = dg_gam_dxNb(xCr, xNb) + a21 * xe_lav_Cr() + a22 * xe_lav_Nb();

	const fp_t detA = a11 * a22 - a12 * a21;
	const fp_t detB = b1  * a22 - a12 * b2;
	const fp_t detC = a11 * b2  - b1  * a21;

	*par_xCr = detB / detA;
	*par_xNb = detC / detA;

	const fp_t G_gamma = g_gam(xCr, xNb)
                        - dg_gam_dxCr(xCr, xNb) * (xCr - *par_xCr)
                        - dg_gam_dxNb(xCr, xNb) * (xNb - *par_xNb);
	const fp_t G_laves = g_lav(*par_xCr, *par_xNb);

	*dG = (G_gamma - G_laves);
}

void nucleation_probability_sphere(const fp_t& xCr, const fp_t& xNb,
                                   const fp_t& par_xCr, const fp_t& par_xNb,
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
	const fp_t dc_Cr = fabs(par_xCr - xCr);
	const fp_t dc_Nb = fabs(par_xNb - xNb);

	const fp_t JCr = k1Cr * exp(-k2 / dc_Cr);
	const fp_t JNb = k1Nb * exp(-k2 / dc_Nb);

    const fp_t PCr = exp(-JCr * dt * dV);
    const fp_t PNb = exp(-JNb * dt * dV);

	*P_nuc = 1. - PCr * PNb;
	*Rstar = (2 * sigma) / dG_chem;

    #ifdef DEBUG
    printf("         Zeldov:        %9.2e\n", Zeldov);
    printf("         Bstar(Cr,Nb):  %9.2e  %9.2e\n", BstarCr, BstarNb);
    printf("         k1(Cr,Nb):     %9.2e  %9.2e\n", k1Cr, k1Nb);
    printf("         k2, kT:        %9.2e  %9.2e\n", k2, kT());
    printf("         dc_(Cr,Nb):    %9.2e  %9.2e\n", dc_Cr, dc_Nb);
    printf("         J(Cr,Nb):      %9.2e  %9.2e\n", JCr, JNb);
    printf("         P(Cr,Nb):      %9.2e  %9.2e\n", PCr, PNb);
    printf("         P:             %9.2e\n", *P_nuc);
    #endif
}
