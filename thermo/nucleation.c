// nucleation.c

#include <math.h>
#include <cassert>
#include "globals.h"
#include "nucleation.h"
#include "parabola625.h"
#ifdef DEBUG
#include <stdio.h>
#endif

void nucleation_driving_force_delta(const fp_t& xCr, const fp_t& xNb, fp_t* dG)
{
    // TKR5p269
    const fp_t & yNb = xe_del_Nb();
    const fp_t & yCr = xe_del_Cr();

    *dG = - g_gam(xCr, xNb)
          - dg_gam_dxNb(xCr, xNb) * (yNb - xNb)
          - dg_gam_dxCr(xCr, xNb) * (yCr - xCr)
          - dg_del_dxNb(yCr, yNb) * (yNb - yCr)
          - dg_del_dxCr(yCr, yNb) * (yCr - yNb);
}

void nucleation_driving_force_laves(const fp_t& xCr, const fp_t& xNb, fp_t* dG)
{
    // TKR5p269
    const fp_t & yNb = xe_lav_Nb();
    const fp_t & yCr = xe_lav_Cr();

    *dG = - g_gam(xCr, xNb)
          - dg_gam_dxNb(xCr, xNb) * (yNb - xNb)
          - dg_gam_dxCr(xCr, xNb) * (yCr - xCr)
          - dg_lav_dxNb(yCr, yNb) * (yNb - yCr)
          - dg_lav_dxCr(yCr, yNb) * (yCr - yNb);
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
    const fp_t sigma_cubed = sigma * sigma * sigma;
    const fp_t Zeldov = (Vatom * dG_chem * dG_chem)
                      / (8. * M_PI * sqrt(kT() * sigma_cubed));
    const fp_t Gstar = (16. * M_PI * sigma_cubed) / (3. * dG_chem * dG_chem);
    const fp_t BstarCr = (3. * Gstar * D_CrCr * xCr) / (sigma * pow(lattice_const, 4));
    const fp_t BstarNb = (3. * Gstar * D_NbNb * xNb) / (sigma * pow(lattice_const, 4));

    const fp_t k1Cr = BstarCr * Zeldov * n_gam;
    const fp_t k1Nb = BstarNb * Zeldov * n_gam;

    const fp_t k2 = Gstar / kT();

    const fp_t dc_Cr =-xCr + xe_gam_Cr();
    const fp_t dc_Nb = xNb - xe_gam_Nb();

    const fp_t JCr = k1Cr * exp(-k2 / dc_Cr);
    const fp_t JNb = k1Nb * exp(-k2 / dc_Nb);

    const fp_t P0 = exp(-dt * dV * (JCr + JNb)); // zero-event probability

    *P_nuc = 1. - P0;
    *Rstar = -(2. * sigma) / dG_chem;

    #ifdef DEBUG
    printf("         Zeldov, G*:    %9.2e  %9.2e\n", Zeldov, Gstar);
    printf("         Bstar(Cr,Nb):  %9.2e  %9.2e\n", BstarCr, BstarNb);
    printf("         k1(Cr,Nb):     %9.2e  %9.2e\n", k1Cr, k1Nb);
    printf("         k2, kT:        %9.2e  %9.2e\n", k2, kT());
    printf("         dc_(Cr,Nb):    %9.2e  %9.2e\n", dc_Cr, dc_Nb);
    printf("         J(Cr,Nb):      %9.2e  %9.2e\n", JCr, JNb);
    printf("         Pnuc, P:       %9.2e  %9.2e\n", *P_nuc, P0);
    #endif
}
