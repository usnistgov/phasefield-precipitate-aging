// check-nucleation.cpp

#include <math.h>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>

#include "parabola625.h"

// Number of precipitates and components (for array allocation)
#define NP 2
#define NC 2

typedef double fp_t;

// Kinetic and model parameters
const fp_t meshres = 0.25e-9;       // grid spacing (m)
const fp_t LinStab = 1. / 14.56876; // threshold of linear (von Neumann) stability
const fp_t lattice_const  = 0.352e-9;      // lattice spacing of FCC nickel (m)

// Diffusion constants in FCC Ni from Xu (m^2/s)
//                     Cr        Nb
const fp_t D_Cr[NC] = {2.16e-15, 0.56e-15}; // first column of diffusivity matrix
const fp_t D_Nb[NC] = {2.97e-15, 4.29e-15}; // second column of diffusivity matrix

// Choose numerical diffusivity to lock chemical and transformational timescales
//                        delta    Laves
const fp_t kappa[NP] = {1.24e-8,   1.24e-8};   // gradient energy coefficient (J/m)
const fp_t Lmob[NP]  = {2.904e-11, 2.904e-11}; // numerical mobility (m^2/Ns)
const fp_t sigma[NP] = {s_delta(), s_laves()}; // interfacial energy (J/m^2)

// Compute interfacial width (nm) and well height (J/m^3)
const fp_t ifce_width = 10. * meshres;
const fp_t width_factor = 2.2; // 2.2 if interface is [0.1,0.9]; 2.94 if [0.05,0.95]
const fp_t omega[NP] = {3.0 * width_factor* sigma[0] / ifce_width,  // delta
                        3.0 * width_factor* sigma[1] / ifce_width}; // Laves

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

	const fp_t G_matrix = g_gam(xCr, xNb)
                        - dg_gam_dxCr(xCr, xNb) * (xCr - *par_xCr)
                        - dg_gam_dxNb(xCr, xNb) * (xNb - *par_xNb);
	const fp_t G_precip = g_del(*par_xCr, *par_xNb);

	*dG = (G_matrix - G_precip);
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

	const fp_t G_matrix = g_gam(xCr, xNb)
                        - dg_gam_dxCr(xCr, xNb) * (xCr - *par_xCr)
                        - dg_gam_dxNb(xCr, xNb) * (xNb - *par_xNb);
	const fp_t G_precip = g_lav(*par_xCr, *par_xNb);

	*dG = (G_matrix - G_precip);
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

    printf("         Zeldov:        %9.2e\n", Zeldov);
    printf("         Bstar(Cr,Nb):  %9.2e  %9.2e\n", BstarCr, BstarNb);
    printf("         k1(Cr,Nb):     %9.2e  %9.2e\n", k1Cr, k1Nb);
    printf("         k2, kT:        %9.2e  %9.2e\n", k2, kT());
    printf("         dc_(Cr,Nb):    %9.2e  %9.2e\n", dc_Cr, dc_Nb);
    printf("         J(Cr,Nb):      %9.2e  %9.2e\n", JCr, JNb);
    printf("         P(Cr,Nb):      %9.2e  %9.2e\n", PCr, PNb);
    printf("         P:             %9.2e\n", *P_nuc);
}

int main()
{
    std::chrono::high_resolution_clock::time_point beginning = std::chrono::high_resolution_clock::now();
    std::default_random_engine generator;
    std::uniform_real_distribution<fp_t> enrich_Nb_range(enrich_min_Nb(), enrich_max_Nb());
    std::uniform_real_distribution<fp_t> enrich_Cr_range(enrich_min_Cr(), enrich_max_Cr());
    unsigned int seed = (std::chrono::high_resolution_clock::now() - beginning).count();
    generator.seed(seed);

    fp_t xCr = enrich_Cr_range(generator);
    fp_t xNb = enrich_Nb_range(generator);

    const fp_t dtDiffusionLimited = (meshres*meshres) / (4. * std::max(D_Cr[0], D_Nb[1]));
    const fp_t dt = 5 * LinStab * dtDiffusionLimited;

    const fp_t dV = meshres * meshres * meshres;

    const fp_t vFccNi = lattice_const * lattice_const * lattice_const / 4.;
    const fp_t n_gam = M_PI / (3. * sqrt(2.) * vFccNi);

    // Test a delta particle
    printf("Delta particle:\n");
    fp_t dG_chem_del, P_nuc_del, Rstar_del;
    fp_t del_xCr = 0., del_xNb = 0.;
    nucleation_driving_force_delta(xCr, xNb, &del_xCr, &del_xNb, &dG_chem_del);
    nucleation_probability_sphere(xCr, xNb, del_xCr, del_xNb,
                                  dG_chem_del, D_Cr[0], D_Nb[1],
                                  sigma[0],
                                  vFccNi, n_gam, dV, dt,
                                  &Rstar_del, &P_nuc_del);

    // Test a Laves particle
    printf("Laves particle:\n");
    fp_t dG_chem_lav, P_nuc_lav, Rstar_lav;
    fp_t lav_xCr = 0., lav_xNb = 0.;
    nucleation_driving_force_laves(xCr, xNb, &lav_xCr, &lav_xNb, &dG_chem_lav);
    nucleation_probability_sphere(xCr, xNb, lav_xCr, lav_xNb,
                                  dG_chem_lav, D_Cr[0], D_Nb[1],
                                  sigma[1],
                                  vFccNi, n_gam, dV, dt,
                                  &Rstar_lav, &P_nuc_lav);

    printf("Composition: %9.4f  %9.4f\n", xCr, xNb);
    printf("Interval:    %9.3e\n", dt);
    printf("Delta offst: %9.4f  %9.4f\n", del_xCr - xe_del_Cr(), del_xNb - xe_del_Nb());
    printf("Laves offst: %9.4f  %9.4f\n", lav_xCr - xe_lav_Cr(), lav_xNb - xe_lav_Nb());
    printf("Driving frc: %9.2e  %9.2e\n", dG_chem_del, dG_chem_lav);
    printf("Crit. radii: %9.2e  %9.2e\n", Rstar_del, Rstar_lav);
    printf("Probability: %9.2e  %9.2e\n", P_nuc_del, P_nuc_lav);

    return 0;
}
