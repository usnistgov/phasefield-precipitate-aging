// check-nucleation.cpp

#include <cmath>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>

#include "nucleation.h"
#include "parabola625.h"

// Number of precipitates and components (for array allocation)
#define NP 2
#define NC 2

// Kinetic and model parameters
const double meshres = 0.25e-9;         // grid spacing (m)
const double LinStab = 1. / 7.28437875; // threshold of linear (von Neumann) stability
const double aFccNi  = 0.352e-9;        // lattice spacing of FCC nickel (m)
const double hCyl    = aFccNi;          // height of the cylinder (m)

// Diffusion constants in FCC Ni from Xu (m^2/s)
//                     Cr        Nb
const double D_Cr[NC] = {2.16e-15, 0.56e-15}; // first column of diffusivity matrix
const double D_Nb[NC] = {2.97e-15, 4.29e-15}; // second column of diffusivity matrix

// Choose numerical diffusivity to lock chemical and transformational timescales
//                          delta      Laves
const double kappa[NP]   = {1.24e-8, 1.24e-8};     // gradient energy coefficient (J/m)
const double Lmob[NP]    = {2.904e-11, 2.904e-11}; // numerical mobility (m^2/Ns)
const double sigma[NP]   = {1.010, 1.010};     // interfacial energy (J/m^2)
const double rPrecip[NP] = {1e-9, 1e-9};           // precipitate radii (m)

// Compute interfacial width (nm) and well height (J/m^3)
const double ifce_width = 10. * meshres;
const double width_factor = 2.2; // 2.2 if interface is [0.1,0.9]; 2.94 if [0.05,0.95]
const double omega[NP] = {3.0 * width_factor* sigma[0] / ifce_width,  // delta
                          3.0 * width_factor* sigma[1] / ifce_width}; // Laves

void nucleation_driving_force_delta(const double& xCr, const double& xNb,
                                    double* par_xCr, double* par_xNb, double* dG)
{
	/* compute thermodynamic driving force for nucleation */
	const double a11 = d2g_del_dxCrCr();
	const double a12 = d2g_del_dxCrNb();
	const double a21 = d2g_del_dxNbCr();
	const double a22 = d2g_del_dxNbNb();

	const double b1 = dg_gam_dxCr(xCr, xNb) + a11 * xe_del_Cr() + a12 * xe_del_Nb();
	const double b2 = dg_gam_dxNb(xCr, xNb) + a21 * xe_del_Cr() + a22 * xe_del_Nb();

	const double detA = a11 * a22 - a12 * a21;
	const double detB = b1  * a22 - a12 * b2;
	const double detC = a11 * b2  - b1  * a21;

	*par_xCr = detB / detA;
	*par_xNb = detC / detA;

	const double G_matrix = Vm() * g_gam(xCr, xNb)
                          - Vm() * dg_gam_dxCr(xCr, xNb) * (xCr - *par_xCr)
                          - Vm() * dg_gam_dxNb(xCr, xNb) * (xNb - *par_xNb);
	const double G_precip = Vm() * g_del(*par_xCr, *par_xNb);

	*dG = (G_matrix - G_precip);
}

void nucleation_driving_force_laves(const double& xCr, const double& xNb,
                                    double* par_xCr, double* par_xNb, double* dG)
{
	/* compute thermodynamic driving force for nucleation */
	const double a11 = d2g_lav_dxCrCr();
	const double a12 = d2g_lav_dxCrNb();
	const double a21 = d2g_lav_dxNbCr();
	const double a22 = d2g_lav_dxNbNb();

	const double b1 = dg_gam_dxCr(xCr, xNb) + a11 * xe_lav_Cr() + a12 * xe_lav_Nb();
	const double b2 = dg_gam_dxNb(xCr, xNb) + a21 * xe_lav_Cr() + a22 * xe_lav_Nb();

	const double detA = a11 * a22 - a12 * a21;
	const double detB = b1  * a22 - a12 * b2;
	const double detC = a11 * b2  - b1  * a21;

	*par_xCr = detB / detA;
	*par_xNb = detC / detA;

	const double G_matrix = Vm() * g_gam(xCr, xNb)
                          - Vm() * dg_gam_dxCr(xCr, xNb) * (xCr - *par_xCr)
                          - Vm() * dg_gam_dxNb(xCr, xNb) * (xNb - *par_xNb);
	const double G_precip = Vm() * g_lav(*par_xCr, *par_xNb);

	*dG = (G_matrix - G_precip);
}

void nucleation_probability_cylinder(const double& xCr, const double& xNb,
                                     const double& par_xCr, const double& par_xNb,
                                     const double& dG_chem, const double& D_CrCr, const double& D_NbNb,
                                     const double& sigma, const double& hCyl, const double& Vatom,
                                     const double& N_gam, const double& dV, const double& dt,
                                     double* Rstar, double* P_nuc)
{
	const double denom = hCyl * dG_chem - 2. * sigma;
	const double Zeldov = Vatom * sqrt(6. * denom*denom*denom / RT())
	                    / (M_PI*M_PI * hCyl*hCyl * sigma);

	const double numer = 2. * M_PI * sigma * (hCyl * dG_chem - sigma);
    const double denomSq = hCyl*hCyl * denom*denom;
    const double BstarCr = (D_CrCr * xCr * numer) / denomSq;
	const double BstarNb = (D_NbNb * xNb * numer) / denomSq;

	const double k1Cr = BstarCr * Zeldov * N_gam / dV;
	const double k1Nb = BstarNb * Zeldov * N_gam / dV;

	const double k2 = dG_chem / RT();
	const double dc_Cr = fabs(par_xCr - xCr);
	const double dc_Nb = fabs(par_xNb - xNb);

	const double JCr = k1Cr * exp(-k2 / dc_Cr);
	const double JNb = k1Nb * exp(-k2 / dc_Nb);

	*Rstar = (sigma * hCyl) / denom;
    const double PCr = exp(-JCr * dt * dV);
    const double PNb = exp(-JNb * dt * dV);

    const double algebraic_mean = 0.5 * (PCr + PNb);
    const double geometric_mean = sqrt(PCr * PNb);
    const double harmonic_mean  = 2. / (1. / PCr + 1. / PNb);

	*P_nuc = 1. - harmonic_mean;

    printf("         denom, Zeldov: %9.2e  %9.2e\n", denom, Zeldov);
    printf("         Bstar(Cr,Nb):  %9.2e  %9.2e\n", BstarCr, BstarNb);
    printf("         k1(Cr,Nb):     %9.2e  %9.2e\n", k1Cr, k1Nb);
    printf("         k2, kT:        %9.2e  %9.2e\n", k2, RT());
    printf("         dc_(Cr,Nb):    %9.2e  %9.2e\n", dc_Cr, dc_Nb);
    printf("         J(Cr,Nb):      %9.2e  %9.2e\n", JCr, JNb);
    printf("         P(Cr,Nb):      %9.2e  %9.2e\n", PCr, PNb);
    printf("         Pal, Pge, Pha: %9.2e  %9.2e  %9.2e\n", algebraic_mean, geometric_mean, harmonic_mean);
}

void nucleation_probability_sphere(const double& xCr, const double& xNb,
                                   const double& par_xCr, const double& par_xNb,
                                   const double& dG_chem, const double& D_CrCr, const double& D_NbNb,
                                   const double& sigma, const double& hCyl, const double& Vatom,
                                   const double& N_gam, const double& dV, const double& dt,
                                   double* Rstar, double* P_nuc)
{
    const double Zeldov = (Vatom * dG_chem * dG_chem)
                        / (8 * M_PI * sqrt(RT() * sigma*sigma*sigma));

    const double BstarCr = (16 * M_PI * sigma * sigma * D_CrCr * xCr)
                         / (dG_chem * dG_chem * pow(aFccNi, 4));
    const double BstarNb = (16 * M_PI * sigma * sigma * D_NbNb * xNb)
                         / (dG_chem * dG_chem * pow(aFccNi, 4));

	const double k1Cr = BstarCr * Zeldov * N_gam / dV;
	const double k1Nb = BstarNb * Zeldov * N_gam / dV;

	const double k2 = dG_chem / RT();
	const double dc_Cr = fabs(par_xCr - xCr);
	const double dc_Nb = fabs(par_xNb - xNb);

	const double JCr = k1Cr * exp(-k2 / dc_Cr);
	const double JNb = k1Nb * exp(-k2 / dc_Nb);

	*Rstar = (2 * sigma) / dG_chem;
    const double PCr = exp(-JCr * dt * dV);
    const double PNb = exp(-JNb * dt * dV);

    const double harmonic_mean  = 2. / (1. / PCr + 1. / PNb);

	*P_nuc = 1. - harmonic_mean;

    printf("         Zeldov:        %9.2e\n", Zeldov);
    printf("         Bstar(Cr,Nb):  %9.2e  %9.2e\n", BstarCr, BstarNb);
    printf("         k1(Cr,Nb):     %9.2e  %9.2e\n", k1Cr, k1Nb);
    printf("         k2, kT:        %9.2e  %9.2e\n", k2, RT());
    printf("         dc_(Cr,Nb):    %9.2e  %9.2e\n", dc_Cr, dc_Nb);
    printf("         J(Cr,Nb):      %9.2e  %9.2e\n", JCr, JNb);
    printf("         P(Cr,Nb):      %9.2e  %9.2e\n", 1. - PCr, 1. - PNb);
}

int main()
{
  std::chrono::high_resolution_clock::time_point beginning = std::chrono::high_resolution_clock::now();
  std::default_random_engine generator;
  std::uniform_real_distribution<double> enrich_Nb_range(0.1693, 0.1760);
  std::uniform_real_distribution<double> enrich_Cr_range(0.2720, 0.3214);
  unsigned int seed = (std::chrono::high_resolution_clock::now() - beginning).count();
  generator.seed(seed);

  double xCr = enrich_Cr_range(generator);
  double xNb = enrich_Nb_range(generator);

  const double dtDiffusionLimited = (meshres*meshres) / (4. * std::max(D_Cr[0], D_Nb[1]));
  const double dt = 0.02e-3 / (LinStab * dtDiffusionLimited);

  const double dx = meshres;
  const double dy = meshres;
  const double dV = dx * dy * hCyl;

  const double nx = 4000;
  const double ny =  625;

  const double vFccNi = aFccNi*aFccNi*aFccNi / 4.;
  const double N_gam = (nx*dx * ny*dy * hCyl * M_PI) / (3. * sqrt(2.) * vFccNi);

  // Test a delta particle
  printf("Delta particle:\n");
  double dG_chem_del, P_nuc_del, Rstar_del;
  double del_xCr = 0., del_xNb = 0.;
  nucleation_driving_force_delta(xCr, xNb, &del_xCr, &del_xNb, &dG_chem_del);
  nucleation_probability_sphere(xCr, xNb, del_xCr, del_xNb,
                                dG_chem_del, D_Cr[0], D_Nb[1],
                                sigma[0], hCyl,
                                vFccNi, N_gam, dV, dt,
                                &Rstar_del, &P_nuc_del);

  // Test a Laves particle
  printf("Laves particle:\n");
  double dG_chem_lav, P_nuc_lav, Rstar_lav;
  double lav_xCr = 0., lav_xNb = 0.;
  nucleation_driving_force_laves(xCr, xNb,
                               &lav_xCr, &lav_xNb, &dG_chem_lav);
  nucleation_probability_sphere(xCr, xNb, lav_xCr, lav_xNb,
                                dG_chem_lav, D_Cr[0], D_Nb[1],
                                sigma[1], hCyl,
                                vFccNi, N_gam, dV, dt,
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
