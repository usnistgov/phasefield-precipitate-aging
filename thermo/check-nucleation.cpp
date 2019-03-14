// check-nucleation.cpp

#include <iostream>
#include <cstdlib>
#include <math.h>
#include "nucleation.h"
#include "parabola625.h"

// Number of precipitates and components (for array allocation)
#define NP 2
#define NC 2

// Kinetic and model parameters
const double meshres = 0.25e-9;        // grid spacing (m)
const double aCyl = 3 * meshres;       // height of the cylinder (m)
const double alpha = 1.07e11;          // three-phase coexistence coefficient (J/m^3)
const double LinStab = 1.0 / 19.42501; // threshold of linear (von Neumann) stability

// Diffusion constants in FCC Ni from Xu (m^2/s)
//                     Cr        Nb
const double D_Cr[NC] = {2.16e-15, 0.56e-15}; // first column of diffusivity matrix
const double D_Nb[NC] = {2.97e-15, 4.29e-15}; // second column of diffusivity matrix
const double aFccNi = 0.352e-9;               // lattice spacing of FCC nickel (m)

// Define st.dev. of bell curves for alloying element segregation
//                       Cr      Nb
const double bell[NC] = {150e-9, 50e-9}; // est. between 80-200 nm from SEM

// Precipitate radii: minimum for thermodynamic stability is 7.5 nm,
// minimum for numerical stability is 14*dx (due to interface width).
const double rPrecip[NP] = {7.5 * 5e-9 / meshres,  // delta
                            7.5 * 5e-9 / meshres}; // Laves

// Choose numerical diffusivity to lock chemical and transformational timescales
//                      delta      Laves
const double kappa[NP] = {1.24e-8, 1.24e-8};     // gradient energy coefficient (J/m)
const double Lmob[NP]  = {2.904e-11, 2.904e-11}; // numerical mobility (m^2/Ns)
const double sigma[NP] = {1.010/6, 1.010/4};     // interfacial energy (J/m^2)

// Compute interfacial width (nm) and well height (J/m^3)
const double ifce_width = 10. * meshres;
const double width_factor = 2.2; // 2.2 if interface is [0.1,0.9]; 2.94 if [0.05,0.95]
const double omega[NP] = {3.0 * width_factor* sigma[0] / ifce_width,  // delta
                          3.0 * width_factor* sigma[1] / ifce_width}; // Laves

void nucleation_driving_force_del(const double& xCr, const double& xNb,
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

	const double G_matrix = g_gam(xCr, xNb) + dg_gam_dxCr(xCr, xNb) * (*par_xCr - xCr)
                          + dg_gam_dxNb(xCr, xNb) * (*par_xNb - xNb);
	const double G_precip = g_del(*par_xCr, *par_xNb);

	*dG = (G_matrix - G_precip);
}

void nucleation_driving_force_lav(const double& xCr, const double& xNb,
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

	const double G_matrix = g_gam(xCr, xNb) + dg_gam_dxCr(xCr, xNb) * (*par_xCr - xCr)
	                      + dg_gam_dxNb(xCr, xNb) * (*par_xNb - xNb);
	const double G_precip = g_lav(*par_xCr, *par_xNb);

	*dG = (G_matrix - G_precip);
}

void nucleation_probability(const double& xCr, const double& xNb,
                            const double& par_xCr, const double& par_xNb,
                            const double& dG_chem, const double& D_CrCr, const double& D_NbNb,
                            const double& sigma, const double& aCyl, const double& Vatom,
                            const double& N_gam, const double& dV, const double& dt,
                            double* Rstar, double* P_nuc)
{
	const double denom = aCyl * dG_chem - 2. * sigma;
	const double Zeldov = Vatom * sqrt(6. * denom*denom*denom / kT())
	                    / (M_PI*M_PI * aCyl*aCyl * sigma);

	const double numer = 2. * M_PI * sigma * (aCyl * dG_chem - sigma);
    const double denom2 = aCyl*aCyl * denom*denom;
    const double BstarCr = (D_CrCr * xCr * numer) / denom2;
	const double BstarNb = (D_NbNb * xNb * numer) / denom2;

	const double k1Cr = BstarCr * Zeldov * N_gam / dV;
	const double k1Nb = BstarNb * Zeldov * N_gam / dV;

	const double k2 = fabs(dG_chem / kT());
	const double dc_Cr = fabs(par_xCr - xCr);
	const double dc_Nb = fabs(par_xNb - xNb);

	const double JCr = k1Cr * exp(-k2 / dc_Cr);
	const double JNb = k1Nb * exp(-k2 / dc_Nb);

	*Rstar = (sigma * aCyl) / denom;
    const double PCr = exp(-JCr * dt * dV);
    const double PNb = exp(-JNb * dt * dV);

    const double algebraic_mean = 0.5 * (PCr + PNb);
    const double geometric_mean = sqrt(PCr * PNb);
    const double harmonic_mean  = 2. / (1. / PCr + 1. / PNb);

	*P_nuc = 1. - harmonic_mean;

    printf("  denom, Zeldov: %9.2e  %9.2e\n", denom, Zeldov);
    printf("  Bstar(Cr,Nb):  %9.2e  %9.2e\n", BstarCr, BstarNb);
    printf("  k1(Cr,Nb):     %9.2e  %9.2e\n", k1Cr, k1Nb);
    printf("  k2, kT:        %9.2e  %9.2e\n", k2, kT());
    printf("  dc_(Cr,Nb):    %9.2e  %9.2e\n", dc_Cr, dc_Nb);
    printf("  J(Cr,Nb):      %9.2e  %9.2e\n", JCr, JNb);
    printf("  P(Cr,Nb):      %9.2e  %9.2e\n", PCr, PNb);
    printf("  Pal, Pge, Pha: %9.2e  %9.2e  %9.2e\n", algebraic_mean, geometric_mean, harmonic_mean);
}


int main(int argc, char* argv[])
{
  if (argc != 4) {
    std::cerr << "Error: Incorrect arguments.\n"
              << "Usage: " << argv[0] << " dt x_Cr x_Nb" << std::endl;
    std::exit(-1);
  }

  const double dt  = std::atof(argv[1]);
  const double xCr = std::atof(argv[2]);
  const double xNb = std::atof(argv[3]);

  const double dx = meshres;
  const double dy = meshres;
  const double dV = dx * dy * aCyl;

  const double nx = 4000;
  const double ny =  625;

  const double Vatom = aFccNi*aFccNi*aFccNi / 4.;
  const double N_gam = (nx*dx * ny*dy * aCyl * M_PI) / (3. * sqrt(2.) * Vatom);

  // Test a delta particle
  printf("Delta particle:\n");
  double dG_chem_del, P_nuc_del, Rstar_del;
  double del_xCr = 0., del_xNb = 0.;
  nucleation_driving_force_del(xCr, xNb, &del_xCr, &del_xNb, &dG_chem_del);
  nucleation_probability(xCr, xNb, del_xCr, del_xNb,
                         dG_chem_del, D_Cr[0], D_Nb[1],
                         sigma[0], aCyl,
                         Vatom, N_gam, dV, dt,
                         &Rstar_del, &P_nuc_del);

  // Test a Laves particle
  printf("Laves particle:\n");
  double dG_chem_lav, P_nuc_lav, Rstar_lav;
  double lav_xCr = 0., lav_xNb = 0.;
  nucleation_driving_force_lav(xCr, xNb,
                               &lav_xCr, &lav_xNb, &dG_chem_lav);
  nucleation_probability(xCr, xNb, lav_xCr, lav_xNb,
                         dG_chem_lav, D_Cr[0], D_Nb[1],
                         sigma[1], aCyl,
                         Vatom, N_gam, dV, dt,
                         &Rstar_lav, &P_nuc_lav);

  printf("Composition: %9.4f  %9.4f\n", xCr, xNb);
  printf("Delta offst: %9.4f  %9.4f\n", del_xCr - xe_del_Cr(), del_xNb - xe_del_Nb());
  printf("Laves offst: %9.4f  %9.4f\n", lav_xCr - xe_lav_Cr(), lav_xNb - xe_lav_Nb());
  printf("Driving frc: %9.2e  %9.2e\n", dG_chem_del, dG_chem_lav);
  printf("Crit. radii: %9.2e  %9.2e\n", Rstar_del, Rstar_lav);
  printf("Probability: %9.2e  %9.2e\n", P_nuc_del, P_nuc_lav);

  return 0;
}
