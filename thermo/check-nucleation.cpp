// check-nucleation.cpp

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <math.h>

#include "nucleation.h"
#include "parabola625.h"
#include "parameters.h"

// Constants
const fp_t dV = meshres * meshres * meshres;
const fp_t vFccNi = lattice_const * lattice_const * lattice_const / 4.;
const fp_t n_gam = M_PI / (3. * sqrt(2.) * vFccNi);
const size_t iters = 50000;

fp_t timestep(const fp_t xCr, const fp_t xNb, const fp_t f_del, const fp_t f_gam, const fp_t f_lav)
{
	const fp_t D[12] = {
		// D_gam
		std::fabs(f_gam * ( M_CrCr(xCr, xNb) * d2g_gam_dxCrCr() + M_CrNb(xCr, xNb) * d2g_gam_dxCrNb())), // D11
		std::fabs(f_gam * ( M_CrCr(xCr, xNb) * d2g_gam_dxNbCr() + M_CrNb(xCr, xNb) * d2g_gam_dxNbNb())), // D12
		std::fabs(f_gam * ( M_NbCr(xCr, xNb) * d2g_gam_dxCrCr() + M_NbNb(xCr, xNb) * d2g_gam_dxCrNb())), // D21
		std::fabs(f_gam * ( M_NbCr(xCr, xNb) * d2g_gam_dxNbCr() + M_NbNb(xCr, xNb) * d2g_gam_dxNbNb())), // D22
		// D_del
		std::fabs(f_del * ( M_CrCr(xCr, xNb) * d2g_del_dxCrCr() + M_CrNb(xCr, xNb) * d2g_del_dxCrNb())), // D11
		std::fabs(f_del * ( M_CrCr(xCr, xNb) * d2g_del_dxNbCr() + M_CrNb(xCr, xNb) * d2g_del_dxNbNb())), // D12
		std::fabs(f_del * ( M_NbCr(xCr, xNb) * d2g_del_dxCrCr() + M_NbNb(xCr, xNb) * d2g_del_dxCrNb())), // D21
		std::fabs(f_del * ( M_NbCr(xCr, xNb) * d2g_del_dxNbCr() + M_NbNb(xCr, xNb) * d2g_del_dxNbNb())), // D22
		// D_lav
		std::fabs(f_lav * ( M_CrCr(xCr, xNb) * d2g_lav_dxCrCr() + M_CrNb(xCr, xNb) * d2g_lav_dxCrNb())), // D11
		std::fabs(f_lav * ( M_CrCr(xCr, xNb) * d2g_lav_dxNbCr() + M_CrNb(xCr, xNb) * d2g_lav_dxNbNb())), // D12
		std::fabs(f_lav * ( M_NbCr(xCr, xNb) * d2g_lav_dxCrCr() + M_NbNb(xCr, xNb) * d2g_lav_dxCrNb())), // D21
		std::fabs(f_lav * ( M_NbCr(xCr, xNb) * d2g_lav_dxNbCr() + M_NbNb(xCr, xNb) * d2g_lav_dxNbNb()))  // D22
	};

	const fp_t dtDiffusionLimited = (meshres * meshres) / (4.0 * *(std::max_element(D, D + 12)));

	return LinStab * dtDiffusionLimited;
}

struct Properties {
	fp_t xCr;
	fp_t xNb;
	fp_t dGlav;
	fp_t dGdel;
	fp_t Pdel;
	fp_t Plav;
	fp_t Rdel;
	fp_t Rlav;
	fp_t dt;
};

void zero_out(struct Properties* p)
{
	p->xCr = 0.;
	p->xNb = 0;
	p->dGdel = 0;
	p->dGlav = 0;
	p->Pdel = 0;
	p->Plav = 0;
	p->Rdel = 0;
	p->Rlav = 0;
}

Properties describe(const std::vector<fp_t>& xCr, const std::vector<fp_t>& xNb)
{
	const size_t N = xCr.size();
	std::vector<fp_t> dGdel(N, 0.);
	std::vector<fp_t> dGlav(N, 0.);
	std::vector<fp_t> Pdel(N, 0.);
	std::vector<fp_t> Plav(N, 0.);
	std::vector<fp_t> Rdel(N, 0.);
	std::vector<fp_t> Rlav(N, 0.);

	Properties Mean;
	zero_out(&Mean);
	for (size_t i=0; i<N; i++) {
		Mean.xCr   += xCr[i]   / N;
		Mean.xNb   += xNb[i]   / N;
	}

	const fp_t pDel = p(0.3);
	const fp_t pLav = p(0.3);
	const fp_t pGam = 1.0 - pDel - pLav;

	Mean.dt = timestep(Mean.xCr, Mean.xNb, pDel, pGam, pLav);

	for (size_t i=0; i<xCr.size(); i++) {
		nucleation_driving_force_delta(xCr[i], xNb[i], &(dGdel[i]));
		nucleation_probability_sphere(xCr[i], xNb[i],
		                              dGdel[i],
									  pGam * (M_CrCr(xCr[i], xNb[i]) * d2g_gam_dxCrCr() + M_CrNb(xCr[i], xNb[i]) * d2g_gam_dxCrNb()),
									  pGam * (M_NbCr(xCr[i], xNb[i]) * d2g_gam_dxNbCr() + M_NbNb(xCr[i], xNb[i]) * d2g_gam_dxNbNb()),
		                              s_delta(),
		                              vFccNi, n_gam, dV, Mean.dt,
		                              &(Rdel[i]), &(Pdel[i]));
		nucleation_driving_force_laves(xCr[i], xNb[i], &(dGlav[i]));
		nucleation_probability_sphere(xCr[i], xNb[i],
		                              dGlav[i],
									  pGam * (M_CrCr(xCr[i], xNb[i]) * d2g_gam_dxCrCr() + M_CrNb(xCr[i], xNb[i]) * d2g_gam_dxCrNb()),
									  pGam * (M_NbCr(xCr[i], xNb[i]) * d2g_gam_dxNbCr() + M_NbNb(xCr[i], xNb[i]) * d2g_gam_dxNbNb()),
		                              s_laves(),
		                              vFccNi, n_gam, dV, Mean.dt,
		                              &(Rlav[i]), &(Plav[i]));
	}

	for (size_t i=0; i<N; i++) {
		Mean.dGdel += dGdel[i] / N;
		Mean.dGlav += dGlav[i] / N;
		Mean.Pdel  += Pdel[i]  / N;
		Mean.Plav  += Plav[i]  / N;
		Mean.Rdel  += Rdel[i]  / N;
		Mean.Rlav  += Rlav[i]  / N;
	}

	Properties StDev;
	zero_out(&StDev);
	if (N > 1) {
		for (size_t i=0; i<N; i++) {
			StDev.xCr   += std::pow(Mean.xCr   - xCr[i], 2);
			StDev.xNb   += std::pow(Mean.xNb   - xNb[i], 2);
			StDev.dGdel += std::pow(Mean.dGdel - dGdel[i], 2);
			StDev.dGlav += std::pow(Mean.dGlav - dGlav[i], 2);
			StDev.Pdel  += std::pow(Mean.Pdel  - Pdel[i], 2);
			StDev.Plav  += std::pow(Mean.Plav  - Plav[i], 2);
			StDev.Rdel  += std::pow(Mean.Rdel  - Rdel[i], 2);
			StDev.Rlav  += std::pow(Mean.Rlav  - Rlav[i], 2);
		}

		StDev.xCr   = std::sqrt(StDev.xCr   / (N - 1));
		StDev.xNb   = std::sqrt(StDev.xNb   / (N - 1));
		StDev.dGdel = std::sqrt(StDev.dGdel / (N - 1));
		StDev.dGlav = std::sqrt(StDev.dGlav / (N - 1));
		StDev.Pdel  = std::sqrt(StDev.Pdel  / (N - 1));
		StDev.Plav  = std::sqrt(StDev.Plav  / (N - 1));
		StDev.Rdel  = std::sqrt(StDev.Rdel  / (N - 1));
		StDev.Rlav  = std::sqrt(StDev.Rlav  / (N - 1));
	}

	// Report
	printf("Composition: %9.6f ± %5.4f    %9.6f ± %5.4f\n", Mean.xCr,   StDev.xCr,   Mean.xNb,   StDev.xNb);
	printf("Driving frc: %9.2e ± %5.2e  %9.2e ± %5.2e\n",   Mean.dGdel, StDev.dGdel, Mean.dGlav, StDev.dGlav);
	printf("Crit. radius:%9.2e ± %5.2e  %9.2e ± %5.2e\n",   Mean.Rdel,  StDev.Rdel,  Mean.Rlav,  StDev.Rlav);
	printf("            (%9.4f Δx          %9.4f Δx)\n",    Mean.Rdel/meshres,       Mean.Rlav/meshres);
	printf("Probability: %9.2e ± %5.2e  %9.2e ± %5.2e\n",   Mean.Pdel,  StDev.Pdel,  Mean.Plav,  StDev.Plav);

	return Mean;
}

int main()
{
	// Randomization
	std::chrono::high_resolution_clock::time_point beginning = std::chrono::high_resolution_clock::now();
	const unsigned int seed = (std::chrono::high_resolution_clock::now() - beginning).count();
	std::default_random_engine generator(seed); // linear congruential engine

	std::uniform_real_distribution<fp_t> matrix_Nb_range(matrix_min_Nb(), matrix_max_Nb());
	std::uniform_real_distribution<fp_t> matrix_Cr_range(matrix_min_Cr(), matrix_max_Cr());

	std::uniform_real_distribution<fp_t> enrich_Nb_range(enrich_min_Nb(), enrich_max_Nb());
	std::uniform_real_distribution<fp_t> enrich_Cr_range(enrich_min_Cr(), enrich_max_Cr());

	Properties Mean;
	zero_out(&Mean);

	printf("=== Gamma-Rich ===\n");

	std::vector<fp_t> xCrR;
	std::vector<fp_t> xNbR;

	xCrR.push_back(0.5175);
	xNbR.push_back(0.0220);

	Mean = describe(xCrR, xNbR);

	printf("=== Matrix ===\n");

	std::vector<fp_t> xCrM;
	std::vector<fp_t> xNbM;

	for (size_t i=0; i<iters; i++) {
		xCrM.push_back(matrix_Cr_range(generator));
		xNbM.push_back(matrix_Nb_range(generator));
	}

	Mean = describe(xCrM, xNbM);

	printf("=== Enrichment ===\n");

	std::vector<fp_t> xCrE;
	std::vector<fp_t> xNbE;

	for (size_t i=0; i<iters; i++) {
		xCrE.push_back(enrich_Cr_range(generator));
		xNbE.push_back(enrich_Nb_range(generator));
	}

	Mean = describe(xCrE, xNbE);

	// === Sweep sigma ===

	FILE* csv = fopen("sigma.csv", "w");
	fprintf(csv, "sigma,Pdel,Plav\n");
	const double ds = 0.001;
	const fp_t pGam = 1.0 - Mean.Pdel - Mean.Plav;
	for (double s = ds; s < 0.5; s += ds) {
		fprintf(csv, "%.3f,", s);
		nucleation_driving_force_delta(Mean.xCr, Mean.xNb, &Mean.dGdel);
		nucleation_probability_sphere(Mean.xCr, Mean.xNb,
		                              Mean.dGdel,
									  pGam * (M_CrCr(Mean.xCr, Mean.xNb) * d2g_gam_dxCrCr() + M_CrNb(Mean.xCr, Mean.xNb) * d2g_gam_dxCrNb()),
									  pGam * (M_NbCr(Mean.xCr, Mean.xNb) * d2g_gam_dxNbCr() + M_NbNb(Mean.xCr, Mean.xNb) * d2g_gam_dxNbNb()),
		                              s,
		                              vFccNi, n_gam, dV, Mean.dt,
		                              &Mean.Rdel, &Mean.Pdel);
		fprintf(csv, "%12.4e,", Mean.Pdel);

		nucleation_driving_force_laves(Mean.xCr, Mean.xNb, &Mean.dGlav);
		nucleation_probability_sphere(Mean.xCr, Mean.xNb,
		                              Mean.dGlav,
									  pGam * (M_CrCr(Mean.xCr, Mean.xNb) * d2g_gam_dxCrCr() + M_CrNb(Mean.xCr, Mean.xNb) * d2g_gam_dxCrNb()),
									  pGam * (M_NbCr(Mean.xCr, Mean.xNb) * d2g_gam_dxNbCr() + M_NbNb(Mean.xCr, Mean.xNb) * d2g_gam_dxNbNb()),
		                              s,
		                              vFccNi, n_gam, dV, Mean.dt,
		                              &Mean.Rlav, &Mean.Plav);
		fprintf(csv, "%12.4e\n", Mean.Plav);
	}
	fclose(csv);

	/*
	// === Sweep xNb ===
	csv = fopen("composition.csv", "w");
	fprintf(csv, "xCr,xNb,dGdel,Pdel,dGlav,Plav\n");
	for (double xNb = ds; xNb < 1. - ds; xNb += ds) {
	    double xCr = 0.01;
	    fprintf(csv, "%.3f,%.3f,", xCr, xNb);
	    nucleation_driving_force_delta(xCr, xNb, &adGdelE);
	    fprintf(csv, "%12.4e,", adGdelE);
	    nucleation_probability_sphere(xCr, xNb,
	                                  xe_del_Cr(), xe_del_Nb(),
	                                  adGdelE,
									  M_CrCr(xCr[i], xNb[i]) * d2g_gam_dxCrCr() + M_CrNb(xCr[i], xNb[i]) * d2g_gam_dxCrNb(),
									  M_NbCr(xCr[i], xNb[i]) * d2g_gam_dxNbCr() + M_NbNb(xCr[i], xNb[i]) * d2g_gam_dxNbNb(),
	                                  s_delta(),
	                                  vFccNi, n_gam, dV, Mean.dt,
	                                  &aRdelE, &aPdelE);
	    fprintf(csv, "%12.4e,", aPdelE);

	    nucleation_driving_force_laves(xCr, xNb, &adGlavE);
	    fprintf(csv, "%12.4e,", adGlavE);
	    nucleation_probability_sphere(xCr, xNb,
	                                  xe_lav_Cr(), xe_lav_Nb(),
	                                  adGlavE,
									  M_CrCr(xCr[i], xNb[i]) * d2g_gam_dxCrCr() + M_CrNb(xCr[i], xNb[i]) * d2g_gam_dxCrNb(),
									  M_NbCr(xCr[i], xNb[i]) * d2g_gam_dxNbCr() + M_NbNb(xCr[i], xNb[i]) * d2g_gam_dxNbNb(),
	                                  s_laves(),
	                                  vFccNi, n_gam, dV, Mean.dt,
	                                  &aRlavE, &aPlavE);
	    fprintf(csv, "%12.4e\n", aPlavE);
	}
	fclose(csv);
	*/

	return 0;
}
