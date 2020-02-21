/**
 \file cuda625.cpp
 \brief Algorithms for 2D and 3D isotropic Cr-Nb-Ni alloy phase transformations
 This implementation depends on the GNU Scientific Library for multivariate root
 finding algorithms, and Mesoscale Microstructure Simulation Project for high-
 performance grid operations in parallel.
*/

#ifndef __CUDA625_CPP__
#define __CUDA625_CPP__

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <random>
#include <set>
#include <sstream>
#include <vector>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_multiroots.h>
#ifdef _OPENMP
#include "omp.h"
#endif

#include "MMSP.hpp"
#include "alloy625.hpp"
#include "data.cuh"
#include "discretization.cuh"
#include "enrichment.h"
#include "mesh.h"
#include "nucleation.h"
#include "output.h"
#include "parabola625.h"
#include "parameters.h"
#include "phasefrac.h"

// Host diffusivity arrays
fp_t D_Cr[NC] = {0}; // populated in main.cpp
fp_t D_Nb[NC] = {0};

// Host mobility array
fp_t Lmob[NP] = {0}; // populated in main.cpp

namespace MMSP
{

template<int dim, class T>
void system_composition(MMSP::grid<dim, MMSP::vector<T> > const& grid, fp_t& xCr0, fp_t& xNb0)
{
	double xCr = 0.0;
	double xNb = 0.0;
	unsigned N = MMSP::nodes(grid);

	for (unsigned n = 0; n < N; n++) {
		MMSP::vector<T>& GridN = grid(n);
		xCr += GridN[0];
		xNb += GridN[1];
	}

	#ifdef MPI_VERSION
	MPI_Request reqs[3];
	MPI_Status stat[3];

	double myCr(xCr);
	double myNb(xNb);
	unsigned myN(N);

	MPI_Allreduce(&myCr, &xCr, 1, MPI_DOUBLE,   MPI_SUM, MPI_COMM_WORLD, reqs[0]);
	MPI_Allreduce(&myNb, &xNb, 1, MPI_DOUBLE,   MPI_SUM, MPI_COMM_WORLD, reqs[1]);
	MPI_Allreduce(&myN,  &N,   1, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD, reqs[2]);
	MPI_Wait(reqs, 3, stat);
	#endif

	xCr0 = xCr / N;
	xNb0 = xNb / N;
}

void set_diffusion_matrix(const fp_t xCr0, const fp_t xNb0, fp_t* DCr, fp_t* DNb, fp_t* L_mob, int verbose)
{
	assert(DCr != NULL);
	assert(DNb != NULL);
	assert(L_mob != NULL);

	int rank = 0;
	#ifdef MPI_VERSION
	rank = MPI::COMM_WORLD.Get_rank();
	#endif

	DCr[0] = M_CrCr(xCr0, xNb0) * d2g_gam_dxCrCr() + M_CrNb(xCr0, xNb0) * d2g_gam_dxCrNb();
	DCr[1] = M_CrCr(xCr0, xNb0) * d2g_gam_dxNbCr() + M_CrNb(xCr0, xNb0) * d2g_gam_dxNbNb();

	DNb[0] = M_NbCr(xCr0, xNb0) * d2g_gam_dxCrCr() + M_NbNb(xCr0, xNb0) * d2g_gam_dxCrNb();
	DNb[1] = M_NbCr(xCr0, xNb0) * d2g_gam_dxNbCr() + M_NbNb(xCr0, xNb0) * d2g_gam_dxNbNb();

	if (rank == 0 && verbose) {
		printf("Diffusion matrix is [%10.4g %10.4g]\n", DCr[0], DCr[1]);
		printf("                    [%10.4g %10.4g]\n", DNb[0], DNb[1]);
	}

	const fp_t Dmin = std::min(std::fabs(DCr[0]), std::fabs(DNb[1]));
	L_mob[0] = MobStab * Dmin / (ifce_width * ifce_width * RT() / Vm()); // numerical mobility (m^2/(Ns))
	L_mob[1] = Lmob[0];                                                  // Ref: TKR5p315

	if (rank == 0 && verbose) {
		printf("Phase mobility is [%10.4g %10.4g]\n", Lmob[0], Lmob[1]);
	}

}

double timestep(const GRID2D& grid, const fp_t* DCr, const fp_t* DNb)
{
	const fp_t D[4] = {DCr[0], DCr[1], DNb[0], DNb[1]};
	double dt = (meshres * meshres) / (4.0 * *(std::max_element(D, D + 4)));

	#ifdef MPI_VERSION
	double mydt(dt);
	MPI::COMM_WORLD.Allreduce(&mydt, &dt, 1, MPI_DOUBLE, MPI_MIN);
	#endif

	return dt;
}

double timestep(const GRID2D& grid)
{
	double dt = 1.0;

	#ifdef _OPENMP
	#pragma omp parallel for reduction(min:dt)
	#endif
	for (int n = 0; n < nodes(grid); n++) {
		vector<fp_t>& GridN = grid(n);
		const fp_t xCr = GridN[0];
		const fp_t xNb = GridN[1];
		const fp_t pDel = p(GridN[NC]);
		const fp_t pLav = p(GridN[NC+1]);
		const fp_t pGam = 1.0 - pDel - pLav;
		const fp_t mCrCr = M_CrCr(xCr, xNb);
		const fp_t mCrNb = M_CrNb(xCr, xNb);
		const fp_t mNbCr = M_NbCr(xCr, xNb);
		const fp_t mNbNb = M_NbNb(xCr, xNb);

		const fp_t D[4] = {
			// D_gam
			std::fabs(pGam * (mCrCr * d2g_gam_dxCrCr() + mCrNb * d2g_gam_dxCrNb())), // D11
			std::fabs(pGam * (mCrCr * d2g_gam_dxNbCr() + mCrNb * d2g_gam_dxNbNb())), // D12
			std::fabs(pGam * (mNbCr * d2g_gam_dxCrCr() + mNbNb * d2g_gam_dxCrNb())), // D21
			std::fabs(pGam * (mNbCr * d2g_gam_dxNbCr() + mNbNb * d2g_gam_dxNbNb()))  // D22
			/*
			// D_del
			std::fabs(pDel * (mCrCr * d2g_del_dxCrCr() + mCrNb * d2g_del_dxCrNb())), // D11
			std::fabs(pDel * (mCrCr * d2g_del_dxNbCr() + mCrNb * d2g_del_dxNbNb())), // D12
			std::fabs(pDel * (mNbCr * d2g_del_dxCrCr() + mNbNb * d2g_del_dxCrNb())), // D21
			std::fabs(pDel * (mNbCr * d2g_del_dxNbCr() + mNbNb * d2g_del_dxNbNb())), // D22
			// D_lav
			std::fabs(pLav * (mCrCr * d2g_lav_dxCrCr() + mCrNb * d2g_lav_dxCrNb())), // D11
			std::fabs(pLav * (mCrCr * d2g_lav_dxNbCr() + mCrNb * d2g_lav_dxNbNb())), // D12
			std::fabs(pLav * (mNbCr * d2g_lav_dxCrCr() + mNbNb * d2g_lav_dxCrNb())), // D21
			std::fabs(pLav * (mNbCr * d2g_lav_dxNbCr() + mNbNb * d2g_lav_dxNbNb()))  // D22
			*/
		};

		const double local_dt = (meshres * meshres) / (4.0 * *(std::max_element(D, D + 4)));

		dt = std::min(local_dt, dt);
	}

	#ifdef MPI_VERSION
	double mydt(dt);
	MPI::COMM_WORLD.Allreduce(&mydt, &dt, 1, MPI_DOUBLE, MPI_MIN);
	#endif

	return dt;
}

void init_flat_composition(GRID2D& grid, std::mt19937& mtrand, fp_t& xCr0, fp_t& xNb0)
{
	/* Randomly choose enriched compositions in a rectangular region of the phase diagram
	 * corresponding to enriched IN625 per DICTRA simulations (mole fraction)
	 */

	#ifdef PLANAR
	double xCrE = 0.3125;
	double xNbE = 0.1575;
	#else
	std::uniform_real_distribution<double> enrichCrDist(enrich_min_Cr(), enrich_max_Cr());
	std::uniform_real_distribution<double> enrichNbDist(enrich_min_Nb(), enrich_max_Nb());

	double xCrE = enrichCrDist(mtrand);
	double xNbE = enrichNbDist(mtrand);
	#endif

	#ifdef MPI_VERSION
	MPI::COMM_WORLD.Barrier();
	MPI::COMM_WORLD.Bcast(&xCrE, 1, MPI_DOUBLE, 0);
	MPI::COMM_WORLD.Bcast(&xNbE, 1, MPI_DOUBLE, 0);
	#endif

	vector<fp_t> init(fields(grid), 0.);
	init[0] = xCrE;
	init[1] = xNbE;

	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (int n = 0; n < nodes(grid); n++) {
		grid(n) = init;
	}

	xCr0 = xCrE;
	xNb0 = xNbE;
}

void init_gaussian_enrichment(GRID2D& grid, std::mt19937& mtrand, fp_t& xCr0, fp_t& xNb0)
{
	/* Randomly choose enriched compositions in a rectangular region of the phase diagram
	 * corresponding to enriched IN625 per DICTRA simulations (mole fraction)
	 */
	std::uniform_real_distribution<double> matrixCrDist(matrix_min_Cr(), matrix_max_Cr());
	std::uniform_real_distribution<double> matrixNbDist(matrix_min_Nb(), matrix_max_Nb());
	std::uniform_real_distribution<double> enrichCrDist(enrich_min_Cr(), enrich_max_Cr());
	std::uniform_real_distribution<double> enrichNbDist(enrich_min_Nb(), enrich_max_Nb());

	double xCrM = matrixCrDist(mtrand);
	double xNbM = matrixNbDist(mtrand);
	double xCrE = enrichCrDist(mtrand);
	double xNbE = enrichNbDist(mtrand);

	#ifdef MPI_VERSION
	MPI::COMM_WORLD.Barrier();
	MPI::COMM_WORLD.Bcast(&xCrM, 1, MPI_DOUBLE, 0);
	MPI::COMM_WORLD.Bcast(&xNbM, 1, MPI_DOUBLE, 0);
	MPI::COMM_WORLD.Bcast(&xCrE, 1, MPI_DOUBLE, 0);
	MPI::COMM_WORLD.Bcast(&xNbE, 1, MPI_DOUBLE, 0);
	#endif

	// Zero initial condition
	const vector<fp_t> blank(fields(grid), 0.);

	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (int n = 0; n < nodes(grid); n++) {
		grid(n) = blank;
	}

	// Initialize matrix (gamma phase): bell curve along x-axis for Cr and Nb composition
	const int Nx = g1(grid, 0) - g0(grid, 0);
	const double a = -Nx * meshres / 3;
	const double b =  Nx * meshres / 3;

	vector<int> x(2, 0);
	for (x[1] = y0(grid); x[1] < y1(grid); x[1]++) {
		for (x[0] = x0(grid); x[0] < x1(grid); x[0]++) {
			const double pos = dx(grid, 0) * x[0];
			const double matrixCr = bell_curve(a, b, bell[0], pos, xCrM, xCrE);
			const double matrixNb = bell_curve(a, b, bell[1], pos, xNbM, xNbE);
			grid(x)[0] = std::min(1.0, std::max(0.0, matrixCr));
			grid(x)[1] = std::min(1.0, std::max(0.0, matrixNb));
		}
	}

	xCr0 = xCrM;
	xNb0 = xNbM;
}

void embed_OPC(GRID2D& grid,
               const vector<int>& x,
               const fp_t& par_xe_Cr,
               const fp_t& par_xe_Nb,
               const fp_t& R_precip,
               const int pid)
{
	/*
	const fp_t R_depletion_Cr = R_precip * sqrt(1.0 + (par_xe_Cr - xCr) / (xCr - xe_gam_Cr()));
	const fp_t R_depletion_Nb = R_precip * sqrt(1.0 + (par_xe_Nb - xNb) / (xNb - xe_gam_Nb()));
	*/
	const int R = int(//std::max(std::max(R_depletion_Cr, R_depletion_Nb),
	                           R_precip + 4 * ifce_width / meshres);

	for (int i = -R; i < R + 1; i++) {
		for (int j = -R; j < R + 1; j++) {
			vector<int> y(x);
			y[0] += i;
			y[1] += j;
			vector<fp_t>& GridY = grid(y);
			const fp_t& xCr = GridY[0];
			const fp_t& xNb = GridY[1];

			const fp_t r = sqrt(fp_t(i * i + j * j)); // distance to seed center
			const fp_t z = meshres * (r - R_precip);
			const fp_t inflation = 0.5;

			// Smoothly interpolate through the interface , TKR5p276
			const fp_t phi = tanh_interp(z, inflation * ifce_width);
			GridY[pid] = phi;
			GridY[0] = p(phi) * par_xe_Cr + p(1.0 - phi) * xCr;
			GridY[1] = p(phi) * par_xe_Nb + p(1.0 - phi) * xNb;
		}
	}
}

void seed_solitaire_delta(GRID2D& grid,
                          const fp_t sigma_del,
                          const fp_t lattice_const, const fp_t ifce_width,
                          const fp_t dx, const fp_t dt)
{
	fp_t dG_chem = 0.;
	fp_t R_star = 0., P_nuc = 0.;
	fp_t xCr = 0., xNb = 0.;

	const fp_t dV = dx * dx * dx;
	const fp_t Vatom = 0.25 * lattice_const * lattice_const * lattice_const; // assuming FCC
	const fp_t n_gam = dV / Vatom;
	const vector<int> x(2, 0);

	xCr = grid(x)[0];
	xNb = grid(x)[1];
	const fp_t pDel = p(grid(x)[NC]);

	nucleation_driving_force_delta(xCr, xNb, &dG_chem);
	nucleation_probability_sphere(xCr, xNb,
	                              dG_chem,
	                              pDel * (M_CrCr(xCr, xNb) * d2g_del_dxCrCr() + M_CrNb(xCr, xNb) * d2g_del_dxCrNb()),
	                              pDel * (M_NbCr(xCr, xNb) * d2g_del_dxNbCr() + M_NbNb(xCr, xNb) * d2g_del_dxNbNb()),
	                              sigma_del,
	                              Vatom,
	                              n_gam,
	                              dV, dt,
	                              &R_star,
	                              &P_nuc);
	if (R_star > 0.) {
		const fp_t R_precip = precip_stabilizer * R_star / dx;
		embed_OPC(grid, x,
		          xr_del_Cr(precip_stabilizer * R_star, 1.0),
		          xr_del_Nb(precip_stabilizer * R_star, 1.0),
		          R_precip,
		          NC);
	}
}

void seed_solitaire_laves(GRID2D& grid,
                          const fp_t sigma_lav,
                          const fp_t lattice_const, const fp_t ifce_width,
                          const fp_t dx, const fp_t dt)
{
	fp_t dG_chem = 0.;
	fp_t R_star = 0., P_nuc = 0.;
	fp_t xCr = 0., xNb = 0.;

	const fp_t dV = dx * dx * dx;
	const fp_t Vatom = 0.25 * lattice_const * lattice_const * lattice_const; // assuming FCC
	const fp_t n_gam = dV / Vatom;
	const vector<int> x(2, 0);

	xCr = grid(x)[0];
	xNb = grid(x)[1];
	const fp_t pLav = p(grid(x)[NC + 1]);

	nucleation_driving_force_laves(xCr, xNb, &dG_chem);
	nucleation_probability_sphere(xCr, xNb,
	                              dG_chem,
	                              pLav * (M_CrCr(xCr, xNb) * d2g_lav_dxCrCr() + M_CrNb(xCr, xNb) * d2g_lav_dxCrNb()),
	                              pLav * (M_NbCr(xCr, xNb) * d2g_lav_dxNbCr() + M_NbNb(xCr, xNb) * d2g_lav_dxNbNb()),
	                              sigma_lav,
	                              Vatom,
	                              n_gam,
	                              dV, dt,
	                              &R_star,
	                              &P_nuc);
	if (R_star > 0.) {
		const fp_t R_precip = precip_stabilizer * R_star / dx;
		embed_OPC(grid, x,
		          xr_lav_Cr(1.0, precip_stabilizer * R_star),
		          xr_lav_Nb(1.0, precip_stabilizer * R_star),
		          R_precip,
		          NC + 1);
	}
}

void seed_planar_delta(GRID2D& grid, const int w_precip)
{
	fp_t xCr = 0.0, xNb = 0.0;
	for (int n=0; n<nodes(grid); n++) {
		xCr += grid(n)[0];
		xNb += grid(n)[1];
	}
	xCr /= nodes(grid);
	xNb /= nodes(grid);

	const fp_t R_precip = meshres * w_precip;
	const fp_t inflation = 0.5;

	vector<int> x(2, 0);
	for (x[1] = x0(grid, 1); x[1] < x1(grid, 1); x[1]++) {
		for (x[0] = x0(grid, 0); x[0] < x1(grid, 0); x[0]++) {
			// Smoothly interpolate through the interface, TKR5p276
			vector<fp_t>& GridN = grid(x);
			const fp_t r = meshres * (x[0] - g0(grid, 0));
			const fp_t phi = tanh_interp(r - R_precip, inflation * ifce_width);
			GridN[NC] = phi;
			GridN[0] = p(phi) * xe_del_Cr() + p(1.0 - phi) * xCr;
			GridN[1] = p(phi) * xe_del_Nb() + p(1.0 - phi) * xNb;
		}
	}
}

void seed_planar_laves(GRID2D& grid, const int w_precip)
{
	fp_t xCr = 0.0, xNb = 0.0;
	for (int n=0; n<nodes(grid); n++) {
		xCr += grid(n)[0];
		xNb += grid(n)[1];
	}
	xCr /= nodes(grid);
	xNb /= nodes(grid);

	const fp_t R_precip = meshres * w_precip;
	const fp_t inflation = 0.5;

	vector<int> x(2, 0);
	for (x[1] = x0(grid, 1); x[1] < x1(grid, 1); x[1]++) {
		for (x[0] = x0(grid, 0); x[0] < x1(grid, 0); x[0]++) {
			// Smoothly interpolate through the interface, TKR5p276
			vector<fp_t>& GridN = grid(x);
			const fp_t r = meshres * (x[0] - g0(grid, 0));
			const fp_t phi = tanh_interp(r - R_precip, inflation * ifce_width);
			GridN[NC+1] = phi;
			GridN[0] = p(phi) * xe_lav_Cr() + p(1.0 - phi) * xCr;
			GridN[1] = p(phi) * xe_lav_Nb() + p(1.0 - phi) * xNb;
		}
	}
}

void init_tanh(GRID2D& grid, fp_t& xCr0, fp_t& xNb0, int index)
{
	const fp_t frac = 0.5;
	const fp_t xCrPre = (index == NC) ? xe_del_Cr() : xe_lav_Cr();
	const fp_t xNbPre = (index == NC) ? xe_del_Nb() : xe_lav_Nb();
	const fp_t xCrMat = xe_gam_Cr();
	const fp_t xNbMat = xe_gam_Nb();

	xCr0 = frac * xCrPre + (1.0 - frac) * xCrMat;
	xNb0 = frac * xNbPre + (1.0 - frac) * xNbMat;

	const fp_t L = frac * meshres * (MMSP::g1(grid, 0) - MMSP::g0(grid, 0));
	const fp_t w = 0.375 * L;

	vector<int> x(2, 0);
	for (x[1] = x0(grid, 1); x[1] < x1(grid, 1); x[1]++) {
		for (x[0] = x0(grid, 0); x[0] < x1(grid, 0); x[0]++) {
			// Smoothly interpolate through the interface, TKR5p276
			vector<fp_t>& GridN = grid(x);
			const fp_t r = meshres * (x[0] - g0(grid, 0));
			const fp_t phi = tanh_interp(r - L, w);
			GridN[0] = p(phi) * xCrPre + p(1.0 - phi) * xCrMat;
			GridN[1] = p(phi) * xNbPre + p(1.0 - phi) * xNbMat;
			GridN[NC ]  = 0.0;
			GridN[NC+1] = 0.0;
			GridN[index] = phi;
		}
	}
}

void init_tanh_delta(GRID2D& grid, fp_t& xCr0, fp_t& xNb0)
{
	init_tanh(grid, xCr0, xNb0, NC);
}

void init_tanh_laves(GRID2D& grid, fp_t& xCr0, fp_t& xNb0)
{
	init_tanh(grid, xCr0, xNb0, NC+1);
}

void seed_pair(GRID2D& grid,
               const fp_t sigma_del, const fp_t sigma_lav,
               const fp_t lattice_const, const fp_t ifce_width,
               const fp_t dx, const fp_t dt)
{
	fp_t dG_chem = 0.;
	fp_t R_star = 0., P_nuc = 0.;
	fp_t xCr = 0., xNb = 0.;
	vector<int> x(2, 0);

	const fp_t dV = dx * dx * dx;
	const fp_t n_gam = dV / vFccNi;

	// Embed a delta particle
	x[0] =-(g1(grid, 0) - g0(grid, 0)) / 5;
	x[1] = 0;
	xCr = grid(x)[0];
	xNb = grid(x)[1];
	const fp_t pDel = p(grid(x)[NC]);
	const fp_t pLav = p(grid(x)[NC + 1]);

	nucleation_driving_force_delta(xCr, xNb, &dG_chem);
	nucleation_probability_sphere(xCr, xNb,
	                              dG_chem,
	                              pDel * (M_CrCr(xCr, xNb) * d2g_del_dxCrCr() + M_CrNb(xCr, xNb) * d2g_del_dxCrNb()),
	                              pDel * (M_NbCr(xCr, xNb) * d2g_del_dxNbCr() + M_NbNb(xCr, xNb) * d2g_del_dxNbNb()),
	                              sigma_del,
	                              vFccNi,
	                              n_gam,
	                              dV,
								  dt,
	                              &R_star,
	                              &P_nuc);
	if (R_star > 0.) {
		const fp_t R_precip = precip_stabilizer * R_star / dx;
		embed_OPC(grid, x,
		          xe_del_Cr(),
		          xe_del_Nb(),
		          R_precip,
		          NC);
	}

	// Embed a Laves particle
	x[0] *= -1;
	x[1] *= -1;
	xCr = grid(x)[0];
	xNb = grid(x)[1];

	nucleation_driving_force_laves(xCr, xNb, &dG_chem);
	nucleation_probability_sphere(xCr, xNb,
	                              dG_chem,
	                              pLav * (M_CrCr(xCr, xNb) * d2g_lav_dxCrCr() + M_CrNb(xCr, xNb) * d2g_lav_dxCrNb()),
	                              pLav * (M_NbCr(xCr, xNb) * d2g_lav_dxNbCr() + M_NbNb(xCr, xNb) * d2g_lav_dxNbNb()),
	                              sigma_lav,
	                              vFccNi,
	                              n_gam,
	                              dV,
								  dt,
	                              &R_star,
	                              &P_nuc);
	if (R_star > 0.) {
		const fp_t R_precip = precip_stabilizer * R_star / dx;
		embed_OPC(grid, x,
		          xe_lav_Cr(),
		          xe_lav_Nb(),
		          R_precip,
		          NC + 1);
	}
}

void generate(int dim, const char* filename)
{
	assert (ifce_width > 4.5 * meshres);
	std::chrono::high_resolution_clock::time_point beginning = std::chrono::high_resolution_clock::now();

	int rank = 0;
	#ifdef MPI_VERSION
	rank = MPI::COMM_WORLD.Get_rank();
	#endif

	FILE* cfile = NULL;

	if (rank == 0)
		cfile = fopen("c.log", "w"); // existing log will be overwritten

	// Initialize pseudo-random number generator
	std::mt19937 mtrand; // Mersenne Twister PRNG
	const unsigned int seed = (std::chrono::high_resolution_clock::now() - beginning).count();
	mtrand.seed(seed);

	if (dim == 2) {
		#if defined(PLANAR)
		const int Nx = 768 * 0.5e-9 / meshres;
		const int Ny =  17;
		#elif defined(TANH)
		const int Nx = 1536 * 0.5e-9 / meshres;
		const int Ny =  17;
		#else
		/*
		const int Nx = 4000;
		const int Ny = 2500;
		*/
		const int Nx = 2000;
		const int Ny = 1250;
		#endif

		double Ntot = 1.0;
		GRID2D initGrid(NC + NP, -Nx / 2, Nx / 2, -Ny / 2, Ny / 2);
		for (int d = 0; d < dim; d++) {
			dx(initGrid, d) = meshres;
			Ntot *= g1(initGrid, d) - g0(initGrid, d);
			if (x0(initGrid, d) == g0(initGrid, d))
				b0(initGrid, d) = Neumann;
			if (x1(initGrid, d) == g1(initGrid, d))
				b1(initGrid, d) = Neumann;
		}

		fp_t xCr0 = 0.0, xNb0 = 0.0;

		#ifdef PLANAR
		init_flat_composition(initGrid, mtrand, xCr0, xNb0);
		#elif defined(TANH)
		if (rank == 0)
			std::cout << "Initializing tanh profile" << std::endl;
		init_tanh_delta(initGrid, xCr0, xNb0);
		#else
		init_gaussian_enrichment(initGrid, mtrand, xCr0, xNb0);
		#endif

		// === Sanity Check ===
		/*
		printf("System composition is %10.4g Cr, %10.4g Nb\n", xCr0, xNb0);
		const fp_t mCrCr = M_CrCr(xCr0, xNb0);
		const fp_t mCrNb = M_CrNb(xCr0, xNb0);
		const fp_t mNbNb = M_NbNb(xCr0, xNb0);
		printf("Mobility matrix is [%10.4g %10.4g]\n                   [%10.4g %10.4g]\n", mCrCr, mCrNb, mCrNb, mNbNb);
		*/

		const double del_frac = estimate_fraction_del(xCr0, xNb0);

		#ifdef PLANAR
		const int w_precip = 9 * ifce_width / meshres;
		seed_planar_delta(initGrid, w_precip);
		#elif defined(TANH)
		// no op
		#elif defined(PAIR)
		const fp_t nuc_dt = 1.0e-3;
		seed_pair(initGrid, s_delta(), s_laves(), lattice_const, ifce_width, meshres, nuc_dt);
        #elif !defined(NUCLEATION)
		const fp_t nuc_dt = 1.0e-3;
		seed_solitaire_delta(initGrid, s_delta(), lattice_const,
		                     ifce_width, meshres, nuc_dt);
		#endif

		ghostswap(initGrid);

		// write initial condition image
		const int nm = 3;

		// initialize memory

		fp_t** xCr = (fp_t**)calloc((Nx + 2), sizeof(fp_t*));
		fp_t** xNb = (fp_t**)calloc((Nx + 2), sizeof(fp_t*));
		fp_t** gamCr = (fp_t**)calloc((Nx + 2), sizeof(fp_t*));
		fp_t** gamNb = (fp_t**)calloc((Nx + 2), sizeof(fp_t*));
		fp_t** pDel = (fp_t**)calloc((Nx + 2), sizeof(fp_t*));
		fp_t** pLav = (fp_t**)calloc((Nx + 2), sizeof(fp_t*));
		fp_t** chem_nrg = (fp_t**)calloc((Nx + 2), sizeof(fp_t*));
		fp_t** grad_nrg = (fp_t**)calloc((Nx + 2), sizeof(fp_t*));

		xCr[0] = (fp_t*)calloc((Nx + 2) * (Ny + 2), sizeof(fp_t));
		xNb[0] = (fp_t*)calloc((Nx + 2) * (Ny + 2), sizeof(fp_t));
		gamCr[0] = (fp_t*)calloc((Nx + 2) * (Ny + 2), sizeof(fp_t));
		gamNb[0] = (fp_t*)calloc((Nx + 2) * (Ny + 2), sizeof(fp_t));
		pDel[0] = (fp_t*)calloc((Nx + 2) * (Ny + 2), sizeof(fp_t));
		pLav[0] = (fp_t*)calloc((Nx + 2) * (Ny + 2), sizeof(fp_t));
		chem_nrg[0] = (fp_t*)calloc((Nx + 2) * (Ny + 2), sizeof(fp_t));
		grad_nrg[0] = (fp_t*)calloc((Nx + 2) * (Ny + 2), sizeof(fp_t));

		for (int i = 1; i < Ny + 2; i++) {
			xCr[i] = &(xCr[0])[(Nx + 2) * i];
			xNb[i] = &(xNb[0])[(Nx + 2) * i];
			gamCr[i] = &(gamCr[0])[(Nx + 2) * i];
			gamNb[i] = &(gamNb[0])[(Nx + 2) * i];
			pDel[i] = &(pDel[0])[(Nx + 2) * i];
			pLav[i] = &(pLav[0])[(Nx + 2) * i];
			chem_nrg[i] = &(chem_nrg[0])[(Nx + 2) * i];
			grad_nrg[i] = &(grad_nrg[0])[(Nx + 2) * i];
		}

		#ifdef _OPENMP
		#pragma omp parallel for
		#endif
		for (int n = 0; n < nodes(initGrid); n++) {
			vector<int> x = position(initGrid, n);
			const int i = x[0] - g0(initGrid, 0) + 1;
			const int j = x[1] - g0(initGrid, 1) + 1; // offset required for proper imshow result
			xCr[j][i] = initGrid(n)[0];
			xNb[j][i] = initGrid(n)[1];
			pDel[j][i] = p(initGrid(n)[NC]);
			pLav[j][i] = p(initGrid(n)[NC+1]);

			double pGam = 1.0 - pDel[j][i] - pLav[j][i];
			double INV_DET = inv_fict_det(pDel[j][i], pGam, pLav[j][i]);
			gamCr[j][i] = fict_gam_Cr(INV_DET, xCr[j][i], xNb[j][i], pDel[j][i], pGam, pLav[j][i]);
			gamNb[j][i] = fict_gam_Nb(INV_DET, xCr[j][i], xNb[j][i], pDel[j][i], pGam, pLav[j][i]);
		}

		set_diffusion_matrix(xCr0, xNb0, D_Cr, D_Nb, Lmob, 0);

		vector<double> summary = summarize_fields(initGrid);
		const double energy = summarize_energy(initGrid, chem_nrg, grad_nrg);

		#ifdef TANH
		FILE* csv;
		csv = fopen("tanh.csv", "w");
		fprintf(csv, "X,xCr,xNb,pDel,pGam,pLav,xGCr,xGNb,xDCr,xDNb,xLCr,xLNb,fchem,fgrad\n");
		const int j = 0.5 * ylength(initGrid);
		for (int i = 1; i < xlength(initGrid) - 1; i++) {
			double pGam = 1.0 - pDel[j][i] - pLav[j][i];
			double INV_DET = inv_fict_det(pDel[j][i], pGam, pLav[j][i]);
			const double dCr = fict_del_Cr(INV_DET, xCr[j][i], xNb[j][i], pDel[j][i], pGam, pLav[j][i]);
			const double dNb = fict_del_Nb(INV_DET, xCr[j][i], xNb[j][i], pDel[j][i], pGam, pLav[j][i]);
			const double lCr = fict_lav_Cr(INV_DET, xCr[j][i], xNb[j][i], pDel[j][i], pGam, pLav[j][i]);
			const double lNb = fict_lav_Nb(INV_DET, xCr[j][i], xNb[j][i], pDel[j][i], pGam, pLav[j][i]);
			fprintf(csv, "%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g\n",
					meshres * (i + x0(initGrid)),
					xCr[j][i], xNb[j][i],
					pDel[j][i], pGam, pLav[j][i],
					gamCr[j][i], gamNb[j][i],
					dCr, dNb,
					lCr, lNb,
					chem_nrg[j][i],
					grad_nrg[j][i]);
		}
		fclose(csv);
		#endif

		const double dtTransformLimited = (meshres * meshres) / (std::pow(2.0, dim) * Lmob[0] * kappa[0]);
		fp_t dtDiffusionLimited = MMSP::timestep(initGrid, D_Cr, D_Nb);
		double round_dt = 0.0;
		double roundoff = 4e6;
		while (round_dt < 1e-20) {
			roundoff *= 10;
			round_dt = std::floor(roundoff * LinStab * std::min(dtTransformLimited, dtDiffusionLimited)) / roundoff;
		}
		const double dt = round_dt;

		if (rank == 0) {
			std::cout << "Timestep dt=" << dt
			          << ". Linear stability limit = " << dtDiffusionLimited
			          << ", interface limit = " << dtTransformLimited
			          << ".\nWith xCr = " << xCr0 << " and xNb = " << xNb0
			          << ", eqm. frac. Delta = " << del_frac << '.'
			          << std::endl;
			fprintf(cfile, "%10s %9s %9s %12s %12s %12s %12s\n",
			        "time", "x_Cr", "x_Nb", "gamma", "delta", "Laves", "energy");
			fprintf(cfile, "%10g %9g %9g %12g %12g %12g %12g\n",
			        0., summary[0], summary[1], summary[2], summary[3], summary[4], energy);

			printf("%9s %9s %9s %9s %9s\n",
			       "x_Cr", "x_Nb", " p_g", " p_d", " p_l");
			printf("%9g %9g %9g %9g %9g\n",
			       summary[0], summary[1], summary[2], summary[3], summary[4]);
		}

		output(initGrid, filename);

		const int step = 0;
		std::string imgname(filename);
		imgname.replace(imgname.find("dat"), 3, "png");
		write_matplotlib(xCr, xNb, pDel, pLav, chem_nrg, grad_nrg, gamCr, gamNb, Nx, Ny, nm, meshres, step, dt, imgname.c_str());

		free(xCr[0]);
		free(xNb[0]);
		free(gamCr[0]);
		free(gamNb[0]);
		free(pDel[0]);
		free(pLav[0]);
		free(chem_nrg[0]);
		free(grad_nrg[0]);
		free(xCr);
		free(xNb);
		free(gamCr);
		free(gamNb);
		free(pDel);
		free(pLav);
		free(chem_nrg);
		free(grad_nrg);

	} else {
		std::cerr << "Error: " << dim << "-dimensional grids unsupported." << std::endl;
		Abort(EXIT_FAILURE);
	}

	if (rank == 0)
		fclose(cfile);
}

} // namespace MMSP


double radius(const MMSP::vector<int>& a, const MMSP::vector<int>& b, const double& dx)
{
	double r = 0.0;
	for (int i = 0; i < a.length() && i < b.length(); i++)
		r += std::pow(a[i] - b[i], 2.0);
	return dx * std::sqrt(r);
}

template <typename T>
T gibbs(const MMSP::vector<T>& v)
{
	// Derivation: TKR5p280, Eq. (4)
	const T xCr = v[0];
	const T xNb = v[1];
	const T pDel = p(v[NC  ]);
	const T pLav = p(v[NC + 1]);

	const T pGam = 1.0 - pDel - pLav;
	const T inv_det = inv_fict_det(pDel, pGam, pLav);
	const T gam_Cr = fict_gam_Cr(inv_det, xCr, xNb, pDel, pGam, pLav);
	const T gam_Nb = fict_gam_Nb(inv_det, xCr, xNb, pDel, pGam, pLav);
	const T del_Cr = fict_del_Cr(inv_det, xCr, xNb, pDel, pGam, pLav);
	const T del_Nb = fict_del_Nb(inv_det, xCr, xNb, pDel, pGam, pLav);
	const T lav_Cr = fict_lav_Cr(inv_det, xCr, xNb, pDel, pGam, pLav);
	const T lav_Nb = fict_lav_Nb(inv_det, xCr, xNb, pDel, pGam, pLav);

	MMSP::vector<T> phiSq(NP);

	for (int i = 0; i < NP; i++)
		phiSq[i] = v[NC + i] * v[NC + i];

	T g  = pGam * g_gam(gam_Cr, gam_Nb);
	  g += pDel * g_del(del_Cr, del_Nb);
	  g += pLav * g_lav(lav_Cr, lav_Nb);

	for (int i = 0; i < NP; i++)
		g += omega[i] * phiSq[i] * pow(1.0 - v[NC + i], 2);

	// Trijunction penalty
	for (int i = 0; i < NP - 1; i++)
		for (int j = i + 1; j < NP; j++)
			g += 2.0 * alpha * phiSq[i] * phiSq[j];

	return g;
}

template <int dim, typename T>
MMSP::vector<T> maskedgradient(const MMSP::grid<dim, MMSP::vector<T> >& GRID,
                               const MMSP::vector<int>& x,
                               const int N)
{
	MMSP::vector<T> gradient(dim);
	MMSP::vector<int> s = x;

	for (int i = 0; i < dim; i++) {
		s[i] += 1;
		const T& yh = GRID(s)[N];
		s[i] -= 2;
		const T& yl = GRID(s)[N];
		s[i] += 1;

		double weight = 1.0 / (2.0 * dx(GRID, i));
		gradient[i] = weight * (yh - yl);
	}

	return gradient;
}

template<int dim, class T>
MMSP::vector<double> summarize_fields(MMSP::grid<dim, MMSP::vector<T> > const& GRID)
{
	#ifdef MPI_VERSION
	MPI_Request reqs;
	MPI_Status stat;
	#endif

	double Ntot = 1.0;
	for (int d = 0; d < dim; d++)
		Ntot *= double(MMSP::g1(GRID, d) - MMSP::g0(GRID, d));

	MMSP::vector<double> summary(NC + NP + 1, 0.0);

	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (int n = 0; n < MMSP::nodes(GRID); n++) {
		MMSP::vector<int> x = MMSP::position(GRID, n);
		MMSP::vector<T>& gridN = GRID(n);
		MMSP::vector<double> mySummary(NC + NP + 1, 0.0);

		for (int i = 0; i < NC; i++)
			mySummary[i] = gridN[i]; // compositions

		mySummary[NC] = 1.0; // gamma fraction init

		for (int i = 0; i < NP; i++) {
			const T newPhaseFrac = p(gridN[NC + i]);

			mySummary[NC + i + 1] = newPhaseFrac; // secondary phase fraction
			mySummary[NC    ] -= newPhaseFrac; // contributes to gamma phase;
		}

		#ifdef _OPENMP
		#pragma omp critical (critSum)
		#endif
		{
			summary += mySummary;
		}
	}

	for (int i = 0; i < NC + NP + 1; i++)
		summary[i] /= Ntot;

	#ifdef MPI_VERSION
	MMSP::vector<double> temp(summary);
	MPI_Ireduce(&(temp[0]), &(summary[0]), NC + NP + 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD, &reqs);
	MPI_Wait(&reqs, &stat);
	#endif

	return summary;
}

template<int dim, class T>
double summarize_energy(MMSP::grid<dim, MMSP::vector<T> > const& GRID, fp_t** chem_nrg, fp_t** grad_nrg)
{
	#ifdef MPI_VERSION
	MPI_Request reqs;
	MPI_Status stat;
	#endif

	double energy = 0.0;

	double dV = 1.0;
	for (int d = 0; d < dim; d++)
		dV *= MMSP::dx(GRID, d);

	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (int n = 0; n < MMSP::nodes(GRID); n++) {
		MMSP::vector<int> x = MMSP::position(GRID, n);
		MMSP::vector<T>& gridN = GRID(n);

		double chemEnergy = gibbs(gridN); // energy density init
		double gradEnergy = 0.;

		for (int i = 0; i < NP; i++) {
			const MMSP::vector<T> gradPhi = maskedgradient(GRID, x, NC + i);
			const T gradSq = (gradPhi * gradPhi); // vector inner product

			gradEnergy += kappa[i] * gradSq; // gradient contributes to energy
		}

		chemEnergy *= dV;
		gradEnergy *= dV;

		if (chem_nrg != NULL) {
			const int i = x[0] - MMSP::g0(GRID, 0) + 1; // nm / 2;
			const int j = x[1] - MMSP::g0(GRID, 1) + 1; // nm / 2;
			chem_nrg[j][i] = chemEnergy;
			grad_nrg[j][i] = gradEnergy;
		}

		#ifdef _OPENMP
		#pragma omp atomic
		#endif
		energy += chemEnergy + gradEnergy;
	}

	#ifdef MPI_VERSION
	double temp(energy);
	MPI_Ireduce(&temp, &energy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD, &reqs);
	MPI_Wait(&reqs, &stat);
	#endif

	return energy;
}

#endif

#include "main.cpp"
