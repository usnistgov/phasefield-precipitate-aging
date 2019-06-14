/**
 \file cuda625.cpp
 \brief Algorithms for 2D and 3D isotropic Cr-Nb-Ni alloy phase transformations
 This implementation depends on the GNU Scientific Library for multivariate root
 finding algorithms, and Mesoscale Microstructure Simulation Project for high-
 performance grid operations in parallel.
*/

#ifndef __CUDA625_CPP__
#define __CUDA625_CPP__

#include <cassert>
#include <chrono>
#include <cmath>
#include <random>
#include <set>
#include <sstream>
#include <vector>
#ifdef _OPENMP
#include "omp.h"
#endif

#include "MMSP.hpp"
#include "alloy625.hpp"
#include "cuda_data.h"
#include "enrichment.h"
#include "globals.h"
#include "mesh.h"
#include "nucleation.h"
#include "numerics.h"
#include "output.h"
#include "parabola625.h"

namespace MMSP
{

void init_flat_composition(GRID2D& grid, std::mt19937& mtrand)
{
	/* Randomly choose enriched compositions in a rectangular region of the phase diagram
	 * corresponding to enriched IN625 per DICTRA simulations (mole fraction)
	 */
	std::uniform_real_distribution<double> enrichCrDist(enrich_min_Cr(), enrich_max_Cr());
	std::uniform_real_distribution<double> enrichNbDist(enrich_min_Nb(), enrich_max_Nb());

	double xCrE = enrichCrDist(mtrand);
	double xNbE = enrichNbDist(mtrand);

	/* Favor a high delta-phase fraction, with a composition
	 * chosen from the gamma-delta edge of the three-phase field
	 */
	/*
	double xCrE = 0.415625;
	double xNbE = 0.0625;
	*/

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
}

void init_gaussian_enrichment(GRID2D& grid, std::mt19937& mtrand)
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
	const double a = -Nx * meshres / 2;
	const double b =  Nx * meshres / 2;

	vector<int> x(2, 0);
	for (x[1] = y0(grid); x[1] < y1(grid); x[1]++) {
		for (x[0] = x0(grid); x[0] < x1(grid); x[0]++) {
			const double pos = dx(grid, 0) * x[0];
			const double matrixCr = bell_curve(a, b, bell[0], pos, xCrM, xCrE);
			const double matrixNb = bell_curve(a, b, bell[1], pos, xNbM, xNbE);
			grid(x)[0] = matrixCr;
			grid(x)[1] = matrixNb;
		}
	}
}

void embed_OPC(GRID2D& grid,
               const vector<int>& x,
               const fp_t& xCr,
               const fp_t& xNb,
               const fp_t& par_xe_Cr,
               const fp_t& par_xe_Nb,
               const fp_t& R_precip,
               const int pid)
{
	const fp_t R_depletion_Cr = R_precip * sqrt(1. + (par_xe_Cr - xCr) / (xCr - xe_gam_Cr()));
	const fp_t R_depletion_Nb = R_precip * sqrt(1. + (par_xe_Nb - xNb) / (xNb - xe_gam_Nb()));
	const fp_t R = std::max(std::max(R_depletion_Cr, R_depletion_Nb),
							R_precip + 4 * ifce_width / meshres);

	for (int i = -R; i < R + 1; i++) {
		for (int j = -R; j < R + 1; j++) {
			vector<int> y(x);
			y[0] += i;
			y[1] += j;
			vector<fp_t>& GridY = grid(y);
			const fp_t r = sqrt(fp_t(i * i + j * j)); // distance to seed center
			const fp_t z = meshres * (r - R_precip);

			// Smoothly interpolate through the interface , TKR5p276
			GridY[0] = xe_gam_Cr()
				+ (par_xe_Cr - xe_gam_Cr()) * tanh_interp(z, 0.1 * ifce_width)
			    + (xCr - xe_gam_Cr()) * tanh_interp(meshres * (R_depletion_Cr - r), 0.1 * ifce_width);

			GridY[1] = xe_gam_Nb()
				+ (par_xe_Nb - xe_gam_Nb()) * tanh_interp(z, 0.1 * ifce_width)
			    + (xNb - xe_gam_Nb()) * tanh_interp(meshres * (R_depletion_Nb - r), 0.1 * ifce_width);

			GridY[pid] = tanh_interp(z, 0.5 * ifce_width);
		}
	}
}

void seed_solitaire(GRID2D& grid,
                    const fp_t D_CrCr, const fp_t D_NbNb,
                    const fp_t sigma_del, const fp_t sigma_lav,
                    const fp_t lattice_const, const fp_t ifce_width,
                    const fp_t dx, const fp_t dt, std::mt19937& mtrand)
{
	fp_t dG_chem = 0.;
	fp_t R_star = 0., P_nuc = 0.;
	fp_t xCr = 0., xNb = 0.;

	const fp_t dV = dx * dx * dx;
	const fp_t Vatom = 0.25 * lattice_const * lattice_const * lattice_const; // assuming FCC
	const fp_t n_gam = dV / Vatom;
	const vector<int> x(2, 0);

	#ifdef CONVERGENCE
	const double penny = 0.;
	#else
	std::uniform_real_distribution<double> heads_or_tails(0, 1);
	const double penny = heads_or_tails(mtrand);
	#endif

	#ifdef MPI_VERSION
	MPI::COMM_WORLD.Barrier();
	MPI::COMM_WORLD.Bcast(&penny, 1, MPI_DOUBLE, 0);
	#endif

	if (penny < 0.5) {
		// Embed a delta particle
		xCr = grid(x)[0];
		xNb = grid(x)[1];

		nucleation_driving_force_delta(xCr, xNb, &dG_chem);
		nucleation_probability_sphere(xCr, xNb,
		                              dG_chem,
		                              D_CrCr, D_NbNb,
		                              sigma_del,
		                              Vatom,
		                              n_gam,
		                              dV, dt,
		                              &R_star,
		                              &P_nuc);
		if (R_star > 0.) {
			const fp_t R_precip = precip_stabilizer * R_star / dx;
			embed_OPC(grid, x,
			          xCr, xNb,
			          xe_del_Cr(),
			          xe_del_Nb(),
			          R_precip,
			          NC);
		}
	} else {
		// Embed a Laves particle
		xCr = grid(x)[0];
		xNb = grid(x)[1];

		nucleation_driving_force_laves(xCr, xNb, &dG_chem);
		nucleation_probability_sphere(xCr, xNb,
		                              dG_chem,
		                              D_CrCr, D_NbNb,
		                              sigma_lav,
		                              Vatom,
		                              n_gam,
		                              dV, dt,
		                              &R_star,
		                              &P_nuc);
		if (R_star > 0.) {
			const fp_t R_precip = precip_stabilizer * R_star / dx;
			embed_OPC(grid, x,
			          xCr, xNb,
			          xe_lav_Cr(),
			          xe_lav_Nb(),
			          R_precip,
			          NC + 1);
		}
	}
}

void seed_planar_delta(GRID2D& grid, const int w_precip)
{
	const int mid = nodes(grid) / 2;
	const fp_t& xCr = grid(mid)[0];

	const fp_t& xNb = grid(mid)[1];

	const int R_depletion_Cr = w_precip * (1. + (xe_del_Cr() - xCr) / (xCr - xe_gam_Cr()));
	const int R_depletion_Nb = w_precip * (1. + (xe_del_Nb() - xNb) / (xNb - xe_gam_Nb()));

	for (int n = 0; n < nodes(grid); n++) {
		vector<fp_t>& GridN = grid(n);
		vector<int> x = position(grid, n);
		const int r = x[0] - g0(grid, 0);
		if (r <= w_precip) {
			GridN[0] = xe_del_Cr();
			GridN[1] = xe_del_Nb();
			GridN[2] = 1.;
		} else {
			if (r <= R_depletion_Cr) {
				GridN[0] = xe_gam_Cr();
			}
			if (r <= R_depletion_Nb) {
				GridN[1] = xe_gam_Nb();
			}
		}
	}
}

void seed_pair(GRID2D& grid,
               const fp_t D_CrCr, const fp_t D_NbNb,
               const fp_t sigma_del, const fp_t sigma_lav,
               const fp_t lattice_const, const fp_t ifce_width,
               const fp_t dx, const fp_t dt)
{
	fp_t dG_chem = 0.;
	fp_t R_star = 0., P_nuc = 0.;
	fp_t xCr = 0., xNb = 0.;
	vector<int> x(2, 0);

	const fp_t dV = dx * dx * dx;
	const fp_t Vatom = 0.25 * lattice_const * lattice_const * lattice_const; // assuming FCC
	const fp_t n_gam = dV / Vatom;

	// Embed a delta particle
	x[0] = -16;
	x[1] = (g1(grid, 1) - g0(grid, 1)) / 8;
	xCr = grid(x)[0];
	xNb = grid(x)[1];

	nucleation_driving_force_delta(xCr, xNb, &dG_chem);
	nucleation_probability_sphere(xCr, xNb,
	                              dG_chem,
	                              D_CrCr, D_NbNb,
	                              sigma_del,
	                              Vatom,
	                              n_gam,
	                              dV, dt,
	                              &R_star,
	                              &P_nuc);
	if (R_star > 0.) {
		const fp_t R_precip = precip_stabilizer * R_star / dx;
		embed_OPC(grid, x,
		          xCr, xNb,
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
	                              D_CrCr, D_NbNb,
	                              sigma_lav,
	                              Vatom,
	                              n_gam,
	                              dV, dt,
	                              &R_star,
	                              &P_nuc);
	if (R_star > 0.) {
		const fp_t R_precip = precip_stabilizer * R_star / dx;
		embed_OPC(grid, x,
		          xCr, xNb,
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

	const double dtTransformLimited = (meshres * meshres) / (std::pow(2.0, dim) * Lmob[0] * kappa[0]);
	const double dtDiffusionLimited = (meshres * meshres) / (std::pow(2.0, dim) * std::max(D_Cr[0], D_Nb[1]));
	const fp_t dt = LinStab * std::min(dtTransformLimited, dtDiffusionLimited);

	// Initialize pseudo-random number generator
	std::mt19937 mtrand; // Mersenne Twister PRNG
	const unsigned int seed = (std::chrono::high_resolution_clock::now() - beginning).count();
	mtrand.seed(seed);

	if (dim == 2) {
		#ifndef CONVERGENCE
		const int Nx = 4000;
		const int Ny = 2500;
		#else
		const int Nx = 768;
		const int Ny = 768;
		#endif
		double Ntot = 1.0;
		GRID2D initGrid(2 * NC + NP, -Nx / 2, Nx / 2, -Ny / 2, Ny / 2);
		for (int d = 0; d < dim; d++) {
			dx(initGrid, d) = meshres;
			Ntot *= g1(initGrid, d) - g0(initGrid, d);
			if (x0(initGrid, d) == g0(initGrid, d))
				b0(initGrid, d) = Neumann;
			if (x1(initGrid, d) == g1(initGrid, d))
				b1(initGrid, d) = Neumann;
		}
		grid<2, double> nickGrid(initGrid, 1);

		if (rank == 0)
			std::cout << "Timestep dt=" << dt
			          << ". Linear stability limits: dtTransformLimited=" << dtTransformLimited
			          << ", dtDiffusionLimited=" << dtDiffusionLimited
			          << '.' << std::endl;

		#ifdef CONVERGENCE
		init_flat_composition(initGrid, mtrand);
		#else
		init_gaussian_enrichment(initGrid, mtrand);
		#endif

		#ifdef CONVERGENCE
		// Embed a particle
		/*
		const int w_precip = glength(initGrid, 0) / 6;
		seed_planar_delta(initGrid, w_precip);
		*/
		seed_solitaire(initGrid, D_Cr[0], D_Nb[1],
		               s_delta(), s_laves(),
		               lattice_const, ifce_width, meshres, dt, mtrand);
		#endif

		// Update fictitious compositions
		for (int n = 0; n < nodes(initGrid); n++)
			update_compositions(initGrid(n));

		ghostswap(initGrid);

		vector<double> summary = summarize_fields(initGrid);
		double energy = summarize_energy(initGrid);

		if (rank == 0) {
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

		// write initial condition image
		fp_t** xNi = (fp_t**)calloc(Nx, sizeof(fp_t*));
		xNi[0]     = (fp_t*)calloc(Nx * Ny, sizeof(fp_t));

		fp_t** phi = (fp_t**)calloc(Nx, sizeof(fp_t*));
		phi[0]     = (fp_t*)calloc(Nx * Ny, sizeof(fp_t));

		for (int i = 1; i < Ny; i++) {
			xNi[i] = &(xNi[0])[Nx * i];
			phi[i] = &(phi[0])[Nx * i];
		}
		const int xoff = x0(initGrid);
		const int yoff = y0(initGrid);

		#ifdef _OPENMP
		#pragma omp parallel for
		#endif
		for (int n = 0; n < nodes(initGrid); n++) {
			vector<int> x = position(nickGrid, n);
			const int i = x[0] - xoff;
			const int j = x[1] - yoff;
			xNi[j][i] = 1. - initGrid(n)[0] - initGrid(n)[1];
			phi[j][i] = h(initGrid(n)[2]) + h(initGrid(n)[3]);
		}

		std::string imgname(filename);
		imgname.replace(imgname.find("dat"), 3, "png");

		#ifdef MPI_VERSION
		std::cerr << "Error: cannot write images in parallel." << std::endl;
		Abort(-1);
		#endif
		const int nm = 0, step = 0;
		const double dt = 1.;
		write_matplotlib(xNi, phi, Nx, Ny, nm, meshres, step, dt, imgname.c_str());

		free(xNi[0]);
		free(xNi);
	} else {
		std::cerr << "Error: " << dim << "-dimensional grids unsupported." << std::endl;
		Abort(-1);
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

template<typename T>
void update_compositions(MMSP::vector<T>& GRIDN)
{
	const T& xcr = GRIDN[0];
	const T& xnb = GRIDN[1];

	const T fdel = h(GRIDN[2]);
	const T flav = h(GRIDN[3]);
	const T fgam = 1. - fdel - flav;

	const T inv_det = inv_fict_det(fdel, fgam, flav);
	GRIDN[NC + NP  ] = fict_gam_Cr(inv_det, xcr, xnb, fdel, fgam, flav);
	GRIDN[NC + NP + 1] = fict_gam_Nb(inv_det, xcr, xnb, fdel, fgam, flav);
}

template <typename T>
T gibbs(const MMSP::vector<T>& v)
{
	const T xCr = v[0];
	const T xNb = v[1];
	const T f_del = h(v[NC  ]);
	const T f_lav = h(v[NC + 1]);
	const T f_gam = 1.0 - f_del - f_lav;
	const T inv_det = inv_fict_det(f_del, f_gam, f_lav);
	const T gam_Cr = v[NC + NP];
	const T gam_Nb = v[NC + NP];
	const T del_Cr = fict_del_Cr(inv_det, xCr, xNb, f_del, f_gam, f_lav);
	const T del_Nb = fict_del_Nb(inv_det, xCr, xNb, f_del, f_gam, f_lav);
	const T lav_Cr = fict_lav_Cr(inv_det, xCr, xNb, f_del, f_gam, f_lav);
	const T lav_Nb = fict_lav_Nb(inv_det, xCr, xNb, f_del, f_gam, f_lav);

	MMSP::vector<T> vsq(NP);

	for (int i = 0; i < NP; i++)
		vsq[i] = v[NC + i] * v[NC + i];

	T g  = f_gam * g_gam(gam_Cr, gam_Nb);
	g += f_del * g_del(del_Cr, del_Nb);
	g += f_lav * g_lav(lav_Cr, lav_Nb);

	for (int i = 0; i < NP; i++)
		g += omega[i] * vsq[i] * pow(1.0 - v[NC + i], 2);

	// Trijunction penalty
	for (int i = 0; i < NP - 1; i++)
		for (int j = i + 1; j < NP; j++)
			g += 2.0 * alpha * vsq[i] * vsq[j];

	return g;
}

template <int dim, typename T>
MMSP::vector<T> maskedgradient(const MMSP::grid<dim, MMSP::vector<T> >& GRID, const MMSP::vector<int>& x, const int N)
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
			const T newPhaseFrac = h(gridN[NC + i]);

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
double summarize_energy(MMSP::grid<dim, MMSP::vector<T> > const& GRID)
{
	#ifdef MPI_VERSION
	MPI_Request reqs;
	MPI_Status stat;
	#endif

	double dV = 1.0;
	for (int d = 0; d < dim; d++)
		dV *= MMSP::dx(GRID, d);

	double energy = 0.0;

	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (int n = 0; n < MMSP::nodes(GRID); n++) {
		MMSP::vector<int> x = MMSP::position(GRID, n);
		MMSP::vector<T>& gridN = GRID(n);

		double myEnergy = dV * gibbs(gridN); // energy density init

		for (int i = 0; i < NP; i++) {
			const MMSP::vector<T> gradPhi = maskedgradient(GRID, x, NC + i);
			const T gradSq = (gradPhi * gradPhi); // vector inner product

			myEnergy += dV * kappa[i] * gradSq; // gradient contributes to energy
		}

		#ifdef _OPENMP
		#pragma omp atomic
		#endif
		energy += myEnergy;
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
