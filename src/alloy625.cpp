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
#include "data.cuh"
#include "discretization.cuh"
#include "enrichment.h"
#include "mesh.h"
#include "nucleation.h"
#include "output.h"
#include "parabola625.h"
#include "parameters.h"

namespace MMSP
{

fp_t timestep(const GRID2D& grid)
{
	fp_t dt = 1.0;

	#ifdef _OPENMP
	#pragma omp parallel for reduction(min:dt)
	#endif
	for (int n = 0; n < nodes(grid); n++) {
		vector<fp_t>& GridN = grid(n);
		const fp_t xCr = GridN[0];
		const fp_t xNb = GridN[1];
		const fp_t f_del = h(GridN[NC]);
		const fp_t f_lav = h(GridN[NC+1]);

		const fp_t D11 = D_CrCr(xCr, xNb, f_del, f_lav);
		const fp_t D22 = D_NbNb(xCr, xNb, f_del, f_lav);
		const fp_t local_dt = (meshres * meshres) /
			(4.0 * std::max(std::fabs(D11), std::fabs(D22)));

		dt = std::min(local_dt, dt);
	}

	return dt;
}


void init_flat_composition(GRID2D& grid, std::mt19937& mtrand)
{
	/* Randomly choose enriched compositions in a rectangular region of the phase diagram
	 * corresponding to enriched IN625 per DICTRA simulations (mole fraction)
	 */

	/*
	std::uniform_real_distribution<double> enrichCrDist(enrich_min_Cr(), enrich_max_Cr());
	std::uniform_real_distribution<double> enrichNbDist(enrich_min_Nb(), enrich_max_Nb());

	double xCrE = enrichCrDist(mtrand);
	double xNbE = enrichNbDist(mtrand);
	*/

	/* Favor a high delta-phase fraction, with a composition
	 * chosen from the gamma-delta edge of the three-phase field
	 */
	double xCrE = 0.415625;
	double xNbE = 0.0625;

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
	const fp_t R_depletion_Cr = R_precip * sqrt(1.0 + (par_xe_Cr - xCr) / (xCr - xe_gam_Cr()));
	const fp_t R_depletion_Nb = R_precip * sqrt(1.0 + (par_xe_Nb - xNb) / (xNb - xe_gam_Nb()));
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

	std::uniform_real_distribution<double> heads_or_tails(0, 1);
	const double penny = heads_or_tails(mtrand);

	#ifdef MPI_VERSION
	MPI::COMM_WORLD.Barrier();
	MPI::COMM_WORLD.Bcast(&penny, 1, MPI_DOUBLE, 0);
	#endif

	if (penny < 0.5) {
		// Embed a delta particle
		xCr = grid(x)[0];
		xNb = grid(x)[1];
		const fp_t phi_del = h(grid(x)[NC]);
		const fp_t phi_lav = h(grid(x)[NC+1]);

		nucleation_driving_force_delta(xCr, xNb, &dG_chem);
		nucleation_probability_sphere(xCr, xNb,
		                              dG_chem,
		                              D_CrCr(xCr, xNb, phi_del, phi_lav),
									  D_NbNb(xCr, xNb, phi_del, phi_lav),
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
		const fp_t phi_del = h(grid(x)[NC]);
		const fp_t phi_lav = h(grid(x)[NC+1]);

		nucleation_driving_force_laves(xCr, xNb, &dG_chem);
		nucleation_probability_sphere(xCr, xNb,
		                              dG_chem,
		                              D_CrCr(xCr, xNb, phi_del, phi_lav),
									  D_NbNb(xCr, xNb, phi_del, phi_lav),
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
	fp_t xCr = 0.0, xNb = 0.0;
	for (int n=0; n<nodes(grid); n++) {
		xCr += grid(n)[0];
		xNb += grid(n)[1];
	}
	xCr /= nodes(grid);
	xNb /= nodes(grid);

	const fp_t R_precip = meshres * w_precip;
	const fp_t R_depletion_Cr = R_precip * (1.0 + (xe_del_Cr() - xCr) / (xCr - xe_gam_Cr()));
	const fp_t R_depletion_Nb = R_precip * (1.0 + (xe_del_Nb() - xNb) / (xNb - xe_gam_Nb()));

	vector<int> x(2, 0);
	for (x[1] = x0(grid, 1); x[1] < x1(grid, 1); x[1]++) {
		for (x[0] = x0(grid, 0); x[0] < x1(grid, 0); x[0]++) {
			// Smoothly interpolate through the interface, TKR5p276
			vector<fp_t>& GridN = grid(x);
			const fp_t r = meshres * (x[0] - g0(grid, 0));
			GridN[NC] = tanh_interp(r - R_precip, ifce_width);
			GridN[0] = xe_del_Cr() * tanh_interp(r - R_precip, ifce_width)
			         + xe_gam_Cr() * tanh_interp(R_precip - r, ifce_width)
			         - xe_gam_Cr() * tanh_interp(R_depletion_Cr - r, ifce_width)
			         + xCr * tanh_interp(R_depletion_Cr - r, ifce_width);
			GridN[1] = xe_del_Nb() * tanh_interp(r - R_precip, ifce_width)
			         + xe_gam_Nb() * tanh_interp(R_precip - r, ifce_width)
			         - xe_gam_Nb() * tanh_interp(R_depletion_Nb - r, ifce_width)
			         + xNb * tanh_interp(R_depletion_Nb - r, ifce_width);
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

	// Initialize pseudo-random number generator
	std::mt19937 mtrand; // Mersenne Twister PRNG
	const unsigned int seed = (std::chrono::high_resolution_clock::now() - beginning).count();
	mtrand.seed(seed);

	if (dim == 2) {
		/*
		const int Nx = 4000;
		const int Ny = 2500;
		*/
		const int Nx = 768;
		const int Ny =  32;

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

		init_flat_composition(initGrid, mtrand);
		/*
		init_gaussian_enrichment(initGrid, mtrand);
		*/

		const int w_precip = glength(initGrid, 0) / 6;
		seed_planar_delta(initGrid, w_precip);
		/*
		seed_solitaire(initGrid, s_delta(), s_laves(), lattice_const,
					   ifce_width, meshres, dt, mtrand);
		*/

		ghostswap(initGrid);

		vector<double> summary = summarize_fields(initGrid);
		const double energy = summarize_energy(initGrid);

		const double dtTransformLimited = (meshres * meshres) / (std::pow(2.0, dim) * Lmob[0] * kappa[0]);
		fp_t dtDiffusionLimited = MMSP::timestep(initGrid);
		const double dt = std::floor(4e11 * LinStab * std::min(dtTransformLimited, dtDiffusionLimited)) / 4e11;

		if (rank == 0) {
			std::cout << "Timestep dt=" << dt
			          << ". Linear stability limits: dtTransformLimited=" << dtTransformLimited
			          << ", dtDiffusionLimited=" << dtDiffusionLimited
			          << '.' << std::endl;
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

		fp_t** xCr = (fp_t**)calloc((Nx + 2), sizeof(fp_t*));
		fp_t** xNb = (fp_t**)calloc((Nx + 2), sizeof(fp_t*));
		fp_t** pDel = (fp_t**)calloc((Nx + 2), sizeof(fp_t*));
		fp_t** pLav = (fp_t**)calloc((Nx + 2), sizeof(fp_t*));
		xCr[0] = (fp_t*)calloc((Nx + 2) * (Ny + 2), sizeof(fp_t));
		xNb[0] = (fp_t*)calloc((Nx + 2) * (Ny + 2), sizeof(fp_t));
		pDel[0] = (fp_t*)calloc((Nx + 2) * (Ny + 2), sizeof(fp_t));
		pLav[0] = (fp_t*)calloc((Nx + 2) * (Ny + 2), sizeof(fp_t));

		for (int i = 1; i < Ny + 2; i++) {
			xCr[i] = &(xCr[0])[(Nx + 2) * i];
			xNb[i] = &(xNb[0])[(Nx + 2) * i];
			pDel[i] = &(pDel[0])[(Nx + 2) * i];
			pLav[i] = &(pLav[0])[(Nx + 2) * i];
		}

		#ifdef _OPENMP
		#pragma omp parallel for
		#endif
		for (int n = 0; n < nodes(initGrid); n++) {
			vector<int> x = position(initGrid, n);
			const int i = x[0] - g0(initGrid, 0) + 1;
			const int j = x[1] - g0(initGrid, 1) + 1; // offset required for proper imshow result
			                                          // (mmsp2png of mesh is correct)
			xCr[j][i] = initGrid(n)[0];
			xNb[j][i] = initGrid(n)[1];
			pDel[j][i] = initGrid(n)[NC];
			pLav[j][i] = initGrid(n)[NC+1];
		}

		std::string imgname(filename);
		imgname.replace(imgname.find("dat"), 3, "png");

		#ifdef MPI_VERSION
		std::cerr << "Error: cannot write images in parallel." << std::endl;
		Abort(EXIT_FAILURE);
		#endif
		const int nm = 3, step = 0;
		write_matplotlib(xCr, xNb, pDel, pLav, Nx, Ny, nm, meshres, step, dt, imgname.c_str());

		free(xCr[0]);
		free(xNb[0]);
		free(pDel[0]);
		free(pLav[0]);
		free(xCr);
		free(xNb);
		free(pDel);
		free(pLav);

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
	const T f_del = h(v[NC  ]);
	const T f_lav = h(v[NC + 1]);

	const T f_gam = 1.0 - f_del - f_lav;
	const T inv_det = inv_fict_det(f_del, f_gam, f_lav);
	const T gam_Cr = fict_gam_Cr(inv_det, xCr, xNb, f_del, f_gam, f_lav);
	const T gam_Nb = fict_gam_Nb(inv_det, xCr, xNb, f_del, f_gam, f_lav);
	const T del_Cr = fict_del_Cr(inv_det, xCr, xNb, f_del, f_gam, f_lav);
	const T del_Nb = fict_del_Nb(inv_det, xCr, xNb, f_del, f_gam, f_lav);
	const T lav_Cr = fict_lav_Cr(inv_det, xCr, xNb, f_del, f_gam, f_lav);
	const T lav_Nb = fict_lav_Nb(inv_det, xCr, xNb, f_del, f_gam, f_lav);

	MMSP::vector<T> phiSq(NP);

	for (int i = 0; i < NP; i++)
		phiSq[i] = v[NC + i] * v[NC + i];

	T g  = f_gam * g_gam(gam_Cr, gam_Nb);
	g += f_del * g_del(del_Cr, del_Nb);
	g += f_lav * g_lav(lav_Cr, lav_Nb);

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
