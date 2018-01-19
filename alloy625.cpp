/*************************************************************************************
 * File: alloy625.cpp                                                                *
 * Algorithms for 2D and 3D isotropic Cr-Nb-Ni alloy phase transformations           *
 * This implementation depends on the GNU Scientific Library for multivariate root   *
 * finding algorithms, and Mesoscale Microstructure Simulation Project for high-     *
 * performance grid operations in parallel.                                          *
 *                                                                                   *
 * Questions/comments to trevor.keller@nist.gov (Trevor Keller, Ph.D.)               *
 *                                                                                   *
 * This software was developed at the National Institute of Standards and Technology *
 * by employees of the Federal Government in the course of their official duties.    *
 * Pursuant to title 17 section 105 of the United States Code this software is not   *
 * subject to copyright protection and is in the public domain. NIST assumes no      *
 * responsibility whatsoever for the use of this code by other parties, and makes no *
 * guarantees, expressed or implied, about its quality, reliability, or any other    *
 * characteristic. We would appreciate acknowledgement if the software is used.      *
 *                                                                                   *
 * This software can be redistributed and/or modified freely provided that any       *
 * derivative works bear some notice that they are derived from it, and any modified *
 * versions bear some notice that they have been modified. Derivative works that     *
 * include MMSP or other software licensed under the GPL may be subject to the GPL.  *
 *************************************************************************************/


#ifndef ALLOY625_UPDATE
#define ALLOY625_UPDATE
#include<set>
#include<cmath>
#include<random>
#include<sstream>
#include<vector>
#ifdef _OPENMP
#include"omp.h"
#endif
#include<gsl/gsl_blas.h>
#include<gsl/gsl_math.h>
#include<gsl/gsl_roots.h>
#include<gsl/gsl_vector.h>
#include<gsl/gsl_multiroots.h>
#include"MMSP.hpp"
#include"alloy625.hpp"

// Taylor series is your best bet.
#if defined CALPHAD
	#include"energy625.c"
#elif defined PARABOLA
	#include"parabola625.c"
#else
	#include"taylor625.c"
#endif


// Note: alloy625.hpp contains important declarations and comments. Have a look.

/* Free energy expressions are generated from CALPHAD using pycalphad and SymPy
 * by the Python script CALPHAD_energy.py. It produces three versions:
 * * energy625.c:   pure CALPHAD expression with derivatives; neither smooth nor continuously differentiable
 * * taylor625.c:   Taylor series expansion of CALPHAD surface about invariant points on phase diagram, RECOMMENDED
 */


/* =============================================== *
 * Implement MMSP kernels: generate() and update() *
 * =============================================== */

/* Representation includes eleven field variables:
 *
 * X0.  molar fraction of Cr + Mo
 * X1.  molar fraction of Nb
 *
 * P2.  phase fraction of delta
 * P3.  phase fraction of Laves
 *
 * C4.  Cr molar fraction in pure gamma
 * C5.  Nb molar fraction in pure gamma
 *
 * C6.  Cr molar fraction in pure delta
 * C7.  Nb molar fraction in pure delta
 *
 * C8. Cr molar fraction in pure Laves
 * C9. Nb molar fraction in pure Laves
 *
 * D10. debugging information (deviation of field variables or max. field velocity normal to the interface)
 */

/* Based on experiments (EDS) and simulations (DICTRA),
 * additively manufactured IN625 has these compositions:
 *
 * Element  Nominal  Interdendritic (mol %)
 * Cr+Mo      30%      31%
 * Nb          2%      13%
 * Ni         68%      56%
 */


// Define equilibrium phase compositions at global scope. Gamma is nominally 30% Cr, 2% Nb,
// as defined in the first elements of the two following arrays. The generate() function
// will adjust the initial gamma composition depending on the type, amount, and composition
// of the secondary phases to maintain the system's nominal composition.
#ifdef PARABOLA
// Parabolic free energy requires a system composition inside the three-phase coexistence region
//                        nominal   delta        laves         gamma-enrichment
const field_t xCr[NP+2] = {0.250,   xe_del_Cr(), xe_lav_Cr(),  0.005};
const field_t xNb[NP+2] = {0.150,   xe_del_Nb(), xe_lav_Nb(),  0.050};
#else
// CALPHAD and Taylor approximations admit any system composition
//                        nominal   delta        laves         gamma-enrichment
const field_t xCr[NP+2] = {0.300,   xe_del_Cr(), xe_lav_Cr(),  0.310 - 0.300};
const field_t xNb[NP+2] = {0.020,   xe_del_Nb(), xe_lav_Nb(),  0.130 - 0.020};
#endif

// Define st.dev. of Gaussians for alloying element segregation
//                       Cr        Nb
const double bell[NC] = {150e-9, 50e-9}; // est. between 80-200 nm from SEM

// Kinetic and model parameters
const double meshres = 5e-9; // grid spacing (m)
const field_t alpha = 1.07e11;  // three-phase coexistence coefficient (J/m^3)

// Diffusion constants in FCC Ni from Xu (m^2/s)
//                        Cr        Nb
const field_t D_Cr[NC] = {2.16e-15, 0.56e-15}; // first column of diffusivity matrix
const field_t D_Nb[NC] = {2.97e-15, 4.29e-15}; // second column of diffusivity matrix

//                         delta    Laves
const field_t kappa[NP] = {1.24e-8, 1.24e-8}; // gradient energy coefficient (J/m)

// Choose numerical diffusivity to lock chemical and transformational timescales
//                        delta      Laves
const field_t Lmob[NP] = {2.904e-11, 2.904e-11}; // numerical mobility (m^2/Ns), Zhou's numbers
//const field_t Lmob[NP] = {1.92e-12, 1.92e-12}; // numerical mobility (m^2/Ns), Xu's numbers

//                        delta  Laves
const field_t sigma[NP] = {1.01, 1.01}; // (J/m^2)

// Interfacial width
const field_t width_factor = 2.2; // 2.2 if interface is [0.1,0.9]; 2.94 if [0.05,0.95]
const field_t ifce_width = 10.0*meshres; // ensure at least 7 points through the interface
const field_t omega[NP] = {3.0 * width_factor * sigma[0] / ifce_width, // delta
                           3.0 * width_factor * sigma[1] / ifce_width  // Laves
                          };       // multiwell height (J/m^3)

// Numerical considerations
const bool useNeumann = true;     // apply zero-flux boundaries (Neumann type)?
const bool useTanh = false;       // apply tanh profile to initial profile of composition and phase
const double epsilon = 1e-12;     // what to consider zero to avoid log(c) explosions

const field_t root_tol  = 1e-4;   // residual tolerance (default is 1e-7)
const int root_max_iter = 1e6;    // default is 1000, increasing probably won't change anything but your runtime

#ifdef PARABOLA
const field_t LinStab = 1.0 / 19.42501; // threshold of linear stability (von Neumann stability condition)
#else
const field_t LinStab = 1.0 / 2913.753; // threshold of linear stability (von Neumann stability condition)
#endif

namespace MMSP
{

void generate(int dim, const char* filename)
{
	int rank = 0;
	#ifdef MPI_VERSION
	rank = MPI::COMM_WORLD.Get_rank();
 	#endif

	FILE* cfile = NULL;

	if (rank == 0)
		cfile = fopen("c.log", "w"); // existing log will be overwritten

	const double dtp = (meshres*meshres)/(std::pow(2.0, dim) * Lmob[0]*kappa[0]); // transformation-limited timestep
	const double dtc = (meshres*meshres)/(std::pow(2.0, dim) * std::max(D_Cr[0], D_Nb[1])); // diffusion-limited timestep
	double dt = LinStab * std::min(dtp, dtc);

	// Initialize pseudo-random number generator
	std::random_device rd; // PRNG seed generator
	std::mt19937 mtrand(rd()); // Mersenne Twister
	std::uniform_real_distribution<double> unidist(0, 1); // uniform distribution on [0, 1)

	if (dim==1) {
		// Construct grid
		const int Nx = 768; // divisible by 12 and 64
		double dV = 1.0;
		double Ntot = 1.0;
		GRID1D initGrid(NC+NP+NC*(NP+1)+1, 0, Nx);
		for (int d = 0; d < dim; d++) {
			dx(initGrid,d)=meshres;
			dV *= meshres;
			Ntot *= g1(initGrid, d) - g0(initGrid, d);
			if (useNeumann) {
				if (x0(initGrid, d) == g0(initGrid, d))
					b0(initGrid, d) = Neumann;
				if (x1(initGrid, d) == g1(initGrid, d))
					b1(initGrid, d) = Neumann;
			}
		}

		// Sanity check on system size and  particle spacing
		if (rank == 0)
			std::cout << "Timestep dt=" << dt << ". Linear stability limits: dtp=" << dtp << " (transformation-limited), dtc="<< dtc << " (diffusion-limited)." << std::endl;

		// Precipitate radii: minimum for thermodynamic stability is 7.5 nm,
		//                    minimum for numerical stability is 14*dx (due to interface width).
		const field_t rPrecip[NP] = {5.0*7.5e-9 / dx(initGrid,0),  // delta
		                             5.0*7.5e-9 / dx(initGrid,0)}; // Laves

		// Zero initial condition
		const vector<field_t> blank(fields(initGrid), 0);
		#ifdef _OPENMP
		#pragma omp parallel for
		#endif
		for (int n = 0; n < nodes(initGrid); n++) {
			initGrid(n) = blank;
		}

		// Initialize matrix (gamma phase): bell curve along x, each stripe in y is identical (with small fluctuations)
		Composition comp;
		//comp += enrichMatrix(initGrid, bell[0], bell[1]);



		if (1) {
			/* ============================================= *
			 * Three-phase Stripe Growth test configuration  *
			 *                                               *
			 * Seed a 1.0 um domain with 3 stripes (one of   *
			 * each secondary phase) to test competitive     *
			 * growth without curvature                      *
			 * ============================================= */

			vector<int> origin(dim, 0);
			const int xoffset[NP] = {int(-16 * (5e-9 / meshres)), int(16 * (5e-9 / meshres))}; //  80 nm

			for (int j = 0; j < NP; j++) {
				origin[0] = Nx / 2 + xoffset[j];
				comp += embedStripe(initGrid, origin, j+NC, rPrecip[j], xCr[j+1], xNb[j+1]);
			}

		} else {
			if (rank == 0)
				std::cerr<<"Error: specify an initial condition!"<<std::endl;
			MMSP::Abort(-1);
		}

		// Synchronize global initial condition parameters
		#ifdef MPI_VERSION
		Composition myComp;
		myComp += comp;
		MPI::COMM_WORLD.Barrier();
		// Caution: Primitive. Will not scale to large MPI systems.
		MPI::COMM_WORLD.Allreduce(&(myComp.N[0]), &(comp.N[0]), NP+1, MPI_INT, MPI_SUM);
		for (int j = 0; j < NP+1; j++) {
			MPI::COMM_WORLD.Barrier();
			MPI::COMM_WORLD.Allreduce(&(myComp.x[j][0]), &(comp.x[j][0]), NC, MPI_DOUBLE, MPI_SUM);
		}
		#endif


		// Initialize matrix to achieve specified system composition
		field_t matCr = Ntot * xCr[0];
		field_t matNb = Ntot * xNb[0];
		double Nmat  = Ntot;
		for (int i = 0; i < NP+1; i++) {
			Nmat  -= comp.N[i];
			matCr -= comp.x[i][0];
			matNb -= comp.x[i][1];
		}
		matCr /= Nmat;
		matNb /= Nmat;


		/* =========================== *
		 * Solve for parallel tangents *
		 * =========================== */

		rootsolver parallelTangentSolver;
		unsigned int totBadTangents = 0;

		#ifdef _OPENMP
		#pragma omp parallel for private(parallelTangentSolver)
		#endif
		for (int n = 0; n < nodes(initGrid); n++) {
			field_t nx = 0.0;
			vector<field_t>& initGridN = initGrid(n);

			for (int i = NC; i < NC+NP; i++)
				nx += h(initGridN[i]);

			if (nx < 0.01) { // pure gamma
				initGridN[0] += matCr;
				initGridN[1] += matNb;
			}

			// Initialize compositions in a manner compatible with OpenMP and MPI parallelization
			guessGamma(initGridN);
			guessDelta(initGridN);
			guessLaves(initGridN);

			double res = parallelTangentSolver.solve(initGridN);

			if (res>root_tol) {
				// Invalid roots: substitute guesses.
				#ifdef _OPENMP
				#pragma omp atomic
				#endif
				totBadTangents++;

				initGridN[fields(initGrid)-1] = res;
			} else {
				initGridN[fields(initGrid)-1] = 0.0;
			}
		}

		ghostswap(initGrid);

		#ifdef MPI_VERSION
		MPI::COMM_WORLD.Barrier();
		unsigned int myBad(totBadTangents);
		MPI::COMM_WORLD.Reduce(&myBad, &totBadTangents, 1, MPI_UNSIGNED, MPI_SUM, 0);
		MPI::COMM_WORLD.Barrier();
		double mydt(dt);
		MPI::COMM_WORLD.Allreduce(&mydt, &dt, 1, MPI_DOUBLE, MPI_MIN);
		#endif

		vector<double> summary = summarize_fields(initGrid);
		double energy = summarize_energy(initGrid);

		if (rank == 0) {
			fprintf(cfile, "%9s\t%9s\t%9s\t%9s\t%9s\t%9s\t%9s\t%9s\t%9s\t%9s\n",
			"ideal", "timestep", "x_Cr", "x_Nb", "gamma", "delta", "Laves", "bad_roots", "free_energy", "velocity");
			fprintf(cfile, "%9g\t%9g\t%9g\t%9g\t%9g\t%9g\t%9g\t%9u\t%9g\t%9g\n",
			dt, dt, summary[0], summary[1], summary[2], summary[3], summary[4], totBadTangents, energy, 0.0);

			printf("%9s %9s %9s %9s %9s %9s\n",
			"x_Cr", "x_Nb", "x_Ni", " p_g", " p_d", "p_l");
			printf("%9g %9g %9g %9g %9g %9g\n",
			summary[0], summary[1], 1.0-summary[0]-summary[1], summary[2], summary[3], summary[4]);
		}

		#ifndef NDEBUG
		// Output compositions where rootsolver failed (and succeeded)
		std::set<std::vector<short> > badpoints;
		std::set<std::vector<short> > gudpoints;
		for (int n = 0; n < nodes(initGrid); n++) {
			if (initGrid(n)[fields(initGrid)-1] > root_tol) {
				std::vector<short> point(int(NC*(NP+1)), short(0));
				for (unsigned int i = 0; i < point.size(); i++)
					point[i] = short(10000 * initGrid(n)[NC+NP+i]);
				badpoints.insert(point);
			} else {
				std::vector<short> point(int(NC*(NP+1)), short(0));
				for (unsigned int i = 0; i < point.size(); i++)
					point[i] = short(10000 * initGrid(n)[NC+NP+i]);
				gudpoints.insert(point);
			}
		}

		FILE* badfile = NULL;
		FILE* gudfile = NULL;
		if (rank == 0) {
			badfile = fopen("badroots.log", "w"); // old results will be deleted
			gudfile = fopen("gudroots.log", "w"); // old results will be deleted
		} else {
			badfile = fopen("badroots.log", "a"); // new results will be appended
			gudfile = fopen("gudroots.log", "a"); // new results will be appended
		}

		#ifdef MPI_VERSION
		for (int r = 0; r < MPI::COMM_WORLD.Get_size(); r++) {
			MPI::COMM_WORLD.Barrier();
			if (rank == r) {
		#endif
				for (std::set<std::vector<short> >::const_iterator i = badpoints.begin(); i != badpoints.end(); i++)
					fprintf(badfile, "%f,%f,%f,%f,%f,%f\n",
					(*i)[0]/10000.0, (*i)[1]/10000.0, (*i)[2]/10000.0, (*i)[3]/10000.0, (*i)[4]/10000.0, (*i)[5]/10000.0);
		#ifdef MPI_VERSION
			}
			MPI::COMM_WORLD.Barrier();
		}
		#endif

		#ifdef MPI_VERSION
		for (int r = 0; r < MPI::COMM_WORLD.Get_size(); r++) {
			MPI::COMM_WORLD.Barrier();
			if (rank == r) {
		#endif
				for (std::set<std::vector<short> >::const_iterator i = gudpoints.begin(); i != gudpoints.end(); i++)
					fprintf(gudfile, "%f,%f,%f,%f,%f,%f\n",
					(*i)[0]/10000.0, (*i)[1]/10000.0, (*i)[2]/10000.0, (*i)[3]/10000.0, (*i)[4]/10000.0, (*i)[5]/10000.0);
		#ifdef MPI_VERSION
			}
			MPI::COMM_WORLD.Barrier();
		}
		#endif

		fclose(badfile);
		fclose(gudfile);
		#endif

		output(initGrid,filename);




	} else if (dim==2) {




		// Construct grid
		const int Nx = 320; // divisible by 12 and 64
		const int Ny = 192;
		double dV = 1.0;
		double Ntot = 1.0;
		GRID2D initGrid(NC+NP+NC*(NP+1)+1, -Nx/2, Nx/2, -Ny/2, Ny/2);
		for (int d = 0; d < dim; d++) {
			dx(initGrid,d)=meshres;
			dV *= meshres;
			Ntot *= g1(initGrid, d) - g0(initGrid, d);
			if (useNeumann) {
				if (x0(initGrid, d) == g0(initGrid, d))
					b0(initGrid, d) = Neumann;
				if (x1(initGrid, d) == g1(initGrid, d))
					b1(initGrid, d) = Neumann;
			}
		}

		// Precipitate radii: minimum for thermodynamic stability is 7.5 nm,
		//                    minimum for numerical stability is 14*dx (due to interface width).
		const field_t rPrecip[NP] = {5.0*7.5e-9 / dx(initGrid,0),  // delta
		                             5.0*7.5e-9 / dx(initGrid,0)}; // Laves


		// Sanity check on system size and  particle spacing
		if (rank == 0)
			std::cout << "Timestep dt=" << dt << ". Linear stability limits: dtp=" << dtp << " (transformation-limited), dtc="<< dtc << " (diffusion-limited)." << std::endl;

		for (int i = 0; i < NP; i++) {
			if (rPrecip[i] > Ny/2)
				std::cerr << "Warning: domain too small to accommodate phase " << i << ", expand beyond " << 2.0*rPrecip[i] << " pixels." << std::endl;
		}

		// Zero initial condition
		const vector<field_t> blank(fields(initGrid), 0);
		#ifdef _OPENMP
		#pragma omp parallel for
		#endif
		for (int n = 0; n < nodes(initGrid); n++) {
			initGrid(n) = blank;
		}

		// Initialize matrix (gamma phase): bell curve along x, each stripe in y is identical (with small fluctuations)
		Composition comp;
		field_t matCr = 0.0;
		field_t matNb = 0.0;
		//comp += enrichMatrix(initGrid, bell[0], bell[1]);

		if (0) {
			/* ================================================ *
			 * Multi-precipitate particle test configuration    *
			 *                                                  *
			 * Seed a 1.0 um x 0.25 um domain with N particles  *
			 * test full numerical and physical model           *
			 * ================================================ */

			std::vector<vector<int> > seeds;
			vector<int> seed(dim+2, 0);
			unsigned int nprcp = std::pow(2, std::ceil(std::log2(96*Nx*Ny/(768*192))));

			#ifdef MPI_VERSION
			int np = MPI::COMM_WORLD.Get_size();
			nprcp = std::max(1, int(nprcp / np));
			#endif
			unsigned int attempts = 0;

			// Generate non-overlapping seeds with random size and phase
			while (seeds.size() < nprcp && attempts < 500 * nprcp) {
				// Set precipitate ID between [0, NP+1)
				double rnd = unidist(mtrand);
				seed[dim] = std::floor(rnd * (NP + 1));

				// Set precipitate radius
				rnd = unidist(mtrand);
				seed[dim+1] = std::floor((0.525 + rnd) * rPrecip[seed[dim]]);

				// Set precipitate origin
				for (int d = 0; d < dim; d++) {
					rnd = unidist(mtrand);
					seed[d] = x0(initGrid, d) + seed[dim+1] + int(rnd * (x1(initGrid, d) - x0(initGrid, d) - 2 * seed[dim+1]));
				}

				bool clearSpace = true;
				for (size_t i = 0; clearSpace && i < seeds.size(); i++) {
					double r = 0;
					for (int d = 0; d < dim; d++)
						r += pow(seeds[i][d] - seed[d], 2);
					if (std::sqrt(r) < seed[dim+1] + seeds[i][dim+1])
						clearSpace = false;
				}

				if (clearSpace)
					seeds.push_back(seed);

				attempts++;
			}

			for (size_t i = 0; i < seeds.size(); i++) {
				vector<int> origin(dim, 0);
				for (int d = 0; d < dim; d++)
					origin[d] = seeds[i][d];
				int pid = seeds[i][dim];
				assert(pid < NP+1);

				// Embed this precipitate
				comp += embedParticle(initGrid, origin, NC+pid, seeds[i][dim+1], xCr[pid+1], xNb[pid+1]);
			}

			// Synchronize global initial condition parameters
			#ifdef MPI_VERSION
			Composition myComp;
			myComp += comp;
			MPI::COMM_WORLD.Barrier();
			// Caution: Primitive. Will not scale to large MPI systems.
			MPI::COMM_WORLD.Allreduce(&(myComp.N[0]), &(comp.N[0]), NP+1, MPI_INT, MPI_SUM);
			for (int j = 0; j < NP+1; j++) {
				MPI::COMM_WORLD.Barrier();
				MPI::COMM_WORLD.Allreduce(&(myComp.x[j][0]), &(comp.x[j][0]), NC, MPI_DOUBLE, MPI_SUM);
			}
			#endif

			// Initialize matrix to achieve specified system composition
			matCr = Ntot * xCr[0];
			matNb = Ntot * xNb[0];
			double Nmat  = Ntot;
			for (int i = 0; i < NP+1; i++) {
				Nmat  -= comp.N[i];
				matCr -= comp.x[i][0];
				matNb -= comp.x[i][1];
			}
			matCr /= Nmat;
			matNb /= Nmat;

		} else if (0) {
			/* =============================================== *
			 * Two-phase Particle Growth test configuration    *
			 *                                                 *
			 * Seed a 1.0 um x 0.05 um domain with 2 particles *
			 * (one of each secondary phase) in a single row   *
			 * to test competitive growth with Gibbs-Thomson   *
			 * =============================================== */

			vector<int> origin(dim, 0);

			// Set precipitate radius
			int r = std::floor((0.525 + unidist(mtrand)) * rPrecip[0]);

			// Set precipitate separation (min=2r, max=Nx-2r
			int d = unidist(mtrand) * (g1(initGrid, 0) - g0(initGrid, 0) - 2*std::max(r, 70));

			// Set system composition
			double xCr0 = 0.05 + unidist(mtrand) * (0.45 - 0.05);
			double xNb0 = 0.15 + unidist(mtrand) * (0.25 - 0.15);

			#ifdef MPI_VERSION
			MPI::COMM_WORLD.Bcast(&r,    1, MPI_INT,    0);
			MPI::COMM_WORLD.Bcast(&d,    1, MPI_INT,    0);
			MPI::COMM_WORLD.Bcast(&xCr0, 1, MPI_DOUBLE, 0);
			MPI::COMM_WORLD.Bcast(&xNb0, 1, MPI_DOUBLE, 0);
			#endif

			for (int j = 0; j < NP; j++) {
				origin[0] = (j%2==0) ? -d/2 : d/2;
				origin[1] = 0;
				comp += embedParticle(initGrid, origin, j+NC, r, xCr[j+1], xNb[j+1]);
			}

			// Synchronize global initial condition parameters
			#ifdef MPI_VERSION
			Composition myComp;
			myComp += comp;
			MPI::COMM_WORLD.Barrier();
			// Caution: Primitive. Will not scale to large MPI systems.
			MPI::COMM_WORLD.Allreduce(&(myComp.N[0]), &(comp.N[0]), NP+1, MPI_INT, MPI_SUM);
			for (int j = 0; j < NP+1; j++) {
				MPI::COMM_WORLD.Barrier();
				MPI::COMM_WORLD.Allreduce(&(myComp.x[j][0]), &(comp.x[j][0]), NC, MPI_DOUBLE, MPI_SUM);
			}
			#endif

			// Initialize matrix to achieve specified system composition
			matCr = Ntot * xCr0;
			matNb = Ntot * xNb0;
			double Nmat  = Ntot;
			for (int i = 0; i < NP+1; i++) {
				Nmat  -= comp.N[i];
				matCr -= comp.x[i][0];
				matNb -= comp.x[i][1];
			}
			matCr /= Nmat;
			matNb /= Nmat;

		} else if (1) {
			/* =============================================== *
			 * Two-phase Particle Growth test configuration    *
			 *            Small-Precipitate Region             *
			 * Seed a 1.0 um x 0.05 um domain with 2 particles *
			 * (one of each secondary phase) in a single row   *
			 * to test competitive growth with Gibbs-Thomson   *
			 * =============================================== */

			vector<int> origin(dim, 0);

			// Set precipitate radius
			int r = std::floor((0.525 + unidist(mtrand)) * rPrecip[0]);

			// Set precipitate separation (min=2r, max=Nx-2r
			int d = unidist(mtrand) * 16 + (g1(initGrid, 0) - g0(initGrid, 0))/2;

			// Set system composition
			bool withinRange = false;
			double xCr0 = 0.45;
			double xNb0 = 0.07;
			while (!withinRange) {
				xCr0 = 0.43 + unidist(mtrand) * (0.47 - 0.43);
				xNb0 = 0.05 + unidist(mtrand) * (0.09 - 0.05);
				withinRange = (std::pow(xCr0 - 0.45, 2.0) + std::pow(xNb0 - 0.07, 2.0) < std::pow(0.02, 2.0));
			}

			#ifdef MPI_VERSION
			MPI::COMM_WORLD.Barrier();
			MPI::COMM_WORLD.Bcast(&r,    1, MPI_INT,    0);
			MPI::COMM_WORLD.Bcast(&d,    1, MPI_INT,    0);
			MPI::COMM_WORLD.Bcast(&xCr0, 1, MPI_DOUBLE, 0);
			MPI::COMM_WORLD.Bcast(&xNb0, 1, MPI_DOUBLE, 0);
			#endif

			for (int j = 0; j < NP; j++) {
				origin[0] = (j%2==0) ? -d/2 : d/2;
				origin[1] = 0;
				comp += embedParticle(initGrid, origin, j+NC, r, xCr[j+1], xNb[j+1]);
			}

			// Synchronize global initial condition parameters
			#ifdef MPI_VERSION
			Composition myComp;
			myComp += comp;
			MPI::COMM_WORLD.Barrier();
			// Caution: Primitive. Will not scale to large MPI systems.
			MPI::COMM_WORLD.Allreduce(&(myComp.N[0]), &(comp.N[0]), NP+1, MPI_INT, MPI_SUM);
			for (int j = 0; j < NP+1; j++) {
				MPI::COMM_WORLD.Barrier();
				MPI::COMM_WORLD.Allreduce(&(myComp.x[j][0]), &(comp.x[j][0]), NC, MPI_DOUBLE, MPI_SUM);
			}
			#endif

			// Initialize matrix to achieve specified system composition
			matCr = Ntot * xCr0;
			matNb = Ntot * xNb0;
			double Nmat  = Ntot;
			for (int i = 0; i < NP+1; i++) {
				Nmat  -= comp.N[i];
				matCr -= comp.x[i][0];
				matNb -= comp.x[i][1];
			}
			matCr /= Nmat;
			matNb /= Nmat;

		} else {
			if (rank == 0)
				std::cerr<<"Error: specify an initial condition!"<<std::endl;
			MMSP::Abort(-1);
		}


		/* =========================== *
		 * Solve for parallel tangents *
		 * =========================== */

		rootsolver parallelTangentSolver;
		unsigned int totBadTangents = 0;

		#ifdef _OPENMP
		#pragma omp parallel for private(parallelTangentSolver)
		#endif
		for (int n = 0; n < nodes(initGrid); n++) {
			field_t nx = 0.0;
			vector<field_t>& initGridN = initGrid(n);

			for (int i = NC; i < NC+NP; i++)
				nx += h(initGridN[i]);

			if (nx < 0.01) { // pure gamma
				initGridN[0] += matCr;
				initGridN[1] += matNb;
			}

			// Initialize compositions in a manner compatible with OpenMP and MPI parallelization
			guessGamma(initGridN);
			guessDelta(initGridN);
			guessLaves(initGridN);

			double res = parallelTangentSolver.solve(initGridN);

			if (res>root_tol) {
				// Invalid roots: substitute guesses.
				#ifdef _OPENMP
				#pragma omp atomic
				#endif
				totBadTangents++;

				initGridN[fields(initGrid)-1] = res;
			} else {
				initGridN[fields(initGrid)-1] = 0.0;
			}
		}

		ghostswap(initGrid);

		#ifdef MPI_VERSION
		MPI::COMM_WORLD.Barrier();
		unsigned int myBad(totBadTangents);
		MPI::COMM_WORLD.Reduce(&myBad, &totBadTangents, 1, MPI_UNSIGNED, MPI_SUM, 0);
		#endif

		vector<double> summary = summarize_fields(initGrid);
		double energy = summarize_energy(initGrid);

		if (rank == 0) {
			fprintf(cfile, "%9s\t%9s\t%9s\t%9s\t%9s\t%9s\t%9s\t%9s\t%9s\t%9s\n",
			"ideal", "timestep", "x_Cr", "x_Nb", "gamma", "delta", "Laves", "bad_roots", "free_energy", "velocity");
			fprintf(cfile, "%9g\t%9g\t%9g\t%9g\t%9g\t%9g\t%9g\t%9u\t%9g\t%9g\n",
			dt, dt, summary[0], summary[1], summary[2], summary[3], summary[4], totBadTangents, energy, 0.0);

			printf("%9s %9s %9s %9s %9s %9s\n",
			"x_Cr", "x_Nb", "x_Ni", " p_g", " p_d", " p_l");
			printf("%9g %9g %9g %9g %9g %9g\n",
			summary[0], summary[1], 1.0-summary[0]-summary[1], summary[2], summary[3], summary[4]);
		}

		#ifndef NDEBUG
		// Output compositions where rootsolver failed (and succeeded)
		std::set<std::vector<short> > badpoints;
		std::set<std::vector<short> > gudpoints;
		for (int n = 0; n < nodes(initGrid); n++) {
			if (initGrid(n)[fields(initGrid)-1] > root_tol) {
				std::vector<short> point(int(NC*(NP+1)), short(0));
				for (unsigned int i = 0; i < point.size(); i++)
					point[i] = short(10000 * initGrid(n)[NC+NP+i]);
				badpoints.insert(point);
			} else {
				std::vector<short> point(int(NC*(NP+1)), short(0));
				for (unsigned int i = 0; i < point.size(); i++)
					point[i] = short(10000 * initGrid(n)[NC+NP+i]);
				gudpoints.insert(point);
			}
		}

		FILE* badfile = NULL;
		FILE* gudfile = NULL;
		if (rank == 0) {
			badfile = fopen("badroots.log", "w"); // old results will be deleted
			gudfile = fopen("gudroots.log", "w"); // old results will be deleted
		} else {
			badfile = fopen("badroots.log", "a"); // new results will be appended
			gudfile = fopen("gudroots.log", "a"); // new results will be appended
		}

		#ifdef MPI_VERSION
		for (int r = 0; r < MPI::COMM_WORLD.Get_size(); r++) {
			MPI::COMM_WORLD.Barrier();
			if (rank == r) {
		#endif
				for (std::set<std::vector<short> >::const_iterator i = badpoints.begin(); i != badpoints.end(); i++)
					fprintf(badfile, "%f,%f,%f,%f,%f,%f\n",
					(*i)[0]/10000.0, (*i)[1]/10000.0, (*i)[2]/10000.0, (*i)[3]/10000.0, (*i)[4]/10000.0, (*i)[5]/10000.0);
		#ifdef MPI_VERSION
			}
			MPI::COMM_WORLD.Barrier();
		}
		#endif

		#ifdef MPI_VERSION
		for (int r = 0; r < MPI::COMM_WORLD.Get_size(); r++) {
			MPI::COMM_WORLD.Barrier();
			if (rank == r) {
		#endif
				for (std::set<std::vector<short> >::const_iterator i = gudpoints.begin(); i != gudpoints.end(); i++)
					fprintf(gudfile, "%f,%f,%f,%f,%f,%f\n",
					(*i)[0]/10000.0, (*i)[1]/10000.0, (*i)[2]/10000.0, (*i)[3]/10000.0, (*i)[4]/10000.0, (*i)[5]/10000.0);
		#ifdef MPI_VERSION
			}
			MPI::COMM_WORLD.Barrier();
		}
		#endif

		fclose(badfile);
		fclose(gudfile);
		#endif

		output(initGrid,filename);



	} else
		std::cerr << "Error: " << dim << "-dimensional grids unsupported." << std::endl;

	if (rank == 0)
		fclose(cfile);

}





template <int dim, typename T> void update(grid<dim,vector<T> >& oldGrid, int steps)
{
	int rank = 0;
	#ifdef MPI_VERSION
	rank = MPI::COMM_WORLD.Get_rank();
	#endif

	grid<dim,vector<T> > newGrid(oldGrid);

	const double dtp = (meshres*meshres)/(2.0 * dim * Lmob[0]*kappa[0]); // transformation-limited timestep
	const double dtc = (meshres*meshres)/(2.0 * dim * std::max(D_Cr[0], D_Nb[1])); // diffusion-limited timestep
	const double dt = LinStab * std::min(dtp, dtc);

	double dV = 1.0;
	double Ntot = 1.0;
	for (int d = 0; d < dim; d++) {
		dx(oldGrid,d) = meshres;
		dx(newGrid,d) = meshres;
		dV *= dx(oldGrid,d);
		Ntot *= double(g1(oldGrid, d) - g0(oldGrid, d));
		if (useNeumann) {
			if (x0(oldGrid, d) == g0(oldGrid, d)) {
				b0(oldGrid, d) = Neumann;
				b0(newGrid, d) = Neumann;
			}
			if (x1(oldGrid, d) == g1(oldGrid, d)) {
				b1(oldGrid, d) = Neumann;
				b1(newGrid, d) = Neumann;
			}
		}
	}

	ghostswap(oldGrid);

	std::ofstream cfile;
	if (rank == 0)
		cfile.open("c.log", std::ofstream::out | std::ofstream::app); // new results will be appended


	double current_time = 0.0;
	static double current_dt = dt;
	static int logcount = 1;

	const int logstep = std::min(1000, steps); // steps between logging status
	const T advectionlimit = 0.1 * meshres;

	T velocity_range[2] = {1.0, -1.0};

	std::stringstream ostr;

	#ifdef ADAPTIVE_TIMESTEPS
	// reference values for adaptive timestepper
	const double run_time = dt * steps;
	const double timelimit = 4.0 * LinStab * std::min(dtp, dtc) / dim;
	const T scaleup = 1.000001; // how fast dt will rise when stable
	const T scaledn = 0.9;      // how fast dt will fall when unstable

	#ifdef MPI_VERSION
	double mydt(current_dt);
	MPI::COMM_WORLD.Allreduce(&mydt, &current_dt, 1, MPI_DOUBLE, MPI_MIN);
	#endif

	if (rank == 0)
		print_progress(0, steps);

	while (current_time < run_time && current_dt > 0.0) {

		/* Partial timestep trips up parallel jobs.
		current_dt = std::min(current_dt, run_time - current_time);
		*/

	#else
	for (int step = 0; step < steps; step++) {

		if (rank == 0)
			print_progress(step, steps);

	#endif

		rootsolver parallelTangentSolver;
		unsigned int totBadTangents = 0;

		#ifdef _OPENMP
		#pragma omp parallel for private(parallelTangentSolver)
		#endif
		for (int n = 0; n < nodes(oldGrid); n++) {
			/* ============================================== *
			 * Point-wise kernel for parallel PDE integration *
			 * ============================================== */

			vector<int> x = position(oldGrid,n);
			const vector<T>& oldGridN = oldGrid(n);
			vector<T>& newGridN = newGrid(n);


			/* ================= *
			 * Collect Constants *
			 * ================= */

			const T PhaseEnergy[NP+1] = {g_del(oldGridN[2*NC+NP],  oldGridN[2*NC+NP+1]),
			                             g_lav(oldGridN[3*NC+NP],  oldGridN[3*NC+NP+1]),
			                             g_gam(oldGridN[  NC+NP],  oldGridN[  NC+NP+1]) // matrix phase last
			                            };

			// Diffusion potential in matrix (will equal chemical potential at equilibrium)
			const field_t dgGdxCr = dg_gam_dxCr(oldGridN[NC+NP], oldGridN[NC+NP+1]);
			const field_t dgGdxNb = dg_gam_dxNb(oldGridN[NC+NP], oldGridN[NC+NP+1]);
			const T chempot[NC] = {dgGdxCr, dgGdxNb};

			// sum of phase fractions-squared
			T sumPhiSq = 0.0;
			for (int i = NC; i < NC+NP; i++)
				sumPhiSq += oldGridN[i] * oldGridN[i];

			// Laplacians of field variables, including fictitious compositions of matrix phase
			const vector<T> laplac = maskedlaplacian(oldGrid, x, 2*NC+NP);


			/* ============================================= *
			 * Solve the Equation of Motion for Compositions *
			 * ============================================= */

			for (int i = 0; i < NC; i++) // Recall that D_Cr and D_Nb are columns of diffusivity matrix
				newGridN[i] = oldGridN[i] + current_dt * ( D_Cr[i] * laplac[NC+NP  ]
				                                         + D_Nb[i] * laplac[NC+NP+1]);


			/* ======================================== *
			 * Solve the Equation of Motion for Phases  *
			 * ======================================== */

			for (int j = 0; j < NP; j++) {
				const T& phiOld = oldGridN[NC+j];
				assert(phiOld > -1);

				// "Pressure" between matrix and precipitate phase
				T Pressure = PhaseEnergy[NP] - PhaseEnergy[j];
				for (int i = 0; i < NC; i++)
					Pressure -= (oldGridN[NC+NP+i] - oldGridN[NC+NP+i+NC*(j+1)]) * chempot[i];

				// Variational derivatives (scalar minus gradient term in Euler-Lagrange eqn)
				T delF_delPhi = -hprime(phiOld) * Pressure;
				delF_delPhi += 2.0 * omega[j] * phiOld * (phiOld - 1.0) * (2.0*phiOld - 1.0);
				delF_delPhi += 2.0 * alpha * phiOld * (sumPhiSq - phiOld * phiOld);
				delF_delPhi -= kappa[j] * laplac[NC+j];

				newGridN[NC+j] = phiOld - current_dt * Lmob[j] * delF_delPhi;
			}


			/* =========================== *
			 * Solve for parallel tangents *
			 * =========================== */

			// Copy old values as initial guesses
			for (int i = NC+NP; i < NC+NP+NC*(NP+1); i++)
				newGridN[i] = oldGridN[i];

			double res = parallelTangentSolver.solve(newGridN);

			if (res > root_tol) {
				// Invalid roots: substitute guesses.
				#ifdef _OPENMP
				#pragma omp atomic
				#endif
				totBadTangents++;

				/*
				guessGamma(newGridN);
				guessDelta(newGridN);
				guessLaves(newGridN);
				*/
				newGridN[fields(newGrid)-1] = res;
			} else {
				newGridN[fields(newGrid)-1] = 0.0;
			}

			/* ======= *
			 * ~ fin ~ *
			 * ======= */
		}

		swap(oldGrid, newGrid);
		ghostswap(oldGrid);

		// Update timestep based on interfacial velocity. If v==0, there's no interface: march ahead with current dt.
		double interfacialVelocity = maxVelocity(newGrid, current_dt, oldGrid);
		double ideal_dt = (interfacialVelocity>epsilon) ? advectionlimit / interfacialVelocity : current_dt;

		#ifdef MPI_VERSION
		MPI::COMM_WORLD.Barrier();
		double myt(ideal_dt);
		MPI::COMM_WORLD.Allreduce(&myt, &ideal_dt, 1, MPI_DOUBLE, MPI_MIN);
		#endif

		if (current_dt < ideal_dt + epsilon) {
			// Update succeeded: process solution
			current_time += current_dt; // increment before output block

			if (interfacialVelocity < velocity_range[0])
				velocity_range[0] = interfacialVelocity;
			else if (interfacialVelocity > velocity_range[1])
				velocity_range[1] = interfacialVelocity;

			/* ====================================================================== *
			 * Collate summary & diagnostic data in OpenMP- and MPI-compatible manner *
			 * ====================================================================== */

			#ifdef MPI_VERSION
			MPI::COMM_WORLD.Barrier();
			unsigned int myBad(totBadTangents);
			MPI::COMM_WORLD.Reduce(&myBad, &totBadTangents, 1, MPI_UNSIGNED, MPI_SUM, 0);
			#endif

			if (logcount >= logstep) {
				logcount = 0;

				// Warning: placement matters for MPI. Be careful.
				vector<double> summary = summarize_fields(newGrid);
				double energy = summarize_energy(newGrid);

				char buffer[4096];

				if (rank == 0) {
					sprintf(buffer, "%9g\t%9g\t%9g\t%9g\t%9g\t%9g\t%9g\t%9u\t%9g\t%9g\t(%9g\t%9g)\n",
					ideal_dt, current_dt, summary[0], summary[1], summary[2], summary[3], summary[4], totBadTangents, energy, interfacialVelocity, velocity_range[0], velocity_range[1]);
					ostr << buffer;
				}
			}

			#ifdef ADAPTIVE_TIMESTEPS
			current_dt = std::min(std::min(ideal_dt, current_dt*scaleup), timelimit);
			#endif

		} else {
			#ifdef MPI_VERSION
			MPI::COMM_WORLD.Barrier();
			#endif

			// Update failed: solution is unstable
			#ifdef ADAPTIVE_TIMESTEPS

			current_dt *= scaledn;

			swap(oldGrid, newGrid);
			ghostswap(oldGrid);
			#else
			if (rank == 0) {
				std::cerr<<"ERROR: Interface swept more than (dx/"<<meshres/advectionlimit<<"), timestep is too aggressive!"<<std::endl;
				cfile.close();
			}

			MMSP::Abort(-1);
			#endif
		}

		logcount++; // increment after output block

		#ifdef MPI_VERSION
		// Synchronize watches
		MPI::COMM_WORLD.Barrier();
		int myL(logcount);
		MPI::COMM_WORLD.Allreduce(&myL, &logcount, 1, MPI_INT, MPI_MAX);
		MPI::COMM_WORLD.Barrier();
		double mydt(current_dt);
		MPI::COMM_WORLD.Allreduce(&mydt, &current_dt, 1, MPI_DOUBLE, MPI_MIN);
		#endif

		assert(current_dt > epsilon);
	}

	if (rank == 0) {
		cfile << ostr.str(); // write log data to disk
		ostr.str(""); // clear log data
	}

	#ifndef NDEBUG
	// Output compositions where rootsolver failed (and succeeded)
	std::set<std::vector<short> > badpoints;
	std::set<std::vector<short> > gudpoints;
	for (int n = 0; n < nodes(oldGrid); n++) {
		if (newGrid(n)[fields(newGrid)-1] > root_tol) {
			std::vector<short> point(int(NC*(NP+1)), short(0));
			for (unsigned int i = 0; i < point.size(); i++)
				point[i] = short(10000 * oldGrid(n)[NC+NP+i]);
			badpoints.insert(point);
		} else {
			std::vector<short> point(int(NC*(NP+1)), short(0));
			for (unsigned int i = 0; i < point.size(); i++)
				point[i] = short(10000 * oldGrid(n)[NC+NP+i]);
			gudpoints.insert(point);
		}
	}

	FILE* badfile = NULL;
	FILE* gudfile = NULL;
	badfile = fopen("badroots.log", "a"); // new results will be appended
	gudfile = fopen("gudroots.log", "a"); // new results will be appended

	#ifdef MPI_VERSION
	for (int r = 0; r < MPI::COMM_WORLD.Get_size(); r++) {
		MPI::COMM_WORLD.Barrier();
		if (rank == r) {
	#endif
			for (std::set<std::vector<short> >::const_iterator i = badpoints.begin(); i != badpoints.end(); i++)
				fprintf(badfile, "%f,%f,%f,%f,%f,%f\n",
				(*i)[0]/10000.0, (*i)[1]/10000.0, (*i)[2]/10000.0, (*i)[3]/10000.0, (*i)[4]/10000.0, (*i)[5]/10000.0);
	#ifdef MPI_VERSION
		}
		MPI::COMM_WORLD.Barrier();
	}
	#endif

	#ifdef MPI_VERSION
	for (int r = 0; r < MPI::COMM_WORLD.Get_size(); r++) {
		MPI::COMM_WORLD.Barrier();
		if (rank == r) {
	#endif
			for (std::set<std::vector<short> >::const_iterator i = gudpoints.begin(); i != gudpoints.end(); i++)
				fprintf(gudfile, "%f,%f,%f,%f,%f,%f\n",
				(*i)[0]/10000.0, (*i)[1]/10000.0, (*i)[2]/10000.0, (*i)[3]/10000.0, (*i)[4]/10000.0, (*i)[5]/10000.0);
	#ifdef MPI_VERSION
		}
		MPI::COMM_WORLD.Barrier();
	}
	#endif

	fclose(badfile);
	fclose(gudfile);
	#endif

	if (rank == 0) {
		cfile.close();
		#ifdef ADAPTIVE_TIMESTEPS
		print_progress(steps-1, steps); // floating-point comparison misses the endpoint
		#endif
	}
}


} // namespace MMSP





double radius(const MMSP::vector<int>& a, const MMSP::vector<int>& b, const double& dx)
{
	double r = 0.0;
	for (int i = 0; i < a.length() && i < b.length(); i++)
		r += std::pow(a[i] - b[i], 2.0);
	return dx*std::sqrt(r);
}


double bellCurve(double x, double m, double s)
{
	return std::exp(-std::pow(x-m,2.0) / (2.0*s*s));
}


// Initial guesses for gamma, delta, and Laves equilibrium compositions
template<typename T>
void guessGamma(MMSP::vector<T>& GRIDN)
{
	// Coarsely approximate gamma using a line compound with x_Nb = 0.025
	// Interpolate x_Cr from (-0.01, 1.01) into (0.2, 0.4)

	const T& xcr = GRIDN[0];
	const T  xnb = 0.10;

	GRIDN[NC+NP  ] = 0.20 + 0.20/1.02 * (xcr + 0.01);
	GRIDN[NC+NP+1] = xnb;
}


template<typename T>
void guessDelta(MMSP::vector<T>& GRIDN)
{
	// Coarsely approximate delta using a line compound with x_Nb = 0.225
	// Interpolate x_Cr from (-0.01, 1.01) into (0.0, 0.05)

	const T& xcr = GRIDN[0];
	const T  xnb = 0.225;

	GRIDN[2*NC+NP  ] = 0.05/1.02 * (xcr + 0.01);
	GRIDN[2*NC+NP+1] = xnb;
}


template<typename T>
void guessLaves(MMSP::vector<T>& GRIDN)
{
	// Coarsely approximate Laves using a line compound with x_Nb = 30.0%
	// Interpolate x_Cr from (-0.01, 1.01) into (0.30, 0.45)

	const T& xcr = GRIDN[0];
	const T  xnb = 0.25;

	GRIDN[3*NC+NP  ] = 0.30 + 0.15/1.02 * (xcr + 0.01);
	GRIDN[3*NC+NP+1] = xnb;
}


template<int dim, typename T>
Composition enrichMatrix(MMSP::grid<dim,MMSP::vector<T> >& GRID, const double bellCr, const double bellNb)
{
	/* Not the most efficient: to simplify n-dimensional grid compatibility,   *
	 * this function computes the excess compositions at each point. A slight  *
	 * speed-up could be obtained by allocating an array of size Nx, computing *
	 * the excess for each entry, then copying from the array into the grid.   *
	 *                                                                         *
	 * For small grids (e.g., 768 x 192), the speedup is not worth the effort. */

	const int Nx = MMSP::g1(GRID, 0) - MMSP::g0(GRID, 0);
	const double h = MMSP::dx(GRID, 0);
	Composition comp;

	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (int n = 0; n < MMSP::nodes(GRID); n++) {
		MMSP::vector<int> x = MMSP::position(GRID, n);
		T matrixCr = xCr[NP+1] * bellCurve(h*x[0], h*(Nx/2), bellCr); // centerline
		//    matrixCr += xCr[4] * bellCurve(h*x[0], 0,        bellCr); // left wall
		//    matrixCr += xCr[4] * bellCurve(h*x[0], h*Nx,     bellCr); // right wall
		T matrixNb = xNb[NP+1] * bellCurve(h*x[0], h*(Nx/2), bellNb); // centerline
		//    matrixNb += xNb[4] * bellCurve(h*x[0], 0,        bellNb); // left wall
		//    matrixNb += xNb[4] * bellCurve(h*x[0], h*Nx,     bellNb); // right wall

		GRID(n)[0] = matrixCr;
		GRID(n)[1] = matrixNb;

		comp.x[NP][0] += matrixCr;
		comp.x[NP][1] += matrixNb;
	}

	return comp;
}

template<typename T>
void embedParticle(MMSP::vector<T>& v, const int& pid, const T& xCr, const T& xNb, Composition& comp)
{
	v[0] = xCr;
	v[1] = xNb;
	v[pid] = 1.0;
	comp.x[pid-NC][0] += xCr;
	comp.x[pid-NC][1] += xNb;
	comp.N[pid-NC] += 1;
}

template<typename T>
void applyParticleTanh(MMSP::vector<T>& v, const int& pid, const T& tanh)
{
	v[0]  -=  tanh*v[0];
	v[1]  -=  tanh*v[1];
	v[pid] -= tanh*v[pid];
}

template<int dim, typename T>
Composition embedParticle(MMSP::grid<dim,MMSP::vector<T> >& GRID,
                          const MMSP::vector<int>& origin,
                          const int pid,
                          const double rprcp,
                          const T& xCr, const T& xNb)
{
	MMSP::vector<int> x(origin);
	const int R = rprcp;
	Composition comp;

	if (dim==1) {
		for (x[0] = origin[0] - R; x[0] <= origin[0] + R; x[0]++) {
			if (x[0] < x0(GRID) || x[0] >= x1(GRID))
				continue;
			const double r = radius(origin, x, 1);
			if (r < rprcp) // point falls within particle
				embedParticle(GRID(x), pid, xCr, xNb, comp);
		}
	} else if (dim==2) {
		for (x[0] = origin[0] - R; x[0] <= origin[0] + R; x[0]++) {
			if (x[0] < x0(GRID) || x[0] >= x1(GRID))
				continue;
			for (x[1] = origin[1] - R; x[1] <= origin[1] + R; x[1]++) {
				if (x[1] < x0(GRID, 1) || x[1] >= x1(GRID, 1))
					continue;
				const double r = radius(origin, x, 1);
				if (r < rprcp) // point falls within particle
					embedParticle(GRID(x), pid, xCr, xNb, comp);
			}
		}
	} else if (dim==3) {
		for (x[0] = origin[0] - R; x[0] <= origin[0] + R; x[0]++) {
			if (x[0] < x0(GRID) || x[0] >= x1(GRID))
				continue;
			for (x[1] = origin[1] - R; x[1] <= origin[1] + R; x[1]++) {
				if (x[1] < x0(GRID, 1) || x[1] >= x1(GRID, 1))
					continue;
				for (x[2] = origin[2] - R; x[2] <= origin[2] + R; x[2]++) {
					if (x[2] < x0(GRID, 2) || x[2] >= x1(GRID, 2))
						continue;
					const double r = radius(origin, x, 1);
					if (r < rprcp) // point falls within particle
						embedParticle(GRID(x), pid, xCr, xNb, comp);
				}
			}
		}
	}

	if (useTanh) {
		// Create a tanh profile in composition surrounding the precipitate to reduce initial Laplacian
		double del = 3.0 + epsilon;
		if (dim==1) {
			for (x[0] = origin[0] - R - 2*del; x[0] <= origin[0] + R + 2*del; x[0]++) {
				if (x[0] < x0(GRID) || x[0] >= x1(GRID))
					continue;
				const double r = radius(origin, x, 1);
				if (r >= R - del && r < R + del) {
					const T tanhprof = 0.5*(1.0 + std::tanh(double(r - R - del)/(del)));
					applyParticleTanh(GRID(x), pid, tanhprof);
				}
			}
		} else if (dim==2) {
			for (x[0] = origin[0] - R - 2*del; x[0] <= origin[0] + R + 2*del; x[0]++) {
				if (x[0] < x0(GRID) || x[0] >= x1(GRID))
					continue;
				for (x[1] = origin[1] - R - 2*del; x[1] <= origin[1] + R + 2*del; x[1]++) {
					if (x[1] < x0(GRID, 1) || x[1] >= x1(GRID, 1))
						continue;
					const double r = radius(origin, x, 1);
					if (r >= R - del && r < R + del) {
						const T tanhprof = 0.5*(1.0 + std::tanh(double(r - R - del)/(del)));
						applyParticleTanh(GRID(x), pid, tanhprof);
					}
				}
			}
		} else if (dim==3) {
			for (x[0] = origin[0] - R - 2*del; x[0] <= origin[0] + R + 2*del; x[0]++) {
				if (x[0] < x0(GRID) || x[0] >= x1(GRID))
					continue;
				for (x[1] = origin[1] - R - 2*del; x[1] <= origin[1] + R + 2*del; x[1]++) {
					if (x[1] < x0(GRID, 1) || x[1] >= x1(GRID, 1))
						continue;
					for (x[2] = origin[2] - R - 2*del; x[2] <= origin[2] + R + 2*del; x[2]++) {
						if (x[2] < x0(GRID, 2) || x[2] >= x1(GRID, 2))
							continue;
						const double r = radius(origin, x, 1);
						if (r >= R - del && r < R + del) {
							const T tanhprof = 0.5*(1.0 + std::tanh(double(r - R - del)/(del)));
							applyParticleTanh(GRID(x), pid, tanhprof);
						}
					}
				}
			}
		}
	}

	return comp;
}


template<typename T>
void embedStripe(MMSP::vector<T>& v, const int& pid, const T& xCr, const T& xNb, Composition& comp)
{
	v[0] = xCr;
	v[1] = xNb;
	v[pid] = 1.0;
	comp.x[pid-NC][0] += xCr;
	comp.x[pid-NC][1] += xNb;
	comp.N[pid-NC] += 1;
}

template<typename T>
void applyStripeTanh(MMSP::vector<T>& v, const bool lowSide, const int& p, const T& xCr, const T& xNb, const T& tanh)
{
	if (lowSide) {
		v[0] = (1.0 - tanh) * v[0] + tanh * xCr;
		v[1] = (1.0 - tanh) * v[1] + tanh * xNb;
		v[p] = (1.0 - tanh) * v[p] + tanh;
	} else {
		v[0] = (1.0 - tanh) * xCr + tanh * v[0];
		v[1] = (1.0 - tanh) * xNb + tanh * v[1];
		v[p] = (1.0 - tanh)       + tanh * v[p];
	}
}

template<int dim, typename T>
Composition embedStripe(MMSP::grid<dim,MMSP::vector<T> >& GRID,
                        const MMSP::vector<int>& origin,
                        const int pid,
                        const double rprcp,
                        const T& xCr, const T& xNb)
{
	MMSP::vector<int> x(origin);
	const int R(rprcp);
	Composition comp;

	if (dim==1) {
		for (x[0] = origin[0] - R; x[0] < origin[0] + R; x[0]++) {
			if (x[0] < x0(GRID) || x[0] >= x1(GRID))
				continue;
			embedStripe(GRID(x), pid, xCr, xNb, comp);
		}
	} else if (dim==2) {
		for (x[0] = origin[0] - R; x[0] < origin[0] + R; x[0]++) {
			if (x[0] < x0(GRID) || x[0] >= x1(GRID))
				continue;
			for (x[1] = x0(GRID, 1); x[1] < x1(GRID, 1); x[1]++) {
				embedStripe(GRID(x), pid, xCr, xNb, comp);
			}
		}
	} else if (dim==3) {
		for (x[0] = origin[0] - R; x[0] < origin[0] + R; x[0]++) {
			if (x[0] < x0(GRID) || x[0] >= x1(GRID))
				continue;
			for (x[1] = x0(GRID, 1); x[1] < x1(GRID, 1); x[1]++) {
				for (x[2] = x0(GRID, 2); x[2] < x1(GRID, 2); x[2]++) {
					embedStripe(GRID(x), pid, xCr, xNb, comp);
				}
			}
		}
	}

	if (useTanh) {
		// Create a tanh profile in composition surrounding the precipitate to reduce initial Laplacian
		T del = 4.3875e-9 / meshres; // empirically determined tanh profile thickness
		if (dim==1) {
			for (x[0] = origin[0] - R - 2*del; x[0] < origin[0] - R; x[0]++) {
				if (x[0] < x0(GRID) || x[0] >= x1(GRID))
					continue;
				const T tanhprof = 0.5*(1.0 + std::tanh(double(x[0] - origin[0] + R + del)/(del)));
				applyStripeTanh(GRID(x), true, pid, xCr, xNb, tanhprof);
			}
		} else if (dim==2) {
			for (x[0] = origin[0] - R - 2*del; x[0] < origin[0] - R; x[0]++) {
				if (x[0] < x0(GRID) || x[0] >= x1(GRID))
					continue;
				for (x[1] = x0(GRID, 1); x[1] < x1(GRID, 1); x[1]++) {
					const T tanhprof = 0.5*(1.0 + std::tanh(double(x[0] - origin[0] + R + del)/(del)));
					applyStripeTanh(GRID(x), true, pid, xCr, xNb, tanhprof);
				}
			}
		} else if (dim==3) {
			for (x[0] = origin[0] - R - 2*del; x[0] < origin[0] - R; x[0]++) {
				if (x[0] < x0(GRID) || x[0] >= x1(GRID))
					continue;
				for (x[1] = x0(GRID, 1); x[1] < x1(GRID, 1); x[1]++) {
					for (x[2] = x0(GRID, 2); x[2] < x1(GRID, 2); x[2]++) {
						const T tanhprof = 0.5*(1.0 + std::tanh(double(x[0] - origin[0] + R + del)/(del)));
						applyStripeTanh(GRID(x), true, pid, xCr, xNb, tanhprof);
					}
				}
			}
		}
		if (dim==1) {
			for (x[0] = origin[0] + R; x[0] < origin[0] + R + 2*del; x[0]++) {
				if (x[0] < x0(GRID) || x[0] >= x1(GRID))
					continue;
				const T tanhprof = 0.5*(1.0 + std::tanh(double(x[0] - origin[0] - R - del)/(del)));
				applyStripeTanh(GRID(x), false, pid, xCr, xNb, tanhprof);
			}
		} else if (dim==2) {
			for (x[0] = origin[0] + R; x[0] < origin[0] + R + 2*del; x[0]++) {
				if (x[0] < x0(GRID) || x[0] >= x1(GRID))
					continue;
				for (x[1] = x0(GRID, 1); x[1] < x1(GRID, 1); x[1]++) {
					const T tanhprof = 0.5*(1.0 + std::tanh(double(x[0] - origin[0] - R - del)/(del)));
					applyStripeTanh(GRID(x), false, pid, xCr, xNb, tanhprof);
				}
			}
		} else if (dim==3) {
			for (x[0] = origin[0] + R; x[0] < origin[0] + R + 2*del; x[0]++) {
				if (x[0] < x0(GRID) || x[0] >= x1(GRID))
					continue;
				for (x[1] = x0(GRID, 1); x[1] < x1(GRID, 1); x[1]++) {
					for (x[2] = x0(GRID, 2); x[2] < x1(GRID, 2); x[2]++) {
						const T tanhprof = 0.5*(1.0 + std::tanh(double(x[0] - origin[0] - R - del)/(del)));
						applyStripeTanh(GRID(x), false, pid, xCr, xNb, tanhprof);
					}
				}
			}
		}
	}

	return comp;
}


template <typename T>
T gibbs(const MMSP::vector<T>& v)
{
	const T h_del = h(v[NC  ]);
	const T h_lav = h(v[NC+1]);
	const T h_gam = 1.0 - h_del - h_lav;

	MMSP::vector<T> vsq(NP);

	for (int i = 0; i < NP; i++)
		vsq[i] = v[NC+i]*v[NC+i];

	T g  = h_gam * g_gam(v[  NC+NP], v[  NC+NP+1]);
	  g += h_del * g_del(v[2*NC+NP], v[2*NC+NP+1]);
	  g += h_lav * g_lav(v[3*NC+NP], v[3*NC+NP+1]);

	for (int i = 0; i < NP; i++)
		g += omega[i] * vsq[i] * pow(1.0 - v[NC+i], 2);

	// Trijunction penalty
	for (int i = 0; i < NP-1; i++)
		for (int j = i+1; j < NP; j++)
			g += 2.0 * alpha * vsq[i] *vsq[j];

	return g;
}


template <int dim, typename T>
MMSP::vector<T> maskedgradient(const MMSP::grid<dim,MMSP::vector<T> >& GRID, const MMSP::vector<int>& x, const int N)
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


template <int dim, typename T>
MMSP::vector<T> maskedlaplacian(const MMSP::grid<dim,MMSP::vector<T> >& GRID, const MMSP::vector<int>& x, const int N)
{
	// Compute Laplacian of first N fields, ignore the rest
	MMSP::vector<T> laplacian(N, 0.0);
	MMSP::vector<int> s = x;

	const MMSP::vector<T>& y = GRID(x);

	for (int i = 0; i < dim; i++) {
		s[i] += 1;
		const MMSP::vector<T>& yh = GRID(s);
		s[i] -= 2;
		const MMSP::vector<T>& yl = GRID(s);
		s[i] += 1;

		const double weight = 1.0 / (MMSP::dx(GRID, i) * MMSP::dx(GRID, i));
		for (int j = 0; j < N; j++)
			laplacian[j] += weight * (yh[j] - 2.0 * y[j] + yl[j]);
	}
	return laplacian;
}



/* ========================================= *
 * Invoke GSL to solve for parallel tangents *
 * ========================================= */

int parallelTangent_f(const gsl_vector* x, void* params, gsl_vector* f)
{
	/* ======================================================= *
	 * Build Vector of Mass and Chemical Potential Differences *
	 * ======================================================= */

	// Initialize vector
	gsl_vector_set_zero(f);

	// Prepare constants
	const field_t x_Cr = ((struct rparams*) params)->x_Cr;
	const field_t x_Nb = ((struct rparams*) params)->x_Nb;
	const field_t h_del = ((struct rparams*) params)->h_del;
	const field_t h_lav = ((struct rparams*) params)->h_lav;
	const field_t h_gam = 1.0 - h_del - h_lav;

	// Prepare variables
	const field_t C_gam_Cr = gsl_vector_get(x, 0);
	const field_t C_gam_Nb = gsl_vector_get(x, 1);

	const field_t C_del_Cr = gsl_vector_get(x, 2);
	const field_t C_del_Nb = gsl_vector_get(x, 3);

	const field_t C_lav_Cr  = gsl_vector_get(x, 4);
	const field_t C_lav_Nb  = gsl_vector_get(x, 5);


	// Prepare derivatives
	const field_t dgGdxCr = dg_gam_dxCr(C_gam_Cr, C_gam_Nb);
	const field_t dgGdxNb = dg_gam_dxNb(C_gam_Cr, C_gam_Nb);

	const field_t dgDdxCr = dg_del_dxCr(C_del_Cr, C_del_Nb);
	const field_t dgDdxNb = dg_del_dxNb(C_del_Cr, C_del_Nb);

	const field_t dgLdxCr = dg_lav_dxCr(C_lav_Cr, C_lav_Nb);
	const field_t dgLdxNb = dg_lav_dxNb(C_lav_Cr, C_lav_Nb);


	// Update vector
	gsl_vector_set(f, 0, x_Cr - h_gam*C_gam_Cr - h_del*C_del_Cr - h_lav*C_lav_Cr);
	gsl_vector_set(f, 1, x_Nb - h_gam*C_gam_Nb - h_del*C_del_Nb - h_lav*C_lav_Nb);

	gsl_vector_set(f, 2, dgGdxCr - dgDdxCr);
	gsl_vector_set(f, 3, dgGdxNb - dgDdxNb);

	gsl_vector_set(f, 4, dgGdxCr - dgLdxCr);
	gsl_vector_set(f, 5, dgGdxNb - dgLdxNb);


	return GSL_SUCCESS;
}


int parallelTangent_df(const gsl_vector* x, void* params, gsl_matrix* J)
{
	/* ========================================================= *
	 * Build Jacobian of Mass and Chemical Potential Differences *
	 * ========================================================= */

	// Prepare constants
	const double h_del = ((struct rparams*) params)->h_del;
	const double h_lav = ((struct rparams*) params)->h_lav;
	const double h_gam = 1.0 - h_del - h_lav;

	// Prepare variables
	#ifndef PARABOLA
	const double C_gam_Cr = gsl_vector_get(x, 0);
	const double C_gam_Nb = gsl_vector_get(x, 1);

	const double C_del_Cr = gsl_vector_get(x, 2);
	const double C_del_Nb = gsl_vector_get(x, 3);

	const double C_lav_Cr = gsl_vector_get(x, 4);
	const double C_lav_Nb = gsl_vector_get(x, 5);
	#endif

	gsl_matrix_set_zero(J);

	// Conservation of mass (Cr, Nb)
	gsl_matrix_set(J, 0, 0, -h_gam);
	gsl_matrix_set(J, 1, 1, -h_gam);

	gsl_matrix_set(J, 0, 2, -h_del);
	gsl_matrix_set(J, 1, 3, -h_del);

	gsl_matrix_set(J, 0, 4, -h_lav);
	gsl_matrix_set(J, 1, 5, -h_lav);


	// Equal chemical potential involving gamma phase (Cr, Nb, Ni)
	// Cross-derivatives must needs be equal, d2G_dxCrNb == d2G_dxNbCr. Cf. Arfken Sec. 1.9.
	#ifdef PARABOLA
	const double jac_gam_CrCr = d2g_gam_dxCrCr();
	const double jac_gam_CrNb = d2g_gam_dxCrNb();
	const double jac_gam_NbCr = jac_gam_CrNb;
	const double jac_gam_NbNb = d2g_gam_dxNbNb();
	#else
	const double jac_gam_CrCr = d2g_gam_dxCrCr(C_gam_Cr, C_gam_Nb);
	const double jac_gam_CrNb = d2g_gam_dxCrNb(C_gam_Cr, C_gam_Nb);
	const double jac_gam_NbCr = jac_gam_CrNb;
	const double jac_gam_NbNb = d2g_gam_dxNbNb(C_gam_Cr, C_gam_Nb);
	#endif

	gsl_matrix_set(J, 2, 0, jac_gam_CrCr);
	gsl_matrix_set(J, 2, 1, jac_gam_CrNb);
	gsl_matrix_set(J, 3, 0, jac_gam_NbCr);
	gsl_matrix_set(J, 3, 1, jac_gam_NbNb);

	gsl_matrix_set(J, 4, 0, jac_gam_CrCr);
	gsl_matrix_set(J, 4, 1, jac_gam_CrNb);
	gsl_matrix_set(J, 5, 0, jac_gam_NbCr);
	gsl_matrix_set(J, 5, 1, jac_gam_NbNb);


	// Equal chemical potential involving delta phase (Cr, Nb)
	#ifdef PARABOLA
	const double jac_del_CrCr = d2g_del_dxCrCr();
	const double jac_del_CrNb = d2g_del_dxCrNb();
	const double jac_del_NbCr = jac_del_CrNb;
	const double jac_del_NbNb = d2g_del_dxNbNb();
	#elif defined CALPHAD
	const double jac_del_CrCr = d2g_del_dxCrCr(C_del_Cr, C_del_Nb);
	const double jac_del_CrNb = d2g_del_dxCrNb(C_del_Cr, C_del_Nb);
	const double jac_del_NbCr = jac_del_CrNb;
	const double jac_del_NbNb = d2g_del_dxNbNb(C_del_Cr, C_del_Nb);
	#else
	const double jac_del_CrCr = d2g_del_dxCrCr(C_del_Cr);
	const double jac_del_CrNb = d2g_del_dxCrNb(C_del_Nb);
	const double jac_del_NbCr = jac_del_CrNb;
	const double jac_del_NbNb = d2g_del_dxNbNb(C_del_Cr, C_del_Nb);
	#endif

	gsl_matrix_set(J, 2, 2, -jac_del_CrCr);
	gsl_matrix_set(J, 2, 3, -jac_del_CrNb);
	gsl_matrix_set(J, 3, 2, -jac_del_NbCr);
	gsl_matrix_set(J, 3, 3, -jac_del_NbNb);


	// Equal chemical potential involving Laves phase (Nb, Ni)
	#ifdef PARABOLA
	const double jac_lav_CrCr = d2g_lav_dxCrCr();
	const double jac_lav_CrNb = d2g_lav_dxCrNb();
	const double jac_lav_NbCr = jac_lav_CrNb;
	const double jac_lav_NbNb = d2g_lav_dxNbNb();
	#else
	const double jac_lav_CrCr = d2g_lav_dxCrCr(C_lav_Cr, C_lav_Nb);
	const double jac_lav_CrNb = d2g_lav_dxCrNb(C_lav_Cr, C_lav_Nb);
	const double jac_lav_NbCr = jac_lav_CrNb;
	const double jac_lav_NbNb = d2g_lav_dxNbNb(C_lav_Cr, C_lav_Nb);
	#endif

	gsl_matrix_set(J, 4, 4, -jac_lav_CrCr);
	gsl_matrix_set(J, 4, 5, -jac_lav_CrNb);
	gsl_matrix_set(J, 5, 4, -jac_lav_NbCr);
	gsl_matrix_set(J, 5, 5, -jac_lav_NbNb);

	return GSL_SUCCESS;
}


int parallelTangent_fdf(const gsl_vector* x, void* params, gsl_vector* f, gsl_matrix* J)
{
	parallelTangent_f( x, params, f);
	parallelTangent_df(x, params, J);

	return GSL_SUCCESS;
}


rootsolver::rootsolver() :
	n(NC*(NP+1)), // one equation per component per phase, including the matrix but not Ni
	maxiter(root_max_iter),
	tolerance(root_tol)
{
	x = gsl_vector_alloc(n);

	/* Choose the multidimensional root finding algorithm.
	 * Do the math and specify the Jacobian if at all possible. Consult the GSL manual for details:
	 * https://www.gnu.org/software/gsl/manual/html_node/Multidimensional-Root_002dFinding.html
	 *
	 * If GSL finds the matrix to be singular, select a hybrid algorithm, then consult a numerical
	 * methods reference (human or paper) to get your system of equations sorted.
	 *
	 * Available algorithms are, in order of *decreasing* efficiency:
	 * 1. gsl_multiroot_fdfsolver_hybridsj
	 * 2. gsl_multiroot_fdfsolver_hybridj
	 * 3. gsl_multiroot_fdfsolver_newton
	 * 4. gsl_multiroot_fdfsolver_gnewton
	 */
	algorithm = gsl_multiroot_fdfsolver_hybridsj;
	solver = gsl_multiroot_fdfsolver_alloc(algorithm, n);
	mrf = {&parallelTangent_f, &parallelTangent_df, &parallelTangent_fdf, n, &par};
}


template<typename T>
double rootsolver::solve(MMSP::vector<T>& GRIDN)
{
	int status;
	size_t iter = 0;

	par.x_Cr = GRIDN[0];
	par.x_Nb = GRIDN[1];

	par.h_del = h(GRIDN[NC  ]);
	par.h_lav = h(GRIDN[NC+1]);


	// copy initial guesses from grid
	for (int i = 0; i < int(n); i++)
		gsl_vector_set(x, i, static_cast<double>(GRIDN[NC+NP+i]));

	gsl_multiroot_fdfsolver_set(solver, &mrf, x);

	do {
		iter++;

		status = gsl_multiroot_fdfsolver_iterate(solver);
		if (status) // solver either converged or got stuck
			break;

		status = gsl_multiroot_test_residual(solver->f, tolerance);
	} while (status == GSL_CONTINUE && iter < maxiter);

	double residual = gsl_blas_dnrm2(solver->f);

	if (status == GSL_SUCCESS)
		for (int i = 0; i < int(n); i++)
			GRIDN[NC+NP+i] = static_cast<T>(gsl_vector_get(solver->x, i));

	return residual;
}


rootsolver::~rootsolver()
{
	gsl_multiroot_fdfsolver_free(solver);
	gsl_vector_free(x);
}



template<int dim,class T>
T maxVelocity(MMSP::grid<dim,MMSP::vector<T> > const & oldGrid, const double& dt,
              MMSP::grid<dim,MMSP::vector<T> > const & newGrid)
{
	double vmax = 0.0;

	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (int n = 0; n<MMSP::nodes(newGrid); n++) {

		MMSP::vector<int> x = MMSP::position(newGrid,n);

		const MMSP::vector<T>& oldGridN = oldGrid(n);
		MMSP::vector<T>& newGridN = newGrid(n);

		T myVel = 0.0;

		for (int i = 0; i < NP; i++) {
			const T newPhaseFrac = h(newGridN[i+NC]);
			if (newPhaseFrac > 0.4 && newPhaseFrac < 0.6) {
				const MMSP::vector<T> gradPhi = maskedgradient(newGrid, x, i+NC);
				const T magGrad = std::sqrt(gradPhi * gradPhi);
				if (magGrad > epsilon) {
					const T oldPhaseFrac = h(oldGridN[i+NC]);
					T dphidt = std::fabs(newPhaseFrac - oldPhaseFrac) / dt;
					T v = (dphidt > epsilon) ? dphidt / magGrad : 0.0;
					myVel = std::max(myVel, v);
				}
			}
		}

		#ifdef _OPENMP
		#pragma omp critical (critVel)
		{
		#endif
		vmax = std::max(vmax, myVel);
		#ifdef _OPENMP
		}
		#endif
	}

	#ifdef MPI_VERSION
	MPI::COMM_WORLD.Barrier();
	double temp(vmax);
	MPI::COMM_WORLD.Allreduce(&temp, &vmax, 1, MPI_DOUBLE, MPI_MAX);
	#endif

	return vmax;
}


template<int dim,class T>
MMSP::vector<double> summarize_fields(MMSP::grid<dim,MMSP::vector<T> > const & GRID)
{
	/* ================================================================== *
	 * Integrate composition and phase fractions over the whole grid      *
	 * to make sure mass is conserved and phase transformations are sane  *
	 * ================================================================== */

	double Ntot = 1.0;
	for (int d = 0; d<dim; d++)
		Ntot *= double(MMSP::g1(GRID, d) - MMSP::g0(GRID, d));

	MMSP::vector<double> summary(NC+NP+1, 0.0);

	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (int n = 0; n<MMSP::nodes(GRID); n++) {
		MMSP::vector<int> x = MMSP::position(GRID,n);
		MMSP::vector<T>& gridN = GRID(n);
		MMSP::vector<double> mySummary(NC+NP+1, 0.0);

		for (int i = 0; i < NC; i++)
			mySummary[i] = gridN[i]; // compositions

		mySummary[NC] = 1.0; // gamma fraction init

		for (int i = 0; i < NP; i++) {
			const T newPhaseFrac = h(gridN[NC+i]);

			mySummary[NC+i+1] = newPhaseFrac;  // secondary phase fraction
			mySummary[NC    ] -= newPhaseFrac; // contributes to gamma phase;
		}

		#ifdef _OPENMP
		#pragma omp critical (critSum)
		{
		#endif
		summary += mySummary;
		#ifdef _OPENMP
		}
		#endif
	}

	for (int i = 0; i < NC+NP+1; i++)
		summary[i] /= Ntot;

	#ifdef MPI_VERSION
	MMSP::vector<double> temp(summary);
	MPI::COMM_WORLD.Reduce(&(temp[0]), &(summary[0]), NC+NP+1, MPI_DOUBLE, MPI_SUM, 0);
	#endif

	return summary;
}

template<int dim,class T>
double summarize_energy(MMSP::grid<dim,MMSP::vector<T> > const & GRID)
{
	/* ============================================================================= *
	 * Integrate free energy over the whole grid to make sure it decreases with time *
	 * ============================================================================= */

	double dV = 1.0;
	for (int d = 0; d<dim; d++)
		dV *= MMSP::dx(GRID, d);

	double energy = 0.0;

	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (int n = 0; n<MMSP::nodes(GRID); n++) {
		MMSP::vector<int> x = MMSP::position(GRID,n);
		MMSP::vector<T>& gridN = GRID(n);

		double myEnergy = dV * gibbs(gridN); // energy density init

		for (int i = 0; i < NP; i++) {
			const MMSP::vector<T> gradPhi = maskedgradient(GRID, x, NC+i);
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
	MPI::COMM_WORLD.Reduce(&temp, &energy, 1, MPI_DOUBLE, MPI_SUM, 0);
	#endif

	return energy;
}

#endif

#include"MMSP.main.hpp"
