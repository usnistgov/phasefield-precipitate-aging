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
#include<cmath>

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
//                        Nominal | phase diagram  | Enriched
//                        gamma   | delta   laves  | gamma (Excess)
const field_t xCr[NP+2] = {0.30,    0.0125, 0.3875,  0.31-0.30};
const field_t xNb[NP+2] = {0.02,    0.2500, 0.2500,  0.13-0.02};

// Define st.dev. of Gaussians for alloying element segregation
//                       Cr        Nb
const double bell[NC] = {150.0e-9, 50.0e-9}; // est. between 80-200 nm from SEM

// Kinetic and model parameters
const double meshres = 5.0e-9; // grid spacing (m)
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
//const field_t Lmob[NP] = {1.92e-12, 1.92e-12, 1.92e-12}; // numerical mobility (m^2/Ns), Xu's numbers

//                        delta  Laves
const field_t sigma[NP] = {1.01, 1.01}; // J/m^2

// Interfacial width
//const field_t omega[NP] = {9.5e8, 9.5e8}; // multiwell height (m^2/Nsm^2), Zhou's numbers
// Note that ifce width was not considered in Zhou's model, but is in vanilla KKS
const field_t width_factor = 2.2;  // 2.2 if interface is [0.1,0.9]; 2.94 if [0.05,0.95]
const field_t ifce_width = 10.0*meshres; // ensure at least 7 points through the interface
const field_t omega[NP] = {3.0 * width_factor * sigma[0] / ifce_width, // delta
                           3.0 * width_factor * sigma[2] / ifce_width  // Laves
                         }; // multiwell height (J/m^3)

// Numerical considerations
const bool useNeumann = true;    // apply zero-flux boundaries (Neumann type)?
const bool tanh_init = false;    // apply tanh profile to initial profile of composition and phase
const field_t x_min = 1.0e-8;    // what to consider zero to avoid log(c) explosions
const double epsilon = 1.0e-14;  // what to consider zero to avoid log(c) explosions

//const field_t devn_tol = 1.0e-6;  // deviation of field parameters above which parallel tangent needs evaluation
const field_t root_tol = 1.0e-4;  // residual tolerance (default is 1e-7)
const int root_max_iter = 50000; // default is 1000, increasing probably won't change anything but your runtime

/*
#ifndef CALPHAD
const field_t LinStab = 1.0 / 1456.875; // threshold of linear stability (von Neumann stability condition)
#else
*/
const field_t LinStab = 1.0 / 5827.508;  // threshold of linear stability (von Neumann stability condition)
/* #endif */

namespace MMSP
{

void generate(int dim, const char* filename)
{
	int rank = 0;
	#ifdef MPI_VERSION
	rank = MPI::COMM_WORLD.Get_rank();
 	#endif

	FILE* cfile = NULL;
	#ifdef ADAPTIVE_TIMESTEPS
	FILE* tfile = NULL;
	#endif

	if (rank == 0) {
		cfile = fopen("c.log", "w"); // existing log will be overwritten
		#ifdef ADAPTIVE_TIMESTEPS
		tfile = fopen("t.log", "w"); // existing log will be overwritten
		#endif
	}

	const double dtp = (meshres*meshres)/(2.0 * dim * Lmob[0]*kappa[0]); // transformation-limited timestep
	const double dtc = (meshres*meshres)/(2.0 * dim * std::max(D_Cr[0], D_Nb[1])); // diffusion-limited timestep
	double dt = LinStab * std::min(dtp, dtc);

	if (dim==1) {
		// Construct grid
		const int Nx = 768; // divisible by 12 and 64
		double dV = 1.0;
		double Ntot = 1.0;
		GRID1D initGrid(14, 0, Nx);
		for (int d = 0; d<dim; d++) {
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

		/*
		const field_t Csstm[NC] = {0.1500, 0.1500}; // gamma-delta system Cr, Nb composition
		const field_t Cprcp[NC] = {0.0125, 0.2500}; // delta precipitate  Cr, Nb composition
		const int mid = 0;    // matrix phase
		const int pid = NC+0; // delta phase

		const field_t Csstm[NC] = {0.3625, 0.1625}; // gamma-Laves system Cr, Nb composition
		const field_t Cprcp[NC] = {0.3625, 0.2750}; // Laves precipitate  Cr, Nb composition
		const int mid = 0;    // matrix phase
		const int pid = NC+1; // Laves phase

		const field_t Csstm[NC] = {0.2500, 0.2500}; // delta-Laves system Cr, Nb composition
		const field_t Cprcp[NC] = {0.0125, 0.2500}; // delta precipitate  Cr, Nb composition
		const int mid = NC+1; // Laves phase
		const int pid = NC+0; // delta phase
		*/

		/* ============================================ *
		 * Two-phase test configuration                 *
		 *                                              *
		 * Seed a 1.0 um domain with a planar interface *
		 * between gamma and precipitate as a simple    *
		 * test of diffusion and phase equations        *
		 * ============================================ */

		/*
		const int Nprcp = Nx / 3;
		const int Nmtrx = Nx - Nprcp;
		const vector<field_t> blank(fields(initGrid), 0.0);

		for (int n= 0; n<nodes(initGrid); n++) {
			vector<int> x = position(initGrid, n);
			vector<field_t>& initGridN = initGrid(n);

			initGridN = blank;

			if (x[0] < Nprcp) {
				// Initialize precipitate with equilibrium composition (from phase diagram)
				initGridN[0] = Cprcp[0];
				initGridN[1] = Cprcp[1];
				initGridN[pid] = 1.0 - x_min;
			} else {
				// Initialize gamma to satisfy system composition
				initGridN[0] = (Csstm[0] * Nx - Cprcp[0] * Nprcp) / Nmtrx;
				initGridN[1] = (Csstm[1] * Nx - Cprcp[1] * Nprcp) / Nmtrx;
				if (mid == pid)
					initGridN[mid] = -1.0 + x_min;
				else if (mid != 0)
					initGridN[mid] = 1.0 - x_min;
			}

		}
		*/


		/* ============================= *
		 * Four-phase test configuration *
		 * ============================= */

		const int Nprcp[NP] = {Nx / 8, Nx / 8}; // grid points per seed
		const int Noff = Nx / 8 + 2 * ifce_width / meshres; // grid points between seeds
		int Nmtrx = Nx; // grid points of matrix phase

		//const field_t Csstm[NC] = {0.3000, 0.1625}; // system Cr, Nb composition
		const field_t Csstm[NC] = {0.20, 0.15}; // system Cr, Nb composition
		const field_t Cprcp[NP][NC] = {{0.0125, 0.2500}, // delta
		                              {0.3625, 0.2750}  // Laves
		                            }; // precipitate  Cr, Nb composition

		field_t matCr = Csstm[0] * Nx;
		field_t matNb = Csstm[1] * Nx;
		for (int pid = 0; pid < NP; pid++) {
			matCr -= Cprcp[pid][0] * Nprcp[pid];
			matNb -= Cprcp[pid][1] * Nprcp[pid];
			Nmtrx -= Nprcp[pid];
		}
		matCr /= Nmtrx;
		matNb /= Nmtrx;

		const vector<field_t> blank(fields(initGrid), 0.0);

		#ifdef _OPENMP
		#pragma omp parallel for
		#endif
		for (int n = 0; n<nodes(initGrid); n++) {
			vector<int> x = position(initGrid, n);
			vector<field_t>& initGridN = initGrid(n);

			initGridN = blank;

			// Initialize gamma to satisfy system composition
			initGridN[0] = matCr;
			initGridN[1] = matNb;

			for (int pid = 0; pid < NP; pid++) {
				const int Nstart = Nx / 4 + pid*Noff;
				if (x[0]>= Nstart && x[0] < Nstart + Nprcp[pid]) {
					// Initialize precipitate with equilibrium composition (from phase diagram)
					initGridN[0] = Cprcp[pid][0];
					initGridN[1] = Cprcp[pid][1];
					initGridN[NC+pid] = 1.0 - x_min;
				}
			}
		}

		unsigned int totBadTangents = 0;

		#ifdef _OPENMP
		#pragma omp parallel for
		#endif
		for (int n = 0; n<nodes(initGrid); n++) {
			vector<field_t>& initGridN = initGrid(n);

			// Initialize compositions in a manner compatible with OpenMP and MPI parallelization
			guessGamma(initGridN);
			guessDelta(initGridN);
			guessLaves(initGridN);

			/* =========================== *
			 * Solve for parallel tangents *
			 * =========================== */

			rootsolver parallelTangentSolver;
			double res = parallelTangentSolver.solve(initGridN);

			if (res>root_tol) {
				// Invalid roots: substitute guesses.

				#ifdef _OPENMP
				#pragma omp critical (iniCrit1)
				#endif
				{
					totBadTangents++;
				}

				guessGamma(initGridN);
				guessDelta(initGridN);
				guessLaves(initGridN);
				initGridN[fields(initGrid)-1] = 1.0;
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

		vector<double> summary = summarize(initGrid, dt, initGrid);

		if (rank == 0) {
			fprintf(cfile, "%9s\t%9s\t%9s\t%9s\t%9s\t%9s\t%9s\t%9s\t%9s\t%9s\n",
			"ideal", "timestep", "x_Cr", "x_Nb", "gamma", "delta", "Laves", "bad_roots", "free_energy", "velocity");
			fprintf(cfile, "%9g\t%9g\t%9g\t%9g\t%9g\t%9g\t%9g\t%9u\t%9g\t%9g\n",
			dt, dt, summary[0], summary[1], summary[2], summary[3], summary[4], totBadTangents, summary[5], summary[6]);

			#ifdef ADAPTIVE_TIMESTEPS
			fprintf(tfile, "%9g\t%9g\t%9g\n", 0.0, 1.0, dt);
			#endif

			printf("%9s %9s %9s %9s %9s %9s\n",
			"x_Cr", "x_Nb", "x_Ni", " p_g", " p_d", "p_l");
			printf("%9g %9g %9g %9g %9g %9g\n",
			summary[0], summary[1], 1.0-summary[0]-summary[1], summary[2], summary[3], summary[4]);
		}

		output(initGrid,filename);




	} else if (dim==2) {




		// Construct grid
		const int Nx = 768; // divisible by 12 and 64
		const int Ny = 192;
		double dV = 1.0;
		double Ntot = 1.0;
		GRID2D initGrid(14, 0, Nx, 0, Ny);
		for (int d = 0; d<dim; d++) {
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
		for (int n = 0; n<nodes(initGrid); n++) {
			vector<field_t>& initGridN = initGrid(n);
			for (int i = NC; i < fields(initGrid); i++)
				initGridN[i] = 0.0;
		}

		// Initialize matrix (gamma phase): bell curve along x, each stripe in y is identical (with small fluctuations)
		Composition comp;
		comp += enrichMatrix(initGrid, bell[0], bell[1]);


		// Seed precipitates: four of each, arranged along the centerline to allow for pairwise coarsening.
		const int xoffset = 16 * (5.0e-9 / meshres); //  80 nm
		const int yoffset = 32 * (5.0e-9 / meshres); // 160 nm
		vector<int> origin(2, 0);

		if (1) {
			/* ================================================ *
			 * Pairwise precipitate particle test configuration *
			 *                                                  *
			 * Seed a 1.0 um x 0.25 um domain with 12 particles *
			 * (four of each secondary phase) in heterogeneous  *
			 * pairs to test full numerical and physical model  *
			 * ================================================ */

			// Initialize delta precipitates
			int j = 0;
			origin[0] = Nx / 2;
			origin[1] = Ny - yoffset + yoffset/2;
			comp += embedParticle(initGrid, origin, j+NC, rPrecip[j], xCr[j+1], xNb[j+1],  1.0 - x_min);
			origin[0] = Nx/2 + xoffset;
			origin[1] = Ny - 5*yoffset + yoffset/2;
			comp += embedParticle(initGrid, origin, j+NC, rPrecip[j], xCr[j+1], xNb[j+1],  1.0 - x_min);
			origin[0] = Nx/2;
			origin[1] = Ny - 3*yoffset + yoffset/2;
			comp += embedParticle(initGrid, origin, j+NC, rPrecip[j], xCr[j+1], xNb[j+1], -1.0 + x_min);
			origin[0] = Nx/2 - xoffset;
			origin[1] = Ny - 6*yoffset + yoffset/2;
			comp += embedParticle(initGrid, origin, j+NC, rPrecip[j], xCr[j+1], xNb[j+1], -1.0 + x_min);

			// Initialize Laves precipitates
			j = 1;
			origin[0] = Nx/2 + xoffset;
			origin[1] = Ny - yoffset + yoffset/2;
			comp += embedParticle(initGrid, origin, j+NC, rPrecip[j], xCr[j+1], xNb[j+1],  1.0 - x_min);
			origin[0] = Nx/2;
			origin[1] = Ny - 4*yoffset + yoffset/2;
			comp += embedParticle(initGrid, origin, j+NC, rPrecip[j], xCr[j+1], xNb[j+1],  1.0 - x_min);
			origin[0] = Nx/2 - xoffset;
			origin[1] = Ny - 2*yoffset + yoffset/2;
			comp += embedParticle(initGrid, origin, j+NC, rPrecip[j], xCr[j+1], xNb[j+1], -1.0 + x_min);
			origin[0] = Nx/2;
			origin[1] = Ny - 6*yoffset + yoffset/2;
			comp += embedParticle(initGrid, origin, j+NC, rPrecip[j], xCr[j+1], xNb[j+1], -1.0 + x_min);
		} else if (0) {
			/* =============================================== *
			 * Two-phase Particle Growth test configuration  *
			 *                                                 *
			 * Seed a 1.0 um x 0.05 um domain with 3 particles *
			 * (one of each secondary phase) in a single row   *
			 * to test competitive growth with Gibbs-Thomson   *
			 * =============================================== */

			// Initialize delta precipitates
			int j = 0;
			origin[0] = Nx / 2;
			origin[1] = Ny / 2;
			comp += embedParticle(initGrid, origin, j+NC, rPrecip[j], xCr[j+1], xNb[j+1],  1.0 - x_min);

			// Initialize Laves precipitates
			j = 1;
			origin[0] = Nx / 2 + xoffset;
			comp += embedParticle(initGrid, origin, j+NC, rPrecip[j], xCr[j+1], xNb[j+1],  1.0 - x_min);
		} else if (0) {
			/* ============================================= *
			 * Two-phase Stripe Growth test configuration  *
			 *                                               *
			 * Seed a 1.0 um x 0.05 um domain with 3 stripes *
			 * (one of each secondary phase) in a single row *
			 * to test competitive growth without curvature  *
			 * ============================================= */

			// Initialize delta stripe
			int j = 0;
			origin[0] = Nx / 2;
			origin[1] = Ny / 2;
			comp += embedStripe(initGrid, origin, j+NC, rPrecip[j], xCr[j+1], xNb[j+1],  1.0 - x_min);

			// Initialize Laves stripe
			j = 1;
			origin[0] = Nx / 2 + xoffset;
			comp += embedStripe(initGrid, origin, j+NC, rPrecip[j], xCr[j+1], xNb[j+1],  1.0 - x_min);

		} else if (0) {
			/* ============================================= *
			 * Two-phase Planar Interface test configuration *
			 *                                               *
			 * Seed a 1.0 um x 0.004 um domain with a planar *
			 * interface between gamma and delta as a simple *
			 * test of diffusion and phase equations         *
			 * ============================================= */

			// Initialize planar interface between gamma and delta

			origin[0] = Nx / 4;
			origin[1] = Ny / 2;
			const field_t delCr = 0.15; // Choose initial composition carefully!
			const field_t delNb = 0.15;
			comp += embedStripe(initGrid, origin, 2, Nx/4, delCr, delNb,  1.0-x_min);
		} else {
			if (rank == 0)
				std::cerr<<"Error: specify an initial condition!"<<std::endl;
			MMSP::Abort(-1);
		}

		// Synchronize global intiial condition parameters
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

		unsigned int totBadTangents = 0;

		#ifdef _OPENMP
		#pragma omp parallel for
		#endif
		for (int n = 0; n<nodes(initGrid); n++) {
			field_t nx = 0.0;
			vector<field_t>& initGridN = initGrid(n);

			for (int i = NC; i < NC+NP; i++)
				nx += h(fabs(initGridN[i]));

			if (1.01 * nx < NP*x_min) { // pure gamma
				initGridN[0] += matCr;
				initGridN[1] += matNb;
			}

			// Initialize compositions in a manner compatible with OpenMP and MPI parallelization
			guessGamma(initGridN);
			guessDelta(initGridN);
			guessLaves(initGridN);


			/* =========================== *
			 * Solve for parallel tangents *
			 * =========================== */

			rootsolver parallelTangentSolver;
			double res = parallelTangentSolver.solve(initGridN);

			if (res>root_tol) {
				// Invalid roots: substitute guesses.
				#ifdef _OPENMP
				#pragma omp critical (iniCrit2)
				#endif
				{
					totBadTangents++;
				}

				guessGamma(initGridN);
				guessDelta(initGridN);
				guessLaves(initGridN);
				initGridN[fields(initGrid)-1] = 1.0;
			}
		}

		ghostswap(initGrid);

		#ifdef MPI_VERSION
		MPI::COMM_WORLD.Barrier();
		unsigned int myBad(totBadTangents);
		MPI::COMM_WORLD.Reduce(&myBad, &totBadTangents, 1, MPI_UNSIGNED, MPI_SUM, 0);
		#endif

		vector<double> summary = summarize(initGrid, dt, initGrid);

		if (rank == 0) {
			fprintf(cfile, "%9s\t%9s\t%9s\t%9s\t%9s\t%9s\t%9s\t%9s\t%9s\t%9s\n",
			"ideal", "timestep", "x_Cr", "x_Nb", "gamma", "delta", "Laves", "bad_roots", "free_energy", "velocity");
			fprintf(cfile, "%9g\t%9g\t%9g\t%9g\t%9g\t%9g\t%9g\t%9u\t%9g\t%9g\n",
			dt, dt, summary[0], summary[1], summary[2], summary[3], summary[4], totBadTangents, summary[5], summary[6]);

			#ifdef ADAPTIVE_TIMESTEPS
			fprintf(tfile, "%9g\t%9g\t%9g\n", 0.0, 1.0, dt);
			#endif

			printf("%9s %9s %9s %9s %9s %9s\n",
			"x_Cr", "x_Nb", "x_Ni", " p_g", " p_d", " p_l");
			printf("%9g %9g %9g %9g %9g %9g\n",
			summary[0], summary[1], 1.0-summary[0]-summary[1], summary[2], summary[3], summary[4]);
		}

		output(initGrid,filename);



	} else
		std::cerr << "Error: " << dim << "-dimensional grids unsupported." << std::endl;

	if (rank == 0) {
		fclose(cfile);
		#ifdef ADAPTIVE_TIMESTEPS
		fclose(tfile);
		#endif
	}

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
	for (int d = 0; d<dim; d++) {
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

	FILE* cfile = NULL;
	#ifdef ADAPTIVE_TIMESTEPS
	FILE* tfile = NULL;
	#endif

	if (rank == 0) {
		cfile = fopen("c.log", "a"); // new results will be appended
		#ifdef ADAPTIVE_TIMESTEPS
		tfile = fopen("t.log", "a"); // new results will be appended
		#endif
	}


	double current_time = 0.0;
	static double current_dt = dt;
	static int logcount = 1;

	const int logstep = std::min(1000, steps); // steps between logging status
	const field_t advectionlimit = 0.1 * meshres;

	field_t velocity_range[2] = {1.0, -1.0};

	#ifdef ADAPTIVE_TIMESTEPS
	// reference values for adaptive timestepper
	const double run_time = dt * steps;
	const double timelimit = 4.0 * LinStab * std::min(dtp, dtc) / dim;
	const field_t scaleup = 1.00001; // how fast will dt rise when stable
	const field_t scaledn = 0.9; // how fast will dt fall when unstable

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
	for (int step = 0; step<steps; step++) {

		if (rank == 0)
			print_progress(step, steps);

	#endif

		unsigned int totBadTangents = 0;

		#ifdef _OPENMP
		#pragma omp parallel for
		#endif
		for (int n = 0; n<nodes(oldGrid); n++) {
			/* ============================================== *
			 * Point-wise kernel for parallel PDE integration *
			 * ============================================== */

			vector<int> x = position(oldGrid,n);
			const vector<T>& oldGridN = oldGrid(n);
			vector<T>& newGridN = newGrid(n);


			/* ================= *
			 * Collect Constants *
			 * ================= */

			const field_t PhaseEnergy[NP+1] = {g_del(oldGridN[NC+NP+2],  oldGridN[NC+NP+3]),
			                                   g_lav(oldGridN[NC+NP+4],  oldGridN[NC+NP+5]),
			                                   g_gam(oldGridN[NC+NP  ],  oldGridN[NC+NP+1])
			                                  };

			// Diffusion potential in matrix (will equal chemical potential at equilibrium)
			const field_t chempot[NC] = {dg_gam_dxCr(oldGridN[NC+NP], oldGridN[NC+NP+1]), dg_gam_dxNb(oldGridN[NC+NP], oldGridN[NC+NP+1])};

			// sum of phase fractions-squared
			field_t sumPhiSq = 0.0;
			for (int i = NC; i < NC+NP; i++)
				sumPhiSq += oldGridN[i] * oldGridN[i];

			// Laplacians of field variables
			const vector<field_t> laplac = maskedlaplacian(oldGrid, x, 7);


			/* ============================================= *
			 * Solve the Equation of Motion for Compositions *
			 * ============================================= */

			for (int i = 0; i < NC; i++) // Recall that D_Cr and D_Nb are columns of diffusivity matrix
				newGridN[i] = oldGridN[i] + current_dt * (D_Cr[i] * laplac[5] + D_Nb[i] * laplac[6]);


			/* ======================================== *
			 * Solve the Equation of Motion for Phases  *
			 * ======================================== */

			for (int j = 0; j < NP; j++) {
				const T& phiOld = oldGridN[j+NC];
				const field_t absPhi = fabs(phiOld);

				// "Pressure" between matrix and precipitate phase
				field_t Pressure = PhaseEnergy[NP] - PhaseEnergy[j];
				for (int i = 0; i < NC; i++)
					Pressure -= (oldGridN[i+5] - oldGridN[i+2*j+7]) * chempot[i];

				// Variational derivatives (scalar minus gradient term in Euler-Lagrange eqn)
				field_t delF_delPhi = -sign(phiOld) * hprime(absPhi) * Pressure;
				delF_delPhi += 2.0 * omega[j] * phiOld * (1.0 - absPhi) * (1.0 - h(absPhi) - sign(phiOld) * phiOld);
				delF_delPhi += 2.0 * alpha * phiOld * (sumPhiSq - phiOld * phiOld);
				delF_delPhi -= kappa[j] * laplac[j+NC];

				newGridN[j+NC] = phiOld - current_dt * Lmob[j] * delF_delPhi;
			}


			/* =========================== *
			 * Solve for parallel tangents *
			 * =========================== */

			// Copy old values as initial guesses
			for (int i = NC+NP; i < fields(newGrid)-1; i++)
				newGridN[i] = oldGridN[i];

			double deviation = 0.0;

			// examine local deviation of the "real" field variables
			for (int i = 0; i < NC+NP; i++)
				deviation += fabs(newGridN[i] - oldGridN[i]);

			// copy deviation into debugging slot
			newGridN[fields(newGrid)-1] = deviation;

			// evaluate parallel tangents if fields have changed
			// Note: This creates a highly imbalanced workload!
			//if (deviation > devn_tol) {
			rootsolver parallelTangentSolver;
			double res = parallelTangentSolver.solve(newGridN);

			if (res>root_tol) {
				// Invalid roots: substitute guesses.
				#ifdef _OPENMP
				#pragma omp critical (updCrit1)
				#endif
				{
					totBadTangents++;
				}
				guessGamma(newGridN);
				guessDelta(newGridN);
				guessLaves(newGridN);
				newGridN[fields(newGrid)-1] = 1.0;
			}
			//}

			/* ======= *
			 * ~ fin ~ *
			 * ======= */
		}

		swap(oldGrid, newGrid);
		ghostswap(oldGrid);

		// Update timestep based on interfacial velocity. If v=0, there's no interface: march ahead with current dt.
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

			vector<double> summary = summarize(newGrid, current_dt, oldGrid);

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

				if (rank == 0)
					fprintf(cfile, "%9g\t%9g\t%9g\t%9g\t%9g\t%9g\t%9g\t%9u\t%9g\t%9g\t(%9g\t%9g)\n",
					ideal_dt, current_dt, summary[0], summary[1], summary[2], summary[3], summary[4], totBadTangents, summary[5], summary[6], velocity_range[0], velocity_range[1]);

				#ifdef ADAPTIVE_TIMESTEPS
				if (rank == 0)
					fprintf(tfile, "%9g\t%9g\t%9g\t%9g\n", interfacialVelocity, std::min(dtp, dtc) / current_dt, ideal_dt, current_dt);
				#endif

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
			if (rank == 0)
				fprintf(tfile, "%9g\t%9g\t%9g\t%9g F\n", interfacialVelocity, std::min(dtp, dtc) / current_dt, ideal_dt, current_dt);

			current_dt *= scaledn;

			swap(oldGrid, newGrid);
			ghostswap(oldGrid);
			#else
			if (rank == 0) {
				std::cerr<<"ERROR: Interface swept more than ("<<meshres/advectionlimit<<")dx, timestep is too aggressive!"<<std::endl;
				fclose(cfile);
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

	// vector<field_t> summary = summarize(newGrid, current_dt, oldGrid);

	if (rank == 0) {
		fclose(cfile);
		#ifdef ADAPTIVE_TIMESTEPS
		fclose(tfile);
		print_progress(steps-1, steps); // floating-point comparison misses the endpoint
		#endif
	}

}


} // namespace MMSP





double radius(const MMSP::vector<int>& a, const MMSP::vector<int>& b, const double& dx)
{
	double r = 0.0;
	for (int i = 0; i < a.length() && i < b.length(); i++)
		r += std::pow(a[i]-b[i],2.0);
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
	// Coarsely approximate gamma using a line compound with x_Nb = 0.015

	const T& xcr = GRIDN[0];
	const T  xnb = 0.015;
	const T  xni = std::max(x_min, 1.0 - xcr - GRIDN[1]);

	GRIDN[NC+NP] = xcr/(xcr + xnb + xni);
	GRIDN[NC+NP+1] = xnb;
}


template<typename T>
void guessDelta(MMSP::vector<T>& GRIDN)
{
	// Coarsely approximate delta using a line compound with x_Ni = 0.75

	const T& xcr = GRIDN[0];
	const T& xnb = GRIDN[1];
	const T  xni = 0.75;

	GRIDN[NC+NP+2] = xcr/(xcr + xnb + xni);
	GRIDN[NC+NP+3] = xnb/(xcr + xnb + xni);
}


template<typename T>
void guessLaves(MMSP::vector<T>& GRIDN)
{
	// Coarsely approximate Laves using a line compound with x_Nb = 30.0%

	const T& xcr = GRIDN[0];
	const T  xnb = 0.30;
	const T  xni = std::max(x_min, 1.0 - xcr - GRIDN[1]);

	GRIDN[NC+NP+4] = xcr/(xcr + xnb + xni);
	GRIDN[NC+NP+5] = xnb;
}


template<int dim,typename T>
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

	for (int n = 0; n<MMSP::nodes(GRID); n++) {
		MMSP::vector<int> x = MMSP::position(GRID, n);
		field_t matrixCr = xCr[NP+1] * bellCurve(h*x[0], h*(Nx/2), bellCr); // centerline
		//    matrixCr += xCr[4] * bellCurve(h*x[0], 0,        bellCr); // left wall
		//    matrixCr += xCr[4] * bellCurve(h*x[0], h*Nx,     bellCr); // right wall
		field_t matrixNb = xNb[NP+1] * bellCurve(h*x[0], h*(Nx/2), bellNb); // centerline
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
Composition embedParticle(MMSP::grid<2,MMSP::vector<T> >& GRID,
                          const MMSP::vector<int>& origin,
                          const int pid,
                          const double rprcp,
                          const T& xCr, const T& xNb,
                          const T phi)
{
	MMSP::vector<int> x(origin);
	const int R = rprcp; //std::max(rdpltCr, rdpltNb);
	Composition comp;

	for (x[0] = origin[0] - R; x[0] <= origin[0] + R; x[0]++) {
		if (x[0] < x0(GRID) || x[0] >= x1(GRID))
			continue;
		for (x[1] = origin[1] - R; x[1] <= origin[1] + R; x[1]++) {
			if (x[1] < y0(GRID) || x[1] >= y1(GRID))
				continue;
			const double r = radius(origin, x, 1);
			if (r < rprcp) { // point falls within particle
				GRID(x)[0] = xCr;
				GRID(x)[1] = xNb;
				GRID(x)[pid] = phi;
				comp.x[pid-NC][0] += xCr;
				comp.x[pid-NC][1] += xNb;
				comp.N[pid-NC] += 1;
			}
		}
	}

	if (tanh_init) {
		// Create a tanh profile in composition surrounding the precipitate to reduce initial Laplacian
		double del = 3.0 + epsilon;
		for (x[0] = origin[0] - R - 2*del; x[0] <= origin[0] + R + 2*del; x[0]++) {
			if (x[0] < x0(GRID) || x[0] >= x1(GRID))
				continue;
			for (x[1] = origin[1] - R - 2*del; x[1] <= origin[1] + R + 2*del; x[1]++) {
				if (x[1] < y0(GRID) || x[1] >= y1(GRID))
					continue;
				const double r = radius(origin, x, 1);
				if (r >= R - del && r < R + del) {
					field_t tanhprof = 0.5*(1.0 + std::tanh(double(r - R - del)/(del)));
					GRID(x)[0]  -=  tanhprof*GRID(x)[0];//  -  GRID(origin)[0]);
					GRID(x)[1]  -=  tanhprof*GRID(x)[1];//  -  GRID(origin)[1]);
					GRID(x)[pid] -= tanhprof*GRID(x)[pid];// - GRID(origin)[pid]);
				}
			}
		}
	}

	return comp;
}


template<typename T>
Composition embedStripe(MMSP::grid<2,MMSP::vector<T> >& GRID,
                        const MMSP::vector<int>& origin,
                        const int pid,
                        const double rprcp,
                        const T& xCr, const T& xNb,
                        const T phi)
{
	MMSP::vector<int> x(origin);
	const int R(rprcp); //std::max(rdpltCr, rdpltNb);
	Composition comp;

	for (x[0] = origin[0] - R; x[0] < origin[0] + R; x[0]++) {
		if (x[0] < x0(GRID) || x[0] >= x1(GRID))
			continue;
		for (x[1] = y0(GRID); x[1] < y1(GRID); x[1]++) {
			GRID(x)[0] = xCr;
			GRID(x)[1] = xNb;
			GRID(x)[pid] = phi;
			comp.x[pid-NC][0] += xCr;
			comp.x[pid-NC][1] += xNb;
			comp.N[pid-NC] += 1;
		}
	}

	if (tanh_init) {
		// Create a tanh profile in composition surrounding the precipitate to reduce initial Laplacian
		field_t del = 4.3875e-9 / meshres; // empirically determined tanh profile thickness
		for (x[0] = origin[0] - R - 2*del; x[0] < origin[0] - R; x[0]++) {
			if (x[0] < x0(GRID) || x[0] >= x1(GRID))
				continue;
			field_t tanhprof = 0.5*(1.0 + std::tanh(double(x[0] - origin[0] + R + del)/(del)));
			for (x[1] = y0(GRID); x[1] < y1(GRID); x[1]++) {
				GRID(x)[0] = GRID(x)[0] - tanhprof*(GRID(x)[0] - xCr);
				GRID(x)[1] = GRID(x)[1] - tanhprof*(GRID(x)[1] - xNb);
				GRID(x)[pid] = GRID(x)[pid] - tanhprof*(GRID(x)[pid] - phi);
			}
		}
		for (x[0] = origin[0] + R; x[0] < origin[0] + R + 2*del; x[0]++) {
			if (x[0] < x0(GRID) || x[0] >= x1(GRID))
				continue;
			field_t tanhprof = 0.5*(1.0 + std::tanh(double(x[0] - origin[0] - R - del)/(del)));
			for (x[1] = y0(GRID); x[1] < y1(GRID); x[1]++) {
				GRID(x)[0] = xCr - tanhprof*(xCr - GRID(x)[0]);
				GRID(x)[1] = xNb - tanhprof*(xNb - GRID(x)[1]);
				GRID(x)[pid] = phi - tanhprof*(phi - GRID(x)[pid]);
			}
		}
	}

	return comp;
}


template <typename T>
T gibbs(const MMSP::vector<T>& v)
{
	const T n_del = h(fabs(v[NC]));
	const T n_lav = h(fabs(v[NC+1]));
	const T n_gam = 1.0 - n_del - n_lav;

	MMSP::vector<T> vsq(NP);

	for (int i = 0; i < NP; i++)
		vsq[i] = v[NC+i]*v[NC+i];

	T g  = n_gam * g_gam(v[NC+NP  ], v[NC+NP+1]);
	  g += n_del * g_del(v[NC+NP+2], v[NC+NP+3]);
	  g += n_lav * g_lav(v[NC+NP+4], v[NC+NP+5]);
	  g += omega[0] * vsq[2-NC] * pow(1.0 - fabs(v[NC]), 2);
	  g += omega[1] * vsq[3-NC] * pow(1.0 - fabs(v[NC+1]), 2);

	// Trijunction penalty
	for (int i = 0; i < NP-1; i++)
		for (int j=i+1; j < NP; j++)
			g += 2.0 * alpha * vsq[i] *vsq[j];

	return g;
}


template <int dim, typename T>
MMSP::vector<T> maskedgradient(const MMSP::grid<dim, MMSP::vector<T> >& GRID, const MMSP::vector<int>& x, const int N)
{
    MMSP::vector<T> gradient(dim);
	MMSP::vector<int> s = x;

	for (int i=0; i<dim; i++) {
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
MMSP::vector<T> maskedlaplacian(const MMSP::grid<dim, MMSP::vector<T> >& GRID, const MMSP::vector<int>& x, const int N)
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
	const field_t n_del = ((struct rparams*) params)->n_del;
	const field_t n_lav = ((struct rparams*) params)->n_lav;
	const field_t n_gam = 1.0 - n_del - n_lav;

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
	gsl_vector_set(f, 0, x_Cr - n_gam*C_gam_Cr - n_del*C_del_Cr - n_lav*C_lav_Cr);
	gsl_vector_set(f, 1, x_Nb - n_gam*C_gam_Nb - n_del*C_del_Nb - n_lav*C_lav_Nb);

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
	const double n_del = ((struct rparams*) params)->n_del;
	const double n_lav = ((struct rparams*) params)->n_lav;
	const double n_gam = 1.0 - n_del - n_lav;

	// Prepare variables
	/* #ifdef CALPHAD */
	const double C_gam_Cr = gsl_vector_get(x, 0);
	const double C_gam_Nb = gsl_vector_get(x, 1);

	const double C_del_Cr = gsl_vector_get(x, 2);
	const double C_del_Nb = gsl_vector_get(x, 3);

	const double C_lav_Cr = gsl_vector_get(x, 4);
	const double C_lav_Nb = gsl_vector_get(x, 5);
	/* #endif */

	gsl_matrix_set_zero(J);

	// Conservation of mass (Cr, Nb)
	gsl_matrix_set(J, 0, 0, -n_gam);
	gsl_matrix_set(J, 1, 1, -n_gam);

	gsl_matrix_set(J, 0, 2, -n_del);
	gsl_matrix_set(J, 1, 3, -n_del);

	gsl_matrix_set(J, 0, 4, -n_lav);
	gsl_matrix_set(J, 1, 5, -n_lav);


	// Equal chemical potential involving gamma phase (Cr, Nb, Ni)
	// Cross-derivatives must needs be equal, d2G_dxCrNb == d2G_dxNbCr. Cf. Arfken Sec. 1.9.
	/*
	#ifndef CALPHAD
	const double jac_gam_CrCr = d2g_gam_dxCrCr();
	const double jac_gam_CrNb = d2g_gam_dxCrNb();
	const double jac_gam_NbCr = jac_gam_CrNb;
	const double jac_gam_NbNb = d2g_gam_dxNbNb();
	#else
	*/
	const double jac_gam_CrCr = d2g_gam_dxCrCr(C_gam_Cr, C_gam_Nb);
	const double jac_gam_CrNb = d2g_gam_dxCrNb(C_gam_Cr, C_gam_Nb);
	const double jac_gam_NbCr = jac_gam_CrNb;
	const double jac_gam_NbNb = d2g_gam_dxNbNb(C_gam_Cr, C_gam_Nb);
	/* #endif */

	gsl_matrix_set(J, 2, 0, jac_gam_CrCr);
	gsl_matrix_set(J, 2, 1, jac_gam_CrNb);
	gsl_matrix_set(J, 3, 0, jac_gam_NbCr);
	gsl_matrix_set(J, 3, 1, jac_gam_NbNb);

	gsl_matrix_set(J, 4, 0, jac_gam_CrCr);
	gsl_matrix_set(J, 4, 1, jac_gam_CrNb);
	gsl_matrix_set(J, 5, 0, jac_gam_NbCr);
	gsl_matrix_set(J, 5, 1, jac_gam_NbNb);


	// Equal chemical potential involving delta phase (Cr, Nb)
	/*
	#ifndef CALPHAD
	const double jac_del_CrCr = d2g_del_dxCrCr();
	const double jac_del_CrNb = d2g_del_dxCrNb();
	const double jac_del_NbCr = jac_del_CrNb;
	const double jac_del_NbNb = d2g_del_dxNbNb();
	#else
	*/
	const double jac_del_CrCr = d2g_del_dxCrCr(C_del_Cr, C_del_Nb);
	const double jac_del_CrNb = d2g_del_dxCrNb(C_del_Cr, C_del_Nb);
	const double jac_del_NbCr = jac_del_CrNb;
	const double jac_del_NbNb = d2g_del_dxNbNb(C_del_Cr, C_del_Nb);
	/* #endif */

	gsl_matrix_set(J, 2, 2, -jac_del_CrCr);
	gsl_matrix_set(J, 2, 3, -jac_del_CrNb);
	gsl_matrix_set(J, 3, 2, -jac_del_NbCr);
	gsl_matrix_set(J, 3, 3, -jac_del_NbNb);


	// Equal chemical potential involving Laves phase (Nb, Ni)
	/*
	#ifndef CALPHAD
	const double jac_lav_CrCr = d2g_lav_dxCrCr();
	const double jac_lav_CrNb = d2g_lav_dxCrNb();
	const double jac_lav_NbCr = jac_lav_CrNb;
	const double jac_lav_NbNb = d2g_lav_dxNbNb();
	#else
	*/
	const double jac_lav_CrCr = d2g_lav_dxCrCr(C_lav_Cr, C_lav_Nb);
	const double jac_lav_CrNb = d2g_lav_dxCrNb(C_lav_Cr, C_lav_Nb);
	const double jac_lav_NbCr = jac_lav_CrNb;
	const double jac_lav_NbNb = d2g_lav_dxNbNb(C_lav_Cr, C_lav_Nb);
	/* #endif */

	gsl_matrix_set(J, 4, 4, -jac_lav_CrCr);
	gsl_matrix_set(J, 4, 5, -jac_lav_CrNb);
	gsl_matrix_set(J, 5, 4, -jac_lav_NbCr);
	gsl_matrix_set(J, 5, 5, -jac_lav_NbNb);

	return GSL_SUCCESS;
}


int parallelTangent_fdf(const gsl_vector* x, void* params, gsl_vector* f, gsl_matrix* J)
{
	parallelTangent_f(x,  params, f);
	parallelTangent_df(x, params, J);

	return GSL_SUCCESS;
}


rootsolver::rootsolver() :
	n(NC*(NP+1)), // one equation per component per phase: eight total
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
	 * hybridsj, hybridj, newton, gnewton
	 */
	algorithm = gsl_multiroot_fdfsolver_hybridsj;
	solver = gsl_multiroot_fdfsolver_alloc(algorithm, n);
	mrf = {&parallelTangent_f, &parallelTangent_df, &parallelTangent_fdf, n, &par};
}


template<typename T> double
rootsolver::solve(MMSP::vector<T>& GRIDN)
{
	int status;
	size_t iter = 0;

	par.x_Cr = GRIDN[0];
	par.x_Nb = GRIDN[1];

	par.n_del = h(fabs(GRIDN[NC  ]));
	par.n_lav = h(fabs(GRIDN[NC+1]));


	// copy initial guesses from grid
	for (int i = 0; i < NC*(NP+1); i++)
		gsl_vector_set(x, i, static_cast<double>(GRIDN[NC+NP+i]));

	gsl_multiroot_fdfsolver_set(solver, &mrf, x);

	do {
		iter++;
		status = gsl_multiroot_fdfsolver_iterate(solver);
		if (status) // extra points for finishing early!
			break;
		status = gsl_multiroot_test_residual(solver->f, tolerance);
	} while (status == GSL_CONTINUE && iter < maxiter);

	double residual = gsl_blas_dnrm2(solver->f);

	if (status == GSL_SUCCESS)
		for (int i = 0; i < NC*(NP+1); i++)
			GRIDN[NC+NP+i] = static_cast<T>(gsl_vector_get(solver->x, i));

	return residual;
}


rootsolver::~rootsolver()
{
	gsl_multiroot_fdfsolver_free(solver);
	gsl_vector_free(x);
}



template<int dim,class T>
T maxVelocity(MMSP::grid<dim, MMSP::vector<T> > const & oldGrid, const double& dt,
              MMSP::grid<dim, MMSP::vector<T> > const & newGrid)
{
	double vmax = 0.0;

	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (int n = 0; n<MMSP::nodes(newGrid); n++) {
		MMSP::vector<int> x = MMSP::position(newGrid,n);

		const MMSP::vector<T>& oldGridN = oldGrid(n);
		MMSP::vector<T>& newGridN = newGrid(n);

		T myVelocity = 0.0;

		for (int i = 0; i < NP; i++) {
			const T newPhaseFrac = h(fabs(newGridN[i+NC]));
			if (newPhaseFrac > 0.1 && newPhaseFrac < 0.9) {
				const MMSP::vector<T> gradPhi = maskedgradient(newGrid, x, i+NC);
				const T magGrad = std::sqrt(gradPhi * gradPhi);
				if (magGrad > epsilon) {
					const T oldPhaseFrac = h(fabs(oldGridN[i+NC]));
					T dphidt = std::fabs(newPhaseFrac - oldPhaseFrac) / dt;
					T v = (dphidt > epsilon) ? dphidt / magGrad : 0.0;
					myVelocity = std::max(myVelocity, v);
				}
			}
		}

		#ifdef _OPENMP
		#pragma omp critical
		{
		#endif
			vmax = std::max(vmax, myVelocity);
		#ifdef _OPENMP
		}
		#endif
	}

	#ifdef MPI_VERSION
	MPI::COMM_WORLD.Barrier();
	double myv(vmax);
	MPI::COMM_WORLD.Allreduce(&myv, &vmax, 1, MPI_DOUBLE, MPI_MAX);
	#endif

	return vmax;
}


template<int dim,class T>
MMSP::vector<double> summarize(MMSP::grid<dim, MMSP::vector<T> > const & oldGrid, const double& dt,
                               MMSP::grid<dim, MMSP::vector<T> > const & newGrid)
{
	/* =========================================================================== *
	 * Integrate composition, phase fractions, and free energy over the whole grid *
	 * to make sure mass is conserved, phase transformations are sane, and energy  *
	 * decreases with time. Store either energy-density _or_ interfacial velocity  *
	 * in the field of each grid point for analysis.                               *
	 * =========================================================================== */

	double Ntot = 1.0;
	double dV = 1.0;
	for (int d = 0; d<dim; d++) {
		Ntot *= double(MMSP::g1(newGrid, d) - MMSP::g0(newGrid, d));
		dV *= MMSP::dx(newGrid, d);
	}
	MMSP::vector<double> summary(NC + NP + 3, 0.0);

	#ifdef _OPENMP
	#pragma omp parallel for shared(summary)
	#endif
	for (int n = 0; n<MMSP::nodes(newGrid); n++) {
		MMSP::vector<int> x = MMSP::position(newGrid,n);
		MMSP::vector<T>& newGridN = newGrid(n);
		const MMSP::vector<T>& oldGridN = oldGrid(n);
		MMSP::vector<double> mySummary(NC + NP + 3, 0.0);
		double myVelocity = 0.0;

		for (int i = 0; i < NC; i++)
			mySummary[i] = newGridN[i];      // compositions

		mySummary[NC] = 1.0;                 // gamma fraction init
		mySummary[NC + NP + 3 - 2] = dV * gibbs(newGridN); // energy density init
		// fields 3, 4, 5, 7 are initialized to zero

		for (int i = 0; i < NP; i++) {
			const MMSP::vector<T> gradPhi = maskedgradient(newGrid, x, i+NC);
			const T gradSq = (gradPhi * gradPhi); // vector inner product
			const T newPhaseFrac = h(fabs(newGridN[i+NC]));

			mySummary[i+NC+1] = newPhaseFrac;       // secondary phase fraction
			mySummary[NC] -= newPhaseFrac;           // contributes to gamma phase;
			mySummary[NC + NP + 3 - 2] += dV * kappa[i] * gradSq; // gradient contributes to energy

			if (std::sqrt(gradSq) > epsilon && newPhaseFrac > 0.1 && newPhaseFrac < 0.9) {
				const T oldPhaseFrac = h(fabs(oldGridN[i+NC]));
				const T dphidt = std::fabs(newPhaseFrac - oldPhaseFrac) / dt;
				const T v = (dphidt > epsilon) ? dphidt / std::sqrt(gradSq) : 0.0;
				myVelocity = std::max(myVelocity, v);
			}
		}


		// Record local velocity
		//newGridN[fields(newGrid)-1] = myVelocity;

		// Sum up mass and phase fractions. Since mySummary[7]=0, it is not accumulated.
		// Get maximum interfacial velocity from separate variable
		#ifdef _OPENMP
		#pragma omp critical (sumCrit1)
		{
		#endif
		summary += mySummary;
		#ifdef _OPENMP
		}
		#pragma omp critical (sumCrit2)
		{
		#endif
		summary[NC + NP + 3 - 1] = std::max(myVelocity, summary[NC + NP + 3 - 1]);
		#ifdef _OPENMP
		}
		#endif
	}

	for (int i = 0; i < NC + NP + 3 - 2; i++)
		summary[i] /= Ntot;

	#ifdef MPI_VERSION
	MMSP::vector<double> tmpSummary(summary);
	MPI::COMM_WORLD.Reduce(&(tmpSummary[0]), &(summary[0]), NC + NP + 3 - 1, MPI_DOUBLE, MPI_SUM, 0);
	MPI::COMM_WORLD.Allreduce(&(tmpSummary[NC + NP + 3 - 1]), &(summary[NC + NP + 3 - 1]), 1, MPI_DOUBLE, MPI_MAX); // maximum velocity
	#endif

	return summary;
}

#endif

#include"MMSP.main.hpp"
