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
 * versions bear some notice that they have been modified.                           *
 *                                                                                   *
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
#if defined PARABOLIC
#include"parabola625.c"
#elif defined calphad
#include"energy625.c"
#else
#include"taylor625.c"
#endif

// Note: alloy625.hpp contains important declarations and comments. Have a look.
//       energy625.c is generated from CALPHAD using pycalphad and SymPy, in CALPHAD_extraction.ipynb.

/* =============================================== *
 * Implement MMSP kernels: generate() and update() *
 * =============================================== */

/* Representation includes thirteen field variables:
 *
 * X0.  molar fraction of Cr + Mo
 * X1.  molar fraction of Nb
 * bal. molar fraction of Ni
 *
 * P2.  phase fraction of delta
 * P3.  phase fraction of mu
 * P4.  phase fraction of Laves
 * bal. phase fraction of gamma
 *
 * C5.  Cr molar fraction in pure gamma
 * C6.  Nb molar fraction in pure gamma
 * bal. Ni molar fraction in pure gamma
 *
 * C7.  Cr molar fraction in pure delta
 * C8.  Nb molar fraction in pure delta
 * bal. Ni molar fraction in pure delta
 *
 * C9.  Cr molar fraction in pure mu
 * C10. Nb molar fraction in pure mu
 * bal. Ni molar fraction in pure mu
 *
 * C11. Cr molar fraction in pure Laves
 * C12. Nb molar fraction in pure Laves
 * bal. Ni molar fraction in pure Laves
 */

/* Based on experiments (EDS) and simulations (DICTRA), additively manufactured
 * IN625 has the following compositions:
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
//                        Nominal |     phase diagram      | Enriched
//                        gamma   | delta    mu     laves  | gamma (Excess)
const double xCr[NP+2] = {0.30,     0.0125,  0.04,  0.3875,  0.31-0.30};
const double xNb[NP+2] = {0.02,     0.2500,  0.50,  0.2500,  0.13-0.02};

// Define st.dev. of Gaussians for alloying element segregation
//                         Cr      Nb
const double bell[NC] = {150.0e-9, 50.0e-9}; // est. between 80-200 nm from SEM

// Kinetic and model parameters
const double meshres = 5.0e-9; //1.25e-9;    // grid spacing (m)
//const double Vm = 1.0e-5;         // molar volume (m^3/mol)
const double alpha = 1.07e11;     // three-phase coexistence coefficient (J/m^3)

// Diffusion constants from Xu
const double D_CrCr = 2.42e-15; // diffusivity in FCC Ni (m^2/s)
const double D_CrNb = 2.47e-15; // diffusivity in FCC Ni (m^2/s)
const double D_NbCr = 0.43e-15; // diffusivity in FCC Ni (m^2/s)
const double D_NbNb = 3.32e-15; // diffusivity in FCC Ni (m^2/s)
//#endif


const double kappa_del = 1.24e-8; // gradient energy coefficient (J/m)
const double kappa_mu  = 1.24e-8; // gradient energy coefficient (J/m)
const double kappa_lav = 1.24e-8; // gradient energy coefficient (J/m)

// Choose numerical diffusivity to lock chemical and transformational timescales
// Zhou's numbers
const double L_del = 2.904e-11;   // numerical mobility (m^2/Ns)
const double L_mu  = 2.904e-11;   // numerical mobility (m^2/Ns)
const double L_lav = 2.904e-11;   // numerical mobility (m^2/Ns)
/*
const double L_del = 1.92e-12;   // numerical mobility (m^2/Ns)
const double L_mu  = 1.92e-12;   // numerical mobility (m^2/Ns)
const double L_lav = 1.92e-12;   // numerical mobility (m^2/Ns)
*/

const double sigma_del = 1.01;    // J/m^2
const double sigma_mu  = 1.01;    // J/m^2
const double sigma_lav = 1.01;    // J/m^2

// Note that ifce width was not considered in Zhou's model, but is in vanilla KKS
/*
// Zhou's numbers
const double omega_del = 9.5e8;  // multiwell height (m^2/Nsm^2)
const double omega_mu  = 9.5e8;  // multiwell height (m^2/Nsm^2)
const double omega_lav = 9.5e8;  // multiwell height (m^2/Nsm^2)
*/
const double width_factor = 2.2;  // 2.2 if interface is [0.1,0.9]; 2.94 if [0.05,0.95]
const double ifce_width = 10.0*meshres; // ensure at least 7 points through the interface
const double omega_del = 3.0 * width_factor * sigma_del / ifce_width; // 9.5e8;  // multiwell height (J/m^3)
const double omega_mu  = 3.0 * width_factor * sigma_mu  / ifce_width; // 9.5e8;  // multiwell height (J/m^3)
const double omega_lav = 3.0 * width_factor * sigma_lav / ifce_width; // 9.5e8;  // multiwell height (J/m^3)

// Numerical considerations
const bool useNeumann = true;    // apply zero-flux boundaries (Neumann type)?
const bool adaptStep = true;     // apply adaptive time-stepping?
const bool tanh_init = false;    // apply tanh profile to initial profile of composition and phase
const double epsilon = 1.0e-10;  // what to consider zero to avoid log(c) explosions

const double root_tol = 1.0e-3;   // residual tolerance (default is 1e-7)
const int root_max_iter = 500000; // default is 1000, increasing probably won't change anything but your runtime

const int logstep = 1000;         // steps between logging status

//const double LinStab = 0.00125; // threshold of linear stability (von Neumann stability condition)
#ifndef CALPHAD
const double LinStab = 1.0 / 15.06022;  // threshold of linear stability (von Neumann stability condition)
#else
const double LinStab = 1.0 / 37650.55;  // threshold of linear stability (von Neumann stability condition)
#endif

namespace MMSP
{

void generate(int dim, const char* filename)
{
	int rank=0;
	#ifdef MPI_VERSION
	rank = MPI::COMM_WORLD.Get_rank();
 	#endif

	FILE* cfile = NULL;
	FILE* tfile = NULL;

	if (rank==0) {
		cfile = fopen("c.log", "w"); // existing log will be overwritten
		if (adaptStep)
			tfile = fopen("t.log", "w"); // existing log will be overwritten
	}

	const double dtp = (meshres*meshres)/(2.0 * dim * L_del*kappa_del); // transformation-limited timestep
	//const double dtc = (meshres*meshres)/(8.0 * Vm*Vm * std::max(M_Cr*d2g_gam_dxCrCr(xe_gam_Cr(), xe_gam_Nb()), M_Nb*d2g_gam_dxNbNb()))); // diffusion-limited timestep
	const double dtc = (meshres*meshres)/(2.0 * dim * std::max(D_CrCr, D_NbNb)); // diffusion-limited timestep
	const double dt = LinStab * std::min(dtp, dtc);

	if (dim==1) {
		// Construct grid
		const int Nx = 768; // divisible by 12 and 64
		double dV = 1.0;
		double Ntot = 1.0;
		GRID1D initGrid(14, 0, Nx);
		for (int d=0; d<dim; d++) {
			dx(initGrid,d)=meshres;
			dV *= meshres;
			Ntot *= g1(initGrid, d) - g0(initGrid, d);
			if (useNeumann) {
				b0(initGrid, d) = Neumann;
				b1(initGrid, d) = Neumann;
			}
		}

		// Sanity check on system size and  particle spacing
		if (rank==0)
			std::cout << "Timestep dt=" << dt << ". Linear stability limits: dtp=" << dtp << " (transformation-limited), dtc="<< dtc << " (diffusion-limited)." << std::endl;

		/*
		const double Csstm[2] = {0.1500, 0.1500}; // gamma-delta system Cr, Nb composition
		const double Cprcp[2] = {0.0125, 0.2500}; // delta precipitate  Cr, Nb composition
		const int mid = 0;    // matrix phase
		const int pid = NC+0; // delta phase

		const double Csstm[2] = {0.0500, 0.3500}; // gamma-mu system Cr, Nb composition
		const double Cprcp[2] = {0.0500, 0.4500}; // mu  precipitate Cr, Nb composition
		const int mid = 0;    // matrix phase
		const int pid = NC+1; // mu phase

		const double Csstm[2] = {0.3625, 0.1625}; // gamma-Laves system Cr, Nb composition
		const double Cprcp[2] = {0.3625, 0.2750}; // Laves precipitate  Cr, Nb composition
		const int mid = 0;    // matrix phase
		const int pid = NC+2; // Laves phase

		const double Csstm[2] = {0.2500, 0.2500}; // delta-Laves system Cr, Nb composition
		const double Cprcp[2] = {0.0125, 0.2500}; // delta precipitate  Cr, Nb composition
		const int mid = NC+2; // Laves phase
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
		const vector<double> blank(fields(initGrid), 0.0);

		for (int n=0; n<nodes(initGrid); n++) {
			vector<int> x = position(initGrid, n);
			vector<double>& initGridN = initGrid(n);

			initGridN = blank;

			if (x[0] < Nprcp) {
				// Initialize precipitate with equilibrium composition (from phase diagram)
				initGridN[0] = Cprcp[0];
				initGridN[1] = Cprcp[1];
				initGridN[pid] = 1.0 - epsilon;
			} else {
				// Initialize gamma to satisfy system composition
				initGridN[0] = (Csstm[0] * Nx - Cprcp[0] * Nprcp) / Nmtrx;
				initGridN[1] = (Csstm[1] * Nx - Cprcp[1] * Nprcp) / Nmtrx;
				if (mid == pid)
					initGridN[mid] = -1.0 + epsilon;
				else if (mid != 0)
					initGridN[mid] = 1.0 - epsilon;
			}

		}
		*/


		/* ============================= *
		 * Four-phase test configuration *
		 * ============================= */

		const int Nprcp[NP] = {Nx / 8, Nx / 8, Nx / 8}; // grid points per seed
		const int Noff = Nx / NP; // grid points between seeds
		int Nmtrx = Nx; // grid points of matrix phase

		const double Csstm[2] = {0.3000, 0.1625}; // system Cr, Nb composition
		const double Cprcp[NP][2] = {{0.0125, 0.2500}, // delta
		                             {0.0500, 0.4500}, // mu
		                             {0.3625, 0.2750}  // Laves
		                            }; // precipitate  Cr, Nb composition

		double matCr = Csstm[0] * Nx;
		double matNb = Csstm[1] * Nx;
		for (int pid=0; pid < NP; pid++) {
			matCr -= Cprcp[pid][0] * Nprcp[pid];
			matNb -= Cprcp[pid][1] * Nprcp[pid];
			Nmtrx -= Nprcp[pid];
		}
		matCr /= Nmtrx;
		matNb /= Nmtrx;

		const vector<double> blank(fields(initGrid), 0.0);

		unsigned int totBadTangents = 0;

		#ifdef _OPENMP
		#pragma omp parallel for
		#endif
		for (int n=0; n<nodes(initGrid); n++) {
			vector<int> x = position(initGrid, n);
			vector<double>& initGridN = initGrid(n);

			initGrid(n) = blank;

			// Initialize gamma to satisfy system composition
			initGridN[0] = matCr;
			initGridN[1] = matNb;

			for (int pid=0; pid < NP; pid++) {
				if (x[0]>= pid*Noff && x[0] < pid*Noff + Nprcp[pid]) {
					// Initialize precipitate with equilibrium composition (from phase diagram)
					initGridN[0] = Cprcp[pid][0];
					initGridN[1] = Cprcp[pid][1];
					initGridN[NC+pid] = 1.0 - epsilon;
				}
			}

			// Initialize compositions in a manner compatible with OpenMP and MPI parallelization
			guessGamma(initGrid(n));
			guessDelta(initGrid(n));
			guessMu(   initGrid(n));
			guessLaves(initGrid(n));

			/* =========================== *
			 * Solve for parallel tangents *
			 * =========================== */

			rootsolver parallelTangentSolver;
			//double res = parallelTangentSolver.solve(initGridN);
			double res = parallelTangentSolver.solve(initGrid(n));

			if (res>root_tol) {
				// Invalid roots: substitute guesses.

				#ifdef _OPENMP
				#pragma omp critical (iniCrit1)
				#endif
				{
					totBadTangents++;
				}

				guessGamma(initGrid(n));
				guessDelta(initGrid(n));
				guessMu(   initGrid(n));
				guessLaves(initGrid(n));
			}
		}

		ghostswap(initGrid);

		#ifdef MPI_VERSION
		unsigned int myBad(totBadTangents);
		MPI::COMM_WORLD.Reduce(&myBad, &totBadTangents, 1, MPI_UNSIGNED, MPI_SUM, 0);
		#endif

		vector<double> summary = summarize(initGrid, dt, initGrid);

		if (rank==0) {

			fprintf(cfile, "%11.9g\t%11.9g\t%11.9g\t%11.9g\t%11.9g\t%11.9g\t%11.9g\t%11.9g\t%11u\n",
			dt, summary[0], summary[1], summary[2], summary[3], summary[4], summary[5], summary[7], totBadTangents);

			if (adaptStep)
				fprintf(tfile, "%11.9g\t%11.9g\t%11.9g\n", 0.0, 1.0, dt);

			std::cout << "       x_Cr        x_Nb        x_Ni         p_g         p_d         p_m         p_l\n";
			printf("%11.9g %11.9g %11.9g %11.9g %11.9g %11.9g %11.9g\n", summary[0], summary[1], 1.0-summary[0]-summary[1],
			                                                             summary[2], summary[3], summary[4], summary[5]);
		}

		output(initGrid,filename);




	} else if (dim==2) {




		// Construct grid
		const int Nx = 768; // divisible by 12 and 64
		const int Ny = 192;
		double dV = 1.0;
		double Ntot = 1.0;
		GRID2D initGrid(14, 0, Nx, 0, Ny);
		for (int d=0; d<dim; d++) {
			dx(initGrid,d)=meshres;
			dV *= meshres;
			Ntot *= g1(initGrid, d) - g0(initGrid, d);
			if (useNeumann) {
				b0(initGrid, d) = Neumann;
				b1(initGrid, d) = Neumann;
			}
		}

		// Precipitate radii: minimum for thermodynamic stability is 7.5 nm,
		//                    minimum for numerical stability is 14*dx (due to interface width).
		const double rPrecip[NP] = {3.0*7.5e-9 / dx(initGrid,0),  // delta
		                            3.0*7.5e-9 / dx(initGrid,0),  // mu
		                            3.0*7.5e-9 / dx(initGrid,0)}; // Laves


		// Sanity check on system size and  particle spacing
		if (rank==0)
			std::cout << "Timestep dt=" << dt << ". Linear stability limits: dtp=" << dtp << " (transformation-limited), dtc="<< dtc << " (diffusion-limited)." << std::endl;

		for (int i=0; i<NP; i++) {
			if (rPrecip[i] > Ny/2)
				std::cerr << "Warning: domain too small to accommodate phase " << i << ", expand beyond " << 2.0*rPrecip[i] << " pixels." << std::endl;
		}

		// Zero initial condition
		for (int n=0; n<nodes(initGrid); n++) {
			vector<double>& initGridN = initGrid(n);
			for (int i=NC; i<fields(initGrid); i++)
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
			comp += embedParticle(initGrid, origin, j+2, rPrecip[j], xCr[j+1], xNb[j+1],  1.0 - epsilon);
			origin[0] = Nx/2 + xoffset;
			origin[1] = Ny - 5*yoffset + yoffset/2;
			comp += embedParticle(initGrid, origin, j+2, rPrecip[j], xCr[j+1], xNb[j+1],  1.0 - epsilon);
			origin[0] = Nx/2;
			origin[1] = Ny - 3*yoffset + yoffset/2;
			comp += embedParticle(initGrid, origin, j+2, rPrecip[j], xCr[j+1], xNb[j+1], -1.0 + epsilon);
			origin[0] = Nx/2 - xoffset;
			origin[1] = Ny - 6*yoffset + yoffset/2;
			comp += embedParticle(initGrid, origin, j+2, rPrecip[j], xCr[j+1], xNb[j+1], -1.0 + epsilon);

			// Initialize mu precipitates
			j = 1;
			origin[0] = Nx / 2;
			origin[1] = Ny - 2*yoffset + yoffset/2;
			comp += embedParticle(initGrid, origin, j+2, rPrecip[j], xCr[j+1], xNb[j+1],  1.0 - epsilon);
			origin[0] = Nx/2 - xoffset;
			origin[1] = Ny - 4*yoffset + yoffset/2;
			comp += embedParticle(initGrid, origin, j+2, rPrecip[j], xCr[j+1], xNb[j+1],  1.0 - epsilon);
			origin[0] = Nx/2 + xoffset;
			origin[1] = Ny - 3*yoffset + yoffset/2;
			comp += embedParticle(initGrid, origin, j+2, rPrecip[j], xCr[j+1], xNb[j+1], -1.0 + epsilon);
			origin[0] = Nx/2;
			origin[1] = Ny - 5*yoffset + yoffset/2;
			comp += embedParticle(initGrid, origin, j+2, rPrecip[j], xCr[j+1], xNb[j+1], -1.0 + epsilon);

			// Initialize Laves precipitates
			j = 2;
			origin[0] = Nx/2 + xoffset;
			origin[1] = Ny - yoffset + yoffset/2;
			comp += embedParticle(initGrid, origin, j+2, rPrecip[j], xCr[j+1], xNb[j+1],  1.0 - epsilon);
			origin[0] = Nx/2;
			origin[1] = Ny - 4*yoffset + yoffset/2;
			comp += embedParticle(initGrid, origin, j+2, rPrecip[j], xCr[j+1], xNb[j+1],  1.0 - epsilon);
			origin[0] = Nx/2 - xoffset;
			origin[1] = Ny - 2*yoffset + yoffset/2;
			comp += embedParticle(initGrid, origin, j+2, rPrecip[j], xCr[j+1], xNb[j+1], -1.0 + epsilon);
			origin[0] = Nx/2;
			origin[1] = Ny - 6*yoffset + yoffset/2;
			comp += embedParticle(initGrid, origin, j+2, rPrecip[j], xCr[j+1], xNb[j+1], -1.0 + epsilon);
		} else if (0) {
			/* =============================================== *
			 * Three-phase Particle Growth test configuration  *
			 *                                                 *
			 * Seed a 1.0 um x 0.05 um domain with 3 particles *
			 * (one of each secondary phase) in a single row   *
			 * to test competitive growth with Gibbs-Thomson   *
			 * =============================================== */

			// Initialize delta precipitates
			int j = 0;
			origin[0] = Nx / 2;
			origin[1] = Ny / 2;
			comp += embedParticle(initGrid, origin, j+2, rPrecip[j], xCr[j+1], xNb[j+1],  1.0 - epsilon);

			// Initialize mu precipitates
			j = 1;
			origin[0] = Nx / 2 - xoffset;
			comp += embedParticle(initGrid, origin, j+2, rPrecip[j], xCr[j+1], xNb[j+1],  1.0 - epsilon);

			// Initialize Laves precipitates
			j = 2;
			origin[0] = Nx / 2 + xoffset;
			comp += embedParticle(initGrid, origin, j+2, rPrecip[j], xCr[j+1], xNb[j+1],  1.0 - epsilon);
		} else if (0) {
			/* ============================================= *
			 * Three-phase Stripe Growth test configuration  *
			 *                                               *
			 * Seed a 1.0 um x 0.05 um domain with 3 stripes *
			 * (one of each secondary phase) in a single row *
			 * to test competitive growth without curvature  *
			 * ============================================= */

			// Initialize delta stripe
			int j = 0;
			origin[0] = Nx / 2;
			origin[1] = Ny / 2;
			comp += embedStripe(initGrid, origin, j+2, rPrecip[j], xCr[j+1], xNb[j+1],  1.0 - epsilon);

			// Initialize mu stripe
			j = 1;
			origin[0] = Nx / 2 - xoffset;
			comp += embedStripe(initGrid, origin, j+2, rPrecip[j], xCr[j+1], xNb[j+1],  1.0 - epsilon);

			// Initialize Laves stripe
			j = 2;
			origin[0] = Nx / 2 + xoffset;
			comp += embedStripe(initGrid, origin, j+2, rPrecip[j], xCr[j+1], xNb[j+1],  1.0 - epsilon);

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
			const double delCr = 0.15; // Choose initial composition carefully!
			const double delNb = 0.15;
			comp += embedStripe(initGrid, origin, 2, Nx/4, delCr, delNb,  1.0-epsilon);
		} else {
			if (rank==0)
				std::cerr<<"Error: specify an initial condition!"<<std::endl;
			MMSP::Abort(-1);
		}

		// Synchronize global intiial condition parameters
		Composition myComp;
		myComp += comp;

		#ifdef MPI_VERSION
		// Caution: Primitive. Will not scale to large MPI systems.
		for (int j=0; j<NP+1; j++) {
			MPI::COMM_WORLD.Allreduce(&myComp.N[j], &comp.N[j], 1, MPI_INT, MPI_SUM);
			for (int i=0; i<NC; i++) {
				MPI::COMM_WORLD.Allreduce(&myComp.x[j][i], &comp.x[j][i], 1, MPI_DOUBLE, MPI_SUM);
			}
		}
		#endif


		// Initialize matrix to achieve specified system composition
		double matCr = Ntot * xCr[0];
		double matNb = Ntot * xNb[0];
		double Nmat  = Ntot;
		for (int i=0; i<NP+1; i++) {
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
		for (int n=0; n<nodes(initGrid); n++) {
			double nx = 0.0;
			vector<double>& initGridN = initGrid(n);

			for (int i=NC; i<NC+NP; i++)
				nx += h(fabs(initGridN[i]));

			if (nx < epsilon) { // pure gamma
				initGridN[0] += matCr;
				initGridN[1] += matNb;
			}

			// Initialize compositions in a manner compatible with OpenMP and MPI parallelization
			guessGamma(initGrid(n));
			guessDelta(initGrid(n));
			guessMu(   initGrid(n));
			guessLaves(initGrid(n));


			/* =========================== *
			 * Solve for parallel tangents *
			 * =========================== */

			rootsolver parallelTangentSolver;
			//double res = parallelTangentSolver.solve(initGridN);
			double res = parallelTangentSolver.solve(initGrid(n));

			if (res>root_tol) {
				// Invalid roots: substitute guesses.
				#ifdef _OPENMP
				#pragma omp critical (iniCrit2)
				#endif
				{
					totBadTangents++;
				}

				guessGamma(initGrid(n));
				guessDelta(initGrid(n));
				guessMu(   initGrid(n));
				guessLaves(initGrid(n));
			}
		}

		ghostswap(initGrid);

		#ifdef MPI_VERSION
		unsigned int myBad(totBadTangents);
		MPI::COMM_WORLD.Reduce(&myBad, &totBadTangents, 1, MPI_UNSIGNED, MPI_SUM, 0);
		#endif

		vector<double> summary = summarize(initGrid, dt, initGrid);

		if (rank==0) {

			fprintf(cfile, "%11.9g\t%11.9g\t%11.9g\t%11.9g\t%11.9g\t%11.9g\t%11.9g\t%11.9g\t%11u\n",
			dt, summary[0], summary[1], summary[2], summary[3], summary[4], summary[5], summary[7], totBadTangents);

			if (adaptStep)
				fprintf(tfile, "%11.9g\t%11.9g\t%11.9g\n", 0.0, 1.0, dt);

			std::cout << "       x_Cr        x_Nb        x_Ni         p_g         p_d         p_m         p_l\n";
			printf("%11.9g %11.9g %11.9g %11.9g %11.9g %11.9g %11.9g\n", summary[0], summary[1], 1.0-summary[0]-summary[1],
			                                                             summary[2], summary[3], summary[4], summary[5]);
		}

		output(initGrid,filename);



	} else
		std::cerr << "Error: " << dim << "-dimensional grids unsupported." << std::endl;

	if (rank == 0) {
		fclose(cfile);
		if (adaptStep)
			fclose(tfile);
	}

}





template <int dim, typename T> void update(grid<dim,vector<T> >& oldGrid, int steps)
{
	int rank=0;
	#ifdef MPI_VERSION
	rank = MPI::COMM_WORLD.Get_rank();
	#endif

	ghostswap(oldGrid);
	grid<dim,vector<T> > newGrid(oldGrid);

	const double dtp = (meshres*meshres)/(2.0 * dim * L_del*kappa_del); // transformation-limited timestep
	//const double dtc = (meshres*meshres)/(8.0 * Vm*Vm * std::max(M_Cr*d2g_gam_dxCrCr(xe_gam_Cr(), xe_gam_Nb()), M_Nb*d2g_gam_dxNbNb()))); // diffusion-limited timestep
	const double dtc = (meshres*meshres)/(2.0 * dim * std::max(D_CrCr, D_NbNb)); // diffusion-limited timestep
	const double dt = LinStab * std::min(dtp, dtc);

	double dV = 1.0;
	double Ntot = 1.0;
	for (int d=0; d<dim; d++) {
		dx(oldGrid,d) = meshres;
		dx(newGrid,d) = meshres;
		dV *= dx(oldGrid,d);
		Ntot *= double(g1(oldGrid, d) - g0(oldGrid, d));
		if (useNeumann && x0(oldGrid,d) == g0(oldGrid,d)) {
			b0(oldGrid,d) = Neumann;
			b0(newGrid,d) = Neumann;
		} else if (useNeumann && x1(oldGrid,d) == g1(oldGrid,d)) {
			b1(oldGrid,d) = Neumann;
			b1(newGrid,d) = Neumann;
		}
	}

	static int logcount = 1;

	FILE* cfile = NULL;
	FILE* tfile = NULL;

	if (rank==0) {
		cfile = fopen("c.log", "a"); // new results will be appended
		if (adaptStep)
			tfile = fopen("t.log", "a"); // new results will be appended
	}


	// Prepare reference and live values for adaptive timestepper
	double current_time = 0.0;
	double current_dt = dt;

	const double run_time = dt * steps;
	const double timelimit = std::min(dtp, dtc) / 5.0;
	const double speedlimit = meshres / 8.0;
	const double scaleup = 1.1; // how fast to raise dt when stable
	const double scaledn = 0.8; // how fast to drop dt when unstable
	int timeratchet = 0;


	while (current_time < run_time && current_dt > 0.0) {
	//for (int step = 0; step < steps; step++) {
		if (adaptStep)
			current_dt = std::min(current_dt, run_time - current_time);

		// Bug: when dt increases, progress bar reprints step zero
		if (rank==0) {
			timeratchet = std::max(timeratchet, int(floor(current_time / current_dt)));
			//print_progress(step, steps);
			print_progress(timeratchet, ceil(run_time / current_dt));
			//std::cout<<'('<<current_time / current_dt<<','<<run_time / current_dt<<')'<<std::endl;
		}

		unsigned int totBadTangents = 0;

		#ifdef _OPENMP
		#pragma omp parallel for
		#endif
		for (int n=0; n<nodes(oldGrid); n++) {
			/* ============================================== *
			 * Point-wise kernel for parallel PDE integration *
			 * ============================================== */

			vector<int> x = position(oldGrid,n);
			vector<T>& oldGridN = oldGrid(n);
			vector<T>& newGridN = newGrid(n);

			/* ================= *
			 * Collect Constants *
			 * ================= */

			const T& xCr = oldGridN[0];
			const T& xNb = oldGridN[1];

			const T& phi_del  = oldGridN[2]; // phase fraction of delta
			const T& phi_mu   = oldGridN[3]; // phase fraction of mu
			const T& phi_lav  = oldGridN[4]; // phase fraction of Laves

			const T abs_phi_del = fabs(phi_del);
			const T abs_phi_mu  = fabs(phi_mu);
			const T abs_phi_lav = fabs(phi_lav);

			const T& C_gam_Cr = oldGridN[5]; // Cr molar fraction in pure gamma
			const T& C_gam_Nb = oldGridN[6]; // Nb molar fraction in pure gamma

			const T& C_del_Cr = oldGridN[7]; // Cr molar fraction in pure delta
			const T& C_del_Nb = oldGridN[8]; // Nb molar fraction in pure delta

			const T& C_mu_Cr  = oldGridN[9];  // Cr molar fraction in pure mu
			const T& C_mu_Nb  = oldGridN[10]; // Nb molar fraction in pure mu

			const T& C_lav_Cr = oldGridN[11]; // Cr molar fraction in pure Laves
			const T& C_lav_Nb = oldGridN[12]; // Nb molar fraction in pure Laves

			const double Ggam = g_gam(C_gam_Cr, C_gam_Nb);
			const double Gdel = g_del(C_del_Cr, C_del_Nb);
			const double Gmu  = g_mu( C_mu_Cr,  C_mu_Nb );
			const double Glav = g_lav(C_lav_Cr, C_lav_Nb);

			/* =================== *
			 * Compute Derivatives *
			 * =================== */

			// Laplacians of field variables
			const vector<double> laplac = laplacian(oldGrid, x);

			const double& lapPhi_del = laplac[2];
			const double& lapPhi_mu  = laplac[3];
			const double& lapPhi_lav = laplac[4];

			const double& lapxCr_gam = laplac[5];
			const double& lapxNb_gam = laplac[6];


			// Diffusion potentials (at equilibrium, equal to chemical potentials)
			const double pot_Cr = dg_gam_dxCr(C_gam_Cr, C_gam_Nb);
			const double pot_Nb = dg_gam_dxNb(C_gam_Cr, C_gam_Nb);

			// "Pressures" between matrix and precipitate phases
			const double P_del = Ggam - Gdel - (C_gam_Cr - C_del_Cr) * pot_Cr - (C_gam_Nb - C_del_Nb) * pot_Nb;
			const double P_mu  = Ggam - Gmu  - (C_gam_Cr - C_mu_Cr)  * pot_Cr - (C_gam_Nb - C_mu_Nb)  * pot_Nb;
			const double P_lav = Ggam - Glav - (C_gam_Cr - C_lav_Cr) * pot_Cr - (C_gam_Nb - C_lav_Nb) * pot_Nb;

			// Variational derivatives (scalar minus gradient term in Euler-Lagrange eqn)
			double delF_delPhi_del = -sign(phi_del) * hprime(abs_phi_del) * P_del;
			double delF_delPhi_mu  = -sign(phi_mu)  * hprime(abs_phi_mu)  * P_mu;
			double delF_delPhi_lav = -sign(phi_lav) * hprime(abs_phi_lav) * P_lav;

			delF_delPhi_del += 2.0 * omega_del * phi_del * (1.0 - abs_phi_del) * (1.0 - abs_phi_del - sign(phi_del) * phi_del);
			delF_delPhi_mu  += 2.0 * omega_mu  * phi_mu  * (1.0 - abs_phi_mu)  * (1.0 - abs_phi_mu  - sign(phi_mu)  * phi_mu);
			delF_delPhi_lav += 2.0 * omega_lav * phi_lav * (1.0 - abs_phi_lav) * (1.0 - abs_phi_lav - sign(phi_lav) * phi_lav);

			delF_delPhi_del += 4.0 * alpha * phi_del * (phi_mu  * phi_mu  + phi_lav * phi_lav);
			delF_delPhi_mu  += 4.0 * alpha * phi_mu  * (phi_del * phi_del + phi_lav * phi_lav);
			delF_delPhi_lav += 4.0 * alpha * phi_lav * (phi_del * phi_del + phi_mu  * phi_mu);


			delF_delPhi_del -= kappa_del * lapPhi_del;
			delF_delPhi_mu  -= kappa_mu  * lapPhi_mu;
			delF_delPhi_lav -= kappa_lav * lapPhi_lav;


			/* ============================================= *
			 * Solve the Equation of Motion for Compositions *
			 * ============================================= */

			newGridN[0] = xCr + current_dt * (D_CrCr * lapxCr_gam + D_CrNb * lapxNb_gam);
			newGridN[1] = xNb + current_dt * (D_NbCr * lapxCr_gam + D_NbNb * lapxNb_gam);


			/* ======================================== *
			 * Solve the Equation of Motion for Phases  *
			 * ======================================== */

			newGridN[2] = phi_del - current_dt * L_del * delF_delPhi_del;
			newGridN[3] = phi_mu  - current_dt * L_mu  * delF_delPhi_mu ;
			newGridN[4] = phi_lav - current_dt * L_lav * delF_delPhi_lav;

			/* =========================== *
			 * Solve for parallel tangents *
			 * =========================== */

			// Copy old values as initial guesses
			for (int i=NC+NP; i<fields(newGrid)-1; i++)
				newGridN[i] = oldGridN[i];

			rootsolver parallelTangentSolver;
			//double res = parallelTangentSolver.solve(newGridN);
			double res = parallelTangentSolver.solve(newGrid(n));

			if (res>root_tol) {
				// Invalid roots: substitute guesses.
				#ifdef _OPENMP
				#pragma omp critical (updCrit1)
				#endif
				{
					totBadTangents++;
				}

				guessGamma(newGrid(n));
				guessDelta(newGrid(n));
				guessMu(   newGrid(n));
				guessLaves(newGrid(n));
			}

			/* ======= *
			 * ~ fin ~ *
			 * ======= */
		}

		ghostswap(newGrid);

		// Update timestep based on interfacial velocity
		const double interfacialVelocity = maxVelocity(oldGrid, current_dt, newGrid);

		if (interfacialVelocity < speedlimit) {
			// Update succeeded: process solution
			current_time += current_dt; // increment before output block

			/* ====================================================================== *
			 * Collate summary & diagnostic data in OpenMP- and MPI-compatible manner *
			 * ====================================================================== */

			if (logcount == logstep) {
				logcount = 0;

				#ifdef MPI_VERSION
				unsigned int myBad(totBadTangents);
				MPI::COMM_WORLD.Reduce(&myBad, &totBadTangents, 1, MPI_UNSIGNED, MPI_SUM, 0);
				MPI::COMM_WORLD.Barrier();
				#endif

				vector<double> summary = summarize(oldGrid, current_dt, newGrid);

				if (rank==0)
					fprintf(cfile, "%11.9g\t%11.9g\t%11.9g\t%11.9g\t%11.9g\t%11.9g\t%11.9g\t%11.9g\t%11u\n",
					current_dt, summary[0], summary[1], summary[2], summary[3], summary[4], summary[5], interfacialVelocity, totBadTangents);

			}

			logcount++; // increment after output block

			if (adaptStep) {
				if (rank==0)
					fprintf(tfile, "%11.9g\t%11.9g\t%11.9g\n", interfacialVelocity, std::min(dtp, dtc) / current_dt, current_dt);

				if (interfacialVelocity > 0.0) {
					current_dt *= scaleup;
					current_dt = std::min(current_dt, timelimit);
				}
			}

			swap(oldGrid, newGrid);

		} else {
			// Update failed: solution is unstable
			if (adaptStep) {
				current_dt *= scaledn;
			} else {
				if (rank==0) {
					std::cerr<<"ERROR: Interfacial velocity exceeded speed limit, timestep is too aggressive!"<<std::endl;
					fclose(cfile);
					if (adaptStep)
						fclose(tfile);
				}
				MMSP::Abort(-1);
			}
		}
	}

	if (rank==0) {
		fclose(cfile);
		if (adaptStep) {
			fclose(tfile);
			print_progress(steps-1, steps);
		}
	}

}


} // namespace MMSP





double radius(const MMSP::vector<int>& a, const MMSP::vector<int>& b, const double& dx)
{
	double r = 0.0;
	for (int i=0; i<a.length() && i<b.length(); i++)
		r += std::pow(a[i]-b[i],2.0);
	return dx*std::sqrt(r);
}


double bellCurve(double x, double m, double s)
{
	return std::exp(-std::pow(x-m,2.0) / (2.0*s*s));
}


// Initial guesses for gamma, mu, and delta equilibrium compositions
template<typename T>
void guessGamma(MMSP::vector<T>& GRIDN)
{
	// Coarsely approximate gamma using a line compound with x_Nb = 0.015

	const T xcr = GRIDN[0];
	//const T xnb = GRIDN[1];
	const T xnb = 0.015;
	const T xni = std::max(epsilon, 1.0 - xcr - GRIDN[1]);

	GRIDN[5] = xcr/(xcr + xnb + xni);
	GRIDN[6] = xnb;
}


template<typename T>
void guessDelta(MMSP::vector<T>& GRIDN)
{
	// Coarsely approximate delta using a line compound with x_Ni = 0.75

	const T xcr = GRIDN[0];
	const T xnb = GRIDN[1];
	const T xni = 0.75;

	GRIDN[7] = xcr/(xcr + xnb + xni);
	GRIDN[8] = xnb/(xcr + xnb + xni);
}


template<typename T>
void guessMu(MMSP::vector<T>& GRIDN)
{
	// Coarsely approximate mu using a line compound with x_Nb=52.5%

	const T xcr = GRIDN[0];
	//const T xnb = GRIDN[1];
	const T xnb = 0.525;
	const T xni = std::max(epsilon, 1.0 - xcr - GRIDN[1]);

	GRIDN[9] = xcr/(xcr + xnb + xni);
	GRIDN[10] = xnb;
}


template<typename T>
void guessLaves(MMSP::vector<T>& GRIDN)
{
	// Coarsely approximate Laves using a line compound with x_Nb = 30.0%

	const T xcr = GRIDN[0];
	//const T xnb = GRIDN[1];
	const T xnb = 0.30;
	const T xni = std::max(epsilon, 1.0 - xcr - GRIDN[1]);

	GRIDN[11] = xcr/(xcr + xnb + xni);
	GRIDN[12] = xnb;
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

	for (int n=0; n<MMSP::nodes(GRID); n++) {
		MMSP::vector<int> x = MMSP::position(GRID, n);
		double matrixCr = xCr[4] * bellCurve(h*x[0], h*(Nx/2), bellCr); // centerline
		//    matrixCr += xCr[4] * bellCurve(h*x[0], 0,        bellCr); // left wall
		//    matrixCr += xCr[4] * bellCurve(h*x[0], h*Nx,     bellCr); // right wall
		double matrixNb = xNb[4] * bellCurve(h*x[0], h*(Nx/2), bellNb); // centerline
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
	double R = rprcp; //std::max(rdpltCr, rdpltNb);
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
	int R(rprcp); //std::max(rdpltCr, rdpltNb);
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
		double del = 4.3875e-9 / meshres; // empirically determined tanh profile thickness
		for (x[0] = origin[0] - R - 2*del; x[0] < origin[0] - R; x[0]++) {
			if (x[0] < x0(GRID) || x[0] >= x1(GRID))
				continue;
			double tanhprof = 0.5*(1.0 + std::tanh(double(x[0] - origin[0] + R + del)/(del)));
			for (x[1] = y0(GRID); x[1] < y1(GRID); x[1]++) {
				GRID(x)[0] = GRID(x)[0] - tanhprof*(GRID(x)[0] - xCr);
				GRID(x)[1] = GRID(x)[1] - tanhprof*(GRID(x)[1] - xNb);
				GRID(x)[pid] = GRID(x)[pid] - tanhprof*(GRID(x)[pid] - phi);
			}
		}
		for (x[0] = origin[0] + R; x[0] < origin[0] + R + 2*del; x[0]++) {
			if (x[0] < x0(GRID) || x[0] >= x1(GRID))
				continue;
			double tanhprof = 0.5*(1.0 + std::tanh(double(x[0] - origin[0] - R - del)/(del)));
			for (x[1] = y0(GRID); x[1] < y1(GRID); x[1]++) {
				GRID(x)[0] = xCr - tanhprof*(xCr - GRID(x)[0]);
				GRID(x)[1] = xNb - tanhprof*(xNb - GRID(x)[1]);
				GRID(x)[pid] = phi - tanhprof*(phi - GRID(x)[pid]);
			}
		}
	}

	return comp;
}


double h(const double p)
{
	return p * p * p * (6.0 * p * p - 15.0 * p + 10.0);
}


double hprime(const double p)
{
	return 30.0 * p * p * (1.0 - p) * (1.0 - p);
}


double gibbs(const MMSP::vector<double>& v)
{
	double n_del = h(fabs(v[2]));
	double n_mu  = h(fabs(v[3]));
	double n_lav = h(fabs(v[4]));
	double n_gam = 1.0 - n_del - n_mu - n_lav;

	MMSP::vector<double> vsq(NP);
	double diagonal = 0.0;

	for (int i=0; i<NP; i++) {
		vsq[i] = v[NC+i]*v[NC+i];
		diagonal += vsq[i];
	}

	double g  = n_gam * g_gam(v[5],  v[6]);
	       g += n_del * g_del(v[7],  v[8]);
	       g += n_mu  * g_mu( v[9],  v[10]);
	       g += n_lav * g_lav(v[11], v[12]);
	       g += omega_del * vsq[2-NC] * pow(1.0 - fabs(v[2]), 2);
	       g += omega_mu  * vsq[3-NC] * pow(1.0 - fabs(v[3]), 2);
	       g += omega_lav * vsq[4-NC] * pow(1.0 - fabs(v[4]), 2);

	// Trijunction penalty, using dot product
	g += alpha * (vsq*vsq - diagonal);

	return g;
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
	const double x_Cr = ((struct rparams*) params)->x_Cr;
	const double x_Nb = ((struct rparams*) params)->x_Nb;
	const double n_del = ((struct rparams*) params)->n_del;
	const double n_mu  = ((struct rparams*) params)->n_mu;
	const double n_lav = ((struct rparams*) params)->n_lav;
	const double n_gam = 1.0 - n_del - n_mu - n_lav;

	// Prepare variables
	const double C_gam_Cr = gsl_vector_get(x, 0);
	const double C_gam_Nb = gsl_vector_get(x, 1);

	const double C_del_Cr = gsl_vector_get(x, 2);
	const double C_del_Nb = gsl_vector_get(x, 3);

	const double C_mu_Cr  = gsl_vector_get(x, 4);
	const double C_mu_Nb  = gsl_vector_get(x, 5);

	const double C_lav_Cr = gsl_vector_get(x, 6);
	const double C_lav_Nb = gsl_vector_get(x, 7);


	// Prepare derivatives
	const double dgGdxCr = dg_gam_dxCr(C_gam_Cr, C_gam_Nb);
	const double dgGdxNb = dg_gam_dxNb(C_gam_Cr, C_gam_Nb);

	const double dgDdxCr = dg_del_dxCr(C_del_Cr, C_del_Nb);
	const double dgDdxNb = dg_del_dxNb(C_del_Cr, C_del_Nb);

	const double dgUdxCr = dg_mu_dxCr(C_mu_Cr, C_mu_Nb);
	const double dgUdxNb = dg_mu_dxNb(C_mu_Cr, C_mu_Nb);

	const double dgLdxCr = dg_lav_dxCr(C_lav_Cr, C_lav_Nb);
	const double dgLdxNb = dg_lav_dxNb(C_lav_Cr, C_lav_Nb);


	// Update vector
	gsl_vector_set(f, 0, x_Cr - n_gam*C_gam_Cr - n_del*C_del_Cr - n_mu*C_mu_Cr - n_lav*C_lav_Cr);
	gsl_vector_set(f, 1, x_Nb - n_gam*C_gam_Nb - n_del*C_del_Nb - n_mu*C_mu_Nb - n_lav*C_lav_Nb);

	gsl_vector_set(f, 2, dgGdxCr - dgDdxCr);
	gsl_vector_set(f, 3, dgGdxNb - dgDdxNb);

	gsl_vector_set(f, 4, dgGdxCr - dgUdxCr);
	gsl_vector_set(f, 5, dgGdxNb - dgUdxNb);

	gsl_vector_set(f, 6, dgGdxCr - dgLdxCr);
	gsl_vector_set(f, 7, dgGdxNb - dgLdxNb);


	return GSL_SUCCESS;
}


int parallelTangent_df(const gsl_vector* x, void* params, gsl_matrix* J)
{
	/* ========================================================= *
	 * Build Jacobian of Mass and Chemical Potential Differences *
	 * ========================================================= */

	// Prepare constants
	const double n_del = ((struct rparams*) params)->n_del;
	const double n_mu  = ((struct rparams*) params)->n_mu;
	const double n_lav = ((struct rparams*) params)->n_lav;
	const double n_gam = 1.0 - n_del - n_mu - n_lav;

	// Prepare variables
	#ifdef CALPHAD
	const double C_gam_Cr = gsl_vector_get(x, 0);
	const double C_gam_Nb = gsl_vector_get(x, 1);

	const double C_del_Cr = gsl_vector_get(x, 2);
	const double C_del_Nb = gsl_vector_get(x, 3);

	const double C_mu_Cr  = gsl_vector_get(x, 4);
	const double C_mu_Nb  = gsl_vector_get(x, 5);

	const double C_lav_Cr = gsl_vector_get(x, 6);
	const double C_lav_Nb = gsl_vector_get(x, 7);
	#endif

	gsl_matrix_set_zero(J);

	// Conservation of mass (Cr, Nb)
	gsl_matrix_set(J, 0, 0, -n_gam);
	gsl_matrix_set(J, 1, 1, -n_gam);

	gsl_matrix_set(J, 0, 2, -n_del);
	gsl_matrix_set(J, 1, 3, -n_del);

	gsl_matrix_set(J, 0, 4, -n_mu);
	gsl_matrix_set(J, 1, 5, -n_mu);

	gsl_matrix_set(J, 0, 6, -n_lav);
	gsl_matrix_set(J, 1, 7, -n_lav);


	// Equal chemical potential involving gamma phase (Cr, Nb, Ni)
	// Cross-derivatives must needs be equal, d2G_dxCrNb == d2G_dxNbCr. Cf. Arfken Sec. 1.9.
	#ifndef CALPHAD
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

	gsl_matrix_set(J, 6, 0, jac_gam_CrCr);
	gsl_matrix_set(J, 6, 1, jac_gam_CrNb);
	gsl_matrix_set(J, 7, 0, jac_gam_CrNb);
	gsl_matrix_set(J, 7, 1, jac_gam_NbNb);


	// Equal chemical potential involving delta phase (Cr, Nb)
	#ifndef CALPHAD
	const double jac_del_CrCr = d2g_del_dxCrCr();
	const double jac_del_CrNb = d2g_del_dxCrNb();
	const double jac_del_NbCr = jac_del_CrNb;
	const double jac_del_NbNb = d2g_del_dxNbNb();
	#else
	const double jac_del_CrCr = d2g_del_dxCrCr(C_del_Cr, C_del_Nb);
	const double jac_del_CrNb = d2g_del_dxCrNb(C_del_Cr, C_del_Nb);
	const double jac_del_NbCr = jac_del_CrNb;
	const double jac_del_NbNb = d2g_del_dxNbNb(C_del_Cr, C_del_Nb);
	#endif

	gsl_matrix_set(J, 2, 2, -jac_del_CrCr);
	gsl_matrix_set(J, 2, 3, -jac_del_CrNb);
	gsl_matrix_set(J, 3, 2, -jac_del_NbCr);
	gsl_matrix_set(J, 3, 3, -jac_del_NbNb);


	// Equal chemical potential involving mu phase (Cr, Ni)
	#ifndef CALPHAD
	const double jac_mu_CrCr = d2g_mu_dxCrCr();
	const double jac_mu_CrNb = d2g_mu_dxCrNb();
	const double jac_mu_NbCr = jac_mu_CrNb;
	const double jac_mu_NbNb = d2g_mu_dxNbNb();
	#else
	const double jac_mu_CrCr = d2g_mu_dxCrCr(C_mu_Cr, C_mu_Nb);
	const double jac_mu_CrNb = d2g_mu_dxCrNb(C_mu_Cr, C_mu_Nb);
	const double jac_mu_NbCr = jac_mu_CrNb;
	const double jac_mu_NbNb = d2g_mu_dxNbNb(C_mu_Cr, C_mu_Nb);
	#endif

	gsl_matrix_set(J, 4, 4, -jac_mu_CrCr);
	gsl_matrix_set(J, 4, 5, -jac_mu_CrNb);
	gsl_matrix_set(J, 5, 4, -jac_mu_NbCr);
	gsl_matrix_set(J, 5, 5, -jac_mu_NbNb);


	// Equal chemical potential involving Laves phase (Nb, Ni)
	#ifndef CALPHAD
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

	gsl_matrix_set(J, 6, 6, -jac_lav_CrCr);
	gsl_matrix_set(J, 6, 7, -jac_lav_CrNb);
	gsl_matrix_set(J, 7, 6, -jac_lav_NbCr);
	gsl_matrix_set(J, 7, 7, -jac_lav_NbNb);

	return GSL_SUCCESS;
}


int parallelTangent_fdf(const gsl_vector* x, void* params, gsl_vector* f, gsl_matrix* J)
{
	parallelTangent_f(x,  params, f);
	parallelTangent_df(x, params, J);

	return GSL_SUCCESS;
}


rootsolver::rootsolver() :
	n(8), // one equation per component per phase: eight total
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

	par.n_del = h(fabs(GRIDN[2]));
	par.n_mu =  h(fabs(GRIDN[3]));
	par.n_lav = h(fabs(GRIDN[4]));

	// copy initial guesses from grid

	gsl_vector_set(x, 0, static_cast<double>(GRIDN[5]));  // gamma Cr
	gsl_vector_set(x, 1, static_cast<double>(GRIDN[6]));  //       Nb

	gsl_vector_set(x, 2, static_cast<double>(GRIDN[7]));  // delta Cr
	gsl_vector_set(x, 3, static_cast<double>(GRIDN[8]));  //       Nb

	gsl_vector_set(x, 4, static_cast<double>(GRIDN[9]));  // mu    Cr
	gsl_vector_set(x, 5, static_cast<double>(GRIDN[10])); //       Nb

	gsl_vector_set(x, 6, static_cast<double>(GRIDN[11])); // Laves Cr
	gsl_vector_set(x, 7, static_cast<double>(GRIDN[12])); //       Nb


	gsl_multiroot_fdfsolver_set(solver, &mrf, x);

	do {
		iter++;
		status = gsl_multiroot_fdfsolver_iterate(solver);
		if (status) // extra points for finishing early!
			break;
		status = gsl_multiroot_test_residual(solver->f, tolerance);
	} while (status == GSL_CONTINUE && iter < maxiter);

	double residual = gsl_blas_dnrm2(solver->f);

	if (status == GSL_SUCCESS) {
		GRIDN[5]  = static_cast<T>(gsl_vector_get(solver->x, 0)); // gamma Cr
		GRIDN[6]  = static_cast<T>(gsl_vector_get(solver->x, 1)); //       Nb

		GRIDN[7]  = static_cast<T>(gsl_vector_get(solver->x, 2)); // delta Cr
		GRIDN[8]  = static_cast<T>(gsl_vector_get(solver->x, 3)); //       Nb

		GRIDN[9]  = static_cast<T>(gsl_vector_get(solver->x, 4)); // mu    Cr
		GRIDN[10] = static_cast<T>(gsl_vector_get(solver->x, 5)); //       Nb

		GRIDN[11] = static_cast<T>(gsl_vector_get(solver->x, 6)); // Laves Cr
		GRIDN[12] = static_cast<T>(gsl_vector_get(solver->x, 7)); //       Nb
	}

	return residual;
}


rootsolver::~rootsolver()
{
	gsl_multiroot_fdfsolver_free(solver);
	gsl_vector_free(x);
}



template<int dim,class T>
double maxVelocity(MMSP::grid<dim, MMSP::vector<T> > const & oldGrid, double const dt,
                   MMSP::grid<dim, MMSP::vector<T> > const & newGrid)
{
	double vmax = -epsilon;
	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (int n=0; n<MMSP::nodes(newGrid); n++) {
		MMSP::vector<int> x = MMSP::position(newGrid,n);
		MMSP::vector<T>& gridN = newGrid(n);

		const double phFrac[NP] = {h(fabs(gridN[2])), h(fabs(gridN[3])), h(fabs(gridN[4]))};

		for (int i=0; i<NP; i++) {
			if (phFrac[i] > 0.05 && phFrac[i] < 0.95) {
				MMSP::vector<double> gradPhi = MMSP::gradient(newGrid, x, i+NC);

				const double magGrad = std::sqrt(gradPhi * gradPhi);
				double dphidt = std::fabs(gridN[i+NC] - oldGrid(n)[i+NC]) / dt;
				double myv = (dphidt>epsilon && magGrad>epsilon) ? dphidt / magGrad : 0.0;
				#ifdef _OPENMP
				#pragma omp critical
				#endif
				{
					vmax = std::max(vmax, myv);
				}
			}
		}

	}

	#ifdef MPI_VERSION
	double myv(vmax);
	MPI::COMM_WORLD.Allreduce(&myv, &vmax, 1, MPI_DOUBLE, MPI_MAX);
	#endif

	return std::max(0.0, vmax);
}


template<int dim,class T>
MMSP::vector<double> summarize(MMSP::grid<dim, MMSP::vector<T> > const & oldGrid, double const dt,
                               MMSP::grid<dim, MMSP::vector<T> >& newGrid)
{
	double Ntot = 1.0;
	double dV = 1.0;
	for (int d=0; d<dim; d++) {
		Ntot *= double(MMSP::g1(newGrid, d) - MMSP::g0(newGrid, d));
		dV *= MMSP::dx(newGrid, d);
	}
	MMSP::vector<double> summary(8, 0.0);

	#ifdef _OPENMP
	#pragma omp parallel for shared(summary)
	#endif
	for (int n=0; n<MMSP::nodes(newGrid); n++) {
		MMSP::vector<int> x = MMSP::position(newGrid,n);
		MMSP::vector<T>& gridN = newGrid(n);
		MMSP::vector<double> mySummary(8, 0.0);
		const double phi_del = h(fabs(gridN[2]));
		const double phi_mu  = h(fabs(gridN[3]));
		const double phi_lav = h(fabs(gridN[4]));

		MMSP::vector<double> gradPhi_del = MMSP::gradient(newGrid, x, 2);
		MMSP::vector<double> gradPhi_mu  = MMSP::gradient(newGrid, x, 3);
		MMSP::vector<double> gradPhi_lav = MMSP::gradient(newGrid, x, 4);

		mySummary[0] = gridN[0]; // x_Cr
		mySummary[1] = gridN[1]; // x_Nb
		mySummary[2] = 1.0 - phi_del - phi_mu - phi_lav; // phi_gam
		mySummary[3] = phi_del;
		mySummary[4] = phi_mu;
		mySummary[5] = phi_lav;
		// free energy density:
		mySummary[6] = dV * (gibbs(gridN) + kappa_del * (gradPhi_del * gradPhi_del)
		                                  + kappa_mu  * (gradPhi_mu  * gradPhi_mu )
		                                  + kappa_lav * (gradPhi_lav * gradPhi_lav));

		double magGrad[NP] = {std::sqrt(gradPhi_del * gradPhi_del),
		                      std::sqrt(gradPhi_mu  * gradPhi_mu ),
		                      std::sqrt(gradPhi_lav * gradPhi_lav)
		                     };

		double myVelocity = 0.0;
		double secondaries = 1.0 / (phi_del + phi_mu + phi_lav);
		for (int i=0; i<NP; i++) {
			if (magGrad[i] > 0.05 && magGrad[i] < 0.95) {
				double dphidt = std::fabs(gridN[i+NC] - oldGrid(n)[i+NC]) / dt;
				double v = (magGrad[i]>epsilon) ? dphidt / magGrad[i] : 0.0;
				myVelocity += v * mySummary[NC+1+i] * secondaries;
				//myVelocity = std::max(myVelocity, v);
			}
		}
		// Record local velocity
		gridN[fields(newGrid)-1] = static_cast<T>(myVelocity);

		// Sum up mass and phase fractions
		#ifdef _OPENMP
		#pragma omp critical (sumCrit1)
		#endif
		{
			summary += mySummary;
		}

		// Get maximum interfacial velocity
		#ifdef _OPENMP
		#pragma omp critical (sumCrit2)
		#endif
		{
			summary[7] = std::max(myVelocity, summary[7]);
		}
	}

	for (int i=0; i<summary.length()-2; i++)
		summary[i] /= Ntot;

	#ifdef MPI_VERSION
	MPI::COMM_WORLD.Barrier();
	MMSP::vector<double> tmpSummary(summary);

	for (int i=0; i<summary.length()-2; i++) {
		MPI::COMM_WORLD.Reduce(&tmpSummary[i], &summary[i], 1, MPI_DOUBLE, MPI_SUM, 0);
		MPI::COMM_WORLD.Barrier(); // probably not necessary
	}
	MPI::COMM_WORLD.Reduce(&tmpSummary[6], &summary[6], 1, MPI_DOUBLE, MPI_SUM, 0); // free energy
	MPI::COMM_WORLD.Allreduce(&tmpSummary[7], &summary[7], 1, MPI_DOUBLE, MPI_MAX); // maximum velocity
	#endif

	return summary;
}

#endif

#include"MMSP.main.hpp"
