// alloy625.cpp
// Algorithms for 2D and 3D isotropic Cr-Nb-Ni alloy phase transformations
// Questions/comments to trevor.keller@nist.gov (Trevor Keller)

// This implementation depends on the GNU Scientific Library
// for multivariate root finding algorithms, and
// The Mesoscale Microstructure Simulation Project for
// high-performance grid operations in parallel. Use of these
// software packages does not constitute endorsement by the
// National Institute of Standards and Technology.


#ifndef ALLOY625_UPDATE
#define ALLOY625_UPDATE
#include<cmath>
#include<random>
#include<gsl/gsl_blas.h>
#include<gsl/gsl_vector.h>
#include<gsl/gsl_multiroots.h>
#include<gsl/gsl_interp2d.h>
#include<gsl/gsl_spline2d.h>

#include"MMSP.hpp"
#include"alloy625.hpp"
#include"energy625.c"

// Number of phases and components (for array allocation)
#define NP 4
#define NC 2

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

/* Based on experiments (EDS) and simulations (DICTRA), additively manufactured IN625
 * has the following compositions:
 *
 * Element  Nominal  Interdendritic (mol %)
 * Cr+Mo      30%      31%
 * Nb          2%      13%
 * Ni         68%      56%
 */


// Define equilibrium phase compositions at global scope. Gamma is nominally 30% Cr, 2% Nb,
// but microsegregation should deplete that somewhat. Compare against the reported total
// system composition and adjust gamma accordingly in the following arrays.
//                        gamma    delta    mu     laves   Enriched gamma
const double xCr[NP+1] = {0.2945,  0.00625, 0.025, 0.32,   0.31};
const double xNb[NP+1] = {0.00254,    0.25, 0.525, 0.33,   0.13};
const double xCrdep = 0.5*xCr[0]; // leftover Cr in depleted gamma phase near precipitate particle
const double xNbdep = 0.5*xNb[0]; // leftover Nb in depleted gamma phase near precipitate particle

// Define st.dev. of Gaussians for alloying element segregation
//                         Cr      Nb
const double bell[NC] = {150e-9, 50e-9}; // est. between 80-200 nm from SEM

// Kinetic and model parameters
const double meshres = 1.25e-9;    // grid spacing (m)
const double Vm = 1.0e-5;         // molar volume (m^3/mol)
const double alpha = 1.07e11;     // three-phase coexistence coefficient (J/m^3)

const double M_Cr = 1.6e-17;      // mobility in FCC Ni (mol^2/Nsm^2)
const double M_Nb = 1.7e-18;      // mobility in FCC Ni (mol^2/Nsm^2)

const double L_del = 2.904e-11;   // numerical mobility (m^2/Nsm^2)
const double L_mu  = 2.904e-11;   // numerical mobility (m^2/Nsm^2)
const double L_lav = 2.904e-11;   // numerical mobility (m^2/Nsm^2)

const double kappa_del = 1.24e-8; // gradient energy coefficient (J/m)
const double kappa_mu  = 1.24e-8; // gradient energy coefficient (J/m)
const double kappa_lav = 1.24e-8; // gradient energy coefficient (J/m)

const double sigma_del = 1.01;    // J/m^2
const double sigma_mu  = 1.01;    // J/m^2
const double sigma_lav = 1.01;    // J/m^2

const double width_factor = 2.2;  // 2.2 if interface is [0.1,0.9]; 2.94 if [0.05,0.95]
const double ifce_width = 7.0*meshres; // ensure at least 7 points through the interface
const double omega_del = 3.0 * width_factor * sigma_del / ifce_width; // 9.5e8;  // multiwell height (m^2/Nsm^2)
const double omega_mu  = 3.0 * width_factor * sigma_del / ifce_width; // 9.5e8;  // multiwell height (m^2/Nsm^2)
const double omega_lav = 3.0 * width_factor * sigma_del / ifce_width; // 9.5e8;  // multiwell height (m^2/Nsm^2)

// Numerical considerations
const bool   useNeumann = false;  // apply zero-flux boundaries (Neumann type)
const double epsilon = 1.0e-12;   // what to consider zero to avoid log(c) explosions
const double noise_amp = 0.0;     // 1.0e-8;
const double init_amp = 0.0;      // 1.0e-8;

const double CFL = 1.0/500.0;     // numerical stability
const double dtp = CFL*(meshres*meshres)/(4.0*L_del*kappa_del); // transformation-limited timestep
const double dtc = CFL*(meshres*meshres)/(4.0*Vm*Vm*M_Cr); // diffusion-limited timestep
const double dt = std::min(dtp, dtc);

namespace MMSP
{

void generate(int dim, const char* filename)
{
	int rank=0;
	#ifdef MPI_VERSION
	rank = MPI::COMM_WORLD.Get_rank();
	#endif
	// Utilize Mersenne Twister from C++11 standard
	std::mt19937_64 mt_rand(time(NULL)+rank);
	std::uniform_real_distribution<double> real_gen(-1,1);

	double totCr  = 0.0;
	double totNb  = 0.0;
	double totDel = 0.0;
	double totMu  = 0.0;
	double totLav = 0.0;
	double totF   = 0.0;

	std::ofstream cfile;
	if (rank==0)
		cfile.open("c.log",std::ofstream::out);

	if (dim==2) {
		// Construct grid
		const int Nx = 768; // divisible by 12 and 64
		const int Ny = 768;
		double dV = 1.0;
		double Ntot = 1.0;
		GRID2D initGrid(13, 0, Nx, 0, Ny);
		for (int d=0; d<dim; d++) {
			dx(initGrid,d)=meshres;
			dV *= meshres;
			Ntot *= g1(initGrid, d) - g0(initGrid, d);
			if (useNeumann) {
				b0(initGrid, d) = Neumann;
				b1(initGrid, d) = Neumann;
			}
		}

		// Precipitate radii: minimum for stability is 7.5 nm, or 1.5*dx. Ensure that radius is greater than 7*dx, or the
		// particle will dissolve as the initially sharp interface becomes diffuse. Numerical dissolution is unphysical.
		const double rPrecip[NP-1] = {1.5*7.5e-9 / dx(initGrid,0),  // delta
		                              1.5*7.5e-9 / dx(initGrid,0),  // mu
		                              1.5*7.5e-9 / dx(initGrid,0)}; // Laves


		// depletion region surrounding precipitates
		double rDepltCr[NP-1] = {0.0};
		double rDepltNb[NP-1] = {0.0};
		for (int i=0; i<NP-1; i++) {
			rDepltCr[i] = std::sqrt(1.0+2.0*xCr[i+1]/xCrdep) * rPrecip[i];
			rDepltNb[i] = std::sqrt(1.0+2.0*xNb[i+1]/xNbdep) * rPrecip[i];
		}


		// Sanity check on system size and  particle spacing
		if (rank==0 && dtp < dtc)
			std::cout<<"Timestep (dt="<<dt<<") is transformation limited (diffusion-limited dt="<<dtc<<")."<<std::endl;
		else if (rank==0 && dtc < dtp)
			std::cout<<"Timestep (dt="<<dt<<") is diffusion limited (transformation-limited dt="<<dtp<<")."<<std::endl;

		for (int i=0; i<NP-1; i++) {
			/*
			double rmax = std::max(rDepltCr[i], rDepltNb[i]);
			if (rmax > Ny/2)
				std::cerr<<"Warning: domain too small to accommodate phase "<<i<<", expand beyond "<<2.0*rmax<<" pixels."<<std::endl;
			*/
			if (rPrecip[i] > Ny/2)
				std::cerr<<"Warning: domain too small to accommodate phase "<<i<<", expand beyond "<<2.0*rPrecip[i]<<" pixels."<<std::endl;
		}

		vector<int> x(2, 0);

		// Initialize matrix (gamma phase): bell curve along x, each stripe in y is identical (with small fluctuations)
		for (x[0]=x0(initGrid); x[0]<x1(initGrid); x[0]++) {
			double matrixCr = (xCr[4]-xCr[0]) * bellCurve(dx(initGrid,0)*x[0], dx(initGrid,0)*(Nx/2), bell[0]);          // centerline
			//    matrixCr += 0.5*(xCr[4]-xCr[0]) * bellCurve(dx(initGrid,0)*x[0], dx(initGrid,0)*(Nx/2 + 10), bell[0]); // centerline
			//    matrixCr += (xCr[4]-xCr[0]) * bellCurve(dx(initGrid,0)*x[0], 0,                          bell[0]);     // left wall
			//    matrixCr += (xCr[4]-xCr[0]) * bellCurve(dx(initGrid,0)*x[0], dx(initGrid,0)*Nx,          bell[0]);     // right wall
			double matrixNb = (xNb[4]-xNb[0]) * bellCurve(dx(initGrid,0)*x[0], dx(initGrid,0)*(Nx/2), bell[1]);          // centerline
			//    matrixNb += 0.5*(xNb[4]-xNb[0]) * bellCurve(dx(initGrid,0)*x[0], dx(initGrid,0)*(Nx/2 + 10), bell[1]); // centerline
			//    matrixNb += (xNb[4]-xNb[0]) * bellCurve(dx(initGrid,0)*x[0], 0,                          bell[1]);     // left wall
			//    matrixNb += (xNb[4]-xNb[0]) * bellCurve(dx(initGrid,0)*x[0], dx(initGrid,0)*Nx,          bell[1]);     // right wall

			for (x[1]=y0(initGrid); x[1]<y1(initGrid); x[1]++) {
				initGrid(x)[0] = matrixCr + xCr[0] * (1.0 + init_amp*real_gen(mt_rand));
				initGrid(x)[1] = matrixNb + xNb[0] * (1.0 + init_amp*real_gen(mt_rand));

				// tiny bit of noise to avoid zeros
				for (int i=NC; i<NC+NP-1; i++)
					initGrid(x)[i] = init_amp*real_gen(mt_rand);
				for (int i=NC+NP-1; i<fields(initGrid); i++)
					initGrid(x)[i] = 0.0;
			}
		}


		// Seed precipitates: four of each, arranged along the centerline to allow for pairwise coarsening.
		const int xoffset = 16 * (5.0e-9 / meshres); //  80 nm
		const int yoffset = 32 * (5.0e-9 / meshres); // 160 nm
		if (Ny > 5.5*yoffset) { // seed full pairwise pattern
			vector<int> origin(2, 0);

			// Initialize delta precipitates
			int j = 0;
			origin[0] = Nx / 2;
			origin[1] = Ny - yoffset + yoffset/2;
			embedParticle(initGrid, origin, j+2, rPrecip[j], rDepltCr[j], rDepltNb[j], xCr[j+1], xNb[j+1], xCrdep, xNbdep, 1.0 - epsilon);
			origin[0] = Nx/2 + xoffset;
			origin[1] = Ny - 5*yoffset + yoffset/2;
			embedParticle(initGrid, origin, j+2, rPrecip[j], rDepltCr[j], rDepltNb[j], xCr[j+1], xNb[j+1], xCrdep, xNbdep,  1.0 - epsilon);
			origin[0] = Nx/2;
			origin[1] = Ny - 3*yoffset + yoffset/2;
			embedParticle(initGrid, origin, j+2, rPrecip[j], rDepltCr[j], rDepltNb[j], xCr[j+1], xNb[j+1], xCrdep, xNbdep, -1.0 + epsilon);
			origin[0] = Nx/2 - xoffset;
			origin[1] = Ny - 6*yoffset + yoffset/2;
			embedParticle(initGrid, origin, j+2, rPrecip[j], rDepltCr[j], rDepltNb[j], xCr[j+1], xNb[j+1], xCrdep, xNbdep, -1.0 + epsilon);

			// Initialize mu precipitates
			j = 1;
			origin[0] = Nx / 2;
			origin[1] = Ny - 2*yoffset + yoffset/2;
			embedParticle(initGrid, origin, j+2, rPrecip[j], rDepltCr[j], rDepltNb[j], xCr[j+1], xNb[j+1], xCrdep, xNbdep,  1.0 - epsilon);
			origin[0] = Nx/2 - xoffset;
			origin[1] = Ny - 4*yoffset + yoffset/2;
			embedParticle(initGrid, origin, j+2, rPrecip[j], rDepltCr[j], rDepltNb[j], xCr[j+1], xNb[j+1], xCrdep, xNbdep,  1.0 - epsilon);
			origin[0] = Nx/2 + xoffset;
			origin[1] = Ny - 3*yoffset + yoffset/2;
			embedParticle(initGrid, origin, j+2, rPrecip[j], rDepltCr[j], rDepltNb[j], xCr[j+1], xNb[j+1], xCrdep, xNbdep, -1.0 + epsilon);
			origin[0] = Nx/2;
			origin[1] = Ny - 5*yoffset + yoffset/2;
			embedParticle(initGrid, origin, j+2, rPrecip[j], rDepltCr[j], rDepltNb[j], xCr[j+1], xNb[j+1], xCrdep, xNbdep, -1.0 + epsilon);

			// Initialize Laves precipitates
			j = 2;
			origin[0] = Nx/2 + xoffset;
			origin[1] = Ny - yoffset + yoffset/2;
			embedParticle(initGrid, origin, j+2, rPrecip[j], rDepltCr[j], rDepltNb[j], xCr[j+1], xNb[j+1], xCrdep, xNbdep,  1.0 - epsilon);
			origin[0] = Nx/2;
			origin[1] = Ny - 4*yoffset + yoffset/2;
			embedParticle(initGrid, origin, j+2, rPrecip[j], rDepltCr[j], rDepltNb[j], xCr[j+1], xNb[j+1], xCrdep, xNbdep,  1.0 - epsilon);
			origin[0] = Nx/2 - xoffset;
			origin[1] = Ny - 2*yoffset + yoffset/2;
			embedParticle(initGrid, origin, j+2, rPrecip[j], rDepltCr[j], rDepltNb[j], xCr[j+1], xNb[j+1], xCrdep, xNbdep, -1.0 + epsilon);
			origin[0] = Nx/2;
			origin[1] = Ny - 6*yoffset + yoffset/2;
			embedParticle(initGrid, origin, j+2, rPrecip[j], rDepltCr[j], rDepltNb[j], xCr[j+1], xNb[j+1], xCrdep, xNbdep, -1.0 + epsilon);
		} else { // seed one row with all phi>0
			vector<int> origin(2, 0);

			// Initialize delta precipitates
			int j = 0;
			origin[0] = Nx / 2;
			origin[1] = Ny / 2;
			embedParticle(initGrid, origin, j+2, rPrecip[j], rDepltCr[j], rDepltNb[j], xCr[j+1], xNb[j+1], xCrdep, xNbdep,  1.0 - epsilon);

			// Initialize mu precipitates
			j = 1;
			origin[0] = Nx / 2 - xoffset;
			embedParticle(initGrid, origin, j+2, rPrecip[j], rDepltCr[j], rDepltNb[j], xCr[j+1], xNb[j+1], xCrdep, xNbdep,  1.0 - epsilon);

			// Initialize Laves precipitates
			j = 2;
			origin[0] = Nx / 2 + xoffset;
			embedParticle(initGrid, origin, j+2, rPrecip[j], rDepltCr[j], rDepltNb[j], xCr[j+1], xNb[j+1], xCrdep, xNbdep,  1.0 - epsilon);
		}


		// Initialize compositions in a manner compatible with OpenMP and MPI parallelization
		#ifdef MPI_VERSION
		for (int n=0; n<nodes(initGrid); n++) {
			guessGamma(initGrid, n, mt_rand, real_gen, noise_amp);
			guessDelta(initGrid, n, mt_rand, real_gen, noise_amp);
			guessMu(   initGrid, n, mt_rand, real_gen, noise_amp);
			guessLaves(initGrid, n, mt_rand, real_gen, noise_amp);
		}
		#else
		#pragma omp parallel for
		for (int n=0; n<nodes(initGrid); n++) {
			#pragma omp critical
			{
				guessGamma(initGrid, n, mt_rand, real_gen, noise_amp);
			}
			#pragma omp critical
			{
				guessDelta(initGrid, n, mt_rand, real_gen, noise_amp);
			}
			#pragma omp critical
			{
				guessMu(   initGrid, n, mt_rand, real_gen, noise_amp);
			}
			#pragma omp critical
			{
				guessLaves(initGrid, n, mt_rand, real_gen, noise_amp);
			}
		}
		#endif


		#ifndef MPI_VERSION
		#pragma omp parallel for
		#endif
		for (int n=0; n<nodes(initGrid); n++) {
			rootsolver CommonTangentSolver;
			CommonTangentSolver.solve(initGrid, n);
		}

		ghostswap(initGrid);

		#ifndef MPI_VERSION
		#pragma omp parallel for
		#endif
		for (int n=0; n<nodes(initGrid); n++) {
			vector<double> gradPhi_del = gradient(initGrid, x, 2);
			vector<double> gradPhi_mu  = gradient(initGrid, x, 3);
			vector<double> gradPhi_lav = gradient(initGrid, x, 4);

			double myCr = initGrid(n)[0];
			double myNb = initGrid(n)[1];
			double myDel = fabs(initGrid(n)[2]);
			double myMu  = fabs(initGrid(n)[3]);
			double myLav = fabs(initGrid(n)[4]);
			double myf = dV*(gibbs(initGrid(n)) + kappa_del * (gradPhi_del * gradPhi_del)
			                                    + kappa_mu  * (gradPhi_mu  * gradPhi_mu )
			                                    + kappa_lav * (gradPhi_lav * gradPhi_lav));

			#ifndef MPI_VERSION
			#pragma omp critical
			{
			#endif
				totCr  += myCr;  // total Cr mass
				totNb  += myNb;  // total Nb mass
				totDel += myDel; // total delta volume
				totMu  += myMu;  // total mu volume
				totLav += myLav; // total Laves volume
				totF   += myf;   // total free energy
			#ifndef MPI_VERSION
			}
			#endif
		}

		totCr /= Ntot;
		totNb /= Ntot;

		#ifdef MPI_VERSION
		double myCr(totCr);
		double myNb(totNb);
		double myDel(totDel);
		double myMu(totMu);
		double myLav(totLav);
		double myF(totF);
		MPI::COMM_WORLD.Reduce(&myCr,  &totCr,  1, MPI_DOUBLE, MPI_SUM, 0);
		MPI::COMM_WORLD.Reduce(&myNb,  &totNb,  1, MPI_DOUBLE, MPI_SUM, 0);
		MPI::COMM_WORLD.Reduce(&myDel, &totDel, 1, MPI_DOUBLE, MPI_SUM, 0);
		MPI::COMM_WORLD.Reduce(&myMu,  &totMu,  1, MPI_DOUBLE, MPI_SUM, 0);
		MPI::COMM_WORLD.Reduce(&myLav, &totLav, 1, MPI_DOUBLE, MPI_SUM, 0);
		MPI::COMM_WORLD.Reduce(&myF,   &totF,   1, MPI_DOUBLE, MPI_SUM, 0);
		#endif

		#ifdef MPI_VERSION
		#endif
		if (rank==0) {
			cfile<<totCr<<'\t'<<totNb<<'\t'<<totDel<<'\t'<<totMu<<'\t'<<totLav<<'\t'<<totF<<std::endl;
			cfile.close();
		}

		print_values(initGrid);
		output(initGrid,filename);

	} else
		std::cerr<<"Error: "<<dim<<"-dimensional grids unsupported."<<std::endl;
}

template <int dim, typename T> void update(grid<dim,vector<T> >& oldGrid, int steps)
{
	int rank=0;
	#ifdef MPI_VERSION
	rank = MPI::COMM_WORLD.Get_rank();
	#endif

	ghostswap(oldGrid);
	grid<dim,vector<T> > newGrid(oldGrid);
	grid<dim,vector<T> > potGrid(oldGrid, NC); // storage for chemical potentials

	// Construct the parallel tangent solver
	std::mt19937_64 mt_rand(time(NULL)+rank);
	std::uniform_real_distribution<double> real_gen(-1,1);

	double dV = 1.0;
	double Ntot = 1.0;
	for (int d=0; d<dim; d++) {
		dx(oldGrid,d) = meshres;
		dx(newGrid,d) = meshres;
		dx(potGrid,d) = meshres;
		dV *= dx(oldGrid,d);
		Ntot *= g1(oldGrid, d) - g0(oldGrid, d);
		if (useNeumann && x0(oldGrid,d) == g0(oldGrid,d)) {
			b0(oldGrid,d) = Neumann;
			b0(newGrid,d) = Neumann;
			b0(potGrid,d) = Neumann;
		} else if (useNeumann && x1(oldGrid,d) == g1(oldGrid,d)) {
			b1(oldGrid,d) = Neumann;
			b1(newGrid,d) = Neumann;
			b1(potGrid,d) = Neumann;
		}
	}

	std::ofstream cfile;
	if (rank==0)
		cfile.open("c.log",std::ofstream::out | std::ofstream::app);

	for (int step=0; step<steps; step++) {
		if (rank==0)
			print_progress(step, steps);

		#ifndef MPI_VERSION
		#pragma omp parallel for
		#endif
		for (int n=0; n<nodes(oldGrid); n++) {
			const T& C_gam_Cr = oldGrid(n)[5];
			const T& C_gam_Nb = oldGrid(n)[6];
			const T  C_gam_Ni = 1.0 - C_gam_Cr - C_gam_Nb;

			potGrid(n)[0] = dg_gam_dxCr(C_gam_Cr, C_gam_Nb, C_gam_Ni);
			potGrid(n)[1] = dg_gam_dxNb(C_gam_Cr, C_gam_Nb, C_gam_Ni);
		}

		ghostswap(potGrid);

		#ifndef MPI_VERSION
		#pragma omp parallel for
		#endif
		for (int n=0; n<nodes(oldGrid); n++) {
			/* ============================================== *
			 * Point-wise kernel for parallel PDE integration *
			 * ============================================== */

			vector<int> x = position(oldGrid,n);


			/* ============================================= *
			 * Solve the Equation of Motion for Compositions *
			 * ============================================= */

			const T& x_Cr = oldGrid(n)[0]; // molar fraction of Cr + Mo
			const T& x_Nb = oldGrid(n)[1]; // molar fraction of Nb

			double gradSqPot_Cr = laplacian(potGrid, x, 0);
			double gradSqPot_Nb = laplacian(potGrid, x, 1);

			newGrid(n)[0] = x_Cr + dt * Vm*Vm * M_Cr * gradSqPot_Cr;
			newGrid(n)[1] = x_Nb + dt * Vm*Vm * M_Nb * gradSqPot_Nb;


			/* ============================== *
			 * Kick stray values into (0, 1) *
			 * ============================== */

			for (int i=0; i<NC; i++) {
				if (newGrid(n)[i] < 0.0)
					newGrid(n)[i] = epsilon;
				if (newGrid(n)[i] > 1.0)
					newGrid(n)[i] = 1.0 - epsilon;
			}

			/* ======================================== *
			 * Solve the Equation of Motion for Phases  *
			 * ======================================== */

			const T& phi_del  = oldGrid(n)[2]; // phase fraction of delta
			const T& phi_mu   = oldGrid(n)[3]; // phase fraction of mu
			const T& phi_lav  = oldGrid(n)[4]; // phase fraction of Laves

			const T& C_gam_Cr = oldGrid(n)[5]; // Cr molar fraction in pure gamma
			const T& C_gam_Nb = oldGrid(n)[6]; // Nb molar fraction in pure gamma
			const T  C_gam_Ni = 1.0 - C_gam_Cr - C_gam_Nb;

			const T& C_del_Cr = oldGrid(n)[7]; // Cr molar fraction in pure delta
			const T& C_del_Nb = oldGrid(n)[8]; // Nb molar fraction in pure delta

			const T& C_mu_Cr  = oldGrid(n)[9];  // Cr molar fraction in pure mu
			const T& C_mu_Nb  = oldGrid(n)[10]; // Nb molar fraction in pure mu
			const T  C_mu_Ni  = 1.0 - C_mu_Cr - C_mu_Nb;

			const T& C_lav_Cr = oldGrid(n)[11]; // Cr molar fraction in pure Laves
			const T& C_lav_Nb = oldGrid(n)[12]; // Nb molar fraction in pure Laves
			const T  C_lav_Ni = 1.0 - C_lav_Cr - C_lav_Nb;

			double df_dphi_del = sign(phi_del) * (-hprime(fabs(phi_del))*g_gam(C_gam_Cr, C_gam_Nb, C_gam_Ni)
			                                      +hprime(fabs(phi_del))*g_del(C_del_Cr, C_del_Nb)
			                                      -2.0*omega_del*phi_del*phi_del*(1.0-fabs(phi_del)))
			                    + 2.0*omega_del*phi_del*pow(1.0-fabs(phi_del),2)
			                    + alpha*fabs(phi_del)*(phi_mu*phi_mu + phi_lav*phi_lav);


			double df_dphi_mu = sign(phi_mu) * (-hprime(fabs(phi_mu))*g_gam(C_gam_Cr, C_gam_Nb, C_gam_Ni)
			                                   + hprime(fabs(phi_mu))*g_mu(C_mu_Cr, C_mu_Nb, C_mu_Ni)
			                                   - 2.0*omega_mu*phi_mu*phi_mu*(1.0-fabs(phi_mu)))
			                    + 2.0*omega_mu*phi_mu*pow(1.0-fabs(phi_mu),2)
			                    + alpha*fabs(phi_mu)*(phi_del*phi_del + phi_lav*phi_lav);


			double df_dphi_lav = sign(phi_lav) * (-hprime(fabs(phi_lav))*g_gam(C_gam_Cr, C_gam_Nb, C_gam_Ni)
			                                     + hprime(fabs(phi_lav))*g_lav(C_lav_Nb, C_lav_Ni)
			                                     - 2.0*omega_lav*phi_lav*phi_lav*(1.0-fabs(phi_lav)))
			                    + 2.0*omega_lav*phi_lav*pow(1.0-fabs(phi_lav),2)
			                    + alpha*fabs(phi_lav)*(phi_del*phi_del + phi_mu*phi_mu);


			double gradSqPhi_del = laplacian(oldGrid, x, 2);
			double gradSqPhi_mu  = laplacian(oldGrid, x, 3);
			double gradSqPhi_lav = laplacian(oldGrid, x, 4);


			newGrid(n)[2] = phi_del - dt * L_del * (df_dphi_del - kappa_del*gradSqPhi_del);
			newGrid(n)[3] = phi_mu  - dt * L_mu  * (df_dphi_mu  - kappa_mu *gradSqPhi_mu );
			newGrid(n)[4] = phi_lav - dt * L_lav * (df_dphi_lav - kappa_lav*gradSqPhi_lav);


			/* ============================== *
			 * Kick stray values into (-1, 1) *
			 * ============================== */

			for (int i=NC; i<NC+NP-1; i++) {
				if (newGrid(n)[i] < -1.0)
					newGrid(n)[i] = -1.0 + epsilon;
				if (newGrid(n)[i] > 1.0)
					newGrid(n)[i] = 1.0 - epsilon;
			}


			/* ====================================== *
			 * Guess at parallel tangent compositions *
			 * ====================================== */

			#ifdef MPI_VERSION
			guessGamma(newGrid, n, mt_rand, real_gen, noise_amp);
			guessDelta(newGrid, n, mt_rand, real_gen, noise_amp);
			guessMu(   newGrid, n, mt_rand, real_gen, noise_amp);
			guessLaves(newGrid, n, mt_rand, real_gen, noise_amp);
			#else
			#pragma omp critical
			{
				guessGamma(newGrid, n, mt_rand, real_gen, noise_amp);
			}
			#pragma omp critical
			{
				guessDelta(newGrid, n, mt_rand, real_gen, noise_amp);
			}
			#pragma omp critical
			{
				guessMu(   newGrid, n, mt_rand, real_gen, noise_amp);
			}
			#pragma omp critical
			{
				guessLaves(newGrid, n, mt_rand, real_gen, noise_amp);
			}
			#endif
			/*
			// It's possible the previous concentration is already the best guess -- let them evolve, instead of resetting each time?
			// Historically, this does not work well...
			for (int i=NC+NP-1; i<fields(oldGrid); i++)
				newGrid(n)[i] = oldGrid(n)[i];
			*/


			/* =========================== *
			 * Solve for parallel tangents *
			 * =========================== */

			rootsolver CommonTangentSolver;
			CommonTangentSolver.solve(newGrid, n);


			/* ======= *
			 * ~ fin ~ *
			 * ======= */
		}
		swap(oldGrid,newGrid);
		ghostswap(oldGrid);


		/* ====================================================================== *
		 * Collate summary & diagnostic data in OpenMP- and MPI-compatible manner *
		 * ====================================================================== */

		double totCr = 0.0;
		double totNb = 0.0;
		double totDel = 0.0;
		double totMu  = 0.0;
		double totLav = 0.0;
		double totF  = 0.0;

		#ifndef MPI_VERSION
		#pragma omp parallel for
		#endif
		for (int n=0; n<nodes(oldGrid); n++) { // Note: The latest values are now stored in oldGrid, *not* newGrid!
			vector<int> x = position(oldGrid,n);

			vector<double> gradPhi_del = gradient(oldGrid, x, 2);
			vector<double> gradPhi_mu  = gradient(oldGrid, x, 3);
			vector<double> gradPhi_lav = gradient(oldGrid, x, 4);

			double myCr = oldGrid(n)[0];
			double myNb = oldGrid(n)[1];
			double myDel = fabs(oldGrid(n)[2]);
			double myMu  = fabs(oldGrid(n)[3]);
			double myLav = fabs(oldGrid(n)[4]);
			double myf = dV*(gibbs(oldGrid(n)) + kappa_del * (gradPhi_del * gradPhi_del)
			                                   + kappa_mu  * (gradPhi_mu  * gradPhi_mu )
			                                   + kappa_lav * (gradPhi_lav * gradPhi_lav));

			#ifndef MPI_VERSION
			#pragma omp critical
			{
			#endif
				totCr  += myCr;  // total Cr mass
				totNb  += myNb;  // total Nb mass
				totDel += myDel; // total delta volume
				totMu  += myMu;  // total mu volume
				totLav += myLav; // total Laves volume
				totF   += myf;   // total free energy
			#ifndef MPI_VERSION
			}
			#endif

		}

		totCr /= Ntot;
		totNb /= Ntot;

		#ifdef MPI_VERSION
		double myCr(totCr);
		double myNb(totNb);
		double myDel(totDel);
		double myMu(totMu);
		double myLav(totLav);
		double myF(totF);
		MPI::COMM_WORLD.Reduce(&myCr,  &totCr,  1, MPI_DOUBLE, MPI_SUM, 0);
		MPI::COMM_WORLD.Reduce(&myNb,  &totNb,  1, MPI_DOUBLE, MPI_SUM, 0);
		MPI::COMM_WORLD.Reduce(&myDel, &totDel, 1, MPI_DOUBLE, MPI_SUM, 0);
		MPI::COMM_WORLD.Reduce(&myMu,  &totMu,  1, MPI_DOUBLE, MPI_SUM, 0);
		MPI::COMM_WORLD.Reduce(&myLav, &totLav, 1, MPI_DOUBLE, MPI_SUM, 0);
		MPI::COMM_WORLD.Reduce(&myF,   &totF,   1, MPI_DOUBLE, MPI_SUM, 0);
		#endif
		if (rank==0)
			cfile<<totCr<<'\t'<<totNb<<'\t'<<totDel<<'\t'<<totMu<<'\t'<<totLav<<'\t'<<totF<<'\n';
	}
	if (rank==0)
		cfile.close();

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
template<int dim,typename T>
void guessGamma(MMSP::grid<dim,MMSP::vector<T> >& GRID, int n, std::mt19937_64& mt_rand, std::uniform_real_distribution<T>& real_gen, const T& amp)
{
	const T& xcr = fabs(GRID(n)[0]);
	const T& xnb = fabs(GRID(n)[1]);
	// if it's inside the gamma field, don't change it
	bool below_upper = (xcr < -0.45*(xnb-0.075)/0.075);
	bool nb_rich = (xnb > 0.075);
	if (below_upper) {
		GRID(n)[5] = xcr;
		GRID(n)[6] = xnb;
	} else if (nb_rich) {
		GRID(n)[5] = xcr/(xcr+xnb+0.9);
		GRID(n)[6] = xnb/(xcr+xnb+0.9);
	} else {
		GRID(n)[5] = -0.45*(xnb-0.075)/0.075;
		GRID(n)[6] = xnb;
	}
	GRID(n)[5] += amp*real_gen(mt_rand);
	GRID(n)[6] += amp*real_gen(mt_rand);
}

template<int dim,typename T>
void guessDelta(MMSP::grid<dim,MMSP::vector<T> >& GRID, int n, std::mt19937_64& mt_rand, std::uniform_real_distribution<T>& real_gen, const T& amp)
{
	const T& xcr = GRID(n)[0];
	const T& xnb = GRID(n)[1];
	GRID(n)[7] = fabs(xcr/(xcr+xnb+0.75) + amp*real_gen(mt_rand));
	GRID(n)[8] = fabs(xnb/(xcr+xnb+0.75) + amp*real_gen(mt_rand));
}

template<int dim,typename T>
void guessMu(MMSP::grid<dim,MMSP::vector<T> >& GRID, int n, std::mt19937_64& mt_rand, std::uniform_real_distribution<T>& real_gen, const T& amp)
{
	const T& xcr = fabs(GRID(n)[0]);
	const T& xnb = fabs(GRID(n)[1]);
	// if it's inside the mu field, don't change it
	bool below_upper = (xcr < 0.325*(xnb-0.475)/0.2);
	bool above_lower = (xcr > -0.5375*(xnb-0.5625)/0.1);
	bool ni_poor = (1.0-xcr-xnb < 0.5);
	if (ni_poor && below_upper && above_lower) {
		GRID(n)[9] = xcr;
		GRID(n)[10] = xnb;
	} else if (ni_poor && below_upper) {
		GRID(n)[9] = -0.5375*(xnb-0.5625)/0.1;
		GRID(n)[10] = xnb;
	} else if (ni_poor && above_lower) {
		GRID(n)[9] = 0.325*(xnb-0.475)/0.2;
		GRID(n)[10] = xnb;
	} else {
		GRID(n)[9] = 0.02;
		GRID(n)[10] = 0.5;
	}
	GRID(n)[9] += amp*real_gen(mt_rand);
	GRID(n)[10] += amp*real_gen(mt_rand);
}

template<int dim,typename T>
void guessLaves(MMSP::grid<dim,MMSP::vector<T> >& GRID, int n, std::mt19937_64& mt_rand, std::uniform_real_distribution<T>& real_gen, const T& amp)
{
	const T& xcr = fabs(GRID(n)[0]);
	const T& xnb = fabs(GRID(n)[1]);
	// if it's inside the Laves field, don't change it
	bool below_upper = (xcr < 0.68*(xnb-0.2)/0.12);
	bool above_lower = (xcr > 0.66*(xnb-0.325)/0.015);
	bool ni_poor = (1.0-xcr-xnb < 0.4);
	if (ni_poor && below_upper && above_lower) {
		GRID(n)[11] = xcr;
		GRID(n)[12] = xnb;
	} else if (ni_poor && below_upper) {
		GRID(n)[11] = 0.66*(xnb-0.325)/0.015;
		GRID(n)[12] = xnb;
	} else if (ni_poor && above_lower) {
		GRID(n)[11] = 0.68*(xnb-0.2)/0.12;
		GRID(n)[12] = xnb;
	} else {
		GRID(n)[11] = 0.332;
		GRID(n)[12] = 0.334;
	}
	GRID(n)[11] += amp*real_gen(mt_rand);
	GRID(n)[12] += amp*real_gen(mt_rand);
}

template<typename T>
void embedParticle(MMSP::grid<2,MMSP::vector<T> >& GRID, const MMSP::vector<int>& origin, const int pid,
                const double rprcp, const double rdpltCr, const double rdpltNb,
                const T& xCr, const T& xNb,
                const double& xCrdep, const double& xNbdep, const T phi)
{
	MMSP::vector<int> x(origin);
	double R = rprcp; //std::max(rdpltCr, rdpltNb);
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
			}
			/*
			else if (fabs(GRID(x)[2]) + fabs(GRID(x)[3]) + fabs(GRID(x)[4])>epsilon) {
				// Don't deplete neighboring precipitates, only matrix
				continue;
			} else {
				if (r<rdpltCr) { // point falls within Cr depletion region
					T dxCr = xCrdep - xCrdep*(r-rprcp)/(rdpltCr-rprcp);
					GRID(x)[0] = std::max(xCrdep, GRID(x)[0]-dxCr);
				}
				if (r<rdpltNb) { // point falls within Nb depletion region
					T dxNb = xNbdep - xNbdep*(r-rprcp)/(rdpltNb-rprcp);
					GRID(x)[1] = std::max(xNbdep, GRID(x)[1]-dxNb);
				}
			}
			*/
		}
	}
}

template<int dim, typename T>
void print_values(const MMSP::grid<dim,MMSP::vector<T> >& GRID)
{
	int rank = 0;
	int Nx = MMSP::g1(GRID, 0) - MMSP::g0(GRID, 0);
	int Ny = MMSP::g1(GRID, 1) - MMSP::g0(GRID, 1);
	#ifdef MPI_VERSION
	rank = MPI::COMM_WORLD.Get_rank();
	#endif
	MMSP::vector<double> totals(NC+NP-1, 0.0);
	for (int n=0; n<MMSP::nodes(GRID); n++) {
		for (int i=0; i<NC; i++)
			totals[i] += GRID(n)[i];
		for (int i=NC; i<NC+NP-1; i++)
			totals[i] += std::fabs(GRID(n)[i]);
	}

	#ifdef MPI_VERSION
	MMSP::vector<double> myTot(totals);
	for (int i=0; i<NC+NP-1; i++) {
		MPI::COMM_WORLD.Reduce(&myTot[i], &totals[i], 1, MPI_DOUBLE, MPI_SUM, 0);
		MPI::COMM_WORLD.Barrier();
	}
	#endif

	for (int i=0; i<NC+NP-1; i++)
		totals[i] /= double(Nx*Ny);
	if (rank==0) {
		std::cout<<"    x_Cr       x_Nb       x_Ni       p_g        p_d        p_m        p_l\n";
		printf("%10.4g %10.4g %10.4g %10.4g %10.4g %10.4g %10.4g\n", totals[0], totals[1], 1.0-totals[0]-totals[1],
		                                                             1.0-totals[2]-totals[3]-totals[4], totals[2], totals[3], totals[4]);
	}
}


double h(const double p)
{
	return pow(fabs(p), 3) * (16.0*p*p - 15.0*fabs(p) + 10);
}

double hprime(const double p)
{
	return 2.0*p * (sign(p)*(8.0*p*p*p - 15.0*p*fabs(p) + 5.0*p) + 32.0*p*p*fabs(p) - 15.0*p*p + 10.0*fabs(p) );
}

double gibbs(const MMSP::vector<double>& v)
{
	double n_del = h(fabs(v[2]));
	double n_mu  = h(fabs(v[3]));
	double n_lav = h(fabs(v[4]));
	double n_gam = 1.0 - n_del - n_mu - n_lav;
	double g  = n_gam * g_gam(v[5],v[6],1.0-v[5]-v[6]);
	       g += n_del * g_del(v[7],v[8]);
	       g += n_mu  * g_mu( v[9],v[10],1.0-v[9]-v[10]);
	       g += n_lav * g_lav(v[12],1.0-v[11]-v[12]);
	       g += omega_del * v[2]*v[2] * pow(1.0 - fabs(v[2]), 2);
	       g += omega_mu  * v[3]*v[3] * pow(1.0 - fabs(v[3]), 2);
	       g += omega_lav * v[4]*v[4] * pow(1.0 - fabs(v[4]), 2);
	for (int i=NC; i<NC+NP-2; i++)
		for (int j=i+1; j<NC+NP-1; j++)
			g += 2.0 * alpha * v[i]*v[i] * v[j]*v[j];

	return g;
}

void simple_progress(int step, int steps)
{
	if (step==0)
		std::cout<<" ["<<std::flush;
	else if (step==steps-1)
		std::cout<<"•] "<<std::endl;
	else if (step % (steps/20) == 0)
		std::cout<<"• "<<std::flush;
}




/* ========================================= *
 * Invoke GSL to solve for parallel tangents *
 * ========================================= */

/* Given const phase fraction (p) and molar fraction (c), iteratively determine the
 * molar fractions in each phase that satisfy equal chemical potential.
 */

int commonTangent_f(const gsl_vector* x, void* params, gsl_vector* f)
{
	// Prepare constants
	const double x_Cr = ((struct rparams*) params)->x_Cr;
	const double x_Nb = ((struct rparams*) params)->x_Nb;
	const double n_del = ((struct rparams*) params)->n_del;
	const double n_mu = ((struct rparams*) params)->n_mu;
	const double n_lav = ((struct rparams*) params)->n_lav;
	const double n_gam = 1.0 - n_del - n_mu - n_lav;

	// Prepare variables
	const double C_gam_Cr = gsl_vector_get(x, 0);
	const double C_gam_Nb = gsl_vector_get(x, 1);
	const double C_gam_Ni = 1.0 - C_gam_Cr - C_gam_Nb;

	const double C_del_Cr = gsl_vector_get(x, 2);
	const double C_del_Nb = gsl_vector_get(x, 3);

	const double C_mu_Cr  = gsl_vector_get(x, 4);
	const double C_mu_Ni  = gsl_vector_get(x, 5);
	const double C_mu_Nb = 1.0 - C_mu_Cr - C_mu_Ni;

	const double C_lav_Nb = gsl_vector_get(x, 6);
	const double C_lav_Ni = gsl_vector_get(x, 7);
	const double C_lav_Cr = 1.0 - C_lav_Nb - C_lav_Ni;

	gsl_vector_set_zero(f); // handy!

	gsl_vector_set(f, 0, x_Cr - n_gam*C_gam_Cr - n_del*C_del_Cr - n_mu*C_mu_Cr - n_lav*C_lav_Cr );
	gsl_vector_set(f, 1, x_Nb - n_gam*C_gam_Nb - n_del*C_del_Nb - n_mu*C_mu_Nb - n_lav*C_lav_Nb );

	gsl_vector_set(f, 2, dg_gam_dxCr(C_gam_Cr, C_gam_Nb, C_gam_Ni) - dg_del_dxCr(C_del_Cr, C_del_Nb));
	gsl_vector_set(f, 3, dg_gam_dxNb(C_gam_Cr, C_gam_Nb, C_gam_Ni) - dg_del_dxNb(C_del_Cr, C_del_Nb));

	gsl_vector_set(f, 4, dg_gam_dxCr(C_gam_Cr, C_gam_Nb, C_gam_Ni) - dg_mu_dxCr(C_mu_Cr, C_mu_Nb, C_mu_Ni));
	gsl_vector_set(f, 5, dg_gam_dxNi(C_gam_Cr, C_gam_Nb, C_gam_Ni) - dg_mu_dxNi(C_mu_Cr, C_mu_Nb, C_mu_Ni));

	gsl_vector_set(f, 6, dg_gam_dxNb(C_mu_Cr, C_mu_Nb, C_mu_Ni) - dg_lav_dxNb(C_lav_Nb, C_lav_Ni));
	gsl_vector_set(f, 7, dg_gam_dxNi(C_mu_Cr, C_mu_Nb, C_mu_Ni) - dg_lav_dxNi(C_lav_Nb, C_lav_Ni));

	return GSL_SUCCESS;
}


int commonTangent_df(const gsl_vector* x, void* params, gsl_matrix* J)
{
	// Prepare constants
	const double x_Cr = ((struct rparams*) params)->x_Cr;
	const double x_Nb = ((struct rparams*) params)->x_Nb;
	const double n_del = ((struct rparams*) params)->n_del;
	const double n_mu = ((struct rparams*) params)->n_mu;
	const double n_lav = ((struct rparams*) params)->n_lav;
	const double n_gam = 1.0 - n_del - n_mu - n_lav;

	// Prepare variables
	const double C_gam_Cr = gsl_vector_get(x, 0);
	const double C_gam_Nb = gsl_vector_get(x, 1);
	const double C_gam_Ni = 1.0 - C_gam_Cr - C_gam_Nb;

	const double C_del_Cr = gsl_vector_get(x, 2);
	const double C_del_Nb = gsl_vector_get(x, 3);

	const double C_mu_Cr  = gsl_vector_get(x, 4);
	const double C_mu_Ni  = gsl_vector_get(x, 5);
	const double C_mu_Nb = 1.0 - C_mu_Cr - C_mu_Ni;

	const double C_lav_Nb = gsl_vector_get(x, 6);
	const double C_lav_Ni = gsl_vector_get(x, 7);
	const double C_lav_Cr = 1.0 - C_lav_Nb - C_lav_Ni;

	gsl_matrix_set_zero(J); // handy!

	// Conservation of mass (Cr, Nb)
	gsl_matrix_set(J, 0, 0, -n_gam);
	gsl_matrix_set(J, 1, 1, -n_gam);

	gsl_matrix_set(J, 0, 2, -n_del);
	gsl_matrix_set(J, 1, 3, -n_del);

	gsl_matrix_set(J, 0, 4, -n_mu);
	gsl_matrix_set(J, 1, 4,  n_mu);
	gsl_matrix_set(J, 1, 5,  n_mu);

	gsl_matrix_set(J, 0, 6,  n_lav);
	gsl_matrix_set(J, 0, 7,  n_lav);
	gsl_matrix_set(J, 1, 6, -n_lav);


	// Equal chemical potential involving gamma phase (Cr, Nb, Ni)
	const double J20 = d2g_gam_dxCrCr(C_gam_Cr, C_gam_Nb, C_gam_Ni);
	const double J21 = d2g_gam_dxCrNb(C_gam_Cr, C_gam_Nb, C_gam_Ni);
	const double J30 = d2g_gam_dxNbCr(C_gam_Cr, C_gam_Nb, C_gam_Ni);
	const double J31 = d2g_gam_dxNbNb(C_gam_Cr, C_gam_Nb, C_gam_Ni);
	const double J50 = d2g_gam_dxNiCr(C_gam_Cr, C_gam_Nb, C_gam_Ni);
	const double J51 = d2g_gam_dxNiNb(C_gam_Cr, C_gam_Nb, C_gam_Ni);
	gsl_matrix_set(J, 2, 0, J20);
	gsl_matrix_set(J, 2, 1, J21);
	gsl_matrix_set(J, 3, 0, J30);
	gsl_matrix_set(J, 3, 1, J31);

	gsl_matrix_set(J, 4, 0, J20);
	gsl_matrix_set(J, 4, 1, J21);
	gsl_matrix_set(J, 5, 0, J50);
	gsl_matrix_set(J, 5, 1, J51);

	gsl_matrix_set(J, 6, 0, J30);
	gsl_matrix_set(J, 6, 1, J31);
	gsl_matrix_set(J, 7, 0, J50);
	gsl_matrix_set(J, 7, 1, J51);


	// Equal chemical potential involving delta phase (Cr, Nb)
	gsl_matrix_set(J, 2, 2,-d2g_del_dxCrCr(C_del_Cr, C_del_Nb));
	gsl_matrix_set(J, 2, 3,-d2g_del_dxCrNb(C_del_Cr, C_del_Nb));
	gsl_matrix_set(J, 3, 2,-d2g_del_dxNbCr(C_del_Cr, C_del_Nb));
	gsl_matrix_set(J, 3, 3,-d2g_del_dxNbNb(C_del_Cr, C_del_Nb));


	// Equal chemical potential involving mu phase (Cr, Ni)
	gsl_matrix_set(J, 4, 4,-d2g_mu_dxCrCr(C_mu_Cr, C_mu_Nb, C_mu_Ni));
	gsl_matrix_set(J, 4, 5,-d2g_mu_dxCrNi(C_mu_Cr, C_mu_Nb, C_mu_Ni));
	gsl_matrix_set(J, 5, 4,-d2g_mu_dxNiCr(C_mu_Cr, C_mu_Nb, C_mu_Ni));
	gsl_matrix_set(J, 5, 5,-d2g_mu_dxNiNi(C_mu_Cr, C_mu_Nb, C_mu_Ni));


	// Equal chemical potential involving Laves phase (Nb, Ni)
	gsl_matrix_set(J, 6, 6,-d2g_lav_dxNbNb(C_lav_Nb, C_lav_Ni));
	gsl_matrix_set(J, 6, 7,-d2g_lav_dxNbNi(C_lav_Nb, C_lav_Ni));
	gsl_matrix_set(J, 7, 6,-d2g_lav_dxNiNb(C_lav_Nb, C_lav_Ni));
	gsl_matrix_set(J, 7, 7,-d2g_lav_dxNiNi(C_lav_Nb, C_lav_Ni));

	return GSL_SUCCESS;
}


int commonTangent_fdf(const gsl_vector* x, void* params, gsl_vector* f, gsl_matrix* J)
{
	commonTangent_f(x,  params, f);
	commonTangent_df(x, params, J);

	return GSL_SUCCESS;
}


rootsolver::rootsolver() :
	n(8), // eight equations
	maxiter(1000),
	tolerance(std::min(epsilon, 1.0e-12))
{
	x = gsl_vector_alloc(n);

	/* Choose the multidimensional root finding algorithm.
	 * If the Jacobian matrix is not available, or suspected to be unreliable, use a derivative-free
	 * algorithm. These are expensive since, in every case, the Jacobian is estimated internally
	 * using finite differences. Do the math and specify the Jacobian if at all possible.
	 * Consult the GSL manual for details:
	 * https://www.gnu.org/software/gsl/manual/html_node/Multidimensional-Root_002dFinding.html
	 *
	 * If GSL finds the matrix to be singular, select a hybrid algorithm, then consult a numerical
	 * methods reference (human or paper) to get your system of equations sorted.
	 */
	#ifndef JACOBIAN
	algorithm = gsl_multiroot_fsolver_hybrids; // hybrids, hybrid, dnewton, broyden
	solver = gsl_multiroot_fsolver_alloc(algorithm, n);
	mrf = {&commonTangent_f, n, &par};
	#else
	algorithm = gsl_multiroot_fdfsolver_hybridsj; // hybridsj, hybridj, newton, gnewton
	solver = gsl_multiroot_fdfsolver_alloc(algorithm, n);
	mrf = {&commonTangent_f, &commonTangent_df, &commonTangent_fdf, n, &par};
	#endif
}

template<int dim,typename T> double
rootsolver::solve(MMSP::grid<dim,MMSP::vector<T> >& GRID, int n)
{
	int status;
	size_t iter = 0;

	// Prepare variables
	MMSP::vector<double> C(x->size, 0.0);

	C[0] = static_cast<double>(GRID(n)[5]);                      // gamma Cr
	C[1] = static_cast<double>(GRID(n)[6]);                      //       Nb

	C[2] = static_cast<double>(GRID(n)[7]);                      // delta Cr
	C[3] = static_cast<double>(GRID(n)[8]);                      //       Nb

	C[4] = static_cast<double>(GRID(n)[9]);                      // mu    Cr
	C[5] = static_cast<double>(1.0 - GRID(n)[9] - GRID(n)[10]);  //       Ni

	C[6] = static_cast<double>(GRID(n)[12]);                     // Laves Nb
	C[7] = static_cast<double>(1.0 - GRID(n)[11] - GRID(n)[12]); //       Ni

	// initial guesses
	par.x_Cr = GRID(n)[0];
	par.x_Nb = GRID(n)[1];

	par.n_del = h(fabs(GRID(n)[2]));
	par.n_mu =  h(fabs(GRID(n)[3]));
	par.n_lav = h(fabs(GRID(n)[4]));

	for (int i=0; i<x->size; i++)
		gsl_vector_set(x, i, C[i]);

	#ifndef JACOBIAN
	gsl_multiroot_fsolver_set(solver, &mrf, x);
	#else
	gsl_multiroot_fdfsolver_set(solver, &mrf, x);
	#endif

	do {
		iter++;
		#ifndef JACOBIAN
		status = gsl_multiroot_fsolver_iterate(solver);
		#else
		status = gsl_multiroot_fdfsolver_iterate(solver);
		#endif
		if (status) // extra points for finishing early!
			break;
		status = gsl_multiroot_test_residual(solver->f, tolerance);
	} while (status==GSL_CONTINUE && iter<maxiter);

	GRID(n)[5] = static_cast<T>(C[0]);               // gamma Cr
	GRID(n)[6] = static_cast<T>(C[1]);               //       Nb

	GRID(n)[7] = static_cast<T>(C[2]);               // delta Cr
	GRID(n)[8] = static_cast<T>(C[3]);               //       Nb

	GRID(n)[9] = static_cast<T>(C[4]);               // mu    Cr
	GRID(n)[10] = static_cast<T>(1.0 - C[4] - C[5]); //       Nb

	GRID(n)[11] = static_cast<T>(1.0 - C[6] - C[7]); // Laves Cr
	GRID(n)[12] = static_cast<T>(C[6]);              //       Nb

	double residual = gsl_blas_dnrm2(solver->f);

	return residual;
}

rootsolver::~rootsolver()
{
	#ifndef JACOBIAN
	gsl_multiroot_fsolver_free(solver);
	#else
	gsl_multiroot_fdfsolver_free(solver);
	#endif
	gsl_vector_free(x);
}

template<int dim,typename T>
void print_matrix(MMSP::grid<dim,MMSP::vector<T> >& GRID, int n)
{
	MMSP::vector<int> x = MMSP::position(GRID, n);
	T xCr = GRID(n)[0];
	T xNb = GRID(n)[1];

	T n_del = h(fabs(GRID(n)[2]));
	T n_mu  = h(fabs(GRID(n)[3]));
	T n_lav = h(fabs(GRID(n)[4]));
	T n_gam = 1.0 - n_del - n_mu - n_lav;

	T C_gam_Cr = GRID(n)[5];
	T C_gam_Nb = GRID(n)[6];
	T C_gam_Ni = 1.0 - C_gam_Cr - C_gam_Nb;

	T C_del_Cr = GRID(n)[7];
	T C_del_Nb = GRID(n)[8];

	T C_mu_Cr = GRID(n)[9];
	T C_mu_Nb = GRID(n)[10];
	T C_mu_Ni = 1.0 - C_mu_Cr - C_mu_Nb;

	T C_lav_Cr = GRID(n)[11];
	T C_lav_Nb = GRID(n)[12];
	T C_lav_Ni = 1.0 - C_lav_Cr - C_lav_Nb;

	std::cout<<"\nValues at ("<<x[0]<<','<<x[1]<<"):\n";
	printf("%11.3g %11.3g %11.3g %11.3g %11.3g %11.3g %11.3g %11.3g %11.3g %11.3g %11.3g %11.3g %11.3g\n", xCr, xNb, n_del, n_mu, n_lav, C_gam_Cr, C_gam_Nb, C_del_Cr, C_del_Nb, C_mu_Cr, C_mu_Nb, C_lav_Cr, C_lav_Nb);

	std::cout<<"Equations at ("<<x[0]<<','<<x[1]<<"):\n";
	printf("%11.3g %11.3g %11.3g %11.3g %11.3g %11.3g %11.3g %11.3g\n",
	xCr - n_gam*C_gam_Cr - n_del*C_del_Cr - n_mu*C_mu_Cr - n_lav*C_lav_Cr,
	xNb - n_gam*C_gam_Nb - n_del*C_del_Nb - n_mu*C_mu_Nb - n_lav*C_lav_Nb,
	dg_gam_dxCr(C_gam_Cr, C_gam_Nb, C_gam_Ni) - dg_del_dxCr(C_del_Cr, C_del_Nb),
	dg_gam_dxNb(C_gam_Cr, C_gam_Nb, C_gam_Ni) - dg_del_dxNb(C_del_Cr, C_del_Nb),
	dg_gam_dxCr(C_gam_Cr, C_gam_Nb, C_gam_Ni) - dg_mu_dxCr(C_mu_Cr, C_mu_Nb, C_mu_Ni),
	dg_gam_dxNi(C_gam_Cr, C_gam_Nb, C_gam_Ni) - dg_mu_dxNi(C_mu_Cr, C_mu_Nb, C_mu_Ni),
	dg_gam_dxNb(C_mu_Cr, C_mu_Nb, C_mu_Ni) - dg_lav_dxNb(C_lav_Nb, C_lav_Ni),
	dg_gam_dxNi(C_mu_Cr, C_mu_Nb, C_mu_Ni) - dg_lav_dxNi(C_lav_Nb, C_lav_Ni));

	std::cout<<"Jacobian at ("<<x[0]<<','<<x[1]<<"):\n";
	printf("%11.3g %11.3g %11.3g %11.3g %11.3g %11.3g %11.3g %11.3g\n", -n_gam,    0.0, -n_del,    0.0, -n_mu,  0.0,  n_lav, n_lav);
	printf("%11.3g %11.3g %11.3g %11.3g %11.3g %11.3g %11.3g %11.3g\n",    0.0, -n_gam,    0.0, -n_del,  n_mu, n_mu, -n_lav,   0.0);

	printf("%11.3g %11.3g %11.3g %11.3g %11.3g %11.3g %11.3g %11.3g\n", d2g_gam_dxCrCr(C_gam_Cr, C_gam_Nb, C_gam_Ni), d2g_gam_dxCrNb(C_gam_Cr, C_gam_Nb, C_gam_Ni), -d2g_del_dxCrCr(C_del_Cr, C_del_Nb), -d2g_del_dxCrNb(C_del_Cr, C_del_Nb), 0.0, 0.0, 0.0, 0.0);
	printf("%11.3g %11.3g %11.3g %11.3g %11.3g %11.3g %11.3g %11.3g\n", d2g_gam_dxNbCr(C_gam_Cr, C_gam_Nb, C_gam_Ni), d2g_gam_dxNbNb(C_gam_Cr, C_gam_Nb, C_gam_Ni), -d2g_del_dxNbCr(C_del_Cr, C_del_Nb), -d2g_del_dxNbNb(C_del_Cr, C_del_Nb), 0.0, 0.0, 0.0, 0.0);

	printf("%11.3g %11.3g %11.3g %11.3g %11.3g %11.3g %11.3g %11.3g\n", d2g_gam_dxCrCr(C_gam_Cr, C_gam_Nb, C_gam_Ni), d2g_gam_dxCrNb(C_gam_Cr, C_gam_Nb, C_gam_Ni), 0.0, 0.0, -d2g_mu_dxCrCr(C_mu_Cr, C_mu_Nb, C_mu_Ni), -d2g_mu_dxCrNi(C_mu_Cr, C_mu_Nb, C_mu_Ni), 0.0, 0.0);
	printf("%11.3g %11.3g %11.3g %11.3g %11.3g %11.3g %11.3g %11.3g\n", -d2g_mu_dxNiCr(C_mu_Cr, C_mu_Nb, C_mu_Ni),   -d2g_mu_dxNiNi(C_mu_Cr, C_mu_Nb, C_mu_Ni),     0.0, 0.0, -d2g_mu_dxNiCr(C_mu_Cr, C_mu_Nb, C_mu_Ni), -d2g_mu_dxNiNi(C_mu_Cr, C_mu_Nb, C_mu_Ni), 0.0, 0.0);

	printf("%11.3g %11.3g %11.3g %11.3g %11.3g %11.3g %11.3g %11.3g\n", d2g_gam_dxNbCr(C_gam_Cr, C_gam_Nb, C_gam_Ni), d2g_gam_dxNbNb(C_gam_Cr, C_gam_Nb, C_gam_Ni), 0.0, 0.0, 0.0, 0.0, -d2g_lav_dxNbNb(C_lav_Nb, C_lav_Ni), -d2g_lav_dxNbNi(C_lav_Nb, C_lav_Ni));
	printf("%11.3g %11.3g %11.3g %11.3g %11.3g %11.3g %11.3g %11.3g\n", d2g_gam_dxNiCr(C_gam_Cr, C_gam_Nb, C_gam_Ni), d2g_gam_dxNiNb(C_gam_Cr, C_gam_Nb, C_gam_Ni), 0.0, 0.0, 0.0, 0.0, -d2g_lav_dxNiNb(C_lav_Nb, C_lav_Ni), -d2g_lav_dxNiNi(C_lav_Nb, C_lav_Ni));
}

#endif

#include"MMSP.main.hpp"
