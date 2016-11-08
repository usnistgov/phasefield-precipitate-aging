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
#include<cassert>
#include<gsl/gsl_blas.h>
#include<gsl/gsl_math.h>
#include<gsl/gsl_roots.h>
#include<gsl/gsl_vector.h>
#include<gsl/gsl_multiroots.h>

#include"MMSP.hpp"
#include"alloy625.hpp"
#ifdef PARABOLIC
#include"parabola625.c"
#else
#include"energy625.c"
#endif

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
//                        gamma   delta     mu     laves    Enriched gamma
const double xCr[NP+1] = {0.2945, 0.003125, 0.05,   0.325,  0.31};
const double xNb[NP+1] = {0.015,  0.24375,  0.4875, 0.2875, 0.12};
//const double xCr[NP+1] = {0.2945,  0.003125, 0.05,   0.325,  0.31};   // for dx=1.25 nm
//const double xNb[NP+1] = {0.00254,   0.24375,  0.4875, 0.2875, 0.12}; // for dx=1.25 nm
const double xCrdep = 0.5*xCr[0]; // leftover Cr in depleted gamma phase near precipitate particle
const double xNbdep = 0.5*xNb[0]; // leftover Nb in depleted gamma phase near precipitate particle

// Define st.dev. of Gaussians for alloying element segregation
//                         Cr      Nb
const double bell[NC] = {150.0e-9, 50.0e-9}; // est. between 80-200 nm from SEM

// Kinetic and model parameters
const double meshres = 5.0e-9; //1.25e-9;    // grid spacing (m)
const double Vm = 1.0e-5;         // molar volume (m^3/mol)
const double alpha = 1.07e11;     // three-phase coexistence coefficient (J/m^3)

#ifndef PARABOLIC
// Zhou's numbers (Al/Nb)
const double M_Cr = 2.42e-18;      // mobility in FCC Ni (mol^2/Nsm^2)
const double M_Nb = 2.42e-18;      // mobility in FCC Ni (mol^2/Nsm^2)
/*
const double M_Cr = 1.6e-17;      // mobility in FCC Ni (mol^2/Nsm^2)
const double M_Nb = 1.7e-18;      // mobility in FCC Ni (mol^2/Nsm^2)
*/

const double D_Cr = Vm*Vm*M_Cr;   // diffusivity in FCC Ni (m^4/Ns)
const double D_Nb = Vm*Vm*M_Nb;   // diffusivity in FCC Ni (m^4/Ns)
#else
// Diffusion constant from Karunaratne and Reed
const double D_Nb = 1.582202e-16; // diffusivity in FCC Ni (m^2/s)
const double D_Cr = D_Nb;         // diffusivity in FCC Ni (m^2/s)
#endif


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
const double ifce_width = 7.0*meshres; // ensure at least 7 points through the interface
const double omega_del = 3.0 * width_factor * sigma_del / ifce_width; // 9.5e8;  // multiwell height (J/m^3)
const double omega_mu  = 3.0 * width_factor * sigma_mu  / ifce_width; // 9.5e8;  // multiwell height (J/m^3)
const double omega_lav = 3.0 * width_factor * sigma_lav / ifce_width; // 9.5e8;  // multiwell height (J/m^3)

// Numerical considerations
const bool useNeumann = true;    // apply zero-flux boundaries (Neumann type)?
const bool tanh_init = false;    // apply tanh profile to initial profile of composition and phase
const double epsilon = 1.0e-10;  // what to consider zero to avoid log(c) explosions
const double noise_amp = 1.0e-5; // 1.0e-8;
const double init_amp = 0.0; //1.0e-3;  // 1.0e-8;

const double root_tol = 1.0e-3;   // residual tolerance (default is 1e-7)
const int root_max_iter = 500000; // default is 1000, increasing probably won't change anything but your runtime


const double LinStab = 0.001; // threshold of linear stability (von Neumann stability condition)
const double dtp = (meshres*meshres)/(4.0 * L_del*kappa_del); // transformation-limited timestep
#ifndef PARABOLIC
const double dtc = (meshres*meshres)/(8.0 * Vm*Vm * std::max(M_Cr*C_gam_Cr(), M_Nb*C_gam_Nb())); // diffusion-limited timestep
#else
const double dtc = (meshres*meshres)/(4.0 * std::max(D_Cr, D_Nb)); // diffusion-limited timestep
#endif
const double dt = LinStab * std::min(dtp, dtc);

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

	double totF = 0.0;
	double totCr = 0.0;
	double totNb = 0.0;
	double totDel = 0.0;
	double totMu  = 0.0;
	double totLav = 0.0;
	double totGam = 0.0;

	std::ofstream cfile;
	if (rank==0)
		cfile.open("c.log",std::ofstream::out);

	if (dim==2) {
		// Construct grid
		const int Nx = 192; // divisible by 12 and 64
		//const int Ny = 768;
		//const int Ny = 10;
		const int Ny = 5;
		//const int Ny = 15;
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
		const double rPrecip[NP-1] = {3.0*7.5e-9 / dx(initGrid,0),  // delta
		                              3.0*7.5e-9 / dx(initGrid,0),  // mu
		                              3.0*7.5e-9 / dx(initGrid,0)}; // Laves

		// depletion region surrounding precipitates
		double rDepltCr[NP-1] = {0.0};
		double rDepltNb[NP-1] = {0.0};
		for (int i=0; i<NP-1; i++) {
			rDepltCr[i] = std::sqrt(1.0+2.0*xCr[i+1]/xCrdep) * rPrecip[i];
			rDepltNb[i] = std::sqrt(1.0+2.0*xNb[i+1]/xNbdep) * rPrecip[i];
		}


		// Sanity check on system size and  particle spacing
		if (rank==0)
			std::cout << "Timestep dt=" << dt << ". Linear stability limits: dtp=" << dtp << " (transformation-limited), dtc="<< dtc << " (diffusion-limited)." << std::endl;

		for (int i=0; i<NP-1; i++) {
			/*
			double rmax = std::max(rDepltCr[i], rDepltNb[i]);
			if (rmax > Ny/2)
				std::cerr << "Warning: domain too small to accommodate phase " << i << ", expand beyond " << 2.0*rmax << " pixels." << std::endl;
			*/
			if (rPrecip[i] > Ny/2)
				std::cerr << "Warning: domain too small to accommodate phase " << i << ", expand beyond " << 2.0*rPrecip[i] << " pixels." << std::endl;
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
				initGrid(x)[0] = xe_gam_Cr(); //matrixCr + xCr[0]; // * (1.0 + init_amp*real_gen(mt_rand));
				initGrid(x)[1] = xe_gam_Nb(); //matrixNb + xNb[0]; // * (1.0 + init_amp*real_gen(mt_rand));

				// tiny bit of noise to avoid zeros
				for (int i=NC; i<NC+NP-1; i++)
					initGrid(x)[i] = 0.0; //init_amp*real_gen(mt_rand);
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
			embedParticle(initGrid, origin, j+2, rPrecip[j], rDepltCr[j], rDepltNb[j], xCr[j+1], xNb[j+1], xCrdep, xNbdep,  1.0 - epsilon);
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

			/*
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
			*/


			/*
			// Initialize delta stripe
			int j = 0;
			origin[0] = Nx / 2;
			origin[1] = Ny / 2;
			embedStripe(initGrid, origin, j+2, rPrecip[j], rDepltCr[j], rDepltNb[j], xCr[j+1], xNb[j+1], xCrdep, xNbdep,  1.0 - epsilon);

			// Initialize mu stripe
			j = 1;
			origin[0] = Nx / 2 - xoffset;
			embedStripe(initGrid, origin, j+2, rPrecip[j], rDepltCr[j], rDepltNb[j], xCr[j+1], xNb[j+1], xCrdep, xNbdep,  1.0 - epsilon);

			// Initialize Laves stripe
			j = 2;
			origin[0] = Nx / 2 + xoffset;
			embedStripe(initGrid, origin, j+2, rPrecip[j], rDepltCr[j], rDepltNb[j], xCr[j+1], xNb[j+1], xCrdep, xNbdep,  1.0 - epsilon);
			*/

			// Initialize planar interface between gamma and delta
			origin[0] = Nx / 4;
			origin[1] = Ny / 2;
			embedStripe(initGrid, origin, 2, Nx/4, 0, 0, 1.1*xe_del_Cr(), 1.1*xe_del_Nb(), xe_del_Cr(), xe_del_Nb(),  1.0-epsilon);
		}


		// Initialize compositions in a manner compatible with OpenMP and MPI parallelization
		#ifdef MPI_VERSION
		for (int n=0; n<nodes(initGrid); n++) {
			guessGamma(initGrid, n, mt_rand, real_gen, init_amp);
			guessDelta(initGrid, n, mt_rand, real_gen, init_amp);
			guessMu(   initGrid, n, mt_rand, real_gen, init_amp);
			guessLaves(initGrid, n, mt_rand, real_gen, init_amp);
		}
		#else
		#pragma omp parallel for
		for (int n=0; n<nodes(initGrid); n++) {
			#pragma omp critical
			{
				guessGamma(initGrid, n, mt_rand, real_gen, init_amp);
			}
			#pragma omp critical
			{
				guessDelta(initGrid, n, mt_rand, real_gen, init_amp);
			}
			#pragma omp critical
			{
				guessMu(   initGrid, n, mt_rand, real_gen, init_amp);
			}
			#pragma omp critical
			{
				guessLaves(initGrid, n, mt_rand, real_gen, init_amp);
			}
		}
		#endif


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
			double myDel = h(fabs(initGrid(n)[2]));
			double myMu  = h(fabs(initGrid(n)[3]));
			double myLav = h(fabs(initGrid(n)[4]));
			double myGam = 1.0 - myDel - myMu - myLav;
			double myf = dV*(gibbs(initGrid(n)) + kappa_del * (gradPhi_del * gradPhi_del)
			                                    + kappa_mu  * (gradPhi_mu  * gradPhi_mu )
			                                    + kappa_lav * (gradPhi_lav * gradPhi_lav));
			initGrid(n)[fields(initGrid)-1] = myf;

			#ifndef MPI_VERSION
			#pragma omp critical
			{
			#endif
				totCr  += myCr;  // total Cr mass
				totNb  += myNb;  // total Nb mass
				totDel += myDel; // total delta volume
				totMu  += myMu;  // total mu volume
				totLav += myLav; // total Laves volume
				totGam += myGam; // total gamma volume
				totF   += myf;   // total free energy
			#ifndef MPI_VERSION
			}
			#endif
		}

		totCr /= Ntot;
		totNb /= Ntot;
		totGam /= Ntot;
		totDel /= Ntot;
		totMu  /= Ntot;
		totLav /= Ntot;

		#ifdef MPI_VERSION
		double myCr(totCr);
		double myNb(totNb);
		double myDel(totDel);
		double myMu(totMu);
		double myGam(totGam);
		double myLav(totLav);
		double myF(totF);
		MPI::COMM_WORLD.Reduce(&myCr,  &totCr,  1, MPI_DOUBLE, MPI_SUM, 0);
		MPI::COMM_WORLD.Reduce(&myNb,  &totNb,  1, MPI_DOUBLE, MPI_SUM, 0);
		MPI::COMM_WORLD.Reduce(&myDel, &totDel, 1, MPI_DOUBLE, MPI_SUM, 0);
		MPI::COMM_WORLD.Reduce(&myMu,  &totMu,  1, MPI_DOUBLE, MPI_SUM, 0);
		MPI::COMM_WORLD.Reduce(&myLav, &totLav, 1, MPI_DOUBLE, MPI_SUM, 0);
		MPI::COMM_WORLD.Reduce(&myGam, &totGam, 1, MPI_DOUBLE, MPI_SUM, 0);
		MPI::COMM_WORLD.Reduce(&myF,   &totF,   1, MPI_DOUBLE, MPI_SUM, 0);
		#endif

		#ifdef MPI_VERSION
		#endif
		if (rank==0) {
			cfile << totCr  << '\t' << totNb  << '\t'
			      << totGam << '\t' << totDel << '\t' << totMu << '\t' << totLav << '\t'
			      << totF   << '\t' << '0'    << std::endl;
			cfile.close();
		}

		//print_values(initGrid);
	if (rank==0) {
		std::cout << "    x_Cr       x_Nb       x_Ni       p_g        p_d        p_m        p_l\n";
		printf("%10.4g %10.4g %10.4g %10.4g %10.4g %10.4g %10.4g\n", totCr, totNb, 1.0-totCr-totNb,
		                                                             totGam, totDel, totMu, totLav);
	}
		output(initGrid,filename);

	} else
		std::cerr << "Error: " << dim << "-dimensional grids unsupported." << std::endl;
}

template <int dim, typename T> void update(grid<dim,vector<T> >& oldGrid, int steps)
{
	int rank=0;
	#ifdef MPI_VERSION
	rank = MPI::COMM_WORLD.Get_rank();
	#endif

	ghostswap(oldGrid);
	grid<dim,vector<T> > newGrid(oldGrid);
	#ifndef PARABOLIC
	grid<dim,vector<T> > potGrid(oldGrid, NC); // storage for chemical potentials
	#endif

	// Construct the parallel tangent solver
	std::mt19937_64 mt_rand(time(NULL)+rank);
	std::uniform_real_distribution<double> real_gen(-1,1);

	double dV = 1.0;
	double Ntot = 1.0;
	for (int d=0; d<dim; d++) {
		dx(oldGrid,d) = meshres;
		dx(newGrid,d) = meshres;
		#ifndef PARABOLIC
		dx(potGrid,d) = meshres;
		#endif
		dV *= dx(oldGrid,d);
		Ntot *= g1(oldGrid, d) - g0(oldGrid, d);
		if (useNeumann && x0(oldGrid,d) == g0(oldGrid,d)) {
			b0(oldGrid,d) = Neumann;
			b0(newGrid,d) = Neumann;
			#ifndef PARABOLIC
			b0(potGrid,d) = Neumann;
			#endif
		} else if (useNeumann && x1(oldGrid,d) == g1(oldGrid,d)) {
			b1(oldGrid,d) = Neumann;
			b1(newGrid,d) = Neumann;
			#ifndef PARABOLIC
			b1(potGrid,d) = Neumann;
			#endif
		}
	}

	std::ofstream cfile;
	if (rank==0)
		cfile.open("c.log",std::ofstream::out | std::ofstream::app);

	for (int step=0; step<steps; step++) {
		if (rank==0)
			print_progress(step, steps);

		double totBadTangents = 0.0;

		#ifndef MPI_VERSION
		#pragma omp parallel for
		#endif
		for (int n=0; n<nodes(oldGrid); n++) {
			vector<int> x = position(oldGrid,n);

			/* =========================== *
			 * Solve for parallel tangents *
			 * =========================== */

			rootsolver parallelTangentSolver;
			double res = parallelTangentSolver.solve(oldGrid, n);

			if (res>root_tol) {
				#ifndef MPI_VERSION
				#pragma omp critical
				{
				totBadTangents += 1;
				}
				#else
				totBadTangents += 1;
				#endif

				// If guesses are invalid, this may cause errors.
				#ifdef MPI_VERSION
				guessGamma(oldGrid, n, mt_rand, real_gen, noise_amp);
				guessDelta(oldGrid, n, mt_rand, real_gen, noise_amp);
				guessMu(   oldGrid, n, mt_rand, real_gen, noise_amp);
				guessLaves(oldGrid, n, mt_rand, real_gen, noise_amp);
				#else
				#pragma omp critical
				{
					guessGamma(oldGrid, n, mt_rand, real_gen, noise_amp);
				}
				#pragma omp critical
				{
					guessDelta(oldGrid, n, mt_rand, real_gen, noise_amp);
				}
				#pragma omp critical
				{
					guessMu(   oldGrid, n, mt_rand, real_gen, noise_amp);
				}
				#pragma omp critical
				{
					guessLaves(oldGrid, n, mt_rand, real_gen, noise_amp);
				}
				#endif
			}

			/* ========================= *
			 * Store chemical potentials *
			 * ========================= */

			const T& C_gam_Cr = oldGrid(n)[5];
			const T& C_gam_Nb = oldGrid(n)[6];
			const T  C_gam_Ni = 1.0 - C_gam_Cr - C_gam_Nb;

			#ifndef PARABOLIC
			potGrid(x)[0] = dg_gam_dxCr(C_gam_Cr, C_gam_Nb, C_gam_Ni);
			potGrid(x)[1] = dg_gam_dxNb(C_gam_Cr, C_gam_Nb, C_gam_Ni);
			#endif

		}

		#ifndef PARABOLIC
		ghostswap(potGrid);
		#endif

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


			const T xCr = oldGrid(n)[0];
			const T xNb = oldGrid(n)[1];

			#ifndef PARABOLIC
			vector<T> lapPot = laplacian(potGrid, x);

			newGrid(x)[0] = xCr + dt * D_Cr * lapPot[0];
			newGrid(x)[1] = xNb + dt * D_Nb * lapPot[1];

			#else
			newGrid(x)[0] = xCr + dt * D_Cr * laplacian(oldGrid, x, 5);
			newGrid(x)[1] = xNb + dt * D_Nb * laplacian(oldGrid, x, 6);

			#endif


			/* ======================================== *
			 * Solve the Equation of Motion for Phases  *
			 * ======================================== */

			const T phi_del  = oldGrid(n)[2]; // phase fraction of delta
			const T phi_mu   = oldGrid(n)[3]; // phase fraction of mu
			const T phi_lav  = oldGrid(n)[4]; // phase fraction of Laves

			const T C_gam_Cr = oldGrid(n)[5]; // Cr molar fraction in pure gamma
			const T C_gam_Nb = oldGrid(n)[6]; // Nb molar fraction in pure gamma
			const T C_gam_Ni = 1.0 - C_gam_Cr - C_gam_Nb;

			const T C_del_Cr = oldGrid(n)[7]; // Cr molar fraction in pure delta
			const T C_del_Nb = oldGrid(n)[8]; // Nb molar fraction in pure delta

			const T C_mu_Cr  = oldGrid(n)[9];  // Cr molar fraction in pure mu
			const T C_mu_Nb  = oldGrid(n)[10]; // Nb molar fraction in pure mu
			const T C_mu_Ni  = 1.0 - C_mu_Cr - C_mu_Nb;

			const T C_lav_Cr = oldGrid(n)[11]; // Cr molar fraction in pure Laves
			const T C_lav_Nb = oldGrid(n)[12]; // Nb molar fraction in pure Laves
			const T C_lav_Ni = 1.0 - C_lav_Cr - C_lav_Nb;

			const double ndel = h(fabs(phi_del));
			const double nmu  = h(fabs(phi_mu));
			const double nlav = h(fabs(phi_lav));
			const double ngam = 1.0 - ndel - nmu - nlav;


			double lapPhi_del = laplacian(oldGrid, x, 2);
			double lapPhi_mu  = laplacian(oldGrid, x, 3);
			double lapPhi_lav = laplacian(oldGrid, x, 4);


			#ifdef PARABOLIC
			double mu_Cr = dg_gam_dxCr(C_gam_Cr);
			double mu_Nb = dg_gam_dxNb(C_gam_Nb);
			double mu_Ni = dg_gam_dxNi(C_gam_Cr, C_gam_Nb, C_gam_Ni);

			double df_dphi_del = -sign(phi_del) * hprime(fabs(phi_del)) * (g_gam(C_gam_Cr, C_gam_Nb) - g_del(C_del_Cr, C_del_Nb)
			                                                               - (C_gam_Cr - C_del_Cr) * mu_Cr
			                                                               - (C_gam_Nb - C_del_Nb) * mu_Nb);

			double df_dphi_mu  = -sign(phi_mu)  * hprime(fabs(phi_mu))  * (g_gam(C_gam_Cr, C_gam_Nb) - g_mu(C_mu_Cr, C_mu_Ni)
			                                                               - (C_gam_Cr - C_mu_Cr) * mu_Cr
			                                                               - (C_gam_Ni - C_mu_Ni) * mu_Ni);

			double df_dphi_lav = -sign(phi_lav) * hprime(fabs(phi_lav)) * (g_gam(C_gam_Cr, C_gam_Nb) - g_lav(C_lav_Nb, C_lav_Ni)
			                                                               - (C_gam_Nb - C_lav_Nb) * mu_Nb
			                                                               - (C_gam_Ni - C_lav_Ni) * mu_Ni);
			#else
			double mu_Cr = dg_gam_dxCr(C_gam_Cr, C_gam_Nb, C_gam_Ni);
			double mu_Nb = dg_gam_dxNb(C_gam_Cr, C_gam_Nb, C_gam_Ni);
			double mu_Ni = dg_gam_dxNi(C_gam_Cr, C_gam_Nb, C_gam_Ni);

			double df_dphi_del = -sign(phi_del) * hprime(fabs(phi_del)) * (g_gam(C_gam_Cr, C_gam_Nb, C_gam_Ni) - g_del(C_del_Cr, C_del_Nb)
			                                                               - (C_gam_Cr - C_del_Cr) * mu_Cr
			                                                               - (C_gam_Nb - C_del_Nb) * mu_Nb);

			double df_dphi_mu  = -sign(phi_mu)  * hprime(fabs(phi_mu))  * (g_gam(C_gam_Cr, C_gam_Nb, C_gam_Ni) - g_mu(C_mu_Cr, C_mu_Nb, C_mu_Ni)
			                                                               - (C_gam_Cr - C_mu_Cr) * mu_Cr
			                                                               - (C_gam_Ni - C_mu_Ni) * mu_Ni);

			double df_dphi_lav = -sign(phi_lav) * hprime(fabs(phi_lav)) * (g_gam(C_gam_Cr, C_gam_Nb, C_gam_Ni) - g_lav(C_lav_Nb, C_lav_Ni)
			                                                               - (C_gam_Nb - C_lav_Nb) * mu_Nb
			                                                               - (C_gam_Ni - C_lav_Ni) * mu_Ni);
			#endif

			df_dphi_del += 2.0 * omega_del * phi_del * (1.0 - fabs(phi_del)) * (1.0 - fabs(phi_del) - sign(phi_del) * phi_del);
			df_dphi_mu  += 2.0 * omega_mu  * phi_mu  * (1.0 - fabs(phi_mu))  * (1.0 - fabs(phi_mu)  - sign(phi_mu)  * phi_mu);
			df_dphi_lav += 2.0 * omega_lav * phi_lav * (1.0 - fabs(phi_lav)) * (1.0 - fabs(phi_lav) - sign(phi_lav) * phi_lav);

			df_dphi_del += 4.0 * alpha * phi_del * (phi_mu  * phi_mu  + phi_lav * phi_lav);
			df_dphi_mu  += 4.0 * alpha * phi_mu  * (phi_del * phi_del + phi_lav * phi_lav);
			df_dphi_lav += 4.0 * alpha * phi_lav * (phi_del * phi_del + phi_mu  * phi_mu);


			newGrid(x)[2] = phi_del + dt * L_del * (kappa_del * lapPhi_del - df_dphi_del);
			newGrid(x)[3] = phi_mu  + dt * L_mu  * (kappa_mu  * lapPhi_mu  - df_dphi_mu );
			newGrid(x)[4] = phi_lav + dt * L_lav * (kappa_lav * lapPhi_lav - df_dphi_lav);



			/* ======================= *
			 * Store parallel tangents *
			 * ======================= */

			for (int i=NC+NP-1; i<fields(newGrid); i++)
				newGrid(x)[i] = oldGrid(n)[i];


			/* ======= *
			 * ~ fin ~ *
			 * ======= */
		}
		swap(oldGrid,newGrid);
		ghostswap(oldGrid);


		/* ====================================================================== *
		 * Collate summary & diagnostic data in OpenMP- and MPI-compatible manner *
		 * ====================================================================== */


		double totF = 0.0;
		double totCr = 0.0;
		double totNb = 0.0;
		double totDel = 0.0;
		double totMu  = 0.0;
		double totLav = 0.0;
		double totGam = 0.0;

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
			double myDel = h(fabs(oldGrid(n)[2]));
			double myMu  = h(fabs(oldGrid(n)[3]));
			double myLav = h(fabs(oldGrid(n)[4]));
			double myGam = 1.0 - myDel - myMu - myLav;
			double myf = dV*(gibbs(oldGrid(n)) + kappa_del * (gradPhi_del * gradPhi_del)
			                                   + kappa_mu  * (gradPhi_mu  * gradPhi_mu )
			                                   + kappa_lav * (gradPhi_lav * gradPhi_lav));
			oldGrid(n)[fields(oldGrid)-1] = static_cast<T>(myf);

			#ifndef MPI_VERSION
			#pragma omp critical
			{
			#endif
				totCr  += myCr;  // total Cr mass
				totNb  += myNb;  // total Nb mass
				totDel += myDel; // total delta volume
				totMu  += myMu;  // total mu volume
				totLav += myLav; // total Laves volume
				totGam += myGam; // total gamma volume
				totF   += myf;   // total free energy
			#ifndef MPI_VERSION
			}
			#endif

		}

		totCr /= Ntot;
		totNb /= Ntot;
		totGam /= Ntot;
		totDel /= Ntot;
		totMu  /= Ntot;
		totLav /= Ntot;
		totBadTangents /= Ntot;

		#ifdef MPI_VERSION
		double myCr(totCr);
		double myNb(totNb);
		double myDel(totDel);
		double myMu(totMu);
		double myLav(totLav);
		double myGam(totGam);
		double myF(totF);
		double myBad(totBadTangents);
		MPI::COMM_WORLD.Reduce(&myCr,  &totCr,  1, MPI_DOUBLE, MPI_SUM, 0);
		MPI::COMM_WORLD.Reduce(&myNb,  &totNb,  1, MPI_DOUBLE, MPI_SUM, 0);
		MPI::COMM_WORLD.Reduce(&myDel, &totDel, 1, MPI_DOUBLE, MPI_SUM, 0);
		MPI::COMM_WORLD.Reduce(&myMu,  &totMu,  1, MPI_DOUBLE, MPI_SUM, 0);
		MPI::COMM_WORLD.Reduce(&myLav, &totLav, 1, MPI_DOUBLE, MPI_SUM, 0);
		MPI::COMM_WORLD.Reduce(&myGam, &totGam, 1, MPI_DOUBLE, MPI_SUM, 0);
		MPI::COMM_WORLD.Reduce(&myF,   &totF,   1, MPI_DOUBLE, MPI_SUM, 0);
		MPI::COMM_WORLD.Reduce(&myBad, &totBadTangents,   1, MPI_DOUBLE, MPI_SUM, 0);
		#endif
		if (rank==0)
			cfile << totCr  << '\t' << totNb << '\t'
			      << totGam << '\t' << totDel << '\t' << totMu << '\t' << totLav << '\t'
			      << totF   << '\t' << totBadTangents << '\n';

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
	MMSP::vector<int> x = MMSP::position(GRID, n);

	/*
	const T xcr = (0.30 + GRID(x)[0])/2.0;
	const T xnb = (0.01 + GRID(x)[1])/2.0;
	const T xcr = 0.30;
	const T xnb = 0.01;
	*/

	// Coarsely approximate gamma using a line compound with x_Nb = 0.015

	const T xcr = GRID(x)[0];
	//const T xnb = GRID(x)[1];
	const T xnb = 0.015;
	const T xni = std::max(epsilon, 1.0 - xcr - GRID(x)[1]);

	GRID(x)[5] = xcr/(xcr + xnb + xni) + amp*real_gen(mt_rand);
	GRID(x)[6] = xnb + amp*real_gen(mt_rand);
}

template<int dim,typename T>
void guessDelta(MMSP::grid<dim,MMSP::vector<T> >& GRID, int n, std::mt19937_64& mt_rand, std::uniform_real_distribution<T>& real_gen, const T& amp)
{
	MMSP::vector<int> x = MMSP::position(GRID, n);

	/*
	const T xcr = (0.003125 + GRID(x)[0])/2.0;
	const T xnb = (0.24375 + GRID(x)[1])/2.0;
	const T xcr = 0.003125;
	const T xnb = 0.24375;
	*/

	// Coarsely approximate delta using a line compound with x_Ni = 0.75

	const T xcr = GRID(x)[0];
	const T xnb = GRID(x)[1];
	const T xni = 0.75;

	GRID(x)[7] = xcr/(xcr + xnb + xni) + amp*real_gen(mt_rand);
	GRID(x)[8] = xnb/(xcr + xnb + xni) + amp*real_gen(mt_rand);
}

template<int dim,typename T>
void guessMu(MMSP::grid<dim,MMSP::vector<T> >& GRID, int n, std::mt19937_64& mt_rand, std::uniform_real_distribution<T>& real_gen, const T& amp)
{
	MMSP::vector<int> x = MMSP::position(GRID, n);

	/*
	const T xcr = 0.05;
	const T xnb = 0.4875;
	const T xcr = (0.05 + GRID(x)[0])/2.0;
	const T xnb = (0.4875 + GRID(x)[1])/2.0;
	*/

	// Coarsely approximate mu using a line compound with x_Nb=52.5%

	const T xcr = GRID(x)[0];
	//const T xnb = GRID(x)[1];
	const T xnb = 0.525;
	const T xni = std::max(epsilon, 1.0 - xcr - GRID(x)[1]);

	GRID(x)[9] = xcr/(xcr + xnb + xni) + amp*real_gen(mt_rand);
	GRID(x)[10] = xnb + amp*real_gen(mt_rand);
	/*
	// if it's inside the mu field, don't change it
	bool below_upper = (xcr < 0.325*(xnb-0.475)/0.2);
	bool above_lower = (xcr > -0.5375*(xnb-0.5625)/0.1);
	bool ni_poor = (1.0-xcr-xnb < 0.5);
	if (ni_poor && below_upper && above_lower) {
		GRID(x)[9] = xcr;
		GRID(x)[10] = xnb;
	} else if (ni_poor && below_upper) {
		GRID(x)[9] = -0.5375*(xnb-0.5625)/0.1;
		GRID(x)[10] = xnb;
	} else if (ni_poor && above_lower) {
		GRID(x)[9] = 0.325*(xnb-0.475)/0.2;
		GRID(x)[10] = xnb;
	} else {
		GRID(x)[9] = 0.02;
		GRID(x)[10] = 0.5;
	}
	//GRID(x)[9]  = xcr; // + amp*real_gen(mt_rand);
	//GRID(x)[10] = xnb; // + amp*real_gen(mt_rand);
	GRID(x)[9]  += amp*real_gen(mt_rand);
	GRID(x)[10] += amp*real_gen(mt_rand);
	*/
}

template<int dim,typename T>
void guessLaves(MMSP::grid<dim,MMSP::vector<T> >& GRID, int n, std::mt19937_64& mt_rand, std::uniform_real_distribution<T>& real_gen, const T& amp)
{
	MMSP::vector<int> x = MMSP::position(GRID, n);

	/*
	const T xcr = 0.325;
	const T xnb = 0.2875;
	const T xcr = (0.325 + GRID(x)[0])/2.0;
	const T xnb = (0.2875 + GRID(x)[1])/2.0;
	*/

	// Coarsely approximate Laves using a line compound with x_Nb = 30.0%

	const T xcr = GRID(x)[0];
	//const T xnb = GRID(x)[1];
	const T xnb = 0.30;
	const T xni = std::max(epsilon, 1.0 - xcr - GRID(x)[1]);

	GRID(x)[11] = xcr/(xcr + xnb + xni) + amp*real_gen(mt_rand);
	GRID(x)[12] = xnb + amp*real_gen(mt_rand);

	/*
	// if it's inside the Laves field, don't change it
	bool below_upper = (xcr < 0.68*(xnb-0.2)/0.12);
	bool above_lower = (xcr > 0.66*(xnb-0.325)/0.015);
	bool ni_poor = (1.0-xcr-xnb < 0.4);
	if (ni_poor && below_upper && above_lower) {
		GRID(x)[11] = xcr;
		GRID(x)[12] = xnb;
	} else if (ni_poor && below_upper) {
		GRID(x)[11] = 0.66*(xnb-0.325)/0.015;
		GRID(x)[12] = xnb;
	} else if (ni_poor && above_lower) {
		GRID(x)[11] = 0.68*(xnb-0.2)/0.12;
		GRID(x)[12] = xnb;
	} else {
		GRID(x)[11] = 0.332;
		GRID(x)[12] = 0.334;
	}
	//GRID(x)[11] = xcr; // + amp*real_gen(mt_rand);
	//GRID(x)[12] = xnb; // + amp*real_gen(mt_rand);
	GRID(x)[11] += amp*real_gen(mt_rand);
	GRID(x)[12] += amp*real_gen(mt_rand);
	*/
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
			else if (h(fabs(GRID(x)[2])) + h(fabs(GRID(x)[3])) + h(fabs(GRID(x)[4]))>epsilon) {
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

template<typename T>
void embedStripe(MMSP::grid<2,MMSP::vector<T> >& GRID, const MMSP::vector<int>& origin, const int pid,
                const double rprcp, const double rdpltCr, const double rdpltNb,
                const T& xCr, const T& xNb,
                const double& xCrdep, const double& xNbdep, const T phi)
{
	MMSP::vector<int> x(origin);
	int R(rprcp); //std::max(rdpltCr, rdpltNb);

	for (x[0] = origin[0] - R; x[0] < origin[0] + R; x[0]++) {
		if (x[0] < x0(GRID) || x[0] >= x1(GRID))
			continue;
		for (x[1] = y0(GRID); x[1] < y1(GRID); x[1]++) {
			GRID(x)[0] = xCr;
			GRID(x)[1] = xNb;
			GRID(x)[pid] = phi;
		}
	}

	if (tanh_init) {
		// Create a tanh profile in composition surrounding the precipitate to reduce initial Laplacian
		double del = 4.3875e-9 / meshres;
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
}

template<int dim, typename T>
void print_values(const MMSP::grid<dim,MMSP::vector<T> >& GRID)
{
	int rank = 0;
	#ifdef MPI_VERSION
	rank = MPI::COMM_WORLD.Get_rank();
	#endif
	double Ntot = 1.0;
	for (int d=0; d<dim; d++)
		Ntot *= double(MMSP::g1(GRID, d) - MMSP::g0(GRID, d));

	double totCr = 0.0;
	double totNb = 0.0;
	double totDel = 0.0;
	double totMu  = 0.0;
	double totLav = 0.0;
	double totGam = 0.0;

	#ifndef MPI_VERSION
	#pragma omp parallel
	#endif
	for (int n=0; n<MMSP::nodes(GRID); n++) {
		double myCr = GRID(n)[0];
		double myNb = GRID(n)[1];
		double myDel = h(fabs(GRID(n)[2]));
		double myMu  = h(fabs(GRID(n)[3]));
		double myLav = h(fabs(GRID(n)[4]));
		double myGam = 1.0 - myDel - myMu - myLav;

		#ifndef MPI_VERSION
		#pragma omp critical
		{
		#endif
			totCr  += myCr;  // total Cr mass
			totNb  += myNb;  // total Nb mass
			totDel += myDel; // total delta volume
			totMu  += myMu;  // total mu volume
			totLav += myLav; // total Laves volume
			totGam += myGam; // total gamma volume
		#ifndef MPI_VERSION
		}
		#endif
	}

	totCr /= Ntot;
	totNb /= Ntot;
	totGam /= Ntot;
	totDel /= Ntot;
	totMu  /= Ntot;
	totLav /= Ntot;

	#ifdef MPI_VERSION
	double myCr(totCr);
	double myNb(totNb);
	double myDel(totDel);
	double myMu(totMu);
	double myLav(totLav);
	double myGam(totGam);
	MPI::COMM_WORLD.Reduce(&myCr,  &totCr,  1, MPI_DOUBLE, MPI_SUM, 0);
	MPI::COMM_WORLD.Reduce(&myNb,  &totNb,  1, MPI_DOUBLE, MPI_SUM, 0);
	MPI::COMM_WORLD.Reduce(&myDel, &totDel, 1, MPI_DOUBLE, MPI_SUM, 0);
	MPI::COMM_WORLD.Reduce(&myMu,  &totMu,  1, MPI_DOUBLE, MPI_SUM, 0);
	MPI::COMM_WORLD.Reduce(&myLav, &totLav, 1, MPI_DOUBLE, MPI_SUM, 0);
	MPI::COMM_WORLD.Reduce(&myGam, &totGam, 1, MPI_DOUBLE, MPI_SUM, 0);
	#endif

	if (rank==0) {
		std::cout << "    x_Cr       x_Nb       x_Ni       p_g        p_d        p_m        p_l\n";
		printf("%10.4g %10.4g %10.4g %10.4g %10.4g %10.4g %10.4g\n", totCr, totNb, 1.0-totCr-totNb,
		                                                             totGam, totDel, totMu, totLav);
	}
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

	#ifdef PARABOLIC
	double g  = n_gam * g_gam(v[5],  v[6]);
	       g += n_del * g_del(v[7],  v[8]);
	       g += n_mu  * g_mu( v[9],  1.0-v[9]-v[10]);
	       g += n_lav * g_lav(v[12], 1.0-v[11]-v[12]);
	       g += omega_del * v[2]*v[2] * pow(1.0 - fabs(v[2]), 2);
	       g += omega_mu  * v[3]*v[3] * pow(1.0 - fabs(v[3]), 2);
	       g += omega_lav * v[4]*v[4] * pow(1.0 - fabs(v[4]), 2);
	#else
	double g  = n_gam * g_gam(v[5],v[6],1.0-v[5]-v[6]);
	       g += n_del * g_del(v[7],v[8]);
	       g += n_mu  * g_mu( v[9],v[10],1.0-v[9]-v[10]);
	       g += n_lav * g_lav(v[12],1.0-v[11]-v[12]);
	       g += omega_del * v[2]*v[2] * pow(1.0 - fabs(v[2]), 2);
	       g += omega_mu  * v[3]*v[3] * pow(1.0 - fabs(v[3]), 2);
	       g += omega_lav * v[4]*v[4] * pow(1.0 - fabs(v[4]), 2);
	#endif

	for (int i=NC; i<NC+NP-2; i++)
		for (int j=i+1; j<NC+NP-1; j++)
			g += 2.0 * alpha * v[i]*v[i] * v[j]*v[j];

	return g;
}

void simple_progress(int step, int steps)
{
	if (step==0)
		std::cout << " [" << std::flush;
	else if (step==steps-1)
		std::cout << "•] " << std::endl;
	else if (step % (steps/20) == 0)
		std::cout << "• " << std::flush;
}




/* ========================================= *
 * Invoke GSL to solve for parallel tangents *
 * ========================================= */

/* Given const phase fraction (p) and molar fraction (c), iteratively determine the
 * molar fractions in each phase that satisfy equal chemical potential.
 */

int parallelTangent_f(const gsl_vector* x, void* params, gsl_vector* f)
{
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


	#ifdef PARABOLIC
	gsl_vector_set(f, 0, x_Cr - n_gam*C_gam_Cr - n_del*C_del_Cr - n_mu*C_mu_Cr - n_lav*C_lav_Cr );
	gsl_vector_set(f, 1, x_Nb - n_gam*C_gam_Nb - n_del*C_del_Nb - n_mu*C_mu_Nb - n_lav*C_lav_Nb );

	gsl_vector_set(f, 2, dg_gam_dxCr(C_gam_Cr) - dg_del_dxCr(C_del_Cr));
	gsl_vector_set(f, 3, dg_gam_dxNb(C_gam_Nb) - dg_del_dxNb(C_del_Nb));

	gsl_vector_set(f, 4, dg_gam_dxCr(C_gam_Cr) - dg_mu_dxCr(C_mu_Cr));
	gsl_vector_set(f, 5, dg_gam_dxNb(C_gam_Nb) - dg_mu_dxNb(C_mu_Nb));

	gsl_vector_set(f, 6, dg_gam_dxCr(C_gam_Cr) - dg_lav_dxCr(C_lav_Cr, C_lav_Nb));
	gsl_vector_set(f, 7, dg_gam_dxNb(C_gam_Nb) - dg_lav_dxNb(C_lav_Cr, C_lav_Nb));



	#else
	gsl_vector_set(f, 0, x_Cr - n_gam*C_gam_Cr - n_del*C_del_Cr - n_mu*C_mu_Cr - n_lav*C_lav_Cr );
	gsl_vector_set(f, 1, x_Nb - n_gam*C_gam_Nb - n_del*C_del_Nb - n_mu*C_mu_Nb - n_lav*C_lav_Nb );

	gsl_vector_set(f, 2, dg_gam_dxCr(C_gam_Cr, C_gam_Nb, C_gam_Ni) - dg_del_dxCr(C_del_Cr, C_del_Nb));
	gsl_vector_set(f, 3, dg_gam_dxNb(C_gam_Cr, C_gam_Nb, C_gam_Ni) - dg_del_dxNb(C_del_Cr, C_del_Nb));

	gsl_vector_set(f, 4, dg_gam_dxCr(C_gam_Cr, C_gam_Nb, C_gam_Ni) - dg_mu_dxCr(C_mu_Cr, C_mu_Nb, C_mu_Ni));
	gsl_vector_set(f, 5, dg_gam_dxNb(C_gam_Cr, C_gam_Nb, C_gam_Ni) - dg_mu_dxNb(C_mu_Cr, C_mu_Nb, C_mu_Ni));

	gsl_vector_set(f, 6, dg_gam_dxCr(C_gam_Cr, C_gam_Nb, C_gam_Ni) - dg_lav_dxCr(C_lav_Nb, C_lav_Ni));
	gsl_vector_set(f, 7, dg_gam_dxNb(C_gam_Cr, C_gam_Nb, C_gam_Ni) - dg_lav_dxNb(C_lav_Nb, C_lav_Ni));



	#endif

	return GSL_SUCCESS;
}


int parallelTangent_df(const gsl_vector* x, void* params, gsl_matrix* J)
{
	// Prepare constants
	const double n_del = ((struct rparams*) params)->n_del;
	const double n_mu  = ((struct rparams*) params)->n_mu;
	const double n_lav = ((struct rparams*) params)->n_lav;
	const double n_gam = 1.0 - n_del - n_mu - n_lav;

	// Prepare variables
	const double C_gam_Cr = gsl_vector_get(x, 0);
	const double C_gam_Nb = gsl_vector_get(x, 1);
	const double C_gam_Ni = 1.0 - C_gam_Cr - C_gam_Nb;

	const double C_del_Cr = gsl_vector_get(x, 2);
	const double C_del_Nb = gsl_vector_get(x, 3);

	const double C_mu_Cr  = gsl_vector_get(x, 4);
	const double C_mu_Nb  = gsl_vector_get(x, 5);
	const double C_mu_Ni = 1.0 - C_mu_Cr - C_mu_Nb;

	const double C_lav_Cr = gsl_vector_get(x, 6);
	const double C_lav_Nb = gsl_vector_get(x, 7);
	const double C_lav_Ni = 1.0 - C_lav_Cr - C_lav_Nb;

	gsl_matrix_set_zero(J); // handy!

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
	#ifdef PARABOLIC
	const double J20 = d2g_gam_dxCrCr();
	const double J21 = d2g_gam_dxCrNb();
	const double J30 = d2g_gam_dxNbCr();
	const double J31 = d2g_gam_dxNbNb();
	#else
	const double J20 = d2g_gam_dxCrCr(C_gam_Cr, C_gam_Ni);
	const double J21 = d2g_gam_dxCrNb(C_gam_Cr, C_gam_Nb, C_gam_Ni);
	const double J30 = d2g_gam_dxNbCr(C_gam_Cr, C_gam_Nb, C_gam_Ni);
	const double J31 = d2g_gam_dxNbNb(C_gam_Nb, C_gam_Ni);
	#endif

	gsl_matrix_set(J, 2, 0, J20);
	gsl_matrix_set(J, 2, 1, J21);
	gsl_matrix_set(J, 3, 0, J30);
	gsl_matrix_set(J, 3, 1, J31);

	gsl_matrix_set(J, 4, 0, J20);
	gsl_matrix_set(J, 4, 1, J21);
	gsl_matrix_set(J, 5, 0, J30);
	gsl_matrix_set(J, 5, 1, J31);

	gsl_matrix_set(J, 6, 0, J20);
	gsl_matrix_set(J, 6, 1, J21);
	gsl_matrix_set(J, 7, 0, J30);
	gsl_matrix_set(J, 7, 1, J31);

	// Equal chemical potential involving delta phase (Cr, Nb)
	#ifdef PARABOLIC
	gsl_matrix_set(J, 2, 2, -d2g_del_dxCrCr());
	gsl_matrix_set(J, 2, 3, -d2g_del_dxCrNb());
	gsl_matrix_set(J, 3, 2, -d2g_del_dxNbCr());
	gsl_matrix_set(J, 3, 3, -d2g_del_dxNbNb());
	#else
	gsl_matrix_set(J, 2, 2, -d2g_del_dxCrCr(C_del_Cr, C_del_Nb));
	gsl_matrix_set(J, 2, 3, -d2g_del_dxCrNb(C_del_Cr, C_del_Nb));
	gsl_matrix_set(J, 3, 2, -d2g_del_dxNbCr(C_del_Cr, C_del_Nb));
	gsl_matrix_set(J, 3, 3, -d2g_del_dxNbNb(C_del_Cr, C_del_Nb));
	#endif


	// Equal chemical potential involving mu phase (Cr, Ni)
	#ifdef PARABOLIC
	gsl_matrix_set(J, 4, 4, -d2g_mu_dxCrCr());
	gsl_matrix_set(J, 4, 5, -d2g_mu_dxCrNb());
	gsl_matrix_set(J, 5, 4, -d2g_mu_dxNbCr());
	gsl_matrix_set(J, 5, 5, -d2g_mu_dxNbNb());
	#else
	gsl_matrix_set(J, 4, 4, -d2g_mu_dxCrCr(C_mu_Cr, C_mu_Nb, C_mu_Ni));
	gsl_matrix_set(J, 4, 5, -d2g_mu_dxCrNb(C_mu_Cr, C_mu_Nb, C_mu_Ni));
	gsl_matrix_set(J, 5, 4, -d2g_mu_dxNbCr(C_mu_Cr, C_mu_Nb, C_mu_Ni));
	gsl_matrix_set(J, 5, 5, -d2g_mu_dxNbNb(C_mu_Cr, C_mu_Nb, C_mu_Ni));
	#endif

	// Equal chemical potential involving Laves phase (Nb, Ni)
	#ifdef PARABOLIC
	gsl_matrix_set(J, 6, 6, -d2g_lav_dxCrCr());
	gsl_matrix_set(J, 6, 7, -d2g_lav_dxCrNb());
	gsl_matrix_set(J, 7, 6, -d2g_lav_dxNbCr());
	gsl_matrix_set(J, 7, 7, -d2g_lav_dxNbNb());
	#else
	gsl_matrix_set(J, 6, 6, -d2g_lav_dxCrCr(C_lav_Nb, C_lav_Ni));
	gsl_matrix_set(J, 6, 7, -d2g_lav_dxCrNb(C_lav_Nb, C_lav_Ni));
	gsl_matrix_set(J, 7, 6, -d2g_lav_dxNbCr(C_lav_Nb, C_lav_Ni));
	gsl_matrix_set(J, 7, 7, -d2g_lav_dxNbNb(C_lav_Nb, C_lav_Ni));
	#endif

	return GSL_SUCCESS;
}


int parallelTangent_fdf(const gsl_vector* x, void* params, gsl_vector* f, gsl_matrix* J)
{
	parallelTangent_f(x,  params, f);
	parallelTangent_df(x, params, J);

	return GSL_SUCCESS;
}


rootsolver::rootsolver() :
	n(8), // eight equations
	maxiter(root_max_iter),
	tolerance(root_tol)
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
	// Available algorithms are: hybrids, hybrid, dnewton, broyden (in order of *decreasing* efficiency)
	algorithm = gsl_multiroot_fsolver_hybrids;
	solver = gsl_multiroot_fsolver_alloc(algorithm, n);
	mrf = {&parallelTangent_f, n, &par};
	#else
	// Available algorithms are: hybridsj, hybridj, newton, gnewton (in order of *decreasing* efficiency)
	algorithm = gsl_multiroot_fdfsolver_hybridsj;
	solver = gsl_multiroot_fdfsolver_alloc(algorithm, n);
	mrf = {&parallelTangent_f, &parallelTangent_df, &parallelTangent_fdf, n, &par};
	#endif
}

template<int dim,typename T> double
rootsolver::solve(MMSP::grid<dim,MMSP::vector<T> >& GRID, int n)
{
	int status;
	size_t iter = 0;

	// fixed values

	par.x_Cr = GRID(n)[0];
	par.x_Nb = GRID(n)[1];

	par.n_del = h(fabs(GRID(n)[2]));
	par.n_mu =  h(fabs(GRID(n)[3]));
	par.n_lav = h(fabs(GRID(n)[4]));

	// initial guesses

	gsl_vector_set(x, 0, static_cast<double>(GRID(n)[5]));  // gamma Cr
	gsl_vector_set(x, 1, static_cast<double>(GRID(n)[6]));  //       Nb

	gsl_vector_set(x, 2, static_cast<double>(GRID(n)[7]));  // delta Cr
	gsl_vector_set(x, 3, static_cast<double>(GRID(n)[8]));  //       Nb

	gsl_vector_set(x, 4, static_cast<double>(GRID(n)[9]));  // mu    Cr
	gsl_vector_set(x, 5, static_cast<double>(GRID(n)[10])); //       Ni

	gsl_vector_set(x, 6, static_cast<double>(GRID(n)[11])); // Laves Nb
	gsl_vector_set(x, 7, static_cast<double>(GRID(n)[12])); //       Ni


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
	} while (status == GSL_CONTINUE && iter < maxiter);

	double residual = gsl_blas_dnrm2(solver->f);

	/*
	#ifdef PARABOLIC
	if (status == GSL_SUCCESS) {
	#else
	if (1) {
	#endif
	*/
	/* If res>tol, returned values will be replaced by guesses
	if (1) {
	*/
	if (status == GSL_SUCCESS) {
		GRID(n)[5]  = static_cast<T>(gsl_vector_get(solver->x, 0));  // gamma Cr
		GRID(n)[6]  = static_cast<T>(gsl_vector_get(solver->x, 1));  //       Nb

		GRID(n)[7]  = static_cast<T>(gsl_vector_get(solver->x, 2));  // delta Cr
		GRID(n)[8]  = static_cast<T>(gsl_vector_get(solver->x, 3));  //       Nb

		GRID(n)[9]  = static_cast<T>(gsl_vector_get(solver->x, 4));  // mu    Cr
		GRID(n)[10] = static_cast<T>(gsl_vector_get(solver->x, 5)); //       Nb

		GRID(n)[11] = static_cast<T>(gsl_vector_get(solver->x, 6)); // Laves Cr
		GRID(n)[12] = static_cast<T>(gsl_vector_get(solver->x, 7)); //       Nb
	}

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



/* ========================================== *
 * Invoke GSL to solve for valid phase fields *
 * ========================================== */

double phaseKicker_f(double x, void* params)
{
	double n = ((struct nparams*) params)->n;
	return n - h(fabs(x));
}

double phaseKicker_df(double x, void* params)
{
	return -sign(x) * hprime(fabs(x));
}

void phaseKicker_fdf(double x, void* params, double* y, double* dy)
{
	*y  = phaseKicker_f(x, params);
	*dy = phaseKicker_df(x, params);
}

phasekicker::phasekicker() :
    maxiter(root_max_iter),
    tolerance(1.0e-6)
{
	// Available algorithms are: newton, secant, steffenson (decreasing convergence rate)
	algorithm = gsl_root_fdfsolver_newton;
	solver = gsl_root_fdfsolver_alloc(algorithm);
}


phasekicker::~phasekicker()
{
	gsl_root_fdfsolver_free(solver);
}

template<typename T>
int phasekicker::kick(const double& n, T& p)
{
	int status = GSL_SUCCESS;

	if (n < epsilon) {
		p = 0.0;
	} else {
		double x0 = sign(p);
		double r = std::min(sign(p)*1.1, 2.0*p); // try descending from high side
		int iter = 0;

		par.n = n;

		F.f   = &phaseKicker_f;
		F.df  = &phaseKicker_df;
		F.fdf = &phaseKicker_fdf;
		F.params = &par;

		gsl_root_fdfsolver_set(solver, &F, r);

		do {
			iter++;
			status = gsl_root_fdfsolver_iterate(solver);
			x0 = r;
			r = gsl_root_fdfsolver_root(solver);
			status = gsl_root_test_delta(r, x0, 0, tolerance);
		} while (status == GSL_CONTINUE && iter < maxiter);

		if (status == GSL_SUCCESS)
			p = r;
	}
	return status;
}


#endif

#include"MMSP.main.hpp"
