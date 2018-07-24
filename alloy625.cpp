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

#ifndef __ALLOY625_CPP__
#define __ALLOY625_CPP__

#include <set>
#include <cmath>
#include <random>
#include <sstream>
#include <vector>
#ifdef _OPENMP
#include "omp.h"
#endif
#include "MMSP.hpp"
#include "alloy625.hpp"
#include "parabola625.c"

/* Free energy expressions are generated from CALPHAD using pycalphad and SymPy
 * by the Python script CALPHAD_energy.py. Run it (python CALPHAD_energy.py)
 * to generate parabola625.c.
 *
 * Representation includes ten field variables:
 *
 * X0. molar fraction of Cr + Mo
 * X1. molar fraction of Nb
 *
 * P2. phase fraction of delta
 * P3. phase fraction of Laves
 *
 * C4. Cr molar fraction in pure gamma
 * C5. Nb molar fraction in pure gamma
 *
 * C6. Cr molar fraction in pure delta
 * C7. Nb molar fraction in pure delta
 *
 * C8. Cr molar fraction in pure Laves
 * C9. Nb molar fraction in pure Laves
 *
 * Based on experiments (EDS) and simulations (DICTRA),
 * additively manufactured IN625 has these compositions:
 *
 * Element  Nominal  Interdendritic (mol %)
 * Cr+Mo      30%      31%
 * Nb          2%      13%
 * Ni         68%      56%
 */

/* Define equilibrium phase compositions at global scope. Gamma is nominally
   30% Cr, 2% Nb, as defined in the first elements of the two following arrays.
   The generate() function will adjust the initial gamma composition depending
   on the type, amount, and composition of the secondary phases to maintain the
   system's nominal composition. Parabolic free energy requires a system
   composition inside the three-phase coexistence region.
                          nominal   delta        laves         enriched       */
const field_t xCr[NP+2] = {0.250,   xe_del_Cr(), xe_lav_Cr(),  0.005};
const field_t xNb[NP+2] = {0.150,   xe_del_Nb(), xe_lav_Nb(),  0.050};

/* Define st.dev. of Gaussians for alloying element segregation
                         Cr      Nb                                           */
const double bell[NC] = {150e-9, 50e-9}; // est. between 80-200 nm from SEM

// Kinetic and model parameters
const double meshres = 5e-9; // grid spacing (m)
const field_t alpha = 1.07e11;  // three-phase coexistence coefficient (J/m^3)

/* Diffusion constants in FCC Ni from Xu (m^2/s)
                          Cr        Nb                                        */
const field_t D_Cr[NC] = {2.16e-15, 0.56e-15}; // first column of diffusivity matrix
const field_t D_Nb[NC] = {2.97e-15, 4.29e-15}; // second column of diffusivity matrix

/* Choose numerical diffusivity to lock chemical and transformational timescales
                           delta      Laves                                   */
const field_t kappa[NP] = {1.24e-8, 1.24e-8};     // gradient energy coefficient (J/m)
const field_t Lmob[NP]  = {2.904e-11, 2.904e-11}; // numerical mobility (m^2/Ns)
const field_t sigma[NP] = {1.010, 1.010};         // interfacial energy (J/m^2)

// Compute interfacial width (nm) and well height (J/m^3)
const field_t ifce_width = 10. * meshres;
const field_t width_factor = 2.2; // 2.2 if interface is [0.1,0.9]; 2.94 if [0.05,0.95]
const field_t omega[NP] = {3.0 * width_factor* sigma[0] / ifce_width,  // delta
                           3.0 * width_factor* sigma[1] / ifce_width}; // Laves

// Numerical considerations
const double epsilon = 1e-12;           // what to consider zero to avoid log(c) explosions
const field_t LinStab = 1.0 / 19.42501; // threshold of linear (von Neumann) stability

/* Precipitate radii: minimum for thermodynamic stability is 7.5 nm,
   minimum for numerical stability is 14*dx (due to interface width).         */
const field_t rPrecip[NP] = {7.5 * 5e-9 / meshres,  // delta
                             7.5 * 5e-9 / meshres}; // Laves


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

	const double dtTransformLimited = (meshres*meshres) / (std::pow(2.0, dim) * Lmob[0]*kappa[0]);
	const double dtDiffusionLimited = (meshres*meshres) / (std::pow(2.0, dim) * std::max(D_Cr[0], D_Nb[1]));
	double dt = LinStab * std::min(dtTransformLimited, dtDiffusionLimited);

	// Initialize pseudo-random number generator
	std::random_device rd; // PRNG seed generator
	std::mt19937 mtrand(rd()); // Mersenne Twister
	std::uniform_real_distribution<double> unidist(0, 1);

	if (dim==1) {
		/* ============================================= *
		 * Three-phase Stripe Growth test configuration  *
		 *                                               *
		 * Seed a 1.0 um domain with 3 stripes (one of   *
		 * each secondary phase) to test competitive     *
		 * growth without curvature                      *
		 * ============================================= */

		const int Nx = 768; // divisible by 12 and 64
		double dV = 1.0;
		double Ntot = 1.0;
		GRID1D initGrid(NC+NP+NC*(NP+1), 0, Nx);
		for (int d = 0; d < dim; d++) {
			dx(initGrid,d) = meshres;
			dV *= meshres;
			Ntot *= g1(initGrid, d) - g0(initGrid, d);
			if (x0(initGrid, d) == g0(initGrid, d))
				b0(initGrid, d) = Neumann;
			if (x1(initGrid, d) == g1(initGrid, d))
				b1(initGrid, d) = Neumann;
		}

		// Sanity check on system size and  particle spacing
		if (rank == 0)
			std::cout << "Timestep dt=" << dt
					  << ". Linear stability limits: dtTransformLimited=" << dtTransformLimited
					  << ", dtDiffusionLimited=" << dtDiffusionLimited
					  << '.' << std::endl;

		// Zero initial condition
		const vector<field_t> blank(fields(initGrid), 0);
		#ifdef _OPENMP
		#pragma omp parallel for
		#endif
		for (int n = 0; n < nodes(initGrid); n++) {
			initGrid(n) = blank;
		}

		Composition comp;
		vector<int> origin(dim, 0);
		const int xoffset[NP] = {int(-16 * (5e-9 / meshres)), int(16 * (5e-9 / meshres))}; //  80 nm

		for (int j = 0; j < NP; j++) {
			origin[0] = Nx / 2 + xoffset[j];
			comp += embedStripe(initGrid, origin, j+NC, rPrecip[j], xCr[j+1], xNb[j+1]);
		}

		// Synchronize global initial condition parameters
		#ifdef MPI_VERSION
		Composition myComp;
		myComp += comp;
		MPI::COMM_WORLD.Allreduce(&(myComp.N[0]), &(comp.N[0]), NP+1, MPI_INT, MPI_SUM);
		MPI::COMM_WORLD.Barrier();
		for (int j = 0; j < NP+1; j++) {
			MPI::COMM_WORLD.Barrier();
			MPI::COMM_WORLD.Allreduce(&(myComp.x[j][0]), &(comp.x[j][0]), NC, MPI_DOUBLE, MPI_SUM);
		}
		#endif

		// Initialize matrix phase to achieve specified system composition
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

		#ifdef _OPENMP
		#pragma omp parallel for
		#endif
		for (int n = 0; n < nodes(initGrid); n++) {
			vector<field_t>& initGridN = initGrid(n);
			field_t nx = 0.0;

			for (int i = NC; i < NC+NP; i++)
				nx += h(initGridN[i]);

			if (nx < 0.01) { // pure gamma
				initGridN[0] += matCr;
				initGridN[1] += matNb;
			}

			update_compositions(initGridN);
		}

		ghostswap(initGrid);

		#ifdef MPI_VERSION
		MPI::COMM_WORLD.Barrier();
		double mydt(dt);
		MPI::COMM_WORLD.Allreduce(&mydt, &dt, 1, MPI_DOUBLE, MPI_MIN);
		#endif

		vector<double> summary = summarize_fields(initGrid);
		double energy = summarize_energy(initGrid);

		if (rank == 0) {
			fprintf(cfile, "%9s\t%9s\t%9s\t%9s\t%9s\t%9s\t%9s\t%9s\t%9s\n",
			               "ideal", "timestep", "x_Cr", "x_Nb", "gamma", "delta", "Laves", "free_energy", "ifce_vel");
			fprintf(cfile, "%9g\t%9g\t%9g\t%9g\t%9g\t%9g\t%9g\t%9g\n",
			               dt, dt, summary[0], summary[1], summary[2], summary[3], summary[4], energy);

			printf("%9s %9s %9s %9s %9s %9s\n",
			       "x_Cr", "x_Nb", "x_Ni", " p_g", " p_d", "p_l");
			printf("%9g %9g %9g %9g %9g %9g\n",
			       summary[0], summary[1], 1.0-summary[0]-summary[1], summary[2], summary[3], summary[4]);
		}

		output(initGrid,filename);



	} else if (dim==2) {
		/* =============================================== *
		 * Two-phase Particle Growth test configuration    *
		 *            Small-Precipitate Region             *
		 * Seed a 1.6 um x 0.96 um domain with 2 particles *
		 * (one of each secondary phase) in a single row   *
		 * to test competitive growth with Gibbs-Thomson   *
		 * =============================================== */

		const int Nx = 320; // product divisible by 12 and 64
		const int Ny = 192;
		double dV = 1.0;
		double Ntot = 1.0;
		// GRID2D initGrid(NC+NP+NC*(NP+1), -Nx/2, Nx/2, -Ny/2, Ny/2);
		// GRID2D initGrid(NC+NP+NC*(NP+1), -5*Nx/2, 5*Nx/2, -4*Ny/2, 4*Ny/2);
		GRID2D initGrid(NC+NP+NC*(NP+1), -10*Nx/2, 10*Nx/2, -9*Ny/2, 9*Ny/2);
		for (int d = 0; d < dim; d++) {
			dx(initGrid,d)=meshres;
			dV *= meshres;
			Ntot *= g1(initGrid, d) - g0(initGrid, d);
			if (x0(initGrid, d) == g0(initGrid, d))
				b0(initGrid, d) = Neumann;
			if (x1(initGrid, d) == g1(initGrid, d))
				b1(initGrid, d) = Neumann;
		}

		// Sanity check on system size and  particle spacing
		if (rank == 0)
			std::cout << "Timestep dt=" << dt
					  << ". Linear stability limits: dtTransformLimited=" << dtTransformLimited
					  << ", dtDiffusionLimited=" << dtDiffusionLimited
					  << '.' << std::endl;

		/* Randomly choose system composition in a circular
		 * region of the phase diagram near the gamma corner
		 * of three-phase coexistence triangular phase field
		 */
		// Set system composition
		const double xo = 0.4125;
		const double yo = 0.1000;
		const double ro = 0.0025;
		bool withinRange = false;
		double xCr0;
		double xNb0;

		while (!withinRange) {
			xCr0 = xo + ro * (unidist(mtrand) - 1.);
			xNb0 = yo + ro * (unidist(mtrand) - 1.);
			withinRange = (std::pow(xCr0 - xo, 2.0) + std::pow(xNb0 - yo, 2.0)
						   < std::pow(ro, 2.0));
		}

		#ifdef MPI_VERSION
		MPI::COMM_WORLD.Barrier();
		MPI::COMM_WORLD.Bcast(&xCr0, 1, MPI_DOUBLE, 0);
		MPI::COMM_WORLD.Bcast(&xNb0, 1, MPI_DOUBLE, 0);
		#endif

		// Zero initial condition
		const MMSP::vector<field_t> blank(MMSP::fields(initGrid), 0);
		#ifdef _OPENMP
		#pragma omp parallel for
		#endif
		for (int n = 0; n < MMSP::nodes(initGrid); n++) {
			initGrid(n) = blank;
		}

		Composition comp = init_2D_tiles(initGrid, Ntot, Nx, Ny, xCr0, xNb0, unidist, mtrand);

		// Initialize matrix to achieve specified system composition
		field_t matCr = Ntot * xCr0;
		field_t matNb = Ntot * xNb0;
		double Nmat  = Ntot;
		for (int i = 0; i < NP+1; i++) {
			Nmat  -= comp.N[i];
			matCr -= comp.x[i][0];
			matNb -= comp.x[i][1];
		}
		matCr /= Nmat;
		matNb /= Nmat;

		#ifdef _OPENMP
		#pragma omp parallel for
		#endif
		for (int n = 0; n < nodes(initGrid); n++) {
			MMSP::vector<field_t>& initGridN = initGrid(n);
			field_t nx = 0.0;

			for (int i = NC; i < NC+NP; i++)
				nx += h(initGridN[i]);

			if (nx < 0.01) { // pure gamma
				initGridN[0] += matCr;
				initGridN[1] += matNb;
			}

			update_compositions(initGridN);
		}

		ghostswap(initGrid);

		vector<double> summary = summarize_fields(initGrid);
		double energy = summarize_energy(initGrid);

		if (rank == 0) {
			fprintf(cfile, "%9s\t%9s\t%9s\t%9s\t%9s\t%9s\t%9s\t%9s\t%9s\n",
			               "ideal", "timestep", "x_Cr", "x_Nb", "gamma", "delta", "Laves", "free_energy", "ifce_vel");
			fprintf(cfile, "%9g\t%9g\t%9g\t%9g\t%9g\t%9g\t%9g\t%9g\n",
			               dt, dt, summary[0], summary[1], summary[2], summary[3], summary[4], energy);

			printf("%9s %9s %9s %9s %9s %9s\n",
			       "x_Cr", "x_Nb", "x_Ni", " p_g", " p_d", " p_l");
			printf("%9g %9g %9g %9g %9g %9g\n",
			       summary[0], summary[1], 1.0-summary[0]-summary[1], summary[2], summary[3], summary[4]);
		}

		output(initGrid,filename);



	} else {
		std::cerr << "Error: " << dim << "-dimensional grids unsupported." << std::endl;
		MMSP::Abort(-1);
	}

	if (rank == 0)
		fclose(cfile);
}




template <int dim, typename T> void update(grid<dim,vector<T> >& oldGrid, int steps)
{
	int rank = 0;
	#ifdef MPI_VERSION
	rank = MPI::COMM_WORLD.Get_rank();
	MPI_Request reqs;
	MPI_Status stat;
	#endif

	grid<dim,vector<T> > newGrid(oldGrid);
	grid<dim,vector<T> > lapGrid(oldGrid, 2*NC+NP); // excludes secondary phases' fictitious compositions

	const double dtTransformLimited = (meshres*meshres) / (2.0 * dim * Lmob[0]*kappa[0]);
	const double dtDiffusionLimited = (meshres*meshres) / (2.0 * dim * std::max(D_Cr[0], D_Nb[1]));
	const double dt = LinStab * std::min(dtTransformLimited, dtDiffusionLimited);

	double dV = 1.0;
	double Ntot = 1.0;
	for (int d = 0; d < dim; d++) {
		dx(oldGrid,d) = meshres;
		dx(newGrid,d) = meshres;
		dV *= dx(oldGrid,d);
		Ntot *= double(g1(oldGrid, d) - g0(oldGrid, d));
		if (x0(oldGrid, d) == g0(oldGrid, d)) {
			b0(oldGrid, d) = Neumann;
			b0(newGrid, d) = Neumann;
		}
		if (x1(oldGrid, d) == g1(oldGrid, d)) {
			b1(oldGrid, d) = Neumann;
			b1(newGrid, d) = Neumann;
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

	std::stringstream ostr;

	for (int step = 0; step < steps; step++) {

		if (rank == 0)
			print_progress(step, steps);

		/* ================= *
		 * Compute Laplacian *
		 * ================= */
		#ifdef _OPENMP
		#pragma omp parallel for
		#endif
		for (int n = 0; n < nodes(oldGrid); n++) {
			vector<int> x = position(oldGrid, n);
			lapGrid(n) = pointerlaplacian(oldGrid, x, 2*NC+NP);
		}

		#ifdef _OPENMP
		#pragma omp parallel for
		#endif
		for (int n = 0; n < nodes(oldGrid); n++) {
			/* ============================================== *
			 * Point-wise kernel for parallel PDE integration *
			 * ============================================== */

			vector<int> x = position(oldGrid,n);
			const vector<T>& oldGridN = oldGrid(n);
			vector<T>& newGridN = newGrid(n);

			const T PhaseEnergy[NP+1] = {g_del(oldGridN[2*NC+NP],  oldGridN[2*NC+NP+1]),
			                             g_lav(oldGridN[3*NC+NP],  oldGridN[3*NC+NP+1]),
			                             g_gam(oldGridN[  NC+NP],  oldGridN[  NC+NP+1])};

			// Diffusion potential in matrix (will equal chemical potential at equilibrium)
			const field_t dgGdxCr = dg_gam_dxCr(oldGridN[NC+NP], oldGridN[NC+NP+1]);
			const field_t dgGdxNb = dg_gam_dxNb(oldGridN[NC+NP], oldGridN[NC+NP+1]);
			const T chempot[NC] = {dgGdxCr, dgGdxNb};

			// sum of phase fractions-squared
			T sumPhiSq = 0.0;
			for (int i = NC; i < NC+NP; i++)
				sumPhiSq += oldGridN[i] * oldGridN[i];

			// Laplacians of field variables, including fictitious compositions of matrix phase
			const vector<T>& laplac = lapGrid(n);

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

			/* =============================== *
			 * Compute Fictitious Compositions *
			 * =============================== */
			update_compositions(newGridN);

		} // end loop over grid points

		swap(oldGrid, newGrid);
		ghostswap(oldGrid);

		if (logcount >= logstep) {
			logcount = 0;

			// Update timestep based on interfacial velocity.
			// If v==0, there's no interface: march ahead with current dt.
			double interfacialVelocity = maxVelocity(newGrid, current_dt, oldGrid);
			double ideal_dt = (interfacialVelocity>epsilon) ? advectionlimit / interfacialVelocity : current_dt;

			#ifdef MPI_VERSION
			double myt(ideal_dt);
			MPI_Iallreduce(&myt, &ideal_dt, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD, &reqs);
			MPI_Wait(&reqs, &stat);
			#endif

			if (current_dt < ideal_dt + epsilon) {
				// Update succeeded: process solution
				current_time += current_dt;

				/* ====================================================================== *
				 * Collate summary & diagnostic data in OpenMP- and MPI-compatible manner *
				 * ====================================================================== */

				vector<double> summary = summarize_fields(newGrid);
				double energy = summarize_energy(newGrid);
				char buffer[4096];

				if (rank == 0) {
					sprintf(buffer, "%9g\t%9g\t%9g\t%9g\t%9g\t%9g\t%9g\t%9g\t%9g\n",
					        ideal_dt, current_dt, summary[0], summary[1], summary[2], summary[3], summary[4], energy, interfacialVelocity);
					ostr << buffer;
				}
			} else {
				if (rank == 0) {
					cfile.close();
				}
				std::cerr << "ERROR: Interface swept more than (dx/" << meshres/advectionlimit << "), timestep is too aggressive!" << std::endl;
				MMSP::Abort(-1);
			}

			assert(current_dt > epsilon);
		}

		logcount++; // increment after output block

		/*
		#ifdef MPI_VERSION
		MPI::COMM_WORLD.Barrier();
		double mydt(current_dt);
		MPI::COMM_WORLD.Allreduce(&mydt, &current_dt, 1, MPI_DOUBLE, MPI_MIN);
		#endif
		*/
	} // end timestepping loop

	if (rank == 0) {
		cfile << ostr.str(); // write log data to disk
		ostr.str(""); // clear log data
	}

	if (rank == 0) {
		cfile.close();
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

template<typename T>
void update_compositions(MMSP::vector<T>& GRIDN)
{
	const T& xcr = GRIDN[0];
	const T& xnb = GRIDN[1];

	const T fdel = h(GRIDN[2]);
	const T flav = h(GRIDN[3]);
	const T fgam = 1. - fdel - flav;

	GRIDN[1*NC+NP  ] = fict_gam_Cr(xcr, xnb, fdel, fgam, flav);
	GRIDN[1*NC+NP+1] = fict_gam_Nb(xcr, xnb, fdel, fgam, flav);
	GRIDN[2*NC+NP  ] = fict_del_Cr(xcr, xnb, fdel, fgam, flav);
	GRIDN[2*NC+NP+1] = fict_del_Nb(xcr, xnb, fdel, fgam, flav);
	GRIDN[3*NC+NP  ] = fict_lav_Cr(xcr, xnb, fdel, fgam, flav);
	GRIDN[3*NC+NP+1] = fict_lav_Nb(xcr, xnb, fdel, fgam, flav);
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

	return comp;
}

template<int dim, typename T>
Composition init_2D_tiles(MMSP::grid<dim,MMSP::vector<T> >& GRID, const double Ntot,
						  const int width, const int height,
						  const double xCr0, const double xNb0,
						  std::uniform_real_distribution<double>& unidist, std::mt19937& mtrand)
{
	#ifdef MPI_VERSION
	MPI_Request* reqs = new MPI_Request[NP + 2];
	MPI_Status* stat = new MPI_Status[NP + 2];
	#endif

	// Set constant precipitate radii and separation
	int r = rPrecip[0];
	int d = 8 + height / 2;

	/*
    // Set precipitate radii and separation with jitter
	const int d = 8 + height / 2 - (unidist(mtrand) * height)/4;
	const int r = std::floor((3. + 4.5 * unidist(mtrand)) * (5.0e-9 / meshres));

	#ifdef MPI_VERSION
	MPI::COMM_WORLD.Barrier();
	MPI::COMM_WORLD.Bcast(&r,1, MPI_INT, 0);
	MPI::COMM_WORLD.Bcast(&d,1, MPI_INT, 0);
	#endif
	*/

	// curvature-dependent initial compositions
	const field_t P_del = 2.0 * sigma[0] / (rPrecip[0] * meshres);
	const field_t P_lav = 2.0 * sigma[1] / (rPrecip[1] * meshres);
	const field_t xrCr[NP+1] = {xr_gam_Cr(P_del, P_lav),
								xr_del_Cr(P_del, P_lav),
								xr_lav_Cr(P_del, P_lav)};
	const field_t xrNb[NP+1] = {xr_gam_Nb(P_del, P_lav),
								xr_del_Nb(P_del, P_lav),
								xr_lav_Nb(P_del, P_lav)};

	Composition comp;
	MMSP::vector<int> tileOrigin(dim, 0);
	int n = 0;

	for (tileOrigin[1] = MMSP::g0(GRID,1) - MMSP::g0(GRID,1) % height; tileOrigin[1] < 1 + MMSP::g1(GRID,1); tileOrigin[1] += height) {
		for (tileOrigin[0] = MMSP::g0(GRID,0) - (n%2)*(3*width)/8 - MMSP::g0(GRID,0) % width; tileOrigin[0] < 1 + MMSP::g1(GRID,0); tileOrigin[0] += width) {
			MMSP::vector<int> origin = tileOrigin;
			for (int j = 0; j < NP; j++) {
				origin[0] = tileOrigin[0] + ((j%2==0) ? -d/2 : d/2);
				origin[1] = tileOrigin[1];
				comp += embedParticle(GRID, origin, j+NC, r, xrCr[j+1], xrNb[j+1]);
			}
		}
		n++;
	}

	// Synchronize global initial condition parameters
	#ifdef MPI_VERSION
	Composition myComp;
	myComp += comp;
	MPI::COMM_WORLD.Barrier();
	MPI_Iallreduce(&(myComp.N[0]), &(comp.N[0]), NP+1, MPI_INT, MPI_SUM, MPI_COMM_WORLD, &(reqs[0]));
	for (int j = 0; j < NP+1; j++) {
		MPI_Iallreduce(&(myComp.x[j][0]), &(comp.x[j][0]), NC, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &(reqs[1+j]));
	}
	MPI_Waitall(NP+2, reqs, stat);
	#endif

	#ifdef MPI_VERSION
	delete [] reqs;
	delete [] stat;
	#endif

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

template <int dim, typename T>
MMSP::vector<T> pointerlaplacian(const MMSP::grid<dim,MMSP::vector<T> >& GRID, const MMSP::vector<int>& x, const int N)
{
	if (dim > 2) {
		std::cerr << "ERROR: pointerlaplacian is only available for 1- and 2-D fields." << std::endl;
		MMSP::Abort(-1);
	}

	MMSP::vector<T> laplacian(N, 0.0);

	const double wx = 1.0 / (MMSP::dx(GRID, 0) * MMSP::dx(GRID, 0));
	const int deltax = (dim == 1) ? 1 : 2 * MMSP::ghosts(GRID) + MMSP::y1(GRID) - MMSP::y0(GRID);
	const int deltay = 1;

	const MMSP::vector<T>* const c = &(GRID(x));
	const MMSP::vector<T>* const l = (MMSP::b0(GRID,0)==MMSP::Neumann && x[0]==MMSP::x0(GRID)  ) ? c : c - deltax;
	const MMSP::vector<T>* const h = (MMSP::b1(GRID,0)==MMSP::Neumann && x[0]==MMSP::x1(GRID)-1) ? c : c + deltax;
	for (int j = 0; j < N; j++)
		laplacian[j] += wx * ((*h)[j] - 2. * (*c)[j] + (*l)[j]);

	if (dim == 2) {
		const double wy = 1.0 / (MMSP::dx(GRID, 1) * MMSP::dx(GRID, 1));
		const MMSP::vector<T>* const cl = (MMSP::b0(GRID,1)==MMSP::Neumann && x[1]==MMSP::y0(GRID)  ) ? c : c - deltay;
		const MMSP::vector<T>* const ch = (MMSP::b1(GRID,1)==MMSP::Neumann && x[1]==MMSP::y1(GRID)-1) ? c : c + deltay;
		for (int j = 0; j < N; j++) {
			laplacian[j] +=  wy * ((*ch)[j] - 2. * (*c)[j] + (*cl)[j]);
		}
	}

	return laplacian;
}


template<int dim,class T>
T maxVelocity(MMSP::grid<dim,MMSP::vector<T> > const& oldGrid, const double& dt,
              MMSP::grid<dim,MMSP::vector<T> > const& newGrid)
{
	#ifdef MPI_VERSION
	MPI_Request reqs;
	MPI_Status stat;
	#endif

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
	double temp(vmax);
	MPI_Iallreduce(&temp, &vmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD, &reqs);
	MPI_Wait(&reqs, &stat);
	#endif

	return vmax;
}


template<int dim,class T>
MMSP::vector<double> summarize_fields(MMSP::grid<dim,MMSP::vector<T> > const& GRID)
{
	#ifdef MPI_VERSION
	MPI_Request reqs;
	MPI_Status stat;
	#endif

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
	MPI_Ireduce(&(temp[0]), &(summary[0]), NC+NP+1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD, &reqs);
	MPI_Wait(&reqs, &stat);
	#endif

	return summary;
}

template<int dim,class T>
double summarize_energy(MMSP::grid<dim,MMSP::vector<T> > const& GRID)
{
	#ifdef MPI_VERSION
	MPI_Request reqs;
	MPI_Status stat;
	#endif

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
	MPI_Ireduce(&temp, &energy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD, &reqs);
	MPI_Wait(&reqs, &stat);
	#endif

	return energy;
}

#endif

#include"MMSP.main.hpp"
