// alloy625.cpp
// Algorithms for 2D and 3D isotropic binary alloy solidification
// using RWTH database for Cu-Ni system, CuNi_RWTH.tdb,
// extracted from the COST507 Light Alloys Database.
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
#include<gsl/gsl_blas.h>
#include<gsl/gsl_vector.h>
#include<gsl/gsl_multiroots.h>
#include<gsl/gsl_interp2d.h>
#include<gsl/gsl_spline2d.h>

#include"MMSP.hpp"
#include"alloy625.hpp"

// Note: alloy625.hpp contains important declarations and comments. Have a look.

// Equilibrium compositions from CALPHAD: gamma, gamma-prime, gamma-double-prime, delta
const double cAlEq[4] = {0.5413, 0.0, 0.0, 0.3940};
const double cNbEq[4] = {0.5413, 0.0, 0.0, 0.3940};

// Numerical stability (Courant-Friedrich-Lewy) parameters
const double epsilon = 1.0e-10;  // what to consider zero to avoid log(c) explosions
const double CFL = 0.2; // controls timestep


const bool useNeumann = true;

const bool planarTest = false;


// Kinetic and model parameters
const double meshres = 0.075; // dx=dy
const double eps_sq = 1.25;
const double a_int = 2.5; // alpha, prefactor of interface width
const double halfwidth = 5.0*meshres; // half the interface width
const double omega = 2.0*eps_sq*pow(a_int/halfwidth,2.0);
const double dt_plimit = CFL*meshres/eps_sq;          // speed limit based on numerical viscosity
const double dt_climit = CFL*pow(meshres,2.0)/eps_sq; // speed limit based on diffusion timescale
const double dt = std::min(dt_plimit, dt_climit);
const double ps0 = 1.0, pl0 = 0.0; // initial phase fractions
const double cAl[6] = {Cle + 0.50*(Cse-Cle)}; // initial Al concentration
const double cNb[6] = {Cle + 0.50*(Cse-Cle)}; // initial Nb concentration


/* =============================================== *
 * Implement MMSP kernels: generate() and update() *
 * =============================================== */

namespace MMSP
{

/* Representation includes five field variables:
 *
 * X0.  molar fraction of Al, Cr, Mo
 * X1.  molar fraction of Nb, Fe
 * bal. molar fraction of Ni
 *
 * P2.  phase fraction of gamma'
 * P3.  phase fraction of gamma''
 * P4.  phase fraction of delta
 * bal. phase fraction of gamma
 */

const double noise_amp = 0.00625;
const double Calpha = 0.05;
const double Cbeta = 0.95;
const double Cmatrix = 0.5*(Calpha+Cbeta);
const double A = 2.0;
const double B = A/pow(Cbeta-Cmatrix,2);
const double gamma = 2.0/pow(Cbeta-Calpha,2);
const double delta = 1.0;
const double epsilon = 3.0;
const double Dalpha = gamma/pow(delta,2);
const double Dbeta = gamma/pow(delta,2);
const double kappa = 2.0;


// Define equilibrium phase compositions at global scope
//                     gamma   gamma'  gamma'' delta
const double xAl[3] = {0.0161, 0.0007, 0.1870, 0.0007};
const double xNb[3] = {0.0072, 0.0196, 0.0157, 0.0196};
const double xNbdep = 0.5*xNb[0]; // leftover Nb in depleted gamma phase near delta particle

// Define st.dev. of Gaussians for alloying element segregation
//                     Al      Nb
const double bell[2] = {0.0625, 0.025};

double radius(const vector<int>& a, const vector<int>& b, const double& dx)
{
	double r = 0.0;
	for (int i=0; i<length(a) && i<length(b); i++)
		r += std::pow(a[i]-b[i],2.0);
	return dx*std::sqrt(r);
}

double bellCurve(double x, double m, double s)
{
	return std::exp( -std::pow(x-m,2.0)/(2.0*s*s) );
}

void generate(int dim, const char* filename)
{
	int rank=0;
	#ifdef MPI_VERSION
	rank = MPI::COMM_WORLD.Get_rank();
	#endif
	// Utilize Mersenne Twister from C++11 standard
	std::mt19937_64 mt_rand(time(NULL)+rank);
	std::uniform_real_distribution<double> real_gen(-1,1);

	if (dim==2) {
		const int edge = 128;
		GRID2D initGrid(5,0,edge,0,edge);
		vector<int> origin(2,edge/2);
		for (int d=0; d<dim; d++)
			dx(initGrid,d)=0.5;

		const double rDelta = 6.5*dx(initGrid,0);
		const double rDeplt = rDelta*std::sqrt(1.0+2.0*xNb[1]/xNbdep); // radius of annular depletion region
		if (rDeplt > edge/2)
			std::cerr<<"Warning: domain too small to accommodate particle. Push beyond "<<rDeplt<<".\n"<<std::endl;

		double delta_mass=0.0;
		for (int n=0; n<nodes(initGrid); n++) {
			vector<int> x = position(initGrid, n);
			const double r = radius(origin, x, dx(initGrid,0));
			if (r > rDelta) {
				// Gamma matrix
				initGrid(n)[0] = xAl[0]*(1.0 + noise_amp*real_gen(mt_rand))
				                 + 0.25*xAl[0]*bellCurve(dx(initGrid,0)*x[0], 0,                     bell[0]*dx(initGrid,0)*edge)
				                 + 0.25*xAl[0]*bellCurve(dx(initGrid,0)*x[0], dx(initGrid,0)*edge/2, bell[0]*dx(initGrid,0)*edge)
				                 + 0.25*xAl[0]*bellCurve(dx(initGrid,0)*x[0], dx(initGrid,0)*edge,   bell[0]*dx(initGrid,0)*edge);
				initGrid(n)[1] = xNb[0]*(1.0 + noise_amp*real_gen(mt_rand))
				                 + xNb[0]*bellCurve(dx(initGrid,0)*x[0], 0,                     bell[1]*dx(initGrid,0)*edge)
				                 + xNb[0]*bellCurve(dx(initGrid,0)*x[0], dx(initGrid,0)*edge/2, bell[1]*dx(initGrid,0)*edge)
				                 + xNb[0]*bellCurve(dx(initGrid,0)*x[0], dx(initGrid,0)*edge,   bell[1]*dx(initGrid,0)*edge);
				for (int i=2; i<fields(initGrid); i++)
					initGrid(n)[i] = 0.01*real_gen(mt_rand);

				if (r<rDeplt) { // point falls within the depletion region
					//double deltaxNb = xNb[0]-xNbdep - xNbdep*(r-rDelta)/(rDeplt-rDelta);
					double deltaxNb = xNbdep - xNbdep*(r-rDelta)/(rDeplt-rDelta);
					initGrid(n)[1] -= deltaxNb;
				}
			} else {
				// Delta particle
				delta_mass+=1.0;
				initGrid(n)[0] = xAl[3]*(1.0 + noise_amp*real_gen(mt_rand));
				initGrid(n)[1] = xNb[3]*(1.0 + noise_amp*real_gen(mt_rand));
				initGrid(n)[4] = 1.0;
				for (int i=2; i<4; i++)
					initGrid(n)[i] = 0.0;
			}
		}

		vector<double> totals(fields(initGrid),0.0);
		for (int n=0; n<nodes(initGrid); n++) {
			for (int i=0; i<2; i++)
				totals[i] += initGrid(n)[i];
			for (int i=2; i<fields(initGrid); i++)
				totals[i] += std::fabs(initGrid(n)[i]);
		}

		for (int i=0; i<fields(initGrid); i++)
			totals[i] /= double(edge*edge);
		#ifdef MPI_VERSION
		vector<double> myTot(totals);
		for (int i=0; i<fields(initGrid); i++) {
			MPI::COMM_WORLD.Reduce(&myTot[i], &totals[i], 1, MPI_DOUBLE, MPI_SUM, 0);
			MPI::COMM_WORLD.Barrier();
		}
		#endif
		if (rank==0) {
			std::cout<<"x_Ni      x_Al      x_Nb\n";
			printf("%.6f  %1.2e  %1.2e\n\n", 1.0-totals[0]-totals[1], totals[0], totals[1]);
			std::cout<<"p_g       p_g'      p_g''     p_d\n";
			printf("%.6f  %1.2e  %1.2e  %1.2e\n", 1.0-totals[2]-totals[3]-totals[4], totals[2], totals[3], totals[4]);
		}

		output(initGrid,filename);
	}

}

template <int dim, typename T> void update(grid<dim,vector<T> >& oldGrid, int steps)
{
	int rank=0;
	#ifdef MPI_VERSION
	rank = MPI::COMM_WORLD.Get_rank();
	#endif

	// Construct the common tangent solver
	rootsolver CommonTangentSolver;

	ghostswap(oldGrid);
	grid<dim,vector<T> > newGrid(oldGrid);
	double dV=1.0;
	for (int d=0; d<dim; d++) {
		dx(oldGrid,d) = meshres;
		dx(newGrid,d) = meshres;
		dV *= dx(oldGrid,d);
		if (useNeumann && x0(oldGrid,d) == g0(oldGrid,d)) {
			b0(oldGrid,d) = Neumann;
			b0(newGrid,d) = Neumann;
		} else if (useNeumann && x1(oldGrid,d) == g1(oldGrid,d)) {
			b1(oldGrid,d) = Neumann;
			b1(newGrid,d) = Neumann;
		}
	}

	std::ofstream cfile;
	if (rank==0)
		cfile.open("c.log",std::ofstream::out | std::ofstream::app);

	for (int step=0; step<steps; step++) {
		if (rank==0)
			print_progress(step, steps);

		double ctot=0.0, ftot=0.0, utot=0.0, vmax=0.0;
		#ifndef MPI_VERSION
		#pragma omp parallel for
		#endif
		for (int n=0; n<nodes(oldGrid); n++) {
			/* ============================================== *
			 * Point-wise kernel for parallel PDE integration *
			 * ============================================== */

			vector<int> x = position(oldGrid,n);

			// Cache some frequently-used reference values
			const T& phi_old = oldGrid(n)[0];
			const T& c_old   = oldGrid(n)[1];
			const T& Cs_old  = oldGrid(n)[2];
			const T& Cl_old  = oldGrid(n)[3];


			/* ======================================= *
			 * Compute Second-Order Finite Differences *
			 * ======================================= */

			double divGradP = 0.0;
			double divGradC = 0.0;
			double lapPhi = 0.0;
			double gradPsq = 0.0;
			vector<int> s(x);
			for (int d=0; d<dim; d++) {
				double weight = 1.0/pow(dx(oldGrid,d), 2.0);

				if (x[d] == x0(oldGrid,d) &&
				    x0(oldGrid,d) == g0(oldGrid,d) &&
				    useNeumann) {
					// Central second-order difference at lower boundary:
					// Flux_lo = grad(phi)_(i-1/2) is defined to be 0
					// Get high values
					s[d] += 1;
					const T& ph = oldGrid(s)[0];
					const T& ch = oldGrid(s)[1];
					const T& Sh = oldGrid(s)[2];
					const T& Lh = oldGrid(s)[3];
					const T Mph = Q(ph,Sh,Lh)*hprime(ph)*(Lh-Sh);
					const T Mch = Q(ph,Sh,Lh);
					// Get central values
					s[d] -= 1;
					const T& pc = oldGrid(s)[0];
					const T& cc = oldGrid(s)[1];
					const T& Sc = oldGrid(s)[2];
					const T& Lc = oldGrid(s)[3];
					const T Mpc = Q(pc,Sc,Lc)*hprime(pc)*(Lc-Sc);
					const T Mcc = Q(pc,Sc,Lc);

					// Put 'em all together
					divGradP += 0.5*weight*( (Mph+Mpc)*(ph-pc) );
					divGradC += 0.5*weight*( (Mch+Mcc)*(ch-cc) );
					lapPhi   += weight*(ph-pc);
				} else if (x[d] == x1(oldGrid,d)-1 &&
				           x1(oldGrid,d) == g1(oldGrid,d) &&
				           useNeumann) {
					// Central second-order difference at upper boundary:
					// Flux_hi = grad(phi)_(i+1/2) is defined to be 0
					// Get low values
					s[d] -= 1;
					const T& pl = oldGrid(s)[0];
					const T& cl = oldGrid(s)[1];
					const T& Sl = oldGrid(s)[2];
					const T& Ll = oldGrid(s)[3];
					const T Mpl = Q(pl,Sl,Ll)*hprime(pl)*(Ll-Sl);
					const T Mcl = Q(pl,Sl,Ll);
					// Get central values
					s[d] += 1;
					const T& pc = oldGrid(s)[0];
					const T& cc = oldGrid(s)[1];
					const T& Sc = oldGrid(s)[2];
					const T& Lc = oldGrid(s)[3];
					const T Mpc = Q(pc,Sc,Lc)*hprime(pc)*(Lc-Sc);
					const T Mcc = Q(pc,Sc,Lc);

					// Put 'em all together
					divGradP += 0.5*weight*( (Mpc+Mpl)*(pl-pc) );
					divGradC += 0.5*weight*( (Mcc+Mcl)*(cl-cc) );
					lapPhi   += weight*(pl-pc);
				} else {
					// Central second-order difference
					// Get low values
					s[d] -= 1;
					const T& pl = oldGrid(s)[0];
					const T& cl = oldGrid(s)[1];
					const T& Sl = oldGrid(s)[2];
					const T& Ll = oldGrid(s)[3];
					const T Mpl = Q(pl,Sl,Ll)*hprime(pl)*(Ll-Sl);
					const T Mcl = Q(pl,Sl,Ll);
					// Get high values
					s[d] += 2;
					const T& ph = oldGrid(s)[0];
					const T& ch = oldGrid(s)[1];
					const T& Sh = oldGrid(s)[2];
					const T& Lh = oldGrid(s)[3];
					const T Mph = Q(ph,Sh,Lh)*hprime(ph)*(Lh-Sh);
					const T Mch = Q(ph,Sh,Lh);
					// Get central values
					s[d] -= 1;
					const T& pc = oldGrid(s)[0];
					const T& cc = oldGrid(s)[1];
					const T& Sc = oldGrid(s)[2];
					const T& Lc = oldGrid(s)[3];
					const T Mpc = Q(pc,Sc,Lc)*hprime(pc)*(Lc-Sc);
					const T Mcc = Q(pc,Sc,Lc);

					// Put 'em all together
					divGradP += 0.5*weight*( (Mph+Mpc)*(ph-pc) - (Mpc+Mpl)*(pc-pl) );
					divGradC += 0.5*weight*( (Mch+Mcc)*(ch-cc) - (Mcc+Mcl)*(cc-cl) );
					lapPhi   += weight*( (ph-pc) - (pc-pl) );
					gradPsq  += weight * pow(0.5*(ph-pl), 2.0);
				}
			}

			/* ==================================================================== *
			 * Solve the Equation of Motion for phi: Kim, Kim, & Suzuki Equation 31 *
			 * ==================================================================== */

			newGrid(n)[0] = phi_old + dt*( eps_sq*lapPhi - omega*gprime(phi_old)
			                               + hprime(phi_old)*( fl(Cl_old)-fs(Cs_old)-(Cl_old-Cs_old)*dfl_dc(Cl_old) ));


			/* ================================================================== *
			 * Solve the Equation of Motion for c: Kim, Kim, & Suzuki Equation 33 *
			 * ================================================================== */

			newGrid(n)[1] = c_old + dt*(divGradC + divGradP);


			/* ============================== *
			 * Solve for common tangent plane *
			 * ============================== */

			CommonTangentSolver.solve(newGrid(n)[0], newGrid(n)[1], newGrid(n)[2], newGrid(n)[3]);


			/* ====================================================================== *
			 * Collate summary & diagnostic data in OpenMP- and MPI-compatible manner *
			 * ====================================================================== */

			double myc = dV*newGrid(n)[1];
			double myf = dV*(0.5*eps_sq*gradPsq + f(newGrid(n)[0], newGrid(n)[1], newGrid(n)[2], newGrid(n)[3]));
			double myv = 0.0;
			if (newGrid(n)[0]>0.3 && newGrid(n)[0]<0.7) {
				gradPsq = 0.0;
				for (int d=0; d<dim; d++) {
					double weight = 1.0/pow(dx(newGrid,d), 2.0);
					s[d] -= 1;
					const T& pl = newGrid(s)[0];
					s[d] += 2;
					const T& ph = newGrid(s)[0];
					s[d] -= 1;
					gradPsq  += weight * pow(0.5*(ph-pl), 2.0);
				}
				myv = (newGrid(n)[0] - phi_old) / (dt * std::sqrt(gradPsq));
			}
			double myu = (fl(newGrid(n)[3])-fs(newGrid(n)[2]))/(Cle - Cse);

			#ifndef MPI_VERSION
			#pragma omp critical
			{
			#endif
				vmax = std::max(vmax,myv); // maximum velocity
				ctot += myc;               // total mass
				ftot += myf;               // total free energy
				utot += myu*myu;           // deviation from equilibrium
				#ifndef MPI_VERSION
			}
				#endif

			/* ======= *
			 * ~ fin ~ *
			 * ======= */
		}
		swap(oldGrid,newGrid);
		ghostswap(oldGrid);

		double ntot(nodes(oldGrid));
		#ifdef MPI_VERSION
		double myvm(vmax);
		double myct(ctot);
		double myft(ftot);
		double myut(utot);
		double myn(ntot);
		MPI::COMM_WORLD.Allreduce(&myct, &ctot, 1, MPI_DOUBLE, MPI_SUM);
		MPI::COMM_WORLD.Allreduce(&myft, &ftot, 1, MPI_DOUBLE, MPI_SUM);
		MPI::COMM_WORLD.Allreduce(&myvm, &vmax, 1, MPI_DOUBLE, MPI_MAX);
		MPI::COMM_WORLD.Allreduce(&myut, &utot, 1, MPI_DOUBLE, MPI_SUM);
		MPI::COMM_WORLD.Allreduce(&myn,  &ntot, 1, MPI_DOUBLE, MPI_SUM);
		#endif
		double CFLmax = (vmax * dt) / meshres;
		utot = std::sqrt(utot/ntot);
		if (rank==0)
			cfile<<ctot<<'\t'<<ftot<<'\t'<<CFLmax<<'\t'<<utot<<std::endl;
	}
	if (rank==0)
		cfile.close();

	print_values(oldGrid, rank);
}


} // namespace MMSP

template<int dim, typename T>
void print_values(const MMSP::grid<dim,MMSP::vector<T> >& oldGrid, const int rank)
{
	double pTot=0.0;
	double cTot=0.0;
	unsigned int nTot = nodes(oldGrid);
	for (int n=0; n<nodes(oldGrid); n++) {
		pTot += oldGrid(n)[0];
		cTot += oldGrid(n)[1];
	}

	#ifdef MPI_VERSION
	double myP(pTot), myC(cTot);
	unsigned int myN(nTot);
	MPI::COMM_WORLD.Allreduce(&myP, &pTot, 1, MPI_DOUBLE, MPI_SUM);
	MPI::COMM_WORLD.Allreduce(&myC, &cTot, 1, MPI_DOUBLE, MPI_SUM);
	MPI::COMM_WORLD.Allreduce(&myN, &nTot, 1, MPI_UNSIGNED, MPI_SUM);
	#endif
	cTot /= nTot;
	double wps = (100.0*pTot)/nTot;
	double wpl = (100.0*(nTot-pTot))/nTot;
	double fs = 100.0*(cTot - Cle)/(Cse-Cle);
	double fl = 100.0*(Cse - cTot)/(Cse-Cle);
	if (rank==0)
		printf("System has %.2f%% solid, %.2f%% liquid, and composition %.2f%% B. Equilibrium is %.2f%% solid, %.2f%% liquid.\n",
		       wps, wpl, 100.0*cTot, fs, fl);
}


double h(const double p)
{
	return p;
}

double hprime(const double p)
{
	return 1.0;
}

double g(const double p)
{
	return pow(p,2.0) * pow(1.0-p,2.0);
}

double gprime(const double p)
{
	return 2.0*p * (1.0-p)*(1.0-2.0*p);
}

double gibbs(const vector<double>& v)
{
	double g  = f_gam(v[0],v[1]) * (1.0 - (h(abs(v[2])) + h(abs(v[3])) + h(abs(v[4]))));
	       g += f_gpr(v[0],v[1]) * h(abs(v[2]));
	       g += f_gdp(v[0],v[1]) * h(abs(v[3]));
	       g += f_del(v[0],v[1]) * h(abs(v[4]));
	       g += w_gpr * v[2]*v[2] * (1.0 - abs(v[2])*(1.0 - abs(v[2]);
	       g += w_gdp * v[3]*v[3] * (1.0 - abs(v[3])*(1.0 - abs(v[3]);
	       g += w_del * v[4]*v[4] * (1.0 - abs(v[4])*(1.0 - abs(v[4]);
	for (int i=2; i<v.length(); i++)
		for (int j=i+1; j<v.length(); j++)
			g += 2.0 * alpha * v[i]*v[i] * v[j]*v[j];

	return omega*g(p) + h(p)*fs(Cs) + (1.0-h(p))*fl(Cl);
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




/* ====================================== *
 * Invoke GSL to solve for common tangent *
 * ====================================== */

/* Given const phase fraction (p) and concentration (c), iteratively determine
 * the solid (Cs) and liquid (Cl) fictitious concentrations that satisfy the
 * equal chemical potential constraint. Pass p and c by const value,
 * Cs and Cl by non-const reference to update in place. This allows use of this
 * single function to both populate the LUT and interpolate values based thereupon.
 */

struct rparams {
	// Composition fields
	double x_Al;
	double x_Nb;

	// Structure fields
	double p_gpr_Al;
	double p_gdp_Al;
	double p_del_Al;

	double p_gpr_Nb;
	double p_gdp_Nb;
	double p_del_Nb;
};


int commonTangent_f(const gsl_vector* x, void* params, gsl_vector* f)
{
	const double x_Al = ((struct rparams*) params)->x_Al;
	const double x_Nb = ((struct rparams*) params)->x_Nb;
	const double p_gpr = ((struct rparams*) params)->p_gpr;
	const double p_gdp = ((struct rparams*) params)->p_gdp;
	const double p_del = ((struct rparams*) params)->p_del;

	const double C_gam_Al = gsl_vector_get(x, 0);
	const double C_gpr_Al = gsl_vector_get(x, 1);
	const double C_gdp_Al = gsl_vector_get(x, 2);
	const double C_del_Al = gsl_vector_get(x, 3);

	const double C_gam_Nb = gsl_vector_get(x, 4);
	const double C_gpr_Nb = gsl_vector_get(x, 5);
	const double C_gdp_Nb = gsl_vector_get(x, 6);
	const double C_del_Nb = gsl_vector_get(x, 7);

	const double f1 = h(p)*Cs + (1.0-h(p))*Cl - c;
	const double f2 = dfs_dc(Cs) - dfl_dc(Cl);

	gsl_vector_set(f, 0, f1);
	gsl_vector_set(f, 1, f2);

	return GSL_SUCCESS;
}


int commonTangent_df(const gsl_vector* x, void* params, gsl_matrix* J)
{
	const double x_Al = ((struct rparams*) params)->x_Al;
	const double x_Nb = ((struct rparams*) params)->x_Nb;
	const double p_gpr = ((struct rparams*) params)->p_gpr;
	const double p_gdp = ((struct rparams*) params)->p_gdp;
	const double p_del = ((struct rparams*) params)->p_del;

	const double C_gam_Al = gsl_vector_get(x, 0);
	const double C_gpr_Al = gsl_vector_get(x, 1);
	const double C_gdp_Al = gsl_vector_get(x, 2);
	const double C_del_Al = gsl_vector_get(x, 3);

	const double C_gam_Nb = gsl_vector_get(x, 4);
	const double C_gpr_Nb = gsl_vector_get(x, 5);
	const double C_gdp_Nb = gsl_vector_get(x, 6);
	const double C_del_Nb = gsl_vector_get(x, 7);

	// Jacobian matrix
	const double sum = h(abs(p_gpr)) + h(abs(p_gdp)) + h(abs(p_del)) ;

	// Need to know the functional form of the chemical potentials to proceed

	gsl_matrix_set(J, 0, 0, 1.0-sum);
	gsl_matrix_set(J, 0, 1, h(abs(p_gpr)));
	gsl_matrix_set(J, 0, 2, h(abs(p_gdp)));
	gsl_matrix_set(J, 0, 3, h(abs(p_del)));

	gsl_matrix_set(J, 1, 0,  d2f_gam_dxAl2(C_gam_Al, x_Nb)));
	gsl_matrix_set(J, 1, 1, -d2f_gpr_dxAl2(C_gpr_Al, x_Nb)));
	gsl_matrix_set(J, 1, 2, 0.0);
	gsl_matrix_set(J, 1, 3, 0.0);

	gsl_matrix_set(J, 2, 0, 0.0);
	gsl_matrix_set(J, 2, 1,  d2f_gpr_dxAl2(C_gpr_Al, x_Nb)));
	gsl_matrix_set(J, 2, 2, -d2f_gdp_dxAl2(C_gdp_Al, x_Nb)));
	gsl_matrix_set(J, 2, 3, 0.0);

	gsl_matrix_set(J, 3, 0, 0.0);
	gsl_matrix_set(J, 3, 1, 0.0);
	gsl_matrix_set(J, 3, 2,  d2f_gdp_dxAl2(C_gdp_Al, x_Nb)));
	gsl_matrix_set(J, 3, 3, -d2f_del_dxAl2(C_del_Al, x_Nb)));

	for (int i=0; i<4; i++)
		for (for j=4; j<8; j++)
			gsl_matrix_set(J, i, j, 0.0);

	gsl_matrix_set(J, 4, 4, 1.0-sum);
	gsl_matrix_set(J, 4, 5, h(abs(p_gpr)));
	gsl_matrix_set(J, 4, 6, h(abs(p_gdp)));
	gsl_matrix_set(J, 4, 7, h(abs(p_del)));

	gsl_matrix_set(J, 5, 4,  d2f_gam_dxAl2(x_Al, C_gam_Nb)));
	gsl_matrix_set(J, 5, 5, -d2f_gpr_dxAl2(x_Al, C_gpr_Nb)));
	gsl_matrix_set(J, 5, 6, 0.0);
	gsl_matrix_set(J, 5, 7, 0.0);

	gsl_matrix_set(J, 6, 4, 0.0);
	gsl_matrix_set(J, 6, 5,  d2f_gpr_dxAl2(x_Al, C_gpr_Nb)));
	gsl_matrix_set(J, 6, 6, -d2f_gdp_dxAl2(x_Al, C_gdp_Nb)));
	gsl_matrix_set(J, 6, 7, 0.0);

	gsl_matrix_set(J, 7, 4, 0.0);
	gsl_matrix_set(J, 7, 5, 0.0);
	gsl_matrix_set(J, 7, 6,  d2f_gdp_dxAl2(x_Al, C_gdp_Nb)));
	gsl_matrix_set(J, 7, 7, -d2f_del_dxAl2(x_Al, C_del_Nb)));

	for (int i=4; i<8; i++)
		for (int j=0; j<4; j++)
			gsl_matrix_set(J, i, j, 0.0);

	return GSL_SUCCESS;
}


int commonTangent_fdf(const gsl_vector* x, void* params, gsl_vector* f, gsl_matrix* J)
{
	commonTangent_f(x, params, f);
	commonTangent_df(x, params, J);

	return GSL_SUCCESS;
}


rootsolver::rootsolver() :
	n(8), // eight equations
	maxiter(5000),
	tolerance(1.0e-12)
{
	x = gsl_vector_alloc(n);

	// configure algorithm
	algorithm = gsl_multiroot_fdfsolver_gnewton; // gnewton, hybridj, hybridsj, newton
	solver = gsl_multiroot_fdfsolver_alloc(algorithm, n);

	mrf = {&commonTangent_f, &commonTangent_df, &commonTangent_fdf, n, &par};
}

template <typename T> double
rootsolver::solve(const T& x_Al, const T& x_Nb, const T& p_gpr, const T& p_gdp, const T& p_del,
                  T& C_gam_Al, T& C_gpr_Al, T& C_gdp_Al, t& C_del_Al, T& C_gam_Nb, T& C_gpr_Nb, T& C_gdp_Nb, t& C_del_Nb)
{
	int status;
	size_t iter = 0;

	// initial guesses
	par.x_Al = x_Al;
	par.x_Nb = x_Nb;
	par.p_gpr = p_gpr;
	par.p_gdp = p_gdp;
	par.p_del = p_del;

	gsl_vector_set(x, 0, C_gam_Al);
	gsl_vector_set(x, 1, C_gpr_Al);
	gsl_vector_set(x, 2, C_gdp_Al);
	gsl_vector_set(x, 3, C_del_Al);

	gsl_vector_set(x, 4, C_gam_Nb);
	gsl_vector_set(x, 5, C_gpr_Nb);
	gsl_vector_set(x, 6, C_gdp_Nb);
	gsl_vector_set(x, 7, C_del_Nb);

	gsl_multiroot_fdfsolver_set(solver, &mrf, x);

	do {
		iter++;
		status = gsl_multiroot_fdfsolver_iterate(solver);
		if (status) // extra points for finishing early!
			break;
		status = gsl_multiroot_test_residual(solver->f, tolerance);
	} while (status==GSL_CONTINUE && iter<maxiter);

	C_gam_Al = static_cast<T>(gsl_vector_get(solver->x, 0));
	C_gpr_Al = static_cast<T>(gsl_vector_get(solver->x, 1));
	C_gdp_Al = static_cast<T>(gsl_vector_get(solver->x, 2));
	C_del_Al = static_cast<T>(gsl_vector_get(solver->x, 3));

	C_gam_Nb = static_cast<T>(gsl_vector_get(solver->x, 4));
	C_gpr_Nb = static_cast<T>(gsl_vector_get(solver->x, 5));
	C_gdp_Nb = static_cast<T>(gsl_vector_get(solver->x, 6));
	C_del_Nb = static_cast<T>(gsl_vector_get(solver->x, 7));

	double residual = gsl_blas_dnrm2(solver->f);

	return residual;
}

rootsolver::~rootsolver()
{
	gsl_multiroot_fdfsolver_free(solver);
	gsl_vector_free(x);
}


#endif

#include"MMSP.main.hpp"
