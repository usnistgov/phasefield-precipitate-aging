// alloy718.cpp
// Algorithms for 2D and 3D solid state transformation in nickel-based superalloy
// Questions/comments to trevor.keller@nist.gov (Trevor Keller)

#ifndef ALLOY718_UPDATE
#define ALLOY718_UPDATE
#include<cmath>
#include<random>
#include"MMSP.hpp"
#include"alloy718.hpp"

namespace MMSP{

/* Representation includes seven field variables:
 * X0. molar fraction of Al, Cr, Mo
 * X1. molar fraction of Nb, Fe
 * 1.0-X0-X1 molar fraction of Ni
 * P2. phase fraction of gamma'
 * P3. phase fraction of gamma'', variant 1
 * P4. phase fraction of gamma'', variant 2
 * P5. phase fraction of gamma'', variant 3
 * P6. phase fraction of delta
 * 1.0-P2-P3-P4-P5-P6 phase fraction of gamma
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


//                     gamma   delta   gamma'
const double xAl[3] = {0.0161, 0.0007, 0.1870};
const double xNb[3] = {0.0072, 0.0196, 0.0157};
const double xNbdep = xNb[0]/2; // leftover Nb in depleted gamma phase near delta particle

//                     Al      Nb
const double sig[2] = {0.0625, 0.025};

double radius(const vector<int>& a, const vector<int>& b, const double& dx) {
	double r = 0.0;
	for (int i=0; i<length(a) && i<length(b); i++)
		r += std::pow(a[i]-b[i],2.0);
	return dx*std::sqrt(r);
}

double bellCurve(double x, double m, double s) {
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

	if (dim==1) {
		const int edge = 128;
		GRID1D initGrid(7,0,edge);
		vector<int> origin(1,edge/2);
		for (int d=0; d<dim; d++)
			dx(initGrid,d)=0.5;

		const double rDelta = 6.5*dx(initGrid,0);
		const double rDeplt = rDelta*(1.0+2.0*xNb[1]/xNbdep); // radius of annular depletion region
		if (rDeplt > edge/2)
			std::cerr<<"Warning: domain too small to accommodate particle. Push beyond "<<rDeplt<<".\n"<<std::endl;

		for (int n=0; n<nodes(initGrid); n++) {
			vector<int> x = position(initGrid, n);
			const double r = radius(origin, x, dx(initGrid,0));
			if (r > rDelta) {
				// Gamma matrix
				initGrid(n)[0] = xAl[0]*(1.0 + noise_amp*real_gen(mt_rand))
				               + 0.25*xAl[0]*bellCurve(dx(initGrid,0)*x[0], 0,                     sig[0]*dx(initGrid,0)*edge)
				               + 0.25*xAl[0]*bellCurve(dx(initGrid,0)*x[0], dx(initGrid,0)*edge/2, sig[0]*dx(initGrid,0)*edge)
				               + 0.25*xAl[0]*bellCurve(dx(initGrid,0)*x[0], dx(initGrid,0)*edge,   sig[0]*dx(initGrid,0)*edge);
				initGrid(n)[1] = xNb[0]*(1.0 + noise_amp*real_gen(mt_rand))
				               + xNb[0]*bellCurve(dx(initGrid,0)*x[0], 0,                     sig[1]*dx(initGrid,0)*edge)
				               + xNb[0]*bellCurve(dx(initGrid,0)*x[0], dx(initGrid,0)*edge/2, sig[1]*dx(initGrid,0)*edge)
				               + xNb[0]*bellCurve(dx(initGrid,0)*x[0], dx(initGrid,0)*edge,   sig[1]*dx(initGrid,0)*edge);
				for (int i=2; i<7; i++)
					initGrid(n)[i] = 0.01*real_gen(mt_rand);

				if (r<rDeplt) { // point falls within the depletion region
					double deltaxNb = xNb[0]-xNbdep - xNbdep*(r-rDelta)/(rDeplt-rDelta);
					initGrid(n)[1] -= deltaxNb;
				}
			} else {
				// Delta particle
				initGrid(n)[0] = xAl[1]*(1.0 + noise_amp*real_gen(mt_rand));
				initGrid(n)[1] = xNb[1]*(1.0 + noise_amp*real_gen(mt_rand));
				initGrid(n)[6] = 1.0;
				for (int i=2; i<6; i++)
					initGrid(n)[i] = 0.0;
			}
		}

		vector<double> totals(7,0.0);
		for (int n=0; n<nodes(initGrid); n++) {
			for (int i=0; i<2; i++)
				totals[i] += initGrid(n)[i];
			for (int i=2; i<7; i++)
				totals[i] += std::fabs(initGrid(n)[i]);
		}

		for (int i=0; i<7; i++)
			totals[i] /= 1.0*edge;
		#ifdef MPI_VERSION
		vector<double> myTot = totals;
		for (int i=0; i<7; i++)
			MPI::COMM_WORLD.Reduce(&myTot[i], &totals[i], 1, MPI_DOUBLE, MPI_SUM, 0);
		#endif
		if (rank==0) {
			std::cout<<"x_Ni      x_Al      x_Nb\n";
			printf("%.6f  %1.2e  %1.2e\n\n", 1.0-totals[0]-totals[1], totals[0], totals[1]);
			std::cout<<"p_g       p_g'      p_g''     p_g''     p_g''     p_d\n";
			printf("%.6f  %1.2e  %1.2e  %1.2e  %1.2e  %1.2e\n", 1.0-totals[2]-totals[3]-totals[4]-totals[5]-totals[6], totals[2], totals[3], totals[4], totals[5], totals[6]);
		}

		output(initGrid,filename);
	}

	if (dim==2) {
		const int edge = 128;
		GRID2D initGrid(7,0,edge,0,edge);
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
				               + 0.25*xAl[0]*bellCurve(dx(initGrid,0)*x[0], 0,                     sig[0]*dx(initGrid,0)*edge)
				               + 0.25*xAl[0]*bellCurve(dx(initGrid,0)*x[0], dx(initGrid,0)*edge/2, sig[0]*dx(initGrid,0)*edge)
				               + 0.25*xAl[0]*bellCurve(dx(initGrid,0)*x[0], dx(initGrid,0)*edge,   sig[0]*dx(initGrid,0)*edge);
				initGrid(n)[1] = xNb[0]*(1.0 + noise_amp*real_gen(mt_rand))
				               + xNb[0]*bellCurve(dx(initGrid,0)*x[0], 0,                     sig[1]*dx(initGrid,0)*edge)
				               + xNb[0]*bellCurve(dx(initGrid,0)*x[0], dx(initGrid,0)*edge/2, sig[1]*dx(initGrid,0)*edge)
				               + xNb[0]*bellCurve(dx(initGrid,0)*x[0], dx(initGrid,0)*edge,   sig[1]*dx(initGrid,0)*edge);
				for (int i=2; i<7; i++)
					initGrid(n)[i] = 0.01*real_gen(mt_rand);

				if (r<rDeplt) { // point falls within the depletion region
					double deltaxNb = xNb[0]-xNbdep - xNbdep*(r-rDelta)/(rDeplt-rDelta);
					initGrid(n)[1] -= deltaxNb;
				}
			} else {
				// Delta particle
				delta_mass+=1.0;
				initGrid(n)[0] = xAl[1]*(1.0 + noise_amp*real_gen(mt_rand));
				initGrid(n)[1] = xNb[1]*(1.0 + noise_amp*real_gen(mt_rand));
				initGrid(n)[6] = 1.0;
				for (int i=2; i<6; i++)
					initGrid(n)[i] = 0.0;
			}
		}

		vector<double> totals(7,0.0);
		for (int n=0; n<nodes(initGrid); n++) {
			for (int i=0; i<2; i++)
				totals[i] += initGrid(n)[i];
			for (int i=2; i<7; i++)
				totals[i] += std::fabs(initGrid(n)[i]);
		}

		for (int i=0; i<7; i++)
			totals[i] /= 1.0*edge*edge;
		#ifdef MPI_VERSION
		vector<double> myTot(totals);
		for (int i=0; i<7; i++) {
			MPI::COMM_WORLD.Reduce(&myTot[i], &totals[i], 1, MPI_DOUBLE, MPI_SUM, 0);
			MPI::COMM_WORLD.Barrier();
		}
		#endif
		if (rank==0) {
			std::cout<<"x_Ni      x_Al      x_Nb\n";
			printf("%.6f  %1.2e  %1.2e\n\n", 1.0-totals[0]-totals[1], totals[0], totals[1]);
			std::cout<<"p_g       p_g'      p_g''     p_g''     p_g''     p_d\n";
			printf("%.6f  %1.2e  %1.2e  %1.2e  %1.2e  %1.2e\n", 1.0-totals[2]-totals[3]-totals[4]-totals[5]-totals[6], totals[2], totals[3], totals[4], totals[5], totals[6]);
		}

		output(initGrid,filename);
	}

	if (dim==3) {
		const int edge = 64;
		GRID3D initGrid(7,0,edge,0,edge,0,edge);
		vector<int> origin(3,edge/2);
		for (int d=0; d<dim; d++)
			dx(initGrid,d)=0.5;

		const double rDelta = 6.5*dx(initGrid,0);
		const double rDeplt = rDelta*std::pow(1.0+2.0*xNb[1]/xNbdep,1.0/3); // radius of annular depletion region
		if (rDeplt > edge/2)
			std::cerr<<"Warning: domain too small to accommodate particle. Push beyond "<<rDeplt<<".\n"<<std::endl;

		for (int n=0; n<nodes(initGrid); n++) {
			vector<int> x = position(initGrid, n);
			const double r = radius(origin, x, dx(initGrid,0));
			if (r > rDelta) {
				// Gamma matrix
				initGrid(n)[0] = xAl[0]*(1.0 + noise_amp*real_gen(mt_rand))
				               + 0.25*xAl[0]*bellCurve(dx(initGrid,0)*x[0], 0,                     sig[0]*dx(initGrid,0)*edge)
				               + 0.25*xAl[0]*bellCurve(dx(initGrid,0)*x[0], dx(initGrid,0)*edge/2, sig[0]*dx(initGrid,0)*edge)
				               + 0.25*xAl[0]*bellCurve(dx(initGrid,0)*x[0], dx(initGrid,0)*edge,   sig[0]*dx(initGrid,0)*edge);
				initGrid(n)[1] = xNb[0]*(1.0 + noise_amp*real_gen(mt_rand))
				               + xNb[0]*bellCurve(dx(initGrid,0)*x[0], 0,                     sig[1]*dx(initGrid,0)*edge)
				               + xNb[0]*bellCurve(dx(initGrid,0)*x[0], dx(initGrid,0)*edge/2, sig[1]*dx(initGrid,0)*edge)
				               + xNb[0]*bellCurve(dx(initGrid,0)*x[0], dx(initGrid,0)*edge,   sig[1]*dx(initGrid,0)*edge);
				for (int i=2; i<7; i++)
					initGrid(n)[i] = 0.01*real_gen(mt_rand);

				if (r<rDeplt) { // point falls within the depletion region
					double deltaxNb = xNb[0]-xNbdep - xNbdep*(r-rDelta)/(rDeplt-rDelta);
					initGrid(n)[1] -= deltaxNb;
				}
			} else {
				// Delta particle
				initGrid(n)[0] = xAl[1]*(1.0 + noise_amp*real_gen(mt_rand));
				initGrid(n)[1] = xNb[1]*(1.0 + noise_amp*real_gen(mt_rand));
				initGrid(n)[6] = 1.0;
				for (int i=2; i<6; i++)
					initGrid(n)[i] = 0.0;
			}
		}

		vector<double> totals(7,0.0);
		for (int n=0; n<nodes(initGrid); n++) {
			for (int i=0; i<2; i++)
				totals[i] += initGrid(n)[i];
			for (int i=2; i<7; i++)
				totals[i] += std::fabs(initGrid(n)[i]);
		}

		for (int i=0; i<7; i++)
			totals[i] /= 1.0*edge*edge*edge;
		#ifdef MPI_VERSION
		vector<double> myTot(totals);
		for (int i=0; i<7; i++) {
			MPI::COMM_WORLD.Reduce(&myTot[i], &totals[i], 1, MPI_DOUBLE, MPI_SUM, 0);
			MPI::COMM_WORLD.Barrier();
		}
		#endif
		if (rank==0) {
			std::cout<<"x_Ni      x_Al      x_Nb\n";
			printf("%.6f  %1.2e  %1.2e\n", 1.0-totals[0]-totals[1], totals[0], totals[1]);
			std::cout<<"p_g       p_g'      p_g''     p_g''     p_g''     p_d\n";
			printf("%.6f  %1.2e  %1.2e  %1.2e  %1.2e  %1.2e\n", 1.0-totals[2]-totals[3]-totals[4]-totals[5]-totals[6], totals[2], totals[3], totals[4], totals[5], totals[6]);
		}

		output(initGrid,filename);
	}
}

template <int dim, typename T>
void update(grid<dim,vector<T> >& oldGrid, int steps)
{
	grid<dim,vector<T> > newGrid(oldGrid);
	grid<dim,T> wspace(oldGrid,1);

	double dt = 0.01;
	double L = 1.0;
	double D = 1.0;


	for (int step=0; step<steps; step++) {
		for (int n=0; n<nodes(oldGrid); n++) {
			vector<int> x = position(oldGrid, n);
			double sum = 0.0;
			for (int i=1; i<fields(oldGrid); i++)
				sum += pow(oldGrid(n)[i],2);

			double C = oldGrid(n)[0];
			double lap = laplacian(oldGrid, x, 0); // take Laplacian of field 0 only

			wspace(x) = -A*(C-Cmatrix)+B*pow(C-Cmatrix,3)
			            +Dalpha*pow(C-Calpha,3)+Dbeta*pow(C-Cbeta,3)
			            -gamma*(C-Calpha)*sum-kappa*lap;
		}
		ghostswap(wspace);

		for (int n=0; n<nodes(oldGrid); n++) {
			vector<int> x = position(oldGrid, n);
			double C = oldGrid(n)[0];
			double lap = laplacian(wspace, x);
			newGrid(x)[0] = C+dt*D*lap;

			double sum = 0.0;
			for (int i=1; i<fields(oldGrid); i++)
				sum += pow(oldGrid(n)[i],2);

			vector<double> vlap = laplacian(oldGrid, x);
			for (int i=1; i<fields(oldGrid); i++) {
				double value = oldGrid(n)[i];

				newGrid(x)[i] = value-dt*L*(-gamma*pow(C-Calpha,2)+delta*pow(value,3)
				                           +epsilon*value*(sum-pow(value,2))-kappa*vlap[i]);
			}
		}
		swap(oldGrid,newGrid);
		ghostswap(oldGrid);
	}
}

} // namespace MMSP

#endif

#include"MMSP.main.hpp"
