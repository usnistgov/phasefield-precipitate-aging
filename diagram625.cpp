// diagram625.cpp
// Code to export raw data to draw a ternary isothermal phase diagram
// for Mo-Nb-Ni system with FCC, BCC, delta, and mu phases
// Questions/comments to trevor.keller@nist.gov (Trevor Keller)

// This implementation depends on the GNU Scientific Library
// for multivariate root finding algorithms.

#ifndef DIAGRAM625_
#define DIAGRAM625_
#include<cmath>
#include<gsl/gsl_blas.h>
#include<gsl/gsl_vector.h>
#include<gsl/gsl_multiroots.h>
#include<gsl/gsl_interp2d.h>
#include<gsl/gsl_spline2d.h>

#include"energy625.c"

// Note: energy625.c is generated from CALPHAD using pycalphad and SymPy, in CALPHAD_extraction.ipynb.

double g_gam(const double& x_Mo, const double& x_Nb, const double& T)
{
	// Two sub-lattices, but #2 contains vacancies only
	const double x_Ni = 1.0 - x_Mo - x_Nb;
	const double y_Va = 1.0;

	const double sites = 1.0;
	return g_gam(x_Mo, x_Nb, x_Ni, y_Va, T) / sites;
}

double g_bcc(const double& x_Mo, const double& x_Nb, const double& T)
{
	// Two sub-lattices, but #2 contains vacancies only
	const double x_Ni = 1.0 - x_Mo - x_Nb;
	const double y_Va = 1.0;

	const double sites = 1.0;
	return g_bcc(x_Mo, x_Nb, x_Ni, y_Va, T) / sites;
}

double g_mu(const double& x_Mo, const double& x_Nb, const double& T)
{
	// Three sub-lattices, but can over-simplify using y2_Nb=y3_Nb
	const double A = 13.0/7;
	const double B = 13.0/3;

	const double y1_Ni = 1.0;
	const double y2_Mo = 1.0 - A*x_Nb;
	const double y2_Nb = A*x_Nb;
	const double y3_Mo = 3.0 - A*x_Nb - B*x_Ni;
	const double y3_Nb = y2_Nb;
	const double y3_Ni = B*x_Ni - 2.0;

	const double sites = 13.0;
	return g_mu(y1_Ni, y2_Nb, y2_Ni, y3_Mo, y3_Nb, y3_Ni, T) / sites;
}

double g_del(const double x_Mo, const double x_Nb, const double& T)
{
	// Two sub-lattices, but can approximate as a stoichiometric phase by setting y1_Nb=0
	const double y1_Ni = 1.0;
	const double y1_Nb = 0.0;
	const double y2_Nb = 4.0*x_Nb;
	const double y2_Mo = 4.0*x_Mo;
	const double y2_Ni = 4.0*x_Ni - 3.0;

	const double sites = 4.0;
	return g_del(y1_Nb, y1_Ni, y2_Mo, y2_Nb, y2_Ni, T) / sites;
}

double gibbs(const vector<double>& v)
{
	double g  = g_gam(v[0],v[1]) * (1.0 - (h(abs(v[2])) + h(abs(v[3])) + h(abs(v[4]))));
	       g += g_mu(v[0],v[1]) * h(abs(v[2]));
	       g += g_del(v[0],v[1]) * h(abs(v[3]));
	       g += w_mu * v[2]*v[2] * (1.0 - abs(v[2])*(1.0 - abs(v[2]);
	       g += w_del * v[3]*v[3] * (1.0 - abs(v[3])*(1.0 - abs(v[3]);
	for (int i=2; i<v.length(); i++)
		for (int j=i+1; j<v.length(); j++)
			g += 2.0 * alpha * v[i]*v[i] * v[j]*v[j];

	return g;
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
	double x_Mo;
	double x_Nb;

	// Structure fields
	double p_mu_Mo;
	double p_del_Mo;

	double p_mu_Nb;
	double p_del_Nb;
};


int commonTangent_f(const gsl_vector* x, void* params, gsl_vector* f)
{
	const double x_Mo = ((struct rparams*) params)->x_Mo;
	const double x_Nb = ((struct rparams*) params)->x_Nb;
	const double p_mu = ((struct rparams*) params)->p_mu;
	const double p_del = ((struct rparams*) params)->p_del;

	const double C_gam_Mo = gsl_vector_get(x, 0);
	const double C_mu_Mo = gsl_vector_get(x, 2);
	const double C_del_Mo = gsl_vector_get(x, 3);

	const double C_gam_Nb = gsl_vector_get(x, 4);
	const double C_mu_Nb = gsl_vector_get(x, 6);
	const double C_del_Nb = gsl_vector_get(x, 7);

	const double f1 = (1.0 - h(fabs(p_mu)) - h(fabs(p_del)))*C_gam_Mo
	                + h(fabs(p_mu))*C_mu_Mo
	                + h(fabs(p_del))*C_del_Mo
	                - x_Mo;
	const double f2 = dg_gam_dxMo(C_gam_Mo) - dg_mu_dxMo(C_mu_Mo);
	const double f3 = dg_mu_dxMo(C_mu_Mo) - dg_del_dxMo(C_del_Mo);

	const double f4 = (1.0 - h(fabs(p_mu)) - h(fabs(p_del)))*C_gam_Nb
	                + h(fabs(p_mu))*C_mu_Nb
	                + h(fabs(p_del))*C_del_Nb
	                - x_Nb;
	const double f5 = dg_gam_dxNb(C_gam_Nb) - dg_mu_dxNb(C_mu_Nb);
	const double f6 = dg_mu_dxNb(C_mu_Nb) - dg_del_dxNb(C_del_Nb);

	gsl_vector_set(f, 0, f1);
	gsl_vector_set(f, 1, f2);
	gsl_vector_set(f, 2, f3);

	gsl_vector_set(f, 3, f4);
	gsl_vector_set(f, 4, f5);
	gsl_vector_set(f, 5, f6);

	return GSL_SUCCESS;
}


int commonTangent_df(const gsl_vector* x, void* params, gsl_matrix* J)
{
	const double x_Mo = ((struct rparams*) params)->x_Mo;
	const double x_Nb = ((struct rparams*) params)->x_Nb;
	const double p_mu = ((struct rparams*) params)->p_mu;
	const double p_del = ((struct rparams*) params)->p_del;

	const double C_gam_Mo = gsl_vector_get(x, 0);
	const double C_mu_Mo = gsl_vector_get(x, 2);
	const double C_del_Mo = gsl_vector_get(x, 3);

	const double C_gam_Nb = gsl_vector_get(x, 4);
	const double C_mu_Nb = gsl_vector_get(x, 6);
	const double C_del_Nb = gsl_vector_get(x, 7);

	// Jacobian matrix
	const double sum = h(abs(p_mu)) + h(abs(p_del)) ;

	// Need to know the functional form of the chemical potentials to proceed

	gsl_matrix_set(J, 0, 0, 1.0-sum);
	gsl_matrix_set(J, 0, 1, h(abs(p_mu)));
	gsl_matrix_set(J, 0, 2, h(abs(p_del)));

	gsl_matrix_set(J, 1, 0,  d2g_gam_dxMo2(C_gam_Mo, x_Nb)));
	gsl_matrix_set(J, 1, 1, -d2g_mu_dxMo2(C_mu_Mo, x_Nb)));
	gsl_matrix_set(J, 1, 2, 0.0);

	gsl_matrix_set(J, 2, 0, 0.0);
	gsl_matrix_set(J, 2, 1,  d2g_mu_dxMo2(C_mu_Mo, x_Nb)));
	gsl_matrix_set(J, 2, 2, -d2g_del_dxMo2(C_del_Mo, x_Nb)));


	for (int i=0; i<3; i++)
		for (for j=3; j<6; j++)
			gsl_matrix_set(J, i, j, 0.0);

	gsl_matrix_set(J, 3, 3, 1.0-sum);
	gsl_matrix_set(J, 3, 4, h(abs(p_mu)));
	gsl_matrix_set(J, 3, 5, h(abs(p_del)));

	gsl_matrix_set(J, 4, 3,  d2g_gam_dxMo2(x_Mo, C_gam_Nb)));
	gsl_matrix_set(J, 4, 4, -d2g_mu_dxMo2(x_Mo, C_mu_Nb)));
	gsl_matrix_set(J, 4, 5, 0.0);

	gsl_matrix_set(J, 5, 3, 0.0);
	gsl_matrix_set(J, 5, 4,  d2g_mu_dxMo2(x_Mo, C_mu_Nb)));
	gsl_matrix_set(J, 5, 5, -d2g_del_dxMo2(x_Mo, C_del_Nb)));

	for (int i=3; i<6; i++)
		for (int j=0; j<3; j++)
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
rootsolver::solve(const T& x_Mo, const T& x_Nb, const T& p_mu, const T& p_del,
                  T& C_gam_Mo, T& C_mu_Mo, t& C_del_Mo, T& C_gam_Nb, T& C_mu_Nb, t& C_del_Nb)
{
	int status;
	size_t iter = 0;

	// initial guesses
	par.x_Mo = x_Mo;
	par.x_Nb = x_Nb;
	par.p_mu = p_mu;
	par.p_del = p_del;

	gsl_vector_set(x, 0, C_gam_Mo);
	gsl_vector_set(x, 1, C_mu_Mo);
	gsl_vector_set(x, 2, C_del_Mo);

	gsl_vector_set(x, 3, C_gam_Nb);
	gsl_vector_set(x, 4, C_mu_Nb);
	gsl_vector_set(x, 5, C_del_Nb);

	gsl_multiroot_fdfsolver_set(solver, &mrf, x);

	do {
		iter++;
		status = gsl_multiroot_fdfsolver_iterate(solver);
		if (status) // extra points for finishing early!
			break;
		status = gsl_multiroot_test_residual(solver->f, tolerance);
	} while (status==GSL_CONTINUE && iter<maxiter);

	C_gam_Mo = static_cast<T>(gsl_vector_get(solver->x, 0));
	C_mu_Mo  = static_cast<T>(gsl_vector_get(solver->x, 1));
	C_del_Mo = static_cast<T>(gsl_vector_get(solver->x, 2));

	C_gam_Nb = static_cast<T>(gsl_vector_get(solver->x, 3));
	C_mu_Nb  = static_cast<T>(gsl_vector_get(solver->x, 4));
	C_del_Nb = static_cast<T>(gsl_vector_get(solver->x, 5));

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
