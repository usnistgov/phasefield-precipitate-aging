#include <gsl/gsl_blas.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_multiroots.h>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>

#include "parabola625.h"

double euclidean(double xACr, double xANb, double xBCr, double xBNb)
{
	return std::sqrt((xACr - xBCr)*(xACr - xBCr) + (xANb - xBNb)*(xANb - xBNb));
}

struct rparams {
	double xCr0;
	double xNb0;
};

int common_tan_del_f(const gsl_vector* x, void* params, gsl_vector* f)
{
	const double xNb = ((struct rparams *) params)->xNb0;
	const double xCr = ((struct rparams *) params)->xCr0;

	const double xNbA = gsl_vector_get(x, 0);
	const double xCrA = gsl_vector_get(x, 1);
	const double xNbB = gsl_vector_get(x, 2);
	const double xCrB = gsl_vector_get(x, 3);

	const double fA = g_gam(xCrA, xNbA);
	const double fB = g_del(xCrB, xNbB);
	const double dfAdxNb = dg_gam_dxNb(xCrA, xNbA);
	const double dfAdxCr = dg_gam_dxCr(xCrA, xNbA);
	const double dfBdxNb = dg_del_dxNb(xCrB, xNbB);
	const double dfBdxCr = dg_del_dxCr(xCrB, xNbB);
	const double dxNb = xNbA - xNbB;
	const double dxCr = xCrA - xCrB;

	gsl_vector_set(f, 0, dfAdxNb - dfBdxNb);
	gsl_vector_set(f, 1, dfAdxCr - dfBdxCr);
	gsl_vector_set(f, 2, fA - dfAdxNb * dxNb - dfAdxCr * dxCr - fB);
	gsl_vector_set(f, 3, (xNb - xNbB) * dxCr - dxNb * (xCr - xCrB));

	return GSL_SUCCESS;
}

int common_tan_del_df(const gsl_vector* x, void* params, gsl_matrix* J)
{
	const double xNb = ((struct rparams *) params)->xNb0;
	const double xCr = ((struct rparams *) params)->xCr0;

	const double xNbA = gsl_vector_get(x, 0);
	const double xCrA = gsl_vector_get(x, 1);
	const double xNbB = gsl_vector_get(x, 2);
	const double xCrB = gsl_vector_get(x, 3);

	const double dfAdxNb = dg_gam_dxNb(xCrA, xNbA);
	const double dfAdxCr = dg_gam_dxCr(xCrA, xNbA);
	const double dfBdxNb = dg_del_dxNb(xCrB, xNbB);
	const double dfBdxCr = dg_del_dxCr(xCrB, xNbB);
	const double d2fAdxNbNb = d2g_gam_dxNbNb();
	const double d2fAdxNbCr = d2g_gam_dxNbCr();
	const double d2fAdxCrCr = d2g_gam_dxCrCr();
	const double d2fBdxNbNb = d2g_del_dxNbNb();
	const double d2fBdxNbCr = d2g_del_dxNbCr();
	const double d2fBdxCrCr = d2g_del_dxCrCr();
	const double dxNb = xNbA - xNbB;
	const double dxCr = xCrA - xCrB;

	gsl_matrix_set(J, 0, 0, d2fAdxNbNb);
	gsl_matrix_set(J, 0, 1, d2fAdxNbCr);
	gsl_matrix_set(J, 0, 2,-d2fBdxNbNb);
	gsl_matrix_set(J, 0, 3,-d2fBdxNbCr);

	gsl_matrix_set(J, 1, 0, d2fAdxNbCr);
	gsl_matrix_set(J, 1, 1, d2fAdxCrCr);
	gsl_matrix_set(J, 1, 2,-d2fBdxNbCr);
	gsl_matrix_set(J, 1, 3,-d2fBdxCrCr);

	gsl_matrix_set(J, 2, 0, -d2fAdxNbNb * dxNb - d2fAdxNbCr * dxCr);
	gsl_matrix_set(J, 2, 1, -d2fAdxNbCr * dxNb - d2fAdxCrCr * dxCr);
	gsl_matrix_set(J, 2, 2, dfAdxNb - dfBdxNb);
	gsl_matrix_set(J, 2, 3, dfAdxCr - dfBdxCr);

	gsl_matrix_set(J, 3, 0, xCrB - xCr);
	gsl_matrix_set(J, 3, 1, xNb - xNbB);
	gsl_matrix_set(J, 3, 2, xCr - xCrA);
	gsl_matrix_set(J, 3, 3, xNbA - xNb);

	return GSL_SUCCESS;
}

int common_tan_del_fdf(const gsl_vector* x, void* params, gsl_vector* f, gsl_matrix* J)
{
	common_tan_del_f(x, params, f);
	common_tan_del_df(x, params, J);

	return GSL_SUCCESS;
}

double estimate_fraction_del(const double xCr, const double xNb)
{
	int status;
	size_t iter = 0, max_iter = 1000;
	const size_t n = 4;
	const double tolerance = 1e-14;

	gsl_vector* x = gsl_vector_alloc(n);
	gsl_vector_set(x, 0, 0.5 * (xNb + xe_gam_Nb()));
	gsl_vector_set(x, 1, 0.5 * (xCr + xe_gam_Cr()));
	gsl_vector_set(x, 2, 0.5 * (xNb + xe_del_Nb()));
	gsl_vector_set(x, 3, 0.5 * (xCr + xe_del_Cr()));

	struct rparams p = {xCr, xNb};

	const gsl_multiroot_fdfsolver_type* algorithm = gsl_multiroot_fdfsolver_hybridsj;
	gsl_multiroot_fdfsolver* solver = gsl_multiroot_fdfsolver_alloc(algorithm, n);
	gsl_multiroot_function_fdf mrf = {&common_tan_del_f, &common_tan_del_df, &common_tan_del_fdf, n, &p};

	gsl_multiroot_fdfsolver_set(solver, &mrf, x);

	do {
		iter++;
		status = gsl_multiroot_fdfsolver_iterate(solver);
		if (status)   /* check if solver is stuck */
			break;
		status = gsl_multiroot_test_residual(solver->f, tolerance);
	} while (status == GSL_CONTINUE && iter < max_iter);

	const double xGNb = gsl_vector_get(solver->x, 0);
	const double xGCr = gsl_vector_get(solver->x, 1);
	const double xDNb = gsl_vector_get(solver->x, 2);
	const double xDCr = gsl_vector_get(solver->x, 3);

	gsl_multiroot_fdfsolver_free(solver);
	gsl_vector_free(x);

	const double full = euclidean(xGCr, xGNb, xDCr, xDNb);
	return euclidean(xGCr, xGNb, xCr, xNb) / full;
}


int common_tan_lav_f(const gsl_vector* x, void* params, gsl_vector* f)
{
	const double xNb = ((struct rparams *) params)->xNb0;
	const double xCr = ((struct rparams *) params)->xCr0;

	const double xNbA = gsl_vector_get(x, 0);
	const double xCrA = gsl_vector_get(x, 1);
	const double xNbB = gsl_vector_get(x, 2);
	const double xCrB = gsl_vector_get(x, 3);

	const double fA = g_gam(xCrA, xNbA);
	const double fB = g_lav(xCrB, xNbB);
	const double dfAdxNb = dg_gam_dxNb(xCrA, xNbA);
	const double dfAdxCr = dg_gam_dxCr(xCrA, xNbA);
	const double dfBdxNb = dg_lav_dxNb(xCrB, xNbB);
	const double dfBdxCr = dg_lav_dxCr(xCrB, xNbB);
	const double dxNb = xNbA - xNbB;
	const double dxCr = xCrA - xCrB;

	gsl_vector_set(f, 0, dfAdxNb - dfBdxNb);
	gsl_vector_set(f, 1, dfAdxCr - dfBdxCr);
	gsl_vector_set(f, 2, fA - dfAdxNb * dxNb - dfAdxCr * dxCr - fB);
	gsl_vector_set(f, 3, (xNb - xNbB) * dxCr - dxNb * (xCr - xCrB));

	return GSL_SUCCESS;
}

int common_tan_lav_df(const gsl_vector* x, void* params, gsl_matrix* J)
{
	const double xNb = ((struct rparams *) params)->xNb0;
	const double xCr = ((struct rparams *) params)->xCr0;

	const double xNbA = gsl_vector_get(x, 0);
	const double xCrA = gsl_vector_get(x, 1);
	const double xNbB = gsl_vector_get(x, 2);
	const double xCrB = gsl_vector_get(x, 3);

	const double dfAdxNb = dg_gam_dxNb(xCrA, xNbA);
	const double dfAdxCr = dg_gam_dxCr(xCrA, xNbA);
	const double dfBdxNb = dg_lav_dxNb(xCrB, xNbB);
	const double dfBdxCr = dg_lav_dxCr(xCrB, xNbB);
	const double d2fAdxNbNb = d2g_gam_dxNbNb();
	const double d2fAdxNbCr = d2g_gam_dxNbCr();
	const double d2fAdxCrCr = d2g_gam_dxCrCr();
	const double d2fBdxNbNb = d2g_lav_dxNbNb();
	const double d2fBdxNbCr = d2g_lav_dxNbCr();
	const double d2fBdxCrCr = d2g_lav_dxCrCr();
	const double dxNb = xNbA - xNbB;
	const double dxCr = xCrA - xCrB;

	gsl_matrix_set(J, 0, 0, d2fAdxNbNb);
	gsl_matrix_set(J, 0, 1, d2fAdxNbCr);
	gsl_matrix_set(J, 0, 2,-d2fBdxNbNb);
	gsl_matrix_set(J, 0, 3,-d2fBdxNbCr);

	gsl_matrix_set(J, 1, 0, d2fAdxNbCr);
	gsl_matrix_set(J, 1, 1, d2fAdxCrCr);
	gsl_matrix_set(J, 1, 2,-d2fBdxNbCr);
	gsl_matrix_set(J, 1, 3,-d2fBdxCrCr);

	gsl_matrix_set(J, 2, 0, -d2fAdxNbNb * dxNb - d2fAdxNbCr * dxCr);
	gsl_matrix_set(J, 2, 1, -d2fAdxNbCr * dxNb - d2fAdxCrCr * dxCr);
	gsl_matrix_set(J, 2, 2, dfAdxNb - dfBdxNb);
	gsl_matrix_set(J, 2, 3, dfAdxCr - dfBdxCr);

	gsl_matrix_set(J, 3, 0, xCrB - xCr);
	gsl_matrix_set(J, 3, 1, xNb - xNbB);
	gsl_matrix_set(J, 3, 2, xCr - xCrA);
	gsl_matrix_set(J, 3, 3, xNbA - xNb);

	return GSL_SUCCESS;
}

int common_tan_lav_fdf(const gsl_vector* x, void* params, gsl_vector* f, gsl_matrix* J)
{
	common_tan_lav_f(x, params, f);
	common_tan_lav_df(x, params, J);

	return GSL_SUCCESS;
}

double estimate_fraction_lav(const double xCr, const double xNb)
{
	int status;
	size_t iter = 0, max_iter = 1000;
	const size_t n = 4;
	const double tolerance = 1e-14;

	gsl_vector* x = gsl_vector_alloc(n);
	gsl_vector_set(x, 0, 0.5 * (xNb + xe_gam_Nb()));
	gsl_vector_set(x, 1, 0.5 * (xCr + xe_gam_Cr()));
	gsl_vector_set(x, 2, 0.5 * (xNb + xe_lav_Nb()));
	gsl_vector_set(x, 3, 0.5 * (xCr + xe_lav_Cr()));

	struct rparams p = {xCr, xNb};

	const gsl_multiroot_fdfsolver_type* algorithm = gsl_multiroot_fdfsolver_hybridsj;
	gsl_multiroot_fdfsolver* solver = gsl_multiroot_fdfsolver_alloc(algorithm, n);
	gsl_multiroot_function_fdf mrf = {&common_tan_lav_f, &common_tan_lav_df, &common_tan_lav_fdf, n, &p};

	gsl_multiroot_fdfsolver_set(solver, &mrf, x);

	do {
		iter++;
		status = gsl_multiroot_fdfsolver_iterate(solver);
		if (status)   /* check if solver is stuck */
			break;
		status = gsl_multiroot_test_residual(solver->f, tolerance);
	} while (status == GSL_CONTINUE && iter < max_iter);

	const double xGNb = gsl_vector_get(solver->x, 0);
	const double xGCr = gsl_vector_get(solver->x, 1);
	const double xLNb = gsl_vector_get(solver->x, 2);
	const double xLCr = gsl_vector_get(solver->x, 3);

	gsl_multiroot_fdfsolver_free(solver);
	gsl_vector_free(x);

	const double full = euclidean(xGCr, xGNb, xLCr, xLNb);
	return euclidean(xGCr, xGNb, xCr, xNb) / full;
}
