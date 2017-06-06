/*************************************************************************************
 * File: equilibrium.cpp                                                             *
 * Implementation of equilibrium solver for Cr-Nb-Ni alloy phase diagrams            *
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
 * versions bear some notice that they have been modified.
 *************************************************************************************/

#ifndef ALLOY625_DIAGRAM
#define ALLOY625_DIAGRAM
#include<cmath>
#include<random>
#include<iostream>
#include<ciso646>
#include<ctime>
#include<vector>
#include<cstring>
#ifdef _OPENMP
#include"omp.h"
#endif
#include<gsl/gsl_blas.h>
#include<gsl/gsl_math.h>
#include<gsl/gsl_roots.h>
#include<gsl/gsl_vector.h>
#include<gsl/gsl_multiroots.h>

using std::vector;

#include"equilibrium.hpp"

/* ============================ rootsolver Class =========================== */

/* Given const phase fractions (in gamma, delta, and Laves) and concentrations of
 * Cr and Nb, iteratively determine the fictitious concentrations in each phase
 * that satisfy the constraints of equal chemical potential at equilibrium and
 * conservation of mass at all times. Pass constants (p and x) by const value,
 * C by non-const reference to update in place.
 */

#ifdef PARABOLA
const size_t max_root_iter = 1e4;
const double root_tol = 1e-10;
#elif defined CALPHAD
const int max_root_iter = 1e4;
const double root_tol = 1e-6;
#else
const int max_root_iter = 1e4;
const double root_tol = 1e-8;
#endif

const int rand_iters = 1e5;
const double maxNi = 1.0;

class rootsolver
{
public:
	rootsolver(gsl_multiroot_function_fdf equations);
	~rootsolver();
	double solve(const vector<double>& origin, vector<double>& data);

private:
	const size_t n;
	const size_t maxiter;
	const double tolerance;
	gsl_vector* x;
	struct rparams par;
	const gsl_multiroot_fdfsolver_type* algorithm;
	gsl_multiroot_fdfsolver* solver;
	gsl_multiroot_function_fdf mrf;
};

rootsolver::rootsolver(gsl_multiroot_function_fdf equations) :
	n(4),
	maxiter(max_root_iter),
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
	mrf.f = equations.f;
	mrf.df = equations.df;
	mrf.fdf = equations.fdf;
	mrf.n = this->n;
	mrf.params = &(this->par);
}


double rootsolver::solve(const vector<double>& origin, vector<double>& data)
{
	int status;
	size_t iter = 0;

	par.x_Cr = origin[0];
	par.x_Nb = origin[1];

	// copy initial guesses from grid
	for (int i = 0; i < int(n); i++)
		gsl_vector_set(x, i, data[i]);

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
			data[i] = gsl_vector_get(solver->x, i);

	return residual;
}


rootsolver::~rootsolver()
{
	gsl_multiroot_fdfsolver_free(solver);
	gsl_vector_free(x);
}

/* ========================================================================= */

// Initial guesses for gamma, delta, and Laves equilibrium compositions
vector<double> guessGammaDelta(const vector<double>& origin)
{
    // Coarsely approximate gamma using a line compound with x_Nb = 0.025
    // Interpolate x_Cr from (-0.01, 1.01) into (0.2, 0.4)
    // Coarsely approximate delta using a line compound with x_Nb = 0.225
    // Interpolate x_Cr from (-0.01, 1.01) into (0.0, 0.05)

	vector<double> data(4, 0.0);

    const double& xcr = origin[0];

	// gamma
    data[0] = 0.20 + 0.20/1.02 * (xcr + 0.01);
    data[1] = 0.10;

    // delta
	data[2] = 0.05/1.02 * (xcr + 0.01);
    data[3] = 0.225;

	return data;
}


vector<double> guessGammaLaves(const vector<double>& origin)
{
    // Coarsely approximate gamma using a line compound with x_Nb = 0.025
    // Interpolate x_Cr from (-0.01, 1.01) into (0.2, 0.4)
    // Coarsely approximate Laves using a line compound with x_Nb = 30.0%
    // Interpolate x_Cr from (-0.01, 1.01) into (0.30, 0.45)

	vector<double> data(4, 0.0);

    const double& xcr = origin[0];

    // gamma
	data[0] = 0.20 + 0.20/1.02 * (xcr + 0.01);
    data[1] = 0.10;

    // laves
    data[2] = 0.30 + 0.15/1.02 * (xcr + 0.01);
    data[3] = 0.25;

	return data;
}


vector<double> guessDeltaLaves(const vector<double>& origin)
{
    // Coarsely approximate delta using a line compound with x_Nb = 0.225
    // Interpolate x_Cr from (-0.01, 1.01) into (0.0, 0.05)
    // Coarsely approximate Laves using a line compound with x_Nb = 30.0%
    // Interpolate x_Cr from (-0.01, 1.01) into (0.30, 0.45)

	vector<double> data(4, 0.0);

    const double& xcr = origin[0];

    // delta
	data[0] = 0.05/1.02 * (xcr + 0.01);
    data[1] = 0.225;

    // laves
    data[2] = 0.30 + 0.15/1.02 * (xcr + 0.01);
    data[3] = 0.25;

	return data;
}

/* ========================================================================= */

int main(int argc, char* argv[])
{
	// Initialize pseudo-random number generator
	std::random_device rd; // PRNG seed generator
	std::mt19937 mtrand(rd()); // Mersenne Twister
	std::uniform_real_distribution<double> unidist(0, 1); // uniform distribution on [0, 1)

	const unsigned int maxiter = rand_iters;
	unsigned int iter = 0;

	gsl_multiroot_function_fdf GammaDeltaEquations = {&GammaDelta_f, &GammaDelta_df, &GammaDelta_fdf};
	gsl_multiroot_function_fdf GammaLavesEquations = {&GammaLaves_f, &GammaLaves_df, &GammaLaves_fdf};
	gsl_multiroot_function_fdf DeltaLavesEquations = {&DeltaLaves_f, &DeltaLaves_df, &DeltaLaves_fdf};

	rootsolver gamDelSolver(GammaDeltaEquations);
	rootsolver gamLavSolver(GammaLavesEquations);
	rootsolver delLavSolver(DeltaLavesEquations);

	vector<vector<double> > GammaDeltaRoots;
	vector<vector<double> > GammaLavesRoots;
	vector<vector<double> > DeltaLavesRoots;
	vector<double> origin(2, 1.0);

	do {
		print_progress(iter, maxiter);

		// Select a random point on the simplex
		double xni = 2.0;
		while (xni > maxNi) {
			origin[0] = unidist(mtrand);
			origin[1] = unidist(mtrand);
			xni = 1.0 - origin[0] - origin[1];
		}

		// Create initial guesses from random point
		vector<double> GammaDeltaResult = guessGammaDelta(origin);
		vector<double> GammaLavesResult = guessGammaLaves(origin);
		vector<double> DeltaLavesResult = guessDeltaLaves(origin);

		if (gamDelSolver.solve(origin, GammaDeltaResult) < root_tol) {
			if (GammaDeltaResult[0] > 0.0 && GammaDeltaResult[0] < 1.0 &&
			    GammaDeltaResult[1] > 0.0 && GammaDeltaResult[1] < 1.0 &&
			    GammaDeltaResult[2] > 0.0 && GammaDeltaResult[2] < 1.0 &&
			    GammaDeltaResult[3] > 0.0 && GammaDeltaResult[3] < 1.0 &&
			    GammaDeltaResult[0] + GammaDeltaResult[1] < 1.0 &&
			    GammaDeltaResult[2] + GammaDeltaResult[3] < 1.0 )
					GammaDeltaRoots.push_back(GammaDeltaResult);
		}

		if (gamLavSolver.solve(origin, GammaLavesResult) < root_tol) {
			if (GammaLavesResult[0] > 0.0 && GammaLavesResult[0] < 1.0 &&
			    GammaLavesResult[1] > 0.0 && GammaLavesResult[1] < 1.0 &&
			    GammaLavesResult[2] > 0.0 && GammaLavesResult[2] < 1.0 &&
			    GammaLavesResult[3] > 0.0 && GammaLavesResult[3] < 1.0 &&
			    GammaLavesResult[0] + GammaLavesResult[1] < 1.0 &&
			    GammaLavesResult[2] + GammaLavesResult[3] < 1.0 )
					GammaLavesRoots.push_back(GammaLavesResult);
		}

		if (delLavSolver.solve(origin, DeltaLavesResult) < root_tol) {
			if (DeltaLavesResult[0] > 0.0 && DeltaLavesResult[0] < 1.0 &&
			    DeltaLavesResult[1] > 0.0 && DeltaLavesResult[1] < 1.0 &&
			    DeltaLavesResult[2] > 0.0 && DeltaLavesResult[2] < 1.0 &&
			    DeltaLavesResult[3] > 0.0 && DeltaLavesResult[3] < 1.0 &&
			    DeltaLavesResult[0] + DeltaLavesResult[1] < 1.0 &&
			    DeltaLavesResult[2] + DeltaLavesResult[3] < 1.0 )
			DeltaLavesRoots.push_back(DeltaLavesResult);
		}

		iter++;
	} while (iter < maxiter);

	if (GammaDeltaRoots.size() > 0) {
		FILE* of = NULL;
		of = fopen("gamma_delta_eqm.txt", "w"); // existing data will be overwritten
		for (unsigned int i=0; i<GammaDeltaRoots.size(); i++)
			fprintf(of, "%f\t%f\t%f\t%f\n", GammaDeltaRoots[i][0], GammaDeltaRoots[i][1], GammaDeltaRoots[i][2], GammaDeltaRoots[i][3]);
		fclose(of);
	}

	if (GammaLavesRoots.size() > 0) {
		FILE* of = NULL;
		of = fopen("gamma_laves_eqm.txt", "w"); // existing data will be overwritten
		for (unsigned int i=0; i<GammaLavesRoots.size(); i++)
			fprintf(of, "%f\t%f\t%f\t%f\n", GammaLavesRoots[i][0], GammaLavesRoots[i][1], GammaLavesRoots[i][2], GammaLavesRoots[i][3]);
		fclose(of);
	}

	if (DeltaLavesRoots.size() > 0) {
		FILE* of = NULL;
		of = fopen("delta_laves_eqm.txt", "w"); // existing data will be overwritten
		for (unsigned int i=0; i<DeltaLavesRoots.size(); i++)
			fprintf(of, "%f\t%f\t%f\t%f\n", DeltaLavesRoots[i][0], DeltaLavesRoots[i][1], DeltaLavesRoots[i][2], DeltaLavesRoots[i][3]);
		fclose(of);
	}


	return 0;
}



#endif
