/*************************************************************************************
 * File: equilibrium.hpp                                                             *
 * Implementation of equations for Cr-Nb-Ni alloy phase diagrams                     *
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

#ifndef ALLOY625_EQUATIONS
#define ALLOY625_EQUATIONS

// Taylor series is your best bet.
#if defined CALPHAD
	#include"energy625.c"
#elif defined PARABOLA
	#include"parabola625.c"
#else
	#include"taylor625.c"
#endif



/* ======================================= *
 * Equations for Gamma - Delta Equilibrium *
 * ======================================= */

// Struct to hold parallel tangent solver constants
struct rparams {
	// Composition fields
	double x_Cr;
	double x_Nb;
};

int GammaDelta_f(const gsl_vector* x, void* params, gsl_vector* f)
{
	gsl_vector_set_zero(f);

	const double x_Cr = ((struct rparams*) params)->x_Cr;
	const double x_Nb = ((struct rparams*) params)->x_Nb;

	const double C_gam_Cr = gsl_vector_get(x, 0);
	const double C_gam_Nb = gsl_vector_get(x, 1);

	const double C_del_Cr = gsl_vector_get(x, 2);
	const double C_del_Nb = gsl_vector_get(x, 3);

	const double fg = g_gam(C_gam_Cr, C_gam_Nb);
	const double fd = g_del(C_del_Cr, C_del_Nb);

	const double dgGdxCr = dg_gam_dxCr(C_gam_Cr, C_gam_Nb);
	const double dgGdxNb = dg_gam_dxNb(C_gam_Cr, C_gam_Nb);

	const double dgDdxCr = dg_del_dxCr(C_del_Cr, C_del_Nb);
	const double dgDdxNb = dg_del_dxNb(C_del_Cr, C_del_Nb);

	gsl_vector_set(f, 0, dgGdxCr - dgDdxCr);
	gsl_vector_set(f, 1, dgGdxNb - dgDdxNb);

	gsl_vector_set(f, 2, fg - fd - (C_gam_Cr - C_del_Cr) * dgGdxCr - (C_gam_Nb - C_del_Nb) * dgGdxNb);
	gsl_vector_set(f, 3, (x_Cr - C_del_Cr)/(x_Nb - C_del_Nb) - (C_gam_Cr - C_del_Cr)/(C_gam_Nb - C_del_Nb));

	return GSL_SUCCESS;
}

int GammaDelta_df(const gsl_vector* x, void* params, gsl_matrix* J)
{
	gsl_matrix_set_zero(J);

	const double x_Cr = ((struct rparams*) params)->x_Cr;
	const double x_Nb = ((struct rparams*) params)->x_Nb;

	const double C_gam_Cr = gsl_vector_get(x, 0);
	const double C_gam_Nb = gsl_vector_get(x, 1);

	const double C_del_Cr = gsl_vector_get(x, 2);
	const double C_del_Nb = gsl_vector_get(x, 3);

	#ifdef PARABOLA
	const double d2gam_CrCr = d2g_gam_dxCrCr();
	const double g2gam_CrNb = d2g_gam_dxCrNb();
	const double d2gam_NbNb = d2g_gam_dxNbNb();
	#else
	const double d2gam_CrCr = d2g_gam_dxCrCr(C_gam_Cr, C_gam_Nb);
	const double g2gam_CrNb = d2g_gam_dxCrNb(C_gam_Cr, C_gam_Nb);
	const double d2gam_NbNb = d2g_gam_dxNbNb(C_gam_Cr, C_gam_Nb);
	#endif

	// Conservation of mass (Cr, Nb)
	gsl_matrix_set(J, 0, 0, d2gam_CrCr);
	gsl_matrix_set(J, 0, 1, g2gam_CrNb);
	gsl_matrix_set(J, 1, 0, g2gam_CrNb);
	gsl_matrix_set(J, 1, 1, d2gam_NbNb);

	// Equal chemical potential involving delta phase (Cr, Nb)
	#ifdef PARABOLA
	const double d2del_CrCr = d2g_del_dxCrCr();
	const double d2del_CrNb = d2g_del_dxCrNb();
	const double d2del_NbNb = d2g_del_dxNbNb();
	#elif defined CALPHAD
	const double d2del_CrCr = d2g_del_dxCrCr(C_del_Cr, C_del_Nb);
	const double d2del_CrNb = d2g_del_dxCrNb(C_del_Cr, C_del_Nb);
	const double d2del_NbNb = d2g_del_dxNbNb(C_del_Cr, C_del_Nb);
	#else
	const double d2del_CrCr = d2g_del_dxCrCr(C_del_Cr);
	const double d2del_CrNb = d2g_del_dxCrNb(C_del_Nb);
	const double d2del_NbNb = d2g_del_dxNbNb(C_del_Cr, C_del_Nb);
	#endif

	gsl_matrix_set(J, 0, 2, -d2del_CrCr);
	gsl_matrix_set(J, 0, 3, -d2del_CrNb);
	gsl_matrix_set(J, 1, 2, -d2del_CrNb);
	gsl_matrix_set(J, 1, 3, -d2del_NbNb);

	const double dgam_Cr = dg_gam_dxCr(C_gam_Cr, C_gam_Nb);
	const double dgam_Nb = dg_gam_dxNb(C_gam_Cr, C_gam_Nb);

	const double ddel_Cr = dg_del_dxCr(C_del_Cr, C_del_Nb);
	const double ddel_Nb = dg_del_dxNb(C_del_Cr, C_del_Nb);

	const double J20 = (C_del_Cr - C_gam_Cr) * d2gam_CrCr + (C_del_Nb - C_gam_Nb) * g2gam_CrNb;
	const double J21 = (C_del_Cr - C_gam_Cr) * g2gam_CrNb + (C_del_Nb - C_gam_Nb) * d2gam_NbNb;
	const double J22 = dgam_Cr - ddel_Cr;
	const double J23 = dgam_Nb - ddel_Nb;

	gsl_matrix_set(J, 2, 0, J20);
	gsl_matrix_set(J, 2, 1, J21);
	gsl_matrix_set(J, 2, 2, J22);
	gsl_matrix_set(J, 2, 3, J23);

	const double J30 = -1.0 / (C_gam_Nb - C_del_Nb);
	const double J31 = (C_gam_Cr - C_del_Cr) / pow(C_gam_Nb - C_del_Nb, 2);
	const double J32 = 1.0 / (C_gam_Nb - C_del_Nb) - 1.0 / (x_Nb - C_del_Nb);
	const double J33 = (x_Cr - C_del_Cr) / pow(x_Nb - C_del_Nb, 2) - (C_gam_Cr - C_del_Cr) / pow(C_gam_Nb - C_del_Nb, 2);

	gsl_matrix_set(J, 3, 0, J30);
	gsl_matrix_set(J, 3, 1, J31);
	gsl_matrix_set(J, 3, 3, J32);
	gsl_matrix_set(J, 3, 3, J33);

	return GSL_SUCCESS;
}

int GammaDelta_fdf(const gsl_vector* x, void* params, gsl_vector* f, gsl_matrix* J)
{
	GammaDelta_f( x, params, f);
	GammaDelta_df(x, params, J);

	return GSL_SUCCESS;
}




/* ======================================= *
 * Equations for Gamma - Laves Equilibrium *
 * ======================================= */

int GammaLaves_f(const gsl_vector* x, void* params, gsl_vector* f)
{
	gsl_vector_set_zero(f);

	const double x_Cr = ((struct rparams*) params)->x_Cr;
	const double x_Nb = ((struct rparams*) params)->x_Nb;

	const double C_gam_Cr = gsl_vector_get(x, 0);
	const double C_gam_Nb = gsl_vector_get(x, 1);

	const double C_lav_Cr = gsl_vector_get(x, 2);
	const double C_lav_Nb = gsl_vector_get(x, 3);

	const double fg = g_gam(C_gam_Cr, C_gam_Nb);
	const double fl = g_lav(C_lav_Cr, C_lav_Nb);

	const double dgGdxCr = dg_gam_dxCr(C_gam_Cr, C_gam_Nb);
	const double dgGdxNb = dg_gam_dxNb(C_gam_Cr, C_gam_Nb);

	const double dgLdxCr = dg_lav_dxCr(C_lav_Cr, C_lav_Nb);
	const double dgLdxNb = dg_lav_dxNb(C_lav_Cr, C_lav_Nb);

	gsl_vector_set(f, 0, dgGdxCr - dgLdxCr);
	gsl_vector_set(f, 1, dgGdxNb - dgLdxNb);

	gsl_vector_set(f, 2, fg - fl - (C_gam_Cr - C_lav_Cr) * dgGdxCr - (C_gam_Nb - C_lav_Nb) * dgGdxNb);
	gsl_vector_set(f, 3, (x_Cr - C_lav_Cr)/(x_Nb - C_lav_Nb) - (C_gam_Cr - C_lav_Cr)/(C_gam_Nb - C_lav_Nb));

	return GSL_SUCCESS;
}

int GammaLaves_df(const gsl_vector* x, void* params, gsl_matrix* J)
{
	gsl_matrix_set_zero(J);

	const double x_Cr = ((struct rparams*) params)->x_Cr;
	const double x_Nb = ((struct rparams*) params)->x_Nb;

	const double C_gam_Cr = gsl_vector_get(x, 0);
	const double C_gam_Nb = gsl_vector_get(x, 1);

	const double C_lav_Cr = gsl_vector_get(x, 2);
	const double C_lav_Nb = gsl_vector_get(x, 3);

	#ifdef PARABOLA
	const double d2gam_CrCr = d2g_gam_dxCrCr();
	const double g2gam_CrNb = d2g_gam_dxCrNb();
	const double d2gam_NbNb = d2g_gam_dxNbNb();
	#else
	const double d2gam_CrCr = d2g_gam_dxCrCr(C_gam_Cr, C_gam_Nb);
	const double g2gam_CrNb = d2g_gam_dxCrNb(C_gam_Cr, C_gam_Nb);
	const double d2gam_NbNb = d2g_gam_dxNbNb(C_gam_Cr, C_gam_Nb);
	#endif

	// Conservation of mass (Cr, Nb)
	gsl_matrix_set(J, 0, 0, d2gam_CrCr);
	gsl_matrix_set(J, 0, 1, g2gam_CrNb);
	gsl_matrix_set(J, 1, 0, g2gam_CrNb);
	gsl_matrix_set(J, 1, 1, d2gam_NbNb);

	// Equal chemical potential involving laves phase (Cr, Nb)
	#ifdef PARABOLA
	const double d2lav_CrCr = d2g_lav_dxCrCr();
	const double d2lav_CrNb = d2g_lav_dxCrNb();
	const double d2lav_NbNb = d2g_lav_dxNbNb();
	#elif defined CALPHAD
	const double d2lav_CrCr = d2g_lav_dxCrCr(C_lav_Cr, C_lav_Nb);
	const double d2lav_CrNb = d2g_lav_dxCrNb(C_lav_Cr, C_lav_Nb);
	const double d2lav_NbNb = d2g_lav_dxNbNb(C_lav_Cr, C_lav_Nb);
	#else
	const double d2lav_CrCr = d2g_lav_dxCrCr(C_lav_Cr, C_lav_Nb);
	const double d2lav_CrNb = d2g_lav_dxCrNb(C_lav_Cr, C_lav_Nb);
	const double d2lav_NbNb = d2g_lav_dxNbNb(C_lav_Cr, C_lav_Nb);
	#endif

	gsl_matrix_set(J, 0, 2, -d2lav_CrCr);
	gsl_matrix_set(J, 0, 3, -d2lav_CrNb);
	gsl_matrix_set(J, 1, 2, -d2lav_CrNb);
	gsl_matrix_set(J, 1, 3, -d2lav_NbNb);

	const double dgam_Cr = dg_gam_dxCr(C_gam_Cr, C_gam_Nb);
	const double dgam_Nb = dg_gam_dxNb(C_gam_Cr, C_gam_Nb);

	const double dlav_Cr = dg_lav_dxCr(C_lav_Cr, C_lav_Nb);
	const double dlav_Nb = dg_lav_dxNb(C_lav_Cr, C_lav_Nb);

	const double J20 = (C_lav_Cr - C_gam_Cr) * d2gam_CrCr + (C_lav_Nb - C_gam_Nb) * g2gam_CrNb;
	const double J21 = (C_lav_Cr - C_gam_Cr) * g2gam_CrNb + (C_lav_Nb - C_gam_Nb) * d2gam_NbNb;
	const double J22 = dgam_Cr - dlav_Cr;
	const double J23 = dgam_Nb - dlav_Nb;

	gsl_matrix_set(J, 2, 0, J20);
	gsl_matrix_set(J, 2, 1, J21);
	gsl_matrix_set(J, 2, 2, J22);
	gsl_matrix_set(J, 2, 3, J23);

	const double J30 = -1.0 / (C_gam_Nb - C_lav_Nb);
	const double J31 = (C_gam_Cr - C_lav_Cr) / pow(C_gam_Nb - C_lav_Nb, 2);
	const double J32 = 1.0 / (C_gam_Nb - C_lav_Nb) - 1.0 / (x_Nb - C_lav_Nb);
	const double J33 = (x_Cr - C_lav_Cr) / pow(x_Nb - C_lav_Nb, 2) - (C_gam_Cr - C_lav_Cr) / pow(C_gam_Nb - C_lav_Nb, 2);

	gsl_matrix_set(J, 3, 0, J30);
	gsl_matrix_set(J, 3, 1, J31);
	gsl_matrix_set(J, 3, 3, J32);
	gsl_matrix_set(J, 3, 3, J33);

	return GSL_SUCCESS;
}

int GammaLaves_fdf(const gsl_vector* x, void* params, gsl_vector* f, gsl_matrix* J)
{
	GammaLaves_f( x, params, f);
	GammaLaves_df(x, params, J);

	return GSL_SUCCESS;
}

/* ======================================= *
 * Equations for Delta - Laves Equilibrium *
 * ======================================= */

int DeltaLaves_f(const gsl_vector* x, void* params, gsl_vector* f)
{
	gsl_vector_set_zero(f);

	const double x_Cr = ((struct rparams*) params)->x_Cr;
	const double x_Nb = ((struct rparams*) params)->x_Nb;

	const double C_del_Cr = gsl_vector_get(x, 0);
	const double C_del_Nb = gsl_vector_get(x, 1);

	const double C_lav_Cr = gsl_vector_get(x, 2);
	const double C_lav_Nb = gsl_vector_get(x, 3);

	const double fd = g_del(C_del_Cr, C_del_Nb);
	const double fl = g_lav(C_lav_Cr, C_lav_Nb);

	const double dgDdxCr = dg_del_dxCr(C_del_Cr, C_del_Nb);
	const double dgDdxNb = dg_del_dxNb(C_del_Cr, C_del_Nb);

	const double dgLdxCr = dg_lav_dxCr(C_lav_Cr, C_lav_Nb);
	const double dgLdxNb = dg_lav_dxNb(C_lav_Cr, C_lav_Nb);

	gsl_vector_set(f, 0, dgDdxCr - dgLdxCr);
	gsl_vector_set(f, 1, dgDdxNb - dgLdxNb);

	gsl_vector_set(f, 2, fd - fl - (C_del_Cr - C_lav_Cr) * dgDdxCr - (C_del_Nb - C_lav_Nb) * dgDdxNb);
	gsl_vector_set(f, 3, (x_Cr - C_lav_Cr)/(x_Nb - C_lav_Nb) - (C_del_Cr - C_lav_Cr)/(C_del_Nb - C_lav_Nb));

	return GSL_SUCCESS;
}

int DeltaLaves_df(const gsl_vector* x, void* params, gsl_matrix* J)
{
	gsl_matrix_set_zero(J);

	const double x_Cr = ((struct rparams*) params)->x_Cr;
	const double x_Nb = ((struct rparams*) params)->x_Nb;

	const double C_del_Cr = gsl_vector_get(x, 0);
	const double C_del_Nb = gsl_vector_get(x, 1);

	const double C_lav_Cr = gsl_vector_get(x, 2);
	const double C_lav_Nb = gsl_vector_get(x, 3);

	#ifdef PARABOLA
	const double d2del_CrCr = d2g_del_dxCrCr();
	const double g2del_CrNb = d2g_del_dxCrNb();
	const double d2del_NbNb = d2g_del_dxNbNb();
	#elif defined CALPHAD
	const double d2del_CrCr = d2g_del_dxCrCr(C_del_Cr, C_del_Nb);
	const double g2del_CrNb = d2g_del_dxCrNb(C_del_Cr, C_del_Nb);
	const double d2del_NbNb = d2g_del_dxNbNb(C_del_Cr, C_del_Nb);
	#else
	const double d2del_CrCr = d2g_del_dxCrCr(C_del_Cr);
	const double g2del_CrNb = d2g_del_dxCrNb(C_del_Nb);
	const double d2del_NbNb = d2g_del_dxNbNb(C_del_Cr, C_del_Nb);
	#endif

	// Conservation of mass (Cr, Nb)
	gsl_matrix_set(J, 0, 0, d2del_CrCr);
	gsl_matrix_set(J, 0, 1, g2del_CrNb);
	gsl_matrix_set(J, 1, 0, g2del_CrNb);
	gsl_matrix_set(J, 1, 1, d2del_NbNb);

	// Equal chemical potential involving laves phase (Cr, Nb)
	#ifdef PARABOLA
	const double d2lav_CrCr = d2g_lav_dxCrCr();
	const double d2lav_CrNb = d2g_lav_dxCrNb();
	const double d2lav_NbNb = d2g_lav_dxNbNb();
	#elif defined CALPHAD
	const double d2lav_CrCr = d2g_lav_dxCrCr(C_lav_Cr, C_lav_Nb);
	const double d2lav_CrNb = d2g_lav_dxCrNb(C_lav_Cr, C_lav_Nb);
	const double d2lav_NbNb = d2g_lav_dxNbNb(C_lav_Cr, C_lav_Nb);
	#else
	const double d2lav_CrCr = d2g_lav_dxCrCr(C_lav_Cr, C_lav_Nb);
	const double d2lav_CrNb = d2g_lav_dxCrNb(C_lav_Cr, C_lav_Nb);
	const double d2lav_NbNb = d2g_lav_dxNbNb(C_lav_Cr, C_lav_Nb);
	#endif

	gsl_matrix_set(J, 0, 2, -d2lav_CrCr);
	gsl_matrix_set(J, 0, 3, -d2lav_CrNb);
	gsl_matrix_set(J, 1, 2, -d2lav_CrNb);
	gsl_matrix_set(J, 1, 3, -d2lav_NbNb);

	const double ddel_Cr = dg_del_dxCr(C_del_Cr, C_del_Nb);
	const double ddel_Nb = dg_del_dxNb(C_del_Cr, C_del_Nb);

	const double dlav_Cr = dg_lav_dxCr(C_lav_Cr, C_lav_Nb);
	const double dlav_Nb = dg_lav_dxNb(C_lav_Cr, C_lav_Nb);

	const double J20 = (C_lav_Cr - C_del_Cr) * d2del_CrCr + (C_lav_Nb - C_del_Nb) * g2del_CrNb;
	const double J21 = (C_lav_Cr - C_del_Cr) * g2del_CrNb + (C_lav_Nb - C_del_Nb) * d2del_NbNb;
	const double J22 = ddel_Cr - dlav_Cr;
	const double J23 = ddel_Nb - dlav_Nb;

	gsl_matrix_set(J, 2, 0, J20);
	gsl_matrix_set(J, 2, 1, J21);
	gsl_matrix_set(J, 2, 2, J22);
	gsl_matrix_set(J, 2, 3, J23);

	const double J30 = -1.0 / (C_del_Nb - C_lav_Nb);
	const double J31 = (C_del_Cr - C_lav_Cr) / pow(C_del_Nb - C_lav_Nb, 2);
	const double J32 = 1.0 / (C_del_Nb - C_lav_Nb) - 1.0 / (x_Nb - C_lav_Nb);
	const double J33 = (x_Cr - C_lav_Cr) / pow(x_Nb - C_lav_Nb, 2) - (C_del_Cr - C_lav_Cr) / pow(C_del_Nb - C_lav_Nb, 2);

	gsl_matrix_set(J, 3, 0, J30);
	gsl_matrix_set(J, 3, 1, J31);
	gsl_matrix_set(J, 3, 3, J32);
	gsl_matrix_set(J, 3, 3, J33);

	return GSL_SUCCESS;
}

int DeltaLaves_fdf(const gsl_vector* x, void* params, gsl_vector* f, gsl_matrix* J)
{
	DeltaLaves_f( x, params, f);
	DeltaLaves_df(x, params, J);

	return GSL_SUCCESS;
}

#endif
