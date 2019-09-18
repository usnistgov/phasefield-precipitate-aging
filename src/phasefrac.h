/**
 \file phasefrac.h
 Use GSL to solve for tie lines and phase fractions of secondary phases
*/

/**
 \brief Returns Euclidean distance between compositions
*/
double euclidean(double xACr, double xANb, double xBCr, double xBNb);

/**
 \brief Container for fixed properties (system composition)
*/
struct rparams {
	double xCr0;
	double xNb0;
};

/**
 \brief Function definition (delta)
*/
int common_tan_del_f(const gsl_vector* x, void* params, gsl_vector* f);

/**
 \brief Jacobian definition (delta)
*/
int common_tan_del_df(const gsl_vector* x, void* params, gsl_matrix* J);

/**
 \brief Wrapper for Function & Jacobian (delta)
*/
int common_tan_del_fdf(const gsl_vector* x, void* params, gsl_vector* f, gsl_matrix* J);

/**
 \brief Compute phase fraction of delta
*/
double estimate_fraction_del(const double xCr, const double xNb);

/**
 \brief Function definition (Laves)
*/
int common_tan_lav_f(const gsl_vector* x, void* params, gsl_vector* f);

/**
 \brief Jacobian definition (Laves)
*/
int common_tan_lav_df(const gsl_vector* x, void* params, gsl_matrix* J);

/**
 \brief Wrapper for Function & Jacobian (Laves)
*/
int common_tan_lav_fdf(const gsl_vector* x, void* params, gsl_vector* f, gsl_matrix* J);

/**
 \brief Compute phase fraction of Laves
*/
double estimate_fraction_lav(const double xCr, const double xNb);
