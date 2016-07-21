/* alloy625.hpp
 * Declarations for 2D and 3D isotropic binary alloy solidification
 * Questions/comments to trevor.keller@nist.gov (Trevor Keller)
 */

std::string PROGRAM = "alloy625";
std::string MESSAGE = "Isotropic phase field solidification example code";

typedef MMSP::grid<1,MMSP::vector<double> > GRID1D;
typedef MMSP::grid<2,MMSP::vector<double> > GRID2D;
typedef MMSP::grid<3,MMSP::vector<double> > GRID3D;

// Interface interpolation function
double h(const double p);
double hprime(const double p);

// Double well potential
double g(const double p);
double gprime(const double p);

// pure phase free energy densities
double f_gam(const double x_Al, const double x_Nb);
double f_gpr(const double x_Al, const double x_Nb);
double f_gdp(const double x_Al, const double x_Nb);
double f_del(const double x_Al, const double x_Nb);

// first derivatives of pure phase free energy densities
double df_gam_dx_Al(const double x_Al, const double x_Nb);
double df_gpr_dx_Al(const double x_Al, const double x_Nb);
double df_gdp_dx_Al(const double x_Al, const double x_Nb);
double df_del_dx_Al(const double x_Al, const double x_Nb);

double df_gam_dx_Nb(const double x_Al, const double x_Nb);
double df_gpr_dx_Nb(const double x_Al, const double x_Nb);
double df_gdp_dx_Nb(const double x_Al, const double x_Nb);
double df_del_dx_Nb(const double x_Al, const double x_Nb);

// chemical potentials of pure phases
double d2f_gam_dx2_Al(const double x_Al, const double x_Nb);
double d2f_gpr_dx2_Al(const double x_Al, const double x_Nb);
double d2f_gdp_dx2_Al(const double x_Al, const double x_Nb);
double d2f_del_dx2_Al(const double x_Al, const double x_Nb);

double d2f_gam_dx2_Nb(const double x_Al, const double x_Nb);
double d2f_gpr_dx2_Nb(const double x_Al, const double x_Nb);
double d2f_gdp_dx2_Nb(const double x_Al, const double x_Nb);
double d2f_del_dx2_Nb(const double x_Al, const double x_Nb);

// Gibbs free energy density
double gibbs(const MMSP::vector<double>& v);

void simple_progress(int step, int steps); // thread-compatible pared-down version of print_progress

template<int dim, typename T> void print_values(const MMSP::grid<dim,MMSP::vector<T> >& oldGrid, const int rank);


/* Given const phase fractions (in gamma prime variants 1,2,3; gamma double-prime; and delta)
 * and concentration, iteratively determine the fictitious concentrations in each phase that
 * satisfy the equal chemical potential constraint.
 * Pass p and x by const value, C by non-const reference to update in place.
 * This allows use of this single function to solve for both components, Al and Nb, in each phase.
 */

int commonTangent_f(const gsl_vector* x, void* params, gsl_vector* f);
int commonTangent_df(const gsl_vector* x, void* params, gsl_matrix* J);
int commonTangent_fdf(const gsl_vector* x, void* params, gsl_vector* f, gsl_matrix* J);

class rootsolver
{
public:
	// constructor
	rootsolver();
	// destructor
	~rootsolver();
	// accessor
	template <typename T> double solve(const T& x_Al, const T& x_Nb, const T& p_gpr, const T& p_gdp, const T& p_del,
	                                   T& C_gam_Al, T& C_gpr_Al, T& C_gdp_Al, t& C_del_Al, T& C_gam_Nb, T& C_gpr_Nb, T& C_gdp_Nb, t& C_del_Nb);

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
