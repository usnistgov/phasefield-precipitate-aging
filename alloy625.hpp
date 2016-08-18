/* alloy625.hpp
 * Declarations for 2D and 3D isotropic Mo-Nb-Ni alloy phase transformations
 * Questions/comments to trevor.keller@nist.gov (Trevor Keller)
 */

std::string PROGRAM = "alloy625";
std::string MESSAGE = "Isotropic Mo-Nb-Ni alloy phase transformation code";

typedef MMSP::grid<1,MMSP::vector<double> > GRID1D;
typedef MMSP::grid<2,MMSP::vector<double> > GRID2D;
typedef MMSP::grid<3,MMSP::vector<double> > GRID3D;

// Interface interpolation function
double h(const double p);
double hprime(const double p);

// Double well potential
double g(const double p);
double gprime(const double p);

// Gibbs free energy density
double gibbs(const MMSP::vector<double>& v);

void simple_progress(int step, int steps); // thread-compatible pared-down version of print_progress

template<int dim, typename T> void print_values(const MMSP::grid<dim,MMSP::vector<T> >& oldGrid, const int rank);


/* Given const phase fractions (in gamma, mu, and delta)
 * and concentrations of Mo and Nb, iteratively determine the fictitious concentrations in each phase that
 * satisfy the constraints of equal chemical potential and conservation of mass.
 * Pass p and x by const value, C by non-const reference to update in place.
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
	template <typename T> double solve(const T& x_Mo, const T& x_Nb, const T& p_mu, const T& p_del,
	                                   T& C_gam_Mo, T& C_mu_Mo, T& C_del_Mo,
	                                   T& C_gam_Nb, T& C_mu_Nb, T& C_del_Nb,
	                                   T& C_gam_Ni, T& C_mu_Ni, T& C_del_Ni);

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
