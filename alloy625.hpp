/* alloy625.hpp
 * Declarations for 2D and 3D isotropic Cr-Nb-Ni alloy phase transformations
 * Questions/comments to trevor.keller@nist.gov (Trevor Keller)
 */

std::string PROGRAM = "alloy625";
std::string MESSAGE = "Isotropic Cr-Nb-Ni alloy phase transformation code";

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

// Initial guesses for gamma, mu, and delta equilibrium compositions
void guessGamma(const double& xcr, const double& xnb, double& ccr, double& cnb);
void guessDelta(const double& xcr, const double& xnb, double& ccr, double& cnb);
void guessMu(   const double& xcr, const double& xnb, double& ccr, double& cnb);
void guessLaves(const double& xcr, const double& xnb, double& ccr, double& cnb);
template<typename T>
void embedParticle(MMSP::grid<2,MMSP::vector<T> >& GRID, const MMSP::vector<int>& origin, const int pid,
                const double rprcp, const double rdpltCr, const double rdeplNb,
                const double& xCr, const double& xNb, const T phi);

// Phi spans [-1,+1], need to know its sign without divide-by-zero errors
double sign(double x) {return (x<0) ? -1.0 : 1.0;}

void simple_progress(int step, int steps); // thread-compatible pared-down version of print_progress

template<int dim, typename T> void print_values(const MMSP::grid<dim,MMSP::vector<T> >& oldGrid, const int rank);


/* Given const phase fractions (in gamma, mu, and delta)
 * and concentrations of Cr and Nb, iteratively determine the fictitious concentrations in each phase that
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
	template <typename T> double solve(const T& x_Cr, const T& x_Nb,
	                                   const T& p_del, const T& p_mu, const T& p_lav,
	                                   T& C_gam_Cr, T& C_del_Cr, T& C_mu_Cr, T& C_lav_Cr,
	                                   T& C_gam_Nb, T& C_del_Nb, T& C_mu_Nb, T& C_lav_Nb);

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
