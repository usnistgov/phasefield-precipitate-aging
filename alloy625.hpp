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

// Gibbs free energy density
double gibbs(const MMSP::vector<double>& v);

// Initial ization and guesses for gamma, mu, and delta equilibrium compositions

double radius(const MMSP::vector<int>& a, const MMSP::vector<int>& b, const double& dx);

double bellCurve(double x, double m, double s);

double sign(double x) {return (x<0) ? -1.0 : 1.0;}


template<int dim,typename T>
void guessGamma(MMSP::grid<dim,MMSP::vector<T> >& GRID, int n, std::mt19937_64& mt_rand, std::uniform_real_distribution<T>& real_gen, const T& amp);

template<int dim,typename T>
void guessDelta(MMSP::grid<dim,MMSP::vector<T> >& GRID, int n, std::mt19937_64& mt_rand, std::uniform_real_distribution<T>& real_gen, const T& amp);

template<int dim,typename T>
void guessMu(MMSP::grid<dim,MMSP::vector<T> >& GRID, int n, std::mt19937_64& mt_rand, std::uniform_real_distribution<T>& real_gen, const T& amp);

template<int dim,typename T>
void guessLaves(MMSP::grid<dim,MMSP::vector<T> >& GRID, int n, std::mt19937_64& mt_rand, std::uniform_real_distribution<T>& real_gen, const T& amp);

template<typename T>
void embedParticle(MMSP::grid<2,MMSP::vector<T> >& GRID, const MMSP::vector<int>& origin, const int pid,
                const double rprcp, const double rdpltCr, const double rdeplNb,
                const T& xCr, const T& xNb,
                const double& xCrdep, const double& xNbdep, const T phi);

template<typename T>
void embedStripe(MMSP::grid<2,MMSP::vector<T> >& GRID, const MMSP::vector<int>& origin, const int pid,
                const double rprcp, const double rdpltCr, const double rdeplNb,
                const T& xCr, const T& xNb,
                const double& xCrdep, const double& xNbdep, const T phi);

// Phi spans [-1,+1], need to know its sign without divide-by-zero errors

void simple_progress(int step, int steps); // thread-compatible pared-down version of print_progress

template<int dim, typename T>
void print_values(const MMSP::grid<dim,MMSP::vector<T> >& GRID);

template<int dim,typename T>
void print_matrix(MMSP::grid<dim,MMSP::vector<T> >& GRID, int n);


/* Given const phase fractions (in gamma, mu, and delta)
 * and concentrations of Cr and Nb, iteratively determine the fictitious concentrations in each phase that
 * satisfy the constraints of equal chemical potential and conservation of mass.
 * Pass p and x by const value, C by non-const reference to update in place.
 */

struct rparams {
	// Composition fields
	double x_Cr;
	double x_Nb;

	// Structure fields
	double n_del;
	double n_mu;
	double n_lav;
};

int commonTangent_f(const gsl_vector* x, void* params, gsl_vector* f);
int commonTangent_df(const gsl_vector* x, void* params, gsl_matrix* J);
int commonTangent_fdf(const gsl_vector* x, void* params, gsl_vector* f, gsl_matrix* J);

class rootsolver
{
public:
	rootsolver();
	~rootsolver();
	template<int dim,typename T>
	double solve(MMSP::grid<dim,MMSP::vector<T> >& GRID, int n);

private:
	const size_t n;
	const size_t maxiter;
	const double tolerance;
	gsl_vector* x;
	struct rparams par;
	#ifndef JACOBIAN
	const gsl_multiroot_fsolver_type* algorithm;
	gsl_multiroot_fsolver* solver;
	gsl_multiroot_function mrf;
	#else
	const gsl_multiroot_fdfsolver_type* algorithm;
	gsl_multiroot_fdfsolver* solver;
	gsl_multiroot_function_fdf mrf;
	#endif
};


/* Solve for the phase field satisfying the given phase fraction */

struct nparams {
	double n;
};

double phaseKicker_f(double x, void* params);

class phasekicker
{
public:
	phasekicker();
	~phasekicker();
	template<typename T>
	int kick(const double& n, T& p);

private:
	const size_t maxiter;
	const double tolerance;
	struct nparams par;
	const gsl_root_fsolver_type* algorithm;
	gsl_root_fsolver* solver;
	gsl_function F;
};
