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

double k(); // equilibrium partition coefficient for solidification

// phase-field diffusivity
double Q(const double p, const double Cs, const double Cl);

double fl(const double c);       // liquid free energy density

double fs(const double c);       // solid free energy density

double dfl_dc(const double c);   // first derivative of fl w.r.t. c

double dfs_dc(const double c);   // first derivative of fs w.r.t. c

double d2fl_dc2(const double c); // second derivative of fl w.r.t. c

double d2fs_dc2(const double c); // second derivative of fs w.r.t. c

double R(const double p, const double Cs, const double Cl); // denominator for dCs, dCl, df

double dCl_dc(const double p, const double Cs, const double Cl); // first derivative of Cl w.r.t. c

double dCs_dc(const double p, const double Cs, const double Cl); // first derivative of Cs w.r.t. c

double f(const double p, const double c, const double Cs, const double Cl); // free energy density

double d2f_dc2(const double p, const double c, const double Cs, const double Cl); // second derivative of f w.r.t. c

void simple_progress(int step, int steps); // thread-compatible pared-down version of print_progress

template<int dim, typename T> void print_values(const MMSP::grid<dim,MMSP::vector<T> >& oldGrid, const int rank);


/* Given const phase fractions (in gamma prime variants 1,2,3; gamma double-prime; and delta)
 * and concentration, iteratively determine the fictitious concentrations in each phase that
 * satisfy the equal chemical potential constraint.
 * Pass p[2-6] and x by const value, C[0-4] by non-const reference to update in place.
 * This allows use of this single function to solve for both components, Al and Nb, in each phase.
 */

int commonTangent_f(const gsl_vector* x, void* params, gsl_vector* f);
int commonTangent_df(const gsl_vector* x, void* params, gsl_matrix* J);
int commonTangent_fdf(const gsl_vector* x, void* params, gsl_vector* f, gsl_matrix* J);


struct rparams {
	// Composition fields
	double x;

	// Structure fields
	double p_gp1;
	double p_gp2;
	double p_gp3;
	double p_gdp;
	double p_del;
};


class rootsolver
{
public:
	// constructor
	rootsolver();
	// destructor
	~rootsolver();
	// accessor
	template <typename T> double solve(const T& c,
	                                   const T& p_gp1, const T& p_gp2, const T& p_gp3, const T& p_gdp, const T& p_del,
	                                   T& C_gam, T& C_gpr, T& C_gdp, t& C_del);

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


void export_energy(rootsolver& NRGsolver); // exports free energy curves to energy.csv
