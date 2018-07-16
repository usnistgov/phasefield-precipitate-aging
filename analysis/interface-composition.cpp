// FILE:   ifcomp.cpp
// INPUT:  MMSP grid containing vector data with at least two fields
// OUTPUT: Matrix and precipitate compositions at the interfaces
// Questions/comments to trevor.keller@nist.gov (Trevor Keller)

#include "MMSP.hpp"
#include <iomanip>
#include <vector>
#include <algorithm>

#if defined CALPHAD
	#include"energy625.c"
#elif defined PARABOLA
	#include"parabola625.c"
#else
	#include"taylor625.c"
#endif

#define NC 2
#define NP 2

/* Representation includes ten field variables:
 *
 * X0.  molar fraction of Cr + Mo
 * X1.  molar fraction of Nb
 *
 * P2.  phase fraction of delta
 * P3.  phase fraction of Laves
 *
 * C4.  Cr molar fraction in pure gamma
 * C5.  Nb molar fraction in pure gamma
 *
 * C6.  Cr molar fraction in pure delta
 * C7.  Nb molar fraction in pure delta
 *
 * C8. Cr molar fraction in pure Laves
 * C9. Nb molar fraction in pure Laves
 */

template <typename T>
double h(const T& p) {return p * p * p * (6.0 * p * p - 15.0 * p + 10.0);}

template<int dim, typename T>
void vectorComp(const MMSP::grid<dim,MMSP::vector<T> >& GRID)
{
	std::cout << std::setprecision(6);

	MMSP::vector<int> x(dim, 0);
	double dV = 1.0;
	for (int d=0; d<dim; d++)
		dV *= MMSP::dx(GRID,d);

	// compute phase fractions
	double fDel = 0.0;
	double fLav = 0.0;
	const int N = MMSP::nodes(GRID);
	for (int n=0; n<N; n++) {
		fDel += h(GRID(n)[2]);
		fLav += h(GRID(n)[3]);
	}
	fDel /= double(N);
	fLav /= double(N);

	// approximate precipitate radii and "pressures"
	const double rDel = std::sqrt(N * dV * fDel / M_PI);
	const double rLav = std::sqrt(N * dV * fLav / M_PI);
	const double pDel = 2.0 * s_delta() / rDel;
	const double pLav = 2.0 * s_laves() / rLav;


	// Find composition at gamma-delta interface
	x[0] = -8 - 0.5 * (MMSP::g1(GRID, 0) - MMSP::g0(GRID, 0));
	x[1] = MMSP::g1(GRID, 1);
	while (h((GRID(x)[2])) > 0.5)
		x[1]--;
	const int delTop = x[1];

	T xCr = GRID(x)[0];
	T xNb = GRID(x)[1];
	T matCr = GRID(x)[4];
	T matNb = GRID(x)[5];
	T preCr = GRID(x)[6];
	T preNb = GRID(x)[7];

	x[1] = MMSP::g0(GRID, 1);
	while (h((GRID(x)[2])) > 0.5)
		x[1]++;
	const int delBot = x[1];

	T dx1 = matCr - xe_gam_Cr();
	T dx2 = matNb - xe_gam_Nb();
	T dx3 = preCr - xe_del_Cr();
	T dx4 = preNb - xe_del_Nb();

	T A1 = d2g_gam_dxCrCr();
	T A2 = d2g_gam_dxCrNb();
	T A3 = d2g_del_dxCrCr();
	T A4 = d2g_del_dxCrNb();

	T B1 = d2g_gam_dxNbCr();
	T B2 = d2g_gam_dxNbNb();
	T B3 = d2g_del_dxNbCr();
	T B4 = d2g_del_dxNbNb();

	T C1 = matCr * d2g_gam_dxCrCr() + matNb * d2g_gam_dxNbCr();
	T C2 = matCr * d2g_gam_dxCrNb() + matNb * d2g_gam_dxNbNb();
	T C3 = preCr * d2g_del_dxCrCr() + preNb * d2g_del_dxNbCr();
	T C4 = preCr * d2g_del_dxCrNb() + preNb * d2g_del_dxNbNb();

	double r1 = std::pow(A1*dx1 + A2*dx2 - A3*dx3 - A4*dx4, 2.);
	double r2 = std::pow(B1*dx1 + B2*dx2 - B3*dx3 - B4*dx4, 2.);
	double r3 = std::pow(C1*dx1 + C2*dx2 - C3*dx3 - C4*dx4 + pDel, 2.);
	double res = std::sqrt((r1+r2+r3) / 3.);
	double rad = MMSP::dx(GRID,1) * (delTop - delBot);

	std::cout << rad;
	std::cout << ',' << xCr << ',' << xNb;
	std::cout << ',' << matCr << ',' << matNb;
	std::cout << ',' << preCr << ',' << preNb;
	std::cout << ',' << res;

	// Find composition at gamma-delta interface
	x[0] = 8 + 0.5 * (MMSP::g1(GRID, 0) - MMSP::g0(GRID, 0));
	x[1] = MMSP::g1(GRID, 1);
	while (h((GRID(x)[3])) > 0.5)
		x[1]--;
	const int lavTop = x[1];

	xCr = GRID(x)[0];
	xNb = GRID(x)[1];
	matCr = GRID(x)[4];
	matNb = GRID(x)[5];
	preCr = GRID(x)[8];
	preNb = GRID(x)[9];

	x[1] = MMSP::g0(GRID, 1);
	while (h((GRID(x)[3])) > 0.5)
		x[1]++;
	const int lavBot = x[1];

	dx1 = matCr - xe_gam_Cr();
	dx2 = matNb - xe_gam_Nb();
	dx3 = preCr - xe_lav_Cr();
	dx4 = preNb - xe_lav_Nb();

	A1 = d2g_gam_dxCrCr();
	A2 = d2g_gam_dxCrNb();
	A3 = d2g_lav_dxCrCr();
	A4 = d2g_lav_dxCrNb();

	B1 = d2g_gam_dxNbCr();
	B2 = d2g_gam_dxNbNb();
	B3 = d2g_lav_dxNbCr();
	B4 = d2g_lav_dxNbNb();

	C1 = matCr * d2g_gam_dxCrCr() + matNb * d2g_gam_dxNbCr();
	C2 = matCr * d2g_gam_dxCrNb() + matNb * d2g_gam_dxNbNb();
	C3 = preCr * d2g_lav_dxCrCr() + preNb * d2g_lav_dxNbCr();
	C4 = preCr * d2g_lav_dxCrNb() + preNb * d2g_lav_dxNbNb();

	r1 = std::pow(A1*dx1 + A2*dx2 - A3*dx3 - A4*dx4, 2.);
	r2 = std::pow(B1*dx1 + B2*dx2 - B3*dx3 - B4*dx4, 2.);
	r3 = std::pow(C1*dx1 + C2*dx2 - C3*dx3 - C4*dx4 + pLav, 2.);
	res = std::sqrt((r1+r2+r3) / 3.);
	rad = MMSP::dx(GRID,1) * (lavTop - lavBot);

	std::cout << ',' << rad;
	std::cout << ',' << xCr << ',' << xNb;
	std::cout << ',' << matCr << ',' << matNb;
	std::cout << ',' << preCr << ',' << preNb;
	std::cout << ',' << res << '\n';
}

int main(int argc, char* argv[]) {
	// command line error check
	if (argc < 2) {
		std::cout << "Usage: " << argv[0] << " [--help] infile\n\n";
		exit(-1);
	}

	// help diagnostic
	if (std::string(argv[1]) == "--help") {
		std::cout << argv[0] << ": print position and composition of precipitate interfaces.\n";
		std::cout << "Usage: " << argv[0] << " [--help] infile\n\n";
		std::cout << "Questions/comments to trevor.keller@nist.gov (Trevor Keller).\n\n";
		exit(0);
	}

	// file open error check
	std::ifstream input(argv[1]);
	if (!input) {
		std::cerr << "File input error: could not open " << argv[1] << ".\n\n";
		exit(-1);
	}

	// read data type
	std::string type;
	getline(input, type, '\n');

	// grid type error check
	if (type.substr(0, 4) != "grid") {
		std::cerr << "File input error: file does not contain grid data." << std::endl;
		exit(-1);
	}

	// data type error check
	bool vector_type = (type.find("vector") != std::string::npos);
	if (not vector_type) {
		std::cerr << "File input error: grid does not contain vector data." << std::endl;
		exit(-1);
	}


	// parse data type
	bool bool_type = (type.find("bool") != std::string::npos);
	bool char_type = (type.find("char") != std::string::npos);
	bool unsigned_char_type = (type.find("unsigned char") != std::string::npos);
	bool int_type = (type.find("int") != std::string::npos);
	bool unsigned_int_type = (type.find("unsigned int") != std::string::npos);
	bool long_type = (type.find("long") != std::string::npos);
	bool unsigned_long_type = (type.find("unsigned long") != std::string::npos);
	bool short_type = (type.find("short") != std::string::npos);
	bool unsigned_short_type = (type.find("unsigned short") != std::string::npos);
	bool float_type = (type.find("float") != std::string::npos);
	bool double_type = (type.find("double") != std::string::npos);
	bool long_double_type = (type.find("long double") != std::string::npos);


	if (not bool_type    and
	    not char_type    and  not unsigned_char_type   and
	    not int_type     and  not unsigned_int_type    and
	    not long_type    and  not unsigned_long_type   and
	    not short_type   and  not unsigned_short_type  and
	    not float_type   and
	    not double_type  and  not long_double_type) {
		std::cerr << "File input error: unknown grid data type." << std::endl;
		exit(-1);
	}

	// read grid dimension
	int dim;
	input >> dim;

	// read number of fields
	int fields;
	input >> fields;


	// write grid data
	if (vector_type && fields>1) {
		if (float_type) {
			if (dim == 1) {
				MMSP::grid<1, MMSP::vector<float> > GRID(argv[1]);
				vectorComp(GRID);
			} else if (dim == 2) {
				MMSP::grid<2, MMSP::vector<float> > GRID(argv[1]);
				vectorComp(GRID);
			} else if (dim == 3) {
				MMSP::grid<3, MMSP::vector<float> > GRID(argv[1]);
				vectorComp(GRID);
			}
		}
		if (double_type) {
			if (dim == 1) {
				MMSP::grid<1, MMSP::vector<double> > GRID(argv[1]);
				vectorComp(GRID);
			} else if (dim == 2) {
				MMSP::grid<2, MMSP::vector<double> > GRID(argv[1]);
				vectorComp(GRID);
			} else if (dim == 3) {
				MMSP::grid<3, MMSP::vector<double> > GRID(argv[1]);
				vectorComp(GRID);
			}
		}
		if (long_double_type) {
			if (dim == 1) {
				MMSP::grid<1, MMSP::vector<long double> > GRID(argv[1]);
				vectorComp(GRID);
			} else if (dim == 2) {
				MMSP::grid<2, MMSP::vector<long double> > GRID(argv[1]);
				vectorComp(GRID);
			} else if (dim == 3) {
				MMSP::grid<3, MMSP::vector<long double> > GRID(argv[1]);
				vectorComp(GRID);
			}
		}
	}

	return 0;
}
