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
T h(const T& p) {return p * p * p * (6.0 * p * p - 15.0 * p + 10.0);}

template<int dim, typename T>
void vectorComp(const MMSP::grid<dim,MMSP::vector<T> >& GRID)
{
	std::cout << std::setprecision(6);

	// Perform a line-scan parallel to the x-axis through the center of the grid
	MMSP::vector<int> x(dim, 0);
	for (int d=0; d<dim; d++)
		x[d] = (MMSP::g1(GRID,d)+MMSP::g0(GRID,d))/2;

	// Find gamma-delta interface
	x[0] = 0;
	while (h((GRID(x)[2])) < 0.5)
		x[0]--;

	T xCr = GRID(x)[0];
	T xNb = GRID(x)[1];
	T matCr = GRID(x)[4];
	T matNb = GRID(x)[5];
	T preCr = GRID(x)[6];
	T preNb = GRID(x)[7];

	std::cout << x[0];
	std::cout << ',' << xCr << ',' << xNb;
	std::cout << ',' << matCr << ',' << matNb;
	std::cout << ',' << preCr << ',' << preNb;

	// Find gamma-Laves interface
	x[0] = 0;
	while (h((GRID(x)[3])) < 0.5)
		x[0]++;

	xCr = GRID(x)[0];
	xNb = GRID(x)[1];
	matCr = GRID(x)[4];
	matNb = GRID(x)[5];
	preCr = GRID(x)[8];
	preNb = GRID(x)[9];

	std::cout << ',' << x[0];
	std::cout << ',' << xCr << ',' << xNb;
	std::cout << ',' << matCr << ',' << matNb;
	std::cout << ',' << preCr << ',' << preNb;
	std::cout << '\n';
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
