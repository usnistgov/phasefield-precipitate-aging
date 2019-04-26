/**
 \file adsorption.cpp
 \brief Compute interface adsorption
 Input: MMSP grid containing stable interface between two phases
 Output: adsorption of species A, B, C=1-A-B on phase boundaries

 Reference: W. Villanueva, W.J. Boettinger, G.B. McFadden, J.A. Warren.
 "A Diffuse-interface model of reactive wetting with intermetallic formation."
 Acta Mater. 60 (2012) 3799-3814, Eqn. 30. This implementation defines the
 integral equal to zero far from the interface.
*/

#include <vector>
#include "MMSP.hpp"
#include "parabola625.c"

/* Representation includes ten field variables:
 *
 * X0. molar fraction of Cr + Mo
 * X1. molar fraction of Nb
 *
 * P2. phase fraction of delta
 * P3. phase fraction of Laves
 *
 * C4. Cr molar fraction in pure gamma
 * C5. Nb molar fraction in pure gamma
 *
 * C6. Cr molar fraction in pure delta
 * C7. Nb molar fraction in pure delta
 *
 * C8. Cr molar fraction in pure Laves
 * C9. Nb molar fraction in pure Laves
 */

// Coefficients
double coeffB(double xah, double xbh, double xch, double xal, double xbl, double xcl)
{
	return (xch*xal - xcl*xah)/(xch*xbl - xcl*xbh);
}

double coeffC(double xah, double xbh, double xch, double xal, double xbl, double xcl)
{
	return (xah*xbl - xal*xbh)/(xch*xbl - xcl*xbh);
}

int main(int argc, char* argv[])
{
	// command line error check
	if (argc < 2) {
		std::cout << "Usage: " << argv[0] << " [--help] infile [outfile]\n\n";
		exit(-1);
	}

	// help diagnostic
	if (std::string(argv[1]) == "--help") {
		std::cout << argv[0] << ": convert MMSP grid data to (p,c) points.\n";
		std::cout << "Usage: " << argv[0] << " [--help] infile [outfile]\n\n";
		std::cout << "Questions/comments to trevor.keller@gmail.com (Trevor Keller).\n\n";
		exit(0);
	}

	// file open error check
	std::ifstream input(argv[1]);
	if (!input) {
		std::cerr << "File input error: could not open " << argv[1] << ".\n\n";
		exit(-1);
	}

	// generate output file name
	std::stringstream filename;
	if (argc < 3)
		filename << std::string(argv[1]).substr(0, std::string(argv[1]).find_last_of(".")) << ".xy";
	else
		filename << argv[2];

	// file open error check
	std::ofstream output(filename.str().c_str());
	if (!output) {
		std::cerr << "File output error: could not open ";
		std::cerr << filename.str() << "." << std::endl;
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

	const double inVm = 1.0 / 1.0e-5;

	MMSP::vector<double> gamma(3, 0.0);

	if (vector_type && fields>1) {
		if (float_type) {
			if (dim == 1) {
				MMSP::grid<1, MMSP::vector<float> > GRID(argv[1]);

				const int N = MMSP::nodes(GRID);
				const double dz = MMSP::dx(GRID, 0);

				// Far-field composition (high side)
				const double xah = GRID(0)[0];
				const double xbh = GRID(0)[1];
				const double xch = 1.0 - xah - xbh;

				// Far-field composition (low side)
				const double xal = GRID(N-1)[0];
				const double xbl = GRID(N-1)[1];
				const double xcl = 1.0 - xal - xbl;

				for (int n = 0; n < N; n++) {
					double xa = GRID(n)[0];
					double xb = GRID(n)[1];
					double xc = 1.0 - xa - xb;

					// Adsorption of A
					gamma[0] += inVm * (xa - coeffB(xah, xbh, xch, xal, xbl, xcl) * xb
					                       - coeffC(xah, xbh, xch, xal, xbl, xcl) * xc) * dz;

					// Adsorption of B (xa->xz, xb->xa, xz->xb)
					gamma[1] += inVm * (xb - coeffB(xbh, xah, xch, xbl, xal, xcl) * xa
					                       - coeffC(xbh, xah, xch, xbl, xal, xcl) * xc) * dz;

					// Adsorption of C (xa->xz, xc->xa, xz->xc
					gamma[2] += inVm * (xc - coeffB(xch, xbh, xah, xcl, xbl, xal) * xb
					                       - coeffC(xch, xbh, xah, xcl, xbl, xal) * xa) * dz;

				}
			} else {
				std::cerr<<"Adsorption equation is only defined for 1D."<<std::endl;
				exit(-1);
			}
		}
		if (double_type) {
			if (dim == 1) {
				MMSP::grid<1, MMSP::vector<double> > GRID(argv[1]);
				const int N = MMSP::nodes(GRID);
				const double dz = MMSP::dx(GRID, 0);

				// Far-field composition (high side)
				const double xah = GRID(0)[0];
				const double xbh = GRID(0)[1];
				const double xch = 1.0 - xah - xbh;

				// Far-field composition (low side)
				const double xal = GRID(N-1)[0];
				const double xbl = GRID(N-1)[1];
				const double xcl = 1.0 - xal - xbl;

				for (int n = 0; n < N; n++) {
					double xa = GRID(n)[0];
					double xb = GRID(n)[1];
					double xc = 1.0 - xa - xb;

					// Adsorption of A
					gamma[0] += inVm * (xa - coeffB(xah, xbh, xch, xal, xbl, xcl) * xb
					                       - coeffC(xah, xbh, xch, xal, xbl, xcl) * xc) * dz;

					// Adsorption of B (xa->xz, xb->xa, xz->xb)
					gamma[1] += inVm * (xb - coeffB(xbh, xah, xch, xbl, xal, xcl) * xa
					                       - coeffC(xbh, xah, xch, xbl, xal, xcl) * xc) * dz;

					// Adsorption of C (xa->xz, xc->xa, xz->xc
					gamma[2] += inVm * (xc - coeffB(xch, xbh, xah, xcl, xbl, xal) * xb
					                       - coeffC(xch, xbh, xah, xcl, xbl, xal) * xa) * dz;

				}
			} else {
				std::cerr<<"Adsorption equation is only defined for 1D."<<std::endl;
				exit(-1);
			}
		}
		if (long_double_type) {
			if (dim == 1) {
				MMSP::grid<1, MMSP::vector<long double> > GRID(argv[1]);
				const int N = MMSP::nodes(GRID);
				const double dz = MMSP::dx(GRID, 0);

				// Far-field composition (high side)
				const double xah = GRID(0)[0];
				const double xbh = GRID(0)[1];
				const double xch = 1.0 - xah - xbh;

				// Far-field composition (low side)
				const double xal = GRID(N-1)[0];
				const double xbl = GRID(N-1)[1];
				const double xcl = 1.0 - xal - xbl;

				for (int n = 0; n < N; n++) {
					double xa = GRID(n)[0];
					double xb = GRID(n)[1];
					double xc = 1.0 - xa - xb;

					// Adsorption of A
					gamma[0] += inVm * (xa - coeffB(xah, xbh, xch, xal, xbl, xcl) * xb
					                       - coeffC(xah, xbh, xch, xal, xbl, xcl) * xc) * dz;

					// Adsorption of B (xa->xz, xb->xa, xz->xb)
					gamma[1] += inVm * (xb - coeffB(xbh, xah, xch, xbl, xal, xcl) * xa
					                       - coeffC(xbh, xah, xch, xbl, xal, xcl) * xc) * dz;

					// Adsorption of C (xa->xz, xc->xa, xz->xc
					gamma[2] += inVm * (xc - coeffB(xch, xbh, xah, xcl, xbl, xal) * xb
					                       - coeffC(xch, xbh, xah, xcl, xbl, xal) * xa) * dz;

				}
			} else {
				std::cerr<<"Adsorption equation is only defined for 1D."<<std::endl;
				exit(-1);
			}
		}
	}

	output.close();

	printf("Adsorption of A (element 0) is\t%g\n", gamma[0]);
	printf("Adsorption of B (element 1) is\t%g\n", gamma[1]);
	printf("Adsorption of C ( balance ) is\t%g\n", gamma[2]);

	return 0;
}
