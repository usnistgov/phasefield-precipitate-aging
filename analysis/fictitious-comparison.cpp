// fictitious-comparison.cpp

#include <string>
#include <sstream>
#include <fstream>
#include <cmath>
#include "MMSP.hpp"
#include "parabola625.c"

template<typename T>
double h(T x){return x*x*x*(6.*x*x - 15.*x + 10.);}

template<int dim, typename T>
void print_vectors(std::ofstream& fstr, const MMSP::grid<dim,MMSP::vector<T> >& GRID, double& resCr, double& resNb)
{
	if (dim==1) {
		MMSP::vector<int> x(1,0);
		for (x[0]=MMSP::x0(GRID); x[0]<MMSP::x1(GRID); x[0]++) {
			const MMSP::vector<T>& v = GRID(x);
			const T& xcr = v[0];
			const T& xnb = v[1];

			const T& del = v[2];
			const T& lav = v[3];

			const T& gCr = v[4];
			const T& gNb = v[5];

			const T& dCr = v[6];
			const T& dNb = v[7];

			const T& lCr = v[8];
			const T& lNb = v[9];

			const T fdel = h(del);
			const T flav = h(lav);
			const T fgam = 1.-fdel-flav;

			const T fgCr = fict_gam_Cr(xcr, xnb, fdel, fgam, flav);
			const T fgNb = fict_gam_Nb(xcr, xnb, fdel, fgam, flav);
			const T fdCr = fict_del_Cr(xcr, xnb, fdel, fgam, flav);
			const T fdNb = fict_del_Nb(xcr, xnb, fdel, fgam, flav);
			const T flCr = fict_lav_Cr(xcr, xnb, fdel, fgam, flav);
			const T flNb = fict_lav_Nb(xcr, xnb, fdel, fgam, flav);

			const double dxCr = (fgCr - gCr)*(fgCr - gCr)
				              + (fdCr - dCr)*(fdCr - dCr)
				              + (flCr - lCr)*(flCr - lCr);
			const double dxNb = (fgNb - gNb)*(fgNb - gNb)
				              + (fdNb - dNb)*(fdNb - dNb)
				              + (flNb - lNb)*(flNb - lNb);

			resCr += dxCr/3.;
			resNb += dxNb/3.;

			fstr << std::sqrt(dxCr/3.) << " " << std::sqrt(dxNb/3.) << " ";
		}

	} else if (dim==2) {
		MMSP::vector<int> x(2,0);
		for (x[1]=MMSP::y0(GRID); x[1]<MMSP::y1(GRID); x[1]++) {
			for (x[0]=MMSP::x0(GRID); x[0]<MMSP::x1(GRID); x[0]++) {
				const MMSP::vector<T>& v = GRID(x);
				const T& xcr = v[0];
				const T& xnb = v[1];

				const T& del = v[2];
				const T& lav = v[3];

				const T& gCr = v[4];
				const T& gNb = v[5];

				const T& dCr = v[6];
				const T& dNb = v[7];

				const T& lCr = v[8];
				const T& lNb = v[9];

				const T fdel = h(del);
				const T flav = h(lav);
				const T fgam = 1.-fdel-flav;

				const T fgCr = fict_gam_Cr(xcr, xnb, fdel, fgam, flav);
				const T fgNb = fict_gam_Nb(xcr, xnb, fdel, fgam, flav);
				const T fdCr = fict_del_Cr(xcr, xnb, fdel, fgam, flav);
				const T fdNb = fict_del_Nb(xcr, xnb, fdel, fgam, flav);
				const T flCr = fict_lav_Cr(xcr, xnb, fdel, fgam, flav);
				const T flNb = fict_lav_Nb(xcr, xnb, fdel, fgam, flav);

				const double dxCr = (fgCr - gCr)*(fgCr - gCr)
					              + (fdCr - dCr)*(fdCr - dCr)
					              + (flCr - lCr)*(flCr - lCr);
				const double dxNb = (fgNb - gNb)*(fgNb - gNb)
					              + (fdNb - dNb)*(fdNb - dNb)
					              + (flNb - lNb)*(flNb - lNb);

				resCr += dxCr/3.;
				resNb += dxNb/3.;

				fstr << std::sqrt(dxCr/3.) << " " << std::sqrt(dxNb/3.) << " ";
			}
		}

	} else if (dim==3) {
		MMSP::vector<int> x(3,0);
		for (x[2]=MMSP::z0(GRID); x[2]<MMSP::z1(GRID); x[2]++) {
			for (x[1]=MMSP::y0(GRID); x[1]<MMSP::y1(GRID); x[1]++) {
				for (x[0]=MMSP::x0(GRID); x[0]<MMSP::x1(GRID); x[0]++) {
					const MMSP::vector<T>& v = GRID(x);
					const T& xcr = v[0];
					const T& xnb = v[1];

					const T& del = v[2];
					const T& lav = v[3];

					const T& gCr = v[4];
					const T& gNb = v[5];

					const T& dCr = v[6];
					const T& dNb = v[7];

					const T& lCr = v[8];
					const T& lNb = v[9];

					const T fdel = h(del);
					const T flav = h(lav);
					const T fgam = 1.-fdel-flav;

					const T fgCr = fict_gam_Cr(xcr, xnb, fdel, fgam, flav);
					const T fgNb = fict_gam_Nb(xcr, xnb, fdel, fgam, flav);
					const T fdCr = fict_del_Cr(xcr, xnb, fdel, fgam, flav);
					const T fdNb = fict_del_Nb(xcr, xnb, fdel, fgam, flav);
					const T flCr = fict_lav_Cr(xcr, xnb, fdel, fgam, flav);
					const T flNb = fict_lav_Nb(xcr, xnb, fdel, fgam, flav);

					const double dxCr = (fgCr - gCr)*(fgCr - gCr)
						              + (fdCr - dCr)*(fdCr - dCr)
						              + (flCr - lCr)*(flCr - lCr);
					const double dxNb = (fgNb - gNb)*(fgNb - gNb)
						              + (fdNb - dNb)*(fdNb - dNb)
						              + (flNb - lNb)*(flNb - lNb);

					resCr += dxCr / 3.;
					resNb += dxNb / 3.;

					fstr << std::sqrt(dxCr/3.) << " " << std::sqrt(dxNb/3.) << " ";
				}
			}
		}
	}
}

int main(int argc, char* argv[])
{
	// command line error check
	if (argc < 2) {
		std::cout << "Usage: " << argv[0] << " [--help] [--mag|--max] infile [outfile]\n";
		std::exit(-1);
	}

	// help diagnostic
	if (std::string(argv[1]) == "--help") {
		std::cout << argv[0] << ": convert MMSP grid data to VTK image data format.\n";
		std::cout << "Usage: " << argv[0] << " [--help] [--mag|--max|--field=n] infile [outfile]\n";
		std::cout << "       Select either --mag or --max to flatten vector or sparse data by the specified method.\n";
		std::cout << "Questions/comments to gruberja@gmail.com (Jason Gruber).\n";
		std::exit(0);
	}

	int fileindex = 1; // in typical usage, filename comes immediately after executable

	// file open error check
	std::ifstream input(argv[fileindex]);
	if (!input) {
		std::cerr << "File input error: could not open " << argv[fileindex] << ".\n";
		exit(-1);
	}

	// generate output file name
	std::stringstream filename;
	if (argc < 3)
		filename << std::string(argv[fileindex]).substr(0, std::string(argv[fileindex]).find_last_of(".")) << ".vti";
	else
		filename << argv[fileindex+1];

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

	bool vector_type = (type.find("vector") != std::string::npos);

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
	if (dim < 1 or dim > 3) {
		std::cerr << "File input error: grid dimension must be 1, 2, or 3." << std::endl;
		exit(-1);
	}

	// read number of fields
	int fields, outfields=2;
	input >> fields;

	// read grid sizes
	int x0[3] = {0, 0, 0};
	int x1[3] = {0, 0, 0};
	for (int i = 0; i < dim; i++)
		input >> x0[i] >> x1[i];

	// read cell spacing
	float dx[3] = {1.0, 1.0, 1.0};
	for (int i = 0; i < dim; i++)
		input >> dx[i];

	// ignore trailing endlines
	input.ignore(10, '\n');


	// determine byte order: 01 AND 01 = 01; 01 AND 10 = 00.
	std::string byte_order;
	if (0x01 & static_cast<int>(1)) byte_order = "LittleEndian";
	else byte_order = "BigEndian";

	// output header markup
	output << "<?xml version=\"1.0\"?>\n";
	output << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"" << byte_order << "\">\n";

	if (dim == 1) {
		output << "  <ImageData WholeExtent=\"" << x0[0] << " " << x1[0] << " 0 0 0 0\"";
		output << "   Origin=\"0 0 0\" Spacing=\"" << dx[0] << " 1 1\">\n";
	} else if (dim == 2) {
		output << "  <ImageData WholeExtent=\"" << x0[0] << " " << x1[0] << " "
		                                        << x0[1] << " " << x1[1]
		                                        << " 0 0\"";
		output << "   Origin=\"0 0 0\" Spacing=\"" << dx[0] << " " << dx[1] << " 1\">\n";
	} else if (dim == 3) {
		output << "  <ImageData WholeExtent=\"" << x0[0] << " " << x1[0] << " "
		                                        << x0[1] << " " << x1[1] << " "
		                                        << x0[2] << " " << x1[2] << "\"";
		output << "   Origin=\"0 0 0\" Spacing=\"" << dx[0] << " " << dx[1] << " " << dx[2] << "\">\n";
	} else {
		std::cerr<<"Error: "<<dim<<"-dimensional data not supported."<<std::endl;
		std::exit(-1);
	}

	double resCr = 0.0, resNb = 0.0, N = 0;

	// read number of blocks
	int blocks;
	input.read(reinterpret_cast<char*>(&blocks), sizeof(blocks));

	for (int i = 0; i < blocks; i++) {
		// read block limits
		int lmin[3] = {0, 0, 0};
		int lmax[3] = {0, 0, 0};
		for (int j = 0; j < dim; j++) {
			input.read(reinterpret_cast<char*>(&lmin[j]), sizeof(lmin[j]));
			input.read(reinterpret_cast<char*>(&lmax[j]), sizeof(lmax[j]));
		}
		int blo[3];
		int bhi[3];
		// read boundary conditions
		for (int j = 0; j < dim; j++) {
			input.read(reinterpret_cast<char*>(&blo[j]), sizeof(blo[j]));
			input.read(reinterpret_cast<char*>(&bhi[j]), sizeof(bhi[j]));
		}

		// write header markup
		if (dim == 1)
			output << "    <Piece Extent=\"" << lmin[0] << " " << lmax[0] << " 0 0 0 0\">\n";
		if (dim == 2)
			output << "    <Piece Extent=\"" << lmin[0] << " " << lmax[0] << " "
			                                 << lmin[1] << " " << lmax[1] << " 0 0\">\n";
		if (dim == 3)
			output << "    <Piece Extent=\"" << lmin[0] << " " << lmax[0] << " "
			                                 << lmin[1] << " " << lmax[1] << " "
			                                 << lmin[2] << " " << lmax[2] << "\">\n";

		// write cell data markup
		if (vector_type) {
			output << "      <CellData>\n";
			output << "        <DataArray Name=\"vector_data\" NumberOfComponents=\"" << outfields << "\"";
		}
		else { /* built-in data types */
			output << "      <CellData>\n";
			output << "        <DataArray Name=\"scalar_data\"";
		}

		if (bool_type)
			output << " type=\"UInt8\" format=\"ascii\">\n";
		else if (char_type)
			output << " type=\"Int8\" format=\"ascii\">\n";
		else if (unsigned_char_type)
			output << " type=\"UInt8\" format=\"ascii\">\n";
		else if (int_type)
			output << " type=\"Int32\" format=\"ascii\">\n";
		else if (unsigned_int_type)
			output << " type=\"UInt32\" format=\"ascii\">\n";
		else if (long_type)
			output << " type=\"Int32\" format=\"ascii\">\n";
		else if (unsigned_long_type)
			output << " type=\"UInt32\" format=\"ascii\">\n";
		else if (short_type)
			output << " type=\"Int16\" format=\"ascii\">\n";
		else if (unsigned_short_type)
			output << " type=\"UInt16\" format=\"ascii\">\n";
		else if (float_type)
			output << " type=\"Float32\" format=\"ascii\">\n";
		else if (double_type)
			output << " type=\"Float64\" format=\"ascii\">\n";
		else if (long_double_type)
		   	output << " type=\"Float128\" format=\"ascii\">\n";


		// read grid data
		unsigned long size, rawSize;
		input.read(reinterpret_cast<char*>(&rawSize), sizeof(rawSize)); // read raw size
		input.read(reinterpret_cast<char*>(&size), sizeof(size)); // read compressed size
		char* compressed_buffer = new char[size];
		input.read(compressed_buffer, size);
		char* buffer = NULL;
		if (size != rawSize) {
			// Decompress data
			buffer = new char[rawSize];
			int status;
			status = uncompress(reinterpret_cast<unsigned char*>(buffer), &rawSize, reinterpret_cast<unsigned char*>(compressed_buffer), size);
			switch(status) {
			case Z_OK:
				break;
			case Z_MEM_ERROR:
				std::cerr << "Uncompress: out of memory." << std::endl;
				exit(1);
				break;
			case Z_BUF_ERROR:
				std::cerr << "Uncompress: output buffer wasn't large enough." << std::endl;
				exit(1);
				break;
			}
			delete [] compressed_buffer;
			compressed_buffer = NULL;
		} else {
			buffer = compressed_buffer;
			compressed_buffer = NULL;
		}

		// write grid data
		if (vector_type) {
			if (bool_type) {
				if (dim == 1) {
					MMSP::grid<1, MMSP::vector<bool> > GRID(fields, lmin, lmax);
					GRID.from_buffer(buffer);
					print_vectors(output, GRID, resCr, resNb);
					N = MMSP::nodes(GRID);
				} else if (dim == 2) {
					MMSP::grid<2, MMSP::vector<bool> > GRID(fields, lmin, lmax);
					GRID.from_buffer(buffer);
					print_vectors(output, GRID, resCr, resNb);
					N = MMSP::nodes(GRID);
				} else if (dim == 3) {
					MMSP::grid<3, MMSP::vector<bool> > GRID(fields, lmin, lmax);
					GRID.from_buffer(buffer);
					print_vectors(output, GRID, resCr, resNb);
					N = MMSP::nodes(GRID);
				}
			}
			else if (unsigned_char_type) {
				if (dim == 1) {
					MMSP::grid<1, MMSP::vector<unsigned char> > GRID(fields, lmin, lmax);
					GRID.from_buffer(buffer);
					N = MMSP::nodes(GRID);
					print_vectors(output, GRID, resCr, resNb);
				} else if (dim == 2) {
					MMSP::grid<2, MMSP::vector<unsigned char> > GRID(fields, lmin, lmax);
					GRID.from_buffer(buffer);
					print_vectors(output, GRID, resCr, resNb);
					N = MMSP::nodes(GRID);
				} else if (dim == 3) {
					MMSP::grid<3, MMSP::vector<unsigned char> > GRID(fields, lmin, lmax);
					GRID.from_buffer(buffer);
					print_vectors(output, GRID, resCr, resNb);
					N = MMSP::nodes(GRID);
				}
			}
			else if (char_type) {
				if (dim == 1) {
					MMSP::grid<1, MMSP::vector<char> > GRID(fields, lmin, lmax);
					GRID.from_buffer(buffer);
					print_vectors(output, GRID, resCr, resNb);
					N = MMSP::nodes(GRID);
				} else if (dim == 2) {
					MMSP::grid<2, MMSP::vector<char> > GRID(fields, lmin, lmax);
					GRID.from_buffer(buffer);
					print_vectors(output, GRID, resCr, resNb);
					N = MMSP::nodes(GRID);
				} else if (dim == 3) {
					MMSP::grid<3, MMSP::vector<char> > GRID(fields, lmin, lmax);
					GRID.from_buffer(buffer);
					print_vectors(output, GRID, resCr, resNb);
					N = MMSP::nodes(GRID);
				}
			}
			else if (unsigned_int_type) {
				if (dim == 1) {
					MMSP::grid<1, MMSP::vector<unsigned int> > GRID(fields, lmin, lmax);
					GRID.from_buffer(buffer);
					print_vectors(output, GRID, resCr, resNb);
					N = MMSP::nodes(GRID);
				} else if (dim == 2) {
					MMSP::grid<2, MMSP::vector<unsigned int> > GRID(fields, lmin, lmax);
					GRID.from_buffer(buffer);
					print_vectors(output, GRID, resCr, resNb);
					N = MMSP::nodes(GRID);
				} else if (dim == 3) {
					MMSP::grid<3, MMSP::vector<unsigned int> > GRID(fields, lmin, lmax);
					GRID.from_buffer(buffer);
					print_vectors(output, GRID, resCr, resNb);
					N = MMSP::nodes(GRID);
				}
			}
			else if (int_type) {
				if (dim == 1) {
					MMSP::grid<1, MMSP::vector<int> > GRID(fields, lmin, lmax);
					GRID.from_buffer(buffer);
					print_vectors(output, GRID, resCr, resNb);
					N = MMSP::nodes(GRID);
				} else if (dim == 2) {
					MMSP::grid<2, MMSP::vector<int> > GRID(fields, lmin, lmax);
					GRID.from_buffer(buffer);
					print_vectors(output, GRID, resCr, resNb);
					N = MMSP::nodes(GRID);
				} else if (dim == 3) {
					MMSP::grid<3, MMSP::vector<int> > GRID(fields, lmin, lmax);
					GRID.from_buffer(buffer);
					print_vectors(output, GRID, resCr, resNb);
					N = MMSP::nodes(GRID);
				}
			}
			else if (unsigned_long_type) {
				if (dim == 1) {
					MMSP::grid<1, MMSP::vector<unsigned long> > GRID(fields, lmin, lmax);
					GRID.from_buffer(buffer);
					print_vectors(output, GRID, resCr, resNb);
					N = MMSP::nodes(GRID);
				} else if (dim == 2) {
					MMSP::grid<2, MMSP::vector<unsigned long> > GRID(fields, lmin, lmax);
					GRID.from_buffer(buffer);
					print_vectors(output, GRID, resCr, resNb);
					N = MMSP::nodes(GRID);
				} else if (dim == 3) {
					MMSP::grid<3, MMSP::vector<unsigned long> > GRID(fields, lmin, lmax);
					GRID.from_buffer(buffer);
					print_vectors(output, GRID, resCr, resNb);
					N = MMSP::nodes(GRID);
				}
			}
			else if (long_type) {
				if (dim == 1) {
					MMSP::grid<1, MMSP::vector<long> > GRID(fields, lmin, lmax);
					GRID.from_buffer(buffer);
					print_vectors(output, GRID, resCr, resNb);
					N = MMSP::nodes(GRID);
				} else if (dim == 2) {
					MMSP::grid<2, MMSP::vector<long> > GRID(fields, lmin, lmax);
					GRID.from_buffer(buffer);
					print_vectors(output, GRID, resCr, resNb);
					N = MMSP::nodes(GRID);
				} else if (dim == 3) {
					MMSP::grid<3, MMSP::vector<long> > GRID(fields, lmin, lmax);
					GRID.from_buffer(buffer);
					print_vectors(output, GRID, resCr, resNb);
					N = MMSP::nodes(GRID);
				}
			}
			else if (unsigned_short_type) {
				if (dim == 1) {
					MMSP::grid<1, MMSP::vector<unsigned short> > GRID(fields, lmin, lmax);
					GRID.from_buffer(buffer);
					print_vectors(output, GRID, resCr, resNb);
					N = MMSP::nodes(GRID);
				} else if (dim == 2) {
					MMSP::grid<2, MMSP::vector<unsigned short> > GRID(fields, lmin, lmax);
					GRID.from_buffer(buffer);
					print_vectors(output, GRID, resCr, resNb);
					N = MMSP::nodes(GRID);
				} else if (dim == 3) {
					MMSP::grid<3, MMSP::vector<unsigned short> > GRID(fields, lmin, lmax);
					GRID.from_buffer(buffer);
					print_vectors(output, GRID, resCr, resNb);
					N = MMSP::nodes(GRID);
				}
			}
			else if (short_type) {
				if (dim == 1) {
					MMSP::grid<1, MMSP::vector<short> > GRID(fields, lmin, lmax);
					GRID.from_buffer(buffer);
					print_vectors(output, GRID, resCr, resNb);
					N = MMSP::nodes(GRID);
				} else if (dim == 2) {
					MMSP::grid<2, MMSP::vector<short> > GRID(fields, lmin, lmax);
					GRID.from_buffer(buffer);
					print_vectors(output, GRID, resCr, resNb);
					N = MMSP::nodes(GRID);
				} else if (dim == 3) {
					MMSP::grid<3, MMSP::vector<short> > GRID(fields, lmin, lmax);
					GRID.from_buffer(buffer);
					print_vectors(output, GRID, resCr, resNb);
					N = MMSP::nodes(GRID);
				}
			}
			else if (float_type) {
				if (dim == 1) {
					MMSP::grid<1, MMSP::vector<float> > GRID(fields, lmin, lmax);
					GRID.from_buffer(buffer);
					print_vectors(output, GRID, resCr, resNb);
					N = MMSP::nodes(GRID);
				} else if (dim == 2) {
					MMSP::grid<2, MMSP::vector<float> > GRID(fields, lmin, lmax);
					GRID.from_buffer(buffer);
					print_vectors(output, GRID, resCr, resNb);
					N = MMSP::nodes(GRID);
				} else if (dim == 3) {
					MMSP::grid<3, MMSP::vector<float> > GRID(fields, lmin, lmax);
					GRID.from_buffer(buffer);
					print_vectors(output, GRID, resCr, resNb);
					N = MMSP::nodes(GRID);
				}
			}
			else if (long_double_type) {
				if (dim == 1) {
					MMSP::grid<1, MMSP::vector<long double> > GRID(fields, lmin, lmax);
					GRID.from_buffer(buffer);
					print_vectors(output, GRID, resCr, resNb);
					N = MMSP::nodes(GRID);
				} else if (dim == 2) {
					MMSP::grid<2, MMSP::vector<long double> > GRID(fields, lmin, lmax);
					GRID.from_buffer(buffer);
					print_vectors(output, GRID, resCr, resNb);
					N = MMSP::nodes(GRID);
				} else if (dim == 3) {
					MMSP::grid<3, MMSP::vector<long double> > GRID(fields, lmin, lmax);
					GRID.from_buffer(buffer);
					print_vectors(output, GRID, resCr, resNb);
					N = MMSP::nodes(GRID);
				}
			}
			else if (double_type) {
				if (dim == 1) {
					MMSP::grid<1, MMSP::vector<double> > GRID(fields, lmin, lmax);
					GRID.from_buffer(buffer);
					print_vectors(output, GRID, resCr, resNb);
					N = MMSP::nodes(GRID);
				} else if (dim == 2) {
					MMSP::grid<2, MMSP::vector<double> > GRID(fields, lmin, lmax);
					GRID.from_buffer(buffer);
					print_vectors(output, GRID, resCr, resNb);
					N = MMSP::nodes(GRID);
				} else if (dim == 3) {
					MMSP::grid<3, MMSP::vector<double> > GRID(fields, lmin, lmax);
					GRID.from_buffer(buffer);
					print_vectors(output, GRID, resCr, resNb);
					N = MMSP::nodes(GRID);
				}
			}
		}

		// clean up
		delete [] buffer;

		// write closing markup
		output << "\n";
		output << "        </DataArray>\n";
		output << "      </CellData>\n";
		output << "    </Piece>\n";
	}

	// output closing markup
	output << "  </ImageData>\n";
	output << "</VTKFile>\n";

	resCr = std::sqrt(resCr / N);
	resNb = std::sqrt(resNb / N);

	printf("%4.12f\t %4.12f\n", resCr, resNb);

	return 0;
}
