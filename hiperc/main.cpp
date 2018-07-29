/*************************************************************************************
 * File: main.cpp                                                                    *
 *                                                                                   *
 * Questions/comments to trevor.keller@nist.gov (Trevor Keller, Ph.D.)               *
 *                                                                                   *
 * This software was developed at the National Institute of Standards and Technology *
 * by employees of the Federal Government in the course of their official duties.    *
 * Pursuant to title 17 section 105 of the United States Code this software is not   *
 * subject to copyright protection and is in the public domain. NIST assumes no      *
 * responsibility whatsoever for the use of this code by other parties, and makes no *
 * guarantees, expressed or implied, about its quality, reliability, or any other    *
 * characteristic. We would appreciate acknowledgement if the software is used.      *
 *                                                                                   *
 * This software can be redistributed and/or modified freely provided that any       *
 * derivative works bear some notice that they are derived from it, and any modified *
 * versions bear some notice that they have been modified. Derivative works that     *
 * include MMSP or other software licensed under the GPL may be subject to the GPL.  *
 *************************************************************************************/

#ifndef __MMSP_MAIN_CPP__
#define __MMSP_MAIN_CPP__
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cctype>
#include <time.h>

int main(int argc, char* argv[])
{
	MMSP::Init(argc, argv);

	// check argument list
	if (argc < 2) {
		std::cout << PROGRAM << ": bad argument list.  Use\n\n";
		std::cout << "    " << PROGRAM << " --help\n\n";
		std::cout << "to generate help message.\n\n";
		MMSP::Abort(-1);
	}

	// Many of the example programs call rand(). srand() must be called
	// _exactly once_, making this the proper place for it.
	int rank = 0;
	#ifdef MPI_VERSION
	rank = MPI::COMM_WORLD.Get_rank();
	#endif
	srand(time(NULL)+rank);

	// print help message and exit
	if (std::string(argv[1]) == std::string("--help")) {
		std::cout << PROGRAM << ": " << MESSAGE << "\n\n";
		std::cout << "Valid command lines have the form:\n\n";
		std::cout << "    " << PROGRAM << " ";
		std::cout << "[--help] [--example dimension [outfile]] [infile [outfile] steps [increment]]\n\n";
		std::cout << "A few examples of using the command line follow.\n\n";
		std::cout << "The command\n\n";
		std::cout << "    " << PROGRAM << " --help\n\n";
		std::cout << "generates this help message and exits.  ";
		std::cout << "The \"--example\" option can be used to gen-\nerate a relevant test grid, e.g.\n\n";
		std::cout << "    " << PROGRAM << " --example 3\n\n";
		std::cout << "generates an example test problem on a grid of dimension 3 and writes this to the \n";
		std::cout << "file named \"example\", while\n\n";
		std::cout << "    " << PROGRAM << " --example 2 start\n\n";
		std::cout << "generates an example test problem on a grid of dimension 2 and writes this to the \n";
		std::cout << "file named \"start\".\n\n";
		std::cout << "    " << PROGRAM << " start 1000\n\n";
		std::cout << "reads the grid contained within \"start\" and runs a simulation for 1000 time steps.\n";
		std::cout << "The final grid is written to a file named \"start.1000\".\n\n";
		std::cout << "    " << PROGRAM << " start final 1000\n\n";
		std::cout << "reads the grid contained within \"start\" and runs a simulation for 1000 time steps.\n";
		std::cout << "The final grid is written to a file named \"final.1000\".\n\n";
		std::cout << "    " << PROGRAM << " start 1000 100\n\n";
		std::cout << "reads the grid contained within \"start\" and runs a simulation for 1000 time steps.\n";
		std::cout << "The grid is then written to a file every 100 time steps.  ";
		std::cout << "The resulting files are \nnamed \"start.0100\", \"start.0200\", ... \"start.1000\".\n\n";
		std::cout << "    " << PROGRAM << " start final 1000 100\n\n";
		std::cout << "reads the grid contained within \"start\" and runs a simulation for 1000 time steps.\n";
		std::cout << "The grid is then written to a file every 100 time steps.  ";
		std::cout << "The resulting files are \nnamed \"final.0100\", \"final.0200\", ... \"final.1000\".\n\n";
		exit(0);
	}

	// generate example grid
	else if (std::string(argv[1]) == std::string("--example")) {
		// check argument list
		if (argc<3 or argc>4) {
			std::cout << PROGRAM << ": bad argument list.  Use\n\n";
			std::cout << "    " << PROGRAM << " --help\n\n";
			std::cout << "to generate help message.\n\n";
			MMSP::Abort(-1);
		}

		// check problem dimension
		if (std::string(argv[2]).find_first_not_of("0123456789") != std::string::npos) {
			std::cout << PROGRAM << ": example grid must have integral dimension.  Use\n\n";
			std::cout << "    " << PROGRAM << " --help\n\n";
			std::cout << "to generate help message.\n\n";
			MMSP::Abort(-1);
		}

		int dim = atoi(argv[2]);

		// set output file name
		std::string outfile;
		if (argc < 4) outfile = "example";
		else outfile = argv[3];

		char* filename = new char[outfile.length()+1];
		for (unsigned int i=0; i<outfile.length(); i++)
			filename[i] = outfile[i];
		filename[outfile.length()]='\0';

		// generate test problem
		MMSP::generate(dim, filename);

		delete [] filename;
	}

	// run simulation
	else {
		// bad argument list
		if (argc<3 or argc>5) {
			std::cout << PROGRAM << ": bad argument list.  Use\n\n";
			std::cout << "    " << PROGRAM << " --help\n\n";
			std::cout << "to generate help message.\n\n";
			MMSP::Abort(-1);
		}

		int steps;
		int increment;
		std::string outfile;

		if (std::string(argv[2]).find_first_not_of("0123456789") == std::string::npos) {
			// set output file name
			outfile = argv[1];

			// must have integral number of time steps
			if (std::string(argv[2]).find_first_not_of("0123456789") != std::string::npos) {
				std::cout << PROGRAM << ": number of time steps must have integral value.  Use\n\n";
				std::cout << "    " << PROGRAM << " --help\n\n";
				std::cout << "to generate help message.\n\n";
				MMSP::Abort(-1);
			}

			steps = atoi(argv[2]);
			increment = steps;

			if (argc > 3) {
				// must have integral output increment
				if (std::string(argv[3]).find_first_not_of("0123456789") != std::string::npos) {
					std::cout << PROGRAM << ": output increment must have integral value.  Use\n\n";
					std::cout << "    " << PROGRAM << " --help\n\n";
					std::cout << "to generate help message.\n\n";
					MMSP::Abort(-1);
				}

				increment = atoi(argv[3]);

				// output increment must be smaller than number of steps
				if (increment > steps) {
					std::cout << PROGRAM << ": output increment must be smaller than number of time steps.  Use\n\n";
					std::cout << "    " << PROGRAM << " --help\n\n";
					std::cout << "to generate help message.\n\n";
					MMSP::Abort(-1);
				}
			}
		}

		else {
			// set output file name
			outfile = argv[2];

			// set number of time steps
			if (std::string(argv[3]).find_first_not_of("0123456789") != std::string::npos) {
				// must have integral number of time steps
				std::cout << PROGRAM << ": number of time steps must have integral value.  Use\n\n";
				std::cout << "    " << PROGRAM << " --help\n\n";
				std::cout << "to generate help message.\n\n";
				MMSP::Abort(-1);
			}

			steps = atoi(argv[3]);
			increment = steps;

			if (argc > 4) {
				// must have integral output increment
				if (std::string(argv[4]).find_first_not_of("0123456789") != std::string::npos) {
					std::cout << PROGRAM << ": output increment must have integral value.  Use\n\n";
					std::cout << "    " << PROGRAM << " --help\n\n";
					std::cout << "to generate help message.\n\n";
					MMSP::Abort(-1);
				}

				increment = atoi(argv[4]);

				// output increment must be smaller than number of steps
				if (increment > steps) {
					std::cout << PROGRAM << ": output increment must be smaller than number of time steps.  Use\n\n";
					std::cout << "    " << PROGRAM << " --help\n\n";
					std::cout << "to generate help message.\n\n";
					MMSP::Abort(-1);
				}
			}
		}

		// file open error check
		std::ifstream input(argv[1]);
		if (!input) {
			std::cerr << "File input error: could not open " << argv[1] << ".\n\n";
			MMSP::Abort(-1);
		}

		// read data type
		std::string type;
		getline(input, type, '\n');

		// grid type error check
		if (type.substr(0, 4) != "grid") {
			std::cerr << "File input error: file does not contain grid data." << std::endl;
			MMSP::Abort(-1);
		}

		// read grid dimension
		int dim;
		input >> dim;
		input.close();

		// set output file basename
		int iterations_start(0);
		if (outfile.find_first_of(".") != outfile.find_last_of(".")) {
			std::string number = outfile.substr(outfile.find_first_of(".") + 1, outfile.find_last_of(".") - 1);
			iterations_start = atoi(number.c_str());
		}
		std::string base;
		if (outfile.find(".", outfile.find_first_of(".") + 1) == std::string::npos) // only one dot found
			base = outfile.substr(0, outfile.find_last_of(".")) + ".";
		else {
			int last_dot = outfile.find_last_of(".");
			int prev_dot = outfile.rfind('.', last_dot - 1);
			std::string number = outfile.substr(prev_dot + 1, last_dot - prev_dot - 1);
			bool isNumeric(true);
			for (unsigned int i = 0; i < number.size(); ++i) {
				if (!isdigit(number[i])) isNumeric = false;
			}
			if (isNumeric)
				base = outfile.substr(0, outfile.rfind(".", outfile.find_last_of(".") - 1)) + ".";
			else base = outfile.substr(0, outfile.find_last_of(".")) + ".";
		}

		// set output file suffix
		std::string suffix = "";
		if (outfile.find_last_of(".") != std::string::npos)
			suffix = outfile.substr(outfile.find_last_of("."), std::string::npos);

		// set output filename length
		int length = base.length() + suffix.length();
		if (1) {
			std::stringstream slength;
			slength << steps;
			length += slength.str().length();
		}

		if (dim == 2) {
			// construct grid objects
			GRID2D grid(argv[1]);

			/* declare host and device data structures */
			struct HostData host;
			struct CudaData dev;

			/* declare default materials and numerical parameters */
			int bx=32, by=32, nm=3, code=53;
			param_parser(&bx, &by, &code, &nm);
			const int nx = g1(grid, 0) - g0(grid, 0) + nm - 1;
			const int ny = g1(grid, 1) - g0(grid, 1) + nm - 1;

			/* initialize memory */
			make_arrays(&host, nx, ny, nm);
			set_mask(dx(grid, 0), dx(grid, 1), code, host.mask_lap, nm);

			for (int n = 0; n < MMSP::nodes(grid); n++) {
				MMSP::vector<fp_t>& gridN = grid(n);
				MMSP::vector<int> x = MMSP::position(grid, n);
				const int i = x[0] - MMSP::g0(grid, 0) + nm / 2;
				const int j = x[1] - MMSP::g0(grid, 1) + nm / 2;
				host.conc_Cr_old[j][i] = gridN[0];
				host.conc_Nb_old[j][i] = gridN[1];
				host.phi_del_old[j][i] = gridN[2];
				host.phi_lav_old[j][i] = gridN[3];
				host.gam_Cr_old[j][i]  = gridN[4];
				host.gam_Nb_old[j][i]  = gridN[5];
			}

			/* initialize GPU */
			init_cuda(&host, nx, ny, nm, &dev);

			const double dtTransformLimited = (meshres*meshres) / (2.0 * dim * Lmob[0]*kappa[0]);
			const double dtDiffusionLimited = (meshres*meshres) / (2.0 * dim * std::max(D_Cr[0], D_Nb[1]));
			const double dt = LinStab * std::min(dtTransformLimited, dtDiffusionLimited);

			// perform computation
			for (int i = iterations_start; i < steps; i += increment) {
				for (int j = 0; j < increment; j++) {
					print_progress(j, increment);
					/* === Start Architecture-Specific Kernel === */
					device_boundaries(&dev, nx, ny, nm, bx, by);

					device_laplacian(&dev, nx, ny, nm, bx, by);

					device_fict_boundaries(&dev, nx, ny, nm, bx, by);

					device_evolution(&dev, nx, ny, nm, bx, by,
					                 D_Cr[0], D_Cr[1], D_Nb[0], D_Nb[1],
					                 alpha, kappa[0], omega[0],
					                 Lmob[0], Lmob[1], dt);

					swap_pointers_1D(&(dev.conc_Cr_old), &(dev.conc_Cr_new));
					swap_pointers_1D(&(dev.conc_Nb_old), &(dev.conc_Nb_new));
					swap_pointers_1D(&(dev.phi_del_old), &(dev.phi_del_new));
					swap_pointers_1D(&(dev.phi_lav_old), &(dev.phi_lav_new));
					swap_pointers_1D(&(dev.gam_Cr_old),  &(dev.gam_Cr_new));
					swap_pointers_1D(&(dev.gam_Nb_old),  &(dev.gam_Nb_new));
					/* === Finish Architecture-Specific Kernel === */
				}

				read_out_result(&dev, &host, nx, ny);

				// generate output filename
				std::stringstream outstr;
				int n = outstr.str().length();
				for (int j = 0; n < length; j++) {
					outstr.str("");
					outstr << base;
					for (int k = 0; k < j; k++) outstr << 0;
					outstr << i + increment << suffix;
					n = outstr.str().length();
				}

				char filename[FILENAME_MAX] = {}; // initialize null characters
				for (unsigned int j=0; j<outstr.str().length(); j++)
					filename[j]=outstr.str()[j];

				// write grid output to file
				for (int n = 0; n < MMSP::nodes(grid); n++) {
					MMSP::vector<fp_t>& gridN = grid(n);
					MMSP::vector<int> x = MMSP::position(grid, n);
					const int i = x[0] - MMSP::g0(grid, 0) + nm / 2;
					const int j = x[1] - MMSP::g0(grid, 1) + nm / 2;
					gridN[0] = host.conc_Cr_old[j][i];
					gridN[1] = host.conc_Nb_old[j][i];
					gridN[2] = host.phi_del_old[j][i];
					gridN[3] = host.phi_lav_old[j][i];
					gridN[4] = host.gam_Cr_old[j][i];
					gridN[5] = host.gam_Nb_old[j][i];
				}
				MMSP::output(grid, filename);
				print_progress(increment, increment);
			}
			free_cuda(&dev);
			free_arrays(&host);
		} else {
			std::cerr << dim << "-dimensional grids are unsupported in the CUDA version." << std::endl;
			MMSP::Abort(-1);
		}

	}

	MMSP::Finalize();
	return 0;
}

#endif
