#ifndef __MMSP_MAIN_CPP__
#define __MMSP_MAIN_CPP__

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <time.h>

int main(int argc, char* argv[])
{
	MMSP::Init(argc, argv);
	const std::string PROGRAM(argv[0]);

	// check argument list
	if (argc < 2) {
		std::cout << PROGRAM << ": bad argument list.  Use\n\n";
		std::cout << "    " << PROGRAM << " --help\n\n";
		std::cout << "to generate help message.\n\n";
		MMSP::Abort(EXIT_FAILURE);
	}

	// srand() must be called exactly once to seed rand().
	int rank = 0;
	#ifdef MPI_VERSION
	rank = MPI::COMM_WORLD.Get_rank();
	#endif
	srand(time(NULL)+rank);

	// print help message and exit
	if (std::string(argv[1]) == std::string("--help")) {
		if (rank == 0) {
			std::cout << PROGRAM << ": " << MESSAGE << "\n\n";
			std::cout << "Valid command lines have the form:\n\n";
			std::cout << "    " << PROGRAM << " ";
			std::cout << "[--help] [--init dimension [outfile]] [infile [outfile] runtime [increment]]\n\n";
			std::cout << "A few examples of using the command line follow.\n\n";
			std::cout << "The command\n\n";
			std::cout << "    " << PROGRAM << " --help\n\n";
			std::cout << "generates this help message and exits.  ";
			std::cout << "The \"--init\" option can be used to generate a relevant test grid, e.g.\n\n";
			std::cout << "    " << PROGRAM << " --init 2\n\n";
			std::cout << "generates an initial condition on a grid of dimension 3 and writes this to the \n";
			std::cout << "file named \"init\", while\n\n";
			std::cout << "    " << PROGRAM << " --init 2 start.dat\n\n";
			std::cout << "generates an initial condition on a grid of dimension 2 and writes this to the \n";
			std::cout << "file named \"start.dat\".\n\n";
			std::cout << "    " << PROGRAM << " start.dat 1000\n\n";
			std::cout << "reads the grid contained within \"start.dat\" and runs a simulation for 1000 seconds.\n";
			std::cout << "The final grid is written to a file named \"start.1000.0000.dat\".\n\n";
			std::cout << "    " << PROGRAM << " start.dat 1000 100\n\n";
			std::cout << "reads the grid contained within \"start.dat\" and runs a simulation for 1000 seconds.\n";
			std::cout << "The grid is then written to a file every 100 seconds.  ";
			std::cout << "The resulting files are \nnamed \"start.0100.0000.dat\", \"start.0200.0000.dat\", ... \"start.1000.0000.dat\".\n\n";
		}
		MMSP::Abort(EXIT_SUCCESS);
	}

	// generate init grid
	else if (std::string(argv[1]) == std::string("--init")) {
		// check argument list
		if (argc<3 or argc>4) {
			std::cout << PROGRAM << ": bad argument list.  Use\n\n";
			std::cout << "    " << PROGRAM << " --help\n\n";
			std::cout << "to generate help message.\n\n";
			MMSP::Abort(EXIT_FAILURE);
		}

		// check problem dimension
		if (std::string(argv[2]).find_first_not_of("0123456789") != std::string::npos) {
			std::cout << PROGRAM << ": init grid must have integral dimension.  Use\n\n";
			std::cout << "    " << PROGRAM << " --help\n\n";
			std::cout << "to generate help message.\n\n";
			MMSP::Abort(EXIT_FAILURE);
		}

		int dim = atoi(argv[2]);

		// set output file name
		std::string outfile;
		if (argc < 4) outfile = "init";
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
			MMSP::Abort(EXIT_FAILURE);
		}

		double sim_start  = 0;
		double sim_finish = 0;
		double sim_step   = 0;
		std::string outfile = "";

		if (std::string(argv[2]).find_first_not_of("0123456789.") == std::string::npos) {
			// set output file name
			outfile = argv[1];

			if (std::string(argv[2]).find_first_not_of("0123456789.") != std::string::npos) {
				std::cout << PROGRAM << ": runtime (sec) must be a positive number.  Use\n\n";
				std::cout << "    " << PROGRAM << " --help\n\n";
				std::cout << "to generate help message.\n\n";
				MMSP::Abort(EXIT_FAILURE);
			}

			sim_finish = atof(argv[2]);
			sim_step = sim_finish;

			if (argc > 3) {
				if (std::string(argv[3]).find_first_not_of("0123456789.") != std::string::npos) {
					std::cout << PROGRAM << ": output increment (sec) must be a positive number.  Use\n\n";
					std::cout << "    " << PROGRAM << " --help\n\n";
					std::cout << "to generate help message.\n\n";
					MMSP::Abort(EXIT_FAILURE);
				}

				sim_step = atof(argv[3]);

				if (sim_step > sim_finish) {
					std::cout << PROGRAM << ": output increment must be smaller than runtime.  Use\n\n";
					std::cout << "    " << PROGRAM << " --help\n\n";
					std::cout << "to generate help message.\n\n";
					MMSP::Abort(EXIT_FAILURE);
				}
			}
		}

		else {
			// set output file name
			outfile = argv[2];

			// set number of time steps
			if (std::string(argv[3]).find_first_not_of("0123456789.") != std::string::npos) {
				std::cout << PROGRAM << ": runtime (sec) must be a positive number.  Use\n\n";
				std::cout << "    " << PROGRAM << " --help\n\n";
				std::cout << "to generate help message.\n\n";
				MMSP::Abort(EXIT_FAILURE);
			}

			sim_finish = atof(argv[3]);
			sim_step = sim_finish;

			if (argc > 4) {
				if (std::string(argv[4]).find_first_not_of("0123456789") != std::string::npos) {
					std::cout << PROGRAM << ": output increment (sec) must be a positive number.  Use\n\n";
					std::cout << "    " << PROGRAM << " --help\n\n";
					std::cout << "to generate help message.\n\n";
					MMSP::Abort(EXIT_FAILURE);
				}

				sim_step = atof(argv[4]);

				if (sim_step > sim_finish) {
					std::cout << PROGRAM << ": output increment must be smaller than number of time steps.  Use\n\n";
					std::cout << "    " << PROGRAM << " --help\n\n";
					std::cout << "to generate help message.\n\n";
					MMSP::Abort(EXIT_FAILURE);
				}
			}
		}

		// file open error check
		std::ifstream input(argv[1]);
		if (!input) {
			std::cerr << "File input error: could not open " << argv[1] << ".\n\n";
			MMSP::Abort(EXIT_FAILURE);
		}

		// read data type
		std::string type;
		getline(input, type, '\n');

		// grid type error check
		if (type.substr(0, 4) != "grid") {
			std::cerr << "File input error: file does not contain grid data." << std::endl;
			MMSP::Abort(EXIT_FAILURE);
		}

		// read grid dimension
		int dim;
		input >> dim;
		input.close();

		// get starting time
		if (outfile.find_first_of(".") != outfile.find_last_of(".")) {
			std::string number = outfile.substr(outfile.find_first_of(".") + 1, outfile.find_last_of(".") - 1);
			sim_start = atof(number.c_str());
		}

		// set output file basename
		std::string base = outfile.substr(0, outfile.find_first_of(".")) + ".";

		// set output file suffix
		std::string suffix = "";
		if (outfile.find_last_of(".") != std::string::npos)
			suffix = outfile.substr(outfile.find_last_of("."), std::string::npos);

		if (dim == 2) {
			// setup logging
			FILE* cfile = NULL;
			if (rank == 0)
				cfile = fopen("c.log", "a"); // existing log will be appended

			// construct grid object
			GRID2D grid(argv[1]);

			// declare host and device data structures
			struct HostData host;
			struct CudaData dev;

			// numerical parameters
			const int bx = 13;
			const int by = 13;
			const int nm = 3;
			const int nx = g1(grid, 0) - g0(grid, 0) + nm - 1;
			const int ny = g1(grid, 1) - g0(grid, 1) + nm - 1;

			// initialize memory
			make_arrays(&host, nx, ny, nm);
			five_point_Laplacian_stencil(dx(grid, 0), dx(grid, 1), host.mask_lap, nm);

			for (int n = 0; n < MMSP::nodes(grid); n++) {
				MMSP::vector<fp_t>& gridN = grid(n);
				MMSP::vector<int> x = MMSP::position(grid, n);
				const int i = x[0] - MMSP::g0(grid, 0) + nm / 2;
				const int j = x[1] - MMSP::g0(grid, 1) + nm / 2;
				host.conc_Cr_old[j][i] = gridN[0];
				host.conc_Nb_old[j][i] = gridN[1];
				host.phi_del_old[j][i] = gridN[NC];
				host.phi_lav_old[j][i] = gridN[NC+1];
			}

			// initialize GPU
			fp_t xCr0, xNb0;
			system_composition(grid, xCr0, xNb0);

			MMSP::set_diffusion_matrix(0.30, 0.02, D_Cr, D_Nb, Lmob, 1);

			init_cuda(&host, nx, ny, nm, kappa, omega, Lmob, D_Cr, D_Nb, &dev);
			device_init_prng(&dev, nx, ny, nm, bx, by);

			// determine timestep
			const double dtTransformLimited = (meshres*meshres) / (2.0 * dim * Lmob[0]*kappa[0]);
			fp_t dtDiffusionLimited = MMSP::timestep(grid, D_Cr, D_Nb);
			double round_dt = 0.0;
			double roundoff = 4e6;
			while (round_dt < 1e-20) {
				roundoff *= 10;
				round_dt = std::floor(roundoff * LinStab * std::min(dtTransformLimited, dtDiffusionLimited)) / roundoff;
			}
			const double dt = round_dt;

			// set intervals for checkpoint I/O and nucleation
			const uint64_t io_interval = std::min(uint64_t(sim_step / dt), uint64_t(5.0 / dt));
			#ifdef NUCLEATION
			const uint64_t nuc_interval = (uint64_t)(0.0001 / dt);
			#endif

			// perform computation
			std::vector<std::string> cstr;
			for (double sim_time = sim_start; sim_time < sim_finish; sim_time += sim_step) {
				/* start update() */
				// convert times to steps (for integer arithmetic)
				const uint64_t kernel_start = sim_time / dt;
				const uint64_t kernel_finish = std::min(sim_time + sim_step, sim_finish) / dt;
				char cbuf[8192];

				for (uint64_t kernel_time = kernel_start; kernel_time < kernel_finish; kernel_time++) {
					print_progress(kernel_time - kernel_start, kernel_finish);

					// === Start Architecture-Specific Kernel ===
					device_boundaries(&dev, nx, ny, nm, bx, by);

					device_fictitious(&dev, nx, ny, nm, bx, by);

					fictitious_boundaries(&dev, nx, ny, nm, bx, by);

					/*
					device_mobilities(&dev, nx, ny, nm, bx, by);

					mobility_boundaries(&dev, nx, ny, nm, bx, by);
					*/

					device_laplacian(&dev, nx, ny, nm, bx, by, dx(grid, 0), dx(grid, 1));

					device_laplacian_boundaries(&dev, nx, ny, nm, bx, by);

					device_evolution(&dev, nx, ny, nm, bx, by, alpha, dt);

					#ifdef NUCLEATION
					const bool nuc_step = (kernel_time % nuc_interval == 0);
					if (nuc_step) {
						device_nucleation(&dev, nx, ny, nm, bx, by,
						                  sigma[0], sigma[1],
						                  lattice_const, ifce_width,
						                  meshres, meshres, meshres,
						                  nuc_interval * dt);
					}
					#endif

					swap_pointers_1D(&(dev.conc_Cr_old), &(dev.conc_Cr_new));
					swap_pointers_1D(&(dev.conc_Nb_old), &(dev.conc_Nb_new));
					swap_pointers_1D(&(dev.phi_del_old), &(dev.phi_del_new));
					swap_pointers_1D(&(dev.phi_lav_old), &(dev.phi_lav_new));

					const bool io_step = (kernel_time+1) % io_interval == 0;
					if (io_step) {
						device_dataviz(&dev, &host, nx, ny, nm, bx, by);

						read_out_result(&dev, &host, nx, ny);

						for (int n = 0; n < MMSP::nodes(grid); n++) {
							MMSP::vector<fp_t>& gridN = grid(n);
							MMSP::vector<int> x = MMSP::position(grid, n);
							const int i = x[0] - MMSP::g0(grid, 0) + nm / 2;
							const int j = x[1] - MMSP::g0(grid, 1) + nm / 2;
							gridN[0]       = host.conc_Cr_new[j][i];
							gridN[1]       = host.conc_Nb_new[j][i];
							gridN[NC]      = host.phi_del_new[j][i];
							gridN[NC+1]    = host.phi_lav_new[j][i];

							// Write fictitious compositions
							double pDel = p(gridN[NC]);
							double pLav = p(gridN[NC+1]);
							double pGam = 1.0 - pDel - pLav;
							double XCR = gridN[0];
							double XNB = gridN[1];

							double INV_DET = inv_fict_det(pDel, pGam, pLav);
							host.conc_Cr_gam[j][i] = fict_gam_Cr(INV_DET, XCR, XNB, pDel, pGam, pLav);
							host.conc_Nb_gam[j][i] = fict_gam_Nb(INV_DET, XCR, XNB, pDel, pGam, pLav);
						}

						ghostswap(grid);

						// Log compositions, phase fractions

						MMSP::vector<double> summary = summarize_fields(grid);
						const double energy = summarize_energy(grid, host.chem_nrg, host.grad_nrg);

						if (rank == 0) {
							sprintf(cbuf, "%10g %9g %9g %12g %12g %12g %12g\n",
							        dt * (kernel_time+1), summary[0], summary[1], summary[2], summary[3], summary[4], energy);
							cstr.push_back(std::string(cbuf));
						}

						// Write image
						char imgname[FILENAME_MAX] = {0};
						for (size_t i = 0; i < base.length(); i++)
							imgname[i] = base[i];

						int namelength = base.length() + sprintf(imgname+base.length(), "%09.4f.png", dt * (kernel_time + 1));
						if (namelength >= FILENAME_MAX) {
							if (rank == 0)
								std::cerr << "Error: Filename " << imgname << " is too long!" << std::endl;
							MMSP::Abort(EXIT_FAILURE);
						}

						write_matplotlib(host.conc_Cr_new, host.conc_Nb_new,
						                 host.phi_del_new, host.phi_lav_new,
										 host.chem_nrg, host.grad_nrg,
										 host.conc_Cr_gam, host.conc_Nb_gam,
						                 nx, ny, nm, MMSP::dx(grid),
						                 kernel_time+1, dt, imgname);

						dtDiffusionLimited = MMSP::timestep(grid, D_Cr, D_Nb);
						if (LinStab * dtDiffusionLimited < 0.2 * dt) {
							std::cout << "ERROR: Timestep is too large! Decrease by a factor of at least "
							          << dt / (LinStab * dtDiffusionLimited) << std::endl;
							std::exit(EXIT_FAILURE);
						}
					}
					// === Finish Architecture-Specific Kernel ===
				}

				for (size_t i = 0; i < cstr.size(); i++)
					fprintf(cfile, "%s", cstr[i].c_str());
				fflush(cfile);
				cstr.clear();

				// Write image
				char filename[FILENAME_MAX] = {0};
				for (size_t i = 0; i < base.length(); i++)
					filename[i] = base[i];
				int namelength = base.length() + sprintf(filename + base.length(), "%09.4f%s", sim_time + sim_step, suffix.c_str());
				if (namelength >= FILENAME_MAX) {
					if (rank == 0)
						std::cerr << "Error: Filename " << filename << " is too long!" << std::endl;
					MMSP::Abort(EXIT_FAILURE);
				}

				MMSP::output(grid, filename);

				print_progress(kernel_finish, kernel_finish);
				/* finish update() */
			}

			free_cuda(&dev);
			free_arrays(&host);
			if (rank == 0)
				fclose(cfile);
		} else {
			std::cerr << dim << "-dimensional grids are unsupported in the CUDA version." << std::endl;
			MMSP::Abort(EXIT_FAILURE);
		}

	}

	MMSP::Finalize();
	return EXIT_SUCCESS;
}

#endif
