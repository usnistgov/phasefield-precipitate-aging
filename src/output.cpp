/**
 \file  output.c
 \brief Implementation of file output functions for spinodal decomposition benchmarks
*/

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <iso646.h>
#include <cmath>
#include "parabola625.h"
#include "output.h"
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

void param_parser(int* bx, int* by, int* code, int* nm)
{
	FILE* input = fopen("params.txt", "r");
	if (input == NULL) {
		printf("Warning: unable to open parameter file 'params.txt'. Marching with default values.\n");
		fflush(stdout);
	} else {
		char buffer[FILENAME_MAX];
		char* pch;
		int ibx=0, iby=0, isc=0;

		// read parameters
		while ( !feof(input)) {
			// process key-value pairs line-by-line
			if (fgets(buffer, 256, input) != NULL) {
				pch = strtok(buffer, " ");

				if (strcmp(pch, "bx") == 0) {
					pch = strtok(NULL, " ");
					*bx = atoi(pch);
					ibx = 1;
				} else if (strcmp(pch, "by") == 0) {
					pch = strtok(NULL, " ");
					*by = atoi(pch);
					iby = 1;
				} else if (strcmp(pch, "sc") == 0) {
					pch = strtok(NULL, " ");
					*nm = atoi(pch);
					pch = strtok(NULL, " ");
					*code = atoi(pch);
					isc = 1;
				} else {
					printf("Warning: unknown key %s. Ignoring value.\n", pch);
				}
			}
		}

		// make sure we got everyone
		if (! ibx)
			printf("Warning: parameter %s undefined. Using default value, %i.\n", "bx", *bx);
		else if (! iby)
			printf("Warning: parameter %s undefined. Using default value, %i.\n", "by", *by);
		else if (! isc)
			printf("Warning: parameter %s undefined. Using default values, %i and %i.\n", "sc", *nm, *code);

		fclose(input);
	}
}

void print_progress(const int step, const int steps)
{
	static unsigned long tstart;
	time_t rawtime;

	if (step==0) {
		char timestring[FILENAME_MAX] = {'\0'};
		struct tm* timeinfo;
		tstart = time(NULL);
		time( &rawtime );
		timeinfo = localtime( &rawtime );
		strftime(timestring, 80, "%c", timeinfo);
		printf("%s [", timestring);
		fflush(stdout);
	} else if (step==steps) {
		unsigned long deltat = time(NULL)-tstart;
		printf("•] %2luh:%2lum:%2lus\n",deltat/3600,(deltat%3600)/60,deltat%60);
		fflush(stdout);
	} else if ((20 * step) % steps == 0) {
		printf("• ");
		fflush(stdout);
	}
}

void write_csv(fp_t** conc, const int nx, const int ny, const fp_t dx, const fp_t dy, const uint64_t step)
{
	FILE* output;
	char name[FILENAME_MAX];
	char num[20];
	int i, j;

	// generate the filename
	sprintf(num, "%07lu", step);
	strcpy(name, "spinodal.");
	strcat(name, num);
	strcat(name, ".csv");

	// open the file
	output = fopen(name, "w");
	if (output == NULL) {
		printf("Error: unable to open %s for output. Check permissions.\n", name);
		exit(EXIT_FAILURE);
	}

	// write csv data
	fprintf(output, "x,y,c\n");
	for (j = 1; j < ny-1; j++) {
		fp_t y = dy * (j - 1);
		for (i = 1; i < nx-1; i++)	{
			fp_t x = dx * (i - 1);
			fprintf(output, "%f,%f,%f\n", x, y, conc[j][i]);
		}
	}

	fclose(output);
}

void figure(int w, int h, size_t dpi)
{
	PyObject* size = PyTuple_New(2);
	PyTuple_SetItem(size, 0, PyFloat_FromDouble(static_cast<double>(w)/dpi));
	PyTuple_SetItem(size, 1, PyFloat_FromDouble(static_cast<double>(h)/dpi));

	PyObject* kwargs = PyDict_New();
	PyDict_SetItemString(kwargs, "figsize", size);
	PyDict_SetItemString(kwargs, "dpi", PyLong_FromSize_t(dpi));
	Py_DECREF(size);

	PyObject* res = PyObject_Call(plt::detail::_interpreter::get().s_python_function_figure,
	                              plt::detail::_interpreter::get().s_python_empty_tuple,
	                              kwargs);
	Py_DECREF(kwargs);

	if (!res) throw std::runtime_error("Call to figure() failed.");
	Py_DECREF(res);
}

int write_dummy(fp_t** conc_Cr, fp_t** conc_Nb,
                fp_t** phi_del, fp_t** phi_lav,
                const int nx, const int ny, const int nm,
                const fp_t deltax,
                const int step, const fp_t dt, const char* filename)
{
	return 0;
}

int write_matplotlib(fp_t** conc_Cr, fp_t** conc_Nb,
                     fp_t** phi_del, fp_t** phi_lav,
                     const int nx, const int ny, const int nm,
                     const fp_t deltax,
                     const uint64_t step, const fp_t dt, const char* filename)
{
	plt::backend("Agg");

	int w = nx - nm/2;
	int h = ny - nm/2;

	std::vector<float> c_Ni(w * h);
	std::vector<float> p_gam(w * h);

	std::vector<float> d(w);
	std::vector<float> c_Cr_bar(w);
	std::vector<float> c_Nb_bar(w);
	std::vector<float> c_Ni_bar(w);
	std::vector<float> p_del_bar(w);
	std::vector<float> p_lav_bar(w);

	PyObject* mat;

	for (int i = 0; i < w; ++i) {
		d.at(i) = 1e6 * deltax * i;
		for (int j = 0; j < h; ++j) {
			c_Ni.at(w * j + i) = 1.0 - conc_Cr[j+nm/2][i+nm/2] - conc_Nb[j+nm/2][i+nm/2];
			p_gam.at(w * j + i) = 1.0 - phi_del[j+nm/2][i+nm/2] - phi_lav[j+nm/2][i+nm/2];

			if (j == h/2) {
				c_Cr_bar.at(i) = conc_Cr[j+nm/2][i+nm/2];
				c_Nb_bar.at(i) = conc_Nb[j+nm/2][i+nm/2];
				c_Ni_bar.at(i) = c_Ni.at(w * j + i);
				p_del_bar.at(i) = phi_del[j+nm/2][i+nm/2];
				p_lav_bar.at(i) = phi_lav[j+nm/2][i+nm/2];
			}
		}
	}
	const int colors = 1;

	figure(3000, 2400, 200);

	std::map<std::string, std::string> str_kw;
	str_kw["cmap"] = "viridis_r";
	str_kw["interpolation"] = "nearest";

	std::map<std::string, double> num_kw;
	num_kw["vmin"] = 0.;
	num_kw["vmax"] = 1.;

	char timearr[FILENAME_MAX] = {0};
	sprintf(timearr, "$t=%.3f\\ \\mathrm{s}$\n", dt * step);
	plt::suptitle(std::string(timearr));

	const long nrows = 3;
	const long ncols = 5;

	// subplot2grid(shape=(nrows, ncols), loc=(row, col), rowspan=1, colspan=1, fig=None, **kwargs)
	plt::subplot2grid(nrows, ncols, 0, 0, 1, ncols - 1);
	mat = plt::imshow(&(p_gam[0]), h, w, colors, str_kw, num_kw);
	plt::title("$\\phi^{\\gamma}$");
	plt::axis("off");

	plt::subplot2grid(nrows, ncols, 1, 0, 1, ncols - 1);
	mat = plt::imshow(&(c_Ni[0]), h, w, colors, str_kw, num_kw);
	plt::title("$x_{\\mathrm{Ni}}$");
	plt::axis("off");

	plt::subplot2grid(nrows, ncols, 0, ncols - 1, 2, 1);
	std::map<std::string, float> bar_opts;
	bar_opts["shrink"] = 0.75;
	plt::colorbar(mat, bar_opts);
	plt::axis("off");
	Py_DECREF(mat);

	plt::subplot2grid(nrows, ncols, nrows-1, 0, 1, ncols - 1);
	plt::xlim(0., 1e6 * deltax * nx);
	plt::ylim(0., 1.);
	plt::xlabel("$X\\ /\\ [\\mathrm{\\mu m}]$");
	plt::ylabel("$x_{\\mathrm{Ni}}(Y=0)$");

	str_kw.clear();
	str_kw["label"] = "$x_{\\mathrm{Nb}}$";
	plt::plot(d, c_Nb_bar, str_kw);
	str_kw["label"] = "$x_{\\mathrm{Cr}}$";
	plt::plot(d, c_Cr_bar, str_kw);
	/*
	str_kw["label"] = "$x_{\\mathrm{Ni}}$";
	plt::plot(d, c_Ni_bar, str_kw);
	*/
	str_kw["label"] = "$\\phi^{\\mathrm{\\delta}}$";
	plt::plot(d, p_del_bar, str_kw);
	str_kw["label"] = "$\\phi^{\\mathrm{\\lambda}}$";
	plt::plot(d, p_lav_bar, str_kw);

	plt::legend();

	plt::save(filename);
	plt::close();

	return 0;
}

int write_matplotlib(fp_t** conc_Cr, fp_t** conc_Nb,
                     fp_t** phi_del, fp_t** phi_lav,
                     fp_t** chem_nrg, fp_t** grad_nrg,
                     fp_t** gam_Cr, fp_t** gam_Nb,
                     const int nx, const int ny, const int nm,
                     const fp_t deltax,
                     const uint64_t step, const fp_t dt, const char* filename)
{
	#ifdef MPI_VERSION
	std::cerr << "Error: cannot write images in parallel." << std::endl;
	MPI::Abort(EXIT_FAILURE);
	#endif

	plt::backend("Agg");

	int L = nx - nm;
	int dL = std::max(1, nx / 1024);
	int w = std::min(L, 1024);
	int h = ny - nm;

	std::vector<float> d(w);
	std::vector<float> f(w);
	std::vector<float> fchem(w);
	std::vector<float> fgrad(w);

	std::vector<float> c_Cr_bar(w);
	std::vector<float> c_Nb_bar(w);

	std::vector<float> c_Cr_gam(w);
	std::vector<float> c_Nb_gam(w);

	std::vector<float> p_del_bar(w);
	std::vector<float> p_lav_bar(w);

	const int j = h/2;
	for (int k = 0; k < w && k*dL < L; ++k) {
		const int i = k * dL;
		d.at(k) = 1e6 * deltax * i;
		if (chem_nrg != NULL) {
			fchem.at(k) = chem_nrg[j+nm/2][i+nm/2];
			fgrad.at(k) = grad_nrg[j+nm/2][i+nm/2];
			f.at(k) = chem_nrg[j+nm/2][i+nm/2] + grad_nrg[j+nm/2][i+nm/2];
		} else {
			fchem.at(k) = 0;
			fgrad.at(k) = 0;
			f.at(k) = 0;
		}

		c_Cr_bar.at(k) = conc_Cr[j+nm/2][i+nm/2];
		c_Nb_bar.at(k) = conc_Nb[j+nm/2][i+nm/2];

		c_Cr_gam.at(k) = gam_Cr[j+nm/2][i+nm/2];
		c_Nb_gam.at(k) = gam_Nb[j+nm/2][i+nm/2];

		p_del_bar.at(k) = phi_del[j+nm/2][i+nm/2];
		p_lav_bar.at(k) = phi_lav[j+nm/2][i+nm/2];
	}

	const long nrows = 2;
	const long ncols = 1;

	figure(3000, 2400, 200);

	char timearr[FILENAME_MAX] = {0};
	sprintf(timearr, "$t=%.3f\\ \\mathrm{s}$\n", dt * step);
	plt::suptitle(std::string(timearr));

	// subplot2grid(shape=(nrows, ncols), loc=(row, col), rowspan=1, colspan=1, fig=None, **kwargs)
	plt::subplot2grid(nrows, ncols, 0, 0, 1, 1);
	plt::xlim(0., 1e6 * deltax * nx);
	plt::xlabel("$X\\ /\\ [\\mathrm{\\mu m}]$");
	plt::ylabel("$\\mathcal{F}(Y=0)\\ /\\ [\\mathrm{J/m^3}]$");

	std::map<std::string, std::string> str_kw;
	str_kw["label"] = "$f$";
	plt::plot(d, f, str_kw);
	str_kw["label"] = "$f_{\\mathrm{chem}}$";
	plt::plot(d, fchem, str_kw);
	str_kw["label"] = "$f_{\\mathrm{grad}}$";
	plt::plot(d, fgrad, str_kw);
	str_kw.clear();

	plt::legend();

	plt::subplot2grid(nrows, ncols, nrows-1, 0, 1, 1);
	plt::xlim(0., 1e6 * deltax * nx);
	plt::ylim(0., 1.);
	plt::xlabel("$X\\ /\\ [\\mathrm{\\mu m}]$");
	plt::ylabel("$x_{\\mathrm{Ni}}(Y=0)$");

	str_kw["label"] = "$x_{\\mathrm{Nb}}$";
	plt::plot(d, c_Nb_bar, str_kw);
	str_kw.clear();
	str_kw["linestyle"] = ":";
	plt::plot(d, c_Nb_gam, str_kw);
	str_kw.clear();

	str_kw["label"] = "$x_{\\mathrm{Cr}}$";
	plt::plot(d, c_Cr_bar, str_kw);
	str_kw.clear();
	str_kw["linestyle"] = ":";
	plt::plot(d, c_Cr_gam, str_kw);
	str_kw.clear();

	str_kw["label"] = "$\\phi^{\\mathrm{\\delta}}$";
	plt::plot(d, p_del_bar, str_kw);

	str_kw["label"] = "$\\phi^{\\mathrm{\\lambda}}$";
	plt::plot(d, p_lav_bar, str_kw);

	plt::legend();

	plt::save(filename);
	plt::close();

	return 0;
}
