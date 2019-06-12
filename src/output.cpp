/**
 \file  output.c
 \brief Implementation of file output functions for spinodal decomposition benchmarks
*/

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <iso646.h>
#include <cmath>
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
		char buffer[256];
		char* pch;
		int ibx=0, iby=0, isc=0;

		/* read parameters */
		while ( !feof(input)) {
			/* process key-value pairs line-by-line */
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

		/* make sure we got everyone */
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
		char timestring[256] = {'\0'};
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

void write_csv(fp_t** conc, const int nx, const int ny, const fp_t dx, const fp_t dy, const int step)
{
	FILE* output;
	char name[256];
	char num[20];
	int i, j;

	/* generate the filename */
	sprintf(num, "%07i", step);
	strcpy(name, "spinodal.");
	strcat(name, num);
	strcat(name, ".csv");

	/* open the file */
	output = fopen(name, "w");
	if (output == NULL) {
		printf("Error: unable to open %s for output. Check permissions.\n", name);
		exit(-1);
	}

	/* write csv data */
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

void write_matplotlib(fp_t** conc, const int nx, const int ny, const int nm,
                      const fp_t deltax,
                      const int step, const fp_t dt, const char* filename)
{
	plt::backend("Agg");

	int w = nx - nm/2;
	int h = ny - nm/2;

	std::vector<float> c(w * h);
	std::vector<float> d(w);
	std::vector<float> cbar(w);
	for (int i = 0; i < w; ++i) {
		d.at(i) = 1e-6 * deltax * i;
		for (int j = 0; j < h; ++j) {
			const float x = conc[j+nm/2][i+nm/2];
			c.at(w * j + i) = x;
			#ifndef CONVERGENCE
			cbar.at(i) += x / h;
			#else
			if (j == h/2)
				cbar.at(i) = x;
			#endif
		}
	}
	const float* z = &(c[0]);
	const int colors = 1;

	figure(3000, 2400, 200);

	std::map<std::string, std::string> str_kw;
	str_kw["cmap"] = "viridis_r";

	std::map<std::string, double> num_kw;
	num_kw["vmin"] = 0.;
	num_kw["vmax"] = 1.;

	char timearr[256] = {0};
	sprintf(timearr, "$t=%07f$ s\n", dt * step);
	plt::suptitle(std::string(timearr));

	long nrows = 3, ncols = 5;
	long spanr = nrows, spanc = ncols;

	spanr -= 1;
	plt::subplot2grid(nrows, ncols, 0, 0, spanr, spanc);
	PyObject* mat = plt::imshow(z, h, w, colors, str_kw, num_kw);
	plt::axis("off");

	std::map<std::string, float> bar_opts;
	bar_opts["shrink"] = 0.75;
	plt::colorbar(mat, bar_opts);

	spanr = 1;
	spanc = ncols-1;
	plt::subplot2grid(nrows, ncols, nrows-1, 0, spanr, spanc);
	plt::plot(d, cbar);
	plt::xlim(0., 1e-6 * deltax * nx);
	#ifndef CONVERGENCE
	plt::ylim(0.4, 0.8);
	#endif
	plt::xlabel("$x\\ /\\ [\\mathrm{\\mu m}]$");
	plt::ylabel("$\\bar{\\chi}_{\\mathrm{Ni}}$");

	plt::save(filename);
	plt::close();
}
