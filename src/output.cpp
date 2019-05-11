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

void subplot(long nrows, long ncols, long plot_number, const std::map<std::string, double> &keywords = {})
{
    // construct positional args
    PyObject* args = PyTuple_New(3);
    PyTuple_SetItem(args, 0, PyFloat_FromDouble(nrows));
    PyTuple_SetItem(args, 1, PyFloat_FromDouble(ncols));
    PyTuple_SetItem(args, 2, PyFloat_FromDouble(plot_number));

	PyObject* kwargs = PyDict_New();
    for (auto it = keywords.begin(); it != keywords.end(); ++it) {
        PyDict_SetItemString(kwargs, it->first.c_str(), PyFloat_FromDouble(it->second));
    }

    PyObject* res = PyObject_Call(plt::detail::_interpreter::get().s_python_function_subplot, args, kwargs);
    if(!res) throw std::runtime_error("Call to subplot() failed.");

    Py_DECREF(args);
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
    for(int j = 0; j < h; ++j) {
        for(int i = 0; i < w; ++i) {
			const float x = conc[j+nm/2][i+nm/2];
            c.at(w * j + i) = x;
			cbar.at(i) += x / h;
			d.at(i) = deltax * i;
        }
	}
    const float* z = &(c[0]);
    const int colors = 1;

	std::map<std::string, std::string> kw;
	kw["cmap"] = "viridis_r";

	char timearr[256] = {0};
	sprintf(timearr, "$t=%07f$ s\n", dt * step);
	plt::text(0.5, -1.5, std::string(timearr));

	plt::figure_size(w, h);

	plt::imshow(z, h, w, colors, kw);
	plt::axis("off");

	plt::tight_layout();
	plt::save(filename);
}
