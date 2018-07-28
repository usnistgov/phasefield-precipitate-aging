/**********************************************************************************
 HiPerC: High Performance Computing Strategies for Boundary Value Problems
 written by Trevor Keller and available from https://github.com/usnistgov/hiperc

 This software was developed at the National Institute of Standards and Technology
 by employees of the Federal Government in the course of their official duties.
 Pursuant to title 17 section 105 of the United States Code this software is not
 subject to copyright protection and is in the public domain. NIST assumes no
 responsibility whatsoever for the use of this software by other parties, and makes
 no guarantees, expressed or implied, about its quality, reliability, or any other
 characteristic. We would appreciate acknowledgement if the software is used.

 This software can be redistributed and/or modified freely provided that any
 derivative works bear some notice that they are derived from it, and any modified
 versions bear some notice that they have been modified.

 Questions/comments to Trevor Keller (trevor.keller@nist.gov)
 **********************************************************************************/

/**
 \file  cuda_main.c
 \brief CUDA implementation of semi-infinite diffusion equation
*/

/* system includes */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* common includes */
#include "mesh.h"
#include "numerics.h"
#include "output.h"
#include "timer.h"

/* specific includes */
#include "cuda_data.h"

/**
 \brief Run simulation using input parameters specified on the command line
*/
int main(int argc, char* argv[])
{
	/* declare host and device data structures */
	struct HostData host;
	struct CudaData dev;

	/* declare default mesh size and resolution */
	int bx=32, by=32, nx=202, ny=202, nm=3, code=53;
	const fp_t dx=1.0, dy=1.0;

	/* declare default materials and numerical parameters */
	fp_t M=5.0, kappa=2.0, linStab=0.25, elapsed=0., energy=0.;
	int step=0, steps=5000000, checks=100000;
	param_parser(argc, argv, &bx, &by, &checks, &code, &M, &kappa, &linStab, &nm, &nx, &ny, &steps);

	const fp_t dt = linStab / (24.0 * M * kappa);

	/* initialize memory */
	make_arrays(&host, &mask_lap, nx, ny, nm);
	set_mask(dx, dy, code, mask_lap, nm);
	apply_initial_conditions(&host, nx, ny, nm);
	init_cuda(&host, mask_lap, nx, ny, nm, &dev);

	/* do the work */
	for (step = 1; step < steps+1; step++) {
		print_progress(step, steps);

		/* === Start Architecture-Specific Kernel === */
		device_boundaries(dev.conc_old, nx, ny, nm, bx, by);

		device_laplacian(dev.conc_old, dev.conc_lap, kappa, nx, ny, nm, bx, by);

		device_boundaries(dev.conc_lap, nx, ny, nm, bx, by);

		device_divergence(dev.conc_lap, dev.conc_div, nx, ny, nm, bx, by);

		device_composition(dev.conc_old, dev.conc_div, dev.conc_new, nx, ny, nm, bx, by, M, dt);

		swap_pointers_1D(&(dev.conc_old), &(dev.conc_new));
		/* === Finish Architecture-Specific Kernel === */

		elapsed += dt;

		if (step % checks == 0) {
			/* transfer result to host (conc_new) from device (dev.conc_old) */
			read_out_result(&dev, &host, nx, ny);
		}
	}

	/* clean up */
	free_arrays(&host, mask_lap);
	free_cuda(&dev);

	return 0;
}
