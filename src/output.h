/**
 \file  output.h
 \brief Declaration of output function prototypes for diffusion benchmarks
*/

/** \cond SuppressGuard */
#ifndef _OUTPUT_H_
#define _OUTPUT_H_
/** \endcond */

#include <iso646.h>
#include "type.h"

/**
 \brief Read parameters from file specified on the command line
*/
void param_parser(int* bx, int* by, int* code, int* nm);

/**
 \brief Prints timestamps and a 20-point progress bar to stdout

 Call inside the timestepping loop, near the top, e.g.
 \code
 for (int step=0; step<steps; step++) {
 	print_progress(step, steps);
 	take_a_step();
 	elapsed += dt;
 }
 \endcode
*/
void print_progress(const int step, const int steps);

/**
 \brief Writes scalar composition field to diffusion.???????.csv
*/
void write_csv(fp_t** conc, const int nx, const int ny, const fp_t dx, const fp_t dy, const uint64_t step);

/**
 \brief Dummy to initialize future object
*/
int write_dummy(fp_t** conc_Cr, fp_t** conc_Nb,
                fp_t** phi_del, fp_t** phi_lav,
                const int nx, const int ny, const int nm,
                const fp_t deltax,
                const int step, const fp_t dt, const char* filename);

/**
 \brief Writes scalar composition field to PNG using matplotlib-cpp
*/
int write_matplotlib(fp_t** conc_Cr, fp_t** conc_Nb,
                     fp_t** phi_del, fp_t** phi_lav,
                     const int nx, const int ny, const int nm,
                     const fp_t deltax,
                     const uint64_t step, const fp_t dt, const char* filename);

/**
   \brief Writes scalar debugging info to PNG using matplotlib-cpp
*/
int write_matplotlib(fp_t** conc_Cr, fp_t** conc_Nb,
                     fp_t** phi_del, fp_t** phi_lav,
                     fp_t** nrg_den,
                     fp_t** gam_Cr, fp_t** gam_Nb,
                     const int nx, const int ny, const int nm,
                     const fp_t deltax,
                     const uint64_t step, const fp_t dt, const char* filename);

/** \cond SuppressGuard */
#endif /* _OUTPUT_H_ */
/** \endcond */
