// check-nucleation.cpp

#include <math.h>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>

#include "globals.h"
#include "nucleation.h"

int main()
{
    std::chrono::high_resolution_clock::time_point beginning = std::chrono::high_resolution_clock::now();
    std::default_random_engine generator;
    std::uniform_real_distribution<fp_t> enrich_Nb_range(enrich_min_Nb(), enrich_max_Nb());
    std::uniform_real_distribution<fp_t> enrich_Cr_range(enrich_min_Cr(), enrich_max_Cr());
    unsigned int seed = (std::chrono::high_resolution_clock::now() - beginning).count();
    generator.seed(seed);

    fp_t xCr = enrich_Cr_range(generator);
    fp_t xNb = enrich_Nb_range(generator);

    const fp_t dtDiffusionLimited = (meshres*meshres) / (4. * std::max(D_Cr[0], D_Nb[1]));
    const fp_t dt = 20 * LinStab * dtDiffusionLimited;

    const fp_t dV = meshres * meshres * meshres;

    const fp_t vFccNi = lattice_const * lattice_const * lattice_const / 4.;
    const fp_t n_gam = M_PI / (3. * sqrt(2.) * vFccNi);

    // Test a delta particle
    #ifdef DEBUG
    printf("Delta particle:\n");
    #endif
    fp_t dG_chem_del, P_nuc_del, Rstar_del;
    fp_t del_xCr = 0., del_xNb = 0.;
    nucleation_driving_force_delta(xCr, xNb, &del_xCr, &del_xNb, &dG_chem_del);
    nucleation_probability_sphere(xCr, xNb, del_xCr, del_xNb,
                                  dG_chem_del, D_Cr[0], D_Nb[1],
                                  0.38, // sigma[0],
                                  vFccNi, n_gam, dV, dt,
                                  &Rstar_del, &P_nuc_del);

    // Test a Laves particle
    #ifdef DEBUG
    printf("Laves particle:\n");
    #endif
    fp_t dG_chem_lav, P_nuc_lav, Rstar_lav;
    fp_t lav_xCr = 0., lav_xNb = 0.;
    nucleation_driving_force_laves(xCr, xNb, &lav_xCr, &lav_xNb, &dG_chem_lav);
    nucleation_probability_sphere(xCr, xNb, lav_xCr, lav_xNb,
                                  dG_chem_lav, D_Cr[0], D_Nb[1],
                                  0.50, // sigma[1],
                                  vFccNi, n_gam, dV, dt,
                                  &Rstar_lav, &P_nuc_lav);

    printf("Interval, D: %9.3e  %9.3e\n", dt,      n_gam);
    printf("Composition: %9.4f  %9.4f\n", xCr,     xNb);
    printf("Delta comp:  %9.4f  %9.4f\n", del_xCr, del_xNb);
    printf("Laves comp:  %9.4f  %9.4f\n", lav_xCr, lav_xNb);
    printf("Driving frc: %9.2e  %9.2e\n", dG_chem_del, dG_chem_lav);
    printf("Crit. radius:%9.2f  %9.2f\n", Rstar_del, Rstar_lav);
    printf("Probability: %9.2e  %9.2e\n", P_nuc_del, P_nuc_lav);
    return 0;
}
