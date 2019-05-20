// check-nucleation.cpp

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <random>
#include <math.h>

#include "globals.h"
#include "nucleation.h"


int main()
{
    const size_t iters = 50000;

    std::chrono::high_resolution_clock::time_point beginning = std::chrono::high_resolution_clock::now();
    const unsigned int seed = (std::chrono::high_resolution_clock::now() - beginning).count();
    std::default_random_engine generator(seed); // linear congruential engine

    // Constants
    const fp_t dtDiffusionLimited = (meshres*meshres) / (4. * std::max(D_Cr[0], D_Nb[1]));
    const fp_t dt = 1000 * LinStab * dtDiffusionLimited;
    const fp_t dV = meshres * meshres * meshres;
    const fp_t vFccNi = lattice_const * lattice_const * lattice_const / 4.;
    const fp_t n_gam = M_PI / (3. * sqrt(2.) * vFccNi);

    printf("=== Matrix ===\n");

    fp_t xCrM[iters]   = {0.};
    fp_t xNbM[iters]   = {0.};
    fp_t dGdelM[iters] = {0.};
    fp_t dGlavM[iters] = {0.};
    fp_t PdelM[iters]  = {0.};
    fp_t PlavM[iters]  = {0.};
    fp_t RdelM[iters]  = {0.};
    fp_t RlavM[iters]  = {0.};

    std::uniform_real_distribution<fp_t> matrix_Nb_range(matrix_min_Nb(), matrix_max_Nb());
    std::uniform_real_distribution<fp_t> matrix_Cr_range(matrix_min_Cr(), matrix_max_Cr());

    for (size_t i=0; i<iters; i++) {
        xCrM[i] = matrix_Cr_range(generator);
        xNbM[i] = matrix_Nb_range(generator);

        nucleation_driving_force_delta(xCrM[i], xNbM[i], &(dGdelM[i]));
        nucleation_probability_sphere(xCrM[i], xNbM[i],
                                      xe_del_Cr(), xe_del_Nb(),
                                      dGdelM[i], D_Cr[0], D_Nb[1],
                                      s_delta(),
                                      vFccNi, n_gam, dV, dt,
                                      &(RdelM[i]), &(PdelM[i]));
        nucleation_driving_force_laves(xCrM[i], xNbM[i], &(dGlavM[i]));
        nucleation_probability_sphere(xCrM[i], xNbM[i],
                                      xe_lav_Cr(), xe_lav_Nb(),
                                      dGlavM[i], D_Cr[0], D_Nb[1],
                                      s_laves(),
                                      vFccNi, n_gam, dV, dt,
                                      &(RlavM[i]), &(PlavM[i]));
    }

    // Mean
    fp_t axCrM(0.),   axNbM(0.);
    fp_t adGdelM(0.), adGlavM(0.);
    fp_t aPdelM(0.),  aPlavM(0.);
    fp_t aRdelM(0.),  aRlavM(0.);
    for (size_t i=0; i<iters; i++) {
        axCrM   += xCrM[i]   / iters;
        axNbM   += xNbM[i]   / iters;
        adGdelM += dGdelM[i] / iters;
        adGlavM += dGlavM[i] / iters;
        aPdelM  += PdelM[i]  / iters;
        aPlavM  += PlavM[i]  / iters;
        aRdelM  += RdelM[i]  / iters;
        aRlavM  += RlavM[i]  / iters;
    }

    // StDev
    fp_t sxCrM(0.),   sxNbM(0.);
    fp_t sdGdelM(0.), sdGlavM(0.);
    fp_t sPdelM(0.),  sPlavM(0.);
    fp_t sRdelM(0.),  sRlavM(0.);
    for (size_t i=0; i<iters; i++) {
        sxCrM   += std::pow(axCrM   - xCrM[i], 2);
        sxNbM   += std::pow(axNbM   - xNbM[i], 2);
        sdGdelM += std::pow(adGdelM - dGdelM[i], 2);
        sdGlavM += std::pow(adGlavM - dGlavM[i], 2);
        sPdelM  += std::pow(aPdelM  - PdelM[i], 2);
        sPlavM  += std::pow(aPlavM  - PlavM[i], 2);
        sRdelM  += std::pow(aRdelM  - RdelM[i], 2);
        sRlavM  += std::pow(aRlavM  - RlavM[i], 2);
    }

    sxCrM   = std::sqrt(sxCrM   / (iters - 1));
    sxNbM   = std::sqrt(sxNbM   / (iters - 1));
    sdGdelM = std::sqrt(sdGdelM / (iters - 1));
    sdGlavM = std::sqrt(sdGlavM / (iters - 1));
    sPdelM  = std::sqrt(sPdelM  / (iters - 1));
    sPlavM  = std::sqrt(sPlavM  / (iters - 1));
    sRdelM  = std::sqrt(sRdelM  / (iters - 1));
    sRlavM  = std::sqrt(sRlavM  / (iters - 1));

    printf("Composition: %9.6f ± %5.4f    %9.6f ± %5.4f\n", axCrM, sxCrM, axNbM, sxNbM);
    printf("Driving frc: %9.2e ± %5.2e  %9.2e ± %5.2e\n", adGdelM, sdGdelM, adGlavM, sdGlavM);
    printf("Crit. radius:%9.2e ± %5.2e  %9.2e ± %5.2e\n", aRdelM, sRdelM, aRlavM, sRlavM);
    printf("            (%9.4f Δx          %9.4f Δx)\n", aRdelM/meshres, aRlavM/meshres);
    printf("Probability: %9.2e ± %5.2e  %9.2e ± %5.2e\n", aPdelM, sPdelM, aPlavM, sPlavM);



    printf("=== Enrichment ===\n");

    fp_t xCrE[iters]   = {0.};
    fp_t xNbE[iters]   = {0.};
    fp_t dGdelE[iters] = {0.};
    fp_t dGlavE[iters] = {0.};
    fp_t PdelE[iters]  = {0.};
    fp_t PlavE[iters]  = {0.};
    fp_t RdelE[iters]  = {0.};
    fp_t RlavE[iters]  = {0.};

    std::uniform_real_distribution<fp_t> enrich_Nb_range(enrich_min_Nb(), enrich_max_Nb());
    std::uniform_real_distribution<fp_t> enrich_Cr_range(enrich_min_Cr(), enrich_max_Cr());

    for (size_t i=0; i<iters; i++) {
        xCrE[i] = enrich_Cr_range(generator);
        xNbE[i] = enrich_Nb_range(generator);

        nucleation_driving_force_delta(xCrE[i], xNbE[i], &(dGdelE[i]));
        nucleation_probability_sphere(xCrE[i], xNbE[i],
                                      xe_del_Cr(), xe_del_Nb(),
                                      dGdelE[i], D_Cr[0], D_Nb[1],
                                      s_delta(),
                                      vFccNi, n_gam, dV, dt,
                                      &(RdelE[i]), &(PdelE[i]));
        nucleation_driving_force_laves(xCrE[i], xNbE[i], &(dGlavE[i]));
        nucleation_probability_sphere(xCrE[i], xNbE[i],
                                      xe_lav_Cr(), xe_lav_Nb(),
                                      dGlavE[i], D_Cr[0], D_Nb[1],
                                      s_laves(),
                                      vFccNi, n_gam, dV, dt,
                                      &(RlavE[i]), &(PlavE[i]));
    }

    // Mean
    fp_t axCrE(0.),   axNbE(0.);
    fp_t adGdelE(0.), adGlavE(0.);
    fp_t aPdelE(0.),  aPlavE(0.);
    fp_t aRdelE(0.),  aRlavE(0.);
    for (size_t i=0; i<iters; i++) {
        axCrE   += xCrE[i]   / iters;
        axNbE   += xNbE[i]   / iters;
        adGdelE += dGdelE[i] / iters;
        adGlavE += dGlavE[i] / iters;
        aPdelE  += PdelE[i]  / iters;
        aPlavE  += PlavE[i]  / iters;
        aRdelE  += RdelE[i]  / iters;
        aRlavE  += RlavE[i]  / iters;
    }

    // StDev
    fp_t sxCrE(0.),   sxNbE(0.);
    fp_t sdGdelE(0.), sdGlavE(0.);
    fp_t sPdelE(0.),  sPlavE(0.);
    fp_t sRdelE(0.),  sRlavE(0.);
    for (size_t i=0; i<iters; i++) {
        sxCrE   += std::pow(axCrE   - xCrE[i],   2);
        sxNbE   += std::pow(axNbE   - xNbE[i],   2);
        sdGdelE += std::pow(adGdelE - dGdelE[i], 2);
        sdGlavE += std::pow(adGlavE - dGlavE[i], 2);
        sPdelE  += std::pow(aPdelE  - PdelE[i],  2);
        sPlavE  += std::pow(aPlavE  - PlavE[i],  2);
        sRdelE  += std::pow(aRdelE  - RdelE[i],  2);
        sRlavE  += std::pow(aRlavE  - RlavE[i],  2);
    }

    sxCrE   = std::sqrt(sxCrE   / (iters - 1));
    sxNbE   = std::sqrt(sxNbE   / (iters - 1));
    sdGdelE = std::sqrt(sdGdelE / (iters - 1));
    sdGlavE = std::sqrt(sdGlavE / (iters - 1));
    sPdelE  = std::sqrt(sPdelE  / (iters - 1));
    sPlavE  = std::sqrt(sPlavE  / (iters - 1));
    sRdelE  = std::sqrt(sRdelE  / (iters - 1));
    sRlavE  = std::sqrt(sRlavE  / (iters - 1));

    printf("Composition: %9.6f ± %5.4f    %9.6f ± %5.4f\n", axCrE, sxCrE, axNbE, sxNbE);
    printf("Driving frc: %9.2e ± %5.2e  %9.2e ± %5.2e\n", adGdelE, sdGdelE, adGlavE, sdGlavE);
    printf("Crit. radius:%9.2e ± %5.2e  %9.2e ± %5.2e\n", aRdelE, sRdelE, aRlavE, sRlavE);
    printf("            (%9.4f Δx          %9.4f Δx)\n", aRdelE/meshres, aRlavE/meshres);
    printf("Probability: %9.2e ± %5.2e  %9.2e ± %5.2e\n", aPdelE, sPdelE, aPlavE, sPlavE);

    // Sweep sigma
    FILE* csv = fopen("sigma.csv", "w");
    fprintf(csv, "sigma,Pdel,Plav\n");
    const double ds = 0.001;
    for (double s = ds; s < 0.5; s += ds) {
        fprintf(csv, "%.3f,", s);
        nucleation_driving_force_delta(axCrE, axNbE, &adGdelE);
        nucleation_probability_sphere(axCrE, axNbE,
                                      xe_del_Cr(), xe_del_Nb(),
                                      adGdelE, D_Cr[0], D_Nb[1],
                                      s,
                                      vFccNi, n_gam, dV, dt,
                                      &aRdelE, &aPdelE);
        fprintf(csv, "%12.4e,", aPdelE);

        nucleation_driving_force_laves(axCrE, axNbE, &adGlavE);
        nucleation_probability_sphere(axCrE, axNbE,
                                      xe_lav_Cr(), xe_lav_Nb(),
                                      adGlavE, D_Cr[0], D_Nb[1],
                                      s,
                                      vFccNi, n_gam, dV, dt,
                                      &aRlavE, &aPlavE);
        fprintf(csv, "%12.4e\n", aPlavE);
    }
    fclose(csv);

    return 0;
}
