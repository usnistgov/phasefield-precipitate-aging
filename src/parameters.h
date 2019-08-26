/**
 \file  parameters.h
 \brief Model parameter definitions
*/

#ifndef __PARAMETERS_H_
#define __PARAMETERS_H_

#include "type.h"
#include "parabola625.h"

// === Discretization Parameters ===

const double meshres = 1.25e-10;     // grid spacing, Δx (m); max. is 2.5 Å
const double ifce_width = 7.5e-10;   // interface thickness, 2λ (m), TKR5p274
const fp_t LinStab = 1. / 50;  // threshold of linear (von Neumann) stability, Co (dimensionless)
const fp_t precip_stabilizer = 13.;  // 1.5 Rc = R0, the root of the curve; fitting parameter
const fp_t lattice_const = 0.352e-9; // lattice spacing of FCC nickel (m)

/* CFL Values
| dx[/nm] | dt[/s]  | CFL       | Note     |
| ------- | ------- | --------- | -------- |
| 0.125   | 1.00e-7 | 9.105475  |          |
| 0.125   | 1.25e-7 | 7.28438   |          |
| 0.125   | 1.50e-7 | 6.0703167 | unstable |
| 0.250   | 5.00e-7 | 7.28438   |          |
*/

/* Stabilizer Values
| dx[/nm] | stabilizer | Delta       | Laves       | Comment               |
| ------- | ---------- | ----------- | ----------- | --------------------- |
| 0.125   |         16 | unstable    | unstable    |                       |
| 0.125   |         17*| stable      | stable      |                       |
| 0.0625  |          7 | unstable    | unstable    | melted before growing |
| 0.0625  |         13 |     |     |  |
*/

// Define st.dev. of bell curves for alloying element segregation
//                       Cr      Nb
const double bell[NC] = {150e-9, 50e-9}; // est. between 80-200 nm from SEM

// === Energetic Parameters ===

// Choose numerical diffusivity to lock chemical and transformational timescales
//                      delta      Laves
const fp_t Lmob[NP]  = {2.904e-11, 2.904e-11}; // numerical mobility (m^2/Ns)
const fp_t sigma[NP] = {s_delta(), s_laves()}; // interfacial energy (J/m^2)
const fp_t alpha = 1.07e11;                    // three-phase coexistence coefficient (J/m^3)

// Compute well height (J/m^3) from interfacial width (nm) and energy (J/m^2)
// Derivation: TKR5p287 -- after Kim, Kim & Suzuki, *Phys. Rev. E* **60** (1999) 7186--7197.
const fp_t width_factor = 2.2; // when "interface" is [0.1,0.9]; 2.94 if [0.05,0.95]
const fp_t kappa[NP] = {3. * sigma[0] * ifce_width / width_factor,
                        3. * sigma[1] * ifce_width / width_factor}; // gradient energy coefficient (J/m)
const fp_t omega[NP] = {18. * omega[0] / kappa[0],  // delta
                        18. * omega[1] / kappa[1]}; // Laves
#endif
