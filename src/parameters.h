/**
 \file  parameters.h
 \brief Model parameter definitions
*/

#ifndef __PARAMETERS_H_
#define __PARAMETERS_H_

#include "type.h"
#include "parabola625.h"

// === Discretization Parameters ===

#if defined(PLANAR) or defined(TANH)
const fp_t meshres = 1.0e-9;    // grid spacing, Δx (m)
const fp_t ifce_width = 20e-9;   // interface thickness, 2λ (m)
#else
const fp_t meshres = 0.3125e-9;  // grid spacing, Δx (m); max. is 2.5 Å
const fp_t ifce_width = 2.5e-9;  // interface thickness, 2λ (m), TKR5p274
#endif
const fp_t LinStab = 2.0e-2;     // threshold of linear (von Neumann) stability, Co (dimensionless)
const fp_t MobStab = 1.0;     // stability factor for diffusion control

// === Materials Parameters ===

const fp_t lattice_const = 0.352e-9; // lattice spacing of FCC nickel (m)
const fp_t vFccNi = lattice_const * lattice_const * lattice_const / 4.; // Volume of an FCC Ni atom
const fp_t precip_stabilizer = 2.;   // 1.5 Rc = R0, the root of the curve; fitting parameter

// Define st.dev. of bell curves for alloying element segregation
//                       Cr      Nb
const double bell[NC] = {150e-9, 50e-9}; // est. between 80-200 nm from SEM

// === Energetic Parameters ===

const fp_t sigma[NP] = {s_delta(), s_laves()}; // interfacial energy (J/m^2)
const fp_t alpha = 1.07e11;                    // three-phase coexistence coefficient (J/m^3)

// Compute well height (J/m^3) from interfacial width (nm) and energy (J/m^2)
// Derivation: TKR5p287 -- after Kim, Kim & Suzuki, *Phys. Rev. E* **60** (1999) 7186--7197.

const fp_t width_factor = 2.2; // when "interface" is [0.1,0.9]; 2.94 if [0.05,0.95]
const fp_t kappa[NP] = {3. * sigma[0]* ifce_width / width_factor,
                        3. * sigma[1]* ifce_width / width_factor
                       }; // gradient energy coefficient (J/m)
const fp_t omega[NP] = {18. * sigma[0] * sigma[0] / kappa[0],
                        18. * sigma[1] * sigma[1] / kappa[1]
                       };
#endif
