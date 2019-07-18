/*************************************************************************************
 * File: cuda625.hpp                                                                 *
 * Declarations for 2D and 3D isotropic Cr-Nb-Ni alloy phase transformations         *
 *                                                                                   *
 * Questions/comments to trevor.keller@nist.gov (Trevor Keller, Ph.D.)               *
 *                                                                                   *
 * This software was developed at the National Institute of Standards and Technology *
 * by employees of the Federal Government in the course of their official duties.    *
 * Pursuant to title 17 section 105 of the United States Code this software is not   *
 * subject to copyright protection and is in the public domain. NIST assumes no      *
 * responsibility whatsoever for the use of this code by other parties, and makes no *
 * guarantees, expressed or implied, about its quality, reliability, or any other    *
 * characteristic. We would appreciate acknowledgement if the software is used.      *
 *                                                                                   *
 * This software can be redistributed and/or modified freely provided that any       *
 * derivative works bear some notice that they are derived from it, and any modified *
 * versions bear some notice that they have been modified. Derivative works that     *
 * include MMSP or other software licensed under the GPL may be subject to the GPL.  *
 *************************************************************************************/

#ifndef __CUDA625_HPP__
#define __CUDA625_HPP__
#include "globals.h"

// Number of precipitates and components (for array allocation)

std::string PROGRAM = "alloy625";
std::string MESSAGE = "Isotropic Cr-Nb-Ni alloy phase transformation code";

typedef MMSP::grid<1, MMSP::vector<fp_t> > GRID1D;
typedef MMSP::grid<2, MMSP::vector<fp_t> > GRID2D;
typedef MMSP::grid<3, MMSP::vector<fp_t> > GRID3D;

/**
 \brief Container for system composition and phase fractions
*/
class Composition
{
public:
	// constructor
	Composition()
	{
		for (int i = 0; i < NP + 1; i++) {
			for (int j = 0; j < NC; j++)
				x[i][j] = 0.0;
		}
		for (int i = 0; i < NP + 1; i++)
			N[i] = 0;
	}
	~Composition() {}

	// modifier
	Composition& operator+=(const Composition& c);

	// representation
	double x[NP + 1][NC]; // composition of each phase
	int    N[NP + 1];   // amount of each phase
};

/**
 \brief Combine composition containers
*/
Composition& Composition::operator+=(const Composition& c)
{
	for (int i = 0; i < NP + 1; i++) {
		for (int j = 0; j < NC; j++)
			this->x[i][j] += c.x[i][j];
	}
	for (int i = 0; i < NP + 1; i++)
		this->N[i] += c.N[i];

	return *this;
}

/**
 \brief Initialize domain with flat composition field
*/
void init_flat_composition(GRID2D& grid, std::mt19937& mtrand);

/**
 \brief Initialize domain with Gaussian peaks in compositions
*/
void init_gaussian_enrichment(GRID2D& grid, std::mt19937& mtrand);

/**
 \brief Embed particle at specified position
 Set <b>O</b>rder <b>P</b>arameter and <b>C</b>omposition fields.
 See TKR5p271.
*/
void embed_OPC(GRID2D& grid,
               const MMSP::vector<int>& x,
               const fp_t& xCr,
               const fp_t& xNb,
               const fp_t& par_xe_Cr,
               const fp_t& par_xe_Nb,
               const fp_t& R_precip,
               const int pid);

/**
 \brief Insert a single particle at the specified location
 Secondary phase will be chosen by random "coin toss".
*/
void seed_solitaire(GRID2D& grid,
                    const fp_t D_CrCr, const fp_t D_NbNb,
                    const fp_t sigma_del, const fp_t sigma_lav,
                    const fp_t lattice_const,
                    const fp_t ifce_width,
                    const fp_t dx,
                    const fp_t dt,
                    std::mt19937& mtrand);

/**
 \brief Insert a slab of delta phase at the left border
*/
void embed_planar_delta(GRID2D& grid, const int w_precip);

/**
 \brief Insert one particle of each secondary phase
 Nuclei will be slightly off-center horizontally, one to each side,
 and separated vertically by an equal distance from the midpoint.
*/
void seed_pair(GRID2D& grid,
               const fp_t D_CrCr, const fp_t D_NbNb,
               const fp_t sigma_del, const fp_t sigma_lav,
               const fp_t lattice_const,
               const fp_t ifce_width,
               const fp_t dx,
               const fp_t dt);

/**
 \brief Compute distance between MMSP coordinates
*/
double radius(const MMSP::vector<int>& a, const MMSP::vector<int>& b, const double& dx);

/**
 \brief Compute fictitious compositions using analytical expressions
*/
template<typename T>
void update_compositions(MMSP::vector<T>& GRIDN);

/**
 \brief Compute Gibbs free energy density
*/
template<typename T>
T gibbs(const MMSP::vector<T>& v);

/**
 \brief Compute gradient of specified field, only
*/
template <int dim, typename T>
MMSP::vector<T> maskedgradient(const MMSP::grid<dim, MMSP::vector<T> >& GRID, const MMSP::vector<int>& x, const int N);

/**
 \brief Summarize field values
 Integrate composition and phase fractions over the whole grid to make sure mass is conserved and phase transformations are sane
*/
template<int dim, typename T>
MMSP::vector<double> summarize_fields(MMSP::grid<dim, MMSP::vector<T> > const& GRID);

/**
 \brief Compute global free energy
 Integrate free energy over the whole grid to make sure it decreases with time
*/
template<int dim, typename T>
double summarize_energy(MMSP::grid<dim, MMSP::vector<T> > const& GRID);

/**
   \brief Compute interface width
   Compute separation between \f$ \phi = 0.1 \f$ to \f$ \phi = 0.9 \f$, <em>viz.</em>,
   the interface width along \f$ y = 0 \f$.
*/
template <int dim, typename T>
double two_lambda(const MMSP::grid<dim,MMSP::vector<T> > GRID);

#endif
