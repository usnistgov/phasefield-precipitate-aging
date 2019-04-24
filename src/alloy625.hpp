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

typedef MMSP::grid<1,MMSP::vector<fp_t> > GRID1D;
typedef MMSP::grid<2,MMSP::vector<fp_t> > GRID2D;
typedef MMSP::grid<3,MMSP::vector<fp_t> > GRID3D;

/**
   Container for system composition and phase fractions
*/
class Composition
{
public:
	// constructor
	Composition()
	{
		for (int i=0; i<NP+1; i++) {
			for (int j=0; j<NC; j++)
				x[i][j] = 0.0;
		}
		for (int i=0; i<NP+1; i++)
			N[i] = 0;
	}
	~Composition() {}

	// modifier
	Composition& operator+=(const Composition& c);

	// representation
	double x[NP+1][NC]; // composition of each phase
	int    N[NP+1];     // amount of each phase
};

/**
   Combine composition containers
*/
Composition& Composition::operator+=(const Composition& c)
{
	for (int i=0; i<NP+1; i++) {
		for (int j=0; j<NC; j++)
			this->x[i][j] += c.x[i][j];
	}
	for (int i=0; i<NP+1; i++)
		this->N[i] += c.N[i];

	return *this;
}

/**
   Compute Gibbs free energy density
*/
template<typename T>
T gibbs(const MMSP::vector<T>& v);

/**
   Compute gradient of specified field, only
*/
template <int dim, typename T>
MMSP::vector<T> maskedgradient(const MMSP::grid<dim,MMSP::vector<T> >& GRID, const MMSP::vector<int>& x, const int N);

/**
   Compute distance between MMSP coordinates
*/
double radius(const MMSP::vector<int>& a, const MMSP::vector<int>& b, const double& dx);

/**
   Compute fictitious compositions using analytical expressions
*/
template<typename T>
void update_compositions(MMSP::vector<T>& GRIDN);

/**
   Insert particle into matrix at specified location with given phase, radius, and composition
*/
template<int dim, typename T>
Composition embedParticle(MMSP::grid<dim,MMSP::vector<T> >& GRID,
                          const MMSP::vector<int>& origin,
                          const int pid,
                          const double rprcp,
                          const T& xCr, const T& xNb);

/**
   Insert stripe into matrix at specified location with given phase, width, and composition
*/
template<int dim, typename T>
Composition embedStripe(MMSP::grid<dim,MMSP::vector<T> >& GRID,
                        const MMSP::vector<int>& origin,
                        const int pid,
                        const double rprcp,
                        const T& xCr, const T& xNb);

/**
   Tile domain with two-particle boxes of uniform size and composition
*/
template<int dim, typename T>
Composition init_2D_tiles(MMSP::grid<dim,MMSP::vector<T> >& GRID, const double Ntot,
                          const int width, const int height,
                          const double xCr0, const double xNb0,
                          std::uniform_real_distribution<double>& unidist, std::mt19937& mtrand);

/**
   Integrate composition and phase fractions over the whole grid to make sure mass is conserved and phase transformations are sane
*/
template<int dim, typename T>
MMSP::vector<double> summarize_fields(MMSP::grid<dim,MMSP::vector<T> > const& GRID);

/**
   Integrate free energy over the whole grid to make sure it decreases with time
*/
template<int dim, typename T>
double summarize_energy(MMSP::grid<dim,MMSP::vector<T> > const& GRID);

#endif
