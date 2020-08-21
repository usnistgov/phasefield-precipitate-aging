/**
 \file  mesh.cpp
 \brief Implemenatation of mesh handling functions
*/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mesh.h"

void make_arrays(struct HostData* host,
                 const int nx, const int ny, const int nm)
{
	int i;

	/* create 2D pointers */
	host->conc_Cr_old = (fp_t**)calloc(nx, sizeof(fp_t*));
	host->conc_Cr_new = (fp_t**)calloc(nx, sizeof(fp_t*));
	host->conc_Cr_lap = (fp_t**)calloc(nx, sizeof(fp_t*));
	host->conc_Cr_gam = (fp_t**)calloc(nx, sizeof(fp_t*));

	host->conc_Nb_old = (fp_t**)calloc(nx, sizeof(fp_t*));
	host->conc_Nb_new = (fp_t**)calloc(nx, sizeof(fp_t*));
	host->conc_Nb_lap = (fp_t**)calloc(nx, sizeof(fp_t*));
	host->conc_Nb_gam = (fp_t**)calloc(nx, sizeof(fp_t*));

	host->conc_Ni     = (fp_t**)calloc(nx, sizeof(fp_t*));

	host->phi_del_old = (fp_t**)calloc(nx, sizeof(fp_t*));
	host->phi_del_new = (fp_t**)calloc(nx, sizeof(fp_t*));
	host->phi_del_lap = (fp_t**)calloc(nx, sizeof(fp_t*));

	host->phi_lav_old = (fp_t**)calloc(nx, sizeof(fp_t*));
	host->phi_lav_new = (fp_t**)calloc(nx, sizeof(fp_t*));
	host->phi_lav_lap = (fp_t**)calloc(nx, sizeof(fp_t*));

	host->phi         = (fp_t**)calloc(nx, sizeof(fp_t*));

	host->chem_nrg     = (fp_t**)calloc(nx, sizeof(fp_t*));
	host->grad_nrg     = (fp_t**)calloc(nx, sizeof(fp_t*));

	host->mask_lap = (fp_t**)calloc(nm, sizeof(fp_t*));

	/* allocate 1D data */
	(host->conc_Cr_old)[0] = (fp_t*)calloc(nx * ny, sizeof(fp_t));
	(host->conc_Cr_new)[0] = (fp_t*)calloc(nx * ny, sizeof(fp_t));
	(host->conc_Cr_lap)[0] = (fp_t*)calloc(nx * ny, sizeof(fp_t));
	(host->conc_Cr_gam)[0] = (fp_t*)calloc(nx * ny, sizeof(fp_t));
	(host->conc_Nb_old)[0] = (fp_t*)calloc(nx * ny, sizeof(fp_t));
	(host->conc_Nb_new)[0] = (fp_t*)calloc(nx * ny, sizeof(fp_t));
	(host->conc_Nb_lap)[0] = (fp_t*)calloc(nx * ny, sizeof(fp_t));
	(host->conc_Nb_gam)[0] = (fp_t*)calloc(nx * ny, sizeof(fp_t));
	(host->conc_Ni)[0]     = (fp_t*)calloc(nx * ny, sizeof(fp_t));

	(host->phi_del_old)[0] = (fp_t*)calloc(nx * ny, sizeof(fp_t));
	(host->phi_del_new)[0] = (fp_t*)calloc(nx * ny, sizeof(fp_t));
	(host->phi_del_lap)[0] = (fp_t*)calloc(nx * ny, sizeof(fp_t));
	(host->phi_lav_old)[0] = (fp_t*)calloc(nx * ny, sizeof(fp_t));
	(host->phi_lav_new)[0] = (fp_t*)calloc(nx * ny, sizeof(fp_t));
	(host->phi_lav_lap)[0] = (fp_t*)calloc(nx * ny, sizeof(fp_t));

	(host->phi)[0]         = (fp_t*)calloc(nx * ny, sizeof(fp_t));

	(host->chem_nrg)[0]     = (fp_t*)calloc(nx * ny, sizeof(fp_t));
	(host->grad_nrg)[0]     = (fp_t*)calloc(nx * ny, sizeof(fp_t));

	(host->mask_lap)[0] = (fp_t*)calloc(nm * nm, sizeof(fp_t));

	/* map 2D pointers onto 1D data */
	for (i = 1; i < ny; i++) {
		(host->conc_Cr_old)[i] = &(host->conc_Cr_old[0])[nx * i];
		(host->conc_Cr_new)[i] = &(host->conc_Cr_new[0])[nx * i];
		(host->conc_Cr_lap)[i] = &(host->conc_Cr_lap[0])[nx * i];
		(host->conc_Cr_gam)[i] = &(host->conc_Cr_gam[0])[nx * i];
		(host->conc_Nb_old)[i] = &(host->conc_Nb_old[0])[nx * i];
		(host->conc_Nb_new)[i] = &(host->conc_Nb_new[0])[nx * i];
		(host->conc_Nb_lap)[i] = &(host->conc_Nb_lap[0])[nx * i];
		(host->conc_Nb_gam)[i] = &(host->conc_Nb_gam[0])[nx * i];
		(host->conc_Ni)[i]     = &(host->conc_Ni[0]    )[nx * i];

		(host->phi_del_old)[i] = &(host->phi_del_old[0])[nx * i];
		(host->phi_del_new)[i] = &(host->phi_del_new[0])[nx * i];
		(host->phi_del_lap)[i] = &(host->phi_del_lap[0])[nx * i];
		(host->phi_lav_old)[i] = &(host->phi_lav_old[0])[nx * i];
		(host->phi_lav_new)[i] = &(host->phi_lav_new[0])[nx * i];
		(host->phi_lav_lap)[i] = &(host->phi_lav_lap[0])[nx * i];
		(host->phi)[i]         = &(host->phi[0]        )[nx * i];

		(host->chem_nrg)[i]    = &(host->chem_nrg[0]    )[nx * i];
		(host->grad_nrg)[i]    = &(host->chem_nrg[0]    )[nx * i];
	}

	for (i = 1; i < nm; i++) {
		(host->mask_lap)[i] = &(host->mask_lap[0])[nm * i];
	}
}

void free_arrays(struct HostData* host)
{
	free((host->conc_Cr_old)[0]);
	free((host->conc_Cr_new)[0]);
	free((host->conc_Cr_lap)[0]);
	free((host->conc_Cr_gam)[0]);

	free((host->conc_Nb_old)[0]);
	free((host->conc_Nb_new)[0]);
	free((host->conc_Nb_lap)[0]);
	free((host->conc_Nb_gam)[0]);

	free((host->conc_Ni)[0]);

	free((host->phi_del_old)[0]);
	free((host->phi_del_new)[0]);
	free((host->phi_del_lap)[0]);

	free((host->phi_lav_old)[0]);
	free((host->phi_lav_new)[0]);
	free((host->phi_lav_lap)[0]);

	free((host->phi)[0]);

	free((host->chem_nrg)[0]);
	free((host->grad_nrg)[0]);

	free((host->mask_lap)[0]);

	free(host->conc_Cr_old);
	free(host->conc_Cr_new);
	free(host->conc_Cr_lap);
	free(host->conc_Cr_gam);

	free(host->conc_Nb_old);
	free(host->conc_Nb_new);
	free(host->conc_Nb_lap);
	free(host->conc_Nb_gam);

	free(host->conc_Ni);

	free(host->phi_del_old);
	free(host->phi_del_new);
	free(host->phi_del_lap);

	free(host->phi_lav_old);
	free(host->phi_lav_new);
	free(host->phi_lav_lap);

	free(host->phi);

	free(host->chem_nrg);
	free(host->grad_nrg);

	free(host->mask_lap);
}

void swap_pointers(fp_t** * conc_old, fp_t** * conc_new)
{
	fp_t** temp = *conc_old;
	*conc_old   = *conc_new;
	*conc_new   = temp;
}

void swap_pointers_1D(fp_t** conc_old, fp_t** conc_new)
{
	fp_t* temp = *conc_old;
	*conc_old  = *conc_new;
	*conc_new  = temp;
}

void set_mask(const fp_t dx, const fp_t dy, const int code, fp_t** mask_lap, const int nm)
{
	switch (code) {
	case 53:
		five_point_Laplacian_stencil(dx, dy, mask_lap, nm);
		break;
	case 93:
		nine_point_Laplacian_stencil(dx, dy, mask_lap, nm);
		break;
	default :
		five_point_Laplacian_stencil(dx, dy, mask_lap, nm);
	}

	assert(nm <= MAX_MASK_W);
	assert(nm <= MAX_MASK_H);
}

void five_point_Laplacian_stencil(const fp_t dx, const fp_t dy, fp_t** mask_lap, const int nm)
{
	assert(nm == 3);

	mask_lap[0][1] =  1. / (dy * dy); /* upper */
	mask_lap[1][0] =  1. / (dx * dx); /* left */
	mask_lap[1][1] = -2. * (dx*dx + dy*dy) / (dx*dx * dy*dy); /* middle */
	mask_lap[1][2] =  1. / (dx * dx); /* right */
	mask_lap[2][1] =  1. / (dy * dy); /* lower */
}

void nine_point_Laplacian_stencil(const fp_t dx, const fp_t dy, fp_t** mask_lap, const int nm)
{
	assert(nm == 3);

	mask_lap[0][0] =   1. / (6. * dx * dy); /* upper-left */
	mask_lap[0][1] =   4. / (6. * dy * dy); /* upper-middle */
	mask_lap[0][2] =   1. / (6. * dx * dy); /* upper-right */

	mask_lap[1][0] =   4. / (6. * dx * dx); /* middle-left */
	mask_lap[1][1] = -10. * (dx*dx + dy*dy) / (6. * dx*dx * dy*dy); /* middle */
	mask_lap[1][2] =   4. / (6. * dx * dx); /* middle-right */

	mask_lap[2][0] =   1. / (6. * dx * dy); /* lower-left */
	mask_lap[2][1] =   4. / (6. * dy * dy); /* lower-middle */
	mask_lap[2][2] =   1. / (6. * dx * dy); /* lower-right */
}

fp_t grad_sq(fp_t** conc, const int x, const int y,
             const fp_t dx, const fp_t dy,
             const int nx, const int ny)
{
	assert(x > 0 && x < nx-1);
	assert(y > 0 && y < ny-1);

	int i=x, j=y;
	fp_t gsq=0.;

	/* gradient in x-direction */
	i += 1;
	const fp_t xhi = conc[j][i];
	i -= 2;
	const fp_t xlo = conc[j][i];
	i += 1;

	gsq += (xhi - xlo) * (xhi - xlo) / (4. * dx * dx);

	/* gradient in y-direction */
	j += 1;
	const fp_t yhi = conc[j][i];
	j -= 2;
	const fp_t ylo = conc[j][i];
	j += 1;

	gsq += (yhi - ylo) * (yhi - ylo) / (4. * dy * dy);

	return gsq;
}
