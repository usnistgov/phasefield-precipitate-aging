/**
 \file  mesh.c
 \brief Implemenatation of mesh handling functions
*/

#include <stdio.h>
#include <stdlib.h>
#include "mesh.h"

void make_arrays(struct HostData* host,
                 const int nx, const int ny, const int nm)
{
	int i;

	/* create 2D pointers */
	host->conc_Cr_old = (fp_t**)calloc(nx, sizeof(fp_t*));
	host->conc_Cr_new = (fp_t**)calloc(nx, sizeof(fp_t*));
	host->conc_Nb_old = (fp_t**)calloc(nx, sizeof(fp_t*));
	host->conc_Nb_new = (fp_t**)calloc(nx, sizeof(fp_t*));
	host->conc_Ni     = (fp_t**)calloc(nx, sizeof(fp_t*));

	host->phi_del_old = (fp_t**)calloc(nx, sizeof(fp_t*));
	host->phi_del_new = (fp_t**)calloc(nx, sizeof(fp_t*));
	host->phi_lav_old = (fp_t**)calloc(nx, sizeof(fp_t*));
	host->phi_lav_new = (fp_t**)calloc(nx, sizeof(fp_t*));
	host->phi         = (fp_t**)calloc(nx, sizeof(fp_t*));

	host->gam_Cr = (fp_t**)calloc(nx, sizeof(fp_t*));
	host->gam_Nb = (fp_t**)calloc(nx, sizeof(fp_t*));

	host->mask_lap = (fp_t**)calloc(nm, sizeof(fp_t*));

	/* allocate 1D data */
	(host->conc_Cr_old)[0] = (fp_t*)calloc(nx * ny, sizeof(fp_t));
	(host->conc_Cr_new)[0] = (fp_t*)calloc(nx * ny, sizeof(fp_t));
	(host->conc_Nb_old)[0] = (fp_t*)calloc(nx * ny, sizeof(fp_t));
	(host->conc_Nb_new)[0] = (fp_t*)calloc(nx * ny, sizeof(fp_t));
	(host->conc_Ni)[0]     = (fp_t*)calloc(nx * ny, sizeof(fp_t));

	(host->phi_del_old)[0] = (fp_t*)calloc(nx * ny, sizeof(fp_t));
	(host->phi_del_new)[0] = (fp_t*)calloc(nx * ny, sizeof(fp_t));
	(host->phi_lav_old)[0] = (fp_t*)calloc(nx * ny, sizeof(fp_t));
	(host->phi_lav_new)[0] = (fp_t*)calloc(nx * ny, sizeof(fp_t));
	(host->phi)[0]         = (fp_t*)calloc(nx * ny, sizeof(fp_t));

	(host->gam_Cr)[0] = (fp_t*)calloc(nx * ny, sizeof(fp_t));
	(host->gam_Nb)[0] = (fp_t*)calloc(nx * ny, sizeof(fp_t));

	(host->mask_lap)[0] = (fp_t*)calloc(nm * nm, sizeof(fp_t));

	/* map 2D pointers onto 1D data */
	for (i = 1; i < ny; i++) {
		(host->conc_Cr_old)[i] = &(host->conc_Cr_old[0])[nx * i];
		(host->conc_Cr_new)[i] = &(host->conc_Cr_new[0])[nx * i];
		(host->conc_Nb_old)[i] = &(host->conc_Nb_old[0])[nx * i];
		(host->conc_Nb_new)[i] = &(host->conc_Nb_new[0])[nx * i];
		(host->conc_Ni)[i]     = &(host->conc_Ni[0]    )[nx * i];

		(host->phi_del_old)[i] = &(host->phi_del_old[0])[nx * i];
		(host->phi_del_new)[i] = &(host->phi_del_new[0])[nx * i];
		(host->phi_lav_old)[i] = &(host->phi_lav_old[0])[nx * i];
		(host->phi_lav_new)[i] = &(host->phi_lav_new[0])[nx * i];
		(host->phi)[i]         = &(host->phi[0]    )[nx * i];

		(host->gam_Cr)[i] = &(host->gam_Cr[0])[nx * i];
		(host->gam_Nb)[i] = &(host->gam_Nb[0])[nx * i];
	}

	for (i = 1; i < nm; i++) {
		(host->mask_lap)[i] = &(host->mask_lap[0])[nm * i];
	}
}

void free_arrays(struct HostData* host)
{
	free((host->conc_Cr_old)[0]);
	free((host->conc_Cr_new)[0]);
	free((host->conc_Nb_old)[0]);
	free((host->conc_Nb_new)[0]);
	free((host->conc_Ni)[0]);
	free((host->phi_del_old)[0]);
	free((host->phi_del_new)[0]);
	free((host->phi_lav_old)[0]);
	free((host->phi_lav_new)[0]);
	free((host->phi)[0]);
	free((host->gam_Cr)[0]);
	free((host->gam_Nb)[0]);
	free((host->mask_lap)[0]);

	free(host->conc_Cr_old);
	free(host->conc_Cr_new);
	free(host->conc_Nb_old);
	free(host->conc_Nb_new);
	free(host->conc_Ni);
	free(host->phi_del_old);
	free(host->phi_del_new);
	free(host->phi_lav_old);
	free(host->phi_lav_new);
	free(host->phi);
	free(host->gam_Cr);
	free(host->gam_Nb);
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
