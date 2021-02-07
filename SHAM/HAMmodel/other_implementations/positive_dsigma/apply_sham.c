/*******************************************************************************
* apply_sham.c: this file is part of the mtsham program.

* mtsham: C code for multi-tracer sub-halo abundance matching.

* Github repository:
        https://github.com/cheng-zhao/mtsham

* Copyright (c) 2020 Cheng Zhao <zhaocheng03@gmail.com>  [MIT license]

*******************************************************************************/

#include "define.h"
#include "mt19937.h"
#include "apply_sham.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/*============================================================================*\
                       Functions for galaxy data handling
\*============================================================================*/

/******************************************************************************
Function `compare_mass`:
  Compare the scattered halo mass of two galaxies for qsort.
Arguments:
  * `a`:        pointer to the first galaxy;
  * `b`:        pointer to the second galaxy.
Return:
  An integer indicating the order of the two galaxies.
******************************************************************************/
static int compare_mass(const void *a, const void *b) {
  if (((DATA *) a)->ms > ((DATA *) b)->ms) return 1;
  else if (((DATA *) a)->ms < ((DATA *) b)->ms) return -1;
  else return 0;
}

/******************************************************************************
Function `assign_gal`:
  Assign galaxy to a halo.
Arguments:
  * `halo`:     the halo to be assigned galaxy;
  * `mass`:     halo mass after scattering;
  * `gal`:      address for storing the galaxy.
******************************************************************************/
static inline void assign_gal(const HDATA *halo, const real mass,
    DATA *gal) {
  gal->x[0] = halo->x[0];
  gal->x[1] = halo->x[1];
  gal->x[2] = halo->x[2];
  gal->v = halo->v;
  gal->m = halo->m;
  gal->ms = mass;
}

/*============================================================================*\
                          Interface for applying SHAM
\*============================================================================*/

/******************************************************************************
Function `apply_sham`:
  Apply the sub-halo abundance matching algorithm.
Arguments:
  * `seed`:     the random seed;
  * `sigm`:     sigma of the Gaussian random number for scattering mass;
  * `sigv`:     sigma of the Gaussian random number for scattering velocity;
  * `mceil`:    maximum allowed mass for assigning galaxies;
  * `halo`:     the input halo catalogue;
  * `nhalo`:    number of input haloes;
  * `bsize`:    box size for the halo catalogue;
  * `rsd`:      indicate whether to apply redshift space distortions;
  * `rsdfac`:   factor for applying redshift space distortions;
  * `gal`:      the output galaxy catalogue;
  * `ngal`:     number of output galaxies.
Return:
  Number of galaxies generated.
******************************************************************************/
int apply_sham(const uint64_t seed, const double sigm, const double sigv,
    const double mceil, const HDATA *halo, const size_t nhalo,
    const double bsize, const bool rsd, const double rsdfac,
    DATA *gal, const int ngal) {
  /* Initialise the random number generator. */
  mt19937_t ran;
  mt19937_seed(&ran, seed);

  /* Apply sub-halo abundance matching for the first ngal haloes. */
  size_t cnth;
  int cntg = 0;
  for (cnth = 0; cnth < nhalo; cnth++) {
    /* Add scatter to the halo mass. */
    double rand = mt19937_get_gauss(&ran, 0, sigm);
    real mass = (rand >= 0) ? halo[cnth].m * (1 + rand)/sigm :
      halo[cnth].m * exp(rand)/sigm;
    /* Mass ceiling. */
    if (mass > mceil) continue;
    assign_gal(halo + cnth, mass, gal + cntg);
    if (++cntg == ngal) break;
  }

  if (cntg != ngal) return cntg;

  /* Sort galaxies with the scattered mass in ascending order. */
  if (cnth != nhalo) qsort(gal, ngal, sizeof(DATA), compare_mass);

  /* Apply SHAM to the rest of haloes and keep the largest ngal ones. */
  for (size_t i = cnth + 1; i < nhalo; i++) {
    double rand = mt19937_get_gauss(&ran, 0, sigm);
    real mass = (rand >= 0) ? halo[i].m * (1 + rand)/sigm :
      halo[i].m * exp(rand)/sigm;
    if (mass > mceil) continue;
    /* Assign a galaxy only if the scattered mass is large enough. */
    if (mass > gal[0].ms) {
      assign_gal(halo + i, mass, gal);
      /* Insert the newly added halo to the heap. */
      for (int j = 0; ; ) {
        int k = (j << 1) + 1;
        if (k > ngal - 1) break;
        if (k != ngal - 1 && gal[k].ms > gal[k + 1].ms) k += 1;
        if (gal[j].ms <= gal[k].ms) break;

        /* Swap the two galaxies: using the last element temporarily. */
        memcpy(gal + ngal, gal + j, sizeof(DATA));
        memcpy(gal + j, gal + k, sizeof(DATA));
        memcpy(gal + k, gal + ngal, sizeof(DATA));

        j = k;
      }
    }
  }

  /* Apply redshift space distortions to the selected galaxies. */
  if (rsd) {
    for (int i = 0; i < cntg; i++) {
      /* Add scatter to the peculiar velocity. */
      gal[i].v += mt19937_get_gauss(&ran, 0, sigv);
      gal[i].x[2] += gal[i].v * rsdfac;
      if (gal[i].x[2] < 0) gal[i].x[2] += bsize;
      else if (gal[i].x[2] >= bsize) gal[i].x[2] -= bsize;
    }
  }

  return cntg;
}
