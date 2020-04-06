#include <math.h>
#include "fcfc.h"

/******************************************************************************
Function `create_dist_bin`:
  Create the distance bins given a range and the number of bins.
Input:
  The distance range of interest and the number of bins, and an array for
  storing distance bins.
Output:
  Return a non-zero integer if there is problem.

Arguments:
  * `min`:      lower limit of the distance range;
  * `max`:      upper limit of the distance range;
  * `n`:        number of distance bins;
  * `bin`:      the array for distance bins.
******************************************************************************/
int create_dist_bin(const double min, const double max, const size_t n,
    double **bin) {
  size_t i;
  double dr;

  MY_ALLOC(*bin, double, n + 1,
      "failed to allocate memory for distance bins.\n");

  dr = (max - min) / n;
  for (i = 0; i <= n; i++)
    (*bin)[i] = min + i * dr;

  return 0;
}


/******************************************************************************
Function `create_1d_counts`:
  Create the arrays for pair counts.
Input:
  The distance range and number of distance bins, and pointers to the array
  for pair counts.
Output:
  Return a non-zero integer if there is problem.

Arguments:
  * `min`:      lower limit of the distance range;
  * `max`:      upper limit of the distance range;
  * `Nbin`:     number of distance bins;
  * `r2prec`:   precision of squared distances;
  * `flag`:     `NUM_COUNT` for number counts, `WT_COUNT` for weight counts;
  * `cnt`:      the array for counts;
  * `Ncnt`:     number of bins for the count array.
******************************************************************************/
int create_1d_counts(const double min, const double max, const size_t Nbin,
    const int r2prec, const int flag, void **cnt, size_t *Ncnt) {
  int prec;
  size_t n, fac;

  // Create the auxiliary count array for approximate squared distances.
  if (r2prec >= - APPROX_R2_PREC && r2prec <= APPROX_R2_PREC) {
    fac = 1;
    prec = (r2prec >= 0) ? r2prec : -r2prec;
    while (prec != 0) {
      fac *= 10;
      prec--;
    }

    if (r2prec == 0)
      n = (size_t) (max * max) - (size_t) (min * min);
    else if (r2prec > 0)
      n = (size_t) (max * max / fac) - (size_t) (min * min / fac);
    else
      n = (size_t) (max * max * fac) - (size_t) (min * min * fac);

    if (flag == NUM_COUNT)
      *cnt = (size_t *) malloc(sizeof(size_t) * n);
    else if (flag == WT_COUNT)
      *cnt = (double *) malloc(sizeof(double) * n);

    if (!(*cnt)) {
      P_ERR("failed to allocate memory for auxiliary pair counts.\n");
      return ERR_MEM;
    }
    *Ncnt = n;
  }
  // Create the count array for exact bin counts.
  else {
    if (flag == NUM_COUNT)
      *cnt = (size_t *) malloc(sizeof(size_t) * Nbin);
    else if (flag == WT_COUNT)
      *cnt = (double *) malloc(sizeof(double) * Nbin);

    if (!(*cnt)) {
      P_ERR("failed to allocate memory for pair counts.\n");
      return ERR_MEM;
    }
    *Ncnt = Nbin;
  }
  return 0;
}


/******************************************************************************
Function `combine_1d_counts`:
  Compute pair counts of distance bins from the auxiliary array.
Input:
  The array storing pair counts and the auxiliary array for pair counts.

Arguments:
  * `r2prec`:   the precision of approximate squared distance;
  * `moment`:   flag for quadrupole correlation;
  * `flag`:     `NUM_COUNT` for number counts, `WT_COUNT` for weight counts;
  * `rbin`:     the array storing distance bins;
  * `auxcnt`:   the auxiliary array counting squared distance bins;
  * `auxqcnt`:  the auxiliary array for quadrupole;
  * `Naux`:     length of the auxiliary array;
  * `cnt`:      the array storing pair counts.
******************************************************************************/
void combine_1d_counts(const int r2prec, const int moment, const int flag,
    const double *rbin, const void *auxcnt, const DOUBLE2 *auxqcnt,
    const size_t Naux, DOUBLE2 *cnt) {
  size_t i, j;
  double fac, max;

  fac = pow(10, r2prec);
  if (flag == NUM_COUNT) {              // Number counts.
    for (i = j = 0; i < Naux; i++) {
      max = ((size_t) (rbin[0] * rbin[0] / fac) + i + 1) * fac;
      if (max <= rbin[j + 1] * rbin[j + 1]) {
        cnt[j].v[0] += ((size_t *) auxcnt)[i];
        if (moment == 2) {
          cnt[j].v[1] += auxqcnt[i].v[0];
          cnt[j].v[2] += auxqcnt[i].v[1];
        }
      }
      else {
        j++;
        i--;
      }
    }
  }
  else if (flag == WT_COUNT) {          // Weight counts.
    for (i = j = 0; i < Naux; i++) {
      max = ((size_t) (rbin[0] * rbin[0] / fac) + i + 1) * fac;
      if (max <= rbin[j + 1] * rbin[j + 1]) {
        cnt[j].v[0] += ((double *) auxcnt)[i];
        if (moment == 2) {
          cnt[j].v[1] += auxqcnt[i].v[0];
          cnt[j].v[2] += auxqcnt[i].v[1];
        }
      }
      else {
        j++;
        i--;
      }
    }
  }
}


/******************************************************************************
Function `find_1d_bin`:
  Find the index of a (squared) distance and the (squared) distance bins,
  using a binary search alhorithm.
Input:
  A (squared) distance and the (squared) distance bins.
Output:
  If the bin is found, return the index; else return -1.

Arguments:
  * `dist`:     the given distance;
  * `rbin`:     the array for distance bins;
  * `n`:        the number of distance bins.
******************************************************************************/
int find_1d_bin(const real dist, const double *rbin, const int n) {
  register int i, l, u;
  l = 0;
  u = n - 1;

  while (l <= u) {
    i = (l + u) >> 1;
    if (rbin[i + 1] <= dist) l = i + 1;
    else if (rbin[i] > dist) u = i - 1;
    else return i;
  }

  return -1;
}

