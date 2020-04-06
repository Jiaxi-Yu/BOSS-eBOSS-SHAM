#include "kdtree.h"

extern inline real squared_distance(const DATA *, const DATA *);
extern inline real min_squared_dist_between_box(const DATA *, const DATA *,
    const DATA *, const DATA *);
extern inline real max_squared_dist_between_box(const DATA *, const DATA *,
    const DATA *, const DATA *);

/******************************************************************************
Function `kdtree_count_1d_auto_pairs`:
  Count 1-D auto pairs based on the k-D tree data structure.
Input:
  Configurations and arrays storing pair counts.

Arguments:
  * `tree`:     the k-D tree;
  * `flag_wt`:  0 for number counts, non-zero for weight counts;
  * `rmin`:     the minimum distance of interest;
  * `rmax`:     the maximum distance of interest;
  * `rnum`:     the number of distance bins;
  * `r2prec`:   the precision of squared distances;
  * `rbin`:     the array for distance bins;
  * `rbin2`:    the array for squared distance bins;
  * `Ncnt`:     number of bins for the counting array;
  * `Nomp`:     number of OpenMP threads;
  * `ncnt`:     the number counting array;
  * `wcnt`:     the weight counting array;
  * `out`:      the output array for pair counts.
******************************************************************************/
void kdtree_count_1d_auto_pairs(const KDT *tree, const int flag_wt,
    const double rmin, const double rmax, const int rnum, const int r2prec,
    const int moment, const double *rbin, const double *rbin2,
    const size_t Ncnt,
#ifdef OMP
    const int Nomp, size_t **ncnt, double **wcnt, DOUBLE2 **qcnt,
#else
    size_t *ncnt, double *wcnt, DOUBLE2 *qcnt,
#endif
    DOUBLE2 *out) {
#ifdef OMP
  int j;
  size_t *tncnt;
  double *twcnt;
  DOUBLE2 *tqcnt;

  tncnt = NULL;
  twcnt = NULL;
  tqcnt = NULL;
#endif
  size_t i;
  double prec;

  if (moment == 2) {    // Quadrupole.
#ifdef OMP
    for (i = 0; i < Nomp; i++)
      memset(qcnt[i], 0, sizeof(DOUBLE2) * Ncnt);
#else
    memset(qcnt, 0, sizeof(DOUBLE2) * Ncnt);
#endif
  }

  if (flag_wt) {        // Counting with weights.
#ifdef OMP
    for (i = 0; i < Nomp; i++)
      memset(wcnt[i], 0, sizeof(double) * Ncnt);
#else
    memset(wcnt, 0, sizeof(double) * Ncnt);
#endif

    // Exact distance bins.
    if (r2prec < -APPROX_R2_PREC || r2prec > APPROX_R2_PREC) {
      if (moment == 2)
#ifdef OMP
#pragma omp parallel
#pragma omp single
#endif
        kdtree_dual_leaf_auto_quad_exact_wt(tree, tree, rbin2, rnum, qcnt,
            wcnt);
      else
#ifdef OMP
#pragma omp parallel
#pragma omp single
#endif
        kdtree_dual_leaf_auto_exact_wt(tree, tree, rbin2, rnum, wcnt);
#ifdef OMP
      for (i = 0; i < Ncnt; i++)
        for (j = 0; j < Nomp; j++) {
          out[i].v[0] += wcnt[j][i];
          if (moment == 2) {
            out[i].v[1] += qcnt[j][i].v[0];
            out[i].v[2] += qcnt[j][i].v[1];
          }
        }
#else
      for (i = 0; i < Ncnt; i++) {
        out[i].v[0] = wcnt[i];
        if (moment == 2) {
          out[i].v[1] = qcnt[i].v[0];
          out[i].v[2] = qcnt[i].v[1];
        }
      }
#endif
    }
    else {              // Approximate distance bins.
#ifdef OMP
      twcnt = (double *) malloc(sizeof(double) * Ncnt);
      if (!twcnt) {
        P_EXT("failed to allocate memory for pair counts.\n");
        exit(ERR_MEM);
      }
      memset(twcnt, 0, sizeof(double) * Ncnt);
      if (moment == 2) {
        tqcnt = (DOUBLE2 *) malloc(sizeof(DOUBLE2) * Ncnt);
        if (!tqcnt) {
          P_EXT("failed to allocate memory for pair counts.\n");
          exit(ERR_MEM);
        }
        memset(tqcnt, 0, sizeof(DOUBLE2) * Ncnt);
      }
#endif
      if (r2prec == 0) {                // Integer distance bins.
        if (rmin < FLT_ERR) {
          if (moment == 2)
#ifdef OMP
#pragma omp parallel
#pragma omp single
#endif
            kdtree_dual_leaf_auto_quad_intbin_rmin0_wt(tree, tree, rmax * rmax,
                qcnt, wcnt);
          else
#ifdef OMP
#pragma omp parallel
#pragma omp single
#endif
            kdtree_dual_leaf_auto_intbin_rmin0_wt(tree, tree, rmax * rmax,
                wcnt);
        }
        else {
          if (moment == 2)
#ifdef OMP
#pragma omp parallel
#pragma omp single
#endif
            kdtree_dual_leaf_auto_quad_intbin_wt(tree, tree, rmin * rmin,
                rmax * rmax, qcnt, wcnt);
          else
#ifdef OMP
#pragma omp parallel
#pragma omp single
#endif
            kdtree_dual_leaf_auto_intbin_wt(tree, tree, rmin * rmin,
                rmax * rmax, wcnt);
        }
      }
      else {
        prec = pow(10, -r2prec);
        if (rmin < FLT_ERR) {
          if (moment == 2)
#ifdef OMP
#pragma omp parallel
#pragma omp single
#endif
            kdtree_dual_leaf_auto_quad_approx_rmin0_wt(tree, tree, rmax * rmax,
                prec, qcnt, wcnt);
          else
#ifdef OMP
#pragma omp parallel
#pragma omp single
#endif
            kdtree_dual_leaf_auto_approx_rmin0_wt(tree, tree, rmax * rmax,
                prec, wcnt);
        }
        else {
          if (moment == 2)
#ifdef OMP
#pragma omp parallel
#pragma omp single
#endif
            kdtree_dual_leaf_auto_quad_approx_wt(tree, tree, rmin * rmin,
                rmax * rmax, prec, qcnt, wcnt);
          else
#ifdef OMP
#pragma omp parallel
#pragma omp single
#endif
            kdtree_dual_leaf_auto_approx_wt(tree, tree, rmin * rmin,
                rmax * rmax, prec, wcnt);
        }
      }

      // Combine the counts.
#ifdef OMP
      for (i = 0; i < Ncnt; i++)
        for (j = 0; j < Nomp; j++) {
          twcnt[i] += wcnt[j][i];
          if (moment == 2) {
            tqcnt[i].v[0] += qcnt[j][i].v[0];
            tqcnt[i].v[1] += qcnt[j][i].v[1];
          }
        }
      combine_1d_counts(r2prec, moment, WT_COUNT, rbin, twcnt, tqcnt, Ncnt,
          out);
      free(twcnt);
      if (moment == 2) free(tqcnt);
#else
      combine_1d_counts(r2prec, moment, WT_COUNT, rbin, wcnt, qcnt, Ncnt, out);
#endif
    }
  }
  else {                // Counting with numbers.
#ifdef OMP
    for (i = 0; i < Nomp; i++)
      memset(ncnt[i], 0, sizeof(size_t) * Ncnt);
#else
    memset(ncnt, 0, sizeof(size_t) * Ncnt);
#endif

    // Exact distance bins.
    if (r2prec < -APPROX_R2_PREC || r2prec > APPROX_R2_PREC) {
      if (moment == 2)
#ifdef OMP
#pragma omp parallel
#pragma omp single
#endif
        kdtree_dual_leaf_auto_quad_exact(tree, tree, rbin2, rnum, qcnt, ncnt);
      else
#ifdef OMP
#pragma omp parallel
#pragma omp single
#endif
        kdtree_dual_leaf_auto_exact(tree, tree, rbin2, rnum, ncnt);
#ifdef OMP
      for (i = 0; i < Ncnt; i++)
        for (j = 0; j < Nomp; j++) {
          out[i].v[0] += ncnt[j][i];
          if (moment == 2) {
            out[i].v[1] += qcnt[j][i].v[0];
            out[i].v[2] += qcnt[j][i].v[1];
          }
        }
#else
      for (i = 0; i < Ncnt; i++) {
        out[i].v[0] = ncnt[i];
        if (moment == 2) {
          out[i].v[1] = qcnt[i].v[0];
          out[i].v[2] = qcnt[i].v[1];
        }
      }
#endif
    }
    else {              // Approximate distance bins.
#ifdef OMP
      tncnt = (size_t *) malloc(sizeof(size_t) * Ncnt);
      if (!tncnt) {
        P_EXT("failed to allocate memory for pair counts.\n");
        exit(ERR_MEM);
      }
      memset(tncnt, 0, sizeof(size_t) * Ncnt);
      if (moment == 2) {
        tqcnt = (DOUBLE2 *) malloc(sizeof(DOUBLE2) * Ncnt);
        if (!tqcnt) {
          P_EXT("failed to allocate memory for pair counts.\n");
          exit(ERR_MEM);
        }
        memset(tqcnt, 0, sizeof(DOUBLE2) * Ncnt);
      }
#endif
      if (r2prec == 0) {          // Integer distance bins.
        if (rmin < FLT_ERR) {
          if (moment == 2)
#ifdef OMP
#pragma omp parallel
#pragma omp single
#endif
            kdtree_dual_leaf_auto_quad_intbin_rmin0(tree, tree, rmax * rmax,
                qcnt, ncnt);
          else
#ifdef OMP
#pragma omp parallel
#pragma omp single
#endif
            kdtree_dual_leaf_auto_intbin_rmin0(tree, tree, rmax * rmax, ncnt);
        }
        else {
          if (moment == 2)
#ifdef OMP
#pragma omp parallel
#pragma omp single
#endif
            kdtree_dual_leaf_auto_quad_intbin(tree, tree, rmin * rmin,
                rmax * rmax, qcnt, ncnt);
          else
#ifdef OMP
#pragma omp parallel
#pragma omp single
#endif
            kdtree_dual_leaf_auto_intbin(tree, tree, rmin * rmin, rmax * rmax,
                ncnt);
        }
      }
      else {
        prec = pow(10, -r2prec);
        if (rmin < FLT_ERR) {
          if (moment == 2)
#ifdef OMP
#pragma omp parallel
#pragma omp single
#endif
            kdtree_dual_leaf_auto_quad_approx_rmin0(tree, tree, rmax * rmax,
                prec, qcnt, ncnt);
          else
#ifdef OMP
#pragma omp parallel
#pragma omp single
#endif
            kdtree_dual_leaf_auto_approx_rmin0(tree, tree, rmax * rmax,
                prec, ncnt);
        }
        else {
          if (moment == 2)
#ifdef OMP
#pragma omp parallel
#pragma omp single
#endif
            kdtree_dual_leaf_auto_quad_approx(tree, tree, rmin * rmin,
                rmax * rmax, prec, qcnt, ncnt);
          else
#ifdef OMP
#pragma omp parallel
#pragma omp single
#endif
            kdtree_dual_leaf_auto_approx(tree, tree, rmin * rmin, rmax * rmax,
                prec, ncnt);
        }
      }

      // Combine the counts.
#ifdef OMP
      for (i = 0; i < Ncnt; i++)
        for (j = 0; j < Nomp; j++) {
          tncnt[i] += ncnt[j][i];
          if (moment == 2) {
            tqcnt[i].v[0] += qcnt[j][i].v[0];
            tqcnt[i].v[1] += qcnt[j][i].v[1];
          }
        }
      combine_1d_counts(r2prec, moment, NUM_COUNT, rbin, tncnt, tqcnt, Ncnt,
          out);
      free(tncnt);
      if (moment == 2) free(tqcnt);
#else
      combine_1d_counts(r2prec, moment, NUM_COUNT, rbin, ncnt, qcnt, Ncnt, out);
#endif
    }
  }

  for (i = 0; i < rnum; i++) {          // No double counting.
    out[i].v[0] *= 2;
    out[i].v[1] *= 2;
  }
}


/******************************************************************************
Function `kdtree_count_1d_cross_pairs`:
  Count 1-D cross pairs based on the k-D tree data structure.
Input:
  Configurations and arrays storing pair counts.

Arguments:
  * `tree1`:    the first k-D tree;
  * `tree2`:    the second k-D tree;
  * `flag_wt`:  0 for number counts, non-zero for weight counts;
  * `rmin`:     the minimum distance of interest;
  * `rmax`:     the maximum distance of interest;
  * `rnum`:     the number of distance bins;
  * `r2prec`:   the precision of squared distances;
  * `rbin`:     the array for distance bins;
  * `rbin2`:    the array for squared distance bins;
  * `Ncnt`:     number of bins for the counting array;
  * `Nomp`:     number of OpenMP threads;
  * `ncnt`:     the number counting array;
  * `wcnt`:     the weight counting array;
  * `out`:      the output array for pair counts.
******************************************************************************/
void kdtree_count_1d_cross_pairs(const KDT *tree1, const KDT *tree2,
    const int flag_wt, const double rmin, const double rmax, const int rnum,
    const int r2prec, const int moment, const double *rbin,
    const double *rbin2, const size_t Ncnt,
#ifdef OMP
    const int Nomp, size_t **ncnt, double **wcnt, DOUBLE2 **qcnt,
#else
    size_t *ncnt, double *wcnt, DOUBLE2 *qcnt,
#endif
    DOUBLE2 *out) {
#ifdef OMP
  int j;
  size_t *tncnt;
  double *twcnt;
  DOUBLE2 *tqcnt;

  tncnt = NULL;
  twcnt = NULL;
  tqcnt = NULL;
#endif
  size_t i;
  double prec;

  if (moment == 2) {    // Quadrupole.
#ifdef OMP
    for (i = 0; i < Nomp; i++)
      memset(qcnt[i], 0, sizeof(DOUBLE2) * Ncnt);
#else
    memset(qcnt, 0, sizeof(DOUBLE2) * Ncnt);
#endif
  }

  if (flag_wt) {        // Counting with weights.
#ifdef OMP
    for (i = 0; i < Nomp; i++)
      memset(wcnt[i], 0, sizeof(double) * Ncnt);
#else
    memset(wcnt, 0, sizeof(double) * Ncnt);
#endif

    // Exact distance bins.
    if (r2prec < -APPROX_R2_PREC || r2prec > APPROX_R2_PREC) {
      if (moment == 2)
#ifdef OMP
#pragma omp parallel
#pragma omp single
#endif
        kdtree_dual_leaf_cross_quad_exact_wt(tree1, tree2, rbin2, rnum, qcnt,
            wcnt);
      else
#ifdef OMP
#pragma omp parallel
#pragma omp single
#endif
        kdtree_dual_leaf_cross_exact_wt(tree1, tree2, rbin2, rnum, wcnt);
#ifdef OMP
      for (i = 0; i < Ncnt; i++)
        for (j = 0; j < Nomp; j++) {
          out[i].v[0] += wcnt[j][i];
          if (moment == 2) {
            out[i].v[1] += qcnt[j][i].v[0];
            out[i].v[2] += qcnt[j][i].v[1];
          }
        }
#else
      for (i = 0; i < Ncnt; i++) {
        out[i].v[0] = wcnt[i];
        if (moment == 2) {
          out[i].v[1] = qcnt[i].v[0];
          out[i].v[2] = qcnt[i].v[1];
        }
      }
#endif
    }
    else {              // Approximate distance bins.
#ifdef OMP
      twcnt = (double *) malloc(sizeof(double) * Ncnt);
      if (!twcnt) {
        P_EXT("failed to allocate memory for pair counts.\n");
        exit(ERR_MEM);
      }
      memset(twcnt, 0, sizeof(double) * Ncnt);
      if (moment == 2) {
        tqcnt = (DOUBLE2 *) malloc(sizeof(DOUBLE2) * Ncnt);
        if (!tqcnt) {
          P_EXT("failed to allocate memory for pair counts.\n");
          exit(ERR_MEM);
        }
        memset(tqcnt, 0, sizeof(DOUBLE2) * Ncnt);
      }
#endif
      if (r2prec == 0) {                // Integer distance bins.
        if (rmin < FLT_ERR) {
          if (moment == 2)
#ifdef OMP
#pragma omp parallel
#pragma omp single
#endif
            kdtree_dual_leaf_cross_quad_intbin_rmin0_wt(tree1, tree2,
                rmax * rmax, qcnt, wcnt);
          else
#ifdef OMP
#pragma omp parallel
#pragma omp single
#endif
            kdtree_dual_leaf_cross_intbin_rmin0_wt(tree1, tree2, rmax * rmax,
                wcnt);
        }
        else {
          if (moment == 2)
#ifdef OMP
#pragma omp parallel
#pragma omp single
#endif
            kdtree_dual_leaf_cross_quad_intbin_wt(tree1, tree2, rmin * rmin,
                rmax * rmax, qcnt, wcnt);
          else
#ifdef OMP
#pragma omp parallel
#pragma omp single
#endif
            kdtree_dual_leaf_cross_intbin_wt(tree1, tree2, rmin * rmin,
                rmax * rmax, wcnt);
        }
      }
      else {
        prec = pow(10, -r2prec);
        if (rmin < FLT_ERR) {
          if (moment == 2)
#ifdef OMP
#pragma omp parallel
#pragma omp single
#endif
            kdtree_dual_leaf_cross_quad_approx_rmin0_wt(tree1, tree2,
                rmax * rmax, prec, qcnt, wcnt);
          else
#ifdef OMP
#pragma omp parallel
#pragma omp single
#endif
            kdtree_dual_leaf_cross_approx_rmin0_wt(tree1, tree2, rmax * rmax,
                prec, wcnt);
        }
        else {
          if (moment == 2)
#ifdef OMP
#pragma omp parallel
#pragma omp single
#endif
            kdtree_dual_leaf_cross_quad_approx_wt(tree1, tree2, rmin * rmin,
                rmax * rmax, prec, qcnt, wcnt);
          else
#ifdef OMP
#pragma omp parallel
#pragma omp single
#endif
            kdtree_dual_leaf_cross_approx_wt(tree1, tree2, rmin * rmin,
                rmax * rmax, prec, wcnt);
        }
      }

      // Combine the counts.
#ifdef OMP
      for (i = 0; i < Ncnt; i++)
        for (j = 0; j < Nomp; j++) {
          twcnt[i] += wcnt[j][i];
          if (moment == 2) {
            tqcnt[i].v[0] += qcnt[j][i].v[0];
            tqcnt[i].v[1] += qcnt[j][i].v[1];
          }
        }
      combine_1d_counts(r2prec, moment, WT_COUNT, rbin, twcnt, tqcnt, Ncnt,
          out);
      free(twcnt);
      if (moment == 2) free(tqcnt);
#else
      combine_1d_counts(r2prec, moment, WT_COUNT, rbin, wcnt, qcnt, Ncnt, out);
#endif
    }
  }
  else {                // Counting with numbers.
#ifdef OMP
    for (i = 0; i < Nomp; i++)
      memset(ncnt[i], 0, sizeof(size_t) * Ncnt);
#else
    memset(ncnt, 0, sizeof(size_t) * Ncnt);
#endif

    // Exact distance bins.
    if (r2prec < -APPROX_R2_PREC || r2prec > APPROX_R2_PREC) {
      if (moment == 2)
#ifdef OMP
#pragma omp parallel
#pragma omp single
#endif
        kdtree_dual_leaf_cross_quad_exact(tree1, tree2, rbin2, rnum, qcnt, ncnt);
      else
#ifdef OMP
#pragma omp parallel
#pragma omp single
#endif
        kdtree_dual_leaf_cross_exact(tree1, tree2, rbin2, rnum, ncnt);
#ifdef OMP
      for (i = 0; i < Ncnt; i++)
        for (j = 0; j < Nomp; j++) {
          out[i].v[0] += ncnt[j][i];
          if (moment == 2) {
            out[i].v[1] += qcnt[j][i].v[0];
            out[i].v[2] += qcnt[j][i].v[1];
          }
        }
#else
      for (i = 0; i < Ncnt; i++) {
        out[i].v[0] = ncnt[i];
        if (moment == 2) {
          out[i].v[1] = qcnt[i].v[0];
          out[i].v[2] = qcnt[i].v[1];
        }
      }
#endif
    }
    else {              // Approximate distance bins.
#ifdef OMP
      tncnt = (size_t *) malloc(sizeof(size_t) * Ncnt);
      if (!tncnt) {
        P_EXT("failed to allocate memory for pair counts.\n");
        exit(ERR_MEM);
      }
      memset(tncnt, 0, sizeof(size_t) * Ncnt);
      if (moment == 2) {
        tqcnt = (DOUBLE2 *) malloc(sizeof(DOUBLE2) * Ncnt);
        if (!tqcnt) {
          P_EXT("failed to allocate memory for pair counts.\n");
          exit(ERR_MEM);
        }
        memset(tqcnt, 0, sizeof(DOUBLE2) * Ncnt);
      }
#endif
      if (r2prec == 0) {          // Integer distance bins.
        if (rmin < FLT_ERR) {
          if (moment == 2)
#ifdef OMP
#pragma omp parallel
#pragma omp single
#endif
            kdtree_dual_leaf_cross_quad_intbin_rmin0(tree1, tree2, rmax * rmax,
                qcnt, ncnt);
          else
#ifdef OMP
#pragma omp parallel
#pragma omp single
#endif
            kdtree_dual_leaf_cross_intbin_rmin0(tree1, tree2, rmax * rmax, ncnt);
        }
        else {
          if (moment == 2)
#ifdef OMP
#pragma omp parallel
#pragma omp single
#endif
            kdtree_dual_leaf_cross_quad_intbin(tree1, tree2, rmin * rmin,
                rmax * rmax, qcnt, ncnt);
          else
#ifdef OMP
#pragma omp parallel
#pragma omp single
#endif
            kdtree_dual_leaf_cross_intbin(tree1, tree2, rmin * rmin, rmax * rmax,
                ncnt);
        }
      }
      else {
        prec = pow(10, -r2prec);
        if (rmin < FLT_ERR) {
          if (moment == 2)
#ifdef OMP
#pragma omp parallel
#pragma omp single
#endif
            kdtree_dual_leaf_cross_quad_approx_rmin0(tree1, tree2, rmax * rmax,
                prec, qcnt, ncnt);
          else
#ifdef OMP
#pragma omp parallel
#pragma omp single
#endif
            kdtree_dual_leaf_cross_approx_rmin0(tree1, tree2, rmax * rmax,
                prec, ncnt);
        }
        else {
          if (moment == 2)
#ifdef OMP
#pragma omp parallel
#pragma omp single
#endif
            kdtree_dual_leaf_cross_quad_approx(tree1, tree2, rmin * rmin,
                rmax * rmax, prec, qcnt, ncnt);
          else
#ifdef OMP
#pragma omp parallel
#pragma omp single
#endif
            kdtree_dual_leaf_cross_approx(tree1, tree2, rmin * rmin, rmax * rmax,
                prec, ncnt);
        }
      }

      // Combine the counts.
#ifdef OMP
      for (i = 0; i < Ncnt; i++)
        for (j = 0; j < Nomp; j++) {
          tncnt[i] += ncnt[j][i];
          if (moment == 2) {
            tqcnt[i].v[0] += qcnt[j][i].v[0];
            tqcnt[i].v[1] += qcnt[j][i].v[1];
          }
        }
      combine_1d_counts(r2prec, moment, NUM_COUNT, rbin, tncnt, tqcnt, Ncnt,
          out);
      free(tncnt);
      if (moment == 2) free(tqcnt);
#else
      combine_1d_counts(r2prec, moment, NUM_COUNT, rbin, ncnt, qcnt, Ncnt, out);
#endif
    }
  }
}


///////////////////////////////////////////////////////////////////////////////
/******************************************************************************
  The following functions are for counting 1-D (isotropic) pairs. They are used
  by the functions above to define the strategies of counting pairs.
  There are serials of similiar functions to achieve high efficiency for
  different cases. Therefore, the codes can be highly redundant.
  To be more concise, the long (ugly) macros below are used. They may be
  difficult to be understand and for debugging though. Try adding the '-E' or
  '-save-temps' flag of `gcc` to check the expanded codes, e.g.
  $ gcc -E kdtree_cnt.c > tmp.c
******************************************************************************/
///////////////////////////////////////////////////////////////////////////////

#define DECLARE_IDX_APPROX      size_t idx
#define DECLARE_IDX_EXACT       int idx
#define DECLARE_DMAX            real dmax
#define DECLARE_QUAD            real dz; real mu

#define RMIN2_APPROX            rmin2
#define RMIN2_EXACT             rbin2[0]
#define RMAX2_APPROX            rmax2
#define RMAX2_EXACT             rbin2[nbin]

#define FIND_1D_BIN_INTBIN_RMIN0        idx = dist;
#define FIND_1D_BIN_INTBIN              idx = dist - rmin2;
#define FIND_1D_BIN_APPROX_RMIN0        idx = dist * r2prec;
#define FIND_1D_BIN_APPROX              idx = (dist - rmin2) * r2prec;
#define FIND_1D_BIN_EXACT       if((idx = find_1d_bin(dist, rbin2, nbin)) >= 0)

#define INCREMENT_COUNT         cnt[idx] ++
#define INCREMENT_WEIGHT        cnt[idx] += a->wt * b->wt
#define INCREMENT_COUNT_OMP     cnt[omp_get_thread_num()][idx] ++
#define INCREMENT_WEIGHT_OMP    cnt[omp_get_thread_num()][idx] += a->wt * b->wt

#define COMP_MU                                                 \
  dz = a->x[2] - b->x[2];                                       \
  if (dz > BOXSIZE / 2) dz -= BOXSIZE;                          \
  if (dz < -BOXSIZE / 2) dz += BOXSIZE;                         \
  if (dist < FLT_ERR) continue;                                 \
  mu = dz * dz / dist;

#define INCREMENT_COUNT_QUAD                                    \
  cnt[idx] ++;                                                  \
  qcnt[idx].v[0] += 2.5 * (3 * mu - 1);                         \
  qcnt[idx].v[1] += 1.125 * (35 * mu * mu - 30 * mu + 3)
#define INCREMENT_WEIGHT_QUAD                                   \
  cnt[idx] += a->wt * b->wt;                                    \
  qcnt[idx].v[0] += a->wt * b->wt * 2.5 * (3 * mu - 1);         \
  qcnt[idx].v[1] +=                                             \
      a->wt * b->wt * 1.125 * (35 * mu * mu - 30 * mu + 3)
#define INCREMENT_COUNT_QUAD_OMP                                \
  cnt[omp_get_thread_num()][idx] ++;                            \
  qcnt[omp_get_thread_num()][idx].v[0] +=                       \
      2.5 * (3 * mu - 1);                                       \
  qcnt[omp_get_thread_num()][idx].v[1] +=                       \
      1.125 * (35 * mu * mu - 30 * mu + 3)
#define INCREMENT_WEIGHT_QUAD_OMP                               \
  cnt[omp_get_thread_num()][idx] += a->wt * b->wt;              \
  qcnt[omp_get_thread_num()][idx].v[0] +=                       \
      a->wt * b->wt * 2.5 * (3 * mu - 1);                       \
  qcnt[omp_get_thread_num()][idx].v[1] +=                       \
      a->wt * b->wt * 1.125 * (35 * mu * mu - 30 * mu + 3)

#define OMP_TASK                _Pragma("omp task")

// Check the maximum possible distance of pairs inside a node.
#define CHECK_DMAX_SINGLE(rmin2)                                \
  dmax = squared_distance(&(node1->min), &(node1->max));        \
  if (dmax < rmin2) return

// Check the maximum possible distance of two nodes.
#define CHECK_DMAX_DUAL(rmin2)                                  \
  dmax = max_squared_dist_between_box(&(node1->min),            \
      &(node1->max), &(node2->min), &(node2->max));             \
  if (dmax < rmin2) return

// Check if the distance is larger than the maximum distance of interest.
#define CHECK_DIST_MAX(rmax2)                                   \
  if (dist >= rmax2) continue
// Check if the distance is outside of the distance range of interest.
#define CHECK_DIST_BOTH(rmin2, rmax2)                           \
  if (dist >= rmax2 || dist < rmin2) continue


/******************************************************************************
Macro `KDTREE_DUAL_LEAF_AUTO`:
  A template for the functions counting auto pairs based on a dual k-D tree
  algorithm, where only leaves are visited.
  This template is designed to avoid double countings.

Arguments:
  * `func`:     the name of the function;
  * `dec_idx`:  declaration of the index for distance bins:
                  approximate distance bins require `size_t` type indices,
                  while exact distance bins require `int` type indices;
  * `dec_dmax`: declaration of the maximum distance between nodes,
                it is only useful for distance range starting with non-zero;
  * `dec_quad`: declaration of variables for quadrupole;
  * `rmax2`:    form of the maximum squared distance of interest:
                  for approximate distance bins it is `rmax2`,
                  while for exact distance bins it is `rbin2[nbin]`;
  * `check_dmax_single`:
                check if the maximum distance of pairs in one node is larger
                than the minimum distance of interest,
                it is only useful for distance range starting with non-zero;
  * `check_dmax_dual`:
                check if the maximum distance of two nodes is larger than
                the minimum distance of interest,
                it is only useful for distance range starting with non-zero;
  * `check_dist`:
                check if the distance is inside the distance range of interest;
  * `find_bin`:
                find the corresponding distance bin given a distance;
  * `increment`:
                increment the counts in a distance bin;
  * `omp_task`:
                preprocessor directive for OpenMP;
  * `...`:      function dependent arguments.
******************************************************************************/
#define KDTREE_DUAL_LEAF_AUTO(func, dec_idx, dec_dmax, dec_quad, rmax2, \
    check_dmax_single, check_dmax_dual, check_dist,                     \
    find_bin, increment, omp_task, ...)                                 \
  size_t i, j;                                                          \
  dec_idx;                                                              \
  real dist, dmin;                                                      \
  dec_dmax;                                                             \
  dec_quad;                                                             \
  DATA *a, *b;                                                          \
  /* Avoid double counting of nodes in reverse turn. */                 \
  if (node1->id > node2->id) return;                                    \
  else if (node1 == node2) {                                            \
    /* Return if the maximum distance of pairs in this node is not
       inside the distance range of interest. */                        \
    check_dmax_single;                                                  \
    /* Compare all the pairs if the node is a leaf. */                  \
    if (node1->left == NULL) {                                          \
      for (i = 0; i < node1->n - 1; i++) {                              \
        for (j = i + 1; j < node2->n; j++) {                            \
          a = node1->data + i;                                          \
          b = node2->data + j;                                          \
          dist = squared_distance(a, b);                                \
          check_dist;                                                   \
          /* Increment the count in the corresponding distance bin. */  \
          find_bin {                                                    \
          increment; }                                                  \
        }                                                               \
      }                                                                 \
    }                                                                   \
    else {      /* The node is not a leaf. */                           \
omp_task                                                                \
      func(node1->left, node2->left, __VA_ARGS__, cnt);                 \
omp_task                                                                \
      func(node1->left, node2->right, __VA_ARGS__, cnt);                \
omp_task                                                                \
      func(node1->right, node2->left, __VA_ARGS__, cnt);                \
omp_task                                                                \
      func(node1->right, node2->right, __VA_ARGS__, cnt);               \
    }                                                                   \
  }                                                                     \
  else {        /* node1->id < node2->id */                             \
    /* Return if the minimum/maximum distance of the two nodes is not
       inside the distance range of interest. */                        \
    dmin = min_squared_dist_between_box(&(node1->min), &(node1->max),   \
        &(node2->min), &(node2->max));                                  \
    if (dmin >= rmax2) return;                                          \
    check_dmax_dual;                                                    \
    /* Compare all the pairs if the nodes are both leaves. */           \
    if (node1->left == NULL && node2->left == NULL) {                   \
      for (i = 0; i < node1->n; i++) {                                  \
        for (j = 0; j < node2->n; j++) {                                \
          a = node1->data + i;                                          \
          b = node2->data + j;                                          \
          dist = squared_distance(a, b);                                \
          check_dist;                                                   \
          /* Increment the count in the corresponding distance bin. */  \
          find_bin {                                                    \
          increment; }                                                  \
        }                                                               \
      }                                                                 \
    }                                                                   \
    else if (node1->left == NULL) {                                     \
omp_task                                                                \
      func(node1, node2->left, __VA_ARGS__, cnt);                       \
omp_task                                                                \
      func(node1, node2->right, __VA_ARGS__, cnt);                      \
    }                                                                   \
    else if (node2->left == NULL) {                                     \
omp_task                                                                \
      func(node1->left, node2, __VA_ARGS__, cnt);                       \
omp_task                                                                \
      func(node1->right, node2, __VA_ARGS__, cnt);                      \
    }                                                                   \
    else {                                                              \
omp_task                                                                \
      func(node1->left, node2->left, __VA_ARGS__, cnt);                 \
omp_task                                                                \
      func(node1->left, node2->right, __VA_ARGS__, cnt);                \
omp_task                                                                \
      func(node1->right, node2->left, __VA_ARGS__, cnt);                \
omp_task                                                                \
      func(node1->right, node2->right, __VA_ARGS__, cnt);               \
    }                                                                   \
  }


/******************************************************************************
Macro `KDTREE_DUAL_LEAF_CROSS`:
  A template for the functions counting cross pairs based on a dual k-D tree
  algorithm, where only leaves are visited.
  The algorithm is similar to that of `KDTREE_DUAL_LEAF_AUTO`.
  This template can also be used for auto pair counts, but it does not prevent
  double counting, and it does not check if two points are identical.

Arguments:
  The arguments are almost the same as those of `KDTREE_DUAL_LEAF_AUTO`, but
  there is only `check_dmax` here, which is equivalent with `check_dmax_dual`
  in macro `KDTREE_DUAL_LEAF_AUTO`.
******************************************************************************/
#define KDTREE_DUAL_LEAF_CROSS(func, dec_idx, dec_dmax, dec_quad,       \
    rmax2, check_dmax, check_dist, find_bin, increment, omp_task, ...)  \
  size_t i, j;                                                          \
  dec_idx;                                                              \
  real dist, dmin;                                                      \
  dec_dmax;                                                             \
  dec_quad;                                                             \
  DATA *a, *b;                                                          \
  /* Return if the minimum/maximum distance of the two nodes is not
     inside the distance range of interest. */                          \
  dmin = min_squared_dist_between_box(&(node1->min), &(node1->max),     \
      &(node2->min), &(node2->max));                                    \
  if (dmin >= rmax2) return;                                            \
  check_dmax;                                                           \
  /* Compare all the pairs if the nodes are both leaves. */             \
  if (node1->left == NULL && node2->left == NULL) {                     \
    for (i = 0; i < node1->n; i++) {                                    \
      for (j = 0; j < node2->n; j++) {                                  \
        a = node1->data + i;                                            \
        b = node2->data + j;                                            \
        dist = squared_distance(a, b);                                  \
        check_dist;                                                     \
        /* Increment the count in the corresponding distance bin. */    \
        find_bin {                                                      \
        increment; }                                                    \
      }                                                                 \
    }                                                                   \
  }                                                                     \
  else {                                                                \
    if (node1->n < node2->n) {                                          \
omp_task                                                                \
      func(node1, node2->left, __VA_ARGS__, cnt);                       \
omp_task                                                                \
      func(node1, node2->right, __VA_ARGS__, cnt);                      \
    }                                                                   \
    else {                                                              \
omp_task                                                                \
      func(node1->left, node2, __VA_ARGS__, cnt);                       \
omp_task                                                                \
      func(node1->right, node2, __VA_ARGS__, cnt);                      \
    }                                                                   \
  }


/******************************************************************************
  Implementations of the functions for pair counts based on the macros above.
******************************************************************************/

///////////////////////////////////////////////////////////////////////////////
//  Functions for auto number counts.
///////////////////////////////////////////////////////////////////////////////

/******************************************************************************
Function `kdtree_dual_leaf_auto_intbin_rmin0`:
  Counting pairs of points from a single dataset in 1-D squared distance bins
  based on the k-D tree data structure and a dual tree algorithm. The upper
  and lower limits of each squared distance bin should be (approximately)
  integers, and the smallest distance of interest should be 0.
Input:
  The k-D tree constructed upon the dataset, the array storing pair counts, and
  the maximum squared distance of interest.

Arguments:
  * `node1`:    a pointer to the first node of the k-D tree;
  * `node2`:    a pointer to the second node of the k-D tree;
  * `rmax2`:    the maximum squared distance of interest;
  * `cnt`:      the array storing pair counts.
******************************************************************************/
void kdtree_dual_leaf_auto_intbin_rmin0(const KDT *node1, const KDT *node2,
    const double rmax2,
#ifdef OMP
    size_t **cnt
#else
    size_t *cnt
#endif
    ) {
  KDTREE_DUAL_LEAF_AUTO(kdtree_dual_leaf_auto_intbin_rmin0,
      DECLARE_IDX_APPROX, /* No dmax */, /* No quadrupole */, RMAX2_APPROX,
      /* No dmax checking */, /* No dmax checking */,
      CHECK_DIST_MAX(RMAX2_APPROX), FIND_1D_BIN_INTBIN_RMIN0,
#ifdef OMP
      INCREMENT_COUNT_OMP, OMP_TASK,
#else
      INCREMENT_COUNT, /* No OpenMP */,
#endif
      rmax2);
}


/******************************************************************************
Function `kdtree_dual_leaf_auto_intbin`:
  Same as `kdtree_dual_leaf_auto_intbin`, but the lowest distance of interest
  is not necessarily 0.
Input:
  The k-D tree constructed upon the dataset, the array storing pair counts, and
  the lower and upper limits of the squared distances of interest.

Arguments:
  * `node1`:    a pointer to the first node of the k-D tree;
  * `node2`:    a pointer to the second node of the k-D tree;
  * `rmin2`:    the minimum squared distance of interest;
  * `rmax2`:    the maximum squared distance of interest;
  * `cnt`:      the array storing pair counts.
******************************************************************************/
void kdtree_dual_leaf_auto_intbin(const KDT *node1, const KDT *node2,
    const double rmin2, const double rmax2,
#ifdef OMP
    size_t **cnt
#else
    size_t *cnt
#endif
    ) {
  KDTREE_DUAL_LEAF_AUTO(kdtree_dual_leaf_auto_intbin,
      DECLARE_IDX_APPROX, DECLARE_DMAX, /* No quadrupole */, RMAX2_APPROX,
      CHECK_DMAX_SINGLE(RMIN2_APPROX), CHECK_DMAX_DUAL(RMIN2_APPROX),
      CHECK_DIST_BOTH(RMIN2_APPROX, RMAX2_APPROX), FIND_1D_BIN_INTBIN,
#ifdef OMP
      INCREMENT_COUNT_OMP, OMP_TASK,
#else
      INCREMENT_COUNT, /* No OpenMP */,
#endif
      rmin2, rmax2);
}


/******************************************************************************
Function `kdtree_dual_leaf_auto_approx_rmin0`:
  Same as `kdtree_dual_leaf_auto_intbin_rmin0`, but allows approximate sqaured
  distance of different precisions.
Input:
  The k-D tree constructed upon the dataset, the array storing pair counts, the
  precision for squared distance approximation, and the maximum squared
  distance of interest.

Arguments:
  * `node1`:    a pointer to the first node of the k-D tree;
  * `node2`:    a pointer to the second node of the k-D tree;
  * `rmax2`:    the maximum squared distance of interest;
  * `r2prec`:   the precision of squared distance;
  * `cnt`:      the array storing pair counts.
******************************************************************************/
void kdtree_dual_leaf_auto_approx_rmin0(const KDT *node1, const KDT *node2,
    const double rmax2, const double r2prec,
#ifdef OMP
    size_t **cnt
#else
    size_t *cnt
#endif
    ) {
  KDTREE_DUAL_LEAF_AUTO(kdtree_dual_leaf_auto_approx_rmin0,
      DECLARE_IDX_APPROX, /* No dmax */, /* No quadrupole */, RMAX2_APPROX,
      /* No dmax checking */, /* No dmax checking */,
      CHECK_DIST_MAX(RMAX2_APPROX), FIND_1D_BIN_APPROX_RMIN0,
#ifdef OMP
      INCREMENT_COUNT_OMP, OMP_TASK,
#else
      INCREMENT_COUNT, /* No OpenMP */,
#endif
      rmax2, r2prec);
}


/******************************************************************************
Function `kdtree_dual_leaf_auto_approx`:
  Same as `kdtree_dual_leaf_auto_intbin`, but allows approximate sqaured
  distance of different precisions.
Input:
  The k-D tree constructed upon the dataset, the array storing pair counts, the
  precision for squared distance approximation, and the minimum and maximum
  squared distance of interest.

Arguments:
  * `node1`:    a pointer to the first node of the k-D tree;
  * `node2`:    a pointer to the second node of the k-D tree;
  * `rmin2`:    the minimum squared distance of interest;
  * `rmax2`:    the maximum squared distance of interest;
  * `r2prec`:   the precision of squared distance;
  * `cnt`:      the array storing pair counts.
******************************************************************************/
void kdtree_dual_leaf_auto_approx(const KDT *node1, const KDT *node2,
    const double rmin2, const double rmax2, const double r2prec,
#ifdef OMP
    size_t **cnt
#else
    size_t *cnt
#endif
    ) {
  KDTREE_DUAL_LEAF_AUTO(kdtree_dual_leaf_auto_approx,
      DECLARE_IDX_APPROX, DECLARE_DMAX, /* No quadrupole */, RMAX2_APPROX,
      CHECK_DMAX_SINGLE(RMIN2_APPROX), CHECK_DMAX_DUAL(RMIN2_APPROX),
      CHECK_DIST_BOTH(RMIN2_APPROX, RMAX2_APPROX), FIND_1D_BIN_APPROX,
#ifdef OMP
      INCREMENT_COUNT_OMP, OMP_TASK,
#else
      INCREMENT_COUNT, /* No OpenMP */,
#endif
      rmin2, rmax2, r2prec);
}


/******************************************************************************
Function `kdtree_dual_leaf_auto_exact`:
  Counting pairs of points from a single dataset in 1-D squared distance bins
  based on the k-D tree data structure and a dual tree algorithm.
  This function uses exact squared distances.
Input:
  The k-D tree constructed upon the dataset, and the arrays storing pair
  counts.

Arguments:
  * `node1`:    a pointer to the first node of the k-D tree;
  * `node2`:    a pointer to the second node of the k-D tree;
  * `rbin2`:    the array storing squared distance bins;
  * `nbin`:     number of distance bins;
  * `cnt`:      the array storing pair counts.
******************************************************************************/
void kdtree_dual_leaf_auto_exact(const KDT *node1, const KDT *node2,
    const double *rbin2, const int nbin,
#ifdef OMP
    size_t **cnt
#else
    size_t *cnt
#endif
    ) {
  KDTREE_DUAL_LEAF_AUTO(kdtree_dual_leaf_auto_exact,
      DECLARE_IDX_EXACT, DECLARE_DMAX, /* No quadrupole */, RMAX2_EXACT,
      CHECK_DMAX_SINGLE(RMIN2_EXACT), CHECK_DMAX_DUAL(RMIN2_EXACT),
      CHECK_DIST_BOTH(RMIN2_EXACT, RMAX2_EXACT), FIND_1D_BIN_EXACT,
#ifdef OMP
      INCREMENT_COUNT_OMP, OMP_TASK,
#else
      INCREMENT_COUNT, /* No OpenMP */,
#endif
      rbin2, nbin);
}


/******************************************************************************
  The following functions are the weight count version of the number count
  functions above, with the same structures. Please consult the number count
  versions for documentations.
******************************************************************************/

///////////////////////////////////////////////////////////////////////////////
//  Functions for auto weight counts.
///////////////////////////////////////////////////////////////////////////////

void kdtree_dual_leaf_auto_intbin_rmin0_wt(const KDT *node1, const KDT *node2,
    const double rmax2,
#ifdef OMP
    double **cnt
#else
    double *cnt
#endif
    ) {
  KDTREE_DUAL_LEAF_AUTO(kdtree_dual_leaf_auto_intbin_rmin0_wt,
      DECLARE_IDX_APPROX, /* No dmax */, /* No quadrupole */, RMAX2_APPROX,
      /* No dmax checking */, /* No dmax checking */,
      CHECK_DIST_MAX(RMAX2_APPROX), FIND_1D_BIN_INTBIN_RMIN0,
#ifdef OMP
      INCREMENT_WEIGHT_OMP, OMP_TASK,
#else
      INCREMENT_WEIGHT, /* No OpenMP */,
#endif
      rmax2);
}


void kdtree_dual_leaf_auto_intbin_wt(const KDT *node1, const KDT *node2,
    const double rmin2, const double rmax2,
#ifdef OMP
    double **cnt
#else
    double *cnt
#endif
    ) {
  KDTREE_DUAL_LEAF_AUTO(kdtree_dual_leaf_auto_intbin_wt,
      DECLARE_IDX_APPROX, DECLARE_DMAX, /* No quadrupole */, RMAX2_APPROX,
      CHECK_DMAX_SINGLE(RMIN2_APPROX), CHECK_DMAX_DUAL(RMIN2_APPROX),
      CHECK_DIST_BOTH(RMIN2_APPROX, RMAX2_APPROX), FIND_1D_BIN_INTBIN,
#ifdef OMP
      INCREMENT_WEIGHT_OMP, OMP_TASK,
#else
      INCREMENT_WEIGHT, /* No OpenMP */,
#endif
      rmin2, rmax2);
}


void kdtree_dual_leaf_auto_approx_rmin0_wt(const KDT *node1, const KDT *node2,
    const double rmax2, const double r2prec,
#ifdef OMP
    double **cnt
#else
    double *cnt
#endif
    ) {
  KDTREE_DUAL_LEAF_AUTO(kdtree_dual_leaf_auto_approx_rmin0_wt,
      DECLARE_IDX_APPROX, /* No dmax */, /* No quadrupole */, RMAX2_APPROX,
      /* No dmax checking */, /* No dmax checking */,
      CHECK_DIST_MAX(RMAX2_APPROX), FIND_1D_BIN_APPROX_RMIN0,
#ifdef OMP
      INCREMENT_WEIGHT_OMP, OMP_TASK,
#else
      INCREMENT_WEIGHT, /* No OpenMP */,
#endif
      rmax2, r2prec);
}


void kdtree_dual_leaf_auto_approx_wt(const KDT *node1, const KDT *node2,
    const double rmin2, const double rmax2, const double r2prec,
#ifdef OMP
    double **cnt
#else
    double *cnt
#endif
    ) {
  KDTREE_DUAL_LEAF_AUTO(kdtree_dual_leaf_auto_approx_wt,
      DECLARE_IDX_APPROX, DECLARE_DMAX, /* No quadrupole */, RMAX2_APPROX,
      CHECK_DMAX_SINGLE(RMIN2_APPROX), CHECK_DMAX_DUAL(RMIN2_APPROX),
      CHECK_DIST_BOTH(RMIN2_APPROX, RMAX2_APPROX), FIND_1D_BIN_APPROX,
#ifdef OMP
      INCREMENT_WEIGHT_OMP, OMP_TASK,
#else
      INCREMENT_WEIGHT, /* No OpenMP */,
#endif
      rmin2, rmax2, r2prec);
}


void kdtree_dual_leaf_auto_exact_wt(const KDT *node1, const KDT *node2,
    const double *rbin2, const int nbin,
#ifdef OMP
    double **cnt
#else
    double *cnt
#endif
    ) {
  KDTREE_DUAL_LEAF_AUTO(kdtree_dual_leaf_auto_exact_wt,
      DECLARE_IDX_EXACT, DECLARE_DMAX, /* No quadrupole */, RMAX2_EXACT,
      CHECK_DMAX_SINGLE(RMIN2_EXACT), CHECK_DMAX_DUAL(RMIN2_EXACT),
      CHECK_DIST_BOTH(RMIN2_EXACT, RMAX2_EXACT), FIND_1D_BIN_EXACT,
#ifdef OMP
      INCREMENT_WEIGHT_OMP, OMP_TASK,
#else
      INCREMENT_WEIGHT, /* No OpenMP */,
#endif
      rbin2, nbin);
}


///////////////////////////////////////////////////////////////////////////////
//  Functions for cross number counts.
///////////////////////////////////////////////////////////////////////////////

/******************************************************************************
Function `kdtree_dual_leaf_cross_intbin_rmin0`:
  Counting pairs of points from two datasets (the two datasets can be the same
  for the auto correlation, but in this case this function is not optimal since
  it double counts the pairs) in 1-D squared distance bins based on the k-D
  tree data structure and a dual tree algorithm. The upper and lower limits of
  each squared distance bin should be (approximately) integers, and the
  smallest distance of interest should be 0.
******************************************************************************/
void kdtree_dual_leaf_cross_intbin_rmin0(const KDT *node1, const KDT *node2,
    const double rmax2,
#ifdef OMP
    size_t **cnt
#else
    size_t *cnt
#endif
    ) {
  KDTREE_DUAL_LEAF_CROSS(kdtree_dual_leaf_cross_intbin_rmin0,
      DECLARE_IDX_APPROX, /* No dmax */, /* No quadrupole */, RMAX2_APPROX,
      /* No dmax checking */,
      CHECK_DIST_MAX(RMAX2_APPROX), FIND_1D_BIN_INTBIN_RMIN0,
#ifdef OMP
      INCREMENT_COUNT_OMP, OMP_TASK,
#else
      INCREMENT_COUNT, /* No OpenMP */,
#endif
      rmax2);
}


/******************************************************************************
Function `kdtree_dual_leaf_cross_intbin`:
  Same as `kdtree_dual_leaf_cross_intbin_rmin0`, but the lowest distance of
  interest is not necessarily 0.
******************************************************************************/
void kdtree_dual_leaf_cross_intbin(const KDT *node1, const KDT *node2,
    const double rmin2, const double rmax2,
#ifdef OMP
    size_t **cnt
#else
    size_t *cnt
#endif
    ) {
  KDTREE_DUAL_LEAF_CROSS(kdtree_dual_leaf_cross_intbin,
      DECLARE_IDX_APPROX, DECLARE_DMAX, /* No quadrupole */, RMAX2_APPROX,
      CHECK_DMAX_DUAL(RMIN2_APPROX),
      CHECK_DIST_BOTH(RMIN2_APPROX, RMAX2_APPROX), FIND_1D_BIN_INTBIN,
#ifdef OMP
      INCREMENT_COUNT_OMP, OMP_TASK,
#else
      INCREMENT_COUNT, /* No OpenMP */,
#endif
      rmin2, rmax2);
}


/******************************************************************************
Function `kdtree_dual_leaf_cross_approx_rmin0`:
  Same as `kdtree_dual_leaf_cross_intbin_rmin0`, but allows approximate squared
  distances with different precisions.
******************************************************************************/
void kdtree_dual_leaf_cross_approx_rmin0(const KDT *node1, const KDT *node2,
    const double rmax2, const double r2prec,
#ifdef OMP
    size_t **cnt
#else
    size_t *cnt
#endif
    ) {
  KDTREE_DUAL_LEAF_CROSS(kdtree_dual_leaf_cross_approx_rmin0,
      DECLARE_IDX_APPROX, /* No dmax */, /* No quadrupole */, RMAX2_APPROX,
      /* No dmax checking */,
      CHECK_DIST_MAX(RMAX2_APPROX), FIND_1D_BIN_APPROX_RMIN0,
#ifdef OMP
      INCREMENT_COUNT_OMP, OMP_TASK,
#else
      INCREMENT_COUNT, /* No OpenMP */,
#endif
      rmax2, r2prec);
}


/******************************************************************************
Function `kdtree_dual_leaf_cross_approx`:
  Same as `kdtree_dual_leaf_cross_intbin`, but allows approximate squread
  distances with different precisions.
******************************************************************************/
void kdtree_dual_leaf_cross_approx(const KDT *node1, const KDT *node2,
    const double rmin2, const double rmax2, const double r2prec,
#ifdef OMP
    size_t **cnt
#else
    size_t *cnt
#endif
    ) {
  KDTREE_DUAL_LEAF_CROSS(kdtree_dual_leaf_cross_approx,
      DECLARE_IDX_APPROX, DECLARE_DMAX, /* No quadrupole */, RMAX2_APPROX,
      CHECK_DMAX_DUAL(RMIN2_APPROX),
      CHECK_DIST_BOTH(RMIN2_APPROX, RMAX2_APPROX), FIND_1D_BIN_APPROX,
#ifdef OMP
      INCREMENT_COUNT_OMP, OMP_TASK,
#else
      INCREMENT_COUNT, /* No OpenMP */,
#endif
      rmin2, rmax2, r2prec);
}


/******************************************************************************
Function `kdtree_dual_leaf_cross_exact`:
  Counting pairs of points from two datasets in 1-D squared distance bins,
  based on the k-D tree data structure and a dual tree algorithm.
  This function uses exact squared distances.
Input:
  The k-D tree constructed on the two datasets, and the arrays storing pair
  counts.

Arguments:
  * `node1`:    a pointer to a node of the first k-D tree;
  * `node2`:    a pointer to a node of the second k-D tree;
  * `rbin2`:    the array storing squared distance bins;
  * `nbin`:     number of distance bins;
  * `cnt`:      the array storing pair counts.
******************************************************************************/
void kdtree_dual_leaf_cross_exact(const KDT *node1, const KDT *node2,
    const double *rbin2, const int nbin,
#ifdef OMP
    size_t **cnt
#else
    size_t *cnt
#endif
    ) {
  KDTREE_DUAL_LEAF_CROSS(kdtree_dual_leaf_cross_exact,
      DECLARE_IDX_EXACT, DECLARE_DMAX, /* No quadrupole */, RMAX2_EXACT,
      CHECK_DMAX_DUAL(RMIN2_EXACT),
      CHECK_DIST_BOTH(RMIN2_EXACT, RMAX2_EXACT), FIND_1D_BIN_EXACT,
#ifdef OMP
      INCREMENT_COUNT_OMP, OMP_TASK,
#else
      INCREMENT_COUNT, /* No OpenMP */,
#endif
      rbin2, nbin);
}


/******************************************************************************
  The following functions are the weight count version of the number count
  functions above, with the same structures. Please consult the number count
  versions for documentations.
******************************************************************************/

///////////////////////////////////////////////////////////////////////////////
//  Functions for cross weight counts.
///////////////////////////////////////////////////////////////////////////////

void kdtree_dual_leaf_cross_intbin_rmin0_wt(const KDT *node1, const KDT *node2,
    const double rmax2,
#ifdef OMP
    double **cnt
#else
    double *cnt
#endif
    ) {
  KDTREE_DUAL_LEAF_CROSS(kdtree_dual_leaf_cross_intbin_rmin0_wt,
      DECLARE_IDX_APPROX, /* No dmax */, /* No quadrupole */, RMAX2_APPROX,
      /* No dmax checking */,
      CHECK_DIST_MAX(RMAX2_APPROX), FIND_1D_BIN_INTBIN_RMIN0,
#ifdef OMP
      INCREMENT_WEIGHT_OMP, OMP_TASK,
#else
      INCREMENT_WEIGHT, /* No OpenMP */,
#endif
      rmax2);
}


void kdtree_dual_leaf_cross_intbin_wt(const KDT *node1, const KDT *node2,
    const double rmin2, const double rmax2,
#ifdef OMP
    double **cnt
#else
    double *cnt
#endif
    ) {
  KDTREE_DUAL_LEAF_CROSS(kdtree_dual_leaf_cross_intbin_wt,
      DECLARE_IDX_APPROX, DECLARE_DMAX, /* No quadrupole */, RMAX2_APPROX,
      CHECK_DMAX_DUAL(RMIN2_APPROX),
      CHECK_DIST_BOTH(RMIN2_APPROX, RMAX2_APPROX), FIND_1D_BIN_INTBIN,
#ifdef OMP
      INCREMENT_WEIGHT_OMP, OMP_TASK,
#else
      INCREMENT_WEIGHT, /* No OpenMP */,
#endif
      rmin2, rmax2);
}


void kdtree_dual_leaf_cross_approx_rmin0_wt(const KDT *node1, const KDT *node2,
    const double rmax2, const double r2prec,
#ifdef OMP
    double **cnt
#else
    double *cnt
#endif
    ) {
  KDTREE_DUAL_LEAF_CROSS(kdtree_dual_leaf_cross_approx_rmin0_wt,
      DECLARE_IDX_APPROX, /* No dmax */, /* No quadrupole */, RMAX2_APPROX,
      /* No dmax checking */,
      CHECK_DIST_MAX(RMAX2_APPROX), FIND_1D_BIN_APPROX_RMIN0,
#ifdef OMP
      INCREMENT_WEIGHT_OMP, OMP_TASK,
#else
      INCREMENT_WEIGHT, /* No OpenMP */,
#endif
      rmax2, r2prec);
}


void kdtree_dual_leaf_cross_approx_wt(const KDT *node1, const KDT *node2,
    const double rmin2, const double rmax2, const double r2prec,
#ifdef OMP
    double **cnt
#else
    double *cnt
#endif
    ) {
  KDTREE_DUAL_LEAF_CROSS(kdtree_dual_leaf_cross_approx_wt,
      DECLARE_IDX_APPROX, DECLARE_DMAX, /* No quadrupole */, RMAX2_APPROX,
      CHECK_DMAX_DUAL(RMIN2_APPROX),
      CHECK_DIST_BOTH(RMIN2_APPROX, RMAX2_APPROX), FIND_1D_BIN_APPROX,
#ifdef OMP
      INCREMENT_WEIGHT_OMP, OMP_TASK,
#else
      INCREMENT_WEIGHT, /* No OpenMP */,
#endif
      rmin2, rmax2, r2prec);
}


void kdtree_dual_leaf_cross_exact_wt(const KDT *node1, const KDT *node2,
    const double *rbin2, const int nbin,
#ifdef OMP
    double **cnt
#else
    double *cnt
#endif
    ) {
  KDTREE_DUAL_LEAF_CROSS(kdtree_dual_leaf_cross_exact_wt,
      DECLARE_IDX_EXACT, DECLARE_DMAX, /* No quadrupole */, RMAX2_EXACT,
      CHECK_DMAX_DUAL(RMIN2_EXACT),
      CHECK_DIST_BOTH(RMIN2_EXACT, RMAX2_EXACT), FIND_1D_BIN_EXACT,
#ifdef OMP
      INCREMENT_WEIGHT_OMP, OMP_TASK,
#else
      INCREMENT_WEIGHT, /* No OpenMP */,
#endif
      rbin2, nbin);
}


/******************************************************************************
  The following functions are the quadrupole version of the functions above,
  with the same structures.
******************************************************************************/

///////////////////////////////////////////////////////////////////////////////
//  Functions for quadrupole counts
///////////////////////////////////////////////////////////////////////////////

void kdtree_dual_leaf_auto_quad_intbin_rmin0(const KDT *node1, const KDT *node2,
    const double rmax2,
#ifdef OMP
    DOUBLE2 **qcnt, size_t **cnt
#else
    DOUBLE2 *qcnt, size_t *cnt
#endif
    ) {
  KDTREE_DUAL_LEAF_AUTO(kdtree_dual_leaf_auto_quad_intbin_rmin0,
      DECLARE_IDX_APPROX, /* No dmax */, DECLARE_QUAD, RMAX2_APPROX,
      /* No dmax checking */, /* No dmax checking */,
      CHECK_DIST_MAX(RMAX2_APPROX), FIND_1D_BIN_INTBIN_RMIN0,
#ifdef OMP
      COMP_MU INCREMENT_COUNT_QUAD_OMP, OMP_TASK,
#else
      COMP_MU INCREMENT_COUNT_QUAD, /* No OpenMP */,
#endif
      rmax2, qcnt);
}


void kdtree_dual_leaf_auto_quad_intbin(const KDT *node1, const KDT *node2,
    const double rmin2, const double rmax2,
#ifdef OMP
    DOUBLE2 **qcnt, size_t **cnt
#else
    DOUBLE2 *qcnt, size_t *cnt
#endif
    ) {
  KDTREE_DUAL_LEAF_AUTO(kdtree_dual_leaf_auto_quad_intbin,
      DECLARE_IDX_APPROX, DECLARE_DMAX, DECLARE_QUAD, RMAX2_APPROX,
      CHECK_DMAX_SINGLE(RMIN2_APPROX), CHECK_DMAX_DUAL(RMIN2_APPROX),
      CHECK_DIST_BOTH(RMIN2_APPROX, RMAX2_APPROX), FIND_1D_BIN_INTBIN,
#ifdef OMP
      COMP_MU INCREMENT_COUNT_QUAD_OMP, OMP_TASK,
#else
      COMP_MU INCREMENT_COUNT_QUAD, /* No OpenMP */,
#endif
      rmin2, rmax2, qcnt);
}


void kdtree_dual_leaf_auto_quad_approx_rmin0(const KDT *node1, const KDT *node2,
    const double rmax2, const double r2prec,
#ifdef OMP
    DOUBLE2 **qcnt, size_t **cnt
#else
    DOUBLE2 *qcnt, size_t *cnt
#endif
    ) {
  KDTREE_DUAL_LEAF_AUTO(kdtree_dual_leaf_auto_quad_approx_rmin0,
      DECLARE_IDX_APPROX, /* No dmax */, DECLARE_QUAD, RMAX2_APPROX,
      /* No dmax checking */, /* No dmax checking */,
      CHECK_DIST_MAX(RMAX2_APPROX), FIND_1D_BIN_APPROX_RMIN0,
#ifdef OMP
      COMP_MU INCREMENT_COUNT_QUAD_OMP, OMP_TASK,
#else
      COMP_MU INCREMENT_COUNT_QUAD, /* No OpenMP */,
#endif
      rmax2, r2prec, qcnt);
}


void kdtree_dual_leaf_auto_quad_approx(const KDT *node1, const KDT *node2,
    const double rmin2, const double rmax2, const double r2prec,
#ifdef OMP
    DOUBLE2 **qcnt, size_t **cnt
#else
    DOUBLE2 *qcnt, size_t *cnt
#endif
    ) {
  KDTREE_DUAL_LEAF_AUTO(kdtree_dual_leaf_auto_quad_approx,
      DECLARE_IDX_APPROX, DECLARE_DMAX, DECLARE_QUAD, RMAX2_APPROX,
      CHECK_DMAX_SINGLE(RMIN2_APPROX), CHECK_DMAX_DUAL(RMIN2_APPROX),
      CHECK_DIST_BOTH(RMIN2_APPROX, RMAX2_APPROX), FIND_1D_BIN_APPROX,
#ifdef OMP
      COMP_MU INCREMENT_COUNT_QUAD_OMP, OMP_TASK,
#else
      COMP_MU INCREMENT_COUNT_QUAD, /* No OpenMP */,
#endif
      rmin2, rmax2, r2prec, qcnt);
}


void kdtree_dual_leaf_auto_quad_exact(const KDT *node1, const KDT *node2,
    const double *rbin2, const int nbin,
#ifdef OMP
    DOUBLE2 **qcnt, size_t **cnt
#else
    DOUBLE2 *qcnt, size_t *cnt
#endif
    ) {
  KDTREE_DUAL_LEAF_AUTO(kdtree_dual_leaf_auto_quad_exact,
      DECLARE_IDX_EXACT, DECLARE_DMAX, DECLARE_QUAD, RMAX2_EXACT,
      CHECK_DMAX_SINGLE(RMIN2_EXACT), CHECK_DMAX_DUAL(RMIN2_EXACT),
      CHECK_DIST_BOTH(RMIN2_EXACT, RMAX2_EXACT), FIND_1D_BIN_EXACT,
#ifdef OMP
      COMP_MU INCREMENT_COUNT_QUAD_OMP, OMP_TASK,
#else
      COMP_MU INCREMENT_COUNT_QUAD, /* No OpenMP */,
#endif
      rbin2, nbin, qcnt);
}


void kdtree_dual_leaf_auto_quad_intbin_rmin0_wt(const KDT *node1,
    const KDT *node2, const double rmax2,
#ifdef OMP
    DOUBLE2 **qcnt, double **cnt
#else
    DOUBLE2 *qcnt, double *cnt
#endif
    ) {
  KDTREE_DUAL_LEAF_AUTO(kdtree_dual_leaf_auto_quad_intbin_rmin0_wt,
      DECLARE_IDX_APPROX, /* No dmax */, DECLARE_QUAD, RMAX2_APPROX,
      /* No dmax checking */, /* No dmax checking */,
      CHECK_DIST_MAX(RMAX2_APPROX), FIND_1D_BIN_INTBIN_RMIN0,
#ifdef OMP
      COMP_MU INCREMENT_WEIGHT_QUAD_OMP, OMP_TASK,
#else
      COMP_MU INCREMENT_WEIGHT_QUAD, /* No OpenMP */,
#endif
      rmax2, qcnt);
}


void kdtree_dual_leaf_auto_quad_intbin_wt(const KDT *node1, const KDT *node2,
    const double rmin2, const double rmax2,
#ifdef OMP
    DOUBLE2 **qcnt, double **cnt
#else
    DOUBLE2 *qcnt, double *cnt
#endif
    ) {
  KDTREE_DUAL_LEAF_AUTO(kdtree_dual_leaf_auto_quad_intbin_wt,
      DECLARE_IDX_APPROX, DECLARE_DMAX, DECLARE_QUAD, RMAX2_APPROX,
      CHECK_DMAX_SINGLE(RMIN2_APPROX), CHECK_DMAX_DUAL(RMIN2_APPROX),
      CHECK_DIST_BOTH(RMIN2_APPROX, RMAX2_APPROX), FIND_1D_BIN_INTBIN,
#ifdef OMP
      COMP_MU INCREMENT_WEIGHT_QUAD_OMP, OMP_TASK,
#else
      COMP_MU INCREMENT_WEIGHT_QUAD, /* No OpenMP */,
#endif
      rmin2, rmax2, qcnt);
}


void kdtree_dual_leaf_auto_quad_approx_rmin0_wt(const KDT *node1,
    const KDT *node2, const double rmax2, const double r2prec,
#ifdef OMP
    DOUBLE2 **qcnt, double **cnt
#else
    DOUBLE2 *qcnt, double *cnt
#endif
    ) {
  KDTREE_DUAL_LEAF_AUTO(kdtree_dual_leaf_auto_quad_approx_rmin0_wt,
      DECLARE_IDX_APPROX, /* No dmax */, DECLARE_QUAD, RMAX2_APPROX,
      /* No dmax checking */, /* No dmax checking */,
      CHECK_DIST_MAX(RMAX2_APPROX), FIND_1D_BIN_APPROX_RMIN0,
#ifdef OMP
      COMP_MU INCREMENT_WEIGHT_QUAD_OMP, OMP_TASK,
#else
      COMP_MU INCREMENT_WEIGHT_QUAD, /* No OpenMP */,
#endif
      rmax2, r2prec, qcnt);
}


void kdtree_dual_leaf_auto_quad_approx_wt(const KDT *node1, const KDT *node2,
    const double rmin2, const double rmax2, const double r2prec,
#ifdef OMP
    DOUBLE2 **qcnt, double **cnt
#else
    DOUBLE2 *qcnt, double *cnt
#endif
    ) {
  KDTREE_DUAL_LEAF_AUTO(kdtree_dual_leaf_auto_quad_approx_wt,
      DECLARE_IDX_APPROX, DECLARE_DMAX, DECLARE_QUAD, RMAX2_APPROX,
      CHECK_DMAX_SINGLE(RMIN2_APPROX), CHECK_DMAX_DUAL(RMIN2_APPROX),
      CHECK_DIST_BOTH(RMIN2_APPROX, RMAX2_APPROX), FIND_1D_BIN_APPROX,
#ifdef OMP
      COMP_MU INCREMENT_WEIGHT_QUAD_OMP, OMP_TASK,
#else
      COMP_MU INCREMENT_WEIGHT_QUAD, /* No OpenMP */,
#endif
      rmin2, rmax2, r2prec, qcnt);
}


void kdtree_dual_leaf_auto_quad_exact_wt(const KDT *node1, const KDT *node2,
    const double *rbin2, const int nbin,
#ifdef OMP
    DOUBLE2 **qcnt, double **cnt
#else
    DOUBLE2 *qcnt, double *cnt
#endif
    ) {
  KDTREE_DUAL_LEAF_AUTO(kdtree_dual_leaf_auto_quad_exact_wt,
      DECLARE_IDX_EXACT, DECLARE_DMAX, DECLARE_QUAD, RMAX2_EXACT,
      CHECK_DMAX_SINGLE(RMIN2_EXACT), CHECK_DMAX_DUAL(RMIN2_EXACT),
      CHECK_DIST_BOTH(RMIN2_EXACT, RMAX2_EXACT), FIND_1D_BIN_EXACT,
#ifdef OMP
      COMP_MU INCREMENT_WEIGHT_QUAD_OMP, OMP_TASK,
#else
      COMP_MU INCREMENT_WEIGHT_QUAD, /* No OpenMP */,
#endif
      rbin2, nbin, qcnt);
}


void kdtree_dual_leaf_cross_quad_intbin_rmin0(const KDT *node1,
    const KDT *node2, const double rmax2,
#ifdef OMP
    DOUBLE2 **qcnt, size_t **cnt
#else
    DOUBLE2 *qcnt, size_t *cnt
#endif
    ) {
  KDTREE_DUAL_LEAF_CROSS(kdtree_dual_leaf_cross_quad_intbin_rmin0,
      DECLARE_IDX_APPROX, /* No dmax */, DECLARE_QUAD, RMAX2_APPROX,
      /* No dmax checking */,
      CHECK_DIST_MAX(RMAX2_APPROX), FIND_1D_BIN_INTBIN_RMIN0,
#ifdef OMP
      COMP_MU INCREMENT_COUNT_QUAD_OMP, OMP_TASK,
#else
      COMP_MU INCREMENT_COUNT_QUAD, /* No OpenMP */,
#endif
      rmax2, qcnt);
}


void kdtree_dual_leaf_cross_quad_intbin(const KDT *node1, const KDT *node2,
    const double rmin2, const double rmax2,
#ifdef OMP
    DOUBLE2 **qcnt, size_t **cnt
#else
    DOUBLE2 *qcnt, size_t *cnt
#endif
    ) {
  KDTREE_DUAL_LEAF_CROSS(kdtree_dual_leaf_cross_quad_intbin,
      DECLARE_IDX_APPROX, DECLARE_DMAX, DECLARE_QUAD, RMAX2_APPROX,
      CHECK_DMAX_DUAL(RMIN2_APPROX),
      CHECK_DIST_BOTH(RMIN2_APPROX, RMAX2_APPROX), FIND_1D_BIN_INTBIN,
#ifdef OMP
      COMP_MU INCREMENT_COUNT_QUAD_OMP, OMP_TASK,
#else
      COMP_MU INCREMENT_COUNT_QUAD, /* No OpenMP */,
#endif
      rmin2, rmax2, qcnt);
}


void kdtree_dual_leaf_cross_quad_approx_rmin0(const KDT *node1,
    const KDT *node2, const double rmax2, const double r2prec,
#ifdef OMP
    DOUBLE2 **qcnt, size_t **cnt
#else
    DOUBLE2 *qcnt, size_t *cnt
#endif
    ) {
  KDTREE_DUAL_LEAF_CROSS(kdtree_dual_leaf_cross_quad_approx_rmin0,
      DECLARE_IDX_APPROX, /* No dmax */, DECLARE_QUAD, RMAX2_APPROX,
      /* No dmax checking */,
      CHECK_DIST_MAX(RMAX2_APPROX), FIND_1D_BIN_APPROX_RMIN0,
#ifdef OMP
      COMP_MU INCREMENT_COUNT_QUAD_OMP, OMP_TASK,
#else
      COMP_MU INCREMENT_COUNT_QUAD, /* No OpenMP */,
#endif
      rmax2, r2prec, qcnt);
}


void kdtree_dual_leaf_cross_quad_approx(const KDT *node1, const KDT *node2,
    const double rmin2, const double rmax2, const double r2prec,
#ifdef OMP
    DOUBLE2 **qcnt, size_t **cnt
#else
    DOUBLE2 *qcnt, size_t *cnt
#endif
    ) {
  KDTREE_DUAL_LEAF_CROSS(kdtree_dual_leaf_cross_quad_approx,
      DECLARE_IDX_APPROX, DECLARE_DMAX, DECLARE_QUAD, RMAX2_APPROX,
      CHECK_DMAX_DUAL(RMIN2_APPROX),
      CHECK_DIST_BOTH(RMIN2_APPROX, RMAX2_APPROX), FIND_1D_BIN_APPROX,
#ifdef OMP
      COMP_MU INCREMENT_COUNT_QUAD_OMP, OMP_TASK,
#else
      COMP_MU INCREMENT_COUNT_QUAD, /* No OpenMP */,
#endif
      rmin2, rmax2, r2prec, qcnt);
}


void kdtree_dual_leaf_cross_quad_exact(const KDT *node1, const KDT *node2,
    const double *rbin2, const int nbin,
#ifdef OMP
    DOUBLE2 **qcnt, size_t **cnt
#else
    DOUBLE2 *qcnt, size_t *cnt
#endif
    ) {
  KDTREE_DUAL_LEAF_CROSS(kdtree_dual_leaf_cross_quad_exact,
      DECLARE_IDX_EXACT, DECLARE_DMAX, DECLARE_QUAD, RMAX2_EXACT,
      CHECK_DMAX_DUAL(RMIN2_EXACT),
      CHECK_DIST_BOTH(RMIN2_EXACT, RMAX2_EXACT), FIND_1D_BIN_EXACT,
#ifdef OMP
      COMP_MU INCREMENT_COUNT_QUAD_OMP, OMP_TASK,
#else
      COMP_MU INCREMENT_COUNT_QUAD, /* No OpenMP */,
#endif
      rbin2, nbin, qcnt);
}


void kdtree_dual_leaf_cross_quad_intbin_rmin0_wt(const KDT *node1,
    const KDT *node2, const double rmax2,
#ifdef OMP
    DOUBLE2 **qcnt, double **cnt
#else
    DOUBLE2 *qcnt, double *cnt
#endif
    ) {
  KDTREE_DUAL_LEAF_CROSS(kdtree_dual_leaf_cross_quad_intbin_rmin0_wt,
      DECLARE_IDX_APPROX, /* No dmax */, DECLARE_QUAD, RMAX2_APPROX,
      /* No dmax checking */,
      CHECK_DIST_MAX(RMAX2_APPROX), FIND_1D_BIN_INTBIN_RMIN0,
#ifdef OMP
      COMP_MU INCREMENT_WEIGHT_QUAD_OMP, OMP_TASK,
#else
      COMP_MU INCREMENT_WEIGHT_QUAD, /* No OpenMP */,
#endif
      rmax2, qcnt);
}


void kdtree_dual_leaf_cross_quad_intbin_wt(const KDT *node1, const KDT *node2,
    const double rmin2, const double rmax2,
#ifdef OMP
    DOUBLE2 **qcnt, double **cnt
#else
    DOUBLE2 *qcnt, double *cnt
#endif
    ) {
  KDTREE_DUAL_LEAF_CROSS(kdtree_dual_leaf_cross_quad_intbin_wt,
      DECLARE_IDX_APPROX, DECLARE_DMAX, DECLARE_QUAD, RMAX2_APPROX,
      CHECK_DMAX_DUAL(RMIN2_APPROX),
      CHECK_DIST_BOTH(RMIN2_APPROX, RMAX2_APPROX), FIND_1D_BIN_INTBIN,
#ifdef OMP
      COMP_MU INCREMENT_WEIGHT_QUAD_OMP, OMP_TASK,
#else
      COMP_MU INCREMENT_WEIGHT_QUAD, /* No OpenMP */,
#endif
      rmin2, rmax2, qcnt);
}


void kdtree_dual_leaf_cross_quad_approx_rmin0_wt(const KDT *node1,
    const KDT *node2, const double rmax2, const double r2prec,
#ifdef OMP
    DOUBLE2 **qcnt, double **cnt
#else
    DOUBLE2 *qcnt, double *cnt
#endif
    ) {
  KDTREE_DUAL_LEAF_CROSS(kdtree_dual_leaf_cross_quad_approx_rmin0_wt,
      DECLARE_IDX_APPROX, /* No dmax */, DECLARE_QUAD, RMAX2_APPROX,
      /* No dmax checking */,
      CHECK_DIST_MAX(RMAX2_APPROX), FIND_1D_BIN_APPROX_RMIN0,
#ifdef OMP
      COMP_MU INCREMENT_WEIGHT_QUAD_OMP, OMP_TASK,
#else
      COMP_MU INCREMENT_WEIGHT_QUAD, /* No OpenMP */,
#endif
      rmax2, r2prec, qcnt);
}


void kdtree_dual_leaf_cross_quad_approx_wt(const KDT *node1, const KDT *node2,
    const double rmin2, const double rmax2, const double r2prec,
#ifdef OMP
    DOUBLE2 **qcnt, double **cnt
#else
    DOUBLE2 *qcnt, double *cnt
#endif
    ) {
  KDTREE_DUAL_LEAF_CROSS(kdtree_dual_leaf_cross_quad_approx_wt,
      DECLARE_IDX_APPROX, DECLARE_DMAX, DECLARE_QUAD, RMAX2_APPROX,
      CHECK_DMAX_DUAL(RMIN2_APPROX),
      CHECK_DIST_BOTH(RMIN2_APPROX, RMAX2_APPROX), FIND_1D_BIN_APPROX,
#ifdef OMP
      COMP_MU INCREMENT_WEIGHT_QUAD_OMP, OMP_TASK,
#else
      COMP_MU INCREMENT_WEIGHT_QUAD, /* No OpenMP */,
#endif
      rmin2, rmax2, r2prec, qcnt);
}


void kdtree_dual_leaf_cross_quad_exact_wt(const KDT *node1, const KDT *node2,
    const double *rbin2, const int nbin,
#ifdef OMP
    DOUBLE2 **qcnt, double **cnt
#else
    DOUBLE2 *qcnt, double *cnt
#endif
    ) {
  KDTREE_DUAL_LEAF_CROSS(kdtree_dual_leaf_cross_quad_exact_wt,
      DECLARE_IDX_EXACT, DECLARE_DMAX, DECLARE_QUAD, RMAX2_EXACT,
      CHECK_DMAX_DUAL(RMIN2_EXACT),
      CHECK_DIST_BOTH(RMIN2_EXACT, RMAX2_EXACT), FIND_1D_BIN_EXACT,
#ifdef OMP
      COMP_MU INCREMENT_WEIGHT_QUAD_OMP, OMP_TASK,
#else
      COMP_MU INCREMENT_WEIGHT_QUAD, /* No OpenMP */,
#endif
      rbin2, nbin, qcnt);
}

