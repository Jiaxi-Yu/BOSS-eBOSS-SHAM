/*********************************************************
**                                                      **
**      k-D Tree (KDT)                                  **
**      Author: Cheng Zhao <zhaocheng03@gmail.com>      **
**                                                      **
*********************************************************/

#ifndef _KDTREE_H_
#define _KDTREE_H_

#include "fcfc.h"
#include "algebra.h"

typedef struct kdtree {
  size_t id;
  size_t n;
  DATA *data;
  DATA min;
  DATA max;
  double wt;
  struct kdtree *left;
  struct kdtree *right;
} KDT;


KDT* kdtree_build(size_t *, DATA *, const size_t);

void kdtree_free(KDT *);

double kdtree_weight(KDT *);

void kdtree_weight_1(KDT *);


/********** kdtree_cnt.c **********/

void kdtree_count_1d_auto_pairs(const KDT *, const int, const double,
    const double, const int, const int, const int, const double *,
    const double *,
    const size_t,
#ifdef OMP
    const int, size_t **, double **, DOUBLE2 **,
#else
    size_t *, double *, DOUBLE2 *,
#endif
    DOUBLE2 *);

void kdtree_count_1d_cross_pairs(const KDT *, const KDT *, const int,
    const double, const double, const int, const int, const int,
    const double *, const double *, const size_t,
#ifdef OMP
    const int, size_t **, double **, DOUBLE2 **,
#else
    size_t *, double *, DOUBLE2 *,
#endif
    DOUBLE2 *);

///////////////////////////////////////////////////////////////////////////////

#ifdef OMP

void kdtree_dual_leaf_auto_intbin_rmin0(const KDT *, const KDT *, const double,
    size_t **);
void kdtree_dual_leaf_auto_intbin(const KDT *, const KDT *, const double,
    const double, size_t **);
void kdtree_dual_leaf_auto_approx_rmin0(const KDT *, const KDT *, const double,
    const double, size_t **);
void kdtree_dual_leaf_auto_approx(const KDT *, const KDT *, const double,
    const double, const double, size_t **);
void kdtree_dual_leaf_auto_exact(const KDT *, const KDT *, const double *,
    const int, size_t **);

void kdtree_dual_leaf_auto_intbin_rmin0_wt(const KDT *, const KDT *,
    const double, double **);
void kdtree_dual_leaf_auto_intbin_wt(const KDT *, const KDT *, const double,
    const double, double **);
void kdtree_dual_leaf_auto_approx_rmin0_wt(const KDT *, const KDT *,
    const double, const double, double **);
void kdtree_dual_leaf_auto_approx_wt(const KDT *, const KDT *, const double,
    const double, const double, double **);
void kdtree_dual_leaf_auto_exact_wt(const KDT *, const KDT *, const double *,
    const int, double **);

void kdtree_dual_leaf_cross_intbin_rmin0(const KDT *, const KDT *,
    const double, size_t **);
void kdtree_dual_leaf_cross_intbin(const KDT *, const KDT *, const double,
    const double, size_t **);
void kdtree_dual_leaf_cross_approx_rmin0(const KDT *, const KDT *,
    const double, const double, size_t **);
void kdtree_dual_leaf_cross_approx(const KDT *, const KDT *, const double,
    const double, const double, size_t **);
void kdtree_dual_leaf_cross_exact(const KDT *, const KDT *, const double *,
    const int, size_t **);

void kdtree_dual_leaf_cross_intbin_rmin0_wt(const KDT *, const KDT *,
    const double, double **);
void kdtree_dual_leaf_cross_intbin_wt(const KDT *, const KDT *, const double,
    const double, double **);
void kdtree_dual_leaf_cross_approx_rmin0_wt(const KDT *, const KDT *,
    const double, const double, double **);
void kdtree_dual_leaf_cross_approx_wt(const KDT *, const KDT *, const double,
    const double, const double, double **);
void kdtree_dual_leaf_cross_exact_wt(const KDT *, const KDT *, const double *,
    const int, double **);

// Quadrupole.

void kdtree_dual_leaf_auto_quad_intbin_rmin0(const KDT *, const KDT *,
    const double, DOUBLE2 **, size_t **);
void kdtree_dual_leaf_auto_quad_intbin(const KDT *, const KDT *, const double,
    const double, DOUBLE2 **, size_t **);
void kdtree_dual_leaf_auto_quad_approx_rmin0(const KDT *, const KDT *,
    const double, const double, DOUBLE2 **, size_t **);
void kdtree_dual_leaf_auto_quad_approx(const KDT *, const KDT *, const double,
    const double, const double, DOUBLE2 **, size_t **);
void kdtree_dual_leaf_auto_quad_exact(const KDT *, const KDT *, const double *,
    const int, DOUBLE2 **, size_t **);

void kdtree_dual_leaf_auto_quad_intbin_rmin0_wt(const KDT *, const KDT *,
    const double, DOUBLE2 **, double **);
void kdtree_dual_leaf_auto_quad_intbin_wt(const KDT *, const KDT *,
    const double, const double, DOUBLE2 **, double **);
void kdtree_dual_leaf_auto_quad_approx_rmin0_wt(const KDT *, const KDT *,
    const double, const double, DOUBLE2 **, double **);
void kdtree_dual_leaf_auto_quad_approx_wt(const KDT *, const KDT *,
    const double, const double, const double, DOUBLE2 **, double **);
void kdtree_dual_leaf_auto_quad_exact_wt(const KDT *, const KDT *,
    const double *, const int, DOUBLE2 **, double **);

void kdtree_dual_leaf_cross_quad_intbin_rmin0(const KDT *, const KDT *,
    const double, DOUBLE2 **, size_t **);
void kdtree_dual_leaf_cross_quad_intbin(const KDT *, const KDT *, const double,
    const double, DOUBLE2 **, size_t **);
void kdtree_dual_leaf_cross_quad_approx_rmin0(const KDT *, const KDT *,
    const double, const double, DOUBLE2 **, size_t **);
void kdtree_dual_leaf_cross_quad_approx(const KDT *, const KDT *, const double,
    const double, const double, DOUBLE2 **, size_t **);
void kdtree_dual_leaf_cross_quad_exact(const KDT *, const KDT *, const double *,
    const int, DOUBLE2 **, size_t **);

void kdtree_dual_leaf_cross_quad_intbin_rmin0_wt(const KDT *, const KDT *,
    const double, DOUBLE2 **, double **);
void kdtree_dual_leaf_cross_quad_intbin_wt(const KDT *, const KDT *,
    const double, const double, DOUBLE2 **, double **);
void kdtree_dual_leaf_cross_quad_approx_rmin0_wt(const KDT *, const KDT *,
    const double, const double, DOUBLE2 **, double **);
void kdtree_dual_leaf_cross_quad_approx_wt(const KDT *, const KDT *,
    const double, const double, const double, DOUBLE2 **, double **);
void kdtree_dual_leaf_cross_quad_exact_wt(const KDT *, const KDT *,
    const double *, const int, DOUBLE2 **, double **);

#else

void kdtree_dual_leaf_auto_intbin_rmin0(const KDT *, const KDT *, const double,
    size_t *);
void kdtree_dual_leaf_auto_intbin(const KDT *, const KDT *, const double,
    const double, size_t *);
void kdtree_dual_leaf_auto_approx_rmin0(const KDT *, const KDT *, const double,
    const double, size_t *);
void kdtree_dual_leaf_auto_approx(const KDT *, const KDT *, const double,
    const double, const double, size_t *);
void kdtree_dual_leaf_auto_exact(const KDT *, const KDT *, const double *,
    const int, size_t *);

void kdtree_dual_leaf_auto_intbin_rmin0_wt(const KDT *, const KDT *,
    const double, double *);
void kdtree_dual_leaf_auto_intbin_wt(const KDT *, const KDT *, const double,
    const double, double *);
void kdtree_dual_leaf_auto_approx_rmin0_wt(const KDT *, const KDT *,
    const double, const double, double *);
void kdtree_dual_leaf_auto_approx_wt(const KDT *, const KDT *, const double,
    const double, const double, double *);
void kdtree_dual_leaf_auto_exact_wt(const KDT *, const KDT *, const double *,
    const int, double *);

void kdtree_dual_leaf_cross_intbin_rmin0(const KDT *, const KDT *,
    const double, size_t *);
void kdtree_dual_leaf_cross_intbin(const KDT *, const KDT *, const double,
    const double, size_t *);
void kdtree_dual_leaf_cross_approx_rmin0(const KDT *, const KDT *,
    const double, const double, size_t *);
void kdtree_dual_leaf_cross_approx(const KDT *, const KDT *, const double,
    const double, const double, size_t *);
void kdtree_dual_leaf_cross_exact(const KDT *, const KDT *, const double *,
    const int, size_t *);

void kdtree_dual_leaf_cross_intbin_rmin0_wt(const KDT *, const KDT *,
    const double, double *);
void kdtree_dual_leaf_cross_intbin_wt(const KDT *, const KDT *, const double,
    const double, double *);
void kdtree_dual_leaf_cross_approx_rmin0_wt(const KDT *, const KDT *,
    const double, const double, double *);
void kdtree_dual_leaf_cross_approx_wt(const KDT *, const KDT *, const double,
    const double, const double, double *);
void kdtree_dual_leaf_cross_exact_wt(const KDT *, const KDT *, const double *,
    const int, double *);

// Quadrupole.

void kdtree_dual_leaf_auto_quad_intbin_rmin0(const KDT *, const KDT *,
    const double, DOUBLE2 *, size_t *);
void kdtree_dual_leaf_auto_quad_intbin(const KDT *, const KDT *, const double,
    const double, DOUBLE2 *, size_t *);
void kdtree_dual_leaf_auto_quad_approx_rmin0(const KDT *, const KDT *,
    const double, const double, DOUBLE2 *, size_t *);
void kdtree_dual_leaf_auto_quad_approx(const KDT *, const KDT *, const double,
    const double, const double, DOUBLE2 *, size_t *);
void kdtree_dual_leaf_auto_quad_exact(const KDT *, const KDT *, const double *,
    const int, DOUBLE2 *, size_t *);

void kdtree_dual_leaf_auto_quad_intbin_rmin0_wt(const KDT *, const KDT *,
    const double, DOUBLE2 *, double *);
void kdtree_dual_leaf_auto_quad_intbin_wt(const KDT *, const KDT *,
    const double, const double, DOUBLE2 *, double *);
void kdtree_dual_leaf_auto_quad_approx_rmin0_wt(const KDT *, const KDT *,
    const double, const double, DOUBLE2 *, double *);
void kdtree_dual_leaf_auto_quad_approx_wt(const KDT *, const KDT *,
    const double, const double, const double, DOUBLE2 *, double *);
void kdtree_dual_leaf_auto_quad_exact_wt(const KDT *, const KDT *,
    const double *, const int, DOUBLE2 *, double *);

void kdtree_dual_leaf_cross_quad_intbin_rmin0(const KDT *, const KDT *,
    const double, DOUBLE2 *, size_t *);
void kdtree_dual_leaf_cross_quad_intbin(const KDT *, const KDT *, const double,
    const double, DOUBLE2 *, size_t *);
void kdtree_dual_leaf_cross_quad_approx_rmin0(const KDT *, const KDT *,
    const double, const double, DOUBLE2 *, size_t *);
void kdtree_dual_leaf_cross_quad_approx(const KDT *, const KDT *, const double,
    const double, const double, DOUBLE2 *, size_t *);
void kdtree_dual_leaf_cross_quad_exact(const KDT *, const KDT *, const double *,
    const int, DOUBLE2 *, size_t *);

void kdtree_dual_leaf_cross_quad_intbin_rmin0_wt(const KDT *, const KDT *,
    const double, DOUBLE2 *, double *);
void kdtree_dual_leaf_cross_quad_intbin_wt(const KDT *, const KDT *,
    const double, const double, DOUBLE2 *, double *);
void kdtree_dual_leaf_cross_quad_approx_rmin0_wt(const KDT *, const KDT *,
    const double, const double, DOUBLE2 *, double *);
void kdtree_dual_leaf_cross_quad_approx_wt(const KDT *, const KDT *,
    const double, const double, const double, DOUBLE2 *, double *);
void kdtree_dual_leaf_cross_quad_exact_wt(const KDT *, const KDT *,
    const double *, const int, DOUBLE2 *, double *);

#endif

/********** kdtree_cnt.c **********/

#endif

