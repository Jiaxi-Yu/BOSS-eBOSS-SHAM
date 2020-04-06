
#ifndef _ALGEBRA_H_
#define _ALGEBRA_H_

#include <stddef.h>
#include "define.h"

DATA median_split(const int, DATA *, const size_t);
int last_compare(const void *, const void *);

/******************************************************************************
Function `kth_compare`:
  Compare the k-th coordinate of two data points in composite space.
  (cf. section 5.5 of de Berg et al. 2008, Computing Geometry 3rd Edition)
Input:
  Two data points `a` and `b`.
Output:
  * 1, if the k-th coordinate of `a` is large than that of `b`;
  * 0, if the two points are identical;
  * -1, if the k-th coordinate of `a` is less than that of `b`.

Arguments:
  * `k`:        the direction of the coordinate to be compared;
  * `a`:        the first data point;
  * `b`:        the second data point.
******************************************************************************/
inline int kth_compare(const int k, const DATA *a, const DATA *b) {
  register int i;
  if (a->x[k] > b->x[k]) return 1;
  if (a->x[k] < b->x[k]) return -1;

  for (i = (k - 1 + DIM) % DIM; i != k;
      i = (i - 1 + DIM) % DIM) {
    if (a->x[i] > b->x[i]) return 1;
    if (a->x[i] < b->x[i]) return -1;
  }
  return 0;
}


/******************************************************************************
Function `squared_distance`:
  Compute the squared Euclidean distance in 3-D space.
Input:
  Two points in 3-D space.
Output:
  The squared Euclidean distance of the two points.

Arguments:
  * `a`:        the first data point;
  * `b`:        the second data point.
******************************************************************************/
inline real squared_distance(const DATA *a, const DATA *b) {
  register real dx, dy, dz;
  dx = a->x[0] - b->x[0];
  if (dx > BOXSIZE / 2) dx -= BOXSIZE;
  if (dx < -BOXSIZE / 2) dx += BOXSIZE;
  dy = a->x[1] - b->x[1];
  if (dy > BOXSIZE / 2) dy -= BOXSIZE;
  if (dy < -BOXSIZE / 2) dy += BOXSIZE;
  dz = a->x[2] - b->x[2];
  if (dz > BOXSIZE / 2) dz -= BOXSIZE;
  if (dz < -BOXSIZE / 2) dz += BOXSIZE;
  return dx * dx + dy * dy + dz * dz;
}


/******************************************************************************
Function `min_squared_dist_between_box`:
  Compute the minimum squared distance between two boxes.
Input:
  Lower and upper corner of the two boxes represented by data points.
Output:
  The minimum squared distance between the two boxes.

Arguments:
  * `min1`:     the lower corner of the first box;
  * `max1`:     the upper corner of the first box;
  * `min2`:     the lower corner of the second box;
  * `max2`:     the upper corner of the second box;
******************************************************************************/
inline real min_squared_dist_between_box(const DATA *min1, const DATA *max1,
    const DATA *min2, const DATA *max2) {
  register int i;
  register real d, sum, d1, d2;

  sum = 0;
  for (i = 0; i < DIM; i++) {
/*
    d = (min1->x[i] < min2->x[i]) ?
      min2->x[i] - max1->x[i] : min1->x[i] - max2->x[i];
    if (d < 0) continue;
*/
    if (min1->x[i] < min2->x[i]) {
      d1 = min2->x[i] - max1->x[i];
      d2 = min1->x[i] - max2->x[i] + BOXSIZE;
      d = (d1 < d2) ? d1 : d2;
      if (d < 0) continue;
    }
    else {
      d1 = min1->x[i] - max2->x[i];
      d2 = min2->x[i] - max1->x[i] + BOXSIZE;
      d = (d1 < d2) ? d1 : d2;
      if (d < 0) continue;
    }
    sum += d * d;
  }
  return sum;
}


/******************************************************************************
Function `max_squared_dist_between_box`:
  Compute the maximum squared distance between two boxes.
Input:
  Lower and upper corner of the two boxes represented by data points.
Output:
  The maximum squared distance between the two boxes.

Arguments:
  * `min1`:     the lower corner of the first box;
  * `max1`:     the upper corner of the first box;
  * `min2`:     the lower corner of the second box;
  * `max2`:     the upper corner of the second box;
******************************************************************************/
inline real max_squared_dist_between_box(const DATA *min1, const DATA *max1,
    const DATA *min2, const DATA *max2) {
  register int i;
  register real d, sum, d1, d2;

  sum = 0;
  for (i = 0; i < DIM; i++) {
/*
    d = (min1->x[i] + max1->x[i] < min2->x[i] + max2->x[i]) ?
      max2->x[i] - min1->x[i] : max1->x[i] - min2->x[i];
*/
    if (min1->x[i] + max1->x[i] < min2->x[i] + max2->x[i]) {
      d1 = max2->x[i] - min1->x[i];
      if (d1 > BOXSIZE / 2) d1 -= BOXSIZE;
      d2 = min2->x[i] - max1->x[i];
    }
    else {
      d1 = max1->x[i] - min2->x[i];
      if (d1 > BOXSIZE / 2) d1 -= BOXSIZE;
      d2 = min1->x[i] - max2->x[i];
    }
    d = (d1 > d2) ? d2 : d1;
    sum += d * d;
  }
  return sum;
}

#endif
