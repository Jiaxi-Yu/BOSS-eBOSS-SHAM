#include "algebra.h"

/******************************************************************************
Function `last_compare`:
  Compare the last coordinate of two data points in composite space.
  Same as the function `kth_compare`, but fix `k` to 0.
  This function serves as the argument for `qsort`.
******************************************************************************/
int last_compare(const void *a, const void *b) {
  register int i;
  if (((DATA *) a)->x[0] > ((DATA *) b)->x[0]) return 1;
  if (((DATA *) a)->x[0] < ((DATA *) b)->x[0]) return -1;

  for (i = DIM - 1; i != 0; i--) {
    if (((DATA *) a)->x[i] > ((DATA *) b)->x[i]) return 1;
    if (((DATA *) a)->x[i] < ((DATA *) b)->x[i]) return -1;
  }
  return 0;
}


/******************************************************************************
Function `median_split`:
  Find the median point of a dataset `data` with `n` points based on the `k`-th
  coordinate comparison in composite space, and move smaller elements to the
  left, larger elements to the right.
  (cf. section 8.5 of Press et al. 2007, Numerical Recipes 3rd Edition)
Input:
  A dataset `data` with `n` points.
Output:
  The median point.

Arguments:
  * `k`:        the direction of the coordinate to be compared;
  * `data`:     the input dataset;
  * `n`:        number of elements of the input dataset.
******************************************************************************/
#define SWAP(a,b) tmp=(a);(a)=(b);(b)=tmp
DATA median_split(const int k, DATA *data, const size_t n) {
  size_t i, ir, j, l, mid, index;
  DATA a, tmp;

  index = n >> 1;         // index of the median point, i.e. n / 2.
  l = 0;
  ir = n - 1;
  for (;;) {
    if (ir <= l + 1) {
      if (ir == l + 1 && kth_compare(k, data + ir, data + l) < 0) {
        SWAP(data[l], data[ir]);
      }
      return data[index];
    }
    else {
      mid = (l + ir) >> 1;
      SWAP(data[mid], data[l + 1]);
      if (kth_compare(k, data + l, data + ir) > 0) {
        SWAP(data[l], data[ir]);
      }
      if (kth_compare(k, data + l + 1, data + ir) > 0) {
        SWAP(data[l + 1], data[ir]);
      }
      if (kth_compare(k, data + l, data + l + 1) > 0) {
        SWAP(data[l], data[l+1]);
      }
      i = l + 1;
      j = ir;
      a = data[l + 1];
      for (;;) {
        do i++; while (kth_compare(k, data + i, &a) < 0);
        do j--; while (kth_compare(k, data + j, &a) > 0);
        if (j < i) break;
        SWAP(data[i], data[j]);
      }
      data[l+1] = data[j];
      data[j] = a;
      if (j >= index) ir = j - 1;
      if (j <= index) l = i;
    }
  }
}

