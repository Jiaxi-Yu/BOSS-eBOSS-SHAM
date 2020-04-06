/**********************************************************
**                                                       **
**      Fast Correlation Function Calculator (FCFC)      **
**      Author: Cheng Zhao <zhaocheng03@gmail.com>       **
**                                                       **
**********************************************************/

#ifndef _FCFC_H_
#define _FCFC_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>
#include "define.h"
#ifdef OMP
#include <omp.h>
#endif
/*
inline real squared_distance(const DATA *a, const DATA *b) {
  register real dx, dy, dz;
  dx = a->x[0] - b->x[0];
  dy = a->x[1] - b->x[1];
  dz = a->x[2] - b->x[2];
  return dx * dx + dy * dy + dz * dz;
}*/


/********** algebra.c **********/
/*
DATA median_split(const int, DATA *, const size_t);

int kth_compare(const int, const DATA *, const DATA *);

int last_compare(const void *, const void *);

//real squared_distance(const DATA *, const DATA *);

real min_squared_dist_between_box(const DATA *, const DATA *,
    const DATA *, const DATA *);

real max_squared_dist_between_box(const DATA *, const DATA *,
    const DATA *, const DATA *);
*/
/********** algebra.c **********/


/********** count_bin.c **********/

int create_dist_bin(const double, const double, const size_t, double **);

int create_1d_counts(const double, const double, const size_t, const int,
    const int, void **, size_t *);

void combine_1d_counts(const int, const int, const int, const double *,
    const void *, const DOUBLE2 *, const size_t, DOUBLE2 *);

int find_1d_bin(const real, const double *, const int);

/********** count_bin.c **********/


/********** read_data.c **********/

int read_cat(const char *, const int, const int, const int, const int,
    const double [2], const double [2], const double[2], const double [2],
    const double [2], DATA **, size_t *, const int);

int read_z2cd(const char *, DOUBLE2 **, size_t *, const int);

int read_1d_dist_bin(const char *, double **, int *, const int);

int read_1d_counts(const char *, const int, double **, DOUBLE2 **,
    int *, const int);

/********** read_data.c **********/


/********** save_res.c **********/

void write_1d_counts(const char *, const size_t, const double, const double *,
    DOUBLE2 *);

void write_1d_2pcf(const char *, const size_t, const double *, const DOUBLE2 *);

/********** save_res.c **********/


/********** rdz2xyz.c **********/

void rdz2xyz(const DOUBLE2 *, const size_t, DATA *, const size_t);

void rdz2xyz_flat_LCDM(const double, DATA *, const size_t);

double comoving_dist_flat_LCDM(double, void *);

double z2cd(double, const double);

/********** rdz2xyz.c **********/


#endif
