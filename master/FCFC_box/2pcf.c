#include "load_conf.h"
#include "kdtree.h"
#include <math.h>

int main(int argc, char *argv[]) {
  char conf_file[MAX_BUF];
  size_t i, Nd, Nr, Ncnt, Nsample;
  DATA *data, *rand;
  KDT *dtree, *rtree;
  size_t *ncnt;
  double *rbin, *rbin2, *wcnt;
  DOUBLE2 *dd, *dr, *rr, *sample, *output, *qcnt;
  double Wd, Wr;
  int ecode, flag_cnvt;
  CONF conf;
#ifdef OMP
  int Nomp, j;
  size_t **pncnt;               // Private number count for threads.
  double **pwcnt;               // Private weight count for threads.
  DOUBLE2 **pqcnt;              // Private multipole count for threads.

  Nomp = omp_get_max_threads();
  if (Nomp < 2)
    P_WRN("this program is compiled with OpenMP, "
        "but the available number of threads is %d.\n"
        "You may want to reset $OMP_NUM_THREADS.\n", Nomp);
  pncnt = NULL;
  pwcnt = NULL;
  pqcnt = NULL;
  omp_set_num_threads(Nomp);
#endif

/******************************************************************************
  Load configuration.
******************************************************************************/
  init_conf(&conf);
  strcpy(conf_file, DEFAULT_CONF_FILE);
  read_opt(argc, argv, conf_file, &conf);

  printf("Loading configuration ... ");
  fflush(stdout);
  read_conf(conf_file, &conf);

  if ((ecode = check_conf(&conf))) {
    printf(FMT_FAIL);
    P_EXT("please check your configuration.\n"
        "Try the -h option for more information.\n");
    return ecode;
  }
  if (conf.verb) print_conf(&conf);
  printf(FMT_DONE);             // Done with loading configuration.

  if (!conf.cmode && !conf.cfmode) {
    printf("Nothing to be done.\n");
    return 0;
  }


/******************************************************************************
  Create distance bins.
******************************************************************************/
  rbin = rbin2 = wcnt = NULL;
  qcnt = NULL;
  ncnt = NULL;
  if (conf.cmode) {     // Pairs will be counted, so distance bins are created.
    printf("Create bins for distances and counts ... ");
    fflush(stdout);
    if (conf.dbtype == 0) {
      if ((ecode = read_1d_dist_bin(conf.dbin, &rbin, &conf.rnum, conf.verb))) {
        printf(FMT_FAIL);
        P_EXT("failed to read distance bins from file.\n");
        return ecode;
      }
      conf.rmin = rbin[0];
      conf.rmax = rbin[conf.rnum];
    }
    else {
      if ((ecode = create_dist_bin(conf.rmin, conf.rmax, conf.rnum, &rbin))) {
        printf(FMT_FAIL);
        P_EXT("failed to create distance bins.\n");
        return ecode;
      }
    }

    if (conf.r2prec < -APPROX_R2_PREC || conf.r2prec > APPROX_R2_PREC) {
      MY_ALLOC(rbin2, double, conf.rnum + 1,
          "failed to allocate memory for squared distance bins.\n");
      for (i = 0; i < conf.rnum + 1; i++)
        rbin2[i] = rbin[i] * rbin[i];
    }

    if (conf.dwt == 0 || conf.rwt == 0) {       // Number counts.
      if ((ecode = create_1d_counts(conf.rmin, conf.rmax, conf.rnum,
          conf.r2prec, NUM_COUNT, (void **) &ncnt, &Ncnt))) {
        printf(FMT_FAIL);
        P_EXT("failed to create count bins.\n");
        return ecode;
      }
#ifdef OMP
      MY_ALLOC(pncnt, size_t *, Nomp,
          "failed to allocate memory of count bins for OpenMP threads.\n");
      for (j = 0; j < Nomp; j++) {
        MY_ALLOC(pncnt[j], size_t, Ncnt,
            "failed to allocate memory of count bins for OpenMP threads.\n");
      }
#endif
    }
    if (conf.dwt != 0 || conf.rwt != 0) {       // Weight counts.
      if ((ecode = create_1d_counts(conf.rmin, conf.rmax, conf.rnum,
          conf.r2prec, WT_COUNT, (void **) &wcnt, &Ncnt))) {
        printf(FMT_FAIL);
        P_EXT("failed to create count bins.\n");
        return ecode;
      }
#ifdef OMP
      MY_ALLOC(pwcnt, double *, Nomp,
          "failed to allocate memory of count bins for OpenMP threads.\n");
      for (j = 0; j < Nomp; j++) {
        MY_ALLOC(pwcnt[j], double, Ncnt,
            "failed to allocate memory of count bins for OpenMP threads.\n");
      }
#endif
    }
    if (conf.moment == 2) {                     // Quadrupole counts.
      MY_ALLOC(qcnt, DOUBLE2 , Ncnt,
          "failed to allocate memory of quadrupole counts.\n");
#ifdef OMP
      MY_ALLOC(pqcnt, DOUBLE2 *, Nomp,
          "failed to allocate memory of quadrupole count bins"
          "for OpenMP threads.\n");
      for (j = 0; j < Nomp; j++) {
        MY_ALLOC(pqcnt[j], DOUBLE2, Ncnt,
            "failed to allocate memory of quadrupole count bins"
            "for OpenMP threads.\n");
      }
#endif
    }
    printf(FMT_DONE);   // Done with creating bins for distances and counts.
  }
  else {        // Pairs will not be counted, so read distance bins from file.
    conf.rnum = 0;
  }

/******************************************************************************
  Read data and construct the tree for data.
******************************************************************************/
  data = rand = NULL;
  dtree = rtree = NULL;
  Wd = Wr = 0;
  flag_cnvt = 0;
  if (conf.cmode != 4 && conf.cmode != 0) {     // Data is required.
    printf("Reading data catalog ... ");
    fflush(stdout);
    if ((ecode = read_cat(conf.data, conf.dskip, conf.dwt, conf.daux,
        conf.dsel, conf.dx, conf.dy, conf.dz, conf.dw, conf.da,
        &data, &Nd, conf.verb))) {
      printf(FMT_FAIL);
      P_EXT("failed to read the data catalog.\n");
      return ecode;
    }

    if (conf.dcnvt && conf.verb) {
      printf("  Converting coordinates ... ");
      fflush(stdout);
    }
    if (conf.dcnvt == 1)                // Flat LCDM.
      rdz2xyz_flat_LCDM(conf.OmegaM, data, Nd);
    else if (conf.dcnvt == 2) {         // Read convention sample from file.
      if ((ecode = read_z2cd(conf.zlist, &sample, &Nsample,
          conf.verb))) {
        printf(FMT_FAIL);
        P_EXT("failed to read the redshift list file.\n");
        return ecode;
      }

      flag_cnvt = 1;
      rdz2xyz(sample, Nsample, data, Nd);
    }
    if (conf.dcnvt && conf.verb)
      printf("\r  Done with coordinate convention.\n");
    printf(FMT_DONE);         // Done with reading data catalog.

    printf("Construct the tree for data ... ");
    fflush(stdout);
    i = 0;
    dtree = kdtree_build(&i, data, Nd);
    if (!dtree) {
      printf(FMT_FAIL);
      P_EXT("failed to construct the tree.\n");
      return ERR_TREE;
    }

    if (conf.dwt) {
      if (conf.verb) printf("\n  Computing weights ... ");
      fflush(stdout);
      Wd = kdtree_weight(dtree);
      if (conf.verb) printf("\r  Done with computing weights.\n");
    }
    else if (conf.flag_dr == 1 && conf.rwt) {
      if (conf.verb) printf("\n  Computing weights ... ");
      fflush(stdout);
      kdtree_weight_1(dtree);
      Wd = Nd;
      for (i = 0; i < Nd; i++) data[i].wt = 1;
      if (conf.verb) printf("\r  Done with computing weights.\n");
    }

    printf(FMT_DONE);           // Done with constructing the tree for data.
  }

/******************************************************************************
  Read random and construct the tree for random.
******************************************************************************/
  if (conf.cmode != 1 && conf.cmode !=0) {      // Random is required.
    printf("Reading random catalog ... ");
    fflush(stdout);
    if ((ecode = read_cat(conf.rand, conf.rskip, conf.rwt, conf.raux,
        conf.rsel, conf.rx, conf.ry, conf.rz, conf.rw, conf.ra,
        &rand, &Nr, conf.verb))) {
      printf(FMT_FAIL);
      P_EXT("failed to read the random catalog.\n");
      return ecode;
    }

    if (conf.rcnvt && conf.verb) {
      printf("  Converting coordinates ... ");
      fflush(stdout);
    }
    if (conf.rcnvt == 1)                // Flat LCDM.
      rdz2xyz_flat_LCDM(conf.OmegaM, rand, Nr);
    else if (conf.rcnvt == 2) {         // Read convention from file.
      if (flag_cnvt == 0) {
        if ((ecode = read_z2cd(conf.zlist, &sample, &Nsample,
            conf.verb))) {
          printf(FMT_FAIL);
          P_EXT("failed to read the redshift list file.\n");
          return ecode;
        }
        flag_cnvt = 1;
      }
      rdz2xyz(sample, Nsample, rand, Nr);
    }
    if (conf.rcnvt && conf.verb)
      printf("\r  Done with coordinate convention.\n");
    printf(FMT_DONE);         // Done with reading random catalog.

    printf("Construct the tree for random ... ");
    fflush(stdout);
    i = 0;
    rtree = kdtree_build(&i, rand, Nr);
    if (!rtree) {
      printf(FMT_FAIL);
      P_EXT("failed to construct the tree.\n");
      return ERR_TREE;
    }

    if (conf.rwt) {
      if (conf.verb) printf("\n  Computing weights ... ");
      fflush(stdout);
      Wr = kdtree_weight(rtree);
      if (conf.verb) printf("\r  Done with computing weights.\n");
    }
    else if (conf.flag_dr == 1 && conf.dwt != 0) {
      if (conf.verb) printf("\n  Computing weights ... ");
      fflush(stdout);
      kdtree_weight_1(rtree);
      Wr = Nr;
      for (i = 0; i < Nr; i++) rand[i].wt = 1;
      if (conf.verb) printf("\r  Done with computing weights.\n");
    }
    printf(FMT_DONE);           // Done with constructing the tree for random.
  }

/******************************************************************************
  Count (read) DD pairs.
******************************************************************************/
  dd = dr = rr = NULL;
  if (conf.flag_dd && conf.rnum != 0) {
    MY_ALLOC(dd, DOUBLE2, conf.rnum,
        "failed to allocate memory for DD pairs.\n");
  }
  if (conf.flag_dd == 1) {              // DD pairs will be computed.
    printf("Counting DD pairs ... ");
    fflush(stdout);
    memset(dd, 0, sizeof(DOUBLE2) * conf.rnum);

#ifdef OMP
    kdtree_count_1d_auto_pairs(dtree, conf.dwt, conf.rmin, conf.rmax,
        conf.rnum, conf.r2prec, conf.moment, rbin, rbin2, Ncnt, Nomp, pncnt,
        pwcnt, pqcnt, dd);
#else
    kdtree_count_1d_auto_pairs(dtree, conf.dwt, conf.rmin, conf.rmax,
        conf.rnum, conf.r2prec, conf.moment, rbin, rbin2, Ncnt, ncnt, wcnt,
        qcnt, dd);
#endif

    if (conf.dwt)               // Counting with weights.
      write_1d_counts(conf.dd, conf.rnum, Wd * Wd, rbin, dd);
    else                        // Counting with numbers.
      write_1d_counts(conf.dd, conf.rnum, (double) Nd * (double) Nd, rbin, dd);

    if(conf.verb) printf("\n  DD counts are saved to file or stdout.\n");
    printf(FMT_DONE);           // Done with DD number counts.
  }
  else if (conf.flag_dd == 2) {         // DD pairs will be read from file.
    printf("Reading DD pairs ... ");
    fflush(stdout);
    if ((ecode = read_1d_counts(conf.dd, conf.rnum, &rbin, &dd, &conf.rnum,
        conf.verb))) {
      printf(FMT_FAIL);
      P_EXT("failed to read DD pairs from file.\n");
      return ecode;
    }
    printf(FMT_DONE);           // Done with reading DD pairs.
  }

/******************************************************************************
  Count (read) DR pairs.
******************************************************************************/
  if (conf.flag_dr && conf.rnum != 0) {
    MY_ALLOC(dr, DOUBLE2, conf.rnum,
        "failed to allocate memory for DR pairs.\n");
  }
  if (conf.flag_dr == 1) {              // DR pairs will be computed.
    printf("Counting DR pairs ... ");
    fflush(stdout);
    memset(dr, 0, sizeof(DOUBLE2) * conf.rnum);

#ifdef OMP
    kdtree_count_1d_cross_pairs(dtree, rtree, conf.dwt + conf.rwt, conf.rmin,
        conf.rmax, conf.rnum, conf.r2prec, conf.moment, rbin, rbin2, Ncnt,
        Nomp, pncnt, pwcnt, pqcnt, dr);
#else
    kdtree_count_1d_cross_pairs(dtree, rtree, conf.dwt + conf.rwt, conf.rmin,
        conf.rmax, conf.rnum, conf.r2prec, conf.moment, rbin, rbin2, Ncnt,
        ncnt, wcnt, qcnt, dr);
#endif

    if (conf.dwt || conf.rwt)   // Counting with weights.
      write_1d_counts(conf.dr, conf.rnum, Wd * Wr, rbin, dr);
    else                        // Counting with numbers.
      write_1d_counts(conf.dr, conf.rnum, (double) Nd * (double) Nr, rbin, dr);

    if(conf.verb) printf("\n  DR counts are saved to file or stdout.\n");
    printf(FMT_DONE);           // Done with DR number counts.
  }
  else if (conf.flag_dr == 2) {         // DR pairs will be read from file.
    printf("Reading DR pairs ... ");
    fflush(stdout);
    if ((ecode = read_1d_counts(conf.dr, conf.rnum, &rbin, &dr, &conf.rnum,
        conf.verb))) {
      printf(FMT_FAIL);
      P_EXT("failed to read DR pairs from file.\n");
      return ecode;
    }
    printf(FMT_DONE);           // Done with reading DR pairs.
  }

  if (conf.flag_dd == 1 || conf.flag_dr == 1) {
    printf("Releasing memory for data ... ");
    fflush(stdout);
    kdtree_free(dtree);
    free(data);
    printf(FMT_DONE);           // Done with releasing memory for data.
  }

/******************************************************************************
  Count (read) RR pairs.
******************************************************************************/
  if ((conf.flag_rr && conf.rnum != 0) || conf.cfmode == 3) {
    MY_ALLOC(rr, DOUBLE2, conf.rnum,
        "failed to allocate memory for RR pairs.\n");
  }
  if (conf.flag_rr == 1) {              // RR pairs will be computed.
    printf("Counting RR pairs ... ");
    fflush(stdout);
    memset(rr, 0, sizeof(DOUBLE2) * conf.rnum);

#ifdef OMP
    kdtree_count_1d_auto_pairs(rtree, conf.rwt, conf.rmin, conf.rmax,
        conf.rnum, conf.r2prec, conf.moment, rbin, rbin2, Ncnt, Nomp, pncnt,
        pwcnt, pqcnt, rr);
#else
    kdtree_count_1d_auto_pairs(rtree, conf.rwt, conf.rmin, conf.rmax,
        conf.rnum, conf.r2prec, conf.moment, rbin, rbin2, Ncnt, ncnt, wcnt,
        qcnt, rr);
#endif

    if (conf.rwt)               // Counting with weights.
      write_1d_counts(conf.rr, conf.rnum, Wr * Wr, rbin, rr);
    else                        // Counting with numbers.
      write_1d_counts(conf.rr, conf.rnum, (double) Nr * (double) Nr, rbin, rr);

    if(conf.verb) printf("\n  RR counts are saved to file or stdout.\n");
    printf(FMT_DONE);           // Done with RR number counts.
  }
  else if (conf.flag_rr == 2) {         // RR pairs will be read from file.
    printf("Reading RR pairs ... ");
    fflush(stdout);
    if ((ecode = read_1d_counts(conf.rr, conf.rnum, &rbin, &rr, &conf.rnum,
        conf.verb))) {
      printf(FMT_FAIL);
      P_EXT("failed to read RR pairs from file.\n");
      return ecode;
    }
    printf(FMT_DONE);           // Done with reading RR pairs.
  }

  if (conf.flag_dr == 1 || conf.flag_rr == 1) {
    printf("Releasing memory for random ... ");
    fflush(stdout);
    kdtree_free(rtree);
    free(rand);
    printf(FMT_DONE);           // Done with releasing memory for random.
  }

/******************************************************************************
  Compute correlation function.
******************************************************************************/
  output = NULL;
  if (conf.cfmode) {
    printf("Computing the correlation function ... ");
    fflush(stdout);
    MY_ALLOC(output, DOUBLE2, conf.rnum,
        "failed to allocate memory for the correlation function.\n");
  }
  if (conf.cfmode == 1) {               // (DD - 2DR + RR) / RR.
    for (i = 0; i < conf.rnum; i++) {
      output[i].v[0] = (dd[i].v[0] - 2 * dr[i].v[0] + rr[i].v[0]) / rr[i].v[0];
      if (conf.moment == 2) {
        output[i].v[1] = (dd[i].v[1] - 2 * dr[i].v[1] + rr[i].v[1])
          / rr[i].v[0];
        output[i].v[2] = (dd[i].v[2] - 2 * dr[i].v[2] + rr[i].v[2])
          / rr[i].v[0];
      }
    }
  }
  else if (conf.cfmode == 2) {          // DD / RR - 1.
    for (i = 0; i < conf.rnum; i++) {
      output[i].v[0] = dd[i].v[0] / rr[i].v[0] - 1;
      if (conf.moment == 2) {
        output[i].v[1] = dd[i].v[1] / rr[i].v[0];
        output[i].v[2] = dd[i].v[2] / rr[i].v[0];
      }
    }
  }
  else if (conf.cfmode == 3) {
    double rr_theo;
    for (i = 0; i < conf.rnum; i++) {
      rr_theo = 4 * PI / 3 / ((double) BOXSIZE * BOXSIZE * BOXSIZE);
      rr_theo *= pow(rbin[i+1], 3) - pow(rbin[i], 3);
      output[i].v[0] = dd[i].v[0] / rr_theo - 1;
      if (conf.moment == 2) {
        output[i].v[1] = dd[i].v[1] / rr_theo;
        output[i].v[2] = dd[i].v[2] / rr_theo * 2;
      }
    }
  }
  if (conf.cfmode) {    // Write to file.
    write_1d_2pcf(conf.output, conf.rnum, rbin, output);
    if (conf.verb)
      printf("\n  The correlation function is saved.\n");
    printf(FMT_DONE);
  }

/******************************************************************************
  Release memory.
******************************************************************************/
  printf("Releasing memory ... ");
  fflush(stdout);
  free(rbin);
  if (conf.cmode) {
    if (conf.r2prec < -APPROX_R2_PREC || conf.r2prec > APPROX_R2_PREC)
      free(rbin2);
    if (conf.dwt == 0 || conf.rwt == 0) {
      free(ncnt);
#ifdef OMP
      for (j = 0; j < Nomp; j++) free(pncnt[j]);
      free(pncnt);
#endif
    }
    if (conf.dwt != 0 || conf.rwt != 0) {
      free(wcnt);
#ifdef OMP
      for (j = 0; j < Nomp; j++) free(pwcnt[j]);
      free(pwcnt);
#endif
    }
    if (conf.moment == 2) {
      free(qcnt);
#ifdef OMP
      for (j = 0; j < Nomp; j++) free(pqcnt[j]);
      free(pqcnt);
#endif
    }
  }

  if (flag_cnvt) free(sample);

  if (conf.flag_dd) free(dd);
  if (conf.flag_dr) free(dr);
  if (conf.flag_rr) free(rr);
  if (conf.cfmode) free(output);
  printf(FMT_DONE);             // Done with releasing memory.

  return 0;
}

