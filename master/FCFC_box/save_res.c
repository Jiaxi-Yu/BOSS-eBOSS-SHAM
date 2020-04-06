#include "fcfc.h"

/******************************************************************************
Function `write_1d_counts`:
  Normalize the pair counts for 1-D (isotropic) correlation and write to a
  file. If the file cannot be written, then write to standard output.
Input:
  The name of the output file and the pair count arrays to be saved.

Arguments:
  * `fname`:    name of the output file;
  * `n`:        number of distance bins;
  * `rbin`:     the array for distance bins;
  * `cnt`:      the array for pair counts;
  * `norm`:     the normalisation factor for pair counts.
******************************************************************************/
void write_1d_counts(const char *fname, const size_t n, const double norm,
    const double *rbin, DOUBLE2 *cnt) {
  FILE *fp;
  size_t i;
  double tmp[3];
  int flag = 0;

  if (!(fp = fopen(fname,"w"))) {
    P_WRN("cannot write to file `%s'.\n"
        "Writing to the standard output instead:\n", fname);
    fp = stdout;
    flag = 1;
  }

  for (i = 0; i < n; i++) {
    tmp[0] = cnt[i].v[0];
    tmp[1] = cnt[i].v[1];
    tmp[2] = cnt[i].v[2];
    cnt[i].v[0] /= norm;
    cnt[i].v[1] /= norm;
    cnt[i].v[2] /= norm;
    fprintf(fp, OFMT_DBL " " OFMT_DBL " " OFMT_DBL " " OFMT_DBL " " OFMT_DBL
        " " OFMT_DBL " " OFMT_DBL " " OFMT_DBL "\n",
        rbin[i], rbin[i + 1], tmp[0], cnt[i].v[0], tmp[1], cnt[i].v[1],
        tmp[2], cnt[i].v[2]);
  }

  if (!flag) fclose(fp);        // Do not close stdout.
}


/******************************************************************************
Function `write_1d_2pcf`:
  Write the 1-D (isotropic) correlation function to a file. If the file cannot
  be written, then write to standard output.
Input:
  The name of the output file and the correlation function to be saved.

Arguments:
  *`fname`:     name of the output file;
  * `n`:        number of bins;
  * `rbin`:     the array for distance bins;
  * `cf`:       the array for correlation function.
******************************************************************************/
void write_1d_2pcf(const char *fname, const size_t n, const double *rbin,
    const DOUBLE2 *cf) {
  FILE *fp;
  size_t i;
  int flag = 0;

  if (!(fp = fopen(fname, "w"))) {
    P_WRN("cannot write to file `%s'.\n"
        "Writing to the standard output instead:\n", fname);
    fp = stdout;
    flag = 1;
  }

  for (i = 0; i < n; i++) {
    fprintf(fp, OFMT_DBL " " OFMT_DBL " " OFMT_DBL " " OFMT_DBL "\n",
        (rbin[i] + rbin[i + 1]) * 0.5, cf[i].v[0], cf[i].v[1], cf[i].v[2]);
  }

  if (!flag) fclose(fp);        // Do not close stdout.
}

