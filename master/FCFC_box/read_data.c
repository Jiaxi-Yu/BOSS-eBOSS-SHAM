#include "fcfc.h"

/******************************************************************************
Function `read_cat`:
  Read 3-D galaxy data from a file. Each line of the file should be a record
  for a galaxy, with the leading 3 columns being the 3-D coordinated of the
  galaxy. The `wcol`th and `acol`th columns are read as weights and the
  auxiliary number for object selection. This function removes objects based
  on the input selection range.
Input:
  The filename and header lines of the input catalog, the columns for weights
  and auxiliary numbers, the range of interest for object selection, and the
  data structure recording information of galaxies.
Output:
  Return a non-zero integer if there is problem.

Arguments:
  * `fname`:    the filename of the input catalog;
  * `header`:   the number of header (skipped) lines for the catalog;
  * `wcol`:     the column for weights;
  * `acol`:     the column for auxilary numbers;
  * `smode`:    method for object selection;
  * `rx`:       the range of interest for the 1st column;
  * `ry`:       the range of interest for the 2nd column;
  * `rz`:       the range of interest for the 3rd column;
  * `rw`:       the range of interest for weights;
  * `ra`:       the range of interest for the auxiliary numbers;
  * `data`:     a pointer to the structure storing galaxy data;
  * `num`:      the number of galaxies;
  * `verbose`:  0 for concise outputs, 1 for detailed outputs.
******************************************************************************/
int read_cat(const char *fname, const int header,
    const int wcol, const int acol, const int smode,
    const double rx[2], const double ry[2], const double rz[2],
    const double rw[2], const double ra[2],
    DATA **data, size_t *num,
    const int verbose) {
  FILE *fcat;
  char line[BUF], *buffer, *p, *end;
  size_t cnt, i, nline, flag[21];
  int j, k, ncol;
  int flag_x, flag_y, flag_z, flag_w, flag_a;
  real *col;

  // Set the length of columns to be read.
  ncol = (wcol > acol) ? wcol : acol;
  if (ncol < 3) ncol = 3;
  // Set the selection flags.
  flag_x = flag_y = flag_z = flag_w = flag_a = 0;
  k = smode;
  if (k % 2 == 1) {
    flag_x = 1;
    k--;
  }
  if (k % 4 == 2) {
    flag_y = 1;
    k -= 2;
  }
  if (k % 8 == 4) {
    flag_z = 1;
    k -= 4;
  }
  if (k % 16 == 8) {
    flag_w = 1;
    k -= 8;
  }
  if (k == 16) {
    flag_a = 1;
  }

  if (verbose) {
    printf("\n  Filename: `%s'.\n  Counting lines ... ", fname);
    fflush(stdout);
  }

  nline = 0;
  MY_ALLOC(buffer, char, CHUNK,
      "failed to allocate memory for reading the file.\n");

  if (!(fcat = fopen(fname, "rb"))) {
    P_ERR("failed to open the file for line counting.\n");
    return ERR_FILE;
  }
  while ((cnt = fread(buffer, sizeof(char), CHUNK, fcat))) {
    p = buffer;
    end = p + cnt;
    while ((p = memchr(p, '\n', end - p))) {
      ++p;
      ++nline;
    }
  }
  fclose(fcat);
  free(buffer);
  nline -= header;

  if (verbose) {
    printf("\r  Number of records: %zu.\n  Allocating memory ... ", nline);
    fflush(stdout);
  }

  MY_ALLOC(col, real, ncol, "failed to allocate memory for reading lines.\n");
  MY_ALLOC(*data, DATA, nline, "failed to allocate memory for the tracers.\n");

  if (verbose) {
    printf("\r  ~ %.3g Mb memory allocated for the tracers.\n"
        "  Reading ...  0%%", sizeof(DATA) * nline / (1024.0 * 1024.0));
    fflush(stdout);
  }

  if (!(fcat = fopen(fname, "r"))) {
    P_ERR("failed to open the file for reading.\n");
    return ERR_FILE;
  }

  // Skip the header.
  for (i = 0; i < header; i++) {
    if (!(fgets(line, BUF, fcat)))
      P_WRN("line %zu of the header may not be correct.\n", i + 1);
  }

  // Set flags to show the percentage of reading procedure.
  for (j = 0; j < 21; j++)
    flag[j] = nline * j * 5 / 100;

  cnt = 0;
  for (j = 1; j < 21; j++) {
    for (i = flag[j - 1]; i < flag[j]; i++) {
      if (!(fgets(line, BUF, fcat))) {
        P_ERR("cannot read line %zu, reading aborted.\n", i + header + 1);
        return ERR_FILE;
      }

      p = strtok(line, DLMT);
      for (k = 0; k < ncol; k++) {
        if (p == NULL) {
          P_ERR("cannot read enough columns from line %zu.\n", i + header + 1);
          return ERR_RANGE;
        }
        if (sscanf(p, FMT_REAL, col + k) != 1) {
          P_ERR("failed to read data from line %zu.\n", i + header + 1);
          return ERR_FILE;
        }
        p = strtok(NULL, DLMT);
      }

      if (flag_x && (col[0] < rx[0] || col[0] >= rx[1])) continue;
      if (flag_y && (col[1] < ry[0] || col[1] >= ry[1])) continue;
      if (flag_z && (col[2] < rz[0] || col[2] >= rz[1])) continue;
      if (flag_w && (col[wcol - 1] < rw[0] || col[wcol - 1] >= rw[1])) continue;
      if (flag_a && (col[acol - 1] < ra[0] || col[acol - 1] >= ra[1])) continue;

      (*data)[cnt].x[0] = col[0];
      (*data)[cnt].x[1] = col[1];
      (*data)[cnt].x[2] = col[2];
      if (wcol) (*data)[cnt].wt = col[wcol - 1];
      cnt++;
    }
    if (verbose && j != 20) {
      printf("\b\b\b\b%3d%%", j * 5);
      fflush(stdout);
    }
  }
  fclose(fcat);
  free(col);

  if (cnt < nline && verbose) {
    printf("\r  %zu objects removed.\n", nline - cnt);
  }

  *num = cnt;

  if (verbose) {
    printf("\r  %zu objects recorded.\n", *num);
  }

  if (*num <= LEAF_SIZE) {
    P_ERR("not enough objects.\n");
    return ERR_RANGE;
  }

  return 0;
}


/******************************************************************************
Function `read_z2cd`:
  Read the redshift list and corresponding radial comoving distances. Each line
  of the file should have two columns, the first for redshift, and the second
  for the corresponding radial comoving distance.
Input:
  The filename of the input file, and the array storing redshift and radial
  comoving distance samples.
Output:
  Return a non-zero integer if there is problem.

Arguments:
  * `fname`:    the filename of the input file;
  * `smp`:      the structure for the redshift and comoving distance sample;
  * `ns`:       number of records for the sample;
  * `verbose`:  0 for concise outputs, 1 for detailed outputs.
******************************************************************************/
int read_z2cd(const char *fname, DOUBLE2 **smp, size_t *ns,
    const int verbose) {
  FILE *fp;
  size_t i;
  char line[BUF];

  if (verbose) {
    printf("\n  Filename: `%s'.\n  Reading ... ", fname);
    fflush(stdout);
  }

  if (!(fp = fopen(fname, "r"))) {
    P_ERR("cannot open `REDSHIFT_LIST`.\n");
    return ERR_FILE;
  }

  // Count number of records.
  i = 0;
  while (fgets(line, BUF, fp) != NULL) {
    sscanf(line, "%*[ |\t]%[^\n]", line);       // Remove leading whitespaces.
    if (line[0] != COMMENT && line[0] != '\0') i++;   // Remove comments.
  }

  if (i < 1) {
    P_ERR("no records found.\n");
    return ERR_INPUT;
  }

  *ns = i;
  MY_ALLOC(*smp, DOUBLE2, i,
      "failed to allocate memory for"
      "the redshift and comoving distance sample.\n");

  // Return to the beginning of the file and start reading.
  fseek(fp, 0, SEEK_SET);
  i = 0;
  while (fgets(line, BUF, fp) != NULL) {
    sscanf(line, "%*[ |\t]%[^\n]", line);       // Remove leading whitespaces.
    if (line[0] == COMMENT || line[0] == '\0') continue;
    if (sscanf(line, "%lf %lf", &(*smp)[i].v[0], &(*smp)[i].v[1]) != 2) {
      P_ERR("failed to read line %zu of the file.\n", i + 1);
      return ERR_FILE;
    }
    i++;
  }
  fclose(fp);

  if (verbose) printf("\r  %zu redshift convention samples added.\n", *ns);

  return 0;
}


/******************************************************************************
Function `read_1d_dist_bin`:
  Read 1-D distance bins from a file. Each line of the file should be a
  distance bins, where the first column is the lower limit, and the second
  column is the upper limit.
Input:
  The filename of the input file, and the array storing distance bins.
Output:
  Return a non-zero integer if there is problem.

Arguments:
  * `fname`:    the filename of the input file;
  * `rbin`:     the array for distance bins;
  * `n`:        number of distance bins;
  * `verbose`:  0 for concise outputs, 1 for detailed outputs.
******************************************************************************/
int read_1d_dist_bin(const char *fname, double **rbin, int *n,
    const int verbose) {
  FILE *fp;
  int i;
  char line[BUF];
  double *r1, *r2;

  if (verbose) {
    printf("\n  Filename: `%s'.\n  Reading ... ", fname);
    fflush(stdout);
  }

  if (!(fp = fopen(fname, "r"))) {
    P_ERR("cannot open `DIST_BIN_FILE`.\n");
    return ERR_FILE;
  }

  // Count number of records.
  i = 0;
  while (fgets(line, BUF, fp) != NULL) {
    sscanf(line, "%*[ |\t]%[^\n]", line);       // Remove leading whitespaces.
    if (line[0] != COMMENT && line[0] != '\0') i++;   // Remove comments.
  }

  if (i < 1) {
    P_ERR("no records found.\n");
    return ERR_INPUT;
  }

  *n = i;
  MY_ALLOC(*rbin, double, i + 1,
      "failed to allocate memory for distance bins.\n");
  MY_ALLOC(r1, double, i,
      "failed to allocate memory for reading the file.\n");
  MY_ALLOC(r2, double, i,
      "failed to allocate memory for reading the file.\n");

  // Return to the beginning of the file and start reading.
  fseek(fp, 0, SEEK_SET);
  i = 0;
  while (fgets(line, BUF, fp) != NULL) {
    sscanf(line, "%*[ |\t]%[^\n]", line);       // Remove leading whitespaces.
    if (line[0] == COMMENT || line[0] == '\0') continue;
    if (sscanf(line, "%lf %lf", r1 + i, r2 + i) != 2) {
      P_ERR("failed to read line %d of the file.\n", i + 1);
      return ERR_FILE;
    }
    i++;
  }
  fclose(fp);

  if (r1[0] < 0) {            // The distances cannot be negative.
    P_ERR("invalid distance bin: (" OFMT_DBL ", " OFMT_DBL ")\n",
        r1[0], r2[0]);
    return ERR_RANGE;
  }
  for (i = 0; i < *n - 1; i++) {      // The bins should be continuous.
    if (r2[i] - r1[i + 1] > FLT_ERR || r2[i] - r1[i + 1] < -FLT_ERR) {
      P_ERR("discontinuous distance bins: (" OFMT_DBL ", " OFMT_DBL ") and ("
          OFMT_DBL ", " OFMT_DBL ")\n", r1[i], r2[i], r1[i + 1], r2[i + 1]);
      return ERR_RANGE;
    }
  }
  for (i = 0; i < *n; i++) {  // The bins should be in increasing order.
    if (r1[i] >= r2[i]) {
      P_ERR("invalid distance bin: (" OFMT_DBL ", " OFMT_DBL ")\n",
          r1[i], r2[i]);
      return ERR_RANGE;
    }
    (*rbin)[i] = r1[i];
  }
  (*rbin)[*n] = r2[*n - 1];

  free(r1);
  free(r2);
  if (verbose) printf("\r  %d distance bins added.\n", *n);

  return 0;
}


/******************************************************************************
Function `read_1d_counts`:
  Read 1-D pair counts that have been computed before from a file.
Input:
  The filename of the input file, and arrays storing pair counts.
Output:
  Return a non-zero integer if there is problem.

Arguments:
  * `fname`:    the filename of the input file;
  * `flag`:     0 for reading distance bins, else for checking distance bins;
  * `rbin`:     the array for distance bins;
  * `cnt`:      the array for pair counts;
  * `n`:        number of distance bins;
  * `verbose`:  0 for concise outputs, 1 for detailed outputs.
******************************************************************************/
int read_1d_counts(const char *fname, const int flag, double **rbin,
    DOUBLE2 **cnt, int *n, const int verbose) {
  FILE *fp;
  int i;
  char line[BUF];
  double *r1, *r2, tmp;

  if (verbose) {
    printf("\n  Filename: `%s'.\n  Reading ... ", fname);
    fflush(stdout);
  }

  if (!(fp = fopen(fname, "r"))) {
    P_ERR("cannot open the pair count file.\n");
    return ERR_FILE;
  }

  // Count number of records.
  i = 0;
  while (fgets(line, BUF, fp) != NULL) {
    sscanf(line, "%*[ |\t]%[^\n]", line);       // Remove leading whitespaces.
    if (line[0] != COMMENT && line[0] != '\0') i++;   // Remove comments.
  }

  if (i < 1) {
    P_ERR("no records found.\n");
    return ERR_INPUT;
  }

  if (flag == 0) {              // Prepare for reading distance bins.
    *n = i;
    MY_ALLOC(*rbin, double, i + 1,
        "failed to allocate memory for distance bins.\n");
    MY_ALLOC(*cnt, DOUBLE2, i,
        "failed to allocate memory for the pair counts.\n");
  }
  else {                        // Check the number of distance bins.
    if (*n != i) {
      P_ERR("number of distance bins (%d) does not match "
          "that from the configuration (%d).\n", i, *n);
      return ERR_INPUT;
    }
  }

  MY_ALLOC(r1, double, i,
      "failed to allocate memory for reading the file.\n");
  MY_ALLOC(r2, double, i,
      "failed to allocate memory for reading the file.\n");

  // Return to the beginning of the file and start reading.
  fseek(fp, 0, SEEK_SET);
  i = 0;
  while (fgets(line, BUF, fp) != NULL) {
    sscanf(line, "%*[ |\t]%[^\n]", line);       // Remove leading whitespaces.
    if (line[0] == COMMENT || line[0] == '\0') continue;
    if (sscanf(line, "%lf %lf %lf %lf %lf %lf",
          r1 + i, r2 + i, &tmp, &(*cnt)[i].v[0], &tmp, &(*cnt)[i].v[1]) != 6) {
      P_ERR("failed to read line %d of the file.\n", i + 1);
      return ERR_FILE;
    }
    i++;
  }
  fclose(fp);

  if (flag == 0) {              // Record distance bins.
    if (r1[0] < 0) {            // The distances cannot be negative.
      P_ERR("invalid distance bin: (" OFMT_DBL ", " OFMT_DBL ")\n",
          r1[0], r2[0]);
      return ERR_RANGE;
    }
    for (i = 0; i < *n - 1; i++) {      // The bins should be continuous.
      if (r2[i] - r1[i + 1] > FLT_ERR || r2[i] - r1[i + 1] < -FLT_ERR) {
        P_ERR("discontinuous distance bins: (" OFMT_DBL ", " OFMT_DBL ") and ("
            OFMT_DBL ", " OFMT_DBL ")\n", r1[i], r2[i], r1[i + 1], r2[i + 1]);
        return ERR_RANGE;
      }
    }
    for (i = 0; i < *n; i++) {  // The bins should be in increasing order.
      if (r1[i] >= r2[i]) {
        P_ERR("invalid distance bin: (" OFMT_DBL ", " OFMT_DBL ")\n",
            r1[i], r2[i]);
        return ERR_RANGE;
      }
      (*rbin)[i] = r1[i];
    }
    (*rbin)[*n] = r2[*n - 1];
  }
  else {                        // Check the distance bins.
    for (i = 0; i < *n; i++) {
      if (r1[i] - (*rbin)[i] > FLT_ERR || r2[i] - (*rbin)[i + 1] > FLT_ERR ||
          r1[i] - (*rbin)[i] < -FLT_ERR || r2[i] - (*rbin)[i + 1] < -FLT_ERR) {
        P_ERR("distance bins do not match those from the configuration "
            "or previous files.\n");
        return ERR_INPUT;
      }
    }
  }

  free(r1);
  free(r2);
  if (verbose) printf("\r  %d records added.\n", *n);

  return 0;
}
