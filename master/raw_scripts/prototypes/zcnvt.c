#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef OMP
#include <omp.h>
#endif

#define BOXSIZE 2500
#define OMEGA_M 0.307115

#define CHUNK 1048576
#define MAXLEN 1024
#define COMMENT '#'

typedef struct {
  double x;
  double y;
  double z;
  double vz;
  double vmax;
} DATA;

size_t safe_strcpy(char *dest, const char *src, const size_t num) {
  size_t i = 0;
  while (i < num - 1 && src[i] != '\0') {
    dest[i] = src[i];
    i++;
  }
  dest[i] = '\0';
  while (src[i] != '\0') i++;
  return i;
}

int main(int argc, char *argv[]) {
  FILE *fp;
  long i, n, ndata, cnt, nrest;
  char *buf, *p, *endl, *end;
  char input[MAXLEN], output[MAXLEN], stmp[MAXLEN];
  double bsize, Om, redshift, hubble, dtmp;
  DATA *data;

  /* Reading command line options */
  if (argc != 4) {
    fprintf(stderr, "Usage: %s INPUT OUTPUT REDSHIFT\n", argv[0]);
    return 1;
  }
  strcpy(input, argv[1]);
  strcpy(output, argv[2]);
  sscanf(argv[3], "%lf", &redshift);
  bsize = BOXSIZE;
  Om = OMEGA_M;
  if (redshift < 0) {
    fprintf(stderr, "Error: incorrect redshift: %g\n", redshift);
    return 1;
  }

  /* Reading input */
  printf("Reading input ...\n");
  if (!(fp = fopen(input, "rb"))) {
    fprintf(stderr, "Error: cannot access the input file.\n");
    return 1;
  }
  buf = (char *) malloc(sizeof(char) * CHUNK);
  if (!buf) {
    fprintf(stderr, "Error: failed memory allocation for reading input.\n");
    return 1;
  }

  /* Counting lines */
  ndata = 0;
  while ((cnt = fread(buf, sizeof(char), CHUNK, fp))) {
    p = buf;
    end = p + cnt;
    while ((p = memchr(p, '\n', end - p))) {
      p++;
      ndata++;
    }
  }
  printf("  Number of records: %ld\n", ndata);

  /* Allocating memory for the data */
  data = (DATA *) malloc(sizeof(DATA) * ndata);
  if (!data) {
    fprintf(stderr, "Error: failed memory allocation for the data.\n");
    return 1;
  }

  /* Reading data from input */
  fseek(fp, 0, SEEK_SET);
  n = nrest = 0;
  while ((cnt = fread(buf + nrest, sizeof(char), CHUNK - nrest, fp))) {
    p = buf;
    end = p + nrest + cnt;
    if (cnt < CHUNK - nrest) *end = '\n';
    while ((endl = memchr(p, '\n', end - p))) {
      *endl = '\0';
      while (p[0] == ' ' || p[0] == '\t') ++p;
      if (p[0] == COMMENT || p[0] == '\0') {
        p = endl + 1;
        continue;
      }
      if(sscanf(p, "%lf %lf %lf %lf %lf %lf %lf",
            &(data[n].x), &(data[n].y), &(data[n].z),
            &dtmp, &dtmp, &(data[n].vz), &(data[n].vmax)) != 7) {
        fprintf(stderr, "Error: failed to read line: %s\n", p);
        return 1;
      }

      n++;
      p = endl + 1;
    }
    nrest = end - p;
    memmove(buf, p, nrest);
  }

  fclose(fp);
  ndata = n;
  printf("  %ld valid objects are recorded.\n", ndata);

  /* Redshift space distortion */
  dtmp = redshift + 1;
  dtmp = dtmp * dtmp * dtmp;
  hubble = 100 * sqrt(dtmp * Om + 1 - Om);
  dtmp = (redshift + 1) / hubble;
#ifdef OMP
  printf("Redshift space convention with %d threads ...\n",
      omp_get_max_threads());
#pragma omp parallel for
#else
  printf("Redshift space convention ...\n");
#endif
  for (i = 0; i < ndata; i++) {
    data[i].z += data[i].vz * dtmp;
    if (data[i].z >= bsize) data[i].z -= bsize;
    if (data[i].z < 0) data[i].z += bsize;
  }

  /* Writing output */
  printf("Writing outputs ...\n");
  if (!(fp = fopen(output, "w"))) {
    fprintf(stderr, "Error: cannot write to file: %s\n", output);
    return 1;
  }

  end = buf;
  for (i = 0; i < ndata; i++) {
    n = snprintf(stmp, MAXLEN, "%.10g %.10g %.10g %.10g\n",
        data[i].x, data[i].y, data[i].z, data[i].vmax);

    if (end - buf + n < CHUNK) {        // there is still space in buf
      cnt = safe_strcpy(end, stmp, n + 1);
      end += cnt;
    }
    else {                              // write buf to file
      if (fwrite(buf, sizeof(char) * (end - buf), 1, fp) != 1) {
        fprintf(stderr, "Error: failed to write to output: %s\n", stmp);
        return 1;
      }
      cnt = safe_strcpy(buf, stmp, n + 1);
      end = buf + cnt;
    }
  }

  if ((n = end - buf) > 0) {
    if (fwrite(buf, sizeof(char) * n, 1, fp) != 1) {
      fprintf(stderr, "Error: failed to write to output: %s\n", stmp);
      return 1;
    }
  }

  fclose(fp);
  free(buf);
  free(data);

  return 0;
}
