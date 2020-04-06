#include <math.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_spline.h>
#include "fcfc.h"

void rdz2xyz(const DOUBLE2 *sample, const size_t ns, DATA *data,
    const size_t n) {
  size_t i;
  double *zsample, *dsample;
  real dist, ra, dec;
  gsl_interp_accel *acc;
  gsl_spline *spline;

  zsample = (double *) malloc(sizeof(double) * ns);
  dsample = (double *) malloc(sizeof(double) * ns);
  if (!zsample || !dsample) {
    P_ERR("failed to allocate memory for"
        "redshift and comoving distance samples.\n");
    exit(ERR_MEM);
  }
  for (i = 0; i < ns; i++) {
    zsample[i] = sample[i].v[0];
    dsample[i] = sample[i].v[1];
  }

  acc = gsl_interp_accel_alloc();
  spline = gsl_spline_alloc(gsl_interp_cspline, ns);
  gsl_spline_init(spline, zsample, dsample, ns);

#ifdef OMP
#pragma omp parallel for private(dist,ra,dec)
#endif
  for (i = 0; i < n; i++) {
    if (data[i].x[2] < zsample[0] || data[i].x[2] >= zsample[ns - 1]) {
#ifdef OMP
#pragma omp critical
#endif
      {
        printf(FMT_FAIL);
        P_EXT("redshift %g is outside the sampled range.\n", data[i].x[2]);
        exit(ERR_RANGE);
      }
    }
    dist = gsl_spline_eval(spline, data[i].x[2], acc);
    ra = data[i].x[0] * PI / 180.0;
    dec = data[i].x[1] * PI / 180.0;

    data[i].x[0] = dist * cos(dec) * cos(ra);
    data[i].x[1] = dist * cos(dec) * sin(ra);
    data[i].x[2] = dist * sin(dec);
  }

  free(zsample);
  free(dsample);
}

void rdz2xyz_flat_LCDM(const double OmegaM, DATA *data, const size_t n) {
  size_t i;
  real dist, ra, dec;

#ifdef OMP
#pragma omp parallel for private(dist,ra,dec)
#endif
  for (i = 0; i < n; i++) {
    dist = z2cd(OmegaM, data[i].x[2]);
    ra = data[i].x[0] * PI / 180.0;
    dec = data[i].x[1] * PI / 180.0;

    data[i].x[0] = dist * cos(dec) * cos(ra);
    data[i].x[1] = dist * cos(dec) * sin(ra);
    data[i].x[2] = dist * sin(dec);
  }
}


double comoving_dist_flat_LCDM(double x, void *OmegaM) {
  double OM = *((double *) OmegaM);
  double H = 100.0 * sqrt(1.0 - OM + OM * pow(1 + x, 3));
  return SPEED_OF_LIGHT / H;
}


double z2cd(double OmegaM, const double z) {
  gsl_integration_workspace *w;
  double result, error;
  gsl_function F;

  F.function = &comoving_dist_flat_LCDM;
  F.params = &OmegaM;

  w = gsl_integration_workspace_alloc(10000);
  gsl_integration_qag(&F, 0, z, 0, INTEG_PREC, 10000,
      GSL_INTEG_GAUSS51, w, &result, &error);
  gsl_integration_workspace_free(w);

  return result;
}
