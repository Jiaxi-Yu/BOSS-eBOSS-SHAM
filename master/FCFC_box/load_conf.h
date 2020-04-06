/*******************************************************
**                                                    **
**     Load the configuration from a file             **
**     Author: Cheng Zhao <zhaocheng03@gmail.com>     **
**                                                    **
*******************************************************/

#ifndef _LOAD_CONF_H_
#define _LOAD_CONF_H_

#include "define.h"

// Codes for (non-short) command line options.
#define OPT_DXL         500
#define OPT_DXU         501
#define OPT_DYL         502
#define OPT_DYU         503
#define OPT_DZL         504
#define OPT_DZU         505
#define OPT_DWL         506
#define OPT_DWU         507
#define OPT_DAL         508
#define OPT_DAU         509
#define OPT_RXL         510
#define OPT_RXU         511
#define OPT_RYL         512
#define OPT_RYU         513
#define OPT_RZL         514
#define OPT_RZU         515
#define OPT_RWL         516
#define OPT_RWU         517
#define OPT_RAL         518
#define OPT_RAU         519
#define OPT_DD          520
#define OPT_DR          521
#define OPT_RR          522

typedef struct config {
  char data[MAX_BUF];           // DATA_CATALOG
  char rand[MAX_BUF];           // RAND_CATALOG
  char zlist[MAX_BUF];          // REDSHIFT_LIST
  char dbin[MAX_BUF];           // DIST_BIN_FILE
  char dd[MAX_BUF];             // DD_FILE
  char dr[MAX_BUF];             // DR_FILE
  char rr[MAX_BUF];             // RR_FILE
  char output[MAX_BUF];         // OUTPUT
  int dskip;                    // DATA_HEADER
  int dwt;                      // DATA_WT_COL
  int daux;                     // DATA_AUX_COL
  int dsel;                     // DATA_SEL_MODE
  int rskip;                    // RAND_HEADER
  int rwt;                      // RAND_WT_COL
  int raux;                     // RAND_AUX_COL
  int rsel;                     // RAND_SEL_MODE
  int dcnvt;                    // DATA_CONVERT
  int rcnvt;                    // RAND_CONVERT
  int dbdim;                    // DIST_BIN_DIM
  int dbtype;                   // DIST_BIN_TYPE
  int rnum;                     // DIST_BIN_RNUM
  int r2prec;                   // DIST_BIN_PREC
  int cmode;                    // COUNT_MODE
  int moment;                   // MOMENT
  int cfmode;                   // CF_MODE
  int force;                    // FORCE
  int verb;                     // VERBOSE
  double dx[2];                 // DATA_X_MIN & DATA_X_MAX
  double dy[2];                 // DATA_Y_MIN & DATA_Y_MAX
  double dz[2];                 // DATA_Z_MIN & DATA_Z_MAX
  double dw[2];                 // DATA_WT_MIN & DATA_WT_MAX
  double da[2];                 // DATA_AUX_MIN & DATA_AUX_MAX
  double rx[2];                 // RAND_X_MIN & RAND_X_MAX
  double ry[2];                 // RAND_Y_MIN & RAND_Y_MAX
  double rz[2];                 // RAND_Z_MIN & RAND_Z_MAX
  double rw[2];                 // RAND_WT_MIN & RAND_WT_MAX
  double ra[2];                 // RAND_AUX_MIN & RAND_AUX_MAX
  double OmegaM;                // OMEGA_M
  double rmin;                  // DIST_BIN_RMIN
  double rmax;                  // DIST_BIN_RMAX
  // Non-input parameters.
  int flag_dd;                  // 0 for not caring about;
  int flag_dr;                  // 1 for computing (output);
  int flag_rr;                  // 2 for reading (input).
} CONF;

void init_conf(CONF *);

void read_opt(const int, char * const [], char *, CONF *);

int read_conf(const char *, CONF *);

int check_conf(CONF *);

int check_input(const char *, const char *);

int check_output(const char *, const char *, const int);

void print_conf(const CONF *);

void temp_conf(void);

void usage(char *);

#endif
